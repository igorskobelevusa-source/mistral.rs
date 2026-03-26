//! Metal implementation of indexed MoE forward for GGUF/ISQ quantized weights.
//!
//! Dispatches a fused Metal compute kernel that handles all (token, expert) pairs
//! in a single GPU pass. QTensor weights are dequantized to f32 once, then the
//! kernel reads expert weights by offset — no index_select, no huge buffers.

use candle_core::{
    backend::BackendStorage,
    quantized::{QMatMul, QTensor},
    DType, Device, Result, Storage, Tensor,
};
use std::sync::Arc;

use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, Device as MetalRawDevice, Library,
};
use objc2_metal::{MTLCompileOptions, MTLMathMode, MTLSize};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

// ── Kernel loading ──

static MOE_LIBRARY: OnceLock<Library> = OnceLock::new();
static MOE_PIPELINES: OnceLock<RwLock<HashMap<String, ComputePipeline>>> = OnceLock::new();

const MOE_METAL_SOURCE: &str = include_str!("../metal_kernels/indexed_moe.metal");

fn load_moe_library(device: &MetalRawDevice) -> Result<Library> {
    if let Some(lib) = MOE_LIBRARY.get() {
        return Ok(lib.clone());
    }
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Fast);
    let lib = device
        .new_library_with_source(MOE_METAL_SOURCE, Some(&opts))
        .map_err(|e| candle_core::Error::Msg(format!("MoE Metal compile error: {e}")))?;
    Ok(MOE_LIBRARY.get_or_init(|| lib).clone())
}

fn load_pipeline(device: &MetalRawDevice, name: &str) -> Result<ComputePipeline> {
    let lock = MOE_PIPELINES.get_or_init(|| RwLock::new(HashMap::new()));
    {
        let cache = lock
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Pipeline read error: {e}")))?;
        if let Some(p) = cache.get(name) {
            return Ok(p.clone());
        }
    }
    let lib = load_moe_library(device)?;
    let func = lib.get_function(name, None).map_err(|e| {
        candle_core::Error::Msg(format!("MoE Metal function '{name}' not found: {e}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::Msg(format!("MoE pipeline error: {e}")))?;
    let mut cache = lock
        .write()
        .map_err(|e| candle_core::Error::Msg(format!("Pipeline write error: {e}")))?;
    cache.insert(name.to_string(), pipeline.clone());
    Ok(pipeline)
}

fn metal_buffer_and_offset(tensor: &Tensor) -> Result<(Buffer, usize)> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(m) => {
            let offset = layout.start_offset() * m.dtype().size_in_bytes();
            Ok((m.buffer().clone(), offset))
        }
        _ => candle_core::bail!("Expected Metal tensor for MoE kernel"),
    }
}

// ── Public API ──

pub fn metal_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => {
            let weights = qtensor.dequantize(x.device())?;
            fused_moe_metal(&weights, x, ids)
        }
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => fused_moe_metal(t, x, ids),
    }
}

/// Dispatch fused Metal MoE kernel.
fn fused_moe_metal(weights: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (_num_experts, out_features, in_features) = weights.dims3()?;

    let (batch, topk, input_dim1, x_flat, orig_dims) = match x.dims() {
        &[b, s, 1, 1, h] => {
            let (_, _, k) = ids.dims3()?;
            let n = b * s;
            (n, k, 1i32, x.reshape((n, h))?, vec![b, s, k])
        }
        &[b, s, k, h] if k > 1 => {
            let n = b * s;
            (n, k, k as i32, x.reshape((n * k, h))?, vec![b, s, k])
        }
        &[n, 1, h] => {
            let (_, k) = ids.dims2()?;
            (n, k, 1i32, x.reshape((n, h))?, vec![n, k])
        }
        dims => candle_core::bail!("fused_moe_metal: unsupported input shape {dims:?}"),
    };

    let flat_ids = ids.reshape((batch * topk,))?.to_dtype(DType::U32)?;

    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("fused_moe_metal: expected Metal device");
    };

    let weights = weights.contiguous()?.to_dtype(DType::F32)?;
    let x_flat = x_flat.contiguous()?.to_dtype(DType::F32)?;
    let flat_ids = flat_ids.contiguous()?;

    let output = Tensor::zeros((batch * topk, out_features), DType::F32, x.device())?;

    // Per-expert dispatch using candle's optimized Metal GEMM.
    // This is faster than our custom kernel because candle uses simdgroup_matrix_multiply.
    // Dequantize once (1GB), then slice per expert and batch-matmul.
    let idx_vec: Vec<u32> = flat_ids.to_vec1()?;

    let mut expert_tokens: Vec<Vec<usize>> = vec![Vec::new(); _num_experts];
    for (pair_idx, &expert_id) in idx_vec.iter().enumerate() {
        expert_tokens[expert_id as usize].push(pair_idx);
    }

    let mut output = Tensor::zeros((batch * topk, out_features), DType::F32, x.device())?;

    for (expert_id, pairs) in expert_tokens.iter().enumerate() {
        if pairs.is_empty() {
            continue;
        }
        let eid = Tensor::new(&[expert_id as u32], x.device())?;
        let expert_w = weights.index_select(&eid, 0)?.squeeze(0)?; // [out, in]

        // Gather input tokens for this expert
        let input_ids: Vec<u32> = if input_dim1 == 1 {
            pairs.iter().map(|&p| (p / topk) as u32).collect()
        } else {
            pairs.iter().map(|&p| p as u32).collect()
        };
        let tok_idx = Tensor::new(input_ids, x.device())?;
        let tokens = x_flat.index_select(&tok_idx, 0)?; // [batch_for_expert, in]

        // Batch matmul via candle's optimized Metal GEMM
        let result = tokens.matmul(&expert_w.t()?)?; // [batch_for_expert, out]

        // Scatter results back
        let pair_idx = Tensor::new(pairs.iter().map(|&i| i as u32).collect::<Vec<_>>(), x.device())?;
        output = output.index_add(&pair_idx, &result, 0)?;
    }

    // Reshape output to match expected shape
    match orig_dims.as_slice() {
        &[b, s, k] => output.reshape((b, s, k, out_features)),
        &[n, k] => output.reshape((n, k, out_features)),
        _ => Ok(output),
    }
}
