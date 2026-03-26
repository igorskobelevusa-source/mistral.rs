//! Metal implementation of indexed MoE forward for GGUF/ISQ quantized weights.
//!
//! Two paths:
//! 1. **Fused kernel** (bf16 weights after dequant): dispatches a Metal compute kernel
//!    that does expert-indexed matmul in one GPU pass.
//! 2. **Per-expert dispatch** (QTensor): dequantizes once, then uses the fused bf16 kernel.

use candle_core::{
    backend::BackendStorage,
    quantized::{QMatMul, QTensor},
    DType, Device, Result, Storage, Tensor,
};
use std::sync::Arc;

use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, Device as MetalRawDevice,
};
use objc2_metal::{MTLCompileOptions, MTLFunctionConstantValues, MTLMathMode, MTLSize};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

// ── Kernel source and pipeline cache ──

static MOE_PIPELINES: OnceLock<RwLock<HashMap<String, ComputePipeline>>> = OnceLock::new();

const MOE_METAL_SOURCE: &str = include_str!("../metal_kernels/indexed_moe.metal");

fn load_moe_pipeline(
    device: &MetalRawDevice,
    name: &str,
    n: u32,
    k: u32,
    batch: u32,
    topk: u32,
    input_dim1: u32,
) -> Result<ComputePipeline> {
    let cache_key = format!("{name}_{n}_{k}_{batch}_{topk}_{input_dim1}");
    let lock = MOE_PIPELINES.get_or_init(|| RwLock::new(HashMap::new()));

    {
        let cache = lock.read().map_err(|e| {
            candle_core::Error::Msg(format!("Pipeline cache read error: {e}"))
        })?;
        if let Some(p) = cache.get(&cache_key) {
            return Ok(p.clone());
        }
    }

    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Fast);
    let lib = device
        .new_library_with_source(MOE_METAL_SOURCE, Some(&opts))
        .map_err(|e| candle_core::Error::Msg(format!("MoE Metal compile error: {e}")))?;

    // Set function constants
    let constants = MTLFunctionConstantValues::new();
    unsafe {
        let n_val = n;
        let k_val = k;
        let batch_val = batch;
        let topk_val = topk;
        let input_dim1_val = input_dim1;
        constants.setConstantValue_type_atIndex(
            &n_val as *const u32 as *const std::ffi::c_void,
            objc2_metal::MTLDataType::UInt,
            0,
        );
        constants.setConstantValue_type_atIndex(
            &k_val as *const u32 as *const std::ffi::c_void,
            objc2_metal::MTLDataType::UInt,
            1,
        );
        constants.setConstantValue_type_atIndex(
            &batch_val as *const u32 as *const std::ffi::c_void,
            objc2_metal::MTLDataType::UInt,
            2,
        );
        constants.setConstantValue_type_atIndex(
            &topk_val as *const u32 as *const std::ffi::c_void,
            objc2_metal::MTLDataType::UInt,
            3,
        );
        constants.setConstantValue_type_atIndex(
            &input_dim1_val as *const u32 as *const std::ffi::c_void,
            objc2_metal::MTLDataType::UInt,
            4,
        );
    }

    let func = lib.get_function(name, Some(&constants)).map_err(|e| {
        candle_core::Error::Msg(format!("MoE Metal function '{name}' not found: {e}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            candle_core::Error::Msg(format!("MoE pipeline creation failed: {e}"))
        })?;

    let mut cache = lock.write().map_err(|e| {
        candle_core::Error::Msg(format!("Pipeline cache write error: {e}"))
    })?;
    cache.insert(cache_key, pipeline.clone());
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

/// Metal indexed MoE forward.
/// Dequantizes QTensor to bf16/f32, then dispatches fused Metal kernel.
pub fn metal_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => {
            // Dequantize to f32 on Metal (creates [n_experts, out, in] tensor ~1GB)
            let weights = qtensor.dequantize(x.device())?;
            // Dispatch fused Metal kernel on the dequantized weights
            fused_moe_metal(&weights, x, ids)
        }
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            fused_moe_metal(t, x, ids)
        }
    }
}

/// Dispatch the fused bf16/f32 Metal MoE kernel.
/// weights: [n_experts, out_features, in_features]
/// x: [batch, 1, hidden] or [b, s, 1, 1, hidden] or [b, s, k, hidden]
/// ids: expert indices
fn fused_moe_metal(weights: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (num_experts, out_features, in_features) = weights.dims3()?;

    // Normalize input shapes
    let (batch, topk, input_dim1, x_flat) = match x.dims() {
        &[b, s, 1, 1, h] => {
            let (_, _, k) = ids.dims3()?;
            let n = b * s;
            (n, k, 1usize, x.reshape((n, h))?)
        }
        &[b, s, k, h] if k > 1 => {
            let n = b * s;
            (n, k, k, x.reshape((n * k, h))?)
        }
        &[n, 1, h] => {
            let (_, k) = ids.dims2()?;
            (n, k, 1, x.reshape((n, h))?)
        }
        dims => candle_core::bail!("fused_moe_metal: unsupported input shape {dims:?}"),
    };

    let flat_ids = ids.reshape((batch * topk,))?.to_dtype(DType::U32)?;

    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("fused_moe_metal: expected Metal device");
    };

    // Ensure contiguous
    let weights = weights.contiguous()?.to_dtype(DType::F32)?;
    let x_flat = x_flat.contiguous()?.to_dtype(DType::F32)?;
    let flat_ids = flat_ids.contiguous()?;

    // Output buffer
    let output = Tensor::zeros((batch * topk, out_features), DType::F32, x.device())?;

    let pipeline = load_moe_pipeline(
        dev.device(),
        "indexed_moe_forward_f32",
        out_features as u32,
        in_features as u32,
        batch as u32,
        topk as u32,
        input_dim1 as u32,
    )?;

    let (w_buf, w_off) = metal_buffer_and_offset(&weights)?;
    let (x_buf, x_off) = metal_buffer_and_offset(&x_flat)?;
    let (id_buf, id_off) = metal_buffer_and_offset(&flat_ids)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(&w_buf), w_off);
    encoder.set_buffer(1, Some(&x_buf), x_off);
    encoder.set_buffer(2, Some(&id_buf), id_off);
    encoder.set_buffer(3, Some(&out_buf), out_off);

    // Grid: (n_output_rows, batch, topk)
    // Each threadgroup: 32 threads (one simdgroup) computing one dot product
    let grid = MTLSize {
        width: out_features as u64,
        height: batch as u64,
        depth: topk as u64,
    };
    let threads = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, threads);

    // Reshape output
    match x.dims() {
        &[b, s, 1, 1, _] => {
            let (_, _, k) = ids.dims3()?;
            output.reshape((b, s, k, out_features))
        }
        &[b, s, k, _] if k > 1 => output.reshape((b, s, k, out_features)),
        &[n, 1, _] => {
            let (_, k) = ids.dims2()?;
            output.reshape((n, k, out_features))
        }
        _ => Ok(output),
    }
}
