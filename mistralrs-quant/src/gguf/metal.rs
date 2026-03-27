//! Metal implementation of indexed MoE forward for GGUF/ISQ quantized weights.
//!
//! Dispatches candle's `kernel_mul_mv_id_q4_K_f32` — the expert-indexed
//! quantized matmul that operates directly on Q4_K blocks with simdgroup ops.
//! No dequantization, no CPU loops, full GPU speed.

use candle_core::{
    backend::BackendStorage,
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Storage, Tensor,
};
use std::sync::Arc;

use candle_metal_kernels::{
    metal::{Buffer, ComputeCommandEncoder, Device as MetalRawDevice},
    set_params, Kernels, Source,
};
use objc2_metal::{MTLResourceUsage, MTLSize};

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

/// Get the raw Metal buffer from a QTensor without dequantizing.
fn qtensor_metal_buffer(qtensor: &QTensor) -> Result<(Buffer, GgmlDType)> {
    let metal_storage = qtensor.metal_storage()?;
    Ok((metal_storage.buffer().clone(), qtensor.dtype()))
}

/// Metal indexed MoE forward — dispatches candle's fused kernel.
/// Operates directly on quantized blocks, no dequantization needed.
pub fn metal_indexed_moe_forward(
    qmatmul: &QMatMul,
    x: &Tensor,
    ids: &Tensor,
    _dequant_cache: &std::sync::Mutex<Option<Tensor>>,
) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => dispatch_quantized_moe(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // Unquantized weights — use regular per-expert matmul
            dispatch_unquantized_moe(t, x, ids)
        }
    }
}

/// Dispatch candle's kernel_mul_mv_id for quantized MoE weights.
/// This calls the fused expert-indexed Q4_K kernel directly.
fn dispatch_quantized_moe(qtensor: &Arc<QTensor>, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("dispatch_quantized_moe: expected Metal device");
    };

    // Get weight buffer directly (no dequantization!)
    let (w_buf, ggml_dtype) = qtensor_metal_buffer(qtensor)?;

    // Weight shape: [n_experts, n_out, n_in] stored as contiguous Q4_K blocks
    // QTensor shape gives us the logical dimensions
    let w_shape = qtensor.shape();
    let (n_experts, n_out, n_in) = match w_shape.dims() {
        &[e, o, i] => (e, o, i),
        dims => candle_core::bail!("Expected 3D weight tensor, got {dims:?}"),
    };

    // Normalize input shape
    let (batch, topk, input_dim1, x_flat) = match x.dims() {
        &[b, s, 1, 1, h] => {
            let (_, _, k) = ids.dims3()?;
            (b * s, k, 1usize, x.reshape((b * s, h))?)
        }
        &[b, s, k, h] if k > 1 => {
            (b * s, k, k, x.reshape((b * s * k, h))?)
        }
        &[n, 1, h] => {
            let (_, k) = ids.dims2()?;
            (n, k, 1, x.reshape((n, h))?)
        }
        dims => candle_core::bail!("dispatch_quantized_moe: unsupported input {dims:?}"),
    };

    // Expert IDs as i32 (candle kernel reads int32_t from raw bytes)
    let flat_ids = ids.reshape((batch, topk))?.to_dtype(DType::U32)?;

    let x_flat = x_flat.contiguous()?.to_dtype(DType::F32)?;
    let flat_ids = flat_ids.contiguous()?;

    // Output: [batch * topk, n_out]
    let output = Tensor::zeros((batch * topk, n_out), DType::F32, x.device())?;

    let (x_buf, x_off) = metal_buffer_and_offset(&x_flat)?;
    let (id_buf, id_off) = metal_buffer_and_offset(&flat_ids)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

    // Kernel parameters matching kernel_mul_mv_id signature
    let nei0 = topk as i64;        // experts per token
    let nei1 = batch as i64;       // number of tokens
    let nbi1 = (topk * 4) as u64;  // stride of ids in bytes (u32 = 4 bytes)

    let ne00 = n_in as i64;        // input dim (K)
    let ne01 = n_out as i64;       // output dim (N)
    let ne02 = 1i64;               // batch dim in weights

    // Byte strides for weights
    let block_size = ggml_dtype.block_size();
    let type_size = ggml_dtype.type_size();
    let nb00 = type_size as u64;
    let blocks_per_row = n_in / block_size;
    let nb01 = (blocks_per_row * type_size) as u64;  // bytes per weight row
    let nb02 = (n_out as u64) * nb01;                 // bytes per expert

    let ne10 = n_in as i64;
    let ne11 = 1i64;               // one token at a time per dispatch
    let ne12 = 1i64;
    let ne13 = 1i64;
    let nb10 = 4u64;               // f32 = 4 bytes
    let nb11 = (n_in * 4) as u64;  // bytes per token
    let nb12 = nb11;

    let ne0 = n_out as i64;
    let ne1 = 1i64;
    let nb1 = (n_out * 4) as u64;

    // Thread group config for Q4_K
    let (nth0, nth1, align) = match ggml_dtype {
        GgmlDType::Q4K => (4usize, 8usize, 4usize),
        GgmlDType::Q2K => (2, 32, 4),
        GgmlDType::Q3K | GgmlDType::Q5K => (2, 32, 4),
        GgmlDType::Q6K => (2, 32, 2),
        GgmlDType::Q8_0 => (8, 8, 8),
        _ => (32, 1, 8),
    };

    let kernel_name = match ggml_dtype {
        GgmlDType::Q4K => "kernel_mul_mv_id_q4_K_f32",
        GgmlDType::Q2K => "kernel_mul_mv_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_id_q3_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_id_q6_K_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        dt => candle_core::bail!("Unsupported GGML dtype for Metal MoE: {dt:?}"),
    };

    fn divide(m: usize, n: usize) -> usize {
        (m + n - 1) / n
    }

    let thread_groups = MTLSize {
        width: divide(n_out, align),
        height: 1, // one token per dispatch in _id mode
        depth: (topk * batch) as usize,
    };
    let threads_per_group = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };

    let pipeline = dev
        .kernels()
        .load_pipeline(dev.device(), Source::Quantized, kernel_name)
        .map_err(|e| candle_core::Error::Msg(format!("MoE kernel load failed: {e}")))?;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Buffer layout for kernel_mul_mv_id:
    // 0: src0s (all expert weights)
    // 1: src1 (input tokens)
    // 2: dst (output)
    // 3: ids (expert indices)
    // 4+: scalar args
    set_params!(
        encoder,
        (
            (&w_buf, 0usize),     // src0s
            (&x_buf, x_off),      // src1
            (&out_buf, out_off),   // dst
            (&id_buf, id_off),     // ids
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    encoder.use_resource(&w_buf, MTLResourceUsage::Read);
    encoder.use_resource(&x_buf, MTLResourceUsage::Read);
    encoder.use_resource(&id_buf, MTLResourceUsage::Read);
    encoder.use_resource(&out_buf, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    // Reshape output
    match x.dims() {
        &[b, s, 1, 1, _] => {
            let (_, _, k) = ids.dims3()?;
            output.reshape((b, s, k, n_out))
        }
        &[b, s, k, _] if k > 1 => output.reshape((b, s, k, n_out)),
        &[n, 1, _] => {
            let (_, k) = ids.dims2()?;
            output.reshape((n, k, n_out))
        }
        _ => Ok(output),
    }
}

/// Fallback for unquantized weights (shared expert, etc.)
fn dispatch_unquantized_moe(weights: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (num_experts, out_features, in_features) = weights.dims3()?;

    let (batch, topk, x_flat) = match x.dims() {
        &[b, s, 1, 1, h] => {
            let (_, _, k) = ids.dims3()?;
            (b * s, k, x.reshape((b * s, h))?)
        }
        &[b, s, k, h] if k > 1 => (b * s, k, x.reshape((b * s * k, h))?),
        &[n, 1, h] => {
            let (_, k) = ids.dims2()?;
            (n, k, x.reshape((n, h))?)
        }
        dims => candle_core::bail!("dispatch_unquantized_moe: unsupported {dims:?}"),
    };

    let flat_ids = ids.reshape((batch * topk,))?;
    let idx_vec: Vec<u32> = flat_ids.to_vec1()?;

    let mut expert_tokens: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
    for (pair_idx, &eid) in idx_vec.iter().enumerate() {
        expert_tokens[eid as usize].push(pair_idx);
    }

    let device = x.device();
    let dtype = x.dtype();
    let mut output = Tensor::zeros((batch * topk, out_features), dtype, device)?;

    for (expert_id, pairs) in expert_tokens.iter().enumerate() {
        if pairs.is_empty() {
            continue;
        }
        let eid = Tensor::new(&[expert_id as u32], device)?;
        let expert_w = weights.index_select(&eid, 0)?.squeeze(0)?;
        let tok_ids: Vec<u32> = pairs
            .iter()
            .map(|&p| (p / topk) as u32)
            .collect();
        let tok_idx = Tensor::new(tok_ids, device)?;
        let tokens = x_flat.index_select(&tok_idx, 0)?;
        let result = tokens.matmul(&expert_w.t()?)?;
        let pair_idx =
            Tensor::new(pairs.iter().map(|&i| i as u32).collect::<Vec<_>>(), device)?;
        output = output.index_add(&pair_idx, &result, 0)?;
    }

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
