//! Metal MoE forward — per-expert dispatch using candle's quantized matmul kernels.
//!
//! Two-phase approach:
//! 1. Rust routes tokens to experts (fast scan of expert IDs)
//! 2. For each active expert, dispatch kernel_mul_mv_q4_K_f32 with buffer offset
//!    — each dispatch handles only that expert's tokens, no ID scanning overhead

use candle_core::{
    backend::BackendStorage,
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Storage, Tensor,
};
use std::sync::Arc;

use candle_metal_kernels::{
    metal::{Buffer, ComputeCommandEncoder},
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

fn qtensor_metal_buffer(qtensor: &QTensor) -> Result<(Buffer, GgmlDType)> {
    let metal_storage = qtensor.metal_storage()?;
    Ok((metal_storage.buffer().clone(), qtensor.dtype()))
}

pub fn metal_indexed_moe_forward(
    qmatmul: &QMatMul,
    x: &Tensor,
    ids: &Tensor,
    _dequant_cache: &std::sync::Mutex<Option<Tensor>>,
) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => dispatch_quantized_moe(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => dispatch_unquantized_moe(t, x, ids),
    }
}

/// Per-expert quantized MoE dispatch.
/// Routes in Rust, dispatches kernel_mul_mv per expert with buffer offset.
fn dispatch_quantized_moe(qtensor: &Arc<QTensor>, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("dispatch_quantized_moe: expected Metal device");
    };

    let (w_buf, ggml_dtype) = qtensor_metal_buffer(qtensor)?;

    let w_shape = qtensor.shape();
    let (n_experts, n_out, n_in) = match w_shape.dims() {
        &[e, o, i] => (e, o, i),
        dims => candle_core::bail!("Expected 3D weight tensor, got {dims:?}"),
    };

    // Normalize input
    let (batch, topk, input_dim1, x_flat) = match x.dims() {
        &[b, s, 1, 1, h] => {
            let (_, _, k) = ids.dims3()?;
            (b * s, k, 1usize, x.reshape((b * s, h))?)
        }
        &[b, s, k, h] if k > 1 => (b * s, k, k, x.reshape((b * s * k, h))?),
        &[n, 1, h] => {
            let (_, k) = ids.dims2()?;
            (n, k, 1, x.reshape((n, h))?)
        }
        dims => candle_core::bail!("dispatch_quantized_moe: unsupported {dims:?}"),
    };

    let flat_ids = ids.reshape((batch * topk,))?.to_dtype(DType::U32)?;
    let idx_vec: Vec<u32> = flat_ids.to_vec1()?;
    let x_flat = x_flat.contiguous()?.to_dtype(DType::F32)?;

    // Phase 1: Route tokens to experts
    let mut expert_pairs: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
    for (pair_idx, &eid) in idx_vec.iter().enumerate() {
        expert_pairs[eid as usize].push(pair_idx);
    }

    // Weight strides
    let block_size = ggml_dtype.block_size();
    let type_size = ggml_dtype.type_size();
    let blocks_per_row = n_in / block_size;
    let row_bytes = blocks_per_row * type_size;
    let expert_bytes = n_out * row_bytes;

    // Kernel config
    let (nth0, nth1, align) = match ggml_dtype {
        GgmlDType::Q4K => (4usize, 8usize, 4usize),
        GgmlDType::Q2K => (2, 32, 4),
        GgmlDType::Q3K | GgmlDType::Q5K => (2, 32, 4),
        GgmlDType::Q6K => (2, 32, 2),
        GgmlDType::Q8_0 => (8, 8, 8),
        _ => (32, 1, 8),
    };

    let kernel_name = match ggml_dtype {
        GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        dt => candle_core::bail!("Unsupported GGML dtype for Metal MoE: {dt:?}"),
    };

    fn divide(m: usize, n: usize) -> usize {
        (m + n - 1) / n
    }

    let pipeline = dev
        .kernels()
        .load_pipeline(dev.device(), Source::Quantized, kernel_name)
        .map_err(|e| candle_core::Error::Msg(format!("MoE kernel load: {e}")))?;

    let (x_buf, _x_off) = metal_buffer_and_offset(&x_flat)?;

    // Phase 2: Dispatch per expert, collect results
    let mut all_results: Vec<Tensor> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    for (expert_id, pairs) in expert_pairs.iter().enumerate() {
        if pairs.is_empty() {
            continue;
        }

        let n_tokens = pairs.len();

        // Gather token indices for this expert
        let tok_indices: Vec<u32> = if input_dim1 == 1 {
            pairs.iter().map(|&p| (p / topk) as u32).collect()
        } else {
            pairs.iter().map(|&p| p as u32).collect()
        };
        let tok_idx_tensor = Tensor::new(tok_indices, x.device())?;
        let gathered = x_flat.index_select(&tok_idx_tensor, 0)?.contiguous()?;
        let (inp_buf, inp_off) = metal_buffer_and_offset(&gathered)?;

        // Per-expert output
        let expert_out = Tensor::zeros((n_tokens, n_out), DType::F32, x.device())?;
        let (eout_buf, eout_off) = metal_buffer_and_offset(&expert_out)?;

        // Weight offset for this expert
        let w_off = expert_id * expert_bytes;

        let ne00 = n_in as i64;
        let ne01 = n_out as i64;
        let ne02 = 1i64;
        let nb00 = 0i64;
        let nb01 = 0i64;
        let nb02 = 0i64;
        let ne10 = n_in as i64;
        let ne11 = n_tokens as i64;
        let ne12 = 1i64;
        let nb10 = 0i64;
        let nb11 = 0i64;
        let nb12 = 0i64;
        let ne0 = n_out as i64;
        let ne1 = n_tokens as i64;
        let r2: u32 = 1;
        let r3: u32 = 1;

        let tg = MTLSize {
            width: divide(n_out, align),
            height: n_tokens,
            depth: 1,
        };
        let tpg = MTLSize { width: nth0, height: nth1, depth: 1 };

        let encoder = dev.command_encoder()?;
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        set_params!(
            encoder,
            (
                (&w_buf, w_off),
                (&inp_buf, inp_off),
                (&eout_buf, eout_off),
                ne00, ne01, ne02,
                nb00, nb01, nb02,
                ne10, ne11, ne12,
                nb10, nb11, nb12,
                ne0, ne1, r2, r3
            )
        );

        encoder.use_resource(&w_buf, MTLResourceUsage::Read);
        encoder.use_resource(&inp_buf, MTLResourceUsage::Read);
        encoder.use_resource(&eout_buf, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(tg, tpg);

        all_results.push(expert_out);
        all_indices.extend(pairs.iter().map(|&i| i as u32));
    }

    // Scatter all expert results into output in one pass
    let all_out = Tensor::cat(&all_results, 0)?;
    let scatter_idx = Tensor::new(all_indices, x.device())?;
    let output = Tensor::zeros((batch * topk, n_out), DType::F32, x.device())?;
    let output = output.index_add(&scatter_idx, &all_out, 0)?;

    // Reshape
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

fn dispatch_unquantized_moe(weights: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (num_experts, out_features, _) = weights.dims3()?;
    let (batch, topk, x_flat) = match x.dims() {
        &[b, s, 1, 1, h] => { let (_, _, k) = ids.dims3()?; (b*s, k, x.reshape((b*s, h))?) }
        &[b, s, k, h] if k > 1 => (b*s, k, x.reshape((b*s*k, h))?),
        &[n, 1, h] => { let (_, k) = ids.dims2()?; (n, k, x.reshape((n, h))?) }
        dims => candle_core::bail!("unsupported {dims:?}"),
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
    for (eid, pairs) in expert_tokens.iter().enumerate() {
        if pairs.is_empty() { continue; }
        let eid_t = Tensor::new(&[eid as u32], device)?;
        let ew = weights.index_select(&eid_t, 0)?.squeeze(0)?;
        let tids: Vec<u32> = pairs.iter().map(|&p| (p / topk) as u32).collect();
        let tokens = x_flat.index_select(&Tensor::new(tids, device)?, 0)?;
        let result = tokens.matmul(&ew.t()?)?;
        let pidx = Tensor::new(pairs.iter().map(|&i| i as u32).collect::<Vec<_>>(), device)?;
        output = output.index_add(&pidx, &result, 0)?;
    }
    match x.dims() {
        &[b, s, 1, 1, _] => { let (_, _, k) = ids.dims3()?; output.reshape((b, s, k, out_features)) }
        &[b, s, k, _] if k > 1 => output.reshape((b, s, k, out_features)),
        &[n, 1, _] => { let (_, k) = ids.dims2()?; output.reshape((n, k, out_features)) }
        _ => Ok(output),
    }
}
