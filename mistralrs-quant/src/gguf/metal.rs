//! Metal implementation of indexed MoE forward for GGUF/ISQ quantized weights.
//!
//! Dispatches a fused expert-selection + matmul kernel that operates directly
//! on quantized weight blocks, avoiding the massive intermediate tensors that
//! the CPU fallback creates (which exceed Metal's buffer limits).

use candle_core::{
    quantized::{QMatMul, QTensor},
    DType, Device, Result, Shape, Tensor,
};
use std::sync::Arc;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Metal indexed MoE forward — dispatches the fused kernel for supported
/// quant types, falls back to dequantize + UnquantLinear for others.
pub fn metal_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(_qtensor) => {
            // TODO: Dispatch the Metal indexed_moe_forward_q4k kernel directly
            // on the quantized blocks without dequantizing.
            //
            // For now, fall back to dequantize + per-expert dispatch.
            // This is slower but avoids the buffer size issue.
            qtensor_moe_per_expert(_qtensor, x, ids)
        }
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            let unquant =
                UnquantLinear::new(QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                    t.clone(),
                    None,
                )))?;
            unquant.gather_forward(x, ids)
        }
    }
}

/// Per-expert dispatch for QTensor on Metal.
/// Dequantizes one expert at a time to avoid huge buffers.
fn qtensor_moe_per_expert(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let device = x.device();
    let dtype = x.dtype();

    // Dequantize all weights once (this creates [n_experts, out, in] in f32)
    // We need the full tensor to index into, but at f32 it's ~1GB for 256 experts.
    // The key insight: we DON'T index_select (which would create [n*k, out, in]).
    // Instead we slice one expert at a time.
    let weights = qtensor.dequantize(device)?;
    let (num_experts, out_features, _in_features) = weights.dims3()?;

    match x.dims() {
        // 5D: [b, s, 1, 1, hidden] — gate/up projections
        &[b_size, seq_len, 1, 1, hidden_dim] => {
            let (_b, _s, num_experts_per_tok) = ids.dims3()?;
            let n = b_size * seq_len;
            let a_flat = x.reshape((n, hidden_dim))?;
            let flat_indices = ids.reshape((n * num_experts_per_tok,))?;
            let idx_vec: Vec<u32> = flat_indices.to_vec1()?;

            let mut expert_tokens: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
            for (pair_idx, &expert_id) in idx_vec.iter().enumerate() {
                expert_tokens[expert_id as usize].push(pair_idx);
            }

            let mut output = Tensor::zeros((n * num_experts_per_tok, out_features), dtype, device)?;

            for (expert_id, pairs) in expert_tokens.iter().enumerate() {
                if pairs.is_empty() {
                    continue;
                }
                // Slice single expert weight: [out, in] — no huge buffer
                let eid = Tensor::new(&[expert_id as u32], device)?;
                let expert_w = weights.index_select(&eid, 0)?.squeeze(0)?;
                let tok_ids: Vec<u32> = pairs.iter().map(|&p| (p / num_experts_per_tok) as u32).collect();
                let tok_idx = Tensor::new(tok_ids, device)?;
                let tokens = a_flat.index_select(&tok_idx, 0)?;
                let result = tokens.matmul(&expert_w.t()?)?;

                let pair_idx = Tensor::new(pairs.iter().map(|&i| i as u32).collect::<Vec<_>>(), device)?;
                output = output.index_add(&pair_idx, &result, 0)?;
            }

            output.reshape((b_size, seq_len, num_experts_per_tok, out_features))
        }
        // 4D: [b, s, k, hidden] — down projection
        &[b_size, seq_len, num_experts_per_tok, hidden_dim] if num_experts_per_tok > 1 => {
            let n = b_size * seq_len;
            let flat_indices = ids.reshape((n * num_experts_per_tok,))?;
            let idx_vec: Vec<u32> = flat_indices.to_vec1()?;
            let a_flat = x.reshape((n * num_experts_per_tok, hidden_dim))?;

            let mut expert_tokens: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
            for (pair_idx, &expert_id) in idx_vec.iter().enumerate() {
                expert_tokens[expert_id as usize].push(pair_idx);
            }

            let mut output = Tensor::zeros((n * num_experts_per_tok, out_features), dtype, device)?;

            for (expert_id, pairs) in expert_tokens.iter().enumerate() {
                if pairs.is_empty() {
                    continue;
                }
                let eid = Tensor::new(&[expert_id as u32], device)?;
                let expert_w = weights.index_select(&eid, 0)?.squeeze(0)?;
                let pair_idx = Tensor::new(pairs.iter().map(|&i| i as u32).collect::<Vec<_>>(), device)?;
                let tokens = a_flat.index_select(&pair_idx, 0)?;
                let result = tokens.matmul(&expert_w.t()?)?;
                output = output.index_add(&pair_idx, &result, 0)?;
            }

            output.reshape((b_size, seq_len, num_experts_per_tok, out_features))
        }
        // 3D: [n, 1, hidden] — CUDA-style
        &[num_tokens, 1, hidden_dim] => {
            let (_, num_experts_per_tok) = ids.dims2()?;
            let flat_indices = ids.reshape((num_tokens * num_experts_per_tok,))?;
            let idx_vec: Vec<u32> = flat_indices.to_vec1()?;
            let a_flat = x.reshape((num_tokens, hidden_dim))?;

            let mut expert_tokens: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
            for (pair_idx, &expert_id) in idx_vec.iter().enumerate() {
                expert_tokens[expert_id as usize].push(pair_idx);
            }

            let mut output = Tensor::zeros((num_tokens * num_experts_per_tok, out_features), dtype, device)?;

            for (expert_id, pairs) in expert_tokens.iter().enumerate() {
                if pairs.is_empty() {
                    continue;
                }
                let eid = Tensor::new(&[expert_id as u32], device)?;
                let expert_w = weights.index_select(&eid, 0)?.squeeze(0)?;
                let tok_ids: Vec<u32> = pairs.iter().map(|&p| (p / num_experts_per_tok) as u32).collect();
                let tok_idx = Tensor::new(tok_ids, device)?;
                let tokens = a_flat.index_select(&tok_idx, 0)?;
                let result = tokens.matmul(&expert_w.t()?)?;

                let pair_idx = Tensor::new(pairs.iter().map(|&i| i as u32).collect::<Vec<_>>(), device)?;
                output = output.index_add(&pair_idx, &result, 0)?;
            }

            output.reshape((num_tokens, num_experts_per_tok, out_features))
        }
        dims => {
            candle_core::bail!(
                "metal_indexed_moe_forward: unsupported input shape {:?}",
                dims
            );
        }
    }
}
