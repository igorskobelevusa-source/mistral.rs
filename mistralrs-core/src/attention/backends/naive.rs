#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::attention::{chunked_attention, SdpaParams};

use std::sync::atomic::{AtomicU64, Ordering};

/// Counter for synchronization throttling during decode
static SYNC_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Synchronize interval - only sync every N calls (0 = disabled)
/// Set to 0 to disable synchronization entirely for maximum decode performance
const SYNC_INTERVAL: u64 = 0;

/// Maybe synchronize the device, but with throttling to avoid stalls during decode.
/// - Metal: Never synchronize (Metal handles memory pressure internally)
/// - CUDA: Only synchronize every SYNC_INTERVAL calls when memory is low
/// - CPU: No synchronization needed
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    // Metal handles memory pressure internally - never sync
    if device.is_metal() {
        return Ok(());
    }

    // CPU doesn't need synchronization
    if device.is_cpu() {
        return Ok(());
    }

    // Synchronization disabled
    if SYNC_INTERVAL == 0 {
        return Ok(());
    }

    // Only check every SYNC_INTERVAL calls to avoid overhead
    let count = SYNC_COUNTER.fetch_add(1, Ordering::Relaxed);
    if count % SYNC_INTERVAL != 0 {
        return Ok(());
    }

    // If less than 1 GB available, synchronize (lowered threshold)
    #[cfg(target_pointer_width = "64")]
    const ONE_GIB: usize = 1024 * 1024 * 1024;
    #[cfg(not(target_pointer_width = "64"))]
    const ONE_GIB: usize = usize::MAX;

    if MemoryUsage.get_memory_available(device)? < ONE_GIB {
        device.synchronize()?;
    }
    Ok(())
}

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    // Use chunked attention with a closure that captures the necessary parameters
    chunked_attention(q, k, v, mask, |q_chunk, k, v, mask_chunk| {
        let mut att =
            MatMul.matmul_affine_mul(q_chunk, &k.t()?, sdpa_params.softmax_scale.into())?;

        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?;
            att = att.tanh()?;
            att = (att * softcap as f64)?;
        }

        if let Some(mask) = mask_chunk {
            att = att.broadcast_add(mask)?;
        }

        att = candle_nn::ops::softmax_last_dim(&att)?;
        MatMul.matmul(&att, v)
    })
}
