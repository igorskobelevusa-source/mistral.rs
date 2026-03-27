//! Metal implementation of indexed MoE forward for GGUF/ISQ quantized weights.
//!
//! Dispatches a fused Metal compute kernel — everything runs on GPU.
//! QTensor weights are dequantized to f32 once per layer, then the kernel
//! reads expert weights by offset. No CPU-side loops or index copies.

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
        let cache = lock.read().map_err(|e| {
            candle_core::Error::Msg(format!("Pipeline read error: {e}"))
        })?;
        if let Some(p) = cache.get(name) {
            return Ok(p.clone());
        }
    }
    let lib = load_moe_library(device)?;
    let func = lib.get_function(name, None).map_err(|e| {
        candle_core::Error::Msg(format!("MoE function '{name}' not found: {e}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::Msg(format!("MoE pipeline error: {e}")))?;
    let mut cache = lock.write().map_err(|e| {
        candle_core::Error::Msg(format!("Pipeline write error: {e}"))
    })?;
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

/// Metal indexed MoE forward with cached dequantization.
/// First call dequantizes and caches; subsequent calls reuse the cache.
pub fn metal_indexed_moe_forward(
    qmatmul: &QMatMul,
    x: &Tensor,
    ids: &Tensor,
    dequant_cache: &std::sync::Mutex<Option<Tensor>>,
) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => {
            // Check cache first
            let weights = {
                let cache = dequant_cache.lock().map_err(|e| {
                    candle_core::Error::Msg(format!("dequant cache lock: {e}"))
                })?;
                cache.clone()
            };
            let weights = match weights {
                Some(w) => w,
                None => {
                    let w = qtensor.dequantize(x.device())?;
                    let mut cache = dequant_cache.lock().map_err(|e| {
                        candle_core::Error::Msg(format!("dequant cache lock: {e}"))
                    })?;
                    *cache = Some(w.clone());
                    w
                }
            };
            dispatch_moe_kernel(&weights, x, ids)
        }
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            dispatch_moe_kernel(t, x, ids)
        }
    }
}

/// Dispatch the Metal MoE kernel — fully GPU, no CPU loops.
fn dispatch_moe_kernel(weights: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (num_experts, out_features, in_features) = weights.dims3()?;

    // Normalize input shape and compute dispatch params
    let (batch, topk, input_dim1, x_flat, reshape_fn): (
        usize, usize, i32, Tensor, Box<dyn Fn(Tensor) -> Result<Tensor>>
    ) = match x.dims() {
        &[b, s, 1, 1, h] => {
            let (_, _, k) = ids.dims3()?;
            let n = b * s;
            let out = out_features;
            (n, k, 1, x.reshape((n, h))?,
             Box::new(move |t| t.reshape((b, s, k, out))))
        }
        &[b, s, k, h] if k > 1 => {
            let n = b * s;
            let out = out_features;
            (n, k, k as i32, x.reshape((n * k, h))?,
             Box::new(move |t| t.reshape((b, s, k, out))))
        }
        &[n, 1, h] => {
            let (_, k) = ids.dims2()?;
            let out = out_features;
            (n, k, 1, x.reshape((n, h))?,
             Box::new(move |t| t.reshape((n, k, out))))
        }
        dims => candle_core::bail!("dispatch_moe_kernel: unsupported input shape {dims:?}"),
    };

    let flat_ids = ids.reshape((batch * topk,))?.to_dtype(DType::U32)?;

    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("dispatch_moe_kernel: expected Metal device");
    };

    // Ensure everything is contiguous and on Metal in f32
    let weights = weights.contiguous()?.to_dtype(DType::F32)?;
    let x_flat = x_flat.contiguous()?.to_dtype(DType::F32)?;
    let flat_ids = flat_ids.contiguous()?;
    let output = Tensor::zeros((batch * topk, out_features), DType::F32, x.device())?;

    let pipeline = load_pipeline(dev.device(), "indexed_moe_forward_f32")?;

    let (w_buf, w_off) = metal_buffer_and_offset(&weights)?;
    let (x_buf, x_off) = metal_buffer_and_offset(&x_flat)?;
    let (id_buf, id_off) = metal_buffer_and_offset(&flat_ids)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

    let n_out = out_features as i32;
    let k_in = in_features as i32;
    let topk_i32 = topk as i32;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(&w_buf), w_off);
    encoder.set_buffer(1, Some(&x_buf), x_off);
    encoder.set_buffer(2, Some(&id_buf), id_off);
    encoder.set_buffer(3, Some(&out_buf), out_off);
    encoder.set_bytes(4, &n_out);
    encoder.set_bytes(5, &k_in);
    encoder.set_bytes(6, &topk_i32);
    encoder.set_bytes(7, &input_dim1);

    // Grid: one threadgroup per (output_row, token, expert_slot)
    // Threads: 32 (one simdgroup) — each computes partial dot product
    let grid = MTLSize {
        width: out_features as usize,
        height: batch as usize,
        depth: topk as usize,
    };
    let threads = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, threads);

    reshape_fn(output)
}
