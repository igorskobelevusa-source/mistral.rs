// Indexed MoE forward kernel for Metal
// Fused expert selection + quantized matmul without materializing weight tensors.
// Each threadgroup computes one output row for one (token, expert_slot) pair.
//
// Port of the CUDA indexed_moe_forward kernel to Metal Shading Language.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ── GGUF block type definitions (matching candle/llama.cpp) ──

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK4_0 32
#define QK8_0 32
#define QK8_1 32

struct block_q8_1 {
    float d;       // delta
    float s;       // d * sum(qs[i])
    int8_t qs[QK8_1]; // quants
};

struct block_q4_K {
    half d;           // super-block scale for quantized scales
    half dmin;        // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];          // 4-bit quants
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

// ── Helper: simdgroup reduction ──

static inline float simd_reduce_sum(float val) {
    return simd_sum(val);
}

// ── Q4_K × Q8_1 dot product (one block) ──
// Computes partial dot product between one Q4_K block and corresponding Q8_1 blocks.
// Adapted from candle-metal-kernels quantized.metal

static float vec_dot_q4_K_q8_1(
    device const block_q4_K * w,
    device const block_q8_1 * y,
    uint tid
) {
    const int n_blocks_per_q4k = QK_K / QK8_1; // 8 blocks of q8_1 per q4k block

    float sumf = 0.0f;

    // Decode scales and mins from the packed 6-bit format
    uint8_t sc[QK_K/32];  // 8 scales
    uint8_t mn[QK_K/32];  // 8 mins

    for (int i = 0; i < 4; i++) {
        sc[2*i+0] = w->scales[i] & 63;
        sc[2*i+1] = w->scales[i+4] & 63;
        mn[2*i+0] = w->scales[i] >> 6 | ((w->scales[i+8] & 0xf) << 2);
        mn[2*i+1] = w->scales[i+4] >> 6 | ((w->scales[i+8] >> 4) << 2);
    }

    float d = (float)w->d;
    float dmin = (float)w->dmin;

    for (int j = 0; j < n_blocks_per_q4k; j++) {
        float scale = d * sc[j];
        float min_val = dmin * mn[j];
        float d_y = y[j].d;
        float s_y = y[j].s;

        // Accumulate: sum of (dequantized_weight * quantized_input)
        sumf -= min_val * s_y;  // min contribution

        int base = j * 32;
        for (int l = 0; l < 32; l++) {
            int qi = (base + l) / 2;
            int shift = ((base + l) % 2) * 4;
            uint8_t q = (w->qs[qi] >> shift) & 0xf;
            sumf += scale * (float)q * d_y * (float)y[j].qs[l];
        }
    }

    return sumf;
}

// ── Q8_0 × Q8_1 dot product ──

static float vec_dot_q8_0_q8_1(
    device const block_q8_0 * w,
    device const block_q8_1 * y,
    uint tid
) {
    float sumf = 0.0f;
    float d_w = (float)w->d;
    float d_y = y->d;

    for (int i = 0; i < QK8_0; i++) {
        sumf += (float)w->qs[i] * (float)y->qs[i];
    }

    return d_w * d_y * sumf;
}

// ── Quantize f32 input row to Q8_1 format ──

static void quantize_row_q8_1(
    device const float * x,
    thread block_q8_1 * y,
    int k
) {
    int num_blocks = k / QK8_1;
    for (int b = 0; b < num_blocks; b++) {
        float amax = 0.0f;
        for (int i = 0; i < QK8_1; i++) {
            amax = max(amax, abs(x[b * QK8_1 + i]));
        }
        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 127.0f / amax : 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < QK8_1; i++) {
            float v = x[b * QK8_1 + i] * id;
            int8_t q = (int8_t)clamp(round(v), -128.0f, 127.0f);
            y[b].qs[i] = q;
            sum += (float)q;
        }
        y[b].d = d;
        y[b].s = d * sum;
    }
}

// ── Main indexed MoE kernel ──
// Grid: (n_output_rows, batch, topk)
// Each threadgroup computes: output[batch_id * topk + topk_id][row] =
//   dot(weight[expert_id][row], input[token_id])
//
// This avoids materializing the full [batch*topk, n, k] selected weight tensor.

constant int MOE_N [[function_constant(0)]];       // output dim (rows of weight)
constant int MOE_K [[function_constant(1)]];       // input dim (cols of weight)
constant int MOE_BATCH [[function_constant(2)]];   // number of tokens
constant int MOE_TOPK [[function_constant(3)]];    // experts per token
constant int MOE_INPUT_DIM1 [[function_constant(4)]]; // 1 for CUDA-style, topk for Metal-style

// Q4_K variant
kernel void indexed_moe_forward_q4k(
    device const char  * all_weights  [[buffer(0)]],   // [n_experts, n, k] in Q4_K blocks
    device const float * all_inputs   [[buffer(1)]],   // [batch, hidden] in f32
    device const uint  * indices      [[buffer(2)]],   // [batch * topk] expert IDs
    device       float * all_outputs  [[buffer(3)]],   // [batch * topk, n] output
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int row = tgpig.x;           // output row index
    const int batch_id = tgpig.y;      // token index
    const int topk_id = tgpig.z;       // expert slot
    const int task_id = batch_id * MOE_TOPK + topk_id;

    if (row >= MOE_N) return;

    // Look up which expert this token uses for this slot
    const uint expert_id = indices[task_id];

    // Input token index
    const int input_idx = (MOE_INPUT_DIM1 == 1) ? batch_id : task_id;

    // Compute byte strides
    const int blocks_per_row = MOE_K / QK_K;
    const int weight_row_bytes = blocks_per_row * sizeof(block_q4_K);
    const int weight_expert_bytes = MOE_N * weight_row_bytes;

    // Pointers to this expert's weight row and this token's input
    device const block_q4_K * w = (device const block_q4_K *)(
        all_weights + expert_id * weight_expert_bytes + row * weight_row_bytes
    );
    device const float * x = all_inputs + input_idx * MOE_K;

    // Quantize input to Q8_1 on the fly (in threadgroup memory for efficiency)
    // For simplicity, each thread processes independently.
    // A production kernel would use shared memory.

    const int q8_blocks = MOE_K / QK8_1;
    float sumf = 0.0f;

    // Direct computation: iterate over Q4_K blocks
    // Each Q4_K block covers QK_K=256 elements = 8 Q8_1 blocks
    const int n_q4k_blocks = blocks_per_row;

    for (int qb = tiisg; qb < n_q4k_blocks; qb += 32) {
        device const block_q4_K * wb = &w[qb];
        float d = (float)wb->d;
        float dmin = (float)wb->dmin;

        // Decode scales and mins
        uint8_t sc[8];
        uint8_t mn[8];
        for (int i = 0; i < 4; i++) {
            sc[2*i+0] = wb->scales[i] & 63;
            sc[2*i+1] = wb->scales[i+4] & 63;
            mn[2*i+0] = wb->scales[i] >> 6 | ((wb->scales[i+8] & 0xf) << 2);
            mn[2*i+1] = wb->scales[i+4] >> 6 | ((wb->scales[i+8] >> 4) << 2);
        }

        int base_idx = qb * QK_K;

        for (int j = 0; j < 8; j++) { // 8 sub-blocks of 32 elements
            float scale = d * sc[j];
            float min_val = dmin * mn[j];
            float sub_sum = 0.0f;
            float sub_x_sum = 0.0f;

            for (int l = 0; l < 32; l++) {
                int idx = base_idx + j * 32 + l;
                if (idx >= MOE_K) break;

                int qi = idx / 2;
                int shift = (idx % 2) * 4;
                uint8_t q = (wb->qs[qi - qb * (QK_K/2)] >> shift) & 0xf;

                float xv = x[idx];
                sub_sum += (float)q * xv;
                sub_x_sum += xv;
            }

            sumf += scale * sub_sum - min_val * sub_x_sum;
        }
    }

    // Simdgroup reduction
    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        all_outputs[task_id * MOE_N + row] = sumf;
    }
}

// BF16 variant
kernel void indexed_moe_forward_bf16(
    device const bfloat * all_weights  [[buffer(0)]],   // [n_experts, n, k] in bf16
    device const float  * all_inputs   [[buffer(1)]],   // [batch, hidden] in f32
    device const uint   * indices      [[buffer(2)]],   // [batch * topk] expert IDs
    device       float  * all_outputs  [[buffer(3)]],   // [batch * topk, n] output
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int row = tgpig.x;
    const int batch_id = tgpig.y;
    const int topk_id = tgpig.z;
    const int task_id = batch_id * MOE_TOPK + topk_id;

    if (row >= MOE_N) return;

    const uint expert_id = indices[task_id];
    const int input_idx = (MOE_INPUT_DIM1 == 1) ? batch_id : task_id;

    device const bfloat * w = all_weights + (expert_id * MOE_N + row) * MOE_K;
    device const float  * x = all_inputs + input_idx * MOE_K;

    float sumf = 0.0f;
    for (int i = tiisg; i < MOE_K; i += 32) {
        sumf += (float)w[i] * x[i];
    }

    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        all_outputs[task_id * MOE_N + row] = sumf;
    }
}

// F32 variant — for dequantized ISQ weights
kernel void indexed_moe_forward_f32(
    device const float * all_weights  [[buffer(0)]],
    device const float * all_inputs   [[buffer(1)]],
    device const uint  * indices      [[buffer(2)]],
    device       float * all_outputs  [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int row = tgpig.x;
    const int batch_id = tgpig.y;
    const int topk_id = tgpig.z;
    const int task_id = batch_id * MOE_TOPK + topk_id;

    if (row >= MOE_N) return;

    const uint expert_id = indices[task_id];
    const int input_idx = (MOE_INPUT_DIM1 == 1) ? batch_id : task_id;

    device const float * w = all_weights + (expert_id * MOE_N + row) * MOE_K;
    device const float * x = all_inputs + input_idx * MOE_K;

    float sumf = 0.0f;
    for (int i = tiisg; i < MOE_K; i += 32) {
        sumf += w[i] * x[i];
    }

    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        all_outputs[task_id * MOE_N + row] = sumf;
    }
}
