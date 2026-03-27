// Indexed MoE forward kernel for Metal — Q4_K quantized weights
// Adapted from candle's kernel_mul_mv_q4_K_f32 with expert index lookup.
// Each threadgroup computes N_DST output rows for one (token, expert_slot) pair
// directly on quantized blocks — no dequantization needed.
//
// Grid: (n_rows/N_DST, batch, topk)
// Threads: (32, 1, 1) — one simdgroup

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define QK_K 256
#define K_SCALE_SIZE 12

// GGUF block types
typedef struct {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

// Number of output rows per threadgroup
#define N_DST 4

// ── Q4_K indexed MoE forward ──
// Computes: output[task_id, row] = dot(weights[expert_id][row], input[token_id])
// where expert_id = indices[task_id]

kernel void indexed_moe_forward_q4k_f32(
    device const  void  * all_weights  [[buffer(0)]],  // all expert weights in Q4_K blocks
    device const  float * all_inputs   [[buffer(1)]],  // [batch_total, k] f32 input
    device const  uint  * indices      [[buffer(2)]],  // [batch * topk] expert IDs
    device        float * all_outputs  [[buffer(3)]],  // [batch * topk, n] output
    constant      int   & n_out        [[buffer(4)]],  // output dim (rows per expert)
    constant      int   & k_in         [[buffer(5)]],  // input dim (cols per expert)
    constant      int   & topk_val     [[buffer(6)]],  // experts per token
    constant      int   & input_dim1   [[buffer(7)]],  // 1=broadcast, topk=per-slot
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int first_row = tgpig.x * N_DST;
    const int batch_id = tgpig.y;
    const int topk_id = tgpig.z;
    const int task_id = batch_id * topk_val + topk_id;

    if (first_row >= n_out) return;

    // Expert selection
    const uint expert_id = indices[task_id];
    const int input_idx = (input_dim1 == 1) ? batch_id : task_id;

    // Compute strides
    const int nb = k_in / QK_K;  // blocks per row
    const int expert_blocks = n_out * nb;  // total blocks per expert

    // Pointers
    device const block_q4_K * x = (device const block_q4_K *)all_weights + expert_id * expert_blocks + first_row * nb;
    device const float      * y = all_inputs + input_idx * k_in;
    device       float      * out = all_outputs + task_id * n_out;

    // Adapted from candle's kernel_mul_mv_q4_K_f32_impl
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    const int ib_row = 0;  // always start at first_row's block offset (already applied to x)

    float yl[16];
    float yh[16];
    float sumf[N_DST] = {0.f};

    const int step = sizeof(block_q4_K) * nb / 2;

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    // How many rows to actually compute (handle tail)
    const int n_rows = min(N_DST, n_out - first_row);

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < n_rows; row++) {

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            sc += step;
            dh += step;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < n_rows; ++row) {
        float all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            out[first_row + row] = all_sum;
        }
    }
}

// F32 variant (for unquantized/shared expert weights)
kernel void indexed_moe_forward_f32(
    device const float * all_weights  [[buffer(0)]],
    device const float * all_inputs   [[buffer(1)]],
    device const uint  * indices      [[buffer(2)]],
    device       float * all_outputs  [[buffer(3)]],
    constant     int   & n_out        [[buffer(4)]],
    constant     int   & k_in         [[buffer(5)]],
    constant     int   & topk_val     [[buffer(6)]],
    constant     int   & input_dim1   [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]]
) {
    const int row = tgpig.x;
    const int batch_id = tgpig.y;
    const int topk_id = tgpig.z;
    const int task_id = batch_id * topk_val + topk_id;

    if (row >= n_out) return;

    const uint expert_id = indices[task_id];
    const int input_idx = (input_dim1 == 1) ? batch_id : task_id;

    device const float * w = all_weights + ((int)expert_id * n_out + row) * k_in;
    device const float * x = all_inputs + input_idx * k_in;

    float sumf = 0.0f;
    for (int i = tiisg; i < k_in; i += 32) {
        sumf += w[i] * x[i];
    }

    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        all_outputs[task_id * n_out + row] = sumf;
    }
}
