// Indexed MoE forward kernel for Metal
// Fused expert selection + matmul without materializing weight tensors.
// Each simdgroup computes one output element for one (token, expert_slot) pair.
//
// Grid: (out_features, batch, topk)
// Threads per group: (32, 1, 1) — one simdgroup

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// F32 variant — for dequantized ISQ weights
kernel void indexed_moe_forward_f32(
    device const float * all_weights  [[buffer(0)]],   // [n_experts, n, k] contiguous
    device const float * all_inputs   [[buffer(1)]],   // [batch_total, k] contiguous
    device const uint  * indices      [[buffer(2)]],   // [batch * topk] expert IDs
    device       float * all_outputs  [[buffer(3)]],   // [batch * topk, n] output
    constant     int   & n_out        [[buffer(4)]],   // output dim
    constant     int   & k_in         [[buffer(5)]],   // input dim
    constant     int   & topk         [[buffer(6)]],   // experts per token
    constant     int   & input_dim1   [[buffer(7)]],   // 1 for broadcast, topk for per-slot
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]]
) {
    const int row = tgpig.x;           // output feature index
    const int batch_id = tgpig.y;      // token index
    const int topk_id = tgpig.z;       // expert slot
    const int task_id = batch_id * topk + topk_id;

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

// BF16 variant
kernel void indexed_moe_forward_bf16(
    device const bfloat * all_weights  [[buffer(0)]],
    device const float  * all_inputs   [[buffer(1)]],
    device const uint   * indices      [[buffer(2)]],
    device       float  * all_outputs  [[buffer(3)]],
    constant     int    & n_out        [[buffer(4)]],
    constant     int    & k_in         [[buffer(5)]],
    constant     int    & topk         [[buffer(6)]],
    constant     int    & input_dim1   [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]]
) {
    const int row = tgpig.x;
    const int batch_id = tgpig.y;
    const int topk_id = tgpig.z;
    const int task_id = batch_id * topk + topk_id;

    if (row >= n_out) return;

    const uint expert_id = indices[task_id];
    const int input_idx = (input_dim1 == 1) ? batch_id : task_id;

    device const bfloat * w = all_weights + ((int)expert_id * n_out + row) * k_in;
    device const float  * x = all_inputs + input_idx * k_in;

    float sumf = 0.0f;
    for (int i = tiisg; i < k_in; i += 32) {
        sumf += (float)w[i] * x[i];
    }

    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        all_outputs[task_id * n_out + row] = sumf;
    }
}
