// Custom indexed MoE forward kernel for Metal with device-buffer routing.
// Based on candle's kernel_mul_mm_id but reads routing table from device memory
// instead of threadgroup memory — eliminates the 32KB threadgroup limit.
//
// Two-phase:
// 1. Rust builds per-expert routing table as device buffer
// 2. This kernel does tiled simdgroup matmul per expert
//
// Grid: (ceil(max_tokens_per_expert/32), ceil(n_out/64), n_experts)
// Threads: 128 (4 simdgroups)

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K 256
#define K_SCALE_SIZE 12

// Block types
typedef struct {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

// Tiling constants (same as candle/llama.cpp)
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4
#define THREAD_MAT_N 2
#define THREAD_PER_ROW 2
#define THREAD_PER_COL 4
#define SG_MAT_SIZE 64

// Q4_K dequantization into half4x4 (16 half values from one block position)
inline void dequantize_q4_K(device const block_q4_K * qb, short il, thread half4x4 & reg) {
    const int ib32 = il / 2;
    const int is  = 2 * ib32;

    device const uint8_t * q = qb->qs + 16 * ib32;
    device const uint8_t * sc = qb->scales;

    // Decode scales/mins
    uint8_t d_sc, d_mn;
    if (is < 4) {
        d_sc = sc[is] & 63;
        d_mn = (sc[is] >> 6) | ((sc[is + 8] & ((is < 2) ? 0x0F : 0xF0)) >> ((is < 2) ? 0 : 2));
    } else {
        d_sc = sc[is] & 63;
        d_mn = (sc[is] >> 6) | ((sc[is + 4] & ((is < 6) ? 0x0F : 0xF0)) >> ((is < 6) ? 0 : 2));
    }

    half d = qb->d;
    half dmin = qb->dmin;
    half scale = d * (half)d_sc;
    half mn = dmin * (half)d_mn;

    int shift = (il % 2) * 4;
    for (int i = 0; i < 16; i++) {
        int qi = q[i];
        half v = scale * (half)((qi >> shift) & 0xF) - mn;
        reg[i / 4][i % 4] = v;
    }
}

// Routed MoE matmul kernel for Q4_K weights.
// route_counts[expert_id] = number of tokens for this expert
// route_indices[offset + i] = (token_idx, expert_slot) packed as uint
// Rust pre-computes offset = sum of counts for experts < expert_id
kernel void moe_mm_q4k_routed(
    device const  uchar  * all_weights     [[buffer(0)]],  // [n_experts, n_out, n_in] Q4_K
    device const  float  * all_inputs      [[buffer(1)]],  // [batch_total, n_in] f32
    device        float  * all_outputs     [[buffer(2)]],  // [batch * topk, n_out] f32
    device const  uint   * route_counts    [[buffer(3)]],  // [n_experts] token count per expert
    device const  uint   * route_tok_ids   [[buffer(4)]],  // [total_pairs] token indices (flat)
    device const  uint   * route_pair_ids  [[buffer(5)]],  // [total_pairs] output pair indices
    device const  uint   * route_offsets   [[buffer(6)]],  // [n_experts] cumulative offset into route arrays
    constant      int    & n_out           [[buffer(7)]],  // output dimension
    constant      int    & n_in            [[buffer(8)]],  // input dimension
    threadgroup   uchar  * shared_memory   [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiitg [[thread_index_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint expert_id = tgpig.z;
    const uint n_tokens_for_expert = route_counts[expert_id];

    if (n_tokens_for_expert == 0) return;

    const uint route_offset = route_offsets[expert_id];
    const uint r0 = tgpig.y; // output tile row
    const uint r1 = tgpig.x; // token tile col

    if (r1 * BLOCK_SIZE_N >= n_tokens_for_expert) return;

    short n_rows = (n_out - (int)(r0 * BLOCK_SIZE_M) < BLOCK_SIZE_M)
                 ? (n_out - (int)(r0 * BLOCK_SIZE_M)) : BLOCK_SIZE_M;
    short n_cols = ((int)n_tokens_for_expert - (int)(r1 * BLOCK_SIZE_N) < BLOCK_SIZE_N)
                 ? ((int)n_tokens_for_expert - (int)(r1 * BLOCK_SIZE_N)) : BLOCK_SIZE_N;

    short thread_row = ((short)tiitg / THREAD_PER_ROW) < n_rows
                     ? ((short)tiitg / THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg / THREAD_PER_COL) < n_cols
                     ? ((short)tiitg / THREAD_PER_COL) : n_cols - 1;

    // Simdgroup matrix accumulators
    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++) {
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // Weight pointer for this expert
    const int blocks_per_row = n_in / QK_K;
    const int expert_stride = n_out * blocks_per_row * (int)sizeof(block_q4_K);

    short il = (tiitg % THREAD_PER_ROW);
    constexpr short nl = 2; // Q4_K: 2 half-blocks per full block
    short offset1 = il / nl;

    device const block_q4_K * x = (device const block_q4_K *)(
        all_weights + expert_id * expert_stride
        + (r0 * BLOCK_SIZE_M + thread_row) * blocks_per_row * sizeof(block_q4_K)
    ) + offset1;

    // Input pointer — read from routed token
    uint local_col = r1 * BLOCK_SIZE_N + thread_col;
    uint tok_idx = (local_col < n_tokens_for_expert)
                 ? route_tok_ids[route_offset + local_col] : 0;

    device const float * y = all_inputs
        + tok_idx * n_in
        + (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL));

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    for (int loop_k = 0; loop_k < n_in; loop_k += BLOCK_SIZE_K) {
        // Dequantize weight block into threadgroup memory
        half4x4 temp_a;
        dequantize_q4_K(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8)
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8)
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        // Load input into threadgroup memory
        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL))
            = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Simdgroup matrix multiply-accumulate
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);

            for (int i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_SIZE * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_SIZE * SG_MAT_SIZE;
        }
    }

    // Write results — scatter to correct output positions
    threadgroup float * temp_str = (threadgroup float *)shared_memory;

    if ((sgitg / 2) == 0) {
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (sgitg % 2) + (i % 4) * BLOCK_SIZE_M + (i / 4) * 8, BLOCK_SIZE_M);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((sgitg / 2) == 1) {
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (sgitg % 2) + (i % 4) * BLOCK_SIZE_M + (i / 4) * 8, BLOCK_SIZE_M);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to output using scattered pair indices
    for (int j = tiitg; j < n_cols * n_rows; j += 128) {
        int col = j / n_rows;
        int row = j % n_rows;

        uint local_idx = r1 * BLOCK_SIZE_N + col;
        if (local_idx >= n_tokens_for_expert) continue;

        uint pair_idx = route_pair_ids[route_offset + local_idx];
        int out_row = r0 * BLOCK_SIZE_M + row;
        if (out_row >= n_out) continue;

        all_outputs[pair_idx * n_out + out_row] = *(temp_str + col * BLOCK_SIZE_M + row);
    }
}
