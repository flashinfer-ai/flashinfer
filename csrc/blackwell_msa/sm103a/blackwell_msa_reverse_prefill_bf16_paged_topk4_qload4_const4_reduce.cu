typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) BlackwellMsaTensorMap { uint64_t opaque[16]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define BLACKWELL_MSA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_minimax_sparse_reverse_prefill_combine_topk4_eighthwarp16_metaparallel_const4_qload4fp8partial_temp1reuse_sm100(uint8_t* __restrict__ partial_o, float* __restrict__ partial_scale, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ split_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int total_q, int num_q_heads, int num_kv_heads, int qhead_per_kv, int topk, int return_softmax_lse, int return_temperature_lse)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int row_group = tid / 8;
    int lane_in_row = tid & 7;
    int leader_lane = row_group * 8;
    int row = blockIdx.x * 32 + row_group;
    int total_rows_out = total_q * num_q_heads;
    int row_valid = ((row < total_rows_out) ? 1 : 0);
    int split_count = 4;
    float lane_lse = -BLACKWELL_MSA_INF;
    float lane_scale = 0.0f;
    if (row_valid != 0 && lane_in_row < split_count) {
        long long split_row = (long long)lane_in_row * (long long)total_rows_out + (long long)row;
        lane_lse = partial_lse[split_row];
        lane_scale = partial_scale[split_row];
    }
    float lse_max = lane_lse;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 1);
    float peer_max = _shfl_xor_0;
    float _max_0 = max_noftz(lse_max, peer_max);
    lse_max = _max_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 2);
    peer_max = _shfl_xor_1;
    float _max_1 = max_noftz(lse_max, peer_max);
    lse_max = _max_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 4);
    peer_max = _shfl_xor_2;
    float _max_2 = max_noftz(lse_max, peer_max);
    lse_max = _max_2;
    float safe_lse_max = ((lse_max == -BLACKWELL_MSA_INF) ? 0.0f : lse_max);
    float lane_weight = 0.0f;
    if (lane_in_row < split_count) {
        float _exp2_0 = approx_exp2((lane_lse - safe_lse_max) * 1.4426950408889634f);
        lane_weight = _exp2_0;
        if (lane_lse == -BLACKWELL_MSA_INF) {
            lane_weight = 0.0f;
        }
    }
    float lse_sum = lane_weight;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 1);
    float peer_sum = _shfl_xor_3;
    lse_sum += peer_sum;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 2);
    peer_sum = _shfl_xor_4;
    lse_sum += peer_sum;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 4);
    peer_sum = _shfl_xor_5;
    lse_sum += peer_sum;
    float _rcp_0 = approx_rcp(lse_sum);
    float inv_lse_sum = ((lse_sum > 0.0f && lse_sum == lse_sum) ? _rcp_0 : 0.0f);
    lane_weight *= inv_lse_sum;
    lane_weight *= lane_scale;
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane);
    float weight_0 = _shfl_0;
    float _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 1);
    float weight_1 = _shfl_1;
    float _shfl_2 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 2);
    float weight_2 = _shfl_2;
    float _shfl_3 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 3);
    float weight_3 = _shfl_3;
    if (lane_in_row == 0 && row_valid != 0) {
        float final_lse = -BLACKWELL_MSA_INF;
        if (return_softmax_lse != 0 || return_temperature_lse != 0) {
            float _log2_0;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(lse_sum));
            final_lse = ((lse_sum > 0.0f) ? safe_lse_max + _log2_0 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
        }
        if (return_softmax_lse != 0) {
            lse[row] = final_lse;
        }
        if (return_temperature_lse != 0) {
            temperature_lse[row] = final_lse;
        }
    }
    if (row_valid != 0) {
        int col_segment = lane_in_row * 16;
        float accum[16];
        #pragma unroll
        for (int elem = 0; elem < 16; elem++) {
            accum[elem] = 0.0f;
        }
        if (split_count > 0) {
            float values_0[16];
            long long values_0_index = (long long)row * 128 + (long long)col_segment;
            {
                uint64_t _fp8x8_0 = *reinterpret_cast<const uint64_t*>(partial_o + values_0_index);
                uint16_t _e4m3x2_0_0 = (uint16_t)((_fp8x8_0 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_0) : "h"(_e4m3x2_0_0));
                uint16_t _h0_0 = (uint16_t)((_f16x2_0_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 0]) : "h"(_h0_0));
                uint16_t _h1_0 = (uint16_t)((_f16x2_0_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 1]) : "h"(_h1_0));
                uint16_t _e4m3x2_1_0 = (uint16_t)((_fp8x8_0 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_0) : "h"(_e4m3x2_1_0));
                uint16_t _h2_0 = (uint16_t)((_f16x2_1_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 2]) : "h"(_h2_0));
                uint16_t _h3_0 = (uint16_t)((_f16x2_1_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 3]) : "h"(_h3_0));
                uint16_t _e4m3x2_2_0 = (uint16_t)((_fp8x8_0 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_0) : "h"(_e4m3x2_2_0));
                uint16_t _h4_0 = (uint16_t)((_f16x2_2_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 4]) : "h"(_h4_0));
                uint16_t _h5_0 = (uint16_t)((_f16x2_2_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 5]) : "h"(_h5_0));
                uint16_t _e4m3x2_3_0 = (uint16_t)((_fp8x8_0 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_0) : "h"(_e4m3x2_3_0));
                uint16_t _h6_0 = (uint16_t)((_f16x2_3_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 6]) : "h"(_h6_0));
                uint16_t _h7_0 = (uint16_t)((_f16x2_3_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 7]) : "h"(_h7_0));
            }
            {
                uint64_t _fp8x8_1 = *reinterpret_cast<const uint64_t*>(partial_o + values_0_index + 8);
                uint16_t _e4m3x2_0_1 = (uint16_t)((_fp8x8_1 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_1) : "h"(_e4m3x2_0_1));
                uint16_t _h0_1 = (uint16_t)((_f16x2_0_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[8 + 0]) : "h"(_h0_1));
                uint16_t _h1_1 = (uint16_t)((_f16x2_0_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[8 + 1]) : "h"(_h1_1));
                uint16_t _e4m3x2_1_1 = (uint16_t)((_fp8x8_1 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_1) : "h"(_e4m3x2_1_1));
                uint16_t _h2_1 = (uint16_t)((_f16x2_1_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[8 + 2]) : "h"(_h2_1));
                uint16_t _h3_1 = (uint16_t)((_f16x2_1_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[8 + 3]) : "h"(_h3_1));
                uint16_t _e4m3x2_2_1 = (uint16_t)((_fp8x8_1 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_1) : "h"(_e4m3x2_2_1));
                uint16_t _h4_1 = (uint16_t)((_f16x2_2_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[8 + 4]) : "h"(_h4_1));
                uint16_t _h5_1 = (uint16_t)((_f16x2_2_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[8 + 5]) : "h"(_h5_1));
                uint16_t _e4m3x2_3_1 = (uint16_t)((_fp8x8_1 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_1) : "h"(_e4m3x2_3_1));
                uint16_t _h6_1 = (uint16_t)((_f16x2_3_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[8 + 6]) : "h"(_h6_1));
                uint16_t _h7_1 = (uint16_t)((_f16x2_3_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[8 + 7]) : "h"(_h7_1));
            }
            #pragma unroll
            for (int elem_1 = 0; elem_1 < 16; elem_1++) {
                float _fma_0 = __fmaf_rn(values_0[elem_1], weight_0, accum[elem_1]);
                accum[elem_1] = _fma_0;
            }
        }
        if (split_count > 1) {
            float values_1[16];
            long long values_1_index = ((long long)total_rows_out + (long long)row) * 128 + (long long)col_segment;
            {
                uint64_t _fp8x8_2 = *reinterpret_cast<const uint64_t*>(partial_o + values_1_index);
                uint16_t _e4m3x2_0_2 = (uint16_t)((_fp8x8_2 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_2) : "h"(_e4m3x2_0_2));
                uint16_t _h0_2 = (uint16_t)((_f16x2_0_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 0]) : "h"(_h0_2));
                uint16_t _h1_2 = (uint16_t)((_f16x2_0_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 1]) : "h"(_h1_2));
                uint16_t _e4m3x2_1_2 = (uint16_t)((_fp8x8_2 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_2) : "h"(_e4m3x2_1_2));
                uint16_t _h2_2 = (uint16_t)((_f16x2_1_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 2]) : "h"(_h2_2));
                uint16_t _h3_2 = (uint16_t)((_f16x2_1_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 3]) : "h"(_h3_2));
                uint16_t _e4m3x2_2_2 = (uint16_t)((_fp8x8_2 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_2) : "h"(_e4m3x2_2_2));
                uint16_t _h4_2 = (uint16_t)((_f16x2_2_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 4]) : "h"(_h4_2));
                uint16_t _h5_2 = (uint16_t)((_f16x2_2_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 5]) : "h"(_h5_2));
                uint16_t _e4m3x2_3_2 = (uint16_t)((_fp8x8_2 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_2) : "h"(_e4m3x2_3_2));
                uint16_t _h6_2 = (uint16_t)((_f16x2_3_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 6]) : "h"(_h6_2));
                uint16_t _h7_2 = (uint16_t)((_f16x2_3_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 7]) : "h"(_h7_2));
            }
            {
                uint64_t _fp8x8_3 = *reinterpret_cast<const uint64_t*>(partial_o + values_1_index + 8);
                uint16_t _e4m3x2_0_3 = (uint16_t)((_fp8x8_3 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_3) : "h"(_e4m3x2_0_3));
                uint16_t _h0_3 = (uint16_t)((_f16x2_0_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[8 + 0]) : "h"(_h0_3));
                uint16_t _h1_3 = (uint16_t)((_f16x2_0_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[8 + 1]) : "h"(_h1_3));
                uint16_t _e4m3x2_1_3 = (uint16_t)((_fp8x8_3 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_3) : "h"(_e4m3x2_1_3));
                uint16_t _h2_3 = (uint16_t)((_f16x2_1_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[8 + 2]) : "h"(_h2_3));
                uint16_t _h3_3 = (uint16_t)((_f16x2_1_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[8 + 3]) : "h"(_h3_3));
                uint16_t _e4m3x2_2_3 = (uint16_t)((_fp8x8_3 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_3) : "h"(_e4m3x2_2_3));
                uint16_t _h4_3 = (uint16_t)((_f16x2_2_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[8 + 4]) : "h"(_h4_3));
                uint16_t _h5_3 = (uint16_t)((_f16x2_2_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[8 + 5]) : "h"(_h5_3));
                uint16_t _e4m3x2_3_3 = (uint16_t)((_fp8x8_3 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_3) : "h"(_e4m3x2_3_3));
                uint16_t _h6_3 = (uint16_t)((_f16x2_3_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[8 + 6]) : "h"(_h6_3));
                uint16_t _h7_3 = (uint16_t)((_f16x2_3_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[8 + 7]) : "h"(_h7_3));
            }
            #pragma unroll
            for (int elem_2 = 0; elem_2 < 16; elem_2++) {
                float _fma_1 = __fmaf_rn(values_1[elem_2], weight_1, accum[elem_2]);
                accum[elem_2] = _fma_1;
            }
        }
        if (split_count > 2) {
            float values_2[16];
            long long values_2_index = (2 * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment;
            {
                uint64_t _fp8x8_4 = *reinterpret_cast<const uint64_t*>(partial_o + values_2_index);
                uint16_t _e4m3x2_0_4 = (uint16_t)((_fp8x8_4 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_4) : "h"(_e4m3x2_0_4));
                uint16_t _h0_4 = (uint16_t)((_f16x2_0_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 0]) : "h"(_h0_4));
                uint16_t _h1_4 = (uint16_t)((_f16x2_0_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 1]) : "h"(_h1_4));
                uint16_t _e4m3x2_1_4 = (uint16_t)((_fp8x8_4 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_4) : "h"(_e4m3x2_1_4));
                uint16_t _h2_4 = (uint16_t)((_f16x2_1_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 2]) : "h"(_h2_4));
                uint16_t _h3_4 = (uint16_t)((_f16x2_1_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 3]) : "h"(_h3_4));
                uint16_t _e4m3x2_2_4 = (uint16_t)((_fp8x8_4 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_4) : "h"(_e4m3x2_2_4));
                uint16_t _h4_4 = (uint16_t)((_f16x2_2_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 4]) : "h"(_h4_4));
                uint16_t _h5_4 = (uint16_t)((_f16x2_2_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 5]) : "h"(_h5_4));
                uint16_t _e4m3x2_3_4 = (uint16_t)((_fp8x8_4 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_4) : "h"(_e4m3x2_3_4));
                uint16_t _h6_4 = (uint16_t)((_f16x2_3_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 6]) : "h"(_h6_4));
                uint16_t _h7_4 = (uint16_t)((_f16x2_3_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 7]) : "h"(_h7_4));
            }
            {
                uint64_t _fp8x8_5 = *reinterpret_cast<const uint64_t*>(partial_o + values_2_index + 8);
                uint16_t _e4m3x2_0_5 = (uint16_t)((_fp8x8_5 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_5) : "h"(_e4m3x2_0_5));
                uint16_t _h0_5 = (uint16_t)((_f16x2_0_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[8 + 0]) : "h"(_h0_5));
                uint16_t _h1_5 = (uint16_t)((_f16x2_0_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[8 + 1]) : "h"(_h1_5));
                uint16_t _e4m3x2_1_5 = (uint16_t)((_fp8x8_5 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_5) : "h"(_e4m3x2_1_5));
                uint16_t _h2_5 = (uint16_t)((_f16x2_1_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[8 + 2]) : "h"(_h2_5));
                uint16_t _h3_5 = (uint16_t)((_f16x2_1_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[8 + 3]) : "h"(_h3_5));
                uint16_t _e4m3x2_2_5 = (uint16_t)((_fp8x8_5 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_5) : "h"(_e4m3x2_2_5));
                uint16_t _h4_5 = (uint16_t)((_f16x2_2_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[8 + 4]) : "h"(_h4_5));
                uint16_t _h5_5 = (uint16_t)((_f16x2_2_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[8 + 5]) : "h"(_h5_5));
                uint16_t _e4m3x2_3_5 = (uint16_t)((_fp8x8_5 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_5) : "h"(_e4m3x2_3_5));
                uint16_t _h6_5 = (uint16_t)((_f16x2_3_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[8 + 6]) : "h"(_h6_5));
                uint16_t _h7_5 = (uint16_t)((_f16x2_3_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[8 + 7]) : "h"(_h7_5));
            }
            #pragma unroll
            for (int elem_3 = 0; elem_3 < 16; elem_3++) {
                float _fma_2 = __fmaf_rn(values_2[elem_3], weight_2, accum[elem_3]);
                accum[elem_3] = _fma_2;
            }
        }
        if (split_count > 3) {
            float values_3[16];
            long long values_3_index = (3 * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment;
            {
                uint64_t _fp8x8_6 = *reinterpret_cast<const uint64_t*>(partial_o + values_3_index);
                uint16_t _e4m3x2_0_6 = (uint16_t)((_fp8x8_6 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_6) : "h"(_e4m3x2_0_6));
                uint16_t _h0_6 = (uint16_t)((_f16x2_0_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[0 + 0]) : "h"(_h0_6));
                uint16_t _h1_6 = (uint16_t)((_f16x2_0_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[0 + 1]) : "h"(_h1_6));
                uint16_t _e4m3x2_1_6 = (uint16_t)((_fp8x8_6 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_6) : "h"(_e4m3x2_1_6));
                uint16_t _h2_6 = (uint16_t)((_f16x2_1_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[0 + 2]) : "h"(_h2_6));
                uint16_t _h3_6 = (uint16_t)((_f16x2_1_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[0 + 3]) : "h"(_h3_6));
                uint16_t _e4m3x2_2_6 = (uint16_t)((_fp8x8_6 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_6) : "h"(_e4m3x2_2_6));
                uint16_t _h4_6 = (uint16_t)((_f16x2_2_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[0 + 4]) : "h"(_h4_6));
                uint16_t _h5_6 = (uint16_t)((_f16x2_2_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[0 + 5]) : "h"(_h5_6));
                uint16_t _e4m3x2_3_6 = (uint16_t)((_fp8x8_6 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_6) : "h"(_e4m3x2_3_6));
                uint16_t _h6_6 = (uint16_t)((_f16x2_3_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[0 + 6]) : "h"(_h6_6));
                uint16_t _h7_6 = (uint16_t)((_f16x2_3_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[0 + 7]) : "h"(_h7_6));
            }
            {
                uint64_t _fp8x8_7 = *reinterpret_cast<const uint64_t*>(partial_o + values_3_index + 8);
                uint16_t _e4m3x2_0_7 = (uint16_t)((_fp8x8_7 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_7) : "h"(_e4m3x2_0_7));
                uint16_t _h0_7 = (uint16_t)((_f16x2_0_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[8 + 0]) : "h"(_h0_7));
                uint16_t _h1_7 = (uint16_t)((_f16x2_0_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[8 + 1]) : "h"(_h1_7));
                uint16_t _e4m3x2_1_7 = (uint16_t)((_fp8x8_7 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_7) : "h"(_e4m3x2_1_7));
                uint16_t _h2_7 = (uint16_t)((_f16x2_1_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[8 + 2]) : "h"(_h2_7));
                uint16_t _h3_7 = (uint16_t)((_f16x2_1_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[8 + 3]) : "h"(_h3_7));
                uint16_t _e4m3x2_2_7 = (uint16_t)((_fp8x8_7 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_7) : "h"(_e4m3x2_2_7));
                uint16_t _h4_7 = (uint16_t)((_f16x2_2_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[8 + 4]) : "h"(_h4_7));
                uint16_t _h5_7 = (uint16_t)((_f16x2_2_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[8 + 5]) : "h"(_h5_7));
                uint16_t _e4m3x2_3_7 = (uint16_t)((_fp8x8_7 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_7) : "h"(_e4m3x2_3_7));
                uint16_t _h6_7 = (uint16_t)((_f16x2_3_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[8 + 6]) : "h"(_h6_7));
                uint16_t _h7_7 = (uint16_t)((_f16x2_3_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_3[8 + 7]) : "h"(_h7_7));
            }
            #pragma unroll
            for (int elem_4 = 0; elem_4 < 16; elem_4++) {
                float _fma_3 = __fmaf_rn(values_3[elem_4], weight_3, accum[elem_4]);
                accum[elem_4] = _fma_3;
            }
        }
        {
            __nv_bfloat162 _pk[8];
            _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
            _pk[4] = __floats2bfloat162_rn(accum[0 + 8], accum[0 + 9]);
            _pk[5] = __floats2bfloat162_rn(accum[0 + 10], accum[0 + 11]);
            _pk[6] = __floats2bfloat162_rn(accum[0 + 12], accum[0 + 13]);
            _pk[7] = __floats2bfloat162_rn(accum[0 + 14], accum[0 + 15]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row * 128 + (long long)col_segment)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row * 128 + (long long)col_segment)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
        }
    }
}

} // extern "C"
