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
kernel_minimax_sparse_reverse_prefill_combine_unscaled_fp8_qagg_pdl_dualcohort_sm100(uint8_t* __restrict__ partial_o, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ split_counts, int* __restrict__ q_order, int* __restrict__ contributor_work_ids, unsigned int* __restrict__ completion_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int total_q, int num_q_heads, int num_kv_heads, int qhead_per_kv, int topk, unsigned int generation, int return_softmax_lse, int return_temperature_lse)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // ---- Role: lower ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // lower_main
            int q_position = blockIdx.x;
            int q_abs = q_order[q_position];
            int warp_id_in_role = (warp - 0);
            int role_tid = warp_id_in_role * 32 + lane;
            int row_group = role_tid / 8;
            int lane_in_row = role_tid & 7;
            int leader_lane = lane / 8 * 8;
            int cohort_linear = q_abs * num_kv_heads;
            if (role_tid < 8) {
                int contributor_slot = role_tid;
                int work_idx = contributor_work_ids[cohort_linear * 8 + contributor_slot];
                {
                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(completion_counts) + (work_idx);
                    while (true) {
                        unsigned int _gca_v;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                        if (_gca_v >= (unsigned int)(generation)) break;
                    }
                }
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            int row = q_abs * num_q_heads + row_group;
            int total_rows_out = total_q * num_q_heads;
            long long split_row = (long long)lane_in_row * (long long)total_rows_out + (long long)row;
            float lane_lse = partial_lse[split_row];
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
            float _exp2_0 = approx_exp2((lane_lse - safe_lse_max) * 1.4426950408889634f);
            float lane_weight = _exp2_0;
            if (lane_lse == -BLACKWELL_MSA_INF) {
                lane_weight = 0.0f;
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
            if (lane_in_row == 0) {
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
            int col_segment = lane_in_row * 16;
            float accum[16];
            #pragma unroll
            for (int elem = 0; elem < 16; elem++) {
                accum[elem] = 0.0f;
            }
            float values[16];
            {
                unsigned _fp8x16_0_0;
                unsigned _fp8x16_0_1;
                unsigned _fp8x16_0_2;
                unsigned _fp8x16_0_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_0_0), "=r"(_fp8x16_0_1), "=r"(_fp8x16_0_2), "=r"(_fp8x16_0_3) : "l"((const void*)(partial_o + ((long long)row * 128 + (long long)col_segment))) : "memory");
                uint16_t _e4m3x2_0_0 = (uint16_t)((_fp8x16_0_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_0) : "h"(_e4m3x2_0_0));
                uint16_t _h0_0 = (uint16_t)((_f16x2_0_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 0]) : "h"(_h0_0));
                uint16_t _h1_0 = (uint16_t)((_f16x2_0_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 1]) : "h"(_h1_0));
                uint16_t _e4m3x2_1_0 = (uint16_t)((_fp8x16_0_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_0) : "h"(_e4m3x2_1_0));
                uint16_t _h2_0 = (uint16_t)((_f16x2_1_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 2]) : "h"(_h2_0));
                uint16_t _h3_0 = (uint16_t)((_f16x2_1_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 3]) : "h"(_h3_0));
                uint16_t _e4m3x2_2_0 = (uint16_t)((_fp8x16_0_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_0) : "h"(_e4m3x2_2_0));
                uint16_t _h4_0 = (uint16_t)((_f16x2_2_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 4]) : "h"(_h4_0));
                uint16_t _h5_0 = (uint16_t)((_f16x2_2_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 5]) : "h"(_h5_0));
                uint16_t _e4m3x2_3_0 = (uint16_t)((_fp8x16_0_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_0) : "h"(_e4m3x2_3_0));
                uint16_t _h6_0 = (uint16_t)((_f16x2_3_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 6]) : "h"(_h6_0));
                uint16_t _h7_0 = (uint16_t)((_f16x2_3_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 7]) : "h"(_h7_0));
                uint16_t _e4m3x2_4_0 = (uint16_t)((_fp8x16_0_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_0) : "h"(_e4m3x2_4_0));
                uint16_t _h8_0 = (uint16_t)((_f16x2_4_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 8]) : "h"(_h8_0));
                uint16_t _h9_0 = (uint16_t)((_f16x2_4_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 9]) : "h"(_h9_0));
                uint16_t _e4m3x2_5_0 = (uint16_t)((_fp8x16_0_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_0) : "h"(_e4m3x2_5_0));
                uint16_t _h10_0 = (uint16_t)((_f16x2_5_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 10]) : "h"(_h10_0));
                uint16_t _h11_0 = (uint16_t)((_f16x2_5_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 11]) : "h"(_h11_0));
                uint16_t _e4m3x2_6_0 = (uint16_t)((_fp8x16_0_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_0) : "h"(_e4m3x2_6_0));
                uint16_t _h12_0 = (uint16_t)((_f16x2_6_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 12]) : "h"(_h12_0));
                uint16_t _h13_0 = (uint16_t)((_f16x2_6_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 13]) : "h"(_h13_0));
                uint16_t _e4m3x2_7_0 = (uint16_t)((_fp8x16_0_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_0) : "h"(_e4m3x2_7_0));
                uint16_t _h14_0 = (uint16_t)((_f16x2_7_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 14]) : "h"(_h14_0));
                uint16_t _h15_0 = (uint16_t)((_f16x2_7_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 15]) : "h"(_h15_0));
            }
            float _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane);
            float weight = _shfl_0;
            #pragma unroll
            for (int elem_1 = 0; elem_1 < 16; elem_1++) {
                float _fma_0 = __fmaf_rn(values[elem_1], weight, accum[elem_1]);
                accum[elem_1] = _fma_0;
            }
            float values_0[16];
            {
                unsigned _fp8x16_1_0;
                unsigned _fp8x16_1_1;
                unsigned _fp8x16_1_2;
                unsigned _fp8x16_1_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_1_0), "=r"(_fp8x16_1_1), "=r"(_fp8x16_1_2), "=r"(_fp8x16_1_3) : "l"((const void*)(partial_o + (((long long)total_rows_out + (long long)row) * 128 + (long long)col_segment))) : "memory");
                uint16_t _e4m3x2_0_1 = (uint16_t)((_fp8x16_1_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_1) : "h"(_e4m3x2_0_1));
                uint16_t _h0_1 = (uint16_t)((_f16x2_0_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 0]) : "h"(_h0_1));
                uint16_t _h1_1 = (uint16_t)((_f16x2_0_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 1]) : "h"(_h1_1));
                uint16_t _e4m3x2_1_1 = (uint16_t)((_fp8x16_1_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_1) : "h"(_e4m3x2_1_1));
                uint16_t _h2_1 = (uint16_t)((_f16x2_1_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 2]) : "h"(_h2_1));
                uint16_t _h3_1 = (uint16_t)((_f16x2_1_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 3]) : "h"(_h3_1));
                uint16_t _e4m3x2_2_1 = (uint16_t)((_fp8x16_1_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_1) : "h"(_e4m3x2_2_1));
                uint16_t _h4_1 = (uint16_t)((_f16x2_2_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 4]) : "h"(_h4_1));
                uint16_t _h5_1 = (uint16_t)((_f16x2_2_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 5]) : "h"(_h5_1));
                uint16_t _e4m3x2_3_1 = (uint16_t)((_fp8x16_1_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_1) : "h"(_e4m3x2_3_1));
                uint16_t _h6_1 = (uint16_t)((_f16x2_3_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 6]) : "h"(_h6_1));
                uint16_t _h7_1 = (uint16_t)((_f16x2_3_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 7]) : "h"(_h7_1));
                uint16_t _e4m3x2_4_1 = (uint16_t)((_fp8x16_1_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_1) : "h"(_e4m3x2_4_1));
                uint16_t _h8_1 = (uint16_t)((_f16x2_4_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 8]) : "h"(_h8_1));
                uint16_t _h9_1 = (uint16_t)((_f16x2_4_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 9]) : "h"(_h9_1));
                uint16_t _e4m3x2_5_1 = (uint16_t)((_fp8x16_1_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_1) : "h"(_e4m3x2_5_1));
                uint16_t _h10_1 = (uint16_t)((_f16x2_5_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 10]) : "h"(_h10_1));
                uint16_t _h11_1 = (uint16_t)((_f16x2_5_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 11]) : "h"(_h11_1));
                uint16_t _e4m3x2_6_1 = (uint16_t)((_fp8x16_1_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_1) : "h"(_e4m3x2_6_1));
                uint16_t _h12_1 = (uint16_t)((_f16x2_6_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 12]) : "h"(_h12_1));
                uint16_t _h13_1 = (uint16_t)((_f16x2_6_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 13]) : "h"(_h13_1));
                uint16_t _e4m3x2_7_1 = (uint16_t)((_fp8x16_1_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_1) : "h"(_e4m3x2_7_1));
                uint16_t _h14_1 = (uint16_t)((_f16x2_7_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 14]) : "h"(_h14_1));
                uint16_t _h15_1 = (uint16_t)((_f16x2_7_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0[0 + 15]) : "h"(_h15_1));
            }
            float _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 1);
            float weight_1 = _shfl_1;
            #pragma unroll
            for (int elem_2 = 0; elem_2 < 16; elem_2++) {
                float _fma_1 = __fmaf_rn(values_0[elem_2], weight_1, accum[elem_2]);
                accum[elem_2] = _fma_1;
            }
            float values_2[16];
            {
                unsigned _fp8x16_2_0;
                unsigned _fp8x16_2_1;
                unsigned _fp8x16_2_2;
                unsigned _fp8x16_2_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_2_0), "=r"(_fp8x16_2_1), "=r"(_fp8x16_2_2), "=r"(_fp8x16_2_3) : "l"((const void*)(partial_o + ((2 * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment))) : "memory");
                uint16_t _e4m3x2_0_2 = (uint16_t)((_fp8x16_2_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_2) : "h"(_e4m3x2_0_2));
                uint16_t _h0_2 = (uint16_t)((_f16x2_0_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 0]) : "h"(_h0_2));
                uint16_t _h1_2 = (uint16_t)((_f16x2_0_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 1]) : "h"(_h1_2));
                uint16_t _e4m3x2_1_2 = (uint16_t)((_fp8x16_2_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_2) : "h"(_e4m3x2_1_2));
                uint16_t _h2_2 = (uint16_t)((_f16x2_1_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 2]) : "h"(_h2_2));
                uint16_t _h3_2 = (uint16_t)((_f16x2_1_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 3]) : "h"(_h3_2));
                uint16_t _e4m3x2_2_2 = (uint16_t)((_fp8x16_2_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_2) : "h"(_e4m3x2_2_2));
                uint16_t _h4_2 = (uint16_t)((_f16x2_2_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 4]) : "h"(_h4_2));
                uint16_t _h5_2 = (uint16_t)((_f16x2_2_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 5]) : "h"(_h5_2));
                uint16_t _e4m3x2_3_2 = (uint16_t)((_fp8x16_2_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_2) : "h"(_e4m3x2_3_2));
                uint16_t _h6_2 = (uint16_t)((_f16x2_3_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 6]) : "h"(_h6_2));
                uint16_t _h7_2 = (uint16_t)((_f16x2_3_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 7]) : "h"(_h7_2));
                uint16_t _e4m3x2_4_2 = (uint16_t)((_fp8x16_2_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_2) : "h"(_e4m3x2_4_2));
                uint16_t _h8_2 = (uint16_t)((_f16x2_4_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 8]) : "h"(_h8_2));
                uint16_t _h9_2 = (uint16_t)((_f16x2_4_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 9]) : "h"(_h9_2));
                uint16_t _e4m3x2_5_2 = (uint16_t)((_fp8x16_2_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_2) : "h"(_e4m3x2_5_2));
                uint16_t _h10_2 = (uint16_t)((_f16x2_5_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 10]) : "h"(_h10_2));
                uint16_t _h11_2 = (uint16_t)((_f16x2_5_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 11]) : "h"(_h11_2));
                uint16_t _e4m3x2_6_2 = (uint16_t)((_fp8x16_2_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_2) : "h"(_e4m3x2_6_2));
                uint16_t _h12_2 = (uint16_t)((_f16x2_6_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 12]) : "h"(_h12_2));
                uint16_t _h13_2 = (uint16_t)((_f16x2_6_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 13]) : "h"(_h13_2));
                uint16_t _e4m3x2_7_2 = (uint16_t)((_fp8x16_2_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_2) : "h"(_e4m3x2_7_2));
                uint16_t _h14_2 = (uint16_t)((_f16x2_7_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 14]) : "h"(_h14_2));
                uint16_t _h15_2 = (uint16_t)((_f16x2_7_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2[0 + 15]) : "h"(_h15_2));
            }
            float _shfl_2 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 2);
            float weight_3 = _shfl_2;
            #pragma unroll
            for (int elem_3 = 0; elem_3 < 16; elem_3++) {
                float _fma_2 = __fmaf_rn(values_2[elem_3], weight_3, accum[elem_3]);
                accum[elem_3] = _fma_2;
            }
            float values_4[16];
            {
                unsigned _fp8x16_3_0;
                unsigned _fp8x16_3_1;
                unsigned _fp8x16_3_2;
                unsigned _fp8x16_3_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_3_0), "=r"(_fp8x16_3_1), "=r"(_fp8x16_3_2), "=r"(_fp8x16_3_3) : "l"((const void*)(partial_o + ((3 * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment))) : "memory");
                uint16_t _e4m3x2_0_3 = (uint16_t)((_fp8x16_3_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_3) : "h"(_e4m3x2_0_3));
                uint16_t _h0_3 = (uint16_t)((_f16x2_0_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 0]) : "h"(_h0_3));
                uint16_t _h1_3 = (uint16_t)((_f16x2_0_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 1]) : "h"(_h1_3));
                uint16_t _e4m3x2_1_3 = (uint16_t)((_fp8x16_3_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_3) : "h"(_e4m3x2_1_3));
                uint16_t _h2_3 = (uint16_t)((_f16x2_1_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 2]) : "h"(_h2_3));
                uint16_t _h3_3 = (uint16_t)((_f16x2_1_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 3]) : "h"(_h3_3));
                uint16_t _e4m3x2_2_3 = (uint16_t)((_fp8x16_3_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_3) : "h"(_e4m3x2_2_3));
                uint16_t _h4_3 = (uint16_t)((_f16x2_2_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 4]) : "h"(_h4_3));
                uint16_t _h5_3 = (uint16_t)((_f16x2_2_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 5]) : "h"(_h5_3));
                uint16_t _e4m3x2_3_3 = (uint16_t)((_fp8x16_3_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_3) : "h"(_e4m3x2_3_3));
                uint16_t _h6_3 = (uint16_t)((_f16x2_3_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 6]) : "h"(_h6_3));
                uint16_t _h7_3 = (uint16_t)((_f16x2_3_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 7]) : "h"(_h7_3));
                uint16_t _e4m3x2_4_3 = (uint16_t)((_fp8x16_3_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_3) : "h"(_e4m3x2_4_3));
                uint16_t _h8_3 = (uint16_t)((_f16x2_4_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 8]) : "h"(_h8_3));
                uint16_t _h9_3 = (uint16_t)((_f16x2_4_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 9]) : "h"(_h9_3));
                uint16_t _e4m3x2_5_3 = (uint16_t)((_fp8x16_3_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_3) : "h"(_e4m3x2_5_3));
                uint16_t _h10_3 = (uint16_t)((_f16x2_5_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 10]) : "h"(_h10_3));
                uint16_t _h11_3 = (uint16_t)((_f16x2_5_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 11]) : "h"(_h11_3));
                uint16_t _e4m3x2_6_3 = (uint16_t)((_fp8x16_3_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_3) : "h"(_e4m3x2_6_3));
                uint16_t _h12_3 = (uint16_t)((_f16x2_6_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 12]) : "h"(_h12_3));
                uint16_t _h13_3 = (uint16_t)((_f16x2_6_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 13]) : "h"(_h13_3));
                uint16_t _e4m3x2_7_3 = (uint16_t)((_fp8x16_3_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_3) : "h"(_e4m3x2_7_3));
                uint16_t _h14_3 = (uint16_t)((_f16x2_7_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 14]) : "h"(_h14_3));
                uint16_t _h15_3 = (uint16_t)((_f16x2_7_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4[0 + 15]) : "h"(_h15_3));
            }
            float _shfl_3 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 3);
            float weight_5 = _shfl_3;
            #pragma unroll
            for (int elem_4 = 0; elem_4 < 16; elem_4++) {
                float _fma_3 = __fmaf_rn(values_4[elem_4], weight_5, accum[elem_4]);
                accum[elem_4] = _fma_3;
            }
            float values_6[16];
            {
                unsigned _fp8x16_4_0;
                unsigned _fp8x16_4_1;
                unsigned _fp8x16_4_2;
                unsigned _fp8x16_4_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_4_0), "=r"(_fp8x16_4_1), "=r"(_fp8x16_4_2), "=r"(_fp8x16_4_3) : "l"((const void*)(partial_o + ((4 * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment))) : "memory");
                uint16_t _e4m3x2_0_4 = (uint16_t)((_fp8x16_4_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_4) : "h"(_e4m3x2_0_4));
                uint16_t _h0_4 = (uint16_t)((_f16x2_0_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 0]) : "h"(_h0_4));
                uint16_t _h1_4 = (uint16_t)((_f16x2_0_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 1]) : "h"(_h1_4));
                uint16_t _e4m3x2_1_4 = (uint16_t)((_fp8x16_4_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_4) : "h"(_e4m3x2_1_4));
                uint16_t _h2_4 = (uint16_t)((_f16x2_1_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 2]) : "h"(_h2_4));
                uint16_t _h3_4 = (uint16_t)((_f16x2_1_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 3]) : "h"(_h3_4));
                uint16_t _e4m3x2_2_4 = (uint16_t)((_fp8x16_4_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_4) : "h"(_e4m3x2_2_4));
                uint16_t _h4_4 = (uint16_t)((_f16x2_2_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 4]) : "h"(_h4_4));
                uint16_t _h5_4 = (uint16_t)((_f16x2_2_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 5]) : "h"(_h5_4));
                uint16_t _e4m3x2_3_4 = (uint16_t)((_fp8x16_4_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_4) : "h"(_e4m3x2_3_4));
                uint16_t _h6_4 = (uint16_t)((_f16x2_3_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 6]) : "h"(_h6_4));
                uint16_t _h7_4 = (uint16_t)((_f16x2_3_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 7]) : "h"(_h7_4));
                uint16_t _e4m3x2_4_4 = (uint16_t)((_fp8x16_4_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_4) : "h"(_e4m3x2_4_4));
                uint16_t _h8_4 = (uint16_t)((_f16x2_4_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 8]) : "h"(_h8_4));
                uint16_t _h9_4 = (uint16_t)((_f16x2_4_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 9]) : "h"(_h9_4));
                uint16_t _e4m3x2_5_4 = (uint16_t)((_fp8x16_4_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_4) : "h"(_e4m3x2_5_4));
                uint16_t _h10_4 = (uint16_t)((_f16x2_5_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 10]) : "h"(_h10_4));
                uint16_t _h11_4 = (uint16_t)((_f16x2_5_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 11]) : "h"(_h11_4));
                uint16_t _e4m3x2_6_4 = (uint16_t)((_fp8x16_4_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_4) : "h"(_e4m3x2_6_4));
                uint16_t _h12_4 = (uint16_t)((_f16x2_6_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 12]) : "h"(_h12_4));
                uint16_t _h13_4 = (uint16_t)((_f16x2_6_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 13]) : "h"(_h13_4));
                uint16_t _e4m3x2_7_4 = (uint16_t)((_fp8x16_4_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_4) : "h"(_e4m3x2_7_4));
                uint16_t _h14_4 = (uint16_t)((_f16x2_7_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 14]) : "h"(_h14_4));
                uint16_t _h15_4 = (uint16_t)((_f16x2_7_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6[0 + 15]) : "h"(_h15_4));
            }
            float _shfl_4 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 4);
            float weight_7 = _shfl_4;
            #pragma unroll
            for (int elem_5 = 0; elem_5 < 16; elem_5++) {
                float _fma_4 = __fmaf_rn(values_6[elem_5], weight_7, accum[elem_5]);
                accum[elem_5] = _fma_4;
            }
            float values_8[16];
            {
                unsigned _fp8x16_5_0;
                unsigned _fp8x16_5_1;
                unsigned _fp8x16_5_2;
                unsigned _fp8x16_5_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_5_0), "=r"(_fp8x16_5_1), "=r"(_fp8x16_5_2), "=r"(_fp8x16_5_3) : "l"((const void*)(partial_o + ((5 * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment))) : "memory");
                uint16_t _e4m3x2_0_5 = (uint16_t)((_fp8x16_5_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_5) : "h"(_e4m3x2_0_5));
                uint16_t _h0_5 = (uint16_t)((_f16x2_0_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 0]) : "h"(_h0_5));
                uint16_t _h1_5 = (uint16_t)((_f16x2_0_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 1]) : "h"(_h1_5));
                uint16_t _e4m3x2_1_5 = (uint16_t)((_fp8x16_5_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_5) : "h"(_e4m3x2_1_5));
                uint16_t _h2_5 = (uint16_t)((_f16x2_1_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 2]) : "h"(_h2_5));
                uint16_t _h3_5 = (uint16_t)((_f16x2_1_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 3]) : "h"(_h3_5));
                uint16_t _e4m3x2_2_5 = (uint16_t)((_fp8x16_5_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_5) : "h"(_e4m3x2_2_5));
                uint16_t _h4_5 = (uint16_t)((_f16x2_2_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 4]) : "h"(_h4_5));
                uint16_t _h5_5 = (uint16_t)((_f16x2_2_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 5]) : "h"(_h5_5));
                uint16_t _e4m3x2_3_5 = (uint16_t)((_fp8x16_5_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_5) : "h"(_e4m3x2_3_5));
                uint16_t _h6_5 = (uint16_t)((_f16x2_3_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 6]) : "h"(_h6_5));
                uint16_t _h7_5 = (uint16_t)((_f16x2_3_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 7]) : "h"(_h7_5));
                uint16_t _e4m3x2_4_5 = (uint16_t)((_fp8x16_5_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_5) : "h"(_e4m3x2_4_5));
                uint16_t _h8_5 = (uint16_t)((_f16x2_4_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 8]) : "h"(_h8_5));
                uint16_t _h9_5 = (uint16_t)((_f16x2_4_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 9]) : "h"(_h9_5));
                uint16_t _e4m3x2_5_5 = (uint16_t)((_fp8x16_5_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_5) : "h"(_e4m3x2_5_5));
                uint16_t _h10_5 = (uint16_t)((_f16x2_5_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 10]) : "h"(_h10_5));
                uint16_t _h11_5 = (uint16_t)((_f16x2_5_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 11]) : "h"(_h11_5));
                uint16_t _e4m3x2_6_5 = (uint16_t)((_fp8x16_5_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_5) : "h"(_e4m3x2_6_5));
                uint16_t _h12_5 = (uint16_t)((_f16x2_6_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 12]) : "h"(_h12_5));
                uint16_t _h13_5 = (uint16_t)((_f16x2_6_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 13]) : "h"(_h13_5));
                uint16_t _e4m3x2_7_5 = (uint16_t)((_fp8x16_5_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_5) : "h"(_e4m3x2_7_5));
                uint16_t _h14_5 = (uint16_t)((_f16x2_7_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 14]) : "h"(_h14_5));
                uint16_t _h15_5 = (uint16_t)((_f16x2_7_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8[0 + 15]) : "h"(_h15_5));
            }
            float _shfl_5 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 5);
            float weight_9 = _shfl_5;
            #pragma unroll
            for (int elem_6 = 0; elem_6 < 16; elem_6++) {
                float _fma_5 = __fmaf_rn(values_8[elem_6], weight_9, accum[elem_6]);
                accum[elem_6] = _fma_5;
            }
            float values_10[16];
            {
                unsigned _fp8x16_6_0;
                unsigned _fp8x16_6_1;
                unsigned _fp8x16_6_2;
                unsigned _fp8x16_6_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_6_0), "=r"(_fp8x16_6_1), "=r"(_fp8x16_6_2), "=r"(_fp8x16_6_3) : "l"((const void*)(partial_o + ((6 * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment))) : "memory");
                uint16_t _e4m3x2_0_6 = (uint16_t)((_fp8x16_6_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_6) : "h"(_e4m3x2_0_6));
                uint16_t _h0_6 = (uint16_t)((_f16x2_0_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 0]) : "h"(_h0_6));
                uint16_t _h1_6 = (uint16_t)((_f16x2_0_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 1]) : "h"(_h1_6));
                uint16_t _e4m3x2_1_6 = (uint16_t)((_fp8x16_6_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_6) : "h"(_e4m3x2_1_6));
                uint16_t _h2_6 = (uint16_t)((_f16x2_1_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 2]) : "h"(_h2_6));
                uint16_t _h3_6 = (uint16_t)((_f16x2_1_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 3]) : "h"(_h3_6));
                uint16_t _e4m3x2_2_6 = (uint16_t)((_fp8x16_6_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_6) : "h"(_e4m3x2_2_6));
                uint16_t _h4_6 = (uint16_t)((_f16x2_2_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 4]) : "h"(_h4_6));
                uint16_t _h5_6 = (uint16_t)((_f16x2_2_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 5]) : "h"(_h5_6));
                uint16_t _e4m3x2_3_6 = (uint16_t)((_fp8x16_6_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_6) : "h"(_e4m3x2_3_6));
                uint16_t _h6_6 = (uint16_t)((_f16x2_3_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 6]) : "h"(_h6_6));
                uint16_t _h7_6 = (uint16_t)((_f16x2_3_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 7]) : "h"(_h7_6));
                uint16_t _e4m3x2_4_6 = (uint16_t)((_fp8x16_6_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_6) : "h"(_e4m3x2_4_6));
                uint16_t _h8_6 = (uint16_t)((_f16x2_4_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 8]) : "h"(_h8_6));
                uint16_t _h9_6 = (uint16_t)((_f16x2_4_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 9]) : "h"(_h9_6));
                uint16_t _e4m3x2_5_6 = (uint16_t)((_fp8x16_6_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_6) : "h"(_e4m3x2_5_6));
                uint16_t _h10_6 = (uint16_t)((_f16x2_5_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 10]) : "h"(_h10_6));
                uint16_t _h11_6 = (uint16_t)((_f16x2_5_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 11]) : "h"(_h11_6));
                uint16_t _e4m3x2_6_6 = (uint16_t)((_fp8x16_6_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_6) : "h"(_e4m3x2_6_6));
                uint16_t _h12_6 = (uint16_t)((_f16x2_6_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 12]) : "h"(_h12_6));
                uint16_t _h13_6 = (uint16_t)((_f16x2_6_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 13]) : "h"(_h13_6));
                uint16_t _e4m3x2_7_6 = (uint16_t)((_fp8x16_6_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_6) : "h"(_e4m3x2_7_6));
                uint16_t _h14_6 = (uint16_t)((_f16x2_7_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 14]) : "h"(_h14_6));
                uint16_t _h15_6 = (uint16_t)((_f16x2_7_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10[0 + 15]) : "h"(_h15_6));
            }
            float _shfl_6 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 6);
            float weight_11 = _shfl_6;
            #pragma unroll
            for (int elem_7 = 0; elem_7 < 16; elem_7++) {
                float _fma_6 = __fmaf_rn(values_10[elem_7], weight_11, accum[elem_7]);
                accum[elem_7] = _fma_6;
            }
            float values_12[16];
            {
                unsigned _fp8x16_7_0;
                unsigned _fp8x16_7_1;
                unsigned _fp8x16_7_2;
                unsigned _fp8x16_7_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_7_0), "=r"(_fp8x16_7_1), "=r"(_fp8x16_7_2), "=r"(_fp8x16_7_3) : "l"((const void*)(partial_o + ((7 * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment))) : "memory");
                uint16_t _e4m3x2_0_7 = (uint16_t)((_fp8x16_7_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_7) : "h"(_e4m3x2_0_7));
                uint16_t _h0_7 = (uint16_t)((_f16x2_0_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 0]) : "h"(_h0_7));
                uint16_t _h1_7 = (uint16_t)((_f16x2_0_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 1]) : "h"(_h1_7));
                uint16_t _e4m3x2_1_7 = (uint16_t)((_fp8x16_7_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_7) : "h"(_e4m3x2_1_7));
                uint16_t _h2_7 = (uint16_t)((_f16x2_1_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 2]) : "h"(_h2_7));
                uint16_t _h3_7 = (uint16_t)((_f16x2_1_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 3]) : "h"(_h3_7));
                uint16_t _e4m3x2_2_7 = (uint16_t)((_fp8x16_7_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_7) : "h"(_e4m3x2_2_7));
                uint16_t _h4_7 = (uint16_t)((_f16x2_2_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 4]) : "h"(_h4_7));
                uint16_t _h5_7 = (uint16_t)((_f16x2_2_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 5]) : "h"(_h5_7));
                uint16_t _e4m3x2_3_7 = (uint16_t)((_fp8x16_7_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_7) : "h"(_e4m3x2_3_7));
                uint16_t _h6_7 = (uint16_t)((_f16x2_3_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 6]) : "h"(_h6_7));
                uint16_t _h7_7 = (uint16_t)((_f16x2_3_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 7]) : "h"(_h7_7));
                uint16_t _e4m3x2_4_7 = (uint16_t)((_fp8x16_7_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_7) : "h"(_e4m3x2_4_7));
                uint16_t _h8_7 = (uint16_t)((_f16x2_4_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 8]) : "h"(_h8_7));
                uint16_t _h9_7 = (uint16_t)((_f16x2_4_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 9]) : "h"(_h9_7));
                uint16_t _e4m3x2_5_7 = (uint16_t)((_fp8x16_7_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_7) : "h"(_e4m3x2_5_7));
                uint16_t _h10_7 = (uint16_t)((_f16x2_5_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 10]) : "h"(_h10_7));
                uint16_t _h11_7 = (uint16_t)((_f16x2_5_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 11]) : "h"(_h11_7));
                uint16_t _e4m3x2_6_7 = (uint16_t)((_fp8x16_7_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_7) : "h"(_e4m3x2_6_7));
                uint16_t _h12_7 = (uint16_t)((_f16x2_6_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 12]) : "h"(_h12_7));
                uint16_t _h13_7 = (uint16_t)((_f16x2_6_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 13]) : "h"(_h13_7));
                uint16_t _e4m3x2_7_7 = (uint16_t)((_fp8x16_7_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_7) : "h"(_e4m3x2_7_7));
                uint16_t _h14_7 = (uint16_t)((_f16x2_7_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 14]) : "h"(_h14_7));
                uint16_t _h15_7 = (uint16_t)((_f16x2_7_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12[0 + 15]) : "h"(_h15_7));
            }
            float _shfl_7 = __shfl_sync(0xFFFFFFFF, lane_weight, leader_lane + 7);
            float weight_13 = _shfl_7;
            #pragma unroll
            for (int elem_8 = 0; elem_8 < 16; elem_8++) {
                float _fma_7 = __fmaf_rn(values_12[elem_8], weight_13, accum[elem_8]);
                accum[elem_8] = _fma_7;
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
    // ---- Role: upper ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // upper_main
            int q_position_1 = blockIdx.x;
            int q_abs_1 = q_order[q_position_1];
            int warp_id_in_role_1 = (warp - 4);
            int role_tid_1 = warp_id_in_role_1 * 32 + lane;
            int row_group_1 = role_tid_1 / 8;
            int lane_in_row_1 = role_tid_1 & 7;
            int leader_lane_1 = lane / 8 * 8;
            int cohort_linear_1 = q_abs_1 * num_kv_heads + 1;
            if (role_tid_1 < 8) {
                int contributor_slot_1 = role_tid_1;
                int work_idx_1 = contributor_work_ids[cohort_linear_1 * 8 + contributor_slot_1];
                {
                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(completion_counts) + (work_idx_1);
                    while (true) {
                        unsigned int _gca_v;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                        if (_gca_v >= (unsigned int)(generation)) break;
                    }
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            int row_1 = q_abs_1 * num_q_heads + qhead_per_kv + row_group_1;
            int total_rows_out_1 = total_q * num_q_heads;
            long long split_row_1 = (long long)lane_in_row_1 * (long long)total_rows_out_1 + (long long)row_1;
            float lane_lse_1 = partial_lse[split_row_1];
            float lse_max_1 = lane_lse_1;
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, lse_max_1, 1);
            float peer_max_1 = _shfl_xor_6;
            float _max_3 = max_noftz(lse_max_1, peer_max_1);
            lse_max_1 = _max_3;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, lse_max_1, 2);
            peer_max_1 = _shfl_xor_7;
            float _max_4 = max_noftz(lse_max_1, peer_max_1);
            lse_max_1 = _max_4;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, lse_max_1, 4);
            peer_max_1 = _shfl_xor_8;
            float _max_5 = max_noftz(lse_max_1, peer_max_1);
            lse_max_1 = _max_5;
            float safe_lse_max_1 = ((lse_max_1 == -BLACKWELL_MSA_INF) ? 0.0f : lse_max_1);
            float _exp2_1 = approx_exp2((lane_lse_1 - safe_lse_max_1) * 1.4426950408889634f);
            float lane_weight_1 = _exp2_1;
            if (lane_lse_1 == -BLACKWELL_MSA_INF) {
                lane_weight_1 = 0.0f;
            }
            float lse_sum_1 = lane_weight_1;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 1);
            float peer_sum_1 = _shfl_xor_9;
            lse_sum_1 += peer_sum_1;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 2);
            peer_sum_1 = _shfl_xor_10;
            lse_sum_1 += peer_sum_1;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 4);
            peer_sum_1 = _shfl_xor_11;
            lse_sum_1 += peer_sum_1;
            float _rcp_1 = approx_rcp(lse_sum_1);
            float inv_lse_sum_1 = ((lse_sum_1 > 0.0f && lse_sum_1 == lse_sum_1) ? _rcp_1 : 0.0f);
            lane_weight_1 *= inv_lse_sum_1;
            if (lane_in_row_1 == 0) {
                float final_lse_1 = -BLACKWELL_MSA_INF;
                if (return_softmax_lse != 0 || return_temperature_lse != 0) {
                    float _log2_1;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(lse_sum_1));
                    final_lse_1 = ((lse_sum_1 > 0.0f) ? safe_lse_max_1 + _log2_1 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
                }
                if (return_softmax_lse != 0) {
                    lse[row_1] = final_lse_1;
                }
                if (return_temperature_lse != 0) {
                    temperature_lse[row_1] = final_lse_1;
                }
            }
            int col_segment_1 = lane_in_row_1 * 16;
            float accum_1[16];
            #pragma unroll
            for (int elem_9 = 0; elem_9 < 16; elem_9++) {
                accum_1[elem_9] = 0.0f;
            }
            float values_1[16];
            {
                unsigned _fp8x16_0_0;
                unsigned _fp8x16_0_1;
                unsigned _fp8x16_0_2;
                unsigned _fp8x16_0_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_0_0), "=r"(_fp8x16_0_1), "=r"(_fp8x16_0_2), "=r"(_fp8x16_0_3) : "l"((const void*)(partial_o + ((long long)row_1 * 128 + (long long)col_segment_1))) : "memory");
                uint16_t _e4m3x2_0_0 = (uint16_t)((_fp8x16_0_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_0) : "h"(_e4m3x2_0_0));
                uint16_t _h0_0 = (uint16_t)((_f16x2_0_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 0]) : "h"(_h0_0));
                uint16_t _h1_0 = (uint16_t)((_f16x2_0_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 1]) : "h"(_h1_0));
                uint16_t _e4m3x2_1_0 = (uint16_t)((_fp8x16_0_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_0) : "h"(_e4m3x2_1_0));
                uint16_t _h2_0 = (uint16_t)((_f16x2_1_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 2]) : "h"(_h2_0));
                uint16_t _h3_0 = (uint16_t)((_f16x2_1_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 3]) : "h"(_h3_0));
                uint16_t _e4m3x2_2_0 = (uint16_t)((_fp8x16_0_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_0) : "h"(_e4m3x2_2_0));
                uint16_t _h4_0 = (uint16_t)((_f16x2_2_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 4]) : "h"(_h4_0));
                uint16_t _h5_0 = (uint16_t)((_f16x2_2_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 5]) : "h"(_h5_0));
                uint16_t _e4m3x2_3_0 = (uint16_t)((_fp8x16_0_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_0) : "h"(_e4m3x2_3_0));
                uint16_t _h6_0 = (uint16_t)((_f16x2_3_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 6]) : "h"(_h6_0));
                uint16_t _h7_0 = (uint16_t)((_f16x2_3_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 7]) : "h"(_h7_0));
                uint16_t _e4m3x2_4_0 = (uint16_t)((_fp8x16_0_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_0) : "h"(_e4m3x2_4_0));
                uint16_t _h8_0 = (uint16_t)((_f16x2_4_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 8]) : "h"(_h8_0));
                uint16_t _h9_0 = (uint16_t)((_f16x2_4_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 9]) : "h"(_h9_0));
                uint16_t _e4m3x2_5_0 = (uint16_t)((_fp8x16_0_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_0) : "h"(_e4m3x2_5_0));
                uint16_t _h10_0 = (uint16_t)((_f16x2_5_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 10]) : "h"(_h10_0));
                uint16_t _h11_0 = (uint16_t)((_f16x2_5_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 11]) : "h"(_h11_0));
                uint16_t _e4m3x2_6_0 = (uint16_t)((_fp8x16_0_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_0) : "h"(_e4m3x2_6_0));
                uint16_t _h12_0 = (uint16_t)((_f16x2_6_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 12]) : "h"(_h12_0));
                uint16_t _h13_0 = (uint16_t)((_f16x2_6_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 13]) : "h"(_h13_0));
                uint16_t _e4m3x2_7_0 = (uint16_t)((_fp8x16_0_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_0) : "h"(_e4m3x2_7_0));
                uint16_t _h14_0 = (uint16_t)((_f16x2_7_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 14]) : "h"(_h14_0));
                uint16_t _h15_0 = (uint16_t)((_f16x2_7_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_1[0 + 15]) : "h"(_h15_0));
            }
            float _shfl_8 = __shfl_sync(0xFFFFFFFF, lane_weight_1, leader_lane_1);
            float weight_2 = _shfl_8;
            #pragma unroll
            for (int elem_10 = 0; elem_10 < 16; elem_10++) {
                float _fma_8 = __fmaf_rn(values_1[elem_10], weight_2, accum_1[elem_10]);
                accum_1[elem_10] = _fma_8;
            }
            float values_0_1[16];
            {
                unsigned _fp8x16_1_0;
                unsigned _fp8x16_1_1;
                unsigned _fp8x16_1_2;
                unsigned _fp8x16_1_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_1_0), "=r"(_fp8x16_1_1), "=r"(_fp8x16_1_2), "=r"(_fp8x16_1_3) : "l"((const void*)(partial_o + (((long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1))) : "memory");
                uint16_t _e4m3x2_0_1 = (uint16_t)((_fp8x16_1_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_1) : "h"(_e4m3x2_0_1));
                uint16_t _h0_1 = (uint16_t)((_f16x2_0_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 0]) : "h"(_h0_1));
                uint16_t _h1_1 = (uint16_t)((_f16x2_0_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 1]) : "h"(_h1_1));
                uint16_t _e4m3x2_1_1 = (uint16_t)((_fp8x16_1_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_1) : "h"(_e4m3x2_1_1));
                uint16_t _h2_1 = (uint16_t)((_f16x2_1_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 2]) : "h"(_h2_1));
                uint16_t _h3_1 = (uint16_t)((_f16x2_1_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 3]) : "h"(_h3_1));
                uint16_t _e4m3x2_2_1 = (uint16_t)((_fp8x16_1_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_1) : "h"(_e4m3x2_2_1));
                uint16_t _h4_1 = (uint16_t)((_f16x2_2_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 4]) : "h"(_h4_1));
                uint16_t _h5_1 = (uint16_t)((_f16x2_2_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 5]) : "h"(_h5_1));
                uint16_t _e4m3x2_3_1 = (uint16_t)((_fp8x16_1_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_1) : "h"(_e4m3x2_3_1));
                uint16_t _h6_1 = (uint16_t)((_f16x2_3_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 6]) : "h"(_h6_1));
                uint16_t _h7_1 = (uint16_t)((_f16x2_3_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 7]) : "h"(_h7_1));
                uint16_t _e4m3x2_4_1 = (uint16_t)((_fp8x16_1_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_1) : "h"(_e4m3x2_4_1));
                uint16_t _h8_1 = (uint16_t)((_f16x2_4_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 8]) : "h"(_h8_1));
                uint16_t _h9_1 = (uint16_t)((_f16x2_4_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 9]) : "h"(_h9_1));
                uint16_t _e4m3x2_5_1 = (uint16_t)((_fp8x16_1_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_1) : "h"(_e4m3x2_5_1));
                uint16_t _h10_1 = (uint16_t)((_f16x2_5_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 10]) : "h"(_h10_1));
                uint16_t _h11_1 = (uint16_t)((_f16x2_5_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 11]) : "h"(_h11_1));
                uint16_t _e4m3x2_6_1 = (uint16_t)((_fp8x16_1_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_1) : "h"(_e4m3x2_6_1));
                uint16_t _h12_1 = (uint16_t)((_f16x2_6_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 12]) : "h"(_h12_1));
                uint16_t _h13_1 = (uint16_t)((_f16x2_6_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 13]) : "h"(_h13_1));
                uint16_t _e4m3x2_7_1 = (uint16_t)((_fp8x16_1_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_1) : "h"(_e4m3x2_7_1));
                uint16_t _h14_1 = (uint16_t)((_f16x2_7_1 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 14]) : "h"(_h14_1));
                uint16_t _h15_1 = (uint16_t)((_f16x2_7_1 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_0_1[0 + 15]) : "h"(_h15_1));
            }
            float _shfl_9 = __shfl_sync(0xFFFFFFFF, lane_weight_1, leader_lane_1 + 1);
            float weight_1_1 = _shfl_9;
            #pragma unroll
            for (int elem_11 = 0; elem_11 < 16; elem_11++) {
                float _fma_9 = __fmaf_rn(values_0_1[elem_11], weight_1_1, accum_1[elem_11]);
                accum_1[elem_11] = _fma_9;
            }
            float values_2_1[16];
            {
                unsigned _fp8x16_2_0;
                unsigned _fp8x16_2_1;
                unsigned _fp8x16_2_2;
                unsigned _fp8x16_2_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_2_0), "=r"(_fp8x16_2_1), "=r"(_fp8x16_2_2), "=r"(_fp8x16_2_3) : "l"((const void*)(partial_o + ((2 * (long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1))) : "memory");
                uint16_t _e4m3x2_0_2 = (uint16_t)((_fp8x16_2_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_2) : "h"(_e4m3x2_0_2));
                uint16_t _h0_2 = (uint16_t)((_f16x2_0_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 0]) : "h"(_h0_2));
                uint16_t _h1_2 = (uint16_t)((_f16x2_0_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 1]) : "h"(_h1_2));
                uint16_t _e4m3x2_1_2 = (uint16_t)((_fp8x16_2_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_2) : "h"(_e4m3x2_1_2));
                uint16_t _h2_2 = (uint16_t)((_f16x2_1_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 2]) : "h"(_h2_2));
                uint16_t _h3_2 = (uint16_t)((_f16x2_1_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 3]) : "h"(_h3_2));
                uint16_t _e4m3x2_2_2 = (uint16_t)((_fp8x16_2_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_2) : "h"(_e4m3x2_2_2));
                uint16_t _h4_2 = (uint16_t)((_f16x2_2_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 4]) : "h"(_h4_2));
                uint16_t _h5_2 = (uint16_t)((_f16x2_2_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 5]) : "h"(_h5_2));
                uint16_t _e4m3x2_3_2 = (uint16_t)((_fp8x16_2_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_2) : "h"(_e4m3x2_3_2));
                uint16_t _h6_2 = (uint16_t)((_f16x2_3_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 6]) : "h"(_h6_2));
                uint16_t _h7_2 = (uint16_t)((_f16x2_3_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 7]) : "h"(_h7_2));
                uint16_t _e4m3x2_4_2 = (uint16_t)((_fp8x16_2_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_2) : "h"(_e4m3x2_4_2));
                uint16_t _h8_2 = (uint16_t)((_f16x2_4_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 8]) : "h"(_h8_2));
                uint16_t _h9_2 = (uint16_t)((_f16x2_4_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 9]) : "h"(_h9_2));
                uint16_t _e4m3x2_5_2 = (uint16_t)((_fp8x16_2_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_2) : "h"(_e4m3x2_5_2));
                uint16_t _h10_2 = (uint16_t)((_f16x2_5_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 10]) : "h"(_h10_2));
                uint16_t _h11_2 = (uint16_t)((_f16x2_5_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 11]) : "h"(_h11_2));
                uint16_t _e4m3x2_6_2 = (uint16_t)((_fp8x16_2_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_2) : "h"(_e4m3x2_6_2));
                uint16_t _h12_2 = (uint16_t)((_f16x2_6_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 12]) : "h"(_h12_2));
                uint16_t _h13_2 = (uint16_t)((_f16x2_6_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 13]) : "h"(_h13_2));
                uint16_t _e4m3x2_7_2 = (uint16_t)((_fp8x16_2_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_2) : "h"(_e4m3x2_7_2));
                uint16_t _h14_2 = (uint16_t)((_f16x2_7_2 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 14]) : "h"(_h14_2));
                uint16_t _h15_2 = (uint16_t)((_f16x2_7_2 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_2_1[0 + 15]) : "h"(_h15_2));
            }
            float _shfl_10 = __shfl_sync(0xFFFFFFFF, lane_weight_1, leader_lane_1 + 2);
            float weight_3_1 = _shfl_10;
            #pragma unroll
            for (int elem_12 = 0; elem_12 < 16; elem_12++) {
                float _fma_10 = __fmaf_rn(values_2_1[elem_12], weight_3_1, accum_1[elem_12]);
                accum_1[elem_12] = _fma_10;
            }
            float values_4_1[16];
            {
                unsigned _fp8x16_3_0;
                unsigned _fp8x16_3_1;
                unsigned _fp8x16_3_2;
                unsigned _fp8x16_3_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_3_0), "=r"(_fp8x16_3_1), "=r"(_fp8x16_3_2), "=r"(_fp8x16_3_3) : "l"((const void*)(partial_o + ((3 * (long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1))) : "memory");
                uint16_t _e4m3x2_0_3 = (uint16_t)((_fp8x16_3_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_3) : "h"(_e4m3x2_0_3));
                uint16_t _h0_3 = (uint16_t)((_f16x2_0_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 0]) : "h"(_h0_3));
                uint16_t _h1_3 = (uint16_t)((_f16x2_0_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 1]) : "h"(_h1_3));
                uint16_t _e4m3x2_1_3 = (uint16_t)((_fp8x16_3_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_3) : "h"(_e4m3x2_1_3));
                uint16_t _h2_3 = (uint16_t)((_f16x2_1_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 2]) : "h"(_h2_3));
                uint16_t _h3_3 = (uint16_t)((_f16x2_1_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 3]) : "h"(_h3_3));
                uint16_t _e4m3x2_2_3 = (uint16_t)((_fp8x16_3_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_3) : "h"(_e4m3x2_2_3));
                uint16_t _h4_3 = (uint16_t)((_f16x2_2_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 4]) : "h"(_h4_3));
                uint16_t _h5_3 = (uint16_t)((_f16x2_2_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 5]) : "h"(_h5_3));
                uint16_t _e4m3x2_3_3 = (uint16_t)((_fp8x16_3_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_3) : "h"(_e4m3x2_3_3));
                uint16_t _h6_3 = (uint16_t)((_f16x2_3_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 6]) : "h"(_h6_3));
                uint16_t _h7_3 = (uint16_t)((_f16x2_3_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 7]) : "h"(_h7_3));
                uint16_t _e4m3x2_4_3 = (uint16_t)((_fp8x16_3_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_3) : "h"(_e4m3x2_4_3));
                uint16_t _h8_3 = (uint16_t)((_f16x2_4_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 8]) : "h"(_h8_3));
                uint16_t _h9_3 = (uint16_t)((_f16x2_4_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 9]) : "h"(_h9_3));
                uint16_t _e4m3x2_5_3 = (uint16_t)((_fp8x16_3_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_3) : "h"(_e4m3x2_5_3));
                uint16_t _h10_3 = (uint16_t)((_f16x2_5_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 10]) : "h"(_h10_3));
                uint16_t _h11_3 = (uint16_t)((_f16x2_5_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 11]) : "h"(_h11_3));
                uint16_t _e4m3x2_6_3 = (uint16_t)((_fp8x16_3_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_3) : "h"(_e4m3x2_6_3));
                uint16_t _h12_3 = (uint16_t)((_f16x2_6_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 12]) : "h"(_h12_3));
                uint16_t _h13_3 = (uint16_t)((_f16x2_6_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 13]) : "h"(_h13_3));
                uint16_t _e4m3x2_7_3 = (uint16_t)((_fp8x16_3_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_3) : "h"(_e4m3x2_7_3));
                uint16_t _h14_3 = (uint16_t)((_f16x2_7_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 14]) : "h"(_h14_3));
                uint16_t _h15_3 = (uint16_t)((_f16x2_7_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_4_1[0 + 15]) : "h"(_h15_3));
            }
            float _shfl_11 = __shfl_sync(0xFFFFFFFF, lane_weight_1, leader_lane_1 + 3);
            float weight_5_1 = _shfl_11;
            #pragma unroll
            for (int elem_13 = 0; elem_13 < 16; elem_13++) {
                float _fma_11 = __fmaf_rn(values_4_1[elem_13], weight_5_1, accum_1[elem_13]);
                accum_1[elem_13] = _fma_11;
            }
            float values_6_1[16];
            {
                unsigned _fp8x16_4_0;
                unsigned _fp8x16_4_1;
                unsigned _fp8x16_4_2;
                unsigned _fp8x16_4_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_4_0), "=r"(_fp8x16_4_1), "=r"(_fp8x16_4_2), "=r"(_fp8x16_4_3) : "l"((const void*)(partial_o + ((4 * (long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1))) : "memory");
                uint16_t _e4m3x2_0_4 = (uint16_t)((_fp8x16_4_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_4) : "h"(_e4m3x2_0_4));
                uint16_t _h0_4 = (uint16_t)((_f16x2_0_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 0]) : "h"(_h0_4));
                uint16_t _h1_4 = (uint16_t)((_f16x2_0_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 1]) : "h"(_h1_4));
                uint16_t _e4m3x2_1_4 = (uint16_t)((_fp8x16_4_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_4) : "h"(_e4m3x2_1_4));
                uint16_t _h2_4 = (uint16_t)((_f16x2_1_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 2]) : "h"(_h2_4));
                uint16_t _h3_4 = (uint16_t)((_f16x2_1_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 3]) : "h"(_h3_4));
                uint16_t _e4m3x2_2_4 = (uint16_t)((_fp8x16_4_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_4) : "h"(_e4m3x2_2_4));
                uint16_t _h4_4 = (uint16_t)((_f16x2_2_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 4]) : "h"(_h4_4));
                uint16_t _h5_4 = (uint16_t)((_f16x2_2_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 5]) : "h"(_h5_4));
                uint16_t _e4m3x2_3_4 = (uint16_t)((_fp8x16_4_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_4) : "h"(_e4m3x2_3_4));
                uint16_t _h6_4 = (uint16_t)((_f16x2_3_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 6]) : "h"(_h6_4));
                uint16_t _h7_4 = (uint16_t)((_f16x2_3_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 7]) : "h"(_h7_4));
                uint16_t _e4m3x2_4_4 = (uint16_t)((_fp8x16_4_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_4) : "h"(_e4m3x2_4_4));
                uint16_t _h8_4 = (uint16_t)((_f16x2_4_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 8]) : "h"(_h8_4));
                uint16_t _h9_4 = (uint16_t)((_f16x2_4_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 9]) : "h"(_h9_4));
                uint16_t _e4m3x2_5_4 = (uint16_t)((_fp8x16_4_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_4) : "h"(_e4m3x2_5_4));
                uint16_t _h10_4 = (uint16_t)((_f16x2_5_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 10]) : "h"(_h10_4));
                uint16_t _h11_4 = (uint16_t)((_f16x2_5_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 11]) : "h"(_h11_4));
                uint16_t _e4m3x2_6_4 = (uint16_t)((_fp8x16_4_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_4) : "h"(_e4m3x2_6_4));
                uint16_t _h12_4 = (uint16_t)((_f16x2_6_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 12]) : "h"(_h12_4));
                uint16_t _h13_4 = (uint16_t)((_f16x2_6_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 13]) : "h"(_h13_4));
                uint16_t _e4m3x2_7_4 = (uint16_t)((_fp8x16_4_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_4) : "h"(_e4m3x2_7_4));
                uint16_t _h14_4 = (uint16_t)((_f16x2_7_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 14]) : "h"(_h14_4));
                uint16_t _h15_4 = (uint16_t)((_f16x2_7_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_6_1[0 + 15]) : "h"(_h15_4));
            }
            float _shfl_12 = __shfl_sync(0xFFFFFFFF, lane_weight_1, leader_lane_1 + 4);
            float weight_7_1 = _shfl_12;
            #pragma unroll
            for (int elem_14 = 0; elem_14 < 16; elem_14++) {
                float _fma_12 = __fmaf_rn(values_6_1[elem_14], weight_7_1, accum_1[elem_14]);
                accum_1[elem_14] = _fma_12;
            }
            float values_8_1[16];
            {
                unsigned _fp8x16_5_0;
                unsigned _fp8x16_5_1;
                unsigned _fp8x16_5_2;
                unsigned _fp8x16_5_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_5_0), "=r"(_fp8x16_5_1), "=r"(_fp8x16_5_2), "=r"(_fp8x16_5_3) : "l"((const void*)(partial_o + ((5 * (long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1))) : "memory");
                uint16_t _e4m3x2_0_5 = (uint16_t)((_fp8x16_5_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_5) : "h"(_e4m3x2_0_5));
                uint16_t _h0_5 = (uint16_t)((_f16x2_0_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 0]) : "h"(_h0_5));
                uint16_t _h1_5 = (uint16_t)((_f16x2_0_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 1]) : "h"(_h1_5));
                uint16_t _e4m3x2_1_5 = (uint16_t)((_fp8x16_5_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_5) : "h"(_e4m3x2_1_5));
                uint16_t _h2_5 = (uint16_t)((_f16x2_1_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 2]) : "h"(_h2_5));
                uint16_t _h3_5 = (uint16_t)((_f16x2_1_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 3]) : "h"(_h3_5));
                uint16_t _e4m3x2_2_5 = (uint16_t)((_fp8x16_5_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_5) : "h"(_e4m3x2_2_5));
                uint16_t _h4_5 = (uint16_t)((_f16x2_2_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 4]) : "h"(_h4_5));
                uint16_t _h5_5 = (uint16_t)((_f16x2_2_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 5]) : "h"(_h5_5));
                uint16_t _e4m3x2_3_5 = (uint16_t)((_fp8x16_5_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_5) : "h"(_e4m3x2_3_5));
                uint16_t _h6_5 = (uint16_t)((_f16x2_3_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 6]) : "h"(_h6_5));
                uint16_t _h7_5 = (uint16_t)((_f16x2_3_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 7]) : "h"(_h7_5));
                uint16_t _e4m3x2_4_5 = (uint16_t)((_fp8x16_5_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_5) : "h"(_e4m3x2_4_5));
                uint16_t _h8_5 = (uint16_t)((_f16x2_4_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 8]) : "h"(_h8_5));
                uint16_t _h9_5 = (uint16_t)((_f16x2_4_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 9]) : "h"(_h9_5));
                uint16_t _e4m3x2_5_5 = (uint16_t)((_fp8x16_5_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_5) : "h"(_e4m3x2_5_5));
                uint16_t _h10_5 = (uint16_t)((_f16x2_5_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 10]) : "h"(_h10_5));
                uint16_t _h11_5 = (uint16_t)((_f16x2_5_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 11]) : "h"(_h11_5));
                uint16_t _e4m3x2_6_5 = (uint16_t)((_fp8x16_5_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_5) : "h"(_e4m3x2_6_5));
                uint16_t _h12_5 = (uint16_t)((_f16x2_6_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 12]) : "h"(_h12_5));
                uint16_t _h13_5 = (uint16_t)((_f16x2_6_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 13]) : "h"(_h13_5));
                uint16_t _e4m3x2_7_5 = (uint16_t)((_fp8x16_5_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_5;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_5) : "h"(_e4m3x2_7_5));
                uint16_t _h14_5 = (uint16_t)((_f16x2_7_5 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 14]) : "h"(_h14_5));
                uint16_t _h15_5 = (uint16_t)((_f16x2_7_5 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_8_1[0 + 15]) : "h"(_h15_5));
            }
            float _shfl_13 = __shfl_sync(0xFFFFFFFF, lane_weight_1, leader_lane_1 + 5);
            float weight_9_1 = _shfl_13;
            #pragma unroll
            for (int elem_15 = 0; elem_15 < 16; elem_15++) {
                float _fma_13 = __fmaf_rn(values_8_1[elem_15], weight_9_1, accum_1[elem_15]);
                accum_1[elem_15] = _fma_13;
            }
            float values_10_1[16];
            {
                unsigned _fp8x16_6_0;
                unsigned _fp8x16_6_1;
                unsigned _fp8x16_6_2;
                unsigned _fp8x16_6_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_6_0), "=r"(_fp8x16_6_1), "=r"(_fp8x16_6_2), "=r"(_fp8x16_6_3) : "l"((const void*)(partial_o + ((6 * (long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1))) : "memory");
                uint16_t _e4m3x2_0_6 = (uint16_t)((_fp8x16_6_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_6) : "h"(_e4m3x2_0_6));
                uint16_t _h0_6 = (uint16_t)((_f16x2_0_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 0]) : "h"(_h0_6));
                uint16_t _h1_6 = (uint16_t)((_f16x2_0_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 1]) : "h"(_h1_6));
                uint16_t _e4m3x2_1_6 = (uint16_t)((_fp8x16_6_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_6) : "h"(_e4m3x2_1_6));
                uint16_t _h2_6 = (uint16_t)((_f16x2_1_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 2]) : "h"(_h2_6));
                uint16_t _h3_6 = (uint16_t)((_f16x2_1_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 3]) : "h"(_h3_6));
                uint16_t _e4m3x2_2_6 = (uint16_t)((_fp8x16_6_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_6) : "h"(_e4m3x2_2_6));
                uint16_t _h4_6 = (uint16_t)((_f16x2_2_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 4]) : "h"(_h4_6));
                uint16_t _h5_6 = (uint16_t)((_f16x2_2_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 5]) : "h"(_h5_6));
                uint16_t _e4m3x2_3_6 = (uint16_t)((_fp8x16_6_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_6) : "h"(_e4m3x2_3_6));
                uint16_t _h6_6 = (uint16_t)((_f16x2_3_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 6]) : "h"(_h6_6));
                uint16_t _h7_6 = (uint16_t)((_f16x2_3_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 7]) : "h"(_h7_6));
                uint16_t _e4m3x2_4_6 = (uint16_t)((_fp8x16_6_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_6) : "h"(_e4m3x2_4_6));
                uint16_t _h8_6 = (uint16_t)((_f16x2_4_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 8]) : "h"(_h8_6));
                uint16_t _h9_6 = (uint16_t)((_f16x2_4_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 9]) : "h"(_h9_6));
                uint16_t _e4m3x2_5_6 = (uint16_t)((_fp8x16_6_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_6) : "h"(_e4m3x2_5_6));
                uint16_t _h10_6 = (uint16_t)((_f16x2_5_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 10]) : "h"(_h10_6));
                uint16_t _h11_6 = (uint16_t)((_f16x2_5_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 11]) : "h"(_h11_6));
                uint16_t _e4m3x2_6_6 = (uint16_t)((_fp8x16_6_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_6) : "h"(_e4m3x2_6_6));
                uint16_t _h12_6 = (uint16_t)((_f16x2_6_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 12]) : "h"(_h12_6));
                uint16_t _h13_6 = (uint16_t)((_f16x2_6_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 13]) : "h"(_h13_6));
                uint16_t _e4m3x2_7_6 = (uint16_t)((_fp8x16_6_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_6;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_6) : "h"(_e4m3x2_7_6));
                uint16_t _h14_6 = (uint16_t)((_f16x2_7_6 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 14]) : "h"(_h14_6));
                uint16_t _h15_6 = (uint16_t)((_f16x2_7_6 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_10_1[0 + 15]) : "h"(_h15_6));
            }
            float _shfl_14 = __shfl_sync(0xFFFFFFFF, lane_weight_1, leader_lane_1 + 6);
            float weight_11_1 = _shfl_14;
            #pragma unroll
            for (int elem_16 = 0; elem_16 < 16; elem_16++) {
                float _fma_14 = __fmaf_rn(values_10_1[elem_16], weight_11_1, accum_1[elem_16]);
                accum_1[elem_16] = _fma_14;
            }
            float values_12_1[16];
            {
                unsigned _fp8x16_7_0;
                unsigned _fp8x16_7_1;
                unsigned _fp8x16_7_2;
                unsigned _fp8x16_7_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_7_0), "=r"(_fp8x16_7_1), "=r"(_fp8x16_7_2), "=r"(_fp8x16_7_3) : "l"((const void*)(partial_o + ((7 * (long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1))) : "memory");
                uint16_t _e4m3x2_0_7 = (uint16_t)((_fp8x16_7_0 >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_7) : "h"(_e4m3x2_0_7));
                uint16_t _h0_7 = (uint16_t)((_f16x2_0_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 0]) : "h"(_h0_7));
                uint16_t _h1_7 = (uint16_t)((_f16x2_0_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 1]) : "h"(_h1_7));
                uint16_t _e4m3x2_1_7 = (uint16_t)((_fp8x16_7_0 >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_7) : "h"(_e4m3x2_1_7));
                uint16_t _h2_7 = (uint16_t)((_f16x2_1_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 2]) : "h"(_h2_7));
                uint16_t _h3_7 = (uint16_t)((_f16x2_1_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 3]) : "h"(_h3_7));
                uint16_t _e4m3x2_2_7 = (uint16_t)((_fp8x16_7_1 >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_7) : "h"(_e4m3x2_2_7));
                uint16_t _h4_7 = (uint16_t)((_f16x2_2_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 4]) : "h"(_h4_7));
                uint16_t _h5_7 = (uint16_t)((_f16x2_2_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 5]) : "h"(_h5_7));
                uint16_t _e4m3x2_3_7 = (uint16_t)((_fp8x16_7_1 >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_7) : "h"(_e4m3x2_3_7));
                uint16_t _h6_7 = (uint16_t)((_f16x2_3_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 6]) : "h"(_h6_7));
                uint16_t _h7_7 = (uint16_t)((_f16x2_3_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 7]) : "h"(_h7_7));
                uint16_t _e4m3x2_4_7 = (uint16_t)((_fp8x16_7_2 >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_7) : "h"(_e4m3x2_4_7));
                uint16_t _h8_7 = (uint16_t)((_f16x2_4_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 8]) : "h"(_h8_7));
                uint16_t _h9_7 = (uint16_t)((_f16x2_4_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 9]) : "h"(_h9_7));
                uint16_t _e4m3x2_5_7 = (uint16_t)((_fp8x16_7_2 >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_7) : "h"(_e4m3x2_5_7));
                uint16_t _h10_7 = (uint16_t)((_f16x2_5_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 10]) : "h"(_h10_7));
                uint16_t _h11_7 = (uint16_t)((_f16x2_5_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 11]) : "h"(_h11_7));
                uint16_t _e4m3x2_6_7 = (uint16_t)((_fp8x16_7_3 >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_7) : "h"(_e4m3x2_6_7));
                uint16_t _h12_7 = (uint16_t)((_f16x2_6_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 12]) : "h"(_h12_7));
                uint16_t _h13_7 = (uint16_t)((_f16x2_6_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 13]) : "h"(_h13_7));
                uint16_t _e4m3x2_7_7 = (uint16_t)((_fp8x16_7_3 >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_7;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_7) : "h"(_e4m3x2_7_7));
                uint16_t _h14_7 = (uint16_t)((_f16x2_7_7 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 14]) : "h"(_h14_7));
                uint16_t _h15_7 = (uint16_t)((_f16x2_7_7 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values_12_1[0 + 15]) : "h"(_h15_7));
            }
            float _shfl_15 = __shfl_sync(0xFFFFFFFF, lane_weight_1, leader_lane_1 + 7);
            float weight_13_1 = _shfl_15;
            #pragma unroll
            for (int elem_17 = 0; elem_17 < 16; elem_17++) {
                float _fma_15 = __fmaf_rn(values_12_1[elem_17], weight_13_1, accum_1[elem_17]);
                accum_1[elem_17] = _fma_15;
            }
            {
                __nv_bfloat162 _pk[8];
                _pk[0] = __floats2bfloat162_rn(accum_1[0 + 0], accum_1[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(accum_1[0 + 2], accum_1[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(accum_1[0 + 4], accum_1[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(accum_1[0 + 6], accum_1[0 + 7]);
                _pk[4] = __floats2bfloat162_rn(accum_1[0 + 8], accum_1[0 + 9]);
                _pk[5] = __floats2bfloat162_rn(accum_1[0 + 10], accum_1[0 + 11]);
                _pk[6] = __floats2bfloat162_rn(accum_1[0 + 12], accum_1[0 + 13]);
                _pk[7] = __floats2bfloat162_rn(accum_1[0 + 14], accum_1[0 + 15]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row_1 * 128 + (long long)col_segment_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row_1 * 128 + (long long)col_segment_1)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
            }
        }
    }
}

} // extern "C"
