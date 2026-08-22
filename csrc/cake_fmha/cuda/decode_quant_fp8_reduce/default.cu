/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeFmhaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeFmhaTensorMapPack { CakeFmhaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define HEAD_DIM 128
#define FIXED_NUM_SPLITS 0
#define USE_PDL 0

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

__global__ __launch_bounds__(32) void
kernel_cake_fmha_decode_quant_fp8_reduce(float* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, uint8_t* __restrict__ O, float* __restrict__ bmm2_scale_ptr, int num_split)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    float output_scale = bmm2_scale_ptr[0];
    int bh = blockIdx.x;
    int split_stride = ((FIXED_NUM_SPLITS != 0) ? FIXED_NUM_SPLITS : num_split);
    int stat_base = bh * split_stride;
    float max_m = -LOOM_INF;
    float fixed_max[8];
    {
        #pragma unroll 8
        for (int s0 = 0; s0 < num_split; s0++) {
            float m_s = partial_max[stat_base + s0];
            float _max_1 = max_noftz(max_m, m_s);
            max_m = _max_1;
        }
    }
    int d_base = lane * 4;
    int po_base = bh * split_stride * HEAD_DIM + d_base;
    float sum_w = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    {
        #pragma unroll 8
        for (int s1 = 0; s1 < num_split; s1++) {
            float m_s1 = partial_max[stat_base + s1];
            if (m_s1 != -LOOM_INF) {
                float _exp2_1 = approx_exp2(m_s1 - max_m);
                float w_s = _exp2_1 * partial_sum[stat_base + s1];
                sum_w = sum_w + w_s;
                int po_off = po_base + s1 * HEAD_DIM;
                float _vec_load_5[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_O + po_off);
                    _vec_load_5[0 + 0] = _v4.x;
                    _vec_load_5[0 + 1] = _v4.y;
                    _vec_load_5[0 + 2] = _v4.z;
                    _vec_load_5[0 + 3] = _v4.w;
                }
                acc0 = acc0 + w_s * _vec_load_5[0];
                acc1 = acc1 + w_s * _vec_load_5[1];
                acc2 = acc2 + w_s * _vec_load_5[2];
                acc3 = acc3 + w_s * _vec_load_5[3];
            }
        }
    }
    float _rcp_0 = approx_rcp(sum_w);
    float final_scale = ((sum_w == 0.0f) ? 0.0f : _rcp_0) * output_scale;
    float out_pair0[2];
    float out_pair1[2];
    out_pair0[0] = acc0 * final_scale;
    out_pair0[1] = acc1 * final_scale;
    out_pair1[0] = acc2 * final_scale;
    out_pair1[1] = acc3 * final_scale;
    int o_off = bh * HEAD_DIM + d_base;
    float out_quad[4];
    out_quad[0] = out_pair0[0];
    out_quad[1] = out_pair0[1];
    out_quad[2] = out_pair1[0];
    out_quad[3] = out_pair1[1];
    {
        unsigned int _fp8_pk[1];
        { unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(out_quad[0 + 1]), "f"(out_quad[0 + 0]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(out_quad[0 + 3]), "f"(out_quad[0 + 2]));
            _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
        }
        *reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned char*>(O + o_off) + (0)) = *reinterpret_cast<unsigned int*>(_fp8_pk);
    }
}

} // extern "C"
