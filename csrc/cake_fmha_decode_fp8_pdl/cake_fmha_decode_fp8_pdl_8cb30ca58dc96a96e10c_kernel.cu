/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define HEAD_DIM 128
#define FIXED_NUM_SPLITS 8
#define USE_PDL 1

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
kernel_cake_fmha_decode_fp8_pdl_8cb30ca58dc96a96e10c(float* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, uint8_t* __restrict__ O, float* __restrict__ bmm2_scale_ptr, int num_split, float bmm2_scalar)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    float output_scale = bmm2_scalar;
    if ((unsigned long long)bmm2_scale_ptr != 0) {
        output_scale = bmm2_scale_ptr[0];
    }
    int bh = blockIdx.x;
    int split_stride = ((FIXED_NUM_SPLITS != 0) ? FIXED_NUM_SPLITS : num_split);
    int stat_base = bh * split_stride;
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    float max_m = -CUDART_INF_F;
    float fixed_max[8];
    float _vec_load_0[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(partial_max + stat_base);
        _vec_load_0[0 + 0] = _v4.x;
        _vec_load_0[0 + 1] = _v4.y;
        _vec_load_0[0 + 2] = _v4.z;
        _vec_load_0[0 + 3] = _v4.w;
    }
    float _vec_load_1[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(partial_max + stat_base + 4);
        _vec_load_1[0 + 0] = _v4.x;
        _vec_load_1[0 + 1] = _v4.y;
        _vec_load_1[0 + 2] = _v4.z;
        _vec_load_1[0 + 3] = _v4.w;
    }
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        fixed_max[s] = _vec_load_0[s];
        fixed_max[s + 4] = _vec_load_1[s];
    }
    #pragma unroll
    for (int s0 = 0; s0 < FIXED_NUM_SPLITS; s0++) {
        float m_s = fixed_max[s0];
        max_m = max_noftz(max_m, m_s);
    }
    int d_base = lane * 4;
    int po_base = bh * split_stride * HEAD_DIM + d_base;
    float sum_w = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float _vec_load_2[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(partial_sum + stat_base);
        _vec_load_2[0 + 0] = _v4.x;
        _vec_load_2[0 + 1] = _v4.y;
        _vec_load_2[0 + 2] = _v4.z;
        _vec_load_2[0 + 3] = _v4.w;
    }
    float _vec_load_3[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(partial_sum + stat_base + 4);
        _vec_load_3[0 + 0] = _v4.x;
        _vec_load_3[0 + 1] = _v4.y;
        _vec_load_3[0 + 2] = _v4.z;
        _vec_load_3[0 + 3] = _v4.w;
    }
    #pragma unroll
    for (int s1 = 0; s1 < FIXED_NUM_SPLITS; s1++) {
        float m_s1 = fixed_max[s1];
        if (m_s1 != -CUDART_INF_F) {
            float split_sum = ((s1 < 4) ? _vec_load_2[s1] : _vec_load_3[s1 - 4]);
            float _exp2_4 = approx_exp2(m_s1 - max_m);
            float w_s = _exp2_4 * split_sum;
            sum_w = sum_w + w_s;
            int po_off = po_base + s1 * HEAD_DIM;
            float _vec_load_4[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_O + po_off);
                _vec_load_4[0 + 0] = _v4.x;
                _vec_load_4[0 + 1] = _v4.y;
                _vec_load_4[0 + 2] = _v4.z;
                _vec_load_4[0 + 3] = _v4.w;
            }
            acc0 = __fmaf_rn(w_s, _vec_load_4[0], acc0);
            acc1 = __fmaf_rn(w_s, _vec_load_4[1], acc1);
            acc2 = __fmaf_rn(w_s, _vec_load_4[2], acc2);
            acc3 = __fmaf_rn(w_s, _vec_load_4[3], acc3);
        }
    }
    float _rcp_6 = approx_rcp(sum_w);
    float final_scale = ((sum_w == 0.0f) ? 0.0f : _rcp_6) * output_scale;
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
    unsigned int _packed_0[1];
    {
        uint32_t _packed;
        asm volatile("{\n\t"
            ".reg .b16 _lo;\n\t"
            ".reg .b16 _hi;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
            "mov.b32 %0, {_lo, _hi};\n\t"
            "}"
            : "=r"(_packed) : "f"(out_quad[0]), "f"(out_quad[1]),
                               "f"(out_quad[2]), "f"(out_quad[3]));
        _packed_0[0] = _packed;
    }
    reinterpret_cast<unsigned*>(O + o_off)[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
}

} // extern "C"
