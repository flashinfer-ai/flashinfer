/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_loss_59bf253fd36d24e3a02a(float* __restrict__ stats, __nv_bfloat16* __restrict__ z, long long* __restrict__ labels, float* __restrict__ infer_logp, float* __restrict__ loss_weights, float* __restrict__ d_in, float* __restrict__ lse, float* __restrict__ logp, float* __restrict__ d, float* __restrict__ term, int rows_c, int row0, int V, int num_tiles, int mode, float loss_div)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int r = bid * 8 + warp;
    int lane_0 = lane;
    int active = ((r < rows_c) ? 1 : 0);
    int r_safe = ((r < rows_c) ? r : rows_c - 1);
    int t = row0 + r_safe;
    long long lbl = labels[t];
    long long v64 = (long long)V;
    unsigned long long stat_row = (unsigned long long)r_safe * (unsigned long long)num_tiles * 2;
    float m = -3.4028234663852886e+38f;
    #pragma unroll 4
    for (int i = lane_0; i < num_tiles; i += 32) {
        float _vec_load_0[2];
        {
            float2 _v2_0 = *reinterpret_cast<const float2*>(stats + (stat_row + (unsigned long long)i * 2) + 0);
            _vec_load_0[0] = _v2_0.x;
            _vec_load_0[0 + 1] = _v2_0.y;
        }
        float _fmax_0 = fmaxf(m, _vec_load_0[0]);
        m = _fmax_0;
    }
    float _warp_reduce_0 = m;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = fmaxf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    float row_max = _warp_reduce_0;
    float s = 0.0f;
    #pragma unroll 4
    for (int i_1 = lane_0; i_1 < num_tiles; i_1 += 32) {
        float _vec_load_1[2];
        {
            float2 _v2_1 = *reinterpret_cast<const float2*>(stats + (stat_row + (unsigned long long)i_1 * 2) + 0);
            _vec_load_1[0] = _v2_1.x;
            _vec_load_1[0 + 1] = _v2_1.y;
        }
        float _exp_0 = expf(_vec_load_1[0] - row_max);
        s += _vec_load_1[1] * _exp_0;
    }
    float _warp_reduce_1 = s;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    float row_sum = _warp_reduce_1;
    float _log_0 = logf(row_sum);
    float lse_v = row_max + _log_0;
    long long zero64 = (long long)0;
    long long last64 = v64 - (long long)1;
    long long lbl_lo = ((lbl >= 0) ? lbl : zero64);
    long long lbl_idx = ((lbl_lo < v64) ? lbl_lo : last64);
    unsigned long long z_off = (unsigned long long)r_safe * (unsigned long long)V + (unsigned long long)lbl_idx;
    float _vec_load_2[1];
    {
        __nv_bfloat16 _bf16_2 = *reinterpret_cast<const __nv_bfloat16*>(z + z_off + 0);
        _vec_load_2[0] = __bfloat162float(_bf16_2);
    }
    float zy = _vec_load_2[0];
    float logp_v = ((lbl >= 0) ? zy - lse_v : 0.0f);
    float term_v = 0.0f;
    float d_v = 0.0f;
    if (mode == 0) {
        term_v = logp_v;
        d_v = ((lbl >= 0) ? -1.0f / loss_div : 0.0f);
    }
    if (mode == 1) {
        float w = loss_weights[t];
        float _exp_1 = expf(logp_v - infer_logp[t]);
        float ratio = _exp_1;
        float clipped = ((ratio < 2.0f) ? ratio : 2.0f);
        term_v = ((lbl >= 0) ? w * clipped : 0.0f);
        float d_grad = ((ratio <= 2.0f) ? (-w) * ratio : 0.0f);
        d_v = ((lbl >= 0) ? d_grad : 0.0f);
    }
    if (mode == 2) {
        float d_ext = d_in[t];
        d_v = ((lbl >= 0) ? d_ext : 0.0f);
    }
    if (active == 1) {
        if (lane_0 == 0) {
            lse[t] = lse_v;
            logp[t] = logp_v;
            d[r_safe] = d_v;
            term[r_safe] = term_v;
        }
    }
}

} // extern "C"
