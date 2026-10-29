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
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_det_train_e5b35ce10bdf1bb1fb38(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, float* __restrict__ g, __nv_bfloat16* __restrict__ beta, float* __restrict__ beta32_out, __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ rstd_q, float* __restrict__ rstd_k, float* __restrict__ g_cs, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int normalize_qk)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int chunk = blockIdx.x;
    int hv = blockIdx.y;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int sub = hv - hq * group;
    int lane_0 = lane;
    float q_frag[4];
    float k_frag[4];
    #pragma unroll
    for (int step = 0; step < 16; step++) {
        int row = sub * 4 + warp + 4 * group * step;
        if (row < n_valid) {
            long long base = ((long long)(tok0 + row) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
            {
                uint2 _vld_0;
                _vld_0 = *reinterpret_cast<const uint2*>(q + base);
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&q_frag[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&q_frag[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1;
                _vld_1 = *reinterpret_cast<const uint2*>(k + base);
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&k_frag[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&k_frag[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
            float q_sq = 0.0f;
            float k_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                q_sq = __fmaf_rn(q_frag[i], q_frag[i], q_sq);
                k_sq = __fmaf_rn(k_frag[i], k_frag[i], k_sq);
            }
            float _warp_reduce_2 = q_sq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
            q_sq = _warp_reduce_2;
            float _warp_reduce_3 = k_sq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
            k_sq = _warp_reduce_3;
            float q_inv = 1.0f;
            float k_inv = 1.0f;
            if (normalize_qk != 0) {
                float _rsqrt_4 = rsqrtf(q_sq + 1e-06f);
                q_inv = _rsqrt_4;
                float _rsqrt_5 = rsqrtf(k_sq + 1e-06f);
                k_inv = _rsqrt_5;
            }
            #pragma unroll
            for (int i2 = 0; i2 < 4; i2++) {
                q_frag[i2] = q_frag[i2] * q_inv;
                k_frag[i2] = k_frag[i2] * k_inv;
            }
            unsigned int _packed_0[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_frag[_lp*2 + 0], q_frag[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_frag[_lp*2 + 2], q_frag[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(qn + (base))[0] = reinterpret_cast<int2*>(_packed_0)[0];
            unsigned int _packed_1[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(k_frag[_lp*2 + 0], k_frag[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(k_frag[_lp*2 + 2], k_frag[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(kn + (base))[0] = reinterpret_cast<int2*>(_packed_1)[0];
            if (lane_0 == 0) {
                long long rstd_index = (long long)(tok0 + row) * (long long)num_heads + (long long)hq;
                rstd_q[rstd_index] = q_inv;
                rstd_k[rstd_index] = k_inv;
            }
        }
    }
    if (warp == 0) {
        float g_lo = 0.0f;
        float g_hi = 0.0f;
        long long gate_index_lo = (long long)(tok0 + lane_0) * (long long)num_v_heads + (long long)hv;
        long long gate_index_hi = (long long)(tok0 + lane_0 + 32) * (long long)num_v_heads + (long long)hv;
        if (lane_0 < n_valid) {
            g_lo = g[gate_index_lo];
        }
        if (n_valid > lane_0 + 32) {
            g_hi = g[gate_index_hi];
        }
        #pragma unroll
        for (int delta = 0; delta < 5; delta++) {
            const int shift = 1 << delta;
            float up_lo = __shfl_up_sync(0xFFFFFFFF, g_lo, shift, 32);
            float up_hi = __shfl_up_sync(0xFFFFFFFF, g_hi, shift, 32);
            if (lane_0 >= shift) {
                g_lo = g_lo + up_lo;
                g_hi = g_hi + up_hi;
            }
        }
        float total_lo = __shfl_sync(0xFFFFFFFF, g_lo, 31);
        g_hi = g_hi + total_lo;
        if (lane_0 < n_valid) {
            g_cs[gate_index_lo] = g_lo * 1.442695f;
            beta32_out[gate_index_lo] = (float)beta[gate_index_lo];
        }
        if (n_valid > lane_0 + 32) {
            g_cs[gate_index_hi] = g_hi * 1.442695f;
            beta32_out[gate_index_hi] = (float)beta[gate_index_hi];
        }
    }
}

} // extern "C"
