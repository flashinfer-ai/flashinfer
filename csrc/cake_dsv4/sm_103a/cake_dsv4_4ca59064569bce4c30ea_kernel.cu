/*
 * Copyright (c) 2023 by FlashInfer team.
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

__global__ __launch_bounds__(128) void
kernel_cake_dsv4_4ca59064569bce4c30ea(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_q_heads, int num_split)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    const int wg_dummy = 0;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y * 4 + warp;
    int stat_base = (batch_idx * num_q_heads + head_idx) * 5;
    int po_head_base = stat_base * 512;
    int d_base = lane * 16;
    float vals[80];
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        float _vec_load_0[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + (po_head_base + s * 512 + d_base) + 0);
            uint4 _vld_0[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int e = 0; e < 16; e++) {
            vals[s * 16 + e] = _vec_load_0[e];
        }
    }
    float local_m = -CAKE_INF;
    if (lane < 5) {
        local_m = partial_lse[stat_base + lane];
    }
    float _warp_reduce_0 = local_m;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    float global_max = _warp_reduce_0;
    float local_w = 0.0f;
    if (lane < 5) {
        float _exp2_0 = approx_exp2(local_m - global_max);
        local_w = ((local_m == -CAKE_INF) ? 0.0f : _exp2_0);
    }
    float _warp_reduce_1 = local_w;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    float global_sum = _warp_reduce_1;
    float _rcp_0 = approx_rcp(global_sum);
    float inv_sum = ((global_sum > 0.0f) ? _rcp_0 : 0.0f);
    float local_weight = local_w * inv_sum;
    int o_head_base = (batch_idx * num_q_heads + head_idx) * 512;
    float acc[16];
    #pragma unroll
    for (int e_1 = 0; e_1 < 16; e_1++) {
        acc[e_1] = 0.0f;
    }
    #pragma unroll
    for (int s_1 = 0; s_1 < 5; s_1++) {
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, local_weight, s_1);
        float split_weight = _shfl_0;
        #pragma unroll
        for (int e_2 = 0; e_2 < 16; e_2++) {
            acc[e_2] = acc[e_2] + split_weight * vals[s_1 * 16 + e_2];
        }
    }
    {
        __nv_bfloat162 _pk[8];
        _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
        _pk[2] = __floats2bfloat162_rn(acc[0 + 4], acc[0 + 5]);
        _pk[3] = __floats2bfloat162_rn(acc[0 + 6], acc[0 + 7]);
        _pk[4] = __floats2bfloat162_rn(acc[0 + 8], acc[0 + 9]);
        _pk[5] = __floats2bfloat162_rn(acc[0 + 10], acc[0 + 11]);
        _pk[6] = __floats2bfloat162_rn(acc[0 + 12], acc[0 + 13]);
        _pk[7] = __floats2bfloat162_rn(acc[0 + 14], acc[0 + 15]);
        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (o_head_base + d_base)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (o_head_base + d_base)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
    }
}

} // extern "C"
