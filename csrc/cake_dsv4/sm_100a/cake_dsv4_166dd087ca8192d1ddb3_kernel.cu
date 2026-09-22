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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
kernel_cake_dsv4_166dd087ca8192d1ddb3(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_q_heads)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    const int wg_dummy = 0;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (batch_idx * num_q_heads + head_idx) * 2;
    float local_lse = -CAKE_INF;
    if (lane < 2) {
        local_lse = partial_lse[(unsigned int)stat_base + lane];
    }
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, local_lse, 0);
    float lse0 = _shfl_0;
    float _shfl_1 = __shfl_sync(0xFFFFFFFF, local_lse, 1);
    float lse1 = _shfl_1;
    float _max_0 = max_noftz(lse0, lse1);
    float global_max = _max_0;
    float _exp2_0 = approx_exp2(lse0 - global_max);
    float weight0 = ((lse0 == -CAKE_INF) ? 0.0f : _exp2_0);
    float _exp2_1 = approx_exp2(lse1 - global_max);
    float weight1 = ((lse1 == -CAKE_INF) ? 0.0f : _exp2_1);
    float global_sum = weight0 + weight1;
    float _rcp_0 = approx_rcp(global_sum);
    float inv_sum = ((global_sum > 0.0f) ? _rcp_0 : 0.0f);
    float split_weights[2];
    split_weights[0] = weight0 * inv_sum;
    split_weights[1] = weight1 * inv_sum;
    int po_head_base = stat_base * 512;
    int o_head_base = (batch_idx * num_q_heads + head_idx) * 512;
    float acc[4];
    int d_base = tid * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        acc[e] = 0.0f;
    }
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        float split_weight = split_weights[s];
        float _vec_load_0[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (po_head_base + s * 512 + d_base) + 0);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_0[0 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _pair * 2])[1])
                    : "r"(_vpairs_0[_pair]));
            }
        }
        #pragma unroll
        for (int e_1 = 0; e_1 < 4; e_1++) {
            acc[e_1] = acc[e_1] + split_weight * _vec_load_0[e_1];
        }
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O + (o_head_base + d_base)))[0]) = _pk2;
    }
}

} // extern "C"
