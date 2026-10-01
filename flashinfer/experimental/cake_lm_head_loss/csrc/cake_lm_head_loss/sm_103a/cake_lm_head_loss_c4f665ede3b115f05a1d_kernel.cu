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
kernel_cake_lm_head_loss_c4f665ede3b115f05a1d(float* __restrict__ acc, float* __restrict__ g, __nv_bfloat16* __restrict__ out, int num_vecs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    float scale = g[0];
    int stride = gridDim.x * 256;
    #pragma unroll 2
    for (int vi = bid * 256 + tid; vi < num_vecs; vi += stride) {
        unsigned long long off = (unsigned long long)vi * 8;
        float _vec_load_0[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(acc + off + 0);
            _vec_load_0[0 + 0] = _v4.x;
            _vec_load_0[0 + 1] = _v4.y;
            _vec_load_0[0 + 2] = _v4.z;
            _vec_load_0[0 + 3] = _v4.w;
        }
        float _vec_load_1[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(acc + (off + 4) + 0);
            _vec_load_1[0 + 0] = _v4.x;
            _vec_load_1[0 + 1] = _v4.y;
            _vec_load_1[0 + 2] = _v4.z;
            _vec_load_1[0 + 3] = _v4.w;
        }
        float o[8];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            o[j] = scale * _vec_load_0[j];
            o[j + 4] = scale * _vec_load_1[j];
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(o[0 + 0], o[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(o[0 + 2], o[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(o[0 + 4], o[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(o[0 + 6], o[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + off))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
    }
}

} // extern "C"
