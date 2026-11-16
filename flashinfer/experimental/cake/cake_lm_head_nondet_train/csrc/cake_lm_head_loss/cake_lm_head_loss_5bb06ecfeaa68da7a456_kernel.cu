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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_loss_5bb06ecfeaa68da7a456(float* __restrict__ dx, float* __restrict__ ws, int ws_slab, int num_vecs, int n_slabs, int row_vecs, int* __restrict__ CNT)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int stride = gridDim.x * 256;
    unsigned long long slab64 = (unsigned long long)ws_slab;
    int n_vecs = num_vecs;
    int live_rows = CNT[0];
    int live_vecs = (live_rows + 255) / 256 * 256 * row_vecs;
    n_vecs = ((live_vecs < num_vecs) ? live_vecs : num_vecs);
    #pragma unroll 2
    for (int vi = bid * 256 + tid; vi < n_vecs; vi += stride) {
        unsigned long long off = (unsigned long long)vi * 8ull;
        float _vec_load_0[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(ws + off + 0);
            _vec_load_0[0 + 0] = _v4.x;
            _vec_load_0[0 + 1] = _v4.y;
            _vec_load_0[0 + 2] = _v4.z;
            _vec_load_0[0 + 3] = _v4.w;
        }
        float _vec_load_1[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(ws + (off + 4ull) + 0);
            _vec_load_1[0 + 0] = _v4.x;
            _vec_load_1[0 + 1] = _v4.y;
            _vec_load_1[0 + 2] = _v4.z;
            _vec_load_1[0 + 3] = _v4.w;
        }
        float o[8];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            o[j] = _vec_load_0[j];
            o[j + 4] = _vec_load_1[j];
        }
        for (int s = 1; s < n_slabs; s++) {
            unsigned long long soff = off + (unsigned long long)s * slab64;
            float _vec_load_2[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(ws + soff + 0);
                _vec_load_2[0 + 0] = _v4.x;
                _vec_load_2[0 + 1] = _v4.y;
                _vec_load_2[0 + 2] = _v4.z;
                _vec_load_2[0 + 3] = _v4.w;
            }
            float _vec_load_3[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(ws + (soff + 4ull) + 0);
                _vec_load_3[0 + 0] = _v4.x;
                _vec_load_3[0 + 1] = _v4.y;
                _vec_load_3[0 + 2] = _v4.z;
                _vec_load_3[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
                o[j_1] = o[j_1] + _vec_load_2[j_1];
                o[j_1 + 4] = o[j_1 + 4] + _vec_load_3[j_1];
            }
        }
        {
            float4 _v4 = make_float4(o[0 + 0], o[0 + 1], o[0 + 2], o[0 + 3]);
            *reinterpret_cast<float4*>((dx + off) + 0) = _v4;
        }
        {
            float4 _v4 = make_float4(o[4 + 0], o[4 + 1], o[4 + 2], o[4 + 3]);
            *reinterpret_cast<float4*>((dx + (off + 4ull)) + 0) = _v4;
        }
    }
    float z[4];
    #pragma unroll
    for (int j_2 = 0; j_2 < 4; j_2++) {
        z[j_2] = __uint_as_float(2147483648u);
    }
    for (int vi_1 = n_vecs + bid * 256 + tid; vi_1 < num_vecs; vi_1 += stride) {
        unsigned long long zoff = (unsigned long long)vi_1 * 8ull;
        {
            float4 _v4 = make_float4(z[0 + 0], z[0 + 1], z[0 + 2], z[0 + 3]);
            *reinterpret_cast<float4*>((dx + zoff) + 0) = _v4;
        }
        {
            float4 _v4 = make_float4(z[0 + 0], z[0 + 1], z[0 + 2], z[0 + 3]);
            *reinterpret_cast<float4*>((dx + (zoff + 4ull)) + 0) = _v4;
        }
    }
}

} // extern "C"
