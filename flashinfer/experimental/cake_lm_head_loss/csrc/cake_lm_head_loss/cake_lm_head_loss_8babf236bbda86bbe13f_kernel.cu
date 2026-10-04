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
kernel_cake_lm_head_loss_8babf236bbda86bbe13f(int* __restrict__ x, int* __restrict__ idx_lo, int* __restrict__ count, int* __restrict__ out, int ld_x_words, int row_vecs, int num_rows, int T)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int n_cnt = count[0];
    int zero = 0;
    int t_last = T - 1;
    int gstride = gridDim.x;
    unsigned long long ldw64 = (unsigned long long)ld_x_words;
    unsigned long long vecs64 = (unsigned long long)row_vecs * 4;
    for (int r = bid; r < num_rows; r += gstride) {
        int _vec_load_0[1];
        {
            _vec_load_0[0] = *reinterpret_cast<const int*>(idx_lo + ((unsigned long long)r * 2) + 0);
        }
        int src_u = _vec_load_0[0];
        int src = ((src_u < t_last) ? src_u : t_last);
        unsigned long long row_in = (unsigned long long)src * ldw64;
        unsigned long long row_out = (unsigned long long)r * vecs64;
        #pragma unroll 2
        for (int c = tid; c < row_vecs; c += 256) {
            unsigned long long off = (unsigned long long)c * 4;
            int _vec_load_1[4];
            {
                const int4* _ivptr_0 = reinterpret_cast<const int4*>(x + (row_in + off) + 0);
                int4 _ivld_0;
                _ivld_0 = *_ivptr_0;
                _vec_load_1[0 + 0] = _ivld_0.x;
                _vec_load_1[0 + 1] = _ivld_0.y;
                _vec_load_1[0 + 2] = _ivld_0.z;
                _vec_load_1[0 + 3] = _ivld_0.w;
            }
            int o[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                o[j] = ((n_cnt > r) ? _vec_load_1[j] : zero);
            }
            {
                int4 _iv4 = make_int4(o[0 + 0], o[0 + 1], o[0 + 2], o[0 + 3]);
                *reinterpret_cast<int4*>(out + (row_out + off) + 0) = _iv4;
            }
        }
    }
}

} // extern "C"
