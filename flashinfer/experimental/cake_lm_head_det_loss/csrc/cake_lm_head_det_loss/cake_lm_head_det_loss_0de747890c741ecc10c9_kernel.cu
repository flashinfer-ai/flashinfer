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
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_det_loss_0de747890c741ecc10c9(__nv_bfloat16* __restrict__ dz, int* __restrict__ ids, int* __restrict__ offsets, __nv_bfloat16* __restrict__ dzblk, int t0, int rows, int v0, int vb, int ldc, int M, int V)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int tid_1 = tid;
    unsigned int zeros[4];
    #pragma unroll
    for (int w = 0; w < 4; w++) {
        zeros[w] = 0;
    }
    int row_vecs = vb / 8;
    for (int r = bid; r < rows; r += gridDim.x) {
        long long row_base = (long long)r * (long long)ldc;
        for (int c = tid_1; c < row_vecs; c += 256) {
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dzblk + (row_base + (long long)c * 8)))[0] = reinterpret_cast<int4*>(zeros)[0];
        }
        __syncthreads();
        int t = t0 + r;
        int r_lo = offsets[t];
        int r_hi = offsets[t + 1];
        int lo = ((r_lo > 0) ? r_lo : 0);
        lo = ((lo < M) ? lo : M);
        int hi = ((r_hi > lo) ? r_hi : lo);
        hi = ((hi < M) ? hi : M);
        for (int i = tid_1; i < hi - lo; i += 256) {
            int j = lo + i;
            int raw = ids[j];
            int v = ((raw > 0) ? raw : 0);
            v = ((v < V) ? v : V - 1);
            if (v >= v0 && v < v0 + vb) {
                dzblk[row_base + (long long)(v - v0)] = dz[j];
            }
        }
    }
}

} // extern "C"
