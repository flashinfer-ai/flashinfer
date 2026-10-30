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
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_eb11410b26f9701f723e(float* __restrict__ dg_part, float* __restrict__ gamma_dot, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, int num_chunks, int num_heads, float* __restrict__ dg_out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int item = blockIdx.x * 128 + tid;
    if (item < num_chunks * num_heads) {
        int chunk = item / num_heads;
        int head = item - chunk * num_heads;
        int clen = chunk_len[chunk];
        if (clen > 0) {
            long long hv64 = (long long)num_heads;
            long long ibase = (long long)chunk * 64 * hv64 + (long long)head;
            long long ebase = (long long)chunk_bos[chunk] * hv64 + (long long)head;
            float vals[64];
            vals[0] = 0.0f;
            vals[1] = 0.0f;
            vals[2] = 0.0f;
            vals[3] = 0.0f;
            vals[4] = 0.0f;
            vals[5] = 0.0f;
            vals[6] = 0.0f;
            vals[7] = 0.0f;
            vals[8] = 0.0f;
            vals[9] = 0.0f;
            vals[10] = 0.0f;
            vals[11] = 0.0f;
            vals[12] = 0.0f;
            vals[13] = 0.0f;
            vals[14] = 0.0f;
            vals[15] = 0.0f;
            vals[16] = 0.0f;
            vals[17] = 0.0f;
            vals[18] = 0.0f;
            vals[19] = 0.0f;
            vals[20] = 0.0f;
            vals[21] = 0.0f;
            vals[22] = 0.0f;
            vals[23] = 0.0f;
            vals[24] = 0.0f;
            vals[25] = 0.0f;
            vals[26] = 0.0f;
            vals[27] = 0.0f;
            vals[28] = 0.0f;
            vals[29] = 0.0f;
            vals[30] = 0.0f;
            vals[31] = 0.0f;
            vals[32] = 0.0f;
            vals[33] = 0.0f;
            vals[34] = 0.0f;
            vals[35] = 0.0f;
            vals[36] = 0.0f;
            vals[37] = 0.0f;
            vals[38] = 0.0f;
            vals[39] = 0.0f;
            vals[40] = 0.0f;
            vals[41] = 0.0f;
            vals[42] = 0.0f;
            vals[43] = 0.0f;
            vals[44] = 0.0f;
            vals[45] = 0.0f;
            vals[46] = 0.0f;
            vals[47] = 0.0f;
            vals[48] = 0.0f;
            vals[49] = 0.0f;
            vals[50] = 0.0f;
            vals[51] = 0.0f;
            vals[52] = 0.0f;
            vals[53] = 0.0f;
            vals[54] = 0.0f;
            vals[55] = 0.0f;
            vals[56] = 0.0f;
            vals[57] = 0.0f;
            vals[58] = 0.0f;
            vals[59] = 0.0f;
            vals[60] = 0.0f;
            vals[61] = 0.0f;
            vals[62] = 0.0f;
            vals[63] = 0.0f;
            #pragma unroll
            for (int s = 0; s < 64; s++) {
                if (clen > s) {
                    vals[s] = dg_part[ibase + (long long)s * hv64];
                }
            }
            float acc = gamma_dot[(long long)chunk * hv64 + (long long)head];
            #pragma unroll
            for (int u = 0; u < 64; u++) {
                const int s_1 = 63 - u;
                if (s_1 < clen) {
                    acc = acc + vals[s_1];
                    dg_out[ebase + (long long)s_1 * hv64] = acc;
                }
            }
        }
    }
}

} // extern "C"
