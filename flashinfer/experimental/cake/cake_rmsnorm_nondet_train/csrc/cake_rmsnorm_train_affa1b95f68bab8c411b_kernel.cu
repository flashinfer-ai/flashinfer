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

__global__ __launch_bounds__(128) void
kernel_cake_rmsnorm_train_affa1b95f68bab8c411b(float* __restrict__ partial, float* __restrict__ dw, int n_chunks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* red_fold = reinterpret_cast<float*>(smem_raw + 0);
    const int red_fold_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int s = bid;
    int rsub = tid / 4;
    int cg = tid - rsub * 4;
    int col = s * 4 + cg;
    float a[1];
    float c[1];
    #pragma unroll
    for (int e = 0; e < 1; e++) {
        a[e] = 0.0f;
        c[e] = 0.0f;
    }
    int n_trips = (n_chunks + 31) / 32;
    int n_waves = (n_trips + 23) / 24;
    float cur[24];
    #pragma unroll
    for (int i = 0; i < 24; i++) {
        cur[i] = 0.0f;
    }
    #pragma unroll
    for (int d = 0; d < 24; d++) {
        int cc = rsub + d * 32;
        if (cc < n_chunks) {
            {
                cur[d] = *reinterpret_cast<const float*>(partial + ((unsigned long long)cc * 256ull + (unsigned long long)col));
            }
        }
    }
    for (int w = 0; w < n_waves; w++) {
        float nxt[24];
        #pragma unroll
        for (int i_1 = 0; i_1 < 24; i_1++) {
            nxt[i_1] = 0.0f;
        }
        #pragma unroll
        for (int d_1 = 0; d_1 < 24; d_1++) {
            int cc_1 = rsub + ((w + 1) * 24 + d_1) * 32;
            if (cc_1 < n_chunks) {
                {
                    nxt[d_1] = *reinterpret_cast<const float*>(partial + ((unsigned long long)cc_1 * 256ull + (unsigned long long)col));
                }
            }
        }
        #pragma unroll
        for (int d_2 = 0; d_2 < 24; d_2++) {
            int cc_2 = rsub + (w * 24 + d_2) * 32;
            if (cc_2 < n_chunks) {
                #pragma unroll
                for (int e_1 = 0; e_1 < 1; e_1++) {
                    float y = __fmaf_rn(cur[d_2 + e_1], 1.0f, -c[e_1]);
                    float t = a[e_1] + y;
                    c[e_1] = t - a[e_1] - y;
                    a[e_1] = t;
                }
            }
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 24; i_2++) {
            cur[i_2] = nxt[i_2];
        }
    }
    #pragma unroll
    for (int e_2 = 0; e_2 < 1; e_2++) {
        red_fold[rsub * 4 + cg + e_2] = a[e_2] - c[e_2];
    }
    __syncthreads();
    if (tid < 4) {
        float o4[1];
        float oc4[1];
        #pragma unroll
        for (int e_3 = 0; e_3 < 1; e_3++) {
            o4[e_3] = 0.0f;
            oc4[e_3] = 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            #pragma unroll
            for (int e_4 = 0; e_4 < 1; e_4++) {
                float d_3 = red_fold[k * 4 + tid + e_4];
                float y_1 = __fmaf_rn(d_3, 1.0f, -oc4[e_4]);
                float t_1 = o4[e_4] + y_1;
                oc4[e_4] = t_1 - o4[e_4] - y_1;
                o4[e_4] = t_1;
            }
        }
        #pragma unroll
        for (int e_5 = 0; e_5 < 1; e_5++) {
            o4[e_5] = o4[e_5] - oc4[e_5];
        }
        int ocol = s * 4 + tid;
        *(reinterpret_cast<float*>(dw) + ((unsigned long long)ocol)) = o4[0];
    }
    __syncthreads();
}

} // extern "C"
