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
kernel_cake_causal_conv1d_train_86b5ecfbd90aca4b52e5(float* __restrict__ partial, float* __restrict__ dw, int C, int main_ctas, int n_ch_tiles, int ct_vec)
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
    int lane_1 = tid & 31;
    int ch = bid * 4 + (tid >> 5);
    if (ch < C) {
        long long C64 = (long long)C;
        int ctc = ch / 8 / ct_vec;
        int G = (main_ctas - ctc + n_ch_tiles - 1) / n_ch_tiles;
        int n_trips = G + 32 - 1 >> 5;
        float a4[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            a4[e] = 0.0f;
        }
        #pragma unroll 4
        for (int k = 0; k < n_trips; k++) {
            int r = lane_1 + (k << 5);
            if (r < G) {
                float _vec_load_0[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial + ((long long)r * C64 + (long long)ch) * 4);
                    _vec_load_0[0 + 0] = _v4.x;
                    _vec_load_0[0 + 1] = _v4.y;
                    _vec_load_0[0 + 2] = _v4.z;
                    _vec_load_0[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int e_1 = 0; e_1 < 4; e_1++) {
                    a4[e_1] = a4[e_1] + _vec_load_0[e_1];
                }
            }
        }
        #pragma unroll
        for (int e_2 = 0; e_2 < 4; e_2++) {
            float other = __shfl_xor_sync(0xFFFFFFFF, a4[e_2], 16);
            a4[e_2] = a4[e_2] + other;
        }
        #pragma unroll
        for (int e_3 = 0; e_3 < 4; e_3++) {
            float other_1 = __shfl_xor_sync(0xFFFFFFFF, a4[e_3], 8);
            a4[e_3] = a4[e_3] + other_1;
        }
        #pragma unroll
        for (int e_4 = 0; e_4 < 4; e_4++) {
            float other_2 = __shfl_xor_sync(0xFFFFFFFF, a4[e_4], 4);
            a4[e_4] = a4[e_4] + other_2;
        }
        #pragma unroll
        for (int e_5 = 0; e_5 < 4; e_5++) {
            float other_3 = __shfl_xor_sync(0xFFFFFFFF, a4[e_5], 2);
            a4[e_5] = a4[e_5] + other_3;
        }
        #pragma unroll
        for (int e_6 = 0; e_6 < 4; e_6++) {
            float other_4 = __shfl_xor_sync(0xFFFFFFFF, a4[e_6], 1);
            a4[e_6] = a4[e_6] + other_4;
        }
        if (lane_1 == 0) {
            {
                float4 _v4 = make_float4(a4[0 + 0], a4[0 + 1], a4[0 + 2], a4[0 + 3]);
                *reinterpret_cast<float4*>(dw + (long long)ch * 4) = _v4;
            }
        }
    }
}

} // extern "C"
