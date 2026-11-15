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
#define NUM_MAIN_1_STAGES 1
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_fp8_attention_8f4172c0adf623a761c9(float* __restrict__ partials, float* __restrict__ mean_k, float* __restrict__ v_scale, int seqlen, int num_tiles)
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
    int chan = tid & 127;
    int is_max = tid >> 7;
    int base = bid * num_tiles * 256 + is_max * 128 + chan;
    float acc = 0.0f;
    #pragma unroll 8
    for (int tile = 0; tile < num_tiles; tile++) {
        float value = partials[base + tile * 256];
        acc = ((is_max == 0) ? acc + value : fmaxf(acc, value));
    }
    if (is_max == 0) {
        float _fdiv_0 = __fdiv_rn(acc, (float)seqlen);
        *(reinterpret_cast<float*>(mean_k + (bid * 128 + chan)) + (0)) = _fdiv_0;
    } else {
        float _fdiv_1 = __fdiv_rn(fmaxf(acc, 1e-12f), 448.0f);
        *(reinterpret_cast<float*>(v_scale + (bid * 128 + chan)) + (0)) = _fdiv_1;
    }
}

} // extern "C"
