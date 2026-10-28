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
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_warp_decode_a7c926e4f62a2e693b44(__nv_bfloat16* __restrict__ route_outputs, __nv_bfloat16* __restrict__ route_weights, int* __restrict__ route_slots, __nv_bfloat16* __restrict__ output, int top_k, int num_tokens, int route_stride, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int feature = blockIdx.x * 128 + tid;
    int token = blockIdx.y;
    int route_base = token * 4;
    float weights[4];
    int rows[4];
    if (feature < M && token < num_tokens) {
        weights[0] = (float)route_weights[route_base + 0];
        weights[1] = (float)route_weights[route_base + 1];
        weights[2] = (float)route_weights[route_base + 2];
        weights[3] = (float)route_weights[route_base + 3];
        #pragma unroll
        for (int slot = 0; slot < 4; slot++) {
            rows[slot] = route_slots[route_base + slot];
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (feature < M && token < num_tokens) {
        __nv_bfloat16 partials[4];
        partials[0] = route_outputs[rows[0] * route_stride + feature];
        partials[1] = route_outputs[rows[1] * route_stride + feature];
        partials[2] = route_outputs[rows[2] * route_stride + feature];
        partials[3] = route_outputs[rows[3] * route_stride + feature];
        float value = 0.0f;
        value = __fmaf_rn((float)partials[0], weights[0], value);
        value = __fmaf_rn((float)partials[1], weights[1], value);
        value = __fmaf_rn((float)partials[2], weights[2], value);
        value = __fmaf_rn((float)partials[3], weights[3], value);
        *(reinterpret_cast<__nv_bfloat16*>(output + (token * M + feature)) + (0)) = (__nv_bfloat16)value;
    }
}

} // extern "C"
