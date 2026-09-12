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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) >= 64, "CUtensorMap CUDA ABI must be at least 64-byte aligned");
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
#define THREADS 32

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_cake_warp_decode_344b2efabee6538b991d(__nv_bfloat16* __restrict__ route_outputs, __nv_bfloat16* __restrict__ route_weights, int* __restrict__ route_slots, __nv_bfloat16* __restrict__ output, int top_k, int num_tokens, int route_stride, int M)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int feature = blockIdx.x * 32 + tid;
    int token = blockIdx.y;
    if (feature < M && token < num_tokens) {
        float value = 0.0f;
        int route_base = token * top_k;
        {
            int remainder = top_k & 3;
            int main_routes = top_k - remainder;
            #pragma unroll 4
            for (int route = 0; route < main_routes; route++) {
                int route_index = route_base + route;
                int output_row = route_index;
                __nv_bfloat16 partial = route_outputs[output_row * route_stride + feature];
                float weight = (float)route_weights[route_index];
                float _cvt_f32_0 = __bfloat162float(partial);
                float _fma_0 = __fmaf_rn(_cvt_f32_0, weight, value);
                value = _fma_0;
            }
            if (remainder >= 2) {
                int route_index_0 = route_base + main_routes;
                int output_row_0 = route_index_0;
                __nv_bfloat16 partial_0 = route_outputs[output_row_0 * route_stride + feature];
                float weight_0 = (float)route_weights[route_index_0];
                float _cvt_f32_1 = __bfloat162float(partial_0);
                float _fma_1 = __fmaf_rn(_cvt_f32_1, weight_0, value);
                value = _fma_1;
                int route_index_1 = route_index_0 + 1;
                int output_row_1 = route_index_1;
                __nv_bfloat16 partial_1 = route_outputs[output_row_1 * route_stride + feature];
                float weight_1 = (float)route_weights[route_index_1];
                float _cvt_f32_2 = __bfloat162float(partial_1);
                float _fma_2 = __fmaf_rn(_cvt_f32_2, weight_1, value);
                value = _fma_2;
            }
            if ((remainder & 1) != 0) {
                int route_index_2 = route_base + top_k - 1;
                int output_row_2 = route_index_2;
                __nv_bfloat16 partial_2 = route_outputs[output_row_2 * route_stride + feature];
                float weight_2 = (float)route_weights[route_index_2];
                float _cvt_f32_3 = __bfloat162float(partial_2);
                float _fma_3 = __fmaf_rn(_cvt_f32_3, weight_2, value);
                value = _fma_3;
            }
        }
        *(reinterpret_cast<__nv_bfloat16*>(output + (token * M + feature)) + (0)) = __float2bfloat16_rn(value);
    }
}

} // extern "C"
