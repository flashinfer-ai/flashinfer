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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

namespace nvfp4_qualified_c346_finalize {
#define kernel_alpha_moe_nvfp4_finalize_bf16_routes_scalar256_top8 kernel_alpha_moe_nvfp4_finalize_bf16_routes_scalar256_top8_nvfp4_qualified_c346_finalize
__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_alpha_moe_nvfp4_finalize_bf16_routes_scalar256_top8(__nv_bfloat16* __restrict__ route_accumulator, int* __restrict__ route_experts, float* __restrict__ output2_scale_scalar, float* __restrict__ topk_weights, float* __restrict__ initial_out, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, float scaling_factor)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int element = blockIdx.x * 256 + tid;
    if (element < M * K) {
        int token = element / K;
        int feature = element % K;
        float total = initial_out[element];
        #pragma unroll
        for (int route_slot = 0; route_slot < 8; route_slot++) {
            int pair = token * top_k + route_slot;
            int expert = route_experts[pair];
            if (expert >= 0) {
                __nv_bfloat16 rounded_down = route_accumulator[pair * K + feature];
                float rounded_float = (float)rounded_down;
                float weighted = rounded_float * topk_weights[pair] * scaling_factor;
                total = total + weighted;
            }
        }
        out[element] = (__nv_bfloat16)total;
    }
}

} // extern "C"


constexpr int kGeneratedThreads = THREADS;
constexpr int kGeneratedSmemTotal = 0;
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef kernel_alpha_moe_nvfp4_finalize_bf16_routes_scalar256_top8
}  // namespace nvfp4_qualified_c346_finalize
