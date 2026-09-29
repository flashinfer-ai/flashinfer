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


namespace nvfp4_qualified_c368_finalize {
#define kernel_alpha_moe_nvfp4_finalize_route_bf16_vector_seed_unit_scale_fma_packed_prefetch4 kernel_alpha_moe_nvfp4_finalize_route_bf16_vector_seed_unit_scale_fma_packed_prefetch4_nvfp4_qualified_c368_finalize
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
kernel_alpha_moe_nvfp4_finalize_route_bf16_vector_seed_unit_scale_fma_packed_prefetch4(__nv_bfloat16* __restrict__ route_accumulator, int* __restrict__ route_experts, float* __restrict__ output2_scale_scalar, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, float scaling_factor)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int element = (blockIdx.x * 256 + tid) * 8;
    if (element < M * K) {
        int token = element / K;
        int feature = element % K;
        float total[8];
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(out + element);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                }
            }
        }
        #pragma unroll
        for (int seed_component = 0; seed_component < 8; seed_component++) {
            total[seed_component] = _vec_load_0[seed_component];
        }
        #pragma unroll 1
        for (int route_base = 0; route_base < top_k; route_base += 4) {
            int active[4];
            float weights[4];
            unsigned int packed_values[16];
            #pragma unroll
            for (int preload_slot = 0; preload_slot < 4; preload_slot++) {
                active[preload_slot] = 0;
                if (route_base + preload_slot < top_k) {
                    int pair = token * top_k + route_base + preload_slot;
                    int expert = route_experts[pair];
                    if (expert >= 0) {
                        int _vec_load_1[4];
                        {
                            const int4* _ivptr_1 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(route_accumulator) + (pair * K + feature) / 2);
                            int4 _ivld_1;
                            _ivld_1 = *_ivptr_1;
                            _vec_load_1[0 + 0] = _ivld_1.x;
                            _vec_load_1[0 + 1] = _ivld_1.y;
                            _vec_load_1[0 + 2] = _ivld_1.z;
                            _vec_load_1[0 + 3] = _ivld_1.w;
                        }
                        weights[preload_slot] = topk_weights[pair];
                        #pragma unroll
                        for (int preload_word = 0; preload_word < 4; preload_word++) {
                            packed_values[preload_slot * 4 + preload_word] = (unsigned int)_vec_load_1[preload_word];
                        }
                        active[preload_slot] = 1;
                    }
                }
            }
            #pragma unroll
            for (int consume_slot = 0; consume_slot < 4; consume_slot++) {
                if (active[consume_slot] != 0) {
                    float route_weight = weights[consume_slot];
                    #pragma unroll
                    for (int component = 0; component < 8; component++) {
                        unsigned int packed_word = packed_values[consume_slot * 4 + component / 2];
                        unsigned int expanded_bits = 0;
                        if (component % 2 == 0) {
                            expanded_bits = packed_word << 16;
                        } else {
                            expanded_bits = packed_word & 4294901760;
                        }
                        float rounded_float = __uint_as_float(expanded_bits);
                        if (scaling_factor == 1.0f) {
                            float _fma_0 = __fmaf_rn(rounded_float, route_weight, total[component]);
                            total[component] = _fma_0;
                        } else {
                            float weighted = rounded_float * route_weight;
                            float _fma_1 = __fmaf_rn(weighted, scaling_factor, total[component]);
                            total[component] = _fma_1;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int output_component = 0; output_component < 8; output_component++) {
            out[element + output_component] = (__nv_bfloat16)total[output_component];
        }
    }
}

} // extern "C"


constexpr int kGeneratedThreads = THREADS;
constexpr int kGeneratedSmemTotal = 0;
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef kernel_alpha_moe_nvfp4_finalize_route_bf16_vector_seed_unit_scale_fma_packed_prefetch4
}  // namespace nvfp4_qualified_c368_finalize
