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
kernel_cake_stepfun_moe_a902bb6315e51b058e4c(const __nv_bfloat16* in_ptr, const float* expert_weights, __nv_bfloat16* out_ptr, const int* expanded_idx_to_permuted_idx, int hidden_dim, int hidden_dim_padded, int num_tokens, int top_k)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;
    int block_dim_x = blockDim.x;
    int grid_dim_x = gridDim.x;
    int grid_dim_y = gridDim.y;
    int hidden_start = tid + block_dim_x * block_idx_x;
    int hidden_stride = block_dim_x * grid_dim_x;
    for (int token_idx = block_idx_y; token_idx < num_tokens; token_idx += grid_dim_y) {
        for (int hidden_idx = hidden_start; hidden_idx < hidden_dim; hidden_idx += hidden_stride) {
            float data = 0.0f;
            for (int k = 0; k < top_k; k++) {
                int expanded_idx = token_idx * top_k + k;
                int permuted_idx = expanded_idx_to_permuted_idx[expanded_idx];
                if (permuted_idx != -1) {
                    float _vec_load_0[1];
                    {
                        _vec_load_0[0] = *reinterpret_cast<const float*>(expert_weights + expanded_idx);
                    }
                    float _vec_load_1[1];
                    {
                        __nv_bfloat16 _bf16_0 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + permuted_idx * hidden_dim_padded + hidden_idx);
                        _vec_load_1[0] = __bfloat162float(_bf16_0);
                    }
                    data = data + _vec_load_0[0] * _vec_load_1[0];
                }
            }
            *(reinterpret_cast<__nv_bfloat16*>(out_ptr + (token_idx * hidden_dim + hidden_idx)) + (0)) = __float2bfloat16_rn(data);
        }
    }
}

} // extern "C"
