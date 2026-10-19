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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_stepfun_moe_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_stepfun_moe_18f756d5d407f5ff75b3(const __nv_bfloat16* in_ptr, const __nv_bfloat16* expert_weights, __nv_bfloat16* out_ptr, const int* expanded_idx_to_permuted_idx, int hidden_dim, int hidden_dim_padded, int num_tokens, int top_k)
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
            float data_h = 0.0f;
            int perm_h[8];
            int perm_hk = 0;
            float scale_h[8];
            float value_h[8];
            perm_h[0] = expanded_idx_to_permuted_idx[token_idx * top_k];
            scale_h[0] = 0.0f;
            value_h[0] = 0.0f;
            perm_h[1] = expanded_idx_to_permuted_idx[token_idx * top_k + 1];
            scale_h[1] = 0.0f;
            value_h[1] = 0.0f;
            perm_h[2] = expanded_idx_to_permuted_idx[token_idx * top_k + 2];
            scale_h[2] = 0.0f;
            value_h[2] = 0.0f;
            perm_h[3] = expanded_idx_to_permuted_idx[token_idx * top_k + 3];
            scale_h[3] = 0.0f;
            value_h[3] = 0.0f;
            perm_h[4] = expanded_idx_to_permuted_idx[token_idx * top_k + 4];
            scale_h[4] = 0.0f;
            value_h[4] = 0.0f;
            perm_h[5] = expanded_idx_to_permuted_idx[token_idx * top_k + 5];
            scale_h[5] = 0.0f;
            value_h[5] = 0.0f;
            perm_h[6] = expanded_idx_to_permuted_idx[token_idx * top_k + 6];
            scale_h[6] = 0.0f;
            value_h[6] = 0.0f;
            perm_h[7] = expanded_idx_to_permuted_idx[token_idx * top_k + 7];
            scale_h[7] = 0.0f;
            value_h[7] = 0.0f;
            if (perm_h[0] != -1) {
                float _vec_load_0[1];
                {
                    __nv_bfloat16 _bf16_0 = *reinterpret_cast<const __nv_bfloat16*>(expert_weights + token_idx * top_k);
                    _vec_load_0[0] = __bfloat162float(_bf16_0);
                }
                scale_h[0] = _vec_load_0[0];
            }
            if (perm_h[1] != -1) {
                float _vec_load_1[1];
                {
                    __nv_bfloat16 _bf16_1 = *reinterpret_cast<const __nv_bfloat16*>(expert_weights + token_idx * top_k + 1);
                    _vec_load_1[0] = __bfloat162float(_bf16_1);
                }
                scale_h[1] = _vec_load_1[0];
            }
            if (perm_h[2] != -1) {
                float _vec_load_2[1];
                {
                    __nv_bfloat16 _bf16_2 = *reinterpret_cast<const __nv_bfloat16*>(expert_weights + token_idx * top_k + 2);
                    _vec_load_2[0] = __bfloat162float(_bf16_2);
                }
                scale_h[2] = _vec_load_2[0];
            }
            if (perm_h[3] != -1) {
                float _vec_load_3[1];
                {
                    __nv_bfloat16 _bf16_3 = *reinterpret_cast<const __nv_bfloat16*>(expert_weights + token_idx * top_k + 3);
                    _vec_load_3[0] = __bfloat162float(_bf16_3);
                }
                scale_h[3] = _vec_load_3[0];
            }
            if (perm_h[4] != -1) {
                float _vec_load_4[1];
                {
                    __nv_bfloat16 _bf16_4 = *reinterpret_cast<const __nv_bfloat16*>(expert_weights + token_idx * top_k + 4);
                    _vec_load_4[0] = __bfloat162float(_bf16_4);
                }
                scale_h[4] = _vec_load_4[0];
            }
            if (perm_h[5] != -1) {
                float _vec_load_5[1];
                {
                    __nv_bfloat16 _bf16_5 = *reinterpret_cast<const __nv_bfloat16*>(expert_weights + token_idx * top_k + 5);
                    _vec_load_5[0] = __bfloat162float(_bf16_5);
                }
                scale_h[5] = _vec_load_5[0];
            }
            if (perm_h[6] != -1) {
                float _vec_load_6[1];
                {
                    __nv_bfloat16 _bf16_6 = *reinterpret_cast<const __nv_bfloat16*>(expert_weights + token_idx * top_k + 6);
                    _vec_load_6[0] = __bfloat162float(_bf16_6);
                }
                scale_h[6] = _vec_load_6[0];
            }
            if (perm_h[7] != -1) {
                float _vec_load_7[1];
                {
                    __nv_bfloat16 _bf16_7 = *reinterpret_cast<const __nv_bfloat16*>(expert_weights + token_idx * top_k + 7);
                    _vec_load_7[0] = __bfloat162float(_bf16_7);
                }
                scale_h[7] = _vec_load_7[0];
            }
            if (perm_h[0] != -1) {
                perm_hk = perm_h[0];
                float _vec_load_8[1];
                {
                    __nv_bfloat16 _bf16_8 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + perm_hk * hidden_dim_padded + hidden_idx);
                    _vec_load_8[0] = __bfloat162float(_bf16_8);
                }
                value_h[0] = _vec_load_8[0];
            }
            if (perm_h[1] != -1) {
                perm_hk = perm_h[1];
                float _vec_load_9[1];
                {
                    __nv_bfloat16 _bf16_9 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + perm_hk * hidden_dim_padded + hidden_idx);
                    _vec_load_9[0] = __bfloat162float(_bf16_9);
                }
                value_h[1] = _vec_load_9[0];
            }
            if (perm_h[2] != -1) {
                perm_hk = perm_h[2];
                float _vec_load_10[1];
                {
                    __nv_bfloat16 _bf16_10 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + perm_hk * hidden_dim_padded + hidden_idx);
                    _vec_load_10[0] = __bfloat162float(_bf16_10);
                }
                value_h[2] = _vec_load_10[0];
            }
            if (perm_h[3] != -1) {
                perm_hk = perm_h[3];
                float _vec_load_11[1];
                {
                    __nv_bfloat16 _bf16_11 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + perm_hk * hidden_dim_padded + hidden_idx);
                    _vec_load_11[0] = __bfloat162float(_bf16_11);
                }
                value_h[3] = _vec_load_11[0];
            }
            if (perm_h[4] != -1) {
                perm_hk = perm_h[4];
                float _vec_load_12[1];
                {
                    __nv_bfloat16 _bf16_12 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + perm_hk * hidden_dim_padded + hidden_idx);
                    _vec_load_12[0] = __bfloat162float(_bf16_12);
                }
                value_h[4] = _vec_load_12[0];
            }
            if (perm_h[5] != -1) {
                perm_hk = perm_h[5];
                float _vec_load_13[1];
                {
                    __nv_bfloat16 _bf16_13 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + perm_hk * hidden_dim_padded + hidden_idx);
                    _vec_load_13[0] = __bfloat162float(_bf16_13);
                }
                value_h[5] = _vec_load_13[0];
            }
            if (perm_h[6] != -1) {
                perm_hk = perm_h[6];
                float _vec_load_14[1];
                {
                    __nv_bfloat16 _bf16_14 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + perm_hk * hidden_dim_padded + hidden_idx);
                    _vec_load_14[0] = __bfloat162float(_bf16_14);
                }
                value_h[6] = _vec_load_14[0];
            }
            if (perm_h[7] != -1) {
                perm_hk = perm_h[7];
                float _vec_load_15[1];
                {
                    __nv_bfloat16 _bf16_15 = *reinterpret_cast<const __nv_bfloat16*>(in_ptr + perm_hk * hidden_dim_padded + hidden_idx);
                    _vec_load_15[0] = __bfloat162float(_bf16_15);
                }
                value_h[7] = _vec_load_15[0];
            }
            if (perm_h[0] != -1) {
                float _fma_0;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(scale_h[0]), "f"(value_h[0]), "f"(data_h));
                data_h = _fma_0;
            }
            if (perm_h[1] != -1) {
                float _fma_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(scale_h[1]), "f"(value_h[1]), "f"(data_h));
                data_h = _fma_1;
            }
            if (perm_h[2] != -1) {
                float _fma_2;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(scale_h[2]), "f"(value_h[2]), "f"(data_h));
                data_h = _fma_2;
            }
            if (perm_h[3] != -1) {
                float _fma_3;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(scale_h[3]), "f"(value_h[3]), "f"(data_h));
                data_h = _fma_3;
            }
            if (perm_h[4] != -1) {
                float _fma_4;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(scale_h[4]), "f"(value_h[4]), "f"(data_h));
                data_h = _fma_4;
            }
            if (perm_h[5] != -1) {
                float _fma_5;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(scale_h[5]), "f"(value_h[5]), "f"(data_h));
                data_h = _fma_5;
            }
            if (perm_h[6] != -1) {
                float _fma_6;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(scale_h[6]), "f"(value_h[6]), "f"(data_h));
                data_h = _fma_6;
            }
            if (perm_h[7] != -1) {
                float _fma_7;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(scale_h[7]), "f"(value_h[7]), "f"(data_h));
                data_h = _fma_7;
            }
            *(reinterpret_cast<__nv_bfloat16*>(out_ptr + (token_idx * hidden_dim + hidden_idx)) + (0)) = __float2bfloat16_rn(data_h);
        }
    }
}

} // extern "C"
