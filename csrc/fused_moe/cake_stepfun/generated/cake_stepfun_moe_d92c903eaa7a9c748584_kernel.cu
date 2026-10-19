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
#define SMEM_SCALE_SMEM_OFF 0
#define SMEM_SCALE_SMEM_STAGE_BYTES 256
#define SMEM_SCALE_SMEM_STRIDE 256
#define SMEM_IDX_SMEM_OFF 256
#define SMEM_IDX_SMEM_STAGE_BYTES 256
#define SMEM_IDX_SMEM_STRIDE 256
#define SMEM_TOTAL 0
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_stepfun_moe_d92c903eaa7a9c748584(const __nv_bfloat16* in_ptr, const float* expert_weights, __nv_bfloat16* out_ptr, const int* expanded_idx_to_permuted_idx, int hidden_dim, int hidden_dim_padded, int num_tokens, int top_k)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[512];

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* scale_smem = reinterpret_cast<float*>(smem_static_raw + 0);
    const int scale_smem_addr = (int)(unsigned long long)__cvta_generic_to_shared(scale_smem);
    int* idx_smem = reinterpret_cast<int*>(smem_static_raw + 256);
    const int idx_smem_addr = (int)(unsigned long long)__cvta_generic_to_shared(idx_smem);

    // === Task calls (dependency order) ===
    long long token_idx = (long long)bid;
    long long num_elems_in_padded_col = (long long)hidden_dim_padded / 8;
    long long num_elems_in_col = (long long)hidden_dim / 8;
    int num_chunks = top_k / 4;
    int block_dim_x = blockDim.x;
    long long stream_bytes = (long long)num_tokens;
    stream_bytes = stream_bytes * (long long)top_k;
    stream_bytes = stream_bytes * (long long)hidden_dim_padded;
    stream_bytes = stream_bytes * 2;
    for (int k_chunk = tid; k_chunk < num_chunks; k_chunk += block_dim_x) {
        int expanded_idx = (int)(token_idx * (long long)top_k) + k_chunk * 4;
        int _vec_load_0[4];
        {
            const int4* _ivptr_0 = reinterpret_cast<const int4*>(expanded_idx_to_permuted_idx + expanded_idx);
            int4 _ivld_0;
            _ivld_0 = *_ivptr_0;
            _vec_load_0[0 + 0] = _ivld_0.x;
            _vec_load_0[0 + 1] = _ivld_0.y;
            _vec_load_0[0 + 2] = _ivld_0.z;
            _vec_load_0[0 + 3] = _ivld_0.w;
        }
        float scale4[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(expert_weights + expanded_idx);
            scale4[0 + 0] = _v4.x;
            scale4[0 + 1] = _v4.y;
            scale4[0 + 2] = _v4.z;
            scale4[0 + 3] = _v4.w;
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            scale_smem[k_chunk * 4 + i] = scale4[i];
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; i_1++) {
            idx_smem[k_chunk * 4 + i_1] = _vec_load_0[i_1];
        }
    }
    long long offset = token_idx * (long long)hidden_dim;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    __syncthreads();
    if (stream_bytes >= 67108864) {
        for (int elem_index = tid; elem_index < (int)num_elems_in_col; elem_index += 256) {
            float thread_output[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                thread_output[j] = 0.0f;
            }
            for (int k_chunk_1 = 0; k_chunk_1 < num_chunks; k_chunk_1++) {
                int permuted[4];
                #pragma unroll
                for (int ki = 0; ki < 4; ki++) {
                    permuted[ki] = idx_smem[k_chunk_1 * 4 + ki];
                }
                float input_elems[32];
                #pragma unroll
                for (int ki_1 = 0; ki_1 < 4; ki_1++) {
                    if (permuted[ki_1] != -1) {
                        int permuted_idx = permuted[ki_1];
                        long long elem = (long long)permuted_idx * num_elems_in_padded_col + (long long)elem_index;
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(in_ptr + elem * 8);
                            uint4 _vld_2[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.nc.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_vld_2[_blk].x), "=r"(_vld_2[_blk].y), "=r"(_vld_2[_blk].z), "=r"(_vld_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)) : "memory");
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&input_elems[ki_1 * 8 + _blk * 8 + _pair * 2])[0]), "=f"((&input_elems[ki_1 * 8 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_2[_pair]));
                                }
                            }
                        }
                    }
                }
                float scale_f32[4];
                #pragma unroll
                for (int ki_2 = 0; ki_2 < 4; ki_2++) {
                    scale_f32[ki_2] = scale_smem[k_chunk_1 * 4 + ki_2];
                }
                #pragma unroll
                for (int ki_3 = 0; ki_3 < 4; ki_3++) {
                    if (permuted[ki_3] != -1) {
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 8; j_1++) {
                            float _fma_0;
                            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(scale_f32[ki_3]), "f"(input_elems[ki_3 * 8 + j_1]), "f"(thread_output[j_1]));
                            thread_output[j_1] = _fma_0;
                        }
                    }
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(thread_output[0 + 0], thread_output[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(thread_output[0 + 2], thread_output[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(thread_output[0 + 4], thread_output[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(thread_output[0 + 6], thread_output[0 + 7]);
                uint4 _st_v4_0 = *reinterpret_cast<uint4*>(&_pk[0]);
                asm volatile(
                    "st.global.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};"
                    :: "l"(&((__nv_bfloat16*)(out_ptr + (offset + (long long)elem_index * 8)))[0]), "r"(_st_v4_0.x), "r"(_st_v4_0.y), "r"(_st_v4_0.z), "r"(_st_v4_0.w) : "memory");
            }
        }
    } else {
        for (int elem_index_1 = tid; elem_index_1 < (int)num_elems_in_col; elem_index_1 += 256) {
            float thread_output_1[8];
            #pragma unroll
            for (int j_2 = 0; j_2 < 8; j_2++) {
                thread_output_1[j_2] = 0.0f;
            }
            for (int k_chunk_2 = 0; k_chunk_2 < num_chunks; k_chunk_2++) {
                int permuted_1[4];
                #pragma unroll
                for (int ki_4 = 0; ki_4 < 4; ki_4++) {
                    permuted_1[ki_4] = idx_smem[k_chunk_2 * 4 + ki_4];
                }
                float input_elems_1[32];
                #pragma unroll
                for (int ki_5 = 0; ki_5 < 4; ki_5++) {
                    if (permuted_1[ki_5] != -1) {
                        int permuted_idx_1 = permuted_1[ki_5];
                        long long elem_1 = (long long)permuted_idx_1 * num_elems_in_padded_col + (long long)elem_index_1;
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(in_ptr + elem_1 * 8);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&input_elems_1[ki_5 * 8 + _blk * 8 + _pair * 2])[0]), "=f"((&input_elems_1[ki_5 * 8 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_3[_pair]));
                                }
                            }
                        }
                    }
                }
                float scale_f32_1[4];
                #pragma unroll
                for (int ki_6 = 0; ki_6 < 4; ki_6++) {
                    scale_f32_1[ki_6] = scale_smem[k_chunk_2 * 4 + ki_6];
                }
                #pragma unroll
                for (int ki_7 = 0; ki_7 < 4; ki_7++) {
                    if (permuted_1[ki_7] != -1) {
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 8; j_3++) {
                            float _fma_1;
                            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(scale_f32_1[ki_7]), "f"(input_elems_1[ki_7 * 8 + j_3]), "f"(thread_output_1[j_3]));
                            thread_output_1[j_3] = _fma_1;
                        }
                    }
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(thread_output_1[0 + 0], thread_output_1[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(thread_output_1[0 + 2], thread_output_1[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(thread_output_1[0 + 4], thread_output_1[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(thread_output_1[0 + 6], thread_output_1[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out_ptr + (offset + (long long)elem_index_1 * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"
