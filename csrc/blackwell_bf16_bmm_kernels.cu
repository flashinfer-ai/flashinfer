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

// Frozen CAKE-generated Loom BF16 BMM dispatcher for NVIDIA GB300 (sm_103a).
// Source commit: 850c3b728d731c9f201c5dc5aad5d1ee51156f57
// Source SHA256: 7fd8f0f03ae9d1026cd9fc69e0fda8c94aa13e3f55ec0265d4d066a8fdc120fe
//
// This file mechanically concatenates 13 generated translation units. The
// shared generated prelude is retained once; every kernel body is unchanged,
// and section-local generated macros are undefined before the next section.
// clang-format off
#include <cstdint>
#include <cuda_bf16.h>
#include <flashinfer/gemm/blackwell_bf16_bmm.cuh>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}


// ---- generic-k64-m16n32 (blackwell_bf16_bmm_sm103_k64.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 2048
#define SMEM_SMEM_A_STRIDE 2048
#define SMEM_SMEM_B_OFF 2048
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 4096
#define SMEM_TOTAL 6144
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k64(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_b_addr = smem + 2048;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = M;
    int effective_n = N;
    int effective_a_stride_b = a_stride_b;
    int effective_a_stride_m = a_stride_m;
    int effective_a_stride_k = a_stride_k;
    int effective_b_stride_b = b_stride_b;
    int effective_b_stride_n = b_stride_n;
    int effective_b_stride_k = b_stride_k;
    int effective_out_type = out_type;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 1; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 128) {
            int copy_row = copy_idx / 8;
            int copy_chunk = copy_idx % 8;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(a_dst), "l"(A + a_src), "r"((effective_m > m_base + copy_row) ? 16 : 0));
            }
        }
    }
    #pragma unroll 4
    for (int copy_iter_b = 0; copy_iter_b < 2; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 256) {
            int copy_row_b = copy_idx_b / 8;
            int copy_chunk_b = copy_idx_b % 8;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(b_dst), "l"(B_storage + b_src), "r"((effective_n > n_base + copy_row_b) ? 16 : 0));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 4
    for (int k_atom = 0; k_atom < 4; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        if (effective_m > m_base + 15 && effective_n > n_warp_base + 7) {
            #pragma unroll
            for (int frag_row = 0; frag_row < 2; frag_row++) {
                int m_idx = m_base + lane / 4 + frag_row * 8;
                int n_idx = n_warp_base + 2 * (lane % 4);
                int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
                const int value_idx = frag_row * 2;
                if (effective_out_type == 0) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                    }
                } else if (effective_out_type == 1) {
                    *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                    *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
                } else {
                    {
                        float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                        *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int frag_row_1 = 0; frag_row_1 < 2; frag_row_1++) {
                int m_idx_1 = m_base + lane / 4 + frag_row_1 * 8;
                int n_idx_1 = n_warp_base + 2 * (lane % 4);
                if (m_idx_1 < effective_m && n_idx_1 < effective_n) {
                    int output_idx_1 = (batch_idx * effective_m + m_idx_1) * effective_n + n_idx_1;
                    const int value_idx_1 = frag_row_1 * 2;
                    if (effective_n > n_idx_1 + 1) {
                        if (effective_out_type == 0) {
                            {
                                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx_1 + 0], accum[value_idx_1 + 1]);
                                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2)))[0]) = _pk;
                            }
                        } else if (effective_out_type == 1) {
                            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum[value_idx_1]);
                            *(reinterpret_cast<__half*>(out_bytes + ((output_idx_1 + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx_1 + 1]);
                        } else {
                            {
                                float2 _v2 = make_float2(accum[value_idx_1 + 0], accum[value_idx_1 + 1]);
                                *reinterpret_cast<float2*>(out_bytes + (output_idx_1 * 4) + 0) = _v2;
                            }
                        }
                    } else if (effective_out_type == 0) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2bfloat16_rn(accum[value_idx_1]);
                    } else {
                        if (effective_out_type == 1) {
                            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum[value_idx_1]);
                        } else {
                            *(reinterpret_cast<float*>(out_bytes + (output_idx_1 * 4)) + (0)) = accum[value_idx_1];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- generic-k256-m16n32 (blackwell_bf16_bmm_sm103_k256.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 8192
#define SMEM_SMEM_B_OFF 8192
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_TOTAL 24576
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int smem_b_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = M;
    int effective_n = N;
    int effective_a_stride_b = a_stride_b;
    int effective_a_stride_m = a_stride_m;
    int effective_a_stride_k = a_stride_k;
    int effective_b_stride_b = b_stride_b;
    int effective_b_stride_n = b_stride_n;
    int effective_b_stride_k = b_stride_k;
    int effective_out_type = out_type;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 512) {
            int copy_row = copy_idx / 32;
            int copy_chunk = copy_idx % 32;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(a_dst), "l"(A + a_src), "r"((effective_m > m_base + copy_row) ? 16 : 0));
            }
        }
    }
    #pragma unroll 8
    for (int copy_iter_b = 0; copy_iter_b < 8; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 1024) {
            int copy_row_b = copy_idx_b / 32;
            int copy_chunk_b = copy_idx_b % 32;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(b_dst), "l"(B_storage + b_src), "r"((effective_n > n_base + copy_row_b) ? 16 : 0));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 16
    for (int k_atom = 0; k_atom < 16; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        if (effective_m > m_base + 15 && effective_n > n_warp_base + 7) {
            #pragma unroll
            for (int frag_row = 0; frag_row < 2; frag_row++) {
                int m_idx = m_base + lane / 4 + frag_row * 8;
                int n_idx = n_warp_base + 2 * (lane % 4);
                int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
                const int value_idx = frag_row * 2;
                if (effective_out_type == 0) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                    }
                } else if (effective_out_type == 1) {
                    *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                    *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
                } else {
                    {
                        float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                        *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int frag_row_1 = 0; frag_row_1 < 2; frag_row_1++) {
                int m_idx_1 = m_base + lane / 4 + frag_row_1 * 8;
                int n_idx_1 = n_warp_base + 2 * (lane % 4);
                if (m_idx_1 < effective_m && n_idx_1 < effective_n) {
                    int output_idx_1 = (batch_idx * effective_m + m_idx_1) * effective_n + n_idx_1;
                    const int value_idx_1 = frag_row_1 * 2;
                    if (effective_n > n_idx_1 + 1) {
                        if (effective_out_type == 0) {
                            {
                                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx_1 + 0], accum[value_idx_1 + 1]);
                                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2)))[0]) = _pk;
                            }
                        } else if (effective_out_type == 1) {
                            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum[value_idx_1]);
                            *(reinterpret_cast<__half*>(out_bytes + ((output_idx_1 + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx_1 + 1]);
                        } else {
                            {
                                float2 _v2 = make_float2(accum[value_idx_1 + 0], accum[value_idx_1 + 1]);
                                *reinterpret_cast<float2*>(out_bytes + (output_idx_1 * 4) + 0) = _v2;
                            }
                        }
                    } else if (effective_out_type == 0) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2bfloat16_rn(accum[value_idx_1]);
                    } else {
                        if (effective_out_type == 1) {
                            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum[value_idx_1]);
                        } else {
                            *(reinterpret_cast<float*>(out_bytes + (output_idx_1 * 4)) + (0)) = accum[value_idx_1];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- generic-k1024-m16n32 (blackwell_bf16_bmm_sm103_k1024.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 32768
#define SMEM_SMEM_B_STAGE_BYTES 65536
#define SMEM_SMEM_B_STRIDE 65536
#define SMEM_TOTAL 98304
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int smem_b_addr = smem + 32768;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = M;
    int effective_n = N;
    int effective_a_stride_b = a_stride_b;
    int effective_a_stride_m = a_stride_m;
    int effective_a_stride_k = a_stride_k;
    int effective_b_stride_b = b_stride_b;
    int effective_b_stride_n = b_stride_n;
    int effective_b_stride_k = b_stride_k;
    int effective_out_type = out_type;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 16; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 2048) {
            int copy_row = copy_idx / 128;
            int copy_chunk = copy_idx % 128;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(a_dst), "l"(A + a_src), "r"((effective_m > m_base + copy_row) ? 16 : 0));
            }
        }
    }
    #pragma unroll 4
    for (int copy_iter_b = 0; copy_iter_b < 32; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 4096) {
            int copy_row_b = copy_idx_b / 128;
            int copy_chunk_b = copy_idx_b % 128;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(b_dst), "l"(B_storage + b_src), "r"((effective_n > n_base + copy_row_b) ? 16 : 0));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 64
    for (int k_atom = 0; k_atom < 64; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        if (effective_m > m_base + 15 && effective_n > n_warp_base + 7) {
            #pragma unroll
            for (int frag_row = 0; frag_row < 2; frag_row++) {
                int m_idx = m_base + lane / 4 + frag_row * 8;
                int n_idx = n_warp_base + 2 * (lane % 4);
                int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
                const int value_idx = frag_row * 2;
                if (effective_out_type == 0) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                    }
                } else if (effective_out_type == 1) {
                    *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                    *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
                } else {
                    {
                        float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                        *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int frag_row_1 = 0; frag_row_1 < 2; frag_row_1++) {
                int m_idx_1 = m_base + lane / 4 + frag_row_1 * 8;
                int n_idx_1 = n_warp_base + 2 * (lane % 4);
                if (m_idx_1 < effective_m && n_idx_1 < effective_n) {
                    int output_idx_1 = (batch_idx * effective_m + m_idx_1) * effective_n + n_idx_1;
                    const int value_idx_1 = frag_row_1 * 2;
                    if (effective_n > n_idx_1 + 1) {
                        if (effective_out_type == 0) {
                            {
                                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx_1 + 0], accum[value_idx_1 + 1]);
                                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2)))[0]) = _pk;
                            }
                        } else if (effective_out_type == 1) {
                            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum[value_idx_1]);
                            *(reinterpret_cast<__half*>(out_bytes + ((output_idx_1 + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx_1 + 1]);
                        } else {
                            {
                                float2 _v2 = make_float2(accum[value_idx_1 + 0], accum[value_idx_1 + 1]);
                                *reinterpret_cast<float2*>(out_bytes + (output_idx_1 * 4) + 0) = _v2;
                            }
                        }
                    } else if (effective_out_type == 0) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2bfloat16_rn(accum[value_idx_1]);
                    } else {
                        if (effective_out_type == 1) {
                            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum[value_idx_1]);
                        } else {
                            *(reinterpret_cast<float*>(out_bytes + (output_idx_1 * 4)) + (0)) = accum[value_idx_1];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b16-m128-n80-k256-m32n40-bf16 (blackwell_bf16_bmm_sm103_k256_m32n40_bf16.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 16384
#define SMEM_SMEM_B_STAGE_BYTES 20480
#define SMEM_SMEM_B_STRIDE 20480
#define SMEM_TOTAL 36864
#define THREADS 160

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(160, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o0_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int smem_b_addr = smem + 16384;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 32;
    int n_base = blockIdx.y * 40;
    const int fixed_m = 128;
    const int fixed_n = 80;
    const int fixed_out_type = 0;
    float accum_lo[4];
    float accum_hi[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum_lo[acc_idx] = 0.0f;
        accum_hi[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 7; copy_iter++) {
        int copy_idx = copy_iter * 160 + tid;
        if (copy_idx < 1024) {
            int copy_row = copy_idx / 32;
            int copy_chunk = copy_idx % 32;
            int a_src = batch_idx * 32768 + (m_base + copy_row) * 256 + copy_chunk * 8;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"((smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 32 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 32 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(A + a_src));
        }
    }
    #pragma unroll 8
    for (int copy_iter_b = 0; copy_iter_b < 8; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 160 + tid;
        if (copy_idx_b < 1280) {
            int copy_row_b = copy_idx_b / 32;
            int copy_chunk_b = copy_idx_b % 32;
            int b_src = batch_idx * 20480 + (n_base + copy_row_b) * 256 + copy_chunk_b * 8;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"((smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 40 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 40 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(B_storage + b_src));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 160;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 16
    for (int k_atom = 0; k_atom < 16; k_atom++) {
        unsigned int a_frag_lo[4];
        unsigned int a_frag_hi[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 4096;
        unsigned int b_group_base = base_b + k_group * 5120;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_lo[0]), "=r"(a_frag_lo[1]), "=r"(a_frag_lo[2]), "=r"(a_frag_lo[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
            : "r"(a_group_base + (row_a + 16) * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum_lo[0]), "+f"(accum_lo[1]), "+f"(accum_lo[2]), "+f"(accum_lo[3])
            : "r"(a_frag_lo[0]), "r"(a_frag_lo[1]), "r"(a_frag_lo[2]), "r"(a_frag_lo[3]), "r"(b_frag[0]), "r"(b_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum_hi[0]), "+f"(accum_hi[1]), "+f"(accum_hi[2]), "+f"(accum_hi[3])
            : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    #pragma unroll
    for (int frag_row = 0; frag_row < 2; frag_row++) {
        int m_idx = m_base + lane / 4 + frag_row * 8;
        int n_idx = n_warp_base + 2 * (lane % 4);
        int output_idx = (batch_idx * fixed_m + m_idx) * fixed_n + n_idx;
        const int value_idx = frag_row * 2;
        if (fixed_out_type == 0) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
            }
        } else if (fixed_out_type == 1) {
            *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum_lo[value_idx]);
            *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum_lo[value_idx + 1]);
        } else {
            {
                float2 _v2 = make_float2(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
            }
        }
    }
    #pragma unroll
    for (int frag_row_1 = 0; frag_row_1 < 2; frag_row_1++) {
        int m_idx_1 = m_base + 16 + lane / 4 + frag_row_1 * 8;
        int n_idx_1 = n_warp_base + 2 * (lane % 4);
        int output_idx_1 = (batch_idx * fixed_m + m_idx_1) * fixed_n + n_idx_1;
        const int value_idx_1 = frag_row_1 * 2;
        if (fixed_out_type == 0) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2)))[0]) = _pk;
            }
        } else if (fixed_out_type == 1) {
            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum_hi[value_idx_1]);
            *(reinterpret_cast<__half*>(out_bytes + ((output_idx_1 + 1) * 2)) + (0)) = __float2half_rn(accum_hi[value_idx_1 + 1]);
        } else {
            {
                float2 _v2 = make_float2(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                *reinterpret_cast<float2*>(out_bytes + (output_idx_1 * 4) + 0) = _v2;
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b16-m128-n80-k256-m32n40-fp16 (blackwell_bf16_bmm_sm103_k256_m32n40_fp16.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 16384
#define SMEM_SMEM_B_STAGE_BYTES 20480
#define SMEM_SMEM_B_STRIDE 20480
#define SMEM_TOTAL 36864
#define THREADS 160

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(160, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o1_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int smem_b_addr = smem + 16384;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 32;
    int n_base = blockIdx.y * 40;
    const int fixed_m = 128;
    const int fixed_n = 80;
    const int fixed_out_type = 1;
    float accum_lo[4];
    float accum_hi[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum_lo[acc_idx] = 0.0f;
        accum_hi[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 7; copy_iter++) {
        int copy_idx = copy_iter * 160 + tid;
        if (copy_idx < 1024) {
            int copy_row = copy_idx / 32;
            int copy_chunk = copy_idx % 32;
            int a_src = batch_idx * 32768 + (m_base + copy_row) * 256 + copy_chunk * 8;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"((smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 32 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 32 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(A + a_src));
        }
    }
    #pragma unroll 8
    for (int copy_iter_b = 0; copy_iter_b < 8; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 160 + tid;
        if (copy_idx_b < 1280) {
            int copy_row_b = copy_idx_b / 32;
            int copy_chunk_b = copy_idx_b % 32;
            int b_src = batch_idx * 20480 + (n_base + copy_row_b) * 256 + copy_chunk_b * 8;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"((smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 40 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 40 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(B_storage + b_src));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 160;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 16
    for (int k_atom = 0; k_atom < 16; k_atom++) {
        unsigned int a_frag_lo[4];
        unsigned int a_frag_hi[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 4096;
        unsigned int b_group_base = base_b + k_group * 5120;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_lo[0]), "=r"(a_frag_lo[1]), "=r"(a_frag_lo[2]), "=r"(a_frag_lo[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
            : "r"(a_group_base + (row_a + 16) * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum_lo[0]), "+f"(accum_lo[1]), "+f"(accum_lo[2]), "+f"(accum_lo[3])
            : "r"(a_frag_lo[0]), "r"(a_frag_lo[1]), "r"(a_frag_lo[2]), "r"(a_frag_lo[3]), "r"(b_frag[0]), "r"(b_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum_hi[0]), "+f"(accum_hi[1]), "+f"(accum_hi[2]), "+f"(accum_hi[3])
            : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    #pragma unroll
    for (int frag_row = 0; frag_row < 2; frag_row++) {
        int m_idx = m_base + lane / 4 + frag_row * 8;
        int n_idx = n_warp_base + 2 * (lane % 4);
        int output_idx = (batch_idx * fixed_m + m_idx) * fixed_n + n_idx;
        const int value_idx = frag_row * 2;
        if (fixed_out_type == 0) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
            }
        } else if (fixed_out_type == 1) {
            *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum_lo[value_idx]);
            *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum_lo[value_idx + 1]);
        } else {
            {
                float2 _v2 = make_float2(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
            }
        }
    }
    #pragma unroll
    for (int frag_row_1 = 0; frag_row_1 < 2; frag_row_1++) {
        int m_idx_1 = m_base + 16 + lane / 4 + frag_row_1 * 8;
        int n_idx_1 = n_warp_base + 2 * (lane % 4);
        int output_idx_1 = (batch_idx * fixed_m + m_idx_1) * fixed_n + n_idx_1;
        const int value_idx_1 = frag_row_1 * 2;
        if (fixed_out_type == 0) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2)))[0]) = _pk;
            }
        } else if (fixed_out_type == 1) {
            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum_hi[value_idx_1]);
            *(reinterpret_cast<__half*>(out_bytes + ((output_idx_1 + 1) * 2)) + (0)) = __float2half_rn(accum_hi[value_idx_1 + 1]);
        } else {
            {
                float2 _v2 = make_float2(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                *reinterpret_cast<float2*>(out_bytes + (output_idx_1 * 4) + 0) = _v2;
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b16-m128-n80-k256-m32n40-fp32 (blackwell_bf16_bmm_sm103_k256_m32n40_fp32.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 16384
#define SMEM_SMEM_B_STAGE_BYTES 20480
#define SMEM_SMEM_B_STRIDE 20480
#define SMEM_TOTAL 36864
#define THREADS 160

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(160, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o2_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int smem_b_addr = smem + 16384;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 32;
    int n_base = blockIdx.y * 40;
    const int fixed_m = 128;
    const int fixed_n = 80;
    const int fixed_out_type = 2;
    float accum_lo[4];
    float accum_hi[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum_lo[acc_idx] = 0.0f;
        accum_hi[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 7; copy_iter++) {
        int copy_idx = copy_iter * 160 + tid;
        if (copy_idx < 1024) {
            int copy_row = copy_idx / 32;
            int copy_chunk = copy_idx % 32;
            int a_src = batch_idx * 32768 + (m_base + copy_row) * 256 + copy_chunk * 8;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"((smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 32 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 32 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(A + a_src));
        }
    }
    #pragma unroll 8
    for (int copy_iter_b = 0; copy_iter_b < 8; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 160 + tid;
        if (copy_idx_b < 1280) {
            int copy_row_b = copy_idx_b / 32;
            int copy_chunk_b = copy_idx_b % 32;
            int b_src = batch_idx * 20480 + (n_base + copy_row_b) * 256 + copy_chunk_b * 8;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"((smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 40 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 40 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(B_storage + b_src));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 160;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 16
    for (int k_atom = 0; k_atom < 16; k_atom++) {
        unsigned int a_frag_lo[4];
        unsigned int a_frag_hi[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 4096;
        unsigned int b_group_base = base_b + k_group * 5120;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_lo[0]), "=r"(a_frag_lo[1]), "=r"(a_frag_lo[2]), "=r"(a_frag_lo[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
            : "r"(a_group_base + (row_a + 16) * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum_lo[0]), "+f"(accum_lo[1]), "+f"(accum_lo[2]), "+f"(accum_lo[3])
            : "r"(a_frag_lo[0]), "r"(a_frag_lo[1]), "r"(a_frag_lo[2]), "r"(a_frag_lo[3]), "r"(b_frag[0]), "r"(b_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum_hi[0]), "+f"(accum_hi[1]), "+f"(accum_hi[2]), "+f"(accum_hi[3])
            : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    #pragma unroll
    for (int frag_row = 0; frag_row < 2; frag_row++) {
        int m_idx = m_base + lane / 4 + frag_row * 8;
        int n_idx = n_warp_base + 2 * (lane % 4);
        int output_idx = (batch_idx * fixed_m + m_idx) * fixed_n + n_idx;
        const int value_idx = frag_row * 2;
        if (fixed_out_type == 0) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
            }
        } else if (fixed_out_type == 1) {
            *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum_lo[value_idx]);
            *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum_lo[value_idx + 1]);
        } else {
            {
                float2 _v2 = make_float2(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
            }
        }
    }
    #pragma unroll
    for (int frag_row_1 = 0; frag_row_1 < 2; frag_row_1++) {
        int m_idx_1 = m_base + 16 + lane / 4 + frag_row_1 * 8;
        int n_idx_1 = n_warp_base + 2 * (lane % 4);
        int output_idx_1 = (batch_idx * fixed_m + m_idx_1) * fixed_n + n_idx_1;
        const int value_idx_1 = frag_row_1 * 2;
        if (fixed_out_type == 0) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2)))[0]) = _pk;
            }
        } else if (fixed_out_type == 1) {
            *(reinterpret_cast<__half*>(out_bytes + (output_idx_1 * 2)) + (0)) = __float2half_rn(accum_hi[value_idx_1]);
            *(reinterpret_cast<__half*>(out_bytes + ((output_idx_1 + 1) * 2)) + (0)) = __float2half_rn(accum_hi[value_idx_1 + 1]);
        } else {
            {
                float2 _v2 = make_float2(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                *reinterpret_cast<float2*>(out_bytes + (output_idx_1 * 4) + 0) = _v2;
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b16-m128-n64-k256-fixed-bf16 (blackwell_bf16_bmm_sm103_k256_m128n64_bf16.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 8192
#define SMEM_SMEM_B_OFF 8192
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_TOTAL 24576
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o0_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int smem_b_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = 128;
    int effective_n = 64;
    int effective_a_stride_b = 32768;
    int effective_a_stride_m = 256;
    int effective_a_stride_k = 1;
    int effective_b_stride_b = 16384;
    int effective_b_stride_n = 256;
    int effective_b_stride_k = 1;
    int effective_out_type = 0;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 512) {
            int copy_row = copy_idx / 32;
            int copy_chunk = copy_idx % 32;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(a_dst), "l"(A + a_src));
            }
        }
    }
    #pragma unroll 8
    for (int copy_iter_b = 0; copy_iter_b < 8; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 1024) {
            int copy_row_b = copy_idx_b / 32;
            int copy_chunk_b = copy_idx_b % 32;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(b_dst), "l"(B_storage + b_src));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 16
    for (int k_atom = 0; k_atom < 16; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        #pragma unroll
        for (int frag_row = 0; frag_row < 2; frag_row++) {
            int m_idx = m_base + lane / 4 + frag_row * 8;
            int n_idx = n_warp_base + 2 * (lane % 4);
            int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
            const int value_idx = frag_row * 2;
            if (effective_out_type == 0) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                }
            } else if (effective_out_type == 1) {
                *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
            } else {
                {
                    float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b16-m128-n64-k256-fixed-fp16 (blackwell_bf16_bmm_sm103_k256_m128n64_fp16.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 8192
#define SMEM_SMEM_B_OFF 8192
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_TOTAL 24576
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o1_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int smem_b_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = 128;
    int effective_n = 64;
    int effective_a_stride_b = 32768;
    int effective_a_stride_m = 256;
    int effective_a_stride_k = 1;
    int effective_b_stride_b = 16384;
    int effective_b_stride_n = 256;
    int effective_b_stride_k = 1;
    int effective_out_type = 1;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 512) {
            int copy_row = copy_idx / 32;
            int copy_chunk = copy_idx % 32;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(a_dst), "l"(A + a_src));
            }
        }
    }
    #pragma unroll 8
    for (int copy_iter_b = 0; copy_iter_b < 8; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 1024) {
            int copy_row_b = copy_idx_b / 32;
            int copy_chunk_b = copy_idx_b % 32;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(b_dst), "l"(B_storage + b_src));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 16
    for (int k_atom = 0; k_atom < 16; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        #pragma unroll
        for (int frag_row = 0; frag_row < 2; frag_row++) {
            int m_idx = m_base + lane / 4 + frag_row * 8;
            int n_idx = n_warp_base + 2 * (lane % 4);
            int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
            const int value_idx = frag_row * 2;
            if (effective_out_type == 0) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                }
            } else if (effective_out_type == 1) {
                *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
            } else {
                {
                    float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b16-m128-n64-k256-fixed-fp32 (blackwell_bf16_bmm_sm103_k256_m128n64_fp32.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 8192
#define SMEM_SMEM_B_OFF 8192
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_TOTAL 24576
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o2_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int smem_b_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = 128;
    int effective_n = 64;
    int effective_a_stride_b = 32768;
    int effective_a_stride_m = 256;
    int effective_a_stride_k = 1;
    int effective_b_stride_b = 16384;
    int effective_b_stride_n = 256;
    int effective_b_stride_k = 1;
    int effective_out_type = 2;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 512) {
            int copy_row = copy_idx / 32;
            int copy_chunk = copy_idx % 32;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(a_dst), "l"(A + a_src));
            }
        }
    }
    #pragma unroll 8
    for (int copy_iter_b = 0; copy_iter_b < 8; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 1024) {
            int copy_row_b = copy_idx_b / 32;
            int copy_chunk_b = copy_idx_b % 32;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(b_dst), "l"(B_storage + b_src));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 16
    for (int k_atom = 0; k_atom < 16; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        #pragma unroll
        for (int frag_row = 0; frag_row < 2; frag_row++) {
            int m_idx = m_base + lane / 4 + frag_row * 8;
            int n_idx = n_warp_base + 2 * (lane % 4);
            int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
            const int value_idx = frag_row * 2;
            if (effective_out_type == 0) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                }
            } else if (effective_out_type == 1) {
                *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
            } else {
                {
                    float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b4-m16-n1024-k1024-fixed-bf16 (blackwell_bf16_bmm_sm103_k1024_m16n1024_bf16.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 32768
#define SMEM_SMEM_B_STAGE_BYTES 65536
#define SMEM_SMEM_B_STRIDE 65536
#define SMEM_TOTAL 98304
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o0_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int smem_b_addr = smem + 32768;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = 16;
    int effective_n = 1024;
    int effective_a_stride_b = 16384;
    int effective_a_stride_m = 1024;
    int effective_a_stride_k = 1;
    int effective_b_stride_b = 1048576;
    int effective_b_stride_n = 1024;
    int effective_b_stride_k = 1;
    int effective_out_type = 0;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 16; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 2048) {
            int copy_row = copy_idx / 128;
            int copy_chunk = copy_idx % 128;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(a_dst), "l"(A + a_src));
            }
        }
    }
    #pragma unroll 4
    for (int copy_iter_b = 0; copy_iter_b < 32; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 4096) {
            int copy_row_b = copy_idx_b / 128;
            int copy_chunk_b = copy_idx_b % 128;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(b_dst), "l"(B_storage + b_src));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 64
    for (int k_atom = 0; k_atom < 64; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        #pragma unroll
        for (int frag_row = 0; frag_row < 2; frag_row++) {
            int m_idx = m_base + lane / 4 + frag_row * 8;
            int n_idx = n_warp_base + 2 * (lane % 4);
            int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
            const int value_idx = frag_row * 2;
            if (effective_out_type == 0) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                }
            } else if (effective_out_type == 1) {
                *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
            } else {
                {
                    float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b4-m16-n1024-k1024-fixed-fp16 (blackwell_bf16_bmm_sm103_k1024_m16n1024_fp16.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 32768
#define SMEM_SMEM_B_STAGE_BYTES 65536
#define SMEM_SMEM_B_STRIDE 65536
#define SMEM_TOTAL 98304
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o1_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int smem_b_addr = smem + 32768;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = 16;
    int effective_n = 1024;
    int effective_a_stride_b = 16384;
    int effective_a_stride_m = 1024;
    int effective_a_stride_k = 1;
    int effective_b_stride_b = 1048576;
    int effective_b_stride_n = 1024;
    int effective_b_stride_k = 1;
    int effective_out_type = 1;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 16; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 2048) {
            int copy_row = copy_idx / 128;
            int copy_chunk = copy_idx % 128;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(a_dst), "l"(A + a_src));
            }
        }
    }
    #pragma unroll 4
    for (int copy_iter_b = 0; copy_iter_b < 32; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 4096) {
            int copy_row_b = copy_idx_b / 128;
            int copy_chunk_b = copy_idx_b % 128;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(b_dst), "l"(B_storage + b_src));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 64
    for (int k_atom = 0; k_atom < 64; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        #pragma unroll
        for (int frag_row = 0; frag_row < 2; frag_row++) {
            int m_idx = m_base + lane / 4 + frag_row * 8;
            int n_idx = n_warp_base + 2 * (lane % 4);
            int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
            const int value_idx = frag_row * 2;
            if (effective_out_type == 0) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                }
            } else if (effective_out_type == 1) {
                *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
            } else {
                {
                    float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b4-m16-n1024-k1024-fixed-fp32 (blackwell_bf16_bmm_sm103_k1024_m16n1024_fp32.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 32768
#define SMEM_SMEM_B_STAGE_BYTES 65536
#define SMEM_SMEM_B_STRIDE 65536
#define SMEM_TOTAL 98304
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o2_fixed(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int smem_b_addr = smem + 32768;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    int effective_m = 16;
    int effective_n = 1024;
    int effective_a_stride_b = 16384;
    int effective_a_stride_m = 1024;
    int effective_a_stride_k = 1;
    int effective_b_stride_b = 1048576;
    int effective_b_stride_n = 1024;
    int effective_b_stride_k = 1;
    int effective_out_type = 2;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 4
    for (int copy_iter = 0; copy_iter < 16; copy_iter++) {
        int copy_idx = copy_iter * 128 + tid;
        if (copy_idx < 2048) {
            int copy_row = copy_idx / 128;
            int copy_chunk = copy_idx % 128;
            int a_src = batch_idx * effective_a_stride_b + (m_base + copy_row) * effective_a_stride_m + copy_chunk * 8 * effective_a_stride_k;
            unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(a_dst), "l"(A + a_src));
            }
        }
    }
    #pragma unroll 4
    for (int copy_iter_b = 0; copy_iter_b < 32; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 128 + tid;
        if (copy_idx_b < 4096) {
            int copy_row_b = copy_idx_b / 128;
            int copy_chunk_b = copy_idx_b % 128;
            int b_src = batch_idx * effective_b_stride_b + (n_base + copy_row_b) * effective_b_stride_n + copy_chunk_b * 8 * effective_b_stride_k;
            unsigned int b_dst = (smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4));
            {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(b_dst), "l"(B_storage + b_src));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int base_a = smem_a_addr;
    unsigned int base_b = smem_b_addr;
    #pragma unroll 64
    for (int k_atom = 0; k_atom < 64; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = base_a + k_group * 2048;
        unsigned int b_group_base = base_b + k_group * 4096;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        #pragma unroll
        for (int frag_row = 0; frag_row < 2; frag_row++) {
            int m_idx = m_base + lane / 4 + frag_row * 8;
            int n_idx = n_warp_base + 2 * (lane % 4);
            int output_idx = (batch_idx * effective_m + m_idx) * effective_n + n_idx;
            const int value_idx = frag_row * 2;
            if (effective_out_type == 0) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                }
            } else if (effective_out_type == 1) {
                *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
            } else {
                {
                    float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                    *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// ---- b2-m8-n1024-k1024-m16n16-tail (blackwell_bf16_bmm_sm103_k1024_n16_m8_tail.cu) ----
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 32768
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_TOTAL 65536
#define THREADS 64

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(64, 1) void
kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_n16_m8_tail(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int smem_b_addr = smem + 32768;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 16;
    const int fixed_m = 8;
    const int fixed_n = 1024;
    const int fixed_out_type = 0;
    float accum[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll 32
    for (int copy_iter = 0; copy_iter < 32; copy_iter++) {
        int copy_idx = copy_iter * 64 + tid;
        int copy_row = copy_idx / 128;
        int copy_chunk = copy_idx % 128;
        int a_src = batch_idx * 8192 + (m_base + copy_row) * 1024 + copy_chunk * 8;
        unsigned int a_dst = (smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4));
        {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(a_dst), "l"(A + a_src), "r"((fixed_m > m_base + copy_row) ? 16 : 0));
        }
    }
    #pragma unroll 32
    for (int copy_iter_b = 0; copy_iter_b < 32; copy_iter_b++) {
        int copy_idx_b = copy_iter_b * 64 + tid;
        int copy_row_b = copy_idx_b / 128;
        int copy_chunk_b = copy_idx_b % 128;
        int b_src = batch_idx * 1048576 + (n_base + copy_row_b) * 1024 + copy_chunk_b * 8;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"((smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 16 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 16 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(B_storage + b_src));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 64;" ::: "memory");
    #pragma unroll 64
    for (int k_atom = 0; k_atom < 64; k_atom++) {
        unsigned int a_frag[4];
        unsigned int b_frag[2];
        unsigned int k_group = k_atom / 4;
        unsigned int atom_in_group = k_atom % 4;
        unsigned int a_group_base = smem_a_addr + k_group * 2048;
        unsigned int b_group_base = smem_b_addr + k_group * 2048;
        unsigned int col_a = 2 * atom_in_group + col_off_a;
        unsigned int col_sw_a = row_a % 8 ^ col_a;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_group_base + row_a * 128 + col_sw_a * 16)
            : "memory");
        unsigned int col_b = 2 * atom_in_group + lane_div8;
        unsigned int col_sw_b = row_b % 8 ^ col_b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_group_base + ((unsigned int)(warp * 8) + row_b) * 128 + col_sw_b * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    }
    {
        #pragma unroll
        for (int frag_row = 0; frag_row < 2; frag_row++) {
            int m_idx = m_base + lane / 4 + frag_row * 8;
            int n_idx = n_warp_base + 2 * (lane % 4);
            if (m_idx < fixed_m && n_idx < fixed_n) {
                int output_idx = (batch_idx * fixed_m + m_idx) * fixed_n + n_idx;
                const int value_idx = frag_row * 2;
                if (fixed_n > n_idx + 1) {
                    if (fixed_out_type == 0) {
                        {
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                        }
                    } else if (fixed_out_type == 1) {
                        *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                        *(reinterpret_cast<__half*>(out_bytes + ((output_idx + 1) * 2)) + (0)) = __float2half_rn(accum[value_idx + 1]);
                    } else {
                        {
                            float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                            *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                        }
                    }
                } else if (fixed_out_type == 0) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_bytes + (output_idx * 2)) + (0)) = __float2bfloat16_rn(accum[value_idx]);
                } else {
                    if (fixed_out_type == 1) {
                        *(reinterpret_cast<__half*>(out_bytes + (output_idx * 2)) + (0)) = __float2half_rn(accum[value_idx]);
                    } else {
                        *(reinterpret_cast<float*>(out_bytes + (output_idx * 4)) + (0)) = accum[value_idx];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef THREADS
