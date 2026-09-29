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
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
kernel_cake_bf16_bmm_bda02d92fb3e9e813ac2(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

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
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
    int n_warp_base = (unsigned int)n_base + warp * 8;
    asm volatile("griddepcontrol.wait;" ::: "memory");
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
            : "r"(b_group_base + (warp * 8 + row_b) * 128 + col_sw_b * 16)
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
        int m_idx = (unsigned int)m_base + lane / 4 + (unsigned int)(frag_row * 8);
        int n_idx = (unsigned int)n_warp_base + 2 * (lane % 4);
        int output_idx = (batch_idx * fixed_m + m_idx) * fixed_n + n_idx;
        const int value_idx = frag_row * 2;
        if (fixed_out_type == 0) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
            }
        } else if (fixed_out_type == 1) {
            uint8_t* pair_dst = out_bytes + (output_idx * 2);
            unsigned long long pair_addr = (unsigned long long)pair_dst;
            if ((pair_addr & 3) == 0) {
                {
                    __half2 _pk = __floats2half2_rn(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                    *reinterpret_cast<__half2*>(&((__half*)(reinterpret_cast<unsigned int*>(pair_dst)))[0]) = _pk;
                }
            } else {
                *(reinterpret_cast<__half*>(pair_dst) + (0)) = __float2half_rn(accum_lo[value_idx]);
                *(reinterpret_cast<__half*>(pair_dst + 2) + (0)) = __float2half_rn(accum_lo[value_idx + 1]);
            }
        } else {
            {
                float2 _v2 = make_float2(accum_lo[value_idx + 0], accum_lo[value_idx + 1]);
                *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
            }
        }
    }
    #pragma unroll
    for (int frag_row_1 = 0; frag_row_1 < 2; frag_row_1++) {
        int m_idx_1 = (unsigned int)(m_base + 16) + lane / 4 + (unsigned int)(frag_row_1 * 8);
        int n_idx_1 = (unsigned int)n_warp_base + 2 * (lane % 4);
        int output_idx_1 = (batch_idx * fixed_m + m_idx_1) * fixed_n + n_idx_1;
        const int value_idx_1 = frag_row_1 * 2;
        if (fixed_out_type == 0) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2)))[0]) = _pk;
            }
        } else if (fixed_out_type == 1) {
            uint8_t* pair_dst_1 = out_bytes + (output_idx_1 * 2);
            unsigned long long pair_addr_1 = (unsigned long long)pair_dst_1;
            if ((pair_addr_1 & 3) == 0) {
                {
                    __half2 _pk = __floats2half2_rn(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                    *reinterpret_cast<__half2*>(&((__half*)(reinterpret_cast<unsigned int*>(pair_dst_1)))[0]) = _pk;
                }
            } else {
                *(reinterpret_cast<__half*>(pair_dst_1) + (0)) = __float2half_rn(accum_hi[value_idx_1]);
                *(reinterpret_cast<__half*>(pair_dst_1 + 2) + (0)) = __float2half_rn(accum_hi[value_idx_1 + 1]);
            }
        } else {
            {
                float2 _v2 = make_float2(accum_hi[value_idx_1 + 0], accum_hi[value_idx_1 + 1]);
                *reinterpret_cast<float2*>(out_bytes + (output_idx_1 * 4) + 0) = _v2;
            }
        }
    }
}

} // extern "C"
