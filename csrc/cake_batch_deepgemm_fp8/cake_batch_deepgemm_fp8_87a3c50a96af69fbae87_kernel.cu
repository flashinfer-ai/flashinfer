/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#define SMEM_STAGING_OFF 0
#define SMEM_STAGING_STAGE_BYTES 10752
#define SMEM_STAGING_STRIDE 10752
#define SMEM_TOTAL 10752
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_batch_deepgemm_fp8_87a3c50a96af69fbae87(int* __restrict__ SFA_bits, int* __restrict__ SFB_bits, unsigned int* __restrict__ SFA_packed, unsigned int* __restrict__ SFB_packed, unsigned int num_groups, unsigned int shape_m, unsigned int N, unsigned int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* staging = reinterpret_cast<int*>(smem_raw + 0);
    const int staging_addr = smem + 0;

    // === Task calls (dependency order) ===
    unsigned int sf_cols = K / 128;
    unsigned int packed_cols = sf_cols / 4;
    unsigned int b_rows = N / 128;
    unsigned int a_blocks = (shape_m + 48 - 1) / 48;
    unsigned int b_output_rows = ((1) ? N : b_rows);
    unsigned int b_blocks = (b_output_rows + 48 - 1) / 48;
    unsigned int blocks_per_group = a_blocks + b_blocks;
    unsigned int group_idx = (unsigned int)bid / blocks_per_group;
    unsigned int local_block = (unsigned int)bid % blocks_per_group;
    int is_a = ((local_block < a_blocks) ? 1 : 0);
    unsigned int scale_block = ((is_a != 0) ? local_block : local_block - a_blocks);
    unsigned int row_count = ((is_a != 0) ? shape_m : b_output_rows);
    unsigned int row_base = scale_block * 48;
    unsigned int rows_left = row_count - row_base;
    unsigned int rows_in_block = ((rows_left > 48) ? (unsigned int)48 : rows_left);
    unsigned int num_values = rows_in_block * sf_cols;
    unsigned int num_vectors = num_values / 4;
    #pragma unroll 1
    for (unsigned int vector_idx = tid; vector_idx < num_vectors; vector_idx += 512) {
        unsigned int value_idx = vector_idx * 4;
        int values[4];
        if (is_a != 0) {
            unsigned int src_idx = (group_idx * shape_m + row_base) * sf_cols + value_idx;
            {
                const int4* _ivptr_0 = reinterpret_cast<const int4*>(SFA_bits + src_idx);
                int4 _ivld_0;
                _ivld_0 = *_ivptr_0;
                values[0 + 0] = _ivld_0.x;
                values[0 + 1] = _ivld_0.y;
                values[0 + 2] = _ivld_0.z;
                values[0 + 3] = _ivld_0.w;
            }
        } else {
            unsigned int src_idx_1 = 0;
            {
                unsigned int source_row = (row_base + value_idx / sf_cols) / 128;
                unsigned int source_col = value_idx % sf_cols;
                src_idx_1 = (group_idx * b_rows + source_row) * sf_cols + source_col;
            }
            {
                const int4* _ivptr_1 = reinterpret_cast<const int4*>(SFB_bits + src_idx_1);
                int4 _ivld_1;
                _ivld_1 = *_ivptr_1;
                values[0 + 0] = _ivld_1.x;
                values[0 + 1] = _ivld_1.y;
                values[0 + 2] = _ivld_1.z;
                values[0 + 3] = _ivld_1.w;
            }
        }
        int* _sv_ptr_0 = reinterpret_cast<int*>(staging + value_idx);
        reinterpret_cast<int4*>(_sv_ptr_0 + 0)[0] = reinterpret_cast<int4*>(values)[0];
    }
    __syncthreads();
    unsigned int num_packed = rows_in_block * packed_cols;
    #pragma unroll 1
    for (unsigned int packed_idx = tid; packed_idx < num_packed; packed_idx += 512) {
        unsigned int packed_k = packed_idx / rows_in_block;
        unsigned int local_row = packed_idx % rows_in_block;
        unsigned int src_base = local_row * sf_cols + packed_k * 4;
        int v0 = staging[src_base];
        int v1 = staging[src_base + 1];
        int v2 = staging[src_base + 2];
        int v3 = staging[src_base + 3];
        unsigned int e0 = (unsigned int)(v0 >> 23) & 255;
        unsigned int e1 = (unsigned int)(v1 >> 23) & 255;
        unsigned int e2 = (unsigned int)(v2 >> 23) & 255;
        unsigned int e3 = (unsigned int)(v3 >> 23) & 255;
        unsigned int word = e0 | e1 << 8 | e2 << 16 | e3 << 24;
        unsigned int global_row = row_base + local_row;
        if (is_a != 0) {
            unsigned int dst_idx = (group_idx * packed_cols + packed_k) * shape_m + global_row;
            SFA_packed[dst_idx] = word;
        } else {
            unsigned int dst_idx_1 = (group_idx * packed_cols + packed_k) * b_output_rows + global_row;
            SFB_packed[dst_idx_1] = word;
        }
    }
}

} // extern "C"
