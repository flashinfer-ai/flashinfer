/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
// BEGIN FROZEN CAKE EXPORT
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_flashinfer_blackwell_batch_deepgemm_fp8_pack_scales_m224(int* __restrict__ SFA_bits, int* __restrict__ SFB_bits, int* __restrict__ SFA_packed, int* __restrict__ SFB_packed, unsigned int num_groups, unsigned int shape_m, unsigned int N, unsigned int K)
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
    int* staging = reinterpret_cast<int*>(smem_raw + 0);
    const int staging_addr = smem + 0;

    // === Task calls (dependency order) ===
    unsigned int sf_cols = K / 128;
    unsigned int packed_cols = sf_cols / 4;
    unsigned int b_rows = N / 128;
    unsigned int a_blocks = (shape_m + 48 - 1) / 48;
    unsigned int b_blocks = (b_rows + 48 - 1) / 48;
    unsigned int blocks_per_group = a_blocks + b_blocks;
    unsigned int group_idx = (unsigned int)bid / blocks_per_group;
    unsigned int local_block = (unsigned int)bid % blocks_per_group;
    int is_a = ((local_block < a_blocks) ? 1 : 0);
    unsigned int scale_block = ((is_a != 0) ? local_block : local_block - a_blocks);
    unsigned int row_count = ((is_a != 0) ? shape_m : b_rows);
    unsigned int row_base = scale_block * 48;
    unsigned int rows_left = row_count - row_base;
    unsigned int rows_in_block = ((rows_left > 48) ? (unsigned int)48 : rows_left);
    unsigned int num_values = rows_in_block * sf_cols;
    #pragma unroll 1
    for (unsigned int value_idx = tid; value_idx < num_values; value_idx += 512) {
        if (is_a != 0) {
            unsigned int src_idx = (group_idx * shape_m + row_base) * sf_cols + value_idx;
            staging[value_idx] = SFA_bits[src_idx];
        } else {
            unsigned int src_idx_1 = (group_idx * b_rows + row_base) * sf_cols + value_idx;
            staging[value_idx] = SFB_bits[src_idx_1];
        }
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
            SFA_packed[dst_idx] = (int)word;
        } else {
            unsigned int dst_idx_1 = (group_idx * packed_cols + packed_k) * b_rows + global_row;
            SFB_packed[dst_idx_1] = (int)word;
        }
    }
}

} // extern "C"

// END FROZEN CAKE EXPORT
// clang-format on
