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
#define SMEM_SCAN_VALUES_OFF 0
#define SMEM_SCAN_VALUES_STAGE_BYTES 4096
#define SMEM_SCAN_VALUES_STRIDE 4096
#define SMEM_TOTAL 4096
#define THREADS 512

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_517e3a1b1cb957f9519e(int* __restrict__ expert_counts, int* __restrict__ expert_tile_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ total_tiles, int num_experts, int tile_n)
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
    int* scan_values = reinterpret_cast<int*>(smem_raw + 0);
    const int scan_values_addr = smem + 0;

    // === Task calls (dependency order) ===
    int expert_init = tid;
    int expert_tiles_init = 0;
    if (expert_init < num_experts) {
        int count_init = expert_counts[expert_init];
        expert_tiles_init = (count_init + tile_n - 1) / tile_n;
    }
    scan_values[expert_init] = expert_tiles_init;
    int expert_init_0 = tid + 512;
    int expert_tiles_init_1 = 0;
    if (expert_init_0 < num_experts) {
        int count_init_1 = expert_counts[expert_init_0];
        expert_tiles_init_1 = (count_init_1 + tile_n - 1) / tile_n;
    }
    scan_values[expert_init_0] = expert_tiles_init_1;
    __syncthreads();
    int scan_index_up = (tid + 1) * 2 - 1;
    if (scan_index_up < 1024) {
        scan_values[scan_index_up] = scan_values[scan_index_up] + scan_values[scan_index_up - 1];
    }
    __syncthreads();
    int scan_index_up_2 = (tid + 1) * 4 - 1;
    if (scan_index_up_2 < 1024) {
        scan_values[scan_index_up_2] = scan_values[scan_index_up_2] + scan_values[scan_index_up_2 - 2];
    }
    __syncthreads();
    int scan_index_up_3 = (tid + 1) * 8 - 1;
    if (scan_index_up_3 < 1024) {
        scan_values[scan_index_up_3] = scan_values[scan_index_up_3] + scan_values[scan_index_up_3 - 4];
    }
    __syncthreads();
    int scan_index_up_4 = (tid + 1) * 16 - 1;
    if (scan_index_up_4 < 1024) {
        scan_values[scan_index_up_4] = scan_values[scan_index_up_4] + scan_values[scan_index_up_4 - 8];
    }
    __syncthreads();
    int scan_index_up_5 = (tid + 1) * 32 - 1;
    if (scan_index_up_5 < 1024) {
        scan_values[scan_index_up_5] = scan_values[scan_index_up_5] + scan_values[scan_index_up_5 - 16];
    }
    __syncthreads();
    int scan_index_up_6 = (tid + 1) * 64 - 1;
    if (scan_index_up_6 < 1024) {
        scan_values[scan_index_up_6] = scan_values[scan_index_up_6] + scan_values[scan_index_up_6 - 32];
    }
    __syncthreads();
    int scan_index_up_7 = (tid + 1) * 128 - 1;
    if (scan_index_up_7 < 1024) {
        scan_values[scan_index_up_7] = scan_values[scan_index_up_7] + scan_values[scan_index_up_7 - 64];
    }
    __syncthreads();
    int scan_index_up_8 = (tid + 1) * 256 - 1;
    if (scan_index_up_8 < 1024) {
        scan_values[scan_index_up_8] = scan_values[scan_index_up_8] + scan_values[scan_index_up_8 - 128];
    }
    __syncthreads();
    int scan_index_up_9 = (tid + 1) * 512 - 1;
    if (scan_index_up_9 < 1024) {
        scan_values[scan_index_up_9] = scan_values[scan_index_up_9] + scan_values[scan_index_up_9 - 256];
    }
    __syncthreads();
    int scan_index_up_10 = (tid + 1) * 1024 - 1;
    if (scan_index_up_10 < 1024) {
        scan_values[scan_index_up_10] = scan_values[scan_index_up_10] + scan_values[scan_index_up_10 - 512];
    }
    __syncthreads();
    if (tid == 0) {
        total_tiles[0] = scan_values[1023];
        scan_values[1023] = 0;
    }
    __syncthreads();
    int scan_index_down = (tid + 1) * 1024 - 1;
    if (scan_index_down < 1024) {
        int scan_left = scan_values[scan_index_down - 512];
        scan_values[scan_index_down - 512] = scan_values[scan_index_down];
        scan_values[scan_index_down] = scan_values[scan_index_down] + scan_left;
    }
    __syncthreads();
    int scan_index_down_11 = (tid + 1) * 512 - 1;
    if (scan_index_down_11 < 1024) {
        int scan_left_1 = scan_values[scan_index_down_11 - 256];
        scan_values[scan_index_down_11 - 256] = scan_values[scan_index_down_11];
        scan_values[scan_index_down_11] = scan_values[scan_index_down_11] + scan_left_1;
    }
    __syncthreads();
    int scan_index_down_12 = (tid + 1) * 256 - 1;
    if (scan_index_down_12 < 1024) {
        int scan_left_2 = scan_values[scan_index_down_12 - 128];
        scan_values[scan_index_down_12 - 128] = scan_values[scan_index_down_12];
        scan_values[scan_index_down_12] = scan_values[scan_index_down_12] + scan_left_2;
    }
    __syncthreads();
    int scan_index_down_13 = (tid + 1) * 128 - 1;
    if (scan_index_down_13 < 1024) {
        int scan_left_3 = scan_values[scan_index_down_13 - 64];
        scan_values[scan_index_down_13 - 64] = scan_values[scan_index_down_13];
        scan_values[scan_index_down_13] = scan_values[scan_index_down_13] + scan_left_3;
    }
    __syncthreads();
    int scan_index_down_14 = (tid + 1) * 64 - 1;
    if (scan_index_down_14 < 1024) {
        int scan_left_4 = scan_values[scan_index_down_14 - 32];
        scan_values[scan_index_down_14 - 32] = scan_values[scan_index_down_14];
        scan_values[scan_index_down_14] = scan_values[scan_index_down_14] + scan_left_4;
    }
    __syncthreads();
    int scan_index_down_15 = (tid + 1) * 32 - 1;
    if (scan_index_down_15 < 1024) {
        int scan_left_5 = scan_values[scan_index_down_15 - 16];
        scan_values[scan_index_down_15 - 16] = scan_values[scan_index_down_15];
        scan_values[scan_index_down_15] = scan_values[scan_index_down_15] + scan_left_5;
    }
    __syncthreads();
    int scan_index_down_16 = (tid + 1) * 16 - 1;
    if (scan_index_down_16 < 1024) {
        int scan_left_6 = scan_values[scan_index_down_16 - 8];
        scan_values[scan_index_down_16 - 8] = scan_values[scan_index_down_16];
        scan_values[scan_index_down_16] = scan_values[scan_index_down_16] + scan_left_6;
    }
    __syncthreads();
    int scan_index_down_17 = (tid + 1) * 8 - 1;
    if (scan_index_down_17 < 1024) {
        int scan_left_7 = scan_values[scan_index_down_17 - 4];
        scan_values[scan_index_down_17 - 4] = scan_values[scan_index_down_17];
        scan_values[scan_index_down_17] = scan_values[scan_index_down_17] + scan_left_7;
    }
    __syncthreads();
    int scan_index_down_18 = (tid + 1) * 4 - 1;
    if (scan_index_down_18 < 1024) {
        int scan_left_8 = scan_values[scan_index_down_18 - 2];
        scan_values[scan_index_down_18 - 2] = scan_values[scan_index_down_18];
        scan_values[scan_index_down_18] = scan_values[scan_index_down_18] + scan_left_8;
    }
    __syncthreads();
    int scan_index_down_19 = (tid + 1) * 2 - 1;
    if (scan_index_down_19 < 1024) {
        int scan_left_9 = scan_values[scan_index_down_19 - 1];
        scan_values[scan_index_down_19 - 1] = scan_values[scan_index_down_19];
        scan_values[scan_index_down_19] = scan_values[scan_index_down_19] + scan_left_9;
    }
    __syncthreads();
    int expert_store = tid;
    if (expert_store < num_experts) {
        expert_tile_offsets[expert_store] = scan_values[expert_store];
        expert_scatter_offsets[expert_store] = 0;
    }
    int expert_store_20 = tid + 512;
    if (expert_store_20 < num_experts) {
        expert_tile_offsets[expert_store_20] = scan_values[expert_store_20];
        expert_scatter_offsets[expert_store_20] = 0;
    }
}

} // extern "C"
