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
#define SMEM_COUNTS_OFF 0
#define SMEM_COUNTS_STAGE_BYTES 4096
#define SMEM_COUNTS_STRIDE 4096
#define SMEM_OFFSETS_OFF 4096
#define SMEM_OFFSETS_STAGE_BYTES 4096
#define SMEM_OFFSETS_STRIDE 4096
#define SMEM_SCATTER_OFFSETS_OFF 8192
#define SMEM_SCATTER_OFFSETS_STAGE_BYTES 4096
#define SMEM_SCATTER_OFFSETS_STRIDE 4096
#define SMEM_CHUNK_PREFIXES_OFF 12288
#define SMEM_CHUNK_PREFIXES_STAGE_BYTES 128
#define SMEM_CHUNK_PREFIXES_STRIDE 128
#define SMEM_TOTAL 12416
#define THREADS 512

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_d11c18ef7d25cdcdccd2(int* __restrict__ topk_ids, int* __restrict__ expert_counts, int* __restrict__ expert_tile_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ route_map, int* __restrict__ token_to_permuted, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ total_tiles, int total_pairs, int num_experts, int max_tiles, int top_k, int tile_n, int* __restrict__ fc2_work_counter, int fc2_pool_ctas)
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
    int* counts = reinterpret_cast<int*>(smem_raw + 0);
    const int counts_addr = smem + 0;
    int* offsets = reinterpret_cast<int*>(smem_raw + 4096);
    const int offsets_addr = smem + 4096;
    int* scatter_offsets = reinterpret_cast<int*>(smem_raw + 8192);
    const int scatter_offsets_addr = smem + 8192;
    int* chunk_prefixes = reinterpret_cast<int*>(smem_raw + 12288);
    const int chunk_prefixes_addr = smem + 12288;

    // === Task calls (dependency order) ===
    int expert_init = tid;
    counts[expert_init] = 0;
    scatter_offsets[expert_init] = 0;
    int expert_init_0 = tid + 512;
    counts[expert_init_0] = 0;
    scatter_offsets[expert_init_0] = 0;
    for (int tile_init = tid; tile_init < max_tiles; tile_init += 512) {
        tile_expert[tile_init] = 0;
        tile_mn_limit[tile_init] = tile_init * tile_n;
    }
    for (int row_init = tid; row_init < max_tiles * tile_n + 1; row_init += 512) {
        route_map[row_init] = 0;
    }
    __syncthreads();
    for (int pair_count = tid; pair_count < total_pairs; pair_count += 512) {
        int expert_count = topk_ids[pair_count];
        atomicAdd(&counts[expert_count], 1);
    }
    __syncthreads();
    int first_expert = tid;
    int second_expert = tid + 512;
    int first_tiles = 0;
    int second_tiles = 0;
    if (first_expert < num_experts) {
        int first_count = counts[first_expert];
        first_tiles = (first_count + tile_n - 1) / tile_n;
    }
    if (second_expert < num_experts) {
        int second_count = counts[second_expert];
        second_tiles = (second_count + tile_n - 1) / tile_n;
    }
    int first_inclusive = first_tiles;
    int second_inclusive = second_tiles;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 1, 32);
    int first_peer = _shfl_up_0;
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 1, 32);
    int second_peer = _shfl_up_1;
    if (lane >= 1) {
        first_inclusive = first_inclusive + first_peer;
        second_inclusive = second_inclusive + second_peer;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 2, 32);
    int first_peer_1 = _shfl_up_2;
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 2, 32);
    int second_peer_2 = _shfl_up_3;
    if (lane >= 2) {
        first_inclusive = first_inclusive + first_peer_1;
        second_inclusive = second_inclusive + second_peer_2;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 4, 32);
    int first_peer_3 = _shfl_up_4;
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 4, 32);
    int second_peer_4 = _shfl_up_5;
    if (lane >= 4) {
        first_inclusive = first_inclusive + first_peer_3;
        second_inclusive = second_inclusive + second_peer_4;
    }
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 8, 32);
    int first_peer_5 = _shfl_up_6;
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 8, 32);
    int second_peer_6 = _shfl_up_7;
    if (lane >= 8) {
        first_inclusive = first_inclusive + first_peer_5;
        second_inclusive = second_inclusive + second_peer_6;
    }
    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 16, 32);
    int first_peer_7 = _shfl_up_8;
    int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 16, 32);
    int second_peer_8 = _shfl_up_9;
    if (lane >= 16) {
        first_inclusive = first_inclusive + first_peer_7;
        second_inclusive = second_inclusive + second_peer_8;
    }
    if (lane == 31) {
        chunk_prefixes[warp] = first_inclusive;
        chunk_prefixes[warp + 16] = second_inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        int chunk_inclusive = chunk_prefixes[lane];
        int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 1, 32);
        int chunk_peer = _shfl_up_10;
        if (lane >= 1) {
            chunk_inclusive = chunk_inclusive + chunk_peer;
        }
        int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 2, 32);
        int chunk_peer_0 = _shfl_up_11;
        if (lane >= 2) {
            chunk_inclusive = chunk_inclusive + chunk_peer_0;
        }
        int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 4, 32);
        int chunk_peer_1 = _shfl_up_12;
        if (lane >= 4) {
            chunk_inclusive = chunk_inclusive + chunk_peer_1;
        }
        int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 8, 32);
        int chunk_peer_2 = _shfl_up_13;
        if (lane >= 8) {
            chunk_inclusive = chunk_inclusive + chunk_peer_2;
        }
        int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 16, 32);
        int chunk_peer_3 = _shfl_up_14;
        if (lane >= 16) {
            chunk_inclusive = chunk_inclusive + chunk_peer_3;
        }
        chunk_prefixes[lane] = chunk_inclusive;
    }
    __syncthreads();
    int first_chunk_base = 0;
    if (warp > 0) {
        first_chunk_base = chunk_prefixes[warp - 1];
    }
    int second_chunk_base = chunk_prefixes[warp + 15];
    if (first_expert < num_experts) {
        int first_offset = first_chunk_base + first_inclusive - first_tiles;
        offsets[first_expert] = first_offset;
        expert_tile_offsets[first_expert] = first_offset;
    }
    if (second_expert < num_experts) {
        int second_offset = second_chunk_base + second_inclusive - second_tiles;
        offsets[second_expert] = second_offset;
        expert_tile_offsets[second_expert] = second_offset;
    }
    if (tid == 0) {
        total_tiles[0] = chunk_prefixes[31];
        fc2_work_counter[0] = fc2_pool_ctas;
    }
    __syncthreads();
    for (int pair_scatter = tid; pair_scatter < total_pairs; pair_scatter += 512) {
        int expert_scatter = topk_ids[pair_scatter];
        int _atomic_old_0 = atomicAdd(&scatter_offsets[expert_scatter], 1);
        int local_row = _atomic_old_0;
        int local_tile = local_row / tile_n;
        int tile = offsets[expert_scatter] + local_tile;
        int grouped_row = tile * tile_n + local_row % tile_n;
        route_map[grouped_row] = pair_scatter / top_k;
        token_to_permuted[pair_scatter] = grouped_row;
        if (local_row % tile_n == 0) {
            int remaining = counts[expert_scatter] - local_tile * tile_n;
            int valid = remaining;
            if (valid > tile_n) {
                valid = tile_n;
            }
            tile_expert[tile] = expert_scatter;
            tile_mn_limit[tile] = tile * tile_n + valid;
        }
    }
    __syncthreads();
    int expert_store = tid;
    if (expert_store < num_experts) {
        expert_counts[expert_store] = counts[expert_store];
        expert_scatter_offsets[expert_store] = scatter_offsets[expert_store];
    }
    int expert_store_9 = tid + 512;
    if (expert_store_9 < num_experts) {
        expert_counts[expert_store_9] = counts[expert_store_9];
        expert_scatter_offsets[expert_store_9] = scatter_offsets[expert_store_9];
    }
}

} // extern "C"
