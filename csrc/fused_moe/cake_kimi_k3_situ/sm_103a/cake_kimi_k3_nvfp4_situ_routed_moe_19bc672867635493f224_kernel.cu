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
#define SMEM_TOTALS_OFF 12416
#define SMEM_TOTALS_STAGE_BYTES 4096
#define SMEM_TOTALS_STRIDE 4096
#define SMEM_TOTAL 16512
#define THREADS 512

#include <math_constants.h>

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
}

extern "C" {

__global__ __launch_bounds__(512) __cluster_dims__(8,1,1) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_19bc672867635493f224(int* __restrict__ topk_ids, int* __restrict__ expert_counts, int* __restrict__ expert_tile_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ route_map, int* __restrict__ token_to_permuted, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ total_tiles, int total_pairs, int num_experts, int max_tiles, int top_k, int tile_n, int* __restrict__ fc2_work_counter, int fc2_pool_ctas)
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
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    int* counts = reinterpret_cast<int*>(smem_raw + 0);
    const int counts_addr = smem + 0;
    int* offsets = reinterpret_cast<int*>(smem_raw + 4096);
    const int offsets_addr = smem + 4096;
    int* scatter_offsets = reinterpret_cast<int*>(smem_raw + 8192);
    const int scatter_offsets_addr = smem + 8192;
    int* chunk_prefixes = reinterpret_cast<int*>(smem_raw + 12288);
    const int chunk_prefixes_addr = smem + 12288;
    int* totals = reinterpret_cast<int*>(smem_raw + 12416);
    const int totals_addr = smem + 12416;

    // === Task calls (dependency order) ===
    int rank_first = cta_rank * 512 + tid;
    int expert_init = tid;
    counts[expert_init] = 0;
    int expert_init_0 = tid + 512;
    counts[expert_init_0] = 0;
    for (int tile_init = rank_first; tile_init < max_tiles; tile_init += 4096) {
        tile_expert[tile_init] = 0;
        tile_mn_limit[tile_init] = tile_init * tile_n;
    }
    for (int row_init = rank_first; row_init < max_tiles * tile_n + 1; row_init += 4096) {
        route_map[row_init] = 0;
    }
    __syncthreads();
    for (int pair_count = rank_first; pair_count < total_pairs; pair_count += 4096) {
        int expert_count = topk_ids[pair_count];
        atomicAdd(&counts[expert_count], 1);
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    int first_expert = tid;
    int second_expert = tid + 512;
    int first_total = 0;
    int second_total = 0;
    int first_base = 0;
    int second_base = 0;
    uint32_t _mapa_0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_0) : "r"(counts_addr + (unsigned int)(first_expert * 4)), "r"(0));
    uint32_t _mapa_1;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_1) : "r"(counts_addr + (unsigned int)(second_expert * 4)), "r"(0));
    int _cluster_ld_0;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_0) : "r"(_mapa_0) : "memory");
    int first_peer_count = _cluster_ld_0;
    int _cluster_ld_1;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_1) : "r"(_mapa_1) : "memory");
    int second_peer_count = _cluster_ld_1;
    first_total = first_total + first_peer_count;
    second_total = second_total + second_peer_count;
    if (cta_rank > 0) {
        first_base = first_base + first_peer_count;
        second_base = second_base + second_peer_count;
    }
    uint32_t _mapa_2;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_2) : "r"(counts_addr + (unsigned int)(first_expert * 4)), "r"(1));
    uint32_t _mapa_3;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_3) : "r"(counts_addr + (unsigned int)(second_expert * 4)), "r"(1));
    int _cluster_ld_2;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_2) : "r"(_mapa_2) : "memory");
    int first_peer_count_1 = _cluster_ld_2;
    int _cluster_ld_3;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_3) : "r"(_mapa_3) : "memory");
    int second_peer_count_2 = _cluster_ld_3;
    first_total = first_total + first_peer_count_1;
    second_total = second_total + second_peer_count_2;
    if (cta_rank > 1) {
        first_base = first_base + first_peer_count_1;
        second_base = second_base + second_peer_count_2;
    }
    uint32_t _mapa_4;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_4) : "r"(counts_addr + (unsigned int)(first_expert * 4)), "r"(2));
    uint32_t _mapa_5;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_5) : "r"(counts_addr + (unsigned int)(second_expert * 4)), "r"(2));
    int _cluster_ld_4;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_4) : "r"(_mapa_4) : "memory");
    int first_peer_count_3 = _cluster_ld_4;
    int _cluster_ld_5;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_5) : "r"(_mapa_5) : "memory");
    int second_peer_count_4 = _cluster_ld_5;
    first_total = first_total + first_peer_count_3;
    second_total = second_total + second_peer_count_4;
    if (cta_rank > 2) {
        first_base = first_base + first_peer_count_3;
        second_base = second_base + second_peer_count_4;
    }
    uint32_t _mapa_6;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_6) : "r"(counts_addr + (unsigned int)(first_expert * 4)), "r"(3));
    uint32_t _mapa_7;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_7) : "r"(counts_addr + (unsigned int)(second_expert * 4)), "r"(3));
    int _cluster_ld_6;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_6) : "r"(_mapa_6) : "memory");
    int first_peer_count_5 = _cluster_ld_6;
    int _cluster_ld_7;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_7) : "r"(_mapa_7) : "memory");
    int second_peer_count_6 = _cluster_ld_7;
    first_total = first_total + first_peer_count_5;
    second_total = second_total + second_peer_count_6;
    if (cta_rank > 3) {
        first_base = first_base + first_peer_count_5;
        second_base = second_base + second_peer_count_6;
    }
    uint32_t _mapa_8;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_8) : "r"(counts_addr + (unsigned int)(first_expert * 4)), "r"(4));
    uint32_t _mapa_9;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_9) : "r"(counts_addr + (unsigned int)(second_expert * 4)), "r"(4));
    int _cluster_ld_8;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_8) : "r"(_mapa_8) : "memory");
    int first_peer_count_7 = _cluster_ld_8;
    int _cluster_ld_9;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_9) : "r"(_mapa_9) : "memory");
    int second_peer_count_8 = _cluster_ld_9;
    first_total = first_total + first_peer_count_7;
    second_total = second_total + second_peer_count_8;
    if (cta_rank > 4) {
        first_base = first_base + first_peer_count_7;
        second_base = second_base + second_peer_count_8;
    }
    uint32_t _mapa_10;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_10) : "r"(counts_addr + (unsigned int)(first_expert * 4)), "r"(5));
    uint32_t _mapa_11;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_11) : "r"(counts_addr + (unsigned int)(second_expert * 4)), "r"(5));
    int _cluster_ld_10;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_10) : "r"(_mapa_10) : "memory");
    int first_peer_count_9 = _cluster_ld_10;
    int _cluster_ld_11;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_11) : "r"(_mapa_11) : "memory");
    int second_peer_count_10 = _cluster_ld_11;
    first_total = first_total + first_peer_count_9;
    second_total = second_total + second_peer_count_10;
    if (cta_rank > 5) {
        first_base = first_base + first_peer_count_9;
        second_base = second_base + second_peer_count_10;
    }
    uint32_t _mapa_12;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_12) : "r"(counts_addr + (unsigned int)(first_expert * 4)), "r"(6));
    uint32_t _mapa_13;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_13) : "r"(counts_addr + (unsigned int)(second_expert * 4)), "r"(6));
    int _cluster_ld_12;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_12) : "r"(_mapa_12) : "memory");
    int first_peer_count_11 = _cluster_ld_12;
    int _cluster_ld_13;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_13) : "r"(_mapa_13) : "memory");
    int second_peer_count_12 = _cluster_ld_13;
    first_total = first_total + first_peer_count_11;
    second_total = second_total + second_peer_count_12;
    if (cta_rank > 6) {
        first_base = first_base + first_peer_count_11;
        second_base = second_base + second_peer_count_12;
    }
    uint32_t _mapa_14;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_14) : "r"(counts_addr + (unsigned int)(first_expert * 4)), "r"(7));
    uint32_t _mapa_15;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_15) : "r"(counts_addr + (unsigned int)(second_expert * 4)), "r"(7));
    int _cluster_ld_14;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_14) : "r"(_mapa_14) : "memory");
    int first_peer_count_13 = _cluster_ld_14;
    int _cluster_ld_15;
    asm volatile(
        "ld.shared::cluster.s32 %0, [%1];"
        : "=r"(_cluster_ld_15) : "r"(_mapa_15) : "memory");
    int second_peer_count_14 = _cluster_ld_15;
    first_total = first_total + first_peer_count_13;
    second_total = second_total + second_peer_count_14;
    if (cta_rank > 7) {
        first_base = first_base + first_peer_count_13;
        second_base = second_base + second_peer_count_14;
    }
    int first_tiles = 0;
    int second_tiles = 0;
    if (first_expert < num_experts) {
        first_tiles = (first_total + tile_n - 1) / tile_n;
    }
    if (second_expert < num_experts) {
        second_tiles = (second_total + tile_n - 1) / tile_n;
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
    int first_peer_15 = _shfl_up_2;
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 2, 32);
    int second_peer_16 = _shfl_up_3;
    if (lane >= 2) {
        first_inclusive = first_inclusive + first_peer_15;
        second_inclusive = second_inclusive + second_peer_16;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 4, 32);
    int first_peer_17 = _shfl_up_4;
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 4, 32);
    int second_peer_18 = _shfl_up_5;
    if (lane >= 4) {
        first_inclusive = first_inclusive + first_peer_17;
        second_inclusive = second_inclusive + second_peer_18;
    }
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 8, 32);
    int first_peer_19 = _shfl_up_6;
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 8, 32);
    int second_peer_20 = _shfl_up_7;
    if (lane >= 8) {
        first_inclusive = first_inclusive + first_peer_19;
        second_inclusive = second_inclusive + second_peer_20;
    }
    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 16, 32);
    int first_peer_21 = _shfl_up_8;
    int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 16, 32);
    int second_peer_22 = _shfl_up_9;
    if (lane >= 16) {
        first_inclusive = first_inclusive + first_peer_21;
        second_inclusive = second_inclusive + second_peer_22;
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
        totals[first_expert] = first_total;
        scatter_offsets[first_expert] = first_base;
        if (cta_rank == 0) {
            expert_tile_offsets[first_expert] = first_offset;
            expert_counts[first_expert] = first_total;
            expert_scatter_offsets[first_expert] = first_total;
        }
    }
    if (second_expert < num_experts) {
        int second_offset = second_chunk_base + second_inclusive - second_tiles;
        offsets[second_expert] = second_offset;
        totals[second_expert] = second_total;
        scatter_offsets[second_expert] = second_base;
        if (cta_rank == 0) {
            expert_tile_offsets[second_expert] = second_offset;
            expert_counts[second_expert] = second_total;
            expert_scatter_offsets[second_expert] = second_total;
        }
    }
    if (tid == 0) {
        if (cta_rank == 0) {
            total_tiles[0] = chunk_prefixes[31];
            fc2_work_counter[0] = fc2_pool_ctas;
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    for (int pair_scatter = rank_first; pair_scatter < total_pairs; pair_scatter += 4096) {
        int expert_scatter = topk_ids[pair_scatter];
        int _atomic_old_0 = atomicAdd(&scatter_offsets[expert_scatter], 1);
        int local_row = _atomic_old_0;
        int local_tile = local_row / tile_n;
        int tile = offsets[expert_scatter] + local_tile;
        int grouped_row = tile * tile_n + local_row % tile_n;
        route_map[grouped_row] = pair_scatter / top_k;
        token_to_permuted[pair_scatter] = grouped_row;
        if (local_row % tile_n == 0) {
            int remaining = totals[expert_scatter] - local_tile * tile_n;
            int valid = remaining;
            if (valid > tile_n) {
                valid = tile_n;
            }
            tile_expert[tile] = expert_scatter;
            tile_mn_limit[tile] = tile * tile_n + valid;
        }
    }
}

} // extern "C"
