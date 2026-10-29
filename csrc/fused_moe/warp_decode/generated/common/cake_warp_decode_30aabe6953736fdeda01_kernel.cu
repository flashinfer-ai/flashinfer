/*
 * Copyright (c) 2026 by FlashInfer team.
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
#define SMEM_EXPERT_COUNTS_OFF 0
#define SMEM_EXPERT_COUNTS_STAGE_BYTES 2048
#define SMEM_EXPERT_COUNTS_STRIDE 2048
#define SMEM_EXPERT_TILE_OFFSETS_OFF 2048
#define SMEM_EXPERT_TILE_OFFSETS_STAGE_BYTES 2052
#define SMEM_EXPERT_TILE_OFFSETS_STRIDE 2052
#define SMEM_EXPERT_SCATTER_OFFSETS_OFF 4112
#define SMEM_EXPERT_SCATTER_OFFSETS_STAGE_BYTES 2048
#define SMEM_EXPERT_SCATTER_OFFSETS_STRIDE 2048
#define SMEM_SCAN_VALUES_OFF 6160
#define SMEM_SCAN_VALUES_STAGE_BYTES 64
#define SMEM_SCAN_VALUES_STRIDE 64
#define SMEM_TOTAL 6272
#define THREADS 256
#define MAX_EXPERTS 512
#define TILE_N 8

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_warp_decode_30aabe6953736fdeda01(int* __restrict__ route_experts, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ route_slots, int* __restrict__ num_non_exiting_ctas, int* __restrict__ fc1_work_counter, int* __restrict__ fc2_work_counter, int route_count, int top_k, int local_expert_offset, int num_experts, int fc1_initial_work, int fc2_initial_work)
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
    int* expert_counts = reinterpret_cast<int*>(smem_raw + 0);
    const int expert_counts_addr = smem + 0;
    int* expert_tile_offsets = reinterpret_cast<int*>(smem_raw + 2048);
    const int expert_tile_offsets_addr = smem + 2048;
    int* expert_scatter_offsets = reinterpret_cast<int*>(smem_raw + 4112);
    const int expert_scatter_offsets_addr = smem + 4112;
    int* scan_values = reinterpret_cast<int*>(smem_raw + 6160);
    const int scan_values_addr = smem + 6160;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int expert_2_0 = tid;
    if (expert_2_0 < num_experts) {
        expert_counts[expert_2_0] = 0;
        expert_tile_offsets[expert_2_0] = 0;
        expert_scatter_offsets[expert_2_0] = 0;
    }
    int expert_2_1 = tid + THREADS;
    if (expert_2_1 < num_experts) {
        expert_counts[expert_2_1] = 0;
        expert_tile_offsets[expert_2_1] = 0;
        expert_scatter_offsets[expert_2_1] = 0;
    }
    if (tid == 0) {
        expert_tile_offsets[num_experts] = 0;
        num_non_exiting_ctas[0] = 0;
        fc1_work_counter[0] = fc1_initial_work;
        fc2_work_counter[0] = fc2_initial_work;
    }
    __syncthreads();
    int rank_expert = -1;
    int count_rank = 0;
    for (int route = tid; route < route_count; route += THREADS) {
        int expert_1 = route_experts[route] - local_expert_offset;
        if (expert_1 >= 0 && expert_1 < num_experts) {
            int counted = atomicAdd(&expert_counts[expert_1], 1);
            rank_expert = expert_1;
            count_rank = counted;
        }
    }
    __syncthreads();
    int first_expert = tid;
    int second_expert = tid + THREADS;
    int first_tiles = 0;
    int second_tiles = 0;
    if (first_expert < num_experts) {
        int first_count = expert_counts[first_expert];
        first_tiles = (first_count + TILE_N - 1) / TILE_N;
    }
    if (second_expert < num_experts) {
        int second_count = expert_counts[second_expert];
        second_tiles = (second_count + TILE_N - 1) / TILE_N;
    }
    int first_inclusive = first_tiles;
    int second_inclusive = second_tiles;
    int first_peer_0 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 1, 32);
    int second_peer_0 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 1, 32);
    if (lane >= 1) {
        first_inclusive = first_inclusive + first_peer_0;
        second_inclusive = second_inclusive + second_peer_0;
    }
    int first_peer_1 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 2, 32);
    int second_peer_1 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 2, 32);
    if (lane >= 2) {
        first_inclusive = first_inclusive + first_peer_1;
        second_inclusive = second_inclusive + second_peer_1;
    }
    int first_peer_2 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 4, 32);
    int second_peer_2 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 4, 32);
    if (lane >= 4) {
        first_inclusive = first_inclusive + first_peer_2;
        second_inclusive = second_inclusive + second_peer_2;
    }
    int first_peer_3 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 8, 32);
    int second_peer_3 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 8, 32);
    if (lane >= 8) {
        first_inclusive = first_inclusive + first_peer_3;
        second_inclusive = second_inclusive + second_peer_3;
    }
    int first_peer_4 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 16, 32);
    int second_peer_4 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 16, 32);
    if (lane >= 16) {
        first_inclusive = first_inclusive + first_peer_4;
        second_inclusive = second_inclusive + second_peer_4;
    }
    if (lane == 31) {
        scan_values[warp] = first_inclusive;
        scan_values[warp + 8] = second_inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        int chunk_prefix = 0;
        if (lane < 16) {
            chunk_prefix = scan_values[lane];
        }
        int _warp_scan_sum_0 = chunk_prefix;
        {
            uint32_t _warp_scan_sum_0_scan = (uint32_t)_warp_scan_sum_0;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_scan, 1, 32); if (lane >= 1) { _warp_scan_sum_0_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_scan, 2, 32); if (lane >= 2) { _warp_scan_sum_0_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_scan, 4, 32); if (lane >= 4) { _warp_scan_sum_0_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_scan, 8, 32); if (lane >= 8) { _warp_scan_sum_0_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_scan, 16, 32); if (lane >= 16) { _warp_scan_sum_0_scan += peer; } }
            _warp_scan_sum_0 = (int)_warp_scan_sum_0_scan;
        }
        chunk_prefix = _warp_scan_sum_0;
        if (lane < 16) {
            scan_values[lane] = chunk_prefix;
        }
    }
    __syncthreads();
    int first_chunk_base = 0;
    if (warp > 0) {
        first_chunk_base = scan_values[warp - 1];
    }
    int second_chunk_base = scan_values[warp + 7];
    if (first_expert < num_experts) {
        expert_tile_offsets[first_expert] = first_chunk_base + first_inclusive - first_tiles;
    }
    if (second_expert < num_experts) {
        expert_tile_offsets[second_expert] = second_chunk_base + second_inclusive - second_tiles;
    }
    int total_tiles = scan_values[15];
    if (tid == 0) {
        num_non_exiting_ctas[0] = total_tiles;
        expert_tile_offsets[num_experts] = total_tiles;
    }
    __syncthreads();
    if (route_count <= THREADS) {
        if (rank_expert >= 0) {
            int rank_tile = expert_tile_offsets[rank_expert] + count_rank / TILE_N;
            int rank_row = count_rank % TILE_N;
            int rank_slot = rank_tile * TILE_N + rank_row;
            route_map[rank_slot] = tid / top_k;
            route_slots[tid] = rank_slot;
            if (rank_row == 0) {
                int rank_remaining = expert_counts[rank_expert] - count_rank;
                int rank_valid_rows = TILE_N;
                if (rank_remaining < TILE_N) {
                    rank_valid_rows = rank_remaining;
                }
                tile_expert[rank_tile] = rank_expert;
                tile_mn_limit[rank_tile] = rank_tile * TILE_N + rank_valid_rows;
            }
        }
    } else {
        for (int route_1 = tid; route_1 < route_count; route_1 += THREADS) {
            int expert_1_1 = route_experts[route_1] - local_expert_offset;
            if (expert_1_1 >= 0 && expert_1_1 < num_experts) {
                int local_row = atomicAdd(&expert_scatter_offsets[expert_1_1], 1);
                int tile = expert_tile_offsets[expert_1_1] + local_row / TILE_N;
                int row = local_row % TILE_N;
                int slot = tile * TILE_N + row;
                route_map[slot] = route_1 / top_k;
                route_slots[route_1] = slot;
                if (row == 0) {
                    int remaining = expert_counts[expert_1_1] - local_row;
                    int valid_rows = TILE_N;
                    if (remaining < TILE_N) {
                        valid_rows = remaining;
                    }
                    tile_expert[tile] = expert_1_1;
                    tile_mn_limit[tile] = tile * TILE_N + valid_rows;
                }
            }
        }
    }
}

} // extern "C"
