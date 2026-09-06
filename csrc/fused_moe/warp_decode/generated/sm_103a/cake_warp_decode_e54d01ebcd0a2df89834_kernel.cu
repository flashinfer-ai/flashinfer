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
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
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
#define SMEM_EXPERT_COUNTS_STAGE_BYTES 256
#define SMEM_EXPERT_COUNTS_STRIDE 256
#define SMEM_EXPERT_TILE_OFFSETS_OFF 256
#define SMEM_EXPERT_TILE_OFFSETS_STAGE_BYTES 260
#define SMEM_EXPERT_TILE_OFFSETS_STRIDE 260
#define SMEM_EXPERT_SCATTER_OFFSETS_OFF 516
#define SMEM_EXPERT_SCATTER_OFFSETS_STAGE_BYTES 256
#define SMEM_EXPERT_SCATTER_OFFSETS_STRIDE 256
#define SMEM_SCAN_VALUES_OFF 772
#define SMEM_SCAN_VALUES_STAGE_BYTES 8
#define SMEM_SCAN_VALUES_STRIDE 8
#define SMEM_TOTAL 896
#define THREADS 256
#define MAX_EXPERTS 64
#define TILE_N 8

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_warp_decode_e54d01ebcd0a2df89834(int* __restrict__ route_experts, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ route_slots, int* __restrict__ num_non_exiting_ctas, int* __restrict__ fc1_work_counter, int* __restrict__ fc2_work_counter, int route_count, int top_k, int local_expert_offset, int num_experts, int fc1_initial_work, int fc2_initial_work)
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
    int* expert_counts = reinterpret_cast<int*>(smem_raw + 0);
    const int expert_counts_addr = smem + 0;
    int* expert_tile_offsets = reinterpret_cast<int*>(smem_raw + 256);
    const int expert_tile_offsets_addr = smem + 256;
    int* expert_scatter_offsets = reinterpret_cast<int*>(smem_raw + 516);
    const int expert_scatter_offsets_addr = smem + 516;
    int* scan_values = reinterpret_cast<int*>(smem_raw + 772);
    const int scan_values_addr = smem + 772;

    // === Task calls (dependency order) ===
    int expert = tid;
    if (expert < num_experts) {
        expert_counts[expert] = 0;
        expert_tile_offsets[expert] = 0;
        expert_scatter_offsets[expert] = 0;
    }
    if (tid == 0) {
        expert_tile_offsets[num_experts] = 0;
        num_non_exiting_ctas[0] = 0;
        fc1_work_counter[0] = fc1_initial_work;
        fc2_work_counter[0] = fc2_initial_work;
    }
    __syncthreads();
    for (int route = tid; route < route_count; route += THREADS) {
        int route_expert = route_experts[route] - local_expert_offset;
        if (route_expert >= 0 && route_expert < num_experts) {
            atomicAdd(&expert_counts[route_expert], 1);
        }
    }
    __syncthreads();
    int tiles = 0;
    int inclusive = 0;
    if (warp < 2) {
        if (expert < num_experts) {
            int count = expert_counts[expert];
            tiles = (count + TILE_N - 1) / TILE_N;
        }
        inclusive = tiles;
        int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
        int peer = _shfl_up_0;
        if (lane >= 1) {
            inclusive = inclusive + peer;
        }
        int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
        int peer_0 = _shfl_up_1;
        if (lane >= 2) {
            inclusive = inclusive + peer_0;
        }
        int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
        int peer_1 = _shfl_up_2;
        if (lane >= 4) {
            inclusive = inclusive + peer_1;
        }
        int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
        int peer_2 = _shfl_up_3;
        if (lane >= 8) {
            inclusive = inclusive + peer_2;
        }
        int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
        int peer_3 = _shfl_up_4;
        if (lane >= 16) {
            inclusive = inclusive + peer_3;
        }
        if (lane == 31) {
            scan_values[warp] = inclusive;
        }
    }
    __syncthreads();
    if (warp < 2) {
        int chunk_base = 0;
        if (warp == 1) {
            chunk_base = scan_values[0];
        }
        if (expert < num_experts) {
            expert_tile_offsets[expert] = chunk_base + inclusive - tiles;
        }
    }
    int total_tiles = scan_values[0] + scan_values[1];
    if (tid == 0) {
        num_non_exiting_ctas[0] = total_tiles;
        expert_tile_offsets[num_experts] = total_tiles;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __syncthreads();
    for (int route_1 = tid; route_1 < route_count; route_1 += THREADS) {
        int route_expert_1 = route_experts[route_1] - local_expert_offset;
        if (route_expert_1 >= 0 && route_expert_1 < num_experts) {
            int _atomic_old_0 = atomicAdd(&expert_scatter_offsets[route_expert_1], 1);
            int local_row = _atomic_old_0;
            int tile = expert_tile_offsets[route_expert_1] + local_row / TILE_N;
            int row = local_row % TILE_N;
            int slot = tile * TILE_N + row;
            route_map[slot] = route_1 / top_k;
            route_slots[route_1] = slot;
            if (row == 0) {
                int remaining = expert_counts[route_expert_1] - local_row;
                int valid_rows = TILE_N;
                if (remaining < TILE_N) {
                    valid_rows = remaining;
                }
                tile_expert[tile] = route_expert_1;
                tile_mn_limit[tile] = tile * TILE_N + valid_rows;
            }
        }
    }
}

} // extern "C"
