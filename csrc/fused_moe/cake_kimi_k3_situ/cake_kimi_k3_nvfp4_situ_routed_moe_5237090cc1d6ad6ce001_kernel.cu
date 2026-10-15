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
#define SMEM_SMEM_RANK_OFF 0
#define SMEM_SMEM_RANK_STAGE_BYTES 3584
#define SMEM_SMEM_RANK_STRIDE 3584
#define SMEM_SMEM_BASE_OFF 3584
#define SMEM_SMEM_BASE_STAGE_BYTES 3584
#define SMEM_SMEM_BASE_STRIDE 3584
#define SMEM_TOTAL 7168
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_5237090cc1d6ad6ce001(int* __restrict__ topk_ids, int* __restrict__ expert_counts, int* __restrict__ expert_tile_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ route_map, int* __restrict__ token_to_permuted, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int total_pairs, int top_k, int tile_n, int tile_n_shift)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    int* smem_rank = reinterpret_cast<int*>(smem_raw + 0);
    const int smem_rank_addr = smem + 0;
    int* smem_base = reinterpret_cast<int*>(smem_raw + 3584);
    const int smem_base_addr = smem + 3584;

    // === Task calls (dependency order) ===
    #pragma unroll 1
    for (int expert_slot = tid; expert_slot < 896; expert_slot += 256) {
        smem_rank[expert_slot] = 0;
    }
    __syncthreads();
    int cta_base = blockIdx.x * 1024;
    int experts[4] = {0};
    int ranks[4] = {0};
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int pair = cta_base + k * 256 + tid;
        if (pair < total_pairs) {
            experts[k] = topk_ids[pair];
            int _atomic_old_0 = atomicAdd(&smem_rank[experts[k]], 1);
            ranks[k] = _atomic_old_0;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int expert_slot_1 = tid; expert_slot_1 < 896; expert_slot_1 += 256) {
        int cta_count = smem_rank[expert_slot_1];
        if (cta_count > 0) {
            int _atomic_old_1 = atomicAdd(&expert_scatter_offsets[expert_slot_1], cta_count);
            smem_base[expert_slot_1] = _atomic_old_1;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < 4; k_1++) {
        int pair_1 = cta_base + k_1 * 256 + tid;
        if (pair_1 < total_pairs) {
            int expert = experts[k_1];
            int local_row = smem_base[expert] + ranks[k_1];
            int local_tile = local_row >> tile_n_shift;
            int tile = expert_tile_offsets[expert] + local_tile;
            int row_in_tile = local_row & tile_n - 1;
            int grouped_row = tile * tile_n + row_in_tile;
            route_map[grouped_row] = pair_1 / top_k;
            token_to_permuted[pair_1] = grouped_row;
            if (row_in_tile == 0) {
                int remaining = expert_counts[expert] - local_tile * tile_n;
                int valid = remaining;
                if (valid > tile_n) {
                    valid = tile_n;
                }
                tile_expert[tile] = expert;
                tile_mn_limit[tile] = tile * tile_n + valid;
            }
        }
    }
}

} // extern "C"
