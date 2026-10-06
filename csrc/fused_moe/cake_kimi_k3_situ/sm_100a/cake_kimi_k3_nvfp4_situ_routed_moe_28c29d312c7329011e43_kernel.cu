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
#define SMEM_WARP_OFFSETS_OFF 0
#define SMEM_WARP_OFFSETS_STAGE_BYTES 64
#define SMEM_WARP_OFFSETS_STRIDE 64
#define SMEM_TOTAL 128
#define THREADS 512

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_28c29d312c7329011e43(int* __restrict__ expert_counts, int* __restrict__ expert_tile_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ total_tiles, int num_experts, int tile_n, int tile_n_shift)
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
    int* warp_offsets = reinterpret_cast<int*>(smem_raw + 0);
    const int warp_offsets_addr = smem + 0;

    // === Task calls (dependency order) ===
    int expert_even = tid * 2;
    int expert_odd = expert_even + 1;
    int even_tiles = 0;
    int odd_tiles = 0;
    if (expert_even < num_experts) {
        int count_even = expert_counts[expert_even];
        even_tiles = count_even + tile_n - 1 >> tile_n_shift;
    }
    if (expert_odd < num_experts) {
        int count_odd = expert_counts[expert_odd];
        odd_tiles = count_odd + tile_n - 1 >> tile_n_shift;
    }
    int pair_tiles = even_tiles + odd_tiles;
    int pair_prefix = pair_tiles;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, pair_prefix, 1, 32);
    int prior = _shfl_up_0;
    if (lane >= 1) {
        pair_prefix = pair_prefix + prior;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, pair_prefix, 2, 32);
    int prior_0 = _shfl_up_1;
    if (lane >= 2) {
        pair_prefix = pair_prefix + prior_0;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, pair_prefix, 4, 32);
    int prior_1 = _shfl_up_2;
    if (lane >= 4) {
        pair_prefix = pair_prefix + prior_1;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, pair_prefix, 8, 32);
    int prior_2 = _shfl_up_3;
    if (lane >= 8) {
        pair_prefix = pair_prefix + prior_2;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, pair_prefix, 16, 32);
    int prior_3 = _shfl_up_4;
    if (lane >= 16) {
        pair_prefix = pair_prefix + prior_3;
    }
    if (lane == 31) {
        warp_offsets[warp] = pair_prefix;
    }
    __syncthreads();
    if (warp == 0) {
        int warp_tiles = 0;
        if (lane < 16) {
            warp_tiles = warp_offsets[lane];
        }
        int warp_prefix = warp_tiles;
        int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, warp_prefix, 1, 32);
        int prior_warp = _shfl_up_5;
        if (lane >= 1) {
            warp_prefix = warp_prefix + prior_warp;
        }
        int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, warp_prefix, 2, 32);
        int prior_warp_0 = _shfl_up_6;
        if (lane >= 2) {
            warp_prefix = warp_prefix + prior_warp_0;
        }
        int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, warp_prefix, 4, 32);
        int prior_warp_1 = _shfl_up_7;
        if (lane >= 4) {
            warp_prefix = warp_prefix + prior_warp_1;
        }
        int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, warp_prefix, 8, 32);
        int prior_warp_2 = _shfl_up_8;
        if (lane >= 8) {
            warp_prefix = warp_prefix + prior_warp_2;
        }
        int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, warp_prefix, 16, 32);
        int prior_warp_3 = _shfl_up_9;
        if (lane >= 16) {
            warp_prefix = warp_prefix + prior_warp_3;
        }
        if (lane < 16) {
            warp_offsets[lane] = warp_prefix - warp_tiles;
        }
        if (lane == 15) {
            total_tiles[0] = warp_prefix;
        }
    }
    __syncthreads();
    int pair_base = warp_offsets[warp] + pair_prefix - pair_tiles;
    if (expert_even < num_experts) {
        expert_tile_offsets[expert_even] = pair_base;
        expert_scatter_offsets[expert_even] = 0;
    }
    if (expert_odd < num_experts) {
        expert_tile_offsets[expert_odd] = pair_base + even_tiles;
        expert_scatter_offsets[expert_odd] = 0;
    }
}

} // extern "C"
