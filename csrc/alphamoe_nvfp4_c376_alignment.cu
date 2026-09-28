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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>


namespace nvfp4_qualified_c376_alignment {
#define kernel_alpha_moe_route_alignment_parallel_map_init kernel_alpha_moe_route_alignment_parallel_map_init_nvfp4_qualified_c376_alignment
__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_COUNTS_OFF 0
#define SMEM_COUNTS_STAGE_BYTES 4096
#define SMEM_COUNTS_STRIDE 4096
#define SMEM_OFFSETS_OFF 4096
#define SMEM_OFFSETS_STAGE_BYTES 4100
#define SMEM_OFFSETS_STRIDE 4100
#define SMEM_SCATTER_OFF 8196
#define SMEM_SCATTER_STAGE_BYTES 4096
#define SMEM_SCATTER_STRIDE 4096
#define SMEM_WARP_PREFIX_OFF 12292
#define SMEM_WARP_PREFIX_STAGE_BYTES 32
#define SMEM_WARP_PREFIX_STRIDE 32
#define SMEM_TOTAL_OFF 12324
#define SMEM_TOTAL_STAGE_BYTES 4
#define SMEM_TOTAL_STRIDE 4
#define SMEM_OWNER_WARP_PREFIX_OFF 12328
#define SMEM_OWNER_WARP_PREFIX_STAGE_BYTES 32
#define SMEM_OWNER_WARP_PREFIX_STRIDE 32
#define SMEM_OWNER_TOTAL_OFF 12360
#define SMEM_OWNER_TOTAL_STAGE_BYTES 4
#define SMEM_OWNER_TOTAL_STRIDE 4
#define SMEM_TOTAL 12416
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_alpha_moe_route_alignment_parallel_map_init(int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ cumsum_buffer, int* __restrict__ compact_owner_plan, int* __restrict__ compact_owner_count, int* __restrict__ route_experts, int num_experts, int block_size, int num_pairs, int sorted_capacity)
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
    int* scatter = reinterpret_cast<int*>(smem_raw + 8196);
    const int scatter_addr = smem + 8196;
    int* warp_prefix = reinterpret_cast<int*>(smem_raw + 12292);
    const int warp_prefix_addr = smem + 12292;
    int* total = reinterpret_cast<int*>(smem_raw + 12324);
    const int total_addr = smem + 12324;
    int* owner_warp_prefix = reinterpret_cast<int*>(smem_raw + 12328);
    const int owner_warp_prefix_addr = smem + 12328;
    int* owner_total = reinterpret_cast<int*>(smem_raw + 12360);
    const int owner_total_addr = smem + 12360;

    // === Task calls (dependency order) ===
    if (blockIdx.x == 0) {
        for (int bin_zero = tid; bin_zero < num_experts; bin_zero += 256) {
            counts[bin_zero] = 0;
            scatter[bin_zero] = 0;
        }
        for (int pad = tid; pad < sorted_capacity; pad += 256) {
            sorted_token_ids[pad] = num_pairs;
        }
        __syncthreads();
        for (int pair_load = tid; pair_load < num_pairs; pair_load += 256) {
            int pair_bin = topk_ids[pair_load] + 1;
            atomicAdd(&counts[pair_bin], 1);
        }
        __syncthreads();
        int padded[4];
        int owner_counts[4];
        int thread_owner_total = 0;
        int thread_total = 0;
        #pragma unroll
        for (int slot = 0; slot < 4; slot++) {
            int bin_scan = tid * 4 + slot;
            int padded_count = 0;
            if (bin_scan < num_experts) {
                int count = counts[bin_scan];
                if (block_size == 8) {
                    padded_count = count + 7 & -8;
                } else {
                    padded_count = count + 15 & -16;
                }
            }
            int owner_count = (padded_count + 31) / 32;
            owner_counts[slot] = owner_count;
            thread_owner_total = thread_owner_total + owner_count;
            padded[slot] = padded_count;
            thread_total = thread_total + padded_count;
        }
        int inclusive = thread_total;
        int owner_inclusive = thread_owner_total;
        int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
        int peer = _shfl_up_0;
        int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, owner_inclusive, 1, 32);
        int owner_peer = _shfl_up_1;
        if (lane >= 1) {
            inclusive = inclusive + peer;
            owner_inclusive = owner_inclusive + owner_peer;
        }
        int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
        int peer_0 = _shfl_up_2;
        int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, owner_inclusive, 2, 32);
        int owner_peer_1 = _shfl_up_3;
        if (lane >= 2) {
            inclusive = inclusive + peer_0;
            owner_inclusive = owner_inclusive + owner_peer_1;
        }
        int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
        int peer_2 = _shfl_up_4;
        int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, owner_inclusive, 4, 32);
        int owner_peer_3 = _shfl_up_5;
        if (lane >= 4) {
            inclusive = inclusive + peer_2;
            owner_inclusive = owner_inclusive + owner_peer_3;
        }
        int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
        int peer_4 = _shfl_up_6;
        int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, owner_inclusive, 8, 32);
        int owner_peer_5 = _shfl_up_7;
        if (lane >= 8) {
            inclusive = inclusive + peer_4;
            owner_inclusive = owner_inclusive + owner_peer_5;
        }
        int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
        int peer_6 = _shfl_up_8;
        int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, owner_inclusive, 16, 32);
        int owner_peer_7 = _shfl_up_9;
        if (lane >= 16) {
            inclusive = inclusive + peer_6;
            owner_inclusive = owner_inclusive + owner_peer_7;
        }
        if (lane == 31) {
            warp_prefix[warp] = inclusive;
            owner_warp_prefix[warp] = owner_inclusive;
        }
        __syncthreads();
        if (warp == 0) {
            int warp_total = 0;
            int owner_warp_total = 0;
            if (lane < 8) {
                warp_total = warp_prefix[lane];
                owner_warp_total = owner_warp_prefix[lane];
            }
            int warp_inclusive = warp_total;
            int owner_warp_inclusive = owner_warp_total;
            int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 1, 32);
            int warp_peer = _shfl_up_10;
            int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, owner_warp_inclusive, 1, 32);
            int owner_warp_peer = _shfl_up_11;
            if (lane >= 1) {
                warp_inclusive = warp_inclusive + warp_peer;
                owner_warp_inclusive = owner_warp_inclusive + owner_warp_peer;
            }
            int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 2, 32);
            int warp_peer_0 = _shfl_up_12;
            int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, owner_warp_inclusive, 2, 32);
            int owner_warp_peer_1 = _shfl_up_13;
            if (lane >= 2) {
                warp_inclusive = warp_inclusive + warp_peer_0;
                owner_warp_inclusive = owner_warp_inclusive + owner_warp_peer_1;
            }
            int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 4, 32);
            int warp_peer_2 = _shfl_up_14;
            int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, owner_warp_inclusive, 4, 32);
            int owner_warp_peer_3 = _shfl_up_15;
            if (lane >= 4) {
                warp_inclusive = warp_inclusive + warp_peer_2;
                owner_warp_inclusive = owner_warp_inclusive + owner_warp_peer_3;
            }
            if (lane < 8) {
                warp_prefix[lane] = warp_inclusive - warp_total;
                owner_warp_prefix[lane] = owner_warp_inclusive - owner_warp_total;
            }
            if (lane == 7) {
                total[0] = warp_inclusive;
                owner_total[0] = owner_warp_inclusive;
            }
        }
        __syncthreads();
        int prefix = warp_prefix[warp] + inclusive - thread_total;
        int owner_prefix = owner_warp_prefix[warp] + owner_inclusive - thread_owner_total;
        #pragma unroll
        for (int publish_slot = 0; publish_slot < 4; publish_slot++) {
            int publish_bin = tid * 4 + publish_slot;
            if (publish_bin < num_experts) {
                offsets[publish_bin] = prefix;
                #pragma unroll 1
                for (int owner = 0; owner < owner_counts[publish_slot]; owner++) {
                    int record = (owner_prefix + owner) * 3;
                    compact_owner_plan[record] = prefix / 8 + owner * 4;
                    compact_owner_plan[record + 1] = publish_bin - 1;
                    int _min_0 = ((padded[publish_slot] / 8 - owner * 4) < (4) ? (padded[publish_slot] / 8 - owner * 4) : (4));
                    compact_owner_plan[record + 2] = _min_0;
                }
            }
            prefix = prefix + padded[publish_slot];
            owner_prefix = owner_prefix + owner_counts[publish_slot];
        }
        if (tid == 0) {
            offsets[num_experts] = total[0];
        }
        __syncthreads();
        for (int pair = tid; pair < num_pairs; pair += 256) {
            int bin_scatter = topk_ids[pair] + 1;
            int _atomic_old_0 = atomicAdd(&scatter[bin_scatter], 1);
            int local_row = _atomic_old_0;
            int destination = offsets[bin_scatter] + local_row;
            sorted_token_ids[destination] = pair;
            if (block_size == 8) {
                if ((local_row & 7) == 0) {
                    expert_ids[destination >> 3] = bin_scatter - 1;
                }
            } else if ((local_row & 15) == 0) {
                expert_ids[destination >> 4] = bin_scatter - 1;
            }
        }
        __syncthreads();
        for (int workspace_bin = tid; workspace_bin < num_experts; workspace_bin += 256) {
            cumsum_buffer[workspace_bin] = offsets[workspace_bin] + scatter[workspace_bin];
        }
        if (tid == 0) {
            cumsum_buffer[num_experts] = total[0];
            num_tokens_post_padded[0] = total[0];
            compact_owner_count[0] = owner_total[0];
        }
    } else {
        int route_index = (blockIdx.x - 1) * 256 + tid;
        if (route_index < num_pairs) {
            route_experts[route_index] = -1;
        }
    }
}

} // extern "C"


constexpr int kGeneratedThreads = THREADS;
constexpr int kGeneratedSmemTotal = SMEM_TOTAL;
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_COUNTS_OFF
#undef SMEM_COUNTS_STAGE_BYTES
#undef SMEM_COUNTS_STRIDE
#undef SMEM_OFFSETS_OFF
#undef SMEM_OFFSETS_STAGE_BYTES
#undef SMEM_OFFSETS_STRIDE
#undef SMEM_SCATTER_OFF
#undef SMEM_SCATTER_STAGE_BYTES
#undef SMEM_SCATTER_STRIDE
#undef SMEM_WARP_PREFIX_OFF
#undef SMEM_WARP_PREFIX_STAGE_BYTES
#undef SMEM_WARP_PREFIX_STRIDE
#undef SMEM_TOTAL_OFF
#undef SMEM_TOTAL_STAGE_BYTES
#undef SMEM_TOTAL_STRIDE
#undef SMEM_OWNER_WARP_PREFIX_OFF
#undef SMEM_OWNER_WARP_PREFIX_STAGE_BYTES
#undef SMEM_OWNER_WARP_PREFIX_STRIDE
#undef SMEM_OWNER_TOTAL_OFF
#undef SMEM_OWNER_TOTAL_STAGE_BYTES
#undef SMEM_OWNER_TOTAL_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef kernel_alpha_moe_route_alignment_parallel_map_init
}  // namespace nvfp4_qualified_c376_alignment
