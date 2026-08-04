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

// Frozen Loom schedule plus its TVM-FFI binding in one translation unit.
//
// Frozen device-source provenance: loom/examples/weave/alpha_moe_fused_router.py
// at Cake commit e2aa03274. The latest source-validation head, def2a9dcb,
// retains the same device body while strengthening route-plan coverage checks.
// generate_kernel(..., arch="sm_100a") and sm_103a both produce the same
// 17,780-byte raw source with SHA256
// ec5bc689e68264a11a56a17fb10f699bc3733a521dea916b71ecda51d4227801.
// Both targets are compiled with the source launcher's --use_fast_math option.
//
// The raw generated prelude is transformed mechanically for embedding:
//   1. generated fixed-width typedefs and the unused Loom/CUtensorMap structs
//      are dropped; the host headers below supply canonical CUDA/stdint types;
//   2. generated CUDA includes are hoisted here;
//   3. generated macros are captured for the launcher, then undefined after
//      the kernel. The generated helper and kernel bodies are unchanged.

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1000 && __CUDA_ARCH__ != 1030
#error "AlphaMoE fused router is supported only on SM100a and SM103a"
#endif

// clang-format off

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_VALUES_OFF 0
#define SMEM_WARP_VALUES_STAGE_BYTES 32
#define SMEM_WARP_VALUES_STRIDE 32
#define SMEM_WARP_INDICES_OFF 32
#define SMEM_WARP_INDICES_STAGE_BYTES 32
#define SMEM_WARP_INDICES_STRIDE 32
#define SMEM_SCORES_OFF 64
#define SMEM_SCORES_STAGE_BYTES 2048
#define SMEM_SCORES_STRIDE 2048
#define SMEM_SCAN_VALUES_OFF 64
#define SMEM_SCAN_VALUES_STAGE_BYTES 2048
#define SMEM_SCAN_VALUES_STRIDE 2048
#define SMEM_TOTAL 2176
#define THREADS 256
#define NUM_WARPS 8
#define MAX_EXPERTS 512
#define MAX_TOP_K 16
#define MAX_BLOCK_M 16

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_alpha_moe_fused_router(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
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
    float* warp_values = reinterpret_cast<float*>(smem_raw + 0);
    const int warp_values_addr = smem + 0;
    int* warp_indices = reinterpret_cast<int*>(smem_raw + 32);
    const int warp_indices_addr = smem + 32;
    float* scores = reinterpret_cast<float*>(smem_raw + 64);
    const int scores_addr = smem + 64;
    int* scan_values = reinterpret_cast<int*>(smem_raw + 64);
    const int scan_values_addr = smem + 64;

    // === Task calls (dependency order) ===
    int global_thread = bid * THREADS + tid;
    int expert_zero = tid;
    if (bid == 0 && expert_zero < E) {
        expert_counts[expert_zero] = 0;
        expert_offsets[expert_zero] = 0;
        expert_scatter_offsets[expert_zero] = 0;
    }
    int expert_zero_0 = tid + THREADS;
    if (bid == 0 && expert_zero_0 < E) {
        expert_counts[expert_zero_0] = 0;
        expert_offsets[expert_zero_0] = 0;
        expert_scatter_offsets[expert_zero_0] = 0;
    }
    if (global_thread == 0) {
        expert_offsets[E] = 0;
        num_tokens_post_padded[0] = 0;
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = bid; token < M; token += num_bids) {
        unsigned long long row_base = (unsigned long long)token * (unsigned long long)E;
        int expert_load = tid;
        if (expert_load < routed_experts) {
            scores[expert_load] = logits[row_base + (unsigned long long)expert_load];
        }
        int expert_load_0 = tid + THREADS;
        if (expert_load_0 < routed_experts) {
            scores[expert_load_0] = logits[row_base + (unsigned long long)expert_load_0];
        }
        __syncthreads();
        unsigned long long output_base = (unsigned long long)token * (unsigned long long)top_k;
        #pragma unroll 1
        for (int route = 0; route < routed_top_k; route++) {
            float local_value = -LOOM_INF;
            int local_index = MAX_EXPERTS;
            int expert_scan = tid;
            if (expert_scan < routed_experts) {
                float candidate = scores[expert_scan];
                if (candidate > local_value || candidate == local_value && expert_scan < local_index) {
                    local_value = candidate;
                    local_index = expert_scan;
                }
            }
            int expert_scan_0 = tid + THREADS;
            if (expert_scan_0 < routed_experts) {
                float candidate_1 = scores[expert_scan_0];
                if (candidate_1 > local_value || candidate_1 == local_value && expert_scan_0 < local_index) {
                    local_value = candidate_1;
                    local_index = expert_scan_0;
                }
            }
            float _warp_reduce_0 = local_value;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
            float warp_value = _warp_reduce_0;
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, local_value == warp_value);
            int owner_mask = _vote_0;
            int _ffs_0 = __ffs(owner_mask);
            int owner_lane = _ffs_0 - 1;
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, local_index, owner_lane);
            int warp_index = _shfl_0;
            if (lane == 0) {
                warp_values[warp] = warp_value;
                warp_indices[warp] = warp_index;
            }
            __syncthreads();
            if (tid == 0) {
                float best_value = warp_values[0];
                int best_index = warp_indices[0];
                #pragma unroll
                for (int warp_1 = 1; warp_1 < NUM_WARPS; warp_1++) {
                    float candidate_value = warp_values[warp_1];
                    int candidate_index = warp_indices[warp_1];
                    if (candidate_value > best_value || candidate_value == best_value && candidate_index < best_index) {
                        best_value = candidate_value;
                        best_index = candidate_index;
                    }
                }
                topk_weights[output_base + (unsigned long long)route] = best_value;
                topk_ids[output_base + (unsigned long long)route] = best_index;
                scores[best_index] = -LOOM_INF;
                atomicAdd(&expert_counts[best_index], 1);
            }
            __syncthreads();
        }
        if (has_shared_expert != 0) {
            if (tid == 0) {
                int shared_expert = E - 1;
                int shared_route = top_k - 1;
                float shared_logit = logits[row_base + (unsigned long long)shared_expert];
                topk_weights[output_base + (unsigned long long)shared_route] = shared_logit;
                topk_ids[output_base + (unsigned long long)shared_route] = shared_expert;
                atomicAdd(&expert_counts[shared_expert], 1);
            }
            __syncthreads();
        }
        if (tid == 0) {
            float selected_max = topk_weights[output_base];
            #pragma unroll 1
            for (int route_max = 1; route_max < top_k; route_max++) {
                float _max_0 = max_noftz(selected_max, topk_weights[output_base + (unsigned long long)route_max]);
                selected_max = _max_0;
            }
            float selected_sum = 0.0f;
            #pragma unroll 1
            for (int route_sum = 0; route_sum < top_k; route_sum++) {
                float _exp2_0 = approx_exp2((topk_weights[output_base + (unsigned long long)route_sum] - selected_max) * 1.4426950408889634f);
                selected_sum += _exp2_0;
            }
            float _rcp_0 = approx_rcp(selected_sum);
            float selected_sum_rcp = _rcp_0;
            #pragma unroll 1
            for (int route_store = 0; route_store < top_k; route_store++) {
                float selected_logit = topk_weights[output_base + (unsigned long long)route_store];
                float _exp2_1 = approx_exp2((selected_logit - selected_max) * 1.4426950408889634f);
                topk_weights[output_base + (unsigned long long)route_store] = _exp2_1 * selected_sum_rcp;
            }
        }
        __syncthreads();
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    if (bid == 0) {
        int expert_scan_init = tid;
        int padded_count_init = 0;
        if (expert_scan_init < E) {
            int count_init = expert_counts[expert_scan_init];
            padded_count_init = (count_init + block_m - 1) / block_m * block_m;
        }
        scan_values[expert_scan_init] = padded_count_init;
        int expert_scan_init_0 = tid + THREADS;
        int padded_count_init_1 = 0;
        if (expert_scan_init_0 < E) {
            int count_init_1 = expert_counts[expert_scan_init_0];
            padded_count_init_1 = (count_init_1 + block_m - 1) / block_m * block_m;
        }
        scan_values[expert_scan_init_0] = padded_count_init_1;
        __syncthreads();
        int scan_index_up = (tid + 1) * 2 - 1;
        if (scan_index_up < MAX_EXPERTS) {
            scan_values[scan_index_up] = scan_values[scan_index_up] + scan_values[scan_index_up - 1];
        }
        __syncthreads();
        int scan_index_up_2 = (tid + 1) * 4 - 1;
        if (scan_index_up_2 < MAX_EXPERTS) {
            scan_values[scan_index_up_2] = scan_values[scan_index_up_2] + scan_values[scan_index_up_2 - 2];
        }
        __syncthreads();
        int scan_index_up_3 = (tid + 1) * 8 - 1;
        if (scan_index_up_3 < MAX_EXPERTS) {
            scan_values[scan_index_up_3] = scan_values[scan_index_up_3] + scan_values[scan_index_up_3 - 4];
        }
        __syncthreads();
        int scan_index_up_4 = (tid + 1) * 16 - 1;
        if (scan_index_up_4 < MAX_EXPERTS) {
            scan_values[scan_index_up_4] = scan_values[scan_index_up_4] + scan_values[scan_index_up_4 - 8];
        }
        __syncthreads();
        int scan_index_up_5 = (tid + 1) * 32 - 1;
        if (scan_index_up_5 < MAX_EXPERTS) {
            scan_values[scan_index_up_5] = scan_values[scan_index_up_5] + scan_values[scan_index_up_5 - 16];
        }
        __syncthreads();
        int scan_index_up_6 = (tid + 1) * 64 - 1;
        if (scan_index_up_6 < MAX_EXPERTS) {
            scan_values[scan_index_up_6] = scan_values[scan_index_up_6] + scan_values[scan_index_up_6 - 32];
        }
        __syncthreads();
        int scan_index_up_7 = (tid + 1) * 128 - 1;
        if (scan_index_up_7 < MAX_EXPERTS) {
            scan_values[scan_index_up_7] = scan_values[scan_index_up_7] + scan_values[scan_index_up_7 - 64];
        }
        __syncthreads();
        int scan_index_up_8 = (tid + 1) * 256 - 1;
        if (scan_index_up_8 < MAX_EXPERTS) {
            scan_values[scan_index_up_8] = scan_values[scan_index_up_8] + scan_values[scan_index_up_8 - 128];
        }
        __syncthreads();
        int scan_index_up_9 = (tid + 1) * 512 - 1;
        if (scan_index_up_9 < MAX_EXPERTS) {
            scan_values[scan_index_up_9] = scan_values[scan_index_up_9] + scan_values[scan_index_up_9 - 256];
        }
        __syncthreads();
        if (tid == 0) {
            int padded_total = scan_values[MAX_EXPERTS - 1];
            num_tokens_post_padded[0] = padded_total;
            scan_values[MAX_EXPERTS - 1] = 0;
        }
        __syncthreads();
        int scan_index_down = (tid + 1) * 512 - 1;
        if (scan_index_down < MAX_EXPERTS) {
            int scan_left = scan_values[scan_index_down - 256];
            scan_values[scan_index_down - 256] = scan_values[scan_index_down];
            scan_values[scan_index_down] = scan_values[scan_index_down] + scan_left;
        }
        __syncthreads();
        int scan_index_down_10 = (tid + 1) * 256 - 1;
        if (scan_index_down_10 < MAX_EXPERTS) {
            int scan_left_1 = scan_values[scan_index_down_10 - 128];
            scan_values[scan_index_down_10 - 128] = scan_values[scan_index_down_10];
            scan_values[scan_index_down_10] = scan_values[scan_index_down_10] + scan_left_1;
        }
        __syncthreads();
        int scan_index_down_11 = (tid + 1) * 128 - 1;
        if (scan_index_down_11 < MAX_EXPERTS) {
            int scan_left_2 = scan_values[scan_index_down_11 - 64];
            scan_values[scan_index_down_11 - 64] = scan_values[scan_index_down_11];
            scan_values[scan_index_down_11] = scan_values[scan_index_down_11] + scan_left_2;
        }
        __syncthreads();
        int scan_index_down_12 = (tid + 1) * 64 - 1;
        if (scan_index_down_12 < MAX_EXPERTS) {
            int scan_left_3 = scan_values[scan_index_down_12 - 32];
            scan_values[scan_index_down_12 - 32] = scan_values[scan_index_down_12];
            scan_values[scan_index_down_12] = scan_values[scan_index_down_12] + scan_left_3;
        }
        __syncthreads();
        int scan_index_down_13 = (tid + 1) * 32 - 1;
        if (scan_index_down_13 < MAX_EXPERTS) {
            int scan_left_4 = scan_values[scan_index_down_13 - 16];
            scan_values[scan_index_down_13 - 16] = scan_values[scan_index_down_13];
            scan_values[scan_index_down_13] = scan_values[scan_index_down_13] + scan_left_4;
        }
        __syncthreads();
        int scan_index_down_14 = (tid + 1) * 16 - 1;
        if (scan_index_down_14 < MAX_EXPERTS) {
            int scan_left_5 = scan_values[scan_index_down_14 - 8];
            scan_values[scan_index_down_14 - 8] = scan_values[scan_index_down_14];
            scan_values[scan_index_down_14] = scan_values[scan_index_down_14] + scan_left_5;
        }
        __syncthreads();
        int scan_index_down_15 = (tid + 1) * 8 - 1;
        if (scan_index_down_15 < MAX_EXPERTS) {
            int scan_left_6 = scan_values[scan_index_down_15 - 4];
            scan_values[scan_index_down_15 - 4] = scan_values[scan_index_down_15];
            scan_values[scan_index_down_15] = scan_values[scan_index_down_15] + scan_left_6;
        }
        __syncthreads();
        int scan_index_down_16 = (tid + 1) * 4 - 1;
        if (scan_index_down_16 < MAX_EXPERTS) {
            int scan_left_7 = scan_values[scan_index_down_16 - 2];
            scan_values[scan_index_down_16 - 2] = scan_values[scan_index_down_16];
            scan_values[scan_index_down_16] = scan_values[scan_index_down_16] + scan_left_7;
        }
        __syncthreads();
        int scan_index_down_17 = (tid + 1) * 2 - 1;
        if (scan_index_down_17 < MAX_EXPERTS) {
            int scan_left_8 = scan_values[scan_index_down_17 - 1];
            scan_values[scan_index_down_17 - 1] = scan_values[scan_index_down_17];
            scan_values[scan_index_down_17] = scan_values[scan_index_down_17] + scan_left_8;
        }
        __syncthreads();
        int expert_scan_store = tid;
        if (expert_scan_store < E) {
            expert_offsets[expert_scan_store] = scan_values[expert_scan_store];
        }
        int expert_scan_store_18 = tid + THREADS;
        if (expert_scan_store_18 < E) {
            expert_offsets[expert_scan_store_18] = scan_values[expert_scan_store_18];
        }
        if (tid == 0) {
            expert_offsets[E] = num_tokens_post_padded[0];
        }
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    for (int scatter_token = bid; scatter_token < M; scatter_token += num_bids) {
        if (tid < top_k) {
            int pair = scatter_token * top_k + tid;
            int pair_expert = topk_ids[pair];
            int _atomic_old_0 = atomicAdd(&expert_scatter_offsets[pair_expert], 1);
            int local_row = _atomic_old_0;
            int grouped_row = expert_offsets[pair_expert] + local_row;
            sorted_token_ids[grouped_row] = pair;
            if (local_row % block_m == 0) {
                expert_ids[grouped_row / block_m] = pair_expert;
            }
        }
    }
    int expert_final = tid;
    if (bid == 0 && expert_final < E) {
        int count_final = expert_counts[expert_final];
        int expert_start = expert_offsets[expert_final];
        int padded_count_final = (count_final + block_m - 1) / block_m * block_m;
        int padding_count = padded_count_final - count_final;
        #pragma unroll
        for (int padding_slot = 0; padding_slot < MAX_BLOCK_M; padding_slot++) {
            if (padding_count > padding_slot) {
                sorted_token_ids[expert_start + count_final + padding_slot] = M * top_k;
            }
        }
    }
    int expert_final_1 = tid + THREADS;
    if (bid == 0 && expert_final_1 < E) {
        int count_final_1 = expert_counts[expert_final_1];
        int expert_start_1 = expert_offsets[expert_final_1];
        int padded_count_final_1 = (count_final_1 + block_m - 1) / block_m * block_m;
        int padding_count_1 = padded_count_final_1 - count_final_1;
        #pragma unroll
        for (int padding_slot_1 = 0; padding_slot_1 < MAX_BLOCK_M; padding_slot_1++) {
            if (padding_count_1 > padding_slot_1) {
                sorted_token_ids[expert_start_1 + count_final_1 + padding_slot_1] = M * top_k;
            }
        }
    }
}

} // extern "C"

constexpr int kGeneratedThreads = THREADS;
constexpr int kGeneratedSmemTotal = SMEM_TOTAL;

#undef LOOM_INF
#undef MAX_BLOCK_M
#undef MAX_EXPERTS
#undef MAX_TOP_K
#undef NUM_MAIN_STAGES
#undef NUM_WARPS
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_SCORES_OFF
#undef SMEM_SCORES_STAGE_BYTES
#undef SMEM_SCORES_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WARP_INDICES_OFF
#undef SMEM_WARP_INDICES_STAGE_BYTES
#undef SMEM_WARP_INDICES_STRIDE
#undef SMEM_WARP_VALUES_OFF
#undef SMEM_WARP_VALUES_STAGE_BYTES
#undef SMEM_WARP_VALUES_STRIDE
#undef THREADS

// clang-format on

namespace flashinfer {
namespace alphamoe_fused_router {

constexpr int64_t kMaxExperts = 512;
constexpr int64_t kMaxTopK = 16;
constexpr int64_t kMaxBlockM = 16;
constexpr int64_t kThreads = 256;
constexpr int64_t kDynamicSmemBytes = 2176;

static_assert(kGeneratedThreads == kThreads);
static_assert(kGeneratedSmemTotal == kDynamicSmemBytes);

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckTensor(const TensorView& tensor, const char* name, DLDataType dtype, int64_t ndim,
                        int32_t device_id) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id << ", got " << tensor.device().device_id;
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK(tensor.dtype() == dtype) << name << " has an unsupported dtype";
  TVM_FFI_ICHECK(tensor.ndim() == ndim)
      << name << " must be rank " << ndim << ", got rank " << tensor.ndim();
}

struct TensorRange {
  uintptr_t begin;
  uintptr_t end;
};

inline TensorRange GetTensorRange(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " must have a byte-addressable dtype";
  const uint64_t bytes_per_element = bits / 8;
  TVM_FFI_ICHECK(static_cast<uint64_t>(tensor.numel()) <=
                 std::numeric_limits<uint64_t>::max() / bytes_per_element)
      << name << " byte count overflows uint64_t";
  const uint64_t bytes = static_cast<uint64_t>(tensor.numel()) * bytes_per_element;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoAlias(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                         const char* rhs_name) {
  const TensorRange lhs_range = GetTensorRange(lhs, lhs_name);
  const TensorRange rhs_range = GetTensorRange(rhs, rhs_name);
  TVM_FFI_ICHECK(!(lhs_range.begin < rhs_range.end && rhs_range.begin < lhs_range.end))
      << lhs_name << " must not overlap " << rhs_name
      << ": the frozen kernel uses __restrict__ pointers";
}

inline int64_t MaxRouteBlocks(int64_t m, int64_t top_k, int64_t num_experts, int64_t block_m) {
  const int64_t pairs = m * top_k;
  const int64_t nonempty = std::min(num_experts, pairs);
  return nonempty + (pairs - nonempty) / block_m;
}

void Run(TensorView logits, TensorView topk_weights, TensorView topk_ids,
         TensorView sorted_token_ids, TensorView expert_ids, TensorView num_tokens_post_padded,
         TensorView expert_counts, TensorView expert_offsets, TensorView expert_scatter_offsets,
         int64_t top_k, int64_t block_m, bool has_shared_expert) {
  TVM_FFI_ICHECK(logits.device().device_type == kDLCUDA) << "logits must be a CUDA tensor";
  const int32_t device_id = logits.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);

  CheckTensor(logits, "logits", dl_float32, 2, device_id);
  CheckTensor(topk_weights, "topk_weights", dl_float32, 2, device_id);
  CheckTensor(topk_ids, "topk_ids", dl_int32, 2, device_id);
  CheckTensor(sorted_token_ids, "sorted_token_ids", dl_int32, 1, device_id);
  CheckTensor(expert_ids, "expert_ids", dl_int32, 1, device_id);
  CheckTensor(num_tokens_post_padded, "num_tokens_post_padded", dl_int32, 1, device_id);
  CheckTensor(expert_counts, "expert_counts", dl_int32, 1, device_id);
  CheckTensor(expert_offsets, "expert_offsets", dl_int32, 1, device_id);
  CheckTensor(expert_scatter_offsets, "expert_scatter_offsets", dl_int32, 1, device_id);

  const int64_t m = logits.size(0);
  const int64_t num_experts = logits.size(1);
  TVM_FFI_ICHECK(m > 0) << "logits must contain at least one token";
  TVM_FFI_ICHECK(num_experts >= 1 && num_experts <= kMaxExperts)
      << "num_experts must be in [1, " << kMaxExperts << "], got " << num_experts;
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= std::min(num_experts, kMaxTopK))
      << "top_k must be in [1, min(num_experts, " << kMaxTopK << ")], got " << top_k;
  TVM_FFI_ICHECK(block_m >= 1 && block_m <= kMaxBlockM)
      << "block_m must be in [1, " << kMaxBlockM << "], got " << block_m;
  TVM_FFI_ICHECK(!has_shared_expert || top_k >= 2) << "a forced shared expert requires top_k >= 2";
  TVM_FFI_ICHECK(m <= std::numeric_limits<int>::max()) << "num_tokens must fit in int32";
  TVM_FFI_ICHECK(m * top_k <= std::numeric_limits<int>::max())
      << "num_tokens * top_k must fit in int32";

  const int64_t max_route_blocks = MaxRouteBlocks(m, top_k, num_experts, block_m);
  TVM_FFI_ICHECK(max_route_blocks <= std::numeric_limits<int>::max() / block_m)
      << "maximum padded route count exceeds int32";
  const int64_t max_padded_pairs = max_route_blocks * block_m;

  TVM_FFI_ICHECK(topk_weights.size(0) == m && topk_weights.size(1) == top_k)
      << "topk_weights must have shape (" << m << ", " << top_k << ")";
  TVM_FFI_ICHECK(topk_ids.size(0) == m && topk_ids.size(1) == top_k)
      << "topk_ids must have shape (" << m << ", " << top_k << ")";
  TVM_FFI_ICHECK(sorted_token_ids.numel() >= max_padded_pairs)
      << "sorted_token_ids capacity must be at least " << max_padded_pairs;
  TVM_FFI_ICHECK(expert_ids.numel() >= max_route_blocks)
      << "expert_ids capacity must be at least " << max_route_blocks;
  TVM_FFI_ICHECK(num_tokens_post_padded.numel() == 1)
      << "num_tokens_post_padded must contain exactly one int32 element";
  TVM_FFI_ICHECK(expert_counts.numel() == num_experts)
      << "expert_counts must have shape (" << num_experts << ",)";
  TVM_FFI_ICHECK(expert_offsets.numel() == num_experts + 1)
      << "expert_offsets must have shape (" << num_experts + 1 << ",)";
  TVM_FFI_ICHECK(expert_scatter_offsets.numel() == num_experts)
      << "expert_scatter_offsets must have shape (" << num_experts << ",)";

  const std::array<const TensorView*, 9> tensors = {
      &logits,           &topk_weights,   &topk_ids,
      &sorted_token_ids, &expert_ids,     &num_tokens_post_padded,
      &expert_counts,    &expert_offsets, &expert_scatter_offsets,
  };
  const std::array<const char*, 9> names = {
      "logits",           "topk_weights",   "topk_ids",
      "sorted_token_ids", "expert_ids",     "num_tokens_post_padded",
      "expert_counts",    "expert_offsets", "expert_scatter_offsets",
  };
  for (size_t i = 0; i < tensors.size(); ++i) {
    for (size_t j = i + 1; j < tensors.size(); ++j) {
      CheckNoAlias(*tensors[i], names[i], *tensors[j], names[j]);
    }
  }

  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, device_id), "cudaGetDeviceProperties");
  TVM_FFI_ICHECK(properties.major == 10 && (properties.minor == 0 || properties.minor == 3))
      << "AlphaMoE fused router requires compute capability 10.0 or 10.3, got " << properties.major
      << "." << properties.minor;
  int cooperative_launch = 0;
  CheckCuda(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device_id),
            "cudaDeviceGetAttribute(cudaDevAttrCooperativeLaunch)");
  TVM_FFI_ICHECK(cooperative_launch != 0)
      << "AlphaMoE fused router requires cooperative-launch support";

  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kDynamicSmemBytes)),
            "cudaFuncSetAttribute(AlphaMoE router dynamic smem)");
  int active_blocks_per_sm = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_sm, kernel_alpha_moe_fused_router, static_cast<int>(kThreads),
                static_cast<size_t>(kDynamicSmemBytes)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE router)");
  TVM_FFI_ICHECK(active_blocks_per_sm > 0)
      << "AlphaMoE fused router has zero cooperative occupancy";

  const int64_t grid_x = std::min<int64_t>(m, properties.multiProcessorCount);
  const int64_t cooperative_capacity =
      static_cast<int64_t>(active_blocks_per_sm) * properties.multiProcessorCount;
  TVM_FFI_ICHECK(grid_x >= 1 && grid_x <= cooperative_capacity)
      << "AlphaMoE fused router grid " << grid_x << " exceeds cooperative residency capacity "
      << cooperative_capacity;

  float* logits_ptr = static_cast<float*>(logits.data_ptr());
  float* topk_weights_ptr = static_cast<float*>(topk_weights.data_ptr());
  int* topk_ids_ptr = static_cast<int*>(topk_ids.data_ptr());
  int* sorted_token_ids_ptr = static_cast<int*>(sorted_token_ids.data_ptr());
  int* expert_ids_ptr = static_cast<int*>(expert_ids.data_ptr());
  int* num_tokens_post_padded_ptr = static_cast<int*>(num_tokens_post_padded.data_ptr());
  int* expert_counts_ptr = static_cast<int*>(expert_counts.data_ptr());
  int* expert_offsets_ptr = static_cast<int*>(expert_offsets.data_ptr());
  int* expert_scatter_offsets_ptr = static_cast<int*>(expert_scatter_offsets.data_ptr());
  int m_arg = static_cast<int>(m);
  int num_experts_arg = static_cast<int>(num_experts);
  int top_k_arg = static_cast<int>(top_k);
  int block_m_arg = static_cast<int>(block_m);
  int has_shared_expert_arg = static_cast<int>(has_shared_expert);
  void* arguments[] = {
      &logits_ptr,
      &topk_weights_ptr,
      &topk_ids_ptr,
      &sorted_token_ids_ptr,
      &expert_ids_ptr,
      &num_tokens_post_padded_ptr,
      &expert_counts_ptr,
      &expert_offsets_ptr,
      &expert_scatter_offsets_ptr,
      &m_arg,
      &num_experts_arg,
      &top_k_arg,
      &block_m_arg,
      &has_shared_expert_arg,
  };

  const cudaStream_t stream = get_stream(logits.device());
  CheckCuda(
      cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel_alpha_moe_fused_router),
                                  dim3(static_cast<unsigned int>(grid_x), 1, 1),
                                  dim3(static_cast<unsigned int>(kThreads), 1, 1), arguments,
                                  static_cast<size_t>(kDynamicSmemBytes), stream),
      "cudaLaunchCooperativeKernel(AlphaMoE fused router)");
}

}  // namespace alphamoe_fused_router
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_router_op, flashinfer::alphamoe_fused_router::Run);
