/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cooperative_groups.h>

#include <cstdint>
#include <type_traits>

#include "flashinfer/exception.h"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

namespace tensorrt_llm::kernels::moe_alltoall {

#define ENABLE_DEBUG_PRINT 0
#define DISABLE_SYNC_FOR_PROFILING 0

constexpr int kEp4Size = 4;
constexpr int kCompactDispatchMaxPayloadBytes = 1024;
constexpr int kCompactDispatchBlockSize = 128;

#ifndef DISABLE_TIMEOUT
#define DISABLE_TIMEOUT 0
#endif

// Helper function for ceiling division
template <typename T>
__host__ __device__ inline T ceilDiv(T m, T n) {
  return (m + n - 1) / n;
}

// Macros for concise launch-time specialization
#define SWITCH_BOOL(flag, NAME, ...) \
  if (flag) {                        \
    constexpr bool NAME = true;      \
    __VA_ARGS__                      \
  } else {                           \
    constexpr bool NAME = false;     \
    __VA_ARGS__                      \
  }

#define SWITCH_TOP_K(top_k, TOP_K, ...)             \
  switch (top_k) {                                  \
    case 22: {                                      \
      constexpr int TOP_K = 22;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 16: {                                      \
      constexpr int TOP_K = 16;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 10: {                                      \
      constexpr int TOP_K = 10;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 8: {                                       \
      constexpr int TOP_K = 8;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 6: {                                       \
      constexpr int TOP_K = 6;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 4: {                                       \
      constexpr int TOP_K = 4;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 2: {                                       \
      constexpr int TOP_K = 2;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 1: {                                       \
      constexpr int TOP_K = 1;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    default: {                                      \
      FLASHINFER_CHECK(false, "Unsupported top_k"); \
    }                                               \
  }

#define SWITCH_DTYPE(dtype, TYPE, ...)                                  \
  switch (dtype) {                                                      \
    case nvinfer1::DataType::kHALF: {                                   \
      using TYPE = half;                                                \
      __VA_ARGS__;                                                      \
      break;                                                            \
    }                                                                   \
    case nvinfer1::DataType::kBF16: {                                   \
      using TYPE = __nv_bfloat16;                                       \
      __VA_ARGS__;                                                      \
      break;                                                            \
    }                                                                   \
    case nvinfer1::DataType::kFLOAT: {                                  \
      using TYPE = float;                                               \
      __VA_ARGS__;                                                      \
      break;                                                            \
    }                                                                   \
    default: {                                                          \
      FLASHINFER_CHECK(false, "Unsupported dtype for moe_a2a_combine"); \
    }                                                                   \
  }

// Quantized combine currently only supports FP16/BF16 inputs.
#define SWITCH_QUANT_MODE(dtype, quant_mode, QUANT_MODE, ...)                          \
  if constexpr (std::is_same_v<dtype, half> || std::is_same_v<dtype, __nv_bfloat16>) { \
    switch (quant_mode) {                                                              \
      case MoeA2ACombineQuantMode::NONE: {                                             \
        constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::NONE;                      \
        __VA_ARGS__;                                                                   \
        break;                                                                         \
      }                                                                                \
      case MoeA2ACombineQuantMode::MXFP8: {                                            \
        constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::MXFP8;                     \
        __VA_ARGS__;                                                                   \
        break;                                                                         \
      }                                                                                \
      case MoeA2ACombineQuantMode::NVFP4: {                                            \
        constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::NVFP4;                     \
        __VA_ARGS__;                                                                   \
        break;                                                                         \
      }                                                                                \
      case MoeA2ACombineQuantMode::MXFP4: {                                            \
        constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::MXFP4;                     \
        __VA_ARGS__;                                                                   \
        break;                                                                         \
      }                                                                                \
      default: {                                                                       \
        FLASHINFER_CHECK(false, "Unsupported quant_mode for moe_a2a_combine");         \
      }                                                                                \
    }                                                                                  \
  } else {                                                                             \
    FLASHINFER_CHECK((quant_mode) == MoeA2ACombineQuantMode::NONE,                     \
                     "All to All Combine Quantization currently only supports "        \
                     "FP16/BF16");                                                     \
    constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::NONE;                          \
    __VA_ARGS__;                                                                       \
  }

#define SWITCH_SWIZZLE_MODE(swizzle_mode, SWIZZLE_MODE, ...)                   \
  switch (swizzle_mode) {                                                      \
    case MoeA2ACombineSwizzleSFMode::LINEAR: {                                 \
      constexpr auto SWIZZLE_MODE = MoeA2ACombineSwizzleSFMode::LINEAR;        \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case MoeA2ACombineSwizzleSFMode::SWIZZLE_128x4: {                          \
      constexpr auto SWIZZLE_MODE = MoeA2ACombineSwizzleSFMode::SWIZZLE_128x4; \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case MoeA2ACombineSwizzleSFMode::SWIZZLE_8x4: {                            \
      constexpr auto SWIZZLE_MODE = MoeA2ACombineSwizzleSFMode::SWIZZLE_8x4;   \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      FLASHINFER_CHECK(false, "Unsupported swizzle_mode for moe_a2a_combine"); \
    }                                                                          \
  }

#if DISABLE_TIMEOUT
#define check_timeout(s) false
#else
// 300 * 2000 MHz - should be high enough on any GPU but will prevent a hang
#define check_timeout(s) ((clock64() - (s)) > (300ll * 2000ll * 1000ll * 1000ll))
#endif

// ============================================================================
// Helper Functions for Expert-to-Rank Mapping
// ============================================================================

__device__ int compute_target_rank_id(int expert_id, int num_experts_per_rank) {
  // Compute which rank owns a given expert using contiguous partitioning
  // Experts are divided evenly across EP ranks:
  // - Rank 0 gets experts [0, num_experts_per_rank)
  // - Rank 1 gets experts [num_experts_per_rank, 2*num_experts_per_rank)
  // - etc.
  // Example: 32 experts, 4 ranks -> 8 experts per rank
  // - Rank 0: experts 0-7
  // - Rank 1: experts 8-15
  // - Rank 2: experts 16-23
  // - Rank 3: experts 24-31
  return expert_id / num_experts_per_rank;
}

// ============================================================================
// Helper Functions for Vectorized Memory Operations
// ============================================================================

template <int VEC_SIZE>
__device__ void vectorized_copy_impl(void* dst, void const* src, int size) {
  using flashinfer::vec_t;

  uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
  uint8_t const* src_ptr = static_cast<uint8_t const*>(src);

  int const stride = blockDim.x * VEC_SIZE;

  for (int offset = threadIdx.x * VEC_SIZE; offset < size; offset += stride) {
    vec_t<uint8_t, VEC_SIZE> v;
    v.load(src_ptr + offset);
    v.store(dst_ptr + offset);
  }
}

__device__ void vectorized_copy(void* dst, void const* src, int size) {
  if (size % 16 == 0) {
    vectorized_copy_impl<16>(dst, src, size);
  } else if (size % 8 == 0) {
    vectorized_copy_impl<8>(dst, src, size);
  } else if (size % 4 == 0) {
    vectorized_copy_impl<4>(dst, src, size);
  } else if (size % 2 == 0) {
    vectorized_copy_impl<2>(dst, src, size);
  } else {
    vectorized_copy_impl<1>(dst, src, size);
  }
}

// Vectorized dispatch: load one vec from source and write to up to TOP_K destinations
template <int VEC_SIZE, int TOP_K>
__device__ void vectorized_dispatch_impl(uint8_t const* src_ptr, int bytes_per_token, int rank_id,
                                         int max_tokens_per_rank, int payload_idx,
                                         DispatchKernelPointers const& ptrs,
                                         int const* topk_target_ranks,
                                         int const* topk_send_indices) {
  using flashinfer::vec_t;

  // Precompute destination base pointers per k
  uint8_t* dst_base_k[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    int dst_idx_k = topk_send_indices[k];
    int target_rank_k = topk_target_ranks[k];
    if (dst_idx_k < 0) {
      dst_base_k[k] = nullptr;
      continue;
    }
    uint8_t* dst_data = static_cast<uint8_t*>(ptrs.recv_buffers[target_rank_k][payload_idx]);
    size_t base_source_rank =
        static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank) +
        static_cast<size_t>(dst_idx_k);
    size_t base_token = base_source_rank * static_cast<size_t>(bytes_per_token);
    dst_base_k[k] = dst_data + base_token;
  }

  // TODO: process all payloads. index could be reused.
  int const stride = blockDim.x * VEC_SIZE;
  for (int offset = threadIdx.x * VEC_SIZE; offset < bytes_per_token; offset += stride) {
    vec_t<uint8_t, VEC_SIZE> v;
    v.load(src_ptr + offset);

#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      uint8_t* dst_base = dst_base_k[k];
      if (dst_base == nullptr) {
        continue;
      }
      v.store(dst_base + offset);
    }
  }
}

template <int TOP_K>
__device__ void vectorized_dispatch(uint8_t const* src_ptr, int bytes_per_token, int rank_id,
                                    int max_tokens_per_rank, int payload_idx,
                                    DispatchKernelPointers const& ptrs,
                                    int const* topk_target_ranks, int const* topk_send_indices) {
  if (bytes_per_token % 16 == 0) {
    vectorized_dispatch_impl<16, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                        payload_idx, ptrs, topk_target_ranks, topk_send_indices);
  } else if (bytes_per_token % 8 == 0) {
    vectorized_dispatch_impl<8, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                       payload_idx, ptrs, topk_target_ranks, topk_send_indices);
  } else if (bytes_per_token % 4 == 0) {
    vectorized_dispatch_impl<4, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                       payload_idx, ptrs, topk_target_ranks, topk_send_indices);
  } else if (bytes_per_token % 2 == 0) {
    vectorized_dispatch_impl<2, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                       payload_idx, ptrs, topk_target_ranks, topk_send_indices);
  } else {
    vectorized_dispatch_impl<1, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                       payload_idx, ptrs, topk_target_ranks, topk_send_indices);
  }
}

__global__ void moeA2APrepareDispatchKernel(int* send_counters, int* local_token_counter,
                                            int ep_size, uint32_t* flag_val_ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Zero send_counters
  if (idx < ep_size) {
    send_counters[idx] = 0;
  }
  // Zero local_token_counter and increment flag_val
  if (idx == 0) {
    *local_token_counter = 0;
    // Increment flag_val for this dispatch round
    *flag_val_ptr = *flag_val_ptr + 1;
  }
}

// ============================================================================
// Generic Dispatch Kernel Implementation
// One CTA processes one token and all its payloads.
// ============================================================================

template <int TOP_K, bool COMPACT_EP4>
__global__ void moeA2ADispatchKernel(
    int32_t const* token_selected_experts,  // [local_num_tokens, TOP_K]
    const DispatchKernelPointers ptrs,      // Struct containing all kernel pointers
    int num_payloads,                       // Number of payloads
    int max_tokens_per_rank,                // Maximum tokens per rank
    int local_num_tokens, int rank_id, int ep_size, int num_experts_per_rank) {
  static_assert(!COMPACT_EP4 || TOP_K > kEp4Size);
  int thread_idx = threadIdx.x;
  int local_token_idx = blockIdx.x;

  if (local_num_tokens == 0) {
    // Special case: If local_num_tokens == 0,
    // we need to keep the threads where local_token_idx == 0 alive to participate in the
    // synchronization. Other threads should return.
    if (local_token_idx > 0) return;
  } else {
    // Threads that do not have a token to process should return.
    if (local_token_idx >= local_num_tokens) return;

    // Prepare per-policy shared-memory tiles for this token
    extern __shared__ int smem[];
    int* smem_topk_target_ranks = smem;
    int* smem_topk_send_indices = nullptr;
    int* smem_compact_send_indices = smem;
    if constexpr (!COMPACT_EP4) {
      smem_topk_send_indices = smem + TOP_K;
    }

    if (thread_idx < warpSize) {
      int lane_id = thread_idx;
      if constexpr (COMPACT_EP4) {
        if (lane_id < kEp4Size) {
          smem_compact_send_indices[lane_id] = -1;
        }
        __syncwarp();
      }

      unsigned topk_mask = __ballot_sync(0xffffffff, lane_id < TOP_K);
      if (lane_id < TOP_K) {
        int k = lane_id;
        int expert_id = token_selected_experts[local_token_idx * TOP_K + k];
        int target_rank = compute_target_rank_id(expert_id, num_experts_per_rank);

        // Elect the first top-k lane for each destination rank.
        unsigned matching_lanes = __match_any_sync(topk_mask, target_rank);
        bool is_first = lane_id == __ffs(matching_lanes) - 1;
        int dst_token_idx = -1;
        if (is_first) {
          dst_token_idx = atomicAdd(&ptrs.send_counters[target_rank], 1);
          if constexpr (COMPACT_EP4) {
            smem_compact_send_indices[target_rank] = dst_token_idx;
          }
        } else {
          target_rank = -1;
        }

        ptrs.topk_target_ranks[local_token_idx * TOP_K + k] = target_rank;
        ptrs.topk_send_indices[local_token_idx * TOP_K + k] = dst_token_idx;
        if constexpr (!COMPACT_EP4) {
          smem_topk_target_ranks[k] = target_rank;
          smem_topk_send_indices[k] = dst_token_idx;
        }
      }
    }
    // Sync before dispatching data
    __syncthreads();

    // EP4 has at most four unique destinations, regardless of TOP_K. The compact
    // specialization avoids expanding those destinations back to TOP_K entries.
    constexpr int NUM_DESTINATIONS = COMPACT_EP4 ? kEp4Size : TOP_K;
    int target_ranks[NUM_DESTINATIONS];
    int send_indices[NUM_DESTINATIONS];
    if constexpr (COMPACT_EP4) {
#pragma unroll
      for (int target_rank = 0; target_rank < NUM_DESTINATIONS; ++target_rank) {
        target_ranks[target_rank] = target_rank;
        send_indices[target_rank] = smem_compact_send_indices[target_rank];
      }
    } else {
#pragma unroll
      for (int k = 0; k < TOP_K; ++k) {
        target_ranks[k] = smem_topk_target_ranks[k];
        send_indices[k] = smem_topk_send_indices[k];
      }
    }

    // Perform a single source load and fan out to each unique destination.
    for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++) {
      uint8_t const* src_data = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]);
      int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];
      uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;

      vectorized_dispatch<NUM_DESTINATIONS>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                            payload_idx, ptrs, target_ranks, send_indices);
    }

    __syncthreads();
  }

  bool is_first_warp = threadIdx.x / warpSize == 0;
  if (is_first_warp) {
    int lane_id = threadIdx.x % warpSize;

    bool is_last_token = false;
    if (lane_id == 0) {
      if (local_num_tokens != 0) {
        int cnt = atomicAdd(ptrs.local_token_counter, 1);
        is_last_token = cnt + 1 == local_num_tokens;
      } else {
        is_last_token = true;
      }
    }
    is_last_token = __shfl_sync(0xffffffff, is_last_token, 0);

    if (is_last_token) {
// Store send_counters to recv_counters
#pragma unroll 1  // No unroll as one iter is typically enough
      for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize) {
        int send_count = ptrs.send_counters[target_rank];
        ptrs.recv_counters[target_rank][rank_id] = send_count;
      }

#if !DISABLE_SYNC_FOR_PROFILING
      uint32_t expected_value = *ptrs.flag_val;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      asm volatile("fence.release.sys;");
#else
      __threadfence_system();
#endif
#pragma unroll 1  // No unroll as one iter is typically enough
      for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize) {
        uint32_t* flag_addr = &ptrs.completion_flags[target_rank][rank_id];
        asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));

#if ENABLE_DEBUG_PRINT
        printf("dispatch: +++Rank %d setting completion flag to %d for rank %d\n", rank_id,
               expected_value, target_rank);
#endif
      }

#pragma unroll 1  // No unroll
      for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize) {
        bool flag_set = false;
        [[maybe_unused]] auto s = clock64();
        do {
          uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
          uint32_t flag_value;
          // Acquire load to ensure visibility of peer's release-store
          asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
          printf(
              "dispatch: ---Rank %d received completion flag from rank %d, flag_value: %d, "
              "expected_value: "
              "%d, address: %p\n",
              rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif
          flag_set = flag_value == expected_value;
        } while (!flag_set && !check_timeout(s));

        if (__builtin_expect(!flag_set, 0)) {
          printf("dispatch: ---Rank %d timed out waiting for completion flag from rank %d\n",
                 rank_id, peer_rank);
          asm volatile("trap;");
          return;
        }
      }
      // asm volatile("fence.acquire.sys;");
#endif
    }
  }
}

void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params) {
  moeA2APrepareDispatchKernel<<<1, params.ep_size, 0, params.stream>>>(
      params.send_counters, params.local_token_counter, params.ep_size, params.flag_val);
}

// ============================================================================
// Launch Functions
// ============================================================================

void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params) {
  // Validate parameters
  TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
  TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
  TLLM_CHECK(params.local_num_tokens >= 0);
  TLLM_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads);

  // Prepare kernel pointers struct
  DispatchKernelPointers kernel_ptrs = {};

  // Fill source data pointers and payload sizes
  for (int i = 0; i < params.num_payloads; i++) {
    kernel_ptrs.src_data_ptrs[i] = params.payloads[i].src_data;
    kernel_ptrs.payload_bytes_per_token[i] =
        params.payloads[i].element_size * params.payloads[i].elements_per_token;
  }

  // Fill receive buffer pointers
  for (int target_rank = 0; target_rank < params.ep_size; target_rank++) {
    kernel_ptrs.recv_counters[target_rank] = params.recv_counters[target_rank];
    for (int payload = 0; payload < params.num_payloads; payload++) {
      kernel_ptrs.recv_buffers[target_rank][payload] = params.recv_buffers[target_rank][payload];
    }
  }

  // Copy completion flag pointers
  for (int i = 0; i < params.ep_size; i++) {
    kernel_ptrs.completion_flags[i] = params.completion_flags[i];
  }
  kernel_ptrs.flag_val = params.flag_val;

  // Copy communication tracking pointers
  kernel_ptrs.send_counters = params.send_counters;
  kernel_ptrs.local_token_counter = params.local_token_counter;
  kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
  kernel_ptrs.topk_send_indices = params.topk_send_indices;

  int block_size = tensorrt_llm::common::getEnvMoeA2ADispatchBlockSize();

  bool const use_compact_ep4 = params.ep_size == kEp4Size && params.top_k > kEp4Size;

  int max_payload_bytes_per_token = 0;
  for (int i = 0; i < params.num_payloads; ++i) {
    if (kernel_ptrs.payload_bytes_per_token[i] > max_payload_bytes_per_token) {
      max_payload_bytes_per_token = kernel_ptrs.payload_bytes_per_token[i];
    }
  }
  // Small payloads do not have enough 16-byte vectors to use larger CTAs. A GB200
  // sweep found 128 threads faster than both 64 and 256 at prefill token counts.
  if (use_compact_ep4 && max_payload_bytes_per_token <= kCompactDispatchMaxPayloadBytes &&
      block_size > kCompactDispatchBlockSize) {
    block_size = kCompactDispatchBlockSize;
  }

  // Configure kernel launch: one block per token
  int grid_size = params.local_num_tokens;
  // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the
  // synchronization.
  if (grid_size == 0) {
    grid_size = 1;
  }
  if (use_compact_ep4) {
    int shared_bytes = kEp4Size * (int)sizeof(int);
    SWITCH_TOP_K(
        params.top_k, TOP_K, if constexpr (TOP_K > kEp4Size) {
          moeA2ADispatchKernel<TOP_K, true><<<grid_size, block_size, shared_bytes, params.stream>>>(
              params.token_selected_experts, kernel_ptrs, params.num_payloads,
              params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank, params.ep_size,
              params.num_experts_per_rank);
        })
  } else {
    int shared_bytes = 2 * params.top_k * (int)sizeof(int);
    SWITCH_TOP_K(params.top_k, TOP_K,
                 moeA2ADispatchKernel<TOP_K, false>
                 <<<grid_size, block_size, shared_bytes, params.stream>>>(
                     params.token_selected_experts, kernel_ptrs, params.num_payloads,
                     params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank,
                     params.ep_size, params.num_experts_per_rank))
  }
}

// ============================================================================
// Combine kernels
// ============================================================================

template <typename T, int ELEMS_PER_VEC>
__device__ __forceinline__ void accumulate_vec(T* dst, T const* src) {
#pragma unroll
  for (int j = 0; j < ELEMS_PER_VEC; ++j) {
    dst[j] += src[j];
  }
}

// Accumulate across all valid ranks into registers, then store once per segment
template <int VEC_SIZE_BYTES, int TOP_K, typename T,
          MoeA2ACombineQuantMode QuantMode = MoeA2ACombineQuantMode::NONE,
          MoeA2ACombineSwizzleSFMode SwizzleMode = MoeA2ACombineSwizzleSFMode::LINEAR>
__device__ void vectorized_combine_impl(void* output_buffer, void* sf_output, int row_idx,
                                        int row_size, int rank_id, int max_tokens_per_rank,
                                        CombineKernelPointers const& ptrs,
                                        float OutputScalarScale = 1.0f) {
  constexpr int elems_per_vec = VEC_SIZE_BYTES / sizeof(T);
  const int size_per_token = row_size * sizeof(T);
  using flashinfer::vec_t;

  uint8_t* dst_bytes;
  if constexpr (QuantMode == MoeA2ACombineQuantMode::NONE) {
    dst_bytes = reinterpret_cast<uint8_t*>(static_cast<T*>(output_buffer) + row_idx * row_size);
  } else {
    // MXFP8 stores one byte per logical element; FP4 packs two e2m1 values per byte, so its
    // rows are half as wide. The accumulation still reads T-sized inputs either way.
    size_t const bytes_per_row = QuantMode == MoeA2ACombineQuantMode::MXFP8
                                     ? static_cast<size_t>(row_size)
                                     : static_cast<size_t>(row_size) / 2;
    dst_bytes = static_cast<uint8_t*>(output_buffer) + static_cast<size_t>(row_idx) * bytes_per_row;
  }

  int const stride = blockDim.x * VEC_SIZE_BYTES;
  int const local_token_idx = blockIdx.x;

  for (int offset = threadIdx.x * VEC_SIZE_BYTES; offset < size_per_token; offset += stride) {
    int logical_offset = offset / sizeof(T);
    vec_t<uint8_t, VEC_SIZE_BYTES> acc[TOP_K];

// Unrolled K accumulation using compact top-k lists
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int target_rank = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
      int dst_idx = ptrs.topk_send_indices[local_token_idx * TOP_K + k];
      if (dst_idx < 0) {
        acc[k].fill(0);
        continue;
      }

      uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
      size_t base_source_rank =
          static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank) +
          static_cast<size_t>(dst_idx);
      size_t base_token = base_source_rank * static_cast<size_t>(size_per_token);

      // Load directly into the per-k accumulator; reduce across k below
      acc[k].load(recv_buffer + base_token + offset);
    }

    // Reduce acc[TOP_K] into acc[0]
    if constexpr (TOP_K == 22) {
      T* a0 = reinterpret_cast<T*>(&acc[0]);
      T* a1 = reinterpret_cast<T*>(&acc[1]);
      T* a2 = reinterpret_cast<T*>(&acc[2]);
      T* a3 = reinterpret_cast<T*>(&acc[3]);
      T* a4 = reinterpret_cast<T*>(&acc[4]);
      T* a5 = reinterpret_cast<T*>(&acc[5]);
      T* a6 = reinterpret_cast<T*>(&acc[6]);
      T* a7 = reinterpret_cast<T*>(&acc[7]);
      T* a8 = reinterpret_cast<T*>(&acc[8]);
      T* a9 = reinterpret_cast<T*>(&acc[9]);
      T* a10 = reinterpret_cast<T*>(&acc[10]);
      T* a11 = reinterpret_cast<T*>(&acc[11]);
      T* a12 = reinterpret_cast<T*>(&acc[12]);
      T* a13 = reinterpret_cast<T*>(&acc[13]);
      T* a14 = reinterpret_cast<T*>(&acc[14]);
      T* a15 = reinterpret_cast<T*>(&acc[15]);
      T* a16 = reinterpret_cast<T*>(&acc[16]);
      T* a17 = reinterpret_cast<T*>(&acc[17]);
      T* a18 = reinterpret_cast<T*>(&acc[18]);
      T* a19 = reinterpret_cast<T*>(&acc[19]);
      T* a20 = reinterpret_cast<T*>(&acc[20]);
      T* a21 = reinterpret_cast<T*>(&acc[21]);
      accumulate_vec<T, elems_per_vec>(a0, a1);
      accumulate_vec<T, elems_per_vec>(a2, a3);
      accumulate_vec<T, elems_per_vec>(a4, a5);
      accumulate_vec<T, elems_per_vec>(a6, a7);
      accumulate_vec<T, elems_per_vec>(a8, a9);
      accumulate_vec<T, elems_per_vec>(a10, a11);
      accumulate_vec<T, elems_per_vec>(a12, a13);
      accumulate_vec<T, elems_per_vec>(a14, a15);
      accumulate_vec<T, elems_per_vec>(a16, a17);
      accumulate_vec<T, elems_per_vec>(a18, a19);
      accumulate_vec<T, elems_per_vec>(a20, a21);

      accumulate_vec<T, elems_per_vec>(a0, a2);
      accumulate_vec<T, elems_per_vec>(a4, a6);
      accumulate_vec<T, elems_per_vec>(a8, a10);
      accumulate_vec<T, elems_per_vec>(a12, a14);
      accumulate_vec<T, elems_per_vec>(a16, a18);

      accumulate_vec<T, elems_per_vec>(a0, a4);
      accumulate_vec<T, elems_per_vec>(a8, a12);
      accumulate_vec<T, elems_per_vec>(a16, a20);

      accumulate_vec<T, elems_per_vec>(a0, a8);
      accumulate_vec<T, elems_per_vec>(a0, a16);
    } else if constexpr (TOP_K == 16) {
      T* a0 = reinterpret_cast<T*>(&acc[0]);
      T* a1 = reinterpret_cast<T*>(&acc[1]);
      T* a2 = reinterpret_cast<T*>(&acc[2]);
      T* a3 = reinterpret_cast<T*>(&acc[3]);
      T* a4 = reinterpret_cast<T*>(&acc[4]);
      T* a5 = reinterpret_cast<T*>(&acc[5]);
      T* a6 = reinterpret_cast<T*>(&acc[6]);
      T* a7 = reinterpret_cast<T*>(&acc[7]);
      T* a8 = reinterpret_cast<T*>(&acc[8]);
      T* a9 = reinterpret_cast<T*>(&acc[9]);
      T* a10 = reinterpret_cast<T*>(&acc[10]);
      T* a11 = reinterpret_cast<T*>(&acc[11]);
      T* a12 = reinterpret_cast<T*>(&acc[12]);
      T* a13 = reinterpret_cast<T*>(&acc[13]);
      T* a14 = reinterpret_cast<T*>(&acc[14]);
      T* a15 = reinterpret_cast<T*>(&acc[15]);
      accumulate_vec<T, elems_per_vec>(a0, a1);
      accumulate_vec<T, elems_per_vec>(a2, a3);
      accumulate_vec<T, elems_per_vec>(a4, a5);
      accumulate_vec<T, elems_per_vec>(a6, a7);
      accumulate_vec<T, elems_per_vec>(a8, a9);
      accumulate_vec<T, elems_per_vec>(a10, a11);
      accumulate_vec<T, elems_per_vec>(a12, a13);
      accumulate_vec<T, elems_per_vec>(a14, a15);

      accumulate_vec<T, elems_per_vec>(a0, a2);
      accumulate_vec<T, elems_per_vec>(a4, a6);
      accumulate_vec<T, elems_per_vec>(a8, a10);
      accumulate_vec<T, elems_per_vec>(a12, a14);

      accumulate_vec<T, elems_per_vec>(a0, a4);
      accumulate_vec<T, elems_per_vec>(a8, a12);

      accumulate_vec<T, elems_per_vec>(a0, a8);
    } else if constexpr (TOP_K == 10) {
      T* a0 = reinterpret_cast<T*>(&acc[0]);
      T* a1 = reinterpret_cast<T*>(&acc[1]);
      T* a2 = reinterpret_cast<T*>(&acc[2]);
      T* a3 = reinterpret_cast<T*>(&acc[3]);
      T* a4 = reinterpret_cast<T*>(&acc[4]);
      T* a5 = reinterpret_cast<T*>(&acc[5]);
      T* a6 = reinterpret_cast<T*>(&acc[6]);
      T* a7 = reinterpret_cast<T*>(&acc[7]);
      T* a8 = reinterpret_cast<T*>(&acc[8]);
      T* a9 = reinterpret_cast<T*>(&acc[9]);
      accumulate_vec<T, elems_per_vec>(a0, a1);
      accumulate_vec<T, elems_per_vec>(a2, a3);
      accumulate_vec<T, elems_per_vec>(a4, a5);
      accumulate_vec<T, elems_per_vec>(a6, a7);
      accumulate_vec<T, elems_per_vec>(a8, a9);

      accumulate_vec<T, elems_per_vec>(a0, a2);
      accumulate_vec<T, elems_per_vec>(a4, a6);

      accumulate_vec<T, elems_per_vec>(a0, a4);
      accumulate_vec<T, elems_per_vec>(a0, a8);
    } else if constexpr (TOP_K == 8) {
      T* a0 = reinterpret_cast<T*>(&acc[0]);
      T* a1 = reinterpret_cast<T*>(&acc[1]);
      T* a2 = reinterpret_cast<T*>(&acc[2]);
      T* a3 = reinterpret_cast<T*>(&acc[3]);
      T* a4 = reinterpret_cast<T*>(&acc[4]);
      T* a5 = reinterpret_cast<T*>(&acc[5]);
      T* a6 = reinterpret_cast<T*>(&acc[6]);
      T* a7 = reinterpret_cast<T*>(&acc[7]);
      accumulate_vec<T, elems_per_vec>(a0, a1);
      accumulate_vec<T, elems_per_vec>(a2, a3);
      accumulate_vec<T, elems_per_vec>(a4, a5);
      accumulate_vec<T, elems_per_vec>(a6, a7);
      accumulate_vec<T, elems_per_vec>(a0, a2);
      accumulate_vec<T, elems_per_vec>(a4, a6);
      accumulate_vec<T, elems_per_vec>(a0, a4);
    } else if constexpr (TOP_K == 6) {
      T* a0 = reinterpret_cast<T*>(&acc[0]);
      T* a1 = reinterpret_cast<T*>(&acc[1]);
      T* a2 = reinterpret_cast<T*>(&acc[2]);
      T* a3 = reinterpret_cast<T*>(&acc[3]);
      T* a4 = reinterpret_cast<T*>(&acc[4]);
      T* a5 = reinterpret_cast<T*>(&acc[5]);
      accumulate_vec<T, elems_per_vec>(a0, a1);
      accumulate_vec<T, elems_per_vec>(a2, a3);
      accumulate_vec<T, elems_per_vec>(a4, a5);
      accumulate_vec<T, elems_per_vec>(a0, a2);
      accumulate_vec<T, elems_per_vec>(a0, a4);
    } else if constexpr (TOP_K == 4) {
      T* a0 = reinterpret_cast<T*>(&acc[0]);
      T* a1 = reinterpret_cast<T*>(&acc[1]);
      T* a2 = reinterpret_cast<T*>(&acc[2]);
      T* a3 = reinterpret_cast<T*>(&acc[3]);
      accumulate_vec<T, elems_per_vec>(a0, a1);
      accumulate_vec<T, elems_per_vec>(a2, a3);
      accumulate_vec<T, elems_per_vec>(a0, a2);
    } else if constexpr (TOP_K == 2) {
      T* a0 = reinterpret_cast<T*>(&acc[0]);
      T* a1 = reinterpret_cast<T*>(&acc[1]);
      accumulate_vec<T, elems_per_vec>(a0, a1);
    } else if constexpr (TOP_K == 1) {
      // nothing to do
    } else {
      // Fallback for any future unspecialized TOP_K instantiations.
      T* a0 = reinterpret_cast<T*>(&acc[0]);
#pragma unroll
      for (int k = 1; k < TOP_K; ++k) {
        T* ak = reinterpret_cast<T*>(&acc[k]);
        accumulate_vec<T, elems_per_vec>(a0, ak);
      }
    }
    if constexpr (QuantMode == MoeA2ACombineQuantMode::NONE) {
      acc[0].store(dst_bytes + offset);
    } else {
      constexpr uint32_t sf_vec_size = QuantMode == MoeA2ACombineQuantMode::NVFP4 ? 16 : 32;
      constexpr uint32_t threads_per_sf = sf_vec_size / elems_per_vec;
      uint8_t scale;
      auto store_sf = [&]() {
        int64_t sf_offset;
        const int64_t num_vecs_per_row = (row_size + sf_vec_size - 1) / sf_vec_size;
        const int64_t sf_idx = logical_offset / sf_vec_size;
        if constexpr (SwizzleMode == MoeA2ACombineSwizzleSFMode::LINEAR) {
          sf_offset = row_idx * num_vecs_per_row + sf_idx;
        } else if constexpr (SwizzleMode == MoeA2ACombineSwizzleSFMode::SWIZZLE_128x4) {
          sf_offset = get_sf_out_offset_128x4(row_idx, sf_idx, num_vecs_per_row);
        } else {  // SWIZZLE_8x4
          sf_offset = get_sf_out_offset_8x4(row_idx, sf_idx, num_vecs_per_row);
        }
        if (threadIdx.x % threads_per_sf == 0) {
          reinterpret_cast<uint8_t*>(sf_output)[sf_offset] = scale;
        }
      };
      auto packed_vec =
          reinterpret_cast<tensorrt_llm::kernels::PackedVec<T, elems_per_vec>&>(acc[0]);
      if constexpr (QuantMode == MoeA2ACombineQuantMode::MXFP8) {
        static_assert(elems_per_vec == 8, "MXFP8 quantization requires 8 elements per vector");
        uint64_t fp8x8 =
            tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<T, 32, elems_per_vec>(packed_vec, &scale);
        reinterpret_cast<uint64_t*>(dst_bytes)[logical_offset / elems_per_vec] = fp8x8;
      } else if constexpr (QuantMode == MoeA2ACombineQuantMode::NVFP4 ||
                           QuantMode == MoeA2ACombineQuantMode::MXFP4) {
        static_assert(elems_per_vec == 8 || elems_per_vec == 16,
                      "FP4 quantization requires 8 or 16 elements per vector");
        constexpr int SF_VEC_SIZE = QuantMode == MoeA2ACombineQuantMode::MXFP4 ? 32 : 16;
        auto fp4_packed = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4 < T, SF_VEC_SIZE,
             elems_per_vec,
             QuantMode == MoeA2ACombineQuantMode::MXFP4 > (packed_vec, OutputScalarScale, &scale);
        // cvt_warp_fp16_to_fp4 returns uint32_t for 8 elems (4 packed bytes) and uint64_t for 16
        // (8 packed bytes); store at the matching width so packed rows stay contiguous and fit the
        // dim/2-byte output buffer (a wider store would overflow into the next token's row).
        reinterpret_cast<decltype(fp4_packed)*>(dst_bytes)[logical_offset / elems_per_vec] =
            fp4_packed;
      }
      store_sf();
    }
  }
}

// Wrapper that selects vector width based on size_per_token alignment
template <int TOP_K, typename T, MoeA2ACombineQuantMode QuantMode = MoeA2ACombineQuantMode::NONE,
          MoeA2ACombineSwizzleSFMode SwizzleMode = MoeA2ACombineSwizzleSFMode::LINEAR>
__device__ void vectorized_combine(void* output_buffer, void* sf_output, int row_idx, int row_size,
                                   int rank_id, int max_tokens_per_rank,
                                   CombineKernelPointers const& ptrs,
                                   float OutputScalarScale = 1.0f) {
  if constexpr (QuantMode != MoeA2ACombineQuantMode::NONE) {
    vectorized_combine_impl<16, TOP_K, T, QuantMode, SwizzleMode>(
        output_buffer, sf_output, row_idx, row_size, rank_id, max_tokens_per_rank, ptrs,
        OutputScalarScale);
  } else {
    if (row_size % 16 == 0) {
      vectorized_combine_impl<16, TOP_K, T>(output_buffer, nullptr, row_idx, row_size, rank_id,
                                            max_tokens_per_rank, ptrs);
    } else if (row_size % 8 == 0) {
      vectorized_combine_impl<8, TOP_K, T>(output_buffer, nullptr, row_idx, row_size, rank_id,
                                           max_tokens_per_rank, ptrs);
    } else if (row_size % 4 == 0) {
      vectorized_combine_impl<4, TOP_K, T>(output_buffer, nullptr, row_idx, row_size, rank_id,
                                           max_tokens_per_rank, ptrs);
    } else if (row_size % 2 == 0) {
      vectorized_combine_impl<2, TOP_K, T>(output_buffer, nullptr, row_idx, row_size, rank_id,
                                           max_tokens_per_rank, ptrs);
    } else {
      vectorized_combine_impl<1, TOP_K, T>(output_buffer, nullptr, row_idx, row_size, rank_id,
                                           max_tokens_per_rank, ptrs);
    }
  }
}

// Copy payload to recv buffer using vectorized copy; one block per token
__global__ void moeA2APrepareCombineKernel(uint8_t* recv_buffer_bytes, uint8_t const* payload_bytes,
                                           int bytes_per_token, int ep_size,
                                           int max_tokens_per_rank, uint32_t* flag_val_ptr,
                                           int const* recv_counters) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // Increment flag_val for this combine round
    *flag_val_ptr = *flag_val_ptr + 1;
  }

  if (payload_bytes == nullptr) return;

  int slot_idx = blockIdx.x;

  int total_slots = ep_size * max_tokens_per_rank;
  if (slot_idx >= total_slots) return;

  // Map global token to (source_rank, token_idx)
  int source_rank = slot_idx / max_tokens_per_rank;
  int token_idx = slot_idx % max_tokens_per_rank;

  // Skip invalid tokens beyond per-source recv count
  if (token_idx >= recv_counters[source_rank]) return;

  // Calculate source and destination pointers for this token
  size_t slot_offset = static_cast<size_t>(slot_idx) * bytes_per_token;
  uint8_t* dst_ptr = recv_buffer_bytes + slot_offset;
  uint8_t const* src_ptr = payload_bytes + slot_offset;

  // Copy one token's data using vectorized copy
  vectorized_copy(dst_ptr, src_ptr, bytes_per_token);
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T, int TOP_K, MoeA2ACombineQuantMode QuantMode = MoeA2ACombineQuantMode::NONE,
          MoeA2ACombineSwizzleSFMode SwizzleMode = MoeA2ACombineSwizzleSFMode::LINEAR>
__global__ void moeA2ACombineKernel(
    const CombineKernelPointers ptrs,  // Combine-specific struct, src_data_ptrs[0] is output
    int max_tokens_per_rank, int elements_per_token, int local_num_tokens, int rank_id, int ep_size,
    float OutputScalarScale) {
  int local_token_idx = blockIdx.x;

  if (local_num_tokens == 0) {
    // Special case: If local_num_tokens == 0,
    // we need to keep the threads where local_token_idx == 0 alive to participate in the
    // synchronization. Other threads should return.
    if (local_token_idx > 0) return;
  } else {
    // Threads that do not have a token to process should return.
    if (local_token_idx >= local_num_tokens) return;
  }

#if !DISABLE_SYNC_FOR_PROFILING
  // In-kernel readiness synchronization at start of combine:
  // - One warp signals readiness to all peers with current flag_val.
  // - The first warp of each block waits for all peers' readiness (equality), then __syncthreads.
  bool is_first_warp = threadIdx.x / warpSize == 0;
  if (is_first_warp) {
    int lane_id = threadIdx.x % warpSize;
    uint32_t expected_value = *ptrs.flag_val;

    if (blockIdx.x == 0) {
      // asm volatile("fence.release.sys;");
#pragma unroll 1  // No unroll
      for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize) {
        uint32_t* flag_addr = &ptrs.completion_flags[peer_rank][rank_id];
        asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));
#if ENABLE_DEBUG_PRINT
        printf("combine: +++Rank %d setting completion flag to %d for rank %d\n", rank_id,
               expected_value, peer_rank);
#endif
      }
    }

#pragma unroll 1  // No unroll
    for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize) {
      bool flag_set = false;
      [[maybe_unused]] auto s = clock64();
      do {
        uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
        uint32_t flag_value;
        // Acquire load to ensure visibility of peer's release-store
        asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
        printf(
            "combine: ---Rank %d received completion flag from rank %d, flag_value: %d, "
            "expected_value: %d, "
            "address: %p\n",
            rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif
        flag_set = flag_value == expected_value;
      } while (!flag_set && !check_timeout(s));

      if (__builtin_expect(!flag_set, 0)) {
        printf("combine: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id,
               peer_rank);
        asm volatile("trap;");
        return;
      }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("fence.acquire.sys;");
#else
    __threadfence_system();
#endif
  }
  __syncthreads();
#endif

  if (local_num_tokens == 0) return;

  // Accumulate across ranks in registers, then store once per segment
  vectorized_combine<TOP_K, T, QuantMode, SwizzleMode>(
      ptrs.src_data_ptrs[0], ptrs.output_scales, local_token_idx, elements_per_token, rank_id,
      max_tokens_per_rank, ptrs, OutputScalarScale);
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params) {
  constexpr int kBlockSize = 256;

  // Calculate bytes per token based on dtype
  int element_size;
  switch (params.dtype) {
    case nvinfer1::DataType::kHALF:
      element_size = sizeof(half);
      break;
    case nvinfer1::DataType::kBF16:
      element_size = sizeof(__nv_bfloat16);
      break;
    case nvinfer1::DataType::kFLOAT:
      element_size = sizeof(float);
      break;
    default:
      FLASHINFER_CHECK(false, "Unsupported dtype for combine prepare");
      return;
  }

  int bytes_per_token = params.elements_per_token * element_size;
  int grid_size_block =
      params.prepare_payload == nullptr ? 1 : params.ep_size * params.max_tokens_per_rank;

  moeA2APrepareCombineKernel<<<grid_size_block, kBlockSize, 0, params.stream>>>(
      static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank])),
      static_cast<uint8_t const*>(params.prepare_payload), bytes_per_token, params.ep_size,
      params.max_tokens_per_rank, params.flag_val, params.recv_counters);
}

// ============================================================================
// Combine Launch Function
// ============================================================================

void moe_a2a_combine_launch(MoeA2ACombineParams const& params) {
  // Validate parameters
  TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
  TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
  TLLM_CHECK(params.local_num_tokens >= 0);
  TLLM_CHECK(params.elements_per_token > 0);

  // Configure kernel launch: one block per token
  int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ACombineBlockSize();
  int grid_size_block = params.local_num_tokens;
  // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the
  // synchronization.
  if (grid_size_block == 0) {
    grid_size_block = 1;
  }

  // Prepare kernel pointers struct for combine
  CombineKernelPointers kernel_ptrs = {};  // Zero-initialize

  // Set output data pointer in src_data_ptrs[0]
  kernel_ptrs.src_data_ptrs[0] = params.output_data;
  kernel_ptrs.output_scales = params.output_scales;

  // Fill recv buffer pointers
  for (int rank = 0; rank < params.ep_size; rank++) {
    kernel_ptrs.recv_buffers[rank][0] = params.recv_buffers[rank];
  }

  // Copy completion flag pointers
  for (int i = 0; i < params.ep_size; i++) {
    kernel_ptrs.completion_flags[i] = params.completion_flags[i];
  }
  kernel_ptrs.flag_val = params.flag_val;

  // Copy communication tracking pointers
  kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
  kernel_ptrs.topk_send_indices = params.topk_send_indices;

  // Launch appropriate kernel with compact macros
  SWITCH_DTYPE(params.dtype, TKernelType, {
    SWITCH_TOP_K(params.top_k, TOP_K, {
      SWITCH_QUANT_MODE(TKernelType, params.quant_mode, QUANT_MODE, {
        SWITCH_SWIZZLE_MODE(params.swizzle_mode, SWIZZLE_MODE, {
          moeA2ACombineKernel<TKernelType, TOP_K, QUANT_MODE, SWIZZLE_MODE>
              <<<grid_size_block, kBlockSize, 0, params.stream>>>(
                  kernel_ptrs, params.max_tokens_per_rank, params.elements_per_token,
                  params.local_num_tokens, params.ep_rank, params.ep_size,
                  params.output_scalar_scale);
        });
      });
    });
  });
}

// Kernel to sanitize expert ids for invalid tokens
__global__ void moeA2ASanitizeExpertIdsKernel(int32_t* expert_ids_ptr,
                                              int32_t const* recv_counters_ptr, int ep_size,
                                              int max_tokens_per_rank, int top_k,
                                              int32_t invalid_id) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_tokens = ep_size * max_tokens_per_rank;
  if (tid >= total_tokens) return;

  int source_rank = tid / max_tokens_per_rank;
  int token_idx = tid % max_tokens_per_rank;

  if (token_idx >= recv_counters_ptr[source_rank]) {
    int32_t* token_expert_ids = expert_ids_ptr + tid * top_k;
    for (int k = 0; k < top_k; ++k) {
      token_expert_ids[k] = invalid_id;
    }
  }
}

void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters,
                                        int32_t invalid_id, int ep_size, int max_tokens_per_rank,
                                        int top_k, cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  int total_tokens = ep_size * max_tokens_per_rank;
  int grid = ceilDiv(total_tokens, kBlockSize);
  moeA2ASanitizeExpertIdsKernel<<<grid, kBlockSize, 0, stream>>>(
      expert_ids, recv_counters, ep_size, max_tokens_per_rank, top_k, invalid_id);
}

}  // namespace tensorrt_llm::kernels::moe_alltoall
