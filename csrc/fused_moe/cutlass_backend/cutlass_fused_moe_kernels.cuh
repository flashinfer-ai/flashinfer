/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/workspace.h"

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
#include "cute/tensor.hpp"
#include "cutlass/arch/memory.h"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif

#include "flashinfer/vec_dtypes.cuh"
#include "moe_kernels.h"
#include "moe_util_kernels.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_small_m_launcher_sm80.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_prebuild.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels::cutlass_kernels {

constexpr int WARP_SIZE = 32;
constexpr int CVT_ELTS_PER_THREAD = 8;

__device__ __forceinline__ float computeSafeFP8QuantScale(float row_amax) {
  if (!(row_amax > 0.0f)) {
    return 1.0f;
  }

  constexpr float FP8_E4M3_MAX = 448.0f;
  constexpr float MIN_AMAX_FOR_FINITE_SCALE = FP8_E4M3_MAX / FLT_MAX;
  // A tiny nonzero amax can make FP8_E4M3_MAX / row_amax overflow to infinity.
  return FP8_E4M3_MAX / fmaxf(row_amax, MIN_AMAX_FOR_FINITE_SCALE);
}

struct FloatMaxOp {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

template <class T>
struct AbsMaxOp {
  using Accum = float;

  __device__ static Accum zero() { return 0.0f; }

  __device__ static Accum abs(T value) { return fabsf(static_cast<float>(value)); }

  __device__ static Accum update(Accum current, T value) { return fmaxf(current, abs(value)); }

  __device__ static float to_float(Accum value) { return value; }
};

template <>
struct AbsMaxOp<half> {
  using Accum = half;

  __device__ static Accum zero() { return __float2half(0.0f); }

  __device__ static Accum abs(half value) { return __habs(value); }

  __device__ static Accum update(Accum current, half value) { return __hmax(current, abs(value)); }

  __device__ static float to_float(Accum value) { return __half2float(value); }
};

template <>
struct AbsMaxOp<__nv_bfloat16> {
  using Accum = __nv_bfloat16;

  __device__ static Accum zero() { return __float2bfloat16(0.0f); }

  __device__ static Accum abs(__nv_bfloat16 value) { return __habs(value); }

  __device__ static Accum update(Accum current, __nv_bfloat16 value) {
    return __hmax(current, abs(value));
  }

  __device__ static float to_float(Accum value) { return __bfloat162float(value); }
};

// Selects the compile-time NVFP4 quantizer variant from the recipe code
// (resolveNVFP4Recipe(); the default code reads FLASHINFER_NVFP4_4OVER6*).
// FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH stays a separate environment knob.
template <typename Fn>
auto dispatchNVFP44Over6Config(int64_t nvfp4_4over6_code, Fn&& fn) {
  auto const recipe = tensorrt_llm::kernels::resolveNVFP4Recipe(nvfp4_4over6_code);
  bool const disableFP4QuantFastMath = tensorrt_llm::common::getEnvDisableFP4QuantFastMath();

  auto dispatchDisableFastMath = [&](auto e4m3MaxTag, auto errModeTag, auto errUseFastMathTag) {
    if (disableFP4QuantFastMath) {
      return fn(std::true_type{},
                NVFP44Over6Config<decltype(e4m3MaxTag)::value, decltype(errModeTag)::value,
                                  decltype(errUseFastMathTag)::value>{});
    }
    return fn(std::false_type{},
              NVFP44Over6Config<decltype(e4m3MaxTag)::value, decltype(errModeTag)::value,
                                decltype(errUseFastMathTag)::value>{});
  };

  if (recipe.use4Over6) {
    NVFP44Over6ErrMode const errMode = recipe.errMode;
    bool const errUseFastMath = recipe.errUseFastMath;
    int const e4m3Max = recipe.e4m3Max;

    auto dispatchErrUseFastMath = [&](auto e4m3MaxTag, auto errModeTag) {
      if (errUseFastMath) {
        return dispatchDisableFastMath(e4m3MaxTag, errModeTag, std::true_type{});
      }
      return dispatchDisableFastMath(e4m3MaxTag, errModeTag, std::false_type{});
    };
    auto dispatchErrMode = [&](auto e4m3MaxTag) {
      switch (errMode) {
        case NVFP44Over6ErrMode::MAE:
          return dispatchErrUseFastMath(
              e4m3MaxTag, std::integral_constant<NVFP44Over6ErrMode, NVFP44Over6ErrMode::MAE>{});
        case NVFP44Over6ErrMode::MSE:
          return dispatchErrUseFastMath(
              e4m3MaxTag, std::integral_constant<NVFP44Over6ErrMode, NVFP44Over6ErrMode::MSE>{});
        default:
          TLLM_CHECK_WITH_INFO(false, "Unsupported NVFP4 4over6 error mode.");
          return dispatchErrUseFastMath(
              e4m3MaxTag, std::integral_constant<NVFP44Over6ErrMode, NVFP44Over6ErrMode::MAE>{});
      }
    };
    if (e4m3Max == 256) {
      return dispatchErrMode(std::integral_constant<int, 256>{});
    }
    TLLM_CHECK_WITH_INFO(e4m3Max == 448, "Unsupported NVFP4 4over6 E4M3 max.");
    return dispatchErrMode(std::integral_constant<int, 448>{});
  }

  if (disableFP4QuantFastMath) {
    return fn(std::true_type{}, std::false_type{});
  }
  return fn(std::false_type{}, std::false_type{});
}

/**
 * Takes the input maps and prepares the expanded maps for min latency
 * @param num_active_experts_per_node: Number of active experts on current node
 * @param experts_to_token_scores: The score of each token for each activated expert. 0 if the
 * expert is not chosen by the token. Only the first num_active_experts_per_ rows are valid
 * @param active_expert_global_ids: The global expert id for each activated expert
 * Only the first num_active_experts_per_ values are valid
 * @param expert_first_token_offset: Store the first token offset for each expert
 */
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void initTensor(T* value, int const tid, int const total_num,
                                           T const init_value) {
  for (int i = tid; i < total_num; i += BLOCK_SIZE) {
    value[i] = init_value;
  }
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void setLocalExperts(int* s_local_experts,
                                                T const* token_selected_experts,
                                                int const total_num_experts, int const tid,
                                                int const start_expert, int const end_expert) {
  for (int i = tid; i < total_num_experts; i += BLOCK_SIZE) {
    int const expert = token_selected_experts[i];

    // If expert is in the current node, subtract start_expert to shift the range to [0,
    // num_experts_per_node)
    bool is_valid_expert = expert >= start_expert && expert < end_expert;
    if (is_valid_expert) {
      int local_expert_id = expert - start_expert;
      if (s_local_experts[local_expert_id] == 0) {
        s_local_experts[local_expert_id] =
            1;  // @TODO: Make sure that we allow duplicated write here
      }
    }
  }
  __syncthreads();
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void prefixSum(T* out, T* in, int const num, int const tid) {
  typedef cub::BlockScan<T, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage tempStorage;

  T threadData = 0;
  if (tid < num) {
    threadData = in[tid];
  }

  BlockScan(tempStorage).InclusiveSum(threadData, threadData);
  __syncthreads();

  if (tid < num) {
    out[tid] = threadData;
  }
  __syncthreads();
}

__device__ __forceinline__ void setActiveNum(int& num_active, int& num_active_offset_start,
                                             int& num_active_offset_end, int const cluster_size,
                                             int const cluster_rank) {
  int num_remainder = num_active % cluster_size;
  int num_active_per_node =
      max(0, num_active - 1) / cluster_size;  // num_active_per_node shouldn't be neg
  if (cluster_rank < num_remainder) {
    num_active = num_active_per_node + 1;
    num_active_offset_start = cluster_rank * num_active;
  } else {
    num_active = num_active_per_node;
    num_active_offset_start = cluster_rank * num_active_per_node + num_remainder;
  }
  num_active_offset_end = num_active_offset_start + num_active;
}

template <int BLOCK_SIZE>
__global__ void buildMinLatencyActiveExpertMapsKernel(
    int* num_active_experts_per_node, float* experts_to_token_scores, int* active_expert_global_ids,
    int64_t* expert_first_token_offset, int const* token_selected_experts,
    float const* token_final_scales, int64_t const num_tokens, int const num_experts_per_token,
    int const start_expert, int const end_expert, int const num_experts_per_node,
    bool const smart_routing, int const cluster_rank, int const cluster_size,
    int const num_experts_smem) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  // Use one block to process the min latency case
  int tid = threadIdx.x;
  // 0. init the global memory experts_to_token_scores [num_experts_per_node, num_token]
  int const total_local_scales = num_experts_per_node * num_tokens;
  initTensor<float, BLOCK_SIZE>(experts_to_token_scores, tid, total_local_scales, 0.0f);
  initTensor<int, BLOCK_SIZE>(active_expert_global_ids, tid, num_experts_per_node, -1);

  __threadfence();  //@Todo: check do I need this fence for previous zero setting

  // 1. mask for the active expert: 1 stands for active
  extern __shared__ int s_local_experts[];
  int* s_store_experts = s_local_experts + num_experts_smem;
  initTensor<int, BLOCK_SIZE>(s_local_experts, tid, num_experts_smem, 0);
  __syncthreads();

  // 2. set the shared array s_local_experts[]
  int const total_num_experts = num_tokens * num_experts_per_token;
  setLocalExperts<int, BLOCK_SIZE>(s_local_experts, token_selected_experts, total_num_experts, tid,
                                   start_expert, end_expert);

  // 3. perform prefix sum to acquire the store position and total active experts
  //@TODO: Use cub first, might need to change it to self-defined api
  prefixSum<int, BLOCK_SIZE>(s_store_experts, s_local_experts, num_experts_smem, tid);

  // 4. store the num of active experts
  int num_active = s_store_experts[num_experts_smem - 1];
  int num_active_offset_start = 0;
  int num_active_offset_end = 0;

  if (smart_routing) {
    setActiveNum(num_active, num_active_offset_start, num_active_offset_end, cluster_size,
                 cluster_rank);
  }

  if (tid == 0) {
    *num_active_experts_per_node = num_active;
  }

  // 5. store the global expert id for each expert
  if (smart_routing) {
    for (int i = tid; i < num_experts_smem; i += BLOCK_SIZE) {
      if (s_local_experts[i]) {
        int offset = s_store_experts[i] - 1;
        if (offset >= num_active_offset_start && offset < num_active_offset_end) {
          active_expert_global_ids[offset - num_active_offset_start] = i;
        } else {
          s_local_experts[i] = 0;
        }
      }
    }
    __syncthreads();  // Need sync to update the s_local_experts
  } else {
    for (int i = tid; i < num_experts_smem; i += BLOCK_SIZE) {
      if (s_local_experts[i]) {
        int offset = s_store_experts[i] - 1;
        active_expert_global_ids[offset] = i + start_expert;
      }
    }
  }

  // 6. store the scale values
  __threadfence();  //@Todo: check do I need this fence for previous zero setting
  for (int i = tid; i < total_num_experts; i += BLOCK_SIZE) {
    int const expert = token_selected_experts[i];

    // If expert is not in the current node, set it to num_experts_per_node
    // If expert is in the current node, subtract start_expert to shift the range to [0,
    // num_experts_per_node)
    bool is_valid_expert =
        smart_routing ? s_local_experts[expert] : (expert >= start_expert && expert < end_expert);

    if (is_valid_expert) {
      int token = i / num_experts_per_token;
      float const scale = token_final_scales[i];
      int offset = s_store_experts[expert - start_expert] - 1 - num_active_offset_start;
      experts_to_token_scores[offset * num_tokens + token] = scale;
    }
  }
  // 7. set default value for redundant memory
  for (int i_exp = num_active + tid; i_exp < num_experts_per_node; i_exp += BLOCK_SIZE) {
    active_expert_global_ids[i_exp] = -1;
  }
  // 8. set expert_first_token_offset
  for (int i_exp = tid; i_exp < num_experts_per_node + 1; i_exp += BLOCK_SIZE) {
    if (i_exp < num_active) {
      expert_first_token_offset[i_exp] = i_exp * num_tokens;
    } else {
      expert_first_token_offset[i_exp] = num_active * num_tokens;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void buildMinLatencyActiveExpertMaps(
    int* num_active_experts_per_node, float* experts_to_token_scores, int* active_expert_global_ids,
    int64_t* expert_first_token_offset, int const* token_selected_experts,
    float const* token_final_scales, int64_t const num_tokens, int const experts_per_token,
    int const start_expert, int const end_expert, int const num_experts_per_node,
    int const cluster_rank, int const cluster_size, int const num_experts_smem, bool enable_pdl,
    cudaStream_t const stream) {
  TLLM_CHECK_WITH_INFO(num_experts_per_node == (end_expert - start_expert),
                       "num_experts_per_node must be equal to end_expert - start_expert");

  TLLM_CHECK_WITH_INFO(num_experts_per_node <= 256,
                       "don't support num_experts_per_node > 256 cases");

  int const threads = 256;
  int const blocks = 1;
  bool const smart_routing = cluster_size > 1;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = num_experts_smem * sizeof(int) * 2;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, buildMinLatencyActiveExpertMapsKernel<threads>,
                     num_active_experts_per_node, experts_to_token_scores, active_expert_global_ids,
                     expert_first_token_offset, token_selected_experts, token_final_scales,
                     num_tokens, experts_per_token, start_expert, end_expert, num_experts_per_node,
                     smart_routing, cluster_rank, cluster_size, num_experts_smem);
}

#if defined(FLASHINFER_FP4_SW_CVT)
constexpr static int SMALL_PROLOGUE_THREADS = 256;
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const*, int64_t, T);

// One CTA, a thread per expanded row: its permuted row counts the (expert, row) keys below its own.
__global__ void smallBuildExpertMapsSortFirstTokenKernel(
    int const* token_selected_experts, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset, int const num_tokens, int const experts_per_token,
    int const num_experts_per_node, int const start_expert) {
  // (expert, row) keys, num_experts_per_node for another node's expert and past the rows; sorted.
  __shared__ int4 keys[SMALL_PROLOGUE_THREADS / 4];
  __shared__ int sorted[SMALL_PROLOGUE_THREADS];
  int const num_rows = num_tokens * experts_per_token;
  int const row = threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int expert = num_experts_per_node;
  if (row < num_rows) {
    int const local = token_selected_experts[row] - start_expert;
    expert = local >= 0 && local < num_experts_per_node ? local : num_experts_per_node;
  }
  int const key = expert * SMALL_PROLOGUE_THREADS + row;
  reinterpret_cast<int*>(keys)[row] = key;
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  int before[4] = {};  // four rows at a time, counted apart
  for (int i = 0; i < num_rows; i += 4) {
    int4 const k = keys[i / 4];
    before[0] += k.x < key;
    before[1] += k.y < key;
    before[2] += k.z < key;
    before[3] += k.w < key;
  }
  if (expert < num_experts_per_node) {
    int const permuted_row = before[0] + before[1] + before[2] + before[3];
    int const unpermuted_row = row % experts_per_token * num_tokens + row / experts_per_token;
    permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
    unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    permuted_token_selected_experts[permuted_row] = expert;
    sorted[permuted_row] = expert;
  }
  int const num_local = __syncthreads_count(expert < num_experts_per_node);
  for (int64_t e = threadIdx.x; e <= num_experts_per_node; e += SMALL_PROLOGUE_THREADS) {
    expert_first_token_offset[e] = findTotalEltsLessThanTarget<int>(sorted, num_local, e);
  }
}
#endif

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel(
    int const* const token_selected_experts, int* const permuted_row_to_unpermuted_row,
    int* const unpermuted_row_to_permuted_row, int64_t* const expert_first_token_offset,
    int64_t const num_tokens, int const experts_per_token, int const start_expert,
    int const end_expert, int const num_experts_per_node) {
  // Only using block wise collective so we can only have one block
  assert(gridDim.x == 1);

  assert(start_expert <= end_expert);
  assert(num_experts_per_node == (end_expert - start_expert));
  assert(num_experts_per_node <= (1 << LOG2_NUM_EXPERTS));

  int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  bool is_valid_token = token < num_tokens;

  // This is the masked expert id for this token
  int local_token_selected_experts[EXPERTS_PER_TOKEN];
  // This is the final permuted rank of this token (ranked by selected expert)
  int local_token_permuted_indices[EXPERTS_PER_TOKEN];

  // Wait PDL before reading token_selected_experts
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

// build expert map
// we need to populate expert ids for all threads, even if there are
// fewer tokens
#pragma unroll
  for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
    int const expert = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i]
                                      : num_experts_per_node;

    // If the token is not valid, set the expert id to num_experts_per_node + 1
    // If expert is not in the current node, set it to num_experts_per_node
    // If expert is in the current node, subtract start_expert to shift the range to [0,
    // num_experts_per_node)
    bool is_valid_expert = expert >= start_expert && expert < end_expert;
    local_token_selected_experts[i] = !is_valid_token   ? num_experts_per_node + 1
                                      : is_valid_expert ? (expert - start_expert)
                                                        : num_experts_per_node;
  }

  // TODO: decompose cub's sort to expose the bucket starts, and just return
  // that to elide the binary search

  // sort the expert map
  using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
  extern __shared__ unsigned char temp_storage[];
  auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

  // Sanity check that the number of bins do correspond to the number of experts
  static_assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= (1 << LOG2_NUM_EXPERTS));
  assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= num_experts_per_node);

  int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];

  cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);
  BlockRadixRank(sort_temp).RankKeys(local_token_selected_experts, local_token_permuted_indices,
                                     extractor, local_expert_first_token_offset);

// We are done with compute, launch the dependent kernels while the stores are in flight
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // write to shared memory and global memory
  if (is_valid_token) {
#pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
      int const unpermuted_row = i * num_tokens + token;
      int const permuted_row = local_token_permuted_indices[i];
      permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
      unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }
  }

#pragma unroll
  for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++) {
    int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
    if (out_expert_id < num_experts_per_node + 1) {
      expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
    }
  }
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenDispatch(
    int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token,
    int const start_expert, int const end_expert, bool enable_pdl, cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(num_experts_per_node == (end_expert - start_expert),
                       "num_experts_per_node must be equal to end_expert - start_expert");
  int const threads = BLOCK_SIZE;
  int const blocks = (num_tokens + threads - 1) / threads;
  TLLM_CHECK_WITH_INFO(blocks == 1, "Current implementation requires single block");

  using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
  size_t shared_size = sizeof(typename BlockRadixRank::TempStorage);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = shared_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  auto kernel =
      &fusedBuildExpertMapsSortFirstTokenKernel<BLOCK_SIZE, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;

  int device = 0;
  int max_smem_per_block = 0;
  check_cuda_error(cudaGetDevice(&device));
  check_cuda_error(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (shared_size >= static_cast<size_t>(max_smem_per_block)) {
    // This should mean that
    // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
    // cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) wouldn't work.
    return false;
  }

  check_cuda_error(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
  check_cuda_error(
      cudaLaunchKernelEx(&config, kernel, token_selected_experts, permuted_row_to_unpermuted_row,
                         unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens,
                         experts_per_token, start_expert, end_expert, num_experts_per_node));

  return true;
}

template <int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(
    int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token,
    int const start_expert, int const end_expert, bool enable_pdl, cudaStream_t stream) {
  int const block_size = num_tokens;
  if (num_tokens > 256) {
    TLLM_LOG_TRACE(
        "Number of tokens %d is greater than 256, which is not supported for fused moe prologues",
        num_tokens);
    return false;
  }

  auto func = &fusedBuildExpertMapsSortFirstTokenDispatch<32, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  if (block_size > 32 && block_size <= 64) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<64, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  } else if (block_size > 64 && block_size <= 128) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<128, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  } else if (block_size > 128 && block_size <= 256) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<256, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  }

  return func(token_selected_experts, permuted_row_to_unpermuted_row,
              unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens,
              num_experts_per_node, experts_per_token, start_expert, end_expert, enable_pdl,
              stream);
}

template <int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(
    int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token,
    int const start_expert, int const end_expert, bool enable_pdl, cudaStream_t stream) {
  auto func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
  switch (experts_per_token) {
    case 1: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
      break;
    }
    case 2: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<2, LOG2_NUM_EXPERTS>;
      break;
    }
    case 4: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<4, LOG2_NUM_EXPERTS>;
      break;
    }
    case 6: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<6, LOG2_NUM_EXPERTS>;
      break;
    }
    case 8: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<8, LOG2_NUM_EXPERTS>;
      break;
    }
    default: {
      TLLM_LOG_TRACE("Top-K value %d does not have supported fused moe prologues",
                     experts_per_token);
      return false;
    }
  }
  return func(token_selected_experts, permuted_row_to_unpermuted_row,
              unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens,
              num_experts_per_node, experts_per_token, start_expert, end_expert, enable_pdl,
              stream);
}

bool fusedBuildExpertMapsSortFirstToken(
    int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token,
    int const start_expert, int const end_expert, bool enable_pdl, cudaStream_t stream) {
  // We need enough bits to represent [0, num_experts_per_node+1] (inclusive) i.e.
  // num_experts_per_node + 2 values This is floor(log2(num_experts_per_node+1)) + 1
  int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;
  if (expert_log <= 9) {
    auto funcs = std::array{&fusedBuildExpertMapsSortFirstTokenBlockSize<1>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<2>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<3>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<4>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<5>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<6>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<7>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<8>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<9>};

    return funcs[expert_log - 1](token_selected_experts, permuted_row_to_unpermuted_row,
                                 unpermuted_row_to_permuted_row, expert_first_token_offset,
                                 num_tokens, num_experts_per_node, experts_per_token, start_expert,
                                 end_expert, enable_pdl, stream);
  }
  TLLM_LOG_TRACE("Experts per node %d does not have supported fused moe prologues",
                 num_experts_per_node);
  return false;
}

int64_t computeNumTokensPerBlock(int64_t const num_tokens, int64_t const num_experts_per_node) {
  for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2) {
    int64_t const num_blocks_per_seq =
        tensorrt_llm::common::ceilDiv(num_tokens, num_tokens_per_block);
    if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
      return num_tokens_per_block;
    }
  }
  return 1024;
}

template <int kNumTokensPerBlock>
__global__ void blockExpertPrefixSumKernel(int const* token_selected_experts,
                                           int* blocked_expert_counts,
                                           int* blocked_row_to_unpermuted_row,
                                           int64_t const num_tokens,
                                           int64_t const num_experts_per_token,
                                           int const start_expert_id) {
  using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // target_expert_id and expert_id are offset by start_expert_id
  int const target_expert_id = blockIdx.x;
  int const block_id = blockIdx.y;
  int const num_blocks_per_seq = gridDim.y;
  int const token_id = block_id * kNumTokensPerBlock + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int expanded_token_id = -1;
  if (token_id < num_tokens) {
    for (int i = 0; i < num_experts_per_token; i++) {
      // TODO(enweiz): Fix uncoalesced access with shared memory.
      int const expert_id =
          token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
      if (expert_id == target_expert_id) {
        expanded_token_id = i * num_tokens + token_id;
        break;
      }
    }
  }

  int const has_matched = expanded_token_id >= 0 ? 1 : 0;
  int index;
  BlockScan(temp_storage).ExclusiveSum(has_matched, index);

  if (has_matched) {
    blocked_row_to_unpermuted_row[target_expert_id * num_tokens + block_id * kNumTokensPerBlock +
                                  index] = expanded_token_id;
  }
  if (threadIdx.x == kNumTokensPerBlock - 1) {
    blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] = index + has_matched;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void blockExpertPrefixSum(int const* token_selected_experts, int* blocked_expert_counts,
                          int* blocked_row_to_unpermuted_row, int64_t const num_tokens,
                          int64_t const num_experts_per_node, int64_t const num_experts_per_token,
                          int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq,
                          int const start_expert_id, bool enable_pdl, cudaStream_t stream) {
  dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
  dim3 const threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  auto func = blockExpertPrefixSumKernel<1024>;
  if (num_tokens_per_block <= 32) {
    func = blockExpertPrefixSumKernel<32>;
  } else if (num_tokens_per_block <= 64) {
    func = blockExpertPrefixSumKernel<64>;
  } else if (num_tokens_per_block <= 128) {
    func = blockExpertPrefixSumKernel<128>;
  } else if (num_tokens_per_block <= 256) {
    func = blockExpertPrefixSumKernel<256>;
  } else if (num_tokens_per_block <= 512) {
    func = blockExpertPrefixSumKernel<512>;
  }
  cudaLaunchKernelEx(&config, func, token_selected_experts, blocked_expert_counts,
                     blocked_row_to_unpermuted_row, num_tokens, num_experts_per_token,
                     start_expert_id);
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumLargeKernel(int const* blocked_expert_counts,
                                                 int* blocked_expert_counts_cumsum,
                                                 int64_t* expert_first_token_offset,
                                                 int64_t const num_experts_per_node,
                                                 int64_t const num_blocks_per_seq,
                                                 int64_t const num_elem_per_thread) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int offset = threadIdx.x * num_elem_per_thread;
  int cnt = 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Note: Because of limited registers, cannot store thread-level prefix sum or enable #pragma
  // unroll
  for (int i = 0; i < num_elem_per_thread; i++) {
    // TODO(enweiz): Fix uncoalesced access with shared memory.
    if (offset + i < num_experts_per_node * num_blocks_per_seq) {
      cnt += blocked_expert_counts[offset + i];
    }
  }

  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  for (int i = 0; i < num_elem_per_thread; i++) {
    if (offset + i < num_experts_per_node * num_blocks_per_seq) {
      blocked_expert_counts_cumsum[offset + i] = cumsum;
      if ((offset + i) % num_blocks_per_seq == 0) {
        expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
      }
      cumsum += blocked_expert_counts[offset + i];
      if ((offset + i) == num_experts_per_node * num_blocks_per_seq - 1) {
        expert_first_token_offset[num_experts_per_node] = cumsum;
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumKernel(int const* blocked_expert_counts,
                                            int* blocked_expert_counts_cumsum,
                                            int64_t* expert_first_token_offset,
                                            int64_t const num_experts_per_node,
                                            int64_t const num_blocks_per_seq) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int const cnt = threadIdx.x < num_experts_per_node * num_blocks_per_seq
                      ? blocked_expert_counts[threadIdx.x]
                      : 0;
  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  if (threadIdx.x < num_experts_per_node * num_blocks_per_seq) {
    blocked_expert_counts_cumsum[threadIdx.x] = cumsum;
    if (threadIdx.x % num_blocks_per_seq == 0) {
      expert_first_token_offset[threadIdx.x / num_blocks_per_seq] = cumsum;
    }
    if (threadIdx.x == num_experts_per_node * num_blocks_per_seq - 1) {
      expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void globalExpertPrefixSum(int const* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                           int64_t* expert_first_token_offset, int64_t const num_experts_per_node,
                           int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq,
                           bool enable_pdl, cudaStream_t stream) {
  int64_t const num_elements = num_experts_per_node * num_blocks_per_seq;

  cudaLaunchConfig_t config;
  config.gridDim = 1;
  config.blockDim = 1024;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  if (num_elements <= 1024) {
    auto func = globalExpertPrefixSumKernel<1024>;
    if (num_elements <= 32) {
      func = globalExpertPrefixSumKernel<32>;
      config.blockDim = 32;
    } else if (num_elements <= 64) {
      func = globalExpertPrefixSumKernel<64>;
      config.blockDim = 64;
    } else if (num_elements <= 128) {
      func = globalExpertPrefixSumKernel<128>;
      config.blockDim = 128;
    } else if (num_elements <= 256) {
      func = globalExpertPrefixSumKernel<256>;
      config.blockDim = 256;
    } else if (num_elements <= 512) {
      func = globalExpertPrefixSumKernel<512>;
      config.blockDim = 512;
    }
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq);
  } else {
    auto func = globalExpertPrefixSumLargeKernel<1024>;
    int64_t const num_elem_per_thread = tensorrt_llm::common::ceilDiv(num_elements, 1024);
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq,
                       num_elem_per_thread);
  }
}

__global__ void mergeExpertPrefixSumKernel(int const* blocked_expert_counts,
                                           int const* blocked_expert_counts_cumsum,
                                           int const* blocked_row_to_unpermuted_row,
                                           int* permuted_token_selected_experts,
                                           int* permuted_row_to_unpermuted_row,
                                           int* unpermuted_row_to_permuted_row,
                                           int const num_tokens) {
  int const target_expert_id = blockIdx.x;
  int const block_id = blockIdx.y;
  int const num_blocks_per_seq = gridDim.y;
  int const token_id = block_id * blockDim.x + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int const cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
  int const offset = blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
  if (threadIdx.x < cnt) {
    int const unpermuted_row =
        blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
    int const permuted_row = offset + threadIdx.x;
    permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
    permuted_token_selected_experts[permuted_row] = target_expert_id;
    unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void mergeExpertPrefixSum(int const* blocked_expert_counts, int const* blocked_expert_counts_cumsum,
                          int const* blocked_row_to_unpermuted_row,
                          int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
                          int* unpermuted_row_to_permuted_row, int64_t const num_tokens,
                          int64_t const num_experts_per_node, int64_t const num_tokens_per_block,
                          int64_t const num_blocks_per_seq, bool enable_pdl, cudaStream_t stream) {
  dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
  dim3 const threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config, mergeExpertPrefixSumKernel, blocked_expert_counts,
                     blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row,
                     permuted_token_selected_experts, permuted_row_to_unpermuted_row,
                     unpermuted_row_to_permuted_row, num_tokens);
}

// threeStepBuildExpertMapsSortFirstToken uses three kernels to achieve the sort of
// token_selected_experts

// 1. blockExpertPrefixSumKernel launches [num_experts_per_node, num_blocks_per_seq] CTAs; each CTA
// has num_tokens_per_block threads. blocked_row_to_unpermuted_row points to a 2D buffer of size
// [num_experts_per_node, num_tokens], which can be viewed as [num_experts_per_node,
// num_blocks_per_seq] blocks, and each block has num_tokens_per_block tokens. Note that each CTA
// corresponds to a block in blocked_row_to_unpermuted_row. Within each CTA, the threads leverage
// cub::BlockScan to compute the offsets of tokens that activate the target expert. If a thread's
// token activates the target expert, the thread stores its unpermuted_row to the buffer block with
// the offset. In addition, the kernel also stores the expert counts for each block to another 2D
// buffer blocked_expert_counts of size [num_experts_per_node, num_blocks_per_seq].

// 2. globalExpertPrefixSumKernel launches 1 CTA; that CTA has num_experts_per_node *
// num_blocks_per_seq threads. The kernel views blocked_expert_counts as a 1D buffer, and leverages
// cub::BlockScan to compute the prefix sum of the expert counts for each block. The prefix sum is
// stored to blocked_expert_counts_cumsum.

// 3. mergeExpertPrefixSumKernel launches [num_experts_per_node, num_blocks_per_seq] CTAs; each CTA
// has num_tokens_per_block threads. Each CTA obtains the block-level offset from
// blocked_expert_counts_cumsum, and thus compacts blocked_row_to_unpermuted_row to
// permuted_row_to_unpermuted_row. In addition, with the block-level offsets, the kernel fills
// permuted_token_selected_experts.

// computeNumTokensPerBlock decides num_tokens_per_block. Note that both blockExpertPrefixSumKernel
// and globalExpertPrefixSumKernel leverage cub::BlockScan, and their CTA sizes are
// num_tokens_per_block and num_experts_per_node * num_blocks_per_seq, respectively.
// computeNumTokensPerBlock tries to find a minimum CTA size for both kernels, so that the
// block-leval cub::BlockScan can be efficient.

void threeStepBuildExpertMapsSortFirstToken(
    int const* token_selected_experts, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset, int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row, int64_t const num_tokens,
    int64_t const num_experts_per_node, int64_t const num_experts_per_token,
    int const start_expert_id, bool enable_pdl, cudaStream_t stream) {
#if defined(FLASHINFER_FP4_SW_CVT)
  if (num_tokens * num_experts_per_token <= SMALL_PROLOGUE_THREADS) {
    launchWithPdlWhenEnabled("smallBuildExpertMapsSortFirstTokenKernel", enable_pdl,
                             smallBuildExpertMapsSortFirstTokenKernel, 1, SMALL_PROLOGUE_THREADS, 0,
                             stream, token_selected_experts, permuted_token_selected_experts,
                             permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
                             expert_first_token_offset, static_cast<int>(num_tokens),
                             static_cast<int>(num_experts_per_token),
                             static_cast<int>(num_experts_per_node), start_expert_id);
    return;
  }
#endif
  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
  int64_t const num_blocks_per_seq =
      tensorrt_llm::common::ceilDiv(num_tokens, num_tokens_per_block);
  int64_t block_threads = num_tokens_per_block;
#if defined(FLASHINFER_FP4_SW_CVT)
  // The global scan's size sets num_tokens_per_block; when the tokens fit in one block, the
  // per-expert blocks only need a power of two covering them.
  if (num_blocks_per_seq == 1) {
    while (block_threads > 32 && block_threads / 2 >= num_tokens) block_threads /= 2;
  }
#endif

  blockExpertPrefixSum(token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
                       num_tokens, num_experts_per_node, num_experts_per_token, block_threads,
                       num_blocks_per_seq, start_expert_id, enable_pdl, stream);
  sync_check_cuda_error(stream);

  globalExpertPrefixSum(blocked_expert_counts, blocked_expert_counts_cumsum,
                        expert_first_token_offset, num_experts_per_node, num_tokens_per_block,
                        num_blocks_per_seq, enable_pdl, stream);
  sync_check_cuda_error(stream);

  mergeExpertPrefixSum(blocked_expert_counts, blocked_expert_counts_cumsum,
                       blocked_row_to_unpermuted_row, permuted_token_selected_experts,
                       permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, num_tokens,
                       num_experts_per_node, block_threads, num_blocks_per_seq, enable_pdl, stream);
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices,
                                                      int64_t const arr_length, T const target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] >= target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

template <class T>
using sizeof_bits = cutlass::sizeof_bits<
    typename cutlass_kernels::TllmToCutlassTypeAdapter<std::remove_cv_t<T>>::type>;

// Function to safely offset an pointer that may contain sub-byte types (FP4/INT4)
template <class T>
__host__ __device__ constexpr T* safe_inc_ptr(T* ptr, size_t offset) {
  constexpr int adjustment = (sizeof_bits<T>::value < 8) ? (8 / sizeof_bits<T>::value) : 1;
  assert(offset % adjustment == 0 && "Attempt to offset index to sub-byte");
  return ptr + offset / adjustment;
}

__host__ __device__ constexpr int64_t getOffsetWeightSF(
    int64_t expert_id, int64_t gemm_n, int64_t gemm_k,
    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  auto function = [=](int64_t min_n_dim_alignment, int64_t min_k_dim_alignment,
                      int64_t block_size) {
    int64_t padded_gemm_n =
        TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_n, min_n_dim_alignment);
    int64_t padded_gemm_k =
        TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k_dim_alignment);
    assert(gemm_k % block_size == 0);
    return expert_id * padded_gemm_n * padded_gemm_k / block_size;
  };
  switch (scaling_type) {
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
      return 0;  // No scaling factors, no offset
  }

  assert(false && "Unrecognized scaling type");
  return 0;
}

__host__ __device__ constexpr int64_t getOffsetActivationSF(
    int64_t expert_id, int64_t token_offset, int64_t gemm_k,
    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  auto function = [=](int64_t min_n_dim_alignment, int64_t min_k_dim_alignment,
                      int64_t block_size) {
    // This formulation ensures that:
    // `sf_offset[i + 1] - sf_offset[i] >= padded(token_offset[i + 1] - token_offset[i])`
    // is true for all possible token distributions.
    int64_t padded_sf_start_offset = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
        token_offset + expert_id * (min_n_dim_alignment - 1), min_n_dim_alignment);
    int64_t padded_gemm_k =
        TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k_dim_alignment);
    assert(gemm_k % block_size == 0);
    assert(padded_gemm_k % block_size == 0);
    return padded_sf_start_offset * padded_gemm_k / block_size;
  };
  switch (scaling_type) {
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
      return 0;  // No scaling factors, no offset
  }

  assert(false && "Unrecognized scaling type");
  return 0;
}

// NVFP4 K padded to the scale-factor alignment, which is also the K step of the SM80 GEMMs.
__host__ __device__ constexpr int64_t padNvfp4K(int64_t k) {
  return TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
      k, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4);
}

// Below SM100 the token rows GEMM1 reads for BF16 W4A16 on the small-M kernel; none from SM100.
inline int const* w4a16TokenRows(QuantParams const& quant_params) {
#if defined(FLASHINFER_FP4_SW_CVT)
  return quant_params.w4a16_token_rows;
#else
  return nullptr;
#endif
}

// 2^x in FP32, -149 <= x <= 127.
__device__ inline float exp2Int(int x) {
  return __int_as_float(x >= -126 ? (x + 127) << 23 : 1 << (x + 149));
}

// x (|x| <= 448) rounded to E4M3, to nearest even.
__device__ inline float roundToE4m3(float x) {
  float const a = fabsf(x);
  uint32_t const bits = __float_as_uint(a);
  float const rounded = a < 0x1p-6f
                            ? __int2float_rn(__float2int_rn(a * 512.f)) * 0x1p-9f
                            : __uint_as_float((bits + 0x7ffffu + (bits >> 20 & 1)) & 0xfff00000u);
  return copysignf(rounded, x);
}

// MXFP8 as eight E4M3 codes times 2^(sf - 127 + shift) in BF16, each rounded once (the exponent
// -135 to 127), NaN with the E8M0 NaN (255) or a NaN code.
__device__ inline void restoreMxfp8x8Scaled(uint64_t codes, uint8_t sf, int shift, uint32_t* out) {
  float const scale = exp2Int(min(int(sf), 254) - 127 + shift);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t const pair = __byte_perm(uint32_t(codes >> 32 * (i / 2)), 0, i % 2 ? 0x4342 : 0x4140);
    // A code's sign and magnitude in FP16's bits are its value times 2^-8.
    uint32_t const bits = (pair & 0x007f007fu) << 7 | (pair & 0x00800080u) << 8;
    float2 const value =
        __half22float2(__hmul2(reinterpret_cast<__half2 const&>(bits), __float2half2_rn(256.f)));
    __nv_bfloat162 const rounded =
        __floats2bfloat162_rn(mx_quantize_mul(value.x, scale), mx_quantize_mul(value.y, scale));
    uint32_t const nan =
        sf == 255 ? ~0u : ((((pair & 0x007f007fu) + 0x00010001u) & 0x00800080u) >> 7) * 0xffffu;
    out[i] = (reinterpret_cast<uint32_t const&>(rounded) & ~nan) | (0x7fc07fc0u & nan);
  }
}

template <class GemmOutputType, class QuantizedType, class ComputeElem, int VecSize,
          bool DISABLE_FP4_QUANT_FAST_MATH = false, typename NVFP4_4OVER6_CONFIG = std::false_type,
          bool RestoreMXFP8 = false>
__device__ auto quantizePackedFPXValue(
    ComputeElem& post_act_val, float global_scale_val, int64_t num_tokens_before_expert,
    int64_t expert_id, int64_t token_id, int64_t elem_idx, int64_t num_cols,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* sf_value = nullptr) {
  constexpr bool is_fp8 = std::is_same_v<QuantizedType, __nv_fp8_e4m3>;
  static constexpr int NumThreadsPerSF = VecSize / CVT_ELTS_PER_THREAD;
  // Quantize the input to FP4
  static_assert(std::is_same_v<GemmOutputType, __nv_bfloat16> ||
                std::is_same_v<GemmOutputType, half>);
  static_assert(ComputeElem::kElements == CVT_ELTS_PER_THREAD);
  PackedVec<GemmOutputType> packed_vec{};
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    packed_vec.elts[i].x = static_cast<GemmOutputType>(post_act_val[i * 2 + 0]);
    packed_vec.elts[i].y = static_cast<GemmOutputType>(post_act_val[i * 2 + 1]);
  }

  // We need to offset into the scaling factors for just this expert
  auto act_sf_expert = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert,
                                                           num_cols, scaling_type);

  // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this
  // expert
  auto sf_out =
      cvt_quant_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF>(
          std::nullopt /* batchIdx */, token_id - num_tokens_before_expert, elem_idx,
          std::nullopt /* numRows */, num_cols / VecSize, act_sf_expert,
          QuantizationSFLayout::SWIZZLED_128x4);

  // Do the conversion and set the output and scaling factor
  auto func = [&]() {
    if constexpr (is_fp8 && RestoreMXFP8) {
      // cvt_warp_fp16_to_mxfp8, then the E4M3 values times 2^(sf - 128) in BF16. Halved, the
      // largest (256 x 2^120) is 2^127, finite, and every nonzero value (at least 2^-126) exact.
      return [](PackedVec<GemmOutputType>& vec, float /* ignored */, uint8_t* SFout,
                float /* ignored */, uint8_t* /* ignored */) -> uint4 {
        static_assert(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize == VecSize);
        float2 values[CVT_ELTS_PER_THREAD / 2];
        uint8_t sf = 255;
        scale_warp_fp16_for_mxfp8<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD>(vec, SFout, &sf,
                                                                                values);
        uint4 out;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        restoreMxfp8x8Scaled(fp32_vec_to_e4m3(values), sf, -1, reinterpret_cast<uint32_t*>(&out));
#else  // software E4M3: the values rounded directly, without the E4M3 bytes
        float const scale = exp2Int(min(int(sf), 254) - 128);
#pragma unroll
        for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; ++i) {
          __nv_bfloat162 const rounded =
              __floats2bfloat162_rn(mx_quantize_mul(roundToE4m3(values[i].x), scale),
                                    mx_quantize_mul(roundToE4m3(values[i].y), scale));
          reinterpret_cast<uint32_t*>(&out)[i] =
              sf == 255 ? 0x7fc07fc0u : reinterpret_cast<uint32_t const&>(rounded);
        }
#endif
        return out;
      };
    } else if constexpr (is_fp8) {
      return [](PackedVec<GemmOutputType>& vec, float /* ignored */, uint8_t* SFout,
                float /* ignored */, uint8_t* SFlocal) -> uint64_t {
        static_assert(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize == VecSize);
        return cvt_warp_fp16_to_mxfp8<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD>(vec, SFout,
                                                                                    SFlocal);
      };
    } else {
      if (scaling_type == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4) {
        return &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD, false,
                                     DISABLE_FP4_QUANT_FAST_MATH, NVFP4_4OVER6_CONFIG>;
      }
      return &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD, true,
                                   DISABLE_FP4_QUANT_FAST_MATH>;
    }
  }();

  return func(packed_vec, global_scale_val, sf_out, 0.0f, sf_value);
}

template <int VecSize, int ElementsPerThread>
__device__ void writeSF(int64_t num_tokens_before_expert, int64_t expert_id,
                        int64_t source_token_id, int64_t token_id, int64_t elem_idx,
                        int64_t num_cols,
                        TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
                        TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf,
                        bool const swizzled_input_sf = true) {
  static constexpr int NumThreadsPerSF = VecSize / ElementsPerThread;

  // We need to offset into the scaling factors for just this expert
  auto act_sf_expert =
      act_sf_flat + getOffsetActivationSF(
                        expert_id, num_tokens_before_expert, num_cols,
                        (VecSize == TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize)
                            ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                            : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);

  // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this
  // expert
  auto sf_out =
      cvt_quant_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF>(
          std::nullopt /* batchIdx */, token_id - num_tokens_before_expert, elem_idx,
          std::nullopt /* numRows */, num_cols / VecSize, act_sf_expert,
          QuantizationSFLayout::SWIZZLED_128x4);
  if (sf_out) {
    if (input_sf) {
      if (swizzled_input_sf) {
        auto const sf_in =
            cvt_quant_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF,
                                        NumThreadsPerSF>(
                std::nullopt /* batchIdx */, source_token_id, elem_idx, std::nullopt /* numRows */,
                num_cols / VecSize,
                const_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf),
                QuantizationSFLayout::SWIZZLED_128x4);
        *sf_out = *sf_in;
      } else {
        auto const sf_in =
            cvt_quant_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF,
                                        NumThreadsPerSF>(
                std::nullopt /* batchIdx */, source_token_id, elem_idx, std::nullopt /* numRows */,
                num_cols / VecSize,
                const_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf),
                QuantizationSFLayout::LINEAR);
        *sf_out = *sf_in;
      }
    } else {
      *sf_out = 0x00;
    }
  }
}

// Eight NVFP4 codes times their E4M3 block scale in FP16/BF16, exact (2 and 4 significant bits).
template <class T>
__device__ inline void restoreNvfp4x8(uint32_t codes, uint8_t sf, uint32_t* out) {
  using T2 = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;
  __nv_fp8_e4m3 scale;
  scale.__x = sf;
  T2 const scale2 = tensorrt_llm::common::cuda_cast<T2>(float(scale));
  flashinfer::e2m1x8_to_16bx8<T>(codes, out);
  auto* pairs = reinterpret_cast<T2*>(out);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pairs[i] = __hmul2(pairs[i], scale2);
  }
}

#if defined(ENABLE_FP4) && defined(FLASHINFER_FP4_SW_CVT)
// A warp per row, restored rows K padded. Restore (NVFP4): codes times their 128x4 block scales,
// permuted rows (per-expert scales) given expert_first_token_offset, else num_rows token rows.
// Pack: permuted rows back to codes, over their block scale (NVFP4) or twice over their E8M0 scale
// (MXFP8), exact.
template <class T, bool Pack, bool IsMX>
__global__ void restoredRowsKernel(T* values, void* codes,
                                   TmaWarpSpecializedGroupedGemmInput::ElementSF const* sf,
                                   int64_t const* expert_first_token_offset,
                                   int64_t const num_experts_per_node, int64_t const num_rows,
                                   int64_t const cols) {
  using Gemm = TmaWarpSpecializedGroupedGemmInput;
  constexpr int64_t VecSize =
      IsMX ? Gemm::MXFPXBlockScaleVectorSize : Gemm::NVFP4BlockScaleVectorSize;
  int64_t const padded_cols =
      Gemm::alignToSfDim(cols, IsMX ? Gemm::MinKDimAlignmentMXFPX : Gemm::MinKDimAlignmentNVFP4);
  int64_t const vecs = cols / VecSize, padded_vecs = padded_cols / VecSize;
  bool const permuted = Pack || expert_first_token_offset;
  int64_t const rows = permuted ? expert_first_token_offset[num_experts_per_node] : num_rows;
  int64_t const warps = int64_t(gridDim.x) * blockDim.x / 32;
  for (int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32; row < rows;
       row += warps) {
    int64_t expert = 0, first = 0;
    if (permuted) {
      expert =
          findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node, row + 1) - 1;
      first = expert_first_token_offset[expert];
    }
    auto* row_sf = const_cast<Gemm::ElementSF*>(sf) +
                   getOffsetActivationSF(
                       expert, first, cols,
                       IsMX ? Gemm::FpXBlockScalingType::MXFPX : Gemm::FpXBlockScalingType::NVFP4);
    for (int64_t v = threadIdx.x % 32; !Pack && v < padded_vecs; v += 32) {
      uint4 out[2] = {};
      if (v < vecs) {
        uint8_t const scale = *cvt_quant_get_sf_out_offset<Gemm::ElementSF, 1>(
            std::nullopt, row - first, v, std::nullopt, padded_vecs, row_sf,
            QuantizationSFLayout::SWIZZLED_128x4);
        uint2 const in = static_cast<uint2 const*>(codes)[row * vecs + v];
        restoreNvfp4x8<T>(in.x, scale, reinterpret_cast<uint32_t*>(&out[0]));
        restoreNvfp4x8<T>(in.y, scale, reinterpret_cast<uint32_t*>(&out[1]));
      }
      reinterpret_cast<uint4*>(values)[2 * (row * padded_vecs + v)] = out[0];
      reinterpret_cast<uint4*>(values)[2 * (row * padded_vecs + v) + 1] = out[1];
    }
    for (int64_t v = threadIdx.x % 32; Pack && v < cols / 8; v += 32) {
      uint8_t const s = *cvt_quant_get_sf_out_offset<Gemm::ElementSF, 1>(
          std::nullopt, row - first, v * 8 / VecSize, std::nullopt, cols / VecSize, row_sf,
          QuantizationSFLayout::SWIZZLED_128x4);
      __nv_fp8_e4m3 scale;
      scale.__x = s;
      float const nvfp4_scale = float(scale) == 0.f ? 1.f : float(scale);  // zero blocks stay 0
      float x[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float const value = static_cast<float>(values[row * padded_cols + v * 8 + i]);
        x[i] = IsMX ? 2.f * mx_quantize_mul(value, exp2Int(127 - min(int(s), 254)))
                    : value / nvfp4_scale;
      }
      if constexpr (IsMX) {
        uint64_t out = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          out |= uint64_t(__nv_fp8_e4m3(x[i]).__x) << (8 * i);
        }
        reinterpret_cast<uint64_t*>(codes)[row * (cols / 8) + v] = out;
      } else {
        reinterpret_cast<uint32_t*>(codes)[row * (cols / 8) + v] = fp32_vec_to_e2m1(x);
      }
    }
  }
}

template <bool Pack, bool IsMX = false, class T>
void restoredRows(T* values, void* codes, TmaWarpSpecializedGroupedGemmInput::ElementSF const* sf,
                  int64_t const* expert_first_token_offset, int64_t const num_experts_per_node,
                  int64_t const num_rows, int64_t const cols, cudaStream_t stream) {
  int64_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
  int64_t const blocks = std::min(smCount * 16, std::max((num_rows + 7) / 8, int64_t{1}));
  restoredRowsKernel<T, Pack, IsMX><<<blocks, 256, 0, stream>>>(
      values, codes, sf, expert_first_token_offset, num_experts_per_node, num_rows, cols);
}

// Token rows restored to 16 bits (K padded) into their local (k, token) permuted rows, a CTA a row
// (its columns split over gridDim.y CTAs when rows are few), with GEMM1's row scale (alpha, times
// the token's per-token NVFP4 scale) and GEMM2's. NVFP4 codes times their scales; W4A16 rows as
// they are; MXFP8 (cols % 128 == 0) times 8 for the SM80 MXFP4 weight scales over 8, GEMM2's alpha
// x 16 (its inputs restored halved), a finite E8M0 >= 244 (to 448 x 2^130, past BF16) flagging the
// row for completeWideMxfp8RowsKernel to recompute in GEMM1.
template <class T, bool IsMX>
__global__ void restoreTokenRowsKernel(T* output, void const* input, uint8_t const* sf,
                                       bool const swizzled_sf, int64_t const num_rows,
                                       int64_t const cols, int const* token_selected_experts,
                                       int const* unpermuted_row_to_permuted_row,
                                       int64_t const experts_per_token, int const start_expert,
                                       int64_t const num_experts_per_node, float const* fc1_alpha,
                                       float const* fc2_alpha, float const* token_scales,
                                       float* fc1_row_scales, float* fc2_row_scales,
                                       int* wide_rows) {
  static_assert(!IsMX || std::is_same_v<T, __nv_bfloat16>);
  int64_t const vecs = cols / 8, padded_vecs = padNvfp4K(cols) / 8;  // of 8 values
  auto block_sf = [&](int64_t row, int64_t block, int64_t blocks) {
    return *cvt_quant_get_sf_out_offset<uint8_t, 1>(
        std::nullopt, row, block, std::nullopt, blocks, const_cast<uint8_t*>(sf),
        swizzled_sf ? QuantizationSFLayout::SWIZZLED_128x4 : QuantizationSFLayout::LINEAR);
  };
  // Words of 8 values v (codes, block scale; zeros in the K padding), read ahead of their stores.
  struct Words {
    uint4 q;
    uint8_t e;
  };
  auto read = [&](int64_t row, int64_t v) {
    Words w{};
    if constexpr (IsMX) {
      w.e = block_sf(row, v / 4, vecs / 4);
      *reinterpret_cast<uint64_t*>(&w.q) = static_cast<uint64_t const*>(input)[row * vecs + v];
    } else if (v < vecs && sf) {
      w.e = block_sf(row, v / 2, swizzled_sf ? padded_vecs / 2 : vecs / 2);
      w.q.x = static_cast<uint32_t const*>(input)[row * vecs + v];
    } else if (v < vecs) {
      w.q = static_cast<uint4 const*>(input)[row * vecs + v];
    }
    return w;
  };
  auto restore = [&](Words const& w, int64_t v, bool& wide) {
    uint4 out = w.q;
    if constexpr (IsMX) {
      wide |= w.e >= 244 && w.e != 255;
      restoreMxfp8x8Scaled(*reinterpret_cast<uint64_t const*>(&w.q), w.e, 3,
                           reinterpret_cast<uint32_t*>(&out));
    } else if (v < vecs && sf) {
      restoreNvfp4x8<T>(w.q.x, w.e, reinterpret_cast<uint32_t*>(&out));
    }
    return out;
  };
  extern __shared__ int permuted_rows[];  // the chunk's, -1 for another node's expert
  int64_t const first_v = blockIdx.y * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.y;
  for (int64_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    // A thread per k, the row restored once per chunk of blockDim.x k.
    for (int64_t k0 = 0; k0 < experts_per_token; k0 += blockDim.x) {
      int64_t const k = k0 + threadIdx.x;
      int64_t const chunk = std::min<int64_t>(experts_per_token - k0, blockDim.x);
      // The row's first words load while its experts, their rows and scales are looked up.
      int const expert =
          k < experts_per_token ? token_selected_experts[row * experts_per_token + k] : 0;
      Words words = first_v < padded_vecs ? read(row, first_v) : Words{};
      int permuted_row = -1;
      float fc1_scale = 0.f, fc2_scale = 0.f;
      if (k < experts_per_token) {
        int const local = expert - start_expert;
        if (local >= 0 && local < num_experts_per_node) {
          permuted_row = unpermuted_row_to_permuted_row[k * num_rows + row];
          fc1_scale = token_scales ? fc1_alpha[local] * token_scales[row] : fc1_alpha[local];
          fc2_scale = IsMX ? fc2_alpha[local] * 16.f : fc2_alpha[local];
        }
        permuted_rows[threadIdx.x] = permuted_row;
      }
      __syncthreads();
      bool wide = false;
      for (int64_t v = first_v; v < padded_vecs; v += stride) {
        Words const next = v + stride < padded_vecs ? read(row, v + stride) : Words{};
        uint4 const out = restore(words, v, wide);
        for (int64_t j = 0; j < chunk; ++j) {
          if (permuted_rows[j] >= 0) {
            reinterpret_cast<uint4*>(output)[permuted_rows[j] * padded_vecs + v] = out;
          }
        }
        words = next;
      }
      wide = __syncthreads_or(wide);  // also the rows read
      if (permuted_row >= 0 && blockIdx.y == 0) {
        fc1_row_scales[permuted_row] = fc1_scale;
        fc2_row_scales[permuted_row] = fc2_scale;
        if constexpr (IsMX) {
          wide_rows[permuted_row] = wide;
        }
      }
    }
  }
}

// GEMM1's flagged rows recomputed from the MXFP8 codes, an expert's 32 at a time (mma m16n8k16,
// codes and weights in BF16, exact): each 32-wide block's sum (two mma steps from zero) times
// 2^(ea + ew - 254) in two steps (exact for a normal product), ea the input's E8M0, ew the weights'
// (BF16 scales 2^(ew - 130)), times the row's GEMM1 scale; NaN with a NaN scale or code. A CTA
// takes an expert's rows and 128 columns, a warp 32, the warps sharing the rows' codes in BF16,
// 4 blocks at a time, the next 4 loading meanwhile (cols % 128 == 0).
template <class OutT>
__global__ void __launch_bounds__(128)
    completeWideMxfp8RowsKernel(OutT* output, QuantParams::RestoredMxfp8Inputs const input,
                                void const* weights, void const* weight_scales,
                                float const* row_scales, int64_t const* expert_first_token_offset,
                                int64_t const num_rows, int64_t const cols, int64_t const n) {
  auto const* codes = static_cast<uint8_t const*>(input.codes);
  int64_t const blocks = cols / TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;
  unsigned const tiles = n / 128;  // column tiles
  int64_t const expert = blockIdx.x / tiles;
  int64_t const first = expert_first_token_offset[expert];
  int64_t const last = expert_first_token_offset[expert + 1];
  int const lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
  unsigned const col0 = blockIdx.x % tiles * 128 + warp * 32;
  // B (see below): the warp's words from column quad col0 / 4, quad q's q * (cols / 64) * 32 words
  // on, a chunk's at most cols / 2 more (32-bit). Scales: columns col0 to col0 + 31, n a block.
  uint32_t const* warp_weights = reinterpret_cast<uint32_t const*>(weights) +
                                 expert * n * cols / 8 + col0 / 4 * (cols / 64) * 32;
  uint32_t const* warp_scales =
      reinterpret_cast<uint32_t const*>(weight_scales) + expert * (cols / 16) * n / 2 + col0 / 2;
  constexpr unsigned ABlocks = 4;
  // A: the tile's 32 rows of ABlocks blocks (codes as loaded, two chunks, then BF16 with 8 values
  // more a row: ldmatrix rows in other banks) and E8M0 scales; rows past count unfilled, unused.
  __shared__ alignas(16) uint8_t tile_codes[2][32][ABlocks * 32];
  __shared__ alignas(16) uint16_t tile_a[32][ABlocks * 32 + 8];
  __shared__ alignas(16) float tile_sf[ABlocks][8][4];  // as 2^(ea - 127): row 8i + g at [g][i]
  static_assert(32 * ABlocks == 128);
  // A warp's B words for two chunks (8 column quads, 4 words padded: 2j, 2j + 1 in other banks)
  // and its scale words.
  __shared__ alignas(16) uint32_t tile_w[2][4][8][ABlocks / 2 * 32 + 4];
  __shared__ alignas(16) uint32_t tile_u[2][4][ABlocks][16];
  // The chunk's tile_u as 2^(ew - 127) in BF16, column col0 + 8j + g's in both halves of [8j + g].
  __shared__ alignas(8) uint32_t tile_sw[4][ABlocks][32];
  __shared__ int64_t tile_rows[32 + 128], tile_tokens[32 + 128];  // flagged, not yet computed
  __shared__ int warp_counts[4];
  int pending = 0;
  for (int64_t start = first; start < last; start += blockDim.x) {
    // This piece's flagged rows after the pending ones, in order.
    int64_t const row = start + threadIdx.x;
    unsigned const ballot = __ballot_sync(~0u, row < last && input.wide_rows[row]);
    if (lane == 0) {
      warp_counts[warp] = __popc(ballot);
    }
    __syncthreads();  // also the last tile's rows read
    int offset = pending, total = pending;
    for (int w = 0; w < 4; ++w) {
      offset += w < warp ? warp_counts[w] : 0;
      total += warp_counts[w];
    }
    if (ballot >> lane & 1) {
      int const pos = offset + __popc(ballot & ((1u << lane) - 1));
      tile_rows[pos] = row;
      tile_tokens[pos] = input.permuted_row_to_unpermuted_row[row] % num_rows;
    }
    __syncthreads();
    bool const end = start + blockDim.x >= last;
    int done = 0;
    while (total - done >= 32 || (end && done < total)) {
      int const count = min(32, total - done);
      // This thread's rows g + 8i (past count not stored), the second 16 (h = 1) only past 16 rows.
      int const halves = count > 16 ? 2 : 1;
      float acc[2][4][4] = {};
      // The chunk from block0 into buffer buf: codes 8 bytes a thread and step (row threadIdx.x /
      // 4, bytes 8 (threadIdx.x % 4) + 32i, at 32 (i ^ row % 4) on in the tile), B 16 bytes a lane
      // and quad (blocks 2 apart), scales 4 lanes a block (cp.async), E8M0 threadIdx.x to sf_next.
      uint8_t sf_next = 0;
      auto const load = [&](unsigned const block0, int const buf) {
        if (int(threadIdx.x / 4) < count) {
          unsigned const row = threadIdx.x / 4;
          uint8_t const* const src =
              codes + tile_tokens[done + row] * cols + block0 * 32 + threadIdx.x % 4 * 8;
#pragma unroll
          for (unsigned i = 0; i < ABlocks; ++i) {
            cutlass::arch::cp_async<8, cutlass::arch::CacheOperation::Always>(
                &tile_codes[buf][row][(i ^ row % 4) * 32 + threadIdx.x % 4 * 8], src + 32 * i);
          }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int const q = 2 * i + lane / 16, part = lane % 16;
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
              &tile_w[buf][warp][q][part * 4],
              warp_weights + q * (cols / 64) * 32 + block0 / 2 * 32 + part * 4);
        }
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            &tile_u[buf][warp][lane / 4 % ABlocks][lane % 4 * 4],
            warp_scales + (block0 + lane / 4 % ABlocks) * n + lane % 4 * 4, lane < 4 * ABlocks);
        cutlass::arch::cp_async_fence();
        if (int(threadIdx.x / ABlocks) < count) {
          sf_next = *cvt_quant_get_sf_out_offset<uint8_t, 1>(
              std::nullopt, tile_tokens[done + threadIdx.x / ABlocks],
              block0 + threadIdx.x % ABlocks, std::nullopt, blocks, const_cast<uint8_t*>(input.sf),
              input.swizzled_sf ? QuantizationSFLayout::SWIZZLED_128x4
                                : QuantizationSFLayout::LINEAR);
        }
      };
      load(0, 0);
      int buf = 0;
      for (unsigned block0 = 0; block0 < blocks; block0 += ABlocks, buf ^= 1) {
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();  // this chunk in buffer buf, the last one's read
        uint8_t const sf_now = sf_next;
        if (block0 + ABlocks < blocks) {
          load(block0 + ABlocks, buf ^ 1);
        }
#pragma unroll
        for (unsigned i = threadIdx.x; i < 32 * ABlocks * 2; i += 128) {
          unsigned const row = i / (ABlocks * 2), part = i % (ABlocks * 2);  // 16 codes
          if (int(row) >= count) {
            break;
          }
          uint4 const v =
              *reinterpret_cast<uint4 const*>(&tile_codes[buf][row][(part ^ row % 4 * 2) * 16]);
          uint32_t w[4] = {v.x, v.y, v.z, v.w}, p[8], nan = 0;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            nan |= (w[k] & 0x7f7f7f7fu) + 0x01010101u;
          }
          // fast_dequant_f8f16x4 is exact but for NaN (codes 0x7f, 0xff: 480).
          if (nan & 0x80808080u) {
            restoreMxfp8x8Scaled(w[0] | uint64_t(w[1]) << 32, 127, 0, p);
            restoreMxfp8x8Scaled(w[2] | uint64_t(w[3]) << 32, 127, 0, p + 4);
          } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              flashinfer::fast_dequant_f8f16x4<__nv_fp8_e4m3, __nv_bfloat16>(
                  &w[k], reinterpret_cast<uint2*>(&p[2 * k]));
            }
          }
          *reinterpret_cast<uint4*>(&tile_a[row][part * 16]) = make_uint4(p[0], p[1], p[2], p[3]);
          *reinterpret_cast<uint4*>(&tile_a[row][part * 16 + 8]) =
              make_uint4(p[4], p[5], p[6], p[7]);
        }
        tile_sf[threadIdx.x % ABlocks][threadIdx.x / ABlocks % 8][threadIdx.x / ABlocks / 8] =
            __uint_as_float(unsigned(sf_now) << 23);
        // With all the chunk's ea 1 to 254 and the warp's ew 21 to 223 (BF16 exponent fields 18 to
        // 220), the sum times 2^(ew - 127), then 2^(ea - 127) in the FMA, gives the same result:
        // for |sum| 0 or 2^-10 to 2^17 both these steps and the two below are exact.
        bool ew = true;
#pragma unroll
        for (int i = 0; i < 2; ++i) {  // the warp's ABlocks * 16 words
          uint32_t const u = (&tile_u[buf][warp][0][0])[lane + 32 * i];
          ew &= ((u & 0xffff) - (18u << 7) < (203u << 7)) & ((u >> 16) - (18u << 7) < (203u << 7));
          if (halves == 2) {
            uint32_t const sw = (u & 0x7f807f80u) + (3u << 23 | 3u << 7);
            reinterpret_cast<uint2*>(&tile_sw[warp][0][0])[lane + 32 * i] =
                make_uint2(__byte_perm(sw, 0, 0x1010), __byte_perm(sw, 0, 0x3232));
          }
        }
        // __syncthreads_and is also the barrier between the stores above and the reads below.
        bool const split =
            __all_sync(~0u, ew) &
            __syncthreads_and(int(threadIdx.x / ABlocks) >= count || unsigned(sf_now - 1) < 254);
        // Block block0 + block of hs halves, the first rs of a thread's 4 results (2: row g's).
        auto const block_step = [&](unsigned const block, bool const fast, int const hs,
                                    int const rs = 4) {
          // A: rows 16h + g, + 8, K 2t, 2t + 1 and 8 more, then 16 more (ldmatrix: lane's row 16h +
          // lane % 16, K 8 (lane / 16)).
          cutlass::Array<unsigned, 4> a[2][2];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              if (h < hs) {
                cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
                    a[h][k], &tile_a[16 * h + lane % 16][block * 32 + k * 16 + lane / 16 * 8]);
              }
            }
          }
          // Each MX scale fills two 16-wide K steps; this thread's columns 2t, 2t + 1 of each j.
          uint32_t u2[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            u2[j] = tile_u[buf][warp][block][4 * j + t];
          }
          float4 const r = *reinterpret_cast<float4 const*>(&tile_sf[block][g][0]);
          float const ra[4] = {r.x, r.y, r.z, r.w};  // rows g + 8i
          int ea[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            ea[i] = __float_as_uint(ra[i]) >> 23;
          }
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            // B: column col0 + 8j + g, bytes 4t to 4t + 3 of its block (cutlass_nvfp4_sm80_weights:
            // column quad, 64-wide K, column, 32-wide K): byte 4t + i holds K 8i + 2t, 8i + 2t + 1.
            uint32_t const w =
                tile_w[buf][warp][2 * j + g / 4][block / 2 * 32 + g % 4 * 8 + block % 2 * 4 + t];
            uint32_t b[4];
            flashinfer::e2m1x8_to_bf16x8(w, b);
            if (fast && hs == 2) {
              // Times column col0 + 8j + g's 2^(ew - 127), exact, as is the sum (products 0 or
              // >= 2^-116, sums below 2^114). One half scales the sums instead (fewer multiplies).
              uint32_t const sw = tile_sw[warp][block][8 * j + g];
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                reinterpret_cast<__nv_bfloat162&>(b[k]) =
                    __hmul2(reinterpret_cast<__nv_bfloat162&>(b[k]),
                            reinterpret_cast<__nv_bfloat162 const&>(sw));
              }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              if (h >= hs) {
                break;
              }
              float sum[4] = {};
              cutlass_kernels_oss::sm80_fp4_small_m::mma_bf16(sum, a[h][0].data(), b);
              cutlass_kernels_oss::sm80_fp4_small_m::mma_bf16(sum, a[h][1].data(), b + 2);
              if (fast) {
                float const rw[2] = {__uint_as_float((u2[j] << 16 & 0x7f800000u) + (3u << 23)),
                                     __uint_as_float((u2[j] & 0x7f800000u) + (3u << 23))};
#pragma unroll
                for (int i = 0; i < rs; ++i) {
                  acc[h][j][i] =
                      fmaf(hs == 2 ? sum[i] : sum[i] * rw[i % 2], ra[2 * h + i / 2], acc[h][j][i]);
                }
                continue;
              }
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                uint32_t const u = u2[j] >> 16 * (i % 2) & 0xffff;
                float s = sum[i];
                if (ea[2 * h + i / 2] == 255 || u >> 7 == 0xff) {
                  s = __uint_as_float(0x7fc00000u);
                }
                int const e = ea[2 * h + i / 2] + (u >> 7 ? (u >> 7) + 3 : __ffs(u) - 4) - 254;
                acc[h][j][i] = fmaf(s * exp2Int(e >> 1), exp2Int(e - (e >> 1)), acc[h][j][i]);
              }
            }
          }
        };
        if (split && halves == 2) {
#pragma unroll
          for (unsigned block = 0; block < ABlocks; ++block) {
            block_step(block, true, 2);
          }
        } else if (split && count > 8) {
#pragma unroll
          for (unsigned block = 0; block < ABlocks; ++block) {
            block_step(block, true, 1);
          }
        } else if (split) {
#pragma unroll
          for (unsigned block = 0; block < ABlocks; ++block) {
            block_step(block, true, 1, 2);
          }
        } else {
#pragma unroll 1
          for (unsigned block = 0; block < ABlocks; ++block) {
            block_step(block, false, halves);
          }
        }
      }
      __syncthreads();  // the tile's buffers read
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (g + 8 * (i / 2) < count) {
          int64_t const out_row = tile_rows[done + g + 8 * (i / 2)];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            output[out_row * n + col0 + 8 * j + 2 * t + i % 2] =
                OutT(acc[i / 4][j][i % 4] * row_scales[out_row]);
          }
        }
      }
      done += count;
    }
    // Move the rest (fewer than 32) to the front.
    pending = total - done;
    int const i = threadIdx.x;
    int64_t const rest_row = i < pending ? tile_rows[done + i] : 0;
    int64_t const rest_token = i < pending ? tile_tokens[done + i] : 0;
    __syncthreads();
    if (i < pending) {
      tile_rows[i] = rest_row;
      tile_tokens[i] = rest_token;
    }
  }
}

// Per-token NVFP4: a permuted row's per-token scale times its expert's alpha, its FP32 GEMM scale.
__global__ void nvfp4RowScalesKernel(float* row_scales, float const* expert_alpha,
                                     int64_t const* expert_first_token_offset,
                                     int64_t const num_experts_per_node) {
  int64_t const rows = expert_first_token_offset[num_experts_per_node];
  for (int64_t row = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; row < rows;
       row += int64_t(gridDim.x) * blockDim.x) {
    int64_t const expert =
        findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node, row + 1) - 1;
    row_scales[row] = expert_alpha[expert] * row_scales[row];
  }
}
#endif

// ====================== Compute FP8 dequant scale only ===============================
__global__ void computeFP8DequantScaleKernel(float const** alpha_scale_ptr_array,
                                             int64_t const num_experts_per_node,
                                             float const* fp8_dequant) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

  assert(fp8_dequant != nullptr);
  alpha_scale_ptr_array[expert] = fp8_dequant + expert;
}

float const** computeFP8DequantScale(float const** alpha_scale_ptr_array,
                                     int const num_experts_per_node, float const* fp8_dequant,
                                     cudaStream_t stream) {
  if (!fp8_dequant) {
    return nullptr;
  }

  int const threads = std::min(1024, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;

  computeFP8DequantScaleKernel<<<blocks, threads, 0, stream>>>(alpha_scale_ptr_array,
                                                               num_experts_per_node, fp8_dequant);

  return alpha_scale_ptr_array;
}

__global__ void prepareProfilerFP8TokenScalePtrArrayKernel(
    float const** alpha_scale_ptr_array, float const* fp8_token_scale,
    int64_t const* expert_first_token_offset, int const num_experts_per_node,
    int const expert_first_token_offset_stride) {
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  int const sample = blockIdx.y;
  if (expert >= num_experts_per_node) {
    return;
  }

  auto const* sample_offsets =
      expert_first_token_offset + sample * expert_first_token_offset_stride;
  alpha_scale_ptr_array[sample * num_experts_per_node + expert] =
      fp8_token_scale + sample_offsets[expert];
}

float const** prepareProfilerFP8TokenScalePtrArray(float const** alpha_scale_ptr_array,
                                                   float const* fp8_token_scale,
                                                   int64_t const* expert_first_token_offset,
                                                   int const num_experts_per_node,
                                                   int const num_samples, cudaStream_t stream) {
  int const threads = std::min(128, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;
  prepareProfilerFP8TokenScalePtrArrayKernel<<<dim3(blocks, num_samples), threads, 0, stream>>>(
      alpha_scale_ptr_array, fp8_token_scale, expert_first_token_offset, num_experts_per_node,
      num_experts_per_node + 1);
  return alpha_scale_ptr_array;
}

template <class BSConfig>
__device__ void setupFP4BlockScalingFactors(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int expert, int gemm_m, int gemm_n, int gemm_k,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* weight_block_scale,
    int64_t num_tokens_before_expert) {
  assert(layout_info.fpX_block_scaling_factors_stride_act);
  assert(layout_info.fpX_block_scaling_factors_stride_weight);

  auto stride_act_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(
      layout_info.fpX_block_scaling_factors_stride_act);
  auto stride_weight_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(
      layout_info.fpX_block_scaling_factors_stride_weight);
  if (layout_info.swap_ab) {
    // M & N swapped for transpose
    stride_act_ptr[expert] = BSConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
    stride_weight_ptr[expert] = BSConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
  } else {
    stride_act_ptr[expert] = BSConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, (int)1));
    stride_weight_ptr[expert] = BSConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, (int)1));
  }

  // This assert validates our current assumption that A&B can be safely transposed without needing
  // to modify
  assert(
      BSConfig::tile_atom_to_shape_SFB(
          cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, 1)) ==
      BSConfig::tile_atom_to_shape_SFA(cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, 1)));

  auto scaling_type =
      std::is_same_v<BSConfig, TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>
          ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
          : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
  layout_info.fpX_block_scaling_factors_act[expert] =
      fp4_act_flat + getOffsetActivationSF(expert, num_tokens_before_expert, gemm_k, scaling_type);

  layout_info.fpX_block_scaling_factors_weight[expert] =
      weight_block_scale + getOffsetWeightSF(expert, gemm_n, gemm_k, scaling_type);
}

__device__ void computeTmaWarpSpecializedInputStrides(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int gemm_m, int gemm_n, int gemm_k,
    int64_t out_idx) {
  if (layout_info.swap_ab) {
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideB*>(
        layout_info.stride_act)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideB{},
                                         cute::make_shape(gemm_m, gemm_k, 1));
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideA*>(
        layout_info.stride_weight)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideA{},
                                         cute::make_shape(gemm_n, gemm_k, 1));
  } else {
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideA*>(
        layout_info.stride_act)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideA{},
                                         cute::make_shape(gemm_m, gemm_k, 1));
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideB*>(
        layout_info.stride_weight)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideB{},
                                         cute::make_shape(gemm_n, gemm_k, 1));
  }
  if (layout_info.stride_c) {
    // TODO Enable 1xN bias matrix as C
    assert(false && "CUTLASS does not support a 1xN bias");
  }
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE ||
      layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION) {
    if (layout_info.swap_ab) {
      reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideD_T*>(
          layout_info.stride_d)[out_idx] =
          cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideD_T{},
                                           cute::make_shape(gemm_n, gemm_m, 1));
    } else {
      reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideD*>(
          layout_info.stride_d)[out_idx] =
          cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideD{},
                                           cute::make_shape(gemm_m, gemm_n, 1));
    }
  }
  if (layout_info.int4_groupwise_params.enabled) {
    layout_info.int4_groupwise_params.stride_s_a[out_idx] = cutlass::make_cute_packed_stride(
        TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::StrideSFA{},
        cute::make_shape(
            gemm_n,
            gemm_k /
                (layout_info.int4_groupwise_params.use_wfp4a16
                     ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size
                     : TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size),
            1));
  }
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__device__ void computeTmaWarpSpecializedInputPointers(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int64_t gemm_m, int64_t gemm_n, int64_t gemm_k,
    int num_tokens_before_expert, int64_t expert, T const* in, WeightType const* weights,
    TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const* w4a8_weight_scale,
    ScaleBiasType const* bias, OutputType* output, float const* router_scales,
    int const* permuted_row_to_unpermuted_row, int64_t const out_idx) {
  // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
  layout_info.ptr_act[out_idx] = safe_inc_ptr(in, num_tokens_before_expert * gemm_k);

  // Each expert's weight matrix is a constant size NxK, get the matrix at index `expert`
  layout_info.ptr_weight[out_idx] = safe_inc_ptr(weights, expert * (gemm_n * gemm_k));

  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE) {
    // The output prior to this contains N elements per token, with `num_tokens_before_expert`
    // tokens
    layout_info.ptr_d[out_idx] = safe_inc_ptr(output, num_tokens_before_expert * gemm_n);
  } else if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION) {
    // The activation epilogue writes the GEMM2 input type (BF16 or MXFP8),
    // which can differ from the ordinary GEMM1 accumulator/output type.
    layout_info.ptr_d[out_idx] =
        safe_inc_ptr(reinterpret_cast<T*>(output), num_tokens_before_expert * gemm_n);
  }
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
    layout_info.fused_finalize_epilogue.ptr_source_token_index[expert] =
        permuted_row_to_unpermuted_row + num_tokens_before_expert;
    layout_info.fused_finalize_epilogue.ptr_router_scales[expert] =
        router_scales + num_tokens_before_expert;
    if (layout_info.fused_finalize_epilogue.ptr_bias != nullptr) {
      layout_info.fused_finalize_epilogue.ptr_bias[expert] = bias + gemm_n * expert;
    }
  }
  if (layout_info.int4_groupwise_params.enabled) {
    // The group size of wfp4a16 is multiplied by 2 because each scale uses 1 byte instead of 2
    // bytes
    layout_info.int4_groupwise_params.ptr_s_a[out_idx] = safe_inc_ptr(
        w4a8_weight_scale,
        expert *
            (gemm_n * gemm_k /
             (layout_info.int4_groupwise_params.use_wfp4a16
                  ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size * 2
                  : TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size)));
  }
}

// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedKernel(
    int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens,
    int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k,
    int64_t const num_experts_per_node, T const* gemm1_in, T const* gemm2_in,
    WeightType const* weights1, WeightType const* weights2, float const* alpha_scale_flat1,
    float const* alpha_scale_flat2,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, OutputType* gemm1_output,
    OutputType* gemm2_output, float const* router_scales,
    int const* permuted_row_to_unpermuted_row) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Both gemms use the same token offset
  auto const num_tokens_before_expert = expert_first_token_offset[expert];
  auto const num_tokens_including_expert = expert_first_token_offset[expert + 1];
  auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
  auto const gemm_m = num_tokens_to_expert;

  if (expert == 0 &&
      layout_info1.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION) {
    *layout_info1.fused_activation_epilogue.norm_constant = 1.0f;
  }
  // M and N transposed since we are using the #tokens as the N dimension
  layout_info1.shape_info.problem_shapes[expert] =
      TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info1.swap_ab ? gemm1_n : gemm_m, layout_info1.swap_ab ? gemm_m : gemm1_n,
          gemm1_k);
  layout_info2.shape_info.problem_shapes[expert] =
      TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info2.swap_ab ? gemm2_n : gemm_m, layout_info2.swap_ab ? gemm_m : gemm2_n,
          gemm2_k);

  if (layout_info1.int4_groupwise_params.enabled) {
    layout_info1.int4_groupwise_params.shape.problem_shapes[expert] =
        TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::
            UnderlyingProblemShape(layout_info1.swap_ab ? gemm1_n : gemm_m,
                                   layout_info1.swap_ab ? gemm_m : gemm1_n, gemm1_k);
  }

  if (layout_info2.int4_groupwise_params.enabled) {
    layout_info2.int4_groupwise_params.shape.problem_shapes[expert] =
        TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::
            UnderlyingProblemShape(layout_info2.swap_ab ? gemm2_n : gemm_m,
                                   layout_info2.swap_ab ? gemm_m : gemm2_n, gemm2_k);
  }

  bool const needs_zero_token_weight_desc =
      layout_info1.int4_groupwise_params.enabled || layout_info2.int4_groupwise_params.enabled;

  auto compute_tma_strides = [&]() {
    assert(gemm_m <= INT32_MAX);
    assert(gemm1_n > 0 && gemm1_n <= INT32_MAX);
    assert(gemm1_k > 0 && gemm1_k <= INT32_MAX);
    assert(gemm2_n > 0 && gemm2_n <= INT32_MAX);
    assert(gemm2_k > 0 && gemm2_k <= INT32_MAX);
    computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert);
    computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert);
  };

  if (gemm_m != 0 || needs_zero_token_weight_desc) {
    compute_tma_strides();
  }

  if (gemm_m == 0 && needs_zero_token_weight_desc) {
    layout_info1.ptr_weight[expert] = safe_inc_ptr(weights1, expert * (gemm1_n * gemm1_k));
    layout_info2.ptr_weight[expert] = safe_inc_ptr(weights2, expert * (gemm2_n * gemm2_k));
  }

  // Skip expensive stride/pointer/SF setup for experts with no assigned tokens.
  // All problem shapes (including int4_groupwise) are initialized above so CUTLASS
  // can correctly traverse the problem list. For groupwise W4 paths, weight
  // strides/pointers are initialized above because the prebuilt global weight
  // TMA descriptor uses group 0 as its base even when that expert has no tokens.
  // The remaining work (alpha scales, block scaling factors, activation/output
  // pointers) is only needed for active experts.
  // For decode (1 token, top_k=8, 128 experts), this skips ~120 of 128 experts.
  if (gemm_m == 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  if (alpha_scale_flat1 && alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + expert;
    layout_info2.alpha_scale_ptr_array[expert] = alpha_scale_flat2 + expert;
  }

  auto setupIfSelected = [&](auto bs_config, auto quant_type) {
    if (quant_type.fc1.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(
          layout_info1, expert, gemm_m, gemm1_n, gemm1_k, fp4_act_flat1,
          quant_type.fc1.weight_block_scale, num_tokens_before_expert);
    }
    if (quant_type.fc2.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(
          layout_info2, expert, gemm_m, gemm2_n, gemm2_k, fp4_act_flat2,
          quant_type.fc2.weight_block_scale, num_tokens_before_expert);
    }
  };

  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig{}, quant_params.fp4);
  // FP8 x MXFP4 has plain FP8 activation and must not fabricate an activation
  // block-scale tensor.  The Humming-style pre-MMA path consumes folded weight
  // scale through int4_groupwise metadata; future post-MMA FP8 x MXFP4 support
  // should use a weight-scale-only path rather than MXFPX activation SF setup.
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{},
                  quant_params.mxfp8_mxfp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{},
                  quant_params.mxfp8_mxfp8);

  if (layout_info1.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION &&
      layout_info1.fpX_block_scaling_type ==
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX) {
    assert(!layout_info1.swap_ab);
    layout_info1.fused_activation_epilogue.ptr_block_scaling_factors[expert] =
        const_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(
            fp4_act_flat2 +
            getOffsetActivationSF(expert, num_tokens_before_expert, gemm1_n,
                                  TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX));
  }

  auto const* fc1_weight_scale =
      quant_params.fp8_mxfp4.fc1.weight_block_scale
          ? reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
                quant_params.fp8_mxfp4.fc1.weight_block_scale)
          : reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
                quant_params.groupwise.fc1.weight_scales);
  auto const* fc2_weight_scale =
      quant_params.fp8_mxfp4.fc2.weight_block_scale
          ? reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
                quant_params.fp8_mxfp4.fc2.weight_block_scale)
          : reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
                quant_params.groupwise.fc2.weight_scales);

  computeTmaWarpSpecializedInputPointers(
      layout_info1, gemm_m, gemm1_n, gemm1_k, num_tokens_before_expert, expert, gemm1_in, weights1,
      fc1_weight_scale, bias1, gemm1_output, nullptr, nullptr, expert);
  computeTmaWarpSpecializedInputPointers(
      layout_info2, gemm_m, gemm2_n, gemm2_k, num_tokens_before_expert, expert, gemm2_in, weights2,
      fc2_weight_scale, bias2, gemm2_output, router_scales, permuted_row_to_unpermuted_row, expert);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ========================== Permutation things =======================================

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
  return converter(input);
}

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with
// finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since
// we will have to duplicate some rows in the input matrix to match the dimensions. Duplicates will
// always get routed to separate experts in the end.

// Note that the permuted_row_to_unpermuted_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ...
// (k-1)*rows_in_input all map to row 0 in the original matrix. Thus, to know where to read in the
// source matrix, we simply take the modulus of the expanded index.

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

template <class InputActivationsType, class ExpandedActivationsType,
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType BlockScalingType,
          bool PRE_QUANT_AWQ, bool DISABLE_FP4_QUANT_FAST_MATH = false,
          typename NVFP4_4OVER6_CONFIG = std::false_type>
__global__ void expandInputRowsKernel(
    InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,
    float const* unpermuted_scales, float* permuted_scales,
    int const* permuted_row_to_unpermuted_row, int const* permuted_token_selected_experts,
    int64_t const num_tokens, int64_t const hidden_size, int64_t const k,
    float const* fc1_act_global_scale, bool use_per_expert_act_scale,
    int64_t const* expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, bool const swizzled_input_sf,
    int64_t const num_experts_per_node, InputActivationsType const* prequant_scales = nullptr,
    float* fp8_token_dequant_scale = nullptr, float const* fp8_expert_residual_scale = nullptr,
    float const** fp8_token_scale_ptr_array = nullptr) {
  static_assert(BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE ||
                    !PRE_QUANT_AWQ,
                "AWQ and Block Scaling are mutually exclusive");
  constexpr bool is_mxfp8 =
      std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX &&
      !PRE_QUANT_AWQ;
  constexpr bool is_mxfp8_input = is_mxfp8 && std::is_same_v<InputActivationsType, __nv_fp8_e4m3>;
  constexpr bool need_mxfp8_quant = is_mxfp8 && !is_mxfp8_input;
  constexpr bool need_per_token_fp8_quant =
      std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE &&
      !PRE_QUANT_AWQ && !std::is_same_v<InputActivationsType, ExpandedActivationsType>;

#ifdef ENABLE_FP4
  constexpr bool is_nvfp4 =
      std::is_same_v<ExpandedActivationsType, Fp4Type> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4 &&
      !PRE_QUANT_AWQ;
  constexpr bool is_nvfp4_input = is_nvfp4 && std::is_same_v<InputActivationsType, Fp4Type>;
  constexpr bool need_nvfp4_quant = is_nvfp4 && !is_nvfp4_input;
#else
  constexpr bool is_nvfp4 = false;
  constexpr bool is_nvfp4_input = false;
  constexpr bool need_nvfp4_quant = false;
#endif

  static_assert(need_nvfp4_quant || need_mxfp8_quant || need_per_token_fp8_quant || PRE_QUANT_AWQ ||
                    std::is_same_v<InputActivationsType, ExpandedActivationsType>,
                "Only NVFP4, MXFP8 and WINT4_AFP8 supports outputting a different format as part "
                "of the expansion");
  static_assert(
      !IsNVFP44Over6Config<NVFP4_4OVER6_CONFIG>::value ||
          BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4,
      "NVFP4 4over6 requires NVFP4 block scaling");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  constexpr int VecSize = is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                   : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;

  constexpr int64_t ELEM_PER_THREAD = (is_nvfp4 || is_mxfp8)
                                          ? CVT_ELTS_PER_THREAD
                                          : (128 / sizeof_bits<InputActivationsType>::value);

  // This should be VecSize * 4 elements
  // We assume at least VecSize alignment or the quantization will fail
  constexpr int64_t min_k_dim_alignment =
      is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
               : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;
  int64_t const padded_hidden_size =
      TmaWarpSpecializedGroupedGemmInput::alignToSfDim(hidden_size, min_k_dim_alignment);

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];

  if (fp8_token_scale_ptr_array && fp8_token_dequant_scale && blockIdx.x == 0) {
    for (int expert = threadIdx.x; expert < num_experts_per_node; expert += blockDim.x) {
      fp8_token_scale_ptr_array[expert] =
          fp8_token_dequant_scale + expert_first_token_offset[expert];
    }
  }

  for (int64_t permuted_row = blockIdx.x; permuted_row < num_valid_tokens;
       permuted_row += gridDim.x) {
    int64_t const unpermuted_row = permuted_row_to_unpermuted_row[permuted_row];

    // Load 128-bits per thread

    constexpr int64_t ELEM_PER_BYTE = is_nvfp4_input ? 2 : 1;
    using DataElem = std::conditional_t<
        is_nvfp4_input, uint32_t,
        std::conditional_t<is_mxfp8_input, uint64_t,
                           cutlass::Array<InputActivationsType, ELEM_PER_THREAD>>>;
    using OutputElem = std::conditional_t<
        is_nvfp4, uint32_t,
        std::conditional_t<is_mxfp8, uint64_t,
                           cutlass::Array<ExpandedActivationsType, ELEM_PER_THREAD>>>;

    // Duplicate and permute rows
    int64_t const source_k_rank = unpermuted_row / num_tokens;
    int64_t const source_row = unpermuted_row % num_tokens;

    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(
        unpermuted_input + source_row * hidden_size / ELEM_PER_BYTE);
    // Cast first to handle when this is FP4
    auto* dest_row_ptr = reinterpret_cast<OutputElem*>(permuted_output) +
                         permuted_row * hidden_size / ELEM_PER_THREAD;
    assert(reinterpret_cast<std::uintptr_t>(source_row_ptr) % sizeof(DataElem) == 0);
    assert(reinterpret_cast<std::uintptr_t>(dest_row_ptr) % sizeof(OutputElem) == 0);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = EXPAND_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = hidden_size / ELEM_PER_THREAD;
    assert(hidden_size % ELEM_PER_THREAD == 0);
    assert(hidden_size % VecSize == 0);

    if constexpr (is_nvfp4 || is_mxfp8) {
      static_assert(ELEM_PER_THREAD == 8, "Expecting 8 elements per thread for quantized types");
      int64_t expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node,
                                                   (int64_t)permuted_row + 1) -
                       1;

      assert(!fc1_act_global_scale || is_nvfp4 && "Global scale is only supported for NVFP4");
      size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
      float global_scale_val = fc1_act_global_scale ? fc1_act_global_scale[act_scale_idx] : 1.0f;
      int64_t num_tokens_before_expert = expert_first_token_offset[expert];

      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        DataElem in_vec;
        cutlass::arch::global_load<DataElem, sizeof(DataElem)>(in_vec, source_row_ptr + elem_index,
                                                               true);
        if constexpr (need_nvfp4_quant || need_mxfp8_quant) {
          auto res =
              quantizePackedFPXValue<InputActivationsType, ExpandedActivationsType, DataElem,
                                     VecSize, DISABLE_FP4_QUANT_FAST_MATH, NVFP4_4OVER6_CONFIG>(
                  in_vec, global_scale_val, num_tokens_before_expert, expert, permuted_row,
                  elem_index, padded_hidden_size, fc1_act_sf_flat,
                  is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                           : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
          static_assert(sizeof(res) == sizeof(*dest_row_ptr),
                        "Quantized value must be the same size as the output");
          auto out_vec = cutlass::platform::bit_cast<OutputElem>(res);
          cutlass::arch::global_store<OutputElem, sizeof(OutputElem)>(
              out_vec, dest_row_ptr + elem_index, true);
        } else {
          assert(act_scale_idx == 0 &&
                 "Cannot use per-expert act scale for pre-quantized activations");
          writeSF<VecSize, ELEM_PER_THREAD>(num_tokens_before_expert, expert, source_row,
                                            permuted_row, elem_index, padded_hidden_size,
                                            fc1_act_sf_flat, input_sf, swizzled_input_sf);
          auto out_vec = cutlass::platform::bit_cast<OutputElem>(in_vec);
          cutlass::arch::global_store<OutputElem, sizeof(OutputElem)>(
              out_vec, dest_row_ptr + elem_index, true);
        }
      }

      // Pad zeros in the extra SFs along the K dimension, we do this to ensure there are no nan
      // values in the padded SF atom Use VecSize per thread since we are just writing out zeros so
      // every thread can process a whole vector
      size_t padding_start_offset = hidden_size / VecSize + start_offset;
      size_t padding_elems_in_col = padded_hidden_size / VecSize;
      for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col;
           elem_index += stride) {
        writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1, permuted_row,
                                  elem_index, padded_hidden_size, fc1_act_sf_flat,
                                  /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
      }
    } else if constexpr (PRE_QUANT_AWQ) {
      static_assert(!is_nvfp4 && !is_mxfp8, "NVFP4 and MXFP8 are not supported for AWQ");
      static_assert(!std::is_same_v<InputActivationsType, ExpandedActivationsType>,
                    "Input and output types must be different for AWQ");
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        DataElem frag_elems;
        cutlass::arch::global_load<DataElem, sizeof(DataElem)>(frag_elems,
                                                               source_row_ptr + elem_index, true);

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ELEM_PER_THREAD; e++) {
          frag_elems[e] = frag_elems[e] * prequant_scales[elem_index * ELEM_PER_THREAD + e];
        }

        auto out_vec = arrayConvert<DataElem, OutputElem>(frag_elems);
        cutlass::arch::global_store<OutputElem, sizeof(OutputElem)>(
            out_vec, dest_row_ptr + elem_index, true);
      }
    } else if constexpr (need_per_token_fp8_quant) {
      using AmaxOp = AbsMaxOp<InputActivationsType>;
      typename AmaxOp::Accum thread_amax = AmaxOp::zero();
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto input_value = source_row_ptr[elem_index];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
          thread_amax = AmaxOp::update(thread_amax, input_value[i]);
        }
      }

      using BlockReduce = cub::BlockReduce<float, EXPAND_THREADS_PER_BLOCK>;
      __shared__ typename BlockReduce::TempStorage reduce_storage;
      __shared__ float shared_token_quant_scale;
      float const row_amax =
          BlockReduce(reduce_storage).Reduce(AmaxOp::to_float(thread_amax), FloatMaxOp{});
      if (threadIdx.x == 0) {
        float const quant = computeSafeFP8QuantScale(row_amax);
        float residual = 1.0f;
        if (fp8_expert_residual_scale) {
          int const expert = permuted_token_selected_experts[permuted_row];
          residual = fp8_expert_residual_scale[expert];
        }
        shared_token_quant_scale = quant;
        fp8_token_dequant_scale[permuted_row] = (1.0f / quant) * residual;
      }
      __syncthreads();

      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto input_value = arrayConvert<DataElem, cutlass::Array<float, ELEM_PER_THREAD>>(
            source_row_ptr[elem_index]);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
          input_value[i] *= shared_token_quant_scale;
        }
        dest_row_ptr[elem_index] =
            arrayConvert<cutlass::Array<float, ELEM_PER_THREAD>, OutputElem>(input_value);
      }
    } else {
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        DataElem in_vec;
        cutlass::arch::global_load<DataElem, sizeof(DataElem)>(in_vec, source_row_ptr + elem_index,
                                                               true);
        auto out_vec = cutlass::platform::bit_cast<OutputElem>(in_vec);
        cutlass::arch::global_store<OutputElem, sizeof(OutputElem)>(
            out_vec, dest_row_ptr + elem_index, true);
      }
    }

    if (permuted_scales && threadIdx.x == 0) {
      int64_t const source_k_idx = source_row * k + source_k_rank;
      permuted_scales[permuted_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // N-dim SF padding (zeroing extra token rows beyond tokens_to_expert up to MinNDimAlignment)
  // is intentionally omitted. The CUTLASS grouped GEMM sets gemm_m = tokens_to_expert per expert
  // and never reads scale factors for rows beyond that. The N-dim padding rows don't correspond
  // to any valid MMA tiles, so their content doesn't affect correctness.
  // K-dim SF padding (above, inside the per-token loop) is still required because MMA tiles may
  // straddle the inter_size boundary within valid rows.
}

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(
    InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,
    float const* unpermuted_scales, float* permuted_scales,
    int const* permuted_row_to_unpermuted_row, int const* permuted_token_selected_experts,
    int64_t const num_rows, int64_t const hidden_size, int const k, int const num_experts_per_node,
    QuantParams const& quant_params, bool use_per_expert_act_scale,
    int64_t* expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, bool const swizzled_input_sf,
    void const* prequant_scales, float* fp8_token_dequant_scale,
    float const* fp8_expert_residual_scale, float const** fp8_token_scale_ptr_array,
    bool enable_pdl, cudaStream_t stream) {
#ifdef ENABLE_FP4
  TLLM_CHECK_WITH_INFO((std::is_same_v<ExpandedActivationsType, Fp4Type> && fc1_act_sf_flat) ||
                           !use_per_expert_act_scale,
                       "Per-expert act scale for FC1 is only supported for NVFP4 activations");
#endif

  TLLM_CHECK_WITH_INFO(((hidden_size * std::min(sizeof_bits<InputActivationsType>::value,
                                                sizeof_bits<ExpandedActivationsType>::value)) %
                        128) == 0,
                       "hidden_size %ld rows are not a multiple of 16B; expandInputRowsKernel "
                       "requires 16B-aligned rows",
                       hidden_size);
  TLLM_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(unpermuted_input) % 16 == 0 &&
                           reinterpret_cast<uintptr_t>(permuted_output) % 16 == 0,
                       "expandInputRowsKernel input/output pointers must be 16B-aligned");

  static int64_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
  // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
  // N-dim SF padding has been removed (CUTLASS grouped GEMM never reads beyond
  // tokens_to_expert), so the grid is driven purely by the expanded token count.
  int64_t const blocks = std::min(smCount * 8, std::max(num_rows * k, int64_t{1}));
  int64_t const threads = EXPAND_THREADS_PER_BLOCK;

  auto func = [&]() {
#ifdef ENABLE_FP8
    // Always MXFP8
    if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> &&
                  !std::is_same_v<InputActivationsType, __nv_fp8_e4m3>) {
      bool const use_per_token_fp8_quant = fp8_token_dequant_scale != nullptr;
      TLLM_CHECK_WITH_INFO(quant_params.mxfp8_mxfp4.fc1.weight_block_scale ||
                               quant_params.mxfp8_mxfp8.fc1.weight_block_scale || prequant_scales ||
                               use_per_token_fp8_quant,
                           "MXFP8 block scaling, prequant_scales or FP8 token scale output "
                           "parameters not provided");
      return prequant_scales
                 ? &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, true>
             : use_per_token_fp8_quant
                 ? &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, false>
                 : &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX, false>;
    }
    // Could be either regular FP8 or MXFP8
    else if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> &&
                       std::is_same_v<InputActivationsType, __nv_fp8_e4m3>) {
      TLLM_CHECK_WITH_INFO(!prequant_scales, "FP8 is not supported for AWQ");
      return (quant_params.mxfp8_mxfp4.fc1.weight_block_scale ||
              quant_params.mxfp8_mxfp8.fc1.weight_block_scale)
                 ? &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX, false>
                 : &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, false>;
    } else
#endif
#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<ExpandedActivationsType, Fp4Type>) {
      TLLM_CHECK_WITH_INFO(quant_params.fp4.fc1.weight_block_scale,
                           "NVFP4 block scaling is expected for FP4xFP4");
      TLLM_CHECK_WITH_INFO(!prequant_scales, "NVFP4 is not supported for AWQ");
      return dispatchNVFP44Over6Config(
          quant_params.fp4.nvfp4_4over6_code,
          [&](auto disableFP4QuantFastMathTag, auto nvfp4_4over6_config_tag) {
            return &expandInputRowsKernel<
                InputActivationsType, ExpandedActivationsType,
                TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4, false,
                decltype(disableFP4QuantFastMathTag)::value, decltype(nvfp4_4over6_config_tag)>;
          });
    } else
#endif
    {
      TLLM_CHECK_WITH_INFO(!prequant_scales,
                           "w4afp8 Prequant scales provided for non-FP8 data type");
      return &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE,
                                    false>;
    }
  }();

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(
      &config, func, unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
      permuted_row_to_unpermuted_row, permuted_token_selected_experts, num_rows, hidden_size, k,
      quant_params.fp4.fc1.act_global_scale, use_per_expert_act_scale, expert_first_token_offset,
      fc1_act_sf_flat, input_sf, swizzled_input_sf, num_experts_per_node,
      reinterpret_cast<InputActivationsType const*>(prequant_scales), fp8_token_dequant_scale,
      fp8_expert_residual_scale, fp8_token_scale_ptr_array);
}

#define INSTANTIATE_EXPAND_INPUT_ROWS(InputActivationsType, ExpandedActivationsType)               \
  template void expandInputRowsKernelLauncher<InputActivationsType, ExpandedActivationsType>(      \
      InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,      \
      float const* unpermuted_scales, float* permuted_scales,                                      \
      int const* permuted_row_to_unpermuted_row, int const* permuted_token_selected_experts,       \
      int64_t const num_rows, int64_t const hidden_size, int const k,                              \
      int const num_experts_per_node, QuantParams const& quant_params,                             \
      bool use_per_expert_act_scale, int64_t* expert_first_token_offset,                           \
      TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,                              \
      TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, bool const swizzled_input_sf, \
      void const* prequant_scales, float* fp8_token_dequant_scale,                                 \
      float const* fp8_expert_residual_scale, float const** fp8_token_scale_ptr_array,             \
      bool enable_pdl, cudaStream_t stream)

// Instantiate the data types that are used by the external pytorch op
// INSTANTIATE_EXPAND_INPUT_ROWS(float, float);
// INSTANTIATE_EXPAND_INPUT_ROWS(half, half);
// #ifdef ENABLE_BF16
// INSTANTIATE_EXPAND_INPUT_ROWS(__nv_bfloat16, __nv_bfloat16);
// #endif

enum class ScaleMode : int {
  NO_SCALE = 0,
  DEFAULT = 1,
};

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;
#if defined(FLASHINFER_FP4_SW_CVT)
// Below SM100 finalize splits a row's columns over gridDim.y CTAs when rows are few, caches the
// first FINALIZE_CACHED_K k (48 KB) in shared memory, and loads k rows FINALIZE_K_LOADS at a time.
constexpr static int FINALIZE_CACHED_K = 4096;
constexpr static int FINALIZE_K_LOADS = 4;
#endif

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip
// connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingKernel(
    GemmOutputType const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* scales, int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts, int64_t const padded_cols, int64_t const unpadded_cols,
    int64_t const experts_per_token, int const num_experts_per_node, int const start_expert_id) {
  assert(padded_cols % 4 == 0);
  assert(unpadded_cols % 4 == 0);
  assert(unpadded_cols <= padded_cols);
  int64_t const original_row = blockIdx.x;
  int64_t const num_rows = gridDim.x;
  auto const offset = original_row * unpadded_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

#if defined(FLASHINFER_FP4_SW_CVT)
  int64_t const start_offset = threadIdx.x + blockIdx.y * blockDim.x;
  int64_t const stride = blockDim.x * gridDim.y;
#else
  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
#endif
  int64_t const num_elems_in_padded_col = padded_cols / FINALIZE_ELEM_PER_THREAD;
  int64_t const num_elems_in_orig_col = unpadded_cols / FINALIZE_ELEM_PER_THREAD;

  using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
#if defined(FLASHINFER_FP4_SW_CVT)
  // GEMM2's rows are in the workspace, aligned for a thread's elements to load at once.
  using InputElem = cutlass::AlignedArray<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
#else
  using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
#endif
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
  auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

#if defined(FLASHINFER_FP4_SW_CVT)
  // Per k: permuted row (-1 to skip), expert and scale, the first padded_k in shared memory.
  struct Entry {
    int row, expert;
    float scale;
  };
  auto lookup = [&](int64_t const k_idx) {
    int64_t const k_offset = original_row * experts_per_token + k_idx;
    int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
    int64_t expanded_permuted_row = -1;
    float scale = 0.f;
    if (expert_id >= 0 && expert_id < num_experts_per_node) {
      expanded_permuted_row = unpermuted_row_to_permuted_row[original_row + k_idx * num_rows];
      if (expanded_permuted_row >= num_rows * experts_per_token) {
        expanded_permuted_row = -1;
      }
      scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
    }
    return Entry{static_cast<int>(std::max<int64_t>(expanded_permuted_row, -1)),
                 static_cast<int>(expert_id), scale};
  };
  constexpr int kLoads = FINALIZE_K_LOADS;
  using KInts = cutlass::AlignedArray<int, kLoads>;
  using KFloats = cutlass::AlignedArray<float, kLoads>;
  int64_t const cached_k = std::min<int64_t>(experts_per_token, FINALIZE_CACHED_K);
  int64_t const padded_k = tensorrt_llm::common::roundUp(cached_k, int64_t{kLoads});
  extern __shared__ KInts finalize_smem[];
  int* const finalize_rows = finalize_smem[0].data();
  int* const finalize_experts = finalize_rows + padded_k;
  float* const finalize_scales = reinterpret_cast<float*>(finalize_experts + padded_k);
  for (int64_t k_idx = threadIdx.x; k_idx < padded_k; k_idx += blockDim.x) {
    Entry const entry = k_idx < cached_k ? lookup(k_idx) : Entry{-1, 0, 0.f};
    finalize_rows[k_idx] = entry.row;
    finalize_experts[k_idx] = entry.expert;
    finalize_scales[k_idx] = entry.scale;
  }
  __syncthreads();

  // The k rows are loaded four at a time and added in k order.
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_orig_col;
       elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    for (int k0 = 0; k0 < experts_per_token; k0 += kLoads) {
      Entry entries[kLoads];
      InputElem expert_rows[kLoads];
      if (k0 < padded_k) {  // a whole batch in shared memory, read without a branch per k
        KInts const rows = reinterpret_cast<KInts const*>(finalize_rows)[k0 / kLoads];
        KInts const experts = reinterpret_cast<KInts const*>(finalize_experts)[k0 / kLoads];
        KFloats const k_scales = reinterpret_cast<KFloats const*>(finalize_scales)[k0 / kLoads];
#pragma unroll
        for (int j = 0; j < kLoads; ++j) {
          entries[j] = Entry{rows[j], experts[j], k_scales[j]};
        }
      } else {
#pragma unroll
        for (int j = 0; j < kLoads; ++j) {
          entries[j] = k0 + j < experts_per_token ? lookup(k0 + j) : Entry{-1, 0, 0.f};
        }
      }
#pragma unroll
      for (int j = 0; j < kLoads; ++j) {
        if (entries[j].row >= 0) {
          expert_rows[j] =
              expanded_permuted_rows_v[entries[j].row * num_elems_in_padded_col + elem_index];
        }
      }
#pragma unroll
      for (int j = 0; j < kLoads; ++j) {
        if (entries[j].row < 0) {
          continue;
        }
        ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(expert_rows[j]);
        if (bias) {
          auto const* bias_ptr = bias_v + entries[j].expert * num_elems_in_padded_col;
          expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
        }
        thread_output = thread_output + entries[j].scale * expert_result;
      }
    }
#else
#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_orig_col; elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
      if (expert_id < 0 || expert_id >= num_experts_per_node) {
        continue;
      }

      int64_t const expanded_original_row = original_row + k_idx * num_rows;
      int64_t const expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];

      int64_t const expanded_rows = num_rows * experts_per_token;
      if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows) {
        continue;
      }

      float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_padded_col;

      ComputeElem expert_result =
          arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
      if (bias) {
        auto const* bias_ptr = bias_v + expert_id * num_elems_in_padded_col;
        expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
      }

      thread_output = thread_output + row_scale * expert_result;
    }
#endif

    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip
// connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingNoFillingKernel(
    GemmOutputType const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* scales, int const* const unpermuted_row_to_permuted_row,
    int const* permuted_row_to_unpermuted_row, int const* token_selected_experts,
    int64_t const* expert_first_token_offset, int64_t const num_rows, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token, int const num_experts_per_node,
    int const start_expert_id) {
  assert(padded_cols % 4 == 0);
  assert(unpadded_cols % 4 == 0);
  assert(unpadded_cols <= padded_cols);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  for (int64_t expanded_permuted_row = blockIdx.x; expanded_permuted_row < num_valid_tokens;
       expanded_permuted_row += gridDim.x) {
    int64_t unpermuted_row = permuted_row_to_unpermuted_row[expanded_permuted_row];

    // Duplicate and permute rows
    int64_t const source_k_rank = unpermuted_row / num_rows;
    int64_t const source_row = unpermuted_row % num_rows;

    // If the expert is the first selected (valid) one of the corresponding token on the current EP
    // rank, do reduction; otherwise, skip.
    bool is_first_selected_expert = true;
    for (int k_idx = 0; k_idx < source_k_rank; ++k_idx) {
      int const expert_id =
          token_selected_experts[source_row * experts_per_token + k_idx] - start_expert_id;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        is_first_selected_expert = false;
        break;
      }
    }
    if (!is_first_selected_expert) {
      continue;
    }

    OutputType* reduced_row_ptr = reduced_unpermuted_output + source_row * unpadded_cols;

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD =
        128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_padded_col = padded_cols / FINALIZE_ELEM_PER_THREAD;
    int64_t const num_elems_in_orig_col = unpadded_cols / FINALIZE_ELEM_PER_THREAD;

    using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
    using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
    auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
    auto const* expanded_permuted_rows_v =
        reinterpret_cast<InputElem const*>(expanded_permuted_rows);
    auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

    for (int elem_index = start_offset; elem_index < num_elems_in_padded_col;
         elem_index += stride) {
      if (elem_index >= num_elems_in_orig_col) continue;  // Skip writing beyond original columns

      ComputeElem thread_output;
      thread_output.fill(0);
      for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
        int64_t const k_offset = source_row * experts_per_token + k_idx;
        int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
        if (expert_id < 0 || expert_id >= num_experts_per_node) {
          continue;
        }

        int64_t const expanded_permuted_row_from_k_idx =
            unpermuted_row_to_permuted_row[source_row + k_idx * num_rows];

        float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

        auto const* expanded_permuted_rows_row_ptr =
            expanded_permuted_rows_v + expanded_permuted_row_from_k_idx * num_elems_in_padded_col;

        ComputeElem expert_result =
            arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);

        if (bias) {
          auto const* bias_ptr = bias_v + expert_id * num_elems_in_padded_col;
          expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
        }

        thread_output = thread_output + row_scale * expert_result;
      }
      OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
      reduced_row_ptr_v[elem_index] = output_elem;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(
    GemmOutputType const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* final_scales, int const* unpermuted_row_to_permuted_row,
    int const* permuted_row_to_unpermuted_row, int const* token_selected_experts,
    int64_t const* expert_first_token_offset, int64_t const num_rows, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token,
    int64_t const num_experts_per_node, MOEParallelismConfig parallelism_config,
    bool const enable_alltoall, bool enable_pdl, cudaStream_t stream) {
  // Only add bias on rank 0 for tensor parallelism
  bool const is_rank_0 = parallelism_config.tp_rank == 0;
  ScaleBiasType const* bias_ptr = is_rank_0 ? bias : nullptr;
  int const start_expert_id = num_experts_per_node * parallelism_config.ep_rank;

  cudaLaunchConfig_t config;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  if (parallelism_config.ep_size > 1 && enable_alltoall) {
    // If all-to-all comm is enabled, finalizeMoeRouting doesn't need to fill the invalid output
    // tokens with zeros.
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
    int64_t const blocks = smCount * 8;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;
    auto func = final_scales
                    ? &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::DEFAULT>
                    : &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::NO_SCALE>;
    cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr,
                       final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
                       token_selected_experts, expert_first_token_offset, num_rows, padded_cols,
                       unpadded_cols, experts_per_token, num_experts_per_node, start_expert_id);
  } else {
    // If all-gather reduce-scatter is used, finalizeMoeRouting must fill invalid output tokens with
    // zeros.
    int64_t const blocks = num_rows;
#if defined(FLASHINFER_FP4_SW_CVT)
    // Few rows: a row's columns split over more CTAs, threads halved to 64 while CTAs have an SM.
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int64_t const elem_per_thread =
        128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);
    int64_t const vecs = unpadded_cols / elem_per_thread;
    int64_t const sms_per_row = std::max<int64_t>(smCount / std::max<int64_t>(blocks, 1), 1);
    int64_t threads = FINALIZE_THREADS_PER_BLOCK;
    while (threads > 64 && tensorrt_llm::common::ceilDiv(vecs, threads / 2) <= sms_per_row) {
      threads /= 2;
    }
    int64_t const splits = std::min(tensorrt_llm::common::ceilDiv(vecs, threads), sms_per_row);
    config.gridDim = dim3(blocks, std::max<int64_t>(splits, 1));
    config.dynamicSmemBytes =
        tensorrt_llm::common::roundUp(std::min<int64_t>(experts_per_token, FINALIZE_CACHED_K),
                                      int64_t{FINALIZE_K_LOADS}) *
        (2 * sizeof(int) + sizeof(float));
#else
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
#endif
    config.blockDim = threads;
    auto func = final_scales ? &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::DEFAULT>
                             : &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::NO_SCALE>;
    cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr,
                       final_scales, unpermuted_row_to_permuted_row, token_selected_experts,
                       padded_cols, unpadded_cols, experts_per_token, num_experts_per_node,
                       start_expert_id);
  }
}

#define INSTANTIATE_FINALIZE_MOE_ROUTING(OutputT, GemmOutputT, ScaleBiasT)                  \
  template void finalizeMoeRoutingKernelLauncher<OutputT, GemmOutputT, ScaleBiasT>(         \
      GemmOutputT const* expanded_permuted_rows, OutputT* reduced_unpermuted_output,        \
      ScaleBiasT const* bias, float const* final_scales,                                    \
      int const* unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row, \
      int const* expert_for_source_row, int64_t const* expert_first_token_offset,           \
      int64_t const num_rows, int64_t const padded_cols, int64_t const actual_cols,         \
      int64_t const experts_per_token, int64_t const num_experts_per_node,                  \
      MOEParallelismConfig parallelism_config, bool const enable_alltoall, bool enable_pdl, \
      cudaStream_t stream);

// // Instantiate the data types that are used by the external pytorch op
// INSTANTIATE_FINALIZE_MOE_ROUTING(half, half, half);
// INSTANTIATE_FINALIZE_MOE_ROUTING(float, float, float);
// #ifdef ENABLE_BF16
// INSTANTIATE_FINALIZE_MOE_ROUTING(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
// #endif

// ============================== Activation Adaptors =================================
template <template <class> class ActFn>
struct IdentityAdaptor {
  constexpr static bool IS_GLU = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T const& x) const {
    ActFn<T> fn{};
    return fn(x);
  }
};

template <template <class> class ActFn>
struct GLUAdaptor {
  constexpr static bool IS_GLU = true;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    ActFn<T> fn{};
    return fn(gate) * linear;
  }
};

struct SwigluBiasAdaptor {
  constexpr static bool IS_GLU = true;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    cutlass::epilogue::thread::Sigmoid<T> fn{};
    T linear_clamped = cutlass::maximum<T>{}(cutlass::minimum<T>{}(linear, limit), -limit);
    T gate_clamped = cutlass::minimum<T>{}(gate, limit);
    return gate_clamped * fn(gate_clamped * alpha) * (linear_clamped + beta);
  }
};

// SwiGLU variant used by Step-3.5/3.7-Flash: out = min(silu(gate), limit) * clamp(up, -limit,
// limit). limit defaults to 7.0 (the Step-3.x model value) when no per-expert swiglu_limit is
// supplied.
struct SwigluStepAdaptor {
  constexpr static bool IS_GLU = true;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = 7.0f;

  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    cutlass::epilogue::thread::SiLu<T> fn{};
    T gate_clamped = cutlass::minimum<T>{}(fn(gate), limit);
    T linear_clamped = cutlass::maximum<T>{}(cutlass::minimum<T>{}(linear, limit), -limit);
    return gate_clamped * linear_clamped;
  }
};

// SiTU-GLU (Kimi-K3 / mistral ffn_activations.situ_glu). A gated activation that transforms both
// branches, evaluated in fp32 (bf16 rounding is visible at the tanh saturation points):
//   out = (beta * tanh(gate / beta) * sigmoid(gate)) * (linear_beta * tanh(up / linear_beta))
// Note the sigmoid reads the *uncapped* gate.
struct SituAdaptor {
  constexpr static bool IS_GLU = true;
  float beta = 4.0f;
  float linear_beta = 25.0f;

  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    cutlass::epilogue::thread::Sigmoid<T> sigmoid{};
    // tanh(z) == 2*sigmoid(2z) - 1. CUTLASS's Sigmoid uses ::expf, whereas its Tanh lowers to
    // tanh.approx.f32 whose 2^-11 absolute error linear_beta=25 would amplify to ~1e-2.
    // The `+ (-1.0f)` is because cutlass::Array has no operator-(Array, scalar).
    auto tanh = [&](T const& z) { return sigmoid(z * 2.0f) * 2.0f + (-1.0f); };
    return (tanh(gate * (1.0f / beta)) * sigmoid(gate) * beta) *
           (tanh(linear * (1.0f / linear_beta)) * linear_beta);
  }
};

// SiTU below SM100: a finite limit (TRT-LLM's gemm1_clamp_limit, as swiglu_limit) clamps up to
// [-limit, limit] and gate from above, NaN passing as in torch.clamp; linear_beta = inf leaves up
// unscaled (the limit of linear_beta * tanh(up / linear_beta)).
struct ClampedSituAdaptor : SituAdaptor {
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T gate, T linear) const {
    constexpr float inf = std::numeric_limits<float>::infinity();
    if (limit != inf) {
      gate = cutlass::minimum<T, true>{}(gate, limit);
      linear = cutlass::maximum<T, true>{}(cutlass::minimum<T, true>{}(linear, limit), -limit);
    }
    if (linear_beta != inf) {
      return SituAdaptor::operator()(gate, linear);
    }
    T up_scale;
    up_scale.fill(inf);
    return SituAdaptor{beta, 1.0f}(gate, up_scale) * linear;
  }
};

// PowLU-GLU. The gate branch replaces
// SiLU's linear factor by a self-adapting power, and the negative half is plain SiLU:
//   PowLU(g) = g > 0 ? g^(m / (sqrt(g) + 1)) * sigmoid(g)
//                    : g * sigmoid(g)
//   out      = min(PowLU(gate), limit) * clamp(up, -limit, limit)
// The two branches agree at g == 0 (both 0) and at g == 1, so PowLU is continuous.
//
// `alpha` carries the exponent numerator m. PowLU needs exactly one scalar besides `limit`, so it
// reuses the generic per-expert swiglu_alpha/swiglu_limit slots instead of widening the
// ActivationParams ABI -- the same trade SwigluStepAdaptor makes for `limit`. Per-expert
// `swiglu_limit` covers split limits (routed experts vs. a fused shared expert).
//
// PowLU is bounded: for m == 2.5 it peaks at 4.02 (g ~= 12.9) and decays back toward 1 as g grows,
// so a `limit` above that peak never clips the gate branch and only clamps `up`.
struct PowLUAdaptor {
  constexpr static bool IS_GLU = true;
  float alpha = 2.5f;  // m
  // Unused by the math; present so the generic setPerExpertActivationParams branch compiles.
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    // Element-wise rather than whole-Array: the gate needs a sign-dependent branch, which the
    // cutlass::Array functors cannot express. T is always cutlass::Array<float, N> here (both
    // dispatch sites call the adaptor with ComputeElem).
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) {
      float const g = gate[i];
      // __expf overflows to +inf for very negative g, giving sigmoid == 0 exactly (not NaN).
      float const sigmoid = 1.0f / (1.0f + __expf(-g));
      // __powf(g, e) is exp2f(e * __log2f(g)). The Triton reference computes exp(e * log(g));
      // the two agree to <= 6e-6 relative, far below bf16 resolution. Shared edge with that
      // reference: an overflowed gate (g == inf) gives e == 0 and 0 * log(inf) == NaN, which the
      // non-propagating fminf below turns into `limit` -- torch.pow(inf, 0) == 1 instead.
      float act = g > 0.0f ? __powf(g, alpha / (sqrtf(g) + 1.0f)) * sigmoid : g * sigmoid;
      act = fminf(act, limit);
      float const u = fmaxf(fminf(linear[i], limit), -limit);
      out[i] = act * u;
    }
    return out;
  }
};

__device__ inline bool hasPerExpertActivationParams(ActivationParams const& params) {
  // clamped_relu2_limit applies model-wide and is consumed only by the fused GEMM1 epilogue.
  return params.swiglu_alpha || params.swiglu_beta || params.swiglu_limit || params.situ_beta ||
         params.situ_linear_beta;
}

// Only assigns what the caller actually supplied, so each adaptor keeps its compile-time default
// (e.g. SwigluStepAdaptor::limit == 7.0, SituAdaptor::beta == 4.0).
template <class ActFn>
__device__ void setPerExpertActivationParams(ActFn& fn, ActivationParams const& params,
                                             int64_t expert) {
  if constexpr (std::is_base_of_v<SituAdaptor, ActFn>) {
    if (params.situ_beta) fn.beta = params.situ_beta[expert];
    if (params.situ_linear_beta) fn.linear_beta = params.situ_linear_beta[expert];
    if constexpr (!std::is_same_v<ActFn, SituAdaptor>) {
      if (params.swiglu_limit) fn.limit = params.swiglu_limit[expert];
    }
  } else {
    if (params.swiglu_alpha) fn.alpha = params.swiglu_alpha[expert];
    if (params.swiglu_beta) fn.beta = params.swiglu_beta[expert];
    if (params.swiglu_limit) fn.limit = params.swiglu_limit[expert];
  }
}

// ============================== Gated Activation =================================
constexpr static int MAX_ACTIVATION_THREADS_PER_BLOCK = 256;

template <class ActivationOutputType, class GemmOutputType, class ActFn>
__global__ void doGatedActivationKernel(ActivationOutputType* output,
                                        GemmOutputType const* gemm_result,
                                        int64_t const* expert_first_token_offset,
                                        int64_t inter_size, int64_t num_experts_per_node,
                                        ActivationParams activation_type) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  if (token >= expert_first_token_offset[num_experts_per_node]) {
    return;
  }

  output = output + token * inter_size;
  gemm_result = gemm_result + token * inter_size * 2;

  constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / sizeof_bits<ActivationOutputType>::value;

  using OutputElem = cutlass::Array<ActivationOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
  auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
  auto output_vec = reinterpret_cast<OutputElem*>(output);
  int64_t const start_offset = tid;
  int64_t const stride = MAX_ACTIVATION_THREADS_PER_BLOCK;
  assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
  int64_t const inter_size_vec = inter_size / ACTIVATION_ELEM_PER_THREAD;

  ActFn fn{};
  if (hasPerExpertActivationParams(activation_type)) {
    int64_t const expert = findTotalEltsLessThanTarget(expert_first_token_offset,
                                                       num_experts_per_node, (int64_t)token + 1) -
                           1;
    setPerExpertActivationParams(fn, activation_type, expert);
  }
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto linear_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
    // BF16 isn't supported, use FP32 for activation function
    auto gate_value =
        arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + inter_size_vec]);
    auto gate_act = fn(gate_value, linear_value);
    output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(gate_act);
  }
}

template <typename ActivationOutputType, typename GemmOutputType>
void doGatedActivation(ActivationOutputType* output, GemmOutputType const* gemm_result,
                       int64_t const* expert_first_token_offset, int64_t inter_size,
                       int64_t num_tokens, int64_t num_experts_per_node,
                       ActivationParams activation_type, cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = MAX_ACTIVATION_THREADS_PER_BLOCK;

  auto* fn = (activation_type == ActivationType::Swiglu)
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType,
                                            GLUAdaptor<cutlass::epilogue::thread::SiLu>>
             : activation_type == ActivationType::Geglu
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType,
                                            GLUAdaptor<cutlass::epilogue::thread::GELU>>
             : activation_type == ActivationType::GegluTanh
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType,
                                            GLUAdaptor<cutlass::epilogue::thread::GELU_taylor>>
             : activation_type == ActivationType::SwigluBias
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SwigluBiasAdaptor>
             : activation_type == ActivationType::SwigluStep
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SwigluStepAdaptor>
             : activation_type == ActivationType::Situ
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SituAdaptor>
             : activation_type == ActivationType::PowLU
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, PowLUAdaptor>
                 : nullptr;
  TLLM_CHECK_WITH_INFO(fn != nullptr, "Invalid activation type");
  fn<<<blocks, threads, 0, stream>>>(output, gemm_result, expert_first_token_offset, inter_size,
                                     num_experts_per_node, activation_type);
}

// ============================== Activation =================================

template <class T, class GemmOutputType, class ScaleBiasType, class ActFn,
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType BlockScalingType,
          bool DISABLE_FP4_QUANT_FAST_MATH = false, typename NVFP4_4OVER6_CONFIG = std::false_type>
__global__ __launch_bounds__(MAX_ACTIVATION_THREADS_PER_BLOCK) void doActivationKernel(
    T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
    ScaleBiasType const* bias_ptr, bool bias_is_broadcast, int64_t const* expert_first_token_offset,
    int const* permuted_token_selected_experts, int num_experts_per_node, int64_t inter_size,
    float const* fc2_act_global_scale, bool use_per_expert_act_scale,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat, float* fp8_token_dequant_scale,
    float const* fp8_expert_residual_scale, ActivationParams activation_params) {
#ifdef ENABLE_FP4
  // FP16/BF16 output of the NVFP4 values restored (code x scale), K padded, for 16-bit GEMMs.
  constexpr bool IsRestoredNVFP4 =
      (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) &&
      std::is_same_v<GemmOutputType, T> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4;
  // And MXFP8 the same way, to BF16.
  constexpr bool IsRestoredMXFP8 =
      std::is_same_v<T, __nv_bfloat16> && std::is_same_v<GemmOutputType, T> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
  constexpr bool IsNVFP4 =
      (std::is_same_v<T, Fp4Type> || IsRestoredNVFP4) &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4;
  constexpr bool IsMXFP8 =
      (std::is_same_v<T, __nv_fp8_e4m3> || IsRestoredMXFP8) &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
#else
  constexpr bool IsRestoredNVFP4 = false;
  constexpr bool IsRestoredMXFP8 = false;
  constexpr bool IsNVFP4 = cute::dependent_false<T>;
  constexpr bool IsMXFP8 = cute::dependent_false<T>;
#endif

  int64_t const tid = threadIdx.x;
  constexpr bool IsGated = ActFn::IS_GLU;
  size_t gated_size_mul = IsGated ? 2 : 1;
  size_t gated_off = IsGated ? inter_size : 0;

  constexpr int64_t VecSize = IsNVFP4
                                  ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                  : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;
  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t ACTIVATION_ELEM_PER_THREAD =
      (IsNVFP4 || IsMXFP8)
          ? CVT_ELTS_PER_THREAD
          : (128 / std::min(sizeof_bits<T>::value, sizeof_bits<GemmOutputType>::value));
  static_assert(!IsNVFP44Over6Config<NVFP4_4OVER6_CONFIG>::value || IsNVFP4,
                "NVFP4 4over6 requires NVFP4 block scaling");

  // This should be VecSize * 4 elements
  // We assume at least VecSize alignment or the quantization will fail
  int64_t const min_k_dim_alignment =
      IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
              : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;
  int64_t const padded_inter_size = ceilDiv(inter_size, min_k_dim_alignment) * min_k_dim_alignment;

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t token = blockIdx.x; token < num_valid_tokens; token += gridDim.x) {
    size_t gemm_result_offset = token * inter_size * gated_size_mul;
    size_t output_offset =
        token * (IsRestoredNVFP4 || IsRestoredMXFP8 ? padded_inter_size : inter_size);

    int64_t expert = 0;
    if (bias_ptr || IsNVFP4 || IsMXFP8 || use_per_expert_act_scale ||
        hasPerExpertActivationParams(activation_params)) {
      expert = permuted_token_selected_experts
                   ? permuted_token_selected_experts[token]
                   : findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node,
                                                 token + 1) -
                         1;
    }

    size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
    float const quant_scale = fp8_quant ? fp8_quant[act_scale_idx] : 1.f;

    // Some globals for FP4
    float global_scale_val = fc2_act_global_scale ? fc2_act_global_scale[act_scale_idx] : 1.0f;
    int64_t num_tokens_before_expert = (IsNVFP4 || IsMXFP8) ? expert_first_token_offset[expert] : 0;

    size_t bias_offset = 0;
    if (bias_ptr) {
      bias_offset = (bias_is_broadcast ? expert * inter_size * gated_size_mul : gemm_result_offset);
    }

    using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using OutputElem =
        std::conditional_t<(IsNVFP4 && !IsRestoredNVFP4), uint32_t,
                           std::conditional_t<(IsMXFP8 && !IsRestoredMXFP8), uint64_t,
                                              cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>>>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    // Aliases gemm_result for non-gated, non-fp8 cases
    auto gemm_result_vec =
        reinterpret_cast<GemmResultElem const*>(gemm_result + gemm_result_offset);
    auto output_vec = reinterpret_cast<OutputElem*>(safe_inc_ptr(output, output_offset));
    auto bias_ptr_vec = reinterpret_cast<BiasElem const*>(bias_ptr + bias_offset);
    assert(reinterpret_cast<std::uintptr_t>(gemm_result_vec) % sizeof(GemmResultElem) == 0);
    assert(reinterpret_cast<std::uintptr_t>(output_vec) % sizeof(OutputElem) == 0);
    assert(reinterpret_cast<std::uintptr_t>(bias_ptr_vec) % sizeof(BiasElem) == 0);

#if defined(FLASHINFER_FP4_SW_CVT)
    int64_t const start_offset = tid + blockIdx.y * blockDim.x;
    int64_t const stride = blockDim.x * gridDim.y;
#else
    int64_t const start_offset = tid;
    int64_t const stride = blockDim.x;
#endif
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

    ActFn fn{};
    setPerExpertActivationParams(fn, activation_params, expert);
    auto compute_activation = [&](int64_t elem_index) {
      GemmResultElem fc1_gemm_value;
      cutlass::arch::global_load<GemmResultElem, sizeof(GemmResultElem)>(
          fc1_gemm_value, gemm_result_vec + elem_index + gated_off_vec, true);
      auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(fc1_gemm_value);
      if (bias_ptr) {
        BiasElem fc1_bias_value;
        cutlass::arch::global_load<BiasElem, sizeof(BiasElem)>(
            fc1_bias_value, bias_ptr_vec + elem_index + gated_off_vec, true);
        fc1_value = fc1_value + arrayConvert<BiasElem, ComputeElem>(fc1_bias_value);
      }

      auto gate_act = [&]() {
        if constexpr (IsGated) {
          GemmResultElem linear_gemm_value;
          cutlass::arch::global_load<GemmResultElem, sizeof(GemmResultElem)>(
              linear_gemm_value, gemm_result_vec + elem_index, true);
          auto linear_value = arrayConvert<GemmResultElem, ComputeElem>(linear_gemm_value);
          if (bias_ptr) {
            BiasElem linear_bias_value;
            cutlass::arch::global_load<BiasElem, sizeof(BiasElem)>(linear_bias_value,
                                                                   bias_ptr_vec + elem_index, true);
            linear_value = linear_value + arrayConvert<BiasElem, ComputeElem>(linear_bias_value);
          }
          return fn(fc1_value, linear_value);
        } else {
          return fn(fc1_value);
        }
      }();
      return gate_act;
    };

    bool const use_per_token_fp8_quant =
        std::is_same_v<T, __nv_fp8_e4m3> &&
        BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE &&
        fp8_token_dequant_scale != nullptr;

    float token_quant_scale = quant_scale;
    if (use_per_token_fp8_quant) {
      float thread_amax = 0.0f;
      for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto gate_act = compute_activation(elem_index);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ComputeElem::kElements; ++i) {
          thread_amax = fmaxf(thread_amax, fabsf(gate_act[i]));
        }
      }

      using BlockReduce = cub::BlockReduce<float, MAX_ACTIVATION_THREADS_PER_BLOCK>;
      __shared__ typename BlockReduce::TempStorage reduce_storage;
      __shared__ float shared_token_quant_scale;
      float const row_amax = BlockReduce(reduce_storage).Reduce(thread_amax, FloatMaxOp{});
      if (tid == 0) {
        float const quant = computeSafeFP8QuantScale(row_amax);
        float residual = 1.0f;
        // The GEMM1 profiler passes no expert map and does not consume the generated FC2 scale.
        if (fp8_expert_residual_scale && permuted_token_selected_experts) {
          int const residual_expert = permuted_token_selected_experts[token];
          residual = fp8_expert_residual_scale[residual_expert];
        }
        shared_token_quant_scale = quant;
        fp8_token_dequant_scale[token] = (1.0f / quant) * residual;
      }
      __syncthreads();
      token_quant_scale = shared_token_quant_scale;
    }

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      auto post_act_val = compute_activation(elem_index) * token_quant_scale;

      if constexpr (IsNVFP4 || IsMXFP8) {
        // We use GemmOutputType as the intermediate compute type as that should always be
        // unquantized
        TmaWarpSpecializedGroupedGemmInput::ElementSF sf = 0;
        auto res = quantizePackedFPXValue<
            GemmOutputType, std::conditional_t<IsRestoredMXFP8, __nv_fp8_e4m3, T>, ComputeElem,
            VecSize, DISABLE_FP4_QUANT_FAST_MATH, NVFP4_4OVER6_CONFIG, IsRestoredMXFP8>(
            post_act_val, global_scale_val, num_tokens_before_expert, expert, token, elem_index,
            inter_size, fc2_act_sf_flat,
            IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                    : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX,
            IsRestoredNVFP4 ? &sf : nullptr);
        OutputElem out_vec;
        if constexpr (IsRestoredNVFP4) {
          restoreNvfp4x8<T>(res, sf, reinterpret_cast<uint32_t*>(&out_vec));
        } else {
          static_assert(sizeof(res) == sizeof(*output_vec),
                        "Quantized value must be the same size as the output");
          out_vec = cutlass::platform::bit_cast<OutputElem>(res);
        }
        cutlass::arch::global_store<OutputElem, sizeof(OutputElem)>(out_vec,
                                                                    output_vec + elem_index, true);
      } else {
        auto out_vec = arrayConvert<ComputeElem, OutputElem>(post_act_val);
        cutlass::arch::global_store<OutputElem, sizeof(OutputElem)>(out_vec,
                                                                    output_vec + elem_index, true);
      }
    }

    // Pad zeros in the extra SFs along the K dimension, we do this to ensure there are no nan
    // values in the padded SF atom
    if constexpr (IsNVFP4 || IsMXFP8) {
      // Use VecSize per thread since we are just writing out zeros so every thread can process a
      // whole vector
      size_t padding_start_offset = inter_size / VecSize + start_offset;
      size_t padding_elems_in_col = padded_inter_size / VecSize;
      for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col;
           elem_index += stride) {
        writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1, token,
                                  elem_index, padded_inter_size, fc2_act_sf_flat,
                                  /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
        if constexpr (IsRestoredNVFP4) {
          auto* padding = reinterpret_cast<uint4*>(output + output_offset) + 2 * elem_index;
          padding[0] = padding[1] = uint4{};
        }
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // N-dim SF padding (zeroing extra token rows beyond tokens_to_expert up to MinNDimAlignment)
  // is intentionally omitted. The CUTLASS grouped GEMM sets gemm_m = tokens_to_expert per expert
  // and never reads scale factors for rows beyond that. The N-dim padding rows don't correspond
  // to any valid MMA tiles, so their content doesn't affect correctness.
  // K-dim SF padding (above, inside the per-token loop) is still required because MMA tiles may
  // straddle the inter_size boundary within valid rows.
}

template <class T, class GemmOutputType, class ScaleBiasType, class SituFn = SituAdaptor>
void doActivation(T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
                  ScaleBiasType const* bias, bool bias_is_broadcast,
                  int64_t const* expert_first_token_offset,
                  int const* permuted_token_selected_experts, int num_experts_per_node,
                  int64_t inter_size, int64_t expanded_num_tokens, ActivationParams activation_type,
                  QuantParams const& quant_params, bool use_per_expert_act_scale,
                  TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat,
                  float* fp8_token_dequant_scale, float const* fp8_expert_residual_scale,
                  bool enable_pdl, cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(
      (inter_size * std::min(sizeof_bits<T>::value, sizeof_bits<GemmOutputType>::value)) % 128 == 0,
      "inter_size %ld rows are not a multiple of 16B; doActivationKernel requires 16B-aligned rows",
      inter_size);
  TLLM_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(gemm_result) % 16 == 0 &&
                           reinterpret_cast<uintptr_t>(output) % 16 == 0 &&
                           reinterpret_cast<uintptr_t>(bias) % 16 == 0,
                       "doActivationKernel input/output/bias pointers must be 16B-aligned");

  // N-dim SF padding has been removed (CUTLASS grouped GEMM never reads beyond
  // tokens_to_expert), so the grid is driven purely by the expanded token count.

  // Select block size (up to MAX_ACTIVATION_THREADS_PER_BLOCK) per the threads
  // needed to process each row to avoid inactive warps and improve occupancy.
#ifdef ENABLE_FP4
  bool const use_mxfp8_block_scaling = quant_params.mxfp8_mxfp4.fc2.weight_block_scale ||
                                       quant_params.mxfp8_mxfp8.fc2.weight_block_scale;
  bool const use_fpx_block_scaling =
      std::is_same_v<T, Fp4Type> || (std::is_same_v<T, __nv_fp8_e4m3> && use_mxfp8_block_scaling);
#else
  constexpr bool use_fpx_block_scaling = false;
#endif
  int64_t const elem_per_thread =
      use_fpx_block_scaling
          ? CVT_ELTS_PER_THREAD
          : (128 / std::min(sizeof_bits<T>::value, sizeof_bits<GemmOutputType>::value));
  int64_t const loads_per_row = std::max<int64_t>(inter_size / elem_per_thread, int64_t{1});
  // The per-token FP8 quant path reduces the row amax with
  // cub::BlockReduce<float, MAX_ACTIVATION_THREADS_PER_BLOCK>, which requires
  // blockDim.x to match the compile-time block size, so force a full block there.
  int64_t const threads =
      fp8_token_dequant_scale != nullptr
          ? MAX_ACTIVATION_THREADS_PER_BLOCK
          : std::min<int64_t>(MAX_ACTIVATION_THREADS_PER_BLOCK,
                              tensorrt_llm::common::roundUp(loads_per_row, int64_t{WARP_SIZE}));

  // Limit the grid size to be proportional to the SM count in order to mitigate
  // the warp scheduling pressure on each SM. The proportion can range from 8 to
  // 32 blocks per SM and is adaptive to the block size.
  static int64_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
  int64_t const blocks_per_sm = std::min<int64_t>(
      int64_t{32},
      tensorrt_llm::common::ceilDiv(int64_t{8} * MAX_ACTIVATION_THREADS_PER_BLOCK, threads));
  int64_t const blocks =
      std::min(smCount * blocks_per_sm, std::max(expanded_num_tokens, int64_t{1}));
  int64_t splits = 1;
#if defined(FLASHINFER_FP4_SW_CVT)
  // Fewer rows: a row's columns split over more CTAs, but not per-token FP8 quant (a row per CTA).
  if (fp8_token_dequant_scale == nullptr) {
    splits = std::min(ceilDiv(loads_per_row, threads), smCount * blocks_per_sm / blocks);
  }
#endif

  auto fn = [&]() {
    auto fn = [&](auto block_scaling_type, auto disableFP4QuantFastMathTag,
                  auto nvfp4_4over6_config_tag) {
      auto fn_list = std::array{
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, IdentityAdaptor<cutlass::epilogue::thread::GELU>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Gelu
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, IdentityAdaptor<cutlass::epilogue::thread::ReLu>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Relu
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, IdentityAdaptor<cutlass::epilogue::thread::SiLu>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Silu
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, GLUAdaptor<cutlass::epilogue::thread::SiLu>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Swiglu
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, GLUAdaptor<cutlass::epilogue::thread::GELU>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Geglu
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, SwigluBiasAdaptor,
                              decltype(block_scaling_type)::value,
                              decltype(disableFP4QuantFastMathTag)::value,
                              decltype(nvfp4_4over6_config_tag)>,  // SwigluBias
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, IdentityAdaptor<cutlass::epilogue::thread::Relu2>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Relu2
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, SwigluStepAdaptor,
                              decltype(block_scaling_type)::value,
                              decltype(disableFP4QuantFastMathTag)::value,
                              decltype(nvfp4_4over6_config_tag)>,  // SwigluStep
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, GLUAdaptor<cutlass::epilogue::thread::GELU_taylor>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // GegluTanh
          &doActivationKernel<T, GemmOutputType, ScaleBiasType,
                              IdentityAdaptor<cutlass::epilogue::thread::Identity>,
                              decltype(block_scaling_type)::value,
                              decltype(disableFP4QuantFastMathTag)::value,
                              decltype(nvfp4_4over6_config_tag)>,  // Identity
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, SituFn,
                              decltype(block_scaling_type)::value,
                              decltype(disableFP4QuantFastMathTag)::value,
                              decltype(nvfp4_4over6_config_tag)>  // Situ
      };
      // PowLU is CUTLASS-only and numbered after InvalidType (like ClampedRelu2), so it is
      // outside the dense enum-indexed range of fn_list and dispatched here instead.
      if (activation_type.activation_type == ActivationType::PowLU) {
        return &doActivationKernel<
            T, GemmOutputType, ScaleBiasType, PowLUAdaptor, decltype(block_scaling_type)::value,
            decltype(disableFP4QuantFastMathTag)::value, decltype(nvfp4_4over6_config_tag)>;
      }
      auto const activation_index = static_cast<size_t>(activation_type.activation_type);
      TLLM_CHECK_WITH_INFO(activation_index < fn_list.size(),
                           "Unsupported activation type in doActivation");
      return fn_list[activation_index];
    };
#ifdef ENABLE_FP4
    auto NVFP4 = tensorrt_llm::common::ConstExprWrapper<
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4>{};
    auto MXFPX = tensorrt_llm::common::ConstExprWrapper<
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX>{};
#endif
    auto NONE = tensorrt_llm::common::ConstExprWrapper<
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE>{};
#ifdef ENABLE_FP4
    if constexpr (std::is_same_v<T, Fp4Type>) {
      TLLM_CHECK_WITH_INFO(quant_params.fp4.fc2.weight_block_scale,
                           "NVFP4 block scaling is expected for FP4xFP4");
      return dispatchNVFP44Over6Config(
          quant_params.fp4.nvfp4_4over6_code,
          [&](auto disableFP4QuantFastMathTag, auto nvfp4_4over6_config_tag) {
            return fn(NVFP4, disableFP4QuantFastMathTag, nvfp4_4over6_config_tag);
          });
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      return use_mxfp8_block_scaling ? fn(MXFPX, std::false_type{}, std::false_type{})
                                     : fn(NONE, std::false_type{}, std::false_type{});
    } else
#endif
    {
#if defined(ENABLE_FP4) && defined(FLASHINFER_FP4_SW_CVT)
      // Below SM100, NVFP4 activations for FC2 are written restored to FP16/BF16.
      if constexpr ((std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) &&
                    std::is_same_v<GemmOutputType, T>) {
        if (quant_params.fp4.fc2.weight_block_scale && fc2_act_sf_flat) {
          if (!quant_params.fp4.fc2.act_global_scale) {  // MXFP8 has no global scale
            TLLM_CHECK_WITH_INFO((std::is_same_v<T, __nv_bfloat16>),
                                 "MXFP8 FC2 activations are restored to BF16");
            return fn(MXFPX, std::false_type{}, std::false_type{});
          }
          return dispatchNVFP44Over6Config(
              quant_params.fp4.nvfp4_4over6_code,
              [&](auto disableFP4QuantFastMathTag, auto nvfp4_4over6_config_tag) {
                return fn(NVFP4, disableFP4QuantFastMathTag, nvfp4_4over6_config_tag);
              });
        }
      }
#endif
      return fn(NONE, std::false_type{}, std::false_type{});
    }
  }();

  cudaLaunchConfig_t config;
  config.gridDim = dim3(blocks, splits);
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, fn, output, gemm_result, fp8_quant, bias, bias_is_broadcast,
                     expert_first_token_offset, permuted_token_selected_experts,
                     num_experts_per_node, inter_size, quant_params.fp4.fc2.act_global_scale,
                     use_per_expert_act_scale, fc2_act_sf_flat, fp8_token_dequant_scale,
                     fp8_expert_residual_scale, activation_type);
}

// ============================== Lora Add Bias =================================
constexpr static int LORA_KERNELS_THREADS_PER_BLOCK = 256;

template <class ScaleBiasType, class LoraType, bool IsGated>
__global__ void loraAddBiasKernel(ScaleBiasType* output, LoraType const* lora_result,
                                  ScaleBiasType const* bias, int64_t const* num_valid_tokens_ptr,
                                  int* permuted_token_selected_experts, int64_t inter_size) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  int64_t const num_tokens = gridDim.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  LoraType const* lora_result_1 = lora_result + token * inter_size;
  int expert_id = permuted_token_selected_experts[token];
  if constexpr (IsGated) {
    output = output + token * inter_size * 2;
    bias = bias + expert_id * inter_size * 2;
  } else {
    output = output + token * inter_size;
    bias = bias + expert_id * inter_size;
  }

  constexpr int64_t LORA_ADD_BIAS_ELEM_PER_THREAD = 128 / sizeof_bits<LoraType>::value;

  using DataElem = cutlass::Array<LoraType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
  using BiasElem = cutlass::Array<ScaleBiasType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
  auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
  auto bias_vec = reinterpret_cast<BiasElem const*>(bias);
  auto output_vec = reinterpret_cast<BiasElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % LORA_ADD_BIAS_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_1_vec[elem_index];
    auto bias_value = bias_vec[elem_index];
    output_vec[elem_index] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
  }

  if constexpr (IsGated) {
    auto lora_result_2_vec =
        reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
    int64_t const inter_size_vec = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      auto lora_value = lora_result_2_vec[elem_index];
      auto bias_value = bias_vec[elem_index + inter_size_vec];
      output_vec[elem_index + inter_size_vec] =
          bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
    }
  }
}

template <class ScaleBiasType, class LoraType>
void loraAddBias(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
                 int64_t const* num_valid_tokens_ptr, int64_t inter_size,
                 int* permuted_token_selected_experts, int64_t num_tokens, bool is_gated_activation,
                 cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

  auto selected_fn = is_gated_activation ? loraAddBiasKernel<ScaleBiasType, LoraType, true>
                                         : loraAddBiasKernel<ScaleBiasType, LoraType, false>;
  selected_fn<<<blocks, threads, 0, stream>>>(output, lora_result, bias, num_valid_tokens_ptr,
                                              permuted_token_selected_experts, inter_size);
}

template <class T>
__global__ void loraReorderKernel(T* output, T const* lora_result,
                                  int64_t const* num_valid_tokens_ptr, int64_t inter_size) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  int64_t const num_tokens = gridDim.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  T const* lora_result_1 = lora_result + token * inter_size;
  output = output + token * inter_size * 2;

  constexpr int64_t LORA_REORDER_ELEM_PER_THREAD = 128 / sizeof_bits<T>::value;

  using DataElem = cutlass::Array<T, LORA_REORDER_ELEM_PER_THREAD>;
  auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
  auto output_vec = reinterpret_cast<DataElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % LORA_REORDER_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / LORA_REORDER_ELEM_PER_THREAD;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_1_vec[elem_index];
    output_vec[elem_index] = lora_value;
  }

  auto lora_result_2_vec =
      reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
  int64_t const inter_size_vec = inter_size / LORA_REORDER_ELEM_PER_THREAD;
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_2_vec[elem_index];
    output_vec[elem_index + inter_size_vec] = lora_value;
  }
}

template <class T>
void loraReorder(T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr,
                 int64_t inter_size, int64_t num_tokens, cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

  loraReorderKernel<T>
      <<<blocks, threads, 0, stream>>>(output, lora_result, num_valid_tokens_ptr, inter_size);
}

// ============================== DEQUANT_FP8 =================================
constexpr static int DEQUANT_KERNELS_THREADS_PER_BLOCK = 256;

template <class OutputType, class InputType>
__global__ void dequantFP8Kernel(OutputType* output, InputType const* input,
                                 int64_t const* num_valid_tokens_ptr, int64_t inter_size,
                                 float const* scale, bool scale_is_dequant) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  output = output + token * inter_size;
  input = input + token * inter_size;

  constexpr int64_t DEQUANT_ELEM_PER_THREAD = 128 / sizeof_bits<InputType>::value;

  using DataElem = cutlass::Array<InputType, DEQUANT_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, DEQUANT_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, DEQUANT_ELEM_PER_THREAD>;
  auto input_vec = reinterpret_cast<DataElem const*>(input);
  auto output_vec = reinterpret_cast<OutputElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = DEQUANT_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % DEQUANT_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / DEQUANT_ELEM_PER_THREAD;

  ComputeElem deqaunt_scale_value;
  float dequant_scale = scale[0];
  if (!scale_is_dequant) {
    dequant_scale = 1.f / dequant_scale;
  }
  deqaunt_scale_value.fill(dequant_scale);

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto input_value = arrayConvert<DataElem, ComputeElem>(input_vec[elem_index]);
    output_vec[elem_index] =
        arrayConvert<ComputeElem, OutputElem>(input_value * deqaunt_scale_value);
  }
}

template <class OutputType, class InputType>
void dequantFP8(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr,
                int64_t inter_size, int64_t num_tokens, float const* scale, bool scale_is_dequant,
                cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = DEQUANT_KERNELS_THREADS_PER_BLOCK;

  dequantFP8Kernel<OutputType, InputType><<<blocks, threads, 0, stream>>>(
      output, input, num_valid_tokens_ptr, inter_size, scale, scale_is_dequant);
}

template <class T, class WeightType, class OutputType>
std::unique_ptr<kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunnerInterface>
makeDeepSeekBlockScaleGemmRunnerIfSupported() {
  // This runner is only for the DeepSeek BF16 activation x FP8 weight
  // block-scale path.  Mixed W4/MXFP4 paths still instantiate CutlassMoeFCRunner
  // but should not construct a DeepSeek block-scale GEMM runner.
  if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<WeightType, __nv_fp8_e4m3> &&
                std::is_same_v<OutputType, __nv_bfloat16>) {
    return std::make_unique<kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<
        __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>();
  } else {
    return nullptr;
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
                   Enable>::CutlassMoeFCRunner()
    : blockscale_gemm_runner_(
          makeDeepSeekBlockScaleGemmRunnerIfSupported<T, WeightType, OutputType>()) {}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
std::map<std::string, std::pair<size_t, size_t>>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
                   Enable>::getWorkspaceDeviceBufferSizes(int64_t const num_rows,
                                                          int64_t const hidden_size,
                                                          int64_t const inter_size,
                                                          int const num_experts_per_node,
                                                          int const experts_per_token,
                                                          ActivationType activation_type,
                                                          bool use_lora,
                                                          bool use_deepseek_fp8_block_scale,
                                                          bool use_mxfp8_act_scaling,
                                                          bool min_latency_mode, bool use_awq) {
  size_t num_moe_inputs =
      min_latency_mode ? num_experts_per_node * num_rows : experts_per_token * num_rows;
  // Below SM100 the restored activations are padded to the NVFP4 K alignment.
  int64_t const k_hidden = use_restored_fp4 ? padNvfp4K(hidden_size) : hidden_size;
  int64_t const k_inter = use_restored_fp4 ? padNvfp4K(inter_size) : inter_size;
  size_t const permuted_elems = num_moe_inputs * k_hidden;
  size_t const interbuf_elems = num_moe_inputs * k_inter;
  size_t glu_inter_elems = 0;
  bool is_gated_activation = isGatedActivation(activation_type);
  if (is_gated_activation) {
    glu_inter_elems = interbuf_elems * 2;
  } else if (mayHaveDifferentGEMMOutputType()) {
    // In this case we are using activation quantization, and some intermediate buffers will be
    // unquantized We need to have separate memory for these as we can no longer alias the output
    // buffer for reuse
    glu_inter_elems = interbuf_elems;
  }

  bool using_tma_ws = moe_gemm_runner_.supportsTmaWarpSpecialized();

  size_t const gemm_output_dtype = sizeof(UnfusedGemmOutputType);

  constexpr float dtype_size =
      (act_fp4 && !use_restored_fp4) ? 0.5f : ((use_w4afp8 || use_restored_fp4) ? 2.0f : sizeof(T));

  size_t const permuted_row_to_unpermuted_row_size =
      min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
  size_t const permuted_token_selected_experts_size =
      min_latency_mode ? 0 : num_moe_inputs * sizeof(int);

  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_rows, num_experts_per_node);
  int64_t const num_blocks_per_seq = tensorrt_llm::common::ceilDiv(num_rows, num_tokens_per_block);
  size_t const blocked_expert_counts_size =
      min_latency_mode ? 0 : num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size =
      min_latency_mode ? 0 : num_experts_per_node * num_rows * sizeof(int);

  size_t const permuted_data_size = permuted_elems * dtype_size;
  size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
  size_t const permuted_token_final_scales_size =
      mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
  size_t const glu_inter_size =
      glu_inter_elems * gemm_output_dtype;  // May be an intermediate type for quantization
  size_t const fc1_result_size =
      interbuf_elems * dtype_size;  // Activation quantizes so back to dtype_size
  size_t const fc2_result_size =
      min_latency_mode ? 0
                       : num_moe_inputs * k_hidden *
                             gemm_output_dtype;  // May be an intermediate type for quantization

  // If topk is greater than num_experts_per_node (i.e. large EP value), then we don't need to
  // allocate for the whole tokens*topk
  auto act_sf_rows =
      min_latency_mode
          ? num_moe_inputs
          : std::min(num_moe_inputs, static_cast<size_t>(num_rows * num_experts_per_node));
#if defined(FLASHINFER_FP4_SW_CVT)
  // Below SM100 the one-CTA sort keeps every expanded row, even an expert a token picks twice
  // (the three-step sort keeps it once).
  if (!min_latency_mode && num_moe_inputs <= SMALL_PROLOGUE_THREADS) {
    act_sf_rows = num_moe_inputs;
  }
#endif
  size_t const sf_size =
      getScalingType() == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX
          ? sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF)
          : sizeof(TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF);

  size_t const fc1_fp4_act_scale_size =
      getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size, getScalingType()) *
      sf_size;
  size_t const fc2_fp4_act_scale_size =
      getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size, getScalingType()) *
      sf_size;
  size_t const fp4_act_scale_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);
  // A fused MXFP8 activation epilogue reads FC1's input scales while writing
  // FC2's input scales, so those buffers cannot alias as they do when a
  // separate activation kernel runs after GEMM1.
  bool const may_fuse_mxfp8_clamped_relu2 = use_mxfp8_act_scaling &&
                                            activation_type == ActivationType::ClampedRelu2 &&
                                            !use_lora && !min_latency_mode;
  size_t const fused_fc2_fp4_act_scale_size =
      may_fuse_mxfp8_clamped_relu2 ? fc2_fp4_act_scale_size : 0;

  size_t const tma_ws_size = using_tma_ws ? TmaWarpSpecializedGroupedGemmInput::workspaceSize(
                                                num_experts_per_node, getScalingType())
                                          : 0;

  size_t const gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);
  size_t const precomputed_scheduler_workspace_size =
      (using_tma_ws && use_sm90_mixed_input_gemm)
          ? cutlass_kernels_oss::detail::precomputed_scheduler_workspace_size(
                num_experts_per_node, num_moe_inputs,
                std::max(hidden_size, is_gated_activation ? inter_size * 2 : inter_size))
          : 0;

  // lora related
  size_t const lora_input_size =
      (use_lora && use_fp8) ? std::max(permuted_elems, interbuf_elems) * sizeof(ScaleBiasType) : 0;
  size_t const lora_fc1_result_size =
      use_lora ? (is_gated_activation ? 2 * interbuf_elems * sizeof(ScaleBiasType)
                                      : interbuf_elems * sizeof(ScaleBiasType))
               : 0;
  size_t const lora_add_bias_size = use_lora ? lora_fc1_result_size : 0;
  size_t const lora_fc2_result_size = use_lora ? permuted_elems * sizeof(ScaleBiasType) : 0;

  // We do some overlapping of the large workspace buffers. Although we could overlap some of the
  // other buffers, they are small enough (i.e no factor of hidden size) they will only be a couple
  // MiB at most, so we don't bother in the case of fused activation we overlap permuted_data and
  // fc2_result in the case of unfused activation we overlap permuted_data and fc1_result we need to
  // calculate the max possible size, so use the max of all three
  size_t overlapped_gemm1_gemm2_inputs_size = std::max(permuted_data_size, fc2_result_size);
  // When glu_inter_elems is 0 we are always fused, otherwise we may need the un-fused case
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_inputs_size =
        std::max(overlapped_gemm1_gemm2_inputs_size, fc1_result_size);
  }

  size_t const alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float*);

  // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
  size_t overlapped_gemm1_gemm2_outputs_size = fc1_result_size;
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_outputs_size =
        std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs_size);
  }

  size_t smoothed_act_size = use_awq
                                 ? std::max(permuted_elems, interbuf_elems) * sizeof(T) * 2
                                 : 0;  // Extra workspace required by AWQ for smoothing activations
  size_t deepseek_fc_workspace_size = 0;
  if (use_deepseek_fp8_block_scale) {
    size_t factor = is_gated_activation ? 2 : 1;
    size_t blockscale_fc1_output_size = factor * interbuf_elems * gemm_output_dtype;
    size_t blockscale_fc2_output_size = permuted_elems * gemm_output_dtype;
    overlapped_gemm1_gemm2_inputs_size =
        std::max(std::max(permuted_data_size, fc1_result_size), blockscale_fc2_output_size);
    overlapped_gemm1_gemm2_outputs_size = blockscale_fc1_output_size;

    auto* blockscale_gemm_runner = getDeepSeekBlockScaleGemmRunner();
    TLLM_CHECK(blockscale_gemm_runner != nullptr);
    auto deepseek_fc1_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
        num_rows, factor * inter_size, hidden_size, experts_per_token, num_experts_per_node);
    auto deepseek_fc2_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
        num_rows, hidden_size, inter_size, experts_per_token, num_experts_per_node);
    deepseek_fc_workspace_size = std::max(deepseek_fc1_workspace_size, deepseek_fc2_workspace_size);
  }

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                                                        \
  do {                                                                              \
    auto aligned_size =                                                             \
        tensorrt_llm::common::alignSize(size, tensorrt_llm::common::kCudaMemAlign); \
    out_map[#name] = std::pair{aligned_size, map_offset};                           \
    map_offset += aligned_size;                                                     \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(permuted_row_to_unpermuted_row);
  ADD(permuted_token_selected_experts);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(expert_first_token_offset);
  ADD(permuted_token_final_scales);
  ADD(overlapped_gemm1_gemm2_inputs);
  ADD(overlapped_gemm1_gemm2_outputs);
  ADD_NAME(alpha_scale_ptr_array_fc1, alpha_scale_ptr_array_size);
  ADD_NAME(alpha_scale_ptr_array_fc2,
           (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0)
               ? 0
               : alpha_scale_ptr_array_size);
  ADD(fp4_act_scale);
  ADD_NAME(act_fp8_token_scale,
           (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0)
               ? num_moe_inputs * sizeof(float)
           // Below SM100 GEMM1's and GEMM2's row scales (MXFP8: and wide flags), each tile-padded.
           : use_restored_fp4 ? (num_moe_inputs + 128) * (use_restored_mx ? 3 : 2) * sizeof(float)
                              : 0);
  ADD(fused_fc2_fp4_act_scale);
  ADD_NAME(tma_ws_gemm1_workspace, tma_ws_size);
  ADD_NAME(tma_ws_gemm2_workspace, tma_ws_size);
  ADD(precomputed_scheduler_workspace);
  ADD(gemm_workspace);
  ADD(lora_input);
  ADD(lora_fc1_result);
  ADD(lora_add_bias);
  ADD(lora_fc2_result);
  ADD(deepseek_fc_workspace);
  ADD(smoothed_act);

  return out_map;

#undef ADD_NAME
#undef ADD
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
size_t
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
                   Enable>::getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size,
                                             int64_t const inter_size, int const num_experts,
                                             int const experts_per_token,
                                             ActivationType activation_type,
                                             MOEParallelismConfig parallelism_config, bool use_lora,
                                             bool use_deepseek_fp8_block_scale,
                                             bool use_mxfp8_act_scaling, bool min_latency_mode,
                                             bool use_awq) {
  int const ep_size = parallelism_config.ep_size;
  TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0,
                       "Number of experts must be a multiple of ep size");
  auto sizes_map = getWorkspaceDeviceBufferSizes(
      num_rows, hidden_size, inter_size, num_experts / ep_size, experts_per_token, activation_type,
      use_lora, use_deepseek_fp8_block_scale, use_mxfp8_act_scaling, min_latency_mode, use_awq);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(),
                 [](auto& v) { return v.second.first; });
  size_t size = tensorrt_llm::common::calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  TLLM_LOG_TRACE("Mixture Of Experts Plugin requires workspace of %2f MiB", size / 1024.f / 1024.f);
  return size;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
void CutlassMoeFCRunner<
    T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
    Enable>::configureWsPtrs(char* ws_ptr, int64_t const num_rows, int64_t const hidden_size,
                             int64_t const inter_size, int const num_experts_per_node,
                             int const experts_per_token, ActivationType activation_type,
                             MOEParallelismConfig parallelism_config, bool use_lora,
                             bool use_deepseek_fp8_block_scale, bool use_mxfp8_act_scaling,
                             bool min_latency_mode, bool use_awq) {
  auto workspaces = getWorkspaceDeviceBufferSizes(
      num_rows, hidden_size, inter_size, num_experts_per_node, experts_per_token, activation_type,
      use_lora, use_deepseek_fp8_block_scale, use_mxfp8_act_scaling, min_latency_mode, use_awq);

  auto getWsPtr = [&](auto type, std::string const& name) {
    return workspaces.at(name).first
               ? reinterpret_cast<decltype(type)*>(ws_ptr + workspaces.at(name).second)
               : nullptr;
  };

  permuted_row_to_unpermuted_row_ = getWsPtr(int{}, "permuted_row_to_unpermuted_row");
  permuted_token_selected_experts_ = getWsPtr(int{}, "permuted_token_selected_experts");
  blocked_expert_counts_ = getWsPtr(int{}, "blocked_expert_counts");
  blocked_expert_counts_cumsum_ = getWsPtr(int{}, "blocked_expert_counts_cumsum");
  blocked_row_to_unpermuted_row_ = getWsPtr(int{}, "blocked_row_to_unpermuted_row");

  expert_first_token_offset_ = getWsPtr(int64_t{}, "expert_first_token_offset");

  // We check if the provided config uses fused finalize and disable it if it does not
  bool gemm2_using_finalize_fusion =
      gemm2_config_->epilogue_fusion_type ==
      cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
  permuted_token_final_scales_ =
      gemm2_using_finalize_fusion ? getWsPtr(float{}, "permuted_token_final_scales") : nullptr;

  bool const is_gated_activation = isGatedActivation(activation_type);
  bool const gemm1_using_fused_moe = moe_gemm_runner_.isFusedGatedActivation(
      *gemm1_config_, activation_type, inter_size, hidden_size);
  bool const gemm1_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_);
  bool const tma_ws_has_glu =
      gemm1_using_tma_ws && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
  // We always use fused path if we can
  bool const non_tma_ws_has_glu = !gemm1_using_fused_moe && is_gated_activation;
  bool const has_glu_inter_result =
      tma_ws_has_glu || non_tma_ws_has_glu || use_fp8 || use_restored_fp4;

  // Always same value, but overlapped with either fc1_result_ or fc2_result_
  permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  // Always same value, ignored if not needed
  glu_inter_result_ =
      has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs") : nullptr;

  // fc1 and fc2 alias one of the above pointers, but it depends on if actfn is fused/unfused which
  // is overlapped NOTE: It is important to get the overlapped pointers correct as the wrong order
  // will cause the buffer to be used as an input and output for the same gemm, which will cause
  // corruption
  fc1_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
  fc2_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");

  if (use_deepseek_fp8_block_scale) {
    permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
    fc1_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
    glu_inter_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
    fc2_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  }

  alpha_scale_ptr_array_fc1_ = getWsPtr((float const*)(nullptr), "alpha_scale_ptr_array_fc1");
  if constexpr (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0) {
    alpha_scale_ptr_array_fc2_ = alpha_scale_ptr_array_fc1_;
  } else {
    alpha_scale_ptr_array_fc2_ = getWsPtr((float const*)(nullptr), "alpha_scale_ptr_array_fc2");
  }

  fc1_fp4_act_scale_ = nullptr;
  fc2_fp4_act_scale_ = nullptr;
  if (use_block_scaling) {
    fc1_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
    auto* const fused_fc2_fp4_act_scale =
        getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fused_fc2_fp4_act_scale");
    fc2_fp4_act_scale_ = fused_fc2_fp4_act_scale ? fused_fc2_fp4_act_scale : fc1_fp4_act_scale_;
    TLLM_CHECK(fc1_fp4_act_scale_ != nullptr);
    TLLM_CHECK(fc2_fp4_act_scale_ != nullptr);
  }
  act_fp8_token_scale_ = getWsPtr(float{}, "act_fp8_token_scale");
  if constexpr (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0) {
    TLLM_CHECK(act_fp8_token_scale_ != nullptr);
  }

  tma_ws_grouped_gemm1_input_ = {};
  tma_ws_grouped_gemm2_input_ = {};
  if (moe_gemm_runner_.supportsTmaWarpSpecialized()) {
    tma_ws_grouped_gemm1_input_.configureWorkspace(
        getWsPtr(int8_t{}, "tma_ws_gemm1_workspace"), num_experts_per_node,
        getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
        getWsPtr(int8_t{}, "precomputed_scheduler_workspace"),
        workspaces.at("precomputed_scheduler_workspace").first, getScalingType());
    tma_ws_grouped_gemm2_input_.configureWorkspace(
        getWsPtr(int8_t{}, "tma_ws_gemm2_workspace"), num_experts_per_node,
        getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
        getWsPtr(int8_t{}, "precomputed_scheduler_workspace"),
        workspaces.at("precomputed_scheduler_workspace").first, getScalingType());
  }

  lora_fc1_result_ = {};
  lora_add_bias_ = {};
  lora_fc2_result_ = {};

  if (use_lora) {
    lora_input_ = getWsPtr(ScaleBiasType{}, "lora_input");
    lora_fc1_result_ = getWsPtr(ScaleBiasType{}, "lora_fc1_result");
    lora_add_bias_ = getWsPtr(ScaleBiasType{}, "lora_add_bias");
    lora_fc2_result_ = getWsPtr(ScaleBiasType{}, "lora_fc2_result");
    TLLM_CHECK_WITH_INFO(!use_fp8 || lora_input_ != nullptr,
                         "LoRA input must not be nullptr if FP8 is enabled");
    TLLM_CHECK(lora_fc1_result_ != nullptr);
    TLLM_CHECK(lora_add_bias_ != nullptr);
    TLLM_CHECK(lora_fc2_result_ != nullptr);
  }

  if (use_deepseek_fp8_block_scale) {
    auto* blockscale_gemm_runner = getDeepSeekBlockScaleGemmRunner();
    TLLM_CHECK(blockscale_gemm_runner != nullptr);
    blockscale_gemm_runner->configureWorkspace(getWsPtr(char{}, "deepseek_fc_workspace"));
  }

  if (use_awq) {
    smoothed_act_ = getWsPtr(int8_t{}, "smoothed_act");
  }
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunnerInterface*
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, IsMXFPX, Sm90Wfp4Afp8Mode,
                   Enable>::getDeepSeekBlockScaleGemmRunner() const {
  TLLM_CHECK_WITH_INFO(
      (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<OutputType, __nv_bfloat16>),
      "Block scale GEMM runner only supports BF16 A/output");
  TLLM_CHECK_WITH_INFO((std::is_same_v<WeightType, __nv_fp8_e4m3>),
                       "Block scale GEMM runner only supports FP8 weights.");
  return blockscale_gemm_runner_.get();
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
void CutlassMoeFCRunner<
    T, WeightType, OutputType, InputType, ScaleBiasType, IsMXFPX, Sm90Wfp4Afp8Mode,
    Enable>::BlockScaleFC1(DeepSeekBlockScaleGemmRunner& gemm_runner, T const* const input,
                           T* const output, void* const gemm_output,
                           int64_t const* const expert_first_token_offset,
                           WeightType const* const fc1_expert_weights,
                           ScaleBiasType const* const fc1_expert_biases,
                           float const* const fc2_fp8_quant, int64_t const num_rows,
                           int64_t const expanded_num_rows, int64_t const hidden_size,
                           int64_t const inter_size, int const num_experts_per_node,
                           ActivationParams fc1_activation_type, QuantParams& quant_params,
                           bool enable_pdl, cudaStream_t stream) {
  bool const is_gated_activation = isGatedActivation(fc1_activation_type);

  int shape_n = is_gated_activation ? inter_size * 2 : inter_size;
  int shape_k = hidden_size;

  // NOTE: we assume gemm_runner.configureWorkspace has already been called.
  gemm_runner.moeGemm(gemm_output, input, fc1_expert_weights, expert_first_token_offset,
                      num_experts_per_node, shape_n, shape_k, stream, nullptr,
                      quant_params.fp8_block_scaling.fc1_scales_ptrs);

  sync_check_cuda_error(stream);
  constexpr bool bias_is_broadcast = true;
  constexpr bool use_per_expert_act_scale = false;
  doActivation<GemmInputType, UnfusedGemmOutputType>(
      reinterpret_cast<GemmInputType*>(output),
      static_cast<UnfusedGemmOutputType const*>(gemm_output), fc2_fp8_quant, fc1_expert_biases,
      bias_is_broadcast, expert_first_token_offset,
      /*permuted_token_selected_experts=*/nullptr, num_experts_per_node, inter_size,
      expanded_num_rows, fc1_activation_type, quant_params, use_per_expert_act_scale, nullptr,
      nullptr, nullptr, enable_pdl, stream);

  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
void CutlassMoeFCRunner<
    T, WeightType, OutputType, InputType, ScaleBiasType, IsMXFPX, Sm90Wfp4Afp8Mode,
    Enable>::BlockScaleFC2(DeepSeekBlockScaleGemmRunner& gemm_runner, T const* const input,
                           void* const gemm_output, OutputType* const final_output,
                           int64_t const* const expert_first_token_offset,
                           WeightType const* const fc2_expert_weights,
                           ScaleBiasType const* const fc2_expert_biases,
                           float const* const unpermuted_final_scales,
                           int const* const unpermuted_row_to_permuted_row,
                           int const* const permuted_row_to_unpermuted_row,
                           int const* const token_selected_experts,
                           int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
                           int64_t const expanded_num_rows, int64_t const hidden_size,
                           int64_t const unpadded_hidden_size, int64_t const inter_size,
                           int64_t const num_experts_per_node, int64_t const k,
                           MOEParallelismConfig parallelism_config, bool const enable_alltoall,
                           QuantParams& quant_params, bool enable_pdl, cudaStream_t stream) {
  int shape_n = hidden_size;
  int shape_k = inter_size;

  // NOTE: we assume gemm_runner.configureWorkspace has already been called.
  gemm_runner.moeGemm(gemm_output, input, fc2_expert_weights, expert_first_token_offset,
                      num_experts_per_node, shape_n, shape_k, stream, nullptr,
                      quant_params.fp8_block_scaling.fc2_scales_ptrs);

  sync_check_cuda_error(stream);

  finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
      static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
      unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
      token_selected_experts, expert_first_token_offset, num_rows, hidden_size,
      unpadded_hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall,
      enable_pdl, stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
T const*
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, IsMXFPX, Sm90Wfp4Afp8Mode,
                   Enable>::applyPrequantScale(void* smoothed_act, void const* permuted_data,
                                               void const* prequant_scales,
                                               int64_t const* num_valid_tokens_ptr,
                                               int64_t const expanded_num_rows,
                                               int64_t const seq_len, bool const use_awq,
                                               cudaStream_t stream) {
  T const* gemm_input;
  bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
  if (use_prequant_scale_kernel) {
    TLLM_CHECK_WITH_INFO((!std::is_same_v<T, WeightType>),
                         "Prequant scales are only used for different weight/activation type!");
    if constexpr (!std::is_same_v<T, WeightType>) {
      tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<UnfusedGemmOutputType, T>(
          reinterpret_cast<T*>(smoothed_act),
          reinterpret_cast<UnfusedGemmOutputType const*>(permuted_data),
          reinterpret_cast<UnfusedGemmOutputType const*>(prequant_scales), expanded_num_rows,
          seq_len, num_valid_tokens_ptr, stream);
    }
    gemm_input = reinterpret_cast<T const*>(smoothed_act);
  } else {
    gemm_input = reinterpret_cast<T const*>(permuted_data);
  }
  sync_check_cuda_error(stream);
  return gemm_input;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
void CutlassMoeFCRunner<
    T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
    Enable>::gemm1(MoeGemmRunner<T, WeightType, BackBoneType, ScaleBiasType, IsMXFPX,
                                 Sm90Wfp4Afp8Mode>& gemm_runner,
                   DeepSeekBlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input,
                   T* const output, void* const intermediate_result,
                   int64_t const* const expert_first_token_offset,
                   int const* const permuted_token_selected_experts,
                   TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
                   WeightType const* const fc1_expert_weights,
                   ScaleBiasType const* const fc1_expert_biases,
                   int64_t const* const num_valid_tokens_ptr,
                   ScaleBiasType const* const fc1_int_scales, float const* const fc1_fp8_dequant,
                   float const* const fc2_fp8_quant, float* const act_fp8_token_scale,
                   TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
                   TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat,
                   QuantParams quant_params, int64_t const num_rows,
                   int64_t const expanded_num_rows, int64_t const hidden_size,
                   int64_t const inter_size, int const num_experts_per_node,
                   ActivationParams fc1_activation_type, float const** alpha_scale_ptr_array,
                   bool bias_is_broadcast, cudaStream_t stream,
                   cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
                   int* num_active_experts_per, int* active_expert_global_ids, bool enable_pdl) {
  if (fp8_blockscale_gemm_runner) {
    TLLM_CHECK(!min_latency_mode);
    Self::BlockScaleFC1(*fp8_blockscale_gemm_runner, input, output, intermediate_result,
                        expert_first_token_offset, fc1_expert_weights, fc1_expert_biases,
                        fc2_fp8_quant, num_rows, expanded_num_rows, hidden_size, inter_size,
                        num_experts_per_node, fc1_activation_type, quant_params, enable_pdl,
                        stream);
    return;
  }

  bool const using_tma_ws_gemm1 = gemm_runner.isTmaWarpSpecialized(config);
  bool const is_gated_activation = isGatedActivation(fc1_activation_type);
  bool const use_ampere_activation_fusion = gemm_runner.isFusedGatedActivation(
      config, fc1_activation_type.activation_type, inter_size, hidden_size);
  bool const use_fused_activation_epilogue =
      using_tma_ws_gemm1 && tma_ws_input_template.fusion ==
                                TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION;
  TLLM_CHECK_WITH_INFO(fc1_activation_type.activation_type != ActivationType::ClampedRelu2 ||
                           use_fused_activation_epilogue,
                       "ClampedRelu2 requires a supported SM10x activation-fusion tactic");
  size_t const fc1_out_size =
      ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  if (min_latency_mode) {
    TLLM_CHECK_WITH_INFO(using_tma_ws_gemm1,
                         "Only TMA warp specialized GEMM is supported in min latency mode.");
    // TODO: as for bias, need to get the correct expert id according to the active expert global
    // ids
    TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Min latency mode does not support bias.");
    TLLM_CHECK(use_fp4);
  }

  if (using_tma_ws_gemm1) {
    TLLM_CHECK(config.is_tma_warp_specialized);
    TLLM_CHECK(!use_ampere_activation_fusion);
    TLLM_CHECK(!use_fp4 || fc1_fp4_act_flat);
    TLLM_CHECK(!use_fp4 || fc2_fp4_act_flat);

    bool has_different_gemm_output_type = using_tma_ws_gemm1 && !std::is_same_v<T, OutputType>;
    bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
    TLLM_CHECK_WITH_INFO(use_fused_activation_epilogue || has_intermediate || input != output,
                         "Input and output buffers are overlapping");
    auto* gemm_output = use_fused_activation_epilogue
                            ? static_cast<void*>(output)
                            : (has_intermediate ? intermediate_result : static_cast<void*>(output));

    auto tma_ws_input = tma_ws_input_template;

    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                     quant_params.groupwise.fc1.alpha, stream);
    }
    if constexpr (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0) {
      TLLM_CHECK(alpha_scale_ptr_array != nullptr);
    }

    auto universal_input = GroupedGemmInput<T, WeightType, BackBoneType, BackBoneType>{
        input,
        total_tokens_including_expert,
        /*weights*/ nullptr,
        /*scales*/ nullptr,
        /*zeros*/ nullptr,
        /*biases*/ nullptr,
        /*C*/ static_cast<BackBoneType*>(gemm_output),
        alpha_scale_ptr_array,
        /*occupancy*/ nullptr,
        fc1_activation_type,
        num_rows,
        /*N*/ int64_t(fc1_out_size),
        /*K*/ hidden_size,
        num_experts_per_node,
        quant_params.groupwise.group_size,
        /*bias_is_broadcast*/ true,
        /*use_fused_moe*/ false,
        stream,
        config};
    gemm_runner.moeGemm(universal_input, tma_ws_input);

    sync_check_cuda_error(stream);

    if (use_fused_activation_epilogue) {
      // GEMM1 already produced the activated tensor. The MXFP8 specialization
      // also generated the UE8M0 block scales consumed by GEMM2.
      return;
    }

    // TODO: when bias_is_broadcast is false, fuse bias to gemm
    using GatedActOutputType = std::conditional_t<use_w4afp8, BackBoneType, GemmInputType>;
    bool use_per_expert_act_scale = use_fp4 ? quant_params.fp4.fc2.use_per_expert_act_scale
                                    : use_wfp4afp8
                                        ? quant_params.fp8_mxfp4.fc2.use_per_expert_act_scale
                                    : use_fp8 ? quant_params.fp8.fc2_use_per_expert_act_scale
                                              : false;
    float* fp8_token_dequant_scale = nullptr;
    float const* fp8_expert_residual_scale = nullptr;
    if constexpr (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0) {
      TLLM_CHECK(quant_params.fp8_mxfp4.fc2.global_scale != nullptr);
      fp8_token_dequant_scale = act_fp8_token_scale;
      fp8_expert_residual_scale = quant_params.fp8_mxfp4.fc2.global_scale;
    }

    doActivation<GatedActOutputType, UnfusedGemmOutputType, ScaleBiasType,
                 std::conditional_t<use_sm90_mx, ClampedSituAdaptor, SituAdaptor>>(
        reinterpret_cast<GatedActOutputType*>(output),
        static_cast<UnfusedGemmOutputType const*>(gemm_output), fc2_fp8_quant, fc1_expert_biases,
        bias_is_broadcast, expert_first_token_offset, permuted_token_selected_experts,
        num_experts_per_node, inter_size, expanded_num_rows, fc1_activation_type, quant_params,
        use_per_expert_act_scale, fc2_fp4_act_flat, fp8_token_dequant_scale,
        fp8_expert_residual_scale, enable_pdl, stream);

    sync_check_cuda_error(stream);
  } else if (use_fp8 || use_restored_fp4) {
    TLLM_CHECK(!use_ampere_activation_fusion);
    TLLM_CHECK(!config.is_tma_warp_specialized);
    TLLM_CHECK(!use_block_scaling || use_restored_fp4);

    // Row scales carry the experts' global scales; with token rows the small-M kernel takes them.
    alpha_scale_ptr_array =
        use_restored_fp4 && (act_fp8_token_scale || w4a16TokenRows(quant_params))
            ? nullptr
            : computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                     use_restored_fp4 ? quant_params.fp4.fc1.global_scale
                                                      : quant_params.fp8.dequant_fc1,
                                     stream);

    auto universal_input = GroupedGemmInput<T, WeightType, BackBoneType, BackBoneType>{
        input,
        total_tokens_including_expert,
        fc1_expert_weights,
        static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales),
        /*zeros*/ nullptr,
        /*biases*/ nullptr,
        reinterpret_cast<UnfusedGemmOutputType*>(intermediate_result),
        alpha_scale_ptr_array,
        /*occupancy*/ nullptr,
        fc1_activation_type,
        expanded_num_rows,
        /*N*/ int64_t(fc1_out_size),
        /*K*/ use_restored_fp4 ? padNvfp4K(hidden_size) : hidden_size,
        num_experts_per_node,
        quant_params.groupwise.group_size,
        /*bias_is_broadcast*/ true,
        /*use_fused_moe*/ false,
        stream,
        config};
    // Below SM100 act_fp8_token_scale is given with the token rows restored: the row scales.
    universal_input.row_scales = use_restored_fp4 ? act_fp8_token_scale : nullptr;
    if (int const* const token_rows = w4a16TokenRows(quant_params)) {
      universal_input.a_rows = token_rows;
      universal_input.a_tokens = static_cast<int>(num_rows);
      universal_input.expert_scales = quant_params.fp4.fc1.global_scale;
    }
    gemm_runner.moeGemm(universal_input, TmaWarpSpecializedGroupedGemmInput{});
#if defined(ENABLE_FP4) && defined(FLASHINFER_FP4_SW_CVT)
    if constexpr (use_restored_mx) {
      if (quant_params.restored_mxfp8.wide_rows) {  // none from the profiler
        completeWideMxfp8RowsKernel<<<num_experts_per_node*(fc1_out_size / 128), 128, 0, stream>>>(
            static_cast<UnfusedGemmOutputType*>(intermediate_result), quant_params.restored_mxfp8,
            fc1_expert_weights, quant_params.groupwise.fc1.weight_scales, act_fp8_token_scale,
            expert_first_token_offset, num_rows, hidden_size, fc1_out_size);
      }
    }
#endif

    bool use_per_expert_act_scale = use_fp8 ? quant_params.fp8.fc2_use_per_expert_act_scale : false;
    if (use_restored_fp4) use_per_expert_act_scale = quant_params.fp4.fc2.use_per_expert_act_scale;
    // FC2 activations stay 16-bit without FC2 scales: per-token (quantized after GEMM1) or W4A16.
    bool const fc2_scales = use_restored_mx || quant_params.fp4.fc2.act_global_scale;
    auto* const act_sf = use_restored_fp4 && fc2_scales ? fc2_fp4_act_flat : nullptr;
    doActivation<GemmInputType, UnfusedGemmOutputType, ScaleBiasType,
                 std::conditional_t<use_restored_fp4, ClampedSituAdaptor, SituAdaptor>>(
        reinterpret_cast<GemmInputType*>(output),
        static_cast<UnfusedGemmOutputType const*>(intermediate_result), fc2_fp8_quant,
        fc1_expert_biases, bias_is_broadcast, expert_first_token_offset,
        permuted_token_selected_experts, num_experts_per_node, inter_size, expanded_num_rows,
        fc1_activation_type, quant_params, use_per_expert_act_scale, act_sf, nullptr, nullptr,
        enable_pdl, stream);

    sync_check_cuda_error(stream);
  } else if (!is_gated_activation) {
    TLLM_CHECK(!use_ampere_activation_fusion);
    TLLM_CHECK(!config.is_tma_warp_specialized);
    TLLM_CHECK(!use_block_scaling);
    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                     quant_params.groupwise.fc1.alpha, stream);
    }
    auto universal_input = GroupedGemmInput<T, WeightType, BackBoneType, BackBoneType>{
        input,
        total_tokens_including_expert,
        fc1_expert_weights,
        /*scales*/ quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales)
            : fc1_int_scales,
        /*zeros*/ quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_zeros)
            : nullptr,
        fc1_expert_biases,
        reinterpret_cast<BackBoneType*>(output),
        alpha_scale_ptr_array,
        /*occupancy*/ nullptr,
        fc1_activation_type,
        expanded_num_rows,
        /*N*/ int64_t(fc1_out_size),
        /*K*/ hidden_size,
        num_experts_per_node,
        quant_params.groupwise.group_size,
        bias_is_broadcast,
        /*use_fused_moe*/ false,
        stream,
        config};
    gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    sync_check_cuda_error(stream);
  } else {
    TLLM_CHECK(!config.is_tma_warp_specialized);
    TLLM_CHECK(is_gated_activation);
    TLLM_CHECK_WITH_INFO(!use_ampere_activation_fusion || input != output,
                         "Input and output buffers are overlapping");
    TLLM_CHECK(!use_block_scaling);
    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                     quant_params.groupwise.fc1.alpha, stream);
    }
    // Run the GEMM with activation function overridden with `Identity`, we do the activation
    // separately
    auto universal_input = GroupedGemmInput<T, WeightType, BackBoneType, BackBoneType>{
        input,
        total_tokens_including_expert,
        fc1_expert_weights,
        /*scales*/ quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales)
            : fc1_int_scales,
        /*zeros*/ quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_zeros)
            : nullptr,
        fc1_expert_biases,
        static_cast<BackBoneType*>(use_ampere_activation_fusion ? output : intermediate_result),
        alpha_scale_ptr_array,
        /*occupancy*/ nullptr,
        use_ampere_activation_fusion ? fc1_activation_type.activation_type
                                     : ActivationType::Identity,
        expanded_num_rows,
        /*N*/ int64_t(fc1_out_size),
        /*K*/ hidden_size,
        num_experts_per_node,
        quant_params.groupwise.group_size,
        bias_is_broadcast,
        use_ampere_activation_fusion,
        stream,
        config};
    gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    sync_check_cuda_error(stream);

    if (!use_ampere_activation_fusion) {
      using GatedActOutputType = std::conditional_t<use_w4afp8, BackBoneType, GemmInputType>;
      doGatedActivation<GatedActOutputType, UnfusedGemmOutputType>(
          reinterpret_cast<GatedActOutputType*>(output),
          static_cast<UnfusedGemmOutputType const*>(intermediate_result), expert_first_token_offset,
          inter_size, expanded_num_rows, num_experts_per_node, fc1_activation_type, stream);

      sync_check_cuda_error(stream);
    }
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Sm90Wfp4Afp8Mode, Enable>::
    gemm2(MoeGemmRunner<T, WeightType, BackBoneType, ScaleBiasType, IsMXFPX, Sm90Wfp4Afp8Mode>&
              gemm_runner,
          DeepSeekBlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input,
          void* const gemm_output, OutputType* const final_output,
          int64_t const* const expert_first_token_offset,
          TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
          WeightType const* const fc2_expert_weights, ScaleBiasType const* const fc2_expert_biases,
          ScaleBiasType const* const fc2_int_scales, float const* const fc2_fp8_dequant,
          TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
          QuantParams quant_params, float const* const unpermuted_final_scales,
          float const* const permuted_final_scales, int const* const unpermuted_row_to_permuted_row,
          int const* permuted_row_to_unpermuted_row, int const* const token_selected_experts,
          int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
          int64_t const expanded_num_rows, int64_t const hidden_size,
          int64_t const unpadded_hidden_size, int64_t const inter_size,
          int const num_experts_per_node, int64_t const k, float const** alpha_scale_ptr_array,
          bool use_lora, void* fc2_lora, cudaStream_t stream,
          MOEParallelismConfig parallelism_config, bool const enable_alltoall,
          cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
          int* num_active_experts_per, int* active_expert_global_ids, bool enable_pdl,
          float const* row_scales, bool skip_finalize) {
  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  bool const using_tma_ws_gemm2 = gemm_runner.isTmaWarpSpecialized(config);

  if (min_latency_mode) {
    TLLM_CHECK_WITH_INFO(using_tma_ws_gemm2,
                         "Only TMA warp specialized GEMM is supported in min latency mode.");
    TLLM_CHECK(use_fp4);
  }

  if (fp8_blockscale_gemm_runner) {
    Self::BlockScaleFC2(*fp8_blockscale_gemm_runner, input, gemm_output, final_output,
                        expert_first_token_offset, fc2_expert_weights, fc2_expert_biases,
                        unpermuted_final_scales, unpermuted_row_to_permuted_row,
                        permuted_row_to_unpermuted_row, token_selected_experts,
                        num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size,
                        unpadded_hidden_size, inter_size, num_experts_per_node, k,
                        parallelism_config, enable_alltoall, quant_params, enable_pdl, stream);
    return;
  }

  TmaWarpSpecializedGroupedGemmInput tma_ws_input{};
  if (using_tma_ws_gemm2) {
    tma_ws_input = tma_ws_input_template;
    if (tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
      // TODO For some reason this has to be done here, it should not overlap with anything else,
      // but doing it in setupTmaWarpSpecializedInputs gives a different result. Ideally, we want
      // this to run on a second stream and overlap with everything else
      //
      // This also means it is included in the timing for the profiler, which is probably more
      // representative until we can overlap it
      check_cuda_error(cudaMemsetAsync(
          final_output, 0x0, sizeof(OutputType) * num_rows * unpadded_hidden_size, stream));
    }
  } else if (use_fp8) {
    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                   fc2_fp8_dequant, stream);
  } else if (use_restored_fp4) {  // no alpha (a null scale) with row scales or the token rows
    bool const no_alpha = row_scales || w4a16TokenRows(quant_params);
    alpha_scale_ptr_array =
        computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                               no_alpha ? nullptr : quant_params.fp4.fc2.global_scale, stream);
  }
  if (use_w4afp8) {
    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                   quant_params.groupwise.fc2.alpha, stream);
  }
  if constexpr (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0) {
    TLLM_CHECK(alpha_scale_ptr_array != nullptr);
  }

  bool fuse_lora_bias = use_lora && !(use_fp8 || using_tma_ws_gemm2);
  // Note: expanded_num_rows, to check this value, it's greater than num_rows * num_experts_per_node
  auto universal_input = GroupedGemmInput<T, WeightType, BackBoneType, BackBoneType>{
      input,
      total_tokens_including_expert,
      fc2_expert_weights,
      quant_params.groupwise.group_size > 0
          ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc2.weight_scales)
          : fc2_int_scales,
      quant_params.groupwise.group_size > 0
          ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc2.weight_zeros)
          : nullptr,
      fuse_lora_bias ? static_cast<ScaleBiasType*>(fc2_lora) : nullptr,
      static_cast<BackBoneType*>(gemm_output),
      alpha_scale_ptr_array,
      /*occupancy*/ nullptr,
      ActivationType::Identity,
      expanded_num_rows,
      /*N*/ hidden_size,
      /*K*/ use_restored_fp4 ? padNvfp4K(inter_size) : inter_size,
      num_experts_per_node,
      quant_params.groupwise.group_size,
      /*bias_is_broadcast*/ false,
      /*use_fused_moe*/ false,
      stream,
      config};
  universal_input.row_scales = row_scales;
  universal_input.expert_scales =
      w4a16TokenRows(quant_params) ? quant_params.fp4.fc2.global_scale : nullptr;
  gemm_runner.moeGemmBiasAct(universal_input, tma_ws_input);
  sync_check_cuda_error(stream);

  if (min_latency_mode) return;

  if (use_lora && !fuse_lora_bias) {
    auto loraBiasApplyFunc =
        doActivation<UnfusedGemmOutputType, UnfusedGemmOutputType, ScaleBiasType>;
    loraBiasApplyFunc(static_cast<UnfusedGemmOutputType*>(gemm_output),
                      static_cast<UnfusedGemmOutputType const*>(gemm_output), nullptr,
                      static_cast<ScaleBiasType const*>(fc2_lora), false, expert_first_token_offset,
                      /*permuted_token_selected_experts=*/nullptr, num_experts_per_node,
                      hidden_size, expanded_num_rows, ActivationParams(ActivationType::Identity),
                      {}, false, nullptr, nullptr, nullptr, enable_pdl, stream);
    sync_check_cuda_error(stream);
  }
  if (skip_finalize) {  // GEMM2's rows, with the FC2 biases, in final_output
    // Only rank 0 adds the biases under tensor parallelism, as finalize does.
    if (fc2_expert_biases && parallelism_config.tp_rank == 0) {
      doActivation<UnfusedGemmOutputType, UnfusedGemmOutputType, ScaleBiasType>(
          reinterpret_cast<UnfusedGemmOutputType*>(final_output),
          static_cast<UnfusedGemmOutputType const*>(gemm_output), nullptr, fc2_expert_biases, true,
          expert_first_token_offset, /*permuted_token_selected_experts=*/nullptr,
          num_experts_per_node, hidden_size, expanded_num_rows,
          ActivationParams(ActivationType::Identity), {}, false, nullptr, nullptr, nullptr,
          enable_pdl, stream);
      sync_check_cuda_error(stream);
    } else if (gemm_output != final_output) {  // the TMA GEMM's output
      TLLM_CUDA_CHECK(cudaMemcpyAsync(final_output, gemm_output,
                                      expanded_num_rows * hidden_size * sizeof(OutputType),
                                      cudaMemcpyDeviceToDevice, stream));
    }
    return;
  }

  bool has_different_output_type_ampere = (use_w4afp8 || use_fp8) && !using_tma_ws_gemm2;
  has_different_output_type_ampere |= use_restored_fp4 && !using_tma_ws_gemm2;
  bool using_fused_finalize =
      tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
  bool has_different_output_type_tma_ws = !using_fused_finalize && using_tma_ws_gemm2;

  if (has_different_output_type_ampere || has_different_output_type_tma_ws) {
    finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
        token_selected_experts, expert_first_token_offset, num_rows, hidden_size,
        unpadded_hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall,
        enable_pdl, stream);
  } else if (!using_tma_ws_gemm2) {
    finalizeMoeRoutingKernelLauncher<OutputType, GemmInputType>(
        static_cast<GemmInputType const*>(gemm_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
        token_selected_experts, expert_first_token_offset, num_rows, hidden_size,
        unpadded_hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall,
        enable_pdl, stream);
  }
  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
bool CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Sm90Wfp4Afp8Mode, Enable>::setupLoraWorkspace(int64_t expanded_num_rows,
                                                                      int64_t num_rows,
                                                                      int64_t inter_size,
                                                                      int64_t hidden_size,
                                                                      int start_expert,
                                                                      bool is_gated_activation,
                                                                      int num_experts_per_node,
                                                                      bool needs_num_valid,
                                                                      LoraParams& lora_params,
                                                                      cudaStream_t stream) {
  std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
  std::vector<void const*>& host_permuted_fc1_weight_ptrs =
      host_lora_workspace_.host_permuted_fc1_weight_ptrs;
  std::vector<void const*>& host_permuted_fc2_weight_ptrs =
      host_lora_workspace_.host_permuted_fc2_weight_ptrs;
  std::vector<void const*>& host_permuted_gated_weight_ptrs =
      host_lora_workspace_.host_permuted_gated_weight_ptrs;

  std::vector<int32_t>& host_permuted_fc1_lora_ranks =
      host_lora_workspace_.host_permuted_fc1_lora_ranks;
  std::vector<int32_t>& host_permuted_fc2_lora_ranks =
      host_lora_workspace_.host_permuted_fc2_lora_ranks;
  std::vector<int32_t>& host_permuted_gated_lora_ranks =
      host_lora_workspace_.host_permuted_gated_lora_ranks;
  std::vector<int64_t>& host_expert_first_token_offset =
      host_lora_workspace_.host_expert_first_token_offset;

  bool all_token_without_lora = true;

  host_permuted_fc1_weight_ptrs.resize(expanded_num_rows * 2);
  host_permuted_fc1_lora_ranks.resize(expanded_num_rows);
  host_permuted_fc2_weight_ptrs.resize(expanded_num_rows * 2);
  host_permuted_fc2_lora_ranks.resize(expanded_num_rows);

  if (is_gated_activation) {
    host_permuted_gated_weight_ptrs.resize(expanded_num_rows * 2);
    host_permuted_gated_lora_ranks.resize(expanded_num_rows);
  }

  TLLM_CUDA_CHECK(cudaEventSynchronize(*(lora_params.memcpy_event_ptr)));

  size_t num_valid_tokens =
      needs_num_valid ? host_expert_first_token_offset[num_experts_per_node] : expanded_num_rows;

  for (int expert_idx = 0; expert_idx < num_experts_per_node; ++expert_idx) {
    int weight_index = expert_idx + start_expert;
    for (size_t i = host_expert_first_token_offset[expert_idx];
         i < host_expert_first_token_offset[expert_idx + 1]; ++i) {
      int source_index = host_permuted_rows[i] % num_rows;
      int32_t lora_rank = lora_params.fc1_lora_ranks[source_index];
      host_permuted_fc1_weight_ptrs[i * 2] =
          reinterpret_cast<ScaleBiasType const*>(
              lora_params.fc1_lora_weight_ptrs[source_index * 2]) +
          weight_index * hidden_size * lora_rank;
      host_permuted_fc1_weight_ptrs[i * 2 + 1] =
          reinterpret_cast<ScaleBiasType const*>(
              lora_params.fc1_lora_weight_ptrs[source_index * 2 + 1]) +
          weight_index * lora_rank * inter_size;
      host_permuted_fc1_lora_ranks[i] = lora_rank;

      lora_rank = lora_params.fc2_lora_ranks[source_index];
      host_permuted_fc2_weight_ptrs[i * 2] =
          reinterpret_cast<ScaleBiasType const*>(
              lora_params.fc2_lora_weight_ptrs[source_index * 2]) +
          weight_index * inter_size * lora_rank;
      host_permuted_fc2_weight_ptrs[i * 2 + 1] =
          reinterpret_cast<ScaleBiasType const*>(
              lora_params.fc2_lora_weight_ptrs[source_index * 2 + 1]) +
          weight_index * lora_rank * hidden_size;
      host_permuted_fc2_lora_ranks[i] = lora_rank;

      if (host_permuted_fc1_lora_ranks[i] || host_permuted_fc2_lora_ranks[i]) {
        all_token_without_lora = false;
      }

      if (is_gated_activation) {
        lora_rank = lora_params.gated_lora_ranks[source_index];
        host_permuted_gated_weight_ptrs[i * 2] =
            reinterpret_cast<ScaleBiasType const*>(
                lora_params.gated_lora_weight_ptrs[source_index * 2]) +
            weight_index * hidden_size * lora_rank;
        host_permuted_gated_weight_ptrs[i * 2 + 1] =
            reinterpret_cast<ScaleBiasType const*>(
                lora_params.gated_lora_weight_ptrs[source_index * 2 + 1]) +
            weight_index * lora_rank * inter_size;
        host_permuted_gated_lora_ranks[i] = lora_rank;

        if (host_permuted_gated_lora_ranks[i]) {
          all_token_without_lora = false;
        }
      }
    }
  }
  return all_token_without_lora;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
auto CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Sm90Wfp4Afp8Mode, Enable>::loraFC1(int64_t expanded_num_rows,
                                                           int64_t inter_size, int64_t hidden_size,
                                                           int num_experts_per_node,
                                                           int start_expert,
                                                           int64_t const* num_valid_tokens_ptr,
                                                           bool is_gated_activation,
                                                           ScaleBiasType const* fc1_expert_biases,
                                                           LoraParams& lora_params,
                                                           float const* input_fp8_dequant,
                                                           cudaStream_t stream)
    -> ScaleBiasType const* {
  TLLM_CHECK_WITH_INFO(!act_fp4, "LoRA does not support FP4 activations");
  std::vector<void const*>& host_permuted_fc1_weight_ptrs =
      host_lora_workspace_.host_permuted_fc1_weight_ptrs;
  std::vector<void const*>& host_permuted_gated_weight_ptrs =
      host_lora_workspace_.host_permuted_gated_weight_ptrs;

  std::vector<int32_t>& host_permuted_fc1_lora_ranks =
      host_lora_workspace_.host_permuted_fc1_lora_ranks;
  std::vector<int32_t>& host_permuted_gated_lora_ranks =
      host_lora_workspace_.host_permuted_gated_lora_ranks;
  std::vector<int64_t>& host_expert_first_token_offset =
      host_lora_workspace_.host_expert_first_token_offset;

  auto fc1_lora_impl = lora_params.fc1_lora_impl;
  int num_reqs = lora_params.num_reqs;

  ScaleBiasType *lora_gated_out = nullptr, *lora_fc1_result = nullptr;

  if (is_gated_activation) {
    lora_gated_out = lora_fc1_result_;
    lora_fc1_result = lora_fc1_result_ + expanded_num_rows * inter_size;
  } else {
    lora_fc1_result = lora_fc1_result_;
  }

  ScaleBiasType* input{};
  if constexpr (use_fp8) {
    TLLM_CHECK(lora_input_);
    bool const scale_is_dequant = true;
    dequantFP8<ScaleBiasType, T>(lora_input_, permuted_data_, num_valid_tokens_ptr, hidden_size,
                                 expanded_num_rows, input_fp8_dequant, scale_is_dequant, stream);
    sync_check_cuda_error(stream);
    input = lora_input_;
  } else if constexpr (!act_fp4) {
    TLLM_CHECK(!lora_input_);
    input = reinterpret_cast<ScaleBiasType*>(permuted_data_);
  }

  void* lora_workspace = lora_params.workspace;
  void* tmp_lora_fc_result = static_cast<void*>(lora_fc1_result);
  int64_t num_valid_tokens = host_expert_first_token_offset[num_experts_per_node];
  int64_t num_reqs_lora =
      std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

  ::tensorrt_llm::kernels::Lora_run(fc1_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
                                    host_permuted_fc1_lora_ranks.data(),
                                    host_permuted_fc1_weight_ptrs.data(), 0, &tmp_lora_fc_result,
                                    lora_workspace, stream);

  if (is_gated_activation) {
    void* tmp_lora_gated_result = static_cast<void*>(lora_gated_out);
    ::tensorrt_llm::kernels::Lora_run(fc1_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
                                      host_permuted_gated_lora_ranks.data(),
                                      host_permuted_gated_weight_ptrs.data(), 0,
                                      &tmp_lora_gated_result, lora_workspace, stream);
  }

  // add bias and reorder
  if (fc1_expert_biases != nullptr) {
    loraAddBias(lora_add_bias_, lora_fc1_result_, fc1_expert_biases, num_valid_tokens_ptr,
                inter_size, permuted_token_selected_experts_, expanded_num_rows,
                is_gated_activation, stream);
    return lora_add_bias_;
  } else if (is_gated_activation) {
    loraReorder(lora_add_bias_, lora_fc1_result_, num_valid_tokens_ptr, inter_size,
                expanded_num_rows, stream);
    return lora_add_bias_;
  } else {
    return lora_fc1_result_;
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Sm90Wfp4Afp8Mode, Enable>::loraFC2(int64_t inter_size, int64_t hidden_size,
                                                           int num_experts_per_node,
                                                           int start_expert,
                                                           int64_t const* num_valid_tokens_ptr,
                                                           int64_t num_tokens,
                                                           LoraParams& lora_params,
                                                           float const* fc2_fp8_quant,
                                                           cudaStream_t stream) {
  std::vector<void const*>& host_permuted_fc2_weight_ptrs =
      host_lora_workspace_.host_permuted_fc2_weight_ptrs;
  std::vector<int32_t>& host_permuted_fc2_lora_ranks =
      host_lora_workspace_.host_permuted_fc2_lora_ranks;
  std::vector<int64_t>& host_expert_first_token_offset =
      host_lora_workspace_.host_expert_first_token_offset;
  auto fc2_lora_impl = lora_params.fc2_lora_impl;
  int num_reqs = lora_params.num_reqs;

  ScaleBiasType* input{};
  if constexpr (use_fp8) {
    TLLM_CHECK(lora_input_);
    bool const scale_is_dequant = false;
    dequantFP8(lora_input_, fc1_result_, num_valid_tokens_ptr, inter_size, num_tokens,
               fc2_fp8_quant, scale_is_dequant, stream);
    sync_check_cuda_error(stream);
    input = lora_input_;
  } else if constexpr (!act_fp4) {
    TLLM_CHECK(!lora_input_);
    input = reinterpret_cast<ScaleBiasType*>(fc1_result_);
  }

  void* lora_workspace = lora_params.workspace;
  int64_t num_valid_tokens = host_expert_first_token_offset[num_experts_per_node];
  void* tmp_lora_fc_result = static_cast<void*>(lora_fc2_result_);
  int64_t num_reqs_lora =
      std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

  ::tensorrt_llm::kernels::Lora_run(fc2_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
                                    host_permuted_fc2_lora_ranks.data(),
                                    host_permuted_fc2_weight_ptrs.data(), 0, &tmp_lora_fc_result,
                                    lora_workspace, stream);
  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
void CutlassMoeFCRunner<
    T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
    Enable>::runMoe(void const* input_activations_void, void const* input_sf_void,
                    bool const swizzled_input_sf, int const* token_selected_experts,
                    float const* token_final_scales, void const* fc1_expert_weights_void,
                    void const* fc1_expert_biases_void, ActivationParams fc1_activation_type,
                    void const* fc2_expert_weights_void, void const* fc2_expert_biases_void,
                    QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size,
                    int64_t const unpadded_hidden_size, int64_t const inter_size,
                    int const full_num_experts, int const experts_per_token, char* workspace_ptr,
                    void* final_output_void, int* unpermuted_row_to_permuted_row,
                    MOEParallelismConfig parallelism_config, bool const enable_alltoall,
                    bool use_lora, LoraParams& lora_params, bool use_deepseek_fp8_block_scale,
                    bool use_mxfp8_act_scaling, bool min_latency_mode,
                    MoeMinLatencyParams& min_latency_params, bool enable_pdl, cudaStream_t stream,
                    float const* per_token_scale, bool skip_finalize) {
  static constexpr bool int_scales_required = std::is_same<WeightType, uint8_t>::value ||
                                              std::is_same<WeightType, cutlass::uint4b_t>::value ||
                                              use_wfp4a16;
  static constexpr bool fp8_scales_required = std::is_same<WeightType, __nv_fp8_e4m3>::value ||
                                              std::is_same<WeightType, __nv_fp8_e5m2>::value;

  auto const* input_activations = static_cast<InputType const*>(input_activations_void);
  auto const* input_sf =
      input_sf_void
          ? reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(input_sf_void)
          : nullptr;
  auto const* fc1_expert_weights = static_cast<WeightType const*>(fc1_expert_weights_void);
  auto const* fc1_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc1_expert_biases_void);
  auto const* fc2_expert_weights = static_cast<WeightType const*>(fc2_expert_weights_void);
  auto const* fc1_int_scales =
      reinterpret_cast<ScaleBiasType const*>(quant_params.wo.fc1_weight_scales);
  auto const* fc2_int_scales =
      reinterpret_cast<ScaleBiasType const*>(quant_params.wo.fc2_weight_scales);

  auto const* fc1_fp8_dequant = quant_params.fp8.dequant_fc1;
  auto const* fc2_fp8_quant = quant_params.fp8.quant_fc2;
  auto const* fc2_fp8_dequant = quant_params.fp8.dequant_fc2;
  // WMXFP8AMXFP8 keeps dequant scales in mxfp8_mxfp8; remap these shared FP8 pointers so
  // downstream GEMM/TMA setup can reuse the common FP8 code path.
  if (fp8_scales_required && use_mxfp8_act_scaling) {
    TLLM_CHECK_WITH_INFO(quant_params.mxfp8_mxfp8.fc1.weight_block_scale != nullptr,
                         "WMXFP8AMXFP8 requires FC1 weight_block_scale to be non-null");
    fc1_fp8_dequant = quant_params.mxfp8_mxfp8.fc1.global_scale;
    fc2_fp8_dequant = quant_params.mxfp8_mxfp8.fc2.global_scale;
  }
  auto const* input_fp8_dequant = quant_params.fp8.dequant_input;

  auto const* fc2_wfp4afp8_quant_scale =
      (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0)
          ? nullptr
          : quant_params.fp8_mxfp4.fc2.act_global_scale;

  auto const* fc2_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc2_expert_biases_void);
  auto* final_output = static_cast<OutputType*>(final_output_void);
  float const* token_topk_unpermuted_scales = token_final_scales;
  // Note: getDeepSeekBlockScaleGemmRunner will do a sanity check on our template parameters.
  auto* blockscale_gemm_runner =
      use_deepseek_fp8_block_scale ? getDeepSeekBlockScaleGemmRunner() : nullptr;

  TLLM_CHECK(input_activations);
  TLLM_CHECK(token_selected_experts);
  TLLM_CHECK(fc1_expert_weights);
  TLLM_CHECK(fc2_expert_weights);
  TLLM_CHECK(workspace_ptr);
  // TLLM_CHECK(token_topk_unpermuted_scales);
  TLLM_CHECK(unpermuted_row_to_permuted_row);
  TLLM_CHECK(full_num_experts % parallelism_config.ep_size == 0);
  TLLM_CHECK(full_num_experts % parallelism_config.cluster_size == 0);

  if (quant_params.mxfp8_mxfp4.fc1.weight_block_scale ||
      quant_params.mxfp8_mxfp8.fc1.weight_block_scale) {
    TLLM_CHECK_WITH_INFO(
        hidden_size % (64 * 8 / sizeof_bits<WeightType>::value) == 0,
        "Hidden size %d does not meet minimum alignment requirements for MXFP8 MOE GEMM %d",
        (int)hidden_size, (int)(64 * 8 / sizeof_bits<WeightType>::value));
    TLLM_CHECK_WITH_INFO(
        inter_size % (64 * 8 / sizeof_bits<WeightType>::value) == 0,
        "Inter size %d does not meet minimum alignment requirements for MXFP8 MOE GEMM %d",
        (int)inter_size, (int)(64 * 8 / sizeof_bits<WeightType>::value));
  } else {
    // For NoSmem epilogue schedule, we need to align the output of the GEMM to 256 bits, for gated
    // activation this is automatic if the usual alignment requirement is met
    if (gemm1_config_->epilogue_schedule == cutlass_extensions::EpilogueScheduleType::NO_SMEM &&
        !isGatedActivation(fc1_activation_type)) {
      TLLM_CHECK_WITH_INFO(
          inter_size % (256 / sizeof_bits<WeightType>::value) == 0,
          "Inter size %d does not meet minimum alignment requirements for MOE GEMM %d",
          (int)inter_size, (int)(256 / sizeof_bits<WeightType>::value));
    }

    if (gemm2_config_->epilogue_schedule == cutlass_extensions::EpilogueScheduleType::NO_SMEM) {
      TLLM_CHECK_WITH_INFO(
          gemm2_config_->epilogue_fusion_type !=
              cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE,
          "Got NoSmem epilogue schedule, which is not supported for finalize fusion");
      TLLM_CHECK_WITH_INFO(
          hidden_size % (256 / sizeof_bits<WeightType>::value) == 0,
          "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d",
          (int)hidden_size, (int)(256 / sizeof_bits<WeightType>::value));
    }

    // Require at least 128 bits of alignment for MOE GEMM
    TLLM_CHECK_WITH_INFO(
        hidden_size % (128 / sizeof_bits<WeightType>::value) == 0,
        "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d",
        (int)hidden_size, (int)(128 / sizeof_bits<WeightType>::value));
    TLLM_CHECK_WITH_INFO(
        inter_size % (128 / sizeof_bits<WeightType>::value) == 0,
        "Inter size %d does not meet minimum alignment requirements for MOE GEMM %d",
        (int)inter_size, (int)(128 / sizeof_bits<WeightType>::value));
  }

  // These values must fit into an int for building the source maps
  TLLM_CHECK_WITH_INFO(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
  TLLM_CHECK_WITH_INFO(num_rows * full_num_experts <= std::numeric_limits<int>::max(),
                       "Number of rows * num_experts is too large");
  TLLM_CHECK_WITH_INFO(experts_per_token * full_num_experts <= std::numeric_limits<int>::max(),
                       "experts_per_token * num_experts is too large");

  TLLM_CHECK_WITH_INFO(gemm1_config_, "MOE GEMM1 Config is not set");
  TLLM_CHECK_WITH_INFO(gemm2_config_, "MOE GEMM2 Config is not set");

  TLLM_CHECK_WITH_INFO(!use_lora || !act_fp4, "MOE does not support LoRA with FP4 model");

  if (int_scales_required) {
    if (!(quant_params.groupwise.fc1.weight_scales && quant_params.groupwise.fc2.weight_scales)) {
      TLLM_CHECK_WITH_INFO(fc1_int_scales != nullptr,
                           "Weight scales expected but scale for first matmul is a null pointer");
      TLLM_CHECK_WITH_INFO(fc2_int_scales != nullptr,
                           "Weight scales expected but scale for second matmul is a null pointer");
    }
    TLLM_CHECK_WITH_INFO(
        fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
        "FP8 scales are provided for integer quantization");
  } else if (fp8_scales_required && use_mxfp8_act_scaling) {
    TLLM_CHECK_WITH_INFO(
        fc1_fp8_dequant != nullptr,
        "WMXFP8AMXFP8 scales expected but dequant scale for FC1 is a null pointer");
    TLLM_CHECK_WITH_INFO(
        fc2_fp8_dequant != nullptr,
        "WMXFP8AMXFP8 scales expected but dequant scale for FC2 is a null pointer");
    TLLM_CHECK_WITH_INFO(fc1_int_scales == nullptr && fc2_int_scales == nullptr,
                         "Integer scales are provided for WMXFP8AMXFP8 quantization");
  } else if (fp8_scales_required && !use_deepseek_fp8_block_scale) {
    TLLM_CHECK_WITH_INFO(fc1_fp8_dequant != nullptr,
                         "FP8 scales expected but dequant scale for FC1 is a null pointer");
    TLLM_CHECK_WITH_INFO(fc2_fp8_quant != nullptr,
                         "FP8 scales expected but quant scale for FC2 is a null pointer");
    TLLM_CHECK_WITH_INFO(fc2_fp8_dequant != nullptr,
                         "FP8 scales expected but quant scale for FC2 is a null pointer");

    TLLM_CHECK_WITH_INFO(fc1_int_scales == nullptr && fc2_int_scales == nullptr,
                         "Integer scales are provided for FP8 quantization");
  } else if (use_lora && use_fp8) {
    TLLM_CHECK_WITH_INFO(input_fp8_dequant != nullptr,
                         "FP8 scales expected but quant scale for input is a null pointer");
  } else {
    TLLM_CHECK_WITH_INFO(fc1_int_scales == nullptr,
                         "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
    TLLM_CHECK_WITH_INFO(fc2_int_scales == nullptr,
                         "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
    TLLM_CHECK_WITH_INFO(
        fc1_fp8_dequant == nullptr,
        "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
    TLLM_CHECK_WITH_INFO(fc2_fp8_quant == nullptr,
                         "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
    TLLM_CHECK_WITH_INFO(fc2_fp8_dequant == nullptr,
                         "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
  }

  bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales &&
                 !use_wfp4a16;
  int const num_experts_per_node = full_num_experts / parallelism_config.ep_size;

  configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node,
                  experts_per_token, fc1_activation_type, parallelism_config, use_lora,
                  use_deepseek_fp8_block_scale, use_mxfp8_act_scaling, min_latency_mode, use_awq);

  int start_expert = num_experts_per_node * parallelism_config.ep_rank;
  int end_expert = start_expert + num_experts_per_node;

  bool const needs_num_valid = parallelism_config.ep_size > 1;
  int64_t const* num_valid_tokens_ptr =
      needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

  auto expanded_num_rows = num_rows * experts_per_token;

  if (min_latency_mode) {
    TLLM_CHECK(use_lora == false);
    TLLM_CHECK(use_awq == false);
    TLLM_CHECK(use_fp4 == true);

    buildMinLatencyActiveExpertMaps(
        min_latency_params.num_active_experts_per_node, min_latency_params.experts_to_token_score,
        min_latency_params.active_expert_global_ids, expert_first_token_offset_,
        token_selected_experts, token_final_scales, num_rows, experts_per_token, start_expert,
        end_expert, num_experts_per_node, parallelism_config.cluster_rank,
        parallelism_config.cluster_size, full_num_experts, enable_pdl, stream);
    sync_check_cuda_error(stream);

    auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(
        num_rows, expanded_num_rows, fc1_activation_type, hidden_size, unpadded_hidden_size,
        inter_size, num_experts_per_node, input_activations_void, input_sf, final_output,
        fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
        min_latency_mode, min_latency_params, use_lora, start_expert, parallelism_config,
        enable_pdl, stream);

    // todo: input_activations_void should be nvfp4, waiting for yuxian's mr ready
    Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner,
                reinterpret_cast<T const*>(input_activations_void), fc1_result_, glu_inter_result_,
                expert_first_token_offset_, /*permuted_token_selected_experts=*/nullptr,
                gemm1_tma_ws_input, fc1_expert_weights, fc1_expert_biases, num_valid_tokens_ptr,
                fc1_int_scales, fc1_fp8_dequant,
                use_wfp4afp8 ? fc2_wfp4afp8_quant_scale : fc2_fp8_quant, act_fp8_token_scale_,
                input_sf /*input fp4 scale or expanded fp4 scale*/, fc2_fp4_act_scale_,
                quant_params, num_rows, expanded_num_rows, hidden_size, inter_size,
                num_experts_per_node, fc1_activation_type, alpha_scale_ptr_array_fc1_, !use_lora,
                stream, *gemm1_config_, true, min_latency_params.num_active_experts_per_node,
                min_latency_params.active_expert_global_ids, enable_pdl);
    sync_check_cuda_error(stream);

    auto gemm2_input =
        applyPrequantScale(smoothed_act_, fc1_result_, quant_params.groupwise.fc2.act_scales,
                           num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
    Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, final_output, nullptr,
                expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights,
                fc2_expert_biases, fc2_int_scales, fc2_fp8_dequant, fc2_fp4_act_scale_,
                quant_params, token_topk_unpermuted_scales, permuted_token_final_scales_,
                unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_,
                token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows,
                hidden_size, unpadded_hidden_size, inter_size, num_experts_per_node,
                experts_per_token, alpha_scale_ptr_array_fc2_, use_lora, lora_fc2_result_, stream,
                parallelism_config, enable_alltoall, *gemm2_config_, true,
                min_latency_params.num_active_experts_per_node,
                min_latency_params.active_expert_global_ids, enable_pdl);
    sync_check_cuda_error(stream);
  } else {
    bool fused_prologue_result = false;
    bool try_fused_prologue = true;
#if defined(FLASHINFER_FP4_SW_CVT)
    // Rows <= experts sort faster in one CTA (scan over rows) than the fused radix rank (experts).
    try_fused_prologue = num_rows * experts_per_token >
                         std::min<int64_t>(SMALL_PROLOGUE_THREADS, num_experts_per_node);
#endif
    if (!use_sm90_mixed_input_gemm && try_fused_prologue) {
      // WAR: fusedBuildExpertMapsSortFirstToken kernel will lead to illegal memory access for
      // W4AFP8
      fused_prologue_result = fusedBuildExpertMapsSortFirstToken(
          token_selected_experts, permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row,
          expert_first_token_offset_, num_rows, num_experts_per_node, experts_per_token,
          start_expert, end_expert, enable_pdl, stream);
    }

    if (!fused_prologue_result) {
      TLLM_LOG_TRACE("Falling back to unfused prologue");
      threeStepBuildExpertMapsSortFirstToken(
          token_selected_experts, permuted_token_selected_experts_, permuted_row_to_unpermuted_row_,
          unpermuted_row_to_permuted_row, expert_first_token_offset_, blocked_expert_counts_,
          blocked_expert_counts_cumsum_, blocked_row_to_unpermuted_row_, num_rows,
          num_experts_per_node, experts_per_token, start_expert, enable_pdl, stream);
    }

    sync_check_cuda_error(stream);

    bool is_gated_activation = isGatedActivation(fc1_activation_type);

    if (use_lora) {
      std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
      std::vector<int64_t>& host_expert_first_token_offset =
          host_lora_workspace_.host_expert_first_token_offset;
      host_permuted_rows.resize(expanded_num_rows);
      TLLM_CUDA_CHECK(tensorrt_llm::common::cudaMemcpyAsyncSanitized(
          host_permuted_rows.data(), permuted_row_to_unpermuted_row_,
          expanded_num_rows * sizeof(int), cudaMemcpyDeviceToHost, stream));
      host_expert_first_token_offset.resize(num_experts_per_node + 1);
      TLLM_CUDA_CHECK(tensorrt_llm::common::cudaMemcpyAsyncSanitized(
          host_expert_first_token_offset.data(), expert_first_token_offset_,
          (num_experts_per_node + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
      TLLM_CUDA_CHECK(cudaEventRecord(*(lora_params.memcpy_event_ptr), stream));
    }

    // Only NVFP4xNVFP4 supports FC1 per-expert act scale
    bool use_per_expert_act_scale = use_fp4 ? quant_params.fp4.fc1.use_per_expert_act_scale : false;
    // Below SM100 the GEMMs take 16-bit restored activations: each token quantized (16-bit input)
    // and restored once, then expanded; with per-expert FC1 activation scales the expanded rows
    // are quantized, as from SM100, and restored.
    bool const restore_tokens = use_restored_fp4 && !use_per_expert_act_scale;
    float* row_scales = nullptr;
    float* fc2_row_scales = nullptr;
    TLLM_CHECK_WITH_INFO(!per_token_scale || (restore_tokens && std::is_same_v<InputType, Fp4Type>),
                         "Per-token NVFP4 activations come quantized, without per-expert scales");
    T* gemm1_input_expand =
        use_w4afp8 ? reinterpret_cast<T*>(smoothed_act_) : reinterpret_cast<T*>(permuted_data_);
    if (use_restored_fp4) gemm1_input_expand = reinterpret_cast<T*>(glu_inter_result_);
    float* fp8_token_dequant_scale = nullptr;
    float const* fp8_expert_residual_scale = nullptr;
    int const* humming_permuted_token_selected_experts = nullptr;
    float const** act_fp8_token_scale_ptr_array = nullptr;
    if constexpr (use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0) {
      TLLM_CHECK(quant_params.fp8_mxfp4.fc1.global_scale != nullptr);
      fp8_token_dequant_scale = act_fp8_token_scale_;
      fp8_expert_residual_scale = quant_params.fp8_mxfp4.fc1.global_scale;
      humming_permuted_token_selected_experts = permuted_token_selected_experts_;
      act_fp8_token_scale_ptr_array = alpha_scale_ptr_array_fc1_;
    }
    if constexpr (use_restored_mx) {
      // MXFP8 input rows are always restored, below.
    } else if (!restore_tokens) {
      expandInputRowsKernelLauncher(
          input_activations, gemm1_input_expand, token_topk_unpermuted_scales,
          permuted_token_final_scales_, permuted_row_to_unpermuted_row_,
          humming_permuted_token_selected_experts, num_rows, hidden_size, experts_per_token,
          num_experts_per_node, quant_params, use_per_expert_act_scale, expert_first_token_offset_,
          fc1_fp4_act_scale_, input_sf, swizzled_input_sf,
          (use_w4afp8 && !use_fp8_input) ? quant_params.groupwise.fc1.act_scales : nullptr,
          fp8_token_dequant_scale, fp8_expert_residual_scale, act_fp8_token_scale_ptr_array,
          enable_pdl, stream);
    }
#if defined(ENABLE_FP4) && defined(FLASHINFER_FP4_SW_CVT)
    if constexpr (use_restored_fp4) {
      auto* scratch = reinterpret_cast<BackBoneType*>(glu_inter_result_);
      // BF16 W4A16 with both GEMMs small-M restores nothing: GEMM1 reads the token rows in place
      // (K checked unpadded: hidden_size is their pitch); both use the experts' global scales.
      if (std::is_same_v<InputType, __nv_bfloat16> && std::is_same_v<BackBoneType, __nv_bfloat16> &&
          restore_tokens && !quant_params.fp4.fc1.act_global_scale && !use_lora &&
          gemm1_config_->enableCudaKernel && gemm2_config_->enableCudaKernel &&
          cutlass_kernels_oss::sm80FP4SmallMSupportsShape(
              inter_size * (is_gated_activation ? 2 : 1), hidden_size, expanded_num_rows,
              num_experts_per_node) &&
          cutlass_kernels_oss::sm80FP4SmallMSupportsShape(
              hidden_size, padNvfp4K(inter_size), expanded_num_rows, num_experts_per_node)) {
        quant_params.w4a16_token_rows = permuted_row_to_unpermuted_row_;
      } else if (restore_tokens) {
        // W4A16 (no FC1 activation global scale) expands the 16-bit input as it is.
        bool const w4a16 = !use_restored_mx && !quant_params.fp4.fc1.act_global_scale;
        TLLM_CHECK_WITH_INFO(!w4a16 || (!std::is_same_v<InputType, Fp4Type> &&
                                        hidden_size % 64 == 0 && inter_size % 64 == 0),
                             "NVFP4 W4A16 takes 16-bit activations, hidden and inter sizes "
                             "divisible by 64");
        void const* codes = input_activations;
        auto const* sf = input_sf;
        bool swizzled = swizzled_input_sf;
        if constexpr (!std::is_same_v<InputType, Fp4Type> && !use_restored_mx) {
          if (!w4a16) {
            invokeFP4Quantization<InputType,
                                  TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize>(
                1, num_rows, hidden_size, input_activations, quant_params.fp4.fc1.act_global_scale,
                reinterpret_cast<int64_t*>(scratch), reinterpret_cast<int32_t*>(fc1_fp4_act_scale_),
                false, QuantizationSFLayout::SWIZZLED_128x4,
                tensorrt_llm::common::getMultiProcessorCount(), enable_pdl, false, false,
                resolveNVFP4Recipe(quant_params.fp4.nvfp4_4over6_code), stream);
            codes = scratch;
            sf = fc1_fp4_act_scale_;
            swizzled = true;
          }
        }
        row_scales = act_fp8_token_scale_;
        fc2_row_scales = row_scales + expanded_num_rows + 128;
        auto* const wide_rows =
            use_restored_mx ? reinterpret_cast<int*>(fc2_row_scales + expanded_num_rows + 128)
                            : nullptr;
        // Few rows: columns split over more CTAs as in finalize, but not MXFP8 (wide flag per row).
        int64_t const sm_count = tensorrt_llm::common::getMultiProcessorCount();
        int64_t const ctas = std::min(sm_count * 8, std::max(num_rows, int64_t{1}));
        int64_t const padded_vecs = padNvfp4K(hidden_size) / 8;
        int64_t const sms_per_row = use_restored_mx ? 1 : std::max<int64_t>(sm_count / ctas, 1);
        int64_t threads = 256;
        while (threads > 64 && ceilDiv(padded_vecs, threads / 2) <= sms_per_row) {
          threads /= 2;
        }
        int64_t const splits = std::min(ceilDiv(padded_vecs, threads), sms_per_row);
        // The chunk's permuted rows, rounded up to 16 bytes: sm_90a loads them 4 at a time.
        size_t const smem = roundUp(std::min<int64_t>(experts_per_token, threads), 4) * sizeof(int);
        restoreTokenRowsKernel<BackBoneType, use_restored_mx>
            <<<dim3(ctas, splits), threads, smem, stream>>>(
                reinterpret_cast<BackBoneType*>(permuted_data_), codes, w4a16 ? nullptr : sf,
                swizzled, num_rows, hidden_size, token_selected_experts,
                unpermuted_row_to_permuted_row, experts_per_token, start_expert,
                num_experts_per_node, quant_params.fp4.fc1.global_scale,
                quant_params.fp4.fc2.global_scale, per_token_scale, row_scales, fc2_row_scales,
                wide_rows);
        if constexpr (use_restored_mx) {
          quant_params.restored_mxfp8 = {input_activations, input_sf, swizzled_input_sf, wide_rows,
                                         permuted_row_to_unpermuted_row_};
        }
      } else {
        restoredRows<false>(reinterpret_cast<BackBoneType*>(permuted_data_), scratch,
                            fc1_fp4_act_scale_, expert_first_token_offset_, num_experts_per_node,
                            expanded_num_rows, hidden_size, stream);
      }
    }
#endif
    auto const* gemm1_input = gemm1_input_expand;

    sync_check_cuda_error(stream);

    auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(
        num_rows, expanded_num_rows, fc1_activation_type, hidden_size, unpadded_hidden_size,
        inter_size, num_experts_per_node, input_activations_void, input_sf, final_output,
        fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
        min_latency_mode, min_latency_params, use_lora, start_expert, parallelism_config,
        enable_pdl, stream);

    auto* gemm1_result = fc1_result_;
    auto* gemm2_result = fc2_result_;
    bool const use_fused_activation_epilogue =
        gemm1_tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION;
    if constexpr (use_mxfp8) {
      if (use_fused_activation_epilogue) {
        // Keep the member workspace aliases stable across calls. The fused
        // MXFP8 path reverses their roles for this invocation only.
        gemm1_result = reinterpret_cast<T*>(fc2_result_);
        gemm2_result = static_cast<void*>(fc1_result_);
      }
    }

    if (use_lora) {
      bool all_token_without_lora = setupLoraWorkspace(
          expanded_num_rows, num_rows, inter_size, hidden_size, start_expert, is_gated_activation,
          num_experts_per_node, needs_num_valid, lora_params, stream);

      if (!all_token_without_lora) {
        fc1_expert_biases =
            loraFC1(expanded_num_rows, inter_size, hidden_size, num_experts_per_node, start_expert,
                    num_valid_tokens_ptr, is_gated_activation, fc1_expert_biases, lora_params,
                    input_fp8_dequant, stream);
        sync_check_cuda_error(stream);
      } else {
        use_lora = false;
      }
    }
    TLLM_CHECK_WITH_INFO(
        use_restored_fp4 || use_sm90_mx || !(lora_params.fc1_deltas || lora_params.fc2_inputs),
        "Only the restored FP4 and SM90 MXFP8 x MXFP4 runners take TRT-LLM's LoRA");
#if defined(ENABLE_FP4) && defined(FLASHINFER_FP4_SW_CVT)
    if constexpr (use_restored_fp4 || use_sm90_mx) {
      if (lora_params.fc1_deltas) {  // GEMM1's row biases, the (k, token) rows permuted
        expandInputRowsKernelLauncher(static_cast<ScaleBiasType const*>(lora_params.fc1_deltas),
                                      static_cast<ScaleBiasType*>(lora_params.fc1_row_biases),
                                      nullptr, nullptr, permuted_row_to_unpermuted_row_, nullptr,
                                      expanded_num_rows, inter_size * (is_gated_activation ? 2 : 1),
                                      1, num_experts_per_node, QuantParams{}, false,
                                      expert_first_token_offset_, nullptr, nullptr, true, nullptr,
                                      nullptr, nullptr, nullptr, enable_pdl, stream);
        fc1_expert_biases = static_cast<ScaleBiasType const*>(lora_params.fc1_row_biases);
      }
    }
#endif

    if constexpr (!use_w4afp8) {
      gemm1_input =
          applyPrequantScale(smoothed_act_, permuted_data_, quant_params.groupwise.fc1.act_scales,
                             num_valid_tokens_ptr, expanded_num_rows, hidden_size, use_awq, stream);
    }
    if (w4a16TokenRows(quant_params)) {
      gemm1_input = static_cast<T const*>(input_activations_void);
    }
    sync_check_cuda_error(stream);
    // Restored FP4: activation reads rows' experts from the unfused prologue's map (fused: none).
    Self::gemm1(
        moe_gemm_runner_, blockscale_gemm_runner, gemm1_input, gemm1_result, glu_inter_result_,
        expert_first_token_offset_,
        use_restored_fp4 && !fused_prologue_result ? permuted_token_selected_experts_
                                                   : humming_permuted_token_selected_experts,
        gemm1_tma_ws_input, fc1_expert_weights, fc1_expert_biases, num_valid_tokens_ptr,
        fc1_int_scales, fc1_fp8_dequant, use_wfp4afp8 ? fc2_wfp4afp8_quant_scale : fc2_fp8_quant,
        use_restored_fp4 ? row_scales : act_fp8_token_scale_, fc1_fp4_act_scale_,
        per_token_scale ? nullptr : fc2_fp4_act_scale_, quant_params, num_rows, expanded_num_rows,
        hidden_size, inter_size, num_experts_per_node, fc1_activation_type,
        alpha_scale_ptr_array_fc1_, !use_lora && !lora_params.fc1_deltas, stream, *gemm1_config_,
        false, nullptr, nullptr, enable_pdl);
    sync_check_cuda_error(stream);
#if defined(ENABLE_FP4) && defined(FLASHINFER_FP4_SW_CVT)
    if constexpr (use_restored_fp4 || use_sm90_mx) {
      if (lora_params.fc2_inputs) {
        // GEMM2's inputs in the permuted rows: restored NVFP4 or MXFP8 packed to their codes, else
        // (W4A16, per-token NVFP4 before its quantization, SM90 MXFP8 codes) as they are.
        if (!use_restored_mx &&
            (use_sm90_mx || !quant_params.fp4.fc2.act_global_scale || per_token_scale)) {
          TLLM_CUDA_CHECK(cudaMemcpyAsync(lora_params.fc2_inputs, gemm1_result,
                                          expanded_num_rows * inter_size * sizeof(GemmInputType),
                                          cudaMemcpyDeviceToDevice, stream));
        } else if constexpr (use_restored_fp4) {
          restoredRows<true, use_restored_mx>(reinterpret_cast<BackBoneType*>(gemm1_result),
                                              lora_params.fc2_inputs, fc2_fp4_act_scale_,
                                              expert_first_token_offset_, num_experts_per_node,
                                              expanded_num_rows, inter_size, stream);
        }
      }
    }
    if constexpr (use_restored_fp4) {
      if (per_token_scale) {
        // Per token: GEMM2's input quantized per row (the FC2 activation global scale as
        // CuTe-DSL's fc2_input_scale), restored, and each row's GEMM2 scale set.
        auto* activations = reinterpret_cast<BackBoneType*>(gemm1_result);
        invokeNvfp4QuantAndPerTokenScale(expanded_num_rows, inter_size, activations, 1.0f, nullptr,
                                         reinterpret_cast<uint8_t*>(glu_inter_result_),
                                         fc2_fp4_act_scale_, fc2_row_scales,
                                         QuantizationSFLayout::SWIZZLED_128x4,
                                         resolveNVFP4Recipe(quant_params.fp4.nvfp4_4over6_code),
                                         stream, enable_pdl, quant_params.fp4.fc2.act_global_scale);
        restoredRows<false>(activations, glu_inter_result_, fc2_fp4_act_scale_, nullptr, 0,
                            expanded_num_rows, inter_size, stream);
        nvfp4RowScalesKernel<<<std::max((expanded_num_rows + 255) / 256, int64_t{1}), 256, 0,
                               stream>>>(fc2_row_scales, quant_params.fp4.fc2.global_scale,
                                         expert_first_token_offset_, num_experts_per_node);
      }
    }
#endif

    if (use_lora) {
      loraFC2(inter_size, hidden_size, num_experts_per_node, start_expert, num_valid_tokens_ptr,
              expanded_num_rows, lora_params, fc2_fp8_quant, stream);
      sync_check_cuda_error(stream);
    }

    auto gemm2_input =
        applyPrequantScale(smoothed_act_, gemm1_result, quant_params.groupwise.fc2.act_scales,
                           num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
    sync_check_cuda_error(stream);
    TLLM_CHECK_WITH_INFO(!skip_finalize || ((use_restored_fp4 || use_sm90_mx) &&
                                            std::is_same_v<OutputType, UnfusedGemmOutputType>),
                         "Only the restored FP4 and SM90 MXFP8 x MXFP4 runners skip finalize, "
                         "into their GEMM output type");
    Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input,
                skip_finalize && use_restored_fp4 ? final_output : gemm2_result, final_output,
                expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights,
                fc2_expert_biases, fc2_int_scales, fc2_fp8_dequant, fc2_fp4_act_scale_,
                quant_params, token_topk_unpermuted_scales, permuted_token_final_scales_,
                unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_,
                token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows,
                hidden_size, unpadded_hidden_size, inter_size, num_experts_per_node,
                experts_per_token, alpha_scale_ptr_array_fc2_, use_lora, lora_fc2_result_, stream,
                parallelism_config, enable_alltoall, *gemm2_config_, false, nullptr, nullptr,
                enable_pdl, fc2_row_scales, skip_finalize);
    sync_check_cuda_error(stream);
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
                   Enable>::
    computeStridesTmaWarpSpecialized(
        int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
        TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens,
        int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
        int64_t gemm2_k, int const num_experts_per_node, T const* gemm1_in, T const* gemm2_in,
        WeightType const* weights1, WeightType const* weights2, float const* fp8_dequant1,
        float const* fp8_dequant2,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2,
        QuantParams quant_params, ScaleBiasType const* bias1, ScaleBiasType const* bias2,
        UnfusedGemmOutputType* gemm1_output, UnfusedGemmOutputType* gemm2_output,
        float const* router_scales, int const* permuted_row_to_unpermuted_row, bool enable_pdl,
        cudaStream_t stream) {
  // Always nullptr
  layout_info1.ptr_c = nullptr;
  layout_info1.stride_c = nullptr;
  layout_info2.ptr_c = nullptr;
  layout_info2.stride_c = nullptr;

  layout_info1.fused_finalize_epilogue.ptr_bias = nullptr;
  if (!bias2) {
    layout_info2.fused_finalize_epilogue.ptr_bias = nullptr;
  }

  constexpr bool use_humming_pre_mma =
      use_wfp4afp8 && Sm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0;
  auto alpha_scale_flat1 = use_fp4 ? quant_params.fp4.fc1.global_scale
                           : (use_wfp4afp8 && !use_humming_pre_mma)
                               ? quant_params.fp8_mxfp4.fc1.global_scale
                           : use_fp8 ? fp8_dequant1
                                     : nullptr;
  auto alpha_scale_flat2 = use_fp4 ? quant_params.fp4.fc2.global_scale
                           : (use_wfp4afp8 && !use_humming_pre_mma)
                               ? quant_params.fp8_mxfp4.fc2.global_scale
                           : use_fp8 ? fp8_dequant2
                                     : nullptr;
  if (!alpha_scale_flat1 && !alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array = nullptr;
    layout_info2.alpha_scale_ptr_array = nullptr;
  }

  layout_info1.int4_groupwise_params.enabled = use_sm90_mixed_input_gemm;
  layout_info2.int4_groupwise_params.enabled = use_sm90_mixed_input_gemm;
  bool const use_mxfp4_weight_scale = use_wfp4a16 || use_wfp4afp8;
  layout_info1.int4_groupwise_params.use_wfp4a16 = use_mxfp4_weight_scale;
  layout_info2.int4_groupwise_params.use_wfp4a16 = use_mxfp4_weight_scale;

  layout_info1.fpX_block_scaling_type = getScalingType();
  layout_info2.fpX_block_scaling_type = getScalingType();

  // Use a smaller block size to spread work across multiple SMs. Each thread handles one expert,
  // so we only need num_experts_per_node threads total. With 1 warp per block, 128 experts
  // yields 4 blocks across 4 SMs instead of 1 block on 1 SM.
  int const threads = std::min(32, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;

  auto* kernel_instance =
      &computeStridesTmaWarpSpecializedKernel<T, WeightType, UnfusedGemmOutputType, ScaleBiasType>;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, kernel_instance, expert_first_token_offset, layout_info1,
                     layout_info2, num_tokens, expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n,
                     gemm2_k, num_experts_per_node, gemm1_in, gemm2_in, weights1, weights2,
                     alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2,
                     quant_params, bias1, bias2, gemm1_output, gemm2_output, router_scales,
                     permuted_row_to_unpermuted_row);

  return std::make_pair(layout_info1, layout_info2);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
                   Enable>::
    computeStridesTmaWarpSpecializedLowLatency(
        TmaWarpSpecializedGroupedGemmInput layout_info1,
        TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n,
        int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k, int const num_experts, T const* input1,
        T const* input2, WeightType const* weights1, WeightType const* weights2,
        float const* fp8_dequant1, float const* fp8_dequant2,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
        QuantParams quant_params, ScaleBiasType const* bias1, ScaleBiasType const* bias2,
        UnfusedGemmOutputType* output1, UnfusedGemmOutputType* output2,
        int const* num_active_experts_per, int const* active_expert_global_ids, int start_expert,
        bool enable_pdl, cudaStream_t stream) {
  TLLM_THROW("Min latency mode is no longer supported");
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Sm90Wfp4Afp8Mode,
                   Enable>::
    setupTmaWarpSpecializedInputs(int64_t num_rows, int64_t expanded_num_rows,
                                  ActivationParams fc1_activation_type, int64_t hidden_size,
                                  int64_t unpadded_hidden_size, int64_t inter_size,
                                  int64_t num_experts_per_node, void const* input_activations_void,
                                  TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf,
                                  void* final_output, WeightType const* fc1_expert_weights,
                                  WeightType const* fc2_expert_weights, QuantParams quant_params,
                                  ScaleBiasType const* fc1_expert_biases,
                                  ScaleBiasType const* fc2_expert_biases, bool min_latency_mode,
                                  MoeMinLatencyParams& min_latency_params, bool use_lora,
                                  int start_expert, MOEParallelismConfig parallelism_config,
                                  bool enable_pdl, cudaStream_t stream) {
  auto gemm1_tma_ws_input = tma_ws_grouped_gemm1_input_;
  auto gemm2_tma_ws_input = tma_ws_grouped_gemm2_input_;

  // Set enable_pdl for both GEMM inputs
  gemm1_tma_ws_input.enable_pdl = enable_pdl;
  gemm2_tma_ws_input.enable_pdl = enable_pdl;
  if (!moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) &&
      !moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_)) {
    return std::make_pair(gemm1_tma_ws_input, gemm2_tma_ws_input);
  }

  bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales &&
                 !use_wfp4a16;

  bool is_gated_activation = isGatedActivation(fc1_activation_type);
  int64_t const fc1_out_size = is_gated_activation ? inter_size * 2 : inter_size;

  bool has_different_gemm_output_type = !std::is_same_v<T, UnfusedGemmOutputType>;
  bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
  auto* gemm1_output = has_intermediate ? glu_inter_result_ : static_cast<void*>(fc1_result_);

  bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
  auto gemm2_input = use_prequant_scale_kernel ? smoothed_act_ : fc1_result_;
  auto* gemm2_output = fc2_result_;

  if (min_latency_mode) {
    auto gemm1_input = input_activations_void;

    gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm1_tma_ws_input.swap_ab = true;
    gemm2_tma_ws_input.swap_ab = true;

    TLLM_CHECK_WITH_INFO(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
    return Self::computeStridesTmaWarpSpecializedLowLatency(
        gemm1_tma_ws_input, gemm2_tma_ws_input, num_rows, fc1_out_size, hidden_size, hidden_size,
        inter_size, num_experts_per_node, reinterpret_cast<T const*>(gemm1_input),
        reinterpret_cast<T const*>(gemm2_input), fc1_expert_weights, fc2_expert_weights,
        quant_params.fp8.dequant_fc1, quant_params.fp8.dequant_fc2, input_sf, fc2_fp4_act_scale_,
        quant_params, nullptr, nullptr, reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
        reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_),
        min_latency_params.num_active_experts_per_node, min_latency_params.active_expert_global_ids,
        start_expert, enable_pdl, stream);
  } else {
    auto gemm1_input = use_prequant_scale_kernel ? smoothed_act_ : permuted_data_;

    gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

    gemm1_tma_ws_input.swap_ab = gemm1_config_->swap_ab;
    gemm2_tma_ws_input.swap_ab = gemm2_config_->swap_ab;
    gemm1_tma_ws_input.precomputed_scheduler_total_routed_tokens = expanded_num_rows;
    gemm2_tma_ws_input.precomputed_scheduler_total_routed_tokens = expanded_num_rows;

    bool const activation_fusion_dtype = use_mxfp8 || (std::is_same_v<T, __nv_bfloat16> &&
                                                       std::is_same_v<WeightType, __nv_bfloat16>);
    bool const use_fused_clamped_relu2 =
        activation_fusion_dtype && gemm1_config_->sm_version >= 100 &&
        gemm1_config_->sm_version < 110 &&
        fc1_activation_type.activation_type == ActivationType::ClampedRelu2 &&
        fc1_activation_type.clamped_relu2_limit != nullptr && fc1_expert_biases == nullptr &&
        !use_lora && !gemm1_tma_ws_input.swap_ab;
    if (use_fused_clamped_relu2) {
      gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION;
      gemm1_tma_ws_input.fused_activation_epilogue.clamp_limit =
          fc1_activation_type.clamped_relu2_limit;
      if constexpr (use_mxfp8) {
        // The ordinary MXFP8 path needs a BF16 GEMM1 intermediate. Reverse the
        // two large scratch roles so the fused epilogue can write MXFP8 without
        // overlapping the permuted input that GEMM1 is still reading.
        gemm1_output = fc2_result_;
        gemm2_input = reinterpret_cast<T*>(fc2_result_);
        gemm2_output = static_cast<void*>(fc1_result_);
        TLLM_CHECK_WITH_INFO(gemm2_input != gemm2_output,
                             "GEMM2 input and output buffers are overlapping");
        TLLM_CHECK_WITH_INFO(fc1_fp4_act_scale_ != fc2_fp4_act_scale_,
                             "Fused GEMM1 input and output scale buffers are overlapping");
      }
    }
    TLLM_CHECK_WITH_INFO(
        (gemm1_tma_ws_input.swap_ab && gemm2_tma_ws_input.swap_ab) || !use_sm90_mixed_input_gemm,
        "Hopper mixed-input grouped GEMM requires swap_ab");

    bool apply_bias = parallelism_config.tp_rank == 0;
    auto* fc2_bias = apply_bias ? fc2_expert_biases : nullptr;
    bool gemm2_using_finalize_fusion =
        gemm2_config_->epilogue_fusion_type ==
        cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
    bool using_fused_finalize = use_fused_finalize_ && gemm2_using_finalize_fusion &&
                                !use_sm90_mixed_input_gemm && !use_lora;
    TLLM_CHECK_WITH_INFO(
        using_fused_finalize == gemm2_using_finalize_fusion,
        "GEMM2 tactic requests finalize fusion, but the runner is not configured to use it");
    if (using_fused_finalize) {
      assert(min_latency_mode == false);
      bool use_reduction = expanded_num_rows > num_rows;
      gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
      gemm2_tma_ws_input.setFinalizeFusionParams(final_output, unpadded_hidden_size, num_rows,
                                                 use_reduction);
    }

    TLLM_CHECK_WITH_INFO(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
    return Self::computeStridesTmaWarpSpecialized(
        expert_first_token_offset_, gemm1_tma_ws_input, gemm2_tma_ws_input, num_rows,
        expanded_num_rows, fc1_out_size, hidden_size, hidden_size, inter_size, num_experts_per_node,
        reinterpret_cast<T const*>(gemm1_input), reinterpret_cast<T const*>(gemm2_input),
        fc1_expert_weights, fc2_expert_weights, quant_params.fp8.dequant_fc1,
        quant_params.fp8.dequant_fc2, fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params,
        fc1_expert_biases, fc2_bias, reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
        reinterpret_cast<UnfusedGemmOutputType*>(gemm2_output), permuted_token_final_scales_,
        permuted_row_to_unpermuted_row_, enable_pdl, stream);
  }
}

// ==================== Helper for getting load balanced routing for profiling
// ==================================

__global__ void prepareFakeRouterBuffers(int* token_selected_experts, int64_t num_tokens, int64_t k,
                                         int64_t num_experts) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t sample = blockIdx.y;
  if (tid >= num_tokens) {
    return;
  }

  // Offset the buffers to the start of the sample
  token_selected_experts += sample * num_tokens * k;

  // This is not perf sensitive we just init the state here every time prepare is called
  // This means the first N tokens will always have the same distribution, regardless of num_tokens
  curandStatePhilox4_32_10_t state;
  curand_init(sample, tid, 0, &state);
  for (int k_idx = 0; k_idx < k; k_idx++) {
    while (true) {
      // curand_uniform includes 1 but not 0, so round up and subtract 1
      int expert = std::ceil(static_cast<float>(num_experts) * curand_uniform(&state)) - 1;

      bool valid = true;
      for (int prev_k = 0; prev_k < k_idx; prev_k++) {
        int prev_expert = token_selected_experts[k * tid + prev_k];
        if (expert == prev_expert) {
          valid = false;
          break;
        }
      }

      if (valid) {
        token_selected_experts[k * tid + k_idx] = expert;
        break;
      }
    }
  }
}

__global__ void populateRandomBufferKernel(void* buffer_void, size_t size) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) {
    return;
  }

  curandStatePhilox4_32_10_t state;
  curand_init(size, tid, 0, &state);

  constexpr int elem_per_thread = 128 / sizeof(uint4);
  auto* buffer = reinterpret_cast<uint4*>(buffer_void);
#pragma unroll
  for (int i = 0; i < elem_per_thread; i++) buffer[tid * elem_per_thread + i] = curand4(&state);
}

__global__ void initializeProfilerClampedRelu2Limit(float* limit) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *limit = 16.0f;
  }
}

template <int BLOCK_SIZE, int NUM_ROUTING_SAMPLES>
__global__ void prepareMinLatencyBuffer(int* num_active_experts_per_node,
                                        int* active_expert_global_ids,
                                        int64_t* expert_first_token_offset, int const num_tokens,
                                        int const num_experts_per_token,
                                        int const num_experts_per_node) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // 0. set offset
  num_active_experts_per_node += bid;
  active_expert_global_ids += bid * num_experts_per_node;
  expert_first_token_offset += bid * (num_experts_per_node + 1);

  // 1. set the num_active_experts_per_node
  int num_active = max(1, (int)(bid * ((float)num_experts_per_node / NUM_ROUTING_SAMPLES)));
  *num_active_experts_per_node = num_active;

  // 2. generate random active experts
  extern __shared__ float s_buf[];
  float* expert_refs = s_buf;
  int* expert_refs_idx = reinterpret_cast<int*>(expert_refs + num_experts_per_node);

  curandState_t local_state;
  curand_init(bid, tid, 0, &local_state);
  for (int i = tid; i < num_experts_per_node; i += BLOCK_SIZE) {
    expert_refs[i] = (float)curand_uniform(&local_state);
    expert_refs_idx[i] = (int)i;
  }
  __syncthreads();

  float thread_key[1];
  int thread_value[1];
  thread_key[0] = std::numeric_limits<float>::max();
  thread_value[0] = num_experts_per_node;

  if (tid < num_experts_per_node) {
    thread_key[0] = expert_refs[tid];
    thread_value[0] = expert_refs_idx[tid];
  }

  using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, 1, int>;
  using BlockRadixSortValue = cub::BlockRadixSort<int, BLOCK_SIZE, 1>;

  union TempStorage {
    typename BlockRadixSort::TempStorage key_value;
    typename BlockRadixSortValue::TempStorage value;
  };
  __shared__ union TempStorage temp_storage;

  BlockRadixSort(temp_storage.key_value).Sort(thread_key, thread_value);
  __syncthreads();

  if (tid > num_active) {
    thread_value[0] = std::numeric_limits<int>::max();
  }
  BlockRadixSortValue(temp_storage.value).Sort(thread_value);
  __syncthreads();

  // 3. set the active_expert_global_ids and expert_first_token_offset
  for (int i = tid; i < num_experts_per_node; i += BLOCK_SIZE) {
    if (i < num_active) {
      active_expert_global_ids[i] = thread_value[0];
      expert_first_token_offset[i] = i * num_tokens;
    } else {
      active_expert_global_ids[i] = -1;
      expert_first_token_offset[i] = num_active * num_tokens;
    }
  }
  if (tid == 0) {
    expert_first_token_offset[num_experts_per_node] = num_active * num_tokens;
  }
}

void populateRandomBuffer(void* buffer_void, size_t size, cudaStream_t stream) {
  // Each thread initialises 128 bytes
  TLLM_CHECK_WITH_INFO(size % 128 == 0, "Unexpected size alignment");
  auto threads = size / 128;
  populateRandomBufferKernel<<<ceilDiv(threads, 128), 128, 0, stream>>>(buffer_void, threads);
}

std::map<std::string, std::pair<size_t, size_t>> GemmProfilerBackend::getProfilerWorkspaces(
    int maxM, bool is_tma_ws_input) {
  size_t k = mK;
  size_t num_expanded_tokens = mMinLatencyMode ? maxM * mNumExpertsPerNode : maxM * k;

  TLLM_CHECK(mDType != nvinfer1::DataType::kINT4);
  // nvllm still uses int64 because torch doesn't have fp4 yet.
  bool is_4bit_act = mDType == nvinfer1::DataType::kFP4 || mDType == nvinfer1::DataType::kINT64;
  bool is_4bit_weight = mWType == nvinfer1::DataType::kINT4 || mWType == nvinfer1::DataType::kFP4 ||
                        mWType == nvinfer1::DataType::kINT64;
  TLLM_CHECK_WITH_INFO(!is_4bit_act || is_4bit_weight,
                       "Cannot have 4-bit activation with non-4-bit weight");
  float dtype_bytes = is_4bit_act ? 0.5f
                                  : static_cast<float>((mWType == nvinfer1::DataType::kINT4 ||
                                                        mWType == nvinfer1::DataType::kUINT8)
                                                           ? getDTypeSize(mOType)
                                                           : getDTypeSize(mDType));
  float weight_bytes = is_4bit_weight ? 0.5f : static_cast<float>(getDTypeSize(mWType));
  size_t output_bytes = getDTypeSize(mOType);
  size_t gemm_output_bytes =
      (mOType == nvinfer1::DataType::kFP8)
          ? sizeof(TmaWarpSpecializedGroupedGemmInput::OutputTypeAdaptor_t<__nv_fp8_e4m3>)
          : output_bytes;

  size_t hidden_size = mExpertHiddenSize;
  size_t inter_size = mExpertInterSize;  // Already divided by TP
  size_t num_experts_per_node = mNumExpertsPerNode;
#if defined(FLASHINFER_FP4_SW_CVT)
  if (is_4bit_act) {
    // Below SM100 the GEMMs take the 4-bit activations restored to 16 bits, K padded.
    dtype_bytes = 2.0f;
    hidden_size = padNvfp4K(hidden_size);
    inter_size = padNvfp4K(inter_size);
  }
#endif

  size_t fc1_out_size = inter_size;
  if (isGatedActivation(mActivationType)) {
    fc1_out_size = inter_size * 2;
  }

  // TODO Needs updated when gather/finalize fusion is integrated
  size_t input_size1 = hidden_size * num_expanded_tokens * dtype_bytes;
  size_t output_size1 = inter_size * num_expanded_tokens * dtype_bytes;

  size_t input_size2 = inter_size * num_expanded_tokens * dtype_bytes;
  size_t output_size2 = hidden_size * output_bytes;

  size_t input_size = mGemmToProfile == GemmToProfile::GEMM_1 ? input_size1 : input_size2;
  size_t output_size = mGemmToProfile == GemmToProfile::GEMM_1 ? output_size1 : output_size2;

  // This may allocate a pointer when not required. That's fine it will be ignored at the cost of
  // some memory
  size_t intermediate_size1 =
      fc1_out_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes
  size_t intermediate_size2 =
      hidden_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes

  size_t intermediate_size =
      mGemmToProfile == GemmToProfile::GEMM_1 ? intermediate_size1 : intermediate_size2;

  size_t weights_1 = hidden_size * fc1_out_size * num_experts_per_node * weight_bytes;
  size_t bias_1 = mBias ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  if (mUseLora && !is_tma_ws_input) bias_1 = output_size1;
  size_t weights_2 = hidden_size * inter_size * num_experts_per_node * weight_bytes;
  size_t bias_2 = mBias ? hidden_size * num_experts_per_node * dtype_bytes : 0;

  size_t weights_size =
      mNeedWeights ? (mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2) : 0;
  size_t bias_size = mGemmToProfile == GemmToProfile::GEMM_1 ? bias_1 : bias_2;

  // TODO Make quant 2 & 4 bigger for FP8 if we ever change to scaling per expert
  bool is_int_w_quant =
      (mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4 ||
       mWType == nvinfer1::DataType::kUINT8) &&
      mGroupSize <= 0;
  bool is_int_groupwise_w_quant =
      (mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4 ||
       mWType == nvinfer1::DataType::kUINT8) &&
      mGroupSize > 0;
  bool is_fp8_act_quant = mDType == nvinfer1::DataType::kFP8;
  bool is_fp8_w_quant = mWType == nvinfer1::DataType::kFP8;
  bool const is_native_wfp4afp8_family = isNativeWfp4Afp8Family();
  // NVFP4 uses the same six quant workspaces as wfp4afp8 (issue #4003).
  bool const is_native_wfp4afp4_family = isNativeWfp4Afp4Family();
  // This predicate identifies the SM90 FP8 activation x packed-MXFP4 storage
  // family.  Sm90Wfp4Afp8ScaleMode selects Humming/pre-MMA vs future post-MMA
  // semantics; do not infer the semantic path from dtype/layout alone.
  bool const is_sm90_wfp4afp8_family = isSm90Wfp4Afp8Family();
  checkSm90Wfp4Afp8ScaleMode();
  bool is_w4afp8_quant = is_int_groupwise_w_quant && is_fp8_act_quant && !is_sm90_wfp4afp8_family;
  bool is_wfp4a16_quant =
      (mDType == nvinfer1::DataType::kHALF || mDType == nvinfer1::DataType::kBF16) &&
      mWType == nvinfer1::DataType::kUINT8;

  // Int sizes
  size_t quant_1_size = is_int_w_quant ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  size_t quant_2_size = is_int_w_quant ? hidden_size * num_experts_per_node * dtype_bytes : 0;
  if (is_int_w_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes;
  } else if ((is_int_groupwise_w_quant && !is_sm90_wfp4afp8_family) || is_wfp4a16_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes * hidden_size / mGroupSize;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes * inter_size / mGroupSize;
  }

  // FP8 sizes
  quant_1_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : quant_1_size;
  quant_2_size = is_fp8_w_quant ? sizeof(float) : quant_2_size;
  size_t quant_3_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : 0;
  size_t quant_4_size = 0;  // Currently ignored by the GEMM
  if (is_int_groupwise_w_quant) {
    quant_3_size = quant_1_size;
    quant_4_size = quant_2_size;
  }

  // FP4 sizes
  bool const use_humming_pre_mma = isHummingPreMmaScaleMode();
  size_t quant_5_size = 0;
  size_t quant_6_size = 0;
  if (is_native_wfp4afp8_family || is_native_wfp4afp4_family) {
    quant_1_size = sizeof(float);
    // fc1 scale factors span fc1_out_size rows (2x inter_size when gated),
    // matching the gemm1_n the profiler consumes them with.
    quant_2_size =
        getOffsetWeightSF(num_experts_per_node, fc1_out_size, hidden_size, mScalingType) *
        sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
    quant_3_size = num_experts_per_node * sizeof(float);
    quant_4_size = sizeof(float);
    quant_5_size = getOffsetWeightSF(num_experts_per_node, hidden_size, inter_size, mScalingType) *
                   sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
    quant_6_size = num_experts_per_node * sizeof(float);
#if defined(FLASHINFER_FP4_SW_CVT)
    // 16-bit weight block scales [E, K / 16, N] for the SM80 GEMMs.
    quant_2_size *= 2;
    quant_5_size *= 2;
#endif
  } else if (is_sm90_wfp4afp8_family) {
    quant_1_size = sizeof(float);
    quant_2_size =
        getOffsetWeightSF(num_experts_per_node, fc1_out_size, hidden_size,
                          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX) *
        sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
    quant_3_size = num_experts_per_node * sizeof(float);
    quant_4_size = sizeof(float);
    quant_5_size =
        getOffsetWeightSF(num_experts_per_node, hidden_size, inter_size,
                          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX) *
        sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
    quant_6_size = num_experts_per_node * sizeof(float);
  }

  // MXFP8xMXFP8 sizes: the FP8 branch above only reserves per-tensor float
  // scalars, but block-scaled MXFP8 needs per-expert weight block SFs and
  // global scales (issue #3558). Layout mirrors QuantParams::MXFP8MXFP8:
  // quant_2/quant_5 = fc1/fc2 weight block SFs, quant_3/quant_6 = fc1/fc2
  // per-expert global scales. fc1 N uses fc1_out_size (doubled when gated).
  bool is_mxfp8_quant =
      is_fp8_act_quant && is_fp8_w_quant &&
      mScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
  if (is_mxfp8_quant) {
    quant_1_size = 0;
    quant_2_size =
        getOffsetWeightSF(num_experts_per_node, fc1_out_size, hidden_size, mScalingType) *
        sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF);
    quant_3_size = num_experts_per_node * sizeof(float);
    quant_4_size = 0;
    quant_5_size = getOffsetWeightSF(num_experts_per_node, hidden_size, inter_size, mScalingType) *
                   sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF);
    quant_6_size = num_experts_per_node * sizeof(float);
  }

  size_t tma_ws_input_workspace_size = 0;
  if (is_tma_ws_input) {
    tma_ws_input_workspace_size =
        TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node, mScalingType) *
        (NUM_ROUTING_SAMPLES * NUM_FUSION_TYPES * NUM_SWAP_AB_TYPES + 1);

    if (is_w4afp8_quant || is_wfp4a16_quant) {
      quant_3_size = 0;
      quant_4_size = 0;
    }
  }

  auto act_sf_rows = mMinLatencyMode ? num_expanded_tokens
                                     : std::min(num_expanded_tokens,
                                                static_cast<size_t>(maxM * num_experts_per_node));
  // getOffsetActivationSF returns zero if scaling_type is NONE
  size_t const fc1_fp4_act_scale_size =
      getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size, mScalingType) *
      sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
  size_t const fc2_fp4_act_scale_size =
      getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size, mScalingType) *
      sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
  size_t const fp4_act_scale_flat_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);
  size_t const fused_fc2_fp4_act_scale_size =
      profilesClampedRelu2Epilogue() && is_mxfp8_quant ? fc2_fp4_act_scale_size : 0;

  size_t w4a8_alpha_size =
      (is_w4afp8_quant || is_wfp4a16_quant) ? num_experts_per_node * sizeof(float) : 0;
  size_t alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);
  if (use_humming_pre_mma) {
    alpha_scale_ptr_array_size *= NUM_ROUTING_SAMPLES;
  }
  size_t gemm_workspace_size = mInterface->getGemmWorkspaceSize(num_experts_per_node);
  size_t precomputed_scheduler_workspace_size =
      (is_tma_ws_input && isSm90MixedInputFamily())
          ? cutlass_kernels_oss::detail::precomputed_scheduler_workspace_size(
                num_experts_per_node, num_expanded_tokens, std::max(hidden_size, fc1_out_size))
          : 0;

  // Routing info
  size_t expert_first_token_offset_size =
      (num_experts_per_node + 1) * sizeof(int64_t) * NUM_ROUTING_SAMPLES;
  size_t map_size = mMinLatencyMode ? 0 : NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t unpermuted_size =
      mMinLatencyMode ? 0 : NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t permuted_size = mMinLatencyMode ? 0 : num_expanded_tokens * sizeof(int);
  size_t token_topk_unpermuted_scales_size =
      mMinLatencyMode ? 0 : num_expanded_tokens * sizeof(float);
  int64_t const num_tokens_per_block = computeNumTokensPerBlock(maxM, num_experts_per_node);
  int64_t const num_blocks_per_seq = tensorrt_llm::common::ceilDiv(maxM, num_tokens_per_block);
  size_t const blocked_expert_counts_size =
      mMinLatencyMode ? 0 : num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size =
      mMinLatencyMode ? 0 : num_experts_per_node * maxM * sizeof(int);

  // The follow buffers are used in min_latency_mode
  size_t num_active_experts_per_node_size =
      mMinLatencyMode ? sizeof(int) * NUM_ROUTING_SAMPLES
                      : 0;  // smaller than or equal to num_experts_per_node
  size_t active_expert_global_ids_size =
      mMinLatencyMode ? mNumExpertsPerNode * sizeof(int) * NUM_ROUTING_SAMPLES : 0;

  bool is_swiglu_bias =
      mActivationType == ActivationType::SwigluBias && mGemmToProfile == GemmToProfile::GEMM_1;
  bool const is_clamped_relu2 =
      mActivationType == ActivationType::ClampedRelu2 && mGemmToProfile == GemmToProfile::GEMM_1;
  size_t swiglu_alpha_size = is_swiglu_bias ? num_experts_per_node * sizeof(float) : 0;
  size_t swiglu_beta_size = is_swiglu_bias ? num_experts_per_node * sizeof(float) : 0;
  size_t swiglu_limit_size = is_swiglu_bias ? num_experts_per_node * sizeof(float)
                                            : (is_clamped_relu2 ? sizeof(float) : 0);

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                              \
  do {                                                    \
    auto aligned_size = alignSize(size, kCudaMemAlign);   \
    out_map[#name] = std::pair{aligned_size, map_offset}; \
    map_offset += aligned_size;                           \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(expert_first_token_offset);
  ADD_NAME(unpermuted_row_to_permuted_row, map_size);
  ADD_NAME(permuted_row_to_unpermuted_row, map_size);
  ADD_NAME(token_selected_experts, unpermuted_size);
  ADD_NAME(permuted_token_selected_experts, permuted_size);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(token_topk_unpermuted_scales);
  ADD_NAME(act_fp8_token_scale,
           isHummingPreMmaScaleMode() ? num_expanded_tokens * sizeof(float) : 0);
  ADD(num_active_experts_per_node);
  ADD(active_expert_global_ids);
  ADD(input);
  ADD(output);
  ADD(intermediate);
  ADD(weights);
  ADD(bias);
  ADD(quant_1);
  ADD(quant_2);
  ADD(quant_3);
  ADD(quant_4);
  ADD(quant_5);
  ADD(quant_6);
  ADD(tma_ws_input_workspace);
  ADD(w4a8_alpha);
  ADD(alpha_scale_ptr_array);
  ADD(fp4_act_scale_flat);
  ADD(fused_fc2_fp4_act_scale);
  ADD(precomputed_scheduler_workspace);
  ADD(gemm_workspace);
  ADD(swiglu_alpha);
  ADD(swiglu_beta);
  ADD(swiglu_limit);
#undef ADD_NAME
#undef ADD

  return out_map;
}

void GemmProfilerBackend::prepareRouting(int num_tokens, char* workspace_ptr_char, bool enable_pdl,
                                         cudaStream_t stream) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR_BASE(type, name)                                                   \
  auto* name##_base =                                                                 \
      (workspaces.at(#name).first                                                     \
           ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
           : nullptr)
#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR_BASE(int64_t*, expert_first_token_offset);
  GET_WS_PTR_BASE(int*, unpermuted_row_to_permuted_row);
  GET_WS_PTR_BASE(int*, permuted_row_to_unpermuted_row);
  GET_WS_PTR_BASE(int*, token_selected_experts);
  GET_WS_PTR(int*, permuted_token_selected_experts);
  GET_WS_PTR(int*, blocked_expert_counts);
  GET_WS_PTR(int*, blocked_expert_counts_cumsum);
  GET_WS_PTR(int*, blocked_row_to_unpermuted_row);
  GET_WS_PTR(int*, num_active_experts_per_node);
  GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR_BASE
#undef GET_WS_PTR

  if (mMinLatencyMode) {
    // expert_first_token_offset for each sample
    TLLM_CHECK_WITH_INFO(mNumExpertsPerNode <= 256,
                         "Min latency mode only supports #experts < 256");
    prepareMinLatencyBuffer<256, NUM_ROUTING_SAMPLES>
        <<<NUM_ROUTING_SAMPLES, 256, mNumExpertsPerNode * (sizeof(float) + sizeof(int)), stream>>>(
            num_active_experts_per_node, active_expert_global_ids, expert_first_token_offset_base,
            num_tokens, mK, mNumExpertsPerNode);
  } else {
    int64_t const num_expanded_tokens = num_tokens * mK;
    int const start_expert_id = mNumExpertsPerNode * mParallelismConfig.ep_rank;

    uint32_t num_threads = 256;
    dim3 grid_dim{(num_tokens + num_threads - 1) / num_threads, NUM_ROUTING_SAMPLES, 1};
    prepareFakeRouterBuffers<<<grid_dim, num_threads, 0, stream>>>(token_selected_experts_base,
                                                                   num_tokens, mK, mNumExperts);
    sync_check_cuda_error(stream);

    for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
      int64_t* expert_first_token_offset =
          expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
      int* unpermuted_row_to_permuted_row =
          unpermuted_row_to_permuted_row_base + i * num_expanded_tokens;
      int* permuted_row_to_unpermuted_row =
          permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;
      int* token_selected_experts = token_selected_experts_base + i * num_expanded_tokens;

      threeStepBuildExpertMapsSortFirstToken(
          token_selected_experts, permuted_token_selected_experts, permuted_row_to_unpermuted_row,
          unpermuted_row_to_permuted_row, expert_first_token_offset, blocked_expert_counts,
          blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row, num_tokens,
          mNumExpertsPerNode, mK, start_expert_id, enable_pdl, stream);
      sync_check_cuda_error(stream);
    }
  }
}

void GemmProfilerBackend::prepareQuantParams(int num_tokens, char* workspace_ptr_char,
                                             cudaStream_t stream) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)
  GET_WS_PTR(void const*, quant_1);
  GET_WS_PTR(void const*, quant_2);
  GET_WS_PTR(void const*, quant_3);
  GET_WS_PTR(void const*, quant_4);
  GET_WS_PTR(void const*, quant_5);
  GET_WS_PTR(void const*, quant_6);
  GET_WS_PTR(float const*, w4a8_alpha);
#undef GET_WS_PTR

  checkSm90Wfp4Afp8ScaleMode();

  if (isSm90Wfp4Afp8Family() && mSm90Wfp4Afp8Mode == Sm90Wfp4Afp8ScaleMode::kPostMmaMxfp8Act) {
    TLLM_CHECK(quant_2 && quant_3 && quant_5 && quant_6);
    // UE8M0 unity weight block scales, as for MXFP8 x MXFP4 from SM100 below.
    TLLM_CUDA_CHECK(
        cudaMemsetAsync(const_cast<void*>(quant_2), 0x7F, workspaces.at("quant_2").first, stream));
    TLLM_CUDA_CHECK(
        cudaMemsetAsync(const_cast<void*>(quant_5), 0x7F, workspaces.at("quant_5").first, stream));
    mQuantParams = QuantParams::MXFP8MXFP4(
        static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_2),
        static_cast<float const*>(quant_3),
        static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_5),
        static_cast<float const*>(quant_6));
    // The SM90 grouped GEMM reads the weight block scales through fp8_mxfp4.
    mQuantParams.fp8_mxfp4.fc1.weight_block_scale = mQuantParams.mxfp8_mxfp4.fc1.weight_block_scale;
    mQuantParams.fp8_mxfp4.fc2.weight_block_scale = mQuantParams.mxfp8_mxfp4.fc2.weight_block_scale;
  } else if (isSm90Wfp4Afp8Family()) {
    TLLM_CHECK(quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP8MXFP4(
        nullptr, static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_2),
        static_cast<float const*>(quant_3),
        isHummingPreMmaScaleMode() ? nullptr : static_cast<float const*>(quant_4),
        static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_5),
        static_cast<float const*>(quant_6), false, false);
  } else if ((mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4 ||
              mWType == nvinfer1::DataType::kUINT8) &&
             mGroupSize < 0) {
    TLLM_CHECK(quant_1 && quant_2);
    mQuantParams = QuantParams::Int(quant_1, quant_2);
  } else if (mWType == nvinfer1::DataType::kINT4 || mWType == nvinfer1::DataType::kUINT8) {
    TLLM_CHECK(quant_1 && quant_2);
    if (mDType == nvinfer1::DataType::kFP8 ||
        (mWType == nvinfer1::DataType::kUINT8 &&
         (mDType == nvinfer1::DataType::kHALF || mDType == nvinfer1::DataType::kBF16))) {
      TLLM_CHECK(w4a8_alpha);
      mQuantParams = QuantParams::GroupWise(mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3,
                                            quant_4, w4a8_alpha, w4a8_alpha);
    } else {
      mQuantParams =
          QuantParams::GroupWise(mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3, quant_4);
    }
  } else if (mDType == nvinfer1::DataType::kFP8 && mWType == nvinfer1::DataType::kFP8 &&
             mScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX) {
#ifdef USING_OSS_CUTLASS_MOE_GEMM
    // MXFP8xMXFP8 must be matched before the generic FP8 branch below: the
    // storage dtypes are identical and only mScalingType distinguishes them
    // (issue #3558).
    TLLM_CHECK(quant_2 && quant_3 && quant_5 && quant_6);
    // Initialize the weight block-SF buffers to E8M0 1.0 (biased exponent
    // 0x7F): the profiler never fills them with real data, and uninitialized
    // exponents make the timed kernels read nondeterministic scale patterns,
    // which can skew tactic rankings vs real traffic.
    TLLM_CUDA_CHECK(
        cudaMemsetAsync(const_cast<void*>(quant_2), 0x7F, workspaces.at("quant_2").first, stream));
    TLLM_CUDA_CHECK(
        cudaMemsetAsync(const_cast<void*>(quant_5), 0x7F, workspaces.at("quant_5").first, stream));
    mQuantParams = QuantParams::MXFP8MXFP8(
        static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_2),
        static_cast<float const*>(quant_3),
        static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_5),
        static_cast<float const*>(quant_6));
    // Backfill the shared FP8 dequant aliases: prepareTmaWsInputs() and the
    // common GEMM/TMA setup read mQuantParams.fp8.dequant_fc1/fc2 directly,
    // mirroring the WMXFP8AMXFP8 remap that runMoe() performs.
    mQuantParams.fp8.dequant_fc1 = static_cast<float const*>(quant_3);
    mQuantParams.fp8.dequant_fc2 = static_cast<float const*>(quant_6);
#else
    TLLM_CHECK_WITH_INFO(false, "MXFP8 x MXFP8 profiling requires OSS Cutlass MoE GEMM");
#endif
  } else if (mWType == nvinfer1::DataType::kFP8) {
    TLLM_CHECK(quant_1 && quant_2 && quant_3);
    mQuantParams =
        QuantParams::FP8(static_cast<float const*>(quant_1), static_cast<float const*>(quant_2),
                         static_cast<float const*>(quant_3), static_cast<float const*>(quant_4));
  } else if (mDType == nvinfer1::DataType::kFP8 &&
             (mWType == nvinfer1::DataType::kFP4 || mWType == nvinfer1::DataType::kINT64)) {
    TLLM_CHECK(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    if (mUseMxfp8ActScaling) {
#ifdef USING_OSS_CUTLASS_MOE_GEMM
      // Normalize the profiler's fabricated weight block scale factors to UE8M0 unity so tactic
      // timing is not driven by random exponents from the workspace.
      TLLM_CUDA_CHECK(cudaMemsetAsync(const_cast<void*>(quant_2), 0x7F,
                                      workspaces.at("quant_2").first, stream));
      TLLM_CUDA_CHECK(cudaMemsetAsync(const_cast<void*>(quant_5), 0x7F,
                                      workspaces.at("quant_5").first, stream));
      mQuantParams = QuantParams::MXFP8MXFP4(
          static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_2),
          static_cast<float const*>(quant_3),
          static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_5),
          static_cast<float const*>(quant_6));
#else
      TLLM_CHECK_WITH_INFO(false, "MXFP8 x MXFP4 profiling requires OSS Cutlass MoE GEMM");
#endif
    } else {
      mQuantParams = QuantParams::FP8MXFP4(
          static_cast<float const*>(quant_1),
          static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_2),
          static_cast<float const*>(quant_3), static_cast<float const*>(quant_4),
          static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_5),
          static_cast<float const*>(quant_6));
    }
  } else if (isNativeWfp4Afp4Family()) {
    // nvllm still uses int64 because torch doesn't have fp4 yet.
    TLLM_CHECK(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP4(
        static_cast<float const*>(quant_1),
        static_cast<TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(quant_2),
        static_cast<float const*>(quant_3), static_cast<float const*>(quant_4),
        static_cast<TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(quant_5),
        static_cast<float const*>(quant_6));
#if defined(FLASHINFER_FP4_SW_CVT)
    mQuantParams.groupwise =
        QuantParams::GroupWise(TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize,
                               quant_2, quant_5)
            .groupwise;
#endif
  }
}

void GemmProfilerBackend::prepareTmaWsInputs(
    int num_tokens, char* workspace_ptr_char, void const* expert_weights,
    TmaWarpSpecializedGroupedGemmInput::EpilogueFusion fusion, bool swap_ab, bool enable_pdl,
    cudaStream_t stream) {
  if (mSM < 90) {
    return;
  }
#if defined(FLASHINFER_FP4_SW_CVT)
  if (isNativeWfp4Afp4Family()) {
    return;  // Below SM100 NVFP4 runs on the SM80 GEMMs.
  }
#endif

  bool const use_sm90_mixed_input_gemm = isSm90MixedInputFamily();
  bool const use_finalize_fusion =
      fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
  bool const use_activation_fusion =
      fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE && !swap_ab &&
      profilesClampedRelu2Epilogue();
  bool const finalize_fusion_not_supported = !mInterface->use_fused_finalize_ || mMinLatencyMode ||
                                             use_sm90_mixed_input_gemm ||
                                             mGemmToProfile != GemmToProfile::GEMM_2;
  if (use_finalize_fusion && finalize_fusion_not_supported) {
    return;
  }

  if (use_sm90_mixed_input_gemm && !swap_ab) {
    return;
  }

  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);

#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR(int64_t*, expert_first_token_offset);
  int64_t* expert_first_token_offset_base = expert_first_token_offset;
  GET_WS_PTR(int*, permuted_row_to_unpermuted_row);
  int* permuted_row_to_unpermuted_row_base = permuted_row_to_unpermuted_row;
  GET_WS_PTR(void*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void*, weights);
  TLLM_CHECK(mNeedWeights == (expert_weights == nullptr));
  void const* weights_sel = mNeedWeights ? weights : expert_weights;
  GET_WS_PTR(void*, bias);
  GET_WS_PTR(float*, token_topk_unpermuted_scales);
  GET_WS_PTR(int8_t*, tma_ws_input_workspace);
  GET_WS_PTR(void*, gemm_workspace);
  GET_WS_PTR(void*, precomputed_scheduler_workspace);
  GET_WS_PTR(float*, alpha_scale_ptr_array);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fused_fc2_fp4_act_scale);
  GET_WS_PTR(float*, swiglu_limit);
  GET_WS_PTR(int*, num_active_experts_per_node);
  GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR

  // For MXFP8, fill the activation block-SF buffer with E8M0 1.0 (biased
  // exponent 0x7F). The profiler never quantizes real activations into it,
  // and uninitialized exponents give the timed kernels a nondeterministic
  // scale pattern, skewing tactic rankings relative to real traffic.
  if (fp4_act_scale_flat != nullptr &&
      mScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX) {
    TLLM_CUDA_CHECK(cudaMemsetAsync(fp4_act_scale_flat, 0x7F,
                                    workspaces.at("fp4_act_scale_flat").first, stream));
  }
  if (fused_fc2_fp4_act_scale != nullptr) {
    TLLM_CUDA_CHECK(cudaMemsetAsync(fused_fc2_fp4_act_scale, 0x7F,
                                    workspaces.at("fused_fc2_fp4_act_scale").first, stream));
  }
  if (use_activation_fusion) {
    TLLM_CHECK(swiglu_limit != nullptr);
    initializeProfilerClampedRelu2Limit<<<1, 1, 0, stream>>>(swiglu_limit);
  }

  size_t tma_ws_size =
      TmaWarpSpecializedGroupedGemmInput::workspaceSize(mNumExpertsPerNode, mScalingType);

  TmaWarpSpecializedGroupedGemmInput dummy_tma_ws_input;
  dummy_tma_ws_input.configureWorkspace(
      tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
      workspaces.at("gemm_workspace").first, precomputed_scheduler_workspace,
      workspaces.at("precomputed_scheduler_workspace").first, mScalingType);
  dummy_tma_ws_input.enable_pdl = enable_pdl;  // Set enable_pdl for dummy input
  tma_ws_input_workspace += tma_ws_size;

  int workspace_index =
      static_cast<int>(use_finalize_fusion) * (NUM_SWAP_AB_TYPES * NUM_ROUTING_SAMPLES) +
      static_cast<int>(swap_ab) * NUM_ROUTING_SAMPLES;
  tma_ws_input_workspace += workspace_index * tma_ws_size;

  size_t num_expanded_tokens = num_tokens * mK;
  dummy_tma_ws_input.precomputed_scheduler_total_routed_tokens = num_expanded_tokens;
  for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
    // Note: Even though we have separate TMA WS inputs for finalize fusion on/off we reuse the same
    // pointers to save space.
    auto& cache_element = mTmaInputCache[use_finalize_fusion][swap_ab][i];
    cache_element.configureWorkspace(
        tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
        workspaces.at("gemm_workspace").first, precomputed_scheduler_workspace,
        workspaces.at("precomputed_scheduler_workspace").first, mScalingType);
    cache_element.enable_pdl = enable_pdl;  // Set enable_pdl for cache element
    cache_element.precomputed_scheduler_total_routed_tokens = num_expanded_tokens;
    tma_ws_input_workspace += tma_ws_size;

    int64_t* expert_first_token_offset =
        expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
    int* permuted_row_to_unpermuted_row =
        permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;

    auto& gemm1_tma_ws_input =
        mGemmToProfile == GemmToProfile::GEMM_1 ? cache_element : dummy_tma_ws_input;
    auto& gemm2_tma_ws_input =
        mGemmToProfile == GemmToProfile::GEMM_2 ? cache_element : dummy_tma_ws_input;
    if (mSM >= 90) {
      auto fc1_output_size =
          isGatedActivation(mActivationType) ? mExpertInterSize * 2 : mExpertInterSize;

      /* GEMM1 */
      gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
      gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

      gemm1_tma_ws_input.swap_ab = swap_ab;
      gemm2_tma_ws_input.swap_ab = swap_ab;

      if (use_finalize_fusion) {
        assert(!mMinLatencyMode);
        gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
        gemm2_tma_ws_input.setFinalizeFusionParams(output, mExpertUnpaddedHiddenSize, num_tokens,
                                                   mK > 1);
      }
      if (use_activation_fusion) {
        gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION;
        gemm1_tma_ws_input.fused_activation_epilogue.clamp_limit = swiglu_limit;
      }

      auto* const fc2_activation_scale = use_activation_fusion && fused_fc2_fp4_act_scale != nullptr
                                             ? fused_fc2_fp4_act_scale
                                             : fp4_act_scale_flat;

      if (mMinLatencyMode) {
        std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) =
            mInterface->computeStridesTmaWarpSpecializedLowLatencyDispatch(
                gemm1_tma_ws_input, gemm2_tma_ws_input, num_tokens, fc1_output_size,
                mExpertHiddenSize, mExpertHiddenSize, mExpertInterSize, mNumExpertsPerNode, input,
                input, weights_sel, weights_sel, mQuantParams.fp8.dequant_fc1,
                mQuantParams.fp8.dequant_fc2, fp4_act_scale_flat, fc2_activation_scale,
                mQuantParams, nullptr, nullptr, intermediate, intermediate,
                num_active_experts_per_node, active_expert_global_ids, 0, enable_pdl, stream);
      } else {
        std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) =
            mInterface->computeStridesTmaWarpSpecializedDispatch(
                expert_first_token_offset, gemm1_tma_ws_input, gemm2_tma_ws_input, num_tokens,
                num_tokens * mK, fc1_output_size, mExpertHiddenSize, mExpertHiddenSize,
                mExpertInterSize, mNumExpertsPerNode, input, input, weights_sel, weights_sel,
                mQuantParams.fp8.dequant_fc1, mQuantParams.fp8.dequant_fc2, fp4_act_scale_flat,
                fc2_activation_scale, mQuantParams, nullptr, nullptr, intermediate, intermediate,
                token_topk_unpermuted_scales, permuted_row_to_unpermuted_row, enable_pdl, stream);
      }
      sync_check_cuda_error(stream);
    }
  }
}

void GemmProfilerBackend::prepare(int num_tokens, char* workspace_ptr_char,
                                  void const* expert_weights, bool enable_pdl,
                                  cudaStream_t stream) {
  mSampleIndex = 0;

  auto workspace_size = getWorkspaceSize(num_tokens);
  populateRandomBuffer(workspace_ptr_char, workspace_size, stream);

  prepareRouting(num_tokens, workspace_ptr_char, enable_pdl, stream);
  prepareQuantParams(num_tokens, workspace_ptr_char, stream);
  if (isHummingPreMmaScaleMode()) {
    auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
    auto* expert_first_token_offset = reinterpret_cast<int64_t*>(
        workspace_ptr_char + workspaces.at("expert_first_token_offset").second);
    auto* act_fp8_token_scale =
        reinterpret_cast<float*>(workspace_ptr_char + workspaces.at("act_fp8_token_scale").second);
    auto* alpha_scale_ptr_array = reinterpret_cast<float const**>(
        workspace_ptr_char + workspaces.at("alpha_scale_ptr_array").second);
    prepareProfilerFP8TokenScalePtrArray(alpha_scale_ptr_array, act_fp8_token_scale,
                                         expert_first_token_offset, mNumExpertsPerNode,
                                         NUM_ROUTING_SAMPLES, stream);
    sync_check_cuda_error(stream);
  }
  for (auto fusion : {TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE,
                      TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE}) {
    for (auto swap_ab : {false, true}) {
      prepareTmaWsInputs(num_tokens, workspace_ptr_char, expert_weights, fusion, swap_ab,
                         enable_pdl, stream);
    }
  }
}

size_t GemmProfilerBackend::getWorkspaceSize(int maxM) {
  auto sizes_map = getProfilerWorkspaces(maxM, mSM >= 90);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(),
                 [](auto& v) { return v.second.first; });
  size_t size = calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  TLLM_LOG_TRACE("MOE profiler workspace size: %zu", size);
  return size;
}

void GemmProfilerBackend::runProfiler(int original_num_tokens, Config const& tactic,
                                      char* workspace_ptr_char, void const* expert_weights,
                                      bool enable_pdl, cudaStream_t const& stream) {
  int64_t expanded_num_tokens = original_num_tokens * mK;
  int64_t num_experts_per_node = mNumExpertsPerNode;

  mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

  auto workspaces = getProfilerWorkspaces(original_num_tokens, tactic.is_tma_warp_specialized);

#define GET_WS_PTR_OFFSET(type, name, offset)                                                    \
  auto* name =                                                                                   \
      (workspaces.at(#name).first                                                                \
           ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) + (offset) \
           : nullptr)
#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR_OFFSET(int64_t const*, expert_first_token_offset,
                    (mSampleIndex * (mNumExpertsPerNode + 1)));
  GET_WS_PTR_OFFSET(int const*, unpermuted_row_to_permuted_row,
                    (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(int const*, permuted_row_to_unpermuted_row,
                    (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(int const*, token_selected_experts, (mSampleIndex * expanded_num_tokens));

  GET_WS_PTR(float const*, token_topk_unpermuted_scales);
  auto const* token_topk_permuted_scales = token_topk_unpermuted_scales;
  GET_WS_PTR(float*, act_fp8_token_scale);

  GET_WS_PTR_OFFSET(int*, num_active_experts_per_node, mSampleIndex);
  GET_WS_PTR_OFFSET(int*, active_expert_global_ids, (mSampleIndex * mNumExpertsPerNode));
  GET_WS_PTR(void const*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void const*, weights);
  TLLM_CHECK(mNeedWeights == (expert_weights == nullptr));
  void const* weights_sel = mNeedWeights ? weights : expert_weights;
  GET_WS_PTR(void const*, bias);

  GET_WS_PTR_OFFSET(float const**, alpha_scale_ptr_array,
                    (isHummingPreMmaScaleMode() ? mSampleIndex * mNumExpertsPerNode : 0));
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fused_fc2_fp4_act_scale);
  GET_WS_PTR(void*, gemm_workspace);

  GET_WS_PTR(float*, swiglu_alpha);
  GET_WS_PTR(float*, swiglu_beta);
  GET_WS_PTR(float*, swiglu_limit);

#undef GET_WS_PTR_OFFSET
#undef GET_WS_PTR

  checkSm90Wfp4Afp8ScaleMode();

  TmaWarpSpecializedGroupedGemmInput tma_ws_input_template;
  if (tactic.is_tma_warp_specialized) {
    // Use non-finalize cache when finalize fusion is not supported for the current GEMM
    bool finalize_supported_this_gemm = (mGemmToProfile == GemmToProfile::GEMM_2) &&
                                        mInterface->use_fused_finalize_ && !mMinLatencyMode &&
                                        !isSm90MixedInputFamily();
    bool request_finalize = tactic.epilogue_fusion_type ==
                            cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
    bool use_finalize_index = request_finalize && finalize_supported_this_gemm;

    tma_ws_input_template = mTmaInputCache[use_finalize_index][tactic.swap_ab][mSampleIndex];
    TLLM_CHECK_WITH_INFO(tma_ws_input_template.isValid(),
                         "TMA WS input template is not initialized");
  }

  mInterface->is_profiler = true;
  auto const* profiler_fc2_activation_quant_scale =
      isHummingPreMmaScaleMode() ? nullptr
                                 : (mQuantParams.fp8_mxfp4.fc2.act_global_scale
                                        ? mQuantParams.fp8_mxfp4.fc2.act_global_scale
                                        : mQuantParams.fp8.quant_fc2);
  if (mGemmToProfile == GemmToProfile::GEMM_1) {
    auto* const fc2_activation_scale =
        tma_ws_input_template.fusion ==
                    TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION &&
                fused_fc2_fp4_act_scale != nullptr
            ? fused_fc2_fp4_act_scale
            : fp4_act_scale_flat;
    auto const activation_params =
        mActivationType == ActivationType::ClampedRelu2
            ? ActivationParams::ClampedRelu2(swiglu_limit)
            : ActivationParams(mActivationType, swiglu_alpha, swiglu_beta, swiglu_limit);
    mInterface->gemm1(input,                                             //
                      output,                                            //
                      intermediate,                                      //
                      expert_first_token_offset,                         //
                      tma_ws_input_template,                             //
                      weights_sel,                                       //
                      bias,                                              //
                      expert_first_token_offset + num_experts_per_node,  //
                      mQuantParams.wo.fc1_weight_scales,                 //
                      mQuantParams.fp8.dequant_fc1,                      //
                      profiler_fc2_activation_quant_scale,               //
                      act_fp8_token_scale,                               //
                      fp4_act_scale_flat,                                //
                      fc2_activation_scale,                              //
                      mQuantParams,                                      //
                      original_num_tokens,                               //
                      expanded_num_tokens,                               //
                      mExpertHiddenSize,                                 //
                      mExpertInterSize,                                  //
                      num_experts_per_node,                              //
                      activation_params,                                 //
                      alpha_scale_ptr_array,                             //
                      !mUseLora,                                         //
                      /*use_deepseek_fp8_block_scale=*/false,            //
                      stream,                                            //
                      tactic,                                            //
                      mMinLatencyMode,                                   //
                      num_active_experts_per_node,                       //
                      active_expert_global_ids,                          //
                      enable_pdl);                                       //
  } else {
    TLLM_CHECK(mGemmToProfile == GemmToProfile::GEMM_2);
    mInterface->gemm2(input,                                           //
                      intermediate,                                    //
                      output,                                          //
                      expert_first_token_offset,                       //
                      tma_ws_input_template,                           //
                      weights_sel,                                     //
                      bias,                                            //
                      mQuantParams.wo.fc2_weight_scales,               //
                      mQuantParams.fp8.dequant_fc2,                    //
                      fp4_act_scale_flat,                              //
                      mQuantParams,                                    //
                      token_topk_unpermuted_scales,                    //
                      token_topk_permuted_scales,                      //
                      unpermuted_row_to_permuted_row,                  //
                      permuted_row_to_unpermuted_row,                  //
                      token_selected_experts,                          //
                      expert_first_token_offset + mNumExpertsPerNode,  //
                      original_num_tokens,                             //
                      expanded_num_tokens,                             //
                      mExpertHiddenSize,                               //
                      mExpertUnpaddedHiddenSize,                       //
                      mExpertInterSize,                                //
                      num_experts_per_node,                            //
                      mK,                                              //
                      alpha_scale_ptr_array,                           //
                      false,                                           //
                      nullptr,                                         //
                      /*use_deepseek_fp8_block_scale=*/false,          //
                      stream,                                          //
                      mParallelismConfig,                              //
                      mEnableAlltoall,                                 //
                      tactic,                                          //
                      mMinLatencyMode,                                 //
                      num_active_experts_per_node,                     //
                      active_expert_global_ids,                        //
                      enable_pdl);                                     //
  }
  mInterface->is_profiler = false;

  sync_check_cuda_error(stream);
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels
