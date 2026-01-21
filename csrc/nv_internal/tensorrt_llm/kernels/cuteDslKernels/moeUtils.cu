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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cuteDslKernels/moeUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cuh"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif
#include <cute/numeric/numeric_types.hpp>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cute_dsl {
namespace {
using ElemCopyType = uint4;
using SFCopyType = uint32_t;

template <typename T>
auto constexpr bitsPerElem() {
#ifdef ENABLE_FP4
  return std::is_same_v<T, __nv_fp4_e2m1> ? 4 : cute::sizeof_bits_v<T>;
#else
  return cute::sizeof_bits_v<T>;
#endif
}

template <typename T>
auto constexpr elemPerCopy() {
  return bitsPerElem<ElemCopyType>() / bitsPerElem<T>();
}

template <typename T>
auto constexpr sfElemPerCopy() {
  return bitsPerElem<SFCopyType>() / bitsPerElem<T>();
}

// Helper to get max active blocks per SM
template <typename KernelFunc>
int32_t getMaxActiveBlocksPerSM(KernelFunc kernel, int32_t threadsPerBlock,
                                size_t dynamicSmemSize) {
  int numBlocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, threadsPerBlock,
                                                dynamicSmemSize);
  return numBlocks;
}

}  // namespace

template <typename InputType, typename SFType, int32_t kSFVecSize, int32_t kThreadsPerBlock>
__global__ void moePermuteKernel(InputType const* input, InputType* permuted_output,
                                 SFType const* input_sf, SFType* permuted_sf,
                                 int32_t const* tile_idx_to_mn_limit,
                                 int32_t const* permuted_idx_to_expanded_idx,
                                 int32_t const* num_non_exiting_tiles, int32_t const hidden_size,
                                 int32_t const top_k, int32_t const tile_size) {
  int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
  [[maybe_unused]] int32_t constexpr kSFElemPerCopy = sfElemPerCopy<SFType>();
  // Need int64_t to prevent overflow when computing pointer offsets.
  int64_t const kCopyPerToken = hidden_size / kElemPerCopy;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
  for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x) {
    int32_t const tile_idx = permuted_idx / tile_size;
    if (permuted_idx >= tile_idx_to_mn_limit[tile_idx]) {
      continue;
    }
    int32_t const expanded_idx = permuted_idx_to_expanded_idx[permuted_idx];
    int32_t const token_idx = expanded_idx / top_k;

    auto const* src_ptr = reinterpret_cast<ElemCopyType const*>(input) + token_idx * kCopyPerToken;
    auto* dst_ptr = reinterpret_cast<ElemCopyType*>(permuted_output) + permuted_idx * kCopyPerToken;
    for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock) {
      dst_ptr[i] = src_ptr[i];
    }

    // Note: FP4 scale factor handling is deferred to Phase 3
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename SFType>
void moePermute(InputType const* input, InputType* permuted_output, SFType const* input_sf,
                SFType* permuted_sf, int32_t const* tile_idx_to_mn_limit,
                int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,
                int32_t const max_num_permuted_tokens, int32_t const hidden_size,
                int32_t const top_k, int32_t const tile_size, bool enable_pdl,
                cudaStream_t stream) {
  int32_t constexpr kThreadsPerBlock = 256;
  int32_t constexpr kSFVecSize = 16;
  int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
  TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.",
                       kElemPerCopy);

  auto kernel = &moePermuteKernel<InputType, SFType, kSFVecSize, kThreadsPerBlock>;
  static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
  int32_t const maxBlocksPerSM = getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
  int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
  int32_t const threads = kThreadsPerBlock;

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
  cudaLaunchKernelEx(&config, kernel, input, permuted_output, input_sf, permuted_sf,
                     tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles,
                     hidden_size, top_k, tile_size);
}

#define INSTANTIATE_MOE_PERMUTE(InputType, SFType)                                           \
  template void moePermute<InputType, SFType>(                                               \
      InputType const* input, InputType* permuted_output, SFType const* input_sf,            \
      SFType* permuted_sf, int32_t const* tile_idx_to_mn_limit,                              \
      int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,     \
      int32_t const max_num_permuted_tokens, int32_t const hidden_size, int32_t const top_k, \
      int32_t const tile_size, bool enable_pdl, cudaStream_t stream)

INSTANTIATE_MOE_PERMUTE(half, uint8_t);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_PERMUTE(__nv_bfloat16, uint8_t);
#endif
#ifdef ENABLE_FP8
INSTANTIATE_MOE_PERMUTE(__nv_fp8_e4m3, uint8_t);
#endif
#ifdef ENABLE_FP4
INSTANTIATE_MOE_PERMUTE(__nv_fp4_e2m1, uint8_t);
#endif
#undef INSTANTIATE_MOE_PERMUTE

template <typename InputType, typename TopKScaleType, int32_t kThreadsPerBlock>
__global__ void moeUnpermuteKernel(InputType const* permuted_input, InputType* output,
                                   int32_t const* expanded_idx_to_permuted_idx,
                                   TopKScaleType const* topk_scales, int32_t const hidden_size,
                                   int32_t const top_k) {
  using AccumType = float;
  int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
  // Need int64_t to prevent overflow when computing pointer offsets.
  int64_t const kCopyPerToken = hidden_size / kElemPerCopy;
  InputType rmem[kElemPerCopy];
  AccumType rmemAccum[kElemPerCopy];

  int32_t const token_idx = blockIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  auto* dst_ptr = reinterpret_cast<ElemCopyType*>(output) + token_idx * kCopyPerToken;
  for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock) {
#pragma unroll
    for (int32_t j = 0; j < kElemPerCopy; j++) {
      rmemAccum[j] = 0;
    }
    for (int32_t k = 0; k < top_k; k++) {
      int32_t const permuted_idx = expanded_idx_to_permuted_idx[token_idx * top_k + k];
      if (permuted_idx < 0) {
        continue;
      }
      auto const* src_ptr =
          reinterpret_cast<ElemCopyType const*>(permuted_input) + permuted_idx * kCopyPerToken;
      *reinterpret_cast<ElemCopyType*>(rmem) = src_ptr[i];
      TopKScaleType const scale = topk_scales[token_idx * top_k + k];

#pragma unroll
      for (int32_t j = 0; j < kElemPerCopy; j++) {
        rmemAccum[j] += static_cast<AccumType>(rmem[j]) * static_cast<AccumType>(scale);
      }
    }
#pragma unroll
    for (int32_t j = 0; j < kElemPerCopy; j++) {
      rmem[j] = static_cast<InputType>(rmemAccum[j]);
    }
    dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename TopKScaleType>
void moeUnpermute(InputType const* permuted_input, InputType* output,
                  int32_t const* expanded_idx_to_permuted_idx, TopKScaleType const* topk_scales,
                  int32_t const num_tokens, int32_t const hidden_size, int32_t const top_k,
                  bool enable_pdl, cudaStream_t stream) {
  int32_t constexpr kThreadsPerBlock = 256;
  int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
  TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.",
                       kElemPerCopy);

  int32_t const blocks = num_tokens;
  int32_t const threads = kThreadsPerBlock;

  auto kernel = &moeUnpermuteKernel<InputType, TopKScaleType, kThreadsPerBlock>;

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
  cudaLaunchKernelEx(&config, kernel, permuted_input, output, expanded_idx_to_permuted_idx,
                     topk_scales, hidden_size, top_k);
}

#define INSTANTIATE_MOE_UNPERMUTE(InputType, TopKScaleType)                                  \
  template void moeUnpermute<InputType>(InputType const* permuted_input, InputType* output,  \
                                        int32_t const* expanded_idx_to_permuted_idx,         \
                                        TopKScaleType const* topk_scales,                    \
                                        int32_t const num_tokens, int32_t const hidden_size, \
                                        int32_t const top_k, bool enable_pdl, cudaStream_t stream)

INSTANTIATE_MOE_UNPERMUTE(half, float);
INSTANTIATE_MOE_UNPERMUTE(half, half);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_UNPERMUTE(__nv_bfloat16, float);
INSTANTIATE_MOE_UNPERMUTE(__nv_bfloat16, __nv_bfloat16);
#endif
#undef INSTANTIATE_MOE_UNPERMUTE

template <typename InputType, int32_t kThreadsPerBlock>
__global__ void moeOutputMemsetKernel(InputType* input, int32_t const* tile_idx_to_mn_limit,
                                      int32_t const* expanded_idx_to_permuted_idx,
                                      int32_t const* permuted_idx_to_expanded_idx,
                                      int32_t const* num_non_exiting_tiles,
                                      int32_t const hidden_size, int32_t const top_k,
                                      int32_t const tile_size) {
  int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
  int64_t const kCopyPerToken = hidden_size / kElemPerCopy;

  InputType rmem[kElemPerCopy];
#pragma unroll
  for (int32_t j = 0; j < kElemPerCopy; j++) {
    rmem[j] = 0;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
  for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x) {
    int32_t const tile_idx = permuted_idx / tile_size;
    if (permuted_idx >= tile_idx_to_mn_limit[tile_idx]) {
      continue;
    }
    int32_t const expanded_idx = permuted_idx_to_expanded_idx[permuted_idx];
    int32_t const token_idx = expanded_idx / top_k;
    int32_t const topk_idx = expanded_idx % top_k;

    bool is_first_in_topk = true;
    for (int32_t k = 0; k < topk_idx; k++) {
      if (expanded_idx_to_permuted_idx[token_idx * top_k + k] >= 0) {
        is_first_in_topk = false;
        break;
      }
    }
    if (!is_first_in_topk) {
      continue;
    }

    auto* dst_ptr = reinterpret_cast<ElemCopyType*>(input) + token_idx * kCopyPerToken;
    for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock) {
      dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType>
void moeOutputMemset(InputType* input, int32_t const* tile_idx_to_mn_limit,
                     int32_t const* expanded_idx_to_permuted_idx,
                     int32_t const* permuted_idx_to_expanded_idx,
                     int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens,
                     int32_t const hidden_size, int32_t const top_k, int32_t const tile_size,
                     bool enable_pdl, cudaStream_t stream) {
  int32_t constexpr kThreadsPerBlock = 256;
  int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
  TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.",
                       kElemPerCopy);

  auto kernel = &moeOutputMemsetKernel<InputType, kThreadsPerBlock>;
  static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
  int32_t const maxBlocksPerSM = getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
  int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
  int32_t const threads = kThreadsPerBlock;

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
  cudaLaunchKernelEx(&config, kernel, input, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx,
                     permuted_idx_to_expanded_idx, num_non_exiting_tiles, hidden_size, top_k,
                     tile_size);
}

#define INSTANTIATE_MOE_OUTPUT_MEMSET(InputType)                                                \
  template void moeOutputMemset<InputType>(                                                     \
      InputType * input, int32_t const* tile_idx_to_mn_limit,                                   \
      int32_t const* expanded_idx_to_permuted_idx, int32_t const* permuted_idx_to_expanded_idx, \
      int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens,              \
      int32_t const hidden_size, int32_t const top_k, int32_t const tile_size, bool enable_pdl, \
      cudaStream_t stream)

INSTANTIATE_MOE_OUTPUT_MEMSET(half);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_OUTPUT_MEMSET(__nv_bfloat16);
#endif
#undef INSTANTIATE_MOE_OUTPUT_MEMSET

// ============================== Activation Kernels ==============================

template <typename InputType, typename ActFn, int32_t kThreadsPerBlock>
__global__ void moeActivationKernel(InputType const* input, InputType* output,
                                    int32_t const* tile_idx_to_mn_limit,
                                    int32_t const* num_non_exiting_tiles, int32_t const interm_size,
                                    int32_t const tile_size) {
  using ComputeType = float;
  int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
  // Need int64_t to prevent overflow when computing pointer offsets.
  int64_t const kCopyPerToken = interm_size / kElemPerCopy;
  InputType rmem[kElemPerCopy];
  InputType rmemGate[kElemPerCopy];
  ActFn act{};

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
  for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x) {
    int32_t const tile_idx = permuted_idx / tile_size;
    if (permuted_idx >= tile_idx_to_mn_limit[tile_idx]) {
      continue;
    }
    auto const* src_ptr = reinterpret_cast<ElemCopyType const*>(input) +
                          permuted_idx * kCopyPerToken * (ActFn::IS_GLU ? 2 : 1);
    auto* dst_ptr = reinterpret_cast<ElemCopyType*>(output) + permuted_idx * kCopyPerToken;
    for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock) {
      *reinterpret_cast<ElemCopyType*>(rmem) = src_ptr[i];
      if constexpr (ActFn::IS_GLU) {
        *reinterpret_cast<ElemCopyType*>(rmemGate) = src_ptr[i + kCopyPerToken];
#pragma unroll
        for (int32_t j = 0; j < kElemPerCopy; j++) {
          rmem[j] = static_cast<InputType>(
              act(static_cast<ComputeType>(rmemGate[j]), static_cast<ComputeType>(rmem[j])));
        }
      } else {
#pragma unroll
        for (int32_t j = 0; j < kElemPerCopy; j++) {
          rmem[j] = static_cast<InputType>(act(static_cast<ComputeType>(rmem[j])));
        }
      }

      dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType>
void moeActivation(InputType const* input, InputType* output, int32_t const* tile_idx_to_mn_limit,
                   int32_t const* num_non_exiting_tiles, MoeActivationType activation_type,
                   int32_t const max_num_permuted_tokens, int32_t const interm_size,
                   int32_t const tile_size, bool enable_pdl, cudaStream_t stream) {
  int32_t constexpr kThreadsPerBlock = 256;
  int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
  TLLM_CHECK_WITH_INFO(interm_size % kElemPerCopy == 0, "interm_size must be divisible by %d.",
                       kElemPerCopy);

  using namespace cutlass_kernels;

  auto get_act_kernel = [](MoeActivationType act_type) -> void (*)(InputType const*, InputType*,
                                                                   int32_t const*, int32_t const*,
                                                                   int32_t const, int32_t const) {
    switch (act_type) {
      case MoeActivationType::Identity:
        return &moeActivationKernel<InputType, IdentityAdaptor<cutlass::epilogue::thread::Identity>,
                                    kThreadsPerBlock>;
      case MoeActivationType::Gelu:
        return &moeActivationKernel<InputType, IdentityAdaptor<cutlass::epilogue::thread::GELU>,
                                    kThreadsPerBlock>;
      case MoeActivationType::Geglu:
        return &moeActivationKernel<InputType, GLUAdaptor<cutlass::epilogue::thread::GELU>,
                                    kThreadsPerBlock>;
      case MoeActivationType::Relu:
        return &moeActivationKernel<InputType, IdentityAdaptor<cutlass::epilogue::thread::ReLu>,
                                    kThreadsPerBlock>;
      case MoeActivationType::Silu:
        return &moeActivationKernel<InputType, IdentityAdaptor<cutlass::epilogue::thread::SiLu>,
                                    kThreadsPerBlock>;
      case MoeActivationType::Swiglu:
        return &moeActivationKernel<InputType, GLUAdaptor<cutlass::epilogue::thread::SiLu>,
                                    kThreadsPerBlock>;
      default:
        TLLM_CHECK_WITH_INFO(false, "Unsupported activation type: %d", static_cast<int>(act_type));
        return nullptr;
    }
  };

  auto kernel = get_act_kernel(activation_type);

  static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
  int32_t const maxBlocksPerSM = getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
  int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
  int32_t const threads = kThreadsPerBlock;

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
  cudaLaunchKernelEx(&config, kernel, input, output, tile_idx_to_mn_limit, num_non_exiting_tiles,
                     interm_size, tile_size);
}

#define INSTANTIATE_MOE_ACTIVATION(InputType)                                                    \
  template void moeActivation<InputType>(                                                        \
      InputType const* input, InputType* output, int32_t const* tile_idx_to_mn_limit,            \
      int32_t const* num_non_exiting_tiles, MoeActivationType activation_type,                   \
      int32_t const max_num_permuted_tokens, int32_t const interm_size, int32_t const tile_size, \
      bool enable_pdl, cudaStream_t stream)

INSTANTIATE_MOE_ACTIVATION(half);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_ACTIVATION(__nv_bfloat16);
#endif
#undef INSTANTIATE_MOE_ACTIVATION

// Note: moeActivationQuantize (fused activation + FP4 quantization) will be added later
// when NVFP4 output support is needed.

}  // namespace kernels::cute_dsl

TRTLLM_NAMESPACE_END
