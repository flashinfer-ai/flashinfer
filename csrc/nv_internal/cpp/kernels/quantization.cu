/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <float.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/quantTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/kernels/quantization.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm {
namespace kernels {

template <typename T>
void invokeQuantization(int8_t* dst, T const* src, int64_t const size, float const* scalePtr,
                        cudaStream_t stream, int maxGridSize) {
  TLLM_CHECK_WITH_INFO(size % 4 == 0,
                       "[ERROR][invokeQuantization] size should be a multiple of 4.\n");

  int numBlocks{static_cast<int>((size + 255) / 256)};
  dim3 grid(std::min(numBlocks, maxGridSize));
  TLLM_CHECK_WITH_INFO(grid.x <= maxGridSize,
                       "[ERROR][invokeQuantization] grid max size is exceeded\n");
  dim3 block(64);
  if (std::is_same_v<T, float>) {
    quantizedKernel<<<grid, block, 0, stream>>>((char4*)dst, (float4 const*)src, size / 4,
                                                scalePtr);
  } else if (std::is_same_v<T, half>) {
    quantizedKernel<<<grid, block, 0, stream>>>((char4*)dst, (half2 const*)src, size / 4, scalePtr);
  }
#ifdef ENABLE_BF16
  else if (std::is_same_v<T, __nv_bfloat16>) {
    quantizedKernel<<<grid, block, 0, stream>>>((char4*)dst, (__nv_bfloat162 const*)src, size / 4,
                                                scalePtr);
  }
#endif
}

template void invokeQuantization<float>(int8_t* dst, float const* src, int64_t const size,
                                        float const* scalePtr, cudaStream_t stream,
                                        int maxGridSize);

template void invokeQuantization<half>(int8_t* dst, half const* src, int64_t const size,
                                       float const* scalePtr, cudaStream_t stream, int maxGridSize);

#ifdef ENABLE_BF16
template void invokeQuantization<__nv_bfloat16>(int8_t* dst, __nv_bfloat16 const* src,
                                                int64_t const size, float const* scalePtr,
                                                cudaStream_t stream, int maxGridSize);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper function for grid configuration with swizzled layouts

inline int computeEffectiveRows(int m, QuantizationSFLayout layout) {
  int effectiveRows = m;
  bool isSfSwizzledLayout = (layout == QuantizationSFLayout::SWIZZLED_128x4 ||
                             layout == QuantizationSFLayout::SWIZZLED_8x4);
  if (isSfSwizzledLayout) {
    int rowTile = (layout == QuantizationSFLayout::SWIZZLED_128x4) ? 128 : 8;
    int numPaddedRows = (m + rowTile - 1) / rowTile * rowTile;  // Round up to rowTile
    effectiveRows = numPaddedRows;
  }
  return effectiveRows;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MXFP8 Quantization

template <typename T>
void invokeMxFP8Quantization(int b, int m, int n, int padded_n, T const* input, int64_t* output,
                             int32_t* SFOuput, QuantizationSFLayout layout, int multiProcessorCount,
                             bool enable_pdl, cudaStream_t stream) {
  // Fixed SF_VEC_SIZE as 32
  static constexpr int SF_VEC_SIZE = 32;

  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(padded_n / CVT_ELTS_PER_THREAD), 512));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = std::max(1u, 2048u / block.x);
  int effectiveRows = computeEffectiveRows(m, layout);
  dim3 grid(std::min(effectiveRows, multiProcessorCount * numBlocksPerSM));

  // Launch the cvt kernel.
  cudaLaunchConfig_t config;
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(
      &config,
      quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_MXFP8, T, SF_VEC_SIZE, true>, b,
      m, n, padded_n, input, nullptr, reinterpret_cast<uint32_t*>(output),
      reinterpret_cast<uint32_t*>(SFOuput), layout);
}

// Do per-token (row) quantization from fp16/bf16/fp32 to int8/fp8_e4m3.
template <typename T, typename QuantT>
void invokePerTokenQuantization(QuantT* dst, T const* src, int64_t const numRows,
                                int64_t const numCols, float const* clampPtr, float* scalePtr,
                                float* sumPtr, QuantMode quantMode, cudaStream_t stream) {
  // each block is responsible for a single row
  dim3 const block(512);
  dim3 const grid(numRows);

  // The number of elements in the packed uint4 vec.
  static constexpr int NUM_ELTS_PER_VEC = sizeof(uint4) / sizeof(T);
  TLLM_CHECK_WITH_INFO(numCols % NUM_ELTS_PER_VEC == 0, "Not supported.");

  // Cache vectors to smem to avoid reloading.
  size_t const dynamicSmemSz = numCols * sizeof(T);
  // Need to check if smem capacity is enough.
  bool useSmem = true;
  if (dynamicSmemSz >= 48 * 1024) {
    cudaError_t res =
        cudaFuncSetAttribute(perTokenQuantization<T, QuantT, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicSmemSz);
    // Fall back to reloading-reversion if smem is not enough.
    useSmem = (res == cudaSuccess);
  }

  // Enable min_scaling_factor if it is fp8 rowwise per-token quantization.
  bool hasFp8MinScaling = quantMode.hasFp8RowWise();
  // Do we use smem ?
  if (useSmem) {
    perTokenQuantization<T, QuantT, true><<<grid, block, dynamicSmemSz, stream>>>(
        dst, src, numRows, numCols, clampPtr, scalePtr, sumPtr, hasFp8MinScaling);
  } else {
    perTokenQuantization<T, QuantT, false><<<grid, block, 0, stream>>>(
        dst, src, numRows, numCols, clampPtr, scalePtr, sumPtr, hasFp8MinScaling);
  }
}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(T, QuantT)                                    \
  template void invokePerTokenQuantization(QuantT* dst, const T* src, const int64_t numRows,    \
                                           const int64_t numCols, float const* clampPtr,        \
                                           float* scalePtr, float* sumPtr, QuantMode quantMode, \
                                           cudaStream_t stream)

INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float, int8_t);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half, int8_t);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16, int8_t);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float, __nv_fp8_e4m3);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4/MXFP8 Quantization

template <typename T, int SF_VEC_SIZE>
void invokeFP4Quantization(int b, int m, int n, T const* input, float const* SFScale,
                           int64_t* output, int32_t* SFOuput, bool useUE8M0,
                           QuantizationSFLayout layout, int multiProcessorCount, bool enable_pdl,
                           cudaStream_t stream) {
#ifdef ENABLE_FP8
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    // Grid, Block size.
    // Each thread converts 16 values.
    dim3 block(std::min(int(n / CVT_FP8_TO_FP4_ELTS_PER_THREAD), 512));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = std::max(1u, 2048u / block.x);
    int effectiveRows = computeEffectiveRows(m, layout);
    dim3 grid(std::min(effectiveRows, multiProcessorCount * numBlocksPerSM));

    // Launch the cvt kernel.
    auto* kernel_instance = useUE8M0
                                ? &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4,
                                                            T, SF_VEC_SIZE, true>
                                : &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4,
                                                            T, SF_VEC_SIZE, false>;
    kernel_instance<<<grid, block, 0, stream>>>(b, m, n, n, input, SFScale,
                                                reinterpret_cast<uint32_t*>(output),
                                                reinterpret_cast<uint32_t*>(SFOuput), layout);

  } else
#endif
  {
    // Grid, Block size.
    // Each thread converts 8 values.
    dim3 block(std::min(int(n / CVT_ELTS_PER_THREAD), 512));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = std::max(1u, 2048u / block.x);
    int effectiveRows = computeEffectiveRows(m, layout);
    dim3 grid(std::min(effectiveRows, multiProcessorCount * numBlocksPerSM));

    // Launch the cvt kernel.
    auto* kernel_instance = useUE8M0
                                ? &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4,
                                                            T, SF_VEC_SIZE, true>
                                : &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4,
                                                            T, SF_VEC_SIZE, false>;

    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                       reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput),
                       layout);
  }
}

template <typename T>
__global__ void block_scale_interleave_kernel(int numBatches, int numRows, int numRowsPadded,
                                              int numCols, int numColsPadded, T const* SFIn,
                                              T* SFOutput) {
  for (int rowIdx = blockIdx.x; rowIdx < numRowsPadded; rowIdx += gridDim.x) {
    for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
      for (int colIdx = threadIdx.x; colIdx < numColsPadded; colIdx += blockDim.x) {
        T sf = 0;
        if (rowIdx < numRows && colIdx < numCols) {
          int64_t inOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
          sf = SFIn[inOffset];
        }

        std::optional<int> batchIdxOpt = batchIdx;
        std::optional<int> numRowsOpt = numRows;

        // Without batching, the math in get_sf_out_offset is the same as
        // int const numSfTilesK = (numCols + 4 - 1) / 4;
        // int const tileOffset = ((mi / 128) * numSfTilesK + ki / 4) * 512;
        // int const dstIdx = tileOffset + (mi % 32) * 16 + ((mi % 128) / 32) * 4 + ki % 4;
        auto dstIdx = get_sf_out_offset_128x4(batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols);
        SFOutput[dstIdx] = sf;
      }
    }
  }
}

__global__ void block_scale_interleave_reverse_kernel(int numBatches, int numRows, int numCols,
                                                      uint8_t const* SFIn, uint8_t* SFOutput) {
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
      for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x) {
        std::optional<int> batchIdxOpt = batchIdx;
        std::optional<int> numRowsOpt = numRows;

        // Get the swizzled input index using the same swizzling pattern
        auto srcIdx = get_sf_out_offset_128x4(batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols);
        auto sf = SFIn[srcIdx];

        // Output goes to linear layout
        int64_t outOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
        SFOutput[outOffset] = sf;
      }
    }
  }
}

// This is intended for weight loading, so m and n are large, b <= 256
template <typename T>
void invokeBlockScaleInterleave(int b, int m, int m_padded, int n, int n_padded, T const* SFIn,
                                T* SFOutput, int multiProcessorCount, cudaStream_t stream) {
  // Each thread reads 1 int8 value
  dim3 block(std::min(n_padded, 1024));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = std::max(1u, 4096u / block.x);
  dim3 grid(std::min(m_padded, multiProcessorCount * numBlocksPerSM));

  block_scale_interleave_kernel<T>
      <<<grid, block, 0, stream>>>(b, m, m_padded, n, n_padded, SFIn, SFOutput);
}

// Explicit template instantiations for the types used by other compilation units
template void invokeBlockScaleInterleave<uint8_t>(int b, int m, int m_padded, int n, int n_padded,
                                                  uint8_t const* SFIn, uint8_t* SFOutput,
                                                  int multiProcessorCount, cudaStream_t stream);
template void invokeBlockScaleInterleave<__nv_bfloat16>(int b, int m, int m_padded, int n,
                                                        int n_padded, __nv_bfloat16 const* SFIn,
                                                        __nv_bfloat16* SFOutput,
                                                        int multiProcessorCount,
                                                        cudaStream_t stream);

// This is intended for weight loading, so m and n are large, b <= 256
void invokeBlockScaleInterleaveReverse(int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput,
                                       int multiProcessorCount, cudaStream_t stream) {
  // Each thread reads 1 int8 value
  dim3 block(std::min(n, 1024));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = std::max(1u, 4096u / block.x);
  dim3 grid(std::min(m, multiProcessorCount * numBlocksPerSM));

  block_scale_interleave_reverse_kernel<<<grid, block, 0, stream>>>(b, m, n, SFIn, SFOutput);
}

template <typename T>
void invokeSiluAndMulNVFP4Quantization(void* output, void* output_scale, void* input,
                                       void* input_global_scale, void* mask, bool use_silu_and_mul,
                                       int m_topk, int k, int n_experts, cudaStream_t stream) {
  int device;
  TLLM_CUDA_CHECK(cudaGetDevice(&device));
  int multiProcessorCount;
  TLLM_CUDA_CHECK(
      cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device));

  // Grid, Block size.
  // Each thread converts 8 values.
  TLLM_CHECK_WITH_INFO(k > 0, "k must be > 0");
  int const workSizePerRow = max(1, k / CVT_ELTS_PER_THREAD);
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(static_cast<int>((totalWorkSize + block.x - 1) / block.x),
                     multiProcessorCount * numBlocksPerSM));
  while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
  }

  // TODO(kaixih@nvidia): Should relax this to allow any grid size.
  // shuw@nvidia.com: only deal with mask case
  TLLM_CHECK_WITH_INFO(mask != nullptr, "mask must be non-null for expert NVFP4 path");
  TLLM_CHECK_WITH_INFO(n_experts > 0, "n_experts must be > 0");
  grid.x = (grid.x + n_experts - 1) / n_experts * n_experts;
  cvt_fp16_to_fp4_expert<T, false><<<grid, block, 0, stream>>>(
      m_topk, k, reinterpret_cast<T*>(input), reinterpret_cast<float*>(input_global_scale),
      reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(output_scale),
      reinterpret_cast<int32_t*>(mask), use_silu_and_mul, n_experts);
  return;
}

// Instantiate the function.
template void invokeFP4Quantization<half, 16>(int b, int m, int n, half const* input,
                                              float const* SFScale, int64_t* output,
                                              int32_t* SFOuput, bool useUE8M0,
                                              QuantizationSFLayout layout, int multiProcessorCount,
                                              bool enable_pdl, cudaStream_t stream);
template void invokeFP4Quantization<half, 32>(int b, int m, int n, half const* input,
                                              float const* SFScale, int64_t* output,
                                              int32_t* SFOuput, bool useUE8M0,
                                              QuantizationSFLayout layout, int multiProcessorCount,
                                              bool enable_pdl, cudaStream_t stream);
template void invokeMxFP8Quantization<half>(int b, int m, int n, int padded_n, half const* input,
                                            int64_t* output, int32_t* SFOuput,
                                            QuantizationSFLayout layout, int multiProcessorCount,
                                            bool enable_pdl, cudaStream_t stream);
template void invokeSiluAndMulNVFP4Quantization<half>(void* output, void* output_scale, void* input,
                                                      void* input_global_scale, void* mask,
                                                      bool use_silu_and_mul, int m_topk, int k,
                                                      int n_experts, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeFP4Quantization<__nv_bfloat16, 16>(
    int b, int m, int n, __nv_bfloat16 const* input, float const* SFScale, int64_t* output,
    int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    bool enable_pdl, cudaStream_t stream);
template void invokeFP4Quantization<__nv_bfloat16, 32>(
    int b, int m, int n, __nv_bfloat16 const* input, float const* SFScale, int64_t* output,
    int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    bool enable_pdl, cudaStream_t stream);
template void invokeMxFP8Quantization<__nv_bfloat16>(int b, int m, int n, int padded_n,
                                                     __nv_bfloat16 const* input, int64_t* output,
                                                     int32_t* SFOuput, QuantizationSFLayout layout,
                                                     int multiProcessorCount, bool enable_pdl,
                                                     cudaStream_t stream);
template void invokeSiluAndMulNVFP4Quantization<__nv_bfloat16>(
    void* output, void* output_scale, void* input, void* input_global_scale, void* mask,
    bool use_silu_and_mul, int m_topk, int k, int n_experts, cudaStream_t stream);

#endif

#ifdef ENABLE_FP8
template void invokeFP4Quantization<__nv_fp8_e4m3, 16>(
    int b, int m, int n, __nv_fp8_e4m3 const* input, float const* SFScale, int64_t* output,
    int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    bool enable_pdl, cudaStream_t stream);
template void invokeFP4Quantization<__nv_fp8_e4m3, 32>(
    int b, int m, int n, __nv_fp8_e4m3 const* input, float const* SFScale, int64_t* output,
    int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    bool enable_pdl, cudaStream_t stream);

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernels
}  // namespace tensorrt_llm
