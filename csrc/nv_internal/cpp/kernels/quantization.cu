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

#include <cuda.h>
#include <cudaTypedefs.h>
#include <float.h>

#include <cub/cub.cuh>

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
                             int32_t* SFOutput, QuantizationSFLayout layout,
                             int multiProcessorCount, bool enable_pdl, cudaStream_t stream) {
  // Fixed SF_VEC_SIZE as 32
  // TODO: TMA quantization for MXFP8 is not supported yet because of SF_VEC_SIZE = 32.
  static constexpr int SF_VEC_SIZE = 32;

  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(padded_n / CVT_FP16_TO_MXFP8_ELTS_PER_THREAD), 512));
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
      reinterpret_cast<uint32_t*>(SFOutput), layout);
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
// TMA tensor map creation helpers

template <typename T>
CUtensorMap make_3d_tma_copy_desc(T* global_address, uint64_t gmem_dim[3],
                                  uint64_t stride_in_bytes[2], uint32_t smem_dim[3],
                                  CUtensorMapSwizzle swizzle_type) {
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 3;
  uint32_t elem_strides[rank] = {1, 1, 1};

  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;

#if CUDA_VERSION >= 12050
  cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                   cudaEnableDefault, &driver_status);
#else
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, cudaEnableDefault,
                          &driver_status);
#endif

  if (driver_status != cudaDriverEntryPointSuccess) {
    TLLM_CHECK_WITH_INFO(false, "Failed to get cuTensorMapEncodeTiled entry point");
  }

  auto encode_func =
      reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);

  CUtensorMapDataType data_type;
  if constexpr (std::is_same_v<T, half>) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }

  CUresult result =
      encode_func(&tensor_map, data_type, rank, global_address, gmem_dim, stride_in_bytes, smem_dim,
                  elem_strides, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type,
                  CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                  CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  TLLM_CHECK_WITH_INFO(result == CUDA_SUCCESS, "Failed to encode TMA tensor map");
  return tensor_map;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Row-wise Amax Helper kernel

template <typename T, uint32_t BLOCK_SIZE>
__global__ void rowWiseAmaxKernel(uint32_t m, uint32_t n, T const* input, float* amaxOutput,
                                  float scale, int32_t* expanded_idx_to_permuted_idx = nullptr) {
  uint32_t rowIdx = blockIdx.x;
  if (rowIdx >= m) return;
  if (expanded_idx_to_permuted_idx != nullptr) {
    rowIdx = expanded_idx_to_permuted_idx[rowIdx];
  }

  float localMax = 0.f;
  for (uint32_t colIdx = threadIdx.x; colIdx < n; colIdx += blockDim.x) {
    T element = input[rowIdx * n + colIdx];
    localMax = fmaxf(localMax, fabsf(static_cast<float>(element) * scale));
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  float blockMax = BlockReduce(tempStorage)
                       .Reduce(
                           localMax,
#if CUDART_VERSION >= 12090
                           cuda::maximum<> {}
#else
                           cub::Max(),
#endif
                       );

  if (threadIdx.x == 0) {
    amaxOutput[rowIdx] = blockMax;
  }
}

template <typename T>
void invokeRowWiseAmax(uint32_t m, uint32_t n, T const* input, float* output, float scale,
                       int32_t* expanded_idx_to_permuted_idx, cudaStream_t stream) {
  constexpr uint32_t BLOCK_SIZE = 256;
  dim3 block(BLOCK_SIZE);
  dim3 grid(m);
  rowWiseAmaxKernel<T, BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(m, n, input, output, scale, expanded_idx_to_permuted_idx);
}

// Instantiate the function.
template void invokeRowWiseAmax<half>(uint32_t m, uint32_t n, half const* input, float* output,
                                      float scale, int32_t* expanded_idx_to_permuted_idx,
                                      cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeRowWiseAmax<__nv_bfloat16>(uint32_t m, uint32_t n, __nv_bfloat16 const* input,
                                               float* output, float scale,
                                               int32_t* expanded_idx_to_permuted_idx,
                                               cudaStream_t stream);
#endif
#ifdef ENABLE_FP8
template void invokeRowWiseAmax<__nv_fp8_e4m3>(uint32_t m, uint32_t n, __nv_fp8_e4m3 const* input,
                                               float* output, float scale,
                                               int32_t* expanded_idx_to_permuted_idx,
                                               cudaStream_t stream);
#endif

template <typename T, uint32_t BLOCK_SIZE, QuantizationSFLayout SF_LAYOUT, bool CACHE_LOCAL_AMAX>
__global__ void nvfp4QuantAndPerTokenScaleKernel(
    // input
    uint32_t m, uint32_t n, T const* input, float globalScaleInv, int32_t* expandedIdxToPermutedIdx,
    // output
    uint8_t* weightOutput, uint8_t* scaleOutput, float* perTokenScaleOutput,
    QuantizationSFLayout layout = QuantizationSFLayout::LINEAR) {
  static constexpr int ELTS_PER_THREAD = CVT_FP16_TO_FP4_ELTS_PER_THREAD;
  static constexpr int SF_VEC_SIZE = 16;
  static constexpr int NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;  // 2
  int rowIdx = blockIdx.x;
  if (rowIdx >= m) return;
  if (expandedIdxToPermutedIdx != nullptr) {
    rowIdx = expandedIdxToPermutedIdx[rowIdx];
  }
  if (rowIdx < 0) return;
  extern __shared__ float
      localAmaxSmem[];  // n / ELTS_PER_THREAD float values to store all local amax
  using VecType = PackedVec<T, ELTS_PER_THREAD>;  // bf16x8
  using PackedFp4Type = std::conditional_t<ELTS_PER_THREAD == 16, uint64_t, uint32_t>;
  VecType vec;
  uint8_t fp8Scale{0};
  PackedFp4Type fp4Vals{0};

  float localAmax = 0.f;
  uint32_t num_vecs_per_row = (n + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD;
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row; vecIdx += blockDim.x) {
    int64_t vecOffset = rowIdx * num_vecs_per_row + vecIdx;
    loadPackedVec(vec, reinterpret_cast<VecType const*>(input) + vecOffset);
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD / 2; ++i) {
      auto element = cuda_abs(vec.elts[i]);
      localAmax = fmaxf(localAmax, static_cast<float>(cuda_max(element.x, element.y)));
    }

    if constexpr (CACHE_LOCAL_AMAX) {
      if constexpr (NUM_THREADS_PER_SF > 1) {
        // use warp shuffle to get the amax of 16 elements and store it to SMEM
        localAmax =
            fmaxf(__shfl_xor_sync(__activemask(), localAmax, NUM_THREADS_PER_SF / 2), localAmax);
      }
      localAmaxSmem[vecIdx] = localAmax;
    }
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  float globalAmax = BlockReduce(tempStorage)
                         .Reduce(
                             localAmax,
#if CUDART_VERSION >= 12090
                             cuda::maximum<> {}
#else
                             cub::Max(),
#endif
                         );

  // save the per-token scale
  float perTokenScale = globalAmax * globalScaleInv;
  if (threadIdx.x == 0) {
    perTokenScaleOutput[rowIdx] = perTokenScale;
  }
  __syncthreads();
  perTokenScale = perTokenScaleOutput[rowIdx];

  // quantize to fp4 with per-token scale
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row; vecIdx += blockDim.x) {
    int64_t vecOffset = rowIdx * num_vecs_per_row + vecIdx;
    loadPackedVec(vec, reinterpret_cast<VecType const*>(input) + vecOffset);

    if constexpr (CACHE_LOCAL_AMAX) {
      localAmax = localAmaxSmem[vecIdx];
      fp4Vals = cvt_warp_fp16_to_fp4_with_vec_max<T, SF_VEC_SIZE, ELTS_PER_THREAD, false>(
          vec, reciprocal_approximate_ftz(perTokenScale), perTokenScale, localAmax, &fp8Scale);
    } else {
      fp4Vals = cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, ELTS_PER_THREAD, false>(
          vec, reciprocal_approximate_ftz(perTokenScale), &fp8Scale);
    }
    reinterpret_cast<PackedFp4Type*>(weightOutput)[vecOffset] = fp4Vals;

    if (threadIdx.x % NUM_THREADS_PER_SF == 0) {
      uint32_t num_sf_vecs_per_row = (n + SF_VEC_SIZE - 1) / SF_VEC_SIZE;
      auto sfVecIdx = vecIdx / NUM_THREADS_PER_SF;
      int64_t sfOffset;
      if constexpr (SF_LAYOUT == QuantizationSFLayout::LINEAR) {
        sfOffset = rowIdx * num_sf_vecs_per_row + sfVecIdx;
      } else if constexpr (SF_LAYOUT == QuantizationSFLayout::SWIZZLED_128x4) {
        sfOffset = get_sf_out_offset_128x4(std::nullopt, rowIdx, sfVecIdx, m, num_sf_vecs_per_row);
      } else {
        sfOffset = get_sf_out_offset_8x4(std::nullopt, rowIdx, sfVecIdx, m, num_sf_vecs_per_row);
      }
      scaleOutput[sfOffset] = fp8Scale;
    }
  }
}

template <typename T>
void invokeNvfp4QuantAndPerTokenScale(uint32_t m, uint32_t n, T const* input, float globalScaleInv,
                                      int32_t* expandedIdxToPermutedIdx, uint8_t* weightOutput,
                                      uint8_t* scaleOutput, float* perTokenScaleOutput,
                                      QuantizationSFLayout sfLayout, cudaStream_t stream) {
  // Kernel packs 16 values per thread via PackedVec load/store.
  TLLM_CHECK_WITH_INFO(n % 16 == 0, "n must be a multiple of 16 for NVFP4 quantization");
  // TODO(siyuan): cache local amax in registers.
  //               currently caching the amax doesn't bring perf improvement.
  constexpr bool CACHE_LOCAL_AMAX = false;

  constexpr uint32_t BLOCK_SIZE = 256;
  uint32_t smem_size = CACHE_LOCAL_AMAX ? n / CVT_FP16_TO_FP4_ELTS_PER_THREAD * sizeof(float)
                                        : 0;  // for caching the local amax
  dim3 block(BLOCK_SIZE);
  dim3 grid(m);
  switch (sfLayout) {
    case QuantizationSFLayout::LINEAR:
      nvfp4QuantAndPerTokenScaleKernel<T, BLOCK_SIZE, QuantizationSFLayout::LINEAR,
                                       CACHE_LOCAL_AMAX><<<grid, block, smem_size, stream>>>(
          m, n, input, globalScaleInv, expandedIdxToPermutedIdx, weightOutput, scaleOutput,
          perTokenScaleOutput);
      break;
    case QuantizationSFLayout::SWIZZLED_128x4:
      nvfp4QuantAndPerTokenScaleKernel<T, BLOCK_SIZE, QuantizationSFLayout::SWIZZLED_128x4,
                                       CACHE_LOCAL_AMAX><<<grid, block, smem_size, stream>>>(
          m, n, input, globalScaleInv, expandedIdxToPermutedIdx, weightOutput, scaleOutput,
          perTokenScaleOutput);
      break;
    case QuantizationSFLayout::SWIZZLED_8x4:
      nvfp4QuantAndPerTokenScaleKernel<T, BLOCK_SIZE, QuantizationSFLayout::SWIZZLED_8x4,
                                       CACHE_LOCAL_AMAX><<<grid, block, smem_size, stream>>>(
          m, n, input, globalScaleInv, expandedIdxToPermutedIdx, weightOutput, scaleOutput,
          perTokenScaleOutput);
      break;
    default:
      TLLM_CHECK_WITH_INFO(false,
                           "Unsupported QuantizationSFLayout. Supported values are: LINEAR,"
                           " SWIZZLED_128x4 and SWIZZLED_8x4.");
  }
}

// Instantiate the function.
template void invokeNvfp4QuantAndPerTokenScale<half>(
    uint32_t m, uint32_t n, half const* input, float globalScaleInv,
    int32_t* expandedIdxToPermutedIdx, uint8_t* weightOutput, uint8_t* scaleOutput,
    float* perTokenScaleOutput, QuantizationSFLayout sfLayout, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>(
    uint32_t m, uint32_t n, __nv_bfloat16 const* input, float globalScaleInv,
    int32_t* expandedIdxToPermutedIdx, uint8_t* weightOutput, uint8_t* scaleOutput,
    float* perTokenScaleOutput, QuantizationSFLayout sfLayout, cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4/MXFP8 Quantization

// Helper function to launch TMA quantization kernel
template <BlockScaleQuantizationType quantization_type, typename T, int SF_VEC_SIZE>
void launchFP4QuantizationTma(int b, int m, int n, T const* input, float const* SFScale,
                              int64_t* output, int32_t* SFOutput, bool useUE8M0,
                              QuantizationSFLayout layout, int multiProcessorCount, bool enable_pdl,
                              bool use_row_wise_scale, bool inverse_scale, cudaStream_t stream) {
  using Traits = TmaKernelTraits<T>;
  constexpr int TMA_ROW_TILE = Traits::TMA_ROW_TILE;
  constexpr int TMA_COL_TILE = Traits::TMA_COL_TILE;
  constexpr int NUM_CONSUMER_WARPS = 8;

  // Compute effective rows for swizzled layouts
  int effectiveRows = computeEffectiveRows(m, layout);

  // Grid and block configuration for TMA kernel
  // TMA kernel uses 288 threads: 1 producer warp + 8 consumer warps
  dim3 block(288);
  // Each block handles TMA_ROW_TILE rows
  int numRowTiles = (effectiveRows + TMA_ROW_TILE - 1) / TMA_ROW_TILE;
  dim3 grid(std::min(numRowTiles, multiProcessorCount * 2));

  // Dynamic shared memory size
  size_t smem_size = get_tma_smem_size<T>();

  // Create 3D TMA tensor map descriptor
  // The TMA kernel loads a box of [TMA_COL_TILE, TMA_ROW_TILE, NUM_CONSUMER_WARPS] elements per TMA
  // call Global tensor is treated as [TMA_COL_TILE, B*M, num_tiles] where num_tiles = N /
  // TMA_COL_TILE. We use b * m (not b * effectiveRows) because batches are stored contiguously
  // without padding between them.
  int num_col_tiles = (n + TMA_COL_TILE - 1) / TMA_COL_TILE;
  uint64_t gmem_dim[3] = {
      static_cast<uint64_t>(TMA_COL_TILE),  // Elements per tile (contiguous in memory)
      static_cast<uint64_t>(b * m),         // Total rows across all batches
      static_cast<uint64_t>(num_col_tiles)  // Number of column tiles
  };
  uint64_t stride_in_bytes[2] = {
      static_cast<uint64_t>(n * sizeof(T)),            // Stride between rows (in bytes)
      static_cast<uint64_t>(TMA_COL_TILE * sizeof(T))  // Stride between tiles (in bytes)
  };
  uint32_t smem_dim[3] = {
      static_cast<uint32_t>(TMA_COL_TILE),       // Elements loaded per tile
      static_cast<uint32_t>(TMA_ROW_TILE),       // Rows loaded per TMA call
      static_cast<uint32_t>(NUM_CONSUMER_WARPS)  // Number of tiles loaded (for 8 consumer warps)
  };

  // CUtensorMap must be 64-byte aligned
  // Use SWIZZLE_128B for half/bf16 (2-byte types), SWIZZLE_NONE for FP8 (1-byte types)
  constexpr CUtensorMapSwizzle swizzle_type =
      (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>)
          ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
          : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
  alignas(64) CUtensorMap tensor_map = make_3d_tma_copy_desc(
      const_cast<T*>(input), gmem_dim, stride_in_bytes, smem_dim, swizzle_type);

  // Select and launch the TMA kernel
  cudaLaunchConfig_t config;
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  if (use_row_wise_scale) {
    if (inverse_scale) {
      auto* kernel_instance =
          useUE8M0
              ? &quantize_with_block_size_tma<quantization_type, T, SF_VEC_SIZE, true, true, true>
              : &quantize_with_block_size_tma<quantization_type, T, SF_VEC_SIZE, false, true, true>;

      // Set max dynamic shared memory for the kernel (required for > 48KB)
      cudaFuncSetAttribute(kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                         reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOutput),
                         layout, tensor_map);
    } else {
      auto* kernel_instance =
          useUE8M0
              ? &quantize_with_block_size_tma<quantization_type, T, SF_VEC_SIZE, true, true, false>
              : &quantize_with_block_size_tma<quantization_type, T, SF_VEC_SIZE, false, true,
                                              false>;
      cudaFuncSetAttribute(kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                         reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOutput),
                         layout, tensor_map);
    }
  } else {
    if (inverse_scale) {
      auto* kernel_instance =
          useUE8M0
              ? &quantize_with_block_size_tma<quantization_type, T, SF_VEC_SIZE, true, false, true>
              : &quantize_with_block_size_tma<quantization_type, T, SF_VEC_SIZE, false, false,
                                              true>;
      cudaFuncSetAttribute(kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                         reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOutput),
                         layout, tensor_map);
    } else {
      auto* kernel_instance =
          useUE8M0
              ? &quantize_with_block_size_tma<quantization_type, T, SF_VEC_SIZE, true, false, false>
              : &quantize_with_block_size_tma<quantization_type, T, SF_VEC_SIZE, false, false,
                                              false>;
      cudaFuncSetAttribute(kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                         reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOutput),
                         layout, tensor_map);
    }
  }
}

template <typename T, int SF_VEC_SIZE>
void invokeFP4Quantization(int b, int m, int n, T const* input, float const* SFScale,
                           int64_t* output, int32_t* SFOutput, bool useUE8M0,
                           QuantizationSFLayout layout, int multiProcessorCount, bool enable_pdl,
                           bool use_row_wise_scale, bool inverse_scale, cudaStream_t stream) {
#ifdef ENABLE_FP8
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    // Use TMA kernel for large m (high throughput mode)
    // TODO: fix the issue when n is not a multiple of NUM_CONSUMER_WARPS * TMA_COL_TILE
    constexpr int TMA_COL_CHUNK = 8 * 64;  // NUM_CONSUMER_WARPS * TMA_COL_TILE
    if constexpr (SF_VEC_SIZE == 16) {
      if (SF_VEC_SIZE == 16 && m >= 1024 && n % TMA_COL_CHUNK == 0) {
        launchFP4QuantizationTma<BlockScaleQuantizationType::FP8_TO_FP4, T, SF_VEC_SIZE>(
            b, m, n, input, SFScale, output, SFOutput, useUE8M0, layout, multiProcessorCount,
            enable_pdl, use_row_wise_scale, inverse_scale, stream);
        return;
      }
    }
    // Original non-TMA path for small m or SF_VEC_SIZE != 16
    // Grid, Block size.
    // Each thread converts 16 values.
    dim3 block(std::min(int(n / CVT_FP8_TO_FP4_ELTS_PER_THREAD), 512));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = std::max(1u, 2048u / block.x);
    int effectiveRows = computeEffectiveRows(m, layout);
    dim3 grid(std::min(effectiveRows, multiProcessorCount * numBlocksPerSM));

    // Launch the cvt kernel.x
    if (use_row_wise_scale) {
      if (inverse_scale) {
        auto* kernel_instance =
            useUE8M0 ? &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T,
                                                 SF_VEC_SIZE, true, true, true>
                     : &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T,
                                                 SF_VEC_SIZE, false, true, true>;
        kernel_instance<<<grid, block, 0, stream>>>(b, m, n, n, input, SFScale,
                                                    reinterpret_cast<uint32_t*>(output),
                                                    reinterpret_cast<uint32_t*>(SFOutput), layout);
      } else {
        auto* kernel_instance =
            useUE8M0 ? &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T,
                                                 SF_VEC_SIZE, true, true, false>
                     : &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T,
                                                 SF_VEC_SIZE, false, true, false>;
        kernel_instance<<<grid, block, 0, stream>>>(b, m, n, n, input, SFScale,
                                                    reinterpret_cast<uint32_t*>(output),
                                                    reinterpret_cast<uint32_t*>(SFOutput), layout);
      }
    } else {
      if (inverse_scale) {
        auto* kernel_instance =
            useUE8M0 ? &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T,
                                                 SF_VEC_SIZE, true, false, true>
                     : &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T,
                                                 SF_VEC_SIZE, false, false, true>;
        kernel_instance<<<grid, block, 0, stream>>>(b, m, n, n, input, SFScale,
                                                    reinterpret_cast<uint32_t*>(output),
                                                    reinterpret_cast<uint32_t*>(SFOutput), layout);
      } else {
        auto* kernel_instance =
            useUE8M0 ? &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T,
                                                 SF_VEC_SIZE, true, false, false>
                     : &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T,
                                                 SF_VEC_SIZE, false, false, false>;
        kernel_instance<<<grid, block, 0, stream>>>(b, m, n, n, input, SFScale,
                                                    reinterpret_cast<uint32_t*>(output),
                                                    reinterpret_cast<uint32_t*>(SFOutput), layout);
      }
    }
  } else
#endif
  {
    // Use TMA kernel for large m (high throughput mode)
    // TODO: fix the issue when n is not a multiple of NUM_CONSUMER_WARPS * TMA_COL_TILE
    constexpr int TMA_COL_CHUNK = 8 * 64;  // NUM_CONSUMER_WARPS * TMA_COL_TILE
    if constexpr (SF_VEC_SIZE == 16) {
      if (SF_VEC_SIZE == 16 && m >= 1024 && n % TMA_COL_CHUNK == 0) {
        launchFP4QuantizationTma<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE>(
            b, m, n, input, SFScale, output, SFOutput, useUE8M0, layout, multiProcessorCount,
            enable_pdl, use_row_wise_scale, inverse_scale, stream);
        return;
      }
    }
    // Original non-TMA path for small m or SF_VEC_SIZE != 16
    // Grid, Block size.
    // Each thread converts 8 values.
    dim3 block(std::min(int(n / CVT_FP16_TO_FP4_ELTS_PER_THREAD), 512));
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
    if (use_row_wise_scale) {
      if (inverse_scale) {
        auto* kernel_instance =
            useUE8M0 ? &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T,
                                                 SF_VEC_SIZE, true, true, true>
                     : &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T,
                                                 SF_VEC_SIZE, false, true, true>;
        cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                           reinterpret_cast<uint32_t*>(output),
                           reinterpret_cast<uint32_t*>(SFOutput), layout);
      } else {
        auto* kernel_instance =
            useUE8M0 ? &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T,
                                                 SF_VEC_SIZE, true, true, false>
                     : &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T,
                                                 SF_VEC_SIZE, false, true, false>;
        cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                           reinterpret_cast<uint32_t*>(output),
                           reinterpret_cast<uint32_t*>(SFOutput), layout);
      }
    } else {
      if (inverse_scale) {
        auto* kernel_instance =
            useUE8M0 ? &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T,
                                                 SF_VEC_SIZE, true, false, true>
                     : &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T,
                                                 SF_VEC_SIZE, false, false, true>;
        cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                           reinterpret_cast<uint32_t*>(output),
                           reinterpret_cast<uint32_t*>(SFOutput), layout);
      } else {
        auto* kernel_instance =
            useUE8M0 ? &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T,
                                                 SF_VEC_SIZE, true, false, false>
                     : &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T,
                                                 SF_VEC_SIZE, false, false, false>;
        cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                           reinterpret_cast<uint32_t*>(output),
                           reinterpret_cast<uint32_t*>(SFOutput), layout);
      }
    }
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
  int const workSizePerRow = max(1, k / CVT_FP16_TO_FP4_ELTS_PER_THREAD);
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
                                              int32_t* SFOutput, bool useUE8M0,
                                              QuantizationSFLayout layout, int multiProcessorCount,
                                              bool enable_pdl, bool use_row_wise_scale,
                                              bool inverse_scale, cudaStream_t stream);
template void invokeFP4Quantization<half, 32>(int b, int m, int n, half const* input,
                                              float const* SFScale, int64_t* output,
                                              int32_t* SFOutput, bool useUE8M0,
                                              QuantizationSFLayout layout, int multiProcessorCount,
                                              bool enable_pdl, bool use_row_wise_scale,
                                              bool inverse_scale, cudaStream_t stream);
template void invokeMxFP8Quantization<half>(int b, int m, int n, int padded_n, half const* input,
                                            int64_t* output, int32_t* SFOutput,
                                            QuantizationSFLayout layout, int multiProcessorCount,
                                            bool enable_pdl, cudaStream_t stream);
template void invokeSiluAndMulNVFP4Quantization<half>(void* output, void* output_scale, void* input,
                                                      void* input_global_scale, void* mask,
                                                      bool use_silu_and_mul, int m_topk, int k,
                                                      int n_experts, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeFP4Quantization<__nv_bfloat16, 16>(
    int b, int m, int n, __nv_bfloat16 const* input, float const* SFScale, int64_t* output,
    int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    bool enable_pdl, bool use_row_wise_scale, bool inverse_scale, cudaStream_t stream);
template void invokeFP4Quantization<__nv_bfloat16, 32>(
    int b, int m, int n, __nv_bfloat16 const* input, float const* SFScale, int64_t* output,
    int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    bool enable_pdl, bool use_row_wise_scale, bool inverse_scale, cudaStream_t stream);
template void invokeMxFP8Quantization<__nv_bfloat16>(int b, int m, int n, int padded_n,
                                                     __nv_bfloat16 const* input, int64_t* output,
                                                     int32_t* SFOutput, QuantizationSFLayout layout,
                                                     int multiProcessorCount, bool enable_pdl,
                                                     cudaStream_t stream);
template void invokeSiluAndMulNVFP4Quantization<__nv_bfloat16>(
    void* output, void* output_scale, void* input, void* input_global_scale, void* mask,
    bool use_silu_and_mul, int m_topk, int k, int n_experts, cudaStream_t stream);

#endif

#ifdef ENABLE_FP8
template void invokeFP4Quantization<__nv_fp8_e4m3, 16>(
    int b, int m, int n, __nv_fp8_e4m3 const* input, float const* SFScale, int64_t* output,
    int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    bool enable_pdl, bool use_row_wise_scale, bool inverse_scale, cudaStream_t stream);
template void invokeFP4Quantization<__nv_fp8_e4m3, 32>(
    int b, int m, int n, __nv_fp8_e4m3 const* input, float const* SFScale, int64_t* output,
    int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    bool enable_pdl, bool use_row_wise_scale, bool inverse_scale, cudaStream_t stream);

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernels
}  // namespace tensorrt_llm
