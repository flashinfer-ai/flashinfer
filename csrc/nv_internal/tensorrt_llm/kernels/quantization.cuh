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

#include <cutlass/arch/barrier.h>
#include <float.h>

#include <cute/arch/copy_sm90_tma.hpp>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

using namespace tensorrt_llm::common;
using Barrier = cutlass::arch::ClusterTransactionBarrier;

namespace tensorrt_llm {
namespace kernels {

__global__ static void quantizedKernel(char4* dst, float4 const* src, int64_t const sizeDiv4,
                                       float const* scalePtr) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4;
       idx += blockDim.x * gridDim.x) {
    float const scale = __ldg(scalePtr);
    char4 tmp;
    float4 const floatTmp = __ldg(src + idx);
    tmp.x = cuda_cast<int8_t>(floatTmp.x * scale);
    tmp.y = cuda_cast<int8_t>(floatTmp.y * scale);
    tmp.z = cuda_cast<int8_t>(floatTmp.z * scale);
    tmp.w = cuda_cast<int8_t>(floatTmp.w * scale);
    dst[idx] = tmp;
  }
}

__global__ static void quantizedKernel(char4* dst, half2 const* src, int64_t const sizeDiv4,
                                       float const* scalePtr) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4;
       idx += blockDim.x * gridDim.x) {
    float const scale = __ldg(scalePtr);
    char4 tmp;
    int srcId = idx << 1;

    uint2 const h2 = __ldg(reinterpret_cast<uint2 const*>(src + srcId));

    half2 const half2Tmp = reinterpret_cast<half2 const&>(h2.x);
    half2 const half2Tmp2 = reinterpret_cast<half2 const&>(h2.y);

    tmp.x = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.x) * scale);
    tmp.y = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.y) * scale);
    tmp.z = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.x) * scale);
    tmp.w = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.y) * scale);
    dst[idx] = tmp;
  }
}

#ifdef ENABLE_BF16
__global__ static void quantizedKernel(char4* dst, __nv_bfloat162 const* src,
                                       int64_t const sizeDiv4, float const* scalePtr) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4;
       idx += blockDim.x * gridDim.x) {
    float const scale = __ldg(scalePtr);
    char4 tmp;
    int srcId = idx << 1;

    uint2 const h2 = __ldg(reinterpret_cast<uint2 const*>(src + srcId));

    __nv_bfloat162 const bfloat162Tmp = reinterpret_cast<__nv_bfloat162 const&>(h2.x);
    __nv_bfloat162 const bfloat162Tmp2 = reinterpret_cast<__nv_bfloat162 const&>(h2.y);

    tmp.x = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp.x) * scale);
    tmp.y = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp.y) * scale);
    tmp.z = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp2.x) * scale);
    tmp.w = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp2.y) * scale);

    dst[idx] = tmp;
  }
}
#endif

template <typename T, typename QuantT, bool USE_SMEM>
__global__ void perTokenQuantization(QuantT* dst, T const* src, int64_t const numRows,
                                     int64_t const numCols, float const* clampPtr, float* scalePtr,
                                     float* sumPtr, bool hasFp8MinScaling) {
  // Smem buffer.
  extern __shared__ uint4 smemBuffer[];

  // The clamping minimum / maximum values.
  T const clampMin = cuda_cast<T>(clampPtr ? clampPtr[0] : -FLT_MAX);
  T const clampMax = cuda_cast<T>(clampPtr ? clampPtr[1] : FLT_MAX);

  // Pack two elements in order to use higher through instructions.
  using T2 = typename packed_as<T, 2>::type;
  using QuantT2 = typename packed_as<QuantT, 2>::type;
  T2 const clampMin2 = cuda_cast<T2, T>(clampMin);
  T2 const clampMax2 = cuda_cast<T2, T>(clampMax);

  // The quantized data type's maximum value (upper-bound).
  static constexpr float MAX_QUANT_VAL = QuantTypeStaticVals<QuantT>::MAX_VAL;
  // The minimum scaling factor (lower-bound).
  static constexpr float MIN_SCALING_FACTOR = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR;
  static constexpr float MIN_SCALING_FACTOR_RCP =
      QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR_RCP;

  // The number of elements in the packed uint4 vec.
  static constexpr int NUM_ELTS_PER_VEC = sizeof(uint4) / sizeof(T);
  static constexpr int NUM_ELTS2_PER_VEC = sizeof(uint4) / sizeof(T2);

  // The number of vectors in the column.
  int const numColVecs = numCols / NUM_ELTS_PER_VEC;
  // The vector pointers for src.
  uint4 const* srcVec = reinterpret_cast<uint4 const*>(src) + blockIdx.x * numColVecs;
  // The pointer for dst.
  QuantT* dstRow = dst + blockIdx.x * numCols;
  // T const* srcRow = src + blockIdx.x * numCols;

  T2 localMax2 = cuda_cast<T2, T>(T(1e-6f));
  float2 localSum2 = {0.f, 0.f};

  for (int i = threadIdx.x; i < numColVecs; i += blockDim.x) {
    uint4 vec = srcVec[i];

#pragma unroll
    for (int j = 0; j < NUM_ELTS2_PER_VEC; ++j) {
      T2& val2 = reinterpret_cast<T2*>(&vec)[j];
      val2 = cuda_clamp(val2, clampMin2, clampMax2);
      localMax2 = cuda_max(localMax2, cuda_abs(val2));
      // TODO: template the version that requires sum to avoid dynamic branching.
      if (sumPtr != nullptr) {
        localSum2.x += cuda_cast<float>(val2.x);
        localSum2.y += cuda_cast<float>(val2.y);
      }
    }
    // Avoid reloading from global memory.
    if constexpr (USE_SMEM) {
      smemBuffer[i] = vec;
    }
  }
  float const rowMax = blockAllReduceMax(cuda_cast<float>(cuda_max<T, T2>(localMax2)));
  if (threadIdx.x == 0) {
    scalePtr[blockIdx.x] = hasFp8MinScaling ? cuda_max(rowMax / MAX_QUANT_VAL, MIN_SCALING_FACTOR)
                                            : (rowMax / MAX_QUANT_VAL);
  }

  if (sumPtr != nullptr) {
    float rowSum[1] = {cuda_sum<float>(localSum2)};
    blockReduceSumV2<float, 1>(rowSum);
    if (threadIdx.x == 0) {
      sumPtr[blockIdx.x] = rowSum[0];
    }
  }

  float const scaleOrigQuant = hasFp8MinScaling
                                   ? fminf(MAX_QUANT_VAL / rowMax, MIN_SCALING_FACTOR_RCP)
                                   : MAX_QUANT_VAL / rowMax;
  for (int i = threadIdx.x; i < numColVecs; i += blockDim.x) {
    uint4 vec = USE_SMEM ? smemBuffer[i] : srcVec[i];
    QuantT2* dstPtr = reinterpret_cast<QuantT2*>(dstRow + i * NUM_ELTS_PER_VEC);
    quantizeAndStore<T2, QuantT2, USE_SMEM>(dstPtr, vec, clampMin2, clampMax2, scaleOrigQuant);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4/MXFP8 Quantization Constants

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;
constexpr int CVT_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_THREADS_PER_WARP = 32;
constexpr int CVT_FP8_TO_FP4_ELTS_PER_THREAD = 16;

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4/MXFP8 Quantization Kernels

template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) quantize_with_block_size(
#else
quantize_with_block_size(
#endif
    int32_t numbatches, int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
    float const* SFScale, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  // The elements per thread.
  static constexpr int ELTS_PER_THREAD = quantization_type == BlockScaleQuantizationType::FP8_TO_FP4
                                             ? CVT_FP8_TO_FP4_ELTS_PER_THREAD
                                             : CVT_ELTS_PER_THREAD;

  using PackedVecT = PackedVec<Type, ELTS_PER_THREAD>;
  static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;  // 2 or 4
  static_assert(sizeof(PackedVecT) == sizeof(Type) * ELTS_PER_THREAD, "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // Is it swizzled layout?
  bool isSfSwizzledLayout = layout == QuantizationSFLayout::SWIZZLED_128x4 ||
                            layout == QuantizationSFLayout::SWIZZLED_8x4;

  // The number of padded rows considering 128x4 or 8x4 SF layout.
  int rowTile = (layout == QuantizationSFLayout::SWIZZLED_128x4) ? 128 : 8;
  int numPaddedRowsForSf = isSfSwizzledLayout ? PadUpFn(numRows, rowTile) : numRows;
  int numColsForSf = isSfSwizzledLayout ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

  // The number of threads in the column dimension。
  // Note that numCols/numPaddedCols/numColsForSf are guaranteed to be multiples of ELTS_PER_THREAD.
  int numColThreads = numCols / ELTS_PER_THREAD;
  int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
  int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

  asm volatile("griddepcontrol.wait;");

  // Input tensor batch/row/col loops.
  // Optimization: Iterate over actual rows first (hot path), then padding rows (cold path)
  // This improves performance for small batch sizes with swizzled layout
  for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x) {
    // Early exit for padding-only blocks: if this block only processes padding rows,
    // we can skip the batch loop and just zero out the scale factors
    bool isRowPadding = (rowIdx >= numRows);

    if (isRowPadding) {
      // Fast path: This row is entirely padding, only zero out scale factors.
      // Note: Padding rows do NOT exist in the output tensor (which is sized [numRows, K]),
      // they only exist in the swizzled scale factor layout. Do NOT write to output buffer here.
      for (int batchIdx = 0; batchIdx < numbatches; batchIdx++) {
        for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x) {
          std::optional<int> optionalBatchIdx = batchIdx;
          std::optional<int> optionalNumRows = numRows;

          // The SF output pointer.
          auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
              optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numColsForSf / SF_VEC_SIZE, SFout,
              layout);

          // Set the SF padding to 0.
          if (sf_out != nullptr) {
            sf_out[0] = 0x00;
          }
        }
      }
    } else {
      // Normal path: This row contains actual data
      for (int batchIdx = 0; batchIdx < numbatches; batchIdx++) {
        for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x) {
          std::optional<int> optionalBatchIdx = batchIdx;
          std::optional<int> optionalNumRows = numRows;

          // The SF output pointer.
          auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
              optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numColsForSf / SF_VEC_SIZE, SFout,
              layout);

          // The input tensor offset.
          int64_t inOffset =
              static_cast<int64_t>(batchIdx * numRows + rowIdx) * numColThreads + colIdx;
          int64_t outOffset =
              static_cast<int64_t>(batchIdx * numRows + rowIdx) * numPaddedColThreads + colIdx;

          // Set the values to 0 of those are padded columns.
          if (colIdx >= numColThreads && colIdx < numPaddedColThreads) {
            // Dispatch the quantization kernel.
            if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_FP4) {
              reinterpret_cast<uint32_t*>(out)[outOffset] = 0u;
            } else if constexpr (quantization_type == BlockScaleQuantizationType::FP8_TO_FP4 ||
                                 quantization_type == BlockScaleQuantizationType::FP16_TO_MXFP8) {
              reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
            }
          }

          // Process actual data or padding
          if (colIdx >= numColThreads) {
            // Column padding: Set the SF padding to 0.
            if (sf_out != nullptr) {
              sf_out[0] = 0x00;
            }
          } else {
            // Load the input vector.
            PackedVecT in_vec = reinterpret_cast<PackedVecT const*>(in)[inOffset];

            // Dispatch the quantization kernel.
            if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_FP4) {
              reinterpret_cast<uint32_t*>(out)[outOffset] =
                  cvt_warp_fp16_to_fp4<Type, SF_VEC_SIZE, ELTS_PER_THREAD, UE8M0_SF>(
                      in_vec, SFScaleVal, sf_out);
            } else if constexpr (quantization_type == BlockScaleQuantizationType::FP8_TO_FP4) {
              reinterpret_cast<uint64_t*>(out)[outOffset] =
                  cvt_warp_fp8_to_fp4<__nv_fp8_e4m3, SF_VEC_SIZE, ELTS_PER_THREAD, UE8M0_SF>(
                      in_vec, SFScaleVal, sf_out);
            } else if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_MXFP8) {
              reinterpret_cast<uint64_t*>(out)[outOffset] =
                  cvt_warp_fp16_to_mxfp8<Type, SF_VEC_SIZE, ELTS_PER_THREAD>(in_vec, sf_out);
            }
          }
        }
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// quantize with TMA in high throughput mode
template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(288, 2) quantize_with_block_size_tma(
#else
quantize_with_block_size_tma(
#endif
    int32_t numbatches, int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
    float const* SFScale, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout,
    const __grid_constant__ CUtensorMap tensor_map) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using Traits = TmaKernelTraits<Type>;
  using SmemType = typename Traits::SmemType;

  static constexpr int ELTS_PER_THREAD = Traits::ELTS_PER_THREAD;
  static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
  static constexpr int TMA_ROW_TILE = Traits::TMA_ROW_TILE;
  static constexpr int TMA_COL_TILE = Traits::TMA_COL_TILE;
  static constexpr int NUM_STAGES = Traits::NUM_STAGES;
  static constexpr int THREADS_PER_ROW = Traits::THREADS_PER_ROW;
  static constexpr int ROWS_PER_WARP = Traits::ROWS_PER_WARP;
  static constexpr int ROW_ITERATIONS = Traits::ROW_ITERATIONS;
  static constexpr int SMEM_STAGE_SIZE = Traits::SMEM_STAGE_SIZE;
  static constexpr int NUM_CONSUMER_WARPS = Traits::NUM_CONSUMER_WARPS;

  using PackedVecT = PackedVec<Type, ELTS_PER_THREAD>;
  static_assert(sizeof(PackedVecT) == sizeof(Type) * ELTS_PER_THREAD, "Vec size is not matched.");
  static_assert(SF_VEC_SIZE == 16, "Only support SF_VEC_SIZE = 16 for TMA quantization.");

  int warpIdx = threadIdx.x / 32;
  int numWarp = blockDim.x / 32;
  int laneIdx = threadIdx.x % 32;

  // IMPORTANT: TMA with SWIZZLE_128B requires 128-byte aligned shared memory.
  extern __shared__ __align__(1024) uint8_t smem_raw[];

  // SMEM data starts at the beginning of dynamic shared memory (128-byte aligned)
  SmemType* smem = reinterpret_cast<SmemType*>(smem_raw);

  // Place barriers at the end of dynamic shared memory
  Barrier* barrier_start_ptr = reinterpret_cast<Barrier*>(smem_raw + Traits::SMEM_DATA_SIZE);

  auto full_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
  auto empty_barriers =
      PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (NUM_STAGES + i); });

  // Get the global scaling factor
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // Is it swizzled layout?
  bool isSfSwizzledLayout = layout == QuantizationSFLayout::SWIZZLED_128x4 ||
                            layout == QuantizationSFLayout::SWIZZLED_8x4;

  // The number of padded rows considering 128x4 or 8x4 SF layout.
  int rowTile = (layout == QuantizationSFLayout::SWIZZLED_128x4) ? 128 : 8;
  int numPaddedRowsForSf = isSfSwizzledLayout ? PadUpFn(numRows, rowTile) : numRows;
  int numColsForSf = isSfSwizzledLayout ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

  asm volatile("griddepcontrol.wait;");

  // TMA barrier initialization.
  if (warpIdx == 0 and laneIdx == 0) {
#pragma unroll
    for (int i = 0; i < NUM_STAGES; i++) {
      full_barriers[i]->init(1);
      empty_barriers[i]->init(NUM_CONSUMER_WARPS);
#pragma unroll
      for (int j = 0; j < NUM_CONSUMER_WARPS; j++) {
        empty_barriers[i]->arrive();
      }
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  uint32_t stage_idx = 0, phase = 0;

  if (warpIdx == 0 and elect_one_sync()) {
    // Producer warp - TMA loads
    for (int rowIdx = blockIdx.x * TMA_ROW_TILE; rowIdx < numPaddedRowsForSf;
         rowIdx += gridDim.x * TMA_ROW_TILE) {
      for (int batchIdx = 0; batchIdx < numbatches; batchIdx++) {
        for (int colIdx = 0; colIdx < numCols; colIdx += NUM_CONSUMER_WARPS * TMA_COL_TILE) {
          empty_barriers[stage_idx]->wait(phase);

          cute::SM90_TMA_LOAD_3D::copy(
              &tensor_map, reinterpret_cast<uint64_t*>(full_barriers[stage_idx]), 0ULL,
              smem + stage_idx * SMEM_STAGE_SIZE, 0, rowIdx, colIdx / TMA_COL_TILE);
          full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_STAGE_SIZE * sizeof(SmemType));

          stage_idx = stage_idx == NUM_STAGES - 1 ? 0 : stage_idx + 1;
          phase ^= stage_idx == 0;
        }
      }
    }
  } else if (warpIdx >= 1 and warpIdx <= 8) {
    // Consumer warps
    int consumerWarpIdx = warpIdx - 1;
    typename Traits::ThreadIndexing tidx(laneIdx, consumerWarpIdx);

    for (int rowIdx = blockIdx.x * TMA_ROW_TILE; rowIdx < numPaddedRowsForSf;
         rowIdx += gridDim.x * TMA_ROW_TILE) {
      for (int batchIdx = 0; batchIdx < numbatches; batchIdx++) {
        std::optional<int> optionalBatchIdx = batchIdx;
        std::optional<int> optionalNumRows = numRows;
        tidx.reset();  // Reset column indices for each row iteration

        int threadRowIdxGlobal;
        int64_t rowOffset, threadOutOffset;

        for (int colIdx = 0; colIdx < numCols; colIdx += NUM_CONSUMER_WARPS * TMA_COL_TILE) {
          threadRowIdxGlobal = rowIdx + tidx.rowIdxLocal;
          rowOffset = static_cast<int64_t>(batchIdx * numRows + threadRowIdxGlobal) * numPaddedCols;
          threadOutOffset = (rowOffset + tidx.colIdx) >> 4;

          full_barriers[stage_idx]->wait(phase);

#pragma unroll
          for (int i = 0; i < ROW_ITERATIONS; i++) {
            auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                optionalBatchIdx, threadRowIdxGlobal, tidx.colVecIdx, optionalNumRows,
                numPaddedCols / SF_VEC_SIZE, SFout, layout);

            // Set padded columns to 0
            if (threadRowIdxGlobal < numRows && tidx.colIdx >= numCols &&
                tidx.colIdx < numPaddedCols) {
              reinterpret_cast<uint64_t*>(out)[threadOutOffset] = 0ull;
            }

            // Set SF padding to 0
            if (threadRowIdxGlobal >= numRows || tidx.colIdx >= numCols) {
              if (sf_out != nullptr) {
                sf_out[0] = 0x00;
              }
            } else {
              SmemType* smem_stage = smem + stage_idx * SMEM_STAGE_SIZE;
              float4 const* base_float4 = reinterpret_cast<float4 const*>(
                  smem_stage + consumerWarpIdx * TMA_COL_TILE * TMA_ROW_TILE +
                  i * TMA_COL_TILE * ROWS_PER_WARP);

              // Load input vector from shared memory
              PackedVecT in_vec = Traits::template load_input_vec<PackedVecT>(
                  base_float4, tidx.rowIdxLocal, tidx.colIdxLocal);

              // Dispatch the quantization kernel
              if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_FP4) {
                reinterpret_cast<uint64_t*>(out)[threadOutOffset] =
                    cvt_warp_fp16_to_fp4<Type, SF_VEC_SIZE, ELTS_PER_THREAD, UE8M0_SF>(
                        in_vec, SFScaleVal, sf_out);
              } else if constexpr (quantization_type == BlockScaleQuantizationType::FP8_TO_FP4) {
                reinterpret_cast<uint64_t*>(out)[threadOutOffset] =
                    cvt_warp_fp8_to_fp4<__nv_fp8_e4m3, SF_VEC_SIZE, ELTS_PER_THREAD, UE8M0_SF>(
                        in_vec, SFScaleVal, sf_out);
              } else if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_MXFP8) {
                reinterpret_cast<uint64_t*>(out)[threadOutOffset] =
                    cvt_warp_fp16_to_mxfp8<Type, SF_VEC_SIZE, ELTS_PER_THREAD>(in_vec, sf_out);
              }
            }

            // Update row index and output offset
            threadRowIdxGlobal += ROWS_PER_WARP;
            rowOffset =
                static_cast<int64_t>(batchIdx * numRows + threadRowIdxGlobal) * numPaddedCols;
            threadOutOffset = (rowOffset + tidx.colIdx) >> 4;
          }

          // Update column offset
          tidx.advance_col();
          threadOutOffset = (rowOffset + tidx.colIdx) >> 4;

          if (laneIdx == 0) {
            empty_barriers[stage_idx]->arrive();
          }

          stage_idx = stage_idx == NUM_STAGES - 1 ? 0 : stage_idx + 1;
          phase ^= stage_idx == 0;
        }
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4_expert(
#else
cvt_fp16_to_fp4_expert(
#endif
    int32_t numRows, int32_t numCols, Type const* in, float const* SFScale, uint32_t* out,
    uint32_t* SFout, int32_t* mask, bool use_silu_and_mul, int n_experts) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVecT = PackedVec<Type, CVT_FP4_ELTS_PER_THREAD>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVecT) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Input tensor row/col loops.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = (gridDim.x * blockDim.x) / n_experts;
  int remainder = (gridDim.x * blockDim.x) % n_experts;
  int expert_idx;
  int tid_in_expert;
  int actual_stride;
  if (remainder > 0) {
    int bound = remainder * (stride + 1);
    if (tid < bound) {
      expert_idx = tid / (stride + 1);
      tid_in_expert = tid % (stride + 1);
      actual_stride = stride + 1;
    } else {
      expert_idx = remainder + (tid - bound) / stride;
      tid_in_expert = (tid - bound) % stride;
      actual_stride = stride;
    }
  } else {
    expert_idx = tid / stride;
    tid_in_expert = tid % stride;
    actual_stride = stride;
  }
  int m = numRows / n_experts;
  int padded_m = (m + (128 - 1)) / 128 * 128;

  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  // TODO(kaixih@nvidia): For now, we assume mask is used together with
  // silu_and_mal. Maybe we want a more general behavior of mask later. In the
  // silu case, the input last dim doubles.
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_silu_and_mul ? colsPerRow * 2 : colsPerRow;

  // Each global thread processes one element
  for (int globalIdx = tid_in_expert + expert_idx * m * colsPerRow;
       globalIdx < (expert_idx + 1) * m * colsPerRow; globalIdx += actual_stride) {
    // Calculate which row and column this global thread should process
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // Find index within the experts
    int rowIdx_in_expert = rowIdx - expert_idx * m;

    // Early exit when using masks.
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      break;
    }

    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVecT in_vec = reinterpret_cast<PackedVecT const*>(in)[inOffset];
    if (use_silu_and_mul) {
      PackedVecT in_vec_mul = reinterpret_cast<PackedVecT const*>(in)[inOffset + colsPerRow];
      silu_and_mul<Type, CVT_FP4_ELTS_PER_THREAD>(in_vec, in_vec_mul);
    }

    // Get the output tensor offset.
    // Same as inOffset because 8 elements are packed into one uint32_t.
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // Get the global scaling factor, which will be applied to the SF.
    // Note SFScale is the same as next GEMM's alpha, which is
    // (448.f / (Alpha_A / 6.f)).
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    // The actual output_scales dim is computed from the padded numCols.
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + expert_idx * padded_m * numCols_SFout;

    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_SF_VEC_SIZE,
                                                     CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, CVT_FP4_SF_VEC_SIZE, CVT_FP4_ELTS_PER_THREAD, UE8M0_SF>(
        in_vec, SFScaleVal, sf_out);
  }
#endif
}

__global__ void block_scale_interleave_kernel(int numbatches, int numRows, int numCols,
                                              uint8_t const* SFIn, uint8_t* SFOutput);
}  // namespace kernels
}  // namespace tensorrt_llm
