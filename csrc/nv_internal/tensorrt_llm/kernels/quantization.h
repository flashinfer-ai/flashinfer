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
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashinfer/fp4_layout.cuh"
#include "tensorrt_llm/common/quantization.h"

namespace tensorrt_llm {
using flashinfer::QuantizationSFLayout;

// This denotes the input and output data types of the block scale quantization.
enum class BlockScaleQuantizationType {
  FP16_TO_FP4 = 0,
  FP8_TO_FP4 = 1,
  FP16_TO_MXFP8 = 2,
};

#define PadUpFn(X, Y) ((X + Y - 1) / (Y) * (Y))

// totalCloumn should be in SFMatrix, not activation Matrix, so no sfVecSize needed.
inline int64_t computeSwizzledLayoutSFSize(int totalRow, int totalColumn, int rowSize = 128) {
  int paddedRow = PadUpFn(totalRow, rowSize);
  int paddedColumn = PadUpFn(totalColumn, 4);
  return static_cast<int64_t>(paddedRow) * paddedColumn;
}

inline int64_t computeLinearLayoutSFSize(int totalRow, int totalColumn) {
  return static_cast<int64_t>(totalRow) * totalColumn;
}

namespace kernels {

template <typename T>
void invokeQuantization(int8_t* dst, T const* src, int64_t const size, float const* scalePtr,
                        cudaStream_t stream = 0, int maxGirdSize = 0);

template <typename T, typename QuantT>
void invokePerTokenQuantization(QuantT* dst, T const* src, int64_t const numRows,
                                int64_t const numCols, float const* clampPtr, float* scalePtr,
                                float* sumPtr, tensorrt_llm::common::QuantMode quantMode,
                                cudaStream_t stream = 0);

template <typename T, int SF_VEC_SIZE>
void invokeFP4Quantization(int b, int m, int n, T const* input, float const* globalScale,
                           int64_t* output, int32_t* SFOuput, bool useUE8M0,
                           QuantizationSFLayout layout, int multiProcessorCount,
                           bool enable_pdl = false, cudaStream_t stream = 0);

template <typename T>
void invokeSiluAndMulNVFP4Quantization(void* output, void* output_scale, void* input,
                                       void* input_global_scale, void* mask, bool use_silu_and_mul,
                                       int m_topk, int k, int n_experts, cudaStream_t stream);

template <typename T>
void invokeBlockScaleInterleave(int b, int m, int m_padded, int n, int n_padded, T const* SFIn,
                                T* SFOutput, int multiProcessorCount, cudaStream_t stream = 0);

void invokeBlockScaleInterleaveReverse(int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput,
                                       int multiProcessorCount, cudaStream_t stream = 0);

template <typename T>
void invokeMxFP8Quantization(int b, int m, int n, int padded_n, T const* input, int64_t* output,
                             int32_t* SFOuput, QuantizationSFLayout layout, int multiProcessorCount,
                             bool enable_pdl = false, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace tensorrt_llm
