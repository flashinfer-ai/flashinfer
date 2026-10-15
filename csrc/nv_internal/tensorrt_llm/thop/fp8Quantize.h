/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <tvm/ffi/container/tuple.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/utils.h"

using tvm::ffi::Tensor;
using tvm::ffi::Tuple;

// colIdx and totalCloumn should be in SFMatrix, not activation Matrix, so no sfVecSize needed.
inline int computeSFIndex(int rowIdx, int colIdx, int totalRow, int totalColumn,
                          tensorrt_llm::QuantizationSFLayout layout, bool useUE8M0 = false) {
  constexpr int kColumnGroup0Size = 4;
  constexpr int kRowGroup0Size = 32;
  constexpr int kRowGroup1Size = kRowGroup0Size * 4;

  // Swizzled layout is used as default layout.
  if (layout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
    // int paddedRow = PadUpFn(totalRow, 128);
    int paddedColumn = PadUpFn(totalColumn, 4);

    int columnIdxInGroup0 = colIdx % kColumnGroup0Size;
    int columnGroupIdx = colIdx / kColumnGroup0Size;
    constexpr int columnGroupStride = kColumnGroup0Size * kRowGroup1Size;

    int rowIdxInGroup0 = rowIdx % kRowGroup0Size;
    int rowIdxInGroup1 = (rowIdx % kRowGroup1Size) / kRowGroup0Size;
    int rowGroupIdx = rowIdx / kRowGroup1Size;
    constexpr int rowGroup1Stride = kColumnGroup0Size;
    constexpr int rowGroup0Stride = kColumnGroup0Size * rowGroup1Stride;
    int rowGroupStride = kRowGroup1Size * paddedColumn;

    return columnIdxInGroup0 + columnGroupIdx * columnGroupStride +
           rowIdxInGroup0 * rowGroup0Stride + rowIdxInGroup1 * rowGroup1Stride +
           rowGroupIdx * rowGroupStride;
  }
  // Linear layout is only used in E2M1AndUFP8SFScaleToFloatV2.
  else if (layout == tensorrt_llm::QuantizationSFLayout::LINEAR) {
    // no padding needed. totalColumn is multiple of kVecSize.
    return rowIdx * totalColumn + colIdx;
  } else {
    TLLM_THROW("Other layout not implemented yet.");
  }
}

// input: [M, K], fp16/bf16_quantized
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in
// linear layout. See QuantizationSFLayout enum for more details about the two layouts.
// alignment: sfVecSize
// returns fp8_quantized and block_scale_factors.
void mxfp8_quantize(TensorView input, TensorView valMxFP8, TensorView scaleFP8SF,
                    int64_t sfSwizzleLayout, int64_t alignment, bool enable_pdl);

// input: [B, M, K], fp16/bf16; or [B, M, 2K] with gate values followed by up values per row when
// use_silu_and_mul is set.
// mask: [B], int32, number of valid rows per expert.
// Writes the first min(mask[b], M) rows of each expert (silu(gate) * up when use_silu_and_mul is
// set) as MXFP8: e4m3 values [B, M, round_up(K, 128)] and UE8M0 block scales [B, round_up(M, 128),
// round_up(K, 128) / 32] in the 128x4 swizzled layout per expert.
void silu_and_mul_mxfp8_experts_quantize(TensorView input, TensorView mask, TensorView valMxFP8,
                                         TensorView scaleFP8SF, bool use_silu_and_mul,
                                         bool enable_pdl);

// x_fp32: [M, K], fp32_quantized (on the host)
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in
// linear layout. See QuantizationSFLayout enum for more details about the two layouts.
// returns fp8_quantized and block_scale_factors (on the host).
void mxfp8_quantize_host(TensorView x_fp32, TensorView fp8_tensor, TensorView scale_tensor,
                         int64_t sfSwizzleLayout = 2);

void mxfp8_dequantize_host(TensorView value_e4m3, TensorView scale_ue8m08sf,
                           TensorView float_tensor, int64_t sfSwizzleLayout = 2);
