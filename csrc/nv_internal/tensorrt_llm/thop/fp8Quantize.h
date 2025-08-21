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
#include <ATen/ATen.h>
#include <ATen/Tensor.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace torch_ext {
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
std::tuple<at::Tensor, at::Tensor> mxfp8_quantize(at::Tensor input, bool is_sf_swizzled_layout,
                                                  int64_t alignment, bool enable_pdl);

// x_fp32: [M, K], fp32_quantized (on the host)
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in
// linear layout. See QuantizationSFLayout enum for more details about the two layouts.
// returns fp8_quantized and block_scale_factors (on the host).
std::tuple<at::Tensor, at::Tensor> mxfp8_quantize_host(at::Tensor x_fp32,
                                                       bool is_sf_swizzled_layout = true);
at::Tensor mxfp8_dequantize_host(at::Tensor value_e4m3, at::Tensor scale_ue8m08sf,
                                 bool is_sf_swizzled_layout = true);

}  // namespace torch_ext
