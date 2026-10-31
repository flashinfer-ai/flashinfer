// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0
// Adapted from ResInfer 51b906836a69fa43915040314df57bc9f23030ad,
// csrc/nvfp4/cuda/nvfp4_quant_kernels.cu (NVIDIA Apache-2.0 base).
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
#pragma once
#include "quant_utils.cuh"

namespace flashinfer::resinfer {
// Production SM100 pack16 path, specialized to adjacent base/residue rows.
template <typename T>
__global__ void __launch_bounds__(512, 4)
    QuantizeRowPairs(int rows, int cols, const T* input, const float* scales, int num_scales,
                     uint32_t* output, uint8_t* output_scales) {
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  if (group >= cols / 16) return;
  float base_scale = scales[0];
  float residue_scale = scales[num_scales == 2 ? 1 : 0];
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    PackedVec<T> lo, hi;
    load_packed_vec16(input + int64_t(row) * cols + group * 16, lo, hi);
    auto base_sf = output_scales + compute_tiled_sf_offset<uint8_t>(2 * row, group, cols);
    auto residue_sf = output_scales + compute_tiled_sf_offset<uint8_t>(2 * row + 1, group, cols);
    auto base_out = output + int64_t(2 * row) * (cols / 8) + group * 2;
    auto residue_out = base_out + cols / 8;
    auto packed = cvt_warp_fp16_to_fp4_mext_r1_fast16<T, false>(lo, hi, base_scale, residue_scale,
                                                                base_sf, residue_out, residue_sf);
    *reinterpret_cast<uint2*>(base_out) = make_uint2(packed.lo, packed.hi);
  }
}
}  // namespace flashinfer::resinfer
