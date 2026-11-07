// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0
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

// Source: ResInfer 51b906836a69fa43915040314df57bc9f23030ad
#pragma once
#include "quant_utils.cuh"
namespace flashinfer::resinfer {
__global__ void QuantizeGroupedQ(const __nv_bfloat16* __restrict__ q, uint8_t* __restrict__ q_pack,
                                 uint8_t* __restrict__ q_sf, int logical_rows, int physical_rows,
                                 int head_dim) {
  const int group = blockIdx.x;
  const int scale_groups = head_dim / CVT_FP4_SF_VEC_SIZE;
  const int base_rows = physical_rows / 2;
  const int64_t input_base = int64_t(group) * logical_rows * head_dim;
  const int64_t data_base = int64_t(group) * physical_rows * (head_dim / 2);
  const int64_t scale_base = int64_t(group) * 128 * scale_groups;
  for (int index = threadIdx.x; index < logical_rows * scale_groups; index += blockDim.x) {
    const int row = index / scale_groups;
    const int column = index % scale_groups;
    PackedVec<__nv_bfloat16> lo{}, hi{};
    load_packed_vec16(q + input_base + int64_t(row) * head_dim + column * CVT_FP4_SF_VEC_SIZE, lo,
                      hi);
    uint8_t* base_data = q_pack + data_base + int64_t(row) * (head_dim / 2) + column * 8;
    uint8_t* residue_data =
        q_pack + data_base + int64_t(base_rows + row) * (head_dim / 2) + column * 8;
    uint8_t* base_scale =
        q_sf + scale_base + compute_tiled_sf_offset<uint8_t>(row, column, head_dim);
    uint8_t* residue_scale =
        q_sf + scale_base + compute_tiled_sf_offset<uint8_t>(base_rows + row, column, head_dim);
    const u32x2 packed = cvt_warp_fp16_to_fp4_mext_r1_fast16<__nv_bfloat16, false>(
        lo, hi, 1.0f, 1.0f, base_scale, reinterpret_cast<uint32_t*>(residue_data), residue_scale);
    *reinterpret_cast<uint2*>(base_data) = make_uint2(packed.lo, packed.hi);
  }
}
}  // namespace flashinfer::resinfer
