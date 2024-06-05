/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GROUP_GEMM_CUTLASS_CUH_
#define FLASHINFER_GROUP_GEMM_CUTLASS_CUH_

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace flashinfer {

namespace group_gemm {

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename T>
__global__ void compute_cutlass_group_gemm_args(cutlass::gemm::GemmCoord* all_problems, T** ptr_x,
                                                T** ptr_w, T** ptr_y, int64_t* ld_x, int64_t* ld_w,
                                                int64_t* ld_y, T* x, T* w, T* y, int64_t* xy_indptr,
                                                int64_t* w_indices, size_t d_in, size_t d_out,
                                                bool w_column_major) {
  int i = blockIdx.x;
  int m = xy_indptr[i + 1] - xy_indptr[i], k = d_in, n = d_out;
  all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
  ptr_w[i] = w + (w_indices == nullptr ? i : w_indices[i]) * d_in * d_out;
  ptr_x[i] = x + xy_indptr[i] * d_in;
  ptr_y[i] = y + xy_indptr[i] * d_out;
  ld_x[i] = k;                       // m * k
  ld_w[i] = w_column_major ? k : n;  // k * n if column major, n * k if row major
  ld_y[i] = n;                       // m * n
}

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_CUTLASS_WRAPPER_CUH_
