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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

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

template <>
struct cutlass_dtype<__nv_fp8_e4m3> {
  using type = cutlass::float_e4m3_t;
};

template <>
struct cutlass_dtype<__nv_fp8_e5m2> {
  using type = cutlass::float_e5m2_t;
};

template <typename DTypeIn, typename DTypeOut>
__global__ void compute_sm80_cutlass_group_gemm_args(
    cutlass::gemm::GemmCoord* all_problems, DTypeIn** x_ptr, DTypeIn** w_ptr, DTypeOut** y_ptr,
    int64_t* x_ld, int64_t* w_ld, int64_t* y_ld, DTypeIn* x, DTypeIn* w, DTypeOut* y,
    int64_t* xy_indptr, int64_t* w_indices, size_t d_in, size_t d_out, bool w_column_major) {
  int i = blockIdx.x;
  int m = xy_indptr[i + 1] - xy_indptr[i], k = d_in, n = d_out;
  all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
  w_ptr[i] = w + (w_indices == nullptr ? i : w_indices[i]) * k * n;
  x_ptr[i] = x + xy_indptr[i] * k;
  y_ptr[i] = y + xy_indptr[i] * n;
  x_ld[i] = k;                       // m * k
  w_ld[i] = w_column_major ? k : n;  // k * n if column major, n * k if row major
  y_ld[i] = n;                       // m * n
}

template <typename DTypeIn, typename DTypeOut, typename ProblemShape, typename StrideA,
          typename StrideB, typename StrideCD>
__global__ void compute_sm90_cutlass_group_gemm_args(
    ProblemShape* all_problems, DTypeIn** x_ptr, DTypeIn** w_ptr, DTypeOut** y_ptr,
    StrideA* x_stride, StrideB* w_stride, StrideCD* y_stride, DTypeIn* x, DTypeIn* w, DTypeOut* y,
    int64_t* xy_indptr, int64_t* w_indices, size_t d_in, size_t d_out, bool w_column_major) {
  int i = blockIdx.x;
  int m = xy_indptr[i + 1] - xy_indptr[i], k = d_in, n = d_out;
  all_problems[i] = ProblemShape(m, n, k);
  w_ptr[i] = w + (w_indices == nullptr ? i : w_indices[i]) * k * n;
  x_ptr[i] = x + xy_indptr[i] * k;
  y_ptr[i] = y + xy_indptr[i] * n;

  x_stride[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  w_stride[i] = w_column_major ? cutlass::make_cute_packed_stride(StrideB{}, {k, n, 1})
                               : cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  y_stride[i] = cutlass::make_cute_packed_stride(StrideCD{}, {m, n, 1});
}

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_CUTLASS_WRAPPER_CUH_
