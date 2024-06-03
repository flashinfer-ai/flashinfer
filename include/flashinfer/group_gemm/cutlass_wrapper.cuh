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

#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace flashinfer {

template <typename DType>
__global__ void group_gemm_args_kernel(
  cutlass::gemm::GemmCoord *all_problems,
  DType **ptr_y,
  DType **ptr_x,
  DType **ptr_w,
  int64_t *ld_y,
  int64_t *ld_x,
  int64_t *ld_w,
  DType *y,
  DType *x,
  DType **w,
  int64_t d_in,
  int64_t d_out
) {
  // TODO(Zihao)
}

}