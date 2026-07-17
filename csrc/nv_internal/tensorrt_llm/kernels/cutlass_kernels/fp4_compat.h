/*
 * Copyright (c) 2026 by FlashInfer team.
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

#include <cuda.h>

#if defined(COMPILE_HOPPER_TMA_GEMMS) || defined(COMPILE_HOPPER_TMA_GROUPED_GEMMS) || \
    defined(CUTLASS_ENABLE_GDC_FOR_SM90)
#include "cutlass/float_subbyte.h"

namespace tensorrt_llm::kernels::cutlass_kernels {
using Fp4Type = cutlass::float_e2m1_t;
}  // namespace tensorrt_llm::kernels::cutlass_kernels

#else
#if CUDA_VERSION < 12080
#error "Native FP4 paths require CUDA 12.8 or newer outside the Hopper CUTLASS FP4 path."
#endif
#include <cuda_fp4.h>

namespace tensorrt_llm::kernels::cutlass_kernels {
using Fp4Type = __nv_fp4_e2m1;
}  // namespace tensorrt_llm::kernels::cutlass_kernels

#endif
