/*
 * Copyright (c) 2025 by FlashInfer team.
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
// Extern template declarations to prevent implicit instantiation in the dispatcher.
// Explicit instantiations are in separate generated files for parallel compilation.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "cutlass/arch/arch.h"

namespace flat {

// clang-format off

#define FOR_EACH_BOOL_4(MACRO, ...) \
    MACRO(false, false, false, false, __VA_ARGS__) \
    MACRO(false, false, false, true, __VA_ARGS__) \
    MACRO(false, false, true, false, __VA_ARGS__) \
    MACRO(false, false, true, true, __VA_ARGS__) \
    MACRO(false, true, false, false, __VA_ARGS__) \
    MACRO(false, true, false, true, __VA_ARGS__) \
    MACRO(false, true, true, false, __VA_ARGS__) \
    MACRO(false, true, true, true, __VA_ARGS__) \
    MACRO(true, false, false, false, __VA_ARGS__) \
    MACRO(true, false, false, true, __VA_ARGS__) \
    MACRO(true, false, true, false, __VA_ARGS__) \
    MACRO(true, false, true, true, __VA_ARGS__) \
    MACRO(true, true, false, false, __VA_ARGS__) \
    MACRO(true, true, false, true, __VA_ARGS__) \
    MACRO(true, true, true, false, __VA_ARGS__) \
    MACRO(true, true, true, true, __VA_ARGS__)

#define DECLARE_TEMPLATE_INSTANCE(is_gva, needs_beta, needs_alpha, init_state, ctype) \
extern template void launch_delta_rule_prefill_kernel_gbai<is_gva, needs_beta, needs_alpha, init_state, cutlass::arch::Sm90, ctype, ctype, float>( \
    cudaStream_t, ctype*, float*, ctype const*, ctype const*, ctype const*, \
    float const*, float const*, float const*, int64_t const*, uint8_t*, int32_t, int32_t, \
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

// Extern template declarations for half
FOR_EACH_BOOL_4(DECLARE_TEMPLATE_INSTANCE, half)

// Extern template declarations for nv_bfloat16
FOR_EACH_BOOL_4(DECLARE_TEMPLATE_INSTANCE, nv_bfloat16)

#undef DECLARE_TEMPLATE_INSTANCE
#undef FOR_EACH_BOOL_4

// clang-format on

}  // namespace flat
