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

namespace flat {

// clang-format off

// Extern template declarations for half (16 combinations)
extern template void launch_delta_rule_prefill_kernel_gbai<false, false, false, false, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, false, false, true, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, false, true, false, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, false, true, true, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, true, false, false, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, true, false, true, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, true, true, false, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, true, true, true, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, false, false, false, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, false, false, true, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, false, true, false, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, false, true, true, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, true, false, false, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, true, false, true, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, true, true, false, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, true, true, true, cutlass::arch::Sm90, half, half, float>(
    cudaStream_t, half*, float*, half const*, half const*, half const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

// Extern template declarations for nv_bfloat16 (16 combinations)
extern template void launch_delta_rule_prefill_kernel_gbai<false, false, false, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, false, false, true, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, false, true, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, false, true, true, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, true, false, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, true, false, true, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, true, true, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<false, true, true, true, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, false, false, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, false, false, true, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, false, true, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, false, true, true, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, true, false, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, true, false, true, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, true, true, false, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

extern template void launch_delta_rule_prefill_kernel_gbai<true, true, true, true, cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t, nv_bfloat16*, float*, nv_bfloat16 const*, nv_bfloat16 const*, nv_bfloat16 const*,
    float const*, float const*, float const*, int64_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

// clang-format on

}  // namespace flat
