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

#include "cutlass_fused_moe_kernels.cuh"
#include "moe_kernels.h"

namespace tensorrt_llm::kernels::cutlass_kernels {
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;
#ifdef ENABLE_FP8
// template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, half>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_fp8_e4m3>;
#endif
#endif
#ifdef ENABLE_FP4
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half>;
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half, half>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half, half>;
template class CutlassMoeFCRunner<half, __nv_fp4_e2m1>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1>;
#endif
#endif

// Explicit instantiations for finalizeMoeRoutingKernelLauncher to ensure
// symbols are emitted in the JIT library for common data types.
INSTANTIATE_FINALIZE_MOE_ROUTING(half, half, half);
INSTANTIATE_FINALIZE_MOE_ROUTING(float, float, float);
#ifdef ENABLE_BF16
INSTANTIATE_FINALIZE_MOE_ROUTING(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
#endif
}  // namespace tensorrt_llm::kernels::cutlass_kernels
