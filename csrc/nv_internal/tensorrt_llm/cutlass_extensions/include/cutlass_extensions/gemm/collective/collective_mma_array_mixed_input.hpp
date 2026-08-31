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

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Pre-MMA scale variant
template <int Stages_, class ClusterShape_ = cute::Shape<cute::_1, cute::_1, cute::_1>,
          class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperative>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInputPreScale {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
      cute::is_same_v<Schedule, KernelPtrArrayTmaWarpSpecializedCooperative> ||
          cute::is_same_v<Schedule, KernelPtrArrayTmaWarpSpecializedPingpong>,
      "KernelSchedule must be one of the Ptr-Array or Grouped GEMM TMA warp specialized policies");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class MixedInputScaleMode {
  kPostMma = 0,
  kPreMmaE8M0,
  // Post-MMA path where the FP8 activation additionally carries a per-block (K group=32) scale that
  // is applied together with the weight scale after each K-chunk MMA. Distinct from kPostMma
  // (per-token only) even though both fold to the upstream post-MMA dispatch policy; the difference
  // is surfaced to the collective through the trailing ElementActivationScale_ template parameter.
  kPostMmaActBlockScale,
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// ElementActivationScale_: trailing gate for the MXFP4-weight x MXFP8-activation per-block
// activation-scale path. Defaults to void so every legacy instantiation (int4a8, mxfp4-bf16,
// per-token post-MMA, pre-MMA E8M0) stays byte-identical: when it is void, HasActivationScale is
// false inside the specialization and all activation-scale code is stripped. The builder passes
// cutlass::bfloat16_t only for MixedInputScaleMode::kPostMmaActBlockScale.
template <class DispatchPolicy, class TileShape, class ElementA, class StrideA, class ElementB,
          class StrideB, class TiledMma, class GmemTiledCopyA, class SmemLayoutAtomA,
          class SmemCopyAtomA, class TransformA, class GmemTiledCopyB, class SmemLayoutAtomB,
          class SmemCopyAtomB, class TransformB, class ElementActivationScale_ = void>
struct CollectiveMmaArrayMixedInput {
  static_assert(cutlass::detail::dependent_false<ElementA>,
                "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp"
#include "cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////
