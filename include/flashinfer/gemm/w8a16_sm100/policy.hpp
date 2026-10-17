/*
 * Copyright (c) 2026 by FlashInfer team.
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
#pragma once

// Dispatch tags for FlashInfer's fork of CUTLASS 4.4's sm100 mixed-input GEMM
// (kernel.hpp / collective.hpp / builder.hpp in this directory, forked from
// cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp,
// cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp and
// cutlass/gemm/collective/builders/sm100_mixed_input_umma_builder.inl).  The fork
// is keyed on these tags so it coexists with the upstream specializations; the
// upstream tags stay untouched.  The tags deliberately do not derive from
// KernelScheduleSm100MixedInputGemm (or KernelScheduleSm100), only from
// KernelSchedule1Sm / 2Sm, so no upstream CollectiveBuilder matches them.

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

// Kernel-level schedule carried by the dispatch policy (matched with is_kernel_tag_of).
template <int SchedulerPipelineStageCount_, int AccumulatorPipelineStageCount_>
struct KernelW8A16MixedInputTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Mainloop dispatch policy (mirrors MainloopSm100TmaUmmaWarpSpecializedMixedInput).
template <int Load2TransformPipelineStageCount_, int Transform2MmaPipelineStageCount_,
          int SchedulerPipelineStageCount_, int AccumulatorPipelineStageCount_,
          class ClusterShape_ = Shape<_1, _1, _1>, class ArchTag_ = arch::Sm100>
struct MainloopW8A16Sm100MixedInput {
  constexpr static int Load2TransformPipelineStageCount = Load2TransformPipelineStageCount_;
  constexpr static int Load2MmaPipelineStageCount = Load2TransformPipelineStageCount_;
  constexpr static int Transform2MmaPipelineStageCount = Transform2MmaPipelineStageCount_;
  constexpr static detail::KernelInputTransformType InputTransformType =
      detail::KernelInputTransformType::MixedInput;
  using ClusterShape = ClusterShape_;
  using ArchTag = ArchTag_;
  using Schedule = KernelW8A16MixedInputTransformSm100<SchedulerPipelineStageCount_,
                                                       AccumulatorPipelineStageCount_>;
  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = Load2TransformPipelineStageCount;
};

// Builder-level schedule tags (what the launcher passes to CollectiveBuilder).
// *Smem* variants keep the converted A operand in shared memory (SS MMA atoms) instead
// of TMEM; that is what frees TMEM for two 256-column accumulator stages (version B4).
struct KernelW8A16MixedInputSm100 {};
struct KernelW8A16MixedInputSmem {};
struct KernelW8A16MixedInput1SmSm100 final : KernelSchedule1Sm, KernelW8A16MixedInputSm100 {};
struct KernelW8A16MixedInput2SmSm100 final : KernelSchedule2Sm, KernelW8A16MixedInputSm100 {};
struct KernelW8A16MixedInput1SmSmemSm100 final : KernelSchedule1Sm,
                                                 KernelW8A16MixedInputSm100,
                                                 KernelW8A16MixedInputSmem {};
struct KernelW8A16MixedInput2SmSmemSm100 final : KernelSchedule2Sm,
                                                 KernelW8A16MixedInputSm100,
                                                 KernelW8A16MixedInputSmem {};

}  // namespace cutlass::gemm
