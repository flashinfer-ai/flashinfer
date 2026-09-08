/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

// Kernel schedule tag for the SM90 grouped cooperative warp-specialized kernel
// whose CONSUMER warpgroups gather operand A themselves with cp.async while
// operand B streams through TMA from the producer warp
// (cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative_gather_a.hpp).
struct KernelPtrArrayTmaWarpSpecializedCooperativeGatherA {};

// SM90 array (grouped) TMA warp-specialized mainloop where operand B streams
// through TMA while operand A is row-GATHERED from an unpermuted activation
// buffer with cp.async, indexed by a per-group int array. Used by the MoE FC1
// GEMM to consume routed tokens without materializing the permuted copy.
//
// The A gather is issued by the CONSUMER warpgroups themselves (each
// warpgroup fills exactly the smem-A rows its own wgmma descriptors read),
// interleaved with the GMMA pipeline inside CollectiveMma::mma_gather(). The
// mainloop producer warp loads B only; the full-barrier arrival count stays
// at the CUTLASS value of 1 (NumProducerThreadEvents). See the collective
// (collective/sm90_mma_array_tma_gmma_ss_warpspecialized_gather_a.hpp) for
// the pipeline and barrier math.
template <int Stages_, class ClusterShape_ = cute::Shape<cute::_1, cute::_1, cute::_1>,
          class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperativeGatherA>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedGatherA {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;

  constexpr static int PipelineAsyncMmaStages = 1;

  static_assert(
      cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperativeGatherA, KernelSchedule>,
      "Gather-A mainloop requires the gather-A grouped cooperative warp-specialized schedule");
  // cp.async cannot multicast: every CTA gathers its own A tile. Restrict to
  // 1x1 clusters so B's TMA multicast masks stay trivial and A traffic is not
  // silently duplicated across a cluster.
  static_assert(cute::size(ClusterShape{}) == 1,
                "Gather-A mainloop v1 supports only 1x1x1 clusters");
};

}  // namespace cutlass::gemm
