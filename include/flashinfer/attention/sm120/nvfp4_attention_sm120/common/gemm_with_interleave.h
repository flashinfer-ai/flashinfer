/*
 * Copyright (c) 2025 by SageAttention team.
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
 *
 * This file adapts cute::SM120::BLOCKSCALED::mma_unpack from CUTLASS/CuTe:
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cute_extension.h"

namespace nvfp4_attention {

using namespace cute;

template <class MMAOp, class GapFn, class TD, class DLayout, class TA, class ALayout, class TB,
          class BLayout, class TC, class CLayout>
CUTE_HOST_DEVICE void mma_unpack_interleaved(MMA_Traits<MMAOp> const&, Tensor<TD, DLayout>& D,
                                             Tensor<TA, ALayout> const& A_zipped,
                                             Tensor<TB, BLayout> const& B_zipped,
                                             Tensor<TC, CLayout> const& C, GapFn&& gap_fn) {
  using RegTypeD = typename remove_extent<typename MMAOp::DRegisters>::type;
  using RegTypeA = typename remove_extent<typename MMAOp::ARegisters>::type;
  using RegTypeB = typename remove_extent<typename MMAOp::BRegisters>::type;
  using RegTypeC = typename remove_extent<typename MMAOp::CRegisters>::type;
  using RegTypeSFA = typename remove_extent<typename MMAOp::SFARegisters>::type;
  using RegTypeSFB = typename remove_extent<typename MMAOp::SFBRegisters>::type;

  auto [A, SFA] = unzip_tensor(A_zipped);
  auto [B, SFB] = unzip_tensor(B_zipped);

  Tensor rA = recast<RegTypeA>(A);
  Tensor rB = recast<RegTypeB>(B);
  Tensor rD = recast<RegTypeD>(D);
  Tensor rC = recast<RegTypeC>(C);
  Tensor rSFA = recast<RegTypeSFA>(filter_zeros(SFA));
  Tensor rSFB = recast<RegTypeSFB>(filter_zeros(SFB));

  cute::SM120::BLOCKSCALED::fma_with_interleave(
      rD(0), rD(1), rD(2), rD(3), rD(4), rD(5), rD(6), rD(7), rD(8), rD(9), rD(10), rD(11), rD(12),
      rD(13), rD(14), rD(15), rA(0), rA(1), rA(2), rA(3), rB(0), rB(1), rB(2), rB(3), rB(4), rB(5),
      rB(6), rB(7), rC(0), rC(1), rC(2), rC(3), rC(4), rC(5), rC(6), rC(7), rC(8), rC(9), rC(10),
      rC(11), rC(12), rC(13), rC(14), rC(15), rSFA(0), rSFB(0), gap_fn);
}

template <class TiledMma, class GapFn, class TA, class ALayout, class TB, class BLayout, class TC,
          class CLayout>
CUTE_HOST_DEVICE void gemm_interleaved(TiledMma const& tiled_mma, Tensor<TC, CLayout>& C,
                                       Tensor<TA, ALayout> const& A, Tensor<TB, BLayout> const& B,
                                       GapFn&& gap_fn) {
  using Traits = typename TiledMma::AtomThrID;

  using MMAOp = typename TiledMma::MMA_Atom_Arch;
  mma_unpack_interleaved(MMA_Traits<MMAOp>{}, C, A, B, C, static_cast<GapFn&&>(gap_fn));
}

}  // namespace nvfp4_attention
