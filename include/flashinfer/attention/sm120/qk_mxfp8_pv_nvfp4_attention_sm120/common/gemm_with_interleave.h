/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include <type_traits>

#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cute_extension.h"

namespace qk_mxfp8_pv_nvfp4_attention {

using namespace cute;

// Unpack the custom N32 FP8 atom exactly as CuTe's mma_unpack does, while
// retaining callback points between the four independent N8 PTX operations.
template <class MMAOp, class GapFn, class TD, class DLayout, class TA, class ALayout, class TB,
          class BLayout, class TC, class CLayout>
CUTE_HOST_DEVICE void mma_unpack_fp8_interleaved(MMA_Traits<MMAOp> const&, Tensor<TD, DLayout>& D,
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

  cute::SM120::BLOCKSCALED::fma_fp8_with_interleave(
      rD(0), rD(1), rD(2), rD(3), rD(4), rD(5), rD(6), rD(7), rD(8), rD(9), rD(10), rD(11), rD(12),
      rD(13), rD(14), rD(15), rA(0), rA(1), rA(2), rA(3), rB(0), rB(1), rB(2), rB(3), rB(4), rB(5),
      rB(6), rB(7), rC(0), rC(1), rC(2), rC(3), rC(4), rC(5), rC(6), rC(7), rC(8), rC(9), rC(10),
      rC(11), rC(12), rC(13), rC(14), rC(15), rSFA(0), rSFB(0), static_cast<GapFn&&>(gap_fn));
}

template <class TiledMma, class GapFn, class TA, class ALayout, class TB, class BLayout, class TC,
          class CLayout>
CUTE_HOST_DEVICE void gemm_fp8_interleaved(TiledMma const&, Tensor<TC, CLayout>& C,
                                           Tensor<TA, ALayout> const& A,
                                           Tensor<TB, BLayout> const& B, GapFn&& gap_fn) {
  using MMAOp = cute::SM120::BLOCKSCALED::SM120_16x32x32_TN_VS_FP8;
  mma_unpack_fp8_interleaved(MMA_Traits<MMAOp>{}, C, A, B, C, static_cast<GapFn&&>(gap_fn));
}

}  // namespace qk_mxfp8_pv_nvfp4_attention
