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

/*! \file
  \brief D = alpha * acc epilogue fusions where alpha may be a device scalar or
  one value per output row/column.

  The broadcast stride is a runtime value, so one instantiation serves both: a
  zero stride reads alpha_ptr[0] for every element, a unit stride reads one
  alpha per row (or column). Use PerRowScaledAcc when tokens are the GEMM's M
  extent, PerColScaledAcc when the kernel swaps A and B so tokens land on N.
*/

#ifndef FLASHINFER_FP4_GEMM_PER_TOKEN_SCALE_H_
#define FLASHINFER_FP4_GEMM_PER_TOKEN_SCALE_H_

#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc, alpha broadcast along N (one value per row of the output).
template <class ElementOutput_, class ElementCompute_, class ElementScalar_ = ElementCompute_,
          int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
          FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct PerRowScaledAcc : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_> {
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr bool IsPerRowScaleSupported = true;
};

template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute,
          class ElementScalar = ElementCompute,
          int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
          FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>
using Sm90PerRowScaledAcc =
    Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
            Sm90ColBroadcast<0, CtaTileShapeMNK, ElementScalar, ElementCompute,
                             Stride<bool, _0, int64_t>, AlignmentScalar>,
            Sm90AccFetch>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementScalar, int AlignmentScalar,
          FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::PerRowScaledAcc<ElementOutput, ElementCompute, ElementScalar, AlignmentScalar,
                            RoundStyle>,
    CtaTileShapeMNK, EpilogueTile>
    : Sm90PerRowScaledAcc<CtaTileShapeMNK,
                          typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
                          ElementCompute, ElementScalar, AlignmentScalar, RoundStyle> {
  using Impl =
      Sm90PerRowScaledAcc<CtaTileShapeMNK,
                          typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
                          ElementCompute, ElementScalar, AlignmentScalar, RoundStyle>;

  struct Arguments {
    using StrideAlpha = Stride<bool, _0, int64_t>;

    // Fallback used only when alpha_ptr is null.
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;
    // {false, _0, 0} broadcasts alpha_ptr[0]; {true, _0, 0} is one per row.
    StrideAlpha dAlpha = {bool(0), _0{}, int64_t(0)};

    operator typename Impl::Arguments() const {
      return {
          {alpha_ptr, alpha, dAlpha},  // alpha
          {},                          // acc
          {}                           // multiplies
      };
    }
  };

  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc, alpha broadcast along M (one value per column of the output).
template <class ElementOutput_, class ElementCompute_, class ElementScalar_ = ElementCompute_,
          int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
          FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct PerColScaledAcc : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_> {
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr bool IsPerColScaleSupported = true;
};

template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute,
          class ElementScalar = ElementCompute,
          int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
          FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>
using Sm90PerColScaledAcc =
    Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
            Sm90RowBroadcast<0, CtaTileShapeMNK, ElementScalar, ElementCompute,
                             Stride<_0, bool, int64_t>, AlignmentScalar>,
            Sm90AccFetch>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementScalar, int AlignmentScalar,
          FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::PerColScaledAcc<ElementOutput, ElementCompute, ElementScalar, AlignmentScalar,
                            RoundStyle>,
    CtaTileShapeMNK, EpilogueTile>
    : Sm90PerColScaledAcc<CtaTileShapeMNK,
                          typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
                          ElementCompute, ElementScalar, AlignmentScalar, RoundStyle> {
  using Impl =
      Sm90PerColScaledAcc<CtaTileShapeMNK,
                          typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
                          ElementCompute, ElementScalar, AlignmentScalar, RoundStyle>;

  struct Arguments {
    using StrideAlpha = Stride<_0, bool, int64_t>;

    // Fallback used only when alpha_ptr is null.
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;
    // {_0, false, 0} broadcasts alpha_ptr[0]; {_0, true, 0} is one per column.
    StrideAlpha dAlpha = {_0{}, bool(0), int64_t(0)};

    operator typename Impl::Arguments() const {
      return {
          {alpha_ptr, alpha, dAlpha},  // alpha
          {},                          // acc
          {}                           // multiplies
      };
    }
  };

  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::epilogue::fusion

#endif  // FLASHINFER_FP4_GEMM_PER_TOKEN_SCALE_H_
