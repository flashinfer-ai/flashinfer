/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FLASHINFER_FP4_GEMM_EPILOGUE_SM103_H_
#define FLASHINFER_FP4_GEMM_EPILOGUE_SM103_H_

#include <cstdint>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/functional.h"

namespace flashinfer {
namespace gemm {
namespace sm103_epilogue {

// CUTLASS does not enable its generic float2 math helper for SM103A. Select
// mul.f32x2 locally so two FP32 alpha * accumulator values share one issued
// instruction without changing CUTLASS-wide architecture feature macros.
template <class Value>
struct PackedF32x2Multiplies : cutlass::multiplies<Value> {};

template <int Size>
struct PackedF32x2Multiplies<cutlass::Array<float, Size>> {
  static_assert(Size % 2 == 0, "f32x2 multiply requires an even fragment size");

  CUTLASS_HOST_DEVICE cutlass::Array<float, Size> operator()(
      cutlass::Array<float, Size> const& lhs, cutlass::Array<float, Size> const& rhs) const {
    cutlass::Array<float, Size> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    union PackedFloat2 {
      float2 values;
      unsigned long long bits;
    };

    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < Size; index += 2) {
      PackedFloat2 lhs_pair{};
      PackedFloat2 rhs_pair{};
      PackedFloat2 result_pair{};
      lhs_pair.values = make_float2(lhs[index], lhs[index + 1]);
      rhs_pair.values = make_float2(rhs[index], rhs[index + 1]);
      asm volatile("mul.f32x2 %0, %1, %2;"
                   : "=l"(result_pair.bits)
                   : "l"(lhs_pair.bits), "l"(rhs_pair.bits));
      result[index] = result_pair.values.x;
      result[index + 1] = result_pair.values.y;
    }
#else
    cutlass::multiplies<float> multiply;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < Size; ++index) {
      result[index] = multiply(lhs[index], rhs[index]);
    }
#endif
    return result;
  }
};

// Marker operation understood by CUTLASS' CollectiveBuilder. It inherits all
// metadata from the stock LinearCombination operation.
template <class ElementOutput, class ElementCompute, class ElementSource = ElementOutput,
          class ElementScalar = ElementCompute,
          cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest>
struct LinearCombinationF32x2Operation
    : cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource,
                                                   ElementScalar, RoundStyle> {};

template <class ElementOutput, class ElementCompute, class ElementSource, class ElementScalar,
          cutlass::FloatRoundStyle RoundStyle>
using LinearCombinationF32x2Impl = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, ElementOutput,
                                           ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar,
                                                   cute::Stride<cute::_0, cute::_0, int64_t>>,
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementSource>,
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<PackedF32x2Multiplies, ElementCompute,
                                               ElementCompute, RoundStyle>,
        cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar,
                                                       cute::Stride<cute::_0, cute::_0, int64_t>>,
        cutlass::epilogue::fusion::Sm90AccFetch>>;

// Layout-compatible replacement for CUTLASS Sm90LinearCombination. Keeping
// the outer beta*C node, even though ElementSource is void here, preserves the
// incumbent callback and shared-storage layout. Only alpha*acc uses FMUL2.
template <class ElementOutput, class ElementCompute, class ElementSource = ElementOutput,
          class ElementScalar = ElementCompute,
          cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest>
struct LinearCombinationF32x2
    : LinearCombinationF32x2Impl<ElementOutput, ElementCompute, ElementSource, ElementScalar,
                                 RoundStyle> {
  using Impl = LinearCombinationF32x2Impl<ElementOutput, ElementCompute, ElementSource,
                                          ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = cute::Stride<cute::_0, cute::_0, int64_t>;
    using StrideBeta = cute::Stride<cute::_0, cute::_0, int64_t>;
    StrideAlpha dAlpha = {cute::_0{}, cute::_0{}, 0};
    StrideBeta dBeta = {cute::_0{}, cute::_0{}, 0};

    operator typename Impl::Arguments() const {
      return {{{beta}, {beta_ptr}, {dBeta}}, {}, {{{alpha}, {alpha_ptr}, {dAlpha}}, {}, {}}, {}};
    }
  };

  using Impl::Impl;
};

}  // namespace sm103_epilogue
}  // namespace gemm
}  // namespace flashinfer

namespace cutlass {
namespace epilogue {
namespace fusion {

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementSource, class ElementScalar,
          FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    flashinfer::gemm::sm103_epilogue::LinearCombinationF32x2Operation<
        ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile>
    : flashinfer::gemm::sm103_epilogue::LinearCombinationF32x2<
          typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute,
          ElementSource, ElementScalar, RoundStyle> {
  using Impl = flashinfer::gemm::sm103_epilogue::LinearCombinationF32x2<
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute,
      ElementSource, ElementScalar, RoundStyle>;
  using Operation = flashinfer::gemm::sm103_epilogue::LinearCombinationF32x2Operation<
      ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Impl::Impl;
};

}  // namespace fusion
}  // namespace epilogue
}  // namespace cutlass

#endif  // FLASHINFER_FP4_GEMM_EPILOGUE_SM103_H_
