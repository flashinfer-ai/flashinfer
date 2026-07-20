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

#include "cute/arch/copy_sm90.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/collective/mixed_input_utils.hpp"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective::detail {

constexpr int int4_group_size = 128;
constexpr int mxfp4_group_size = 32;

template <class ElementWeight>
struct DefaultWeightScaleGroupSize;

template <>
struct DefaultWeightScaleGroupSize<cutlass::float_e2m1_t> {
  static constexpr int value = 32;
};

template <>
struct DefaultWeightScaleGroupSize<cutlass::int4b_t> {
  static constexpr int value = 128;
};

typedef uint32_t __nv_fp4x8_storage_t;
typedef uint32_t __nv_bf16x2_storage_t;
typedef uint32_t __nv_int4x8_storage_t;
typedef uint64_t __nv_fp8x8_storage_t;
typedef cutlass::uint128_t __nv_bf16x8_storage_t;

// -----------------------------------------------------------------------
// Interleaved version of the bits of four consecutive fp4 values (i.e. 16-bits):
//     s000000eem000000         (1st fp4)
//        s000000eem000000      (2nd fp4)
//           s000000eem000000   (3rd fp4)
//     0sm0ee0000000000         (4th fp4)
// -----------------------------------------------------------------------

__device__ __inline__ __nv_bf16x8_storage_t psx_cvt_triton_fp4x8_to_bf16x8_interleaved(
    const __nv_fp4x8_storage_t fp4x8) {
  __nv_bf16x8_storage_t bf16x8_raw;
  __nv_bfloat162* bf16x2_raw = reinterpret_cast<__nv_bfloat162*>(&bf16x8_raw);

  // 0x7e807e80 -> BF16 [126, 126]
  uint32_t bias_raw = 0x7e807e80U;
  __nv_bfloat162 bias = reinterpret_cast<__nv_bfloat162&>(bias_raw);

  __nv_fp4x8_storage_t first_fp4 = fp4x8 & 0x81C081C0U;
  bf16x2_raw[0] = __hmul2(reinterpret_cast<__nv_bfloat162&>(first_fp4), bias);

  __nv_fp4x8_storage_t second_fp4 = (fp4x8 << 3) & 0x81C081C0U;
  bf16x2_raw[1] = __hmul2(reinterpret_cast<__nv_bfloat162&>(second_fp4), bias);

  __nv_fp4x8_storage_t third_fp4 = (fp4x8 << 6) & 0x81C081C0U;
  bf16x2_raw[2] = __hmul2(reinterpret_cast<__nv_bfloat162&>(third_fp4), bias);

  __nv_fp4x8_storage_t fourth_fp4;
  __nv_fp4x8_storage_t fourth_fp4_s = (fp4x8 << 1) & 0x80008000U;
  __nv_fp4x8_storage_t fourth_fp4_e = fp4x8 >> 3;

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  asm volatile(
      "{\n"
      "  lop3.b32 %0, %0, %1, %2, %3;\n"
      "}\n"
      : "+r"(fourth_fp4_e)
      : "n"(0x01800180U), "r"(fourth_fp4_s), "n"(immLut));

  __nv_fp4x8_storage_t fourth_fp4_m = fp4x8 >> 7;

  asm volatile(
      "{\n"
      "  lop3.b32 %0, %1, %2, %3, %4;\n"
      "}\n"
      : "=r"(fourth_fp4)
      : "r"(fourth_fp4_m), "n"(0x00400040U), "r"(fourth_fp4_e), "n"(immLut));

  bf16x2_raw[3] = __hmul2(reinterpret_cast<__nv_bfloat162&>(fourth_fp4), bias);

  return bf16x8_raw;
}

inline __device__ unsigned prmt(unsigned hi, unsigned lo, unsigned select_code) {
  unsigned res = 0;

  asm volatile(
      "{\n"
      "prmt.b32 %0, %1, %2, %3;\n"
      "}\n"
      : "=r"(res)
      : "r"(lo), "r"(hi), "r"(select_code));

  return res;
}

__constant__ static __nv_fp8x4_storage_t HIGH_E4M3s_LUT_[2] = {0x03020100U, 0x03020100U};
__constant__ static __nv_fp8x4_storage_t LOW_E4M3s_LUT_[2] = {0xFFFEFC00U, 0xFFFEFC00U};

__device__ __inline__ __nv_fp8x4_storage_t cvt_lut_fp4_to_bf16(const unsigned index) {
  auto lane_id = threadIdx.x & 0x1;
  __nv_fp8x4_storage_t h4b_lut = HIGH_E4M3s_LUT_[lane_id];
  __nv_fp8x4_storage_t l4b_lut = LOW_E4M3s_LUT_[lane_id];

  __nv_fp8x4_storage_t lut_res = prmt(h4b_lut, l4b_lut, index);

  return lut_res;
}

__device__ __inline__ __nv_bf16x8_storage_t psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved(
    const __nv_fp4x8_storage_t fp4x8) {
  // interleaved version
  // input fp4x8: 7564 3120
  // output bf16x8: 7654 3210

  __nv_bf16x8_storage_t bf16x8_raw;
  __nv_bf16x2_storage_t* bf16x2_raw = reinterpret_cast<__nv_bf16x2_storage_t*>(&bf16x8_raw);

  __nv_fp8x4_storage_t h_fp8x4_0to1_bits = (fp4x8 & 0xC0C0C0C0U) >> 6;  // 7632
  __nv_fp8x4_storage_t l_fp8x4_0to1_bits = (fp4x8 & 0x0C0C0C0CU) >> 2;  // 5410

  unsigned h4b_em_fp4x4 = (fp4x8 & 0x77770000U) >> 16U;
  unsigned l4b_em_fp4x4 = (fp4x8 & 0x00007777U);

  __nv_fp8x4_storage_t h4b_2to9_bits = cvt_lut_fp4_to_bf16(h4b_em_fp4x4);  // 7564
  __nv_fp8x4_storage_t l4b_2to9_bits = cvt_lut_fp4_to_bf16(l4b_em_fp4x4);  // 3120

  bf16x2_raw[0] = prmt(l_fp8x4_0to1_bits, l4b_2to9_bits, 0x5240U) << 6U;  // 1 0
  bf16x2_raw[1] = prmt(h_fp8x4_0to1_bits, l4b_2to9_bits, 0x5341U) << 6U;  // 3 2

  bf16x2_raw[2] = prmt(l_fp8x4_0to1_bits, h4b_2to9_bits, 0x7260U) << 6U;  // 5 4
  bf16x2_raw[3] = prmt(h_fp8x4_0to1_bits, h4b_2to9_bits, 0x7361U) << 6U;  // 7 6

  return bf16x8_raw;
}

// FP4 E2M1 [0, 0.5, 1, 1.5] encoded as FP8 E4M3.
__constant__ static uint32_t FP4_POS_E4M3s_REG1_[2] = {0x3C383000, 0x3C383000};
// FP4 E2M1 [2, 3, 4, 6] encoded as FP8 E4M3.
__constant__ static uint32_t FP4_POS_E4M3s_REG2_[2] = {0x4C484440, 0x4C484440};

__device__ __inline__ __nv_fp8x8_storage_t psx_cvt_lut_prmt_fp4x8_to_fp8x8(
    const __nv_fp4x8_storage_t fp4x8) {
  __nv_fp8x8_storage_t fp8x8_raw;
  __nv_fp8x4_storage_t* fp8x4_raw = reinterpret_cast<__nv_fp8x4_storage_t*>(&fp8x8_raw);

  __nv_fp8x4_storage_t hb_sign_fp8x4 = (fp4x8 & 0x80808080U);
  __nv_fp8x4_storage_t lb_sign_fp8x4 = (fp4x8 & 0x08080808U) << 4U;

  __nv_fp8x4_storage_t h4b_sign_fp8x4 = prmt(hb_sign_fp8x4, lb_sign_fp8x4, 0x7362U);
  __nv_fp8x4_storage_t l4b_sign_fp8x4 = prmt(hb_sign_fp8x4, lb_sign_fp8x4, 0x5140U);

  // PRMT consumes only the low 16 bits of its selector in generic mode, so
  // the low half does not need its high selector half cleared.
  unsigned l4b_em_fp4x4 = fp4x8 & 0x77777777U;
  unsigned h4b_em_fp4x4 = l4b_em_fp4x4 >> 16U;

  auto lane_id = threadIdx.x & 0x1;
  uint32_t h4b_lut = FP4_POS_E4M3s_REG2_[lane_id];
  uint32_t l4b_lut = FP4_POS_E4M3s_REG1_[lane_id];
  __nv_fp8x4_storage_t h4b_em_fp8x4 = prmt(h4b_lut, l4b_lut, h4b_em_fp4x4);
  __nv_fp8x4_storage_t l4b_em_fp8x4 = prmt(h4b_lut, l4b_lut, l4b_em_fp4x4);

  fp8x4_raw[0] = l4b_sign_fp8x4 | l4b_em_fp8x4;
  fp8x4_raw[1] = h4b_sign_fp8x4 | h4b_em_fp8x4;

  return fp8x8_raw;
}

__device__ __inline__ __nv_fp8x8_storage_t psx_cvt_lut_prmt_fp4x8_to_fp8x8_preprocessed_signs(
    const __nv_fp4x8_storage_t fp4x8) {
  __nv_fp8x8_storage_t fp8x8_raw;
  __nv_fp8x4_storage_t* fp8x4_raw = reinterpret_cast<__nv_fp8x4_storage_t*>(&fp8x8_raw);

  // Offline preprocessing keeps each nibble's low 3 EM bits in place, but
  // repacks signs so outputs 0..3 are already in byte bit7 and outputs 4..7
  // are in bit3 of each byte.  That removes the runtime sign-gather PRMTs.
  // PRMT consumes only the low 16 bits of its selector in generic mode, so
  // the low half does not need its high selector half cleared.
  unsigned l4b_em_fp4x4 = fp4x8 & 0x77777777U;
  unsigned h4b_em_fp4x4 = l4b_em_fp4x4 >> 16U;

  auto lane_id = threadIdx.x & 0x1;
  uint32_t h4b_lut = FP4_POS_E4M3s_REG2_[lane_id];
  uint32_t l4b_lut = FP4_POS_E4M3s_REG1_[lane_id];
  __nv_fp8x4_storage_t h4b_em_fp8x4 = prmt(h4b_lut, l4b_lut, h4b_em_fp4x4);
  __nv_fp8x4_storage_t l4b_em_fp8x4 = prmt(h4b_lut, l4b_lut, l4b_em_fp4x4);

  fp8x4_raw[0] = (fp4x8 & 0x80808080U) | l4b_em_fp8x4;
  fp8x4_raw[1] = ((fp4x8 << 4U) & 0x80808080U) | h4b_em_fp8x4;

  return fp8x8_raw;
}

// [ 0,  1,  2,  3] encoded as FP8
__constant__ static uint32_t POS_E4M3s_REG1_[2] = {0x44403800, 0x44403800};
// [ 4,  5,  6,  7] encoded as FP8
__constant__ static uint32_t POS_E4M3s_REG2_[2] = {0x4E4C4A48, 0x4E4C4A48};
// [-8, -7, -6, -5] encoded as FP8
__constant__ static uint32_t NEG_E4M3s_REG1_[2] = {0xCACCCED0, 0xCACCCED0};
// [-4, -3, -2, -1] encoded as FP8
__constant__ static uint32_t NEG_E4M3s_REG2_[2] = {0xB8C0C4C8, 0xB8C0C4C8};

__device__ __inline__ __nv_fp8x8_storage_t psx_cvt_lut_prmt_int4x8_to_fp8x8(
    const __nv_int4x8_storage_t int4x8) {
  __nv_fp8x8_storage_t fp8x8_raw;
  __nv_fp8x4_storage_t* fp8x4_raw = reinterpret_cast<__nv_fp8x4_storage_t*>(&fp8x8_raw);

  // View the input as reg
  uint32_t reg = reinterpret_cast<const uint32_t&>(int4x8);

  // Determines if to get from the signed or unsigned candidates
  uint32_t sign = (reg & 0x88888888) >> 1;

  // Ignore sign bit when indexing into LUT
  uint32_t lut_idx = (reg & 0x77777777);

  // Signed is OR'd with 0x32103210 to find the correct value in the LUT
  const uint32_t final_prmt_base = 0x32103210;

  auto lane_id = threadIdx.x & 0x1;
  uint32_t POS_E4M3s_REG1 = POS_E4M3s_REG1_[lane_id];
  uint32_t POS_E4M3s_REG2 = POS_E4M3s_REG2_[lane_id];
  uint32_t NEG_E4M3s_REG1 = NEG_E4M3s_REG1_[lane_id];
  uint32_t NEG_E4M3s_REG2 = NEG_E4M3s_REG2_[lane_id];

  asm volatile(
      "{\n"
      "  .reg .b32 pos_f8s, neg_f8s;\n"
      "  .reg .b32 lut1, sign1, prmt0, prmt1;\n"
      "  or.b32 prmt0, %4, %3;\n"
      "  prmt.b32 pos_f8s, %5, %6, %2;\n"
      "  prmt.b32 neg_f8s, %7, %8, %2;\n"
      "  prmt.b32 %0, pos_f8s, neg_f8s, prmt0;\n"
      "  shr.u32 lut1, %2, 16;\n"
      "  shr.u32 sign1, %3, 16;\n"
      "  or.b32 prmt1, %4, sign1;\n"
      "  prmt.b32 pos_f8s, %5, %6, lut1;\n"
      "  prmt.b32 neg_f8s, %7, %8, lut1;\n"
      "  prmt.b32 %1, pos_f8s, neg_f8s, prmt1;\n"
      "}\n"
      : "=r"(fp8x4_raw[0]), "=r"(fp8x4_raw[1])
      : "r"(lut_idx), "r"(sign), "r"(final_prmt_base), "r"(POS_E4M3s_REG1), "r"(POS_E4M3s_REG2),
        "r"(NEG_E4M3s_REG1), "r"(NEG_E4M3s_REG2));

  return fp8x8_raw;
}

template <class...>
using MixedInputVoid = void;

template <class Collective, class = void>
struct MixedInputFusedE8M0PreMmaScale {
  static constexpr bool value = false;
};

template <class Collective>
struct MixedInputFusedE8M0PreMmaScale<Collective,
                                      MixedInputVoid<decltype(Collective::FusedE8M0PreMmaScale)>> {
  static constexpr bool value = Collective::FusedE8M0PreMmaScale;
};

template <class Collective, class = void>
struct MixedInputFoldedWeightScaleStorage {
  static constexpr bool value = false;
};

template <class Collective>
struct MixedInputFoldedWeightScaleStorage<
    Collective, MixedInputVoid<decltype(Collective::WeightScaleBulkCopyBytes),
                               decltype(Collective::WeightScaleTransactionBytes)>> {
  static constexpr bool value = true;
};

template <class Collective>
struct MixedGroupedGemmInputUtils {
 private:
  using KernelSchedule = typename Collective::KernelSchedule;
  using ConversionMode = typename Collective::ConversionMode;
  using SmemLayoutA = typename Collective::SmemLayoutA;
  using SmemLayoutB = typename Collective::SmemLayoutB;
  using SmemLayoutScale = typename Collective::SmemLayoutScale;
  using SmemLayoutActivationScale = typename Collective::SmemLayoutActivationScale;
  using SwappedElementA = typename Collective::SwappedElementA;
  using SwappedElementB = typename Collective::SwappedElementB;
  using RealSwappedElementA = typename Collective::RealSwappedElementA;
  using RealSwappedElementB = typename Collective::RealSwappedElementB;
  using ElementScale = typename Collective::ElementScale;
  using ElementZero = typename Collective::ElementZero;
  using NonVoidElementActivationScale = typename Collective::NonVoidElementActivationScale;
  using SmemCopyAtomScale = typename Collective::SmemCopyAtomScale;
  static constexpr auto KernelConversionMode = Collective::KernelConversionMode;
  static constexpr auto ModeHasScales = Collective::ModeHasScales;
  static constexpr auto UseScaleLookupTable = Collective::UseScaleLookupTable;
  static constexpr auto UseFP4ToBF16LookupTable = Collective::UseFP4ToBF16LookupTable;
  static constexpr auto UseFP4ToFP8LookupTable = Collective::UseFP4ToFP8LookupTable;
  static constexpr auto UseInt4ToFP8LookupTable = Collective::UseInt4ToFP8LookupTable;
  static constexpr auto HasActivationScale = Collective::HasActivationScale;
  static constexpr bool FusedE8M0PreMmaScale = MixedInputFusedE8M0PreMmaScale<Collective>::value;
  static constexpr bool HasFoldedWeightScaleStorage =
      MixedInputFoldedWeightScaleStorage<Collective>::value;

 public:
  static constexpr auto elements_per_smem_scale() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    } else if constexpr (ModeHasScales) {
      return cute::cosize_v<SmemLayoutScale>;
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Type not handled in scale smem allocation.");
    }
  }

  static constexpr auto elements_per_smem_zero() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert ||
                  KernelConversionMode == ConversionMode::ConvertAndScale) {
      return 0;
    } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      return cute::cosize_v<SmemLayoutScale>;
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Type not handled in scale smem allocation.");
    }
  }

  // These methods use some the public members of the class. For that reason, we define them after
  // the public section.
  static constexpr uint32_t compute_tma_transaction_bytes_mk() {
    return cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) *
                                  static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementA>));
  }

  static constexpr uint32_t compute_tma_transaction_bytes_nk() {
    return cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) *
                                  static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementB>));
  }

  static constexpr uint32_t compute_tma_transaction_bytes_extra() {
    constexpr uint32_t bulk_copy_alignment_bytes = 16;
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    } else if constexpr (ModeHasScales) {
      constexpr uint32_t scale_tx_bytes =
          cutlass::bits_to_bytes(size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) *
                                 static_cast<uint32_t>(cute::sizeof_bits_v<ElementScale>));
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        if constexpr (FusedE8M0PreMmaScale || HasFoldedWeightScaleStorage) {
          static_assert(Collective::WeightScaleBulkCopyBytes % bulk_copy_alignment_bytes == 0,
                        "Each folded weight-scale bulk copy must be 16B aligned.");
        } else {
          static_assert(scale_tx_bytes % bulk_copy_alignment_bytes == 0,
                        "Each scale bulk copy must be 16B aligned.");
        }
        if constexpr (HasActivationScale) {
          constexpr uint32_t activation_scale_tx_bytes = cutlass::bits_to_bytes(
              size<0>(SmemLayoutActivationScale{}) * size<1>(SmemLayoutActivationScale{}) *
              static_cast<uint32_t>(cute::sizeof_bits_v<NonVoidElementActivationScale>));
          if constexpr (FusedE8M0PreMmaScale || HasFoldedWeightScaleStorage) {
            return Collective::WeightScaleTransactionBytes + activation_scale_tx_bytes;
          } else {
            return scale_tx_bytes + activation_scale_tx_bytes;
          }
        } else {
          if constexpr (FusedE8M0PreMmaScale || HasFoldedWeightScaleStorage) {
            return Collective::WeightScaleTransactionBytes;
          } else {
            return scale_tx_bytes;
          }
        }
      } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        // Scale and zero share smem layout
        static_assert(scale_tx_bytes % bulk_copy_alignment_bytes == 0,
                      "Each scale bulk copy must be 16B aligned.");
        constexpr uint32_t zero_tx_bytes =
            cutlass::bits_to_bytes(size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) *
                                   static_cast<uint32_t>(cute::sizeof_bits_v<ElementZero>));
        static_assert(zero_tx_bytes % bulk_copy_alignment_bytes == 0,
                      "Each zero bulk copy must be 16B aligned.");
        return scale_tx_bytes + zero_tx_bytes;
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Type not handled in tma transaction bytes computation.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Type not handled in tma transaction bytes computation.");
    }
  }

  /// Utilities to copy A from smem to RF
  template <class SmemTiledCopyA, class TensorASmemView, class TensorACopyView>
  CUTLASS_DEVICE static void copy_tensors_A(SmemTiledCopyA const& smem_tiled_copy_A,
                                            TensorASmemView const& tCsA,
                                            TensorACopyView& tCrA_copy_view, int k_block,
                                            int read_stage) {
    if (k_block < size<2>(tCsA.shape())) {
      copy(smem_tiled_copy_A, tCsA(_, _, k_block, read_stage), tCrA_copy_view(_, _, k_block));
    }
  }

  /// Utilities to copy Scales for A from smem to RF
  template <class... Ts, class... Us>
  CUTLASS_DEVICE static void copy_tensors_SFA(cute::tuple<Ts...> const& partitioned_mma_extra_info,
                                              cute::tuple<Us...> const& tiled_copy_and_views,
                                              int k_block, int read_stage) {
    // We are starting a new k-tile so copy the scale
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // nothing to do
    } else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
      auto tCrS_copy_view = cute::get<1>(tiled_copy_and_views);
      auto tCsS = cute::get<0>(partitioned_mma_extra_info);
      copy(smem_tiled_copy_S, tCsS(_, _, k_block, read_stage), tCrS_copy_view(_, _, k_block));
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        // Nothing extra to do
      } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        auto tCsZ = cute::get<2>(partitioned_mma_extra_info);
        auto tCrZ_copy_view = cute::get<2>(tiled_copy_and_views);
        copy(smem_tiled_copy_S, tCsZ(_, _, k_block, read_stage), tCrZ_copy_view(_, _, k_block));
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in A -> RF path.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in A -> RF path.");
    }
  }

  /// Utilities to copy A and extra inputs from smem to RF
  template <class SmemTiledCopyA, class TensorASmemView, class TensorACopyView, class... Ts,
            class... Us>
  CUTLASS_DEVICE static void copy_tensors_MK(SmemTiledCopyA const& smem_tiled_copy_A,
                                             TensorASmemView const& tCsA,
                                             TensorACopyView& tCrA_copy_view,
                                             cute::tuple<Ts...> const& partitioned_mma_extra_info,
                                             cute::tuple<Us...> const& tiled_copy_and_views,
                                             int k_block, int read_stage) {
    if (k_block < size<2>(tCsA.shape())) {
      copy(smem_tiled_copy_A, tCsA(_, _, k_block, read_stage), tCrA_copy_view(_, _, k_block));
    }

    if (k_block == 0) {
      // We are starting a new k-tile so copy the scale
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // nothing to do
      } else if constexpr (ModeHasScales) {
        auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
        auto tCrS_copy_view = cute::get<1>(tiled_copy_and_views);
        auto tCsS = cute::get<0>(partitioned_mma_extra_info);
        copy(smem_tiled_copy_S, tCsS(_, _, k_block, read_stage), tCrS_copy_view(_, _, k_block));
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
          // Nothing extra to do
        } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tCsZ = cute::get<2>(partitioned_mma_extra_info);
          auto tCrZ_copy_view = cute::get<2>(tiled_copy_and_views);
          copy(smem_tiled_copy_S, tCsZ(_, _, k_block, read_stage), tCrZ_copy_view(_, _, k_block));
        } else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                        "Conversion mode not handled in A -> RF path.");
        }
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in A -> RF path.");
      }
    }
  }

  // The core converter uses a lookup table to converts i4 -> 8 bit value.
  template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut, class EngineScale,
            class LayoutScale>
  CUTLASS_DEVICE static void lookup_table_convert(  // Accept mutable temporaries
      Tensor<EngineIn, LayoutIn> const& src, Tensor<EngineOut, LayoutOut>&& dst,
      Tensor<EngineScale, LayoutScale> const& scales_neg,
      Tensor<EngineScale, LayoutScale> const& scales_pos) {
    lookup_table_convert(src, dst, scales_neg, scales_pos);
  }
  template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut, class EngineScale,
            class LayoutScale>
  CUTLASS_DEVICE static void lookup_table_convert(
      Tensor<EngineIn, LayoutIn> const& src, Tensor<EngineOut, LayoutOut>& dst,
      Tensor<EngineScale, LayoutScale> const& scales_neg,
      Tensor<EngineScale, LayoutScale> const& scales_pos) {
    constexpr int N = cute::cosize(LayoutIn{});
    static_assert(N == 4 || N == 8);
    static_assert(cosize(LayoutScale{}) <= N / 4,
                  "at least 4 consecutive weights must share the same scale.");
    using SrcArray = cutlass::Array<cutlass::int4b_t, 8>;
    using DstArray = cutlass::Array<RealSwappedElementB, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, N / 4, sizeof(DstArray)>;

    // View the input as reg
    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r = cute::recast<RegArray>(dst)(0);

    // Determines if to get from the signed or unsigned candidates
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    uint32_t sign;  // ((reg & 0x88888888) | 0x64206420) >> 1
    asm volatile(
        "{\n"
        "  lop3.b32 %0, %1, %2, %3, %4;\n"
        "}\n"
        : "=r"(sign)
        : "r"(src_reg), "n"(0x88888888), "n"(0x64206420), "n"(immLut));
    sign = sign >> 1;

    // Ignore sign bit when indexing into LUT
    uint32_t lut_idx = src_reg & 0x77777777;
    Tensor scales_neg_ = cute::filter(scales_neg);
    Tensor scales_pos_ = cute::filter(scales_pos);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i, lut_idx >>= 16, sign >>= 16) {
      auto&& scale_neg_ = reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_neg_(i));
      auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_pos_(i));
      asm volatile(
          "{\n"
          "  .reg .b32 pos, neg                    ;\n"
          "  prmt .b32 neg, %3, %4, %1             ;\n"
          "  prmt .b32 pos, %5, %6, %1             ;\n"
          "  prmt .b32 %0, pos, neg, %2            ;\n"
          "}\n"
          : "=r"(r[i])
          : "r"(lut_idx), "r"(sign), "r"(scale_neg_[0]), "r"(scale_neg_[1]), "r"(scale_pos_[0]),
            "r"(scale_pos_[1]));
    }
  }

  template <class EngineIn, class LayoutIn, class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE static void fp4tobf16_lookup_table_convert(  // Accept mutable temporaries
      Tensor<EngineIn, LayoutIn> const& src, Tensor<EngineOut, LayoutOut>&& dst) {
    fp4tobf16_lookup_table_convert(src, dst);
  }

  template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut>
  CUTLASS_DEVICE static void fp4tobf16_lookup_table_convert(Tensor<EngineIn, LayoutIn> const& src,
                                                            Tensor<EngineOut, LayoutOut>& dst) {
    // View the input as reg
    auto&& src_ = cute::recast<__nv_fp4x8_storage_t>(src)(0);
    auto&& dst_ = cute::recast<__nv_bf16x8_storage_t>(dst)(0);

    // dst_ = psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved(src_);
    dst_ = psx_cvt_triton_fp4x8_to_bf16x8_interleaved(src_);
  }

  template <class EngineIn, class LayoutIn, class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE static void int4tofp8_lookup_table_convert(  // Accept mutable temporaries
      Tensor<EngineIn, LayoutIn> const& src, Tensor<EngineOut, LayoutOut>&& dst) {
    int4tofp8_lookup_table_convert(src, dst);
  }

  template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut>
  CUTLASS_DEVICE static void int4tofp8_lookup_table_convert(Tensor<EngineIn, LayoutIn> const& src,
                                                            Tensor<EngineOut, LayoutOut>& dst) {
    // View the input as reg
    auto&& src_ = cute::recast<__nv_int4x8_storage_t>(src)(0);
    auto&& dst_ = cute::recast<__nv_fp8x8_storage_t>(dst)(0);

    dst_ = psx_cvt_lut_prmt_int4x8_to_fp8x8(src_);
  }

  template <class EngineIn, class LayoutIn, class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE static void fp4tofp8_lookup_table_convert(  // Accept mutable temporaries
      Tensor<EngineIn, LayoutIn> const& src, Tensor<EngineOut, LayoutOut>&& dst) {
    fp4tofp8_lookup_table_convert(src, dst);
  }

  template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut>
  CUTLASS_DEVICE static void fp4tofp8_lookup_table_convert(Tensor<EngineIn, LayoutIn> const& src,
                                                           Tensor<EngineOut, LayoutOut>& dst) {
    auto&& src_ = cute::recast<__nv_fp4x8_storage_t>(src)(0);
    auto&& dst_ = cute::recast<__nv_fp8x8_storage_t>(dst)(0);

#if defined(CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS)
    dst_ = psx_cvt_lut_prmt_fp4x8_to_fp8x8_preprocessed_signs(src_);
#else
    dst_ = psx_cvt_lut_prmt_fp4x8_to_fp8x8(src_);
#endif
  }

  __device__ __inline__ static void fp4tofp8_fused_e8m0_pre_mma_convert_pair(
      __nv_fp4x8_storage_t fp4x8_0, __nv_fp4x8_storage_t fp4x8_1, __nv_fp8x8_storage_t& fp8x8_raw_0,
      __nv_fp8x8_storage_t& fp8x8_raw_1, uint32_t lo_exp_offset, uint32_t hi_exp_offset) {
    // One WGMMA A operand lane contributes two fp4x8 registers whose low
    // fp8x4 chunks share one row scale, and high chunks share the other.
    __nv_fp8x4_storage_t* fp8x4_raw_0 = reinterpret_cast<__nv_fp8x4_storage_t*>(&fp8x8_raw_0);
    __nv_fp8x4_storage_t* fp8x4_raw_1 = reinterpret_cast<__nv_fp8x4_storage_t*>(&fp8x8_raw_1);

    uint32_t const fp4_raw_0 = reinterpret_cast<uint32_t const&>(fp4x8_0);
    uint32_t const fp4_raw_1 = reinterpret_cast<uint32_t const&>(fp4x8_1);
    uint32_t const em_selector_0 = fp4_raw_0 & 0x77777777U;
    uint32_t const em_selector_1 = fp4_raw_1 & 0x77777777U;
    constexpr uint32_t fp4_codes_0_to_3_em_bias = 0x0c080000U;
    constexpr uint32_t fp4_codes_4_to_7_em_bias = 0x1c181410U;
    uint32_t const lo_l4b_exp_offseted_lut =
        (lo_exp_offset * 0x08080800U) + fp4_codes_0_to_3_em_bias;
    uint32_t const lo_h4b_exp_offseted_lut =
        (lo_exp_offset * 0x08080808U) + fp4_codes_4_to_7_em_bias;
    uint32_t const hi_l4b_exp_offseted_lut =
        (hi_exp_offset * 0x08080800U) + fp4_codes_0_to_3_em_bias;
    uint32_t const hi_h4b_exp_offseted_lut =
        (hi_exp_offset * 0x08080808U) + fp4_codes_4_to_7_em_bias;

    uint32_t const lo_em_fp8x4_0 =
        prmt(lo_h4b_exp_offseted_lut, lo_l4b_exp_offseted_lut, em_selector_0);
    uint32_t const lo_em_fp8x4_1 =
        prmt(lo_h4b_exp_offseted_lut, lo_l4b_exp_offseted_lut, em_selector_1);

#if defined(CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS)
    fp8x4_raw_0[0] = (fp4_raw_0 & 0x80808080U) | lo_em_fp8x4_0;
    fp8x4_raw_1[0] = (fp4_raw_1 & 0x80808080U) | lo_em_fp8x4_1;
#else
    uint32_t const hb_sign_fp8x4_0 = fp4_raw_0 & 0x80808080U;
    uint32_t const hb_sign_fp8x4_1 = fp4_raw_1 & 0x80808080U;
    uint32_t const lb_sign_fp8x4_0 = (fp4_raw_0 & 0x08080808U) << 4U;
    uint32_t const lb_sign_fp8x4_1 = (fp4_raw_1 & 0x08080808U) << 4U;
    uint32_t const l4b_sign_fp8x4_0 = prmt(hb_sign_fp8x4_0, lb_sign_fp8x4_0, 0x5140U);
    uint32_t const l4b_sign_fp8x4_1 = prmt(hb_sign_fp8x4_1, lb_sign_fp8x4_1, 0x5140U);
    uint32_t const h4b_sign_fp8x4_0 = prmt(hb_sign_fp8x4_0, lb_sign_fp8x4_0, 0x7362U);
    uint32_t const h4b_sign_fp8x4_1 = prmt(hb_sign_fp8x4_1, lb_sign_fp8x4_1, 0x7362U);

    fp8x4_raw_0[0] = l4b_sign_fp8x4_0 | lo_em_fp8x4_0;
    fp8x4_raw_1[0] = l4b_sign_fp8x4_1 | lo_em_fp8x4_1;
#endif

    uint32_t const hi_em_fp8x4_0 =
        prmt(hi_h4b_exp_offseted_lut, hi_l4b_exp_offseted_lut, em_selector_0 >> 16U);
    uint32_t const hi_em_fp8x4_1 =
        prmt(hi_h4b_exp_offseted_lut, hi_l4b_exp_offseted_lut, em_selector_1 >> 16U);

#if defined(CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS)
    fp8x4_raw_0[1] = ((fp4_raw_0 << 4U) & 0x80808080U) | hi_em_fp8x4_0;
    fp8x4_raw_1[1] = ((fp4_raw_1 << 4U) & 0x80808080U) | hi_em_fp8x4_1;
#else
    fp8x4_raw_0[1] = h4b_sign_fp8x4_0 | hi_em_fp8x4_0;
    fp8x4_raw_1[1] = h4b_sign_fp8x4_1 | hi_em_fp8x4_1;
#endif
  }

  template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut>
  CUTLASS_DEVICE static void fp4tofp8_fused_e8m0_pre_mma_convert_pair(
      Tensor<EngineIn, LayoutIn> const& src0, Tensor<EngineIn, LayoutIn> const& src1,
      Tensor<EngineOut, LayoutOut>& dst0, Tensor<EngineOut, LayoutOut>& dst1,
      uint32_t lo_exp_offset, uint32_t hi_exp_offset) {
    auto&& src0_ = cute::recast<__nv_fp4x8_storage_t>(src0)(0);
    auto&& src1_ = cute::recast<__nv_fp4x8_storage_t>(src1)(0);
    auto&& dst0_ = cute::recast<__nv_fp8x8_storage_t>(dst0)(0);
    auto&& dst1_ = cute::recast<__nv_fp8x8_storage_t>(dst1)(0);

    fp4tofp8_fused_e8m0_pre_mma_convert_pair(src0_, src1_, dst0_, dst1_, lo_exp_offset,
                                             hi_exp_offset);
  }

  /// Utilities to dequantize A.
  template <class Layout>
  CUTLASS_DEVICE static void static_check_scale(Layout const& tensor) {
    static_assert(shape<0>(Layout{}) >= 4 && stride<0>(Layout{}) == 0,
                  "At least 4 adjacent weights in a thread must share the same scale.");
  }
  template <class Engine, class Layout>
  CUTLASS_DEVICE static void static_check_scale(Tensor<Engine, Layout> const& tensor) {
    static_check_scale(flatten(Layout{}));
  }

  template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut, class... Ts>
  CUTLASS_DEVICE static void dequantize_A_kblock(Tensor<EngineIn, LayoutIn> const& tCrA_load,
                                                 Tensor<EngineOut, LayoutOut>& tCrA_mma,
                                                 cute::tuple<Ts...>& partitioned_extra_info,
                                                 int const k_block) {
    static_assert(is_rmem<EngineIn>::value,
                  "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value,
                  "Output tensor for A conversion must come from registers");
    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma(_, _, k_block);

    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
                         "The first mode of tensor src must be contiguous in memory");
    // try to make the size of the first mode equal to 32bit
    int constexpr NumValPerSrcReg =
        cute::min(decltype(size(src(_, 0)))::value, ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1, -1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1, -1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(dst_vm); ++i) {
        LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
      }
    } else if constexpr (UseScaleLookupTable) {
      constexpr int num_elements = decltype(size(src))::value;
      static_assert(is_same_v<RealSwappedElementA, cutlass::int4b_t>,
                    "Lookup table only supports int4 being the quant type now.");
      static_assert(sizeof_bits_v<ElementScale> == 64,
                    "Lookup table only supports 8 8bit scale values now.");
      static_assert(num_elements % 4 == 0 && num_elements >= 4,
                    "Lookup table requires a vector size of 4x when converting.");

      Tensor tCrS_neg = cute::get<1>(partitioned_extra_info);
      auto&& tCrS_pos =
          cute::get<2>(partitioned_extra_info);  // modification to its value is needed
      Tensor scales_neg = tCrS_neg(_, _, k_block);
      Tensor scales_pos = tCrS_pos(_, _, k_block);
      CUTE_STATIC_ASSERT_V(cute::size(src) == cute::size(scales_neg));

      static_check_scale(scales_neg);
      static_check_scale(scales_pos);
      Tensor scales_neg_vm =
          cute::group_modes<1, -1>(cute::zipped_divide(scales_neg, Int<NumValPerSrcReg>{}));
      Tensor scales_pos_vm =
          cute::group_modes<1, -1>(cute::zipped_divide(scales_pos, Int<NumValPerSrcReg>{}));

      if (k_block == 0) {
        Tensor scales_neg_vm_ = filter(scales_neg_vm);
        Tensor scales_pos_vm_ = filter(scales_pos_vm);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(scales_neg_vm_.layout()); ++i) {
          auto&& scale_neg_ =
              reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_neg_vm_(i));
          auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 2>&>(scales_pos_vm_(i));
          constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;
          asm volatile(
              "{\n"
              "  lop3 .b32 %0, %2, %4, %5, %6;\n"
              "  xor  .b32 %1, %3, %5;        \n"
              "}\n"
              : "=r"(scale_pos_[0]), "=r"(scale_pos_[1])
              : "r"(scale_neg_[0]), "r"(scale_neg_[1]), "n"(0xFFFFFF00), "n"(0x80808080),
                "n"(immLut));
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(dst_vm); ++i) {
        lookup_table_convert(src_vm(_, i), dst_vm(_, i), scales_neg_vm(_, i), scales_pos_vm(_, i));
      }
    } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      Tensor scales = cute::get<1>(partitioned_extra_info)(_, _, k_block);
      CUTE_STATIC_ASSERT_V(size(src) == size(scales));
      Tensor scales_vm =
          cute::group_modes<1, -1>(cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));

      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            dst_vm(j, i) *= scales_vm(j, i);
          }
        }
      } else {
        auto stage = make_tensor_like<ElementScale>(src_vm(_, 0));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), stage);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            stage(j) *= scales_vm(j, i);
          }
          LayoutAwareConvert(stage, dst_vm(_, i));
        }
      }
    } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      static_assert(is_same_v<ElementScale, ElementZero>,
                    "ElementScale and ElementZero must be the same.");
      Tensor scales = cute::get<1>(partitioned_extra_info)(_, _, k_block);
      Tensor zeros = cute::get<3>(partitioned_extra_info)(_, _, k_block);
      CUTE_STATIC_ASSERT_V(size(src) == size(scales));
      CUTE_STATIC_ASSERT_V(size(src) == size(zeros));
      Tensor scales_vm =
          cute::group_modes<1, -1>(cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));
      Tensor zeros_vm =
          cute::group_modes<1, -1>(cute::zipped_divide(zeros, Int<NumValPerSrcReg>{}));

      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            dst_vm(j, i) = dst_vm(j, i) * scales_vm(j, i) + zeros_vm(j, i);
          }
        }
      } else {
        auto stage = make_tensor_like<ElementScale>(src_vm(_, 0));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), stage);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            stage(j) = stage(j) * scales_vm(j, i) + zeros_vm(j, i);
          }
          LayoutAwareConvert(stage, dst_vm(_, i));
        }
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "No A data is loaded.");
    }
  }

  template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut>
  CUTLASS_DEVICE static void convert_A_slot(Tensor<EngineIn, LayoutIn> const& src,
                                            Tensor<EngineOut, LayoutOut>& dst) {
    static_assert(is_rmem<EngineIn>::value,
                  "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value,
                  "Output tensor for A conversion must come from registers");
    using SrcType = typename EngineIn::value_type;

    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
                         "The first mode of tensor src must be contiguous in memory");
    CUTE_STATIC_ASSERT_V(size(src) == size(dst));
    // try to make the size of the first mode equal to 32bit
    int constexpr NumValPerSrcReg =
        cute::min(decltype(size(src(_, 0)))::value, ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1, -1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1, -1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));

    // KernelConversionMode == ConversionMode::DirectConvert
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(dst_vm); ++i) {
      if constexpr (UseFP4ToBF16LookupTable) {
        fp4tobf16_lookup_table_convert(src_vm(_, i), dst_vm(_, i));
      } else if constexpr (UseFP4ToFP8LookupTable) {
        fp4tofp8_lookup_table_convert(src_vm(_, i), dst_vm(_, i));
      } else if constexpr (UseInt4ToFP8LookupTable) {
        int4tofp8_lookup_table_convert(src_vm(_, i), dst_vm(_, i));
      } else {
        LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
      }
    }
  }

  template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut>
  CUTLASS_DEVICE static void convert_A_kblock(Tensor<EngineIn, LayoutIn> const& tCrA_load,
                                              Tensor<EngineOut, LayoutOut>& tCrA_mma,
                                              int const k_block) {
    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma(_, _, k_block);
    convert_A_slot(src, dst);
  }

  template <int KBlock, class EngineIn, class EngineOut, class LayoutIn, class LayoutOut>
  CUTLASS_DEVICE static void convert_A_kblock(Tensor<EngineIn, LayoutIn> const& tCrA_load,
                                              Tensor<EngineOut, LayoutOut>& tCrA_mma,
                                              cute::Int<KBlock> k_block) {
    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma(_, _, k_block);
    convert_A_slot(src, dst);
  }

  template <int KBlock, class EngineIn, class EngineOut, class LayoutIn, class LayoutOut,
            class EngineScale, class LayoutScale>
  CUTLASS_DEVICE static void convert_A_kblock_fused_e8m0_pre_mma_raw_scale_to_slot(
      Tensor<EngineIn, LayoutIn> const& tCrA_load, Tensor<EngineOut, LayoutOut>& tCrA_mma_slot,
      Tensor<EngineScale, LayoutScale>& scale_values, cute::Int<KBlock>) {
    static_assert(FusedE8M0PreMmaScale, "This helper is only for fused e8m0 pre-MMA scale.");
    static_assert(UseFP4ToFP8LookupTable,
                  "Fused e8m0 pre-MMA scale currently supports MXFP4 x FP8 only.");
    static_assert(is_rmem<EngineIn>::value,
                  "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value,
                  "Output tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineScale>::value,
                  "Scale tensor for A conversion must come from registers");
    using SrcType = typename EngineIn::value_type;
    using ScaleScalar = ElementScale;
    static_assert(cute::is_same_v<typename EngineScale::value_type, ScaleScalar>,
                  "Raw fused e8m0 scale tensor must use scalar e8m0 elements.");

    Tensor src = tCrA_load(_, _, cute::Int<KBlock>{});
    Tensor dst = tCrA_mma_slot;
    Tensor scales = scale_values(_, _, cute::Int<KBlock>{});

    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
                         "The first mode of tensor src must be contiguous in memory");
    CUTE_STATIC_ASSERT_V(size(src) == size(dst));
    CUTE_STATIC_ASSERT_V(size(src) == size(scales));

    int constexpr NumValPerSrcReg =
        cute::min(decltype(size(src(_, 0)))::value, ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1, -1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1, -1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));
    Tensor scales_vm =
        cute::group_modes<1, -1>(cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));

    auto scale_values_0 = cute::filter(scales_vm(_, Int<0>{}));
    constexpr int ScaleValueCount = decltype(size(scale_values_0))::value;
    constexpr int DstVecCount = decltype(size<1>(dst_vm))::value;
    static_assert(ScaleValueCount == 2 || ScaleValueCount == NumValPerSrcReg,
                  "Fused e8m0 pre-MMA raw scale expects either two compact row scales or one scale "
                  "per fp4 lane.");
    static_assert((DstVecCount % 2) == 0,
                  "Fused e8m0 pre-MMA pair conversion expects an even number of fp4x8 operands.");

    constexpr int HiScaleIndex = (ScaleValueCount == NumValPerSrcReg) ? (NumValPerSrcReg / 2) : 1;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DstVecCount; i += 2) {
      auto row_scales = cute::filter(scales_vm(_, i));
      ScaleScalar const lo_scale = row_scales(0);
      ScaleScalar const hi_scale = row_scales(HiScaleIndex);
      uint32_t const lo_exp_offset = static_cast<uint32_t>(lo_scale.storage);
      uint32_t const hi_exp_offset = static_cast<uint32_t>(hi_scale.storage);
      auto src_vec0 = src_vm(_, i);
      auto src_vec1 = src_vm(_, i + 1);
      auto dst_vec0 = dst_vm(_, i);
      auto dst_vec1 = dst_vm(_, i + 1);
      fp4tofp8_fused_e8m0_pre_mma_convert_pair(src_vec0, src_vec1, dst_vec0, dst_vec1,
                                               lo_exp_offset, hi_exp_offset);
    }
  }

  template <int KBlock, int ScalePairCount, class EngineScale, class LayoutScale,
            class LoOffsetArray, class HiOffsetArray>
  CUTLASS_DEVICE static void cache_A_kblock_fused_e8m0_pre_mma_exp_offsets(
      Tensor<EngineScale, LayoutScale> const& scales, cute::Int<KBlock>, cute::Int<ScalePairCount>,
      LoOffsetArray& lo_exp_offsets, HiOffsetArray& hi_exp_offsets) {
    static_assert(FusedE8M0PreMmaScale, "This helper is only for fused e8m0 pre-MMA scale.");
    using ScaleScalar = typename EngineScale::value_type;
    constexpr int NumValPerSrcReg = 8;
    Tensor scales_vm =
        cute::group_modes<1, -1>(cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));
    static_assert(decltype(size<1>(scales_vm))::value == ScalePairCount * 2,
                  "Fused e8m0 pre-MMA scale tensor must match A operand pair layout.");

    cute::for_each(cute::make_seq<ScalePairCount>{}, [&](auto pair_c) {
      constexpr int pair = decltype(pair_c)::value;
      constexpr int scale_vec = pair * 2;
      Tensor row_scales = scales_vm(_, Int<scale_vec>{});
      constexpr int ScaleValueCount = decltype(size(row_scales))::value;
      static_assert(ScaleValueCount == 2 || ScaleValueCount == NumValPerSrcReg,
                    "Fused e8m0 pre-MMA raw scale expects either two compact row scales or one "
                    "scale per fp4 lane.");
      constexpr int HiScaleIndex = (ScaleValueCount == NumValPerSrcReg) ? (NumValPerSrcReg / 2) : 1;
      ScaleScalar const lo_scale = row_scales(0);
      ScaleScalar const hi_scale = row_scales(HiScaleIndex);
      constexpr int cache_index = KBlock * ScalePairCount + pair;
      lo_exp_offsets[cache_index] = static_cast<uint32_t>(lo_scale.storage);
      hi_exp_offsets[cache_index] = static_cast<uint32_t>(hi_scale.storage);
    });
  }

  template <int KBlock, int ScalePairCount, class EngineIn, class EngineOut, class LayoutIn,
            class LayoutOut, class LoOffsetArray, class HiOffsetArray>
  CUTLASS_DEVICE static void convert_A_kblock_fused_e8m0_pre_mma_exp_offsets_to_slot(
      Tensor<EngineIn, LayoutIn> const& tCrA_load, Tensor<EngineOut, LayoutOut>& tCrA_mma_slot,
      cute::Int<KBlock>, cute::Int<ScalePairCount>, LoOffsetArray const& lo_exp_offsets,
      HiOffsetArray const& hi_exp_offsets) {
    static_assert(FusedE8M0PreMmaScale, "This helper is only for fused e8m0 pre-MMA scale.");
    static_assert(UseFP4ToFP8LookupTable,
                  "Fused e8m0 pre-MMA scale currently supports MXFP4 x FP8 only.");
    static_assert(is_rmem<EngineIn>::value,
                  "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value,
                  "Output tensor for A conversion must come from registers");
    using SrcType = typename EngineIn::value_type;

    Tensor src = tCrA_load(_, _, cute::Int<KBlock>{});
    Tensor dst = tCrA_mma_slot;

    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
                         "The first mode of tensor src must be contiguous in memory");
    CUTE_STATIC_ASSERT_V(size(src) == size(dst));

    int constexpr NumValPerSrcReg =
        cute::min(decltype(size(src(_, 0)))::value, ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1, -1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1, -1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));

    constexpr int DstVecCount = decltype(size<1>(dst_vm))::value;
    static_assert((DstVecCount % 2) == 0,
                  "Fused e8m0 pre-MMA pair conversion expects an even number of fp4x8 operands.");
    static_assert(
        ScalePairCount * 2 == DstVecCount,
        "Fused e8m0 pre-MMA scale cache must provide one scale pair per fp4x8 operand pair.");

    cute::for_each(cute::make_seq<ScalePairCount>{}, [&](auto pair_c) {
      constexpr int pair = decltype(pair_c)::value;
      constexpr int i = pair * 2;
      auto src_vec0 = src_vm(_, i);
      auto src_vec1 = src_vm(_, i + 1);
      auto dst_vec0 = dst_vm(_, i);
      auto dst_vec1 = dst_vm(_, i + 1);
      constexpr int cache_index = KBlock * ScalePairCount + pair;
      uint32_t const lo_exp_offset = lo_exp_offsets[cache_index];
      uint32_t const hi_exp_offset = hi_exp_offsets[cache_index];
      fp4tofp8_fused_e8m0_pre_mma_convert_pair(src_vec0, src_vec1, dst_vec0, dst_vec1,
                                               lo_exp_offset, hi_exp_offset);
    });
  }

  /// Utilities for any additional inputs inside of the TMA load
  template <class Params, class TensorStorage, class... Ts>
  CUTLASS_DEVICE static auto partition_extra_tma_inputs(Params const& mainloop_params,
                                                        cute::tuple<Ts...> const& load_inputs,
                                                        TensorStorage& shared_tensors,
                                                        uint2 const& cluster_local_block_id,
                                                        int const m_coord, int const l_coord) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple();
    } else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()),
                              SmemLayoutScale{});  // (BLK_M,BLK_K,PIPE)
      Tensor gS_mkl = get<2>(load_inputs);
      auto block_tma_s = mainloop_params.tma_load_scale.get_slice(cluster_local_block_id.y);
      Tensor gS = gS_mkl(_, _, m_coord, _, l_coord);  // (BLK_M,BLK_K,k)

      Tensor tSgS = block_tma_s.partition_S(gS);  // (TMA,TMA_M,TMA_K,k)
      Tensor tSsS = block_tma_s.partition_D(sS);  // (TMA,TMA_M,TMA_K,PIPE)
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tSgS, tSsS);
      } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()),
                                SmemLayoutScale{});  // (BLK_M,BLK_K,PIPE)
        Tensor gZ_mkl = get<3>(load_inputs);
        auto block_tma_z = mainloop_params.tma_load_zero.get_slice(cluster_local_block_id.y);
        Tensor gZ = gZ_mkl(_, _, m_coord, _, l_coord);  // (BLK_M,BLK_K,k)

        Tensor tZgZ = block_tma_z.partition_S(gZ);  // (TMA,TMA_M,TMA_K,k)
        Tensor tZsZ = block_tma_z.partition_D(sZ);  // (TMA,TMA_M,TMA_K,PIPE)
        return cute::make_tuple(tSgS, tSsS, tZgZ, tZsZ);
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled for input partitioning.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled for input partitioning.");
    }
  }

  /// Utilities for partitioning extra inputs for loading from smem in the mainloop.
  template <class ThreadMma, class TensorStorage>
  CUTLASS_DEVICE static auto partition_extra_mma_info(ThreadMma const& mma_thread_slice,
                                                      TensorStorage& shared_tensors) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // nothing to do
      return cute::make_tuple();
    } else if constexpr (UseScaleLookupTable) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()),
                              SmemLayoutScale{});  // (BLK_M,BLK_SCALE_K,PIPE)
      Tensor tCsS = mma_thread_slice.partition_A(sS);
      Tensor tCrS = make_tensor<ElementScale>(
          mma_thread_slice.partition_fragment_A(sS(_, _, Int<0>{})).layout());

      return cute::make_tuple(tCsS, tCrS);
    } else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()),
                              SmemLayoutScale{});  // (BLK_M,BLK_SCALE_K,PIPE)
      Tensor tCsS = mma_thread_slice.partition_A(sS);
      Tensor tCrS = make_tensor<ElementScale>(
          mma_thread_slice.partition_fragment_A(sS(_, _, Int<0>{})).layout());

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tCsS, tCrS);
      } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()),
                                SmemLayoutScale{});  // (BLK_M,BLK_SCALE_K,PIPE)
        Tensor tCsZ = mma_thread_slice.partition_A(sZ);
        Tensor tCrZ = make_tensor<ElementZero>(
            mma_thread_slice.partition_fragment_A(sZ(_, _, Int<0>{})).layout());
        return cute::make_tuple(tCsS, tCrS, tCsZ, tCrZ);
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in A -> RF path.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in A -> RF path.");
    }
  }

  /// Returns the tiled copy and copy views for the extra inputs.
  template <class TiledMma, class... Ts>
  CUTLASS_DEVICE static auto retile_extra_mma_info(TiledMma const& tiled_mma,
                                                   cute::tuple<Ts...>& partitioned_extra_info,
                                                   int const warp_group_thread_idx) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // nothing to do
      return cute::make_tuple();
    } else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S = make_tiled_copy_A(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(warp_group_thread_idx);
      Tensor tCrS_copy_view =
          smem_thr_copy_S.retile_D(cute::get<1>(partitioned_extra_info));  // (CPY,CPY_M,CPY_K)

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view);
      } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor tCrZ_copy_view =
            smem_thr_copy_S.retile_D(cute::get<3>(partitioned_extra_info));  // (CPY,CPY_M,CPY_K)
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view, tCrZ_copy_view);
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in A -> RF path.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in A -> RF path.");
    }
  }
};

template <class Collective>
struct MixedInputUtilsSM100 {
 private:
  using KernelSchedule = typename Collective::KernelSchedule;
  using ConversionMode = typename Collective::ConversionMode;
  using SmemLayoutA = typename Collective::SmemLayoutA;
  using SmemLayoutB = typename Collective::SmemLayoutB;
  using ElementScale = typename Collective::ElementScale;
  using ElementZero = typename Collective::ElementZero;
  static constexpr auto KernelConversionMode = Collective::KernelConversionMode;

 public:
  // Helper functions to select packing for conversion
  template <class SrcType, class DstType, int Cosize>
  struct select_packing {  // Naive packing policy

    static constexpr auto value() {
      return Int<cute::gcd(Cosize,
                           32 / cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>))>{};
    }
  };

  /// (Designed for separate transform pipeline in Blackwell)
  /// Utilities to dequantize A.
  template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut, class... Ts>
  CUTLASS_DEVICE static void dequantize_A_kblock_for_transform(
      Tensor<EngineIn, LayoutIn> const& tArA, Tensor<EngineOut, LayoutOut>& tArACompute,
      cute::tuple<Ts...> const& partitioned_extra_info, int const k_block) {
    static_assert(is_rmem<EngineIn>::value,
                  "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value,
                  "Output tensor for A conversion must come from registers");
    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    auto src = tArA(_, _, _, k_block);
    auto dst = tArACompute(_, _, _, k_block);
    auto pSrc = raw_pointer_cast(src.data());
    auto pDst = const_cast<DstType*>(raw_pointer_cast(dst.data()));
    constexpr int num_elements = decltype(size(src))::value;

    constexpr int pack = decltype(select_packing<SrcType, DstType, num_elements>::value())::value;
    using Converter = cutlass::NumericArrayConverter<DstType, SrcType, pack,
                                                     cutlass::FloatRoundStyle::round_to_nearest>;
    using SrcArray = cutlass::Array<SrcType, pack>;
    using DstArray = cutlass::Array<DstType, pack>;
    constexpr int DstElementsPerReg = 32 / sizeof_bits_v<DstType>;
    using RegArray = cutlass::AlignedArray<uint32_t, pack / DstElementsPerReg, sizeof(DstArray)>;

    auto src_arr = recast<SrcArray>(src);
    auto dst_arr = recast<DstArray>(dst);

    Tensor dst_vm = cute::group_modes<1, -1>(cute::zipped_divide(dst, pack));

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      cute::transform(src_arr, dst_arr, Converter::convert);
    } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      auto const& scales = cute::get<1>(partitioned_extra_info)(_, _, _, k_block);

      CUTE_STATIC_ASSERT_V(size(src) == size(scales));

      if constexpr (is_same_v<DstType, ElementScale>) {
        cute::transform(src_arr, dst_arr, Converter::convert);

        using ScaleArray = cutlass::Array<ElementScale, pack>;
        auto scale_arr = recast<ScaleArray>(filter_zeros(scales));

        if constexpr (is_same_v<DstType, cutlass::bfloat16_t>) {
          Tensor scales_vm = cute::group_modes<1, -1>(cute::zipped_divide(scales, pack));

          for (int i = 0; i < size<1>(dst_vm); ++i) {
            auto&& r = cute::recast<RegArray>(dst_vm(_, i))(0);
            auto&& scale_reg = cute::recast<RegArray>(scales_vm(_, i))(0);
            CUTLASS_PRAGMA_UNROLL
            for (size_t ii = 0; ii < RegArray::kElements; ++ii) {
              __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
              bf16x2_val =
                  __hmul2(bf16x2_val, reinterpret_cast<__nv_bfloat162 const&>(scale_reg[ii]));
            }
          }
        } else {
          cute::transform(dst_arr, scale_arr, dst_arr, cute::multiplies{});
        }
      } else {
        constexpr int pack1 =
            decltype(select_packing<SrcType, ElementScale, num_elements>::value())::value;
        constexpr int pack2 =
            decltype(select_packing<ElementScale, DstType, num_elements>::value())::value;
        constexpr int pack = cute::gcd(pack1, pack2);
        using Converter1 =
            cutlass::NumericArrayConverter<ElementScale, SrcType, pack,
                                           cutlass::FloatRoundStyle::round_to_nearest>;
        using Converter2 =
            cutlass::NumericArrayConverter<DstType, ElementScale, pack,
                                           cutlass::FloatRoundStyle::round_to_nearest>;
        using SrcArray = cutlass::Array<SrcType, pack>;
        using DstArray = cutlass::Array<DstType, pack>;
        using StageArray = cutlass::Array<ElementScale, pack>;
        constexpr int iters = num_elements / pack;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < iters; ++i) {
          SrcArray const* pSrcArr = reinterpret_cast<SrcArray const*>(pSrc) + i;
          DstArray* pDstArr = reinterpret_cast<DstArray*>(pDst) + i;
          StageArray stageArr;
          stageArr = Converter1::convert(*pSrcArr);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < pack; ++j) {
            stageArr[j] = stageArr[j] * scales[i * pack + j];
          }
          *pDstArr = Converter2::convert(stageArr);
        }
      }
    } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      static_assert(is_same_v<ElementScale, ElementZero>,
                    "ElementScale and ElementZero must be the same.");

      auto const& scales = cute::get<1>(partitioned_extra_info)(_, _, _, k_block);
      auto const& zeros = cute::get<3>(partitioned_extra_info)(_, _, _, k_block);
      CUTE_STATIC_ASSERT_V(size(src) == size(scales));
      CUTE_STATIC_ASSERT_V(size(src) == size(zeros));

      if constexpr (is_same_v<DstType, ElementZero>) {
        cute::transform(src_arr, dst_arr, Converter::convert);

        using ScaleArray = cutlass::Array<ElementScale, pack>;
        auto scale_arr = recast<ScaleArray>(filter_zeros(scales));

        using ZeroArray = cutlass::Array<ElementZero, pack>;
        auto zero_arr = recast<ZeroArray>(filter_zeros(zeros));

        if constexpr (is_same_v<DstType, cutlass::bfloat16_t>) {
          Tensor scales_vm = cute::group_modes<1, -1>(cute::zipped_divide(scales, pack));
          Tensor zeros_vm = cute::group_modes<1, -1>(cute::zipped_divide(zeros, pack));

          for (int i = 0; i < size<1>(dst_vm); ++i) {
            auto&& r = cute::recast<RegArray>(dst_vm(_, i))(0);
            auto&& scale_reg = cute::recast<RegArray>(scales_vm(_, i))(0);
            auto&& zero_reg = cute::recast<RegArray>(zeros_vm(_, i))(0);
            CUTLASS_PRAGMA_UNROLL
            for (size_t ii = 0; ii < RegArray::kElements; ++ii) {
              __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
              bf16x2_val =
                  __hmul2(bf16x2_val, reinterpret_cast<__nv_bfloat162 const&>(scale_reg[ii]));
              bf16x2_val =
                  __hadd2(bf16x2_val, reinterpret_cast<__nv_bfloat162 const&>(zero_reg[ii]));
            }
          }
        } else {
          cute::transform(dst_arr, scale_arr, dst_arr, cute::multiplies{});
          cute::transform(dst_arr, zero_arr, dst_arr, cute::plus{});
        }
      } else {
        constexpr int pack1 =
            decltype(select_packing<SrcType, ElementScale, num_elements>::value())::value;
        constexpr int pack2 =
            decltype(select_packing<ElementScale, DstType, num_elements>::value())::value;
        constexpr int pack = cute::gcd(pack1, pack2);
        using Converter1 =
            cutlass::NumericArrayConverter<ElementScale, SrcType, pack,
                                           cutlass::FloatRoundStyle::round_to_nearest>;
        using Converter2 =
            cutlass::NumericArrayConverter<DstType, ElementScale, pack,
                                           cutlass::FloatRoundStyle::round_to_nearest>;
        using SrcArray = cutlass::Array<SrcType, pack>;
        using DstArray = cutlass::Array<DstType, pack>;
        using StageArray = cutlass::Array<ElementScale, pack>;
        constexpr int iters = num_elements / pack;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < iters; ++i) {
          SrcArray const* pSrcArr = reinterpret_cast<SrcArray const*>(pSrc) + i;
          DstArray* pDstArr = reinterpret_cast<DstArray*>(pDst) + i;
          StageArray stageArr;
          stageArr = Converter1::convert(*pSrcArr);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < pack; ++j) {
            stageArr[j] = stageArr[j] * scales[i * pack + j] + zeros[i * pack + j];
          }
          *pDstArr = Converter2::convert(stageArr);
        }
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled for input partitioning.");
    }
  }
};
}  // namespace cutlass::gemm::collective::detail
