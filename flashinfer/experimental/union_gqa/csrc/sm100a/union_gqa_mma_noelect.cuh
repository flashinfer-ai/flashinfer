/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
 * @file union_gqa_mma_noelect.cuh
 * @brief NOELECT variants of CuTe's SM100 UMMA atoms for union-GQA.
 *
 * Derived from CuTe's SM100_MMA_F16BF16_{SS,TS} atoms. The local variants
 * remove the inner elect_one_sync and add matching MMA_Traits specializations.
 *
 * Rationale: CuTe's stock SM100_MMA_F16BF16_{SS,TS} atoms call
 * `cute::elect_one_sync()` inside every fma. When a single MMA warp issues
 * many `cute::gemm` calls in sequence (as the union-GQA mainloop does), that's
 * an elect.sync per call. Hoisting one outer elect_one_sync to the MMA warp
 * entry and using NOELECT atoms cuts the per-call sync entirely; the inner
 * gemms emit a bare tcgen05.mma each.
 *
 * The variants preserve the stock instruction shape without the inner elect.
 */
#pragma once

#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>

namespace cute {

// --------------------------------------------------------------------------
// SS NOELECT: A and B use SMEM descriptors.
// --------------------------------------------------------------------------
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_UnionGQA_SS_NOELECT {
  static_assert(M == 64 || M == 128,
                "SS_NOELECT M-mode size should be 64 or 128 for 1-CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256),
                "SS_NOELECT N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a, uint64_t const& desc_b,
                                   uint32_t const& tmem_c, uint32_t const& scaleC,
                                   uint64_t const& idescE) {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
#else
    CUTE_INVALID_CONTROL_PATH(
        "SM100_MMA_F16BF16_UnionGQA_SS_NOELECT requires CUTE_ARCH_MMA_SM100A_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_UnionGQA_SS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                                        b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "SS_NOELECT supports 16-bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC,
            class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const& traits,
                                                    Tensor<TD, DLayout>& D,
                                                    Tensor<TA, ALayout> const& A,
                                                    Tensor<TB, BLayout> const& B,
                                                    Tensor<TC, CLayout> const& C) {
    static_assert(is_tmem<TD>::value, "Expected tmem D");
    static_assert(is_rmem<TA>::value, "Expected desc A");
    static_assert(is_rmem<TB>::value, "Expected desc B");
    static_assert(is_tmem<TC>::value, "Expected tmem C");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_UnionGQA_SS_NOELECT<a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
                                          b_neg>::fma(desc_a, desc_b, tmem_c,
                                                      uint32_t(traits.accumulate_), idesc);
  }
};

// --------------------------------------------------------------------------
// TS NOELECT: A uses TMEM and B uses an SMEM descriptor.
// --------------------------------------------------------------------------
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One, UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_UnionGQA_TS_NOELECT {
  static_assert(M == 64 || M == 128,
                "TS_NOELECT M-mode size should be 64 or 128 for 1-CTA cluster MMA.");
  static_assert((M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) ||
                    (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
                "TS_NOELECT N-mode size invalid for chosen M.");
  static_assert(a_major == UMMA::Major::K, "TS_NOELECT A (TMEM) must be K-major");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t const& tmem_a, uint64_t const& desc_b,
                                   uint32_t const& tmem_c, uint32_t const& scaleC,
                                   uint64_t const& idescE) {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
#else
    CUTE_INVALID_CONTROL_PATH(
        "SM100_MMA_F16BF16_UnionGQA_TS_NOELECT requires CUTE_ARCH_MMA_SM100A_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_UnionGQA_TS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                                        b_major, a_neg, b_neg, c_sat>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "TS_NOELECT supports 16-bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC,
            class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const& traits,
                                                    Tensor<TD, DLayout>& D,
                                                    Tensor<TA, ALayout> const& A,
                                                    Tensor<TB, BLayout> const& B,
                                                    Tensor<TC, CLayout> const& C) {
    static_assert(is_tmem<TD>::value, "Expected tmem D");
    static_assert(is_tmem<TA>::value, "Expected tmem A");
    static_assert(is_rmem<TB>::value, "Expected desc B");
    static_assert(is_tmem<TC>::value, "Expected tmem C");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_UnionGQA_TS_NOELECT<a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
                                          b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c,
                                                             uint32_t(traits.accumulate_), idesc);
  }
};

}  // namespace cute
