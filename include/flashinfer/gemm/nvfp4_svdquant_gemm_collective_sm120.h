/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// SVDQuant modifications are covered by the following Apache-2.0 notice.
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

// SM120 fused NVFP4 SVDQuant mainloop: the CUTLASS SM120 block-scaled TMA warp-specialized
// collective (sm120_blockscaled_mma_tma.hpp) extended with a fused rank-32 bf16 LoRA-up
// (out += D @ L1^T) that accumulates into the SAME fp32 register accumulator before the
// epilogue. Structure mirrors the SM100 CollectiveMmaLoRA port
// (nvfp4_svdquant_gemm_collective_sm100.h) adapted to the SM120 execution model:
//   - register-fragment accumulator (no TMEM), SMEM->register operand path, 1SM, 1x1x1 cluster;
//   - D/L1 arrive by TMA on one of two Sm120LoRaPath policies (template parameter):
//     kByteExactMainloop: ONE extra producer step after the residual K-loop, overlaying the
//     freed residual stage buffers smem_A/smem_B (no dedicated carveout); the extra step's
//     arrived bytes exactly match the armed residual A+B+SF budget: the D/L1 TMA boxes are
//     byte-exact overlays of one A/B stage (K256 boxes cover 64 bf16 columns with the rank
//     tail TMA zero-filled) and SFA/SFB are dummy-reloaded, matching the byte-exact protocol
//     proven by the SM100 port;
//     kDedicatedTma: a one-slot PipelineTmaAsync with its own smem carveout, barriers, and
//     transaction budget; true rank-32 boxes, no dummy SF reloads, and the residual
//     pipeline's byte budget, buffers, and state are never touched;
//   - the LoRA MMA is a bf16 mma.sync tiled MMA constructed with the residual TiledMma's
//     atom layout and M/N permutations so its accumulator partitioning is IDENTICAL
//     (enforced by static_assert), applied by the kernel AFTER Stream-K fixup and before the
//     epilogue, exactly once per output tile by the epilogue-owning work unit.
//
// This file also provides the kernel driver GemmUniversalLoRaSm120, a copy of the CUTLASS
// SM90 cooperative warp-specialized kernel (sm90_gemm_tma_warpspecialized_cooperative.hpp,
// which drives the stock SM120 block-scaled schedule) with the producer/consumer LoRA hooks.

#pragma once

#include <cuda_bf16.h>

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/workspace.h"

// Hook-enabled epilogue copy + the subtile-hook helpers the kernel below threads
// through the store loop.
#include "flashinfer/gemm/nvfp4_svdquant_gemm_epilogue_sm120.h"

namespace cutlass::gemm::collective {
using namespace cute;

namespace svdquant_sm120_detail {

// Extract the AtomLayoutMNK / PermutationMNK a TiledMMA was built with so the LoRA bf16
// tiled MMA can reuse them (identical accumulator partitioning is required to fold the
// LoRA product into the residual accumulator fragment).
template <class TiledMmaT>
struct TiledMmaTraits;

template <class Atom, class AtomLayoutMNK_, class PermutationMNK_>
struct TiledMmaTraits<cute::TiledMMA<Atom, AtomLayoutMNK_, PermutationMNK_>> {
  using AtomLayoutMNK = AtomLayoutMNK_;
  using PermutationMNK = PermutationMNK_;
};

// Keep the LoRA swizzle type-dependent so K128 configurations do
// not instantiate a 128-byte swizzle over their 64-byte bf16 rows.
template <bool Enable, class Layout>
struct MaybeLoRaSmemSwizzle {
  using type = Layout;
};

template <class Layout>
struct MaybeLoRaSmemSwizzle<true, Layout> {
  using type = decltype(cute::composition(cute::Swizzle<3, 3, 3>{}, Layout{}));
};

}  // namespace svdquant_sm120_detail

// How the rank-32 D/L1 operands reach shared memory:
//   - kByteExactMainloop: one extra step through the residual pipeline whose arrived bytes
//     exactly match the armed residual budget (stage-buffer overlays, dummy SFA/SFB reloads);
//   - kDedicatedTma: a one-slot pipeline with its own carveout, barriers, and transaction
//     budget; the residual pipeline's byte accounting and state are never touched.
enum class Sm120LoRaPath { kByteExactMainloop, kDedicatedTma };

template <class DispatchPolicy_, class TileShape_, class ElementPairA_, class StridePairA_,
          class ElementPairB_, class StridePairB_, class TiledMma_, class GmemTiledCopyPairA_,
          class SmemLayoutAtomsA_, class SmemCopyAtomsA_, class TransformA_,
          class GmemTiledCopyPairB_, class SmemLayoutAtomsB_, class SmemCopyAtomsB_,
          class TransformB_, Sm120LoRaPath LoRaPath_ = Sm120LoRaPath::kByteExactMainloop,
          bool Row80ProductionTraits_ = false, bool CompileInlineDown_ = true, int LoRaK_ = 32>
struct CollectiveMmaLoRaSm120 {
  //
  // Type Aliases (identical to the stock SM120 block-scaled collective)
  //
  using DispatchPolicy = DispatchPolicy_;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using TileShape = TileShape_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                                remove_cvref_t<decltype(get<1>(ElementPairB{}))>>,
                "SFA and SFB data types should be the same");

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA = remove_cvref_t<decltype(get<0>(StridePairA{}))>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB = remove_cvref_t<decltype(get<0>(StridePairB{}))>;

  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  using ArrayElementA = ElementA;
  using ArrayElementB = ElementB;

  using TiledMma = TiledMma_;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;

  static constexpr int SFVecSize = TiledMma::Traits::SFVecSize;
  // NVFP4 block scales are quantized with a 16-element vector; an MXF4-flavored (VS=32)
  // builder result would silently consume the scales with the wrong grouping.
  static_assert(SFVecSize == 16, "SM120 SVDQuant requires the NVFP4 (SFVecSize=16) MMA");
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;
  using GmemTiledCopyA = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopySFA = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopyB = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB{}))>;
  using GmemTiledCopySFB = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB{}))>;

  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;

  using SmemLayoutAtomA = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomB = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsB{}))>;
  using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsB{}))>;

  using SmemCopyAtomsA = SmemCopyAtomsA_;
  using SmemCopyAtomsB = SmemCopyAtomsB_;

  using SmemCopyAtomA = remove_cvref_t<decltype(get<0>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomSFA = remove_cvref_t<decltype(get<1>(SmemCopyAtomsA{}))>;

  using SmemCopyAtomB = remove_cvref_t<decltype(get<0>(SmemCopyAtomsB{}))>;
  using SmemCopyAtomSFB = remove_cvref_t<decltype(get<1>(SmemCopyAtomsB{}))>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr Sm120LoRaPath LoRaPath = LoRaPath_;
  static constexpr bool UseDedicatedLoRaTma = (LoRaPath == Sm120LoRaPath::kDedicatedTma);
  static constexpr bool UseLoRaPipeline = UseDedicatedLoRaTma;
  static constexpr bool CompileInlineDown = CompileInlineDown_;

  static constexpr int ThreadCount = size(TiledMma{});

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  static constexpr int NumProducerThreadEvents = 1;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(not cute::is_void_v<SmemCopyAtomA>,
                "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>,
                "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t<::cutlass::gemm::detail::is_major<0, StrideA>(), Step<_2, _1, _3>,
                    Step<_1, _2, _3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t<::cutlass::gemm::detail::is_major<0, StrideB>(), Step<_2, _1, _3>,
                    Step<_1, _2, _3>>{}));

  using SmemLayoutSFA_ = decltype(make_layout(
      append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
      append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))));

  // The SFA smem atom always spans 128 physical rows (Blk_MN). A smaller M tile keeps the
  // padded rows: the producer loads the full 128-row TMA box and the consumer selects its
  // TileShape.M-row slice, mirroring the small-N SFB mechanism below.
  using TileShapeSFA =
      cute::conditional_t<size<0>(TileShape{}) < 128,
                          decltype(cute::make_shape(Int<128>{}, shape<1>(TileShape{}),
                                                    shape<2>(TileShape{}))),
                          TileShape>;

  using SmemLayoutSFA = cute::conditional_t<
      size<0>(TileShape{}) < 128,
      decltype(cute::logical_divide(SmemLayoutSFA_{}, select<0, 2>(TileShape{}))), SmemLayoutSFA_>;

  using SmemLayoutSFB_ = decltype(make_layout(
      append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
      append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))));

  using TileShapeSFB =
      cute::conditional_t<size<1>(TileShape{}) < 128,
                          decltype(cute::make_shape(shape<0>(TileShape{}), Int<128>{},
                                                    shape<2>(TileShape{}))),
                          TileShape>;

  using SmemLayoutSFB = cute::conditional_t<
      size<1>(TileShape{}) < 128,
      decltype(cute::logical_divide(SmemLayoutSFB_{}, select<1, 2>(TileShape{}))), SmemLayoutSFB_>;

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

  static_assert(DispatchPolicy::Stages >= 2,
                "Specialization requires Stages set to value 2 or more.");
  static_assert(
      not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
          not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
      "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();

  using TmaInternalElementA = cute::conditional_t<
      not IsF8F6F4, ElementA,
      cute::conditional_t<
          cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
          cutlass::detail::float_e2m1_unpacksmem_t,
          cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m3_t>,
                              cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e3m2_t>,
                                                  cutlass::detail::float_e3m2_unpacksmem_t,
                                                  uint_bit_t<sizeof_bits_v<ElementA>>>>>>;

  using TmaInternalElementB = cute::conditional_t<
      not IsF8F6F4, ElementB,
      cute::conditional_t<
          cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
          cutlass::detail::float_e2m1_unpacksmem_t,
          cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m3_t>,
                              cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e3m2_t>,
                                                  cutlass::detail::float_e3m2_unpacksmem_t,
                                                  uint_bit_t<sizeof_bits_v<ElementB>>>>>>;

  using TmaInternalElementSF = ElementSF;

  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeB>;

  // Residual per-stage TMA byte budget (identical formulas to the stock collective).
  static constexpr uint32_t ResidualTmaBytesMK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));

  static constexpr uint32_t ResidualTmaBytesNK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>) +
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));

  static constexpr uint32_t ResidualTmaBytes = ResidualTmaBytesMK + ResidualTmaBytesNK;

  // === LoRA-up: bf16 D[M,r] (A-side) and L1[N,r] (B-side), r = LoRaK. ===
  // The LoRA math below is written against LoRaK throughout -- the mma atom count
  // (LoRaK / LoRaAtomK), the TMA box extents, and the smem overlay views all follow it.
  // What bounds it is storage, asserted where the operands are staged: a rank tile has
  // to fit the residual stage it overlays (see the AStageBf16Elems asserts below), so
  // K128 tiles carry rank 32 and K256 tiles have room for 64.
  static constexpr int LoRaK = LoRaK_;
  static_assert(LoRaK > 0 && LoRaK % 32 == 0, "LoRA rank must be a positive multiple of 32");

  using ElementD = cutlass::bfloat16_t;
  using ElementL1 = cutlass::bfloat16_t;
  // Row-major [M, rank] / [N, rank]: K-contiguous, rank-3 (mode, rank, batch).
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideL1 = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // bf16 columns spanned by one residual A/B stage: K128 fp4 -> 32 (the rank-32 tile is a
  // byte-exact stage overlay), K256 fp4 -> 64 (the TMA box covers 64 bf16 columns; columns
  // 32..63 lie outside the [M/N, 32] gmem tensor and are TMA zero-filled, so the padded
  // K-blocks contribute zero to the LoRA MMA).
  static constexpr int LoRaStorageK =
      static_cast<int>(size<2>(TileShape{}) * sizeof_bits<ElementA>::value /
                       sizeof_bits<cutlass::bfloat16_t>::value);
  static_assert(LoRaStorageK >= LoRaK, "one stage must hold at least the rank tile");

  static constexpr int AStageBf16Elems = static_cast<int>(
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value) /
      sizeof(cutlass::bfloat16_t));
  static constexpr int BStageBf16Elems = static_cast<int>(
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value) /
      sizeof(cutlass::bfloat16_t));

  // The overlay is compact and byte-exact per stage: D-tile == A-stage, L1-tile == B-stage.
  static_assert(size<0>(TileShape{}) * LoRaStorageK == AStageBf16Elems,
                "D storage tile must overlay one residual A stage byte-exactly");
  static_assert(size<1>(TileShape{}) * LoRaStorageK == BStageBf16Elems,
                "L1 storage tile must overlay one residual B stage byte-exactly");
  static_assert((AStageBf16Elems * static_cast<int>(sizeof(cutlass::bfloat16_t))) % 1024 == 0,
                "residual A stage stride must preserve the 1024-byte smem_A alignment");
  static_assert((BStageBf16Elems * static_cast<int>(sizeof(cutlass::bfloat16_t))) % 1024 == 0,
                "residual B stage stride must preserve the 1024-byte smem_B alignment");

  // K-major (M, LoRaStorageK, PIPE) overlay of smem_A / smem_B. The row80
  // production path and the optional build-wide diagnostic use CUTLASS's bf16
  // 128-byte swizzle donor. The consumer names these aliases directly; the TMA
  // producer derives their exact byte-domain physical equivalent below.
  using SmemLayoutDLinear = decltype(make_layout(
      make_shape(shape<0>(TileShape{}), Int<LoRaStorageK>{}, Int<DispatchPolicy::Stages>{}),
      make_stride(Int<LoRaStorageK>{}, Int<1>{}, Int<AStageBf16Elems>{})));
  using SmemLayoutL1Linear = decltype(make_layout(
      make_shape(shape<1>(TileShape{}), Int<LoRaStorageK>{}, Int<DispatchPolicy::Stages>{}),
      make_stride(Int<LoRaStorageK>{}, Int<1>{}, Int<BStageBf16Elems>{})));

  static constexpr bool EnableLoRaSmemSwizzle = Row80ProductionTraits_;
  static constexpr bool UseRow80SideSlot = UseDedicatedLoRaTma && Row80ProductionTraits_;
  static constexpr bool UseLoRaSmemSwizzle =
      EnableLoRaSmemSwizzle &&
      (LoRaPath_ == Sm120LoRaPath::kByteExactMainloop || UseRow80SideSlot) && LoRaStorageK == 64;
  // Row80ProductionTraits_ is the production-row80 selector passed by the driver.
  // Keep the bias prefetch tied to that exact tactic instead of the optional
  // build-wide swizzle diagnostic above.
  static constexpr bool UseRow80BiasPrefetch = Row80ProductionTraits_;
  using SmemLayoutD = typename svdquant_sm120_detail::MaybeLoRaSmemSwizzle<UseLoRaSmemSwizzle,
                                                                           SmemLayoutDLinear>::type;
  using SmemLayoutL1 =
      typename svdquant_sm120_detail::MaybeLoRaSmemSwizzle<UseLoRaSmemSwizzle,
                                                           SmemLayoutL1Linear>::type;

  // Rank-32 payload view of the same overlay: K extent LoRaK, row stride kept at
  // LoRaStorageK so rows still hop over the TMA zero-filled tail columns. Where
  // LoRaStorageK == LoRaK (K128 stages) these collapse to SmemLayoutD/L1 exactly.
  using SmemLayoutDPayloadLinear = decltype(make_layout(
      make_shape(shape<0>(TileShape{}), Int<LoRaK>{}, Int<DispatchPolicy::Stages>{}),
      make_stride(Int<LoRaStorageK>{}, Int<1>{}, Int<AStageBf16Elems>{})));
  using SmemLayoutL1PayloadLinear = decltype(make_layout(
      make_shape(shape<1>(TileShape{}), Int<LoRaK>{}, Int<DispatchPolicy::Stages>{}),
      make_stride(Int<LoRaStorageK>{}, Int<1>{}, Int<BStageBf16Elems>{})));
  using SmemLayoutDPayload =
      typename svdquant_sm120_detail::MaybeLoRaSmemSwizzle<UseLoRaSmemSwizzle,
                                                           SmemLayoutDPayloadLinear>::type;
  using SmemLayoutL1Payload =
      typename svdquant_sm120_detail::MaybeLoRaSmemSwizzle<UseLoRaSmemSwizzle,
                                                           SmemLayoutL1PayloadLinear>::type;

  static_assert(cosize_v<SmemLayoutD> == cosize_v<SmemLayoutDLinear>,
                "D swizzle must preserve the byte-exact stage overlay size");
  static_assert(cosize_v<SmemLayoutL1> == cosize_v<SmemLayoutL1Linear>,
                "L1 swizzle must preserve the byte-exact stage overlay size");
  static_assert(cosize_v<SmemLayoutDPayload> <= cosize_v<SmemLayoutD>,
                "D payload view must stay within its storage overlay");
  static_assert(cosize_v<SmemLayoutL1Payload> <= cosize_v<SmemLayoutL1>,
                "L1 payload view must stay within its storage overlay");

  // TMA encodes swizzles in byte-domain address bits. Recast the bf16
  // consumer layout so Swizzle<3,3,3> becomes its byte-equivalent
  // Swizzle<3,4,3>, then recast the gmem tensor, CTA box, and partition
  // coordinates together so every producer-side extent uses byte units.
  using LoRaTmaInternalElement =
      cute::conditional_t<UseLoRaSmemSwizzle, uint8_t, cutlass::bfloat16_t>;
  using SmemLayoutDTmaRecast =
      decltype(cute::recast_layout<cutlass::bfloat16_t, uint8_t>(SmemLayoutD{}));
  using SmemLayoutL1TmaRecast =
      decltype(cute::recast_layout<cutlass::bfloat16_t, uint8_t>(SmemLayoutL1{}));
  using SmemLayoutDTma = cute::conditional_t<UseLoRaSmemSwizzle, SmemLayoutDTmaRecast, SmemLayoutD>;
  using SmemLayoutL1Tma =
      cute::conditional_t<UseLoRaSmemSwizzle, SmemLayoutL1TmaRecast, SmemLayoutL1>;
  static constexpr int LoRaTmaRankK = LoRaK * sizeof(ElementD) / sizeof(LoRaTmaInternalElement);
  static constexpr int LoRaTmaStorageK =
      LoRaStorageK * sizeof(ElementD) / sizeof(LoRaTmaInternalElement);
  static_assert(cosize_v<SmemLayoutDTma> * sizeof(LoRaTmaInternalElement) ==
                    cosize_v<SmemLayoutD> * sizeof(ElementD),
                "D TMA recast must preserve the overlay byte size");
  static_assert(cosize_v<SmemLayoutL1Tma> * sizeof(LoRaTmaInternalElement) ==
                    cosize_v<SmemLayoutL1> * sizeof(ElementL1),
                "L1 TMA recast must preserve the overlay byte size");

  // Consumer-side overlay view: only the rank-32 payload columns are staged into
  // registers and folded by the tail MMAs; the TMA zero-filled tail columns of
  // K256 stages contribute exact zeros to the accumulator and are skipped. The
  // producer's TMA boxes, dummy SF reloads, and pipeline byte accounting are
  // untouched, so every barrier arms and arrives exactly as the default build.
  using SmemLayoutDConsume = SmemLayoutDPayload;
  using SmemLayoutL1Consume = SmemLayoutL1Payload;

  // The storage-width fallback's tail MMAs over the TMA zero-filled columns
  // compute acc = acc + 0 on every lane, which canonicalizes an exactly -0.0
  // accumulator lane to +0.0. The payload view skips those MMAs, so the same
  // additive identity is applied explicitly to keep the output bit pattern
  // identical to the fallback's on every valid input. K128 stages have no
  // skipped columns and need nothing.
  template <class FrgTensorAcc>
  CUTLASS_DEVICE static void canonicalize_lora_tail_zeros(FrgTensorAcc&& accum) {
    if constexpr (LoRaStorageK != LoRaK) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(accum); ++i) {
        accum(i) = accum(i) + 0.0f;
      }
    }
  }

  // Pipeline byte protocol (byte-exact, matching the proven SM100 scheme): every
  // producer_acquire arms the full residual budget. The LoRA step arrives exactly that
  // budget too: D-box(== A-stage bytes) + L1-box(== B-stage bytes) + dummy SFA/SFB reloads
  // (== SF bytes). No expect-transaction adjustment is needed on any step.
  static constexpr uint32_t LoRaTmaBytes = static_cast<uint32_t>(
      size<0>(TileShape{}) * LoRaStorageK * sizeof(cutlass::bfloat16_t) +
      size<1>(TileShape{}) * LoRaStorageK * sizeof(cutlass::bfloat16_t) +
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>));
  static_assert(LoRaTmaBytes == ResidualTmaBytes,
                "LoRA step bytes must equal the armed residual byte budget");

  static constexpr uint32_t TmaTransactionBytes = ResidualTmaBytes;
  static constexpr uint32_t TmaTransactionBytesMK = ResidualTmaBytesMK;
  static constexpr uint32_t TmaTransactionBytesNK = ResidualTmaBytesNK;

  // === Dedicated LoRA path (kDedicatedTma): true rank-32 staging in a one-slot pipeline.
  // D/L1 arrive against the LoRA barrier only, so the boxes carry no rank padding, no dummy
  // SF reloads exist, and the residual pipeline's byte budget and state stay untouched. ===
  using LoRaPipeline = cutlass::PipelineTmaAsync<1>;
  using LoRaPipelineState = cutlass::PipelineState<1>;

  // Historical dedicated builds use compact K32 tiles. Row80's independent side slot
  // instead slices stage zero from the proven K64 production layouts: the TMA sees the
  // padded 128B-swizzled storage while the consumer sees only its rank-32 payload.
  using SmemLayoutDDedicatedCompact = decltype(make_layout(
      make_shape(shape<0>(TileShape{}), Int<LoRaK>{}), make_stride(Int<LoRaK>{}, Int<1>{})));
  using SmemLayoutL1DedicatedCompact = decltype(make_layout(
      make_shape(shape<1>(TileShape{}), Int<LoRaK>{}), make_stride(Int<LoRaK>{}, Int<1>{})));
  using SmemLayoutDSideSlot = decltype(SmemLayoutD{}(_, _, Int<0>{}));
  using SmemLayoutL1SideSlot = decltype(SmemLayoutL1{}(_, _, Int<0>{}));
  using SmemLayoutDSideSlotConsume = decltype(SmemLayoutDPayload{}(_, _, Int<0>{}));
  using SmemLayoutL1SideSlotConsume = decltype(SmemLayoutL1Payload{}(_, _, Int<0>{}));
  using SmemLayoutDDedicated =
      cute::conditional_t<UseRow80SideSlot, SmemLayoutDSideSlot, SmemLayoutDDedicatedCompact>;
  using SmemLayoutL1Dedicated =
      cute::conditional_t<UseRow80SideSlot, SmemLayoutL1SideSlot, SmemLayoutL1DedicatedCompact>;
  using SmemLayoutDDedicatedConsume =
      cute::conditional_t<UseRow80SideSlot, SmemLayoutDSideSlotConsume,
                          SmemLayoutDDedicatedCompact>;
  using SmemLayoutL1DedicatedConsume =
      cute::conditional_t<UseRow80SideSlot, SmemLayoutL1SideSlotConsume,
                          SmemLayoutL1DedicatedCompact>;

  using LoRaDedicatedTmaInternalElement =
      cute::conditional_t<UseRow80SideSlot, uint8_t, cutlass::bfloat16_t>;
  using SmemLayoutDDedicatedTmaRecast =
      decltype(cute::recast_layout<cutlass::bfloat16_t, uint8_t>(SmemLayoutDDedicated{}));
  using SmemLayoutL1DedicatedTmaRecast =
      decltype(cute::recast_layout<cutlass::bfloat16_t, uint8_t>(SmemLayoutL1Dedicated{}));
  using SmemLayoutDDedicatedTma =
      cute::conditional_t<UseRow80SideSlot, SmemLayoutDDedicatedTmaRecast, SmemLayoutDDedicated>;
  using SmemLayoutL1DedicatedTma =
      cute::conditional_t<UseRow80SideSlot, SmemLayoutL1DedicatedTmaRecast, SmemLayoutL1Dedicated>;
  static constexpr int LoRaDedicatedTmaRankK =
      LoRaK * sizeof(ElementD) / sizeof(LoRaDedicatedTmaInternalElement);
  static constexpr int LoRaDedicatedTmaStorageK = (UseRow80SideSlot ? LoRaStorageK : LoRaK) *
                                                  sizeof(ElementD) /
                                                  sizeof(LoRaDedicatedTmaInternalElement);
  static_assert(cosize_v<SmemLayoutDDedicatedConsume> <= cosize_v<SmemLayoutDDedicated>,
                "the dedicated D payload must fit in its storage slot");
  static_assert(cosize_v<SmemLayoutL1DedicatedConsume> <= cosize_v<SmemLayoutL1Dedicated>,
                "the dedicated L1 payload must fit in its storage slot");

  // Both operands arrive against the single LoRA full barrier; producer_acquire arms
  // exactly this budget from the LoRA pipeline's own params.
  static constexpr uint32_t LoRaDedicatedTmaBytes = static_cast<uint32_t>(
      (size<0>(TileShape{}) + size<1>(TileShape{})) * (UseRow80SideSlot ? LoRaStorageK : LoRaK) *
      sizeof(cutlass::bfloat16_t));

  // === LoRA bf16 tiled MMA: same atom layout and M/N permutations as the residual MMA so
  // the fp32 accumulator partitioning is identical; only the K mode differs (K=16 atom). ===
  using ResidualMmaTraits = svdquant_sm120_detail::TiledMmaTraits<TiledMma>;
  using LoRaAtom = cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>;
  using LoRaPermutationMNK =
      decltype(make_tile(get<0>(typename ResidualMmaTraits::PermutationMNK{}),
                         get<1>(typename ResidualMmaTraits::PermutationMNK{}), Int<16>{}));
  using LoRaMma = decltype(cute::make_tiled_mma(
      LoRaAtom{}, typename ResidualMmaTraits::AtomLayoutMNK{}, LoRaPermutationMNK{}));

  static_assert(size(LoRaMma{}) == size(TiledMma{}),
                "LoRA MMA must use the same thread count as the residual MMA");
  // Accumulator congruence: both tiled MMAs must partition the CTA (M,N) accumulator into
  // byte-identical per-thread fragments.
  static_assert(
      cute::is_same_v<decltype(partition_shape_C(TiledMma{}, make_shape(size<0>(TileShape{}),
                                                                        size<1>(TileShape{})))),
                      decltype(partition_shape_C(LoRaMma{}, make_shape(size<0>(TileShape{}),
                                                                       size<1>(TileShape{}))))>,
      "LoRA MMA accumulator partitioning must match the residual MMA accumulator");

  static constexpr int LoRaAtomK = size<2>(typename LoRaAtom::Shape_MNK{});
  static_assert(LoRaK % LoRaAtomK == 0);

  struct SharedStorageByteExact {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      alignas(16) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      alignas(16) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };

  // Dedicated-path storage: identical residual arrays plus the true-rank D/L1 staging
  // buffers and the one-slot LoRA pipeline barriers. The launcher's StageCountAutoCarveout
  // arithmetic must reserve exactly this growth before the builder selects stage counts.
  struct SharedStorageDedicatedTma {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      alignas(16) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      alignas(16) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
      alignas(128) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutDDedicated>> smem_D;
      alignas(128) cute::ArrayEngine<ElementL1, cute::cosize_v<SmemLayoutL1Dedicated>> smem_L1;
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16, _0> {
      alignas(16) typename MainloopPipeline::SharedStorage mainloop;
      alignas(16) typename LoRaPipeline::SharedStorage lora;
    };
    alignas(16) PipelineStorage pipeline_storage;
  };

  using SharedStorage =
      cute::conditional_t<UseDedicatedLoRaTma, SharedStorageDedicatedTma, SharedStorageByteExact>;

  // The rank-32 overlays must live inside the residual arrays.
  static_assert(cosize_v<SmemLayoutD> <= cosize_v<SmemLayoutA>,
                "D overlay must end at or before the end of smem_A");
  static_assert(cosize_v<SmemLayoutL1> <= cosize_v<SmemLayoutB>,
                "L1 overlay must end at or before the end of smem_B");
  // Each true-rank tile must fit inside one residual stage.
  static_assert(size<0>(TileShape{}) * LoRaK <= AStageBf16Elems,
                "the true-rank D tile must fit at a residual A stage base");
  static_assert(size<1>(TileShape{}) * LoRaK <= BStageBf16Elems,
                "the true-rank L1 tile must fit at a residual B stage base");

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // The kernel driver owns pipeline construction; these map the policy-dependent layout of
  // the pipeline storage block.
  CUTLASS_DEVICE static typename MainloopPipeline::SharedStorage& mainloop_pipeline_storage(
      PipelineStorage& storage) {
    if constexpr (UseLoRaPipeline) {
      return storage.mainloop;
    } else {
      return storage;
    }
  }

  CUTLASS_DEVICE static typename LoRaPipeline::SharedStorage& lora_pipeline_storage(
      PipelineStorage& storage) {
    static_assert(UseLoRaPipeline, "only the dedicated policy owns a LoRA pipeline");
    return storage.lora;
  }

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementSF const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementSF const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
    // LoRA-up: D [M, 32] (1/alpha NOT folded), L1 [N, 32] (1/alpha folded in by caller).
    ElementD const* ptr_D{nullptr};
    StrideD dD{};
    ElementL1 const* ptr_L1{nullptr};
    StrideL1 dL1{};
    int lora_rank{LoRaK};
    // The elected producer lane prefetches the row80 bias line while the
    // residual/LoRA pipeline is still in flight.
    cutlass::bfloat16_t const* ptr_bias{nullptr};
    // Exact M537/K5120 experiment: the complete CTA cooperatively computes one
    // unique 16x32 tile of D before entering the persistent residual mainloop.
    // The counters gate the later D TMA without launching a separate K12 down kernel.
    cutlass::bfloat16_t const* ptr_inline_down_x{nullptr};
    cutlass::bfloat16_t const* ptr_inline_down_l2t{nullptr};
    cutlass::bfloat16_t* ptr_inline_down_output{nullptr};
    uint32_t* ptr_inline_down_counters{nullptr};
    int inline_down_m{0};
    int inline_down_k{0};
    bool inline_down_wait_on_n{false};
  };

  // Device side kernel params
  struct Params {
    using TMA_A =
        decltype(make_tma_copy(GmemTiledCopyA{},
                               make_tensor(recast_ptr<TmaInternalElementA>(nullptr),
                                           repeat_like(StrideA{}, int32_t(0)), StrideA{}),
                               SmemLayoutA{}(_, _, cute::Int<0>{}),
                               make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{}));
    using TMA_B =
        decltype(make_tma_copy(GmemTiledCopyB{},
                               make_tensor(recast_ptr<TmaInternalElementB>(nullptr),
                                           repeat_like(StrideB{}, int32_t(0)), StrideB{}),
                               SmemLayoutB{}(_, _, cute::Int<0>{}),
                               make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), _1{}));

    using TMA_SFA = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFA{}, make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}),
        SmemLayoutSFA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShapeSFA{}), shape<2>(TileShapeSFA{})), _1{}));

    using TMA_SFB = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFB{}, make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFB{}),
        SmemLayoutSFB{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShapeSFB{}), shape<2>(TileShapeSFB{})), _1{}));

    using TMA_D = decltype(make_tma_copy<LoRaTmaInternalElement>(
        SM90_TMA_LOAD{},
        cute::recast<LoRaTmaInternalElement>(make_tensor(
            static_cast<ElementD const*>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{})),
        SmemLayoutDTma{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), Int<LoRaTmaStorageK>{}), _1{}));
    using TMA_L1 = decltype(make_tma_copy<LoRaTmaInternalElement>(
        SM90_TMA_LOAD{},
        cute::recast<LoRaTmaInternalElement>(make_tensor(static_cast<ElementL1 const*>(nullptr),
                                                         repeat_like(StrideL1{}, int32_t(0)),
                                                         StrideL1{})),
        SmemLayoutL1Tma{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), Int<LoRaTmaStorageK>{}), _1{}));

    // Dedicated-path descriptors: true rank-32 boxes into the dedicated staging buffers
    // (the tma_load_d/l1 boxes above are sized by the LoRaStorageK stage overlay instead).
    using TMA_D_Dedicated = decltype(make_tma_copy<LoRaDedicatedTmaInternalElement>(
        SM90_TMA_LOAD{},
        cute::recast<LoRaDedicatedTmaInternalElement>(make_tensor(
            static_cast<ElementD const*>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{})),
        SmemLayoutDDedicatedTma{},
        make_shape(shape<0>(TileShape{}), Int<LoRaDedicatedTmaStorageK>{}), _1{}));
    using TMA_L1_Dedicated = decltype(make_tma_copy<LoRaDedicatedTmaInternalElement>(
        SM90_TMA_LOAD{},
        cute::recast<LoRaDedicatedTmaInternalElement>(
            make_tensor(static_cast<ElementL1 const*>(nullptr), repeat_like(StrideL1{}, int32_t(0)),
                        StrideL1{})),
        SmemLayoutL1DedicatedTma{},
        make_shape(shape<1>(TileShape{}), Int<LoRaDedicatedTmaStorageK>{}), _1{}));

    struct LoRaDedicatedTma {
      TMA_D_Dedicated tma_load_d_dedicated;
      TMA_L1_Dedicated tma_load_l1_dedicated;
    };
    struct LoRaDedicatedTmaNone {};

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
    TMA_D tma_load_d;
    TMA_L1 tma_load_l1;
    // Only the dedicated policy carries side-slot descriptors, preserving the
    // byte-exact Params footprint.
    cute::conditional_t<UseLoRaPipeline, LoRaDedicatedTma, LoRaDedicatedTmaNone> lora_dedicated;
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    cutlass::bfloat16_t const* ptr_bias = nullptr;
    cutlass::bfloat16_t const* ptr_inline_down_x = nullptr;
    cutlass::bfloat16_t const* ptr_inline_down_l2t = nullptr;
    cutlass::bfloat16_t* ptr_inline_down_output = nullptr;
    uint32_t* ptr_inline_down_counters = nullptr;
    int inline_down_m = 0;
    int inline_down_k = 0;
    bool inline_down_wait_on_n = false;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                  Arguments const& args, void* workspace) {
    (void)workspace;

    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M, K, L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N, K, L), args.dB));

    Tensor tensor_sfa = make_tensor(args.ptr_SFA, args.layout_SFA);
    Tensor tensor_sfb = make_tensor(args.ptr_SFB, args.layout_SFB);

    typename Params::TMA_A tma_load_a =
        make_tma_copy(GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_, _, cute::Int<0>{}),
                      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{});
    typename Params::TMA_B tma_load_b =
        make_tma_copy(GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_, _, cute::Int<0>{}),
                      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), _1{});

    typename Params::TMA_SFA tma_load_sfa = make_tma_copy<uint16_t>(
        GmemTiledCopySFA{}, tensor_sfa, SmemLayoutSFA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShapeSFA{}), shape<2>(TileShapeSFA{})), _1{});

    typename Params::TMA_SFB tma_load_sfb = make_tma_copy<uint16_t>(
        GmemTiledCopySFB{}, tensor_sfb, SmemLayoutSFB{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShapeSFB{}), shape<2>(TileShapeSFB{})), _1{});

    // The gmem view is (M/N, LoRaK, L): the rank is this collective's LoRaK (validated in
    // can_implement), so no out-of-bounds chunk handling is needed.
    Tensor tensor_d = make_tensor(args.ptr_D, make_layout(make_shape(M, Int<LoRaK>{}, L), args.dD));
    Tensor tensor_l1 =
        make_tensor(args.ptr_L1, make_layout(make_shape(N, Int<LoRaK>{}, L), args.dL1));
    Tensor tensor_d_tma = cute::recast<LoRaTmaInternalElement>(tensor_d);
    Tensor tensor_l1_tma = cute::recast<LoRaTmaInternalElement>(tensor_l1);

    typename Params::TMA_D tma_load_d = make_tma_copy<LoRaTmaInternalElement>(
        SM90_TMA_LOAD{}, tensor_d_tma, SmemLayoutDTma{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), Int<LoRaTmaStorageK>{}), _1{});
    typename Params::TMA_L1 tma_load_l1 = make_tma_copy<LoRaTmaInternalElement>(
        SM90_TMA_LOAD{}, tensor_l1_tma, SmemLayoutL1Tma{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), Int<LoRaTmaStorageK>{}), _1{});

    Params params{};
    params.tma_load_a = tma_load_a;
    params.tma_load_b = tma_load_b;
    params.tma_load_sfa = tma_load_sfa;
    params.tma_load_sfb = tma_load_sfb;
    params.tma_load_d = tma_load_d;
    params.tma_load_l1 = tma_load_l1;
    params.layout_SFA = args.layout_SFA;
    params.layout_SFB = args.layout_SFB;
    params.tma_transaction_bytes = TmaTransactionBytes;
    params.tma_transaction_bytes_mk = TmaTransactionBytesMK;
    params.tma_transaction_bytes_nk = TmaTransactionBytesNK;
    params.ptr_bias = args.ptr_bias;
    params.ptr_inline_down_x = args.ptr_inline_down_x;
    params.ptr_inline_down_l2t = args.ptr_inline_down_l2t;
    params.ptr_inline_down_output = args.ptr_inline_down_output;
    params.ptr_inline_down_counters = args.ptr_inline_down_counters;
    params.inline_down_m = args.inline_down_m;
    params.inline_down_k = args.inline_down_k;
    params.inline_down_wait_on_n = args.inline_down_wait_on_n;
    if constexpr (UseLoRaPipeline) {
      auto tensor_d_dedicated = cute::recast<LoRaDedicatedTmaInternalElement>(tensor_d);
      auto tensor_l1_dedicated = cute::recast<LoRaDedicatedTmaInternalElement>(tensor_l1);
      params.lora_dedicated.tma_load_d_dedicated = make_tma_copy<LoRaDedicatedTmaInternalElement>(
          SM90_TMA_LOAD{}, tensor_d_dedicated, SmemLayoutDDedicatedTma{},
          make_shape(shape<0>(TileShape{}), Int<LoRaDedicatedTmaStorageK>{}), _1{});
      params.lora_dedicated.tma_load_l1_dedicated = make_tma_copy<LoRaDedicatedTmaInternalElement>(
          SM90_TMA_LOAD{}, tensor_l1_dedicated, SmemLayoutL1DedicatedTma{},
          make_shape(shape<1>(TileShape{}), Int<LoRaDedicatedTmaStorageK>{}), _1{});
    }
    return params;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(ProblemShape const& problem_shape,
                                                Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    constexpr int tma_alignment_bits_A =
        cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B =
        cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A =
        tma_alignment_bits_A / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(
                                         cute::make_shape(M, K, L), StrideA{});
    constexpr int min_tma_aligned_elements_B =
        tma_alignment_bits_B / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
                                         cute::make_shape(N, K, L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for "
          "TMA.\n");
      return implementable;
    }
    if (args.lora_rank != LoRaK) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: this SM120 config stages a different LoRA rank\n");
      return false;
    }
    if (args.ptr_D == nullptr || args.ptr_L1 == nullptr) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: D/L1 pointers must be non-null\n");
      return false;
    }
    // Residual K must be whole tiles: a partial residual k-tile would change that step's
    // TMA byte count and break the fixed expect-transaction protocol.
    if (K % size<2>(TileShape{}) != 0) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: problem K must be a multiple of the CTA K tile\n");
      return false;
    }
    return implementable;
  }

  CUTLASS_DEVICE static void inline_lora_down_m537_k5120(Params const& params, int physical_block,
                                                         int thread, float* accumulator_tiles) {
    constexpr int kInlineM = 537;
    constexpr int kInlineK = 5120;
    constexpr int kInlineRank = 32;
    constexpr int kDownCols = 32;
    // This producer writes a rank-32 down projection. The shape guard below keys
    // on inline_down_m/k and never looks at the rank, so on a collective staging
    // a different rank it would fill the wrong extent silently rather than fail.
    if constexpr (LoRaK != kInlineRank) {
      return;
    }
    constexpr int kDownElements = 16 * kDownCols;
    constexpr int kDownTiles = (kInlineM + 15) / 16;
    if (params.inline_down_m != kInlineM || params.inline_down_k != kInlineK ||
        physical_block >= kDownTiles) {
      return;
    }

    auto const* x = reinterpret_cast<__nv_bfloat16 const*>(params.ptr_inline_down_x);
    auto const* l2t = reinterpret_cast<__nv_bfloat16 const*>(params.ptr_inline_down_l2t);
    auto* down = reinterpret_cast<__nv_bfloat16*>(params.ptr_inline_down_output);
    int const warp = thread / 32;
    int const lane = thread % 32;
    int const warp_count = int(blockDim.x) / 32;
    int const tile_m = physical_block;
    int const group = lane / 4;
    int const pair_col = (lane % 4) * 2;
    int const row = tile_m * 16 + group;
    int const col_base = group;
    float accum[4][4] = {};

#pragma unroll 1
    for (int step = warp * (kInlineK / 16) / warp_count;
         step < (warp + 1) * (kInlineK / 16) / warp_count; ++step) {
      int const k_base = step * 16;
      uint32_t a0 = 0u;
      uint32_t a1 = 0u;
      uint32_t a2 = 0u;
      uint32_t a3 = 0u;
      if (row < kInlineM) {
        auto const* a = x + row * kInlineK + k_base + pair_col;
        a0 = *reinterpret_cast<uint32_t const*>(a);
        a2 = *reinterpret_cast<uint32_t const*>(a + 8);
      }
      if (row + 8 < kInlineM) {
        auto const* a = x + (row + 8) * kInlineK + k_base + pair_col;
        a1 = *reinterpret_cast<uint32_t const*>(a);
        a3 = *reinterpret_cast<uint32_t const*>(a + 8);
      }
      auto const* b = l2t + (k_base + pair_col) * kInlineRank + col_base;
#pragma unroll
      for (int fragment = 0; fragment < 4; ++fragment) {
        auto const* bf = b + fragment * 8;
        uint32_t const b0 =
            static_cast<uint32_t>(*reinterpret_cast<uint16_t const*>(bf)) |
            (static_cast<uint32_t>(*reinterpret_cast<uint16_t const*>(bf + kInlineRank)) << 16);
        uint32_t const b1 =
            static_cast<uint32_t>(*reinterpret_cast<uint16_t const*>(bf + 8 * kInlineRank)) |
            (static_cast<uint32_t>(*reinterpret_cast<uint16_t const*>(bf + 9 * kInlineRank)) << 16);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(accum[fragment][0]), "+f"(accum[fragment][1]), "+f"(accum[fragment][2]),
              "+f"(accum[fragment][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }

#pragma unroll
    for (int fragment = 0; fragment < 4; ++fragment) {
      int const output0 = group * kDownCols + fragment * 8 + pair_col;
      int const output1 = output0 + 8 * kDownCols;
      *reinterpret_cast<float2*>(accumulator_tiles + warp * kDownElements + output0) =
          make_float2(accum[fragment][0], accum[fragment][1]);
      *reinterpret_cast<float2*>(accumulator_tiles + warp * kDownElements + output1) =
          make_float2(accum[fragment][2], accum[fragment][3]);
    }
    __syncthreads();

    for (int element = thread; element < kDownElements; element += int(blockDim.x)) {
      float reduced = 0.0f;
#pragma unroll 1
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        reduced += accumulator_tiles[source_warp * kDownElements + element];
      }
      int const output_row = tile_m * 16 + element / kDownCols;
      int const output_col = element % kDownCols;
      if (output_row < kInlineM) {
        down[output_row * kInlineRank + output_col] = __float2bfloat16_rn(reduced);
      }
    }
    __syncthreads();
    if (thread == 0) {
      __threadfence();
      atomicAdd(params.ptr_inline_down_counters + tile_m / 8, 1u);
    }
  }

  template <class MCoord, class NCoord>
  CUTLASS_DEVICE static void wait_for_inline_lora_down(Params const& params, MCoord const& m_coord,
                                                       NCoord const& n_coord) {
    if (params.ptr_inline_down_counters == nullptr) {
      return;
    }
    int const original_m_tile = params.inline_down_wait_on_n ? int(n_coord) : int(m_coord);
    int const expected_tiles = original_m_tile < 4 ? 8 : 2;
    while (atomicAdd(params.ptr_inline_down_counters + original_m_tile, 0u) <
           static_cast<uint32_t>(expected_tiles)) {
      __nanosleep(64);
    }
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
    if constexpr (UseLoRaPipeline) {
      cute::prefetch_tma_descriptor(
          params.lora_dedicated.tma_load_d_dedicated.get_tma_descriptor());
      cute::prefetch_tma_descriptor(
          params.lora_dedicated.tma_load_l1_dedicated.get_tma_descriptor());
    } else {
      cute::prefetch_tma_descriptor(params.tma_load_d.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_l1.get_tma_descriptor());
    }
  }

  // Temporary adhoc partitioning for scaling factors (identical to stock).
  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr auto thrfrg_SFA(SFATensor&& sfatensor,
                                             TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    using AtomShape_MNK = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);

    auto a_tile =
        make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);

    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);

    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);

    return thr_tensor;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor,
                                             TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    using AtomShape_MNK = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);

    auto a_tile =
        make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);

    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);

    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr auto partition_fragment_SFA(SFATensor&& sfatensor,
                                                         ThrMma& thread_mma) {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                  thrfrg_SFA(sfatensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFA);
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor,
                                                         ThrMma& thread_mma) {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                  thrfrg_SFB(sfbtensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFB);
  }

  template <class TiledMmaT>
  CUTE_HOST_DEVICE constexpr auto get_layoutSFA_TV(TiledMmaT& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    auto atile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<1>{}, Int<0>{})),
                     _));

    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
  }

  template <class TiledMmaT>
  CUTE_HOST_DEVICE constexpr auto get_layoutSFB_TV(TiledMmaT& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    auto btile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<0>{}, Int<1>{})),
                     _));

    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
  }

  /// Set up the data needed by this collective for load and mma.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL,
                                Params const& params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;

    Tensor mA_mkl = params.tma_load_a.get_tma_tensor(make_shape(M, K, L));
    Tensor mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(N, K, L));
    Tensor mSFA_mkl = params.tma_load_sfa.get_tma_tensor(shape(params.layout_SFA));
    Tensor mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(params.layout_SFB));

    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

    Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

    // One rank-32 chunk per output tile: tile shape (BLK_M/BLK_N, box K); the rank mode has
    // exactly one tile. The box K is the stage overlay width (LoRaStorageK) for the
    // byte-exact policy and the true rank for the pipeline-owning policies, whose
    // coordinate tensors must come from their own descriptors.
    if constexpr (UseLoRaPipeline) {
      Tensor mD_mkl = params.lora_dedicated.tma_load_d_dedicated.get_tma_tensor(
          make_shape(M, Int<LoRaDedicatedTmaRankK>{}, L));
      Tensor mL1_nkl = params.lora_dedicated.tma_load_l1_dedicated.get_tma_tensor(
          make_shape(N, Int<LoRaDedicatedTmaRankK>{}, L));
      auto lora_tiler_d = make_shape(shape<0>(TileShape{}), Int<LoRaDedicatedTmaStorageK>{});
      auto lora_tiler_l1 = make_shape(shape<1>(TileShape{}), Int<LoRaDedicatedTmaStorageK>{});
      Tensor gD_mkl = local_tile(mD_mkl, lora_tiler_d, make_coord(_, _, _));     // (BLK_M,32,m,1,l)
      Tensor gL1_nkl = local_tile(mL1_nkl, lora_tiler_l1, make_coord(_, _, _));  // (BLK_N,32,n,1,l)

      return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gD_mkl, gL1_nkl);
    } else {
      Tensor mD_mkl = params.tma_load_d.get_tma_tensor(make_shape(M, Int<LoRaTmaRankK>{}, L));
      Tensor mL1_nkl = params.tma_load_l1.get_tma_tensor(make_shape(N, Int<LoRaTmaRankK>{}, L));
      auto lora_tiler_d = make_shape(shape<0>(TileShape{}), Int<LoRaTmaStorageK>{});
      auto lora_tiler_l1 = make_shape(shape<1>(TileShape{}), Int<LoRaTmaStorageK>{});
      Tensor gD_mkl = local_tile(mD_mkl, lora_tiler_d, make_coord(_, _, _));
      Tensor gL1_nkl = local_tile(mL1_nkl, lora_tiler_l1, make_coord(_, _, _));

      return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gD_mkl, gL1_nkl);
    }
  }

  /// Producer perspective. `do_lora` appends ONE extra D/L1 pipeline step after the
  /// residual K-loop; the kernel passes compute_epilogue(work_tile_info) so exactly the
  /// epilogue-owning work unit produces (and later consumes) the LoRA stage.
  template <class TensorA, class TensorB, class TensorSFA, class TensorSFB, class TensorD,
            class TensorL1, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(
      Params const& params, MainloopPipeline pipeline, PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB, TensorD, TensorL1> const& load_inputs,
      BlockCoord const& blk_coord, KTileIterator k_tile_iter, int k_tile_count, int thread_idx,
      uint32_t block_rank_in_cluster, TensorStorage& shared_tensors, bool do_lora) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
      Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
      Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});
      // The TMA producer uses the byte-domain recast of the consumer's bf16
      // layout when swizzling is enabled; both views address the same bytes.
      Tensor sD = make_tensor(
          make_smem_ptr(recast_ptr<LoRaTmaInternalElement>(shared_tensors.smem_A.begin())),
          SmemLayoutDTma{});
      Tensor sL1 = make_tensor(
          make_smem_ptr(recast_ptr<LoRaTmaInternalElement>(shared_tensors.smem_B.begin())),
          SmemLayoutL1Tma{});

      auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gD_mkl, gL1_nkl] = load_inputs;

      auto block_tma_a = params.tma_load_a.get_slice(0);
      auto block_tma_b = params.tma_load_b.get_slice(0);

      auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
      auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

      auto block_tma_d = params.tma_load_d.get_slice(0);
      auto block_tma_l1 = params.tma_load_l1.get_slice(0);

      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      auto broadcast_m =
          make_layout(make_shape(Int<size<0>(TileShapeSFA{}) / size<0>(TileShape{})>{},
                                 Int<cute::numeric_limits<int>::max()>{}),
                      make_stride(_0{}, size<0>(TileShapeSFA{}) / size<0>(TileShape{})));
      auto broadcast_n =
          make_layout(make_shape(Int<size<1>(TileShapeSFB{}) / size<1>(TileShape{})>{},
                                 Int<cute::numeric_limits<int>::max()>{}),
                      make_stride(_0{}, size<1>(TileShapeSFB{}) / size<1>(TileShape{})));
      Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
      Tensor gSFA = gSFA_mkl(_, _, broadcast_m(m_coord), _, l_coord);
      Tensor gSFB = gSFB_nkl(_, _, broadcast_n(n_coord), _, l_coord);
      Tensor gD = gD_mkl(_, _, m_coord, Int<0>{}, l_coord);
      Tensor gL1 = gL1_nkl(_, _, n_coord, Int<0>{}, l_coord);

      Tensor tAgA = block_tma_a.partition_S(gA);
      Tensor tAsA = block_tma_a.partition_D(sA);

      Tensor tBgB = block_tma_b.partition_S(gB);
      Tensor tBsB = block_tma_b.partition_D(sB);

      Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);
      Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);

      Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);

      Tensor tDgD = block_tma_d.partition_S(gD);
      Tensor tDsD = block_tma_d.partition_D(sD);

      Tensor tL1gL1 = block_tma_l1.partition_S(gL1);
      Tensor tL1sL1 = block_tma_l1.partition_D(sL1);

      // Residual mainloop: each step arms LoRaTmaBytes by default and adds the residual
      // remainder BEFORE issuing its TMAs.
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(smem_pipe_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_, _, _, *k_tile_iter),
             tAsA(_, _, _, write_stage));
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_, _, _, *k_tile_iter),
             tBsB(_, _, _, write_stage));

        copy(params.tma_load_sfa.with(*tma_barrier), tAgSFA(_, _, _, *k_tile_iter),
             tAsSFA(_, _, _, write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier), tBgSFB(_, _, _, *k_tile_iter),
             tBsSFB(_, _, _, write_stage));

        ++k_tile_iter;
        ++smem_pipe_write;
      }

      // One extra producer step: D + L1 into the next freed stage (epilogue owner only).
      // SFA/SFB are dummy-reloaded (k-tile 0) so the arrived bytes exactly match the
      // armed A+B+SF budget; the D/L1 boxes are byte-exact stage overlays.
      if (do_lora) {
        if constexpr (CompileInlineDown) {
          wait_for_inline_lora_down(params, m_coord, n_coord);
        }
        if constexpr (UseRow80BiasPrefetch) {
          if (params.ptr_bias != nullptr) {
            auto const* bias_tile = params.ptr_bias + int(m_coord) * int(size<0>(TileShape{}));
            asm volatile("prefetch.global.L1 [%0];" ::"l"(bias_tile));
          }
        }
        pipeline.producer_acquire(smem_pipe_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(params.tma_load_sfa.with(*tma_barrier), tAgSFA(_, _, _, Int<0>{}),
             tAsSFA(_, _, _, write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier), tBgSFB(_, _, _, Int<0>{}),
             tBsSFB(_, _, _, write_stage));
        copy(params.tma_load_d.with(*tma_barrier), tDgD, tDsD(_, _, _, write_stage));
        copy(params.tma_load_l1.with(*tma_barrier), tL1gL1, tL1sL1(_, _, _, write_stage));

        ++smem_pipe_write;
      }
    }
    __syncwarp();
  }

  /// Producer perspective for the dedicated LoRA policy. The
  /// residual K-loop is unchanged and its pipeline state never accounts for LoRA. For the
  /// epilogue-owning work unit the producer acquires the one-slot LoRA pipeline and issues
  /// exactly two true-rank TMAs (D, then L1) against its full barrier.
  /// The slot is acquired after the residual enqueues so a still-busy slot never
  /// blocks this unit's residual prefetch.
  template <class TensorA, class TensorB, class TensorSFA, class TensorSFB, class TensorD,
            class TensorL1, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(
      Params const& params, MainloopPipeline pipeline, PipelineState smem_pipe_write,
      LoRaPipeline lora_pipeline, LoRaPipelineState lora_pipe_write,
      cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB, TensorD, TensorL1> const& load_inputs,
      BlockCoord const& blk_coord, KTileIterator k_tile_iter, int k_tile_count, int thread_idx,
      uint32_t block_rank_in_cluster, TensorStorage& shared_tensors, bool do_lora) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
      Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
      Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

      auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gD_mkl, gL1_nkl] = load_inputs;

      auto block_tma_a = params.tma_load_a.get_slice(0);
      auto block_tma_b = params.tma_load_b.get_slice(0);

      auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
      auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

      auto block_tma_d = params.lora_dedicated.tma_load_d_dedicated.get_slice(0);
      auto block_tma_l1 = params.lora_dedicated.tma_load_l1_dedicated.get_slice(0);

      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      auto broadcast_m =
          make_layout(make_shape(Int<size<0>(TileShapeSFA{}) / size<0>(TileShape{})>{},
                                 Int<cute::numeric_limits<int>::max()>{}),
                      make_stride(_0{}, size<0>(TileShapeSFA{}) / size<0>(TileShape{})));
      auto broadcast_n =
          make_layout(make_shape(Int<size<1>(TileShapeSFB{}) / size<1>(TileShape{})>{},
                                 Int<cute::numeric_limits<int>::max()>{}),
                      make_stride(_0{}, size<1>(TileShapeSFB{}) / size<1>(TileShape{})));
      Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
      Tensor gSFA = gSFA_mkl(_, _, broadcast_m(m_coord), _, l_coord);
      Tensor gSFB = gSFB_nkl(_, _, broadcast_n(n_coord), _, l_coord);
      Tensor gD = gD_mkl(_, _, m_coord, Int<0>{}, l_coord);    // (BLK_M,32)
      Tensor gL1 = gL1_nkl(_, _, n_coord, Int<0>{}, l_coord);  // (BLK_N,32)

      Tensor tAgA = block_tma_a.partition_S(gA);
      Tensor tAsA = block_tma_a.partition_D(sA);

      Tensor tBgB = block_tma_b.partition_S(gB);
      Tensor tBsB = block_tma_b.partition_D(sB);

      Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);
      Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);

      Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);

      // Gmem-side partitions only; the smem destinations are policy-dependent and are
      // partitioned inside the do_lora branch below.
      Tensor tDgD = block_tma_d.partition_S(gD);
      Tensor tL1gL1 = block_tma_l1.partition_S(gL1);

      // Residual mainloop, byte-identical to the byte-exact policy: every step arms and
      // arrives the full residual budget.
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(smem_pipe_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_, _, _, *k_tile_iter),
             tAsA(_, _, _, write_stage));
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_, _, _, *k_tile_iter),
             tBsB(_, _, _, write_stage));

        copy(params.tma_load_sfa.with(*tma_barrier), tAgSFA(_, _, _, *k_tile_iter),
             tAsSFA(_, _, _, write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier), tBgSFB(_, _, _, *k_tile_iter),
             tBsSFB(_, _, _, write_stage));

        ++k_tile_iter;
        ++smem_pipe_write;
      }

      // Owner work unit only: exactly two TMAs against the LoRA full barrier, whose
      // producer_acquire armed exactly LoRaDedicatedTmaBytes. No dummy SF reloads and no
      // main-pipeline step exist on these policies.
      if (do_lora) {
        if constexpr (CompileInlineDown) {
          wait_for_inline_lora_down(params, m_coord, n_coord);
        }
        if constexpr (UseRow80BiasPrefetch) {
          if (params.ptr_bias != nullptr) {
            auto const* bias_tile = params.ptr_bias + int(m_coord) * int(size<0>(TileShape{}));
            asm volatile("prefetch.global.L1 [%0];" ::"l"(bias_tile));
          }
        }
        {
          lora_pipeline.producer_acquire(lora_pipe_write);

          using LoRaBarrierType = typename LoRaPipeline::ProducerBarrierType;
          LoRaBarrierType* lora_barrier = lora_pipeline.producer_get_barrier(lora_pipe_write);

          // D/L1 stage into their dedicated buffers; the residual arrays are never aliased.
          Tensor sD = make_tensor(make_smem_ptr(recast_ptr<LoRaDedicatedTmaInternalElement>(
                                      shared_tensors.smem_D.begin())),
                                  SmemLayoutDDedicatedTma{});
          Tensor sL1 = make_tensor(make_smem_ptr(recast_ptr<LoRaDedicatedTmaInternalElement>(
                                       shared_tensors.smem_L1.begin())),
                                   SmemLayoutL1DedicatedTma{});
          Tensor tDsD = block_tma_d.partition_D(sD);
          Tensor tL1sL1 = block_tma_l1.partition_D(sL1);

          copy(params.lora_dedicated.tma_load_d_dedicated.with(*lora_barrier), tDgD, tDsD);
          copy(params.lora_dedicated.tma_load_l1_dedicated.with(*lora_barrier), tL1gL1, tL1sL1);
        }

        ++lora_pipe_write;
      }
    }
    __syncwarp();
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Producer epilogue for the dedicated policy: both pipelines quiesce independently.
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write,
                                LoRaPipeline lora_pipeline, LoRaPipelineState lora_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
      lora_pipeline.producer_tail(lora_pipe_write);
    }
  }

  /// Consumer perspective: residual K-loop only (identical to stock). The LoRA product is
  /// applied by mma_lora() below, invoked by the kernel after Stream-K fixup.
  template <class FrgTensorC, class BlockCoord>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                          FrgTensorC& accum, int k_tile_count, int thread_idx,
                          TensorStorage& shared_tensors, [[maybe_unused]] Params const& params,
                          BlockCoord const& blk_coord) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
    Tensor sSFA = [&]() {
      if constexpr (size<0>(TileShape{}) >= 128) {
        return make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
      } else {
        Tensor temp = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
        auto m = get<0>(blk_coord);
        return temp(make_coord(_, m % (size<0>(TileShapeSFA{}) / size<0>(TileShape{}))), _, _);
      }
    }();
    Tensor sSFB = [&]() {
      if constexpr (size<1>(TileShape{}) >= 128) {
        return make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});
      } else {
        Tensor temp = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});
        auto n = get<1>(blk_coord);
        return temp(make_coord(_, n % (size<1>(TileShapeSFB{}) / size<1>(TileShape{}))), _, _);
      }
    }();

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));

    Tensor tCrSFA = partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA =
        make_tiled_copy_impl(SmemCopyAtomSFA{}, get_layoutSFA_TV(tiled_mma),
                             make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA = smem_thr_copy_SFA.partition_S(as_position_independent_swizzle_tensor(sSFA));
    Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

    auto smem_tiled_copy_SFB =
        make_tiled_copy_impl(SmemCopyAtomSFB{}, get_layoutSFB_TV(tiled_mma),
                             make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB = smem_thr_copy_SFB.partition_S(as_position_independent_swizzle_tensor(sSFB));
    Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));

    CUTE_STATIC_ASSERT_V(size<1>(tCsSFA) == size<1>(tCrSFA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCrSFA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCsSFB));
    CUTE_STATIC_ASSERT_V(size<3>(tCsSFA) == size<3>(tCsSFB));
    CUTE_STATIC_ASSERT_V(size<2>(sA) == size<2>(sSFA));
    CUTE_STATIC_ASSERT_V(size<2>(sB) == size<2>(sSFA));

    //
    // PIPELINED MAIN LOOP
    //

    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage = smem_pipe_read.index();
    auto tCsA_stage = tCsA(_, _, _, read_stage);
    auto tCsB_stage = tCsB(_, _, _, read_stage);
    auto tCsSFA_stage = tCsSFA(_, _, _, read_stage);
    auto tCsSFB_stage = tCsSFB(_, _, _, read_stage);

    auto copy_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_A, tCsA_stage(_, _, k_block), tCrA_copy_view(_, _, k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_, _, k_block), tCrB_copy_view(_, _, k_block));

      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_, _, k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_, _, k_block));

      copy(tCsSFA_stage(_, _, k_block), tCrSFA_copy_view(_, _, k_block));
      copy(tCsSFB_stage(_, _, k_block), tCrSFB_copy_view(_, _, k_block));
    };

    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(tiled_mma, make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
                 make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)), accum);
    };

    pipeline.consumer_wait(smem_pipe_read);

    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 1; --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
              thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_stage = tCsA(_, _, _, read_stage);
          tCsB_stage = tCsB(_, _, _, read_stage);
          tCsSFA_stage = tCsSFA(_, _, _, read_stage);
          tCsSFB_stage = tCsSFB(_, _, _, read_stage);
          pipeline.consumer_wait(smem_pipe_read);
        }

        copy_kblock(k_block_next);
        gemm_kblock(k_block);
      });
    }  // k_tile_count

    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }

      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);
    });
  }

  /// Consume the LoRA stage: rank-32 bf16 D@L1^T added into the SAME accumulator fragment.
  /// Called by the kernel AFTER TileScheduler::fixup (accumulator fully reduced) and before
  /// the epilogue, only by the epilogue-owning work unit; consumes exactly one stage.
  template <class FrgTensorC>
  CUTLASS_DEVICE void mma_lora(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                               FrgTensorC& accum, int thread_idx, TensorStorage& shared_tensors) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    Tensor sD =
        make_tensor(make_smem_ptr(recast_ptr<cutlass::bfloat16_t>(shared_tensors.smem_A.begin())),
                    SmemLayoutDConsume{});
    Tensor sL1 =
        make_tensor(make_smem_ptr(recast_ptr<cutlass::bfloat16_t>(shared_tensors.smem_B.begin())),
                    SmemLayoutL1Consume{});

    LoRaMma lora_mma;
    auto thread_lora = lora_mma.get_thread_slice(thread_idx);

    Tensor tCrD = thread_lora.partition_fragment_A(sD(_, _, Int<0>{}));    // (MMA,MMA_M,MMA_K)
    Tensor tCrL1 = thread_lora.partition_fragment_B(sL1(_, _, Int<0>{}));  // (MMA,MMA_N,MMA_K)

    auto smem_tiled_copy_D = make_tiled_copy_A(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cutlass::bfloat16_t>{}, lora_mma);
    auto smem_thr_copy_D = smem_tiled_copy_D.get_thread_slice(thread_idx);
    Tensor tCsD = smem_thr_copy_D.partition_S(sD);
    Tensor tCrD_copy_view = smem_thr_copy_D.retile_D(tCrD);

    auto smem_tiled_copy_L1 = make_tiled_copy_B(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cutlass::bfloat16_t>{}, lora_mma);
    auto smem_thr_copy_L1 = smem_tiled_copy_L1.get_thread_slice(thread_idx);
    Tensor tCsL1 = smem_thr_copy_L1.partition_S(sL1);
    Tensor tCrL1_copy_view = smem_thr_copy_L1.retile_D(tCrL1);

    CUTE_STATIC_ASSERT_V(size<1>(tCrD) == size<1>(accum));   // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrL1) == size<2>(accum));  // MMA_N

    pipeline.consumer_wait(smem_pipe_read);
    int read_stage = smem_pipe_read.index();

    copy(smem_tiled_copy_D, tCsD(_, _, _, read_stage), tCrD_copy_view);
    copy(smem_tiled_copy_L1, tCsL1(_, _, _, read_stage), tCrL1_copy_view);

    // mma.sync accumulates C += A*B; the residual product is already in `accum`.
    auto LORA_K_BLOCK_MAX = size<2>(tCrD);
    for_each(make_int_sequence<LORA_K_BLOCK_MAX>{}, [&](auto k_block) {
      cute::gemm(lora_mma, tCrD(_, _, k_block), tCrL1(_, _, k_block), accum);
    });
    canonicalize_lora_tail_zeros(accum);

    cutlass::arch::NamedBarrier::sync(thr_size(TiledMma{}),
                                      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
    pipeline.consumer_release(smem_pipe_read);
  }

  /// Epilogue-overlapped consumption of the byte-exact LoRA stage, split into two phases:
  ///   begin    - wait for the stage the producer's extra step filled, stage the FULL
  ///              per-thread D/L1 fragments into registers with the same vectorized
  ///              tiled copies as the monolithic path, then release the stage
  ///              IMMEDIATELY (the overlay buffers go back to the producer before the
  ///              epilogue starts, so the next persistent tile's prefetch is never
  ///              throttled). The fragments are small: tile_M x 32 + tile_N x 32 bf16
  ///              spread over all MMA threads (~12-24 registers/thread); the pinned
  ///              ptxas spill gate is the arbiter of that budget.
  ///   subtile  - fold the bf16 product for exactly one epilogue subtile's accumulator
  ///              MMA blocks out of those registers (called by the epilogue's hook
  ///              before the subtile's fragments are visited, overlapping the in-flight
  ///              TMA stores of earlier subtiles). Per-element accumulation order over
  ///              k-blocks matches the monolithic path, so results stay bit-identical.
  /// The producer protocol and byte accounting are untouched.
  CUTLASS_DEVICE auto mma_lora_begin(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                                     int thread_idx, TensorStorage& shared_tensors) {
    using namespace cute;

    Tensor sD =
        make_tensor(make_smem_ptr(recast_ptr<cutlass::bfloat16_t>(shared_tensors.smem_A.begin())),
                    SmemLayoutDConsume{});
    Tensor sL1 =
        make_tensor(make_smem_ptr(recast_ptr<cutlass::bfloat16_t>(shared_tensors.smem_B.begin())),
                    SmemLayoutL1Consume{});

    LoRaMma lora_mma;
    auto thread_lora = lora_mma.get_thread_slice(thread_idx);

    Tensor tCrD = thread_lora.partition_fragment_A(sD(_, _, Int<0>{}));    // (MMA,MMA_M,MMA_K)
    Tensor tCrL1 = thread_lora.partition_fragment_B(sL1(_, _, Int<0>{}));  // (MMA,MMA_N,MMA_K)

    auto smem_tiled_copy_D = make_tiled_copy_A(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cutlass::bfloat16_t>{}, lora_mma);
    auto smem_thr_copy_D = smem_tiled_copy_D.get_thread_slice(thread_idx);
    Tensor tCsD = smem_thr_copy_D.partition_S(sD);
    Tensor tCrD_copy_view = smem_thr_copy_D.retile_D(tCrD);

    auto smem_tiled_copy_L1 = make_tiled_copy_B(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cutlass::bfloat16_t>{}, lora_mma);
    auto smem_thr_copy_L1 = smem_tiled_copy_L1.get_thread_slice(thread_idx);
    Tensor tCsL1 = smem_thr_copy_L1.partition_S(sL1);
    Tensor tCrL1_copy_view = smem_thr_copy_L1.retile_D(tCrL1);

    pipeline.consumer_wait(smem_pipe_read);
    int read_stage = smem_pipe_read.index();

    copy(smem_tiled_copy_D, tCsD(_, _, _, read_stage), tCrD_copy_view);
    copy(smem_tiled_copy_L1, tCsL1(_, _, _, read_stage), tCrL1_copy_view);

    // Every MMA thread must finish its smem reads before ANY thread releases the overlay
    // buffers back to the producer.
    cutlass::arch::NamedBarrier::sync(thr_size(TiledMma{}),
                                      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
    pipeline.consumer_release(smem_pipe_read);

    return cute::make_tuple(tCrD, tCrL1);
  }

  template <class LoRaFrags, class FrgTensorC>
  CUTLASS_DEVICE void mma_lora_subtile(LoRaFrags const& frags, FrgTensorC& accum, int mma_m_begin,
                                       int mma_m_count, int mma_n_begin, int mma_n_count) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    auto const& tCrD = cute::get<0>(frags);
    auto const& tCrL1 = cute::get<1>(frags);
    CUTE_STATIC_ASSERT_V(size<1>(tCrD) == size<1>(accum));   // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrL1) == size<2>(accum));  // MMA_N

    LoRaMma lora_mma;
    auto LORA_K_BLOCK_MAX = size<2>(tCrD);
    for (int m = mma_m_begin; m < mma_m_begin + mma_m_count; ++m) {
      for (int n = mma_n_begin; n < mma_n_begin + mma_n_count; ++n) {
        for_each(make_int_sequence<LORA_K_BLOCK_MAX>{}, [&](auto k_block) {
          cute::gemm(lora_mma, tCrD(_, m, k_block), tCrL1(_, n, k_block), accum(_, m, n));
        });
        canonicalize_lora_tail_zeros(accum(_, m, n));
      }
    }
  }

  /// Consume the dedicated LoRA slot: rank-32 bf16 D@L1^T added into the SAME accumulator
  /// fragment, called at the same points as the byte-exact mma_lora above. The slot is
  /// released as soon as every MMA thread holds its fragments in registers, BEFORE the bf16
  /// MMAs, so the producer may refill it while the tail MMAs execute.
  template <class FrgTensorC>
  CUTLASS_DEVICE void mma_lora(LoRaPipeline lora_pipeline, LoRaPipelineState lora_pipe_read,
                               FrgTensorC& accum, int thread_idx, TensorStorage& shared_tensors) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    Tensor sD =
        make_tensor(make_smem_ptr(shared_tensors.smem_D.begin()), SmemLayoutDDedicatedConsume{});
    Tensor sL1 =
        make_tensor(make_smem_ptr(shared_tensors.smem_L1.begin()), SmemLayoutL1DedicatedConsume{});

    LoRaMma lora_mma;
    auto thread_lora = lora_mma.get_thread_slice(thread_idx);

    Tensor tCrD = thread_lora.partition_fragment_A(sD);    // (MMA,MMA_M,MMA_K)
    Tensor tCrL1 = thread_lora.partition_fragment_B(sL1);  // (MMA,MMA_N,MMA_K)

    // Plain vectorized smem->rmem copies: the dedicated layout is un-swizzled K-major with
    // the same TiledMma-consistent per-thread partitioning as the byte-exact overlay path,
    // minus the stage mode.
    auto smem_tiled_copy_D = make_tiled_copy_A(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cutlass::bfloat16_t>{}, lora_mma);
    auto smem_thr_copy_D = smem_tiled_copy_D.get_thread_slice(thread_idx);
    Tensor tCsD = smem_thr_copy_D.partition_S(sD);
    Tensor tCrD_copy_view = smem_thr_copy_D.retile_D(tCrD);

    auto smem_tiled_copy_L1 = make_tiled_copy_B(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cutlass::bfloat16_t>{}, lora_mma);
    auto smem_thr_copy_L1 = smem_tiled_copy_L1.get_thread_slice(thread_idx);
    Tensor tCsL1 = smem_thr_copy_L1.partition_S(sL1);
    Tensor tCrL1_copy_view = smem_thr_copy_L1.retile_D(tCrL1);

    CUTE_STATIC_ASSERT_V(size<1>(tCrD) == size<1>(accum));   // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrL1) == size<2>(accum));  // MMA_N

    lora_pipeline.consumer_wait(lora_pipe_read);

    copy(smem_tiled_copy_D, tCsD, tCrD_copy_view);
    copy(smem_tiled_copy_L1, tCsL1, tCrL1_copy_view);

    // Every MMA thread must finish its smem reads before ANY thread releases the one-slot
    // buffer back to the producer; removing this barrier is a race against the next refill.
    cutlass::arch::NamedBarrier::sync(thr_size(TiledMma{}),
                                      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
    lora_pipeline.consumer_release(lora_pipe_read);

    // mma.sync accumulates C += A*B; the residual product is already in `accum`.
    auto LORA_K_BLOCK_MAX = size<2>(tCrD);
    for_each(make_int_sequence<LORA_K_BLOCK_MAX>{}, [&](auto k_block) {
      cute::gemm(lora_mma, tCrD(_, _, k_block), tCrL1(_, _, k_block), accum);
    });
  }

  /// Side-slot flavor of mma_lora_begin: stage the independent slot's rank-32
  /// payload into registers, release the one-slot pipeline, and let the epilogue
  /// hook fold the fragments per output subtile.
  CUTLASS_DEVICE auto mma_lora_begin(LoRaPipeline lora_pipeline, LoRaPipelineState lora_pipe_read,
                                     int thread_idx, TensorStorage& shared_tensors) {
    using namespace cute;

    Tensor sD =
        make_tensor(make_smem_ptr(shared_tensors.smem_D.begin()), SmemLayoutDDedicatedConsume{});
    Tensor sL1 =
        make_tensor(make_smem_ptr(shared_tensors.smem_L1.begin()), SmemLayoutL1DedicatedConsume{});

    LoRaMma lora_mma;
    auto thread_lora = lora_mma.get_thread_slice(thread_idx);

    Tensor tCrD = thread_lora.partition_fragment_A(sD);
    Tensor tCrL1 = thread_lora.partition_fragment_B(sL1);

    using LoRaSmemCopyAtomD = cute::conditional_t<
        UseRow80SideSlot, Copy_Atom<SM75_U32x4_LDSM_N, cutlass::bfloat16_t>,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cutlass::bfloat16_t>>;
    auto smem_tiled_copy_D = make_tiled_copy_A(LoRaSmemCopyAtomD{}, lora_mma);
    auto smem_thr_copy_D = smem_tiled_copy_D.get_thread_slice(thread_idx);
    Tensor tCsD = smem_thr_copy_D.partition_S(sD);
    Tensor tCrD_copy_view = smem_thr_copy_D.retile_D(tCrD);

    using LoRaSmemCopyAtomL1 = cute::conditional_t<
        UseRow80SideSlot, Copy_Atom<SM75_U32x2_LDSM_N, cutlass::bfloat16_t>,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cutlass::bfloat16_t>>;
    auto smem_tiled_copy_L1 = make_tiled_copy_B(LoRaSmemCopyAtomL1{}, lora_mma);
    auto smem_thr_copy_L1 = smem_tiled_copy_L1.get_thread_slice(thread_idx);
    Tensor tCsL1 = smem_thr_copy_L1.partition_S(sL1);
    Tensor tCrL1_copy_view = smem_thr_copy_L1.retile_D(tCrL1);

    lora_pipeline.consumer_wait(lora_pipe_read);
    copy(smem_tiled_copy_D, tCsD, tCrD_copy_view);
    copy(smem_tiled_copy_L1, tCsL1, tCrL1_copy_view);
    cutlass::arch::NamedBarrier::sync(thr_size(TiledMma{}),
                                      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
    lora_pipeline.consumer_release(lora_pipe_read);

    return cute::make_tuple(tCrD, tCrL1);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void mma_tail(MainloopPipeline, PipelineState, int) {}
};

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel driver: copy of the CUTLASS SM90 cooperative warp-specialized kernel (which drives
// the stock SM120 block-scaled schedule) with the LoRA producer/consumer hooks:
//   - producer computes do_lora = compute_epilogue(work_tile_info) and passes it to load();
//   - byte-exact policy: producer/consumer mainloop states advance by
//     work_k_tile_count + do_lora; dedicated policy: mainloop states advance by
//     work_k_tile_count only and the LoRA pipeline's own states advance by exactly do_lora;
//   - consumer calls mma_lora() after TileScheduler::fixup and before the epilogue store.
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace flashinfer::gemm::svdquant_sm120 {

using namespace cute;

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_,
          class TileSchedulerTag_>
class GemmUniversalLoRaSm120 {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
                "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);
  static_assert(cute::size(ClusterShape{}) == 1,
                "SM120 fused SVDQuant kernel requires a 1x1x1 cluster");

  static constexpr uint32_t TileSchedulerPipelineStageCount =
      DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  using TileSchedulerTag = TileSchedulerTag_;

  using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
      TileSchedulerTag, ArchTag, TileShape, ClusterShape,
      TileSchedulerPipelineStageCount>::Scheduler;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t NumSchedThreads = cutlass::NumThreadsPerWarp;
  static constexpr uint32_t NumMMAThreads = size(TiledMma{});
  static constexpr uint32_t NumMainloopLoadThreads = cutlass::NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueLoadThreads = cutlass::NumThreadsPerWarp;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = NumMMAThreads / cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock =
      NumMMAThreads + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumFixupBarriers = NumMmaWarpGroups;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;

  // Closed contract over the supported (tile M, MMA threads, warpgroups) families: a
  // 64-row tile runs one consumer warpgroup, every larger tile two. Block size, pipeline
  // quorums, scheduler arrivals, and fixup barriers derive from the TiledMma size above.
  static constexpr bool IsSmallM = size<0>(TileShape{}) == 64;
  static_assert((IsSmallM && NumMMAThreads == 128 && NumMmaWarpGroups == 1) ||
                    (!IsSmallM && NumMMAThreads == 256 && NumMmaWarpGroups == 2),
                "TiledMMA consumer geometry is outside the closed SM120 SVDQuant contract.");
  static_assert(!IsSmallM || (MaxThreadsPerBlock == 256 && NumFixupBarriers == 1),
                "a 64-row tile must derive a 256-thread block and one fixup barrier");

  // The LoRa* types are valid for either policy; the pipeline object exists
  // only for the dedicated side slot.
  static constexpr bool UseDedicatedLoRaTma = CollectiveMainloop::UseDedicatedLoRaTma;
  static constexpr bool UseRow80SideSlot = CollectiveMainloop::UseRow80SideSlot;
  static constexpr bool UseLoRaPipeline = CollectiveMainloop::UseLoRaPipeline;
  using LoRaPipeline = typename CollectiveMainloop::LoRaPipeline;
  using LoRaPipelineState = typename CollectiveMainloop::LoRaPipelineState;

  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread = size<0>(TileShape{}) * size<1>(TileShape{}) / NumMMAThreads *
                                       sizeof(ElementAccumulator) / sizeof(uint32_t);

  // SM120 block-scaled register-accumulator kernels hit high register pressure on
  // >=128-regs-per-thread accumulator tiles (matches the stock SM120 heuristic).
  static constexpr bool HeavyRegisterPressure = (RegsPerThread >= 128);

  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;

  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1, 2>;

  using TileSchedulerPipeline = typename TileScheduler::Pipeline;
  using TileSchedulerPipelineState = typename TileSchedulerPipeline::PipelineState;
  using TileSchedulerStorage = typename TileScheduler::SharedStorage;
  using TileSchedulerThrottlePipeline = typename TileScheduler::ThrottlePipeline;
  using TileSchedulerThrottlePipelineState = typename TileSchedulerThrottlePipeline::PipelineState;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;

    alignas(16) TileSchedulerStorage scheduler;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    cutlass::gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    cutlass::gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };

  //
  // Methods
  //

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM "
          "count.");
      sm_count =
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
    }

    cutlass::KernelHardwareInfo hw_info = args.hw_info;
    hw_info.sm_count = sm_count;
    hw_info.max_active_clusters = max_active_clusters;

    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset, cutlass::MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset +=
        TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = cutlass::round_nearest(workspace_offset, cutlass::MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler,
        scheduler_workspace, NumEpilogueSubTiles);

    return {args.mode,
            problem_shape,
            CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop,
                                                        mainloop_workspace),
            CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue,
                                                        epilogue_workspace),
            hw_info,
            scheduler,
            workspace};
  }

  static bool can_implement(Arguments const& args) {
    bool implementable = (args.mode == cutlass::gemm::GemmUniversalMode::kGemm) or
                         (args.mode == cutlass::gemm::GemmUniversalMode::kBatched &&
                          cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    // Separate-reduction Stream-K units obtain their accumulator purely from workspace and
    // never run the mainloop load path, so they would not receive a LoRA stage. This kernel
    // keeps separate reduction off structurally: to_underlying_arguments passes
    // NumEpilogueSubTiles (store-pipe increment, 1 for this epilogue) to the scheduler and
    // the SM100-wrapped Stream-K scheduler's requires_separate_reduction is always false.
    return implementable;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = cutlass::round_nearest(workspace_size, cutlass::MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_size = cutlass::round_nearest(workspace_size, cutlass::MinWorkspaceAlignment);
    return workspace_size;
  }

  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace = nullptr,
                                              cudaStream_t stream = nullptr,
                                              cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    cutlass::Status status = cutlass::Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(
        args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset, cutlass::MinWorkspaceAlignment);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
        args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info,
        NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset +=
        TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups,
            NumEpilogueSubTiles);
    workspace_offset = cutlass::round_nearest(workspace_offset, cutlass::MinWorkspaceAlignment);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN
                            ? TileScheduler::RasterOrderOptions::AlongN
                            : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{},
                                         ClusterShape{}, params.hw_info, args);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

#if (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
     CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#define ENABLE_SM120_SVDQ_KERNEL_LEVEL 1
#endif

#if !defined(ENABLE_SM120_SVDQ_KERNEL_LEVEL)
    CUTE_INVALID_CONTROL_PATH(
        "ERROR : Arch conditional MMA instruction used without targeting appropriate compute "
        "capability. Aborting.\n");
#else

    // Preconditions (the M/thread-count contract is asserted at class scope)
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L].");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L].");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L].");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L].");

    enum class WarpGroupRole { Producer = 0, Consumer0 = 1, Consumer1 = 2 };
    enum class ProducerWarpRole { Mainloop = 0, Warp1 = 1, Epilogue = 2, MainloopAux = 3 };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = cutlass::canonical_lane_idx();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % NumMMAThreads;
    auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    // Valid consumer roles are the NumMmaWarpGroups warpgroups after the producer;
    // Consumer1 exists only in the two-warpgroup shape.
    auto is_consumer_warp_group = [](WarpGroupRole role) {
      auto wg_idx = static_cast<uint32_t>(role);
      return wg_idx >= NumLoadWarpGroups && wg_idx < NumLoadWarpGroups + NumMmaWarpGroups;
    };
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();

    // TileScheduler pipeline
    typename TileSchedulerPipeline::Params scheduler_pipeline_params;
    typename TileSchedulerThrottlePipeline::Params scheduler_throttle_pipeline_params;
    if constexpr (IsSchedDynamicPersistent) {
      if (warp_group_role == WarpGroupRole::Producer &&
          producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::ProducerConsumer;
      } else {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::Consumer;
      }
      scheduler_pipeline_params.producer_blockid = 0;
      scheduler_pipeline_params.producer_arv_count = 1;
      scheduler_pipeline_params.consumer_arv_count =
          NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads;

      if (is_epi_load_needed) {
        scheduler_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
      }
      scheduler_pipeline_params.transaction_bytes = sizeof(typename TileScheduler::CLCResponse);

      scheduler_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
      scheduler_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
      scheduler_throttle_pipeline_params.dst_blockid = 0;
      scheduler_throttle_pipeline_params.initializing_warp = 3;
      if (warp_group_role == WarpGroupRole::Producer &&
          producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Consumer;
      } else if (warp_group_role == WarpGroupRole::Producer &&
                 producer_warp_role == ProducerWarpRole::Mainloop) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Producer;
      }
    }
    TileSchedulerPipeline scheduler_pipeline(shared_storage.scheduler.pipeline(),
                                             scheduler_pipeline_params);
    TileSchedulerPipelineState scheduler_pipe_consumer_state;

    TileSchedulerThrottlePipeline scheduler_throttle_pipeline(
        shared_storage.scheduler.throttle_pipeline(), scheduler_throttle_pipeline_params);
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_consumer_state;
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_producer_state =
        cutlass::make_producer_start_state<TileSchedulerThrottlePipeline>();

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        (producer_warp_role == ProducerWarpRole::Mainloop ||
         producer_warp_role == ProducerWarpRole::MainloopAux)) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (is_consumer_warp_group(warp_group_role)) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumMMAThreads;
    mainloop_pipeline_params.num_producers = NumProducerThreads;
    // Arms the residual A+B+SF budget on every acquire. Byte-exact policy: the extra LoRA
    // step arrives exactly this budget too. Dedicated policy: the mainloop never sees LoRA
    // bytes; those are armed by the LoRA pipeline's own params below.
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(
        CollectiveMainloop::mainloop_pipeline_storage(shared_storage.pipelines.mainloop),
        mainloop_pipeline_params, ClusterShape{});

    // LoRA (D+L1) pipeline for the pipeline-owning policies: one slot carrying its own
    // transaction budget; its producer is the mainloop producer warp and its consumers are
    // the MMA threads. The policies differ only in where the TMAs land (carveout vs. the
    // borrow stage base), which the pipeline does not see.
    struct LoRaPipelineNone {};
    auto lora_pipeline = [&]() {
      if constexpr (UseLoRaPipeline) {
        typename LoRaPipeline::Params lora_pipeline_params;
        if (warp_group_role == WarpGroupRole::Producer &&
            (producer_warp_role == ProducerWarpRole::Mainloop ||
             producer_warp_role == ProducerWarpRole::MainloopAux)) {
          lora_pipeline_params.role = LoRaPipeline::ThreadCategory::Producer;
        }
        if (is_consumer_warp_group(warp_group_role)) {
          lora_pipeline_params.role = LoRaPipeline::ThreadCategory::Consumer;
        }
        lora_pipeline_params.is_leader = warp_group_thread_idx == 0;
        lora_pipeline_params.num_consumers = NumMMAThreads;
        lora_pipeline_params.num_producers = NumProducerThreads;
        lora_pipeline_params.transaction_bytes = CollectiveMainloop::LoRaDedicatedTmaBytes;
        return LoRaPipeline(
            CollectiveMainloop::lora_pipeline_storage(shared_storage.pipelines.mainloop),
            lora_pipeline_params, ClusterShape{});
      } else {
        return LoRaPipelineNone{};
      }
    }();

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (is_consumer_warp_group(warp_group_role)) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumMMAThreads;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = cutlass::NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order,
                                            params_load_order_barrier);

    // Initialize starting pipeline states for the collectives
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    cutlass::PipelineState mainloop_pipe_producer_state =
        cutlass::make_producer_start_state<MainloopPipeline>();
    cutlass::PipelineState epi_load_pipe_producer_state =
        cutlass::make_producer_start_state<EpiLoadPipeline>();
    cutlass::PipelineState epi_store_pipe_producer_state =
        cutlass::make_producer_start_state<EpiStorePipeline>();

    // Pipeline-owning policies only: LoRA states persist across work units and advance by
    // exactly do_lora per unit, independently of the mainloop states above.
    [[maybe_unused]] LoRaPipelineState lora_pipe_consumer_state;
    [[maybe_unused]] LoRaPipelineState lora_pipe_producer_state =
        cutlass::make_producer_start_state<LoRaPipeline>();

    auto cluster_wait_fn = []() {
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return []() { cute::cluster_wait(); };
      } else {
        __syncthreads();
        return []() {};
      }
    }();

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

    TiledMma tiled_mma;
    auto blk_shape = TileShape{};

    TileScheduler scheduler{params.scheduler};
    if constexpr (IsSchedDynamicPersistent) {
      scheduler.set_data_ptr(shared_storage.scheduler.data());
    }
    typename TileScheduler::WorkTileInfo work_tile_info;

    CollectiveMainloop collective_mainloop;

    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2,
                  "Output of load_init must have at least two elements (A, B)");

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    cluster_wait_fn();

    // Tactic 9 gets a CompileInlineDown=false specialization so this optional
    // experiment cannot perturb its production codegen. Other tactics retain
    // their measured legacy layout, and tactic 81 still serves the explicit
    // inline-down test API when the pointers are populated.
    if constexpr (CollectiveMainloop::CompileInlineDown) {
      if (params.mainloop.ptr_inline_down_counters != nullptr) {
        int const physical_block =
            int(blockIdx.x) + int(gridDim.x) * (int(blockIdx.y) + int(gridDim.y) * int(blockIdx.z));
        auto* inline_down_scratch = reinterpret_cast<float*>(&shared_storage.tensors.mainloop);
        CollectiveMainloop::inline_lora_down_m537_k5120(params.mainloop, physical_block, thread_idx,
                                                        inline_down_scratch);
      }
    }

    if (warp_group_role == WarpGroupRole::Producer) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Scheduler Producer Warp
      if (producer_warp_role == ProducerWarpRole::Warp1) {
        if constexpr (IsSchedDynamicPersistent) {
          bool requires_clc_query = true;
          TileSchedulerPipelineState scheduler_pipe_producer_state =
              cutlass::make_producer_start_state<TileSchedulerPipeline>();

          cutlass::arch::wait_on_dependent_grids();
          while (work_tile_info.is_valid()) {
            if (requires_clc_query) {
              scheduler_throttle_pipeline.consumer_wait(scheduler_pipe_throttle_consumer_state);
              scheduler_throttle_pipeline.consumer_release(scheduler_pipe_throttle_consumer_state);
              ++scheduler_pipe_throttle_consumer_state;

              scheduler_pipe_producer_state =
                  scheduler.advance_to_next_work(scheduler_pipeline, scheduler_pipe_producer_state);
            }

            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }

            work_tile_info = next_work_tile_info;
          }
          scheduler_pipeline.producer_tail(scheduler_pipe_producer_state);
        }
      }  // Scheduler Producer Warp End

      // Mainloop Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Mainloop) {
        cutlass::arch::wait_on_dependent_grids();
        bool do_load_order_arrive = true;
        bool requires_clc_query = true;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next_work_tile_info;
            continue;
          }

          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          auto work_k_tile_count =
              TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)),
                                                       shape<3>(gA_mkl));

          // The epilogue-owning work unit produces one extra D/L1 stage after its K range.
          bool do_lora = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

          if (requires_clc_query) {
            scheduler_throttle_pipeline.producer_acquire(scheduler_pipe_throttle_producer_state);
            scheduler_throttle_pipeline.producer_commit(scheduler_pipe_throttle_producer_state);
            ++scheduler_pipe_throttle_producer_state;
          }

          if constexpr (UseLoRaPipeline) {
            collective_mainloop.load(
                params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, lora_pipeline,
                lora_pipe_producer_state, load_inputs, blk_coord, k_tile_iter, work_k_tile_count,
                lane_idx, block_rank_in_cluster, shared_storage.tensors.mainloop, do_lora);
            // The mainloop state advances by the residual K tiles only; the LoRA pipeline
            // advances by exactly do_lora.
            mainloop_pipe_producer_state.advance(work_k_tile_count);
            lora_pipe_producer_state.advance(do_lora ? 1 : 0);
          } else {
            collective_mainloop.load(
                params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs,
                blk_coord, k_tile_iter, work_k_tile_count, lane_idx, block_rank_in_cluster,
                shared_storage.tensors.mainloop, do_lora);
            // Update starting pipeline state for the next tile (+1 for the LoRA stage).
            mainloop_pipe_producer_state.advance(work_k_tile_count + (do_lora ? 1 : 0));
          }

          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
              work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          work_tile_info = next_work_tile_info;
          if constexpr (IsSchedDynamicPersistent) {
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
        }  // Scheduler work fetch loop

        if constexpr (UseLoRaPipeline) {
          collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state,
                                        lora_pipeline, lora_pipe_producer_state);
        } else {
          collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
        }
      }

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && is_epi_load_needed) {
        cutlass::arch::wait_on_dependent_grids();

        if (!TileScheduler::requires_separate_reduction(params.scheduler) &&
            work_tile_info.is_valid()) {
          load_order_barrier.wait();
        }

        CollectiveEpilogue local_collective_epilogue(params.epilogue,
                                                     shared_storage.tensors.epilogue);

        while (work_tile_info.is_valid()) {
          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            epi_load_pipe_producer_state = local_collective_epilogue.load(
                epi_load_pipeline, epi_load_pipe_producer_state, problem_shape_MNKL, blk_shape,
                blk_coord, tiled_mma, lane_idx, shared_storage.tensors.epilogue,
                work_tile_info.reduction_subtile_idx());
          }

          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
              work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
          work_tile_info = next_work_tile_info;
          if constexpr (IsSchedDynamicPersistent) {
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
        }  // Scheduler work fetch loop

        local_collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      }  // Epilogue Producer Warp End
    }  // Producer Warp Group End

    else if (is_consumer_warp_group(warp_group_role)) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      CollectiveEpilogue local_collective_epilogue(params.epilogue,
                                                   shared_storage.tensors.epilogue);

      bool do_store_tail = false;
      while (work_tile_info.is_valid()) {
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count =
            TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

        auto accumulators =
            partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));  // (MMA,MMA_M,MMA_N)
        bool valid_work = TileScheduler::valid_warpgroup_in_work_tile(work_tile_info);
        bool do_epilogue = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);
        bool do_lora = do_epilogue;
        if (valid_work) {
          collective_mainloop.mma(mainloop_pipeline, mainloop_pipe_consumer_state, accumulators,
                                  work_k_tile_count, mma_thread_idx,
                                  shared_storage.tensors.mainloop, params.mainloop, blk_coord);

          collective_mainloop.mma_tail(mainloop_pipeline, mainloop_pipe_consumer_state,
                                       work_k_tile_count);

          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }

        int consumer_warp_group_idx = cutlass::canonical_warp_group_idx() - NumLoadWarpGroups;

        // Perform reduction across splits, if needed
        TileScheduler::fixup(params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups,
                             consumer_warp_group_idx);

        // On hook-enabled configs, consume the LoRA
        // stage inside the epilogue store loop so the rank-32 tail folds under the store
        // schedule.
        // LoRA-once is preserved - only the epilogue owner produced the stage and only
        // it runs the hooked store. The pre-store fold below remains for the cases the
        // hook does not cover (big-tile configs that keep the builder epilogue,
        // reduction subtiles - unused by our schedulers - and the dedicated pipeline).
        bool lora_in_epilogue = false;
        if constexpr ((!UseDedicatedLoRaTma || UseRow80SideSlot) &&
                      flashinfer::gemm::svdquant_sm120::is_sm120_lora_hooked_epilogue_v<
                          CollectiveEpilogue>) {
          lora_in_epilogue =
              valid_work && do_lora && do_epilogue && work_tile_info.reduction_subtile_idx() == -1;
        }

        // Fold in the rank-32 LoRA-up exactly once per output tile: the accumulator is
        // fully reduced here, and only the epilogue owner produced (and consumes) the stage.
        if (valid_work && do_lora && !lora_in_epilogue) {
          if constexpr (UseDedicatedLoRaTma) {
            collective_mainloop.mma_lora(lora_pipeline, lora_pipe_consumer_state, accumulators,
                                         mma_thread_idx, shared_storage.tensors.mainloop);
            lora_pipe_consumer_state.advance(1);
          } else {
            collective_mainloop.mma_lora(mainloop_pipeline, mainloop_pipe_consumer_state,
                                         accumulators, mma_thread_idx,
                                         shared_storage.tensors.mainloop);
            mainloop_pipe_consumer_state.advance(1);
          }
        }

        if (do_epilogue) {
          auto run_plain_store = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
            auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
                local_collective_epilogue.store(
                    epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline,
                    epi_store_pipe_producer_state, problem_shape_MNKL, blk_shape, blk_coord,
                    accumulators, tiled_mma, mma_thread_idx, shared_storage.tensors.epilogue,
                    work_tile_info.reduction_subtile_idx());
            epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
            epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
            do_store_tail = true;
          };
          if constexpr (flashinfer::gemm::svdquant_sm120::is_sm120_lora_hooked_epilogue_v<
                            CollectiveEpilogue>) {
            if (lora_in_epilogue) {
              // Stage D/L1 into registers and hand the buffers back to the producer
              // BEFORE the epilogue starts; the hook then folds per subtile. The two
              // policies stage different fragment shapes (true-rank vs. padded overlay),
              // hence the immediately-invoked dispatch.
              auto lora_frags = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
                if constexpr (UseRow80SideSlot) {
                  auto frags = collective_mainloop.mma_lora_begin(
                      lora_pipeline, lora_pipe_consumer_state, mma_thread_idx,
                      shared_storage.tensors.mainloop);
                  lora_pipe_consumer_state.advance(1);
                  return frags;
                } else {
                  auto frags = collective_mainloop.mma_lora_begin(
                      mainloop_pipeline, mainloop_pipe_consumer_state, mma_thread_idx,
                      shared_storage.tensors.mainloop);
                  mainloop_pipe_consumer_state.advance(1);
                  return frags;
                }
              }();
              auto lora_hook = make_sm120_lora_subtile_hook(collective_mainloop, lora_frags);
              auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
                  local_collective_epilogue.store(
                      epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline,
                      epi_store_pipe_producer_state, problem_shape_MNKL, blk_shape, blk_coord,
                      accumulators, tiled_mma, mma_thread_idx, shared_storage.tensors.epilogue,
                      work_tile_info.reduction_subtile_idx(), lora_hook);
              epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
              epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
              do_store_tail = true;
            } else {
              run_plain_store();
            }
          } else {
            run_plain_store();
          }
        }

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
            work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
        work_tile_info = next_work_tile_info;
        if constexpr (IsSchedDynamicPersistent) {
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
          }
        }
      }  // Scheduler work fetch loop

      if (do_store_tail) {
        local_collective_epilogue.store_tail(epi_load_pipeline, epi_load_pipe_consumer_state,
                                             epi_store_pipeline, epi_store_pipe_producer_state);
      }
    }  // Consumer Warp Groups End
#endif
  }
};

}  // namespace flashinfer::gemm::svdquant_sm120
