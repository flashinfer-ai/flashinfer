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
// SVDQuant fused NVFP4 GEMM launcher templates (SM120): build the standard SM120 block-scaled
// NVFP4 mainloop via the CUTLASS CollectiveBuilder, re-instantiate the custom
// CollectiveMmaLoRaSm120 (a renamed copy of the SM120 block-scaled warp-specialized collective
// extended with the fused rank-32 LoRA-up) with the Builder's extracted typedefs, and run it
// via the LoRA-aware kernel driver GemmUniversalLoRaSm120. Tactics enumerate
// 4 CTA tiles x swap_ab {false,true} x {persistent, Stream-K}, all with a fixed 1x1x1 cluster.
// The stock SM120 fp4 GEMM template (fp4_gemm_template_sm120.h) is untouched.

#pragma once

#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "flashinfer/gemm/nvfp4_svdquant_gemm_collective_sm120.h"
#include "flashinfer/gemm/nvfp4_svdquant_gemm_epilogue_sm120.h"

namespace flashinfer {
namespace gemm {
namespace svdquant_sm120 {

using namespace cute;

using Arch = cutlass::arch::Sm120;
using ElementType = cutlass::float_e2m1_t;  // NVFP4 operands
using SFType = cutlass::float_ue4m3_t;
using OutElementType = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementC = void;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
static constexpr int AlignA = 32;
static constexpr int AlignB = 32;
static constexpr int AlignC = 8;

using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using ClusterShape = Shape<_1, _1, _1>;  // SM120: fixed, no multicast.
inline constexpr size_t kInlineDownWorkspaceBytes = 256;

// Row80 uses an independent LoRA side slot; all other configurations reuse a
// residual stage with byte-exact TMA transfers.
inline constexpr cutlass::gemm::collective::Sm120LoRaPath kBuildWideLoRaPathSm120 =
    cutlass::gemm::collective::Sm120LoRaPath::kByteExactMainloop;

// The LoRA rank this module was compiled for. One module serves one rank, the way
// the upstream cute-dsl path compiles a specialization per rank; a different rank
// is a separate JIT build under its own cache name (see gen_gemm_sm120_module_
// cutlass_nvfp4_svdquant's lora_rank). Rank 32 leaves the define unset so the
// production module keeps its historical identity byte for byte.
inline constexpr int kSvdqSm120ModuleLoRaRank =
#if defined(SVDQ_SM120_LORA_RANK)
    SVDQ_SM120_LORA_RANK;
#else
    32;
#endif
static_assert(kSvdqSm120ModuleLoRaRank > 0 && kSvdqSm120ModuleLoRaRank % 32 == 0,
              "SVDQ_SM120_LORA_RANK must be a positive multiple of 32");

// swap_ab=false: out[M,N] row-major, per-column bias[N] (matches SM100).
// swap_ab=true: the launcher computes out^T[N,M] = B@A^T + L1@D^T; the output is written
// column-major over the same row-major [M,N] buffer and bias[N] becomes a per-row vector of
// the swapped problem.
using FusionOperationPerCol =
    cutlass::epilogue::fusion::LinCombPerColBias<OutElementType, float, OutElementType, ElementC,
                                                 float>;
using FusionOperationPerRow =
    cutlass::epilogue::fusion::LinCombPerRowBias<OutElementType, float, OutElementType, ElementC,
                                                 float>;

template <class MmaTileShape_, bool SwapAB_, class TileSchedulerTag_>
struct SvdquantGemmConfigSm120 {
  using MmaTileShape = MmaTileShape_;
  static constexpr bool SwapAB = SwapAB_;
  using TileSchedulerTag = TileSchedulerTag_;

  // 64-row tiles run the driver's single-consumer-warpgroup contract. Keep the admitted
  // family closed: the proven 64x128 non-swap rows plus production row80.
  static constexpr bool SmallM = cute::size<0>(MmaTileShape{}) == 64;
  static constexpr bool IsProvenSmallM =
      !SwapAB && cute::size<1>(MmaTileShape{}) == 128 &&
      (cute::size<2>(MmaTileShape{}) == 128 || cute::size<2>(MmaTileShape{}) == 256) &&
      (cute::is_same_v<TileSchedulerTag, cutlass::gemm::PersistentScheduler> ||
       cute::is_same_v<TileSchedulerTag, cutlass::gemm::StaticPersistentScheduler>);
  static constexpr bool IsRow80 =
      SmallM && SwapAB && cute::size<1>(MmaTileShape{}) == 32 &&
      cute::size<2>(MmaTileShape{}) == 256 &&
      cute::is_same_v<TileSchedulerTag, cutlass::gemm::StaticPersistentScheduler>;
  static constexpr bool IsRow80SideSlot = IsRow80;
  static constexpr cutlass::gemm::collective::Sm120LoRaPath LoRaPathSm120 =
      IsRow80SideSlot ? cutlass::gemm::collective::Sm120LoRaPath::kDedicatedTma
                      : kBuildWideLoRaPathSm120;

  // LoRA rank this config can stage. The byte-exact path overlays the rank
  // tile on a residual stage, and one stage spans TileK/4 bf16
  // columns -- 32 for a K128 tile, 64 for K256, where columns 32..63 are TMA
  // zero-filled today precisely because the rank is only 32.
  //
  // A module built for a rank a config cannot stage still compiles that config
  // at rank 32; it is simply never selected, because the collective's own
  // can_implement rejects a mismatched args.lora_rank. Keeping the config
  // compiled rather than dropping it is what keeps kernel ids stable, since the
  // KernelShapeSm120 enum and SVDQ_SM120_DISPATCH are written out by hand and
  // would otherwise have to be regenerated per rank.
  static constexpr int kLoRaStageCapacity = cute::size<2>(MmaTileShape{}) / 4;
  static constexpr int LoRaRank =
      kSvdqSm120ModuleLoRaRank <= kLoRaStageCapacity ? kSvdqSm120ModuleLoRaRank : 32;
  static_assert(!IsRow80SideSlot || IsRow80, "the production side slot is row80-only");
  static_assert(IsRow80 ||
                    LoRaPathSm120 == cutlass::gemm::collective::Sm120LoRaPath::kByteExactMainloop,
                "non-row80 configs must retain the byte-exact LoRA path");
  static constexpr bool IsProductionRow80 =
      IsRow80 && (LoRaPathSm120 == cutlass::gemm::collective::Sm120LoRaPath::kByteExactMainloop ||
                  IsRow80SideSlot);
  static_assert(!IsRow80 || IsProductionRow80, "the row80 diagnostic lost its production traits");
  static_assert(!SmallM || IsProvenSmallM || IsRow80, "unsupported 64-row SM120 SVDQuant tile");

  using LayoutC =
      cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using FusionOperation = cute::conditional_t<SwapAB, FusionOperationPerRow, FusionOperationPerCol>;

  using CollectiveEpilogueBase = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape, EpilogueTileType,
      ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignC, OutElementType, LayoutC,
      AlignC, cutlass::epilogue::TmaWarpSpecialized, FusionOperation>::CollectiveOp;
  // Hook-enabled copy of the builder's epilogue so the rank-32 LoRA tail folds into
  // each epilogue subtile's accumulator MMA blocks inside the store loop. Storage,
  // params, and every non-store path are verbatim. RESTRICTED to tiles at or below
  // 128x128: the 256x128/128x256 TUs run 384 threads against a full 64K register
  // file (their ptxas spills are pinned exactly), and the fold's extra ~12-24
  // registers measurably deepened those spills - they keep the serial tail, whose
  // codegen is byte-identical to the pre-overlap build. Every gate-relevant overlap
  // win measured on the 128x128 rows (t1/t36/t37).
  static constexpr bool UseLoRaEpiOverlap =
      cute::size<0>(MmaTileShape{}) <= 128 && cute::size<1>(MmaTileShape{}) <= 128;
  using CollectiveEpilogue =
      cute::conditional_t<UseLoRaEpiOverlap,
                          typename MakeSm120LoRaOverlapEpilogue<CollectiveEpilogueBase>::Type,
                          CollectiveEpilogueBase>;

  // Shared-memory carveout the residual stage builder must not touch, on top of the
  // epilogue storage. Byte-exact policy: zero, D/L1 overlay the residual A/B stage buffers.
  // Dedicated policy: the true rank-32 bf16 D[M,32]/L1[N,32] staging buffers plus the
  // one-slot LoRA pipeline barriers, rounded to 128 like the builder's own per-stage
  // pipeline allowance. Keep this bound
  // tight: over-counting drops a residual stage (K256 tiles sit directly on the Stages >= 2
  // floor), under-counting over-allocates the smem capacity (checked by the static_assert
  // below).
  static constexpr int Row80SideSlotStorageK = cute::size<2>(MmaTileShape{}) *
                                               cutlass::sizeof_bits<ElementType>::value /
                                               cutlass::sizeof_bits<cutlass::bfloat16_t>::value;
  static constexpr int LoRaSmemCarveoutBytes =
      LoRaPathSm120 == cutlass::gemm::collective::Sm120LoRaPath::kDedicatedTma
          ? static_cast<int>((cute::size<0>(MmaTileShape{}) + cute::size<1>(MmaTileShape{})) *
                             (IsRow80SideSlot ? Row80SideSlotStorageK : 32) *
                             sizeof(cutlass::bfloat16_t)) +
                128
          : 0;

  // Build the standard SM120 block-scaled mainloop, then re-instantiate the LoRA collective
  // with the builder's extracted template arguments and this config's LoRA path policy.
  // The pinned cooperative builder cannot emit a 64-row tile: it hard-selects a 4x2
  // AtomLayout (256 threads) and its SFA smem atom extent collapses at TileM < 128. A
  // 64-row config therefore runs the builder at a 128-row donor shape (same N, K, and
  // carveout) purely to extract the dispatch policy (donor stage count included), operand
  // strides, TMA/smem/copy atoms, and the NVFP4 MMA atom; the 128-thread TiledMma is
  // hand-built below and the collective is re-instantiated at the real tile shape.
  using DonorTileShape =
      cute::conditional_t<SmallM,
                          decltype(cute::make_shape(_128{}, cute::shape<1>(MmaTileShape{}),
                                                    cute::shape<2>(MmaTileShape{}))),
                          MmaTileShape>;

  using AutoMainloopStageCount = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)) + LoRaSmemCarveoutBytes>;
  // NCU exposed pipeline-wait headroom in the three-stage row80 donor. Four
  // stages won the strict-CUPTI comparison without changing CTA residency and
  // use less shared memory than the statistically tied five-stage candidate.
  using MainloopStageCount =
      cute::conditional_t<IsProductionRow80, cutlass::gemm::collective::StageCount<4>,
                          AutoMainloopStageCount>;

  using CollectiveMainloopBase = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassBlockScaledTensorOp, cute::tuple<ElementType, SFType>, LayoutA,
      AlignA, cute::tuple<ElementType, SFType>, LayoutB, AlignB, ElementAccumulator, DonorTileShape,
      ClusterShape, MainloopStageCount,
      cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>::CollectiveOp;
  static_assert(!IsRow80SideSlot || (CollectiveMainloopBase::DispatchPolicy::Stages == 4 &&
                                     Row80SideSlotStorageK == 64),
                "the row80 side slot requires stage4 and a padded K64 slot");

  // Hand-built 64-row MMA: reuse the donor NVFP4 atom and N/K permutations, but run a
  // 2x2 atom layout (128 threads) and rebuild permutation M at the real 64-row extent.
  using DonorTiledMma = typename CollectiveMainloopBase::TiledMma;
  using DonorMmaTraits =
      cutlass::gemm::collective::svdquant_sm120_detail::TiledMmaTraits<DonorTiledMma>;
  using SmallMTiledMma = decltype(cute::make_tiled_mma(
      typename DonorTiledMma::Atom{}, Layout<Shape<_2, _2, _1>>{},
      cute::make_tile(_64{}, cute::get<1>(typename DonorMmaTraits::PermutationMNK{}),
                      cute::get<2>(typename DonorMmaTraits::PermutationMNK{}))));
  using TiledMma = cute::conditional_t<SmallM, SmallMTiledMma, DonorTiledMma>;
  static_assert(cute::size(TiledMma{}) == (SmallM ? 128 : 256),
                "the TiledMma thread count must match the driver's warpgroup contract");
  static_assert(cute::is_same_v<typename TiledMma::ValTypeC, ElementAccumulator>,
                "the residual MMA must accumulate in fp32");

  template <bool CompileInlineDown>
  using CollectiveMainloopT = cutlass::gemm::collective::CollectiveMmaLoRaSm120<
      typename CollectiveMainloopBase::DispatchPolicy, MmaTileShape,
      typename CollectiveMainloopBase::ElementPairA, typename CollectiveMainloopBase::StridePairA,
      typename CollectiveMainloopBase::ElementPairB, typename CollectiveMainloopBase::StridePairB,
      TiledMma, typename CollectiveMainloopBase::GmemTiledCopyPairA,
      typename CollectiveMainloopBase::SmemLayoutAtomsA,
      typename CollectiveMainloopBase::SmemCopyAtomsA, typename CollectiveMainloopBase::TransformA,
      typename CollectiveMainloopBase::GmemTiledCopyPairB,
      typename CollectiveMainloopBase::SmemLayoutAtomsB,
      typename CollectiveMainloopBase::SmemCopyAtomsB, typename CollectiveMainloopBase::TransformB,
      LoRaPathSm120, IsProductionRow80, CompileInlineDown, LoRaRank>;
  using CollectiveMainloop = CollectiveMainloopT<true>;
  using CollectiveMainloopNoInlineDown = CollectiveMainloopT<false>;

  // A 64-row CTA stores one 64x32 epilogue subtile per 32 output columns; the count reaches
  // the Stream-K scheduler sizing through get_store_pipe_increment.
  static_assert(!SmallM || (cute::size<0>(typename CollectiveEpilogue::EpilogueTile{}) == 64 &&
                            cute::size<1>(typename CollectiveEpilogue::EpilogueTile{}) == 32 &&
                            CollectiveEpilogue::get_store_pipe_increment(MmaTileShape{}) ==
                                cute::size<1>(MmaTileShape{}) / 32),
                "a 64-row tile must produce one 64x32 store subtile per 32 columns");

  using GemmKernel = GemmUniversalLoRaSm120<Shape<int, int, int, int>, CollectiveMainloop,
                                            CollectiveEpilogue, TileSchedulerTag>;
  using GemmKernelNoInlineDown =
      GemmUniversalLoRaSm120<Shape<int, int, int, int>, CollectiveMainloopNoInlineDown,
                             CollectiveEpilogue, TileSchedulerTag>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using GemmNoInlineDown = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelNoInlineDown>;

  // The carveout above must upper-bound the storage the LoRA path actually adds; a
  // violation means the builder selected residual stages against bytes that do not exist.
  static_assert(GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes,
                "SM120 SVDQuant kernel shared storage exceeds the SM120 smem capacity");
  static_assert(GemmKernelNoInlineDown::SharedStorageSize == GemmKernel::SharedStorageSize,
                "inline-down codegen selection must not change shared storage sizing");
};

using PersistentTag = cutlass::gemm::PersistentScheduler;
using StreamKTag = cutlass::gemm::StreamKScheduler;

// 16 legacy kernels: shape-major, then swap_ab, then scheduler. IDs are stable.
using Tactic128x128x128Config =
    SvdquantGemmConfigSm120<Shape<_128, _128, _128>, false, PersistentTag>;
using Tactic128x128x128SwapConfig =
    SvdquantGemmConfigSm120<Shape<_128, _128, _128>, true, PersistentTag>;
using Tactic128x128x128SkConfig =
    SvdquantGemmConfigSm120<Shape<_128, _128, _128>, false, StreamKTag>;
using Tactic128x128x128SwapSkConfig =
    SvdquantGemmConfigSm120<Shape<_128, _128, _128>, true, StreamKTag>;
using Tactic128x128x256Config =
    SvdquantGemmConfigSm120<Shape<_128, _128, _256>, false, PersistentTag>;
using Tactic128x128x256SwapConfig =
    SvdquantGemmConfigSm120<Shape<_128, _128, _256>, true, PersistentTag>;
using Tactic128x128x256SkConfig =
    SvdquantGemmConfigSm120<Shape<_128, _128, _256>, false, StreamKTag>;
using Tactic128x128x256SwapSkConfig =
    SvdquantGemmConfigSm120<Shape<_128, _128, _256>, true, StreamKTag>;
using Tactic256x128x128Config =
    SvdquantGemmConfigSm120<Shape<_256, _128, _128>, false, PersistentTag>;
using Tactic256x128x128SwapConfig =
    SvdquantGemmConfigSm120<Shape<_256, _128, _128>, true, PersistentTag>;
using Tactic256x128x128SkConfig =
    SvdquantGemmConfigSm120<Shape<_256, _128, _128>, false, StreamKTag>;
using Tactic256x128x128SwapSkConfig =
    SvdquantGemmConfigSm120<Shape<_256, _128, _128>, true, StreamKTag>;
using Tactic128x256x128Config =
    SvdquantGemmConfigSm120<Shape<_128, _256, _128>, false, PersistentTag>;
using Tactic128x256x128SwapConfig =
    SvdquantGemmConfigSm120<Shape<_128, _256, _128>, true, PersistentTag>;
using Tactic128x256x128SkConfig =
    SvdquantGemmConfigSm120<Shape<_128, _256, _128>, false, StreamKTag>;
using Tactic128x256x128SwapSkConfig =
    SvdquantGemmConfigSm120<Shape<_128, _256, _128>, true, StreamKTag>;

// Small-N swap tiles (kernel ids 16-19): the low-risk vehicle for small-M
// problems - swap_ab maps the problem M onto the CTA tile's N side, so a
// 64/32-column tile quadruples/octuples the CTA count of an M<=512 problem
// without touching the cooperative (128-row, 2-consumer-warpgroup) driver.
// The pinned CUTLASS builder supports TileShape_N >= 32 and pads the SFB smem
// layout to 128 itself.
using Tactic128x64x128SwapConfig =
    SvdquantGemmConfigSm120<Shape<_128, _64, _128>, true, PersistentTag>;
using Tactic128x64x128SwapSkConfig =
    SvdquantGemmConfigSm120<Shape<_128, _64, _128>, true, StreamKTag>;
using Tactic128x64x256SwapConfig =
    SvdquantGemmConfigSm120<Shape<_128, _64, _256>, true, PersistentTag>;
using Tactic128x32x128SwapConfig =
    SvdquantGemmConfigSm120<Shape<_128, _32, _128>, true, PersistentTag>;
// Kernel id 22: Stream-K sibling of the 128x64x256 swap tile, so the thin-tile
// K256 schedule (the small-M winner) can also split its K range natively.
using Tactic128x64x256SwapSkConfig =
    SvdquantGemmConfigSm120<Shape<_128, _64, _256>, true, StreamKTag>;

// Small-M tiles (kernel ids 20-21): 64-row CTAs remove the half-tile waste of M<=64
// problems on the non-swap side, running the driver's single-consumer-warpgroup
// contract (128 MMA threads, 256-thread block). Persistent only; a Stream-K sibling
// is admitted separately once this path is proven.
using Tactic64x128x128Config =
    SvdquantGemmConfigSm120<Shape<_64, _128, _128>, false, PersistentTag>;
using Tactic64x128x256Config =
    SvdquantGemmConfigSm120<Shape<_64, _128, _256>, false, PersistentTag>;

// Per-shape static-scheduler flavors (kernel ids 23-25): the same three tiles the
// measured one-wave winners run, compiled on the static tile scheduler
// (PersistentTileSchedulerSm90). One-wave problems gain from the divmod tile
// assignment (no scheduler-response pipeline, no CLC query warp); many-wave
// problems lose the dynamic load balancing, so the host side exposes these rows
// only inside the one-wave envelope of their own tile grid.
using Tactic128x64x256SwapStaticConfig =
    SvdquantGemmConfigSm120<Shape<_128, _64, _256>, true, cutlass::gemm::StaticPersistentScheduler>;
using Tactic128x32x128SwapStaticConfig =
    SvdquantGemmConfigSm120<Shape<_128, _32, _128>, true, cutlass::gemm::StaticPersistentScheduler>;
using Tactic64x128x128StaticConfig =
    SvdquantGemmConfigSm120<Shape<_64, _128, _128>, false,
                            cutlass::gemm::StaticPersistentScheduler>;

// Fill-geometry kernel (id 26): the 32-column swap tile with K256 steps -
// kernel 19's grid fill (48 CTAs at M=64 on 3072-column problems) combined
// with kernel 18's half K-step count. Exposed like the other thin swap
// tiles (M <= 512).
using Tactic128x32x256SwapConfig =
    SvdquantGemmConfigSm120<Shape<_128, _32, _256>, true, PersistentTag>;

// Kernel id 27: static-scheduler sibling of the fill-geometry kernel. The
// host exposes it only for one-wave grids; keeping the dynamic config as id 26
// preserves the existing row 78 and the multi-wave fallback.
using Tactic128x32x256SwapStaticConfig =
    SvdquantGemmConfigSm120<Shape<_128, _32, _256>, true, cutlass::gemm::StaticPersistentScheduler>;

// Kernel id 28: production row80 combines the proven 64-row single-consumer
// driver with the 32-column K256 donor. At M=64,N=3072 the swapped grid grows
// from 48 to 96 CTAs.
using Tactic64x32x256SwapStaticConfig =
    SvdquantGemmConfigSm120<Shape<_64, _32, _256>, true, cutlass::gemm::StaticPersistentScheduler>;

// Row 81: static K3 sibling of the legacy 256x128x128 swap kernel for the
// 105-tile M537,N5376 grid. It fits a 110-SM part in one wave, where the
// static scheduler avoids the persistent scheduler-response protocol.
// Profitability is restricted to the two measured K shapes by Python.
using Tactic256x128x128SwapStaticConfig =
    SvdquantGemmConfigSm120<Shape<_256, _128, _128>, true,
                            cutlass::gemm::StaticPersistentScheduler>;

// Row 82: the six-wave case-9 winner. Its 651 nearly full CTA slots make the
// static scheduler profitable despite the multi-wave grid, while the 64-column
// tile preserves enough parallelism for M1935,N5376.
using Tactic256x64x128SwapStaticConfig =
    SvdquantGemmConfigSm120<Shape<_256, _64, _128>, true, cutlass::gemm::StaticPersistentScheduler>;

inline constexpr int kNvfp4SvdquantGemmSm120NumLegacyKernels = 16;
inline constexpr int kNvfp4SvdquantGemmSm120NumKernels = 31;

enum class KernelShapeSm120 {
  k128x128x128,
  k128x128x128Swap,
  k128x128x128Sk,
  k128x128x128SwapSk,
  k128x128x256,
  k128x128x256Swap,
  k128x128x256Sk,
  k128x128x256SwapSk,
  k256x128x128,
  k256x128x128Swap,
  k256x128x128Sk,
  k256x128x128SwapSk,
  k128x256x128,
  k128x256x128Swap,
  k128x256x128Sk,
  k128x256x128SwapSk,
  k128x64x128Swap,
  k128x64x128SwapSk,
  k128x64x256Swap,
  k128x32x128Swap,
  k64x128x128,
  k64x128x256,
  k128x64x256SwapSk,  // appended after the driver ids so ids 16-21 keep their meaning
  // per-shape static-scheduler flavors of the one-wave winner tiles
  k128x64x256SwapStatic,
  k128x32x128SwapStatic,
  k64x128x128Static,
  k128x32x256Swap,  // fill-geometry tile: 48-CTA one-wave grids with K256 steps
  k128x32x256SwapStatic,
  k64x32x256SwapStatic,
  k256x128x128SwapStatic,
  k256x64x128SwapStatic,
};

// Legacy kernels follow the id%4 convention; the small-N and small-M additions are
// listed explicitly (the 128x64x128 and 128x64x256 swap tiles have Stream-K siblings).
constexpr bool kernel_is_streamk_sm120(int kernel_id) {
  return kernel_id < kNvfp4SvdquantGemmSm120NumLegacyKernels
             ? (kernel_id % 4) >= 2
             : (kernel_id == static_cast<int>(KernelShapeSm120::k128x64x128SwapSk) ||
                kernel_id == static_cast<int>(KernelShapeSm120::k128x64x256SwapSk));
}

// A runtime tactic is one row of the flattened tactic table: which compiled
// kernel to launch and which scheduler parameters to apply. Rows 0..15 are the
// compiled kernels with scheduler defaults and keep their historical ids and
// meaning; appended rows only add scheduler variants on the same kernels.
enum class RasterOrderSm120 : int { kHeuristic = 0, kAlongM = 1, kAlongN = 2 };

struct RuntimeTacticSm120 {
  int kernel_id;
  int splits;  // 1 = scheduler-default decomposition; >1 = forced Split-K
  RasterOrderSm120 raster;
  int swizzle;  // power of two; 1 = default
};

// 16 legacy kernel rows + 8 legacy Stream-K kernels x splits {2,4} + 8 legacy
// persistent kernels x {AlongM, AlongN, swizzle 2, swizzle 4} + the small-N
// swap kernels (a default row each, plus splits {2,4} for the Stream-K ones) +
// the small-M kernels (a default row each) + the per-shape static-scheduler
// kernels (a default row each) + the fill-geometry kernel and its static
// sibling, production row80, and the case-9 static row82 (a default row each).
inline constexpr int kNvfp4SvdquantGemmSm120NumTactics = 83;

namespace detail {
constexpr std::array<RuntimeTacticSm120, kNvfp4SvdquantGemmSm120NumTactics>
make_runtime_tactic_table() {
  constexpr int kLegacy = kNvfp4SvdquantGemmSm120NumLegacyKernels;
  std::array<RuntimeTacticSm120, kNvfp4SvdquantGemmSm120NumTactics> table{};
  for (int kid = 0; kid < kLegacy; ++kid) {
    table[kid] = {kid, 1, RasterOrderSm120::kHeuristic, 1};
  }
  int row = kLegacy;
  for (int kid = 0; kid < kLegacy; ++kid) {
    if (kernel_is_streamk_sm120(kid)) {
      table[row++] = {kid, 2, RasterOrderSm120::kHeuristic, 1};
      table[row++] = {kid, 4, RasterOrderSm120::kHeuristic, 1};
    }
  }
  for (int kid = 0; kid < kLegacy; ++kid) {
    if (!kernel_is_streamk_sm120(kid)) {
      table[row++] = {kid, 1, RasterOrderSm120::kAlongM, 1};
      table[row++] = {kid, 1, RasterOrderSm120::kAlongN, 1};
      table[row++] = {kid, 1, RasterOrderSm120::kHeuristic, 2};
      table[row++] = {kid, 1, RasterOrderSm120::kHeuristic, 4};
    }
  }
  for (int kid = kLegacy; kid < kNvfp4SvdquantGemmSm120NumKernels; ++kid) {
    table[row++] = {kid, 1, RasterOrderSm120::kHeuristic, 1};
    if (kernel_is_streamk_sm120(kid)) {
      table[row++] = {kid, 2, RasterOrderSm120::kHeuristic, 1};
      table[row++] = {kid, 4, RasterOrderSm120::kHeuristic, 1};
    }
  }
  return table;
}
}  // namespace detail

inline constexpr auto kRuntimeTacticTableSm120 = detail::make_runtime_tactic_table();
inline constexpr int kNoInlineDownRuntimeTactic = 9;

// The historical ids must never change meaning: row t launches kernel t with
// scheduler defaults, exactly as before the table existed.
static_assert(kRuntimeTacticTableSm120[0].kernel_id == 0 &&
                  kRuntimeTacticTableSm120[15].kernel_id == 15 &&
                  kRuntimeTacticTableSm120[15].splits == 1 &&
                  kRuntimeTacticTableSm120[15].swizzle == 1 &&
                  kRuntimeTacticTableSm120[15].raster == RasterOrderSm120::kHeuristic,
              "legacy tactic rows 0-15 must stay bitwise-equivalent to the compiled kernels");
static_assert(kRuntimeTacticTableSm120[kNoInlineDownRuntimeTactic].kernel_id ==
                      static_cast<int>(KernelShapeSm120::k256x128x128Swap) &&
                  kRuntimeTacticTableSm120[kNoInlineDownRuntimeTactic].splits == 1 &&
                  kRuntimeTacticTableSm120[kNoInlineDownRuntimeTactic].swizzle == 1 &&
                  kRuntimeTacticTableSm120[kNoInlineDownRuntimeTactic].raster ==
                      RasterOrderSm120::kHeuristic,
              "the no-inline-down specialization must remain the legacy tactic 9 row");
static_assert(kRuntimeTacticTableSm120[16].kernel_id == 2 &&
                  kRuntimeTacticTableSm120[16].splits == 2 &&
                  kRuntimeTacticTableSm120[31].kernel_id == 15 &&
                  kRuntimeTacticTableSm120[31].splits == 4,
              "Split-K rows must cover exactly the Stream-K kernels");
static_assert(kRuntimeTacticTableSm120[32].kernel_id == 0 &&
                  kRuntimeTacticTableSm120[32].raster == RasterOrderSm120::kAlongM &&
                  kRuntimeTacticTableSm120[63].kernel_id == 13 &&
                  kRuntimeTacticTableSm120[63].swizzle == 4,
              "raster/swizzle rows must cover exactly the persistent kernels");
static_assert(kRuntimeTacticTableSm120[64].kernel_id == 16 &&
                  kRuntimeTacticTableSm120[65].kernel_id == 17 &&
                  kRuntimeTacticTableSm120[66].kernel_id == 17 &&
                  kRuntimeTacticTableSm120[66].splits == 2 &&
                  kRuntimeTacticTableSm120[67].splits == 4 &&
                  kRuntimeTacticTableSm120[68].kernel_id == 18 &&
                  kRuntimeTacticTableSm120[69].kernel_id == 19,
              "small-N swap rows must sit at the table tail in kernel order");
static_assert(kRuntimeTacticTableSm120[70].kernel_id == 20 &&
                  kRuntimeTacticTableSm120[70].splits == 1 &&
                  kRuntimeTacticTableSm120[70].swizzle == 1 &&
                  kRuntimeTacticTableSm120[70].raster == RasterOrderSm120::kHeuristic &&
                  kRuntimeTacticTableSm120[71].kernel_id == 21 &&
                  kRuntimeTacticTableSm120[71].splits == 1 &&
                  kRuntimeTacticTableSm120[71].swizzle == 1 &&
                  kRuntimeTacticTableSm120[71].raster == RasterOrderSm120::kHeuristic,
              "small-M rows must follow the small-N rows as scheduler-default rows");
static_assert(kRuntimeTacticTableSm120[72].kernel_id == 22 &&
                  kRuntimeTacticTableSm120[72].splits == 1 &&
                  kRuntimeTacticTableSm120[73].kernel_id == 22 &&
                  kRuntimeTacticTableSm120[73].splits == 2 &&
                  kRuntimeTacticTableSm120[74].kernel_id == 22 &&
                  kRuntimeTacticTableSm120[74].splits == 4,
              "the K256-swap Stream-K rows are appended after the small-M rows");
static_assert(
    kRuntimeTacticTableSm120[75].kernel_id == 23 && kRuntimeTacticTableSm120[76].kernel_id == 24 &&
        kRuntimeTacticTableSm120[77].kernel_id == 25 && kRuntimeTacticTableSm120[75].splits == 1 &&
        kRuntimeTacticTableSm120[76].splits == 1 && kRuntimeTacticTableSm120[77].splits == 1 &&
        kRuntimeTacticTableSm120[75].swizzle == 1 && kRuntimeTacticTableSm120[76].swizzle == 1 &&
        kRuntimeTacticTableSm120[77].swizzle == 1 &&
        kRuntimeTacticTableSm120[75].raster == RasterOrderSm120::kHeuristic &&
        kRuntimeTacticTableSm120[76].raster == RasterOrderSm120::kHeuristic &&
        kRuntimeTacticTableSm120[77].raster == RasterOrderSm120::kHeuristic,
    "the static-scheduler rows close the table as scheduler-default rows");
static_assert(kRuntimeTacticTableSm120[78].kernel_id == 26 &&
                  kRuntimeTacticTableSm120[78].splits == 1 &&
                  kRuntimeTacticTableSm120[78].swizzle == 1 &&
                  kRuntimeTacticTableSm120[78].raster == RasterOrderSm120::kHeuristic,
              "the fill-geometry row follows as a scheduler-default row");
static_assert(kRuntimeTacticTableSm120[79].kernel_id == 27 &&
                  kRuntimeTacticTableSm120[79].splits == 1 &&
                  kRuntimeTacticTableSm120[79].swizzle == 1 &&
                  kRuntimeTacticTableSm120[79].raster == RasterOrderSm120::kHeuristic,
              "the static fill-geometry sibling is append-only");
static_assert(kRuntimeTacticTableSm120[80].kernel_id == 28 &&
                  kRuntimeTacticTableSm120[80].splits == 1 &&
                  kRuntimeTacticTableSm120[80].swizzle == 1 &&
                  kRuntimeTacticTableSm120[80].raster == RasterOrderSm120::kHeuristic,
              "the production row80 tactic is append-only");
static_assert(kRuntimeTacticTableSm120[81].kernel_id == 29 &&
                  kRuntimeTacticTableSm120[81].splits == 1 &&
                  kRuntimeTacticTableSm120[81].swizzle == 1 &&
                  kRuntimeTacticTableSm120[81].raster == RasterOrderSm120::kHeuristic,
              "the row44 static-scheduler tactic is append-only");
static_assert(kRuntimeTacticTableSm120[82].kernel_id == 30 &&
                  kRuntimeTacticTableSm120[82].splits == 1 &&
                  kRuntimeTacticTableSm120[82].swizzle == 1 &&
                  kRuntimeTacticTableSm120[82].raster == RasterOrderSm120::kHeuristic,
              "the case-9 static-scheduler tactic is append-only");
// The single decode point for every host entry (launch, workspace sizing,
// feasibility): the launcher dispatches on the decoded kernel_id and passes
// the row down so all three paths see identical scheduler parameters.
inline RuntimeTacticSm120 decode_runtime_tactic(int tactic) {
  if (tactic < 0 || tactic >= kNvfp4SvdquantGemmSm120NumTactics) {
    throw std::invalid_argument("nvfp4_svdquant_gemm (sm120): invalid tactic");
  }
  return kRuntimeTacticTableSm120[tactic];
}

// Shape-dependent legality of a row: forced Split-K may not exceed the K-tile
// count (the CUTLASS scheduler validates split syntax but not this bound).
template <class Config>
inline bool runtime_tactic_feasible(RuntimeTacticSm120 const& rt, int k) {
  if (rt.swizzle < 1 || (rt.swizzle & (rt.swizzle - 1)) != 0) return false;
  if (rt.splits > 1) {
    int const cta_k = cute::size<2>(typename Config::MmaTileShape{});
    int const k_tiles = (k + cta_k - 1) / cta_k;
    if (rt.splits > k_tiles) return false;
  }
  return true;
}

// The single scheduler-argument applicator shared by run_tactic,
// workspace_size_for_tactic, and can_implement_tactic. The Stream-K forced-
// splits env hook is applied last so the existing fault-injection tests keep
// overriding whatever the row selected.
template <class Config, class SchedulerArgs>
inline void apply_runtime_scheduler_args(RuntimeTacticSm120 const& rt,
                                         SchedulerArgs& scheduler_args) {
  if constexpr (!std::is_const_v<decltype(scheduler_args.max_swizzle_size)>) {
    scheduler_args.max_swizzle_size = rt.swizzle;
  }
  if constexpr (!std::is_const_v<decltype(scheduler_args.raster_order)>) {
    using Enum_t = decltype(scheduler_args.raster_order);
    scheduler_args.raster_order = rt.raster == RasterOrderSm120::kAlongM   ? Enum_t::AlongM
                                  : rt.raster == RasterOrderSm120::kAlongN ? Enum_t::AlongN
                                                                           : Enum_t::Heuristic;
  }
  if constexpr (cute::is_same_v<typename Config::TileSchedulerTag, StreamKTag>) {
    if (rt.splits > 1) {
      scheduler_args.splits = rt.splits;
      scheduler_args.decomposition_mode = decltype(scheduler_args.decomposition_mode)::SplitK;
    }
  }
}

template <class Config>
size_t shared_storage_size_for_tactic() {
  return static_cast<size_t>(Config::GemmKernel::SharedStorageSize);
}

template <class Config>
size_t workspace_size_for_tactic(RuntimeTacticSm120 const& rt, int m, int n, int k) {
  using Gemm = typename Config::Gemm;
  Gemm gemm;
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  if constexpr (Config::SwapAB) {
    args.problem_shape = cute::make_shape(n, m, k, 1);
  } else {
    args.problem_shape = cute::make_shape(m, n, k, 1);
  }
  apply_runtime_scheduler_args<Config>(rt, args.scheduler);
  return kInlineDownWorkspaceBytes + gemm.get_workspace_size(args);
}

template <class Config>
bool can_implement_tactic(RuntimeTacticSm120 const& rt, int m, int n, int k, int lora_rank) {
  if (!runtime_tactic_feasible<Config>(rt, k)) return false;
  using Gemm = typename Config::Gemm;
  using Sm1xxBlkScaledConfig = typename Config::CollectiveMainloop::Sm1xxBlkScaledConfig;
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  if constexpr (Config::SwapAB) {
    args.problem_shape = cute::make_shape(n, m, k, 1);
  } else {
    args.problem_shape = cute::make_shape(m, n, k, 1);
  }
  args.mainloop.dA = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideA>(k, 0);
  args.mainloop.dB = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideB>(k, 0);
  args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
  args.mainloop.lora_rank = lora_rank;
  // Pointer non-null checks are part of can_implement; use sentinels for the dry-run query.
  args.mainloop.ptr_D = reinterpret_cast<cutlass::bfloat16_t const*>(size_t(256));
  args.mainloop.ptr_L1 = reinterpret_cast<cutlass::bfloat16_t const*>(size_t(256));
  int64_t const r = lora_rank;
  args.mainloop.dD = cute::make_stride(r, cute::_1{}, int64_t(0));
  args.mainloop.dL1 = cute::make_stride(r, cute::_1{}, int64_t(0));
  args.epilogue.dC = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideC>(n, 0);
  args.epilogue.dD = args.epilogue.dC;
  apply_runtime_scheduler_args<Config>(rt, args.scheduler);
  typename Config::Gemm gemm;
  return gemm.can_implement(args) == cutlass::Status::kSuccess;
}

template <class Config, bool CompileInlineDown>
void run_tactic_impl(RuntimeTacticSm120 const& rt, void* out, void const* A, void const* B,
                     void const* sfa, void const* sfb, float const* alpha, void const* D,
                     void const* L1, void const* bias, void const* inline_down_x,
                     void const* inline_down_l2t, int m, int n, int k, int lora_rank, char* ws,
                     size_t wsBytes, cudaStream_t stream, bool enable_pdl) {
  using Gemm = std::conditional_t<CompileInlineDown, typename Config::Gemm,
                                  typename Config::GemmNoInlineDown>;
  using Sm1xxBlkScaledConfig = typename Config::CollectiveMainloop::Sm1xxBlkScaledConfig;
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;

  // This config stages Config::LoRaRank; a launcher that hands it anything else
  // would read the D/L1 tiles at the wrong extent rather than fail, so refuse.
  if (lora_rank != Config::LoRaRank)
    throw std::invalid_argument("nvfp4_svdquant_gemm (sm120): this tactic stages LoRA rank " +
                                std::to_string(Config::LoRaRank) + ", got " +
                                std::to_string(lora_rank));
  int64_t const r = lora_rank;

  if constexpr (Config::SwapAB) {
    // out^T[N,M] = alpha*(B@A^T) + L1@D^T [+ bias per swapped row]: swap the operand roles
    // and the problem M/N; the column-major epilogue writes the same row-major [M,N] buffer.
    args.problem_shape = cute::make_shape(n, m, k, 1);
    args.mainloop.ptr_A = static_cast<ElementType const*>(B);
    args.mainloop.ptr_B = static_cast<ElementType const*>(A);
    args.mainloop.ptr_SFA = static_cast<SFType const*>(sfb);
    args.mainloop.ptr_SFB = static_cast<SFType const*>(sfa);
    args.mainloop.ptr_D = static_cast<cutlass::bfloat16_t const*>(L1);
    args.mainloop.dD = cute::make_stride(r, cute::_1{}, int64_t(n) * r);
    args.mainloop.ptr_L1 = static_cast<cutlass::bfloat16_t const*>(D);
    args.mainloop.dL1 = cute::make_stride(r, cute::_1{}, int64_t(m) * r);
  } else {
    args.problem_shape = cute::make_shape(m, n, k, 1);
    args.mainloop.ptr_A = static_cast<ElementType const*>(A);
    args.mainloop.ptr_B = static_cast<ElementType const*>(B);
    args.mainloop.ptr_SFA = static_cast<SFType const*>(sfa);
    args.mainloop.ptr_SFB = static_cast<SFType const*>(sfb);
    args.mainloop.ptr_D = static_cast<cutlass::bfloat16_t const*>(D);
    args.mainloop.dD = cute::make_stride(r, cute::_1{}, int64_t(m) * r);
    args.mainloop.ptr_L1 = static_cast<cutlass::bfloat16_t const*>(L1);
    args.mainloop.dL1 = cute::make_stride(r, cute::_1{}, int64_t(n) * r);
  }
  args.mainloop.dA = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideA>(k, 0);
  args.mainloop.dB = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideB>(k, 0);
  args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
  args.mainloop.lora_rank = lora_rank;
  bool const use_inline_down = inline_down_x != nullptr && inline_down_l2t != nullptr;
  if constexpr (!CompileInlineDown) {
    if (use_inline_down)
      throw std::invalid_argument("nvfp4_svdquant_gemm (sm120): inline LoRA-down is compiled out");
  }
  if (use_inline_down && (m != 537 || n != 5376 || k != 5120 || !Config::SwapAB))
    throw std::invalid_argument(
        "nvfp4_svdquant_gemm (sm120): inline LoRA-down requires exact "
        "M537/N5376/K5120 swap configuration");
  if (use_inline_down) {
    args.mainloop.ptr_inline_down_x = static_cast<cutlass::bfloat16_t const*>(inline_down_x);
    args.mainloop.ptr_inline_down_l2t = static_cast<cutlass::bfloat16_t const*>(inline_down_l2t);
    args.mainloop.ptr_inline_down_output = static_cast<cutlass::bfloat16_t*>(const_cast<void*>(D));
    args.mainloop.ptr_inline_down_counters = reinterpret_cast<uint32_t*>(ws);
    args.mainloop.inline_down_m = m;
    args.mainloop.inline_down_k = k;
    args.mainloop.inline_down_wait_on_n = Config::SwapAB;
  }
  if constexpr (Config::CollectiveMainloop::UseRow80BiasPrefetch)
    args.mainloop.ptr_bias = static_cast<OutElementType const*>(bias);

  args.epilogue.ptr_C = nullptr;
  args.epilogue.ptr_D = static_cast<OutElementType*>(out);
  args.epilogue.dC = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideC>(n, 0);
  args.epilogue.dD = args.epilogue.dC;
  // out = alpha * acc + bias (alpha via device scalar pointer). A null bias is a no-op.
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.thread.alpha_ptr = alpha;
  args.epilogue.thread.beta_ptr = nullptr;
  args.epilogue.thread.bias_ptr = static_cast<OutElementType const*>(bias);

  if (!runtime_tactic_feasible<Config>(rt, k))
    throw std::invalid_argument("nvfp4_svdquant_gemm (sm120): tactic infeasible for this shape");
  apply_runtime_scheduler_args<Config>(rt, args.scheduler);

  Gemm gemm;
  size_t const gemmWorkspaceBytes = gemm.get_workspace_size(args);
  size_t const requiredWorkspaceBytes = kInlineDownWorkspaceBytes + gemmWorkspaceBytes;
  if (wsBytes < requiredWorkspaceBytes)
    throw std::invalid_argument("nvfp4_svdquant_gemm (sm120): insufficient workspace");
  if (use_inline_down) {
    auto const clear_status = cudaMemsetAsync(ws, 0, kInlineDownWorkspaceBytes, stream);
    if (clear_status != cudaSuccess)
      throw std::runtime_error("nvfp4_svdquant_gemm (sm120): inline-down counter clear failed");
  }
  char* const gemmWorkspace = ws + kInlineDownWorkspaceBytes;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess)
    throw std::runtime_error("nvfp4_svdquant_gemm (sm120): can_implement failed");
  st = gemm.initialize(args, gemmWorkspace, stream);
  if (st != cutlass::Status::kSuccess)
    throw std::runtime_error("nvfp4_svdquant_gemm (sm120): initialize failed");
  st = gemm.run(args, gemmWorkspace, stream, nullptr, enable_pdl);
  if (st != cutlass::Status::kSuccess)
    throw std::runtime_error("nvfp4_svdquant_gemm (sm120): run failed");
}

template <class Config>
void run_tactic(RuntimeTacticSm120 const& rt, void* out, void const* A, void const* B,
                void const* sfa, void const* sfb, float const* alpha, void const* D, void const* L1,
                void const* bias, void const* inline_down_x, void const* inline_down_l2t, int m,
                int n, int k, int lora_rank, char* ws, size_t wsBytes, cudaStream_t stream,
                bool enable_pdl) {
  run_tactic_impl<Config, true>(rt, out, A, B, sfa, sfb, alpha, D, L1, bias, inline_down_x,
                                inline_down_l2t, m, n, k, lora_rank, ws, wsBytes, stream,
                                enable_pdl);
}

template <class Config>
void run_tactic_no_inline_down(RuntimeTacticSm120 const& rt, void* out, void const* A,
                               void const* B, void const* sfa, void const* sfb, float const* alpha,
                               void const* D, void const* L1, void const* bias, int m, int n, int k,
                               int lora_rank, char* ws, size_t wsBytes, cudaStream_t stream,
                               bool enable_pdl) {
  run_tactic_impl<Config, false>(rt, out, A, B, sfa, sfb, alpha, D, L1, bias, nullptr, nullptr, m,
                                 n, k, lora_rank, ws, wsBytes, stream, enable_pdl);
}

// ---------------------------------------------------------------------------
// Per-config selection of the inline-LoRA-down compile path.
//
// 711d72e added an optional CTA-level inline LoRA-down routine to the shared
// collective. No production caller supplies its pointers, so the routine never
// executes, but instantiating it still changes ptxas allocation for the whole
// kernel. Per-thread stack frame measured with `cuobjdump -res-usage` on the
// campaign JIT module, path compiled in -> compiled out:
//
//   256x128x128 persistent  136 B ->   0 B   (kernel ids 8 and 9)
//   128x256x128 persistent  136 B ->  80 B, 144 B -> 88 B
//   128x128x*, 128x64x*, 128x32x* persistent and Stream-K   8-16 B -> 0 B
//   256x128x128 static        8 B -> 128 B   REGRESSES
//   256x128x128 Stream-K     40 B ->  64 B / 72 B   REGRESSES
//   128x256x128 Stream-K     40 B -> 184 B / 192 B  REGRESSES
//
// The choice is therefore per config, not global. Configs that regress keep the
// path compiled in, as does Tactic256x128x128SwapStaticConfig: it is the only
// config nvfp4_svdquant_gemm_run_inline_down (tactic 81) can dispatch to, and it
// is the one caller that really populates the inline-down pointers.
//
// The list drives three things that must stay in lockstep: the selector below,
// the extern declarations, and the per-tactic explicit instantiations emitted by
// csrc/nvfp4_svdquant_gemm_cutlass_sm120.jinja.
#define SVDQ_SM120_NO_INLINE_DOWN_CONFIG_LIST(X) \
  X(Tactic128x128x128Config)                     \
  X(Tactic128x128x128SwapConfig)                 \
  X(Tactic128x128x128SkConfig)                   \
  X(Tactic128x128x128SwapSkConfig)               \
  X(Tactic128x128x256Config)                     \
  X(Tactic128x128x256SwapConfig)                 \
  X(Tactic128x128x256SkConfig)                   \
  X(Tactic128x128x256SwapSkConfig)               \
  X(Tactic256x128x128Config)                     \
  X(Tactic256x128x128SwapConfig)                 \
  X(Tactic128x256x128Config)                     \
  X(Tactic128x256x128SwapConfig)                 \
  X(Tactic128x64x128SwapConfig)                  \
  X(Tactic128x64x128SwapSkConfig)                \
  X(Tactic128x64x256SwapConfig)                  \
  X(Tactic128x64x256SwapSkConfig)                \
  X(Tactic128x32x128SwapConfig)                  \
  X(Tactic128x32x256SwapConfig)

template <class Config>
inline constexpr bool kCompileInlineDownSm120 = true;

#define SVDQ_SM120_DECLARE_NO_INLINE_DOWN(Config) \
  template <>                                     \
  inline constexpr bool kCompileInlineDownSm120<Config> = false;
SVDQ_SM120_NO_INLINE_DOWN_CONFIG_LIST(SVDQ_SM120_DECLARE_NO_INLINE_DOWN)
#undef SVDQ_SM120_DECLARE_NO_INLINE_DOWN

#define INSTANTIATE_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Config)                                      \
  template size_t shared_storage_size_for_tactic<Config>();                                       \
  template size_t workspace_size_for_tactic<Config>(RuntimeTacticSm120 const& rt, int m, int n,   \
                                                    int k);                                       \
  template bool can_implement_tactic<Config>(RuntimeTacticSm120 const& rt, int m, int n, int k,   \
                                             int lora_rank);                                      \
  template void run_tactic<Config>(                                                               \
      RuntimeTacticSm120 const& rt, void* out, void const* A, void const* B, void const* sfa,     \
      void const* sfb, float const* alpha, void const* D, void const* L1, void const* bias,       \
      void const* inline_down_x, void const* inline_down_l2t, int m, int n, int k, int lora_rank, \
      char* ws, size_t wsBytes, cudaStream_t stream, bool enable_pdl);

#define INSTANTIATE_NVFP4_SVDQUANT_GEMM_SM120_NO_INLINE_DOWN_TACTIC(Config)                        \
  template void run_tactic_no_inline_down<Config>(                                                 \
      RuntimeTacticSm120 const& rt, void* out, void const* A, void const* B, void const* sfa,      \
      void const* sfb, float const* alpha, void const* D, void const* L1, void const* bias, int m, \
      int n, int k, int lora_rank, char* ws, size_t wsBytes, cudaStream_t stream,                  \
      bool enable_pdl);

// The per-shape static-scheduler configs alias their base configs under the global
// static-scheduler build; their translation units must then define nothing (the
// base config's TU already holds the one definition per specialization).
#define INSTANTIATE_NVFP4_SVDQUANT_GEMM_SM120_STATIC_TACTIC(Config) \
  INSTANTIATE_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Config)

#define EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Config)                                           \
  extern template size_t shared_storage_size_for_tactic<Config>();                                \
  extern template size_t workspace_size_for_tactic<Config>(RuntimeTacticSm120 const& rt, int m,   \
                                                           int n, int k);                         \
  extern template bool can_implement_tactic<Config>(RuntimeTacticSm120 const& rt, int m, int n,   \
                                                    int k, int lora_rank);                        \
  extern template void run_tactic<Config>(                                                        \
      RuntimeTacticSm120 const& rt, void* out, void const* A, void const* B, void const* sfa,     \
      void const* sfb, float const* alpha, void const* D, void const* L1, void const* bias,       \
      void const* inline_down_x, void const* inline_down_l2t, int m, int n, int k, int lora_rank, \
      char* ws, size_t wsBytes, cudaStream_t stream, bool enable_pdl);

#define EXTERN_NVFP4_SVDQUANT_GEMM_SM120_NO_INLINE_DOWN_TACTIC(Config)                             \
  extern template void run_tactic_no_inline_down<Config>(                                          \
      RuntimeTacticSm120 const& rt, void* out, void const* A, void const* B, void const* sfa,      \
      void const* sfb, float const* alpha, void const* D, void const* L1, void const* bias, int m, \
      int n, int k, int lora_rank, char* ws, size_t wsBytes, cudaStream_t stream,                  \
      bool enable_pdl);
SVDQ_SM120_NO_INLINE_DOWN_CONFIG_LIST(EXTERN_NVFP4_SVDQUANT_GEMM_SM120_NO_INLINE_DOWN_TACTIC)

// The per-tactic kernels are explicitly instantiated in Jinja-generated translation units
// (see csrc/nvfp4_svdquant_gemm_cutlass_sm120.jinja) so they compile in parallel.
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x128x128Config)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x128x128SwapConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x128x128SkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x128x128SwapSkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x128x256Config)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x128x256SwapConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x128x256SkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x128x256SwapSkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic256x128x128Config)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic256x128x128SwapConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic256x128x128SkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic256x128x128SwapSkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x256x128Config)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x256x128SwapConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x256x128SkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x256x128SwapSkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x64x128SwapConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x64x128SwapSkConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x64x256SwapConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x32x128SwapConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic64x128x128Config)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic64x128x256Config)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x64x256SwapSkConfig)
// Under the global static-scheduler build these names alias configs already
// declared above; re-declaring the same specializations is redundant.
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x64x256SwapStaticConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x32x128SwapStaticConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic64x128x128StaticConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x32x256SwapStaticConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic256x128x128SwapStaticConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic128x32x256SwapConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic64x32x256SwapStaticConfig)
EXTERN_NVFP4_SVDQUANT_GEMM_SM120_TACTIC(Tactic256x64x128SwapStaticConfig)

}  // namespace svdquant_sm120
}  // namespace gemm
}  // namespace flashinfer
