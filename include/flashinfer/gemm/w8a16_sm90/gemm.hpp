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
#ifndef FLASHINFER_GEMM_W8A16_SM90_GEMM_HPP_
#define FLASHINFER_GEMM_W8A16_SM90_GEMM_HPP_

// The Hopper W8A16 kernel variants: one CUTLASS GemmUniversal per (tile, decomposition),
// built on the register-sourced mixed-input mainloop
// (sm90_mma_tma_gmma_rs_warpspecialized_mixed_input).
//
//   out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
//
// TMA lands the fp8 weight tile and the 16-bit activation tile in smem, each consumer
// warpgroup pulls its 64 weight rows into registers, converts them (w8a16_convert.hpp) and
// issues wgmma with A from registers and B from smem.  The operand mapping, the fused
// epilogue and the host-side lifecycle (arguments, workspace, run) are in w8a16_host.hpp.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "flashinfer/gemm/w8a16_common.hpp"
#include "flashinfer/gemm/w8a16_convert.hpp"
#include "flashinfer/gemm/w8a16_host.hpp"
#include "flashinfer/gemm/w8a16_sm90/scheduler.hpp"
#include "flashinfer/gemm/w8a16_sm90/tile_order.hpp"

namespace flashinfer::gemm::constants::W8A16Sm90 {

// CTA tile dimensions: W = weight rows, A = activation rows, K = reduction depth.
static constexpr int TileK = 64;
static constexpr int CooperativeW = 128;
static constexpr int PingpongW = 64;
static constexpr int TileAMultiple = 8;
static constexpr int TileA128 = 128;
static constexpr int TileA160 = 160;
static constexpr int TileA192 = 192;
static constexpr int TileA256 = 256;
static constexpr int MaxTileA = TileA256;
static constexpr int DefaultCluster = 1;
static constexpr int MulticastCluster = 2;

// Split-K limits: equal slices of at least MinKTilesPerSplit reduction tiles.
static constexpr int MaxSplits = 8;
static constexpr int MinKTilesPerSplit = 8;
static constexpr long long MaxPartialsBytes = 64LL << 20;

}  // namespace flashinfer::gemm::constants::W8A16Sm90

namespace flashinfer::gemm {

namespace W8A16Sm90 = constants::W8A16Sm90;

// TileW: weight rows per CTA tile (M'): 128 for the cooperative schedule (two consumer
//         warpgroups, 64 rows each), 64 for ping-pong (one warpgroup per tile).
// TileA: activation rows per CTA tile (N' = wgmma N), any multiple of 8 up to 256.
// TileK: 64 fp8 bytes of weight per row per stage; with the 16-bit activation that is
//         8 KB + TileA * 128 B per stage (40 KB for 256 rows: 5 stages in 227 KB).
// Cluster: CTAs per cluster along the weight-row dimension.  2 = the two CTAs take
//         adjacent weight tiles of the same activation tile, which TMA multicasts to both:
//         halves the activation's L2 -> SMEM traffic.
template <int TileW, int TileA, int TileK = W8A16Sm90::TileK,
          int Cluster = W8A16Sm90::DefaultCluster>
struct W8A16Sm90Cfg {
  static_assert(TileW == W8A16Sm90::CooperativeW || TileW == W8A16Sm90::PingpongW,
                "128 rows = cooperative, 64 rows = ping-pong");
  static_assert(TileA % W8A16Sm90::TileAMultiple == 0 && TileA <= W8A16Sm90::MaxTileA,
                "wgmma N is a multiple of 8 up to 256");
  static_assert(Cluster == W8A16Sm90::DefaultCluster || Cluster == W8A16Sm90::MulticastCluster,
                "cluster of 1 or 2 CTAs");
  static constexpr int W = TileW;
  static constexpr int A = TileA;
  static constexpr int K = TileK;
  static constexpr bool Cooperative = TileW == W8A16Sm90::CooperativeW;
  using TileShape = cute::Shape<cute::Int<TileW>, cute::Int<TileA>, cute::Int<TileK>>;
  using ClusterShape = cute::Shape<cute::Int<Cluster>, cute::_1, cute::_1>;
  using MainloopSchedule =
      std::conditional_t<Cooperative, cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                         cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
  using EpilogueSchedule =
      std::conditional_t<Cooperative, cutlass::epilogue::TmaWarpSpecializedCooperative,
                         cutlass::epilogue::TmaWarpSpecialized>;
};

// The instantiated tiles: cooperative 128-row weight tiles at four activation heights (the
// dispatcher picks the height whose tile count best fills the SMs), and two ping-pong tiles
// (two tiles in flight per SM) for short mainloops.  All run at cluster size 1.
using W8A16Sm90Cfg128x256 = W8A16Sm90Cfg<W8A16Sm90::CooperativeW, W8A16Sm90::TileA256>;
using W8A16Sm90Cfg128x192 = W8A16Sm90Cfg<W8A16Sm90::CooperativeW, W8A16Sm90::TileA192>;
using W8A16Sm90Cfg128x160 = W8A16Sm90Cfg<W8A16Sm90::CooperativeW, W8A16Sm90::TileA160>;
using W8A16Sm90Cfg128x128 = W8A16Sm90Cfg<W8A16Sm90::CooperativeW, W8A16Sm90::TileA128>;
using W8A16Sm90Cfg64x256 = W8A16Sm90Cfg<W8A16Sm90::PingpongW, W8A16Sm90::TileA256>;
using W8A16Sm90Cfg64x128 = W8A16Sm90Cfg<W8A16Sm90::PingpongW, W8A16Sm90::TileA128>;

// One kernel variant: a tile configuration under one decomposition (W8A16Decomposition).
// Split-K runs the GEMM as a batch of `splits` K-slices in one launch (batch stride = the
// slice length along K for both operands, one M x N fp32 partial per batch), so an
// under-filled grid becomes `splits` times as many tiles; W8A16SplitKReduceKernel then sums
// the partials and applies scale and bias once.  No CUTLASS stream-K fixup is involved.
template <typename T, typename Cfg, W8A16Decomposition Decomp = W8A16Decomposition::Plain>
struct W8A16Sm90Gemm {
  static constexpr bool IsSplitK = Decomp == W8A16Decomposition::SplitK;
  static constexpr bool IsStreamK = Decomp == W8A16Decomposition::StreamK;
  static constexpr int TileK = Cfg::K;
  using ActType = T;
  using ElementW = cutlass::float_e4m3_t;
  using ElementAct = typename W8A16CutlassType<T>::type;
  using ElementD = std::conditional_t<IsSplitK, float, ElementAct>;
  using ElementBias = ElementAct;
  using ElementAccumulator = float;
  using ElementCompute = float;

  // A = weight [N, K], K contiguous -> RowMajor (K-major, required for the transformed operand).
  using LayoutA = cutlass::layout::RowMajor;
  // B = activation [M, K], K contiguous -> as B[N', K] that is ColumnMajor (K-major).
  using LayoutB = cutlass::layout::ColumnMajor;
  // D [M', N'] column-major == out[M, N] row-major.
  using LayoutD = cutlass::layout::ColumnMajor;
  // Elements per 16-byte access: 16 fp8 weights, 8 16-bit activations, 8 16-bit outputs or
  // 4 fp32 split-K partials.
  static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementW>::value;
  static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementAct>::value;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // D = alpha * acc + bias[row]; alpha is the per-tensor weight scale, bias is per M' (= N).
  // Split-K partials: D = acc (fp32); scale and bias are applied in the reduce.
  using FusionOp = std::conditional_t<
      IsSplitK, cutlass::epilogue::fusion::LinearCombination<float, float, void, float>,
      cutlass::epilogue::fusion::LinCombPerRowBias<ElementD, ElementCompute, ElementBias, ElementD,
                                                   float, AlignD>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, typename Cfg::TileShape,
      typename Cfg::ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute, void, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      typename Cfg::EpilogueSchedule, FusionOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cute::tuple<ElementW>, LayoutA, AlignA,
      ElementAct, LayoutB, AlignB, ElementAccumulator, typename Cfg::TileShape,
      typename Cfg::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename Cfg::MainloopSchedule>::CollectiveOp;

  // Plain persistent variants (cluster 1) run the grouped scheduler (scheduler.hpp); at
  // group 1 it is CUTLASS's own order.
  static constexpr bool IsGrouped = !IsStreamK && cute::size(typename Cfg::ClusterShape{}) == 1;
  using TileSchedulerTag =
      std::conditional_t<IsStreamK, cutlass::gemm::StreamKScheduler,
                         std::conditional_t<IsGrouped, W8A16Sm90GroupedTileSchedulerTag, void>>;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, TileSchedulerTag>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;

  // The shared arguments (w8a16_host.hpp) plus the tile walk order (tile_order.hpp): raster
  // "m" everywhere; the grouped scheduler takes its L2 group size through max_swizzle_size,
  // CUTLASS's own schedulers get 1 (no grouping, no padding).
  static typename Gemm::Arguments MakeArguments(const T* act, const __nv_fp8_e4m3* weight,
                                                const float* scale, const T* bias, void* out, int M,
                                                int N, int K, int device_id, int num_sms,
                                                int splits) {
    auto args = W8A16BaseArguments<W8A16Sm90Gemm>(act, weight, scale, bias, out, M, N, K, device_id,
                                                  num_sms, splits);
    args.scheduler.max_swizzle_size =
        IsGrouped ? W8A16Sm90PickGroup(M, N, K, Cfg::W) : W8A16Sm90CutlassSwizzle;
    args.scheduler.raster_order = W8A16Sm90Raster;
    return args;
  }

  static cutlass::Status Launch(Gemm& gemm, cudaStream_t stream) { return gemm.run(stream); }
};

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM90_GEMM_HPP_
