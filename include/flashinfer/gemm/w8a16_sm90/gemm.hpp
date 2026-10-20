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
// warpgroup pulls its 64 weight rows into registers, converts them (convert.hpp) and issues
// wgmma with A from registers and B from smem.  Operand mapping: the narrow operand must be
// the builder's A, so the weight [N, K] is A with M' = N, the activation [M, K] is B with
// N' = M, D is [M', N'] column-major, which is out[M, N] row-major.  Per-tensor scale =
// epilogue alpha (device pointer), bias = per-row (M' = N) broadcast in the fused epilogue.
// A split-K variant writes fp32 partials instead and applies scale and bias in its reduce.

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
#include "flashinfer/gemm/w8a16_sm90/convert.hpp"
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

// The instantiated tiles.  Cooperative 128-row weight tiles at four activation heights (the
// dispatcher picks the height whose tile count best fills the SMs), and two ping-pong tiles
// (two tiles in flight per SM) for short mainloops.  A cluster-2 multicast pair of the
// 128x256 tile measured within 2% of the cluster-1 tile on every benchmark row, so it is
// not instantiated.
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
  static constexpr int kAlignA = 128 / cutlass::sizeof_bits<ElementW>::value;    // 16
  static constexpr int kAlignB = 128 / cutlass::sizeof_bits<ElementAct>::value;  // 8
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;    // 8

  // D = alpha * acc + bias[row]; alpha is the per-tensor weight scale, bias is per M' (= N).
  // Split-K partials: D = acc (fp32); scale and bias are applied in the reduce.
  using FusionOp = std::conditional_t<
      IsSplitK, cutlass::epilogue::fusion::LinearCombination<float, float, void, float>,
      cutlass::epilogue::fusion::LinCombPerRowBias<ElementD, ElementCompute, ElementBias, ElementD,
                                                   float, kAlignD>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, typename Cfg::TileShape,
      typename Cfg::ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute, void, LayoutD, kAlignD, ElementD, LayoutD, kAlignD,
      typename Cfg::EpilogueSchedule, FusionOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cute::tuple<ElementW>, LayoutA, kAlignA,
      ElementAct, LayoutB, kAlignB, ElementAccumulator, typename Cfg::TileShape,
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

  // `splits` > 1 only with split-K: problem {N, M, K / splits, splits}; A and B batch stride
  // = K / splits elements (the next K-slice, row strides stay K so TMA walks the slices in
  // place), D batch stride = M * N (the next fp32 partial).
  static typename Gemm::Arguments MakeArguments(const T* act, const __nv_fp8_e4m3* weight,
                                                const float* scale, const T* bias, void* out, int M,
                                                int N, int K, int device_id, int num_sms,
                                                int splits = 1) {
    const int kc = K / splits;
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(N, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(M, K, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, 1));
    if (splits > 1) {
      cute::get<2>(stride_a) = kc;
      cute::get<2>(stride_b) = kc;
      cute::get<2>(stride_d) = static_cast<int64_t>(M) * N;
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = num_sms;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, kc, splits},
        {reinterpret_cast<const ElementW*>(weight), stride_a,
         reinterpret_cast<const ElementAct*>(act), stride_b},
        {{}, nullptr, stride_d, reinterpret_cast<ElementD*>(out), stride_d},
        hw_info};
    auto& fusion = args.epilogue.thread;
    fusion.alpha = 1.0f;
    fusion.beta = 0.0f;
    if constexpr (!IsSplitK) {
      fusion.alpha_ptr = scale;  // per-tensor weight scale, read on device
      fusion.beta_ptr = nullptr;
      fusion.bias_ptr = reinterpret_cast<const ElementBias*>(bias);  // nullptr -> 0
      fusion.dBias = {cute::_1{}, cute::_0{}, 0};
    } else {
      (void)scale;
      (void)bias;
    }
    // Tile walk order (tile_order.hpp): raster "m" everywhere; the grouped scheduler takes
    // its L2 group size through max_swizzle_size, CUTLASS's own schedulers get 1 (no
    // grouping, no padding).
    args.scheduler.max_swizzle_size =
        IsGrouped ? W8A16Sm90PickGroup(M, N, K, Cfg::W) : kW8A16Sm90CutlassSwizzle;
    args.scheduler.raster_order = kW8A16Sm90Raster;
    if constexpr (IsStreamK) {
      // CUTLASS's heuristic decomposition (data-parallel full waves, stream-K tail) with
      // the deterministic (K-ordered) peer reduction; the other modes never measured
      // faster.
      args.scheduler.decomposition_mode =
          cutlass::gemm::kernel::detail::DecompositionMode::Heuristic;
      args.scheduler.reduction_mode = cutlass::gemm::kernel::detail::ReductionMode::Deterministic;
    }
    return args;
  }

  static size_t PartialsBytes(int M, int N, int splits) {
    return IsSplitK && splits > 1 ? static_cast<size_t>(splits) * M * N * sizeof(float) : 0;
  }

  // CUTLASS workspace (persistent scheduler: usually 0; stream-K: the peers' fp32
  // partials plus per-tile barriers), rounded to 256 B, then the split-K fp32 partials.
  static size_t WorkspaceBytes(int M, int N, int K, int device_id, int num_sms, int splits = 1) {
    auto args = MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, device_id,
                              num_sms, splits);
    return W8A16Align256(Gemm::get_workspace_size(args)) + PartialsBytes(M, N, splits);
  }

  static cudaError_t Run(const T* act, const __nv_fp8_e4m3* weight, const float* scale,
                         const T* bias, T* out, int M, int N, int K, void* workspace,
                         size_t workspace_bytes, int device_id, int num_sms, cudaStream_t stream,
                         int splits = 1) {
    if (IsSplitK && (splits < 2 || K % (splits * Cfg::K) != 0)) return cudaErrorInvalidValue;
    void* d_out = out;
    if constexpr (IsSplitK) {
      auto probe = MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, device_id,
                                 num_sms, splits);
      const size_t cutlass_ws = W8A16Align256(Gemm::get_workspace_size(probe));
      if (cutlass_ws + PartialsBytes(M, N, splits) > workspace_bytes) return cudaErrorInvalidValue;
      d_out = static_cast<char*>(workspace) + cutlass_ws;
    }
    auto args = MakeArguments(act, weight, scale, bias, d_out, M, N, K, device_id, num_sms, splits);
    Gemm gemm;
    if (Gemm::get_workspace_size(args) > workspace_bytes) return cudaErrorInvalidValue;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorNotSupported;
    if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
    if constexpr (IsSplitK) {
      return W8A16SplitKReduce<T>(static_cast<const float*>(d_out), splits, scale, bias, out, M, N,
                                  stream);
    }
    return cudaSuccess;
  }
};

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM90_GEMM_HPP_
