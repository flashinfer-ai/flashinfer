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
#ifndef FLASHINFER_GEMM_W8A16_SM100_GEMM_HPP_
#define FLASHINFER_GEMM_W8A16_SM100_GEMM_HPP_

// The Blackwell W8A16 kernel variants: one CUTLASS GemmUniversal per (tile, decomposition)
// on FlashInfer's fork of the sm100 mixed-input mainloop (builder / collective / kernel /
// policy .hpp in this directory).
//
//   out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
//
// tcgen05.mma takes both operands from shared memory (or A from TMEM), so the fp8 -> 16-bit
// conversion cannot ride in registers in front of the MMA the way the Hopper kernel does
// it.  The mixed-input collective runs the conversion as a separate transform warpgroup
// between the TMA load and the MMA: TMA lands the fp8 tile in smem, four transform warps
// convert it (convert.hpp) and write the 16-bit copy to TMEM (K-major A path), the MMA
// warp issues tcgen05.mma with A from TMEM and B (the activation) from smem.
//
// Operand mapping ("swap and transpose", as CUTLASS example 86): the narrow operand must be
// the builder's A, so the weight [N, K] is A with M' = N, the activation [M, K] is B with
// N' = M, and D is written as [M', N'] column-major, which is exactly out[M, N] row-major.
// The per-tensor scale is the epilogue alpha (fp32, read from the device pointer), the bias
// is a per-row (M' = N) broadcast in the same fused epilogue.  A split-K variant writes fp32
// partials instead and applies scale and bias in its reduce.

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
#include "flashinfer/gemm/w8a16_sm100/builder.hpp"
#include "flashinfer/gemm/w8a16_sm100/collective.hpp"
#include "flashinfer/gemm/w8a16_sm100/convert.hpp"
#include "flashinfer/gemm/w8a16_sm100/kernel.hpp"
#include "flashinfer/gemm/w8a16_sm100/policy.hpp"
#include "flashinfer/gemm/w8a16_sm100/tile_order.hpp"

namespace flashinfer::gemm {

// Split-K reduce: out[i] = T(sum_s ws[s][i] * scale + bias[i % N]), 8 elements per
// thread (two 16 B loads per split, one 16 B store).  Requires N % 8 == 0 (the launcher
// checks it) and M * N % 8 == 0, which follows.
template <typename T>
__global__ void W8A16Sm100SplitKReduceKernel(const float* __restrict__ ws, int splits,
                                             const float* __restrict__ scale,
                                             const T* __restrict__ bias, T* __restrict__ out,
                                             int64_t mn, int N) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (i >= mn) return;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  for (int sp = 0; sp < splits; ++sp) {
    const float4* src = reinterpret_cast<const float4*>(ws + static_cast<int64_t>(sp) * mn + i);
    const float4 a = src[0], b = src[1];
    acc[0] += a.x;
    acc[1] += a.y;
    acc[2] += a.z;
    acc[3] += a.w;
    acc[4] += b.x;
    acc[5] += b.y;
    acc[6] += b.z;
    acc[7] += b.w;
  }
  const float sc = *scale;
  const int n = static_cast<int>(i % N);
  alignas(16) T v[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float bb = bias != nullptr ? static_cast<float>(bias[n + e]) : 0.f;
    v[e] = static_cast<T>(acc[e] * sc + bb);
  }
  *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(v);
}

template <typename T>
struct W8A16Sm100CutlassType;
template <>
struct W8A16Sm100CutlassType<__nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};
template <>
struct W8A16Sm100CutlassType<half> {
  using type = cutlass::half_t;
};

// kTileW: weight rows per MMA tile (M'), cluster-level for the 2SM schedule.
// kTileA: activation rows per CTA tile (N'), up to the MMA's 256.
// kAInSmem: converted A operand staged in smem (SS MMA) instead of TMEM; required when
// two accumulator stages of kTileA columns already fill TMEM (kTileA == 256).
// kClusterN: CTAs (2SM pairs) along the activation dimension that share one weight tile;
// the weight TMA load is multicast to all of them (version B16 experiment for the weights
// that overflow L2: halves the weight's L2 -> SMEM traffic and doubles the activation band
// per weight footprint).
template <int kTileW, int kTileA, int kTileK, int kClusterM, bool k2SM, bool kAInSmem = false,
          int kClusterN = 1>
struct W8A16Sm100Cfg {
  static constexpr int kW = kTileW;
  static constexpr int kA = kTileA;
  static constexpr int kK = kTileK;
  static constexpr int kCluster = kClusterM;
  static constexpr int kClusterAct = kClusterN;
  static constexpr bool kTwoSM = k2SM;
  using TileShape = cute::Shape<cute::Int<kTileW>, cute::Int<kTileA>, cute::Int<kTileK>>;
  using ClusterShape = cute::Shape<cute::Int<kClusterM>, cute::Int<kClusterN>, cute::_1>;
  using MainloopSchedule = std::conditional_t<
      k2SM,
      std::conditional_t<kAInSmem, cutlass::gemm::KernelW8A16MixedInput2SmSmemSm100,
                         cutlass::gemm::KernelW8A16MixedInput2SmSm100>,
      std::conditional_t<kAInSmem, cutlass::gemm::KernelW8A16MixedInput1SmSmemSm100,
                         cutlass::gemm::KernelW8A16MixedInput1SmSm100>>;
  using EpilogueSchedule = std::conditional_t<k2SM, cutlass::epilogue::TmaWarpSpecialized2Sm,
                                              cutlass::epilogue::TmaWarpSpecialized1Sm>;
};

using W8A16Sm100Cfg2SM256x128 = W8A16Sm100Cfg<256, 128, 128, 2, true>;
using W8A16Sm100Cfg1SM128x128 = W8A16Sm100Cfg<128, 128, 128, 1, false>;
// Version B4: 256-row activation tile halves the transform work per MMA flop; the two
// 256-column accumulator stages take all of TMEM, so A-compute lives in smem and the
// k-tile drops to 64 to keep 4+ pipeline stages.
using W8A16Sm100Cfg2SM256x256 = W8A16Sm100Cfg<256, 256, 64, 2, true, true>;
// Version B8: 192-row activation tile with A still in TMEM (2 x 192 acc + 2 x 64 A = 512 cols):
// 2/3 the transform work per flop of 256x128 on the skeleton whose no-convert ceiling is 99%.
using W8A16Sm100Cfg2SM256x192 = W8A16Sm100Cfg<256, 192, 128, 2, true>;
// Version B14: 160-row activation tile (2 x 160 acc + 2 x 64 A = 448 TMEM cols).  Not a
// faster tile per se: a second tile *height* so the dispatcher can pick the one whose
// tile count lands closest to a whole number of waves.  N=5120 at M=1024: 20 x 8 =
// 160 tiles of 128 rows = 2.16 waves (last wave 16% full) vs 20 x 7 = 140 tiles of 160
// rows = 1.89 waves (89% full); the extra rows of the last M-tile are masked by the
// TMA / epilogue predication, and the transform work per k-tile (the weight tile) is
// the same, so the taller tile costs only its extra MMA.
using W8A16Sm100Cfg2SM256x160 = W8A16Sm100Cfg<256, 160, 128, 2, true>;

// Version B12: split-K.  With kSplitK the kernel runs as a *batch* of `splits` GEMMs over
// K-slices in one launch (batch stride = the slice length along K for A and B, one
// M x N fp32 partial per batch for D), so an under-filled grid becomes `splits` times as
// many tiles for the persistent scheduler; W8A16Sm100SplitKReduceKernel then sums the
// partials, applies the per-tensor scale and the bias and casts.  Scale and bias are
// deferred to the reduce (alpha = 1 here) so the partials are plain accumulators.
// Version B13: stream-K.  With kStreamK the kernel runs under CUTLASS's SM100 stream-K
// tile scheduler instead of the plain persistent one: the full waves stay data-parallel
// (one cluster per output tile, whole K), and the tiles of the partial last wave(s) are
// cut along K into equal iteration counts handed to every resident cluster, so no
// cluster idles while a dozen others walk a whole K.  Peers of a tile reduce their fp32
// partials through the scheduler workspace (deterministic order); only the final peer
// runs the real epilogue (scale, bias, cast), so the output path is the plain one.  The
// mainloop is untouched: kernel.hpp already consumes the per-unit k-range and calls the
// scheduler's fixup before the epilogue.  Mutually exclusive with kSplitK.
template <typename T, typename Cfg, bool kSplitK = false, bool kStreamK = false>
struct W8A16Sm100Gemm {
  static_assert(!(kSplitK && kStreamK), "split-K and stream-K are different variants");
  using ElementW = cutlass::float_e4m3_t;
  using ElementAct = typename W8A16Sm100CutlassType<T>::type;
  using ElementD = std::conditional_t<kSplitK, float, ElementAct>;
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
  // Split-K partials: D = acc (fp32), scale and bias applied in the reduce kernel.
  using FusionOp = std::conditional_t<
      kSplitK, cutlass::epilogue::fusion::LinearCombination<float, float, void, float>,
      cutlass::epilogue::fusion::LinCombPerRowBias<ElementD, ElementCompute, ElementBias, ElementD,
                                                   float, kAlignD>>;

  // fp32 partials double the epilogue's smem per stage (the auto tile is 128x32 for bf16:
  // 18 KB; fp32 would be 33 KB and overshoot the 227 KB budget by 1 KB next to the 6/4
  // mainloop stages), so the split-K variant uses a 128x16 epilogue tile.
  using EpilogueTile = std::conditional_t<kSplitK, cute::Shape<cute::_128, cute::_16>,
                                          cutlass::epilogue::collective::EpilogueTileAuto>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, typename Cfg::TileShape,
      typename Cfg::ClusterShape, EpilogueTile, ElementAccumulator, ElementCompute, void, LayoutD,
      kAlignD, ElementD, LayoutD, kAlignD, typename Cfg::EpilogueSchedule, FusionOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, cute::tuple<ElementW>, LayoutA, kAlignA,
      ElementAct, LayoutB, kAlignB, ElementAccumulator, typename Cfg::TileShape,
      typename Cfg::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename Cfg::MainloopSchedule>::CollectiveOp;

  using TileSchedulerTag = std::conditional_t<kStreamK, cutlass::gemm::StreamKScheduler, void>;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, TileSchedulerTag>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;

  // `splits` > 1 only with kSplitK: problem {N, M, K / splits, splits}, A and B batch
  // stride = K / splits elements (the next K-slice), D batch stride = M * N (the next
  // fp32 partial in `out`).  Row strides stay the full K, so TMA walks the sliced
  // operands in place.
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
    if constexpr (!kSplitK) {
      fusion.alpha_ptr = scale;  // per-tensor weight scale, read on device
      fusion.beta_ptr = nullptr;
      fusion.bias_ptr = reinterpret_cast<const ElementBias*>(bias);  // nullptr -> 0
      fusion.dBias = {cute::_1{}, cute::_0{}, 0};
    } else {
      (void)scale;
      (void)bias;
    }
    // Tile order (tile_order.hpp): how many tiles the in-flight clusters group along the
    // fast axis (L2 reuse of a weight band x activation band instead of a line of tiles),
    // which axis walks fastest, and the TMA loads' L2 eviction priority.
    args.scheduler.max_swizzle_size = W8A16Sm100PickSwizzle(M, N, K);
    args.scheduler.raster_order = kW8A16Sm100Raster;
    args.mainloop.cache_hint_a = kW8A16Sm100WeightCacheHint;
    args.mainloop.cache_hint_b = kW8A16Sm100ActCacheHint;
    if constexpr (kStreamK) {
      // CUTLASS's heuristic decomposition: data-parallel for the full waves, stream-K for
      // the tiles of the last (up to two) waves; collapses to plain split-K when the
      // leftover tiles divide the clusters evenly, and to data-parallel when there is
      // nothing to balance.  Deterministic (K-ordered) peer reduction.
      args.scheduler.decomposition_mode =
          cutlass::gemm::kernel::detail::DecompositionMode::Heuristic;
      args.scheduler.reduction_mode = cutlass::gemm::kernel::detail::ReductionMode::Deterministic;
    }
    return args;
  }

  static size_t PartialsBytes(int M, int N, int splits) {
    return kSplitK && splits > 1 ? static_cast<size_t>(splits) * M * N * sizeof(float) : 0;
  }

  // CUTLASS workspace (persistent scheduler: usually 0; stream-K: the peers' fp32
  // partials plus per-tile barriers) rounded to 256 B, then the fp32 partials for
  // split-K.
  static size_t WorkspaceBytes(int M, int N, int K, int device_id, int num_sms, int splits = 1) {
    auto args = MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, device_id,
                              num_sms, splits);
    const size_t cutlass_ws = (Gemm::get_workspace_size(args) + 255) / 256 * 256;
    return cutlass_ws + PartialsBytes(M, N, splits);
  }

  static cudaError_t Run(const T* act, const __nv_fp8_e4m3* weight, const float* scale,
                         const T* bias, T* out, int M, int N, int K, void* workspace,
                         size_t workspace_bytes, int device_id, int num_sms, cudaStream_t stream,
                         int splits = 1) {
    if (kSplitK && (splits < 2 || K % (splits * Cfg::kK) != 0)) return cudaErrorInvalidValue;
    void* d_out = out;
    size_t cutlass_ws = 0;
    if constexpr (kSplitK) {
      auto probe = MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, device_id,
                                 num_sms, splits);
      cutlass_ws = (Gemm::get_workspace_size(probe) + 255) / 256 * 256;
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
    if (gemm.run(stream, nullptr, /*launch_with_pdl=*/false) != cutlass::Status::kSuccess) {
      return cudaErrorLaunchFailure;
    }
    if constexpr (kSplitK) {
      const int64_t mn = static_cast<int64_t>(M) * N;
      const int threads = 256;
      const int64_t blocks = (mn / 8 + threads - 1) / threads;
      W8A16Sm100SplitKReduceKernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
          static_cast<const float*>(d_out), splits, scale, bias, out, mn, N);
      return cudaGetLastError();
    }
    return cudaSuccess;
  }
};

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM100_GEMM_HPP_
