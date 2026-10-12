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
#ifndef FLASHINFER_GEMM_W8A16_FP8_GEMM_SM100_CUH_
#define FLASHINFER_GEMM_W8A16_FP8_GEMM_SM100_CUH_

// W8A16 GEMM for Blackwell (sm_100a), rung B0: CUTLASS mixed-input mainloop.
//
//   out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
//
// tcgen05.mma takes both operands from shared memory (or A from TMEM), so the
// fp8 -> 16-bit conversion cannot ride in registers in front of the MMA the way
// the Hopper kernel does it.  CUTLASS's sm100 mixed-input collective runs the
// conversion as a separate transform warpgroup between the TMA load and the MMA:
// TMA lands the fp8 tile in smem, four transform warps convert it and write the
// 16-bit copy to TMEM (K-major A path), the MMA warp issues tcgen05.mma with A
// from TMEM and B (the activation) from smem.
//
// Operand mapping ("swap and transpose", as CUTLASS example 86): the narrow
// operand must be the builder's A, so the weight [N, K] is A with M' = N, the
// activation [M, K] is B with N' = M, and D is written as [M', N'] column-major,
// which is exactly out[M, N] row-major.  The per-tensor scale is the epilogue
// alpha (fp32, read from the device pointer), the bias is a per-row (M' = N)
// broadcast in the same fused epilogue.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace flashinfer {
namespace gemm {

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
// kTileA: activation rows per CTA tile (N'); the auto-carveout builder caps it at 128.
template <int kTileW, int kTileA, int kTileK, int kClusterM, bool k2SM>
struct W8A16Sm100Cfg {
  static constexpr int kW = kTileW;
  static constexpr int kA = kTileA;
  static constexpr int kK = kTileK;
  static constexpr int kCluster = kClusterM;
  static constexpr bool kTwoSM = k2SM;
  using TileShape = cute::Shape<cute::Int<kTileW>, cute::Int<kTileA>, cute::Int<kTileK>>;
  using ClusterShape = cute::Shape<cute::Int<kClusterM>, cute::_1, cute::_1>;
  using MainloopSchedule =
      std::conditional_t<k2SM, cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100,
                         cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100>;
  using EpilogueSchedule = std::conditional_t<k2SM, cutlass::epilogue::TmaWarpSpecialized2Sm,
                                              cutlass::epilogue::TmaWarpSpecialized1Sm>;
};

using W8A16Sm100Cfg2SM256x128 = W8A16Sm100Cfg<256, 128, 128, 2, true>;
using W8A16Sm100Cfg1SM128x128 = W8A16Sm100Cfg<128, 128, 128, 1, false>;

template <typename T, typename Cfg>
struct W8A16Sm100Gemm {
  using ElementW = cutlass::float_e4m3_t;
  using ElementAct = typename W8A16Sm100CutlassType<T>::type;
  using ElementD = ElementAct;
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
  using FusionOp =
      cutlass::epilogue::fusion::LinCombPerRowBias<ElementD, ElementCompute, ElementBias, ElementD,
                                                   float, kAlignD>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, typename Cfg::TileShape,
      typename Cfg::ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute, void, LayoutD, kAlignD, ElementD, LayoutD, kAlignD,
      typename Cfg::EpilogueSchedule, FusionOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, cute::tuple<ElementW>, LayoutA, kAlignA,
      ElementAct, LayoutB, kAlignB, ElementAccumulator, typename Cfg::TileShape,
      typename Cfg::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename Cfg::MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;

  static typename Gemm::Arguments MakeArguments(const T* act, const __nv_fp8_e4m3* weight,
                                                const float* scale, const T* bias, T* out, int M,
                                                int N, int K, int device_id, int num_sms) {
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(N, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(M, K, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, 1));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = num_sms;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, K, 1},
        {reinterpret_cast<const ElementW*>(weight), stride_a,
         reinterpret_cast<const ElementAct*>(act), stride_b},
        {{}, nullptr, stride_d, reinterpret_cast<ElementD*>(out), stride_d},
        hw_info};
    auto& fusion = args.epilogue.thread;
    fusion.alpha = 1.0f;
    fusion.beta = 0.0f;
    fusion.alpha_ptr = scale;  // per-tensor weight scale, read on device
    fusion.beta_ptr = nullptr;
    fusion.bias_ptr = reinterpret_cast<const ElementBias*>(bias);  // nullptr -> 0
    fusion.dBias = {cute::_1{}, cute::_0{}, 0};
    return args;
  }

  static size_t WorkspaceBytes(int M, int N, int K, int device_id, int num_sms) {
    auto args =
        MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, device_id, num_sms);
    return Gemm::get_workspace_size(args);
  }

  static cudaError_t Run(const T* act, const __nv_fp8_e4m3* weight, const float* scale,
                         const T* bias, T* out, int M, int N, int K, void* workspace,
                         size_t workspace_bytes, int device_id, int num_sms, cudaStream_t stream) {
    auto args = MakeArguments(act, weight, scale, bias, out, M, N, K, device_id, num_sms);
    Gemm gemm;
    if (Gemm::get_workspace_size(args) > workspace_bytes) return cudaErrorInvalidValue;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorNotSupported;
    if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    if (gemm.run(stream, nullptr, /*launch_with_pdl=*/false) != cutlass::Status::kSuccess) {
      return cudaErrorLaunchFailure;
    }
    return cudaSuccess;
  }
};

enum class W8A16Sm100Tile { kAuto, k2SM256x128 = 1, k1SM128x128 = 2 };

// FLASHINFER_W8A16_FP8_TILE_SM100 = "2sm256x128" | "1sm128x128" pins the tile; read once.
inline W8A16Sm100Tile W8A16Sm100TileEnv() {
  static const W8A16Sm100Tile tile = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_TILE_SM100");
    if (s == nullptr || *s == '\0') return W8A16Sm100Tile::kAuto;
    if (std::strcmp(s, "2sm256x128") == 0) return W8A16Sm100Tile::k2SM256x128;
    if (std::strcmp(s, "1sm128x128") == 0) return W8A16Sm100Tile::k1SM128x128;
    return W8A16Sm100Tile::kAuto;
  }();
  return tile;
}

inline W8A16Sm100Tile W8A16Sm100PickTile(int M, int N, int K) {
  const W8A16Sm100Tile env = W8A16Sm100TileEnv();
  if (env != W8A16Sm100Tile::kAuto) return env;
  (void)M;
  (void)N;
  (void)K;
  return W8A16Sm100Tile::k2SM256x128;
}

template <typename T>
size_t W8A16Fp8GemmSm100WorkspaceBytes(int M, int N, int K, int device_id, int num_sms) {
  switch (W8A16Sm100PickTile(M, N, K)) {
    case W8A16Sm100Tile::k1SM128x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    default:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
  }
}

template <typename T>
cudaError_t W8A16Fp8GemmSm100(const T* act, const __nv_fp8_e4m3* weight, const float* scale,
                              const T* bias, T* out, int M, int N, int K, void* workspace,
                              size_t workspace_bytes, int device_id, int num_sms,
                              cudaStream_t stream) {
  switch (W8A16Sm100PickTile(M, N, K)) {
    case W8A16Sm100Tile::k1SM128x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    default:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
  }
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_SM100_CUH_
