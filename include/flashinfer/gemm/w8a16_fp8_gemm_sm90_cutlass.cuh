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
#ifndef FLASHINFER_GEMM_W8A16_FP8_GEMM_SM90_CUTLASS_CUH_
#define FLASHINFER_GEMM_W8A16_FP8_GEMM_SM90_CUTLASS_CUH_

// W8A16 GEMM for Hopper (sm_90a) on CUTLASS's mixed-input mainloop
// (sm90_mma_tma_gmma_rs_warpspecialized_mixed_input): version H1 of the Hopper
// CUTLASS line, built next to the hand-written kernel of w8a16_fp8_gemm.cuh so the
// two can be measured per shape through the Python op's `algo` argument.
//
//   out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
//
// The mainloop is the register-sourced one: TMA lands the fp8 weight tile and the
// 16-bit activation tile in smem, each consumer warpgroup pulls its 64 weight rows
// into registers, converts them (NumericArrayConverter<T, e4m3, 4>, specialised below)
// and issues wgmma with A from registers and B from smem.  Same operand mapping as
// the Blackwell header: the narrow operand must be the builder's A, so the weight
// [N, K] is A with M' = N, the activation [M, K] is B with N' = M, D is [M', N']
// column-major, which is out[M, N] row-major.  Per-tensor scale = epilogue alpha
// (device pointer), bias = per-row (M' = N) broadcast in the fused epilogue.
//
// What this buys over the hand kernel: the activation tile height (wgmma N) is a
// template argument, so the dispatcher gets a menu of heights (256 / 192 / 160 / 128)
// to land the tile count near a whole number of waves, and CUTLASS's stream-K
// scheduler is a template argument too.  Whether it also matches the hand kernel's
// mainloop speed is what the per-shape sweep decides.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
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
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace flashinfer {
namespace gemm {
namespace w8a16_sm90_detail {

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// Four e4m3 bytes (little-endian in `q`) -> two bf16x2 registers, value order kept.
// Hopper has no packed e4m3 -> bf16 convert (that is sm_100's cvt.rn.bf16x2.e4m3x2);
// CUTLASS's own converter goes e4m3x2 -> f16x2 -> 2 x f32 -> bf16x2, four dependent
// cvt per pair.  This is the ALU path of the hand kernel / Blackwell version B1: a
// sign-replicating prmt puts byte k in the low byte of lane k with its sign copied
// into the high byte, (v << 4) & 0x87F0 leaves the sign at [15] and the 7 magnitude
// bits at [10:4], so the bf16 exponent field holds e4m3's raw exponent, and one
// bf16x2 multiply by 2^(127 - 7) fixes the bias.  Exact for every finite e4m3 code,
// denormals included (they land as bf16 denormals and the multiply rescales them).
__device__ __forceinline__ void e4m3x4_to_bf16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
  const uint32_t lo_s = (prmt_b32(q, 0u, 0x9180u) << 4) & 0x87F087F0u;
  const uint32_t hi_s = (prmt_b32(q, 0u, 0xB3A2u) << 4) & 0x87F087F0u;
  const uint32_t k_bits = 0x7B807B80u;  // bf16x2 {2^120, 2^120}
  const __nv_bfloat162 k = *reinterpret_cast<const __nv_bfloat162*>(&k_bits);
  __nv_bfloat162 l = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&lo_s), k);
  __nv_bfloat162 h = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&hi_s), k);
  lo = *reinterpret_cast<uint32_t*>(&l);
  hi = *reinterpret_cast<uint32_t*>(&h);
}

// f16 has the native packed convert since sm_89 (PTX ISA 7.8).
__device__ __forceinline__ void e4m3x4_to_f16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(q));
}

}  // namespace w8a16_sm90_detail
}  // namespace gemm
}  // namespace flashinfer

// The RS mixed-input transform converts in packs of gcd(fragment, 32 / 8) = 4; these
// specialisations are more specialised than CUTLASS's <T, float_e4m3_t, N, Round> and
// are declared before any instantiation below, so they take over for this module only
// (the Blackwell header declares the same pair for its own module).
namespace cutlass {

template <FloatRoundStyle Round>
struct NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e4m3_t, 4, Round> {
  using result_element = cutlass::bfloat16_t;
  using source_element = cutlass::float_e4m3_t;
  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type out;
    uint32_t* o = reinterpret_cast<uint32_t*>(&out);
    flashinfer::gemm::w8a16_sm90_detail::e4m3x4_to_bf16x4(
        *reinterpret_cast<uint32_t const*>(&source), o[0], o[1]);
    return out;
  }
  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

template <FloatRoundStyle Round>
struct NumericArrayConverter<cutlass::half_t, cutlass::float_e4m3_t, 4, Round> {
  using result_element = cutlass::half_t;
  using source_element = cutlass::float_e4m3_t;
  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type out;
    uint32_t* o = reinterpret_cast<uint32_t*>(&out);
    flashinfer::gemm::w8a16_sm90_detail::e4m3x4_to_f16x4(
        *reinterpret_cast<uint32_t const*>(&source), o[0], o[1]);
    return out;
  }
  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

}  // namespace cutlass

namespace flashinfer {
namespace gemm {

template <typename T>
struct W8A16Sm90CutlassType;
template <>
struct W8A16Sm90CutlassType<__nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};
template <>
struct W8A16Sm90CutlassType<half> {
  using type = cutlass::half_t;
};

// kTileW: weight rows per CTA tile (M'): 128 for the cooperative schedule (two consumer
//         warpgroups, 64 rows each), 64 for ping-pong (one warpgroup per tile).
// kTileA: activation rows per CTA tile (N' = wgmma N), any multiple of 8 up to 256.
// kTileK: 64 fp8 bytes of weight per row per stage; with the 16-bit activation that is
//         8 KB + kTileA * 128 B per stage (40 KB for 256 rows: 5 stages in 227 KB).
// kCluster: CTAs per cluster along the weight-row dimension.  2 = the two CTAs take
//         adjacent weight tiles of the same activation tile, which TMA multicasts to both
//         (the hand kernel's arrangement): halves the activation's L2 -> SMEM traffic.
template <int kTileW, int kTileA, int kTileK = 64, int kCluster = 1>
struct W8A16Sm90Cfg {
  static_assert(kTileW == 128 || kTileW == 64, "128 rows = cooperative, 64 rows = ping-pong");
  static_assert(kTileA % 8 == 0 && kTileA <= 256, "wgmma N is a multiple of 8 up to 256");
  static_assert(kCluster == 1 || kCluster == 2, "cluster of 1 or 2 CTAs");
  static constexpr int kW = kTileW;
  static constexpr int kA = kTileA;
  static constexpr int kK = kTileK;
  static constexpr bool kCooperative = kTileW == 128;
  using TileShape = cute::Shape<cute::Int<kTileW>, cute::Int<kTileA>, cute::Int<kTileK>>;
  using ClusterShape = cute::Shape<cute::Int<kCluster>, cute::_1, cute::_1>;
  using MainloopSchedule =
      std::conditional_t<kCooperative, cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                         cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
  using EpilogueSchedule =
      std::conditional_t<kCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative,
                         cutlass::epilogue::TmaWarpSpecialized>;
};

using W8A16Sm90Cfg128x256 = W8A16Sm90Cfg<128, 256>;  // the hand kernel's default shape
using W8A16Sm90Cfg128x192 = W8A16Sm90Cfg<128, 192>;  // wave-fill heights (see PickTile)
using W8A16Sm90Cfg128x160 = W8A16Sm90Cfg<128, 160>;
using W8A16Sm90Cfg128x128 = W8A16Sm90Cfg<128, 128>;
using W8A16Sm90Cfg64x256 = W8A16Sm90Cfg<64, 256>;  // ping-pong: 2 tiles in flight per SM
// W8A16Sm90Cfg<128, 256, 64, 2> (activation multicast pair) was swept on Tue 13 Oct 2026:
// within +-2% of the cluster-1 tile on all 144 rows, so it is not instantiated.

inline cutlass::gemm::kernel::detail::DecompositionMode W8A16Sm90StreamKModeEnv();
inline cutlass::gemm::kernel::detail::ReductionMode W8A16Sm90StreamKReductionEnv();
inline int W8A16Sm90SwizzleEnv();
inline int W8A16Sm90PickSwizzle(int tiles_w, int tiles_a, int cluster_w, int max_swizzle);
inline cutlass::gemm::kernel::detail::RasterOrderOptions W8A16Sm90RasterEnv();

template <typename T, typename Cfg, bool kStreamK = false>
struct W8A16Sm90Gemm {
  using ElementW = cutlass::float_e4m3_t;
  using ElementAct = typename W8A16Sm90CutlassType<T>::type;
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
  using FusionOp = cutlass::epilogue::fusion::LinCombPerRowBias<ElementD, ElementCompute, ElementBias,
                                                                ElementD, float, kAlignD>;

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

  using TileSchedulerTag = std::conditional_t<kStreamK, cutlass::gemm::StreamKScheduler, void>;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, TileSchedulerTag>;
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
    // Tile walk order of the persistent scheduler (both the plain and the stream-K one):
    // the raster direction and how many tile rows / columns are grouped so that the
    // operand slab they share stays in L2.  CUTLASS's default is no grouping; the group
    // size is capped per problem because the scheduler pads the tile grid to it.
    const int tiles_w = (N + Cfg::kW - 1) / Cfg::kW;
    const int tiles_a = (M + Cfg::kA - 1) / Cfg::kA;
    args.scheduler.max_swizzle_size = W8A16Sm90PickSwizzle(
        tiles_w, tiles_a, cute::size<0>(typename Cfg::ClusterShape{}), W8A16Sm90SwizzleEnv());
    args.scheduler.raster_order = W8A16Sm90RasterEnv();
    if constexpr (kStreamK) {
      args.scheduler.decomposition_mode = W8A16Sm90StreamKModeEnv();
      args.scheduler.reduction_mode = W8A16Sm90StreamKReductionEnv();
    }
    return args;
  }

  // CUTLASS workspace (persistent scheduler: usually 0; stream-K: the peers' fp32
  // partials plus per-tile barriers), rounded to 256 B.
  static size_t WorkspaceBytes(int M, int N, int K, int device_id, int num_sms) {
    auto args =
        MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, device_id, num_sms);
    return (Gemm::get_workspace_size(args) + 255) / 256 * 256;
  }

  static cudaError_t Run(const T* act, const __nv_fp8_e4m3* weight, const float* scale,
                         const T* bias, T* out, int M, int N, int K, void* workspace,
                         size_t workspace_bytes, int device_id, int num_sms,
                         cudaStream_t stream) {
    auto args = MakeArguments(act, weight, scale, bias, out, M, N, K, device_id, num_sms);
    Gemm gemm;
    if (Gemm::get_workspace_size(args) > workspace_bytes) return cudaErrorInvalidValue;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorNotSupported;
    if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
    return cudaSuccess;
  }
};

// Hopper `algo` codes of the Python op (`flashinfer.gemm.w8a16_fp8_gemm(..., algo=...)`
// on sm_90a); they must not change.  0 = the launcher's choice, 1 = the hand-written
// kernel's own dispatcher (w8a16_fp8_gemm.cuh), 2.. = one CUTLASS variant.
enum class W8A16Sm90Algo {
  kAuto = 0,
  kHand = 1,
  kCutlass128x256 = 2,
  kCutlass128x192 = 3,
  kCutlass128x160 = 4,
  kCutlass128x128 = 5,
  kCutlass64x256 = 6,
  kCutlass128x256StreamK = 7,
  kCutlass128x192StreamK = 8,
};
constexpr int kW8A16Sm90MaxAlgoCode = 8;

inline W8A16Sm90Algo W8A16Sm90AlgoFromCode(int code) {
  return (code >= 1 && code <= kW8A16Sm90MaxAlgoCode) ? static_cast<W8A16Sm90Algo>(code)
                                                     : W8A16Sm90Algo::kAuto;
}

// FLASHINFER_W8A16_FP8_ALGO_SM90 = "hand" | "cutlass_128x256" | "cutlass_128x192" |
// "cutlass_128x160" | "cutlass_128x128" | "cutlass_64x256" | "cutlass_128x256_streamk" |
// "cutlass_128x192_streamk" pins the variant; read once.
inline W8A16Sm90Algo W8A16Sm90AlgoEnv() {
  static const W8A16Sm90Algo algo = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_ALGO_SM90");
    if (s == nullptr || *s == '\0') return W8A16Sm90Algo::kAuto;
    if (std::strcmp(s, "hand") == 0) return W8A16Sm90Algo::kHand;
    if (std::strcmp(s, "cutlass_128x256") == 0) return W8A16Sm90Algo::kCutlass128x256;
    if (std::strcmp(s, "cutlass_128x192") == 0) return W8A16Sm90Algo::kCutlass128x192;
    if (std::strcmp(s, "cutlass_128x160") == 0) return W8A16Sm90Algo::kCutlass128x160;
    if (std::strcmp(s, "cutlass_128x128") == 0) return W8A16Sm90Algo::kCutlass128x128;
    if (std::strcmp(s, "cutlass_64x256") == 0) return W8A16Sm90Algo::kCutlass64x256;
    if (std::strcmp(s, "cutlass_128x256_streamk") == 0)
      return W8A16Sm90Algo::kCutlass128x256StreamK;
    if (std::strcmp(s, "cutlass_128x192_streamk") == 0)
      return W8A16Sm90Algo::kCutlass128x192StreamK;
    return W8A16Sm90Algo::kAuto;
  }();
  return algo;
}

// Same knobs as the Blackwell stream-K variant (FLASHINFER_W8A16_FP8_STREAMK_MODE /
// _REDUCTION); read once.
inline cutlass::gemm::kernel::detail::DecompositionMode W8A16Sm90StreamKModeEnv() {
  using Mode = cutlass::gemm::kernel::detail::DecompositionMode;
  static const Mode mode = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_STREAMK_MODE");
    if (s == nullptr || *s == '\0') return Mode::Heuristic;
    if (std::strcmp(s, "streamk") == 0) return Mode::StreamK;
    if (std::strcmp(s, "splitk") == 0) return Mode::SplitK;
    if (std::strcmp(s, "dataparallel") == 0) return Mode::DataParallel;
    return Mode::Heuristic;
  }();
  return mode;
}

inline cutlass::gemm::kernel::detail::ReductionMode W8A16Sm90StreamKReductionEnv() {
  using Mode = cutlass::gemm::kernel::detail::ReductionMode;
  static const Mode mode = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_STREAMK_REDUCTION");
    if (s != nullptr && std::strcmp(s, "nondeterministic") == 0) return Mode::Nondeterministic;
    return Mode::Deterministic;
  }();
  return mode;
}

// FLASHINFER_W8A16_FP8_SM90_RASTER = "m" (default) | "n" | "heuristic": which tiles
// consecutive CTAs of the persistent scheduler take.  "m" walks the weight tiles of one
// activation tile (the activation tile stays in L2, the weight streams), "n" walks the
// activation tiles of one weight tile.  CUTLASS's heuristic picks "n" whenever there are
// at least as many weight tiles as activation tiles (N >= M/2), and for this operand
// layout that is 2x slower once the activation outgrows L2: mlp.gate_up tp1 M=44160 31.1
// vs 14.9 ms, mlp.down tp1 33.2 vs 14.5, gen.qkv tp1 12.9 vs 5.8 (ncu, locked clocks,
// Tue 13 Oct 2026); "m" is never worse than the heuristic on the 144-row table.
//
// FLASHINFER_W8A16_FP8_SM90_SWIZZLE = 1 (default) | 2 | 4 | 8: upper bound on the
// scheduler's L2 tile-group size (W8A16Sm90PickSwizzle may lower it).  Version H1 made 8
// the default because it recovered 12% on gen.qkv tp1 M=8192; that was the heuristic's
// "n" raster being papered over by the grouping.  With "m" the group buys nothing (1.046
// vs 1.054 ms on that shape, equal on five others) and it pads the tile grid, so it is
// off.  Both for benchmarking; read once.
inline int W8A16Sm90SwizzleEnv() {
  static const int v = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_SM90_SWIZZLE");
    const int n = (s != nullptr && *s != '\0') ? std::atoi(s) : 1;
    return (n == 1 || n == 2 || n == 4 || n == 8) ? n : 1;
  }();
  return v;
}

// The swizzle is not free: CUTLASS's persistent and stream-K schedulers round BOTH tile
// counts up to a multiple of (swizzle x cluster) and run the padding tiles (TMA zero-fills
// them, the epilogue drops them; see round_up in tile_scheduler_params.h).  With swizzle 8
// that is 10 -> 16 weight tiles on N=1280 (ncu, Tue 13 Oct 2026: 1.6x the HGMMA count of the
// hand kernel, 247 vs 162 us) and 4 -> 8 activation tiles on M=1024.  CUTLASS also shrinks
// the swizzle on its own when the smaller tile count is under 6 / 3 / 2 (get_log_swizzle_size),
// so the waste is computed for the size it will actually use.  Largest size whose padding
// costs at most 0.5% of the tiles, 1 if none does: even 1.7% of padding (M=44160 = 173
// activation tiles -> 176) cost 3-5% at that size, while the grouping itself is worth 11% on
// gen.qkv tp1 M=8192 (80 x 32 tiles, no padding) and nothing on the M=44160 shapes.
inline int W8A16Sm90PickSwizzle(int tiles_w, int tiles_a, int cluster_w, int max_swizzle) {
  const int min_dim = tiles_w < tiles_a ? tiles_w : tiles_a;
  const long long real = 1LL * tiles_w * tiles_a;
  for (int s = max_swizzle; s >= 1; s /= 2) {
    int e = 1;  // what CUTLASS turns `s` into
    if (s >= 8 && min_dim >= 6) {
      e = 8;
    } else if (s >= 4 && min_dim >= 3) {
      e = 4;
    } else if (s >= 2 && min_dim >= 2) {
      e = 2;
    }
    if (s > 1 && e == 1) continue;  // same as 1: decide there
    const int mw = e * cluster_w;
    const long long padded = 1LL * ((tiles_w + mw - 1) / mw * mw) * ((tiles_a + e - 1) / e * e);
    if (padded * 1000 <= real * 1005) return s;
  }
  return 1;
}

inline cutlass::gemm::kernel::detail::RasterOrderOptions W8A16Sm90RasterEnv() {
  using R = cutlass::gemm::kernel::detail::RasterOrderOptions;
  static const R v = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_SM90_RASTER");
    if (s != nullptr && std::strcmp(s, "n") == 0) return R::AlongN;
    if (s != nullptr && std::strcmp(s, "heuristic") == 0) return R::Heuristic;
    return R::AlongM;
  }();
  return v;
}

// Version H3: the dispatcher over the CUTLASS variants and the hand kernel, fitted to the
// locked-clock sweep of Tue 13 Oct 2026 (H100 SXM ipp2-0148 / ipp2-0069, every launch timed
// by ncu at base clocks, 24 Cosmos3 shapes x 6 M classes, each variant / the bf16 cuBLAS
// GEMM measured next to it) taken AFTER the scheduler's padding tiles were removed (see
// W8A16Sm90PickSwizzle; version H2 was fitted to padded CUTLASS numbers).  Geomean kernel /
// cuBLAS per M class:
//
//   M       hand   CUTLASS 128x256   this rule (measured)   per-row oracle
//   1024    1.32   1.65              1.19                   1.16
//   2048    1.20   1.33              1.10                   1.08
//   4096    1.09   1.15              1.06                   1.04
//   8192    1.04   1.06              1.02                   1.01
//   44160   1.03   1.04              1.01                   1.01
//   88320   1.03   1.02              1.01                   1.00
//
// Without the padding, the CUTLASS 128x256 mainloop and the hand kernel are the same
// kernel to within 2% on every row with N > 512 (same tensor-pipe utilisation, same stall
// mix under ncu); a cluster of 2 with the activation tile multicast changes nothing (+-2%).
// What remains is wave quantisation of the 128x256 tile (units = weight tiles x activation
// tiles over 132 SMs), so the rule reads the wave count and how full the last wave is:
//   * one partial wave (<= 1.5): a long K (>= 4096) with at least 0.7 of a wave -> stream-K
//     splits the tiles along K (attn.out / mlp.down / gen.qkv at M=1024-2048: 25-30% over
//     the hand kernel's split-K); otherwise the 64x256 ping-pong tile, which puts 2x the
//     CTAs on the machine (10-25%).
//   * more than 1.5 waves: a last wave that is >= 0.8 full costs nothing -> 128x256; a
//     sliver of a last wave (< 0.15, e.g. mlp.gate_up tp4 at M=4096: 6.06 waves) -> stream-K
//     (2-10%); in between, ping-pong up to 3 waves (gen.qkv tp8 at M=8192: 2.42 waves,
//     12%), 128x256 beyond.
//   * N <= 512 (und.kv at TP 2-8): the hand kernel's split-K up to M=4096 (CUTLASS's
//     stream-K has a ~40 us floor there), the 128x128 tile for under a wave, 128x256 above.
// Worst row of the rule against the per-row oracle: und.kv tp4 M=4096 (+12%, 27 us).
// The hand kernel is left with 8 of the 144 rows.  128x192 / 128x160 never win a row (the
// register-sourced mainloop converts the weight once per activation tile, so narrower
// tiles convert more per flop); they stay `algo`-only.
inline W8A16Sm90Algo W8A16Sm90PickAlgo(int M, int N, int K, int num_sms, int algo = 0) {
  const W8A16Sm90Algo forced = W8A16Sm90AlgoFromCode(algo);
  if (forced != W8A16Sm90Algo::kAuto) return forced;
  const W8A16Sm90Algo env = W8A16Sm90AlgoEnv();
  if (env != W8A16Sm90Algo::kAuto) return env;
  if (K % 16 != 0 || N % 8 != 0) return W8A16Sm90Algo::kHand;  // CUTLASS alignment
  const long long units = 1LL * ((N + 127) / 128) * ((M + 255) / 256);
  const double waves = static_cast<double>(units) / std::max(num_sms, 1);
  const double frac = waves - std::floor(waves);
  if (N <= 512) {
    if (units <= 32 && M <= 4096) return W8A16Sm90Algo::kHand;
    if (waves < 0.75) return W8A16Sm90Algo::kCutlass128x128;
    return W8A16Sm90Algo::kCutlass128x256;
  }
  if (waves <= 1.5) {
    if (K >= 4096 && waves >= 0.7) return W8A16Sm90Algo::kCutlass128x256StreamK;
    if (N <= 2048 || K <= 3200) return W8A16Sm90Algo::kCutlass64x256;
    return W8A16Sm90Algo::kCutlass128x256;
  }
  if (frac >= 0.8) return W8A16Sm90Algo::kCutlass128x256;
  // A sliver of a last wave is worth spreading along K only while it is a visible share
  // of the run (0.1 of 262 waves is not; stream-K's fix-up traffic is).
  if (frac < 0.15 && K >= 2048 && waves <= 16.0) return W8A16Sm90Algo::kCutlass128x256StreamK;
  if (waves <= 3.0) return W8A16Sm90Algo::kCutlass64x256;
  return W8A16Sm90Algo::kCutlass128x256;
}

// Workspace bytes of a CUTLASS variant (0 for the hand kernel: its own function sizes
// its split-K partials).
template <typename T>
size_t W8A16Fp8GemmSm90CutlassWorkspaceBytes(W8A16Sm90Algo algo, int M, int N, int K,
                                             int device_id, int num_sms) {
  switch (algo) {
    case W8A16Sm90Algo::kCutlass128x256:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x256>::WorkspaceBytes(M, N, K, device_id, num_sms);
    case W8A16Sm90Algo::kCutlass128x192:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x192>::WorkspaceBytes(M, N, K, device_id, num_sms);
    case W8A16Sm90Algo::kCutlass128x160:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x160>::WorkspaceBytes(M, N, K, device_id, num_sms);
    case W8A16Sm90Algo::kCutlass128x128:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x128>::WorkspaceBytes(M, N, K, device_id, num_sms);
    case W8A16Sm90Algo::kCutlass64x256:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg64x256>::WorkspaceBytes(M, N, K, device_id, num_sms);
    case W8A16Sm90Algo::kCutlass128x256StreamK:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x256, true>::WorkspaceBytes(M, N, K, device_id,
                                                                         num_sms);
    case W8A16Sm90Algo::kCutlass128x192StreamK:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x192, true>::WorkspaceBytes(M, N, K, device_id,
                                                                         num_sms);
    default:
      return 0;
  }
}

template <typename T>
cudaError_t W8A16Fp8GemmSm90Cutlass(W8A16Sm90Algo algo, const T* act, const __nv_fp8_e4m3* weight,
                                    const float* scale, const T* bias, T* out, int M, int N, int K,
                                    void* workspace, size_t workspace_bytes, int device_id,
                                    int num_sms, cudaStream_t stream) {
  switch (algo) {
    case W8A16Sm90Algo::kCutlass128x256:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x256>::Run(act, weight, scale, bias, out, M, N, K,
                                                        workspace, workspace_bytes, device_id,
                                                        num_sms, stream);
    case W8A16Sm90Algo::kCutlass128x192:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x192>::Run(act, weight, scale, bias, out, M, N, K,
                                                        workspace, workspace_bytes, device_id,
                                                        num_sms, stream);
    case W8A16Sm90Algo::kCutlass128x160:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x160>::Run(act, weight, scale, bias, out, M, N, K,
                                                        workspace, workspace_bytes, device_id,
                                                        num_sms, stream);
    case W8A16Sm90Algo::kCutlass128x128:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x128>::Run(act, weight, scale, bias, out, M, N, K,
                                                        workspace, workspace_bytes, device_id,
                                                        num_sms, stream);
    case W8A16Sm90Algo::kCutlass64x256:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg64x256>::Run(act, weight, scale, bias, out, M, N, K,
                                                       workspace, workspace_bytes, device_id,
                                                       num_sms, stream);
    case W8A16Sm90Algo::kCutlass128x256StreamK:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x256, true>::Run(act, weight, scale, bias, out, M, N,
                                                              K, workspace, workspace_bytes,
                                                              device_id, num_sms, stream);
    case W8A16Sm90Algo::kCutlass128x192StreamK:
      return W8A16Sm90Gemm<T, W8A16Sm90Cfg128x192, true>::Run(act, weight, scale, bias, out, M, N,
                                                              K, workspace, workspace_bytes,
                                                              device_id, num_sms, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_SM90_CUTLASS_CUH_
