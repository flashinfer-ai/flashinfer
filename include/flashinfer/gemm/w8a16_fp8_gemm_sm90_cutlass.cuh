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
template <int kTileW, int kTileA, int kTileK = 64>
struct W8A16Sm90Cfg {
  static_assert(kTileW == 128 || kTileW == 64, "128 rows = cooperative, 64 rows = ping-pong");
  static_assert(kTileA % 8 == 0 && kTileA <= 256, "wgmma N is a multiple of 8 up to 256");
  static constexpr int kW = kTileW;
  static constexpr int kA = kTileA;
  static constexpr int kK = kTileK;
  static constexpr bool kCooperative = kTileW == 128;
  using TileShape = cute::Shape<cute::Int<kTileW>, cute::Int<kTileA>, cute::Int<kTileK>>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
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

inline cutlass::gemm::kernel::detail::DecompositionMode W8A16Sm90StreamKModeEnv();
inline cutlass::gemm::kernel::detail::ReductionMode W8A16Sm90StreamKReductionEnv();
inline int W8A16Sm90SwizzleEnv();
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
    // operand slab they share stays in L2.  CUTLASS's default is no grouping.
    args.scheduler.max_swizzle_size = W8A16Sm90SwizzleEnv();
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

// FLASHINFER_W8A16_FP8_SM90_SWIZZLE = 1 | 2 | 4 | 8 (default): tile-group size of the
// persistent scheduler's L2 swizzle.  FLASHINFER_W8A16_FP8_SM90_RASTER = "heuristic"
// (default) | "m" | "n": raster direction (M = along the weight tiles, N = along the
// activation tiles).  CUTLASS's own default is no grouping (1), and that is where the
// CUTLASS mainloop looked 12% slower than the hand kernel: ncu at locked clocks on
// gen.qkv tp1 M=8192 (N=10240 K=5120), Tue 13 Oct 2026, 128x256 tile: swizzle 1 =
// 1.261 ms at 56% of DRAM peak, swizzle 8 = 1.117 ms, the hand kernel 1.121 ms.  The
// raster direction does not matter once the groups are 8 wide.  For benchmarking; read once.
inline int W8A16Sm90SwizzleEnv() {
  static const int v = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_SM90_SWIZZLE");
    const int n = (s != nullptr && *s != '\0') ? std::atoi(s) : 8;
    return (n == 1 || n == 2 || n == 4 || n == 8) ? n : 8;
  }();
  return v;
}

inline cutlass::gemm::kernel::detail::RasterOrderOptions W8A16Sm90RasterEnv() {
  using R = cutlass::gemm::kernel::detail::RasterOrderOptions;
  static const R v = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_SM90_RASTER");
    if (s != nullptr && std::strcmp(s, "m") == 0) return R::AlongM;
    if (s != nullptr && std::strcmp(s, "n") == 0) return R::AlongN;
    return R::Heuristic;
  }();
  return v;
}

// Version H1: the launcher's choice is still the hand kernel; the CUTLASS variants are
// reachable through `algo` / the env var until the per-shape sweep says otherwise.
inline W8A16Sm90Algo W8A16Sm90PickAlgo(int M, int N, int K, int num_sms, int algo = 0) {
  const W8A16Sm90Algo forced = W8A16Sm90AlgoFromCode(algo);
  if (forced != W8A16Sm90Algo::kAuto) return forced;
  const W8A16Sm90Algo env = W8A16Sm90AlgoEnv();
  if (env != W8A16Sm90Algo::kAuto) return env;
  (void)M;
  (void)N;
  (void)K;
  (void)num_sms;
  return W8A16Sm90Algo::kHand;
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
