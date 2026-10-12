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

// W8A16 GEMM for Blackwell (sm_100a), rung B9: FlashInfer's fork of CUTLASS's
// mixed-input kernel (w8a16_sm100/) with two transform warpgroups and three 2SM tiles,
// 256x128x128 and 256x192x128 (converted A in TMEM) and 256x256x64 (converted A in
// smem), picked per problem by a wave-fill cost model (W8A16Sm100PickTile).
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
// Rung B0 profile (12800x5120 M=8192): tensor pipe 70%, the four transform warps
// (one per SM sub-partition) issue-latency bound on CUTLASS's converter, which
// goes e4m3x2 -> f16x2 -> 2x f32 -> bf16x2 (4 dependent cvt per pair).  Rung B1
// replaces the 4-element e4m3 -> bf16 / f16 array converter the transform warps
// instantiate with a pure ALU path: a sign-replicating prmt places each byte in
// its 16-bit lane with the sign extended above it, one shift + mask builds a
// 16-bit float whose exponent field holds the raw e4m3 exponent, and one
// bf16x2 / f16x2 multiply by 2^(bias16 - bias8) fixes the bias.  Exact for every
// finite e4m3 value including denormals (they land as 16-bit denormals and the
// multiply rescales them); the two e4m3fn NaN encodings come out as +-480.
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

// FlashInfer's fork of the sm100 mixed-input kernel / collective / builder (rung B2+).
#include "flashinfer/gemm/w8a16_sm100/builder.hpp"
#include "flashinfer/gemm/w8a16_sm100/collective.hpp"
#include "flashinfer/gemm/w8a16_sm100/kernel.hpp"
#include "flashinfer/gemm/w8a16_sm100/policy.hpp"

namespace flashinfer {
namespace gemm {
namespace w8a16_sm100_detail {

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// Four e4m3 bytes (little-endian in `q`) -> two bf16x2 registers, value order kept.
// prmt selector nibbles 0x9180 / 0xB3A2: byte k into the low byte of lane k and its
// sign replicated into the high byte (selector bit 3).  (v << 4) puts the 7 magnitude
// bits at [10:4] and the replicated sign at [15], mask 0x87F0 keeps exactly those.
// The exponent field then holds e4m3's raw exponent, so multiply by 2^(127 - 7) = 2^120
// (bf16 0x7B80).  Denormals: field 0 -> bf16 denormal (m / 8) * 2^-126, times 2^120
// = (m / 8) * 2^-6, the e4m3 denormal value.
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

// Same for f16: (v << 7) puts the magnitude at [13:7] and the replicated sign at [15],
// mask 0xBF80; exponent bias fix is 2^(15 - 7) = 2^8 (f16 0x5C00).
__device__ __forceinline__ void e4m3x4_to_f16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
  const uint32_t lo_s = (prmt_b32(q, 0u, 0x9180u) << 7) & 0xBF80BF80u;
  const uint32_t hi_s = (prmt_b32(q, 0u, 0xB3A2u) << 7) & 0xBF80BF80u;
  const uint32_t k_bits = 0x5C005C00u;  // f16x2 {256, 256}
  const __half2 k = *reinterpret_cast<const __half2*>(&k_bits);
  __half2 l = __hmul2(*reinterpret_cast<const __half2*>(&lo_s), k);
  __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&hi_s), k);
  lo = *reinterpret_cast<uint32_t*>(&l);
  hi = *reinterpret_cast<uint32_t*>(&h);
}

}  // namespace w8a16_sm100_detail
}  // namespace gemm
}  // namespace flashinfer

// The mixed-input transform converts in packs of 4 (gcd(fragment, 32 / 8)); these
// specializations are more specialized than CUTLASS's <T, float_e4m3_t, N, Round> and
// are declared before any instantiation below, so they take over for this module only.
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
    flashinfer::gemm::w8a16_sm100_detail::e4m3x4_to_bf16x4(
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
    flashinfer::gemm::w8a16_sm100_detail::e4m3x4_to_f16x4(
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
template <int kTileW, int kTileA, int kTileK, int kClusterM, bool k2SM, bool kAInSmem = false>
struct W8A16Sm100Cfg {
  static constexpr int kW = kTileW;
  static constexpr int kA = kTileA;
  static constexpr int kK = kTileK;
  static constexpr int kCluster = kClusterM;
  static constexpr bool kTwoSM = k2SM;
  using TileShape = cute::Shape<cute::Int<kTileW>, cute::Int<kTileA>, cute::Int<kTileK>>;
  using ClusterShape = cute::Shape<cute::Int<kClusterM>, cute::_1, cute::_1>;
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
// Rung B4: 256-row activation tile halves the transform work per MMA flop; the two
// 256-column accumulator stages take all of TMEM, so A-compute lives in smem and the
// k-tile drops to 64 to keep 4+ pipeline stages.
using W8A16Sm100Cfg2SM256x256 = W8A16Sm100Cfg<256, 256, 64, 2, true, true>;
// Rung B8: 192-row activation tile with A still in TMEM (2 x 192 acc + 2 x 64 A = 512 cols):
// 2/3 the transform work per flop of 256x128 on the skeleton whose no-convert ceiling is 99%.
using W8A16Sm100Cfg2SM256x192 = W8A16Sm100Cfg<256, 192, 128, 2, true>;

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

enum class W8A16Sm100Tile {
  kAuto,
  k2SM256x128 = 1,
  k1SM128x128 = 2,
  k2SM256x256 = 3,
  k2SM256x192 = 4
};

// FLASHINFER_W8A16_FP8_TILE_SM100 = "2sm256x128" | "1sm128x128" | "2sm256x256" |
// "2sm256x192" pins the tile; read once.
inline W8A16Sm100Tile W8A16Sm100TileEnv() {
  static const W8A16Sm100Tile tile = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_TILE_SM100");
    if (s == nullptr || *s == '\0') return W8A16Sm100Tile::kAuto;
    if (std::strcmp(s, "2sm256x128") == 0) return W8A16Sm100Tile::k2SM256x128;
    if (std::strcmp(s, "1sm128x128") == 0) return W8A16Sm100Tile::k1SM128x128;
    if (std::strcmp(s, "2sm256x256") == 0) return W8A16Sm100Tile::k2SM256x256;
    if (std::strcmp(s, "2sm256x192") == 0) return W8A16Sm100Tile::k2SM256x192;
    return W8A16Sm100Tile::kAuto;
  }();
  return tile;
}

// Rung B9 dispatch: wave-fill cost model over the three 2SM tiles.  Each cluster owns
// one 256(weight rows) x tileA(activation rows) tile; the modelled time is
// ceil(clusters needed / clusters resident) x tileA / eff, with eff = 1.00 for the
// 128-row tile (A in TMEM, 4 transform-to-MMA stages), 1.22 for the 192-row tile (A in
// TMEM, 2 stages, 2/3 the transform work per flop) and 1.10 for the 256-row tile (A in
// smem, half the transform work but a 64-wide k-tile).  Constants fitted offline on the
// B200 tables of rungs B3, B4 and B8: geomean 1.008x the per-row oracle.  Narrow or
// short problems keep the 128-row tile (more clusters to fill the machine), everything
// that fills the machine takes the 192-row tile, the 256-row tile wins a few very wide
// rows.
inline W8A16Sm100Tile W8A16Sm100PickTile(int M, int N, int K, int num_sms) {
  const W8A16Sm100Tile env = W8A16Sm100TileEnv();
  if (env != W8A16Sm100Tile::kAuto) return env;
  (void)K;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long w_tiles = (N + 255) / 256;
  auto cost = [&](int tile_a, double eff) {
    const long long units = w_tiles * ((M + tile_a - 1) / tile_a);
    return static_cast<double>((units + clusters - 1) / clusters) * tile_a / eff;
  };
  const double c128 = cost(128, 1.00);
  const double c192 = cost(192, 1.22);
  const double c256 = cost(256, 1.10);
  if (c192 <= c128 && c192 <= c256) return W8A16Sm100Tile::k2SM256x192;
  if (c256 < c128) return W8A16Sm100Tile::k2SM256x256;
  return W8A16Sm100Tile::k2SM256x128;
}

template <typename T>
size_t W8A16Fp8GemmSm100WorkspaceBytes(int M, int N, int K, int device_id, int num_sms) {
  switch (W8A16Sm100PickTile(M, N, K, num_sms)) {
    case W8A16Sm100Tile::k1SM128x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Tile::k2SM256x256:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x256>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Tile::k2SM256x192:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x192>::WorkspaceBytes(M, N, K, device_id,
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
  switch (W8A16Sm100PickTile(M, N, K, num_sms)) {
    case W8A16Sm100Tile::k1SM128x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Tile::k2SM256x256:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x256>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Tile::k2SM256x192:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x192>::Run(act, weight, scale, bias, out, M, N, K,
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
