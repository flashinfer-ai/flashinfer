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

// Rung B10: the native packed converter.  Blackwell has a one-instruction e4m3x2 ->
// bf16x2 convert (PTX ISA 9.2, `cvt.rn.bf16x2.e4m3x2`, SASS F2FP.BF16.E4M3.UNPACK_B);
// one F2FP per pair replaces the four-instruction ALU chain below.  PerfSim on rung B9
// (gb100, 2SM 256x192x128, M=192 N=256 K=4096) showed why that chain matters: each
// transform warp needs ~200 instructions and ~750 cycles per 128-deep k-block, the MMA
// takes 768, and the TMEM-store -> mbarrier -> 2-CTA barrier -> MMA-issue handoff adds a
// fixed ~226 cycles, so the convert chain runs ~50 cycles longer than the MMA per
// k-block and the tensor pipe idles 6%.  Halving the chain puts the convert back behind
// the MMA.  Exact for every finite e4m3 code (all are representable in bf16, denormals
// included); byte 0 lands in lo[15:0].
//
// The ALU path is kept for toolchains older than CUDA 13.2 (no PTX 9.2) and for
// -DW8A16_FORCE_ALU_CONVERT (A/B testing).
#if !defined(W8A16_FORCE_ALU_CONVERT) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define W8A16_NATIVE_E4M3_BF16_CVT 1
#else
#define W8A16_NATIVE_E4M3_BF16_CVT 0
#endif

// Four e4m3 bytes (little-endian in `q`) -> two bf16x2 registers, value order kept.
__device__ __forceinline__ void e4m3x4_to_bf16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
#if W8A16_NATIVE_E4M3_BF16_CVT
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(q));
#else
  // Rung B1 ALU path.  prmt selector nibbles 0x9180 / 0xB3A2: byte k into the low byte
  // of lane k and its sign replicated into the high byte (selector bit 3).  (v << 4)
  // puts the 7 magnitude bits at [10:4] and the replicated sign at [15], mask 0x87F0
  // keeps exactly those.  The exponent field then holds e4m3's raw exponent, so
  // multiply by 2^(127 - 7) = 2^120 (bf16 0x7B80).  Denormals: field 0 -> bf16 denormal
  // (m / 8) * 2^-126, times 2^120 = (m / 8) * 2^-6, the e4m3 denormal value.
  const uint32_t lo_s = (prmt_b32(q, 0u, 0x9180u) << 4) & 0x87F087F0u;
  const uint32_t hi_s = (prmt_b32(q, 0u, 0xB3A2u) << 4) & 0x87F087F0u;
  const uint32_t k_bits = 0x7B807B80u;  // bf16x2 {2^120, 2^120}
  const __nv_bfloat162 k = *reinterpret_cast<const __nv_bfloat162*>(&k_bits);
  __nv_bfloat162 l = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&lo_s), k);
  __nv_bfloat162 h = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&hi_s), k);
  lo = *reinterpret_cast<uint32_t*>(&l);
  hi = *reinterpret_cast<uint32_t*>(&h);
#endif
}

// Same for f16.  `cvt.rn.f16x2.e4m3x2` has existed since sm_89 (PTX ISA 7.8), so the
// native form needs no toolchain guard; the ALU path ((v << 7) puts the magnitude at
// [13:7] and the replicated sign at [15], mask 0xBF80, bias fix 2^(15 - 7) = 2^8,
// f16 0x5C00) stays behind W8A16_FORCE_ALU_CONVERT.
__device__ __forceinline__ void e4m3x4_to_f16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
#if !defined(W8A16_FORCE_ALU_CONVERT)
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(q));
#else
  const uint32_t lo_s = (prmt_b32(q, 0u, 0x9180u) << 7) & 0xBF80BF80u;
  const uint32_t hi_s = (prmt_b32(q, 0u, 0xB3A2u) << 7) & 0xBF80BF80u;
  const uint32_t k_bits = 0x5C005C00u;  // f16x2 {256, 256}
  const __half2 k = *reinterpret_cast<const __half2*>(&k_bits);
  __half2 l = __hmul2(*reinterpret_cast<const __half2*>(&lo_s), k);
  __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&hi_s), k);
  lo = *reinterpret_cast<uint32_t*>(&l);
  hi = *reinterpret_cast<uint32_t*>(&h);
#endif
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

// Rung B12: split-K.  With kSplitK the kernel runs as a *batch* of `splits` GEMMs over
// K-slices in one launch (batch stride = the slice length along K for A and B, one
// M x N fp32 partial per batch for D), so an under-filled grid becomes `splits` times as
// many tiles for the persistent scheduler; W8A16Sm100SplitKReduceKernel then sums the
// partials, applies the per-tensor scale and the bias and casts.  Scale and bias are
// deferred to the reduce (alpha = 1 here) so the partials are plain accumulators.
template <typename T, typename Cfg, bool kSplitK = false>
struct W8A16Sm100Gemm {
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

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;

  // `splits` > 1 only with kSplitK: problem {N, M, K / splits, splits}, A and B batch
  // stride = K / splits elements (the next K-slice), D batch stride = M * N (the next
  // fp32 partial in `out`).  Row strides stay the full K, so TMA walks the sliced
  // operands in place.
  static typename Gemm::Arguments MakeArguments(const T* act, const __nv_fp8_e4m3* weight,
                                                const float* scale, const T* bias, void* out,
                                                int M, int N, int K, int device_id, int num_sms,
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
    return args;
  }

  static size_t PartialsBytes(int M, int N, int splits) {
    return kSplitK && splits > 1 ? static_cast<size_t>(splits) * M * N * sizeof(float) : 0;
  }

  // CUTLASS workspace (persistent scheduler: usually 0) rounded to 256 B, then the
  // fp32 partials for split-K.
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

// Tile codes.  The numeric values are the public `algo` codes of the Python op
// (`flashinfer.gemm.w8a16_fp8_gemm(..., algo=...)`), so they must not change.
enum class W8A16Sm100Tile {
  kAuto,
  k2SM256x128 = 1,
  k1SM128x128 = 2,
  k2SM256x256 = 3,
  k2SM256x192 = 4,
  k2SM256x128SplitK = 5  // rung B12: 256x128 tile, batched K-slices + fp32 reduce
};
constexpr int kW8A16Sm100MaxTileCode = 5;
constexpr int kW8A16Sm100MaxSplits = 8;

inline W8A16Sm100Tile W8A16Sm100TileFromCode(int code) {
  return (code >= 1 && code <= kW8A16Sm100MaxTileCode) ? static_cast<W8A16Sm100Tile>(code)
                                                      : W8A16Sm100Tile::kAuto;
}

// FLASHINFER_W8A16_FP8_TILE_SM100 = "2sm256x128" | "1sm128x128" | "2sm256x256" |
// "2sm256x192" | "2sm256x128_splitk" pins the tile; read once.
inline W8A16Sm100Tile W8A16Sm100TileEnv() {
  static const W8A16Sm100Tile tile = [] {
    const char* s = std::getenv("FLASHINFER_W8A16_FP8_TILE_SM100");
    if (s == nullptr || *s == '\0') return W8A16Sm100Tile::kAuto;
    if (std::strcmp(s, "2sm256x128") == 0) return W8A16Sm100Tile::k2SM256x128;
    if (std::strcmp(s, "1sm128x128") == 0) return W8A16Sm100Tile::k1SM128x128;
    if (std::strcmp(s, "2sm256x256") == 0) return W8A16Sm100Tile::k2SM256x256;
    if (std::strcmp(s, "2sm256x192") == 0) return W8A16Sm100Tile::k2SM256x192;
    if (std::strcmp(s, "2sm256x128_splitk") == 0) return W8A16Sm100Tile::k2SM256x128SplitK;
    return W8A16Sm100Tile::kAuto;
  }();
  return tile;
}

// Split count for the 256x128 2SM tile (rung B12).  Two cases split, both from the
// back-to-back per-M sweeps of Mon 12 Oct 2026:
//   1. an under-filled grid, fewer cluster tiles than half the resident clusters:
//      the largest split that still fits in one wave (every cluster gets at most one
//      K-slice; the und.kv rows, 8..64 tiles on 74 clusters, run 1.3-1.8x faster);
//   2. a long K (>= 100 k-tiles) whose partial last wave the split removes: the
//      smallest split that keeps >= 50 k-tiles per slice and cuts the modelled
//      mainloop (waves x k-tiles per slice) by at least 15% (mlp.down at M=1024,
//      K=25600: 160 tiles = 2.2 waves, 8% faster split 4-5 ways).
// Short-K problems never split: the fp32 partials (one M x N write and read per slice)
// cost more than the fill buys (attn.out / gen.qkv / mlp.down at K <= 8192 lose 5-25%).
// Every K-slice is a whole number of 128-deep k-tiles (all batches share one slice
// length) and at most kW8A16Sm100MaxSplits slices.  1 = do not split.
inline int W8A16Sm100SplitK(int M, int N, int K, int num_sms) {
  if (K % 128 != 0) return 1;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long units = static_cast<long long>((N + 255) / 256) * ((M + 127) / 128);
  const int ktiles = K / 128;
  if (units * 2 <= clusters) {
    int best = 1;
    for (int s = 2; s <= kW8A16Sm100MaxSplits; ++s) {
      if (ktiles % s != 0 || ktiles / s < 2 || units * s > clusters) continue;
      best = s;
    }
    return best;
  }
  if (ktiles >= 100) {
    const long long waves1 = (units + clusters - 1) / clusters;
    for (int s = 2; s <= kW8A16Sm100MaxSplits; ++s) {
      if (ktiles % s != 0 || ktiles / s < 50) continue;
      const long long waves = (units * s + clusters - 1) / clusters;
      if (waves * (ktiles / s) * 100 <= waves1 * ktiles * 85) return s;
    }
  }
  return 1;
}

// Split count actually used by the split-K variant: the model's choice, or, when the
// variant is forced through `algo` on a grid the model would not split, the smallest
// s >= 2 that divides the k-tile count.  0 = K cannot be split (fewer than 2 k-tiles).
inline int W8A16Sm100SplitsFor(int M, int N, int K, int num_sms) {
  const int s = W8A16Sm100SplitK(M, N, K, num_sms);
  if (s >= 2) return s;
  const int ktiles = K / 128;
  if (K % 128 != 0 || ktiles < 2) return 0;
  for (int c = 2; c <= kW8A16Sm100MaxSplits; ++c) {
    if (ktiles % c == 0) return c;
  }
  return 0;
}

// `algo` is the caller's tile code (0 = auto); an explicit code wins over the env
// override, which wins over the cost model.
inline W8A16Sm100Tile W8A16Sm100PickTile(int M, int N, int K, int num_sms, int algo = 0) {
  const W8A16Sm100Tile forced = W8A16Sm100TileFromCode(algo);
  if (forced != W8A16Sm100Tile::kAuto) return forced;
  const W8A16Sm100Tile env = W8A16Sm100TileEnv();
  if (env != W8A16Sm100Tile::kAuto) return env;
  // Rung B12: under-filled grids and long-K partial waves split K instead of choosing
  // between tiles; see W8A16Sm100SplitK.
  if (W8A16Sm100SplitK(M, N, K, num_sms) > 1) return W8A16Sm100Tile::k2SM256x128SplitK;

  // Rung B11 refit (native converter, rung B10): with the convert off the critical path
  // the 192-row tile's advantage shrinks to a few percent (eff 1.22 -> 1.02) and the
  // 256x256 smem-A tile loses on every one of the 144 benchmark rows, so it is no
  // longer picked (it stays reachable through `algo`).  Fitted on three back-to-back
  // sweeps of all tiles (geomean model pick / per-row oracle 1.025, the B9 constants
  // gave 1.040).  The 1SM 128x128 tile never wins once the per-M classes are measured
  // in isolation (its apparent wins were thermal noise of the full-table sweep), so it
  // is `algo`-only as well.
  (void)K;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long w_tiles = (N + 255) / 256;
  auto cost = [&](int tile_a, double eff) {
    const long long units = w_tiles * ((M + tile_a - 1) / tile_a);
    return static_cast<double>((units + clusters - 1) / clusters) * tile_a / eff;
  };
  const double c128 = cost(128, 1.00);
  const double c192 = cost(192, 1.02);
  if (c192 <= c128) return W8A16Sm100Tile::k2SM256x192;
  return W8A16Sm100Tile::k2SM256x128;
}

template <typename T>
size_t W8A16Fp8GemmSm100WorkspaceBytes(int M, int N, int K, int device_id, int num_sms,
                                       int algo = 0) {
  switch (W8A16Sm100PickTile(M, N, K, num_sms, algo)) {
    case W8A16Sm100Tile::k2SM256x128SplitK:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, true>::WorkspaceBytes(
          M, N, K, device_id, num_sms, std::max(2, W8A16Sm100SplitsFor(M, N, K, num_sms)));
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
                              cudaStream_t stream, int algo = 0) {
  switch (W8A16Sm100PickTile(M, N, K, num_sms, algo)) {
    case W8A16Sm100Tile::k2SM256x128SplitK: {
      const int splits = W8A16Sm100SplitsFor(M, N, K, num_sms);
      if (splits < 2) return cudaErrorNotSupported;
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, true>::Run(
          act, weight, scale, bias, out, M, N, K, workspace, workspace_bytes, device_id, num_sms,
          stream, splits);
    }
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
