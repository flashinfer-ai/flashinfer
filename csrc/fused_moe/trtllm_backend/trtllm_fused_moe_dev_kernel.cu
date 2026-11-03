/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function for array conversion
template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
  return converter(input);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool isAligned16(void const* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 15u) == 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace activation {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float silu(float x) { return x / (1.0f + expf(-x)); }

////////////////////////////////////////////////////////////////////////////////////////////////////

// Gated SwiGLU with the optional OAI controls, matching the fused FC1 epilogue of the trtllm-gen
// cubins:
//   xGlu    = clamp(xGlu, max=limit)
//   xLinear = clamp(xLinear, -limit, limit)
//   out     = xGlu * sigmoid(alpha * xGlu) * (xLinear + beta)
// With no parameters set this is exactly silu(xGlu) * xLinear, bit-for-bit, so the common path is
// unchanged. The per-expert scalars are looked up per element rather than hoisted: they are the
// same address for every thread of a tile and stay resident in L1, and this kernel is bandwidth
// bound on the FC1 output anyway.
template <typename KernelParams>
inline __device__ float gatedSilu(KernelParams const& params, int32_t permutedIdx, float xLinear,
                                  float xGlu) {
  if (params.gatedActAlphaPtr == nullptr && params.gatedActBetaPtr == nullptr &&
      params.gatedActClampLimitPtr == nullptr) {
    return silu(xGlu) * xLinear;
  }

  // Permuted tokens are grouped per expert and padded to tileTokensDim, so the tile index is the
  // FC1 GEMM's batch index. tileTokensDim is a runtime value (not always a power of two), hence
  // the division; it is off the default path and negligible next to the global loads.
  int32_t const localExpertIdx = params.ctaIdxXyToBatchIdx[permutedIdx / params.tileTokensDim];

  if (params.gatedActClampLimitPtr != nullptr) {
    float const limit = params.gatedActClampLimitPtr[localExpertIdx];
    xGlu = fminf(xGlu, limit);
    xLinear = fmaxf(fminf(xLinear, limit), -limit);
  }
  float const alpha =
      params.gatedActAlphaPtr != nullptr ? params.gatedActAlphaPtr[localExpertIdx] : 1.0f;
  float const beta =
      params.gatedActBetaPtr != nullptr ? params.gatedActBetaPtr[localExpertIdx] : 0.0f;
  // x * sigmoid(alpha * x), i.e. silu(x) generalized by alpha.
  float const act = xGlu / (1.0f + expf(-alpha * xGlu));
  return act * (xLinear + beta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationKernel(KernelParams params) {
  using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.z; tokenIdx < params.numTokens; tokenIdx += gridDim.z) {
    // Look over experts per token
    for (int k = blockIdx.y; k < params.topK; k += gridDim.y) {
      int const expandedIdx = tokenIdx * params.topK + k;
      int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
      if (permutedIdx == -1) continue;

      // Loop over hidden dim
      for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
           hiddenIdx += blockDim.x * gridDim.x) {
        // Use int64_t to avoid overflow when permutedIdx * innerDim > INT32_MAX
        int64_t const baseIdx = (int64_t)permutedIdx * params.innerDim + hiddenIdx;

        float x1 = (float)params.inPtr[baseIdx];
        float x2 = (float)params.inPtr[baseIdx + params.innerDim / 2];

        Type out = (Type)gatedSilu(params, permutedIdx, x1, x2);

        int64_t const outIdx = (int64_t)permutedIdx * (params.innerDim / 2) + hiddenIdx;
        params.outPtr[outIdx] = out;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Float4Max {
  __device__ __forceinline__ float4 operator()(float4 const& a, float4 const& b) const {
    float4 result;
    result.x = fmaxf(a.x, b.x);
    result.y = fmaxf(a.y, b.y);
    result.z = fmaxf(a.z, b.z);
    result.w = fmaxf(a.w, b.w);
    return result;
  }
};

struct Float2Max {
  __device__ __forceinline__ float2 operator()(float2 const& a, float2 const& b) const {
    float2 result;
    result.x = fmaxf(a.x, b.x);
    result.y = fmaxf(a.y, b.y);
    return result;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VecType, int size>
__device__ __forceinline__ VecType packedTypeFromArray(float data[size]) {
  return {};
}

template <>
__device__ __forceinline__ float4 packedTypeFromArray<float4, 4>(float data[4]) {
  float4 result;
  result.x = data[0];
  result.y = data[1];
  result.z = data[2];
  result.w = data[3];
  return result;
}

template <>
__device__ __forceinline__ float2 packedTypeFromArray<float2, 2>(float data[2]) {
  float2 result;
  result.x = data[0];
  result.y = data[1];
  return result;
}

template <>
__device__ __forceinline__ float packedTypeFromArray<float, 1>(float data[1]) {
  return data[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PackedType, int size>
__device__ __forceinline__ cutlass::Array<float, size> arrayFromPackedType(PackedType data) {
  return cutlass::Array<float, size>{};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 4> arrayFromPackedType<float4, 4>(float4 data) {
  return cutlass::Array<float, 4>{data.x, data.y, data.z, data.w};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 2> arrayFromPackedType<float2, 2>(float2 data) {
  return cutlass::Array<float, 2>{data.x, data.y};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 1> arrayFromPackedType<float, 1>(float data) {
  return cutlass::Array<float, 1>{data};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_TOKENS_PER_CTA>
struct KernelTraits;

template <>
struct KernelTraits<4> {
  using MaxOp = Float4Max;
  using PackedType = float4;
};

template <>
struct KernelTraits<2> {
  using MaxOp = Float2Max;
  using PackedType = float2;
};

template <>
struct KernelTraits<1> {
  using MaxOp = cuda::maximum<>;
  using PackedType = float;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA = 128;

template <typename KernelParams>
__global__ void activationDeepSeekKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  int32_t constexpr NumTokensPerCta = KernelParams::NumTokensPerCta;
  using KernelTraits = KernelTraits<NumTokensPerCta>;
  using MaxOp = typename KernelTraits::MaxOp;
  using PackedType = typename KernelTraits::PackedType;
  using BlockReduce = cub::BlockReduce<PackedType, DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA>;

  __shared__ float s_scaleOutArr[NumTokensPerCta];
  __shared__ typename BlockReduce::TempStorage tempStorage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // The largest (finite) value that can be represented using E4m3.
  float constexpr E4m3MaxVal{448.f};

  int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
  // Scale block is 128 or 32 output elements; both are powers of two.
  bool const sfBlockIsCta = params.deepSeekSfBlockSize == DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA;
  int const sfBlockLog2 = sfBlockIsCta ? 7 : 5;
  // Loop over tokens
  float scale1Arr[NumTokensPerCta];
  float scale2Arr[NumTokensPerCta];
  float dataX1Arr[NumTokensPerCta];
  float dataX2Arr[NumTokensPerCta];
  float outArr[NumTokensPerCta];
  float absOutArr[NumTokensPerCta];
  int permutedIdxArr[NumTokensPerCta];

  // Loop over tokens
  for (int k = blockIdx.z; k < params.topK; k += gridDim.z) {
    for (int tokenCtaIdx = blockIdx.y * NumTokensPerCta; tokenCtaIdx < params.numTokens;
         tokenCtaIdx += gridDim.y * NumTokensPerCta) {
      for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
           hiddenIdx += blockDim.x * gridDim.x) {
#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
          scale1Arr[tokenInCtaIdx] = 0.0f;
          scale2Arr[tokenInCtaIdx] = 0.0f;
          dataX1Arr[tokenInCtaIdx] = 0.0f;
          dataX2Arr[tokenInCtaIdx] = 0.0f;
          outArr[tokenInCtaIdx] = 0.0f;
          absOutArr[tokenInCtaIdx] = 0.0f;
          // The loop below stops writing this at the tail of the token range, and the
          // activation loop needs a defined value to decide whether a per-expert lookup
          // is in range.
          permutedIdxArr[tokenInCtaIdx] = -1;
        }
#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
          int const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
          if (tokenIdx >= params.numTokens) {
            break;
          }

          int const expandedIdx = tokenIdx * params.topK + k;
          int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
          permutedIdxArr[tokenInCtaIdx] = permutedIdx;
          if (permutedIdx == -1) {
            continue;
          }

          // Process blocks for this CTA
          // Use int64_t to avoid overflow when permutedIdx * innerDim > INT32_MAX
          int64_t const baseIdx = (int64_t)permutedIdx * params.innerDim + hiddenIdx;

          int64_t const scale1Idx =
              (int64_t)permutedIdx + (int64_t)totalNumPaddedTokens * (hiddenIdx >> sfBlockLog2);
          int64_t const scale2Idx =
              (int64_t)permutedIdx +
              (int64_t)totalNumPaddedTokens *
                  ((hiddenIdx >> sfBlockLog2) + ((params.innerDim / 2) >> sfBlockLog2));

          scale1Arr[tokenInCtaIdx] = params.inDqSfsPtr[scale1Idx];
          scale2Arr[tokenInCtaIdx] = params.inDqSfsPtr[scale2Idx];
          dataX1Arr[tokenInCtaIdx] = static_cast<float>(params.inPtr[baseIdx]);
          dataX2Arr[tokenInCtaIdx] =
              static_cast<float>(params.inPtr[baseIdx + params.innerDim / 2]);
        }

#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
          float x1 = scale1Arr[tokenInCtaIdx] * dataX1Arr[tokenInCtaIdx];
          float x2 = scale2Arr[tokenInCtaIdx] * dataX2Arr[tokenInCtaIdx];
          // Padding lanes keep permutedIdx == -1 and contribute a zeroed x1/x2, so skip the
          // per-expert lookup for them instead of indexing ctaIdxXyToBatchIdx out of range.
          int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
          float out = permutedIdx == -1 ? silu(x2) * x1 : gatedSilu(params, permutedIdx, x1, x2);
          outArr[tokenInCtaIdx] = out;
          absOutArr[tokenInCtaIdx] = fabsf(out);
        }

        if (sfBlockIsCta) {
          // One 128-wide scale block per CTA. innerDim / 2 is a multiple of 128, so every
          // thread reaches the CTA-wide reduction and barrier below.
          auto absOutPacked = packedTypeFromArray<PackedType, NumTokensPerCta>(absOutArr);
          auto aMaxPacked = BlockReduce(tempStorage).Reduce(absOutPacked, MaxOp{});
          auto aMaxArr = arrayFromPackedType<PackedType, NumTokensPerCta>(aMaxPacked);

#pragma unroll
          for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
            if (threadIdx.x == 0) {
              auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
              if (tokenIdx >= params.numTokens) {
                break;
              }
              int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
              if (permutedIdx == -1) {
                continue;
              }
              // Make sure the scale is strictly positive to avoid division by zero in case the
              // maximum is zero.
              float scaleOut =
                  fmaxf(aMaxArr[tokenInCtaIdx] / E4m3MaxVal, std::numeric_limits<float>::min());
              s_scaleOutArr[tokenInCtaIdx] = scaleOut;
              int64_t const scaleOut_idx = (int64_t)permutedIdxArr[tokenInCtaIdx] +
                                           (int64_t)totalNumPaddedTokens * (hiddenIdx >> 7);
              params.outDqSfsPtr[scaleOut_idx] = scaleOut;
            }
          }
          __syncthreads();

#pragma unroll
          for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
            auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
            if (tokenIdx >= params.numTokens) {
              break;
            }
            int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
            if (permutedIdx == -1) {
              continue;
            }
            float const scaleOut = s_scaleOutArr[tokenInCtaIdx];
            int64_t const outIdx = (int64_t)permutedIdx * (params.innerDim / 2) + hiddenIdx;
            params.outPtr[outIdx] = static_cast<Type>(outArr[tokenInCtaIdx] / scaleOut);
          }
        } else {
          // One 32-wide scale block per warp. innerDim / 2 is a multiple of 32 but not
          // necessarily of 128, so a CTA may have idle warps: reduce with warp shuffles
          // and never touch a CTA-wide barrier.
          float scaleOutArr[NumTokensPerCta];
#pragma unroll
          for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
            float aMax = absOutArr[tokenInCtaIdx];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
              aMax = fmaxf(aMax, __shfl_xor_sync(0xffffffffu, aMax, offset));
            }
            scaleOutArr[tokenInCtaIdx] =
                fmaxf(aMax / E4m3MaxVal, std::numeric_limits<float>::min());
          }
          int const laneIdx = threadIdx.x % 32;
#pragma unroll
          for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
            auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
            if (tokenIdx >= params.numTokens) {
              break;
            }
            int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
            if (permutedIdx == -1) {
              continue;
            }
            float const scaleOut = scaleOutArr[tokenInCtaIdx];
            if (laneIdx == 0) {
              int64_t const scaleOut_idx =
                  (int64_t)permutedIdx + (int64_t)totalNumPaddedTokens * (hiddenIdx >> 5);
              params.outDqSfsPtr[scaleOut_idx] = scaleOut;
            }
            int64_t const outIdx = (int64_t)permutedIdx * (params.innerDim / 2) + hiddenIdx;
            params.outPtr[outIdx] = static_cast<Type>(outArr[tokenInCtaIdx] / scaleOut);
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Row-parallel DeepSeek FP8 activation: SwiGLU on the FC1 output and re-quantization to e4m3 with
// one fp32 scale per 128 columns.
//
// activationDeepSeekKernel above walks (token, top-k slot) pairs: one byte per thread per load, a
// cub block reduce and three barriers per slot, and under expert parallelism most slots are not on
// this GPU (expandedIdxToPermutedIdx == -1) yet still cost the reduce. This kernel walks the
// permuted rows instead (the local pairs plus the tile padding, 1/EP of the pairs): a group of
// kThreadsPerSfBlock lanes owns one 128-column scale block of one row, each lane 16 consecutive
// bytes of it (one 16-byte load from the linear half, one from the gate half, one 16-byte store),
// and the block amax is a register max followed by log2(kThreadsPerSfBlock) shuffles. No shared
// memory, no block barrier, and the grid is sized by the row count instead of numTokens x topK.
//
// Numerics are those of activationDeepSeekKernel, operation for operation: x = scale * float(e4m3),
// silu(x2) * x1 (gatedSilu with the OAI controls), scale = max(amax / 448, FLT_MIN) and
// e4m3(out / scale) with the saturating round-to-nearest-even conversion. Every row the old kernel
// wrote gets the same bytes. NaN inputs: the block amax is reduced with fmaxf, which drops a NaN,
// as the old kernel's 2- and 4-token instantiations do; its 1-token instantiation reduced with
// cuda::maximum<>, which keeps a NaN only in the left operand, so with a NaN in GEMM1's output that
// variant and this kernel can differ in that row. Padding rows inside an expert's last tile
// (permutedIdxToTokenIdx == -1), which the old kernel left with stale bytes, get zeros and the
// FLT_MIN scale: GEMM2 reads them as tile rows but nobody reads its result rows, and a defined
// buffer is cheaper to reason about. Rows at or past *totalNumPaddedTokens are not touched.

constexpr int DEEP_SEEK_ACTIVATION_V2_THREADS_PER_CTA = 128;
constexpr int DEEP_SEEK_ACTIVATION_V2_NUM_ELTS_PER_SF = 128;
// Grid cap: one resident wave at 128 threads and up to 64 registers per thread (52-56 in the
// sm_100a build), so the CTAs that find their row past the live count never wait for a slot.
constexpr int DEEP_SEEK_ACTIVATION_V2_MAX_CTAS_PER_SM = 8;
// Smallest token count for which this kernel is selected; below it activationDeepSeekKernel runs
// as before. Its 1- and 2-token instantiations finish in 2.1-3.3 us up to 16 tokens where this
// kernel takes 3.7-5.1 us. Activation plus finalize cross over between 32 and 64 tokens in every
// parallel split measured; this kernel alone is at parity at 32 and ahead from 64 tokens for EP8,
// crosses between 32 and 64 for EP4 x TP2, between 64 and 128 for EP2 x TP4 and between 128 and
// 256 for TP8, where the finalize's gain covers its 0.5-1.1 us loss at 64. Own times (PDL off),
// old / new us, GLM-5.3 shapes (H 6144, I 2048, 256 experts, top-k 8): B300, 148 SMs at 2032 MHz,
// 2026-11-03, EP8: 2.7 / 4.0 at T = 8, 4.3 / 4.3 at 32, 7.1 / 4.5 at 64, 9.6 / 4.8 at 128; TP8
// 2.4 / 4.8 at 32 and 3.9 / 5.0 at 64; Rubin sm107, 2026-11-03, EP8: 2.6 / 4.3 at 8, 4.6 / 4.3 at
// 32, 5.8 / 4.3 at 64, 8.0 / 4.6 at 128.
constexpr int DEEP_SEEK_ACTIVATION_V2_MIN_TOKENS = 64;

// Loads one lane's 16 bytes of the linear half and of the gate half of permuted row `row`, column
// block `colBlock`, and the two block scales.
template <typename Type>
__device__ __forceinline__ void loadDeepSeekActivationInputs(
    Type const* inPtr, float const* inDqSfsPtr, int row, int colBlock, int col, int innerDim,
    int outDim, int totalNumPaddedTokens, int numSfBlocksPerRow, uint4& x1Raw, uint4& x2Raw,
    float& scale1, float& scale2) {
  // Use int64_t to avoid overflow when row * innerDim > INT32_MAX
  int64_t const rowBase = (int64_t)row * innerDim;
  // The linear term is the first half of the FC1 row, the gate the second half; the gate's scale
  // blocks follow the linear ones in the column-block-major scale tensor.
  x1Raw = *reinterpret_cast<uint4 const*>(inPtr + rowBase + col);
  x2Raw = *reinterpret_cast<uint4 const*>(inPtr + rowBase + outDim + col);
  scale1 = inDqSfsPtr[(int64_t)row + (int64_t)totalNumPaddedTokens * colBlock];
  scale2 =
      inDqSfsPtr[(int64_t)row + (int64_t)totalNumPaddedTokens * (colBlock + numSfBlocksPerRow)];
}

template <typename KernelParams>
__global__ void __launch_bounds__(DEEP_SEEK_ACTIVATION_V2_THREADS_PER_CTA)
    activationDeepSeekKernelV2(KernelParams params) {
  using Type = typename KernelParams::Type;
  // 16 bytes per lane per load / store.
  constexpr int kEltsPerVec = 16 / sizeof(Type);
  constexpr int kThreadsPerSfBlock = DEEP_SEEK_ACTIVATION_V2_NUM_ELTS_PER_SF / kEltsPerVec;
  constexpr int kSfBlocksPerCta = DEEP_SEEK_ACTIVATION_V2_THREADS_PER_CTA / kThreadsPerSfBlock;
  static_assert(kThreadsPerSfBlock >= 1 && kThreadsPerSfBlock <= 32 &&
                    (kThreadsPerSfBlock & (kThreadsPerSfBlock - 1)) == 0,
                "a scale block must map to a power-of-two number of lanes within one warp");
  static_assert(kEltsPerVec * sizeof(Type) == 16, "a lane owns one 16-byte vector");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // The largest (finite) value that can be represented using E4m3.
  float constexpr E4m3MaxVal{448.f};

  int const outDim = params.innerDim / 2;
  int const numSfBlocksPerRow = outDim / DEEP_SEEK_ACTIVATION_V2_NUM_ELTS_PER_SF;
  int const totalMaxPaddedTokens = params.totalMaxPaddedTokens;
  int const groupIdx = threadIdx.x / kThreadsPerSfBlock;
  int const laneInGroup = threadIdx.x % kThreadsPerSfBlock;
  // Lanes of this scale-block group: the shuffles below only involve them, because groups of one
  // warp can sit in different rows and leave the loop at different times.
  unsigned const groupMask =
      kThreadsPerSfBlock == 32
          ? 0xffffffffu
          : ((1u << kThreadsPerSfBlock) - 1u)
                << ((threadIdx.x % 32) / kThreadsPerSfBlock * kThreadsPerSfBlock);

  // CTA index -> (row, column block). A CTA covers kSfBlocksPerCta scale blocks: whole rows when
  // a row has at most that many (rowsPerCta of them; the surplus groups idle when the row count
  // does not divide), otherwise a kSfBlocksPerCta-wide slice of one row (ctasPerRow CTAs per row;
  // the surplus groups of the last slice idle). All 32-bit: a 64-bit division by a runtime value
  // costs more than a memory round trip at decode sizes.
  bool const wholeRows = numSfBlocksPerRow <= kSfBlocksPerCta;
  int const rowsPerCta = wholeRows ? kSfBlocksPerCta / numSfBlocksPerRow : 1;
  int const ctasPerRow =
      wholeRows ? 1 : (numSfBlocksPerRow + kSfBlocksPerCta - 1) / kSfBlocksPerCta;
  int const localRow = groupIdx / numSfBlocksPerRow;  // 0 for sliced rows
  int const localColBlock = groupIdx - localRow * numSfBlocksPerRow;
  if (localRow >= rowsPerCta) {
    return;  // idle group; uniform within the group, and nothing below synchronizes the CTA
  }
  int const ctaStride = gridDim.x;

  // Software pipeline over this CTA's rows: the token index is fetched two rows ahead, the data
  // one row ahead, so each row costs one memory round trip at most and the first row two (the live
  // count and the first two token indices go out together, then the first row's data).
  int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
  int cta0 = blockIdx.x;
  int row0 = (cta0 / ctasPerRow) * rowsPerCta + localRow;
  int colBlock0 = (cta0 % ctasPerRow) * kSfBlocksPerCta + localColBlock;
  int cta1 = cta0 + ctaStride;
  int row1 = (cta1 / ctasPerRow) * rowsPerCta + localRow;
  int colBlock1 = (cta1 % ctasPerRow) * kSfBlocksPerCta + localColBlock;
  // permutedIdxToTokenIdx has totalMaxPaddedTokens + 1 entries: any row < totalMaxPaddedTokens is
  // in bounds, live or not.
  int tokenIdx0 = row0 < totalMaxPaddedTokens ? params.permutedIdxToTokenIdx[row0] : -1;
  int tokenIdx1 = row1 < totalMaxPaddedTokens ? params.permutedIdxToTokenIdx[row1] : -1;
  uint4 x1Raw = make_uint4(0u, 0u, 0u, 0u);
  uint4 x2Raw = make_uint4(0u, 0u, 0u, 0u);
  float scale1 = 0.0f;
  float scale2 = 0.0f;
  if (row0 < totalNumPaddedTokens && colBlock0 < numSfBlocksPerRow && tokenIdx0 != -1) {
    loadDeepSeekActivationInputs<Type>(
        params.inPtr, params.inDqSfsPtr, row0, colBlock0,
        colBlock0 * DEEP_SEEK_ACTIVATION_V2_NUM_ELTS_PER_SF + laneInGroup * kEltsPerVec,
        params.innerDim, outDim, totalNumPaddedTokens, numSfBlocksPerRow, x1Raw, x2Raw, scale1,
        scale2);
  }

  // Rows grow with the CTA index, so once this group's row is past the live count all later ones
  // are too.
  while (row0 < totalNumPaddedTokens) {
    // Prefetch: token index two rows ahead, data one row ahead.
    int const cta2 = cta1 + ctaStride;
    int const row2 = (cta2 / ctasPerRow) * rowsPerCta + localRow;
    int const colBlock2 = (cta2 % ctasPerRow) * kSfBlocksPerCta + localColBlock;
    int const tokenIdx2 = row2 < totalMaxPaddedTokens ? params.permutedIdxToTokenIdx[row2] : -1;
    uint4 x1Next = make_uint4(0u, 0u, 0u, 0u);
    uint4 x2Next = make_uint4(0u, 0u, 0u, 0u);
    float scale1Next = 0.0f;
    float scale2Next = 0.0f;
    if (row1 < totalNumPaddedTokens && colBlock1 < numSfBlocksPerRow && tokenIdx1 != -1) {
      loadDeepSeekActivationInputs<Type>(
          params.inPtr, params.inDqSfsPtr, row1, colBlock1,
          colBlock1 * DEEP_SEEK_ACTIVATION_V2_NUM_ELTS_PER_SF + laneInGroup * kEltsPerVec,
          params.innerDim, outDim, totalNumPaddedTokens, numSfBlocksPerRow, x1Next, x2Next,
          scale1Next, scale2Next);
    }

    if (colBlock0 < numSfBlocksPerRow) {
      bool const isPadding = tokenIdx0 == -1;
      Type const* x1Vec = reinterpret_cast<Type const*>(&x1Raw);
      Type const* x2Vec = reinterpret_cast<Type const*>(&x2Raw);
      float out[kEltsPerVec];
      float aMax = 0.0f;
#pragma unroll
      for (int i = 0; i < kEltsPerVec; ++i) {
        float const x1 = scale1 * static_cast<float>(x1Vec[i]);
        float const x2 = scale2 * static_cast<float>(x2Vec[i]);
        // Padding rows carry zeros; skip the per-expert lookup for them as the old kernel does.
        out[i] = isPadding ? silu(x2) * x1 : gatedSilu(params, row0, x1, x2);
        aMax = fmaxf(aMax, fabsf(out[i]));
      }
#pragma unroll
      for (int offset = kThreadsPerSfBlock / 2; offset > 0; offset /= 2) {
        aMax = fmaxf(aMax, __shfl_xor_sync(groupMask, aMax, offset));
      }
      // Make sure the scale is strictly positive to avoid division by zero in case the maximum is
      // zero.
      float const scaleOut = fmaxf(aMax / E4m3MaxVal, std::numeric_limits<float>::min());
      uint4 outRaw;
      Type* outVec = reinterpret_cast<Type*>(&outRaw);
#pragma unroll
      for (int i = 0; i < kEltsPerVec; ++i) {
        outVec[i] = static_cast<Type>(out[i] / scaleOut);
      }
      int const col =
          colBlock0 * DEEP_SEEK_ACTIVATION_V2_NUM_ELTS_PER_SF + laneInGroup * kEltsPerVec;
      *reinterpret_cast<uint4*>(params.outPtr + (int64_t)row0 * outDim + col) = outRaw;
      if (laneInGroup == 0) {
        params.outDqSfsPtr[(int64_t)row0 + (int64_t)totalNumPaddedTokens * colBlock0] = scaleOut;
      }
    }

    cta0 = cta1;
    row0 = row1;
    colBlock0 = colBlock1;
    tokenIdx0 = tokenIdx1;
    x1Raw = x1Next;
    x2Raw = x2Next;
    scale1 = scale1Next;
    scale2 = scale2Next;
    cta1 = cta2;
    row1 = row2;
    colBlock1 = colBlock2;
    tokenIdx1 = tokenIdx2;
  }
}

// Launches the row-parallel activation with NumTokensPerCta fixed to 1 (the parameter is unused
// there), i.e. one instantiation per element type and PDL mode.
#define LAUNCH_ACTIVATION_V2(data, kernel, numBlocks, numThreads, smemSize, stream)              \
  if (data.mDtypeElt == tg::Dtype::Fp16) {                                                       \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::half_t, 1), kernel, numBlocks, numThreads,       \
               smemSize, stream);                                                                \
  } else if (data.mDtypeElt == tg::Dtype::E4m3) {                                                \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::float_e4m3_t, 1), kernel, numBlocks, numThreads, \
               smemSize, stream);                                                                \
  } else if (data.mDtypeElt == tg::Dtype::Bfloat16) {                                            \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::bfloat16_t, 1), kernel, numBlocks, numThreads,   \
               smemSize, stream);                                                                \
  } else {                                                                                       \
    FLASHINFER_WARN("Unsupported dtypeElt");                                                     \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  if (data.mDtypeElt == tg::Dtype::E2m1) {
    // Note: this should be unreachable because the options are checked beforehand.
    // E2m1 requires using higher-precision intermediate data (bf16).
    FLASHINFER_CHECK(false, "Activation with E2m1_t isn't supported.");
    return;
  }

  if (data.mUseDeepSeekFp8) {
    constexpr int NUM_ELTS_PER_LOAD = 1;
    constexpr int NUM_ELTS_PER_CTA = DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA;
    // The scale block the row-parallel kernel below is written for.
    constexpr int NUM_ELTS_PER_SF = DEEP_SEEK_ACTIVATION_V2_NUM_ELTS_PER_SF;
    FLASHINFER_CHECK(data.mDeepSeekSfBlockSize == 128 || data.mDeepSeekSfBlockSize == 32,
                     "DeepSeek FP8 activation: scale block must be 128 or 32.");

    int device{-1};
    cudaGetDevice(&device);
    int numSms = 0;
    cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

    // Output dimension is innerDim / 2; each CTA covers 128 elements, i.e. one scale block
    // at block 128 or four warp-owned blocks at block 32.
    int const outputDim = data.innerDim / 2;

    // The row-parallel kernel needs the row -> token map and the row-count bound (callers that
    // build activation::Data without them, such as the standalone activation op, keep the old
    // kernel), the 128-column scale block it is written for (a 32-column block stays on the old
    // kernel), at least one whole scale block (the scale indexing of both kernels assumes whole
    // blocks; the row split divides by their count), 16-byte aligned rows and at least
    // DEEP_SEEK_ACTIVATION_V2_MIN_TOKENS tokens (below that the old kernel's small-batch
    // instantiations are faster; see the constant).
    int const eltBytes = tg::dtypeGetNumBits(data.mDtypeElt) / 8;
    bool const useRowParallel =
        data.permutedIdxToTokenIdx != nullptr && data.totalMaxPaddedTokens > 0 &&
        data.numTokens >= DEEP_SEEK_ACTIVATION_V2_MIN_TOKENS && outputDim > 0 &&
        data.mDeepSeekSfBlockSize == NUM_ELTS_PER_SF && outputDim % NUM_ELTS_PER_SF == 0 &&
        (outputDim * eltBytes) % 16 == 0 && isAligned16(data.inPtr) && isAligned16(data.outPtr);
    if (useRowParallel) {
      int const numSfBlocksPerRow = outputDim / NUM_ELTS_PER_SF;
      int const sfBlocksPerCta = DEEP_SEEK_ACTIVATION_V2_THREADS_PER_CTA /
                                 (NUM_ELTS_PER_SF / (16 / eltBytes));  // 16 for e4m3
      // A CTA covers whole rows (sfBlocksPerCta / numSfBlocksPerRow of them) or, for rows wider
      // than a CTA, one sfBlocksPerCta-wide slice of a row; see the kernel. The live row count is
      // only known on the device, so the grid is sized by the allocation bound, capped at one
      // resident wave, and the kernel grid-strides and stops at the live count.
      int64_t const numCtasNeeded =
          numSfBlocksPerRow <= sfBlocksPerCta
              ? ((int64_t)data.totalMaxPaddedTokens + (sfBlocksPerCta / numSfBlocksPerRow) - 1) /
                    (sfBlocksPerCta / numSfBlocksPerRow)
              : (int64_t)data.totalMaxPaddedTokens *
                    ((numSfBlocksPerRow + sfBlocksPerCta - 1) / sfBlocksPerCta);
      int const numCtas = static_cast<int>(std::min<int64_t>(
          numCtasNeeded, (int64_t)numSms * DEEP_SEEK_ACTIVATION_V2_MAX_CTAS_PER_SM));
      LAUNCH_ACTIVATION_V2(data, activationDeepSeekKernelV2, dim3(numCtas),
                           DEEP_SEEK_ACTIVATION_V2_THREADS_PER_CTA, 0, stream);
      return;
    }

    int const numCtaBlocks = (outputDim + NUM_ELTS_PER_CTA - 1) / NUM_ELTS_PER_CTA;
    int const gridSizeX = (numCtaBlocks + NUM_ELTS_PER_LOAD - 1) / NUM_ELTS_PER_LOAD;

    auto numCtas = gridSizeX * data.numTokens * data.topK;
    // FIXME: This is heruistic based on very short benchmark.
    int numTokensPerCta = 1;
    if (numCtas > numSms * 32) {
      numTokensPerCta = 4;
    } else if (numCtas > numSms * 4) {
      numTokensPerCta = 2;
    } else {
      numTokensPerCta = 1;
    }

    int const gridSizeY = std::min(8192, (data.numTokens + numTokensPerCta - 1) / numTokensPerCta);

    const dim3 grid(gridSizeX, gridSizeY, data.topK);

    LAUNCH_ACTIVATION(data, activationDeepSeekKernel, numTokensPerCta, grid,
                      DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA, 0, stream);
  } else {
    int const numThreads = 256;
    const dim3 grid(data.innerDim / 128, data.topK, std::min(8192, data.numTokens));

    LAUNCH_ACTIVATION(data, activationKernel, 1, grid, numThreads, 0, stream);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace activation

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace convertsf {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

namespace dev {
// Compute the offset that corresponds to (dataRowIdx, dataBlkColIdx) in the SF tensor where
// dataRowIdx and dataBlkColIdx are the respective indices of the row and the block of 16 elts
// from the K dim in the tensor of data.
inline __device__ int64_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx,
                                      int32_t numDataBlksPerRow) {
  // The number of rows of SF per block.
  static int32_t constexpr NumRowsPerSfBlock = 128;
  // The number of cols of SF per block.
  static int32_t constexpr NumColsPerSfBlock = 4;
  // The size of each SF block.
  static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

  // The number of rows of data per SF block.
  static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
  // The number of cols of blocks of data per SF block.
  static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;

  // The row of the SF block in the SF tensor.
  int sfBlkRowIdx = dataRowIdx / NumDataRowsPerSfBlock;
  // The col of the SF block in the SF tensor.
  int sfBlkColIdx = dataBlkColIdx / NumDataBlkColsPerSfBlock;
  // The blocks are stored row-major in the tensor of scaling factors.
  int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumDataBlkColsPerSfBlock + sfBlkColIdx;

  // Find the row in the SF block.
  int sfRowIdx = (dataRowIdx % 32) * 4 + (dataRowIdx % NumDataRowsPerSfBlock) / 32;
  // Find the col in the SF block.
  int sfColIdx = (dataBlkColIdx % 4);

  // Compute the offset in bytes.
  return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Given the GMEM address of an output element, compute the offset of the corresponding scaling
// factor in the SF tensor. Optionally, a startTokenIndex can be provided if the first token is not
// the start token in the SF tensor. This is useful when inflight batching is enabled in TRT-LLM,
// where the context and generation output are stored as one output tensor. In this case, the
// generation output may not start with zero offset in the SF output tensor.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(int64_t gmemOffsetInBytes, int32_t hiddenDim,
                                      int32_t startTokenIdx = 0) {
  // The number of elements per sf.
  int32_t constexpr NumEltsPerSf = 16;
  // The GMEM offset of the output element.
  int64_t gmemOffset = gmemOffsetInBytes * 8 /*bits*/ / NumBitsPerElt;
  // The row/col indices of the corresponding SF element.
  int32_t sfRowIdx = gmemOffset / hiddenDim + startTokenIdx;
  int32_t sfColIdx = (gmemOffset % hiddenDim) / NumEltsPerSf;
  // Compute the SF offset.
  return getSfOffset(sfRowIdx, sfColIdx, hiddenDim / NumEltsPerSf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO(tizheng): Refactor to track gmem offset instead of doing pointer subtraction.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(void const* gmemOutPtr, void const* gmemBasePtr,
                                      int32_t hiddenDim, int32_t startTokenIdx = 0) {
  return getSfOffset<NumBitsPerElt>(
      reinterpret_cast<char const*>(gmemOutPtr) - reinterpret_cast<char const*>(gmemBasePtr),
      hiddenDim, startTokenIdx);
}

}  // namespace dev

// TODO: it would be nice to move some of that logic to Fp4Utils.h
template <tg::SfLayout Layout>
inline __device__ int32_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx,
                                      int32_t numDataBlksPerRow) {
  if constexpr (Layout == tg::SfLayout::Linear) {
    return numDataBlksPerRow * dataRowIdx + dataBlkColIdx;
  } else if constexpr (Layout == tg::SfLayout::R128c4) {
    return static_cast<int32_t>(dev::getSfOffset(dataRowIdx, dataBlkColIdx, numDataBlksPerRow));
  } else if constexpr (Layout == tg::SfLayout::R8c4 || Layout == tg::SfLayout::R8c16) {
    static int32_t constexpr NumRowsPerSfBlock = 8;
    static int32_t constexpr NumColsPerSfBlock = (Layout == tg::SfLayout::R8c4) ? 4 : 16;
    static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;
    int sfBlkRowIdx = dataRowIdx / NumRowsPerSfBlock;
    int sfBlkColIdx = dataBlkColIdx / NumColsPerSfBlock;
    int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumColsPerSfBlock + sfBlkColIdx;
    int sfRowIdx = dataRowIdx % NumRowsPerSfBlock;
    int sfColIdx = dataBlkColIdx % NumColsPerSfBlock;
    return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <tg::SfLayout LayoutSrc, tg::SfLayout LayoutDst, typename KernelParams>
__device__ void convertSfCommon(KernelParams params) {
  // Note: it's assumed that the number of scaling factors per row is a multiple of 4.
  constexpr int VecSize = 4;
  using VecType = uint32_t;
  static_assert(sizeof(VecType) == VecSize);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Immediately trigger the secondary kernel when using PDL, then wait on primary.
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // TODO: consider optimizing if used in production.
  // This is a naive kernel. It's not doing coalesced loads.

  int const numSfPerRow = params.hiddenDimSf;

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    for (int hiddenSfVecIdx = threadIdx.x + blockDim.x * blockIdx.x;
         hiddenSfVecIdx < numSfPerRow / VecSize; hiddenSfVecIdx += blockDim.x * gridDim.x) {
      // Index of the first SF in the vector.
      int const hiddenSfIdx = VecSize * hiddenSfVecIdx;

      // Load scale factors.
      int sfIdxIn = getSfOffset<LayoutSrc>(tokenIdx, hiddenSfIdx, numSfPerRow);
      const VecType sfVec = reinterpret_cast<VecType const*>(params.inSfPtr)[sfIdxIn / VecSize];

      // Store scale factors.
      int const sfIdxOut = getSfOffset<LayoutDst>(tokenIdx, hiddenSfIdx, numSfPerRow);
      reinterpret_cast<VecType*>(params.outSfPtr)[sfIdxOut / VecSize] = sfVec;
    }
  }
}

#define CONVERT_FP4_SF_KERNEL(LayoutSrc, LayoutDst)                                  \
  template <typename KernelParams>                                                   \
  __global__ void convertSf##LayoutSrc##To##LayoutDst##Kernel(KernelParams params) { \
    convertSfCommon<tg::SfLayout::LayoutSrc, tg::SfLayout::LayoutDst>(params);       \
  }
// We only need a conversion to the linear layout.
CONVERT_FP4_SF_KERNEL(R128c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c16, Linear);
#undef CONVERT_FP4_SF_KERNEL

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  constexpr int VecSize = 4;
  int const numThreads = 128;
  int const numBlocksX = (data.hiddenDimSf / VecSize - 1 + numThreads) / numThreads;
  int const numBlocksY = std::min(8192, data.numTokens);
  dim3 numBlocks(numBlocksX, numBlocksY);
#define CONVERT_FP4_SF_LAUNCH(LayoutSrc, LayoutDst)                                             \
  if (data.sfLayoutSrc == tg::SfLayout::LayoutSrc &&                                            \
      data.sfLayoutDst == tg::SfLayout::LayoutDst) {                                            \
    LAUNCH_PDL(data, false, cutlass::float_e4m3_t, convertSf##LayoutSrc##To##LayoutDst##Kernel, \
               numBlocks, numThreads, 0, stream);                                               \
    return;                                                                                     \
  }
  CONVERT_FP4_SF_LAUNCH(R128c4, Linear);
  CONVERT_FP4_SF_LAUNCH(R8c4, Linear);
  CONVERT_FP4_SF_LAUNCH(R8c16, Linear);
#undef CONVERT_FP4_SF_LAUNCH
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace convertsf

namespace permute {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void permuteKernel(KernelParams params) {
  using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      // Load chunk of token into registers
      const Type data = params.inPtr[tokenIdx * params.hiddenDim + hiddenIdx];

      // Write to topK places
      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
        params.outPtr[permutedIdx * params.hiddenDim + hiddenIdx] = data;
      }
    }
    if (params.useDeepSeekFp8) {
      for (int scaleIdx = threadIdx.x + blockDim.x * blockIdx.x; scaleIdx < params.hiddenDim / 128;
           scaleIdx += blockDim.x * gridDim.x) {
        for (int k = 0; k < params.topK; k++) {
          int const expandedIdx = tokenIdx * params.topK + k;
          int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

          int const idx_in = tokenIdx + params.numTokens * scaleIdx;
          int const idx_out = permutedIdx + params.totalNumPaddedTokens[0] * scaleIdx;

          params.outDqSfsPtr[idx_out] = params.inDqSfsPtr[idx_in];
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  int const numThreads = 256;
  int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
  int const numBlocksY = std::min(8192, data.numTokens);
  dim3 numBlocks(numBlocksX, numBlocksY);

  LAUNCH(data, permuteKernel, numBlocks, numThreads, 0, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace permute

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace finalize {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      // Accumulate chunk of token into registers
      float data = 0.0F;

      // Write to topK places
      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

        if (permutedIdx == -1) {
          continue;
        }

        if constexpr (KernelParams::UseSecondInput) {
          float const input =
              float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]} +
              float{params.inPtr2[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
          if (params.expertWeightsPtr != nullptr) {
            TypeExpW const scale = params.expertWeightsPtr[expandedIdx];
            data += float{scale} * input;
          } else {
            data += input;
          }
        } else {
          if (params.expertWeightsPtr != nullptr) {
            TypeExpW const scale = params.expertWeightsPtr[expandedIdx];
            data += float{scale} *
                    float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
          } else {
            data += float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
          }
        }
      }

      params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = static_cast<Type>(data);
    }
  }
}

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

__device__ float4 vectorizedLoadPtx(float4 const* ptr) {
  float4 ret;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
               : "l"(ptr));
  return ret;
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip
// connection.
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int MaxTopK = 64;

typedef struct __CUDA_ALIGN__(4) {
  cutlass::bfloat16_t array[2];
} bfloat16_2;

typedef struct __CUDA_ALIGN__(8) {
  cutlass::bfloat16_t array[4];
} bfloat16_4;

typedef struct __CUDA_ALIGN__(8) {
  half array[4];
} half_4;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int UnrollFactor_, typename TypeExpW_>
struct ScaleTraitsStruct;

template <>
struct ScaleTraitsStruct<1, cutlass::bfloat16_t> {
  using PackedType = cutlass::bfloat16_t;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 1>;
};

template <>
struct ScaleTraitsStruct<2, cutlass::bfloat16_t> {
  using PackedType = bfloat16_2;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 2>;
};

template <>
struct ScaleTraitsStruct<4, cutlass::bfloat16_t> {
  using PackedType = bfloat16_4;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 4>;
};

template <>
struct ScaleTraitsStruct<1, float> {
  using PackedType = float;
  using ArrayType = cutlass::Array<float, 1>;
};

template <>
struct ScaleTraitsStruct<2, float> {
  using PackedType = float2;
  using ArrayType = cutlass::Array<float, 2>;
};

template <>
struct ScaleTraitsStruct<4, float> {
  using PackedType = float4;
  using ArrayType = cutlass::Array<float, 4>;
};

template <>
struct ScaleTraitsStruct<1, half> {
  using PackedType = half;
  using ArrayType = cutlass::Array<half, 1>;
};

template <>
struct ScaleTraitsStruct<2, half> {
  using PackedType = half2;
  using ArrayType = cutlass::Array<half, 2>;
};

template <>
struct ScaleTraitsStruct<4, half> {
  using PackedType = half_4;
  using ArrayType = cutlass::Array<half, 4>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int UnrollFactor_, typename TypeExpW_>
struct FinalizeTraits;

template <typename TypeExpW_>
struct FinalizeTraits<1, TypeExpW_> {
  using IdxPackedType = int;
  using IdxArrayType = cutlass::Array<int, 1>;
  using ScaleTraits = ScaleTraitsStruct<1, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

template <typename TypeExpW_>
struct FinalizeTraits<2, TypeExpW_> {
  using IdxPackedType = int2;
  using IdxArrayType = cutlass::Array<int, 2>;
  using ScaleTraits = ScaleTraitsStruct<2, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

template <typename TypeExpW_>
struct FinalizeTraits<4, TypeExpW_> {
  using IdxPackedType = int4;
  using IdxArrayType = cutlass::Array<int, 4>;
  using ScaleTraits = ScaleTraitsStruct<4, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernelVecLoad(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;
  int constexpr TopKUnrollFactor = KernelParams::TopKUnrollFactor;

  static_assert(TopKUnrollFactor == 1 || TopKUnrollFactor == 2 || TopKUnrollFactor == 4,
                "TopKUnrollFactor must be 1, 2, or 4");
  using FinalizeTraits = FinalizeTraits<TopKUnrollFactor, TypeExpW>;
  using IdxPackedType = typename FinalizeTraits::IdxPackedType;
  using IdxArrayType = typename FinalizeTraits::IdxArrayType;
  using ScalePackedType = typename FinalizeTraits::ScalePackedType;
  using ScaleArrayType = typename FinalizeTraits::ScaleArrayType;

  int const hiddenDimPaddedBits = params.hiddenDimPadded * cutlass::sizeof_bits<Type>::value;
  int const hiddenDimBits = params.hiddenDim * cutlass::sizeof_bits<Type>::value;
  assert(hiddenDimPaddedBits % 128 == 0);
  assert(hiddenDimBits % 128 == 0);

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<Type>::value;
  using InputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

  int64_t const tokenIdx = blockIdx.x;
  int64_t const startOffset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const numElemsInPaddedCol = params.hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
  int64_t const numElemsInCol = params.hiddenDim / FINALIZE_ELEM_PER_THREAD;
  bool const useScale = params.expertWeightsPtr != nullptr;

  __shared__ ScalePackedType scaleArrSmem[MaxTopK / TopKUnrollFactor];
  __shared__ IdxPackedType permutedIdxArrSmem[MaxTopK / TopKUnrollFactor];

  for (int kChunkIdx = threadIdx.x; kChunkIdx < params.topK / TopKUnrollFactor;
       kChunkIdx += blockDim.x) {
    int const expandedIdx = tokenIdx * params.topK + kChunkIdx * TopKUnrollFactor;
    auto permutedIdxPacked = reinterpret_cast<IdxPackedType const*>(
        params.expandedIdxToPermutedIdx)[expandedIdx / TopKUnrollFactor];
    auto scalePacked = useScale ? reinterpret_cast<ScalePackedType const*>(
                                      params.expertWeightsPtr)[expandedIdx / TopKUnrollFactor]
                                : ScalePackedType{TypeExpW(1.f)};

    scaleArrSmem[kChunkIdx] = scalePacked;
    permutedIdxArrSmem[kChunkIdx] = permutedIdxPacked;
  }

  auto const offset = tokenIdx * params.hiddenDim;
  Type* outputPtr = params.outPtr + offset;
  auto* outElemPtr = reinterpret_cast<OutputElem*>(outputPtr);
  auto const* inElemPtr = reinterpret_cast<InputElem const*>(params.inPtr);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif
  __syncthreads();

  auto const* inElemPtr2 = reinterpret_cast<InputElem const*>(params.inPtr2);
  for (int elemIndex = startOffset; elemIndex < numElemsInCol; elemIndex += stride) {
    ComputeElem threadOutput;
    threadOutput.fill(0);
    for (int kChunkIdx = 0; kChunkIdx < params.topK / TopKUnrollFactor; kChunkIdx++) {
      auto permutedIdxArr = *reinterpret_cast<IdxArrayType const*>(&permutedIdxArrSmem[kChunkIdx]);
      InputElem inputElemArr[TopKUnrollFactor];
      InputElem inputElemArr2[TopKUnrollFactor];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        auto const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }

        auto const* inputPermutedPtr = inElemPtr + permutedIdx * numElemsInPaddedCol;
        float4 input =
            vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr[elemIndex]));
        inputElemArr[ki] = *reinterpret_cast<InputElem const*>(&input);
        if constexpr (KernelParams::UseSecondInput) {
          auto const* inputPermutedPtr2 = inElemPtr2 + permutedIdx * numElemsInPaddedCol;
          float4 input2 =
              vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr2[elemIndex]));
          inputElemArr2[ki] = *reinterpret_cast<InputElem const*>(&input2);
        }
      }
      auto scaleArr = *reinterpret_cast<ScaleArrayType const*>(&scaleArrSmem[kChunkIdx]);
      auto const scaleFloatArr =
          arrayConvert<ScaleArrayType, cutlass::Array<float, TopKUnrollFactor>>(scaleArr);

#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        auto const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        auto scale = useScale ? scaleFloatArr[ki] : 1.0f;
        ComputeElem expertResult = arrayConvert<InputElem, ComputeElem>(inputElemArr[ki]);
        if constexpr (KernelParams::UseSecondInput) {
          expertResult = expertResult + arrayConvert<InputElem, ComputeElem>(inputElemArr2[ki]);
        }
        threadOutput = threadOutput + scale * expertResult;
      }
    }
    OutputElem outputElem = arrayConvert<ComputeElem, OutputElem>(threadOutput);
    outElemPtr[elemIndex] = outputElem;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Small-batch finalize: each CTA reduces one hidden-dim slice of one token. The topK reduction is
// split across blockDim.y thread groups so every thread issues at most
// SMALL_BATCH_FINALIZE_MAX_LOADS independent 128-bit loads, and the partial sums are combined
// through shared memory. This spreads a few tokens over many SMs instead of serializing topK
// dependent loads per thread, which dominates finalize latency at decode batch sizes.
constexpr int SMALL_BATCH_FINALIZE_THREADS = 256;
constexpr int SMALL_BATCH_FINALIZE_MAX_LOADS = 4;
// Cap on topK groups: keeps at least 16 threads per block across the hidden dimension.
constexpr int SMALL_BATCH_FINALIZE_MAX_TOPK_GROUPS = 16;

template <typename KernelParams>
__global__ void __launch_bounds__(SMALL_BATCH_FINALIZE_THREADS)
    finalizeKernelSmallBatch(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;
  constexpr int ElemPerVec = 128 / cutlass::sizeof_bits<Type>::value;
  using VecElem = cutlass::Array<Type, ElemPerVec>;
  using ComputeElem = cutlass::Array<float, ElemPerVec>;

  __shared__ float partialSmem[SMALL_BATCH_FINALIZE_THREADS][ElemPerVec];

  int const kSplit = blockDim.y;
  int64_t const tokenIdx = blockIdx.x;
  int const vecIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int const numVecs = params.hiddenDim / ElemPerVec;
  int64_t const numVecsPadded = params.hiddenDimPadded / ElemPerVec;
  bool const active = vecIdx < numVecs;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Let the next kernel launch while this one runs; it waits on this grid before reading the
  // output.
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  // The permutation and the expert weights come from routing, which completed before the GEMMs
  // this kernel depends on could start, so they are read before waiting on the primary kernel
  // (as in finalizeKernelVecLoad). Only the GEMM output is read after the wait.
  int permutedIdx[SMALL_BATCH_FINALIZE_MAX_LOADS];
  float scale[SMALL_BATCH_FINALIZE_MAX_LOADS];
#pragma unroll
  for (int j = 0; j < SMALL_BATCH_FINALIZE_MAX_LOADS; ++j) {
    int const k = threadIdx.y + j * kSplit;
    permutedIdx[j] = -1;
    scale[j] = 0.f;
    if (active && k < params.topK) {
      int64_t const expandedIdx = tokenIdx * params.topK + k;
      permutedIdx[j] = params.expandedIdxToPermutedIdx[expandedIdx];
      scale[j] = params.expertWeightsPtr != nullptr
                     ? static_cast<float>(params.expertWeightsPtr[expandedIdx])
                     : 1.f;
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  ComputeElem acc;
  acc.fill(0.f);
  if (active) {
    VecElem in[SMALL_BATCH_FINALIZE_MAX_LOADS];
    VecElem in2[SMALL_BATCH_FINALIZE_MAX_LOADS];
#pragma unroll
    for (int j = 0; j < SMALL_BATCH_FINALIZE_MAX_LOADS; ++j) {
      if (permutedIdx[j] != -1) {
        float4 const v = vectorizedLoadPtx(reinterpret_cast<float4 const*>(params.inPtr) +
                                           permutedIdx[j] * numVecsPadded + vecIdx);
        in[j] = *reinterpret_cast<VecElem const*>(&v);
        if constexpr (KernelParams::UseSecondInput) {
          float4 const v2 = vectorizedLoadPtx(reinterpret_cast<float4 const*>(params.inPtr2) +
                                              permutedIdx[j] * numVecsPadded + vecIdx);
          in2[j] = *reinterpret_cast<VecElem const*>(&v2);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < SMALL_BATCH_FINALIZE_MAX_LOADS; ++j) {
      if (permutedIdx[j] != -1) {
        ComputeElem expertResult = arrayConvert<VecElem, ComputeElem>(in[j]);
        if constexpr (KernelParams::UseSecondInput) {
          expertResult = expertResult + arrayConvert<VecElem, ComputeElem>(in2[j]);
        }
        acc = acc + scale[j] * expertResult;
      }
    }
  }

  int const tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (threadIdx.y > 0) {
#pragma unroll
    for (int e = 0; e < ElemPerVec; ++e) {
      partialSmem[tid][e] = acc[e];
    }
  }
  __syncthreads();
  if (threadIdx.y == 0 && active) {
    for (int g = 1; g < kSplit; ++g) {
#pragma unroll
      for (int e = 0; e < ElemPerVec; ++e) {
        acc[e] += partialSmem[g * blockDim.x + threadIdx.x][e];
      }
    }
    reinterpret_cast<VecElem*>(params.outPtr + tokenIdx * params.hiddenDim)[vecIdx] =
        arrayConvert<ComputeElem, VecElem>(acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Finalize over a 2-D grid of (token, hidden-dim slice): CTA (blockIdx.x = token, blockIdx.y =
// slice) of FINALIZE_V2_THREADS_PER_CTA threads owns one 16-byte vector of the output row per
// thread. finalizeKernelVecLoad gives one CTA of 256 threads to a whole token, which at decode
// batch sizes leaves most SMs idle and serializes the top-k loads of each thread behind its
// accumulation; here every thread issues all of its top-k loads (in rounds of
// FINALIZE_V2_LOADS_IN_FLIGHT) before it uses any of them, and the grid covers the GPU from a few
// tokens on. The token's permuted indices and weights are read straight from global memory by
// every thread (the same addresses for the whole CTA: one transaction per warp, L1 hits across
// warps), so there is no shared memory and no barrier.
//
// Numerics are those of finalizeKernelVecLoad: fp32 accumulation in slot order k = 0 .. topK - 1
// of weight_k * value_k (the second input, when present, is added to the value first), slots with
// permutedIdx == -1 skipped, one conversion to the output type at the end; tokens without a local
// slot store zeros.

constexpr int FINALIZE_V2_THREADS_PER_CTA = 128;
constexpr int FINALIZE_V2_LOADS_IN_FLIGHT = 8;
// Largest grid, in CTAs per SM, for which this kernel is selected (about four resident waves).
constexpr int FINALIZE_V2_MAX_CTAS_PER_SM = 32;
// The per-token rule in run() calls a grid of fewer than this many 256-thread CTAs small (148 SMs
// x 8 CTAs, see its comment) and gives it finalizeKernelSmallBatch or the scalar finalizeKernel;
// larger grids take finalizeKernelVecLoad. finalizeKernelV2 replaces only the latter: it is
// 1.5-1.7x faster than finalizeKernelVecLoad at 64-128 tokens of width 6144, while below the bound
// the small-batch kernel is 0.0-0.5 us faster than it, in every parallel split measured. Own
// times (PDL off), old / new us, GLM-5.3 shapes at EP8: B300, 148 SMs at 2032 MHz, 2026-11-03:
// small-batch 2.6 / 2.7 at T = 8 and 3.1 / 3.5 at 32, VecLoad 5.9 / 3.7 at 64 and 6.2 / 3.8 at
// 128; Rubin sm107, 2026-11-03: 2.6 / 2.7 at 8, 3.3 / 3.7 at 32, 5.9 / 3.5 at 64,
// 6.2 / 3.8 at 128.
constexpr int FINALIZE_SMALL_GRID_CTA_BOUND = 1184;

template <typename KernelParams>
__global__ void __launch_bounds__(FINALIZE_V2_THREADS_PER_CTA)
    finalizeKernelV2(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;
  constexpr int ElemPerVec = 128 / cutlass::sizeof_bits<Type>::value;
  using VecElem = cutlass::Array<Type, ElemPerVec>;
  using ComputeElem = cutlass::Array<float, ElemPerVec>;

  int64_t const tokenIdx = blockIdx.x;
  int const vecIdx = blockIdx.y * FINALIZE_V2_THREADS_PER_CTA + threadIdx.x;
  int const numVecs = params.hiddenDim / ElemPerVec;
  int64_t const numVecsPadded = params.hiddenDimPadded / ElemPerVec;
  bool const active = vecIdx < numVecs;
  bool const useScale = params.expertWeightsPtr != nullptr;
  int64_t const expandedBase = tokenIdx * params.topK;

  // The permutation and the expert weights come from routing, which completed before the GEMMs
  // this kernel depends on could start, so the first round is read before waiting on the primary
  // kernel (as in finalizeKernelVecLoad). Only the GEMM output is read after the wait.
  int permutedIdx[FINALIZE_V2_LOADS_IN_FLIGHT];
  float scale[FINALIZE_V2_LOADS_IN_FLIGHT];
#pragma unroll
  for (int j = 0; j < FINALIZE_V2_LOADS_IN_FLIGHT; ++j) {
    permutedIdx[j] = -1;
    scale[j] = 1.0f;
    if (j < params.topK) {
      permutedIdx[j] = params.expandedIdxToPermutedIdx[expandedBase + j];
      if (useScale) {
        scale[j] = static_cast<float>(params.expertWeightsPtr[expandedBase + j]);
      }
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  ComputeElem threadOutput;
  threadOutput.fill(0);
  auto const* inVecPtr = reinterpret_cast<float4 const*>(params.inPtr);
  auto const* inVecPtr2 = reinterpret_cast<float4 const*>(params.inPtr2);
  for (int k0 = 0; k0 < params.topK; k0 += FINALIZE_V2_LOADS_IN_FLIGHT) {
    if (k0 > 0) {
#pragma unroll
      for (int j = 0; j < FINALIZE_V2_LOADS_IN_FLIGHT; ++j) {
        int const k = k0 + j;
        permutedIdx[j] = -1;
        scale[j] = 1.0f;
        if (k < params.topK) {
          permutedIdx[j] = params.expandedIdxToPermutedIdx[expandedBase + k];
          if (useScale) {
            scale[j] = static_cast<float>(params.expertWeightsPtr[expandedBase + k]);
          }
        }
      }
    }
    float4 inRaw[FINALIZE_V2_LOADS_IN_FLIGHT];
    float4 inRaw2[FINALIZE_V2_LOADS_IN_FLIGHT];
#pragma unroll
    for (int j = 0; j < FINALIZE_V2_LOADS_IN_FLIGHT; ++j) {
      if (active && permutedIdx[j] != -1) {
        inRaw[j] = vectorizedLoadPtx(inVecPtr + permutedIdx[j] * numVecsPadded + vecIdx);
        if constexpr (KernelParams::UseSecondInput) {
          inRaw2[j] = vectorizedLoadPtx(inVecPtr2 + permutedIdx[j] * numVecsPadded + vecIdx);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < FINALIZE_V2_LOADS_IN_FLIGHT; ++j) {
      if (active && permutedIdx[j] != -1) {
        ComputeElem expertResult =
            arrayConvert<VecElem, ComputeElem>(*reinterpret_cast<VecElem const*>(&inRaw[j]));
        if constexpr (KernelParams::UseSecondInput) {
          expertResult = expertResult + arrayConvert<VecElem, ComputeElem>(
                                            *reinterpret_cast<VecElem const*>(&inRaw2[j]));
        }
        threadOutput = threadOutput + scale[j] * expertResult;
      }
    }
  }
  if (active) {
    reinterpret_cast<VecElem*>(params.outPtr + tokenIdx * params.hiddenDim)[vecIdx] =
        arrayConvert<ComputeElem, VecElem>(threadOutput);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeDeepSeekKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  using BlockReduce = cub::BlockReduce<float, 128>;

  __shared__ float s_scaleOut;
  __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      // Accumulate chunk of token into registers
      float acc = 0.0f;

      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
        if (permutedIdx == -1) {
          continue;
        }
        int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
        int const scaleIdx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
        float const blockScale = params.inDqSfsPtr ? params.inDqSfsPtr[scaleIdx] : 1;

        float const expertProb = (float)params.expertWeightsPtr[tokenIdx * params.topK + k];

        float const scale = expertProb * blockScale;
        acc += scale *
               static_cast<float>(params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]);
      }

      // The largest (finite) value that can be represented using E4m3.
      float constexpr E4m3MaxVal{448.f};

      // Compute the absolute max
      float aMax = BlockReduce(temp_storage).Reduce(fabsf(acc), cuda::maximum<>{});

      if (threadIdx.x == 0) {
        if (params.outDqSfsPtr) {
          s_scaleOut = aMax / E4m3MaxVal;
          int const scaleOut_idx = tokenIdx + hiddenIdx / 128 * params.numTokens;
          params.outDqSfsPtr[scaleOut_idx] = aMax / E4m3MaxVal;
        } else {
          s_scaleOut = 1.0f;
        }
      }
      __syncthreads();
      float const scaleOut = s_scaleOut;
      __syncthreads();
      params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = (Type)(acc / scaleOut);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void run(Data const& data, void* stream) {
  if (data.mUseDeepSeekFp8) {
    FLASHINFER_CHECK(data.inPtr2 == nullptr,
                     "DeepSeek FP8 finalize does not support a second input");
    int const numThreads = 128;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535 specified by CUDA.
    int const numBlocksY = std::min(8192, data.numTokens);
    dim3 numBlocks(numBlocksX, numBlocksY);

    LAUNCH_TOPK_EXPW_SINGLE_INPUT(data, finalizeDeepSeekKernel, numBlocks, numThreads, 0, stream);
  } else {
    int const numThreads = 256;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535 specified by CUDA.
    int const numBlocksY = std::min(8192, data.numTokens);

    int64_t const finalizeElemsPerVector = 128 / tg::dtypeGetNumBits(data.mDtypeElt);
    bool const vectorWidthAligned = data.hiddenDim % finalizeElemsPerVector == 0 &&
                                    data.hiddenDimPadded % finalizeElemsPerVector == 0;
    // The (token, slice) grid kernel replaces the per-token vectorized kernel between the
    // small-grid bound and a few waves (measured: 1.3-1.7x faster than it at 64 and 256 tokens of
    // width 6144, 0.8x at 2048, where the per-token kernel's longer-lived CTAs keep more bytes in
    // flight); above that the per-token vectorized kernel stays, below the bound the small-batch
    // split and the scalar finalizeKernel stay (see FINALIZE_SMALL_GRID_CTA_BOUND). The scalar
    // finalizeKernel also stays for rows that are not 16-byte vectors.
    int device{-1};
    cudaGetDevice(&device);
    int numSms = 0;
    cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);
    int64_t const numVecs = data.hiddenDim / finalizeElemsPerVector;
    int64_t const numSlices =
        (numVecs + FINALIZE_V2_THREADS_PER_CTA - 1) / FINALIZE_V2_THREADS_PER_CTA;
    bool const useSliceGrid =
        vectorWidthAligned && data.topK <= MaxTopK && isAligned16(data.inPtr) &&
        (data.inPtr2 == nullptr || isAligned16(data.inPtr2)) && isAligned16(data.outPtr) &&
        numBlocksX * numBlocksY >= FINALIZE_SMALL_GRID_CTA_BOUND &&
        (int64_t)data.numTokens * numSlices <= (int64_t)numSms * FINALIZE_V2_MAX_CTAS_PER_SM;
    // With topK <= SMALL_BATCH_FINALIZE_MAX_LOADS there is a single topK group, and the
    // shared-memory reduction costs more than the split saves.
    bool const useSmallBatch =
        !useSliceGrid && numBlocksX * numBlocksY < FINALIZE_SMALL_GRID_CTA_BOUND &&
        vectorWidthAligned && data.topK > SMALL_BATCH_FINALIZE_MAX_LOADS &&
        data.topK <= SMALL_BATCH_FINALIZE_MAX_TOPK_GROUPS * SMALL_BATCH_FINALIZE_MAX_LOADS;
    if (useSliceGrid) {
      dim3 numBlocks(data.numTokens, static_cast<unsigned>(numSlices));
      LAUNCH_TOPK_EXPW(data, finalizeKernelV2, numBlocks, FINALIZE_V2_THREADS_PER_CTA, 0, stream);
    } else if (useSmallBatch) {
      // Fewest topK groups that keep each thread at SMALL_BATCH_FINALIZE_MAX_LOADS loads or fewer.
      int kSplit = 1;
      while (kSplit * SMALL_BATCH_FINALIZE_MAX_LOADS < data.topK) {
        kSplit *= 2;
      }
      int const vecsPerBlock = SMALL_BATCH_FINALIZE_THREADS / kSplit;
      int const numVecs = data.hiddenDim / finalizeElemsPerVector;
      dim3 numBlocks(data.numTokens, (numVecs + vecsPerBlock - 1) / vecsPerBlock);
      dim3 blockDim(vecsPerBlock, kSplit);
      LAUNCH_TOPK_EXPW(data, finalizeKernelSmallBatch, numBlocks, blockDim, 0, stream);
    } else if (numBlocksX * numBlocksY < FINALIZE_SMALL_GRID_CTA_BOUND || !vectorWidthAligned) {
      // The number 1184 comes from 148 * 8, where 148 is the number of SMs (Streaming
      // Multiprocessors) in the Blackwell architecture, and the value 8 means that each Streaming
      // Multiprocessor (SM) can hold up to 8 blocks for this kernel. This limitation is intended to
      // ensure that when the number of waves is greater than 1, we choose to use the kernel with
      // vectorized loading.
      dim3 numBlocks(numBlocksX, numBlocksY);
      LAUNCH_TOPK_EXPW(data, finalizeKernel, numBlocks, numThreads, 0, stream);
    } else {
      FLASHINFER_CHECK(
          data.topK <= MaxTopK,
          "Finalize kernel with vectorized loading is not supported for this TopK value: %d",
          data.topK);
      LAUNCH_TOPK_EXPW(data, finalizeKernelVecLoad, /*numBlocks=*/data.numTokens,
                       /*numThreads=*/FINALIZE_THREADS_PER_BLOCK, 0, stream);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace finalize

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace moe::dev
