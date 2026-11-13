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

// Persistent DeepSeek FP8 activation: one kernel for every token count, taking over from the
// per-pair activationDeepSeekKernel for callers that provide the permuted row -> token map (the
// fused MoE runner). Its output is bit-identical to that kernel's; launched behind GEMM1 with
// PDL, as the fused MoE launches it, its critical-path time is at or below the old kernel's at
// every token count (the pull request that added it carries the measurements).
//
// A piece is one scale block (SfBlock columns) of one permuted row; a GEMM1 output row holds the
// linear half then the gate half, innerDim / 2 columns each. L lanes of one warp own a piece,
// each lane SfBlock / L consecutive columns of both halves; the block amax is a register max plus
// log2(L) xor-shuffles. No shared memory, no block barrier.
//
// The grid is fixed (at most DEEP_SEEK_ACTIVATION_PERSISTENT_CTAS_PER_SM blocks per SM, one
// resident wave) and every block strides over the pieces of a work list. The kernel decides on
// the device, uniformly over the grid, from the live padded row count P (a routing output) and
// the host-known pair count numTokens * topK:
//   1. The work list: the permuted rows (tile padding rows, token index -1, are skipped) when
//      P <= pairs, else the (token, slot) pairs (pairs routed to other ranks, permuted index -1,
//      are skipped). Neither padding nor remote pairs cost a data load. The old kernel walks
//      every pair, 7 of 8 remote under EP8; a row walk would visit up to 63 padding rows per live
//      row (a 64-row GEMM tile holding one live row).
//   2. Lanes per piece: 4 bytes per lane (32 lanes per 128-wide e4m3 block, 4 elements each)
//      while the list has at most DEEP_SEEK_ACTIVATION_PERSISTENT_FINE_MAX_PIECES pieces, else
//      16 bytes per lane (8 lanes, 16 elements). Few pieces want many short dependent chains in
//      flight, many pieces want wide loads; the crossover is a measured value (148 SMs, one
//      resident wave).
// The list entry of a piece is fetched two pieces ahead and its data one piece ahead, so each
// piece costs one memory round trip and the first piece's data is requested right after the PDL
// wait.
//
// Pre-wait reads under PDL. P, the two index tables and the tile -> expert map are routing
// outputs. GEMM1, the kernel this one waits on, is a trtllm-gen cubin whose every CTA executes
// griddepcontrol.wait before it triggers the dependent launch or exits (TrtllmGenBatchedGemmRunner
// rules out the kernel options that would allow otherwise). griddepcontrol.wait returns only once
// the routing grid has completed and its writes are visible, so by the time any GEMM1 CTA has
// triggered, those tables are final, whatever point the routing kernel triggered at (the
// small-batch routing kernels trigger at their start). Reading them before
// cudaGridDependencySynchronize() is therefore sound here; it would not be in a kernel launched
// directly after routing. GEMM1's output (inPtr, inDqSfsPtr) is read only after the wait. A
// caller whose primary is not a trtllm-gen GEMM (the standalone activation op) does not provide
// the row map and keeps activationDeepSeekKernel.
//
// Numerics are activationDeepSeekKernel's, operation for operation: x = scale * float(e4m3),
// gatedSilu (its per-expert controls looked up once per piece instead of once per element),
// scale = max(amax / 448, FLT_MIN), e4m3(out / scale) with the saturating round-to-nearest-even
// conversion. The block amax is reduced with fmaxf, as the old kernel's 2- and 4-token
// instantiations do; its 1-token instantiation reduces with cuda::maximum<>, which CCCL
// implements as fmax for floating-point types, so a NaN from GEMM1 is dropped alike in all of
// them. Padding rows are not written, as before.

constexpr int DEEP_SEEK_ACTIVATION_PERSISTENT_THREADS_PER_CTA = 128;
// One resident wave: 8 x 128 threads at <= 64 registers per thread (58 for e4m3 in this module's
// sm_100a build), no shared memory; __launch_bounds__ holds ptxas to the budget.
constexpr int DEEP_SEEK_ACTIVATION_PERSISTENT_CTAS_PER_SM = 8;
// Largest piece count of the chosen list handled at 4 bytes per lane, for 128-wide blocks.
constexpr int DEEP_SEEK_ACTIVATION_PERSISTENT_FINE_MAX_PIECES = 8192;
// The scale block width instantiated below. The kernel template also handles 32 (a quarter-size
// piece: 8 and 2 lanes), but a 32-wide block reaches activation::run only through the standalone
// op, which keeps the old kernel (see persistentActivationSupported).
constexpr int DEEP_SEEK_ACTIVATION_PERSISTENT_SF_BLOCK = 128;

template <int Bytes>
struct PersistentActivationVec;
template <>
struct PersistentActivationVec<16> {
  using Type = uint4;
};
template <>
struct PersistentActivationVec<8> {
  using Type = uint2;
};
template <>
struct PersistentActivationVec<4> {
  using Type = uint32_t;
};

// Lanes per piece for a block of SfBlock elements of Type: fine = 4 bytes per lane (at most one
// warp), coarse = 16 bytes per lane.
template <typename Type, int SfBlock>
struct PersistentActivationLanes {
  static constexpr int kBlockBytes = SfBlock * static_cast<int>(sizeof(Type));
  static constexpr int kFine = kBlockBytes / 4 > 32 ? 32 : kBlockBytes / 4;
  static constexpr int kCoarse = kBlockBytes / 16;
  static_assert(kBlockBytes % 16 == 0 && kCoarse >= 1,
                "a scale block must span a multiple of 16 bytes");
};

// Piece q (row-major over (list entry, scale block)) -> entry and block: a shift when the block
// count is a power of two, a 32-bit division otherwise (I = 640 gives 5 blocks of 128).
struct PersistentActivationPieceMap {
  int numBlocks;  // scale blocks per row
  int shift;      // log2(numBlocks) when it is a power of two, else -1

  __device__ __forceinline__ void split(int q, int& entry, int& block) const {
    if (shift >= 0) {
      entry = q >> shift;
      block = q - (entry << shift);
    } else {
      entry = q / numBlocks;
      block = q - entry * numBlocks;
    }
  }
};

// gatedSilu with the per-expert controls already in registers: the same operations in the same
// order, so the bytes are the same.
__device__ __forceinline__ float gatedSiluFromControls(bool hasControls, bool hasClamp, float alpha,
                                                       float beta, float limit, float xLinear,
                                                       float xGlu) {
  if (!hasControls) {
    return silu(xGlu) * xLinear;
  }
  if (hasClamp) {
    xGlu = fminf(xGlu, limit);
    xLinear = fmaxf(fminf(xLinear, limit), -limit);
  }
  float const act = xGlu / (1.0f + expf(-alpha * xGlu));
  return act * (xLinear + beta);
}

template <typename KernelParams, int SfBlock, int L>
struct PersistentActivationPiece {
  using Type = typename KernelParams::Type;
  static constexpr int kElts = SfBlock / L;
  static constexpr int kBytes = kElts * static_cast<int>(sizeof(Type));
  static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16, "a lane owns 4, 8 or 16 bytes");
  using Vec = typename PersistentActivationVec<kBytes>::Type;

  int row = -1;    // -1: a padding row (rows list) or a pair on another rank (pairs list)
  int block = 0;   // scale block within the row
  int entry = 0;   // the list entry: the row itself (rows list) or the pair (pairs list)
  int expert = 0;  // local expert of the row, only with the per-expert controls
  Vec x1{}, x2{};
  float scale1 = 0.f, scale2 = 0.f;

  // The list entry of piece q: the rows list reads row -> token (the row is the entry), the
  // pairs list reads pair -> row.
  __device__ __forceinline__ void index(KernelParams const& params,
                                        PersistentActivationPieceMap const& map, int q,
                                        bool pairsList) {
    map.split(q, entry, block);
    if (pairsList) {
      row = params.expandedIdxToPermutedIdx[entry];
    } else {
      row = params.permutedIdxToTokenIdx[entry] == -1 ? -1 : entry;
    }
  }

  // Local expert of row r: routing's tile -> batch map, tileTokensDim rows per tile.
  __device__ __forceinline__ void lookUpExpert(KernelParams const& params, int r) {
    expert = params.ctaIdxXyToBatchIdx[r / params.tileTokensDim];
  }

  // One lane's bytes of both halves of the row, and the two block scales (block-major with
  // stride P; the gate blocks follow the linear ones).
  __device__ __forceinline__ void load(KernelParams const& params,
                                       PersistentActivationPieceMap const& map, int P, int outDim,
                                       int lane) {
    int64_t const base = (int64_t)row * params.innerDim;
    int const col = block * SfBlock + lane * kElts;
    x1 = *reinterpret_cast<Vec const*>(params.inPtr + base + col);
    x2 = *reinterpret_cast<Vec const*>(params.inPtr + base + outDim + col);
    scale1 = params.inDqSfsPtr[(int64_t)row + (int64_t)P * block];
    scale2 = params.inDqSfsPtr[(int64_t)row + (int64_t)P * (block + map.numBlocks)];
  }
};

// L lanes per piece. A group's exit depends on blockIdx and the group index only, so it is
// uniform and the shuffles never wait for an exited lane; groups of one warp may sit at different
// pieces, hence the mask covers the group alone.
template <typename KernelParams, int SfBlock, int L>
__device__ __forceinline__ void activationDeepSeekPersistentBody(
    KernelParams const& params, int P, int outDim, PersistentActivationPieceMap const map,
    int numPieces, bool pairsList) {
  using Piece = PersistentActivationPiece<KernelParams, SfBlock, L>;
  using Type = typename KernelParams::Type;
  using Vec = typename Piece::Vec;
  constexpr int kElts = Piece::kElts;
  constexpr int kGroupsPerCta = DEEP_SEEK_ACTIVATION_PERSISTENT_THREADS_PER_CTA / L;
  static_assert(L >= 1 && L <= 32 && (L & (L - 1)) == 0,
                "lanes per piece: a power of two within one warp");
  float constexpr E4m3MaxVal{448.f};

  int const lane = threadIdx.x % L;
  int const group = threadIdx.x / L;
  unsigned const groupMask =
      L == 32 ? 0xffffffffu : ((1u << L) - 1u) << (((threadIdx.x % 32) / L) * L);
  bool const hasControls = params.gatedActAlphaPtr != nullptr ||
                           params.gatedActBetaPtr != nullptr ||
                           params.gatedActClampLimitPtr != nullptr;
  bool const hasClamp = params.gatedActClampLimitPtr != nullptr;

  int const stride = gridDim.x * kGroupsPerCta;
  int q0 = blockIdx.x * kGroupsPerCta + group;
  int q1 = q0 + stride;

  // Routing outputs for the first two pieces, read before the wait (see the kernel comment). With
  // the rows list the expert of a row follows from the row alone; with the pairs list it needs the
  // table entry first, so it moves one stage later.
  Piece p0, p1;
  if (q0 < numPieces) p0.index(params, map, q0, pairsList);
  if (q1 < numPieces) p1.index(params, map, q1, pairsList);
  if (hasControls) {
    if (q0 < numPieces && (!pairsList || p0.row != -1)) {
      p0.lookUpExpert(params, pairsList ? p0.row : p0.entry);
    }
    if (!pairsList && q1 < numPieces) p1.lookUpExpert(params, p1.entry);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif
  if (q0 >= numPieces) {
    return;
  }
  if (p0.row != -1) p0.load(params, map, P, outDim, lane);

  while (q0 < numPieces) {
    // Prefetch: the list entry two pieces ahead, the data one piece ahead.
    int const q2 = q1 + stride;
    Piece p2;
    if (q2 < numPieces) {
      p2.index(params, map, q2, pairsList);
      if (hasControls && !pairsList) p2.lookUpExpert(params, p2.entry);
    }
    if (q1 < numPieces && p1.row != -1) {
      p1.load(params, map, P, outDim, lane);
      if (hasControls && pairsList) p1.lookUpExpert(params, p1.row);
    }

    if (p0.row != -1) {
      float alpha = 1.0f, beta = 0.0f, limit = 0.0f;
      if (hasControls) {
        if (hasClamp) limit = params.gatedActClampLimitPtr[p0.expert];
        if (params.gatedActAlphaPtr != nullptr) alpha = params.gatedActAlphaPtr[p0.expert];
        if (params.gatedActBetaPtr != nullptr) beta = params.gatedActBetaPtr[p0.expert];
      }
      Type const* x1 = reinterpret_cast<Type const*>(&p0.x1);
      Type const* x2 = reinterpret_cast<Type const*>(&p0.x2);
      float out[kElts];
      float aMax = 0.0f;
#pragma unroll
      for (int i = 0; i < kElts; ++i) {
        float const xLinear = p0.scale1 * static_cast<float>(x1[i]);
        float const xGlu = p0.scale2 * static_cast<float>(x2[i]);
        out[i] = gatedSiluFromControls(hasControls, hasClamp, alpha, beta, limit, xLinear, xGlu);
        aMax = fmaxf(aMax, fabsf(out[i]));
      }
#pragma unroll
      for (int offset = L / 2; offset > 0; offset >>= 1) {
        aMax = fmaxf(aMax, __shfl_xor_sync(groupMask, aMax, offset));
      }
      // Strictly positive, so the division below is defined for an all-zero block.
      float const scaleOut = fmaxf(aMax / E4m3MaxVal, std::numeric_limits<float>::min());
      Vec outVec{};
      Type* outElts = reinterpret_cast<Type*>(&outVec);
#pragma unroll
      for (int i = 0; i < kElts; ++i) {
        outElts[i] = static_cast<Type>(out[i] / scaleOut);
      }
      int const col = p0.block * SfBlock + lane * kElts;
      *reinterpret_cast<Vec*>(params.outPtr + (int64_t)p0.row * outDim + col) = outVec;
      if (lane == 0) {
        params.outDqSfsPtr[(int64_t)p0.row + (int64_t)P * p0.block] = scaleOut;
      }
    }

    q0 = q1;
    q1 = q2;
    p0 = p1;
    p1 = p2;
  }
}

template <typename KernelParams, int SfBlock>
__device__ __forceinline__ void activationDeepSeekPersistent(KernelParams const& params) {
  static_assert(SfBlock == 128 || SfBlock == 32, "DeepSeek FP8 scale blocks are 128 or 32 wide");
  using Lanes = PersistentActivationLanes<typename KernelParams::Type, SfBlock>;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Let GEMM2 launch at once; it waits for this kernel's completion itself.
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  int const outDim = params.innerDim / 2;
  PersistentActivationPieceMap map;
  map.numBlocks = outDim / SfBlock;
  map.shift = (map.numBlocks & (map.numBlocks - 1)) == 0 ? __ffs(map.numBlocks) - 1 : -1;
  // The live padded row count: a routing output, read before the wait (see the kernel comment).
  int const P = params.totalNumPaddedTokens[0];
  int const pairs = params.numTokens * params.topK;
  bool const pairsList = P > pairs;
  int const numPieces = (pairsList ? pairs : P) * map.numBlocks;
  // A 32-wide piece is a quarter of a 128-wide one: the same capacity holds four times as many.
  constexpr int kFineMaxPieces = DEEP_SEEK_ACTIVATION_PERSISTENT_FINE_MAX_PIECES * (128 / SfBlock);
  if (numPieces <= kFineMaxPieces) {
    activationDeepSeekPersistentBody<KernelParams, SfBlock, Lanes::kFine>(params, P, outDim, map,
                                                                          numPieces, pairsList);
  } else {
    activationDeepSeekPersistentBody<KernelParams, SfBlock, Lanes::kCoarse>(params, P, outDim, map,
                                                                            numPieces, pairsList);
  }
}

template <typename KernelParams>
__global__ void __launch_bounds__(DEEP_SEEK_ACTIVATION_PERSISTENT_THREADS_PER_CTA,
                                  DEEP_SEEK_ACTIVATION_PERSISTENT_CTAS_PER_SM)
    activationDeepSeekKernelPersistent128(KernelParams params) {
  activationDeepSeekPersistent<KernelParams, 128>(params);
}

// The persistent kernel is launched with NumTokensPerCta fixed to 1 (the parameter is unused
// there); LAUNCH_ACTIVATION would also instantiate the 2- and 4-token variants.
#define LAUNCH_ACTIVATION_PERSISTENT(data, kernel, numBlocks, numThreads, smemSize, stream)      \
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

inline bool isAligned16(void const* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 15u) == 0;
}

// Whether activationDeepSeekKernelPersistent128 can run: the row -> token map and its bound (the
// fused MoE runner sets them; the standalone activation op does not, and its primary is not a
// trtllm-gen GEMM, so it keeps activationDeepSeekKernel), whole scale blocks of the width the
// kernel is built for, 16-byte aligned rows and pointers, and the tile -> expert map when the
// per-expert controls are set.
inline bool persistentActivationSupported(Data const& data) {
  int const outputDim = data.innerDim / 2;
  int const eltBytes = tg::dtypeGetNumBits(data.mDtypeElt) / 8;
  bool const hasControls = data.gatedActAlphaPtr != nullptr || data.gatedActBetaPtr != nullptr ||
                           data.gatedActClampLimitPtr != nullptr;
  return data.permutedIdxToTokenIdx != nullptr && data.totalMaxPaddedTokens > 0 &&
         data.expandedIdxToPermutedIdx != nullptr &&
         data.mDeepSeekSfBlockSize == DEEP_SEEK_ACTIVATION_PERSISTENT_SF_BLOCK && outputDim > 0 &&
         outputDim % DEEP_SEEK_ACTIVATION_PERSISTENT_SF_BLOCK == 0 &&
         (outputDim * eltBytes) % 16 == 0 && isAligned16(data.inPtr) && isAligned16(data.outPtr) &&
         (!hasControls || (data.ctaIdxXyToBatchIdx != nullptr && data.tileTokensDim > 0));
}

// The device walks the shorter of the two lists (the rows, at most the bound, or the pairs); the
// grid covers that list at fine-mode occupancy, capped at one resident wave, and the kernel
// strides over the rest. Any grid is correct; this one only avoids idle blocks at small T.
inline int persistentActivationNumCtas(Data const& data, int numSms) {
  int const outputDim = data.innerDim / 2;
  int const eltBytes = tg::dtypeGetNumBits(data.mDtypeElt) / 8;
  int const numBlocks = outputDim / DEEP_SEEK_ACTIVATION_PERSISTENT_SF_BLOCK;
  int const fineLanes = std::min(32, DEEP_SEEK_ACTIVATION_PERSISTENT_SF_BLOCK * eltBytes / 4);
  int const piecesPerCta = DEEP_SEEK_ACTIVATION_PERSISTENT_THREADS_PER_CTA / fineLanes;
  int64_t const entries =
      std::min<int64_t>(data.totalMaxPaddedTokens, (int64_t)data.numTokens * data.topK);
  int64_t const needed = (entries * numBlocks + piecesPerCta - 1) / piecesPerCta;
  int64_t const cap = (int64_t)numSms * DEEP_SEEK_ACTIVATION_PERSISTENT_CTAS_PER_SM;
  return static_cast<int>(std::min(std::max<int64_t>(needed, 1), cap));
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
    FLASHINFER_CHECK(data.mDeepSeekSfBlockSize == 128 || data.mDeepSeekSfBlockSize == 32,
                     "DeepSeek FP8 activation: scale block must be 128 or 32.");

    int device{-1};
    cudaGetDevice(&device);
    int numSms = 0;
    cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

    if (persistentActivationSupported(data)) {
      dim3 const grid(persistentActivationNumCtas(data, numSms));
      LAUNCH_ACTIVATION_PERSISTENT(data, activationDeepSeekKernelPersistent128, grid,
                                   DEEP_SEEK_ACTIVATION_PERSISTENT_THREADS_PER_CTA, 0, stream);
      return;
    }

    constexpr int NUM_ELTS_PER_LOAD = 1;
    constexpr int NUM_ELTS_PER_CTA = DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA;

    // Output dimension is innerDim / 2; each CTA covers 128 elements, i.e. one scale block
    // at block 128 or four warp-owned blocks at block 32.
    int const outputDim = data.innerDim / 2;
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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Wait on the primary kernel before reading anything another kernel wrote, including the
  // routing outputs below. Under PDL this kernel can start before routing has finished: only
  // this wait guarantees that their writes are visible.
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Wait on the primary kernel before reading anything another kernel wrote, including the
  // permutation and expert weights from routing. Under PDL, the kernels before this one trigger
  // early, so routing may still be running (or its writes not yet visible) when this kernel
  // starts. Only this wait guarantees that their writes are visible.
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

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
    // With topK <= SMALL_BATCH_FINALIZE_MAX_LOADS there is a single topK group, and the
    // shared-memory reduction costs more than the split saves.
    bool const useSmallBatch =
        numBlocksX * numBlocksY < 1184 && vectorWidthAligned &&
        data.topK > SMALL_BATCH_FINALIZE_MAX_LOADS &&
        data.topK <= SMALL_BATCH_FINALIZE_MAX_TOPK_GROUPS * SMALL_BATCH_FINALIZE_MAX_LOADS;
    if (useSmallBatch) {
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
    } else if (numBlocksX * numBlocksY < 1184 || !vectorWidthAligned) {
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
