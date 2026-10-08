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

#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <array>
#include <cuda/std/limits>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../exception.h"
#include "../fp4_layout.cuh"
#include "../logging.h"
#include "../utils.cuh"
#include "trtllm_allreduce_fusion.cuh"

namespace flashinfer {
namespace trtllm_mnnvl_allreduce {

using flashinfer::QuantizationSFLayout;

enum class QuantType : int {
  kNone = 0,
  kFP8 = 1,
  kFP4 = 2,
  kDynamicFP8 = 3,
};

struct AllReduceFusionParams {
  int nRanks;
  int rank;
  int numTokens;
  int tokenDim;
  void** bufferPtrsDev;
  void* bufferPtrLocal;
  void* multicastPtr;
  uint32_t* bufferFlags;
  bool rmsNormFusion;
  bool launchWithPdl;

  void const* input;
  void const* residualIn;
  void const* gamma;
  double epsilon;
  // 0 for standard RMSNorm (out = gamma * x * rsqrt(...)),
  // 1 for Gemma / Qwen3.5 (out = (1 + gamma) * x * rsqrt(...)).
  float weightBias = 0.f;
  float* outputScale = nullptr;
  QuantizationSFLayout sfLayout = QuantizationSFLayout::SWIZZLED_128x4;
  QuantType quantType = QuantType::kNone;

  void* residualOut;
  void* output;
  void* quantOut = nullptr;
  void* scalingFactorOut = nullptr;
  cudaStream_t stream = nullptr;
};

template <typename T>
struct AllReduceKernelParams {
  T* outputPtr;
  T* prenormedPtr;
  T const* shardPtr;
  T const* residualInPtr;
  T const* gammaPtr;
  T** inputPtrs;
  T* mcastPtr;
  T* localBufferPtr;
  int nRanks;
  int rank;
  int numTokens;
  int tokenDim;
  float epsilon;
  float weightBias;
  float* outputScalePtr;
  void* quantOutPtr;
  void* scalingFactorOutPtr;
  QuantizationSFLayout sfLayout;
  uint32_t* bufferFlags;
};

namespace utils {

constexpr uint16_t kNEGZERO_FP16 = 0x8000U;
constexpr uint32_t kNEGZERO_FP32 = 0x80000000U;
constexpr int kMaxClusterSize = 8;
constexpr float kFP8E4M3Max = 448.0f;
// E4M3FN's smallest positive subnormal is 2^-9. Clamp dynamic
// scales to min_subnormal / max_finite so zero/tiny rows do not
// produce a zero scale.
constexpr float kFP8E4M3MinSubnormal = 1.0f / 512.0f;
constexpr float kDynamicFP8MinScale = kFP8E4M3MinSubnormal / kFP8E4M3Max;

template <typename T>
union Fp16BitCast {
  T mFp;
  uint16_t mInt;

  constexpr Fp16BitCast() : mInt(0) {}

  constexpr Fp16BitCast(T val) : mFp(val) {}

  constexpr Fp16BitCast(uint16_t val) : mInt(val) {}
};

template <typename T>
inline __device__ float toFloat(T val) {
  return val;
}

template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
template <>
inline __device__ float toFloat<__nv_half>(__nv_half val) {
  return __half2float(val);
}

template <typename T>
inline __device__ T fromFloat(float val) {
  return val;
}

template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
inline __device__ __nv_half fromFloat<__nv_half>(float val) {
  return __float2half(val);
}

template <typename T>
static constexpr __device__ __host__ T negZero() {
  if constexpr (std::is_same_v<T, float>) {
    return -0.0F;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>) {
    return Fp16BitCast<T>(kNEGZERO_FP16).mFp;
  } else {
    static_assert(sizeof(T) == 0, "negativeZero not specialized for this type");
  }
  return T{};  // Never reached, but needed for compilation
}

// WARNING: the Lamport sentinel is a *bit pattern* (fp32 -0.0 = 0x80000000;
// fp16/bf16 -0.0 = 0x8000). Always compare bit-exact -- do NOT fall back to
// `val == 0.F && signbit(val)`. nvcc emits `setp.eq.f32` with `.ftz=true`
//  which flushes fp32 subnormal operands to +/-0.0 *before*
// the equality while signbit() still reads bit 31, so any fp32 negative
// subnormal pattern (e.g. 0x80010000, which appears when bf16 negative
// subnormals 0x8001-0x807F land in the high half of a 4-byte poll load) would
// falsely match the sentinel and deadlock the polling loop.
template <typename T>
static inline __device__ bool isNegZero(T val) {
  if constexpr (std::is_same_v<T, float>) {
    return __float_as_uint(val) == kNEGZERO_FP32;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>) {
    return Fp16BitCast<T>(val).mInt == kNEGZERO_FP16;
  } else {
    static_assert(sizeof(T) == 0, "isNegZero not specialized for this type");
  }
  return false;  // Never reached, but needed for compilation
}

template <typename PackedType, typename T>
constexpr __device__ __host__ PackedType getPackedLamportInit() {
  static_assert(sizeof(PackedType) % sizeof(T) == 0, "PackedType size must be divisible by T size");
  constexpr int kNumElements = sizeof(PackedType) / sizeof(T);

  union PackedT {
    PackedType mPacked;
    std::array<T, kNumElements> mElements;

    constexpr PackedT() : mElements{} {
      for (int i = 0; i < kNumElements; i++) {
        mElements[i] = negZero<T>();
      }
    }
  };

  PackedT initValue{};
  return initValue.mPacked;
}

// A helper class to get the correct base pointer for a given layout
struct LamportBufferLayout {
  uint32_t numStages = 1;
  uint32_t bytesPerBuffer = 0;
  static constexpr uint32_t sNumLamportBuffers = 3;

  // Implicitly inlined
  [[nodiscard]] __device__ __host__ size_t getTotalBytes() const {
    return numStages * static_cast<size_t>(bytesPerBuffer / numStages) * sNumLamportBuffers;
  }

  // Implicitly inlined
  [[nodiscard]] __device__ __host__ void* getStagePtr(void* bufferBasePtr, uint32_t lamportIndex,
                                                      uint32_t stageIndex) const {
    // Typecast to avoid warnings
    return reinterpret_cast<void*>(
        reinterpret_cast<char*>(bufferBasePtr) +
        static_cast<size_t>((lamportIndex * numStages + stageIndex) *
                            static_cast<size_t>(bytesPerBuffer / numStages)));
  }
};
// Current Index
// Dirty Index
// bytes_per_buffer
// Dirty num_stages
// Dirty bytes_to_clear = {stage0, stage1, stage2, stage3}  # We fix this to 4 stages
// offset_access_ptr

namespace cg = cooperative_groups;

// PackedType is the one used in kernel for Lamport buffer (LDG.128 or LDG.64)
template <typename PackedType = float4, bool UseCGA = false>
__device__ struct __attribute__((aligned(32))) LamportFlags {
 public:
  __device__ explicit LamportFlags(uint32_t* bufferFlags, uint32_t numStages = 1)
      : mBufferFlagsPtr(bufferFlags), mFlagAccessPtr(&bufferFlags[8]) {
    mCurBufferLayout.numStages = numStages;
    uint4 flag = reinterpret_cast<uint4*>(bufferFlags)[0];
    mCurrentIndex = flag.x;
    mDirtyIndex = flag.y;
    // Buffer size is unchanged as the flag should be coupled to each buffer
    mCurBufferLayout.bytesPerBuffer = flag.z;
    mDirtyBufferLayout.bytesPerBuffer = flag.z;
    mDirtyBufferLayout.numStages = flag.w;
    *reinterpret_cast<uint4*>(&mBytesToClear) = reinterpret_cast<uint4*>(bufferFlags)[1];
  }

  // Return the base pointer of the lamport buffer indexed by mCurrentIndex and the stageIdx
  [[nodiscard]] __device__ void* getCurLamportBuf(void* bufferBasePtr, int stageIdx = 0) const {
    return mCurBufferLayout.getStagePtr(bufferBasePtr, mCurrentIndex, stageIdx);
  }

  // Fill the dirty lamport buffer with the init value; Use stageIdx to select the stage to clear,
  // -1 to clear all
  // FIXME: Current kernel may use less stages than the dirty numStages; How to guarantee the
  // correctness? CAUTION: This function requires all threads in the grid to participate and ASSUME
  // 1D thread block layout!
  __device__ void clearDirtyLamportBuf(void* bufferBasePtr, int stageIdx = -1) {
    // Rasterize the threads to 1D for flexible clearing

    uint32_t globalCtaIdx = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t globalTid = globalCtaIdx * blockDim.x + threadIdx.x;
    uint32_t numThreads = gridDim.x * gridDim.y * blockDim.x;

    if (stageIdx == -1) {
      // Clear all stages
      for (uint32_t i = 0; i < mDirtyBufferLayout.numStages; i++) {
        clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[i], mDirtyIndex, i);
      }
    } else if (stageIdx < mDirtyBufferLayout.numStages) {
      clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[stageIdx], mDirtyIndex,
                     stageIdx);
    }
  }

  __device__ void ctaArrive() {
    if constexpr (UseCGA) {
      cg::cluster_group cluster = cg::this_cluster();
      __cluster_barrier_arrive();
      if (cluster.block_rank() == 0 && threadIdx.x < 32) {
        __cluster_barrier_wait();
        arriveCounter(threadIdx.x);
      }
      return;
    }
    uint32_t const barrierThreads = round_up(static_cast<uint32_t>(blockDim.x), 32u);
    // Named CTA barrier avoids a full __syncthreads() while still ordering payload stores
    // before the per-CTA Lamport arrival counter update.
    if (threadIdx.x < 32) {
      asm volatile("barrier.cta.sync 1, %0;" ::"r"(barrierThreads) : "memory");
      arriveCounter(threadIdx.x);
    } else {
      asm volatile("barrier.cta.arrive 1, %0;" ::"r"(barrierThreads) : "memory");
    }
  }

  __device__ void waitAndUpdate(uint4 bytesToClearPerStage) {
    bool isLastCtaT0{false};
    int targetCount{0};
    cg::grid_group grid = cg::this_grid();
    // Use the first thread instead of the last thread as the last thread may exit early
    isLastCtaT0 = grid.thread_rank() == 0;
    if constexpr (UseCGA) {
      cg::cluster_group cluster = cg::this_cluster();
      targetCount = gridDim.x * gridDim.y * gridDim.z / cluster.num_blocks();
    } else {
      targetCount = gridDim.x * gridDim.y * gridDim.z;
    }
    if (isLastCtaT0) {
      uint4* flagPtr = reinterpret_cast<uint4*>(mBufferFlagsPtr);
      while (*reinterpret_cast<uint32_t volatile*>(mFlagAccessPtr) < targetCount) {
      }
      // 'Current' becomes 'Dirty'
      flagPtr[0] = {(mCurrentIndex + 1) % 3,          // Current index
                    mCurrentIndex,                    // Dirty index
                    mCurBufferLayout.bytesPerBuffer,  // Buffer size
                    mCurBufferLayout.numStages};      // Dirty - Number of stages
      flagPtr[1] = bytesToClearPerStage;
      *mFlagAccessPtr = 0;
    }
  }

 private:
  uint32_t* mBufferFlagsPtr;
  uint32_t* mFlagAccessPtr;

  uint32_t mCurrentIndex, mDirtyIndex;
  // So that we can access it with uint4
  alignas(16) std::array<uint32_t, 4> mBytesToClear;
  LamportBufferLayout mCurBufferLayout, mDirtyBufferLayout;

  __device__ void arriveCounter(int tid) {
    if (tid == 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
      asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1)
                   : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
      asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1)
                   : "memory");
#else
      atomicAdd(mFlagAccessPtr, 1);
#endif
    }
  }

  inline __device__ void clearPackedBuf(void* bufferBasePtr, uint32_t globalTid,
                                        uint32_t numThreads, uint32_t bytesToClear,
                                        uint8_t dirtyIndex, uint8_t stageIdx) {
    // Round up to the float4 boundary
    uint32_t clearBoundary = ceil_div<uint32_t>(bytesToClear, sizeof(PackedType));
    for (uint32_t packedIdx = globalTid; packedIdx < clearBoundary; packedIdx += numThreads) {
      reinterpret_cast<PackedType*>(
          mDirtyBufferLayout.getStagePtr(bufferBasePtr, dirtyIndex, stageIdx))[packedIdx] =
          getPackedLamportInit<PackedType, float>();
    }
  }
};

template <typename PackedType, typename T>
union PackedVec {
  PackedType packed;
  T elements[sizeof(PackedType) / sizeof(T)];

  __device__ PackedVec& operator+=(PackedVec& other) {
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++) {
      elements[i] += other.elements[i];
    }
    return *this;
  }

  __device__ PackedVec operator+(PackedVec& other) {
    PackedVec result;
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++) {
      result.elements[i] = elements[i] + other.elements[i];
    }
    return result;
  }
};

template <typename PackedType, typename T>
inline __device__ void sanitizeLamportPayload(PackedVec<PackedType, T>& value) {
#pragma unroll
  for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++) {
    if (isNegZero(value.elements[i])) {
      value.elements[i] = fromFloat<T>(0.f);
    }
  }
}

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr) {
  return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr) {
  return *reinterpret_cast<PackedType const*>(ptr);
}

template <typename PackedType>
union VolatilePackedLoad {
  PackedType packed;
  uint32_t words[sizeof(PackedType) / sizeof(uint32_t)];
};

template <typename PackedType>
inline __device__ VolatilePackedLoad<PackedType> loadPackedVolatile(void const* ptr) {
  static_assert(sizeof(PackedType) == 0, "Not implemented");
  return {};
}

template <>
inline __device__ VolatilePackedLoad<float4> loadPackedVolatile<float4>(void const* ptr) {
  VolatilePackedLoad<float4> returnValue;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(returnValue.words[0]), "=r"(returnValue.words[1]), "=r"(returnValue.words[2]),
                 "=r"(returnValue.words[3])
               : "l"(ptr)
               : "memory");
  return returnValue;
}

template <>
inline __device__ VolatilePackedLoad<float2> loadPackedVolatile<float2>(void const* ptr) {
  VolatilePackedLoad<float2> returnValue;
  asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];\n"
               : "=r"(returnValue.words[0]), "=r"(returnValue.words[1])
               : "l"(ptr)
               : "memory");
  return returnValue;
}

template <typename PackedType>
inline __device__ bool isLamportDirty(VolatilePackedLoad<PackedType> const& value) {
  // The dirty sentinel is a raw word; typed fp compares can flush nearby bit patterns.
  // ptx memeory model only gaurantee atomicity for 64bits granularity so we check every element
  // here to make sure the packed payload is consumed.
  bool dirty = false;
#pragma unroll
  for (int i = 0; i < sizeof(PackedType) / sizeof(uint32_t); i++) {
    dirty |= value.words[i] == kNEGZERO_FP32;
  }
  return dirty;
}

template <int Rank, uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD>
inline __device__ bool pollOneshotRemoteRank(PackedVec<PackedType, T>* remoteValues,
                                             T* stagePtrLocal, int token, int tokenDim,
                                             int packedIdx) {
  if constexpr (Rank == LocalRank) {
    return true;
  } else {
    auto loaded = loadPackedVolatile<PackedType>(
        &stagePtrLocal[token * tokenDim * WorldSize + Rank * tokenDim +
                       packedIdx * kELTS_PER_THREAD]);
    remoteValues[Rank].packed = loaded.packed;
    return !isLamportDirty(loaded);
  }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD, int... Ranks>
inline __device__ bool pollOneshotRemoteRanks(PackedVec<PackedType, T>* remoteValues,
                                              T* stagePtrLocal, int token, int tokenDim,
                                              int packedIdx, std::integer_sequence<int, Ranks...>) {
  bool valid = true;
  ((valid &= pollOneshotRemoteRank<Ranks, WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
        remoteValues, stagePtrLocal, token, tokenDim, packedIdx)),
   ...);
  return valid;
}

template <typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulatePacked(float (&accum)[kELTS_PER_THREAD],
                                        PackedVec<PackedType, T> const& value) {
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] += toFloat<T>(value.elements[i]);
  }
}

template <int Rank, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulateOneshotRank(float (&accum)[kELTS_PER_THREAD],
                                             PackedVec<PackedType, T> const* remoteValues,
                                             PackedVec<PackedType, T> const& localValue) {
  if constexpr (Rank == LocalRank) {
    accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, localValue);
  } else {
    accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, remoteValues[Rank]);
  }
}

template <uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD, int... Ranks>
inline __device__ void accumulateOneshotRanks(float (&accum)[kELTS_PER_THREAD],
                                              PackedVec<PackedType, T> const* remoteValues,
                                              PackedVec<PackedType, T> const& localValue,
                                              std::integer_sequence<int, Ranks...>) {
  (accumulateOneshotRank<Ranks, LocalRank, T, PackedType, kELTS_PER_THREAD>(accum, remoteValues,
                                                                            localValue),
   ...);
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD>
inline __device__ void waitOneshotRemoteRanks(PackedVec<PackedType, T>* remoteValues,
                                              T* stagePtrLocal, int token, int tokenDim,
                                              int packedIdx) {
  static_assert(LocalRank < WorldSize);
  while (1) {
    bool const valid =
        pollOneshotRemoteRanks<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
            remoteValues, stagePtrLocal, token, tokenDim, packedIdx,
            std::make_integer_sequence<int, WorldSize>{});
    if (valid) {
      break;
    }
  }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotDeterministic(
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue) {
  static_assert(LocalRank < WorldSize);
  float accum[kELTS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] = 0.f;
  }
  accumulateOneshotRanks<LocalRank, T, PackedType, kELTS_PER_THREAD>(
      accum, remoteValues, localValue, std::make_integer_sequence<int, WorldSize>{});

  PackedVec<PackedType, T> packedAccum;
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    packedAccum.elements[i] = fromFloat<T>(accum[i]);
  }
  return packedAccum;
}

template <uint8_t WorldSize, int kRankChunk, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceLamportRanksChunked(T* buffer, int token,
                                                                     int tokenDim,
                                                                     int tokenOffset) {
  static_assert(kRankChunk > 0, "kRankChunk must be positive");
  static_assert(WorldSize % kRankChunk == 0, "WorldSize must be divisible by kRankChunk");
  // Chunk ranks for large world sizes to avoid register spills from keeping every rank payload
  // live at once.
  float accum[kELTS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] = 0.f;
  }

#pragma unroll 1
  for (int rankBase = 0; rankBase < WorldSize; rankBase += kRankChunk) {
    float chunkAccum[kELTS_PER_THREAD];
    while (1) {
      bool valid = true;
#pragma unroll
      for (int i = 0; i < kELTS_PER_THREAD; i++) {
        chunkAccum[i] = 0.f;
      }
#pragma unroll
      for (int rr = 0; rr < kRankChunk; rr++) {
        int const r = rankBase + rr;
        auto loaded = loadPackedVolatile<PackedType>(
            &buffer[token * tokenDim * WorldSize + r * tokenDim + tokenOffset]);
        PackedVec<PackedType, T> value;
        value.packed = loaded.packed;
        valid &= !isLamportDirty(loaded);
        accumulatePacked<T, PackedType, kELTS_PER_THREAD>(chunkAccum, value);
      }
      if (valid) {
        break;
      }
    }
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      accum[i] += chunkAccum[i];
    }
  }

  PackedVec<PackedType, T> packedAccum;
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    packedAccum.elements[i] = fromFloat<T>(accum[i]);
  }
  return packedAccum;
}

template <typename T_IN>
inline __device__ void copyF4(T_IN* dst, T_IN const* src) {
  float4* dst4 = reinterpret_cast<float4*>(dst);
  float4 const* src4 = reinterpret_cast<float4 const*>(src);
  __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

uint32_t constexpr kWARP_SIZE = 32U;
uint32_t constexpr kLOG2_WARP_SIZE = 5U;
uint32_t constexpr kLANE_ID_MASK = 0x1f;
uint32_t constexpr kFINAL_MASK = 0xffffffff;

template <typename T>
inline __device__ T warpReduceSumFull(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(kFINAL_MASK, val, mask, kWARP_SIZE);
  }
  return val;
}

template <typename T>
inline __device__ T warpReduceMaxFull(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(kFINAL_MASK, val, mask, kWARP_SIZE));
  }
  return val;
}

template <typename T>
inline __device__ T warpReduceMaxPartial(T val) {
  int laneId = threadIdx.x & kLANE_ID_MASK;
  int warpSize = blockDim.x - (threadIdx.x & ~(kWARP_SIZE - 1));
  unsigned int active_mask = (1U << warpSize) - 1;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    int targetLane = laneId ^ mask;
    auto tmp = __shfl_xor_sync(active_mask, val, mask, kWARP_SIZE);
    val = targetLane < warpSize ? fmaxf(val, tmp) : val;
  }
  return val;
}

template <typename T>
inline __device__ T warpReduceSumPartial(T val) {
  int laneId = threadIdx.x & kLANE_ID_MASK;
  // We make sure only the last warp will call this function
  int warpSize = blockDim.x - (threadIdx.x & ~(kWARP_SIZE - 1));
  unsigned int active_mask = (1U << warpSize) - 1;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    int targetLane = laneId ^ mask;
    auto tmp = __shfl_xor_sync(active_mask, val, mask, kWARP_SIZE);
    val += targetLane < warpSize ? tmp : 0;
  }
  return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceMaxPartial(T val) {
  __shared__ T smem[kWARP_SIZE];
  int laneId = threadIdx.x & kLANE_ID_MASK;
  int warpId = threadIdx.x >> kLOG2_WARP_SIZE;
  int warpNum = (blockDim.x + kWARP_SIZE - 1) >> kLOG2_WARP_SIZE;

  val = (warpId == warpNum - 1) ? warpReduceMaxPartial(val) : warpReduceMaxFull(val);
  if (laneId == 0) {
    smem[warpId] = val;
  }
  __syncthreads();

  if (warpId == 0) {
    val = (laneId < warpNum) ? smem[laneId]
                             : fromFloat<T>(-cuda::std::numeric_limits<float>::infinity());
    val = (warpNum == 1) ? warpReduceMaxPartial(val) : warpReduceMaxFull(val);

    if constexpr (SYNC) {
      if (laneId == 0) {
        smem[warpId] = val;
      }
    }
  }
  if constexpr (SYNC) {
    __syncthreads();
    val = smem[0];
  }
  return val;
}

template <typename T>
inline __device__ T blockReduceMaxFull(T val) {
  __shared__ T smem[kWARP_SIZE];
  int lane_id = threadIdx.x & kLANE_ID_MASK;
  int warp_id = threadIdx.x >> kLOG2_WARP_SIZE;
  int warp_num = blockDim.x >> kLOG2_WARP_SIZE;

  val = warpReduceMaxFull(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();

  val = (lane_id < warp_num) ? smem[lane_id]
                             : fromFloat<T>(-cuda::std::numeric_limits<float>::infinity());
  val = warpReduceMaxFull(val);

  return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceMax(T val) {
  bool hasPartialWarp = (blockDim.x & kLANE_ID_MASK) != 0;
  if (hasPartialWarp) {
    return blockReduceMaxPartial<T, SYNC>(val);
  } else {
    return blockReduceMaxFull<T>(val);
  }
}

// SYNC:
//  - True: share the sum across all threads
//  - False: only thread 0 get the sum; Other thread's value is undefined.
template <typename T, bool SYNC = false>
inline __device__ T blockReduceSumPartial(T val) {
  __shared__ T smem[kWARP_SIZE];
  int laneId = threadIdx.x & kLANE_ID_MASK;
  int warpId = threadIdx.x >> kLOG2_WARP_SIZE;
  int warpNum = (blockDim.x + kWARP_SIZE - 1) >>
                kLOG2_WARP_SIZE;  // Ceiling division to include partial warps

  val = (warpId == warpNum - 1) ? warpReduceSumPartial(val) : warpReduceSumFull(val);
  if (laneId == 0) {
    smem[warpId] = val;
  }
  __syncthreads();

  if (warpId == 0) {
    val = (laneId < warpNum) ? smem[laneId] : (T)0.f;
    // Need to consider the corner case where we only have one warp and it is partial
    val = (warpNum == 1) ? warpReduceSumPartial(val) : warpReduceSumFull(val);

    if constexpr (SYNC) {
      if (laneId == 0) {
        smem[warpId] = val;
      }
    }
  }
  if constexpr (SYNC) {
    __syncthreads();
    val = smem[0];
  }
  return val;
}

template <typename T>
inline __device__ T blockReduceSumFull(T val) {
  __shared__ T smem[kWARP_SIZE];
  int lane_id = threadIdx.x & kLANE_ID_MASK;
  int warp_id = threadIdx.x >> kLOG2_WARP_SIZE;
  int warp_num = blockDim.x >> kLOG2_WARP_SIZE;

  val = warpReduceSumFull(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();

  val = (lane_id < warp_num) ? smem[lane_id] : (T)0.f;
  val = warpReduceSumFull(val);

  return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceSum(T val) {
  bool hasPartialWarp = (blockDim.x & kLANE_ID_MASK) != 0;
  if (hasPartialWarp) {
    return blockReduceSumPartial<T, SYNC>(val);
  } else {
    return blockReduceSumFull<T>(val);
  }
}
// Tune the grid configuration for fused oneshot and RMSNorm kernels.
// Return (block_size, cluster_size, loads_per_thread).
std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread,
                                           bool useCluster) {
  // Step 1: start from the widest cluster we are willing to launch. MNNVL JIT only
  // targets SM90/SM100, but RMSNorm may request a no-CGA fallback for multi-load rows.
  int clusterSize = useCluster ? kMaxClusterSize : 1;
  int blockSize = 128;
  int threadsNeeded = ceil_div(dim, eltsPerThread);
  int loadsPerThread = 1;

  blockSize = ceil_div(threadsNeeded, clusterSize);
  if (useCluster) {
    // Step 2: shrink the cluster until the hidden dimension partitions cleanly across CTAs.
    // This keeps each CTA responsible for an integral chunk of packed elements.
    while (threadsNeeded % clusterSize != 0 && clusterSize > 1) {
      clusterSize /= 2;
    }
    int const maxDivisibleClusterSize = clusterSize;
    blockSize = ceil_div(threadsNeeded, clusterSize);
    // Step 3: if divisibility leaves each CTA too small, trade cluster width for at least
    // a 128-thread CTA. This improves occupancy and avoids tiny CTAs when the row is narrow.
    while (blockSize < 128 && clusterSize >= 2) {
      blockSize *= 2;
      clusterSize /= 2;
    }
    int smCount = GetCudaMultiProcessorCount();
    // Step 4: if the token grid already has enough CTAs to cover the GPU, reduce cluster
    // width and make CTAs larger. This avoids over-partitioning one token across too many CTAs.
    while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512) {
      blockSize *= 2;
      clusterSize /= 2;
    }
    // Step 5: if the token grid still underfills the GPU, restore cluster width up to the
    // divisibility limit. We accept 64-thread CTAs here to expose more CTAs per token.
    while (clusterSize < maxDivisibleClusterSize) {
      int const candidateClusterSize = clusterSize * 2;
      int const candidateBlockSize = ceil_div(threadsNeeded, candidateClusterSize);
      if (candidateBlockSize < 64 || numTokens * candidateClusterSize > smCount) {
        break;
      }
      clusterSize = candidateClusterSize;
      blockSize = candidateBlockSize;
    }
  }
  // Step 6: for very wide rows, first increase cluster width on SM90+ and then increase
  // per-thread loads. The goal is to keep block_size within CUDA's 1024-thread limit.
  while (blockSize > 1024) {
    if (useCluster && clusterSize < kMaxClusterSize) {
      clusterSize = clusterSize << 1;
    } else if (loadsPerThread < 8) {
      loadsPerThread += 1;
    } else {
      break;
    }
    blockSize = ceil_div(threadsNeeded, clusterSize * loadsPerThread);
  }
  return {blockSize, clusterSize, loadsPerThread};
}
};  // namespace utils

using utils::blockReduceMax;
using utils::blockReduceSum;
using utils::fromFloat;
using utils::isLamportDirty;
using utils::isNegZero;
using utils::LamportFlags;
using utils::loadPacked;
using utils::loadPackedVolatile;
using utils::PackedVec;
using utils::sanitizeLamportPayload;
using utils::toFloat;

namespace quant {

template <typename T, typename PackedType, int ELTS_PER_THREAD>
inline __device__ void quant_fp8(PackedVec<PackedType, T> packedAccum, void* quantOutPtr,
                                 float invOutputScale, uint32_t threadOffset) {
  static_assert(ELTS_PER_THREAD == 8 || ELTS_PER_THREAD == 4, "ELTS_PER_THREAD must be 8 or 4");
  using QuantizedPackedType = std::conditional_t<ELTS_PER_THREAD == 8, float2, float>;

  auto quantOut = reinterpret_cast<__nv_fp8_e4m3*>(quantOutPtr);
  PackedVec<QuantizedPackedType, __nv_fp8_e4m3> quantizedAccum;
#pragma unroll
  for (int i = 0; i < ELTS_PER_THREAD; i++) {
    quantizedAccum.elements[i] =
        __nv_fp8_e4m3(toFloat<T>(packedAccum.elements[i]) * invOutputScale);
  }
  reinterpret_cast<QuantizedPackedType*>(&quantOut[threadOffset])[0] = quantizedAccum.packed;
}

template <typename T, typename PackedType, int ELTS_PER_THREAD>
inline __device__ void quant_nvfp4(PackedVec<PackedType, T> packedAccum, void* quantOutPtr,
                                   void* sfOutPtr, float* outputScale, uint32_t tokenIdx,
                                   uint32_t tokenDim, uint32_t packedIdx,
                                   QuantizationSFLayout sfLayout) {
#if CUDA_VERSION >= 12080
  static_assert(
      ELTS_PER_THREAD == 8 && (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>),
      "NVFP4 quantization fusion is only supported for FP16/BF16!");

  auto packedAccumVec = *reinterpret_cast<vec_t<T, ELTS_PER_THREAD>*>(&packedAccum);
  auto sfOut = trtllm_allreduce_fusion::utils::cvt_quant_to_fp4_get_sf_out_offset<
      uint32_t, trtllm_allreduce_fusion::details::CVT_FP4_SF_VEC_SIZE / ELTS_PER_THREAD>(
      cuda::std::nullopt, tokenIdx, packedIdx, cuda::std::nullopt, tokenDim,
      reinterpret_cast<uint32_t*>(sfOutPtr), sfLayout);

  uint32_t quantOutOffset = tokenIdx * tokenDim / ELTS_PER_THREAD + packedIdx;
  reinterpret_cast<uint32_t*>(quantOutPtr)[quantOutOffset] =
      trtllm_allreduce_fusion::utils::cvt_warp_fp16_to_fp4<T, ELTS_PER_THREAD>(packedAccumVec,
                                                                               *outputScale, sfOut);
#else
  (void)packedAccum;
  (void)quantOutPtr;
  (void)sfOutPtr;
  (void)outputScale;
  (void)tokenIdx;
  (void)tokenDim;
  (void)packedIdx;
  (void)sfLayout;
#endif
}

}  // namespace quant

template <uint8_t WorldSize, typename T, bool RMSNormFusion = false,
          QuantType QType = QuantType::kNone, typename PackedType = float4>
__global__ void __launch_bounds__(1024)
    oneshotAllreduceFusionKernel(AllReduceKernelParams<T> params) {
  static_assert(QType == QuantType::kNone || RMSNormFusion,
                "Quant-only pattern without RMSNorm is not supported!");
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kELT_SIZE = sizeof(T);
  int const tokenDim = params.tokenDim;
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int packedIdx = cluster.thread_rank();
  int token = blockIdx.x;
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

  cudaGridDependencySynchronize();
  // We only use 1 stage for the oneshot allreduce
  constexpr bool kUseLamportCGA = true;
  LamportFlags<PackedType, kUseLamportCGA> flag(params.bufferFlags, 1);
  T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, 0));
  T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], 0));

  if (packedIdx * kELTS_PER_THREAD >= tokenDim) {
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);
    return;
  }

  // ==================== Broadcast tokens to each rank =============================
  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
  sanitizeLamportPayload<PackedType, T>(val);

  reinterpret_cast<PackedType*>(
      &stagePtrMcast[token * tokenDim * WorldSize + params.rank * tokenDim])[packedIdx] =
      val.packed;

  flag.ctaArrive();
  // ======================= Lamport Sync and clear the output buffer from previous iteration
  // =============================
  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);

  // ======================= Reduction =============================
  // Fully deterministic: every rank uses the exact same reduction order.
  // For WorldSize <= 8, specialize the local slot so the fast path reuses `val`
  // from registers without a dynamic `remoteValues[params.rank]` store. Larger
  // world sizes use the compact fallback because the benefit is thin but
  // specializing every rank significantly increases JIT compile time.
  PackedVec<PackedType, T> packedAccum;
  if constexpr (WorldSize <= 8) {
    packedAccum = val;
#define RUN_ONESHOT_LOCAL_RANK(LOCAL_RANK)                                                   \
  case LOCAL_RANK:                                                                           \
    if constexpr (WorldSize > LOCAL_RANK) {                                                  \
      PackedVec<PackedType, T> remoteValues[WorldSize];                                      \
      utils::waitOneshotRemoteRanks<WorldSize, LOCAL_RANK, T, PackedType, kELTS_PER_THREAD>( \
          remoteValues, stagePtrLocal, token, tokenDim, packedIdx);                          \
      packedAccum = utils::reduceOneshotDeterministic<WorldSize, LOCAL_RANK, T, PackedType,  \
                                                      kELTS_PER_THREAD>(remoteValues, val);  \
    }                                                                                        \
    break

    switch (params.rank) {
      RUN_ONESHOT_LOCAL_RANK(0);
      RUN_ONESHOT_LOCAL_RANK(1);
      RUN_ONESHOT_LOCAL_RANK(2);
      RUN_ONESHOT_LOCAL_RANK(3);
      RUN_ONESHOT_LOCAL_RANK(4);
      RUN_ONESHOT_LOCAL_RANK(5);
      RUN_ONESHOT_LOCAL_RANK(6);
      RUN_ONESHOT_LOCAL_RANK(7);
    }
#undef RUN_ONESHOT_LOCAL_RANK
  } else {
    // Chunk large-world reductions to avoid register spills from a live values[WorldSize] array.
    packedAccum = utils::reduceLamportRanksChunked<WorldSize, 8, T, PackedType, kELTS_PER_THREAD>(
        stagePtrLocal, token, tokenDim, packedIdx * kELTS_PER_THREAD);
  }

  cudaTriggerProgrammaticLaunchCompletion();
  if constexpr (RMSNormFusion) {
    // =============================== Residual ===============================
    PackedVec<PackedType, T> residualIn;
    residualIn.packed = *reinterpret_cast<PackedType const*>(&params.residualInPtr[threadOffset]);
    packedAccum += residualIn;
    if (params.prenormedPtr != nullptr) {
      *reinterpret_cast<PackedType*>(&params.prenormedPtr[threadOffset]) = packedAccum.packed;
    }
    // =============================== Rmsnorm ================================
    PackedVec<PackedType, T> gamma;
    gamma.packed =
        *reinterpret_cast<PackedType const*>(&params.gammaPtr[packedIdx * kELTS_PER_THREAD]);

    float threadSum = 0.F;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      // FIXME: Use float square if accuracy issue
      threadSum += toFloat<T>(packedAccum.elements[i] * packedAccum.elements[i]);
    }
    float blockSum = blockReduceSum<float, true>(threadSum);

    __shared__ float sharedVal[8];  // Temporary variable to share the sum within block
    float fullSum = blockSum;
    int const numBlocks = cluster.num_blocks();
    if (numBlocks > 1) {
      fullSum = 0.F;
      // Need to reduce over the entire cluster
      int const blockRank = cluster.block_rank();
      if (threadIdx.x < numBlocks) {
        cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
      }
      cluster.barrier_wait(cluster.barrier_arrive());
      for (int i = 0; i < numBlocks; ++i) {
        fullSum += sharedVal[i];
      }
    }
    float rcpRms = rsqrtf(fullSum / tokenDim + params.epsilon);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] = fromFloat<T>(toFloat<T>(packedAccum.elements[i]) * rcpRms *
                                             (params.weightBias + toFloat<T>(gamma.elements[i])));
    }
  }
  if (params.outputPtr != nullptr) {
    reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = packedAccum.packed;
  }
  if constexpr (QType == QuantType::kFP8) {
    float invOutputScale = 1.0f / (*params.outputScalePtr);
    quant::quant_fp8<T, PackedType, kELTS_PER_THREAD>(packedAccum, params.quantOutPtr,
                                                      invOutputScale, threadOffset);
  }
#if CUDA_VERSION >= 12080
  else if constexpr (QType == QuantType::kFP4) {
    quant::quant_nvfp4<T, PackedType, kELTS_PER_THREAD>(
        packedAccum, params.quantOutPtr, params.scalingFactorOutPtr, params.outputScalePtr, token,
        tokenDim, packedIdx, params.sfLayout);
  }
#endif
  else if constexpr (QType == QuantType::kDynamicFP8) {
    float threadMax = 0.F;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      threadMax = fmaxf(threadMax, fabsf(toFloat<T>(packedAccum.elements[i])));
    }
    float tokenMax = blockReduceMax<float, true>(threadMax);
    __shared__ float sharedMax[utils::kMaxClusterSize];
    int const numBlocks = cluster.num_blocks();
    int const blockRank = cluster.block_rank();
    if (numBlocks > 1) {
      float fullMax = 0.F;
      if (threadIdx.x < numBlocks) {
        cluster.map_shared_rank(&sharedMax[0], threadIdx.x)[blockRank] = tokenMax;
      }
      cluster.barrier_wait(cluster.barrier_arrive());
      for (int i = 0; i < numBlocks; ++i) {
        fullMax = fmaxf(fullMax, sharedMax[i]);
      }
      tokenMax = fullMax;
    }
    float tokenScale = fmaxf(tokenMax / utils::kFP8E4M3Max, utils::kDynamicFP8MinScale);
    if (threadIdx.x == 0 && blockRank == 0) {
      reinterpret_cast<float*>(params.scalingFactorOutPtr)[token] = tokenScale;
    }
    using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
    PackedQuantizedType quantPacked;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      float q = toFloat<T>(packedAccum.elements[i]) / tokenScale;
      q = fminf(fmaxf(q, -utils::kFP8E4M3Max), utils::kFP8E4M3Max);
      reinterpret_cast<__nv_fp8_e4m3*>(&quantPacked)[i] = static_cast<__nv_fp8_e4m3>(q);
    }
    reinterpret_cast<PackedQuantizedType*>(
        &reinterpret_cast<__nv_fp8_e4m3*>(params.quantOutPtr)[threadOffset])[0] = quantPacked;
  }
  flag.waitAndUpdate(
      {static_cast<uint32_t>(params.numTokens * tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

using utils::adjustGridConfig;

template <typename T>
cudaError_t oneshotAllreduceFusionDispatch(AllReduceFusionParams const& params) {
  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const eltsPerThread = sizeof(float4) / sizeof(T);

  auto [blockSize, clusterSize, loadsPerThread] =
      adjustGridConfig(numTokens, tokenDim, eltsPerThread, true);
  dim3 grid(numTokens, clusterSize, 1);

  FLASHINFER_LOG_DEBUG(
      "[MNNVL AllReduceOneShot] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: "
      "%d, "
      "loads_per_thread: %d, "
      "threads_needed: %d",
      numTokens, clusterSize, blockSize, clusterSize, loadsPerThread,
      ceil_div(tokenDim, eltsPerThread));

  FLASHINFER_CHECK(blockSize <= 1024 && loadsPerThread == 1,
                   "Hidden Dimension %d exceeds the maximum supported hidden dimension (%d)",
                   tokenDim, 1024 * 8 * eltsPerThread);

  if (!params.rmsNormFusion && params.quantType != QuantType::kNone) {
    FLASHINFER_ERROR("[MNNVL AllReduceOneShot] Quantization requires RMSNorm fusion");
    return cudaErrorInvalidValue;
  }
  if (params.quantType != QuantType::kNone) {
    if (params.quantOut == nullptr) {
      FLASHINFER_ERROR(
          "[MNNVL AllReduceOneShot] quantOut must be non-null when quantization is enabled");
      return cudaErrorInvalidValue;
    }
    if (params.quantType != QuantType::kDynamicFP8 && params.outputScale == nullptr) {
      FLASHINFER_ERROR(
          "[MNNVL AllReduceOneShot] outputScale must be non-null for static quantization");
      return cudaErrorInvalidValue;
    }
  }
  if (params.quantType == QuantType::kDynamicFP8 && params.scalingFactorOut == nullptr) {
    FLASHINFER_ERROR("[MNNVL AllReduceOneShot] scale_out is required for dynamic FP8");
    return cudaErrorInvalidValue;
  }
  if (params.quantType == QuantType::kFP4) {
#if CUDA_VERSION < 12080
    FLASHINFER_ERROR("[MNNVL AllReduceOneShot] FP4 quantization requires CUDA 12.8+");
    return cudaErrorInvalidValue;
#else
    if (params.scalingFactorOut == nullptr) {
      FLASHINFER_ERROR(
          "[MNNVL AllReduceOneShot] scalingFactorOut is required for FP4 quantization");
      return cudaErrorInvalidValue;
    }
    if (tokenDim % trtllm_allreduce_fusion::details::CVT_FP4_SF_VEC_SIZE != 0) {
      FLASHINFER_ERROR("[MNNVL AllReduceOneShot] FP4 quantization requires tokenDim divisible by " +
                       std::to_string(trtllm_allreduce_fusion::details::CVT_FP4_SF_VEC_SIZE));
      return cudaErrorInvalidValue;
    }
#endif
  }
  if (clusterSize > utils::kMaxClusterSize) {
    FLASHINFER_ERROR("[MNNVL AllReduceOneShot] cluster_size " + std::to_string(clusterSize) +
                     " exceeds shared max buffer " + std::to_string(utils::kMaxClusterSize));
    return cudaErrorInvalidValue;
  }

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = 1;
  attrs[1].val.clusterDim.y = clusterSize;
  attrs[1].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{
      .gridDim = grid,
      .blockDim = blockSize,
      .dynamicSmemBytes = 0,
      .stream = params.stream,
      .attrs = attrs,
      .numAttrs = 2,
  };

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, RMSNORM, QTYPE) \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                  \
      &config, &oneshotAllreduceFusionKernel<WORLD_SIZE, T, RMSNORM, QTYPE>, kernelParams));
#define DISPATCH_ALLREDUCE_KERNEL(WORLD_SIZE)                                        \
  if (params.rmsNormFusion) {                                                        \
    switch (params.quantType) {                                                      \
      case QuantType::kFP8:                                                          \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, true, QuantType::kFP8);                  \
        break;                                                                       \
      case QuantType::kFP4:                                                          \
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) { \
          LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, true, QuantType::kFP4);                \
        } else {                                                                     \
          FLASHINFER_ERROR(                                                          \
              "[MNNVL AllReduceOneShot] FP4 quantization is only "                   \
              "supported for FP16/BF16");                                            \
          return cudaErrorInvalidValue;                                              \
        }                                                                            \
        break;                                                                       \
      case QuantType::kDynamicFP8:                                                   \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, true, QuantType::kDynamicFP8);           \
        break;                                                                       \
      case QuantType::kNone:                                                         \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, true, QuantType::kNone);                 \
        break;                                                                       \
      default:                                                                       \
        FLASHINFER_ERROR("[MNNVL AllReduceOneShot] Unsupported quant type " +        \
                         std::to_string(static_cast<int>(params.quantType)));        \
        return cudaErrorInvalidValue;                                                \
    }                                                                                \
  } else {                                                                           \
    LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, false, QuantType::kNone);                    \
  }

  AllReduceKernelParams<T> kernelParams{
      .outputPtr = reinterpret_cast<T*>(params.output),
      .prenormedPtr = reinterpret_cast<T*>(params.residualOut),
      .shardPtr = reinterpret_cast<T const*>(params.input),
      .residualInPtr = reinterpret_cast<T const*>(params.residualIn),
      .gammaPtr = reinterpret_cast<T const*>(params.gamma),
      .inputPtrs = reinterpret_cast<T**>(params.bufferPtrsDev),
      .mcastPtr = reinterpret_cast<T*>(params.multicastPtr),
      .localBufferPtr = reinterpret_cast<T*>(params.bufferPtrLocal),
      .nRanks = params.nRanks,
      .rank = params.rank,
      .numTokens = params.numTokens,
      .tokenDim = params.tokenDim,
      .epsilon = static_cast<float>(params.epsilon),
      .weightBias = params.weightBias,
      .outputScalePtr = params.outputScale,
      .quantOutPtr = params.quantOut,
      .scalingFactorOutPtr = params.scalingFactorOut,
      .sfLayout = params.sfLayout,
      .bufferFlags = params.bufferFlags,
  };

  switch (params.nRanks) {
    case 2:
      DISPATCH_ALLREDUCE_KERNEL(2);
      break;
    case 4:
      DISPATCH_ALLREDUCE_KERNEL(4);
      break;
    case 8:
      DISPATCH_ALLREDUCE_KERNEL(8);
      break;
    case 16:
      DISPATCH_ALLREDUCE_KERNEL(16);
      break;
    case 32:
      DISPATCH_ALLREDUCE_KERNEL(32);
      break;
    case 64:
      DISPATCH_ALLREDUCE_KERNEL(64);
      break;
    default:
      FLASHINFER_ERROR("MNNVL AllReduce: unsupported world_size " + std::to_string(params.nRanks) +
                       ". Supported sizes: {2, 4, 8, 16, 32, 64}");
      return cudaErrorInvalidValue;
  }
#undef DISPATCH_ALLREDUCE_KERNEL
#undef LAUNCH_ALLREDUCE_KERNEL
  return cudaSuccess;
}

enum MNNVLTwoShotStage : uint8_t {
  SCATTER = 0,
  BROADCAST = 1,
  NUM_STAGES = 2,
};

using utils::copyF4;

template <uint8_t WorldSize, typename T, bool WaitForResults = true, typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshotAllreduceKernel(AllReduceKernelParams<T> params) {
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kELT_SIZE = sizeof(T);

  int const packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int const token = blockIdx.x;
  int const tokenOffset = packedIdx * kELTS_PER_THREAD;
  int const threadOffset = token * params.tokenDim + tokenOffset;
  bool const inBounds = tokenOffset < params.tokenDim;
  int const destRank = token % WorldSize;
  int const destTokenOffset = token / WorldSize;

  cudaGridDependencySynchronize();

  LamportFlags<PackedType> flag(params.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
  T* scatterBufLocal = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER));
  T* scatterBufDest = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
  T* broadcastBufW =
      reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, MNNVLTwoShotStage::BROADCAST));
  T* broadcastBufR = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST));

  PackedVec<PackedType, T> val;
  if (inBounds) {
    val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
    sanitizeLamportPayload<PackedType, T>(val);

    reinterpret_cast<PackedType*>(&scatterBufDest[destTokenOffset * params.tokenDim * WorldSize +
                                                  params.rank * params.tokenDim])[packedIdx] =
        val.packed;
  }

  cudaTriggerProgrammaticLaunchCompletion();
  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER);

  if (inBounds && destRank == params.rank) {
    int const localToken = token / WorldSize;
    // Fully deterministic: every rank uses the exact same reduction order. Chunking avoids
    // register spills for large world sizes.
    constexpr int kRankChunk = WorldSize < 16 ? WorldSize : 16;
    PackedVec<PackedType, T> packedAccum =
        utils::reduceLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kELTS_PER_THREAD>(
            scatterBufLocal, localToken, params.tokenDim, tokenOffset);
    // Reduced values can round to the dirty sentinel; sanitize before broadcast polling.
    sanitizeLamportPayload<PackedType, T>(packedAccum);
    reinterpret_cast<PackedType*>(&broadcastBufW[token * params.tokenDim])[packedIdx] =
        packedAccum.packed;
  }

  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST);

  if constexpr (WaitForResults) {
    // OOB threads still arrive so waitAndUpdate sees every CTA.
    flag.ctaArrive();

    if (inBounds) {
      auto loaded = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
      while (isLamportDirty(loaded)) {
        loaded = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
      }
      reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = loaded.packed;
    }

    flag.waitAndUpdate(
        {static_cast<uint32_t>(round_up(params.numTokens, WorldSize) * params.tokenDim * kELT_SIZE),
         static_cast<uint32_t>(params.numTokens * params.tokenDim * kELT_SIZE), 0, 0});
  }
}

template <typename T, QuantType QType = QuantType::kNone, bool UseCGA = false,
          int LoadsPerThread = 1, typename PackedType = float4>
__global__ __launch_bounds__(1024) void rmsNormLamport(AllReduceKernelParams<T> params) {
  constexpr int kELTS_PER_LOAD = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kELT_SIZE = sizeof(T);

  uint32_t const token = blockIdx.x;
  uint32_t const blockSize = blockDim.x;
  uint32_t const threadOffset = threadIdx.x;

  uint32_t numThreads = blockSize;
  uint32_t clusterSize = 1;
  uint32_t clusterBlockRank = 0;
  if constexpr (UseCGA) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    numThreads = cluster.num_threads();
    clusterSize = cluster.num_blocks();
    clusterBlockRank = cluster.block_rank();
  }

  uint32_t const dimPadded = round_up(params.tokenDim, kELTS_PER_LOAD * numThreads);
  uint32_t const elemsPerThread = dimPadded / numThreads;
  uint32_t const loadStride = blockSize;
  uint32_t const blockChunkSize =
      ceil_div(params.tokenDim, clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
  uint32_t const baseTokenOffset = clusterBlockRank * blockChunkSize;

  extern __shared__ uint8_t smem[];
  uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T);
  T* smemInput = reinterpret_cast<T*>(&smem[0]);
  T* smemResidual = reinterpret_cast<T*>(&smem[smemBufferSize]);
  T* smemGamma = reinterpret_cast<T*>(&smem[2 * smemBufferSize]);

  cudaTriggerProgrammaticLaunchCompletion();

  LamportFlags<PackedType, UseCGA> flag(params.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
  T* input = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.localBufferPtr, MNNVLTwoShotStage::BROADCAST));

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    uint32_t const tokenOffset = baseTokenOffset + chunkOffset;
    if (tokenOffset < params.tokenDim) {
      copyF4(&smemResidual[chunkOffset],
             &params.residualInPtr[token * params.tokenDim + tokenOffset]);
    }
  }
  __pipeline_commit();
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    uint32_t const tokenOffset = baseTokenOffset + chunkOffset;
    if (tokenOffset < params.tokenDim) {
      copyF4(&smemGamma[chunkOffset], &params.gammaPtr[tokenOffset]);
    }
  }
  __pipeline_commit();

  flag.ctaArrive();

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    uint32_t const tokenOffset = baseTokenOffset + chunkOffset;
    if (tokenOffset < params.tokenDim) {
      auto loaded = loadPackedVolatile<PackedType>(&input[token * params.tokenDim + tokenOffset]);
      while (isLamportDirty(loaded)) {
        loaded = loadPackedVolatile<PackedType>(&input[token * params.tokenDim + tokenOffset]);
      }
      reinterpret_cast<PackedType*>(&smemInput[chunkOffset])[0] = loaded.packed;
    }
  }

  float rInput[LoadsPerThread * kELTS_PER_LOAD];
  float threadSum = 0.f;

  // Residual and gamma staging use normal cp.async groups; residual is consumed first.
  __pipeline_wait_prior(1);

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    uint32_t const tokenOffset = baseTokenOffset + chunkOffset;
    if (tokenOffset < params.tokenDim) {
      PackedVec<PackedType, T> inp{.packed = loadPacked<PackedType>(&smemInput[chunkOffset])};
      PackedVec<PackedType, T> res{.packed = loadPacked<PackedType>(&smemResidual[chunkOffset])};
      PackedVec<PackedType, T> inpPlusRes = inp + res;
      if (params.prenormedPtr != nullptr) {
        reinterpret_cast<PackedType*>(
            &params.prenormedPtr[token * params.tokenDim + tokenOffset])[0] = inpPlusRes.packed;
      }

#pragma unroll
      for (int j = 0; j < kELTS_PER_LOAD; j++) {
        float const value = toFloat<T>(inpPlusRes.elements[j]);
        rInput[i * kELTS_PER_LOAD + j] = value;
        threadSum += value * value;
      }
    }
  }

  __pipeline_wait_prior(0);

  float blockSum = blockReduceSum<float, true>(threadSum);
  float fullSum = blockSum;
  __shared__ float sharedVal[8];
  if constexpr (UseCGA) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const numBlocks = cluster.num_blocks();
    if (numBlocks > 1) {
      fullSum = 0.F;
      int const blockRank = cluster.block_rank();
      if (threadIdx.x < numBlocks) {
        cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
      }
      cluster.barrier_wait(cluster.barrier_arrive());
      for (int i = 0; i < numBlocks; ++i) {
        fullSum += sharedVal[i];
      }
    }
  }

  float const rcpRms = rsqrtf(fullSum / params.tokenDim + params.epsilon);
  float rNorm[LoadsPerThread * kELTS_PER_LOAD];
  float threadMax = 0.F;
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    uint32_t const tokenOffset = baseTokenOffset + chunkOffset;
    if (tokenOffset < params.tokenDim) {
      PackedVec<PackedType, T> gamma{.packed = loadPacked<PackedType>(&smemGamma[chunkOffset])};
      PackedVec<PackedType, T> out;
#pragma unroll
      for (int j = 0; j < kELTS_PER_LOAD; j++) {
        float normVal = (params.weightBias + toFloat<T>(gamma.elements[j])) *
                        rInput[i * kELTS_PER_LOAD + j] * rcpRms;
        rNorm[i * kELTS_PER_LOAD + j] = normVal;
        threadMax = fmaxf(threadMax, fabsf(normVal));
        out.elements[j] = fromFloat<T>(normVal);
      }
      if (params.outputPtr != nullptr) {
        reinterpret_cast<PackedType*>(&params.outputPtr[token * params.tokenDim + tokenOffset])[0] =
            out.packed;
      }
      if constexpr (QType == QuantType::kFP8) {
        float invOutputScale = 1.0f / (*params.outputScalePtr);
        quant::quant_fp8<T, PackedType, kELTS_PER_LOAD>(out, params.quantOutPtr, invOutputScale,
                                                        token * params.tokenDim + tokenOffset);
      }
#if CUDA_VERSION >= 12080
      else if constexpr (QType == QuantType::kFP4) {
        quant::quant_nvfp4<T, PackedType, kELTS_PER_LOAD>(
            out, params.quantOutPtr, params.scalingFactorOutPtr, params.outputScalePtr, token,
            params.tokenDim, tokenOffset / kELTS_PER_LOAD, params.sfLayout);
      }
#endif
    }
  }

  if constexpr (QType == QuantType::kDynamicFP8) {
    float tokenMax = blockReduceMax<float, true>(threadMax);
    __shared__ float sharedMax[utils::kMaxClusterSize];
    if constexpr (UseCGA) {
      namespace cg = cooperative_groups;
      cg::cluster_group cluster = cg::this_cluster();
      if (clusterSize > 1) {
        float fullMax = 0.F;
        int const blockRank = cluster.block_rank();
        if (threadIdx.x < clusterSize) {
          cluster.map_shared_rank(&sharedMax[0], threadIdx.x)[blockRank] = tokenMax;
        }
        cluster.barrier_wait(cluster.barrier_arrive());
        for (int i = 0; i < clusterSize; ++i) {
          fullMax = fmaxf(fullMax, sharedMax[i]);
        }
        tokenMax = fullMax;
      }
    }
    float tokenScale = fmaxf(tokenMax / utils::kFP8E4M3Max, utils::kDynamicFP8MinScale);
    if (threadIdx.x == 0 && clusterBlockRank == 0) {
      reinterpret_cast<float*>(params.scalingFactorOutPtr)[token] = tokenScale;
    }
    using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
      uint32_t const tokenOffset = baseTokenOffset + chunkOffset;
      if (tokenOffset < params.tokenDim) {
        PackedQuantizedType quantPacked;
#pragma unroll
        for (int j = 0; j < kELTS_PER_LOAD; j++) {
          float q = rNorm[i * kELTS_PER_LOAD + j] / tokenScale;
          q = fminf(fmaxf(q, -utils::kFP8E4M3Max), utils::kFP8E4M3Max);
          reinterpret_cast<__nv_fp8_e4m3*>(&quantPacked)[j] = static_cast<__nv_fp8_e4m3>(q);
        }
        reinterpret_cast<PackedQuantizedType*>(&reinterpret_cast<__nv_fp8_e4m3*>(
            params.quantOutPtr)[token * params.tokenDim + tokenOffset])[0] = quantPacked;
      }
    }
  }

  cudaGridDependencySynchronize();

  flag.waitAndUpdate({static_cast<uint32_t>(round_up(params.numTokens, params.nRanks) *
                                            params.tokenDim * kELT_SIZE),
                      static_cast<uint32_t>(params.numTokens * params.tokenDim * kELT_SIZE), 0, 0});
}

template <typename T>
cudaError_t twoshotAllreduceFusionDispatch(AllReduceFusionParams const& params) {
  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const numEltsPerThread = sizeof(float4) / sizeof(T);
  FLASHINFER_CHECK(tokenDim % numEltsPerThread == 0,
                   "[MNNVL AllReduceTwoShot] token_dim must be divisible by %d", numEltsPerThread);

  if (!params.rmsNormFusion && params.quantType != QuantType::kNone) {
    FLASHINFER_ERROR("[MNNVL AllReduceTwoShot] Quantization requires RMSNorm fusion");
    return cudaErrorInvalidValue;
  }
  if (params.quantType != QuantType::kNone) {
    if (params.quantOut == nullptr) {
      FLASHINFER_ERROR(
          "[MNNVL AllReduceTwoShot] quantOut must be non-null when quantization is enabled");
      return cudaErrorInvalidValue;
    }
    if (params.quantType != QuantType::kDynamicFP8 && params.outputScale == nullptr) {
      FLASHINFER_ERROR(
          "[MNNVL AllReduceTwoShot] outputScale must be non-null for static quantization");
      return cudaErrorInvalidValue;
    }
  }
  if (params.quantType == QuantType::kDynamicFP8 && params.scalingFactorOut == nullptr) {
    FLASHINFER_ERROR("[MNNVL AllReduceTwoShot] scale_out is required for dynamic FP8");
    return cudaErrorInvalidValue;
  }
  if (params.quantType == QuantType::kFP4) {
#if CUDA_VERSION < 12080
    FLASHINFER_ERROR("[MNNVL AllReduceTwoShot] FP4 quantization requires CUDA 12.8+");
    return cudaErrorInvalidValue;
#else
    if (params.scalingFactorOut == nullptr) {
      FLASHINFER_ERROR(
          "[MNNVL AllReduceTwoShot] scalingFactorOut is required for FP4 quantization");
      return cudaErrorInvalidValue;
    }
    if (tokenDim % trtllm_allreduce_fusion::details::CVT_FP4_SF_VEC_SIZE != 0) {
      FLASHINFER_ERROR("[MNNVL AllReduceTwoShot] FP4 quantization requires tokenDim divisible by " +
                       std::to_string(trtllm_allreduce_fusion::details::CVT_FP4_SF_VEC_SIZE));
      return cudaErrorInvalidValue;
    }
#endif
  }

  int const arNumThreads = ceil_div(tokenDim, numEltsPerThread);
  int const arNumBlocksPerToken = ceil_div(arNumThreads, 128);

  AllReduceKernelParams<T> kernelParams{
      .outputPtr = reinterpret_cast<T*>(params.output),
      .prenormedPtr = reinterpret_cast<T*>(params.residualOut),
      .shardPtr = reinterpret_cast<T const*>(params.input),
      .residualInPtr = reinterpret_cast<T const*>(params.residualIn),
      .gammaPtr = reinterpret_cast<T const*>(params.gamma),
      .inputPtrs = reinterpret_cast<T**>(params.bufferPtrsDev),
      .mcastPtr = reinterpret_cast<T*>(params.multicastPtr),
      .localBufferPtr = reinterpret_cast<T*>(params.bufferPtrLocal),
      .nRanks = params.nRanks,
      .rank = params.rank,
      .numTokens = params.numTokens,
      .tokenDim = params.tokenDim,
      .epsilon = static_cast<float>(params.epsilon),
      .weightBias = params.weightBias,
      .outputScalePtr = params.outputScale,
      .quantOutPtr = params.quantOut,
      .scalingFactorOutPtr = params.scalingFactorOut,
      .sfLayout = params.sfLayout,
      .bufferFlags = params.bufferFlags,
  };

  dim3 arGrid(numTokens, arNumBlocksPerToken);

  cudaLaunchAttribute arAttrs[1];
  arAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  arAttrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;

  cudaLaunchConfig_t arConfig{
      .gridDim = arGrid,
      .blockDim = 128,
      .dynamicSmemBytes = 0,
      .stream = params.stream,
      .attrs = arAttrs,
      .numAttrs = 1,
  };

  FLASHINFER_LOG_DEBUG("[MNNVL AllReduceTwoShot] Dispatch: grid size: (%d, %d, 1), block_size: 128",
                       numTokens, arNumBlocksPerToken);

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, WAIT_FOR_RESULTS) \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                    \
      &arConfig, &twoshotAllreduceKernel<WORLD_SIZE, T, WAIT_FOR_RESULTS>, kernelParams));
#define DISPATCH_ALLREDUCE_KERNEL(WAIT_FOR_RESULTS)                        \
  switch (params.nRanks) {                                                 \
    case 2:                                                                \
      LAUNCH_ALLREDUCE_KERNEL(2, WAIT_FOR_RESULTS);                        \
      break;                                                               \
    case 4:                                                                \
      LAUNCH_ALLREDUCE_KERNEL(4, WAIT_FOR_RESULTS);                        \
      break;                                                               \
    case 8:                                                                \
      LAUNCH_ALLREDUCE_KERNEL(8, WAIT_FOR_RESULTS);                        \
      break;                                                               \
    case 16:                                                               \
      LAUNCH_ALLREDUCE_KERNEL(16, WAIT_FOR_RESULTS);                       \
      break;                                                               \
    case 32:                                                               \
      LAUNCH_ALLREDUCE_KERNEL(32, WAIT_FOR_RESULTS);                       \
      break;                                                               \
    case 64:                                                               \
      LAUNCH_ALLREDUCE_KERNEL(64, WAIT_FOR_RESULTS);                       \
      break;                                                               \
    default:                                                               \
      FLASHINFER_ERROR("[MNNVL AllReduceTwoShot] Unsupported world_size" + \
                       std::to_string(params.nRanks) +                     \
                       ". Supported sizes: {2, 4, 8, 16, 32, 64}");        \
      return cudaErrorInvalidValue;                                        \
  }

  if (params.rmsNormFusion) {
    DISPATCH_ALLREDUCE_KERNEL(false);
  } else {
    DISPATCH_ALLREDUCE_KERNEL(true);
    return cudaSuccess;
  }
#undef DISPATCH_ALLREDUCE_KERNEL
#undef LAUNCH_ALLREDUCE_KERNEL

  auto gridConfig = adjustGridConfig(numTokens, tokenDim, numEltsPerThread, true);
  int rnBlockSize = std::get<0>(gridConfig);
  int rnClusterSize = std::get<1>(gridConfig);
  int rnLoadsPerThread = std::get<2>(gridConfig);

  bool rnUseCGA = rnClusterSize > 1 && rnLoadsPerThread == 1;
  if (rnClusterSize > 1 && !rnUseCGA) {
    gridConfig = adjustGridConfig(numTokens, tokenDim, numEltsPerThread, false);
    rnBlockSize = std::get<0>(gridConfig);
    rnClusterSize = std::get<1>(gridConfig);
    rnLoadsPerThread = std::get<2>(gridConfig);
    rnUseCGA = false;
  }

  if (rnBlockSize > 1024 || rnLoadsPerThread > 8) {
    FLASHINFER_ERROR("[MNNVL AllReduceTwoShotRMSNorm] Unsupported hidden dimension " +
                     std::to_string(tokenDim));
    return cudaErrorInvalidValue;
  }
  if (rnClusterSize > utils::kMaxClusterSize) {
    FLASHINFER_ERROR("[MNNVL AllReduceTwoShotRMSNorm] cluster_size " +
                     std::to_string(rnClusterSize) + " exceeds shared max buffer " +
                     std::to_string(utils::kMaxClusterSize));
    return cudaErrorInvalidValue;
  }

  int const rnNumThreads = rnClusterSize * rnBlockSize;
  int const dimPadded = round_up(tokenDim, numEltsPerThread * rnNumThreads);
  int const iters = dimPadded / rnNumThreads;
  size_t const smemSize = 3 * rnBlockSize * iters * sizeof(T);

  dim3 rnGrid(numTokens, rnClusterSize, 1);
  cudaLaunchConfig_t rnConfig;
  cudaLaunchAttribute rnAttrs[2];
  rnConfig.stream = params.stream;
  rnConfig.gridDim = rnGrid;
  rnConfig.blockDim = rnBlockSize;
  rnConfig.dynamicSmemBytes = smemSize;
  rnConfig.attrs = rnAttrs;
  rnAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  rnAttrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
  rnConfig.numAttrs = 1;
  if (rnUseCGA) {
    rnAttrs[1].id = cudaLaunchAttributeClusterDimension;
    rnAttrs[1].val.clusterDim.x = 1;
    rnAttrs[1].val.clusterDim.y = rnClusterSize;
    rnAttrs[1].val.clusterDim.z = 1;
    rnConfig.numAttrs = 2;
  }

  FLASHINFER_LOG_DEBUG(
      "[MNNVL AllReduceTwoShotRMSNorm] Dispatch: grid size: (%d, %d, 1), block_size: %d, "
      "cluster_size: %d, loads_per_thread: %d, threads_needed: %d",
      numTokens, rnClusterSize, rnBlockSize, rnClusterSize, rnLoadsPerThread,
      ceil_div(tokenDim, numEltsPerThread));

#define LAUNCH_RMSNORM_KERNEL(USE_CGA, LOADS_PER_THREAD, QTYPE)                                   \
  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(&rmsNormLamport<T, QTYPE, USE_CGA, LOADS_PER_THREAD>, \
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                                            smemSize));                                           \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                                        \
      &rnConfig, &rmsNormLamport<T, QTYPE, USE_CGA, LOADS_PER_THREAD>, kernelParams));

#define DISPATCH_QUANT(USE_CGA, LOADS_PER_THREAD)                                  \
  switch (params.quantType) {                                                      \
    case QuantType::kFP8:                                                          \
      LAUNCH_RMSNORM_KERNEL(USE_CGA, LOADS_PER_THREAD, QuantType::kFP8);           \
      break;                                                                       \
    case QuantType::kFP4:                                                          \
      if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) { \
        LAUNCH_RMSNORM_KERNEL(USE_CGA, LOADS_PER_THREAD, QuantType::kFP4);         \
      } else {                                                                     \
        FLASHINFER_ERROR(                                                          \
            "[MNNVL AllReduceTwoShotRMSNorm] FP4 quantization is only "            \
            "supported for FP16/BF16");                                            \
        return cudaErrorInvalidValue;                                              \
      }                                                                            \
      break;                                                                       \
    case QuantType::kDynamicFP8:                                                   \
      LAUNCH_RMSNORM_KERNEL(USE_CGA, LOADS_PER_THREAD, QuantType::kDynamicFP8);    \
      break;                                                                       \
    case QuantType::kNone:                                                         \
      LAUNCH_RMSNORM_KERNEL(USE_CGA, LOADS_PER_THREAD, QuantType::kNone);          \
      break;                                                                       \
    default:                                                                       \
      FLASHINFER_ERROR("[MNNVL AllReduceTwoShotRMSNorm] Unsupported quant type " + \
                       std::to_string(static_cast<int>(params.quantType)));        \
      return cudaErrorInvalidValue;                                                \
  }

#define DISPATCH_LOADS(USE_CGA)                                                          \
  switch (rnLoadsPerThread) {                                                            \
    case 1:                                                                              \
      DISPATCH_QUANT(USE_CGA, 1);                                                        \
      break;                                                                             \
    case 2:                                                                              \
      DISPATCH_QUANT(false, 2);                                                          \
      break;                                                                             \
    case 3:                                                                              \
      DISPATCH_QUANT(false, 3);                                                          \
      break;                                                                             \
    case 4:                                                                              \
      DISPATCH_QUANT(false, 4);                                                          \
      break;                                                                             \
    case 5:                                                                              \
      DISPATCH_QUANT(false, 5);                                                          \
      break;                                                                             \
    case 6:                                                                              \
      DISPATCH_QUANT(false, 6);                                                          \
      break;                                                                             \
    case 7:                                                                              \
      DISPATCH_QUANT(false, 7);                                                          \
      break;                                                                             \
    case 8:                                                                              \
      DISPATCH_QUANT(false, 8);                                                          \
      break;                                                                             \
    default:                                                                             \
      FLASHINFER_ERROR("[MNNVL AllReduceTwoShotRMSNorm] Unsupported loads_per_thread " + \
                       std::to_string(rnLoadsPerThread) +                                \
                       ". Supported sizes: {1, 2, 3, 4, 5, 6, 7, 8}");                   \
      return cudaErrorInvalidValue;                                                      \
  }

  if (rnUseCGA) {
    DISPATCH_LOADS(true);
  } else {
    DISPATCH_LOADS(false);
  }

#undef DISPATCH_LOADS
#undef DISPATCH_QUANT
#undef LAUNCH_RMSNORM_KERNEL
  return cudaSuccess;
}
}  // namespace trtllm_mnnvl_allreduce
}  // namespace flashinfer
