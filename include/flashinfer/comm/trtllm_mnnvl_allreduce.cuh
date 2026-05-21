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

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

#include "../exception.h"
#include "../logging.h"
#include "../utils.cuh"
namespace flashinfer {
namespace trtllm_mnnvl_allreduce {

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

  void* residualOut;
  void* output;
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
  uint32_t* bufferFlags;
};

namespace utils {

constexpr uint16_t kNEGZERO_FP16 = 0x8000U;
constexpr uint32_t kNEGZERO_FP32 = 0x80000000U;

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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (UseCGA) {
      cg::cluster_group cluster = cg::this_cluster();
      __cluster_barrier_arrive();
      if (cluster.block_rank() == 0 && threadIdx.x < 32) {
        __cluster_barrier_wait();
        arriveCounter(threadIdx.x);
      }
      return;
    }
#endif
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cg::grid_group grid = cg::this_grid();
    // Use the first thread instead of the last thread as the last thread may exit early
    isLastCtaT0 = grid.thread_rank() == 0;
    if constexpr (UseCGA) {
      cg::cluster_group cluster = cg::this_cluster();
      targetCount = gridDim.x * gridDim.y * gridDim.z / cluster.num_blocks();
    } else {
      targetCount = gridDim.x * gridDim.y * gridDim.z;
    }
#else
    isLastCtaT0 = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0;
    targetCount = gridDim.x * gridDim.y * gridDim.z;
#endif
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
  return value.words[0] == kNEGZERO_FP32;
}

template <typename T_IN>
inline __device__ void copyF4(T_IN* dst, T_IN const* src) {
  float4* dst4 = reinterpret_cast<float4*>(dst);
  float4 const* src4 = reinterpret_cast<float4 const*>(src);
  __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
inline __device__ void mbarrierInit(void* barrier, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(barrier))), "r"(count)
               : "memory");
}

inline __device__ void mbarrierArriveExpectTx(void* barrier, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(barrier))), "r"(bytes)
               : "memory");
}

inline __device__ void mbarrierWait(void* barrier, int phase) {
  uint32_t barrierPtr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni WAIT;\n\t"
      "DONE:\n\t"
      "}"
      :
      : "r"(barrierPtr), "r"(phase)
      : "memory");
}

inline __device__ void cpAsyncBulkGlobalToShared(void* dst, void const* src, void* barrier,
                                                 uint32_t bytes) {
  uint32_t smemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint64_t gmemPtr = static_cast<uint64_t>(__cvta_generic_to_global(const_cast<void*>(src)));
  uint32_t barrierPtr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      :
      : "r"(smemPtr), "l"(gmemPtr), "r"(bytes), "r"(barrierPtr)
      : "memory");
}
#endif

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
// A helper function to tune the grid configuration for fused oneshot and rmsnorm kernels
// Return (block_size, cluster_size, loads_per_thread)
std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread,
                                           int smVersionMajor) {
  // Start with preferred block_size and cluster_size
  int clusterSize = smVersionMajor >= 9 ? 8 : 1;
  int blockSize = 128;
  // ========================== Adjust the grid configuration ==========================
  int threadsNeeded = ceil_div(dim, eltsPerThread);
  int loadsPerThread = 1;

  blockSize = ceil_div(threadsNeeded, clusterSize);
  if (smVersionMajor >= 9) {
    while (threadsNeeded % clusterSize != 0 && clusterSize > 1) {
      clusterSize /= 2;
    }
    int const maxDivisibleClusterSize = clusterSize;
    blockSize = ceil_div(threadsNeeded, clusterSize);
    while (blockSize < 128 && clusterSize >= 2) {
      blockSize *= 2;
      clusterSize /= 2;
    }
    int smCount = GetCudaMultiProcessorCount();
    while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512) {
      blockSize *= 2;
      clusterSize /= 2;
    }
    // If the token grid still underfills the GPU, prefer more CTAs per token even when that
    // makes each CTA smaller than the usual 128-thread target.
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
  // Trying to scale up use multiple loads or CGA
  while (blockSize > 1024) {
    if (smVersionMajor >= 9) {
      if (clusterSize < 8) {
        clusterSize = clusterSize << 1;
      } else {
        if (loadsPerThread < 8) {
          loadsPerThread += 1;
        } else {
          break;
        }
      }
    } else {
      if (loadsPerThread < 8) {
        loadsPerThread += 1;
      } else {
        break;
      }
    }
    blockSize = ceil_div(threadsNeeded, clusterSize * loadsPerThread);
  }
  while (smVersionMajor >= 9 && blockSize > 1024 && loadsPerThread < 8) {
    loadsPerThread += 1;
    blockSize = ceil_div(threadsNeeded, clusterSize * loadsPerThread);
  }
  return {blockSize, clusterSize, loadsPerThread};
}
};  // namespace utils

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

template <uint8_t WorldSize, typename T, bool RMSNormFusion = false, typename PackedType = float4>
__global__ void __launch_bounds__(1024)
    oneshotAllreduceFusionKernel(AllReduceKernelParams<T> params) {
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kELT_SIZE = sizeof(T);
  int const tokenDim = params.tokenDim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int packedIdx = cluster.thread_rank();
  int token = blockIdx.x;
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

  cudaGridDependencySynchronize();
#else
  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  // Offset w.r.t. the input shard
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;
#endif
  // We only use 1 stage for the oneshot allreduce
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  constexpr bool kUseLamportCGA = true;
#else
  constexpr bool kUseLamportCGA = false;
#endif
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
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) val.elements[i] = fromFloat<T>(0.f);
  }

  reinterpret_cast<PackedType*>(
      &stagePtrMcast[token * tokenDim * WorldSize + params.rank * tokenDim])[packedIdx] =
      val.packed;

  flag.ctaArrive();
  // ======================= Lamport Sync and clear the output buffer from previous iteration
  // =============================
  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);

  PackedVec<PackedType, T> remoteValues[WorldSize - 1];
  while (1) {
    bool valid = true;
#pragma unroll
    for (int r = 0; r < WorldSize - 1; r++) {
      int const rankToLoad = (r + params.rank + 1) % WorldSize;
      auto loaded = loadPackedVolatile<PackedType>(
          &stagePtrLocal[token * tokenDim * WorldSize + rankToLoad * tokenDim +
                         packedIdx * kELTS_PER_THREAD]);
      remoteValues[r].packed = loaded.packed;
      // Keep the local value in registers and poll remote ranks in latcomm's cyclic order.
      valid &= !isLamportDirty(loaded);
    }
    if (valid) {
      break;
    }
  }

  // ======================= Reduction =============================
  float accum[kELTS_PER_THREAD];
  PackedVec<PackedType, T> packedAccum;

#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] = toFloat<T>(val.elements[i]);
#pragma unroll
    for (int r = 0; r < WorldSize - 1; r++) {
      accum[i] += toFloat<T>(remoteValues[r].elements[i]);
    }
  }

#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    packedAccum.elements[i] = fromFloat<T>(accum[i]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  if constexpr (RMSNormFusion) {
    // =============================== Residual ===============================
    PackedVec<PackedType, T> residualIn;
    residualIn.packed = *reinterpret_cast<PackedType const*>(&params.residualInPtr[threadOffset]);
    packedAccum += residualIn;
    *reinterpret_cast<PackedType*>(&params.prenormedPtr[threadOffset]) = packedAccum.packed;
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cg::cluster_group cluster = cg::this_cluster();
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
#endif
    float rcpRms = rsqrtf(fullSum / tokenDim + params.epsilon);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] = fromFloat<T>(toFloat<T>(packedAccum.elements[i]) * rcpRms *
                                             (params.weightBias + toFloat<T>(gamma.elements[i])));
    }
  }
  reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = packedAccum.packed;
  flag.waitAndUpdate(
      {static_cast<uint32_t>(params.numTokens * tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

using utils::adjustGridConfig;

template <typename T>
cudaError_t oneshotAllreduceFusionDispatch(AllReduceFusionParams const& params) {
  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const eltsPerThread = sizeof(float4) / sizeof(T);

  static const int kSMVersionMajor = GetCudaComputeCapability().first;

  auto [blockSize, clusterSize, loadsPerThread] =
      adjustGridConfig(numTokens, tokenDim, eltsPerThread, kSMVersionMajor);
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
                   tokenDim, 1024 * (kSMVersionMajor >= 9 ? 8 : 1) * eltsPerThread);

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
      .numAttrs = kSMVersionMajor >= 9 ? 2 : 1,
  };

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, RMSNORM) \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(           \
      &config, &oneshotAllreduceFusionKernel<WORLD_SIZE, T, RMSNORM>, kernelParams));
#define DISPATCH_ALLREDUCE_KERNEL(WORLD_SIZE)   \
  if (params.rmsNormFusion) {                   \
    LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, true);  \
  } else {                                      \
    LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, false); \
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
#undef LAUNCH_ALLREDUCE_KERNEL
  return cudaSuccess;
}

enum MNNVLTwoShotStage : uint8_t {
  SCATTER = 0,
  BROADCAST = 1,
  NUM_STAGES = 2,
};

using utils::copyF4;

template <uint8_t WorldSize, typename T, bool RMSNormFusion = false, bool UseCGA = false,
          int LoadsPerThread = 1, bool ExtraAR = false, typename PackedType = float4>
__global__ __launch_bounds__((RMSNormFusion ? 1024 : 128)) void twoshotAllreduceKernel(
    AllReduceKernelParams<T> params) {
  constexpr int kELTS_PER_LOAD = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kELT_SIZE = sizeof(T);
  constexpr bool kExtraAR = RMSNormFusion && ExtraAR;
  constexpr uint32_t kARLoadsPerThread = kExtraAR ? 1 : LoadsPerThread;

  uint32_t const token = blockIdx.x;
  uint32_t const blockSize = blockDim.x;
  uint32_t const threadOffset = threadIdx.x;
  bool const runRMSNormTail = !kExtraAR || blockIdx.y == 0;

  uint32_t numThreads = blockSize;
  uint32_t clusterSize = 1;
  uint32_t clusterBlockRank = 0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (RMSNormFusion && UseCGA) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    numThreads = cluster.num_threads();
    clusterSize = cluster.num_blocks();
    clusterBlockRank = cluster.block_rank();
  }
#endif

  uint32_t const dimPadded = round_up(params.tokenDim, kELTS_PER_LOAD * numThreads);
  uint32_t const elemsPerThread = dimPadded / numThreads;
  uint32_t const loadStride = blockSize;
  uint32_t const blockChunkSize =
      ceil_div(params.tokenDim, clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
  uint32_t const baseTokenOffset =
      RMSNormFusion ? clusterBlockRank * blockChunkSize : blockIdx.y * blockSize * kELTS_PER_LOAD;
  uint32_t const arBaseTokenOffset =
      kExtraAR ? blockIdx.y * blockSize * kELTS_PER_LOAD : baseTokenOffset;
  uint32_t const rmsBaseTokenOffset = kExtraAR ? 0 : baseTokenOffset;
  uint32_t const rmsBlockChunkSize = kExtraAR ? params.tokenDim : blockChunkSize;
  uint32_t const destRank = token % WorldSize;
  uint32_t const destTokenOffset = token / WorldSize;

  extern __shared__ uint8_t smem[];
  uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T);
  T* smemInput = reinterpret_cast<T*>(&smem[0]);
  T* smemResidual = reinterpret_cast<T*>(&smem[smemBufferSize]);
  T* smemGamma = reinterpret_cast<T*>(&smem[2 * smemBufferSize]);

  alignas(16) __shared__ uint64_t residualStageBarrier;
  alignas(16) __shared__ uint64_t gammaStageBarrier;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (RMSNormFusion) {
    if (runRMSNormTail && threadIdx.x == 0) {
      utils::mbarrierInit(&residualStageBarrier, 1);
      utils::mbarrierInit(&gammaStageBarrier, 1);
    }
    __syncthreads();
  }
  cudaGridDependencySynchronize();
#endif

  LamportFlags<PackedType, RMSNormFusion && UseCGA> flag(params.bufferFlags,
                                                         MNNVLTwoShotStage::NUM_STAGES);
  T* scatterBufLocal = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER));
  T* scatterBufDest = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
  T* broadcastBufW =
      reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, MNNVLTwoShotStage::BROADCAST));
  T* broadcastBufR = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST));

  PackedVec<PackedType, T> shardVals[kARLoadsPerThread];
#pragma unroll
  for (uint32_t i = 0; i < kARLoadsPerThread; i++) {
    uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    uint32_t const tokenOffset = arBaseTokenOffset + chunkOffset;
    if (tokenOffset < params.tokenDim) {
      shardVals[i].packed =
          loadPacked<PackedType>(&params.shardPtr[token * params.tokenDim + tokenOffset]);
#pragma unroll
      for (int j = 0; j < kELTS_PER_LOAD; j++) {
        if (isNegZero(shardVals[i].elements[j])) {
          shardVals[i].elements[j] = fromFloat<T>(0.F);
        }
      }
      reinterpret_cast<PackedType*>(
          &scatterBufDest[destTokenOffset * params.tokenDim * WorldSize +
                          params.rank * params.tokenDim])[tokenOffset / kELTS_PER_LOAD] =
          shardVals[i].packed;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  if constexpr (!kExtraAR) {
    flag.ctaArrive();
  }
  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER);

  if (destRank == params.rank) {
    uint32_t const localToken = token / WorldSize;
#pragma unroll
    for (uint32_t i = 0; i < kARLoadsPerThread; i++) {
      uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
      uint32_t const tokenOffset = arBaseTokenOffset + chunkOffset;
      if (tokenOffset < params.tokenDim) {
        PackedVec<PackedType, T> remoteValues[WorldSize - 1];
        while (1) {
          bool valid = true;
#pragma unroll
          for (int r = 0; r < WorldSize - 1; r++) {
            int const rankToLoad = (r + params.rank + 1) % WorldSize;
            auto loaded = loadPackedVolatile<PackedType>(
                &scatterBufLocal[localToken * params.tokenDim * WorldSize +
                                 rankToLoad * params.tokenDim + tokenOffset]);
            remoteValues[r].packed = loaded.packed;
            valid &= !isLamportDirty(loaded);
          }
          if (valid) {
            break;
          }
        }

        PackedVec<PackedType, T> packedAccum;
#pragma unroll
        for (int j = 0; j < kELTS_PER_LOAD; j++) {
          float accum = toFloat<T>(shardVals[i].elements[j]);
#pragma unroll
          for (int r = 0; r < WorldSize - 1; r++) {
            accum += toFloat<T>(remoteValues[r].elements[j]);
          }
          packedAccum.elements[j] = fromFloat<T>(accum);
        }
        // Reduced values can round to the dirty sentinel; sanitize before broadcast polling.
        sanitizeLamportPayload<PackedType, T>(packedAccum);
        reinterpret_cast<PackedType*>(&broadcastBufW[token * params.tokenDim + tokenOffset])[0] =
            packedAccum.packed;
      }
    }
  }

  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST);

  if (!runRMSNormTail) {
    flag.ctaArrive();
    return;
  }

  uint32_t stageElems = 0;
  uint32_t stageBytes = 0;
  if constexpr (RMSNormFusion) {
    stageElems = rmsBaseTokenOffset < params.tokenDim
                     ? (rmsBlockChunkSize < params.tokenDim - rmsBaseTokenOffset
                            ? rmsBlockChunkSize
                            : params.tokenDim - rmsBaseTokenOffset)
                     : 0;
    stageBytes = stageElems * sizeof(T);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (RMSNormFusion) {
    if (stageBytes > 0 && threadIdx.x == 0) {
      T const* residualSrc = &params.residualInPtr[token * params.tokenDim + rmsBaseTokenOffset];
      T const* gammaSrc = &params.gammaPtr[rmsBaseTokenOffset];
      // The mbarrier expect_tx must be issued before the TMA copy that completes it.
      utils::mbarrierArriveExpectTx(&residualStageBarrier, stageBytes);
      utils::cpAsyncBulkGlobalToShared(smemResidual, residualSrc, &residualStageBarrier,
                                       stageBytes);
      utils::mbarrierArriveExpectTx(&gammaStageBarrier, stageBytes);
      utils::cpAsyncBulkGlobalToShared(smemGamma, gammaSrc, &gammaStageBarrier, stageBytes);
    }
  }
#else
  if constexpr (RMSNormFusion) {
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
      uint32_t const tokenOffset = rmsBaseTokenOffset + chunkOffset;
      if (tokenOffset < params.tokenDim) {
        copyF4(&smemResidual[chunkOffset],
               &params.residualInPtr[token * params.tokenDim + tokenOffset]);
      }
    }
    __pipeline_commit();
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
      uint32_t const tokenOffset = rmsBaseTokenOffset + chunkOffset;
      if (tokenOffset < params.tokenDim) {
        copyF4(&smemGamma[chunkOffset], &params.gammaPtr[tokenOffset]);
      }
    }
    __pipeline_commit();
  }
#endif

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    uint32_t const tokenOffset = rmsBaseTokenOffset + chunkOffset;
    if (tokenOffset < params.tokenDim) {
      auto loaded =
          loadPackedVolatile<PackedType>(&broadcastBufR[token * params.tokenDim + tokenOffset]);
      while (isLamportDirty(loaded)) {
        loaded =
            loadPackedVolatile<PackedType>(&broadcastBufR[token * params.tokenDim + tokenOffset]);
      }
      if constexpr (RMSNormFusion) {
        reinterpret_cast<PackedType*>(&smemInput[chunkOffset])[0] = loaded.packed;
      } else {
        reinterpret_cast<PackedType*>(&params.outputPtr[token * params.tokenDim + tokenOffset])[0] =
            loaded.packed;
      }
    }
  }

  if constexpr (RMSNormFusion) {
    float rInput[LoadsPerThread * kELTS_PER_LOAD];
    float threadSum = 0.f;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (stageElems > 0) {
      utils::mbarrierWait(&residualStageBarrier, 0);
    }
#else
    __pipeline_wait_prior(1);
#endif

#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
      uint32_t const tokenOffset = rmsBaseTokenOffset + chunkOffset;
      if (tokenOffset < params.tokenDim) {
        PackedVec<PackedType, T> inp{.packed = loadPacked<PackedType>(&smemInput[chunkOffset])};
        PackedVec<PackedType, T> res{.packed = loadPacked<PackedType>(&smemResidual[chunkOffset])};
        PackedVec<PackedType, T> inpPlusRes = inp + res;
        reinterpret_cast<PackedType*>(
            &params.prenormedPtr[token * params.tokenDim + tokenOffset])[0] = inpPlusRes.packed;

#pragma unroll
        for (int j = 0; j < kELTS_PER_LOAD; j++) {
          float const value = toFloat<T>(inpPlusRes.elements[j]);
          rInput[i * kELTS_PER_LOAD + j] = value;
          threadSum += value * value;
        }
      }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (stageElems > 0) {
      utils::mbarrierWait(&gammaStageBarrier, 0);
    }
#else
    __pipeline_wait_prior(0);
#endif

    float blockSum = blockReduceSum<float, true>(threadSum);
    float fullSum = blockSum;
    __shared__ float sharedVal[8];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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
#endif

    float const rcpRms = rsqrtf(fullSum / params.tokenDim + params.epsilon);
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t const chunkOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
      uint32_t const tokenOffset = rmsBaseTokenOffset + chunkOffset;
      if (tokenOffset < params.tokenDim) {
        PackedVec<PackedType, T> gamma{.packed = loadPacked<PackedType>(&smemGamma[chunkOffset])};
        PackedVec<PackedType, T> out;
#pragma unroll
        for (int j = 0; j < kELTS_PER_LOAD; j++) {
          out.elements[j] = fromFloat<T>((params.weightBias + toFloat<T>(gamma.elements[j])) *
                                         rInput[i * kELTS_PER_LOAD + j] * rcpRms);
        }
        reinterpret_cast<PackedType*>(&params.outputPtr[token * params.tokenDim + tokenOffset])[0] =
            out.packed;
      }
    }
  }

  if constexpr (kExtraAR) {
    flag.ctaArrive();
  }
  flag.waitAndUpdate(
      {static_cast<uint32_t>(round_up(params.numTokens, WorldSize) * params.tokenDim * kELT_SIZE),
       static_cast<uint32_t>(params.numTokens * params.tokenDim * kELT_SIZE), 0, 0});
}

template <typename T>
cudaError_t twoshotAllreduceFusionDispatch(AllReduceFusionParams const& params) {
  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const numEltsPerThread = sizeof(float4) / sizeof(T);
  FLASHINFER_CHECK(tokenDim % numEltsPerThread == 0,
                   "[MNNVL AllReduceTwoShot] token_dim must be divisible by %d", numEltsPerThread);
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
      .bufferFlags = params.bufferFlags,
  };

  if (!params.rmsNormFusion) {
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

    FLASHINFER_LOG_DEBUG(
        "[MNNVL AllReduceTwoShot] Dispatch: grid size: (%d, %d, 1), block_size: 128", numTokens,
        arNumBlocksPerToken);

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE) \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(  \
      &arConfig, &twoshotAllreduceKernel<WORLD_SIZE, T, false, false, 1>, kernelParams));
    switch (params.nRanks) {
      case 2:
        LAUNCH_ALLREDUCE_KERNEL(2);
        break;
      case 4:
        LAUNCH_ALLREDUCE_KERNEL(4);
        break;
      case 8:
        LAUNCH_ALLREDUCE_KERNEL(8);
        break;
      case 16:
        LAUNCH_ALLREDUCE_KERNEL(16);
        break;
      case 32:
        LAUNCH_ALLREDUCE_KERNEL(32);
        break;
      case 64:
        LAUNCH_ALLREDUCE_KERNEL(64);
        break;
      default:
        FLASHINFER_ERROR("[MNNVL AllReduceTwoShot] Unsupported world_size" +
                         std::to_string(params.nRanks) +
                         ". Supported sizes: {2, 4, 8, 16, 32, 64}");
        return cudaErrorInvalidValue;
    }
#undef LAUNCH_ALLREDUCE_KERNEL
    return cudaSuccess;
  }

  static const int kSMVersionMajor = GetCudaComputeCapability().first;
  auto gridConfig = adjustGridConfig(numTokens, tokenDim, numEltsPerThread, kSMVersionMajor);
  int rnBlockSize = std::get<0>(gridConfig);
  int rnClusterSize = std::get<1>(gridConfig);
  int rnLoadsPerThread = std::get<2>(gridConfig);

  bool rnUseCGA = kSMVersionMajor >= 9 && rnClusterSize > 1 && rnLoadsPerThread == 1;
  if (kSMVersionMajor >= 9 && rnClusterSize > 1 && !rnUseCGA) {
    gridConfig = adjustGridConfig(numTokens, tokenDim, numEltsPerThread, 0);
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

  bool const useExtraAR = params.nRanks == 8 && !rnUseCGA && rnLoadsPerThread == 1 &&
                          arNumBlocksPerToken > rnClusterSize && arNumBlocksPerToken <= 8;
  int const launchBlockSize = useExtraAR ? 128 : rnBlockSize;
  int const launchGridY = useExtraAR ? arNumBlocksPerToken : rnClusterSize;
  int const launchLoadsPerThread = useExtraAR ? arNumBlocksPerToken : rnLoadsPerThread;
  int const rnNumThreads = rnClusterSize * rnBlockSize;
  int const launchNumThreads = useExtraAR ? launchBlockSize : rnNumThreads;
  int const dimPadded = round_up(tokenDim, numEltsPerThread * launchNumThreads);
  int const iters = dimPadded / launchNumThreads;
  size_t const smemSize = 3 * launchBlockSize * iters * sizeof(T);

  dim3 rnGrid(numTokens, launchGridY, 1);
  cudaLaunchConfig_t rnConfig;
  cudaLaunchAttribute rnAttrs[2];
  rnConfig.stream = params.stream;
  rnConfig.gridDim = rnGrid;
  rnConfig.blockDim = launchBlockSize;
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
      "cluster_size: %d, loads_per_thread: %d, threads_needed: %d, extra_ar: %d",
      numTokens, launchGridY, launchBlockSize, rnClusterSize, launchLoadsPerThread,
      ceil_div(tokenDim, numEltsPerThread), static_cast<int>(useExtraAR));

#define LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, USE_CGA, LOADS_PER_THREAD, EXTRA_AR)     \
  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                                             \
      &twoshotAllreduceKernel<WORLD_SIZE, T, true, USE_CGA, LOADS_PER_THREAD, EXTRA_AR>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize));                           \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                               \
      &rnConfig,                                                                         \
      &twoshotAllreduceKernel<WORLD_SIZE, T, true, USE_CGA, LOADS_PER_THREAD, EXTRA_AR>, \
      kernelParams));

#define DISPATCH_LOADS(WORLD_SIZE, USE_CGA, EXTRA_AR)                                    \
  switch (launchLoadsPerThread) {                                                        \
    case 1:                                                                              \
      LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, USE_CGA, 1, EXTRA_AR);                     \
      break;                                                                             \
    case 2:                                                                              \
      LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, false, 2, EXTRA_AR);                       \
      break;                                                                             \
    case 3:                                                                              \
      LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, false, 3, EXTRA_AR);                       \
      break;                                                                             \
    case 4:                                                                              \
      LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, false, 4, EXTRA_AR);                       \
      break;                                                                             \
    case 5:                                                                              \
      LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, false, 5, EXTRA_AR);                       \
      break;                                                                             \
    case 6:                                                                              \
      LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, false, 6, EXTRA_AR);                       \
      break;                                                                             \
    case 7:                                                                              \
      LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, false, 7, EXTRA_AR);                       \
      break;                                                                             \
    case 8:                                                                              \
      LAUNCH_TWOSHOT_FUSION_TYPED(WORLD_SIZE, false, 8, EXTRA_AR);                       \
      break;                                                                             \
    default:                                                                             \
      FLASHINFER_ERROR("[MNNVL AllReduceTwoShotRMSNorm] Unsupported loads_per_thread " + \
                       std::to_string(launchLoadsPerThread) +                            \
                       ". Supported sizes: {1, 2, 3, 4, 5, 6, 7, 8}");                   \
      return cudaErrorInvalidValue;                                                      \
  }

#define DISPATCH_WORLD_SIZE(USE_CGA, EXTRA_AR)                                    \
  switch (params.nRanks) {                                                        \
    case 2:                                                                       \
      DISPATCH_LOADS(2, USE_CGA, EXTRA_AR);                                       \
      break;                                                                      \
    case 4:                                                                       \
      DISPATCH_LOADS(4, USE_CGA, EXTRA_AR);                                       \
      break;                                                                      \
    case 8:                                                                       \
      DISPATCH_LOADS(8, USE_CGA, EXTRA_AR);                                       \
      break;                                                                      \
    case 16:                                                                      \
      DISPATCH_LOADS(16, USE_CGA, EXTRA_AR);                                      \
      break;                                                                      \
    case 32:                                                                      \
      DISPATCH_LOADS(32, USE_CGA, EXTRA_AR);                                      \
      break;                                                                      \
    case 64:                                                                      \
      DISPATCH_LOADS(64, USE_CGA, EXTRA_AR);                                      \
      break;                                                                      \
    default:                                                                      \
      FLASHINFER_ERROR("[MNNVL AllReduceTwoShotRMSNorm] Unsupported world_size" + \
                       std::to_string(params.nRanks) +                            \
                       ". Supported sizes: {2, 4, 8, 16, 32, 64}");               \
      return cudaErrorInvalidValue;                                               \
  }

  if (useExtraAR) {
    DISPATCH_WORLD_SIZE(false, true);
  } else if (rnUseCGA) {
    DISPATCH_WORLD_SIZE(true, false);
  } else {
    DISPATCH_WORLD_SIZE(false, false);
  }

#undef DISPATCH_WORLD_SIZE
#undef DISPATCH_LOADS
#undef LAUNCH_TWOSHOT_FUSION_TYPED
  return cudaSuccess;
}
}  // namespace trtllm_mnnvl_allreduce
}  // namespace flashinfer
