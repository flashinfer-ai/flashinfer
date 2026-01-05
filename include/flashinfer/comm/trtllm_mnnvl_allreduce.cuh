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
#include <vector_types.h>
#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <iostream>
#include <optional>
#include <type_traits>

#include "../exception.h"
#include "../fp4_layout.cuh"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {
namespace trtllm_mnnvl_allreduce {

// TODO: Same; This enum defination is duplicated
enum class QuantType : int {
  kNone = 0,
  kFP8 = 1,
  kFP4 = 2,
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
  bool rmsNormFusion = false;
  bool launchWithPdl = false;

  void const* input;
  void const* residualIn = nullptr;
  void const* gamma = nullptr;
  double epsilon = 1e-5;
  float* outputScale = nullptr;
  QuantizationSFLayout sfLayout = QuantizationSFLayout::SWIZZLED_128x4;
  QuantType quantType = QuantType::kNone;

  void* residualOut = nullptr;
  void* output = nullptr;
  void* quantOut = nullptr;
  void* scalingFactorOut = nullptr;
  cudaStream_t stream = nullptr;
};

namespace utils {

constexpr uint16_t kNEGZERO_FP16 = 0x8000U;

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

template <typename T>
static inline __device__ bool isNegZero(T val) {
  if constexpr (std::is_same_v<T, float>) {
    return val == 0.F && signbit(val);
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
template <typename PackedType = float4>
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
    int tid{0};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    cg::cluster_group cluster = cg::this_cluster();
    // We update the atomic counter per cluster
    tid = cluster.thread_rank();
    cluster.sync();
#else
    tid = threadIdx.x;
    __syncthreads();
#endif
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

  __device__ void waitAndUpdate(uint4 bytesToClearPerStage) {
    bool isLastCtaT0{false};
    int targetCount{0};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cg::grid_group grid = cg::this_grid();
    // Use the first thread instead of the last thread as the last thread may exit early
    isLastCtaT0 = grid.thread_rank() == 0;
    targetCount = grid.num_clusters();
#else
    isLastCtaT0 = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0;
    targetCount = gridDim.x * gridDim.y;
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
inline __device__ PackedType loadPacked(T* ptr) {
  return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr) {
  return *reinterpret_cast<PackedType const*>(ptr);
}

template <typename PackedType>
inline __device__ PackedType loadPackedVolatile(void const* ptr) {
  static_assert(sizeof(PackedType) == 0, "Not implemented");
  return PackedType{};
}

template <>
inline __device__ float4 loadPackedVolatile<float4>(void const* ptr) {
  float4 returnValue;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(returnValue.x), "=f"(returnValue.y), "=f"(returnValue.z), "=f"(returnValue.w)
               : "l"(ptr));
  return returnValue;
}

template <>
inline __device__ float2 loadPackedVolatile<float2>(void const* ptr) {
  float2 returnValue;
  asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];\n"
               : "=f"(returnValue.x), "=f"(returnValue.y)
               : "l"(ptr));
  return returnValue;
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
  }
  // Trying to scale up use multiple loads or CGA
  while (blockSize > 1024) {
    if (smVersionMajor >= 9) {
      if (clusterSize < 8) {
        clusterSize = clusterSize << 1;
      } else {
        break;
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
  return {blockSize, clusterSize, loadsPerThread};
}
};  // namespace utils

using utils::fromFloat;
using utils::PackedVec;
using utils::toFloat;

// TODO: These code are shared with trtllm_allreduce_fusion.cuh, and moe_allreduce_fusion; Should we
// move them to a shared header?
namespace quant {

namespace details {

static constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
static constexpr int CVT_FP4_SF_VEC_SIZE = 16;
static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;

}  // namespace details

namespace maths {
// // ============================== Cast ==============================
template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float2 cuda_cast<float2, int2>(int2 val) {
  return make_float2(val.x, val.y);
}

template <>
__device__ inline float2 cuda_cast<float2, float>(float val) {
  return make_float2(val, val);
}

template <>
__device__ inline float2 cuda_cast<float2, half2>(half2 val) {
  return __half22float2(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float2>(float2 val) {
  return __float22half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float>(float val) {
  return __float2half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, half>(half val) {
  return __half2half2(val);
}

template <>
__device__ inline int8_t cuda_cast<int8_t, half>(half val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  union {
    half fp16;
    int16_t int16_in;
  };

  fp16 = val;
  asm volatile("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
  return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, half2>(half2 val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}

template <>
__device__ inline int8_t cuda_cast<int8_t, float>(float val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
  return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, float2>(float2 val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}

template <>
__device__ inline half2 cuda_cast<half2, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  int16 = val;
  return make_half2(int8[0], int8[1]);
}

template <>
__device__ inline float2 cuda_cast<float2, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  int16 = val;
  return make_float2(int8[0], int8[1]);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast(int32_t val) {
  return static_cast<float>(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast(int8_t val) {
  return static_cast<float>(val);
}

template <>
__device__ inline int8_t cuda_cast(__nv_bfloat16 val) {
  return static_cast<float>(val);
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

template <>
__device__ inline float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val) {
  return bf1622float2(val);
}

template <>
__device__ inline half cuda_cast<half, __nv_bfloat16>(__nv_bfloat16 val) {
  return __float2half(__bfloat162float(val));
}

inline __device__ int16_t bf1622int16(__nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = max(min(__low2float(val), 127.f), -128.f);
  f_val.y = max(min(__high2float(val), 127.f), -128.f);

  union {
    int8_t int8[2];
    int16_t int16;
  };

  int8[0] = static_cast<int8_t>(static_cast<short>(f_val.x));
  int8[1] = static_cast<int8_t>(static_cast<short>(f_val.y));
  return int16;
#else
  val = __hmin2(val, make_bfloat162(127., 127.));
  val = __hmax2(val, make_bfloat162(-128., -128.));

  union {
    int8_t int8[2];
    int16_t int16;
  };

  int8[0] = static_cast<int8_t>(static_cast<short>(val.x));
  int8[1] = static_cast<int8_t>(static_cast<short>(val.y));
  return int16;
#endif
}

template <>
__device__ inline int16_t cuda_cast<int16_t, __nv_bfloat162>(__nv_bfloat162 val) {
  return bf1622int16(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, half>(half val) {
  return __float2bfloat16(__half2float(val));
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __nv_bfloat162 val2;
  val2.x = val;
  val2.y = val;
  return val2;
#else
  return __bfloat162bfloat162(val);
#endif
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_bfloat16>(__nv_bfloat16 val) {
  return bf162bf162(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float>(float val) {
  return __float2bfloat162_rn(val);
}

inline __device__ __nv_bfloat162 float22bf162(const float2 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __floats2bfloat162_rn(val.x, val.y);
#else
  return __float22bfloat162_rn(val);
#endif
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float2>(float2 val) {
  return float22bf162(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  int16 = val;
  __nv_bfloat162 res;
  res.x = cuda_cast<__nv_bfloat16>(int8[0]);
  res.y = cuda_cast<__nv_bfloat16>(int8[1]);
  return res;
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, half2>(half2 val) {
  return float22bf162(__half22float2(val));
}

// // ============================== Abs ==============================
template <typename T>
__device__ inline T cuda_abs(T val) {
  assert(false);
  return {};
}

template <>
__device__ inline float cuda_abs(float val) {
  return fabs(val);
}

template <>
__device__ inline float2 cuda_abs(float2 val) {
  return make_float2(fabs(val.x), fabs(val.y));
}

template <>
__device__ inline half cuda_abs(half val) {
  return __habs(val);
}

template <>
__device__ inline half2 cuda_abs(half2 val) {
  return __habs2(val);
}

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val) {
  return __habs(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val) {
  return __habs2(val);
}
#endif

// // ============================== Max ==============================
template <typename To, typename Ti>
__device__ inline To cuda_max(Ti val) {
  return cuda_cast<To>(val);
};

template <>
__device__ inline float cuda_max(float2 val) {
  return fmaxf(val.x, val.y);
}

template <>
__device__ inline half cuda_max(half2 val) {
  return __hmax(val.x, val.y);
}

template <>
__device__ inline __nv_bfloat16 cuda_max(__nv_bfloat162 val) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  return __hmax(val.x, val.y);
#else
  assert(0);
  asm volatile("brkpt;\n" ::);
  return __nv_bfloat16(0);
#endif
}

// Binary maximum: compute the max of two values.
template <typename T>
__device__ inline T cuda_max(T val1, T val2) {
  return (val1 > val2) ? val1 : val2;
}

template <>
__device__ inline float2 cuda_max(float2 val1, float2 val2) {
  float2 out;
  out.x = fmaxf(val1.x, val2.x);
  out.y = fmaxf(val1.y, val2.y);
  return out;
}

template <>
__device__ inline half2 cuda_max(half2 val1, half2 val2) {
  return __hmax2(val1, val2);
}

template <>
__device__ inline __nv_bfloat162 cuda_max(__nv_bfloat162 val1, __nv_bfloat162 val2) {
  return __hmax2(val1, val2);
}

// // ============================== Reciprocal ==============================
// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}
}  // namespace maths

inline __device__ int64_t get_sf_out_offset_128x4(std::optional<int> batchIdx, int mIdx, int kIdx,
                                                  std::optional<int> numRows, int numCols) {
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

  // batched tensor
  // SF layout [numBTiles, numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [bTileIdx, mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4 * innerKStride;  // 4

  // M tile layout [32, 4] is column-major.
  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * innerMStride;  // 16

  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = 32 * outerMStride;  // 512

  // SF vector size 16. We round the "numCols" up to a multiple of 64.
  int factor = details::CVT_FP4_SF_VEC_SIZE * 4;
  int32_t numKTiles = (numCols + factor - 1) / factor;
  int32_t mTileIdx = mIdx / (32 * 4);
  int64_t mTileStride = numKTiles * kTileStride;

  // Each SF block has 128 rows so pad rows to the multiple of 128.
  int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
  int64_t bTileStride = numMTiles * mTileStride;

  // Compute the global offset.
  int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                     kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride +
                     innerKIdx * innerKStride;

  return SFOffset;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(std::optional<int> batchIdx, int rowIdx,
                                                       int colIdx, std::optional<int> numRows,
                                                       int numCols, SFType* SFout,
                                                       QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    if (layout == QuantizationSFLayout::SWIZZLED_128x4) {
      // SF vector index (16 elements share one SF in the K dimension).
      // numRows and numCols are unpadded.
      int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;

      auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numCols);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == QuantizationSFLayout::LINEAR) {
      // Linear row-major layout, no padding required.
      int32_t KTileIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;

      int32_t numKTiles = numCols / details::CVT_FP4_SF_VEC_SIZE;
      int64_t mTileStride = numKTiles;

      int64_t BTileStride = numRows.value_or(0) * mTileStride;

      int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else {
      return nullptr;
    }
  }
#endif
  return nullptr;
}

__forceinline__ __device__ uint32_t pack_bytes(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3) {
  uint32_t val0 = c0;
  uint32_t val1 = c1;
  uint32_t val2 = c2;
  uint32_t val3 = c3;

  return (val3 << 24) | (val2 << 16) | (val1 << 8) | val0;
}

#if CUDA_VERSION >= 12080
// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
// NOTE: bypass sm_100 requirement by __nv_cvt_float2_to_fp4x2
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]), "f"(array[4]), "f"(array[5]),
        "f"(array[6]), "f"(array[7]));
  return val;
#else
  uint32_t val;
  __nv_fp4x2_storage_t vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    vals[i] = __nv_cvt_float2_to_fp4x2(*(((float2*)array) + i), __NV_E2M1, cudaRoundNearest);
  }
  val = pack_bytes(vals[0], vals[1], vals[2], vals[3]);
  return val;
#endif
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x),
        "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  uint32_t val;
  __nv_fp4x2_storage_t vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    vals[i] = __nv_cvt_float2_to_fp4x2(array[i], __NV_E2M1, cudaRoundNearest);
  }
  val = pack_bytes(vals[0], vals[1], vals[2], vals[3]);
  return val;
#endif
}

// Quantizes the provided PackedVec into the uint32_t output
template <typename T, uint32_t VEC_SIZE, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(vec_t<T, VEC_SIZE>& vec, float SFScaleVal,
                                         uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = maths::cuda_abs(get_vec2_element(vec, 0));

#pragma unroll
  for (int i = 1; i < details::CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = maths::cuda_max(localMax, maths::cuda_abs(get_vec2_element(vec, i)));
  }

  // Get the absolute maximum among all 16 values (two threads).
  localMax = maths::cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // Get the final absolute maximum values.
  float vecMax = float(maths::cuda_max(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * maths::reciprocal_approximate_ftz(6.0f));
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 12080)
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
#else
#error "FP8 E8M0 support requires CUDA 12.8 or newer."
#endif
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
  float outputScale = SFValue != 0 ? maths::reciprocal_approximate_ftz(
                                         SFValue * maths::reciprocal_approximate_ftz(SFScaleVal))
                                   : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[details::CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < details::CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<T, half>) {
      fp2Vals[i] = __half22float2(get_vec2_element(vec, i));
    } else {
      fp2Vals[i] = __bfloat1622float2(get_vec2_element(vec, i));
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

#endif

// ============================== Quant Device Function ==============================
template <typename T, typename PackedType, int ELTS_PER_THREAD>
inline __device__ void quant_fp8(PackedVec<PackedType, T> packed_accum, void* quant_out_ptr,
                                 float* output_scale, uint32_t thread_offset) {
  static_assert(ELTS_PER_THREAD == 8 || ELTS_PER_THREAD == 4, "ELTS_PER_THREAD must be 8 or 4");
  using QuantizedPackedType = std::conditional_t<ELTS_PER_THREAD == 8, float2, float>;

  auto quant_out = reinterpret_cast<__nv_fp8_e4m3*>(quant_out_ptr);
  PackedVec<QuantizedPackedType, __nv_fp8_e4m3> quantized_accum;
#pragma unroll
  for (int i = 0; i < ELTS_PER_THREAD; i++) {
    quantized_accum.elements[i] =
        __nv_fp8_e4m3(toFloat<T>(packed_accum.elements[i]) * (*output_scale));
  }
  reinterpret_cast<QuantizedPackedType*>(&quant_out[thread_offset])[0] = quantized_accum.packed;
}

template <typename T, typename PackedType, int ELTS_PER_THREAD>
inline __device__ void quant_nvfp4(PackedVec<PackedType, T> packed_accum, void* quant_out_ptr,
                                   void* sf_out_ptr, float* output_scale, uint32_t token_idx,
                                   uint32_t token_dim, uint32_t packed_idx,
                                   QuantizationSFLayout sf_layout) {
  static_assert(
      ELTS_PER_THREAD == 8 && (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>),
      "NVFP4 quantization fusion is only supported for FP16/BF16!");

  // Cast the packed accumulator
  auto packed_accum_ = *reinterpret_cast<vec_t<T, ELTS_PER_THREAD>*>(&packed_accum);
  // SFType is only the pointer type; It does not affect the internal logic of offset calculation.
  // Get the target pointer to the SF output.
  auto sf_out =
      cvt_quant_to_fp4_get_sf_out_offset<uint32_t, details::CVT_FP4_SF_VEC_SIZE / ELTS_PER_THREAD>(
          std::nullopt, token_idx, packed_idx, std::nullopt,
          token_dim / details::CVT_FP4_SF_VEC_SIZE, reinterpret_cast<uint32_t*>(sf_out_ptr),
          sf_layout);

  // Calculate the offset in packed item granularity for the quant output
  uint32_t quant_out_offset = token_idx * token_dim / ELTS_PER_THREAD + packed_idx;
  // Each packedvec has 8 elements -> 1 float4 in input -> 1 uint32_t in output
  reinterpret_cast<uint32_t*>(quant_out_ptr)[quant_out_offset] =
      cvt_warp_fp16_to_fp4<T, ELTS_PER_THREAD, false>(packed_accum_, *output_scale, sf_out);
}
};  // namespace quant

using utils::blockReduceSum;
using utils::isNegZero;
using utils::LamportFlags;
using utils::loadPacked;
using utils::loadPackedVolatile;

template <uint8_t WorldSize, typename T, bool RMSNormFusion = false,
          QuantType QType = QuantType::kNone, typename PackedType = float4>
__global__ void __launch_bounds__(1024) oneshotAllreduceFusionKernel(
    /* output ptrs*/
    T* outputPtr, T* prenormedPtr, void* quantOutPtr, void* scalingFactorOutPtr,
    /* input ptrs*/
    T const* shardPtr, T const* residualInPtr, T const* gammaPtr,
    /* Comm buffer params */
    T** inputPtrs, T* mcastPtr, int const rank, uint32_t* bufferFlags,
    /* problem size parameters*/
    int const numTokens, int const tokenDim, float epsilon, float* outputScale,
    QuantizationSFLayout sfLayout) {
  static_assert(QType == QuantType::kNone || RMSNormFusion,
                "Quant-only pattern without RMSNorm is not supported!");
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr int kLAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
  constexpr uint32_t kELT_SIZE = sizeof(T);
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
  LamportFlags<PackedType> flag(bufferFlags, 1);
  T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, 0));
  T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], 0));

  if (packedIdx * kELTS_PER_THREAD >= tokenDim) {
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
    return;
  }

  // ==================== Broadcast tokens to each rank =============================
  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) val.elements[i] = fromFloat<T>(0.f);
  }

  reinterpret_cast<PackedType*>(
      &stagePtrMcast[token * tokenDim * WorldSize + rank * tokenDim])[packedIdx] = val.packed;

  flag.ctaArrive();
  // ======================= Lamport Sync and clear the output buffer from previous iteration
  // =============================
  flag.clearDirtyLamportBuf(inputPtrs[rank], -1);

  PackedVec<PackedType, float> valuesLamport[WorldSize];
  while (1) {
    bool valid = true;
#pragma unroll
    for (int r = 0; r < WorldSize; r++) {
      valuesLamport[r].packed = loadPackedVolatile<PackedType>(
          &stagePtrLocal[token * tokenDim * WorldSize + r * tokenDim +
                         packedIdx * kELTS_PER_THREAD]);

#pragma unroll
      for (int i = 0; i < kLAMPORT_ELTS_PER_PACKED; i++) {
        valid &= !isNegZero(valuesLamport[r].elements[i]);
      }
    }
    if (valid) {
      break;
    }
  }

  auto values = reinterpret_cast<PackedVec<PackedType, T>*>(valuesLamport);
  // ======================= Reduction =============================
  float accum[kELTS_PER_THREAD];
  PackedVec<PackedType, T> packedAccum;

#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] = toFloat<T>(values[0].elements[i]);
  }

#pragma unroll
  for (int r = 1; r < WorldSize; r++) {
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      accum[i] += toFloat<T>(values[r].elements[i]);
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
    residualIn.packed = *reinterpret_cast<PackedType const*>(&residualInPtr[threadOffset]);
    packedAccum += residualIn;
    *reinterpret_cast<PackedType*>(&prenormedPtr[threadOffset]) = packedAccum.packed;
    // =============================== Rmsnorm ================================
    PackedVec<PackedType, T> gamma;
    gamma.packed = *reinterpret_cast<PackedType const*>(&gammaPtr[packedIdx * kELTS_PER_THREAD]);

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
    namespace cg = cooperative_groups;
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
    float rcpRms = rsqrtf(fullSum / tokenDim + epsilon);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] = fromFloat<T>(toFloat<T>(packedAccum.elements[i]) * rcpRms *
                                             toFloat<T>(gamma.elements[i]));
    }
  }
  if (outputPtr != nullptr) {
    reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = packedAccum.packed;
  }

  if constexpr (QType == QuantType::kFP8) {
    quant::quant_fp8<T, PackedType, kELTS_PER_THREAD>(packedAccum, quantOutPtr, outputScale,
                                                      threadOffset);
  } else if constexpr (QType == QuantType::kFP4) {
    quant::quant_nvfp4<T, PackedType, kELTS_PER_THREAD>(packedAccum, quantOutPtr,
                                                        scalingFactorOutPtr, outputScale, token,
                                                        tokenDim, packedIdx, sfLayout);
  }
  flag.waitAndUpdate(
      {static_cast<uint32_t>(numTokens * tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
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

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, RMSNORM, QUANT_TYPE)                               \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                                     \
      &config, &oneshotAllreduceFusionKernel<WORLD_SIZE, T, RMSNORM, QUANT_TYPE>, output,      \
      residualOut, params.quantOut, params.scalingFactorOut, input, residualIn, gamma, ucPtrs, \
      mcPtr, params.rank, params.bufferFlags, numTokens, tokenDim,                             \
      static_cast<float>(params.epsilon), params.outputScale, params.sfLayout));
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
          FLASHINFER_ERROR("FP4 quantization is only supported for FP16/BF16!");     \
          return cudaErrorInvalidValue;                                              \
        }                                                                            \
        break;                                                                       \
      case QuantType::kNone:                                                         \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, true, QuantType::kNone);                 \
        break;                                                                       \
      default:                                                                       \
        FLASHINFER_ERROR("Unsupported quant type! Got " +                            \
                         std::to_string(static_cast<int>(params.quantType)));        \
        return cudaErrorInvalidValue;                                                \
    }                                                                                \
  } else {                                                                           \
    LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, false, QuantType::kNone);                    \
  }

  T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
  T* mcPtr = reinterpret_cast<T*>(params.multicastPtr);
  T* output = reinterpret_cast<T*>(params.output);
  T* residualOut = reinterpret_cast<T*>(params.residualOut);
  T const* input = reinterpret_cast<T const*>(params.input);
  T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
  T const* gamma = reinterpret_cast<T const*>(params.gamma);

  switch (params.nRanks) {
      // FIXME: Do we need other world sizes?
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

template <uint8_t WorldSize, typename T, typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshotAllreduceKernel(
    T* outputPtr, T const* shardPtr, T** inputPtrs, T* mcastPtr, uint32_t const numTokens,
    uint32_t const tokenDim, uint32_t const rank, uint32_t* bufferFlags,
    bool const wait_for_results) {
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr int kLAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
  constexpr uint32_t kELT_SIZE = sizeof(T);

  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  // Offset w.r.t. the input shard
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

  int destRank = token % WorldSize;
  int destTokenOffset = token / WorldSize;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  LamportFlags<PackedType> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);

  T* scatterBufLocal =
      reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::SCATTER));
  T* scatterBufDest =
      reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
  T* broadcastBufW =
      reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, MNNVLTwoShotStage::BROADCAST));
  T* broadcastBufR =
      reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::BROADCAST));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  // Make sure the clear function is called before OOB thread exits
  if (packedIdx * kELTS_PER_THREAD >= tokenDim) {
    flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
    return;
  }

  // =============================== Scatter ===============================

  // Load vectorized data
  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) {
      val.elements[i] = fromFloat<T>(0.F);
    }
  }

  // Store vectorized data
  reinterpret_cast<PackedType*>(
      &scatterBufDest[destTokenOffset * tokenDim * WorldSize + rank * tokenDim])[packedIdx] =
      val.packed;

  flag.clearDirtyLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::SCATTER);

  // =============================== Reduction and Broadcast ===============================

  if ((token % WorldSize) == rank) {
    int localToken = token / WorldSize;
    float accum[kELTS_PER_THREAD] = {0.F};

    // Use float as we only check each float value for validity
    PackedVec<PackedType, float> valuesLamport[WorldSize];
    while (1) {
      bool valid = true;
#pragma unroll
      for (int r = 0; r < WorldSize; r++) {
        valuesLamport[r].packed = loadPackedVolatile<PackedType>(
            &scatterBufLocal[localToken * tokenDim * WorldSize + r * tokenDim +
                             packedIdx * kELTS_PER_THREAD]);

        // Check validity across all elements
#pragma unroll
        for (int i = 0; i < kLAMPORT_ELTS_PER_PACKED; i++) {
          valid &= !isNegZero(valuesLamport[r].elements[i]);
        }
      }
      if (valid) {
        break;
      }
    }

    // Now we view it as the value for reduction
    auto values = reinterpret_cast<PackedVec<PackedType, T>*>(valuesLamport);
#pragma unroll
    for (int r = 0; r < WorldSize; r++) {
#pragma unroll
      for (int i = 0; i < kELTS_PER_THREAD; i++) {
        accum[i] += toFloat<T>(values[r].elements[i]);
      }
    }

    // Store vectorized result
    PackedVec<PackedType, T> packedAccum;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] = fromFloat<T>(accum[i]);
    }
    reinterpret_cast<PackedType*>(&broadcastBufW[token * tokenDim])[packedIdx] = packedAccum.packed;
  }

  flag.clearDirtyLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::BROADCAST);

  // Optionally wait for results if the next layer isn't doing the Lamport check
  if (wait_for_results) {
    // Update the atomic counter to indicate the block has read the offsets
    flag.ctaArrive();

    PackedVec<PackedType, float> valLamport;
    valLamport.packed = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
    while (isNegZero(valLamport.elements[0])) {
      valLamport.packed = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
    }
    if (outputPtr) {
      reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = valLamport.packed;
    }

    // Update the buffer flags
    flag.waitAndUpdate(
        {static_cast<uint32_t>(round_up(numTokens, WorldSize) * tokenDim *
                               kELT_SIZE),                         // Clear Size for scatter stage
         static_cast<uint32_t>(numTokens * tokenDim * kELT_SIZE),  // Clear Size for broadcast stage
         0, 0});
    // If not wait for results, we will rely on the following kernel to update the buffer
  }
}

using utils::copyF4;
// This kernel works performant when loads_per_thread is 1.
// For this mode, we are able to support up to 1024 (threads) x 8 (elements) = 8192 hidden
// dimension. There are two options for further scaling up:
//      1. Use CGA if supported. It expands the hidden dimension to 8k x 8 = 64k.
//      2. Set loads_per_thread >1. Which can be used if CGA is not supported. Note that this will
//      be limited by the shared memory size and register count.
template <typename T, QuantType QType = QuantType::kNone, int LoadsPerThread = 1>
__global__ __launch_bounds__(1024) void rmsNormLamport_fusion(
    /* Output ptrs */
    T* outputPreNorm, T* outputNorm, void* quantOut, void* scalingFactorOut,
    /* Input ptrs */
    T* bufferInput, T const* gamma, T const* residual,
    /* Comm buffer params */
    uint32_t worldSize, uint32_t* bufferFlags,
    /* Problem parameters */
    uint32_t numTokens, uint32_t dim, float epsilon, float* outputScale,
    QuantizationSFLayout sfLayout) {
  static int const kELTS_PER_LOAD = sizeof(float4) / sizeof(T);

  uint32_t const token = blockIdx.x;
  uint32_t const blockSize = blockDim.x;
  uint32_t const threadOffset = threadIdx.x;

  uint32_t numThreads = blockSize;
  uint32_t clusterSize = 1;
  uint32_t blockOffset = 0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  numThreads = cluster.num_threads();
  clusterSize = cluster.num_blocks();
  blockOffset = cluster.block_rank();
#endif
  uint32_t const dimPadded = round_up(dim, kELTS_PER_LOAD * numThreads);
  uint32_t const elemsPerThread = dimPadded / numThreads;
  uint32_t const loadStride = blockSize;

  extern __shared__ uint8_t smem[];
  float rInput[LoadsPerThread * kELTS_PER_LOAD];
  uint32_t offsets[LoadsPerThread * kELTS_PER_LOAD];

  uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T);
  T* smemInput = (T*)&smem[0];
  T* smemResidual = (T*)&smem[smemBufferSize];
  T* smemGamma = (T*)&smem[2 * smemBufferSize];

  LamportFlags<float4> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
  T* input = reinterpret_cast<T*>(
      flag.getCurLamportBuf(reinterpret_cast<void*>(bufferInput), MNNVLTwoShotStage::BROADCAST));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  // The offset that current thread should load from. Note that the hidden dimension is split by CGA
  // size and each block loads a contiguous chunk; The size of chunk that each block processes
  uint32_t const blockChunkSize = ceil_div(dim, clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
  uint32_t const blockLoadOffset = token * dim + blockOffset * blockChunkSize;

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    // Each block load a contiguous chunk of tokens
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    offsets[i] = blockLoadOffset + threadLoadOffset;
  }

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      copyF4(&smemResidual[threadLoadOffset], &residual[blockLoadOffset + threadLoadOffset]);
    }
  }
  __pipeline_commit();
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      copyF4(&smemGamma[threadLoadOffset], &gamma[blockOffset * blockChunkSize + threadLoadOffset]);
    }
  }
  __pipeline_commit();

  flag.ctaArrive();
  bool valid = false;
  // ACQBLK if not lamport
  while (!valid) {
    valid = true;
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;

      if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
        float4* dst4 = reinterpret_cast<float4*>(&smemInput[threadLoadOffset]);
        float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);

        float4 value = loadPackedVolatile<float4>(src4);
        // Assume that the 16B were written atomically, so we only need to check one value
        valid &= !isNegZero(value.x);
        *dst4 = value;
      }
    }
  }

  __pipeline_wait_prior(1);
  __syncthreads();

  float threadSum = 0.f;
#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    int threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T> inp{.packed = loadPacked<float4>(&smemInput[threadLoadOffset])};
      PackedVec<float4, T> res{.packed = loadPacked<float4>(&smemResidual[threadLoadOffset])};

      PackedVec<float4, T> inp_plus_res = inp + res;
#pragma unroll
      for (int j = 0; j < kELTS_PER_LOAD; j++) {
        rInput[i * kELTS_PER_LOAD + j] = toFloat<T>(inp_plus_res.elements[j]);
        threadSum += toFloat<T>(inp_plus_res.elements[j] * inp_plus_res.elements[j]);
      }

      *reinterpret_cast<float4*>(&outputPreNorm[blockLoadOffset + threadLoadOffset]) =
          inp_plus_res.packed;
    }
  }

  __pipeline_wait_prior(0);

  float blockSum = blockReduceSum<float, true>(threadSum);

  float fullSum = blockSum;
  __shared__ float sharedVal[8];
  // Use CGA Reduction if supported
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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

  float rcpRms = rsqrtf(fullSum / dim + epsilon);

#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    PackedVec<float4, T> r_out;
    uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    uint32_t token_offset = blockOffset * blockChunkSize + threadLoadOffset;
    if (token_offset < dim) {
      PackedVec<float4, T> gamma = {.packed = loadPacked<float4>(&smemGamma[threadLoadOffset])};

#pragma unroll
      for (uint32_t j = 0; j < kELTS_PER_LOAD; j++) {
        r_out.elements[j] =
            fromFloat<T>(toFloat<T>(gamma.elements[j]) * rInput[i * kELTS_PER_LOAD + j] * rcpRms);
      }
      if (outputNorm != nullptr) {
        *reinterpret_cast<float4*>(&outputNorm[blockLoadOffset + threadLoadOffset]) = r_out.packed;
      }
      if constexpr (QType == QuantType::kFP8) {
        quant::quant_fp8<T, float4, kELTS_PER_LOAD>(r_out, quantOut, outputScale, threadOffset);
      } else if constexpr (QType == QuantType::kFP4) {
        quant::quant_nvfp4<T, float4, kELTS_PER_LOAD>(r_out, quantOut, scalingFactorOut,
                                                      outputScale, token, dim,
                                                      token_offset / kELTS_PER_LOAD, sfLayout);
      }
    }
  }
  constexpr int kELTS_SIZE = sizeof(T);

  // Assume the previous kernel does not modify the buffer_flags.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  // Update the buffer pointers
  flag.waitAndUpdate({static_cast<uint32_t>(round_up(numTokens, worldSize) * dim * kELTS_SIZE),
                      static_cast<uint32_t>(numTokens * dim * kELTS_SIZE), 0, 0});
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

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE)                                               \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                                \
      &arConfig, &twoshotAllreduceKernel<WORLD_SIZE, T>, output, input, ucPtrs, mcastPtr, \
      numTokens, tokenDim, params.rank, params.bufferFlags, (!params.rmsNormFusion)));
  T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
  T* mcastPtr = reinterpret_cast<T*>(params.multicastPtr);
  T* output = reinterpret_cast<T*>(params.output);
  T const* input = reinterpret_cast<T const*>(params.input);
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
                       std::to_string(params.nRanks) + ". Supported sizes: {2, 4, 8, 16, 32, 64}");
      return cudaErrorInvalidValue;
  }
#undef LAUNCH_ALLREDUCE_KERNEL

  // Launch the rmsnorm lamport kernel if fusion is enabled
  if (params.rmsNormFusion) {
    static const int kSMVersionMajor = GetCudaComputeCapability().first;
    auto gridConfig = adjustGridConfig(numTokens, tokenDim, numEltsPerThread, kSMVersionMajor);
    int rnBlockSize = std::get<0>(gridConfig);
    int rnClusterSize = std::get<1>(gridConfig);
    int rnLoadsPerThread = std::get<2>(gridConfig);

    int rnNumThreads = rnClusterSize * rnBlockSize;
    dim3 rnGrid(numTokens, rnClusterSize, 1);
    cudaLaunchConfig_t rnConfig;
    cudaLaunchAttribute rnAttrs[2];
    rnConfig.stream = params.stream;
    rnConfig.gridDim = rnGrid;
    rnConfig.blockDim = rnBlockSize;
    rnConfig.attrs = rnAttrs;
    rnAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    rnAttrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
    rnAttrs[1].id = cudaLaunchAttributeClusterDimension;
    rnAttrs[1].val.clusterDim.x = 1;
    rnAttrs[1].val.clusterDim.y = rnClusterSize;
    rnAttrs[1].val.clusterDim.z = 1;
    rnConfig.numAttrs = kSMVersionMajor >= 9 ? 2 : 1;

    bool const rnUseCGA = kSMVersionMajor >= 9 && rnClusterSize > 1;
    int const dimPadded = round_up(tokenDim, numEltsPerThread * rnNumThreads);
    int const iters = dimPadded / rnNumThreads;

    size_t const smemSize = 3 * rnBlockSize * iters * sizeof(T);

    FLASHINFER_LOG_DEBUG(
        "[MNNVL AllReduceTwoShotRMSNorm] Dispatch: grid size: (%d, %d, 1), block_size: %d, "
        "cluster_size: %d, "
        "loads_per_thread: %d, "
        "threads_needed: %d",
        numTokens, rnClusterSize, rnBlockSize, rnClusterSize, rnLoadsPerThread,
        ceil_div(tokenDim, numEltsPerThread));

#define RUN_RMSNORM_FUSION_KERNEL_(LOADS_PER_THREAD, QType)                                     \
  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(&rmsNormLamport_fusion<T, QType, LOADS_PER_THREAD>, \
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,        \
                                            smemSize));                                         \
  rnConfig.dynamicSmemBytes = smemSize;                                                         \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                                      \
      &rnConfig, &rmsNormLamport_fusion<T, QType, LOADS_PER_THREAD>, residualOut, output,       \
      params.quantOut, params.scalingFactorOut, bufferInput, gamma, residualIn, params.nRanks,  \
      params.bufferFlags, numTokens, tokenDim, static_cast<float>(params.epsilon),              \
      params.outputScale, params.sfLayout));

#define RUN_RMSNORM_FUSION_KERNEL(LOADS_PER_THREAD)                                \
  switch (params.quantType) {                                                      \
    case QuantType::kFP8:                                                          \
      RUN_RMSNORM_FUSION_KERNEL_(LOADS_PER_THREAD, QuantType::kFP8);               \
      break;                                                                       \
    case QuantType::kFP4:                                                          \
      if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) { \
        RUN_RMSNORM_FUSION_KERNEL_(LOADS_PER_THREAD, QuantType::kFP4);             \
      } else {                                                                     \
        FLASHINFER_ERROR("FP4 quantization is only supported for FP16/BF16!");     \
        return cudaErrorInvalidValue;                                              \
      }                                                                            \
      break;                                                                       \
    case QuantType::kNone:                                                         \
      RUN_RMSNORM_FUSION_KERNEL_(LOADS_PER_THREAD, QuantType::kNone);              \
      break;                                                                       \
    default:                                                                       \
      FLASHINFER_ERROR("[MNNVL AllReduceTwoShotRMSNorm] Unsupported quant type" +  \
                       std::to_string(static_cast<int>(params.quantType)) +        \
                       ". Supported types: {kFP8, kFP4, kNone}");                  \
      return cudaErrorInvalidValue;                                                \
  }

    T* residualOut = reinterpret_cast<T*>(params.residualOut);
    T* output = reinterpret_cast<T*>(params.output);
    T* bufferInput = reinterpret_cast<T*>(params.bufferPtrLocal);
    T const* gamma = reinterpret_cast<T const*>(params.gamma);
    T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
    if (rnUseCGA) {
      RUN_RMSNORM_FUSION_KERNEL(1);
    } else {
      switch (rnLoadsPerThread) {
        case 1:
          RUN_RMSNORM_FUSION_KERNEL(1);
          break;
        case 2:
          RUN_RMSNORM_FUSION_KERNEL(2);
          break;
        case 3:
          RUN_RMSNORM_FUSION_KERNEL(3);
          break;
        case 4:
          RUN_RMSNORM_FUSION_KERNEL(4);
          break;
        case 5:
          RUN_RMSNORM_FUSION_KERNEL(5);
          break;
        case 6:
          RUN_RMSNORM_FUSION_KERNEL(6);
          break;
        case 7:
          RUN_RMSNORM_FUSION_KERNEL(7);
          break;
        case 8:
          RUN_RMSNORM_FUSION_KERNEL(8);
          break;
        default:
          FLASHINFER_ERROR("[MNNVL AllReduceTwoShotRMSNorm] Unsupported loads_per_thread" +
                           std::to_string(rnLoadsPerThread) +
                           ". Supported sizes: {1, 2, 3, 4, 5, 6, 7, 8}");
          return cudaErrorInvalidValue;
      }  // switch (rnLoadsPerThread)
    }  // if (rnUseCGA)
#undef RUN_RMSNORM_FUSION_KERNEL
#undef RUN_RMSNORM_FUSION_KERNEL_

  }  // if (params.rmsNormFusion)
  return cudaSuccess;
}
}  // namespace trtllm_mnnvl_allreduce
}  // namespace flashinfer
