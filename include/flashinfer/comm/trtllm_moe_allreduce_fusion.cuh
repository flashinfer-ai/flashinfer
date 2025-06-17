#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/std/optional>
#include <tuple>
#include <type_traits>

#include "../exception.h"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace trtllm_moe_allreduce_fusion {

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

enum class FP4QuantizationSFLayout {
  // Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
  // blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8
  // values. The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and
  // columns to the nearest multiple of 4.
  //
  // The scale factor block rows map to data block rows in an interleaved pattern:
  // For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
  // Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
  //
  // Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
  SWIZZLED,
  // Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen
  // kernels standard.
  LINEAR
};

namespace utils {
#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

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
                                                       FP4QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    if (layout == FP4QuantizationSFLayout::SWIZZLED) {
      // SF vector index (16 elements share one SF in the K dimension).
      // numRows and numCols are unpadded.
      int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;

      auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numCols);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == FP4QuantizationSFLayout::LINEAR) {
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

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
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
  // static_assert(false, "not supported.");
  return 0;
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
  // static_assert(false, "not supported.");
  return 0;
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
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
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
}  // namespace utils

template <typename T>
struct AllReduceFusionParams {
  int nranks;
  int rank;
  // size = token_num * hidden_dim
  int size;
  int hidden_dim;
  void** workspace;
  void* allreduce_out;
  void* residual_in;
  void* residual_out;
  void* norm_out;
  void* quant_out;
  void* scale_out;
  void* rms_gamma;
  float rms_eps;
  // todo(review): why float* scale_factor in trt-llm?
  float scale_factor;
  FP4QuantizationSFLayout layout = FP4QuantizationSFLayout::SWIZZLED;
  cudaStream_t stream;
};

template <typename T>
struct MoeReductionAllReduceFusionParams : public AllReduceFusionParams<T> {
  // * moe reduction specific params
  // Refer to kernel implementation on layout of those params
  // number of active experts on current device

  // todo(review): why int* moe_reduction_device_num_experts = nullptr; in trt-llm?
  int moe_reduction_device_num_experts = 0;
  // per token per expert fp32 scale
  float* moe_reduction_scale_input = nullptr;
  // per token per expert input
  void* moe_reduction_active_experts_token_input = nullptr;
  // per token input
  void* moe_reduction_token_input = nullptr;
};

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
    flag_value = *flag_ptr;
    int comm_size = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] =
          reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> vec_add(const vec_t<T, VEC_SIZE>& a,
                                                      const vec_t<T, VEC_SIZE>& b) {
  vec_t<T, VEC_SIZE> ret;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    ret[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
  }
  return ret;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(vec_t<T, VEC_SIZE> const& residual,
                                                       vec_t<T, VEC_SIZE> const& gamma,
                                                       float const eps, int hidden_dim) {
  __shared__ float s_val;
  vec_t<T, VEC_SIZE> norm_out;
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float v = static_cast<float>(residual[i]);
    acc += v * v;
  }
  utils::blockReduceSumV2<float, 1>(&acc);
  if (cluster.num_blocks() > 1) {
    if (threadIdx.x == 0) {
      s_val = acc;
      acc = 0.f;
    }
    cluster.sync();
    if (threadIdx.x == 0) {
      for (int i = 0; i < cluster.num_blocks(); ++i) {
        acc += *cluster.map_shared_rank(&s_val, i);
      }
    }
    cluster.sync();
  }
  if (threadIdx.x == 0) {
    s_val = rsqrtf(acc / hidden_dim + eps);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    norm_out[i] = static_cast<float>(residual[i]) * s_val * static_cast<float>(gamma[i]);
  }
  return norm_out;
}

template <bool AllReduceOut, bool ResidualOut, bool NormOut, bool QuantOut, typename T,
          uint32_t VEC_SIZE>
__device__ __forceinline__ void fused_op(vec_t<T, VEC_SIZE> const& val, int access_id, int token_id,
                                         int access_id_in_token, AllReduceFusionParams<T>& params) {
  if constexpr (AllReduceOut) {
    val.store(reinterpret_cast<T*>(params.allreduce_out) + access_id * VEC_SIZE);
  }
  vec_t<T, VEC_SIZE> residual_val;
  residual_val.load(reinterpret_cast<T*>(params.residual_in) + access_id * VEC_SIZE);

  vec_t<T, VEC_SIZE> gamma_val;
  gamma_val.load(reinterpret_cast<T*>(params.rms_gamma) + access_id_in_token * VEC_SIZE);
  residual_val = vec_add<T, VEC_SIZE>(val, residual_val);
  if constexpr (ResidualOut) {
    residual_val.store(reinterpret_cast<T*>(params.residual_out) + access_id * VEC_SIZE);
  }
  vec_t<T, VEC_SIZE> norm_val;
  norm_val = rms_norm<T, VEC_SIZE>(residual_val, gamma_val, params.rms_eps, params.hidden_dim);
  if constexpr (NormOut) {
    norm_val.store(reinterpret_cast<T*>(params.norm_out) + access_id * VEC_SIZE);
  }
  if constexpr (QuantOut) {
    auto sf_out = utils::cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 2>(
        std::nullopt /* batchIdx */, token_id, access_id_in_token, std::nullopt /* numRows */,
        params.hidden_dim, reinterpret_cast<uint32_t*>(params.scale_out), params.layout);
    reinterpret_cast<uint32_t*>(params.quant_out)[access_id] =
        utils::cvt_warp_fp16_to_fp4<T, VEC_SIZE>(norm_val, params.scale_factor, sf_out);
  }
}

template <typename T>
struct neg_zero {
  static constexpr T value = -T(0);
};

template <>
struct neg_zero<half> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __half value = __half_raw{neg_zero_bits};
};

template <>
struct neg_zero<nv_bfloat16> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __nv_bfloat16 value = __nv_bfloat16_raw{neg_zero_bits};
};

template <>
struct neg_zero<float> {
  static constexpr unsigned int neg_zero_bits = 0x80000000U;
};

template <typename T>
__device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T>
__device__ bool is_negative_zero(T) {
  return false;
}

// float specialization
template <>
__device__ bool is_negative_zero<float>(float x) {
  return (__float_as_int(x) == 0x80000000);
}

// double specialization
template <>
__device__ bool is_negative_zero<double>(double x) {
  return (__double_as_longlong(x) == 0x8000000000000000ULL);
}

// __half specialization
template <>
__device__ bool is_negative_zero<__half>(__half x) {
  return (__half_as_ushort(x) == 0x8000);
}

// __nv_bfloat16 specialization
template <>
__device__ bool is_negative_zero<__nv_bfloat16>(__nv_bfloat16 x) {
  return (__bfloat16_as_ushort(x) == 0x8000);
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    if (is_negative_zero(vec[i])) {
      return true;
    }
  }
  return false;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    vec[i] = (is_negative_zero(vec[i])) ? static_cast<T>(0.f) : vec[i];
  }
}

template <typename T>
__device__ __forceinline__ void set_neg_zero(T* addr) {
  vec_t<T, 16 / sizeof(T)> val;
  val.fill(neg_zero_v<T>);
  val.store_global_volatile(addr);
}

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    auto status = cudaGetDevice(&device_id);
    FLASHINFER_CHECK(status == cudaSuccess, "cudaGetDevice failed with error code " +
                                                std::string(cudaGetErrorString(status)));
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    sm_count = device_prop.multiProcessorCount;
  }
  return sm_count;
}

bool use_oneshot(int token_num) { return token_num <= details::kOneShotMaxToken; }

/////////////////////////////////////////////////////////////////
//                  * MoE Reduction Fusion *                   //
/////////////////////////////////////////////////////////////////

template <typename T, int NRanks, bool AllReduceOut, bool ResidualOut, bool NormOut, bool QuantOut>
__global__ void moereduce_allreduce_fusion_kernel_oneshot_lamport(
    MoeReductionAllReduceFusionParams<T> params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  cg::grid_group grid = cg::this_grid();

  // Each token is handled by one cluster
  // which token is handled by current cluster
  int token_id = grid.cluster_rank();
  // total number of token
  int num_token = params.size / params.hidden_dim;
  // Each thread handle VEC_SIZE num elem in token. Total cluster.num_threads() to handle one
  // token For current token, which VEC_SIZE is handled by current thread (in unit of
  // VEC_SIZE)
  int access_id_in_token = cluster.thread_rank();
  // Across all token, which VEC_SIZE is handled by current thread (in unit of
  // VEC_SIZE)
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  int access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
  // Persistent kernel
  // stride to next token handled by current cta
  int token_stride = grid.num_clusters();
  // stride in unit of VEC_SIZE
  int access_stride = token_stride * params.hidden_dim / VEC_SIZE;
  // Total number of access in unit of VEC_SIZE to handle (token_num * hidden_dim)
  // This is within one rank
  int tot_access = params.size / VEC_SIZE;
  vec_t<T, VEC_SIZE> clear_vec;
  clear_vec.fill(neg_zero_v<T>);

  cudaGridDependencySynchronize();
  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / VEC_SIZE;

  // * MoE related
  int threadid_in_cluster = cluster.thread_rank();
  // Start Offset within one token's hidden_size of element
  // Current thread handle token[thread_offset_within_token : thread_offset_within_token +
  // VEC_SIZE]
  int thread_offset_within_token = threadid_in_cluster * VEC_SIZE;

  // Persistent Kernel
  // Each cluster iterate through all token it need to handle
  for (int token_id = grid.cluster_rank(); token_id < num_token; token_id += grid.num_clusters()) {
    if (thread_offset_within_token >= params.hidden_dim) {
      break;
    }

    // * MoE Reduce
    // Offset within (num_token, hidden_size) in unit of element
    int thread_offset_across_token = token_id * params.hidden_dim + thread_offset_within_token;

    vec_t<T, VEC_SIZE> accumulator;
    accumulator.fill(0.f);

    // * Iterate through all active expert
    int num_actexp = params.moe_reduction_device_num_experts;
    for (int actexp_i = 0; actexp_i < num_actexp; ++actexp_i) {
      // * Load active expert i's token j's partial data
      // Offset within (num_act_exp, num_token, hidden_size) in unit of element
      int thread_offset_across_actexp_token =
          actexp_i * (params.hidden_dim * num_token) + thread_offset_across_token;
      vec_t<T, VEC_SIZE> actexp_i_data;
      actexp_i_data.load(reinterpret_cast<T*>(params.moe_reduction_active_experts_token_input) +
                         thread_offset_across_actexp_token);

      // * Load active expert i's token j's scale
      int thread_offset_scale = actexp_i * num_token + token_id;
      float actexp_i_token_j_scale =
          reinterpret_cast<float const*>(params.moe_reduction_scale_input)[thread_offset_scale];

#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        // assume computation is done in ScaleType
        accumulator[i] +=
            static_cast<T>((static_cast<float>(actexp_i_data[i]) * actexp_i_token_j_scale));
      }
    }

    // * FC2 + reduced(gGEMM2)
    vec_t<T, VEC_SIZE> fc2_data;
    fc2_data.load(reinterpret_cast<T*>(params.moe_reduction_token_input) +
                  thread_offset_across_token);
    accumulator = vec_add<T, VEC_SIZE>(accumulator, fc2_data);

    // * AR Store
    int access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
    int idx = access_id;

    remove_neg_zero<T, VEC_SIZE>(accumulator);

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      // STG.128 to remote rank
      int offset = (params.rank * tot_access + idx) * VEC_SIZE;
      accumulator.store(reinterpret_cast<T*>(comm.data_bufs[r]) + offset);
    }
  }

  // * Clear previous buffer
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    int offset = idx * VEC_SIZE;
    clear_vec.store(reinterpret_cast<T*>(comm.clear_buf) + offset);
  }

  // * AR Load + Fusion
  for (int idx = access_id, tidx = token_id; idx < tot_access;
       idx += access_stride, tidx += token_stride) {
    // * AR Load
    vec_t<T, VEC_SIZE> vals[NRanks];
    bool done = false;
    while (!done) {
      // printf("Rank %d poll AR Load with flag %d\n", params.rank, *comm.flag_ptr);
      done = true;
#pragma unroll
      for (int r_i = 0; r_i < NRanks; ++r_i) {
        int r = (r_i + params.rank) % NRanks;
        // LDG.128 from local rank
        vals[r].load_global_volatile(reinterpret_cast<T*>(comm.data_bufs[params.rank]) +
                                     (r * tot_access + idx) * VEC_SIZE);
        done &= !has_neg_zero<T, VEC_SIZE>(vals[r]);
      }
    }

    vec_t<T, VEC_SIZE> sum_val = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
      sum_val = vec_add<T, VEC_SIZE>(sum_val, vals[r]);
    }

    // * Fuse
    fused_op<AllReduceOut, ResidualOut, NormOut, QuantOut, T, VEC_SIZE>(sum_val, idx, tidx,
                                                                        access_id_in_token, params);
  }
  comm.update(params.size * NRanks);
  cudaTriggerProgrammaticLaunchCompletion();
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, int NRanks, bool AllReduceOut, bool ResidualOut, bool NormOut, bool QuantOut>
cudaError_t launch_oneshot_moereduce_lamport(MoeReductionAllReduceFusionParams<T> const& params,
                                             cudaLaunchConfig_t& cfg) {
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &cfg,
      moereduce_allreduce_fusion_kernel_oneshot_lamport<T, NRanks, AllReduceOut, ResidualOut,
                                                        NormOut, QuantOut>,
      params));
  return cudaSuccess;
}

template <typename T, int NRanks, bool AllReduceOut, bool ResidualOut, bool NormOut, bool QuantOut>
cudaError_t moereduction_allreduce_fusion_kernel_launcher(
    MoeReductionAllReduceFusionParams<T> const& params, bool launch_with_pdl) {
  int token_num = params.size / params.hidden_dim;
  bool oneshot = use_oneshot(token_num);
  // todo(yingyi): support token_num > oneshot max token in another kernel
  if (oneshot == false) {
    FLASHINFER_LOG_WARN("expect one shot but got %d tokens, expect performance degradation",
                        token_num);
    oneshot = true;
  }
  // FLASHINFER_CHECK(oneshot, "only support one shot");
  // Each token is handled by one cluster
  int cluster_num = token_num;
  // Total number of threads (within one cluster) that's need to handle one token
  // given that each thread handle kElemsPerAccess
  int threads_per_token = params.hidden_dim * sizeof(T) / details::kBytesPerAccess;
  // Total number of warp (within one cluster) that's need to handle one token
  // given that each thread handle kElemsPerAccess
  int warps_per_token = (threads_per_token + 31) / 32;
  int cluster_size = 8;
  while (warps_per_token % cluster_size != 0) {
    cluster_size /= 2;
  }
  int block_size = warps_per_token / cluster_size * 32;
  FLASHINFER_CHECK(block_size <= 1024 && cluster_size > 0,
                   "block_size <= 1024 && cluster_size > 0");
  int sm_count = get_sm_count();
  int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;
  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute attribute[2];
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = 2;
  if (oneshot) {
    FLASHINFER_CUDA_CALL(
        (launch_oneshot_moereduce_lamport<T, NRanks, AllReduceOut, ResidualOut, NormOut, QuantOut>(
            params, cfg)));
  }
  return cudaSuccess;
}

#define DISPATCH_BOOL_(expr, const_expr, ...) \
  [&]() -> cudaError_t {                      \
    if (expr) {                               \
      constexpr bool const_expr = true;       \
      return __VA_ARGS__();                   \
    } else {                                  \
      constexpr bool const_expr = false;      \
      return __VA_ARGS__();                   \
    }                                         \
  }()

#define _DISPATCH_MOEREDUCTION_CASE(n_ranks_val, N_RANKS_VAR, ar, res, rms, quant, AR, RES, RMS, \
                                    QUANT, ...)                                                  \
  case n_ranks_val: {                                                                            \
    constexpr int N_RANKS_VAR = n_ranks_val;                                                     \
    return DISPATCH_BOOL_(ar, AR, [&]() -> cudaError_t {                                         \
      return DISPATCH_BOOL_(res, RES, [&]() -> cudaError_t {                                     \
        return DISPATCH_BOOL_(rms, RMS, [&]() -> cudaError_t {                                   \
          return DISPATCH_BOOL_(quant, QUANT, [&]() -> cudaError_t { return __VA_ARGS__(); });   \
        });                                                                                      \
      });                                                                                        \
    });                                                                                          \
  }

#define DISPATCH_MOEREDUCTION(n_ranks, ar, res, rms, quant, N_RANKS, AR, RES, RMS, QUANT, ...) \
  [&]() -> cudaError_t {                                                                       \
    switch (n_ranks) {                                                                         \
      _DISPATCH_MOEREDUCTION_CASE(2, N_RANKS, ar, res, rms, quant, AR, RES, RMS, QUANT,        \
                                  __VA_ARGS__)                                                 \
      _DISPATCH_MOEREDUCTION_CASE(4, N_RANKS, ar, res, rms, quant, AR, RES, RMS, QUANT,        \
                                  __VA_ARGS__)                                                 \
      _DISPATCH_MOEREDUCTION_CASE(8, N_RANKS, ar, res, rms, quant, AR, RES, RMS, QUANT,        \
                                  __VA_ARGS__)                                                 \
      _DISPATCH_MOEREDUCTION_CASE(16, N_RANKS, ar, res, rms, quant, AR, RES, RMS, QUANT,       \
                                  __VA_ARGS__)                                                 \
      default:                                                                                 \
        FLASHINFER_CHECK(false, "Unsupported n_ranks");                                        \
        return cudaErrorNotSupported;                                                          \
    }                                                                                          \
  }()

template <typename T>
cudaError_t moereduction_allreduce_fusion_op(MoeReductionAllReduceFusionParams<T> const& params,
                                             bool launch_with_pdl) {
  FLASHINFER_CHECK(params.residual_in && params.rms_gamma, "residual_in and rms_gamma must be set");
  FLASHINFER_CHECK(params.moe_reduction_scale_input &&
                       params.moe_reduction_active_experts_token_input &&
                       params.moe_reduction_token_input,
                   "moe_reduction_scale_input, moe_reduction_active_experts_token_input and "
                   "moe_reduction_token_input must be set");
  FLASHINFER_CHECK(params.size % params.hidden_dim == 0, "size must be a multiple of hidden_dim");
  FLASHINFER_CHECK(params.hidden_dim * sizeof(T) % details::kBytesPerAccess == 0,
                   "hidden_dim * sizeof(T) must be a multiple of kBytesPerAccess");
  FLASHINFER_CHECK(
      params.allreduce_out || params.residual_out || params.norm_out || params.quant_out,
      "at least one of allreduce_out, residual_out, norm_out, quant_out must be set");

  // hidden_dim (d) = 7168 for dpsk moe, and hence 128 tokens as one-shot threshold
  // AR outputs are optional, since we always have fused options followed.
  // pattern1: AR+Residual+Add_RMS+Quant
  // [m, d] bf16 allreduce_in, [m, d] bf16 residual_in
  // [m, d] bf16 residual_out, [m, d] fp4 quant_out

  // pattern2: AR+Add_RMS
  // [m, d] bf16 allreduce_in, [m, d] bf16 residual_in
  // [m, d] bf16 norm_out

  // pattern3: AR+Add_RMS
  // [m, d] bf16 allreduce_in, [m, d] bf16 residual_in
  // [m, d] bf16 norm_out

  // pattern4: AR+Add_RMS
  // [m, d] bf16 allreduce_in, [m, d] bf16 residual_in
  // [m, d] bf16 residual_out, [m, d] bf16 norm_out, [m, d] fp4 quant_out

  auto status = DISPATCH_MOEREDUCTION(
      params.nranks, params.allreduce_out, params.residual_out, params.rms_gamma, params.quant_out,
      N_RANKS, AR, RES, RMS, QUANT, [&]() -> cudaError_t {
        FLASHINFER_CUDA_CALL(
            (moereduction_allreduce_fusion_kernel_launcher<T, N_RANKS, AR, RES, RMS, QUANT>(
                (params), (launch_with_pdl))));
      });
  return status;
}
}  // namespace trtllm_moe_allreduce_fusion
}  // namespace flashinfer
