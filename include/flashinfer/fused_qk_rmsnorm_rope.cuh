/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_FUSED_QK_RMSNORM_ROPE_CUH_
#define FLASHINFER_FUSED_QK_RMSNORM_ROPE_CUH_

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace flashinfer {

#define FLASHINFER_FUSED_CHECK(condition)                                                 \
  do {                                                                                    \
    if (!(condition)) {                                                                   \
      fprintf(stderr, "FLASHINFER_FUSED_CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
              #condition);                                                                \
      abort();                                                                            \
    }                                                                                     \
  } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 1: IntFastDiv — fast signed integer division on GPU
// Based on Hacker's Delight, Second Edition, Chapter 10.
////////////////////////////////////////////////////////////////////////////////////////////////////

class IntFastDiv {
 public:
  __host__ IntFastDiv() : mDivisor(1), mMagicM(0), mMagicS(-1), mAddSign(1) {}

  __host__ IntFastDiv(int divisor) : mDivisor(divisor) {
    if (mDivisor == 0) throw std::runtime_error("IntFastDiv: cannot divide by 0");
    updateMagicNumbers();
  }

  __host__ IntFastDiv& operator=(int divisor) {
    this->mDivisor = divisor;
    if (this->mDivisor == 0) throw std::runtime_error("IntFastDiv: cannot divide by 0");
    updateMagicNumbers();
    return *this;
  }

  __host__ __device__ operator int() const { return mDivisor; }

 private:
  int mDivisor;
  int mMagicM;
  int mMagicS;
  int mAddSign;

  __host__ void updateMagicNumbers() {
    if (mDivisor == 1) {
      mMagicM = 0;
      mMagicS = -1;
      mAddSign = 1;
      return;
    } else if (mDivisor == -1) {
      mMagicM = 0;
      mMagicS = -1;
      mAddSign = -1;
      return;
    }

    int p;
    unsigned int tmpAd, tmpAnc, delta, q1, r1, q2, r2, t;
    unsigned const two31 = 0x80000000;
    tmpAd = abs(mDivisor);
    t = two31 + ((unsigned int)mDivisor >> 31);
    tmpAnc = t - 1 - t % tmpAd;
    p = 31;
    q1 = two31 / tmpAnc;
    r1 = two31 - q1 * tmpAnc;
    q2 = two31 / tmpAd;
    r2 = two31 - q2 * tmpAd;
    do {
      ++p;
      q1 = 2 * q1;
      r1 = 2 * r1;
      if (r1 >= tmpAnc) {
        ++q1;
        r1 -= tmpAnc;
      }
      q2 = 2 * q2;
      r2 = 2 * r2;
      if (r2 >= tmpAd) {
        ++q2;
        r2 -= tmpAd;
      }
      delta = tmpAd - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    this->mMagicM = q2 + 1;
    if (mDivisor < 0) this->mMagicM = -this->mMagicM;
    this->mMagicS = p - 32;

    if ((mDivisor > 0) && (mMagicM < 0))
      mAddSign = 1;
    else if ((mDivisor < 0) && (mMagicM > 0))
      mAddSign = -1;
    else
      mAddSign = 0;
  }

  __host__ __device__ friend int operator/(int const dividend, IntFastDiv const& divisor);
};

__host__ __device__ inline int operator/(int const dividend, IntFastDiv const& divisor) {
  int q;
#ifdef __CUDA_ARCH__
  asm("mul.hi.s32 %0, %1, %2;" : "=r"(q) : "r"(divisor.mMagicM), "r"(dividend));
#else
  q = (((unsigned long long)((long long)divisor.mMagicM * (long long)dividend)) >> 32);
#endif
  q += dividend * divisor.mAddSign;
  if (divisor.mMagicS >= 0) {
    q >>= divisor.mMagicS;
    q += (((unsigned int)q) >> 31);
  }
  return q;
}

__host__ __device__ inline int operator%(int const dividend, IntFastDiv const& divisor) {
  int quotient = dividend / divisor;
  int remainder = dividend - quotient * divisor;
  return remainder;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 2: packed_as — maps a base type + vector width to the appropriate CUDA vector type.
// In a sub-namespace to avoid collision with identically-named templates in norm.cuh.
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace fused_rope_detail {

template <typename T, int N>
struct packed_as {
  static_assert(N == 1, "packed_as only supports N=1, 2, 4");
  using type = T;
};

template <>
struct packed_as<uint, 1> {
  using type = uint;
};

template <>
struct packed_as<uint, 2> {
  using type = uint2;
};

template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

}  // namespace fused_rope_detail

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 3: FP8 E4M3 quantization helpers
// SM89+ (Ada/Hopper/Blackwell): native vectorized PTX conversion
// SM < 89: scalar __nv_fp8_e4m3 constructor fallback
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ __nv_fp8_e4m3 quantize_fp8_e4m3(float val, float scale = 1.0f) {
  return __nv_fp8_e4m3(val * scale);
}

__device__ __forceinline__ uint16_t float2_to_fp8_e4m3_packed(float2 val, float scale = 1.0f) {
  __nv_fp8_e4m3 fp8_0 = quantize_fp8_e4m3(val.x, scale);
  __nv_fp8_e4m3 fp8_1 = quantize_fp8_e4m3(val.y, scale);
  return (*reinterpret_cast<uint8_t*>(&fp8_0)) | ((*reinterpret_cast<uint8_t*>(&fp8_1)) << 8);
}

__device__ __forceinline__ uint32_t float4_to_fp8_e4m3_packed(float f0, float f1, float f2,
                                                              float f3, float scale = 1.0f) {
  float scaled0 = f0 * scale;
  float scaled1 = f1 * scale;
  float scaled2 = f2 * scale;
  float scaled3 = f3 * scale;

  uint32_t result;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "{\n\t"
      ".reg .b16 r_lo, r_hi;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 r_lo, %4, %3;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 r_hi, %2, %1;\n\t"
      "mov.b32 %0, {r_hi, r_lo};\n\t"
      "}"
      : "=r"(result)
      : "f"(scaled0), "f"(scaled1), "f"(scaled2), "f"(scaled3));
#else
  __nv_fp8_e4m3 fp8_0 = __nv_fp8_e4m3(scaled0);
  __nv_fp8_e4m3 fp8_1 = __nv_fp8_e4m3(scaled1);
  __nv_fp8_e4m3 fp8_2 = __nv_fp8_e4m3(scaled2);
  __nv_fp8_e4m3 fp8_3 = __nv_fp8_e4m3(scaled3);
  result = (*reinterpret_cast<uint8_t*>(&fp8_0)) | ((*reinterpret_cast<uint8_t*>(&fp8_1)) << 8) |
           ((*reinterpret_cast<uint8_t*>(&fp8_2)) << 16) |
           ((*reinterpret_cast<uint8_t*>(&fp8_3)) << 24);
#endif

  return result;
}

__device__ __forceinline__ uint2 float8_to_fp8_e4m3_packed(float2 val0, float2 val1, float2 val2,
                                                           float2 val3, float scale = 1.0f) {
  uint32_t packed_lo = float4_to_fp8_e4m3_packed(val0.x, val0.y, val1.x, val1.y, scale);
  uint32_t packed_hi = float4_to_fp8_e4m3_packed(val2.x, val2.y, val3.x, val3.y, scale);
  return make_uint2(packed_lo, packed_hi);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 4: Blackwell FFMA2 intrinsics (SM100+) with scalar fallbacks
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__device__ __forceinline__ float2 fmul2(const float2& a, const float2& b) {
  uint64_t c;
  asm volatile("mul.f32x2 %0, %1, %2;\n"
               : "=l"(c)
               : "l"(reinterpret_cast<const uint64_t&>(a)),
                 "l"(reinterpret_cast<const uint64_t&>(b)));
  return reinterpret_cast<float2&>(c);
}

__device__ __forceinline__ float2 ffma2(const float2& a, const float2& b, const float2& c) {
  uint64_t d;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(d)
               : "l"(reinterpret_cast<const uint64_t&>(a)),
                 "l"(reinterpret_cast<const uint64_t&>(b)),
                 "l"(reinterpret_cast<const uint64_t&>(c)));
  return reinterpret_cast<float2&>(d);
}

__device__ __forceinline__ float2 fadd2(const float2& a, const float2& b) {
  uint64_t c;
  asm volatile("add.f32x2 %0, %1, %2;\n"
               : "=l"(c)
               : "l"(reinterpret_cast<const uint64_t&>(a)),
                 "l"(reinterpret_cast<const uint64_t&>(b)));
  return reinterpret_cast<float2&>(c);
}
#else
__device__ __forceinline__ float2 fmul2(const float2& a, const float2& b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

__device__ __forceinline__ float2 ffma2(const float2& a, const float2& b, const float2& c) {
  return make_float2(a.x * b.x + c.x, a.y * b.y + c.y);
}

__device__ __forceinline__ float2 fadd2(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 5: Vectorized FP8 store helper
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int numElemsPerThread>
__device__ __forceinline__ void quantize_store_fp8(float2 const* elements, __nv_fp8_e4m3* out,
                                                   int offset, float scale) {
  constexpr int numFloat2PerThread = numElemsPerThread / 2;
  if constexpr (numElemsPerThread == 2) {
    uint16_t packed = float2_to_fp8_e4m3_packed(elements[0], scale);
    *reinterpret_cast<uint16_t*>(&out[offset]) = packed;
  } else if constexpr (numElemsPerThread == 4) {
    uint32_t packed = float4_to_fp8_e4m3_packed(elements[0].x, elements[0].y, elements[1].x,
                                                elements[1].y, scale);
    *reinterpret_cast<uint32_t*>(&out[offset]) = packed;
  } else if constexpr (numElemsPerThread == 8) {
    uint2 packed =
        float8_to_fp8_e4m3_packed(elements[0], elements[1], elements[2], elements[3], scale);
    *reinterpret_cast<uint2*>(&out[offset]) = packed;
  } else {
#pragma unroll
    for (int ii = 0; ii < numFloat2PerThread; ii++) {
      out[offset + ii * 2] = quantize_fp8_e4m3(elements[ii].x, scale);
      out[offset + ii * 2 + 1] = quantize_fp8_e4m3(elements[ii].y, scale);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 6: Fused QK RMSNorm + RoPE kernel
//
// Performs across-heads RMSNorm and 3D RoPE in a single kernel (for self-attention).
// Also copies V to a separate contiguous output buffer with optional FP8 quantization.
//
// Architecture:
// - 2D grid: blockIdx.x = tokenIdx, blockIdx.y = block type (0=Q, 1=K, 2=V)
// - Q/K blocks: warps cooperate for RMSNorm reduction, then each warp applies norm + RoPE
// - V blocks: just copy + optional FP8 quantize (no RMSNorm, no RoPE)
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int THREADS_PER_WARP = 32;

template <int head_dim, bool interleave, int MAX_HEADS = 32, bool OUTPUT_FP8 = false,
          bool HAS_YARN = false>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16 const* qkv_in, void* q_out, void* k_out, void* v_out, int const num_heads_q,
    int const num_heads_k, int const num_heads_v, float const eps, __nv_bfloat16 const* q_weight,
    __nv_bfloat16 const* k_weight, float const* freq_table, int const num_tokens,
    IntFastDiv const seq_len, IntFastDiv const ppw, IntFastDiv const pphppw,
    int const num_frame_channels, int const num_height_channels, int const num_width_channels,
    float attention_factor, bool is_qk_norm, float output_quant_scale, float v_quant_scale) {
  static_assert((head_dim & (head_dim - 1)) == 0,
                "head_dim must be a power of 2 (required for bitwise modulo in NeoX RoPE path)");
  static_assert(
      head_dim % (THREADS_PER_WARP * 2) == 0,
      "head_dim must be divisible by 64 (each warp processes one head with even element count)");
  constexpr int log_head_dim = __builtin_ctz(head_dim);
  constexpr int numElemsPerThread = head_dim / THREADS_PER_WARP;
  static_assert(numElemsPerThread % 2 == 0, "numElemsPerThread must be divisible by 2");
  constexpr int numFloat2PerThread = numElemsPerThread / 2;
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename fused_rope_detail::packed_as<uint, vecSize>::type;

  int const warpId = threadIdx.x / THREADS_PER_WARP;
  int const laneId = threadIdx.x % THREADS_PER_WARP;

  int const tokenIdx = blockIdx.x;
  int const blockType = blockIdx.y;

  if (tokenIdx >= num_tokens) return;

  int const num_heads = num_heads_q + num_heads_k + num_heads_v;
  int const baseOffset = tokenIdx * num_heads * head_dim;
  int const headIdx = warpId;

  int const threadHeadOffset = headIdx * head_dim + laneId * numElemsPerThread;

  // ========== V blocks: simple copy + optional FP8 quantize ==========
  if (blockType == 2) {
    if (headIdx >= num_heads_v) return;

    int const v_input_offset =
        baseOffset + (num_heads_q + num_heads_k) * head_dim + threadHeadOffset;
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv_in[v_input_offset]);

    int const v_output_offset = tokenIdx * num_heads_v * head_dim + threadHeadOffset;

    if constexpr (OUTPUT_FP8) {
      float2 elements[numFloat2PerThread];
#pragma unroll
      for (int i = 0; i < vecSize; i++) {
        elements[i] = __bfloat1622float2(
            *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
      }
      quantize_store_fp8<numElemsPerThread>(elements, reinterpret_cast<__nv_fp8_e4m3*>(v_out),
                                            v_output_offset, v_quant_scale);
    } else {
      __nv_bfloat16* bf16_out = reinterpret_cast<__nv_bfloat16*>(v_out);
      *reinterpret_cast<vec_T*>(&bf16_out[v_output_offset]) = vec;
    }
    return;
  }

  // ========== Q/K blocks: RMSNorm + RoPE + optional FP8 quantize ==========
  __shared__ float sharedSumOfSquares[MAX_HEADS];

  bool const isQ = (blockType == 0);

  int const num_heads_this = isQ ? num_heads_q : num_heads_k;

  bool const validHead = (headIdx < num_heads_this);
  int const hidden_dim_this = num_heads_this * head_dim;

  float2 elements[numFloat2PerThread];
  float2 r_weight[numFloat2PerThread];

  int const qkSegmentStart = isQ ? 0 : num_heads_q * head_dim;
  int const inputOffset = baseOffset + qkSegmentStart + threadHeadOffset;

  float2 sumOfSquares = make_float2(0.0f, 0.0f);

  if (validHead) {
    __nv_bfloat16 const* weight_ptr = isQ ? q_weight : k_weight;

    vec_T weight_vec = *reinterpret_cast<vec_T const*>(&weight_ptr[threadHeadOffset]);
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv_in[inputOffset]);

#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      r_weight[i] = __bfloat1622float2(
          *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&weight_vec) + i));

      float2 vals =
          __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));

      sumOfSquares = ffma2(vals, vals, sumOfSquares);

      elements[i] = vals;
    }
  }

  if (is_qk_norm) {
    float sumOfSquaresScalar = sumOfSquares.x + sumOfSquares.y;

#pragma unroll
    for (int step = THREADS_PER_WARP / 2; step > 0; step /= 2) {
      sumOfSquaresScalar += __shfl_xor_sync(0xffffffff, sumOfSquaresScalar, step);
    }

    if (laneId == 0) {
      sharedSumOfSquares[warpId] = validHead ? sumOfSquaresScalar : 0.0f;
    }
    __syncthreads();

    int const warpsPerBlock = blockDim.x / THREADS_PER_WARP;
    float totalSumOfSquares = (laneId < warpsPerBlock) ? sharedSumOfSquares[laneId] : 0.0f;

#pragma unroll
    for (int step = THREADS_PER_WARP / 2; step > 0; step /= 2) {
      totalSumOfSquares += __shfl_xor_sync(0xffffffff, totalSumOfSquares, step);
    }

    float rms_rcp = rsqrtf(totalSumOfSquares / static_cast<float>(hidden_dim_this) + eps);

    if (validHead) {
      float2 rms_rcp_vec = make_float2(rms_rcp, rms_rcp);
#pragma unroll
      for (int i = 0; i < numFloat2PerThread; i++) {
        elements[i] = fmul2(fmul2(elements[i], rms_rcp_vec), r_weight[i]);

        // Round to BF16 and back to match the precision of the unfused reference.
        // Without this, the fused kernel carries extra float32 mantissa bits
        // through RoPE, producing results that differ from the reference.
        __nv_bfloat162 tmp = __float22bfloat162_rn(elements[i]);
        elements[i] = __bfloat1622float2(tmp);
      }
    }
  }

  // Apply RoPE to normalized elements
  if (validHead) {
    float2 elements2[numFloat2PerThread];
    float2 cos_vals[numFloat2PerThread];
    float2 sin_vals[numFloat2PerThread];

    int const token_idx_in_seq = tokenIdx % seq_len;
    int const pos_id_t = token_idx_in_seq / pphppw;
    int const pos_id_x = token_idx_in_seq % pphppw;
    int const pos_id_h = pos_id_x / ppw;
    int const pos_id_w = pos_id_x % ppw;

    int32_t height_slice_start = num_frame_channels;
    int32_t width_slice_start = num_frame_channels + num_height_channels;

    if constexpr (interleave) {
#pragma unroll
      for (int ii = 0; ii < numFloat2PerThread; ii++) {
        int dim_idx_x = laneId * numElemsPerThread + ii * 2;
        int pos_id = dim_idx_x >= width_slice_start    ? pos_id_w
                     : dim_idx_x >= height_slice_start ? pos_id_h
                                                       : pos_id_t;

        float freq_xy = freq_table[dim_idx_x >> 1];
        float theta_xy = pos_id * freq_xy;
        float sin_xy, cos_xy;
        __sincosf(theta_xy, &sin_xy, &cos_xy);

        sin_vals[ii] = make_float2(sin_xy, sin_xy);
        cos_vals[ii] = make_float2(cos_xy, cos_xy);

        elements2[ii] = make_float2(-elements[ii].y, elements[ii].x);
      }
    } else {
      __syncwarp();
#pragma unroll
      for (int ii = 0; ii < numFloat2PerThread; ii++) {
        float elem_x = __shfl_xor_sync(0xffffffff, elements[ii].x, 16);
        float elem_y = __shfl_xor_sync(0xffffffff, elements[ii].y, 16);
        if (laneId < 16) {
          elem_x = -elem_x;
          elem_y = -elem_y;
        }
        elements2[ii] = make_float2(elem_x, elem_y);
      }

#pragma unroll
      for (int ii = 0; ii < numFloat2PerThread; ii++) {
        int dim_idx_x = laneId * numElemsPerThread + ii * 2;
        dim_idx_x = (dim_idx_x * 2) & ((1 << log_head_dim) - 1);
        int pos_id = dim_idx_x >= width_slice_start    ? pos_id_w
                     : dim_idx_x >= height_slice_start ? pos_id_h
                                                       : pos_id_t;

        float freq_x = freq_table[dim_idx_x >> 1];
        float theta_x = pos_id * freq_x;
        float sin_x, cos_x;
        __sincosf(theta_x, &sin_x, &cos_x);

        int dim_idx_y = laneId * numElemsPerThread + ii * 2 + 1;
        dim_idx_y = (dim_idx_y * 2) & ((1 << log_head_dim) - 1);
        pos_id = dim_idx_y >= width_slice_start    ? pos_id_w
                 : dim_idx_y >= height_slice_start ? pos_id_h
                                                   : pos_id_t;

        float freq_y = freq_table[dim_idx_y >> 1];
        float theta_y = pos_id * freq_y;
        float sin_y, cos_y;
        __sincosf(theta_y, &sin_y, &cos_y);

        sin_vals[ii] = make_float2(sin_x, sin_y);
        cos_vals[ii] = make_float2(cos_x, cos_y);
      }
      __syncwarp();
    }

    if constexpr (HAS_YARN) {
      float2 attention_factor_vec = make_float2(attention_factor, attention_factor);
#pragma unroll
      for (int ii = 0; ii < numFloat2PerThread; ii++) {
        elements[ii] = fmul2(ffma2(elements[ii], cos_vals[ii], fmul2(elements2[ii], sin_vals[ii])),
                             attention_factor_vec);
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < numFloat2PerThread; ii++) {
        elements[ii] = ffma2(elements[ii], cos_vals[ii], fmul2(elements2[ii], sin_vals[ii]));
      }
    }

    int const outputBase = tokenIdx * num_heads_this * head_dim;
    int const outputOffset = outputBase + threadHeadOffset;

    if constexpr (OUTPUT_FP8) {
      __nv_fp8_e4m3* fp8_out =
          isQ ? reinterpret_cast<__nv_fp8_e4m3*>(q_out) : reinterpret_cast<__nv_fp8_e4m3*>(k_out);
      quantize_store_fp8<numElemsPerThread>(elements, fp8_out, outputOffset, output_quant_scale);
    } else {
      __nv_bfloat16* bf16_out =
          isQ ? reinterpret_cast<__nv_bfloat16*>(q_out) : reinterpret_cast<__nv_bfloat16*>(k_out);
      vec_T vec;
      for (int ii = 0; ii < vecSize; ii++) {
        __nv_bfloat162 vals = __float22bfloat162_rn(elements[ii]);
        reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint*>(&vec) + ii)) = vals;
      }
      vec_T* outputPtr = reinterpret_cast<vec_T*>(&bf16_out[outputOffset]);
      *outputPtr = vec;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 7: Dispatch macros
////////////////////////////////////////////////////////////////////////////////////////////////////

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

#define DISPATCH_OUTPUT_FP8(output_fp8, OUTPUT_FP8, ...) \
  if (output_fp8) {                                      \
    const bool OUTPUT_FP8 = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool OUTPUT_FP8 = false;                       \
    __VA_ARGS__                                          \
  }

#define DISPATCH_HAS_YARN(has_yarn, HAS_YARN, ...) \
  if (has_yarn) {                                  \
    const bool HAS_YARN = true;                    \
    __VA_ARGS__                                    \
  } else {                                         \
    const bool HAS_YARN = false;                   \
    __VA_ARGS__                                    \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 8: Host-side frequency computation
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float compute_adjusted_freq_host(int half_dim_val, float base, int dim_size,
                                               float factor, float low, float high) {
  float freq = powf(base, -2.0f * half_dim_val / static_cast<float>(dim_size));
  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;

    float high_adj = high;
    if (fabsf(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }
    float linear_func = (static_cast<float>(half_dim_val) - low) / (high_adj - low);
    float ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
    float inv_freq_extrapolation_factor = 1.0f - ramp_func;
    freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor) +
           inv_freq_extrapolation * inv_freq_extrapolation_factor;
  }
  return freq;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 9: Frequency table cache + host launcher
//
// The cache is allocated once and never freed to ensure cudagraph capture compatibility.
////////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  float* d_ptr = nullptr;
  int alloc_floats = 0;
  int head_dim = 0;
  float base = 0.0f;
  float factor = 0.0f;
  float low = 0.0f;
  float high = 0.0f;
  int num_frame_channels = 0;
  int num_height_channels = 0;
  int num_width_channels = 0;
} s_freq_cache;

inline void launchFusedQKNormRope(void const* qkv_in, void* q_out, void* k_out, void* v_out,
                                  int const num_tokens, int const seq_len, int const ppf,
                                  int const pph, int const ppw, int const num_frame_channels,
                                  int const num_height_channels, int const num_width_channels,
                                  int const num_heads_q, int const num_heads_k,
                                  int const num_heads_v, int const head_dim, float const eps,
                                  void const* q_weight, void const* k_weight, float const base,
                                  bool const interleave, float factor, float low, float high,
                                  float attention_factor, cudaStream_t stream, bool is_qk_norm,
                                  int const num_sms, bool output_fp8, float output_quant_scale,
                                  float v_quant_scale) {
  FLASHINFER_FUSED_CHECK((head_dim & (head_dim - 1)) == 0);

  if (factor == 1.0f) {
    FLASHINFER_FUSED_CHECK(attention_factor == 1.0f);
  }

  int const table_size = head_dim / 2;

  if (s_freq_cache.alloc_floats < table_size) {
    if (s_freq_cache.d_ptr != nullptr) {
      cudaFree(s_freq_cache.d_ptr);
    }
    cudaMalloc(&s_freq_cache.d_ptr, table_size * sizeof(float));
    s_freq_cache.alloc_floats = table_size;
  }

  bool cache_miss =
      (s_freq_cache.head_dim != head_dim || s_freq_cache.base != base ||
       s_freq_cache.factor != factor || s_freq_cache.low != low || s_freq_cache.high != high ||
       s_freq_cache.num_frame_channels != num_frame_channels ||
       s_freq_cache.num_height_channels != num_height_channels ||
       s_freq_cache.num_width_channels != num_width_channels);

  if (cache_miss) {
    FLASHINFER_FUSED_CHECK(table_size <= 128);
    float h_freq_table[128];
    int offset = 0;

    for (int i = 0; i < num_frame_channels / 2; i++)
      h_freq_table[offset++] =
          compute_adjusted_freq_host(i, base, num_frame_channels, factor, low, high);

    for (int i = 0; i < num_height_channels / 2; i++)
      h_freq_table[offset++] =
          compute_adjusted_freq_host(i, base, num_height_channels, factor, low, high);

    for (int i = 0; i < num_width_channels / 2; i++)
      h_freq_table[offset++] =
          compute_adjusted_freq_host(i, base, num_width_channels, factor, low, high);

    FLASHINFER_FUSED_CHECK(offset == table_size);

    cudaMemcpy(s_freq_cache.d_ptr, h_freq_table, table_size * sizeof(float),
               cudaMemcpyHostToDevice);

    s_freq_cache.head_dim = head_dim;
    s_freq_cache.base = base;
    s_freq_cache.factor = factor;
    s_freq_cache.low = low;
    s_freq_cache.high = high;
    s_freq_cache.num_frame_channels = num_frame_channels;
    s_freq_cache.num_height_channels = num_height_channels;
    s_freq_cache.num_width_channels = num_width_channels;
  }

  int const maxHeads = max(max(num_heads_q, num_heads_k), num_heads_v);
  int const warpsPerBlock = maxHeads;
  int const blockSize = warpsPerBlock * THREADS_PER_WARP;

  dim3 gridDim(num_tokens, 3);
  dim3 blockDim(blockSize);

  bool const has_yarn = (factor != 1.0f);

#define LAUNCH_KERNEL(HD)                                                                         \
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                                                   \
    DISPATCH_OUTPUT_FP8(output_fp8, OUTPUT_FP8, {                                                 \
      DISPATCH_HAS_YARN(has_yarn, HAS_YARN, {                                                     \
        fusedQKNormRopeKernel<HD, INTERLEAVE, 32, OUTPUT_FP8, HAS_YARN>                           \
            <<<gridDim, blockDim, 0, stream>>>(                                                   \
                reinterpret_cast<__nv_bfloat16 const*>(qkv_in), q_out, k_out, v_out, num_heads_q, \
                num_heads_k, num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),  \
                reinterpret_cast<__nv_bfloat16 const*>(k_weight), s_freq_cache.d_ptr, num_tokens, \
                IntFastDiv(seq_len), IntFastDiv(ppw), IntFastDiv(pph * ppw), num_frame_channels,  \
                num_height_channels, num_width_channels, attention_factor, is_qk_norm,            \
                output_quant_scale, v_quant_scale);                                               \
      });                                                                                         \
    });                                                                                           \
  })

  switch (head_dim) {
    case 64:
      LAUNCH_KERNEL(64);
      break;
    case 128:
      LAUNCH_KERNEL(128);
      break;
    case 256:
      LAUNCH_KERNEL(256);
      break;
    default:
      FLASHINFER_FUSED_CHECK(false);  // Unsupported head_dim
      break;
  }

#undef LAUNCH_KERNEL
}

}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_QK_RMSNORM_ROPE_CUH_
