/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef VEC_DTYPES_CUH_
#define VEC_DTYPES_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace flashinfer {

#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
#define FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
#endif

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

/******************* vec_t type cast *******************/

template <typename dst_t, typename src_t>
struct vec_cast {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(dst_t* dst, const src_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = (dst_t)src[i];
    }
  }
};

template <>
struct vec_cast<float, half> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float* dst, const half* src) {
    if constexpr (vec_size == 1) {
      dst[0] = (float)src[0];
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((float2*)dst)[i] = __half22float2(((half2*)src)[i]);
      }
    }
  }
};

template <>
struct vec_cast<half, float> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(half* dst, const float* src) {
    if constexpr (vec_size == 1) {
      dst[0] = __float2half(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((half2*)dst)[i] = __float22half2_rn(((float2*)src)[i]);
      }
    }
  }
};

template <typename T>
constexpr FLASHINFER_INLINE int get_exponent_bits() {
  if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
    return 4;
  } else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {
    return 5;
  } else if constexpr (std::is_same<T, half>::value) {
    return 5;
  } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
    return 8;
  }
}

template <typename T>
constexpr FLASHINFER_INLINE int get_mantissa_bits() {
  if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
    return 3;
  } else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {
    return 2;
  } else if constexpr (std::is_same<T, half>::value) {
    return 11;
  } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
    return 7;
  }
}

/*!
 * \brief Fallback to software fast dequant implementation if hardware dequantization is not
 * available.
 * \note Inspired by Marlin's fast dequantization, but here we don't have to permute
 * weights order.
 * \ref
 * https://github.com/vllm-project/vllm/blob/6dffa4b0a6120159ef2fe44d695a46817aff65bc/csrc/quantization/fp8/fp8_marlin.cu#L120
 */
template <typename fp8_dtype, typename fp16_dtype>
__device__ void fast_dequant_f8f16x4(uint32_t* input, uint2* output) {
  uint32_t q = *input;
  if constexpr (std::is_same<fp8_dtype, __nv_fp8_e5m2>::value &&
                std::is_same<fp16_dtype, half>::value) {
    output->x = __byte_perm(0U, q, 0x5140);
    output->y = __byte_perm(0U, q, 0x7362);
  } else {
    constexpr int FP8_EXPONENT = get_exponent_bits<fp8_dtype>();
    constexpr int FP8_MANTISSA = get_mantissa_bits<fp8_dtype>();
    constexpr int FP16_EXPONENT = get_exponent_bits<fp16_dtype>();

    constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
    // Calculate MASK for extracting mantissa and exponent
    constexpr int MASK1 = 0x80000000;
    constexpr int MASK2 = MASK1 >> (FP8_EXPONENT + FP8_MANTISSA);
    constexpr int MASK3 = MASK2 & 0x7fffffff;
    constexpr int MASK = MASK3 | (MASK3 >> 16);
    q = __byte_perm(q, q, 0x1302);

    // Extract and shift FP8 values to FP16 format
    uint32_t Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
    uint32_t Out2 = ((q << 8) & 0x80008000) | (((q << 8) & MASK) >> RIGHT_SHIFT);

    constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
    // Construct and apply exponent bias
    if constexpr (std::is_same<fp16_dtype, half>::value) {
      const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

      // Convert to half2 and apply bias
      *(half2*)&(output->x) = __hmul2(*reinterpret_cast<const half2*>(&Out1), bias_reg);
      *(half2*)&(output->y) = __hmul2(*reinterpret_cast<const half2*>(&Out2), bias_reg);
    } else {
      constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
      const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));
      // Convert to bfloat162 and apply bias
      *(nv_bfloat162*)&(output->x) =
          __hmul2(*reinterpret_cast<const nv_bfloat162*>(&Out1), bias_reg);
      *(nv_bfloat162*)&(output->y) =
          __hmul2(*reinterpret_cast<const nv_bfloat162*>(&Out2), bias_reg);
    }
  }
}

template <>
struct vec_cast<nv_bfloat16, __nv_fp8_e4m3> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(nv_bfloat16* dst, const __nv_fp8_e4m3* src) {
    if constexpr (vec_size == 1) {
      dst[0] = nv_bfloat16(src[0]);
    } else if constexpr (vec_size == 2) {
      dst[0] = nv_bfloat16(src[0]);
      dst[1] = nv_bfloat16(src[1]);
    } else {
      static_assert(vec_size % 4 == 0, "vec_size must be a multiple of 4");
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>((uint32_t*)&src[i * 4],
                                                         (uint2*)&dst[i * 4]);
      }
    }
  }
};

template <>
struct vec_cast<nv_bfloat16, __nv_fp8_e5m2> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(nv_bfloat16* dst, const __nv_fp8_e5m2* src) {
    if constexpr (vec_size == 1) {
      dst[0] = nv_bfloat16(src[0]);
    } else if constexpr (vec_size == 2) {
      dst[0] = nv_bfloat16(src[0]);
      dst[1] = nv_bfloat16(src[1]);
    } else {
      static_assert(vec_size % 4 == 0, "vec_size must be a multiple of 4");
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        fast_dequant_f8f16x4<__nv_fp8_e5m2, nv_bfloat16>((uint32_t*)&src[i * 4],
                                                         (uint2*)&dst[i * 4]);
      }
    }
  }
};

template <>
struct vec_cast<__nv_fp8_e4m3, half> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(__nv_fp8_e4m3* dst, const half* src) {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    if constexpr (vec_size == 1) {
      dst[0] = __nv_fp8_e4m3(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        uint16_t y;
        uint32_t x = *(uint32_t*)&src[i * 2];
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(y) : "r"(x));
        *(uint16_t*)&dst[i * 2] = y;
      }
    }
#else
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = __nv_fp8_e4m3(src[i]);
    }
#endif  // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
  }
};

template <>
struct vec_cast<__nv_fp8_e5m2, half> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(__nv_fp8_e5m2* dst, const half* src) {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    if constexpr (vec_size == 1) {
      dst[0] = __nv_fp8_e5m2(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        uint16_t y;
        uint32_t x = *(uint32_t*)&src[i * 2];
        asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(y) : "r"(x));
        *(uint16_t*)&dst[i * 2] = y;
      }
    }
#else
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = __nv_fp8_e5m2(src[i]);
    }
#endif  // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
  }
};

template <>
struct vec_cast<half, __nv_fp8_e4m3> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(half* dst, const __nv_fp8_e4m3* src) {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    if constexpr (vec_size == 1) {
      dst[0] = half(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        uint32_t y;
        uint16_t x = *(uint16_t*)&src[i * 2];
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(y) : "h"(x));
        *(uint32_t*)&dst[i * 2] = y;
      }
    }
#else
    if constexpr (vec_size == 1) {
      dst[0] = half(src[0]);
    } else if constexpr (vec_size == 2) {
      dst[0] = half(src[0]);
      dst[1] = half(src[1]);
    } else {
      static_assert(vec_size % 4 == 0, "vec_size must be a multiple of 4");
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        fast_dequant_f8f16x4<__nv_fp8_e4m3, half>((uint32_t*)&src[i * 4], (uint2*)&dst[i * 4]);
      }
    }
#endif  // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
  }
};

template <>
struct vec_cast<half, __nv_fp8_e5m2> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(half* dst, const __nv_fp8_e5m2* src) {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    if constexpr (vec_size == 1) {
      dst[0] = half(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        uint32_t y;
        uint16_t x = *(uint16_t*)&src[i * 2];
        asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(y) : "h"(x));
        *(uint32_t*)&dst[i * 2] = y;
      }
    }
#else
    if constexpr (vec_size == 1) {
      dst[0] = half(src[0]);
    } else if constexpr (vec_size == 2) {
      dst[0] = half(src[0]);
      dst[1] = half(src[1]);
    } else {
      static_assert(vec_size % 4 == 0, "vec_size must be a multiple of 4");
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        fast_dequant_f8f16x4<__nv_fp8_e5m2, half>((uint32_t*)&src[i * 4], (uint2*)&dst[i * 4]);
      }
    }
#endif  // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
  }
};

template <>
struct vec_cast<float, nv_bfloat16> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float* dst, const nv_bfloat16* src) {
    if constexpr (vec_size == 1) {
      dst[0] = (float)src[0];
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((float2*)dst)[i] = __bfloat1622float2(((nv_bfloat162*)src)[i]);
      }
    }
  }
};

template <>
struct vec_cast<nv_bfloat16, float> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(nv_bfloat16* dst, const float* src) {
    if constexpr (vec_size == 1) {
      dst[0] = nv_bfloat16(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((nv_bfloat162*)dst)[i] = __float22bfloat162_rn(((float2*)src)[i]);
      }
    }
  }
};

template <typename float_t, size_t vec_size>
struct vec_t {
  FLASHINFER_INLINE float_t& operator[](size_t i);
  FLASHINFER_INLINE const float_t& operator[](size_t i) const;
  FLASHINFER_INLINE void fill(float_t val);
  FLASHINFER_INLINE void load(const float_t* ptr);
  FLASHINFER_INLINE void store(float_t* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr);
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const;
  FLASHINFER_INLINE static void memcpy(float_t* dst, const float_t* src);
  FLASHINFER_INLINE float_t* ptr();
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_from_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const vec_t<src_float_t, vec_size>& src) {
  vec_cast<tgt_float_t, src_float_t>::cast<vec_size>(
      dst.ptr(), const_cast<vec_t<src_float_t, vec_size>*>(&src)->ptr());
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_load_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const src_float_t* src_ptr) {
  if constexpr (std::is_same<src_float_t, tgt_float_t>::value) {
    dst.load(src_ptr);
  } else {
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_store_impl(tgt_float_t* dst_ptr,
                                       const vec_t<src_float_t, vec_size>& src) {
  if constexpr (std::is_same<src_float_t, tgt_float_t>::value) {
    src.store(dst_ptr);
  } else {
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

/******************* vec_t<__nv_fp8_e4m3> *******************/

// __nv_fp8_e4m3 x 1
template <>
struct vec_t<__nv_fp8_e4m3, 1> {
  __nv_fp8_e4m3 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 1>::fill(__nv_fp8_e4m3 val) { data = val; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 1>::load(const __nv_fp8_e4m3* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 1>::store(__nv_fp8_e4m3* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 1>::memcpy(__nv_fp8_e4m3* dst,
                                                       const __nv_fp8_e4m3* src) {
  *dst = *src;
}

// __nv_fp8_e4m3 x 2
template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  __nv_fp8x2_e4m3 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::load(const __nv_fp8_e4m3* ptr) {
  data = *((__nv_fp8x2_e4m3*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::store(__nv_fp8_e4m3* ptr) const {
  *((__nv_fp8x2_e4m3*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::memcpy(__nv_fp8_e4m3* dst,
                                                       const __nv_fp8_e4m3* src) {
  *((__nv_fp8x2_e4m3*)dst) = *((__nv_fp8x2_e4m3*)src);
}

// __nv_fp8_e4m3 x 4

template <>
struct vec_t<__nv_fp8_e4m3, 4> {
  __nv_fp8x4_e4m3 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::load(const __nv_fp8_e4m3* ptr) {
  data = *((__nv_fp8x4_e4m3*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::store(__nv_fp8_e4m3* ptr) const {
  *((__nv_fp8x4_e4m3*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::memcpy(__nv_fp8_e4m3* dst,
                                                       const __nv_fp8_e4m3* src) {
  *((__nv_fp8x4_e4m3*)dst) = *((__nv_fp8x4_e4m3*)src);
}

// __nv_fp8_e4m3 x 8

template <>
struct vec_t<__nv_fp8_e4m3, 8> {
  uint2 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 8>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 8>::fill(__nv_fp8_e4m3 val) {
  ((__nv_fp8x4_e4m3*)(&data.x))->__x =
      (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
      (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
  ((__nv_fp8x4_e4m3*)(&data.y))->__x =
      (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
      (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 8>::load(const __nv_fp8_e4m3* ptr) {
  data = *((uint2*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 8>::store(__nv_fp8_e4m3* ptr) const {
  *((uint2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 8>::memcpy(__nv_fp8_e4m3* dst,
                                                       const __nv_fp8_e4m3* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// __nv_fp8_e4m3 x 16 or more
template <size_t vec_size>
struct vec_t<__nv_fp8_e4m3, vec_size> {
  uint4 data[vec_size / 16];

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)data)[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)data)[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((__nv_fp8x4_e4m3*)(&(data[i].x)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].y)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].z)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].w)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
    }
  }
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};

/******************* vec_t<__nv_fp8_e5m2> *******************/

// __nv_fp8_e5m2 x 1
template <>
struct vec_t<__nv_fp8_e5m2, 1> {
  __nv_fp8_e5m2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 1>::fill(__nv_fp8_e5m2 val) { data = val; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 1>::load(const __nv_fp8_e5m2* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 1>::store(__nv_fp8_e5m2* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 1>::memcpy(__nv_fp8_e5m2* dst,
                                                       const __nv_fp8_e5m2* src) {
  *dst = *src;
}

// __nv_fp8_e5m2 x 2
template <>
struct vec_t<__nv_fp8_e5m2, 2> {
  __nv_fp8x2_e5m2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::load(const __nv_fp8_e5m2* ptr) {
  data = *((__nv_fp8x2_e5m2*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::store(__nv_fp8_e5m2* ptr) const {
  *((__nv_fp8x2_e5m2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::memcpy(__nv_fp8_e5m2* dst,
                                                       const __nv_fp8_e5m2* src) {
  *((__nv_fp8x2_e5m2*)dst) = *((__nv_fp8x2_e5m2*)src);
}

// __nv_fp8_e5m2 x 4

template <>
struct vec_t<__nv_fp8_e5m2, 4> {
  __nv_fp8x4_e5m2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::load(const __nv_fp8_e5m2* ptr) {
  data = *((__nv_fp8x4_e5m2*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::store(__nv_fp8_e5m2* ptr) const {
  *((__nv_fp8x4_e5m2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::memcpy(__nv_fp8_e5m2* dst,
                                                       const __nv_fp8_e5m2* src) {
  *((__nv_fp8x4_e5m2*)dst) = *((__nv_fp8x4_e5m2*)src);
}

// __nv_fp8_e5m2 x 8

template <>
struct vec_t<__nv_fp8_e5m2, 8> {
  uint2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 8>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 8>::fill(__nv_fp8_e5m2 val) {
  ((__nv_fp8x4_e5m2*)(&data.x))->__x =
      (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
      (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
  ((__nv_fp8x4_e5m2*)(&data.y))->__x =
      (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
      (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 8>::load(const __nv_fp8_e5m2* ptr) {
  data = *((uint2*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 8>::store(__nv_fp8_e5m2* ptr) const {
  *((uint2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 8>::memcpy(__nv_fp8_e5m2* dst,
                                                       const __nv_fp8_e5m2* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// __nv_fp8_e5m2 x 16 or more

template <size_t vec_size>
struct vec_t<__nv_fp8_e5m2, vec_size> {
  uint4 data[vec_size / 16];

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)data)[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)data)[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((__nv_fp8x4_e5m2*)(&(data[i].x)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2*)(&(data[i].y)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2*)(&(data[i].z)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2*)(&(data[i].w)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
    }
  }
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};

/******************* vec_t<half> *******************/

// half x 1
template <>
struct vec_t<half, 1> {
  half data;

  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val);
  FLASHINFER_INLINE void load(const half* ptr);
  FLASHINFER_INLINE void store(half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(half* dst, const half* src);
};

FLASHINFER_INLINE void vec_t<half, 1>::fill(half val) { data = val; }

FLASHINFER_INLINE void vec_t<half, 1>::load(const half* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<half, 1>::store(half* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<half, 1>::memcpy(half* dst, const half* src) { *dst = *src; }

// half x 2
template <>
struct vec_t<half, 2> {
  half2 data;

  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val);
  FLASHINFER_INLINE void load(const half* ptr);
  FLASHINFER_INLINE void store(half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(half* dst, const half* src);
};

FLASHINFER_INLINE void vec_t<half, 2>::fill(half val) { data = make_half2(val, val); }

FLASHINFER_INLINE void vec_t<half, 2>::load(const half* ptr) { data = *((half2*)ptr); }

FLASHINFER_INLINE void vec_t<half, 2>::store(half* ptr) const { *((half2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<half, 2>::memcpy(half* dst, const half* src) {
  *((half2*)dst) = *((half2*)src);
}

// half x 4

template <>
struct vec_t<half, 4> {
  uint2 data;

  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val);
  FLASHINFER_INLINE void load(const half* ptr);
  FLASHINFER_INLINE void store(half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(half* dst, const half* src);
};

FLASHINFER_INLINE void vec_t<half, 4>::fill(half val) {
  *(half2*)(&data.x) = make_half2(val, val);
  *(half2*)(&data.y) = make_half2(val, val);
}

FLASHINFER_INLINE void vec_t<half, 4>::load(const half* ptr) { data = *((uint2*)ptr); }

FLASHINFER_INLINE void vec_t<half, 4>::store(half* ptr) const { *((uint2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<half, 4>::memcpy(half* dst, const half* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// half x 8 or more

template <size_t vec_size>
struct vec_t<half, vec_size> {
  uint4 data[vec_size / 8];
  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)data)[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)data)[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(half2*)(&(data[i].x)) = make_half2(val, val);
      *(half2*)(&(data[i].y)) = make_half2(val, val);
      *(half2*)(&(data[i].z)) = make_half2(val, val);
      *(half2*)(&(data[i].w)) = make_half2(val, val);
    }
  }
  FLASHINFER_INLINE void load(const half* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(half* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(half* dst, const half* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};

/******************* vec_t<nv_bfloat16> *******************/

// nv_bfloat16 x 1
template <>
struct vec_t<nv_bfloat16, 1> {
  nv_bfloat16 data;
  FLASHINFER_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  FLASHINFER_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(nv_bfloat16 val);
  FLASHINFER_INLINE void load(const nv_bfloat16* ptr);
  FLASHINFER_INLINE void store(nv_bfloat16* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

FLASHINFER_INLINE void vec_t<nv_bfloat16, 1>::fill(nv_bfloat16 val) { data = val; }

FLASHINFER_INLINE void vec_t<nv_bfloat16, 1>::load(const nv_bfloat16* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<nv_bfloat16, 1>::store(nv_bfloat16* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<nv_bfloat16, 1>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *dst = *src;
}

// nv_bfloat16 x 2
template <>
struct vec_t<nv_bfloat16, 2> {
  nv_bfloat162 data;

  FLASHINFER_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  FLASHINFER_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(nv_bfloat16 val);
  FLASHINFER_INLINE void load(const nv_bfloat16* ptr);
  FLASHINFER_INLINE void store(nv_bfloat16* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

FLASHINFER_INLINE void vec_t<nv_bfloat16, 2>::fill(nv_bfloat16 val) {
  data = make_bfloat162(val, val);
}

FLASHINFER_INLINE void vec_t<nv_bfloat16, 2>::load(const nv_bfloat16* ptr) {
  data = *((nv_bfloat162*)ptr);
}

FLASHINFER_INLINE void vec_t<nv_bfloat16, 2>::store(nv_bfloat16* ptr) const {
  *((nv_bfloat162*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<nv_bfloat16, 2>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *((nv_bfloat162*)dst) = *((nv_bfloat162*)src);
}

// nv_bfloat16 x 4

template <>
struct vec_t<nv_bfloat16, 4> {
  uint2 data;

  FLASHINFER_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  FLASHINFER_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(nv_bfloat16 val);
  FLASHINFER_INLINE void load(const nv_bfloat16* ptr);
  FLASHINFER_INLINE void store(nv_bfloat16* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

FLASHINFER_INLINE void vec_t<nv_bfloat16, 4>::fill(nv_bfloat16 val) {
  *(nv_bfloat162*)(&data.x) = make_bfloat162(val, val);
  *(nv_bfloat162*)(&data.y) = make_bfloat162(val, val);
}

FLASHINFER_INLINE void vec_t<nv_bfloat16, 4>::load(const nv_bfloat16* ptr) {
  data = *((uint2*)ptr);
}

FLASHINFER_INLINE void vec_t<nv_bfloat16, 4>::store(nv_bfloat16* ptr) const {
  *((uint2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<nv_bfloat16, 4>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// nv_bfloat16 x 8 or more

template <size_t vec_size>
struct vec_t<nv_bfloat16, vec_size> {
  uint4 data[vec_size / 8];

  FLASHINFER_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)data)[i]; }
  FLASHINFER_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)data)[i];
  }
  FLASHINFER_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(nv_bfloat16 val) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(nv_bfloat162*)(&(data[i].x)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].y)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].z)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].w)) = make_bfloat162(val, val);
    }
  }
  FLASHINFER_INLINE void load(const nv_bfloat16* ptr) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(nv_bfloat16* ptr) const {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};

/******************* vec_t<float> *******************/

// float x 1

template <>
struct vec_t<float, 1> {
  float data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val);
  FLASHINFER_INLINE void load(const float* ptr);
  FLASHINFER_INLINE void store(float* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src);
};

FLASHINFER_INLINE void vec_t<float, 1>::fill(float val) { data = val; }

FLASHINFER_INLINE void vec_t<float, 1>::load(const float* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<float, 1>::store(float* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<float, 1>::memcpy(float* dst, const float* src) { *dst = *src; }

// float x 2

template <>
struct vec_t<float, 2> {
  float2 data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val);
  FLASHINFER_INLINE void load(const float* ptr);
  FLASHINFER_INLINE void store(float* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src);
};

FLASHINFER_INLINE void vec_t<float, 2>::fill(float val) { data = make_float2(val, val); }

FLASHINFER_INLINE void vec_t<float, 2>::load(const float* ptr) { data = *((float2*)ptr); }

FLASHINFER_INLINE void vec_t<float, 2>::store(float* ptr) const { *((float2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<float, 2>::memcpy(float* dst, const float* src) {
  *((float2*)dst) = *((float2*)src);
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  float4 data[vec_size / 4];

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  FLASHINFER_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)dst)[i] = ((float4*)src)[i];
    }
  }
};

}  // namespace flashinfer

#endif  // VEC_DTYPES_CUH_
