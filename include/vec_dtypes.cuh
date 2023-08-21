#ifndef VEC_DTYPES_CUH_
#define VEC_DTYPES_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifdef FLASHINFER_USE_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>

namespace flashinfer {

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__

template <typename float_t, size_t vec_size>
struct vec_t {
  TVM_XINLINE float_t &operator[](size_t i);
  TVM_XINLINE void fill(float_t val);
  TVM_XINLINE void load(const float_t *ptr);
  TVM_XINLINE void store(float_t *ptr);
  TVM_XINLINE static void memcpy(float_t *dst, const float_t *src);
};

#ifdef FLASHINFER_USE_FP8
/******************* vec_t<__nv_fp8_e4m3> *******************/

// __nv_fp8_e4m3 x 1
template <>
struct vec_t<__nv_fp8_e4m3, 1> {
  __nv_fp8_e4m3 data;

  TVM_XINLINE __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)(&data))[i]; }
  TVM_XINLINE void fill(__nv_fp8_e4m3 val);
  TVM_XINLINE void load(const __nv_fp8_e4m3 *ptr);
  TVM_XINLINE void store(__nv_fp8_e4m3 *ptr);
  TVM_XINLINE static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src);
};

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 1>::fill(__nv_fp8_e4m3 val) { data = val; }

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 1>::load(const __nv_fp8_e4m3 *ptr) { data = *ptr; }

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 1>::store(__nv_fp8_e4m3 *ptr) { *ptr = data; }

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 1>::memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
  *dst = *src;
}

// __nv_fp8_e4m3 x 2
template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  __nv_fp8x2_e4m3 data;

  TVM_XINLINE __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)(&data))[i]; }
  TVM_XINLINE void fill(__nv_fp8_e4m3 val);
  TVM_XINLINE void load(const __nv_fp8_e4m3 *ptr);
  TVM_XINLINE void store(__nv_fp8_e4m3 *ptr);
  TVM_XINLINE static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src);
};

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 2>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 2>::load(const __nv_fp8_e4m3 *ptr) {
  data = *((__nv_fp8x2_e4m3 *)ptr);
}

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 2>::store(__nv_fp8_e4m3 *ptr) {
  *((__nv_fp8x2_e4m3 *)ptr) = data;
}

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 2>::memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
  *((__nv_fp8x2_e4m3 *)dst) = *((__nv_fp8x2_e4m3 *)src);
}

// __nv_fp8_e4m3 x 4

template <>
struct vec_t<__nv_fp8_e4m3, 4> {
  __nv_fp8x4_e4m3 data;

  TVM_XINLINE __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)(&data))[i]; }
  TVM_XINLINE void fill(__nv_fp8_e4m3 val);
  TVM_XINLINE void load(const __nv_fp8_e4m3 *ptr);
  TVM_XINLINE void store(__nv_fp8_e4m3 *ptr);
  TVM_XINLINE static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src);
};

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 4>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 4>::load(const __nv_fp8_e4m3 *ptr) {
  data = *((__nv_fp8x4_e4m3 *)ptr);
}

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 4>::store(__nv_fp8_e4m3 *ptr) {
  *((__nv_fp8x4_e4m3 *)ptr) = data;
}

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 4>::memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
  *((__nv_fp8x4_e4m3 *)dst) = *((__nv_fp8x4_e4m3 *)src);
}

// __nv_fp8_e4m3 x 8

template <>
struct vec_t<__nv_fp8_e4m3, 8> {
  uint2 data;

  TVM_XINLINE __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)(&data))[i]; }
  TVM_XINLINE void fill(__nv_fp8_e4m3 val);
  TVM_XINLINE void load(const __nv_fp8_e4m3 *ptr);
  TVM_XINLINE void store(__nv_fp8_e4m3 *ptr);
  TVM_XINLINE static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src);
};

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 8>::fill(__nv_fp8_e4m3 val) {
  ((__nv_fp8x4_e4m3 *)(&data.x))->__x =
      (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
      (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
  ((__nv_fp8x4_e4m3 *)(&data.y))->__x =
      (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
      (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 8>::load(const __nv_fp8_e4m3 *ptr) { data = *((uint2 *)ptr); }

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 8>::store(__nv_fp8_e4m3 *ptr) { *((uint2 *)ptr) = data; }

TVM_XINLINE void vec_t<__nv_fp8_e4m3, 8>::memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
  *((__nv_fp8_e4m3 *)dst) = *((__nv_fp8_e4m3 *)src);
}

/******************* vec_t<__nv_fp8_e5m2> *******************/

// __nv_fp8_e5m2 x 1
template <>
struct vec_t<__nv_fp8_e5m2, 1> {
  __nv_fp8_e5m2 data;

  TVM_XINLINE __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)(&data))[i]; }
  TVM_XINLINE void fill(__nv_fp8_e5m2 val);
  TVM_XINLINE void load(const __nv_fp8_e5m2 *ptr);
  TVM_XINLINE void store(__nv_fp8_e5m2 *ptr);
  TVM_XINLINE static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src);
};

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 1>::fill(__nv_fp8_e5m2 val) { data = val; }

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 1>::load(const __nv_fp8_e5m2 *ptr) { data = *ptr; }

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 1>::store(__nv_fp8_e5m2 *ptr) { *ptr = data; }

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 1>::memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
  *dst = *src;
}

// __nv_fp8_e5m2 x 2
template <>
struct vec_t<__nv_fp8_e5m2, 2> {
  __nv_fp8x2_e5m2 data;

  TVM_XINLINE __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)(&data))[i]; }
  TVM_XINLINE void fill(__nv_fp8_e5m2 val);
  TVM_XINLINE void load(const __nv_fp8_e5m2 *ptr);
  TVM_XINLINE void store(__nv_fp8_e5m2 *ptr);
  TVM_XINLINE static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src);
};

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 2>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 2>::load(const __nv_fp8_e5m2 *ptr) {
  data = *((__nv_fp8x2_e5m2 *)ptr);
}

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 2>::store(__nv_fp8_e5m2 *ptr) {
  *((__nv_fp8x2_e5m2 *)ptr) = data;
}

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 2>::memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
  *((__nv_fp8x2_e5m2 *)dst) = *((__nv_fp8x2_e5m2 *)src);
}

// __nv_fp8_e5m2 x 4

template <>
struct vec_t<__nv_fp8_e5m2, 4> {
  __nv_fp8x4_e5m2 data;

  TVM_XINLINE __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)(&data))[i]; }
  TVM_XINLINE void fill(__nv_fp8_e5m2 val);
  TVM_XINLINE void load(const __nv_fp8_e5m2 *ptr);
  TVM_XINLINE void store(__nv_fp8_e5m2 *ptr);
  TVM_XINLINE static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src);
};

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 4>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 4>::load(const __nv_fp8_e5m2 *ptr) {
  data = *((__nv_fp8x4_e5m2 *)ptr);
}

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 4>::store(__nv_fp8_e5m2 *ptr) {
  *((__nv_fp8x4_e5m2 *)ptr) = data;
}

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 4>::memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
  *((__nv_fp8x4_e5m2 *)dst) = *((__nv_fp8x4_e5m2 *)src);
}

// __nv_fp8_e5m2 x 8

template <>
struct vec_t<__nv_fp8_e5m2, 8> {
  uint2 data;

  TVM_XINLINE __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)(&data))[i]; }
  TVM_XINLINE void fill(__nv_fp8_e5m2 val);
  TVM_XINLINE void load(const __nv_fp8_e5m2 *ptr);
  TVM_XINLINE void store(__nv_fp8_e5m2 *ptr);
  TVM_XINLINE static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src);
};

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 8>::fill(__nv_fp8_e5m2 val) {
  ((__nv_fp8x4_e5m2 *)(&data.x))->__x =
      (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
      (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
  ((__nv_fp8x4_e5m2 *)(&data.y))->__x =
      (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
      (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 8>::load(const __nv_fp8_e5m2 *ptr) { data = *((uint2 *)ptr); }

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 8>::store(__nv_fp8_e5m2 *ptr) { *((uint2 *)ptr) = data; }

TVM_XINLINE void vec_t<__nv_fp8_e5m2, 8>::memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
  *((__nv_fp8_e5m2 *)dst) = *((__nv_fp8_e5m2 *)src);
}

#endif

/******************* vec_t<half> *******************/

// half x 1
template <>
struct vec_t<half, 1> {
  half data;

  TVM_XINLINE half &operator[](size_t i) { return ((half *)(&data))[i]; }
  TVM_XINLINE void fill(half val);
  TVM_XINLINE void load(const half *ptr);
  TVM_XINLINE void store(half *ptr);
  TVM_XINLINE static void memcpy(half *dst, const half *src);
};

TVM_XINLINE void vec_t<half, 1>::fill(half val) { data = val; }

TVM_XINLINE void vec_t<half, 1>::load(const half *ptr) { data = *ptr; }

TVM_XINLINE void vec_t<half, 1>::store(half *ptr) { *ptr = data; }

TVM_XINLINE void vec_t<half, 1>::memcpy(half *dst, const half *src) { *dst = *src; }

// half x 2
template <>
struct vec_t<half, 2> {
  half2 data;

  TVM_XINLINE half &operator[](size_t i) { return ((half *)(&data))[i]; }
  TVM_XINLINE void fill(half val);
  TVM_XINLINE void load(const half *ptr);
  TVM_XINLINE void store(half *ptr);
  TVM_XINLINE static void memcpy(half *dst, const half *src);
};

TVM_XINLINE void vec_t<half, 2>::fill(half val) { data = make_half2(val, val); }

TVM_XINLINE void vec_t<half, 2>::load(const half *ptr) { data = *((half2 *)ptr); }

TVM_XINLINE void vec_t<half, 2>::store(half *ptr) { *((half2 *)ptr) = data; }

TVM_XINLINE void vec_t<half, 2>::memcpy(half *dst, const half *src) {
  *((half2 *)dst) = *((half2 *)src);
}

// half x 4

template <>
struct vec_t<half, 4> {
  uint2 data;

  TVM_XINLINE half &operator[](size_t i) { return ((half *)(&data))[i]; }
  TVM_XINLINE void fill(half val);
  TVM_XINLINE void load(const half *ptr);
  TVM_XINLINE void store(half *ptr);
  TVM_XINLINE static void memcpy(half *dst, const half *src);
};

TVM_XINLINE void vec_t<half, 4>::fill(half val) {
  *(half2 *)(&data.x) = make_half2(val, val);
  *(half2 *)(&data.y) = make_half2(val, val);
}

TVM_XINLINE void vec_t<half, 4>::load(const half *ptr) { data = *((uint2 *)ptr); }

TVM_XINLINE void vec_t<half, 4>::store(half *ptr) { *((uint2 *)ptr) = data; }

TVM_XINLINE void vec_t<half, 4>::memcpy(half *dst, const half *src) {
  *((uint2 *)dst) = *((uint2 *)src);
}

// half x 8

template <>
struct vec_t<half, 8> {
  uint4 data;

  TVM_XINLINE half &operator[](size_t i) { return ((half *)(&data))[i]; }
  TVM_XINLINE void fill(half val);
  TVM_XINLINE void load(const half *ptr);
  TVM_XINLINE void store(half *ptr);
  TVM_XINLINE static void memcpy(half *dst, const half *src);
};

TVM_XINLINE void vec_t<half, 8>::fill(half val) {
  *(half2 *)(&data.x) = make_half2(val, val);
  *(half2 *)(&data.y) = make_half2(val, val);
  *(half2 *)(&data.z) = make_half2(val, val);
  *(half2 *)(&data.w) = make_half2(val, val);
}

TVM_XINLINE void vec_t<half, 8>::load(const half *ptr) { data = *((uint4 *)ptr); }

TVM_XINLINE void vec_t<half, 8>::store(half *ptr) { *((uint4 *)ptr) = data; }

TVM_XINLINE void vec_t<half, 8>::memcpy(half *dst, const half *src) {
  *((uint4 *)dst) = *((uint4 *)src);
}

/******************* vec_t<nv_bfloat16> *******************/

// nv_bfloat16 x 1
template <>
struct vec_t<nv_bfloat16, 1> {
  nv_bfloat16 data;

  TVM_XINLINE nv_bfloat16 &operator[](size_t i) { return ((nv_bfloat16 *)(&data))[i]; }
  TVM_XINLINE void fill(nv_bfloat16 val);
  TVM_XINLINE void load(const nv_bfloat16 *ptr);
  TVM_XINLINE void store(nv_bfloat16 *ptr);
  TVM_XINLINE static void memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src);
};

TVM_XINLINE void vec_t<nv_bfloat16, 1>::fill(nv_bfloat16 val) { data = val; }

TVM_XINLINE void vec_t<nv_bfloat16, 1>::load(const nv_bfloat16 *ptr) { data = *ptr; }

TVM_XINLINE void vec_t<nv_bfloat16, 1>::store(nv_bfloat16 *ptr) { *ptr = data; }

TVM_XINLINE void vec_t<nv_bfloat16, 1>::memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src) {
  *dst = *src;
}

// nv_bfloat16 x 2
template <>
struct vec_t<nv_bfloat16, 2> {
  nv_bfloat162 data;

  TVM_XINLINE nv_bfloat16 &operator[](size_t i) { return ((nv_bfloat16 *)(&data))[i]; }
  TVM_XINLINE void fill(nv_bfloat16 val);
  TVM_XINLINE void load(const nv_bfloat16 *ptr);
  TVM_XINLINE void store(nv_bfloat16 *ptr);
  TVM_XINLINE static void memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src);
};

TVM_XINLINE void vec_t<nv_bfloat16, 2>::fill(nv_bfloat16 val) { data = make_bfloat162(val, val); }

TVM_XINLINE void vec_t<nv_bfloat16, 2>::load(const nv_bfloat16 *ptr) {
  data = *((nv_bfloat162 *)ptr);
}

TVM_XINLINE void vec_t<nv_bfloat16, 2>::store(nv_bfloat16 *ptr) { *((nv_bfloat162 *)ptr) = data; }

TVM_XINLINE void vec_t<nv_bfloat16, 2>::memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src) {
  *((nv_bfloat162 *)dst) = *((nv_bfloat162 *)src);
}

// nv_bfloat16 x 4

template <>
struct vec_t<nv_bfloat16, 4> {
  uint2 data;

  TVM_XINLINE nv_bfloat16 &operator[](size_t i) { return ((nv_bfloat16 *)(&data))[i]; }
  TVM_XINLINE void fill(nv_bfloat16 val);
  TVM_XINLINE void load(const nv_bfloat16 *ptr);
  TVM_XINLINE void store(nv_bfloat16 *ptr);
  TVM_XINLINE static void memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src);
};

TVM_XINLINE void vec_t<nv_bfloat16, 4>::fill(nv_bfloat16 val) {
  *(nv_bfloat162 *)(&data.x) = make_bfloat162(val, val);
  *(nv_bfloat162 *)(&data.y) = make_bfloat162(val, val);
}

TVM_XINLINE void vec_t<nv_bfloat16, 4>::load(const nv_bfloat16 *ptr) { data = *((uint2 *)ptr); }

TVM_XINLINE void vec_t<nv_bfloat16, 4>::store(nv_bfloat16 *ptr) { *((uint2 *)ptr) = data; }

TVM_XINLINE void vec_t<nv_bfloat16, 4>::memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src) {
  *((uint2 *)dst) = *((uint2 *)src);
}

// nv_bfloat16 x 8

template <>
struct vec_t<nv_bfloat16, 8> {
  uint4 data;

  TVM_XINLINE nv_bfloat16 &operator[](size_t i) { return ((nv_bfloat16 *)(&data))[i]; }
  TVM_XINLINE void fill(nv_bfloat16 val);
  TVM_XINLINE void load(const nv_bfloat16 *ptr);
  TVM_XINLINE void store(nv_bfloat16 *ptr);
  TVM_XINLINE static void memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src);
};

TVM_XINLINE void vec_t<nv_bfloat16, 8>::fill(nv_bfloat16 val) {
  *(nv_bfloat162 *)(&data.x) = make_bfloat162(val, val);
  *(nv_bfloat162 *)(&data.y) = make_bfloat162(val, val);
  *(nv_bfloat162 *)(&data.z) = make_bfloat162(val, val);
  *(nv_bfloat162 *)(&data.w) = make_bfloat162(val, val);
}

TVM_XINLINE void vec_t<nv_bfloat16, 8>::load(const nv_bfloat16 *ptr) { data = *((uint4 *)ptr); }

TVM_XINLINE void vec_t<nv_bfloat16, 8>::store(nv_bfloat16 *ptr) { *((uint4 *)ptr) = data; }

TVM_XINLINE void vec_t<nv_bfloat16, 8>::memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src) {
  *((uint4 *)dst) = *((uint4 *)src);
}

/******************* vec_t<float> *******************/

// float x 1

template <>
struct vec_t<float, 1> {
  float data;

  TVM_XINLINE float &operator[](size_t i) { return ((float *)(&data))[i]; }
  TVM_XINLINE void fill(float val);
  TVM_XINLINE void load(const float *ptr);
  TVM_XINLINE void store(float *ptr);
  TVM_XINLINE static void memcpy(float *dst, const float *src);
};

TVM_XINLINE void vec_t<float, 1>::fill(float val) { data = val; }

TVM_XINLINE void vec_t<float, 1>::load(const float *ptr) { data = *ptr; }

TVM_XINLINE void vec_t<float, 1>::store(float *ptr) { *ptr = data; }

TVM_XINLINE void vec_t<float, 1>::memcpy(float *dst, const float *src) { *dst = *src; }

// float x 2

template <>
struct vec_t<float, 2> {
  float2 data;

  TVM_XINLINE float &operator[](size_t i) { return ((float *)(&data))[i]; }
  TVM_XINLINE void fill(float val);
  TVM_XINLINE void load(const float *ptr);
  TVM_XINLINE void store(float *ptr);
  TVM_XINLINE static void memcpy(float *dst, const float *src);
};

TVM_XINLINE void vec_t<float, 2>::fill(float val) { data = make_float2(val, val); }

TVM_XINLINE void vec_t<float, 2>::load(const float *ptr) { data = *((float2 *)ptr); }

TVM_XINLINE void vec_t<float, 2>::store(float *ptr) { *((float2 *)ptr) = data; }

TVM_XINLINE void vec_t<float, 2>::memcpy(float *dst, const float *src) {
  *((float2 *)dst) = *((float2 *)src);
}

// float x 4

template <>
struct vec_t<float, 4> {
  float4 data;

  TVM_XINLINE float &operator[](size_t i) { return ((float *)(&data))[i]; }
  TVM_XINLINE void fill(float val);
  TVM_XINLINE void load(const float *ptr);
  TVM_XINLINE void store(float *ptr);
  TVM_XINLINE static void memcpy(float *dst, const float *src);
};

TVM_XINLINE void vec_t<float, 4>::fill(float val) { data = make_float4(val, val, val, val); }

TVM_XINLINE void vec_t<float, 4>::load(const float *ptr) { data = *((float4 *)ptr); }

TVM_XINLINE void vec_t<float, 4>::store(float *ptr) { *((float4 *)ptr) = data; }

TVM_XINLINE void vec_t<float, 4>::memcpy(float *dst, const float *src) {
  *((float4 *)dst) = *((float4 *)src);
}

template <>
struct vec_t<float, 8> {
  ulonglong4 data;

  TVM_XINLINE float &operator[](size_t i) { return ((float *)(&data))[i]; }
  TVM_XINLINE void fill(float val);
  TVM_XINLINE void load(const float *ptr);
  TVM_XINLINE void store(float *ptr);
  TVM_XINLINE static void memcpy(float *dst, const float *src);
};

TVM_XINLINE void vec_t<float, 8>::fill(float val) {
  *(float2 *)(&data.x) = make_float2(val, val);
  *(float2 *)(&data.y) = make_float2(val, val);
  *(float2 *)(&data.z) = make_float2(val, val);
  *(float2 *)(&data.w) = make_float2(val, val);
}

TVM_XINLINE void vec_t<float, 8>::load(const float *ptr) { data = *((ulonglong4 *)ptr); }

TVM_XINLINE void vec_t<float, 8>::store(float *ptr) { *((ulonglong4 *)ptr) = data; }

TVM_XINLINE void vec_t<float, 8>::memcpy(float *dst, const float *src) {
  *((ulonglong4 *)dst) = *((ulonglong4 *)src);
}

}  // namespace flashinfer

#endif  // VEC_DTYPES_CUH_