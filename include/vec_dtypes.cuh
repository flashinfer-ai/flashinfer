#ifndef VEC_DTYPES_CUH_
#define VEC_DTYPES_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
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
};

// half x 1
template <>
struct vec_t<half, 1> {
  half data;

  TVM_XINLINE half &operator[](size_t i) { return ((half *)(&data))[i]; }
  TVM_XINLINE void fill(half val);
  TVM_XINLINE void load(const half *ptr);
  TVM_XINLINE void store(half *ptr);
};

TVM_XINLINE void vec_t<half, 1>::fill(half val) { data = val; }

TVM_XINLINE void vec_t<half, 1>::load(const half *ptr) { data = *ptr; }

TVM_XINLINE void vec_t<half, 1>::store(half *ptr) { *ptr = data; }

// half x 2
template <>
struct vec_t<half, 2> {
  half2 data;

  TVM_XINLINE half &operator[](size_t i) { return ((half *)(&data))[i]; }
  TVM_XINLINE void fill(half val);
  TVM_XINLINE void load(const half *ptr);
  TVM_XINLINE void store(half *ptr);
};

TVM_XINLINE void vec_t<half, 2>::fill(half val) { data = make_half2(val, val); }

TVM_XINLINE void vec_t<half, 2>::load(const half *ptr) { data = *((half2 *)ptr); }

TVM_XINLINE void vec_t<half, 2>::store(half *ptr) { *((half2 *)ptr) = data; }

// half x 4

template <>
struct vec_t<half, 4> {
  uint2 data;

  TVM_XINLINE half &operator[](size_t i) { return ((half *)(&data))[i]; }
  TVM_XINLINE void fill(half val);
  TVM_XINLINE void load(const half *ptr);
  TVM_XINLINE void store(half *ptr);
};

TVM_XINLINE void vec_t<half, 4>::fill(half val) {
  *(half2 *)(&data.x) = make_half2(val, val);
  *(half2 *)(&data.y) = make_half2(val, val);
}

TVM_XINLINE void vec_t<half, 4>::load(const half *ptr) { data = *((uint2 *)ptr); }

TVM_XINLINE void vec_t<half, 4>::store(half *ptr) { *((uint2 *)ptr) = data; }

// half x 8

template <>
struct vec_t<half, 8> {
  uint4 data;

  TVM_XINLINE half &operator[](size_t i) { return ((half *)(&data))[i]; }
  TVM_XINLINE void fill(half val);
  TVM_XINLINE void load(const half *ptr);
  TVM_XINLINE void store(half *ptr);
};

TVM_XINLINE void vec_t<half, 8>::fill(half val) {
  *(half2 *)(&data.x) = make_half2(val, val);
  *(half2 *)(&data.y) = make_half2(val, val);
  *(half2 *)(&data.z) = make_half2(val, val);
  *(half2 *)(&data.w) = make_half2(val, val);
}

TVM_XINLINE void vec_t<half, 8>::load(const half *ptr) { data = *((uint4 *)ptr); }

TVM_XINLINE void vec_t<half, 8>::store(half *ptr) { *((uint4 *)ptr) = data; }

// float x 1

template <>
struct vec_t<float, 1> {
  float data;

  TVM_XINLINE float &operator[](size_t i) { return ((float *)(&data))[i]; }
  TVM_XINLINE void fill(float val);
  TVM_XINLINE void load(const float *ptr);
  TVM_XINLINE void store(float *ptr);
};

TVM_XINLINE void vec_t<float, 1>::fill(float val) { data = val; }

TVM_XINLINE void vec_t<float, 1>::load(const float *ptr) { data = *ptr; }

TVM_XINLINE void vec_t<float, 1>::store(float *ptr) { *ptr = data; }

// float x 2

template <>
struct vec_t<float, 2> {
  float2 data;

  TVM_XINLINE float &operator[](size_t i) { return ((float *)(&data))[i]; }
  TVM_XINLINE void fill(float val);
  TVM_XINLINE void load(const float *ptr);
  TVM_XINLINE void store(float *ptr);
};

TVM_XINLINE void vec_t<float, 2>::fill(float val) { data = make_float2(val, val); }

TVM_XINLINE void vec_t<float, 2>::load(const float *ptr) { data = *((float2 *)ptr); }

TVM_XINLINE void vec_t<float, 2>::store(float *ptr) { *((float2 *)ptr) = data; }

// float x 4

template <>
struct vec_t<float, 4> {
  float4 data;

  TVM_XINLINE float &operator[](size_t i) { return ((float *)(&data))[i]; }
  TVM_XINLINE void fill(float val);
  TVM_XINLINE void load(const float *ptr);
  TVM_XINLINE void store(float *ptr);
};

TVM_XINLINE void vec_t<float, 4>::fill(float val) { data = make_float4(val, val, val, val); }

TVM_XINLINE void vec_t<float, 4>::load(const float *ptr) { data = *((float4 *)ptr); }

TVM_XINLINE void vec_t<float, 4>::store(float *ptr) { *((float4 *)ptr) = data; }

template <>
struct vec_t<float, 8> {
  ulonglong4 data;

  TVM_XINLINE float &operator[](size_t i) { return ((float *)(&data))[i]; }
  TVM_XINLINE void fill(float val);
  TVM_XINLINE void load(const float *ptr);
  TVM_XINLINE void store(float *ptr);
};

TVM_XINLINE void vec_t<float, 8>::fill(float val) {
  *(float2 *)(&data.x) = make_float2(val, val);
  *(float2 *)(&data.y) = make_float2(val, val);
  *(float2 *)(&data.z) = make_float2(val, val);
  *(float2 *)(&data.w) = make_float2(val, val);
}

TVM_XINLINE void vec_t<float, 8>::load(const float *ptr) { data = *((ulonglong4 *)ptr); }

TVM_XINLINE void vec_t<float, 8>::store(float *ptr) { *((ulonglong4 *)ptr) = data; }

}  // namespace flashinfer

#endif  // VEC_DTYPES_CUH_