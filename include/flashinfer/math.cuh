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
#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

#ifdef USE_ROCM

#include <hip/hip_runtime.h>
// TODO (yiakwy) : functions not included
#include <hip/amd_detail/amd_warp_sync_functions.h>
#include "flashinfer/hip_warp_sync_functions.h"
#include "flashinfer/hip_cuda_type_utils.h"

// CUDA API Portable interfaces
#include "flashinfer/hip_defs.h"

#else

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#endif // USE_ROCM-1

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }


#ifdef USE_ROCM

#include <hip/hip_fp16.h>

namespace amdgpu {

// ROCM exp c primitive, which computes 2^x in fp8/fp16/bf16/fp32
template<typename T>
__forceinline__ __device__ T exp2(T);

template<typename T>
__forceinline__ __device__ T log2(T);

template<typename T>
__forceinline__ __device__ T rcp(T);

template<typename T>
__forceinline__ __device__ T shfl_xor_sync(T, int);

template<typename T>
__forceinline__ __device__ T rsqrt(T);

template<typename T>
__forceinline__ __device__ T tanh(T);

// sepicalization

// TODO (yiakwy) : add equivalent asm version for fast exp computation (polynomial approx)
template<>
inline __device__ float exp2(float x) {
  return exp2f(x);
}

template<>
inline __device__ half exp2(half x) {
  return hexp2(x);
}

template<>
inline __device__ half2 exp2(half2 x) {
  return h2exp2(x);
}

template<>
__forceinline__ __device__ float log2(float x) {
  return log2f(x);
}

template<>
inline __device__ half log2(half x) {
  return hlog2(x);
}

template<>
__forceinline__ __device__ float rcp(float x) {
  // TODO (yiakwy) : __frcp_rn is not supported in ROCM 6.2
  // TODO (yiakwy) : accelerate __frcp_rn for float input with fast rcp algorithm
  // return __frcp_rn(x);
  return 1.f / x;
}

// TODO (yiakwy) : verify; see details from here : https://rocm.docs.amd.com/projects/HIP/en/develop/reference/kernel_language.html
template<>
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  // note AMD uses 8 byte mask (i.e. long datatype) to allow all 64 threads participate in
  // TODO (yiakwy) : this does not work
  // return __shfl_xor_sync(0xffffffffffffffff, x, lane_mask);
  // TODO (yiakwy) : workaround
  return __shfl_xor(x, lane_mask);
}

template<>
__forceinline__ __device__ half shfl_xor_sync(half x, int lane_mask) {
  // note AMD uses 8 byte mask (i.e. long datatype)
  // TODO (yiakwy) : this does not work
  // return __shfl_xor_sync(0xffffffffffffffff, x, lane_mask);
  // TODO (yiakwy) : workaround
  return __shfl_xor(x, lane_mask);
}

template<>
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  // note AMD uses 8 byte mask (i.e. long datatype)
  // TODO (yiakwy) : this does not work
  // return __shfl_xor_sync(0xffffffffffffffff, x, lane_mask);
  // TODO (yiakwy) : workaround
  return __shfl_xor(x, lane_mask);
}

template<>
__forceinline__ __device__ float rsqrt(float x) {
  return rsqrtf(x);
}

template<>
__forceinline__ __device__ float tanh(float x) {
  return tanhf(x);
}

template<>
__forceinline__ __device__ half tanh(half x) {
  // TODO (yiakwy) : SDK 6.2 does not define htanh
  /*
  return htanh(x);
  */
  // TODO (yiakwy) : optimize this with fast polynomial fitting
  half a = hexp(x);
  half b = hexp(-x);
  return (a - b) / (a + b);
}

template<>
__forceinline__ __device__ half2 tanh(half2 x) {
  // TODO (yiakwy) : SDK 6.2 does not define h2tanh
  /*
  return h2tanh(x);
  */
  return half2{tanh(x.x), tanh(x.y)};
}

} // amdgpu

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  return amdgpu::exp2(x);
}

__forceinline__ __device__ half ptx_exp2(half x) {
  return amdgpu::exp2(x);
}

__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  return amdgpu::exp2(x);
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  return amdgpu::log2(x);
}


/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  return amdgpu::rcp(x);
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  return amdgpu::shfl_xor_sync(x, lane_mask);
}

__forceinline__ __device__ half shfl_xor_sync(half x, int lane_mask) {
  return amdgpu::shfl_xor_sync(x, lane_mask);
}

__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return amdgpu::shfl_xor_sync(x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  return amdgpu::rsqrt(x);
}

__forceinline__ __device__ float tanh(float x) {
  return amdgpu::tanh(x);
}

__forceinline__ __device__ half tanh(half x) {
  return amdgpu::tanh(x);
}

__forceinline__ __device__ half2 tanh(half2 x) {
  return amdgpu::tanh(x);
}

#else

// NVIDIA PTX exlusive codes

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return __shfl_xor_sync(0xffffffff, x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

#endif // USE_ROCM-2

}  // namespace math
}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CUH_
