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

// FP4 KV cache quantization kernels with linear (non-swizzled) block scale layout.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_fp8.h>
typedef __nv_fp8_e4m3 fp8_e4m3;
#define HAS_FP8_SUPPORT 1
#else
typedef uint8_t fp8_e4m3;
#define HAS_FP8_SUPPORT 0
#endif

#include "tvm_ffi_utils.h"

// Helper functions
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ __nv_bfloat162 cuda_abs(__nv_bfloat162 a) {
  __nv_bfloat162 result;
  float fx = fabsf(__bfloat162float(a.x));
  float fy = fabsf(__bfloat162float(a.y));
  result.x = __float2bfloat16(fx);
  result.y = __float2bfloat16(fy);
  return result;
}

__device__ __forceinline__ half2 cuda_abs(half2 a) { return __habs2(a); }

__device__ __forceinline__ __nv_bfloat162 cuda_max(__nv_bfloat162 a, __nv_bfloat162 b) {
  __nv_bfloat162 result;
  result.x = __bfloat162float(a.x) > __bfloat162float(b.x) ? a.x : b.x;
  result.y = __bfloat162float(a.y) > __bfloat162float(b.y) ? a.y : b.y;
  return result;
}

__device__ __forceinline__ half2 cuda_max(half2 a, half2 b) { return __hmax2(a, b); }

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
  return 0;
#endif
}

// Quantize 8 FP16/BF16 values to E2M1 with FP8 E4M3 block scaling
template <typename InType>
__device__ uint32_t quantize_fp16_to_e2m1_with_scaling(InType (&vec)[4], float global_scale,
                                                       uint8_t* block_scale_out) {
  constexpr int SF_VEC_SIZE = 16;
  constexpr int CVT_ELTS_PER_THREAD = 8;
  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;

  auto localMax = cuda_abs(vec[0]);

#pragma unroll
  for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++) {
    localMax = cuda_max(localMax, cuda_abs(vec[i]));
  }

  localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }

  float vecMax;
  if constexpr (std::is_same_v<InType, __nv_bfloat162>) {
    auto max_single =
        __bfloat162float(localMax.x) > __bfloat162float(localMax.y) ? localMax.x : localMax.y;
    vecMax = __bfloat162float(max_single);
  } else {
    vecMax = fmaxf(__half2float(localMax.x), __half2float(localMax.y));
  }

  uint8_t fp8_scale_val = 0;
  float output_scale = 0.0f;

  auto sf_value =
      reciprocal_approximate_ftz(global_scale) * (vecMax * reciprocal_approximate_ftz(6.0f));

#if HAS_FP8_SUPPORT
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf_value);
  fp8_scale_val = tmp.__x;
  sf_value = static_cast<float>(tmp);
#else
  fp8_scale_val = static_cast<uint8_t>(fminf(fmaxf(sf_value, 0.0f), 255.0f));
  sf_value = static_cast<float>(fp8_scale_val);
#endif

  output_scale = vecMax != 0 ? reciprocal_approximate_ftz(sf_value * global_scale) : 0.0f;

  if (block_scale_out) {
    *block_scale_out = fp8_scale_val;
  }

  float2 fp2_vals[CVT_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<InType, __nv_bfloat162>) {
      fp2_vals[i] = __bfloat1622float2(vec[i]);
    } else {
      fp2_vals[i] = __half22float2(vec[i]);
    }
    fp2_vals[i].x *= output_scale;
    fp2_vals[i].y *= output_scale;
  }

  uint32_t e2m1_vec = fp32_vec_to_e2m1(fp2_vals);

  return e2m1_vec;
}

// Quantization kernel for BF16 to NVFP4
template <int BLOCK_SIZE = 128, int ELTS_PER_THREAD = 16>
__global__ void nvfp4_quant_from_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                             const float* __restrict__ global_scale_ptr,
                                             uint8_t* __restrict__ fp4_output,
                                             uint8_t* __restrict__ block_scales, const int M,
                                             const int K) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= M) return;

  __shared__ float global_scale;

  if (tid == 0) {
    global_scale = *global_scale_ptr;
  }
  __syncthreads();

  constexpr int CVT_ELTS_PER_THREAD = 8;
  constexpr int PACKED_PER_THREAD = CVT_ELTS_PER_THREAD / 2;
  const int elts_per_block = BLOCK_SIZE * CVT_ELTS_PER_THREAD;

  const __nv_bfloat16* row_input = input + row * K;
  uint8_t* row_fp4 = fp4_output + row * (K / 2);
  uint8_t* row_scales = block_scales + row * (K / 16);

  for (int base_col = 0; base_col < K; base_col += elts_per_block) {
    const int col_start = base_col + tid * CVT_ELTS_PER_THREAD;

    if (col_start >= K) break;

    __nv_bfloat162 vec[4];

#pragma unroll
    for (int i = 0; i < PACKED_PER_THREAD; ++i) {
      const int col = col_start + i * 2;
      if (col + 1 < K) {
        vec[i] = *reinterpret_cast<const __nv_bfloat162*>(&row_input[col]);
      } else if (col < K) {
        vec[i].x = row_input[col];
        vec[i].y = __float2bfloat16(0.0f);
      } else {
        vec[i] = __float2bfloat162_rn(0.0f);
      }
    }

    const int block_idx = col_start / 16;
    uint8_t* scale_out = (tid % 2 == 0) ? &row_scales[block_idx] : nullptr;

    uint32_t e2m1_vals = quantize_fp16_to_e2m1_with_scaling(vec, global_scale, scale_out);

    const int packed_idx = col_start / 2;
    if (packed_idx + 3 < K / 2) {
      *reinterpret_cast<uint32_t*>(&row_fp4[packed_idx]) = e2m1_vals;
    } else {
      uint8_t* bytes = reinterpret_cast<uint8_t*>(&e2m1_vals);
      for (int i = 0; i < 4 && packed_idx + i < K / 2; ++i) {
        row_fp4[packed_idx + i] = bytes[i];
      }
    }
  }
}

// Quantization kernel for FP16 to NVFP4
template <int BLOCK_SIZE = 128, int ELTS_PER_THREAD = 16>
__global__ void nvfp4_quant_from_fp16_kernel(const half* __restrict__ input,
                                             const float* __restrict__ global_scale_ptr,
                                             uint8_t* __restrict__ fp4_output,
                                             uint8_t* __restrict__ block_scales, const int M,
                                             const int K) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= M) return;

  __shared__ float global_scale;

  if (tid == 0) {
    global_scale = *global_scale_ptr;
  }
  __syncthreads();

  constexpr int CVT_ELTS_PER_THREAD = 8;
  constexpr int PACKED_PER_THREAD = CVT_ELTS_PER_THREAD / 2;
  const int elts_per_block = BLOCK_SIZE * CVT_ELTS_PER_THREAD;

  const half* row_input = input + row * K;
  uint8_t* row_fp4 = fp4_output + row * (K / 2);
  uint8_t* row_scales = block_scales + row * (K / 16);

  for (int base_col = 0; base_col < K; base_col += elts_per_block) {
    const int col_start = base_col + tid * CVT_ELTS_PER_THREAD;

    if (col_start >= K) break;

    half2 vec[4];

#pragma unroll
    for (int i = 0; i < PACKED_PER_THREAD; ++i) {
      const int col = col_start + i * 2;
      if (col + 1 < K) {
        vec[i] = *reinterpret_cast<const half2*>(&row_input[col]);
      } else if (col < K) {
        vec[i].x = row_input[col];
        vec[i].y = __float2half(0.0f);
      } else {
        vec[i] = __float2half2_rn(0.0f);
      }
    }

    const int block_idx = col_start / 16;
    uint8_t* scale_out = (tid % 2 == 0) ? &row_scales[block_idx] : nullptr;

    uint32_t e2m1_vals = quantize_fp16_to_e2m1_with_scaling(vec, global_scale, scale_out);

    const int packed_idx = col_start / 2;
    if (packed_idx + 3 < K / 2) {
      *reinterpret_cast<uint32_t*>(&row_fp4[packed_idx]) = e2m1_vals;
    } else {
      uint8_t* bytes = reinterpret_cast<uint8_t*>(&e2m1_vals);
      for (int i = 0; i < 4 && packed_idx + i < K / 2; ++i) {
        row_fp4[packed_idx + i] = bytes[i];
      }
    }
  }
}

void nvfp4_kv_quant(TensorView input, TensorView global_scale, TensorView fp4_output,
                    TensorView block_scales) {
  CHECK_INPUT(input);
  CHECK_CUDA(global_scale);
  CHECK_INPUT(fp4_output);
  CHECK_INPUT(block_scales);

  const int M = input.size(0);
  const int K = input.size(1);

  TVM_FFI_ICHECK(input.ndim() == 2) << "input must be 2D";
  TVM_FFI_ICHECK(K % 16 == 0) << "K dimension must be divisible by 16";
  TVM_FFI_ICHECK(fp4_output.ndim() == 2) << "fp4_output must be 2D";
  TVM_FFI_ICHECK(fp4_output.size(0) == M) << "fp4_output row count mismatch";
  TVM_FFI_ICHECK(fp4_output.size(1) == K / 2) << "fp4_output column count mismatch";
  TVM_FFI_ICHECK(block_scales.ndim() == 2) << "block_scales must be 2D";
  TVM_FFI_ICHECK(block_scales.size(0) == M) << "block_scales row count mismatch";
  TVM_FFI_ICHECK(block_scales.size(1) == K / 16) << "block_scales column count mismatch";
  TVM_FFI_ICHECK(global_scale.device().device_id == input.device().device_id)
      << "global_scale must be on the same device as input";
  TVM_FFI_ICHECK(fp4_output.device().device_id == input.device().device_id)
      << "fp4_output must be on the same device as input";
  TVM_FFI_ICHECK(block_scales.device().device_id == input.device().device_id)
      << "block_scales must be on the same device as input";

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  cudaStream_t stream = get_stream(input.device());

  const float* scale_ptr = static_cast<const float*>(global_scale.data_ptr());

  constexpr int BLOCK_SIZE = 128;
  dim3 grid(M);
  dim3 block(BLOCK_SIZE);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    if constexpr (std::is_same_v<c_type, nv_bfloat16>) {
      nvfp4_quant_from_bf16_kernel<BLOCK_SIZE, 16>
          <<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(input.data_ptr()),
                                       scale_ptr, static_cast<uint8_t*>(fp4_output.data_ptr()),
                                       static_cast<uint8_t*>(block_scales.data_ptr()), M, K);
    } else {
      nvfp4_quant_from_fp16_kernel<BLOCK_SIZE, 16>
          <<<grid, block, 0, stream>>>(static_cast<const half*>(input.data_ptr()), scale_ptr,
                                       static_cast<uint8_t*>(fp4_output.data_ptr()),
                                       static_cast<uint8_t*>(block_scales.data_ptr()), M, K);
    }
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_kv_quant, nvfp4_kv_quant);
