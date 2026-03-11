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

// FP4 KV cache dequantization kernels with linear (non-swizzled) block scale layout.

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

// E2M1 lookup table
__device__ __constant__ float E2M1_LUT[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                               4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                               -2.0f, -3.0f, -4.0f, -6.0f};

// Dequantize 4 FP4 values (packed in uint16_t) to float2x2
__device__ __forceinline__ void dequant_fp4x4_to_float2x2(uint16_t packed_fp4, float scale_0,
                                                          float scale_1, float global_scale,
                                                          float2& out0, float2& out1) {
  uint8_t fp4_0 = (packed_fp4 >> 0) & 0xF;
  uint8_t fp4_1 = (packed_fp4 >> 4) & 0xF;
  uint8_t fp4_2 = (packed_fp4 >> 8) & 0xF;
  uint8_t fp4_3 = (packed_fp4 >> 12) & 0xF;

  out0.x = E2M1_LUT[fp4_0] * scale_0 * global_scale;
  out0.y = E2M1_LUT[fp4_1] * scale_0 * global_scale;
  out1.x = E2M1_LUT[fp4_2] * scale_1 * global_scale;
  out1.y = E2M1_LUT[fp4_3] * scale_1 * global_scale;
}

template <typename OutType, int BLOCK_SIZE = 128, int ELTS_PER_THREAD = 16>
__global__ void nvfp4_dequant_vectorized_kernel(const uint8_t* __restrict__ fp4_data,
                                                const uint8_t* __restrict__ block_scales,
                                                const float* __restrict__ global_scale_ptr,
                                                OutType* __restrict__ output, const int M,
                                                const int K) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= M) return;

  __shared__ float global_scale;
  __shared__ uint8_t smem_scales[512];

  if (tid == 0) {
    global_scale = *global_scale_ptr;
  }

  const int K_scales = K / 16;
  const int K_packed = K / 2;

  for (int i = tid; i < K_scales; i += BLOCK_SIZE) {
    smem_scales[i] = block_scales[row * K_scales + i];
  }
  __syncthreads();

  constexpr int PACKED_PER_THREAD = ELTS_PER_THREAD / 2;
  const int elts_per_block = BLOCK_SIZE * ELTS_PER_THREAD;

  const uint8_t* row_fp4 = fp4_data + row * K_packed;
  OutType* row_output = output + row * K;

  for (int base_col = 0; base_col < K; base_col += elts_per_block) {
    const int col_start = base_col + tid * ELTS_PER_THREAD;

    if (col_start >= K) break;

#pragma unroll
    for (int i = 0; i < PACKED_PER_THREAD / 2; ++i) {
      const int col = col_start + i * 4;
      if (col + 3 >= K) break;

      const int packed_idx = col / 2;
      uint16_t packed_fp4 = *reinterpret_cast<const uint16_t*>(&row_fp4[packed_idx]);

      const int scale_idx_0 = col / 16;
      const int scale_idx_1 = (col + 2) / 16;

      const uint8_t scale_fp8_0 = smem_scales[scale_idx_0];
      const uint8_t scale_fp8_1 = smem_scales[scale_idx_1];

#if HAS_FP8_SUPPORT
      const float scale_0 =
          static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&scale_fp8_0));
      const float scale_1 =
          static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&scale_fp8_1));
#else
      const float scale_0 = 1.0f;
      const float scale_1 = 1.0f;
#endif

      float2 out0, out1;
      dequant_fp4x4_to_float2x2(packed_fp4, scale_0, scale_1, global_scale, out0, out1);

      if constexpr (std::is_same_v<OutType, __nv_bfloat16>) {
        __nv_bfloat162 bf16_0 = __float22bfloat162_rn(out0);
        __nv_bfloat162 bf16_1 = __float22bfloat162_rn(out1);

        *reinterpret_cast<__nv_bfloat162*>(&row_output[col]) = bf16_0;
        *reinterpret_cast<__nv_bfloat162*>(&row_output[col + 2]) = bf16_1;
      } else if constexpr (std::is_same_v<OutType, half>) {
        half2 h2_0 = __float22half2_rn(out0);
        half2 h2_1 = __float22half2_rn(out1);

        *reinterpret_cast<half2*>(&row_output[col]) = h2_0;
        *reinterpret_cast<half2*>(&row_output[col + 2]) = h2_1;
      }
    }
  }
}

void nvfp4_kv_dequant(TensorView fp4_data, TensorView block_scales, TensorView global_scale,
                      TensorView output, bool scale_on_host) {
  CHECK_INPUT(fp4_data);
  CHECK_INPUT(block_scales);
  CHECK_INPUT(output);

  const int M = fp4_data.size(0);
  const int K = fp4_data.size(1) * 2;

  TVM_FFI_ICHECK(fp4_data.ndim() == 2) << "fp4_data must be 2D";
  TVM_FFI_ICHECK(block_scales.ndim() == 2) << "block_scales must be 2D";
  TVM_FFI_ICHECK(output.ndim() == 2) << "output must be 2D";
  TVM_FFI_ICHECK(output.size(0) == M) << "output row count mismatch";
  TVM_FFI_ICHECK(output.size(1) == K) << "output column count mismatch";
  TVM_FFI_ICHECK(block_scales.size(0) == M) << "block_scales row count mismatch";
  TVM_FFI_ICHECK(block_scales.size(1) == K / 16) << "block_scales column count mismatch";

  ffi::CUDADeviceGuard device_guard(fp4_data.device().device_id);
  cudaStream_t stream = get_stream(output.device());

  const float* scale_ptr;
  ffi::Tensor device_scale_buf;

  if (scale_on_host) {
    CHECK_CPU(global_scale);
    device_scale_buf = alloc_tensor({1}, dl_float32, fp4_data.device());
    cudaMemcpyAsync(device_scale_buf.data_ptr(), global_scale.data_ptr(),
                    sizeof(float), cudaMemcpyHostToDevice, stream);
    scale_ptr = static_cast<const float*>(device_scale_buf.data_ptr());
  } else {
    CHECK_CUDA(global_scale);
    scale_ptr = static_cast<const float*>(global_scale.data_ptr());
  }

  constexpr int BLOCK_SIZE = 128;
  dim3 grid(M);
  dim3 block(BLOCK_SIZE);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(output.dtype(), c_type, [&] {
    nvfp4_dequant_vectorized_kernel<c_type, BLOCK_SIZE, 16><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(fp4_data.data_ptr()),
        static_cast<const uint8_t*>(block_scales.data_ptr()),
        scale_ptr,
        static_cast<c_type*>(output.data_ptr()), M, K);
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_kv_dequant, nvfp4_kv_dequant);
