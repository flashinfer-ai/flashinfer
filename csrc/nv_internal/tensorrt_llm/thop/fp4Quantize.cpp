/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/thop/fp4Quantize.h"

#include <cuda_fp16.h>

#include <cstdint>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/utils.h"

// self: [M, K], fp16/bf16/fp8_quantized
// globalScale: [1] float, = (448 * 6) / self.abs().max()
// nvfp4: sfVecSize = 16, sfUseUE8M0 = false
// mxfp4: sfVecSize = 32, sfUseUE8M0 = true
// alignment: sfVecSize
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in
// linear layout. See QuantizationSFLayout enum for more details about the two layouts. returns
// self_fp4, self_block_scale_factors self_fp4: [M, K / 2], FLOAT4_E2M1X2 self_block_scale_factors:
// ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
void fp4_quantize(TensorView self, Optional<TensorView> const& globalScale, TensorView valueE2M1,
                  TensorView scaleFP8SF, int64_t sfVecSize, bool sfUseUE8M0,
                  bool isSfSwizzledLayout, bool isSf8x4Layout, bool enable_pdl) {
  CHECK_CUDA(self);
  CHECK_CONTIGUOUS(self);
  if (sfUseUE8M0) {
    TVM_FFI_ICHECK_EQ(sfVecSize, 32) << "sfVecSize can only be 32, when sfUseUE8M0 is true";
  } else {
    TVM_FFI_ICHECK(globalScale.has_value()) << "globalScale is required when sfUseUE8M0 is false";
    // CHECK_INPUT_AND_TYPE(globalScale.value(), torch::kFloat32);
    TVM_FFI_ICHECK_EQ(sfVecSize, 16) << "sfVecSize can only be 16, when sfUseUE8M0 is false";
  }

  float* globalScalePtr{nullptr};
  if (globalScale.has_value()) {
    globalScalePtr = static_cast<float*>(globalScale.value().data_ptr());
  }

  auto const& inputShape = self.sizes();
  auto const& rank = inputShape.size();

  TVM_FFI_ICHECK_GE(rank, 2) << "Input should be >=2D tensor.";
  int64_t m = 1;
  for (size_t i = 0; i < rank - 1; i++) {
    m *= inputShape[i];
  }
  auto const k = inputShape[rank - 1];
  TVM_FFI_ICHECK_EQ(k % sfVecSize, 0);

  const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

  auto layout = tensorrt_llm::QuantizationSFLayout::LINEAR;
  layout = isSfSwizzledLayout ? (isSf8x4Layout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4
                                               : tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4)
                              : tensorrt_llm::QuantizationSFLayout::LINEAR;

#define LAUNCH_FP4_QUANTIZE_KERNEL(T, SF_VEC_SIZE)                                                 \
  tensorrt_llm::kernels::invokeFP4Quantization<T, SF_VEC_SIZE>(                                    \
      1, m, k, reinterpret_cast<T*>(self.data_ptr()), globalScalePtr,                              \
      reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                                            \
      reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), sfUseUE8M0, layout, mMultiProcessorCount, \
      enable_pdl, get_stream(self.device()));

  if (sfUseUE8M0) {
    if (self.dtype() == dl_float16) {
      LAUNCH_FP4_QUANTIZE_KERNEL(half, 32)
    } else if (self.dtype() == dl_bfloat16) {
#ifdef ENABLE_BF16
      LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 32)
#else
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "BFloat16 must be enabled to quantize an bf16 tensor to fp4.";
#endif
    } else if (self.dtype() == dl_float8_e4m3fn) {
#ifdef ENABLE_FP8
      LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3, 32)
#else
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "FP8 must be enabled to quantize an fp8 tensor to fp4.";
#endif
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3.";
    }
  } else {
    if (self.dtype() == dl_float16) {
      LAUNCH_FP4_QUANTIZE_KERNEL(half, 16)
    } else if (self.dtype() == dl_bfloat16) {
#ifdef ENABLE_BF16
      LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 16)
#else
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "BFloat16 must be enabled to quantize an bf16 tensor to fp4.";
#endif
    } else if (self.dtype() == dl_float8_e4m3fn) {
#ifdef ENABLE_FP8
      LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3, 16)
#else
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "FP8 must be enabled to quantize an fp8 tensor to fp4.";
#endif
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3.";
    }
  }

#undef LAUNCH_FP4_QUANTIZE_KERNEL
}

// self: [B, M, K], fp16/bf16/fp8_quantized
// globalScale: [1] float, = (448 * 6) / self.abs().max()
// nvfp4: sfVecSize = 16, sfUseUE8M0 = false
// mxfp4: sfVecSize = 32 (not supported yet), sfUseUE8M0 = true
// alignment: sfVecSize
// returns self_fp4, self_block_scale_factors
// self_fp4: [B, M, K / 2], FLOAT4_E2M1X2
// self_block_scale_factors:
//   [B, ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4], SF_DTYPE (UE4M3 or UE8M0)
void fp4_batched_quantize(Tensor self, Tensor globalScale, Tensor valueE2M1, Tensor scaleFP8SF,
                          int64_t sfVecSize, bool sfUseUE8M0) {
  CHECK_CUDA(self);
  CHECK_CONTIGUOUS(self);
  auto fp32_dtype = DLDataType{kDLFloat, 32, 1};
  CHECK_INPUT_TYPE(globalScale, fp32_dtype);
  TVM_FFI_ICHECK_EQ(sfVecSize, 16) << "sfVecSize can only be 16";

  auto const& inputShape = self.sizes();
  auto const& rank = inputShape.size();

  TVM_FFI_ICHECK_EQ(rank, 3) << "Input should be 3D tensor.";

  int64_t b = inputShape[0];
  int64_t m = inputShape[1];
  int64_t k = inputShape[2];

  TVM_FFI_ICHECK_EQ(k % sfVecSize, 0);

  std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
  outputShape[rank - 1] = k / 2;

  const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
  auto layout = tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4;

#define LAUNCH_FP4_QUANTIZE_KERNEL(T, SF_VEC_SIZE)                                                 \
  tensorrt_llm::kernels::invokeFP4Quantization<T, SF_VEC_SIZE>(                                    \
      b, m, k, reinterpret_cast<T*>(self.data_ptr()), static_cast<float*>(globalScale.data_ptr()), \
      reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                                            \
      reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), sfUseUE8M0, layout, mMultiProcessorCount, \
      /*enable_pdl=*/false, get_stream(self.device()));

  if (self.dtype() == dl_float16) {
    LAUNCH_FP4_QUANTIZE_KERNEL(half, 16)
  } else if (self.dtype() == dl_bfloat16) {
#ifdef ENABLE_BF16
    LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 16)
#else
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "BFloat16 must be enabled to quantize an bf16 tensor to fp4.";
#endif
  } else if (self.dtype() == dl_float8_e4m3fn) {
#ifdef ENABLE_FP8
    LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3, 16)
#else
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "FP8 must be enabled to quantize an fp8 tensor to fp4.";
#endif
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3.";
  }

#undef LAUNCH_FP4_QUANTIZE_KERNEL
}

void silu_and_mul_scaled_nvfp4_experts_quantize(Tensor output, Tensor output_scale,
                                                Tensor const input, Tensor const input_global_scale,
                                                Tensor const mask, bool use_silu_and_mul) {
  auto fp32_dtype = DLDataType{kDLFloat, 32, 1};
  auto int32_dtype = DLDataType{kDLInt, 32, 1};
  auto uint8_dtype = DLDataType{kDLUInt, 8, 1};
  CHECK_CUDA(output);
  CHECK_CUDA(output_scale);
  CHECK_CUDA(input);
  CHECK_CUDA(input_global_scale);
  CHECK_CUDA(mask);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(output_scale);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(input_global_scale);
  CHECK_CONTIGUOUS(mask);

  TVM_FFI_ICHECK_EQ(mask.ndim(), 1);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);
  TVM_FFI_ICHECK_EQ(output_scale.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input_global_scale.ndim(), 1);

  CHECK_INPUT_TYPE(input_global_scale, fp32_dtype);
  CHECK_INPUT_TYPE(mask, int32_dtype);
  // output is uint8 (two nvfp4 values are packed into one uint8)
  // output_scale is int32 (four fp8 values are packed into one int32)
  CHECK_INPUT_TYPE(output, uint8_dtype);
  CHECK_INPUT_TYPE(output_scale, int32_dtype);

  constexpr int BLOCK_SIZE = 16;
  auto m_topk = input.shape()[0];
  auto k_by_2 = input.shape()[1];
  auto k = k_by_2;
  if (use_silu_and_mul) {
    TVM_FFI_ICHECK_EQ(k_by_2 % 2, 0) << "k must be a multiple of 2";
    k = k_by_2 / 2;
  }
  TVM_FFI_ICHECK_EQ(k % BLOCK_SIZE, 0) << "k must be a multiple of 16";
  auto n_experts = input_global_scale.shape()[0];
  TVM_FFI_ICHECK_EQ(mask.shape()[0], n_experts);
  TVM_FFI_ICHECK_EQ(output.shape()[0], m_topk);
  TVM_FFI_ICHECK_EQ(output.shape()[1], k / 2);
  int scales_k = k / BLOCK_SIZE;
  // 4 means the swizzle requirement by nvidia nvfp4.
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4 means 4 fp8 values are packed into one int32
  TVM_FFI_ICHECK_EQ(output_scale.shape()[1] * 4, padded_k);

  auto in_dtype = input.dtype();
  const cudaStream_t stream = get_stream(input.device());
  if (in_dtype == dl_float16) {
    tensorrt_llm::kernels::invokeSiluAndMulNVFP4Quantization<half>(
        output.data_ptr(), output_scale.data_ptr(), input.data_ptr(), input_global_scale.data_ptr(),
        mask.data_ptr(), use_silu_and_mul, m_topk, k, n_experts, stream);
  } else if (in_dtype == dl_bfloat16) {
    tensorrt_llm::kernels::invokeSiluAndMulNVFP4Quantization<__nv_bfloat16>(
        output.data_ptr(), output_scale.data_ptr(), input.data_ptr(), input_global_scale.data_ptr(),
        mask.data_ptr(), use_silu_and_mul, m_topk, k, n_experts, stream);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "silu_and_mul_scaled_nvfp4_experts_quantize only "
                                                  "supports input tensor with dtypes fp16/bf16.";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_quantize, fp4_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_batched_quantize, fp4_batched_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_and_mul_scaled_nvfp4_experts_quantize,
                              silu_and_mul_scaled_nvfp4_experts_quantize);
