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
    globalScalePtr = static_cast<float*>(globalScale.value()->data);
  }

  auto const& inputShape = self.shape();
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

#define LAUNCH_FP4_QUANTIZE_KERNEL(T, SF_VEC_SIZE)                                               \
  tensorrt_llm::kernels::invokeFP4Quantization<T, SF_VEC_SIZE>(                                  \
      1, m, k, reinterpret_cast<T*>(self->data), globalScalePtr,                                 \
      reinterpret_cast<int64_t*>(valueE2M1->data), reinterpret_cast<int32_t*>(scaleFP8SF->data), \
      sfUseUE8M0, layout, mMultiProcessorCount, /*mask=*/nullptr, enable_pdl,                    \
      get_stream(self->device));

  if (sfUseUE8M0) {
    if (self->dtype == dl_float16) {
      LAUNCH_FP4_QUANTIZE_KERNEL(half, 32)
    } else if (self->dtype == dl_bfloat16) {
#ifdef ENABLE_BF16
      LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 32)
#else
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "BFloat16 must be enabled to quantize an bf16 tensor to fp4.";
#endif
    } else if (self->dtype == dl_float8_e4m3fn) {
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
    if (self->dtype == dl_float16) {
      LAUNCH_FP4_QUANTIZE_KERNEL(half, 16)
    } else if (self->dtype == dl_bfloat16) {
#ifdef ENABLE_BF16
      LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 16)
#else
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "BFloat16 must be enabled to quantize an bf16 tensor to fp4.";
#endif
    } else if (self->dtype == dl_float8_e4m3fn) {
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
void fp4_batched_quantize(TensorView self, Optional<TensorView> const& mask, TensorView globalScale,
                          TensorView valueE2M1, TensorView scaleFP8SF, int64_t sfVecSize,
                          bool sfUseUE8M0) {
  CHECK_CUDA(self);
  CHECK_CONTIGUOUS(self);
  auto fp32_dtype = DLDataType{kDLFloat, 32, 1};
  CHECK_INPUT_TYPE(globalScale, fp32_dtype);
  TVM_FFI_ICHECK_EQ(sfVecSize, 16) << "sfVecSize can only be 16";

  auto const& inputShape = self.shape();
  auto const& rank = inputShape.size();

  TVM_FFI_ICHECK_EQ(rank, 3) << "Input should be 3D tensor.";

  int64_t b = inputShape[0];
  int64_t m = inputShape[1];
  int64_t k = inputShape[2];

  TVM_FFI_ICHECK_EQ(k % sfVecSize, 0);
  bool use_mask = mask.has_value();
  if (use_mask) {
    auto const& mask_rank = mask.value().shape().size();
    TVM_FFI_ICHECK_EQ(mask_rank, 1) << "Mask should be 1D tensor.";
    TVM_FFI_ICHECK_EQ(mask.value().shape()[0], b);
  }

  std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
  outputShape[rank - 1] = k / 2;

  const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
  auto layout = tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4;

#define LAUNCH_FP4_QUANTIZE_KERNEL(T, SF_VEC_SIZE)                                               \
  tensorrt_llm::kernels::invokeFP4Quantization<T, SF_VEC_SIZE>(                                  \
      b, m, k, reinterpret_cast<T*>(self->data), static_cast<float*>(globalScale->data),         \
      reinterpret_cast<int64_t*>(valueE2M1->data), reinterpret_cast<int32_t*>(scaleFP8SF->data), \
      sfUseUE8M0, layout, mMultiProcessorCount,                                                  \
      use_mask ? static_cast<int32_t*>(mask.value()->data) : nullptr, /*enable_pdl=*/false,      \
      get_stream(self->device));

  if (self->dtype == dl_float16) {
    LAUNCH_FP4_QUANTIZE_KERNEL(half, 16)
  } else if (self->dtype == dl_bfloat16) {
#ifdef ENABLE_BF16
    LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 16)
#else
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "BFloat16 must be enabled to quantize an bf16 tensor to fp4.";
#endif
  } else if (self->dtype == dl_float8_e4m3fn) {
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

void silu_and_mul_nvfp4_batched_quantize(TensorView const& self, TensorView const& mask,
                                         TensorView const& globalScale, TensorView valueE2M1,
                                         TensorView scaleFP8SF, int64_t sfVecSize) {
  // TODO(shuw): mask can be none
  CHECK_CUDA(self);
  CHECK_CONTIGUOUS(self);
  auto fp32_dtype = DLDataType{kDLFloat, 32, 1};
  CHECK_INPUT_TYPE(globalScale, fp32_dtype);
  TVM_FFI_ICHECK_EQ(sfVecSize, 16) << "sfVecSize can only be 16";

  auto const& inputShape = self.shape();
  auto const& rank = inputShape.size();
  auto const& mask_rank = mask.shape().size();

  TVM_FFI_ICHECK_EQ(rank, 3) << "Input should be 3D tensor.";
  TVM_FFI_ICHECK_EQ(mask_rank, 1) << "Mask should be 1D tensor.";

  int64_t b = inputShape[0];
  int64_t m = inputShape[1];
  int64_t k_by_2 = inputShape[2];
  int64_t k = k_by_2 / 2;

  TVM_FFI_ICHECK_EQ(k % sfVecSize, 0);
  TVM_FFI_ICHECK_EQ(mask.shape()[0], b);

  std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
  outputShape[rank - 1] = k / 2;

  const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
  auto layout = tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4;

#define LAUNCH_SILU_AND_MUL_NVFP4_QUANTIZE_KERNEL(T, SF_VEC_SIZE)                             \
  tensorrt_llm::kernels::invokeSiluAndMulFP4Quantization<T, SF_VEC_SIZE>(                     \
      b, m, k_by_2, reinterpret_cast<T*>(self->data), static_cast<float*>(globalScale->data), \
      static_cast<int32_t*>(mask->data), reinterpret_cast<int64_t*>(valueE2M1->data),         \
      reinterpret_cast<int32_t*>(scaleFP8SF->data), layout, mMultiProcessorCount,             \
      get_stream(self->device));

  if (self->dtype == dl_float16) {
    LAUNCH_SILU_AND_MUL_NVFP4_QUANTIZE_KERNEL(half, 16)
  } else if (self->dtype == dl_bfloat16) {
#ifdef ENABLE_BF16
    LAUNCH_SILU_AND_MUL_NVFP4_QUANTIZE_KERNEL(__nv_bfloat16, 16)
#else
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "BFloat16 must be enabled to quantize an bf16 tensor to fp4.";
#endif
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "fp4_quantize only supports input tensor with dtypes fp16/bf16.";
  }

#undef LAUNCH_SILU_AND_MUL_NVFP4_QUANTIZE_KERNEL
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_quantize, fp4_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_batched_quantize, fp4_batched_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_and_mul_nvfp4_batched_quantize,
                              silu_and_mul_nvfp4_batched_quantize);
