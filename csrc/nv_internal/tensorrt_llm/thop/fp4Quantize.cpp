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

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "pytorch_extension_utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace torch_ext {
// self: [M, K], fp16/bf16/fp8_quantized
// globalScale: [1] float, = (448 * 6) / self.abs().max()
// nvfp4: sfVecSize = 16, sfUseUE8M0 = false
// mxfp4: sfVecSize = 32 (not supported yet), sfUseUE8M0 = true
// alignment: sfVecSize
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in
// linear layout. See FP4QuantizationSFLayout enum for more details about the two layouts. returns
// self_fp4, self_block_scale_factors self_fp4: [M, K / 2], FLOAT4_E2M1X2 self_block_scale_factors:
// ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
std::tuple<at::Tensor, at::Tensor> fp4_quantize(at::Tensor const& self,
                                                at::Tensor const& globalScale, int64_t sfVecSize,
                                                bool sfUseUE8M0, bool isSfSwizzledLayout) {
  CHECK_TH_CUDA(self);
  CHECK_CONTIGUOUS(self);
  CHECK_INPUT_TYPE(globalScale, c10::ScalarType::Float);
  TORCH_CHECK(sfVecSize == 16, "sfVecSize can only be 16");

  auto const& inputShape = self.sizes();
  auto const& rank = inputShape.size();

  TORCH_CHECK(rank >= 2, "Input should be >=2D tensor.");
  int64_t m = 1;
  for (size_t i = 0; i < rank - 1; i++) {
    m *= inputShape[i];
  }
  auto const k = inputShape[rank - 1];
  TORCH_CHECK(k % sfVecSize == 0);

  std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
  outputShape[rank - 1] = k / 2;

  at::Tensor valueE2M1 =
      at::detail::empty_cuda(outputShape, FLOAT4_E2M1X2, self.device(), /* stride */ std::nullopt);

  int64_t SFSize = isSfSwizzledLayout
                       ? tensorrt_llm::computeFP4SwizzledLayoutSFSize(m, k / sfVecSize)
                       : tensorrt_llm::computeFP4LinearLayoutSFSize(m, k / sfVecSize);

  at::Tensor scaleFP8SF = at::detail::empty_cuda({SFSize}, SF_DTYPE, self.device(),
                                                 /* stride */ std::nullopt);  // 1D tensor

  const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

  auto const layout = isSfSwizzledLayout ? tensorrt_llm::FP4QuantizationSFLayout::SWIZZLED
                                         : tensorrt_llm::FP4QuantizationSFLayout::LINEAR;

#define LAUNCH_FP4_QUANTIZE_KERNEL(T)                                                              \
  tensorrt_llm::kernels::invokeFP4Quantization(                                                    \
      m, k, reinterpret_cast<T*>(self.data_ptr()), globalScale.data_ptr<float>(),                  \
      reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                                            \
      reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), sfUseUE8M0, layout, mMultiProcessorCount, \
      at::cuda::getCurrentCUDAStream(self.get_device()));

  if (self.scalar_type() == at::ScalarType::Half) {
    LAUNCH_FP4_QUANTIZE_KERNEL(half)
  } else if (self.scalar_type() == at::ScalarType::BFloat16) {
#ifdef ENABLE_BF16
    LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16)
#else
    C10_THROW_ERROR(NotImplementedError,
                    "BFloat16 must be enabled to quantize an bf16 tensor to fp4.");
#endif
  } else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn) {
#ifdef ENABLE_FP8
    LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3)
#else
    C10_THROW_ERROR(NotImplementedError, "FP8 must be enabled to quantize an fp8 tensor to fp4.");
#endif
  } else {
    C10_THROW_ERROR(NotImplementedError,
                    "fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3.");
  }

#undef LAUNCH_FP4_QUANTIZE_KERNEL

  return {valueE2M1, scaleFP8SF};
}

void fp4_swizzle_blockscale(at::Tensor const& unswizzled_sf, at::Tensor& swizzled_sf, int64_t b,
                            int64_t m, int64_t n) {
  CHECK_TH_CUDA(unswizzled_sf);
  CHECK_CONTIGUOUS(unswizzled_sf);
  CHECK_TH_CUDA(swizzled_sf);
  CHECK_CONTIGUOUS(swizzled_sf);
  CHECK_INPUT_TYPE(unswizzled_sf, c10::ScalarType::Byte);
  CHECK_INPUT_TYPE(swizzled_sf, c10::ScalarType::Byte);

  TORCH_CHECK(unswizzled_sf.sizes() == swizzled_sf.sizes(),
              "Input and output tensors must have the same shape");

  const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
  tensorrt_llm::kernels::invokeNVFP4BlockScaleInterleave(
      b, m, n, reinterpret_cast<uint8_t const*>(unswizzled_sf.data_ptr()),
      reinterpret_cast<uint8_t*>(swizzled_sf.data_ptr()), mMultiProcessorCount,
      at::cuda::getCurrentCUDAStream(unswizzled_sf.get_device()));
}
}  // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_quantize", &torch_ext::fp4_quantize);
  m.def("fp4_swizzle_blockscale", &torch_ext::fp4_swizzle_blockscale);
}
