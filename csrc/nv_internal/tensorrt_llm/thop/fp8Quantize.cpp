/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/thop/fp8Quantize.h"

#include <cstdint>

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/thop/utils.h"
#include "tvm/ffi/error.h"

// input: [M, K], fp32/fp16/bf16/fp8_quantized
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in
// linear layout. See QuantizationSFLayout enum for more details about the two layouts.
// returns
void mxfp8_quantize(TensorView input, TensorView valMxFP8, TensorView scaleFP8SF,
                    bool isSfSwizzledLayout, int64_t alignment, bool enable_pdl) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);

  // Fixed SF_VEC_SIZE as 32
  static constexpr int SF_VEC_SIZE = 32;
  TVM_FFI_ICHECK_EQ(alignment % SF_VEC_SIZE, 0)
      << "alignment must be divisible by SF_VEC_SIZE = 32";

  auto const& inputShape = input.sizes();
  auto const& rank = inputShape.size();

  TVM_FFI_ICHECK_GE(rank, 2) << "Input should be >=2D tensor.";
  int64_t m = 1;
  for (size_t i = 0; i < rank - 1; i++) {
    m *= inputShape[i];
  }
  auto const k = inputShape[rank - 1];
  TVM_FFI_ICHECK_EQ(k % SF_VEC_SIZE, 0) << "k must be divisible by SF_VEC_SIZE = 32";
  auto const padded_k = ((k + alignment - 1) / alignment) * alignment;

  const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

  auto const layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                                         : tensorrt_llm::QuantizationSFLayout::LINEAR;

#define LAUNCH_MXFP8_QUANTIZE_KERNEL(T)                                                            \
  tensorrt_llm::kernels::invokeMxFP8Quantization(                                                  \
      1, m, k, padded_k, reinterpret_cast<T*>(input.data_ptr()),                                   \
      reinterpret_cast<int64_t*>(valMxFP8.data_ptr()),                                             \
      reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), layout, mMultiProcessorCount, enable_pdl, \
      get_stream(input.device()));

  if (input.dtype() == dl_float16) {
    LAUNCH_MXFP8_QUANTIZE_KERNEL(half)
  } else if (input.dtype() == dl_bfloat16) {
#ifdef ENABLE_BF16
    LAUNCH_MXFP8_QUANTIZE_KERNEL(__nv_bfloat16)
#else
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "BFloat16 must be enabled to quantize an bf16 tensor to mxfp8.";
#endif
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "mxfp8_quantize only supports input tensor with dtypes fp16/bf16.";
  }

#undef LAUNCH_MXFP8_QUANTIZE_KERNEL
}

inline uint8_t float_to_ue8m0(float value) {
  if (value == 0.0f) {
    return 0x00;
  }
  constexpr uint32_t FP32_MANTISSA_BITS = 23;
  uint32_t val_u32 = *reinterpret_cast<uint32_t*>(&value);
  uint8_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
  uint32_t mantissa = val_u32 & 0x7FFFFF;
  // Round up exponent and deal with satfinite.
  if ((mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)) {
    ++exponent;
  }
  return exponent;
}

// Used in tests to quantize mxe4m3 tensors on host.
void mxfp8_quantize_host(TensorView x_fp32, TensorView fp8_tensor, TensorView scale_tensor,
                         bool is_sf_swizzled_layout) {
  int32_t const sf_vec_size = 32;
  auto fp32_dtype = DLDataType{kDLFloat, 32, 1};
  CHECK_INPUT_TYPE(x_fp32, fp32_dtype);
  auto data_shape = x_fp32.sizes();
  TVM_FFI_ICHECK_EQ(data_shape.size(), 2) << "x_fp32 should be 2D tensor.";
  int num_tokens = data_shape[0];
  int hidden_dim = data_shape[1];
  int groups_per_hidden_dim = hidden_dim / sf_vec_size;

  tensorrt_llm::QuantizationSFLayout layout =
      is_sf_swizzled_layout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                            : tensorrt_llm::QuantizationSFLayout::LINEAR;

  for (size_t ti = 0; ti < static_cast<size_t>(data_shape[0]); ++ti) {
    for (int group = 0; group < groups_per_hidden_dim; ++group) {
      float* fp32_ptr =
          static_cast<float*>(x_fp32.data_ptr()) + ti * hidden_dim + group * sf_vec_size;
      uint8_t* fp8_ptr =
          static_cast<uint8_t*>(fp8_tensor.data_ptr()) + ti * hidden_dim + group * sf_vec_size;

      uint8_t* scale_ue8m08sf_ptr = static_cast<uint8_t*>(scale_tensor.data_ptr());

      float local_amax = 0.0f;
      for (int ki = 0; ki < sf_vec_size; ++ki) {
        local_amax = std::max(std::abs(fp32_ptr[ki]), local_amax);
      }

      local_amax *= (1.f / 448.0f);

      uint8_t scale_ue8m0 = float_to_ue8m0(local_amax);
      auto const inv_scale = (scale_ue8m0 == 0) ? 1 : exp2f(127 - static_cast<float>(scale_ue8m0));

      scale_ue8m08sf_ptr[computeSFIndex(ti, group, data_shape[0], groups_per_hidden_dim, layout)] =
          scale_ue8m0;

      for (int ki = 0; ki < sf_vec_size; ++ki) {
        float const scaled_fp32_value = fp32_ptr[ki] * inv_scale;
        auto fp8_value = cutlass::float_e4m3_t{scaled_fp32_value};
        fp8_ptr[ki] = *reinterpret_cast<uint8_t*>(&fp8_value);
      }
    }
  }
}

// Used in tests to dequantize mxe4m3 tensors on host.
void mxfp8_dequantize_host(TensorView value_e4m3, TensorView scale_ue8m08sf,
                           TensorView float_tensor, bool is_sf_swizzled_layout) {
  int32_t const sf_vec_size = 32;
  CHECK_INPUT_TYPE(value_e4m3, dl_uint8);
  CHECK_INPUT_TYPE(scale_ue8m08sf, dl_uint8);
  auto data_shape = value_e4m3.sizes();
  auto scale_shape = scale_ue8m08sf.sizes();
  TVM_FFI_ICHECK_EQ(data_shape.size(), 2) << "value_e4m3 should be 2D tensor.";
  TVM_FFI_ICHECK_EQ(scale_shape.size(), 1) << "scale_ue8m08sf should be 1D tensor.";

  int hidden_dim = data_shape[1];
  int groups_per_hidden_dim = hidden_dim / sf_vec_size;

  tensorrt_llm::QuantizationSFLayout layout =
      is_sf_swizzled_layout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                            : tensorrt_llm::QuantizationSFLayout::LINEAR;
  for (size_t ti = 0; ti < static_cast<size_t>(data_shape[0]); ++ti) {
    for (int group = 0; group < groups_per_hidden_dim; ++group) {
      float* float_ptr =
          static_cast<float*>(float_tensor.data_ptr()) + ti * hidden_dim + group * sf_vec_size;
      uint8_t* fp8_ptr =
          static_cast<uint8_t*>(value_e4m3.data_ptr()) + ti * hidden_dim + group * sf_vec_size;
      uint8_t* scale_ue8m08sf_ptr = static_cast<uint8_t*>(scale_ue8m08sf.data_ptr());
      uint8_t fp8_scale = scale_ue8m08sf_ptr[computeSFIndex(ti, group, data_shape[0],
                                                            groups_per_hidden_dim, layout)];

      float scale_float;
      uint32_t scale_float_u32 = uint32_t(fp8_scale) << 23;
      memcpy(&scale_float, &scale_float_u32, sizeof(scale_float));

      for (int ki = 0; ki < sf_vec_size; ++ki) {
        uint8_t fp8_u8_repr = fp8_ptr[ki];
        auto fp32 = static_cast<float>(*reinterpret_cast<cutlass::float_e4m3_t*>(&fp8_u8_repr));
        float value = fp32 * scale_float;
        float_ptr[ki] = value;
      }
    }
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp8_dequantize_host, mxfp8_dequantize_host);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp8_quantize_host, mxfp8_quantize_host);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp8_quantize, mxfp8_quantize);
