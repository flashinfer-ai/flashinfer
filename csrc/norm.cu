/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <flashinfer/norm.cuh>
#include <flashinfer/norm/fused_dit_layernorm.cuh>
#include <flashinfer/norm/fused_qk_rmsnorm_rope.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

void rmsnorm(TensorView output, TensorView input, TensorView weight, double eps, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(1, weight);  // weight: (hidden_size)

  auto input_ndim = input.ndim();
  if (input_ndim == 2) {
    // Normal RMSNorm: [batch_size, hidden_size]
    // Use CTA parallelization for better parallelism
    CHECK_DIM(2, output);
    TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
    unsigned int batch_size = input.size(0);
    unsigned int hidden_size = input.size(1);
    TVM_FFI_ICHECK_EQ(output.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(output.size(1), hidden_size);
    ffi::CUDADeviceGuard device_guard(input.device().device_id);
    const cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
      cudaError_t status = norm::RMSNorm(
          static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
          static_cast<c_type*>(output.data_ptr()), batch_size, hidden_size, input.stride(0),
          output.stride(0), eps, enable_pdl, stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "RMSNorm failed with error code " << cudaGetErrorString(status);
      return true;
    });
  } else if (input_ndim == 3) {
    // QK RMSNorm: [batch_size, num_heads, head_dim]
    // Use warp-level parallization
    CHECK_DIM(3, output);  // output: (batch_size, num_heads, hidden_size)
    TVM_FFI_ICHECK_EQ(input.size(2), weight.size(0));
    unsigned int batch_size = input.size(0);
    unsigned int num_heads = input.size(1);
    unsigned int hidden_size = input.size(2);
    TVM_FFI_ICHECK_EQ(output.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(output.size(1), num_heads);
    TVM_FFI_ICHECK_EQ(output.size(2), hidden_size);

    ffi::CUDADeviceGuard device_guard(input.device().device_id);
    const cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
      cudaError_t status = norm::QKRMSNorm(
          static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
          static_cast<c_type*>(output.data_ptr()), batch_size, num_heads, hidden_size,
          input.stride(0), input.stride(1), output.stride(0), output.stride(1), eps, enable_pdl,
          stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "QKRMSNorm failed with error code " << cudaGetErrorString(status);
      return true;
    });
  } else {
    TVM_FFI_ICHECK(false) << "Unsupported input dimension: " << input_ndim;
  }
}

void rmsnorm_quant(TensorView output, TensorView input, TensorView weight, TensorView scale,
                   double eps, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DEVICE(input, scale);
  CHECK_DIM(1, weight);  // weight: (hidden_size)
  TVM_FFI_ICHECK_EQ(scale.numel(), 1);

  auto input_ndim = input.ndim();
  if (input_ndim == 2) {
    // Normal RMSNorm: [batch_size, hidden_size]
    // Use CTA parallelization for better parallelism
    CHECK_DIM(2, output);
    TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
    unsigned int batch_size = input.size(0);
    unsigned int hidden_size = input.size(1);
    TVM_FFI_ICHECK_EQ(output.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(output.size(1), hidden_size);
    ffi::CUDADeviceGuard device_guard(input.device().device_id);
    const cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(output.dtype(), o_type, [&] {
        cudaError_t status = norm::RMSNormQuant(
            static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
            static_cast<o_type*>(output.data_ptr()), batch_size, hidden_size, input.stride(0),
            output.stride(0), static_cast<float*>(scale.data_ptr()), eps, enable_pdl, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "RMSNormQuant failed with error code " << cudaGetErrorString(status);
        return true;
      });
    });
  } else {
    TVM_FFI_ICHECK(false) << "Unsupported input dimension: " << input_ndim;
  }
}

void fused_add_rmsnorm(TensorView input, TensorView residual, TensorView weight, double eps,
                       bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, residual);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
  CHECK_DIM(2, residual);  // residual: (batch_size, hidden_size)
  CHECK_DIM(1, weight);    // weight: (hidden_size)
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  TVM_FFI_ICHECK_EQ(residual.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(residual.size(1), hidden_size);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden_size);
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    cudaError_t status = norm::FusedAddRMSNorm(
        static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(residual.data_ptr()),
        static_cast<c_type*>(weight.data_ptr()), batch_size, hidden_size, input.stride(0),
        residual.stride(0), eps, enable_pdl, stream);

    TVM_FFI_ICHECK(status == cudaSuccess)
        << "FusedAddRMSNorm failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void fused_add_rmsnorm_quant(TensorView output, TensorView input, TensorView residual,
                             TensorView weight, TensorView scale, double eps, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_DEVICE(input, residual);
  CHECK_DEVICE(input, weight);
  CHECK_DEVICE(input, output);
  CHECK_DEVICE(input, scale);
  CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
  CHECK_DIM(2, residual);  // residual: (batch_size, hidden_size)
  CHECK_DIM(1, weight);    // weight: (hidden_size)
  CHECK_DIM(2, output);
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  TVM_FFI_ICHECK_EQ(residual.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(residual.size(1), hidden_size);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden_size);
  TVM_FFI_ICHECK_EQ(scale.numel(), 1);
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(output.dtype(), o_type, [&] {
      cudaError_t status = norm::FusedAddRMSNormQuant(
          static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(residual.data_ptr()),
          static_cast<c_type*>(weight.data_ptr()), static_cast<o_type*>(output.data_ptr()),
          batch_size, hidden_size, input.stride(0), residual.stride(0), output.stride(0),
          static_cast<float*>(scale.data_ptr()), eps, enable_pdl, stream);

      TVM_FFI_ICHECK(status == cudaSuccess)
          << "FusedAddRMSNormQuant failed with error code " << cudaGetErrorString(status);
      return true;
    });
  });
}

void gemma_rmsnorm(TensorView output, TensorView input, TensorView weight, double eps,
                   bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(2, input);   // input: (batch_size, hidden_size)
  CHECK_DIM(1, weight);  // weight: (hidden_size)
  TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  TVM_FFI_ICHECK_EQ(output.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden_size);
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    cudaError_t status = norm::GemmaRMSNorm(
        static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
        static_cast<c_type*>(output.data_ptr()), batch_size, hidden_size, input.stride(0),
        output.stride(0), eps, enable_pdl, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "GemmaRMSNorm failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void gemma_fused_add_rmsnorm(TensorView input, TensorView residual, TensorView weight, double eps,
                             bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, residual);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
  CHECK_DIM(2, residual);  // residual: (batch_size, hidden_size)
  CHECK_DIM(1, weight);    // weight: (hidden_size)
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  TVM_FFI_ICHECK_EQ(residual.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(residual.size(1), hidden_size);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden_size);
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    cudaError_t status = norm::GemmaFusedAddRMSNorm(
        static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(residual.data_ptr()),
        static_cast<c_type*>(weight.data_ptr()), batch_size, hidden_size, input.stride(0),
        residual.stride(0), eps, enable_pdl, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "GemmaFusedAddRMSNorm failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void layernorm(Tensor output, Tensor input, Tensor gamma, Tensor beta, double eps) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(gamma);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(beta);
  CHECK_DEVICE(input, gamma);
  CHECK_DEVICE(input, beta);
  CHECK_DIM(2, input);  // input: (batch_size, hidden_size)
  CHECK_DIM(1, gamma);  // gamma: (hidden_size)
  CHECK_DIM(1, beta);   // beta: (hidden_size)
  TVM_FFI_ICHECK_EQ(input.size(1), gamma.size(0));
  TVM_FFI_ICHECK_EQ(input.size(1), beta.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  TVM_FFI_ICHECK_EQ(output.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden_size);
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());
  // TODO(kaixih): This is currently our only use case; Add more if needed.
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_bfloat16) << "input must be bfloat16";
  TVM_FFI_ICHECK_EQ(gamma.dtype(), dl_float32) << "gamma must be float32";
  TVM_FFI_ICHECK_EQ(beta.dtype(), dl_float32) << "beta must be float32";

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    cudaError_t status = norm::LayerNorm(
        static_cast<c_type*>(input.data_ptr()), static_cast<float*>(gamma.data_ptr()),
        static_cast<float*>(beta.data_ptr()), static_cast<c_type*>(output.data_ptr()), batch_size,
        hidden_size, eps, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "LayerNorm failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void fused_dit_layernorm_run(TensorView input, TensorView residual, TensorView gate,
                             TensorView gate_bias, TensorView gamma, TensorView beta,
                             TensorView scale, TensorView scale_bias, TensorView shift,
                             TensorView shift_bias, TensorView residual_out, TensorView norm_out,
                             TensorView sf_out, TensorView sf_scale, TensorView input_sf_scale,
                             double epsilon, int64_t mode, int64_t output_format) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  CHECK_INPUT_TYPE(input, dl_bfloat16);

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  int batch_size = static_cast<int>(input.size(0));
  int num_rows = static_cast<int>(input.size(1));
  int hidden_size = static_cast<int>(input.size(2));

  auto get_bf16_ptr = [](TensorView& t) -> __nv_bfloat16* {
    return t.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(t.data_ptr()) : nullptr;
  };
  auto get_float_ptr = [](TensorView& t) -> float* {
    return t.numel() > 0 ? reinterpret_cast<float*>(t.data_ptr()) : nullptr;
  };
  auto get_const_bf16_ptr = [](TensorView& t) -> const __nv_bfloat16* {
    return t.numel() > 0 ? reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()) : nullptr;
  };
  auto get_const_float_ptr = [](TensorView& t) -> const float* {
    return t.numel() > 0 ? reinterpret_cast<const float*>(t.data_ptr()) : nullptr;
  };

  fused_layernorm::launchFusedLayerNorm(
      get_bf16_ptr(residual_out), reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), get_const_float_ptr(gamma),
      get_const_float_ptr(beta), get_const_bf16_ptr(gate), get_const_float_ptr(gate_bias),
      get_const_bf16_ptr(residual), get_const_bf16_ptr(scale), get_const_float_ptr(scale_bias),
      get_const_bf16_ptr(shift), get_const_float_ptr(shift_bias), static_cast<float>(epsilon),
      batch_size, num_rows, hidden_size, stream,
      static_cast<fused_layernorm::FusedLayerNormMode>(mode),
      static_cast<fused_layernorm::OutputFormat>(output_format),
      sf_out.numel() > 0 ? reinterpret_cast<uint32_t*>(sf_out.data_ptr()) : nullptr,
      get_float_ptr(sf_scale), get_float_ptr(input_sf_scale));
}

void fused_qk_rmsnorm_rope_run(TensorView qkv_in, TensorView q_weight, TensorView k_weight,
                               TensorView q_out, TensorView k_out, TensorView v_out,
                               int64_t num_tokens, int64_t seq_len, int64_t ppf, int64_t pph,
                               int64_t ppw, int64_t num_frame_channels, int64_t num_height_channels,
                               int64_t num_width_channels, int64_t num_heads_q, int64_t num_heads_k,
                               int64_t num_heads_v, int64_t head_dim, double eps, double base,
                               bool interleave, double factor, double low, double high,
                               double attention_factor, bool is_qk_norm, bool output_fp8,
                               double output_quant_scale, double v_quant_scale) {
  CHECK_INPUT(qkv_in);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_CUDA(q_out);
  CHECK_CONTIGUOUS(q_out);
  CHECK_CUDA(k_out);
  CHECK_CONTIGUOUS(k_out);
  CHECK_CUDA(v_out);
  CHECK_CONTIGUOUS(v_out);

  CHECK_INPUT_TYPE(qkv_in, dl_bfloat16);
  CHECK_INPUT_TYPE(q_weight, dl_bfloat16);
  CHECK_INPUT_TYPE(k_weight, dl_bfloat16);

  ffi::CUDADeviceGuard device_guard(qkv_in.device().device_id);
  const cudaStream_t stream = get_stream(qkv_in.device());

  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, qkv_in.device().device_id);

  launchFusedQKNormRope(
      qkv_in.data_ptr(), q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), num_tokens,
      static_cast<int>(seq_len), static_cast<int>(ppf), static_cast<int>(pph),
      static_cast<int>(ppw), static_cast<int>(num_frame_channels),
      static_cast<int>(num_height_channels), static_cast<int>(num_width_channels),
      static_cast<int>(num_heads_q), static_cast<int>(num_heads_k), static_cast<int>(num_heads_v),
      static_cast<int>(head_dim), static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
      static_cast<float>(base), interleave, static_cast<float>(factor), static_cast<float>(low),
      static_cast<float>(high), static_cast<float>(attention_factor), stream, is_qk_norm, num_sms,
      output_fp8, static_cast<float>(output_quant_scale), static_cast<float>(v_quant_scale));
}
