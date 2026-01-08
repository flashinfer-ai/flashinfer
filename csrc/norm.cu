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

void rmsnorm_quant(TensorView output, TensorView input, TensorView weight, double scale, double eps,
                   bool enable_pdl) {
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
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(output.dtype(), o_type, [&] {
        cudaError_t status = norm::RMSNormQuant(
            static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
            static_cast<o_type*>(output.data_ptr()), batch_size, hidden_size, input.stride(0),
            output.stride(0), static_cast<float>(scale), eps, enable_pdl, stream);
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
                             TensorView weight, double scale, double eps, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_DEVICE(input, residual);
  CHECK_DEVICE(input, weight);
  CHECK_DEVICE(input, output);
  CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
  CHECK_DIM(2, residual);  // residual: (batch_size, hidden_size)
  CHECK_DIM(1, weight);    // weight: (hidden_size)
  CHECK_DIM(2, output);
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  TVM_FFI_ICHECK_EQ(residual.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(residual.size(1), hidden_size);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden_size);
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(output.dtype(), o_type, [&] {
      cudaError_t status = norm::FusedAddRMSNormQuant(
          static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(residual.data_ptr()),
          static_cast<c_type*>(weight.data_ptr()), static_cast<o_type*>(output.data_ptr()),
          batch_size, hidden_size, input.stride(0), residual.stride(0), output.stride(0), scale,
          eps, enable_pdl, stream);

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
