// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
// SPDX-License-Identifier: Apache-2.0
// FlashInfer TVM-FFI binding for ResInfer's MEXT quantizer.
#include <tvm/ffi/extra/cuda/device_guard.h>

#include <climits>
#include <flashinfer/experimental/resinfer/quantize.cuh>

#include "tvm_ffi_utils.h"

void quantize(TensorView input, TensorView scales, TensorView output, TensorView output_scales) {
  CHECK_INPUT(input);
  CHECK_INPUT(scales);
  CHECK_INPUT(output);
  CHECK_INPUT(output_scales);
  CHECK_DEVICE(input, scales);
  CHECK_DEVICE(input, output);
  CHECK_DEVICE(input, output_scales);
  CHECK_DIM(2, input);
  CHECK_DIM(2, output);
  check_dtype(scales, dl_float32, "scales");
  check_dtype(output, dl_uint8, "output");
  check_dtype(output_scales, dl_uint8, "output_scales");
  auto m = input.size(0), k = input.size(1);
  TVM_FFI_ICHECK(m > 0 && m <= INT_MAX / 2 && k > 0 && k <= INT_MAX && k % 16 == 0);
  TVM_FFI_ICHECK(scales.numel() == 1 || scales.numel() == 2);
  TVM_FFI_ICHECK(output.size(0) == 2 * m && output.size(1) == k / 2);
  TVM_FFI_ICHECK_EQ(output_scales.numel(), ((2 * m + 127) / 128) * ((k + 63) / 64) * 512);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(input.data_ptr()) % 32, 0);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % 8, 0);
  ffi::CUDADeviceGuard guard(input.device().device_id);
  int major, minor, sms;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, input.device().device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, input.device().device_id);
  TVM_FFI_ICHECK(major == 10 && minor == 0) << "ResInfer MEXT currently supports SM100";
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, input.device().device_id);
  int groups = k / 16;
  int desired_y = std::min<int>((((m + 127) / 128) * 128 + m - 1) / m, groups);
  int threads = std::min(512, std::max(32, ((groups + desired_y - 1) / desired_y + 31) / 32 * 32));
  dim3 block(threads), grid(std::min<int64_t>(m, sms * 4), (groups + threads - 1) / threads);
  auto stream = get_stream(input.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), T, [&] {
    flashinfer::resinfer::QuantizeRowPairs<T><<<grid, block, 0, stream>>>(
        m, k, static_cast<const T*>(input.data_ptr()), static_cast<const float*>(scales.data_ptr()),
        scales.numel(), static_cast<uint32_t*>(output.data_ptr()),
        static_cast<uint8_t*>(output_scales.data_ptr()));
    return true;
  });
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize, quantize);
