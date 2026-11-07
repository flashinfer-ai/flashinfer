// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
// SPDX-License-Identifier: Apache-2.0
// TVM-FFI adapter for ResInfer's grouped Q residue quantizer.
#include <climits>
#include <flashinfer/experimental/resinfer/grouped_q.cuh>

#include "tvm_ffi_utils.h"

void quantize_grouped_q(TensorView q, TensorView packed, TensorView scales) {
  CHECK_INPUT(q);
  CHECK_INPUT(packed);
  CHECK_INPUT(scales);
  CHECK_DEVICE(q, packed);
  CHECK_DEVICE(q, scales);
  CHECK_DIM(4, q);
  CHECK_DIM(4, packed);
  check_dtype(q, dl_bfloat16, "q");
  check_dtype(packed, dl_uint8, "packed");
  check_dtype(scales, dl_uint8, "scales");
  auto b = q.size(0), h = q.size(1), g = q.size(2), d = q.size(3);
  TVM_FFI_ICHECK(b > 0 && h > 0 && b <= INT_MAX / h);
  TVM_FFI_ICHECK(g == 4 || g == 6 || g == 8);
  TVM_FFI_ICHECK(d == 128 || d == 256);
  int physical = g == 4 ? 8 : 16;
  TVM_FFI_ICHECK(packed.size(0) == b && packed.size(1) == h && packed.size(2) == physical &&
                 packed.size(3) == d / 2);
  TVM_FFI_ICHECK_EQ(scales.numel(), b * h * 128 * (d / 16));
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(q.data_ptr()) % 32, 0);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 8, 0);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.device().device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.device().device_id);
  TVM_FFI_ICHECK(major == 10 && minor == 0) << "ResAttn currently supports SM100";
  flashinfer::resinfer::QuantizeGroupedQ<<<b * h, 128, 0, get_stream(q.device())>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()), static_cast<uint8_t*>(packed.data_ptr()),
      static_cast<uint8_t*>(scales.data_ptr()), g, physical, d);
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize, quantize_grouped_q);
