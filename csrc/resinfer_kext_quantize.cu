// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
// SPDX-License-Identifier: Apache-2.0
// TVM-FFI adapter for ResInfer's selected-channel K-extension quantizer.
#include <tvm/ffi/extra/cuda/device_guard.h>

#include <climits>
#include <flashinfer/experimental/resinfer/kext_pack16.cuh>
#include <flashinfer/experimental/resinfer/kext_pack8.cuh>

#include "tvm_ffi_utils.h"

void quantize(TensorView input, TensorView scales, TensorView masks, TensorView output,
              TensorView output_scales, int64_t residue_per_8, int64_t elts_per_thread) {
  CHECK_INPUT(input);
  CHECK_INPUT(scales);
  CHECK_INPUT(masks);
  CHECK_INPUT(output);
  CHECK_INPUT(output_scales);
  CHECK_DEVICE(input, scales);
  CHECK_DEVICE(input, masks);
  CHECK_DEVICE(input, output);
  CHECK_DEVICE(input, output_scales);
  CHECK_DIM(2, input);
  CHECK_DIM(1, masks);
  CHECK_DIM(2, output);
  check_dtype(scales, dl_float32, "scales");
  check_dtype(masks, dl_uint8, "masks");
  check_dtype(output, dl_uint8, "output");
  check_dtype(output_scales, dl_uint8, "output_scales");
  auto m = input.size(0), k = input.size(1);
  TVM_FFI_ICHECK(m > 0 && m <= INT_MAX - 127 && k > 0 && k <= INT_MAX / 2 && k % 256 == 0);
  TVM_FFI_ICHECK(residue_per_8 == 1 || residue_per_8 == 2 || residue_per_8 == 4);
  TVM_FFI_ICHECK(elts_per_thread == 8 || elts_per_thread == 16);
  auto ext = k + k / 8 * residue_per_8;
  TVM_FFI_ICHECK_EQ(scales.numel(), 1);
  TVM_FFI_ICHECK_EQ(masks.numel(), k / 8);
  TVM_FFI_ICHECK(output.size(0) == m && output.size(1) == ext / 2);
  TVM_FFI_ICHECK_EQ(output_scales.numel(), ((m + 127) / 128) * ((ext + 63) / 64) * 512);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(input.data_ptr()) % 32, 0);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % 8, 0);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(scales.data_ptr()) % 4, 0);
  ffi::CUDADeviceGuard guard(input.device().device_id);
  int major, minor, sms, max_threads;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, input.device().device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, input.device().device_id);
  TVM_FFI_ICHECK(major == 10 && minor == 0) << "ResInfer KEXT currently supports SM100";
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, input.device().device_id);
  cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor,
                         input.device().device_id);
  int groups = k / elts_per_thread;
  int threads = std::min(512, std::max(32, (groups + 31) / 32 * 32));
  int blocks_per_sm = std::max(1, std::min(4, max_threads / threads));
  dim3 block(threads), grid(std::min<int64_t>(((m + 127) / 128) * 128, sms * blocks_per_sm),
                            (groups + threads - 1) / threads);
  auto stream = get_stream(input.device());
#define LAUNCH(RATIO)                                                                       \
  if (elts_per_thread == 8) {                                                               \
    flashinfer::resinfer::cvt_fp16_to_fp4_residue_pack8_##RATIO##_no_swap<T, false>         \
        <<<grid, block, 0, stream>>>(m, ext, k, in, sf, out, out_sf, mask);                 \
  } else if (k % 512 == 0) {                                                                \
    flashinfer::resinfer::cvt_fp16_to_fp4_residue_pack16_##RATIO##_no_swap<T, false, false> \
        <<<grid, block, 0, stream>>>(m, ext, k, in, sf, out, out_sf, mask);                 \
  } else {                                                                                  \
    flashinfer::resinfer::cvt_fp16_to_fp4_residue_pack16_##RATIO##_no_swap<T, false, true>  \
        <<<grid, block, 0, stream>>>(m, ext, k, in, sf, out, out_sf, mask);                 \
  }
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), T, [&] {
    auto in = static_cast<const T*>(input.data_ptr());
    auto sf = static_cast<const float*>(scales.data_ptr());
    auto mask = static_cast<const uint8_t*>(masks.data_ptr());
    auto out = static_cast<uint32_t*>(output.data_ptr());
    auto out_sf = static_cast<uint32_t*>(output_scales.data_ptr());
    // Launch geometry and pack selection follow the public ResInfer producer.
    switch (residue_per_8) {
      case 1:
        LAUNCH(r0125);
        break;
      case 2:
        LAUNCH(r025);
        break;
      case 4:
        LAUNCH(r050);
        break;
    }
    return true;
  });
#undef LAUNCH
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize, quantize);
