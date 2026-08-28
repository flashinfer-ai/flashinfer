/*
 * Copyright (c) 2026 by FlashInfer team.
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
#include <flashinfer/vibecuda/softmax.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void softmax_vibecuda(TensorView logits, TensorView output,
                      Optional<TensorView> maybe_temperature_arr, double temperature_val,
                      bool enable_pdl) {
  CHECK_INPUT(logits);
  CHECK_INPUT(output);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  CHECK_DIM(2, output);  // output: (batch_size, vocab_size)
  CHECK_INPUT_TYPE(logits, dl_float32);
  CHECK_INPUT_TYPE(output, dl_float32);
  CHECK_SHAPE(logits, output);

  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);

  bool has_temperature_arr = maybe_temperature_arr.has_value();
  if (has_temperature_arr) {
    CHECK_INPUT(maybe_temperature_arr.value());
    CHECK_DIM(1, maybe_temperature_arr.value());
    CHECK_INPUT_TYPE(maybe_temperature_arr.value(), dl_float32);
    TVM_FFI_ICHECK_EQ(maybe_temperature_arr.value().size(0), batch_size)
        << "temperature tensor length must match batch_size";
    CHECK_DEVICE(maybe_temperature_arr.value(), logits);
  }

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  // The tuned cluster/pipe paths require SM100-class hardware (thread
  // clusters over 8+ CTAs, DSM pair pools, packed f32x2 exp-argument math).
  // Fail loudly when the vibecuda backend is selected on an unsupported
  // architecture instead of silently routing to another kernel.
  int cc_major = 0, cc_minor = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor,
                                        logits.device().device_id) == cudaSuccess &&
                 cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor,
                                        logits.device().device_id) == cudaSuccess)
      << "softmax_vibecuda: failed to query device compute capability";
  TVM_FFI_ICHECK(cc_major >= 10)
      << "softmax_vibecuda: the vibecuda softmax backend requires SM100-class "
         "GPUs (compute capability >= 10.0); got "
      << cc_major << "." << cc_minor;

  auto stream = get_stream(logits.device());
  cudaError_t status = flashinfer::vibecuda::Softmax(
      static_cast<float*>(logits.data_ptr()), static_cast<float*>(output.data_ptr()), batch_size,
      vocab_size,
      has_temperature_arr ? static_cast<float*>(maybe_temperature_arr.value().data_ptr()) : nullptr,
      static_cast<float>(temperature_val), enable_pdl, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "flashinfer::vibecuda::Softmax failed with error code " << cudaGetErrorString(status);
}
