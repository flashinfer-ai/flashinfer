/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/fused_rmsnorm_silu/fusedRMSNormSiLU.h>

#include "tvm_ffi_utils.h"

void fused_rmsnorm_silu(TensorView output, TensorView input, TensorView weight, double eps,
                        double scale) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(2, input);   // [num_tokens, hidden_size]
  CHECK_DIM(2, output);  // [num_tokens, hidden_size]
  CHECK_DIM(1, weight);  // [hidden_size]

  unsigned int num_tokens = input.size(0);
  unsigned int hidden_size = input.size(1);
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden_size);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden_size);

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  // Get SM count for persistent CTA kernels
  int device_id = input.device().device_id;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  int num_sms = prop.multiProcessorCount;

  // Launch the fused RMSNorm + SiLU kernel (standard layout, no NDHWC)
  launchFusedRMSNormSiLU<false>(
      input.data_ptr(),                     // input
      output.data_ptr(),                    // output
      static_cast<int>(num_tokens),         // num_tokens
      static_cast<int>(hidden_size),        // hidden_dim
      static_cast<int>(input.stride(0)),    // input_stride_token
      num_sms,                              // num_sms
      static_cast<float>(eps),              // eps
      weight.data_ptr(),                    // weight
      static_cast<float>(scale),            // scale
      nullptr,                              // bias (not used)
      static_cast<int>(output.stride(0)),   // output_stride_token
      0, 0, 0,                              // DHW, HW, W (unused for standard layout)
      0, 0, 0, 0,                           // output strides N,D,H,W (unused)
      0,                                    // output_D_offset (unused)
      stream);
}
