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
#include <flashinfer/mamba/selective_state_update.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

namespace flashinfer::mamba {

// TODO: Implement the launcher function that:
// 1. Validates tensor dimensions and devices
// 2. Extracts raw pointers from TensorView
// 3. Calls the framework-agnostic kernel from include/flashinfer/mamba/selective_state_update.cuh
//
// Example signature (adjust based on your kernel requirements):
void selective_state_update(TensorView state, TensorView x, TensorView dt, TensorView output,
                            TensorView A, TensorView B, TensorView C, TensorView D,
                            Optional<TensorView> z, Optional<TensorView> dt_bias,
                            bool dt_softplus, Optional<TensorView> state_batch_indices,
                            int64_t pad_slot_id) {
  throw std::runtime_error("selective_state_update is not implemented yet.");

  // TODO: Add input validation
  // CHECK_LAST_DIM_CONTIGUOUS_INPUT(state);
  // CHECK_LAST_DIM_CONTIGUOUS_INPUT(x);
  // CHECK_DEVICE(state, x);
  // TVM_FFI_ICHECK_EQ(state.ndim(), 3);  // Example dimension check

  // TODO: Extract dimensions from tensors
  // unsigned int batch_size = state.size(0);
  // unsigned int dim_state = state.size(1);
  // unsigned int dim_input = x.size(1);

  // TODO: Get CUDA device and stream
  // ffi::CUDADeviceGuard device_guard(state.device().device_id);
  // const cudaStream_t stream = get_stream(state.device());

  // TODO: Dispatch based on dtype and call kernel
  // DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(state.dtype(), c_type, [&] {
  //   cudaError_t status = mamba::SelectiveStateUpdate(
  //       static_cast<c_type*>(state.data_ptr()),
  //       static_cast<const c_type*>(x.data_ptr()),
  //       static_cast<const c_type*>(dt.data_ptr()),
  //       static_cast<const c_type*>(A.data_ptr()),
  //       static_cast<const c_type*>(B.data_ptr()),
  //       static_cast<const c_type*>(C.data_ptr()),
  //       batch_size, dim_state, dim_input, stream);
  //   TVM_FFI_ICHECK(status == cudaSuccess)
  //       << "SelectiveStateUpdate failed with error code " << cudaGetErrorString(status);
  //   return true;
  // });
}

}  // namespace flashinfer::mamba
