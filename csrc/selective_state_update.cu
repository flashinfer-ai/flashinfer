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

void selective_state_update(TensorView state, TensorView x, TensorView dt, TensorView output,
                            TensorView A, TensorView B, TensorView C, TensorView D,
                            Optional<TensorView> z, Optional<TensorView> dt_bias, bool dt_softplus,
                            Optional<TensorView> state_batch_indices, int64_t pad_slot_id) {
  auto const batch = x.size(0);
  auto const state_cache_size = state.size(0);
  auto const nheads = state.size(1);
  auto const dim = state.size(2);
  auto const dstate = state.size(3);
  auto const ngroups = B.size(1);

  FLASHINFER_CHECK(state_cache_size >= batch, "state.size(0) must be >= x.size(0)");

  FLASHINFER_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");

  // Check x shape and strides
  CHECK_DIM(3, x);
  FLASHINFER_CHECK(x.size(1) == nheads, "x.size(1) must equal nheads");
  FLASHINFER_CHECK(x.size(2) == dim, "x.size(2) must equal dim");
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(x);
  FLASHINFER_CHECK(x.stride(1) == dim, "x.stride(1) must equal dim, got ", x.stride(1),
                   " expected ", x.size(2));

  // Check output shape and strides
  CHECK_DIM(3, output);
  CHECK_LAST_DIM_CONTIGUOUS(output);
  FLASHINFER_CHECK(output.size(1) == nheads, "output.size(1) must equal nheads");
  FLASHINFER_CHECK(output.size(2) == dim, "output.size(2) must equal dim");
  FLASHINFER_CHECK(output.stride(1) == dim, "output.stride(1) must equal dim");

  // Check dt shape and strides
  CHECK_CUDA(dt);
  CHECK_DIM(3, dt);  // dt: {batch, nheads, dim}
  FLASHINFER_CHECK(dt.size(1) == nheads, "dt.size(1) must equal nheads");
  FLASHINFER_CHECK(dt.size(2) == dim, "dt.size(2) must equal dim");
  FLASHINFER_CHECK(dt.stride(1) == 1, "dt.stride(1) must be 1, got ", dt.stride(1));
  FLASHINFER_CHECK(dt.stride(2) == 0, "dt.stride(2) must be 0 (broadcasted), got ", dt.stride(2));

  // Check state - fully contiguous
  CHECK_INPUT(state);   // CUDA + fully contiguous (uses TVM FFI)
  CHECK_DIM(4, state);  // state: {state_cache_size, nheads, dim, dstate}

  // Check B shape and strides
  CHECK_CUDA(B);
  CHECK_DIM(3, B);  // B: {batch, B.size(1), dstate}
  FLASHINFER_CHECK(B.size(0) == batch, "B.size(0) must equal batch");
  FLASHINFER_CHECK(B.size(1) == ngroups, "B.size(1) must equal ngroups");
  FLASHINFER_CHECK(B.size(2) == dstate, "B.size(2) must equal dstate");
  CHECK_LAST_DIM_CONTIGUOUS(B);  // stride(2) == 1
  FLASHINFER_CHECK(B.stride(1) == B.size(2), "B.stride(1) must equal dstate, got ", B.stride(1),
                   " expected ", B.size(2));

  // Check C shape and strides
  CHECK_CUDA(C);
  CHECK_LAST_DIM_CONTIGUOUS(C);  // stride(2) == 1
  CHECK_DIM(3, C);               // C: {batch, C.size(1), dstate}
  FLASHINFER_CHECK(C.stride(1) == C.size(2), "C.stride(1) must equal dstate, got ", C.stride(1),
                   " expected ", C.size(2));
  FLASHINFER_CHECK(C.size(0) == batch, "C.size(0) must equal batch");
  FLASHINFER_CHECK(C.size(1) == ngroups, "C.size(1) must equal ngroups");
  FLASHINFER_CHECK(C.size(2) == dstate, "C.size(2) must equal dstate");

  // Check D - specific stride patterns indicating broadcasting
  CHECK_CUDA(D);
  CHECK_DIM(2, D);  // D: {nheads, dim}
  FLASHINFER_CHECK(D.size(0) == nheads, "D.size(0) must equal nheads");
  FLASHINFER_CHECK(D.size(1) == dim, "D.size(1) must equal dim");
  FLASHINFER_CHECK(D.stride(0) == 1, "D.stride(0) must be 1, got ", D.stride(0));
  FLASHINFER_CHECK(D.stride(1) == 0, "D.stride(1) must be 0 (broadcasted), got ", D.stride(1));

  // Check A - specific stride patterns indicating broadcasting
  CHECK_CUDA(A);
  CHECK_DIM(3, A);  // A: {nheads, dim, dstate}
  FLASHINFER_CHECK(A.size(0) == nheads, "A.size(0) must equal nheads");
  FLASHINFER_CHECK(A.size(1) == dim, "A.size(1) must equal dim");
  FLASHINFER_CHECK(A.size(2) == dstate, "A.size(2) must equal dstate");
  FLASHINFER_CHECK(A.stride(1) == 0, "A.stride(1) must be 0 (broadcasted), got ", A.stride(1));
  FLASHINFER_CHECK(A.stride(2) == 0, "A.stride(2) must be 0 (broadcasted), got ", A.stride(2));

  // Optional dt_bias check
  if (dt_bias.has_value()) {
    auto& bias = dt_bias.value();
    CHECK_CUDA(bias);
    CHECK_DIM(2, bias);  // dt_bias: {nheads, dim}
    FLASHINFER_CHECK(bias.size(0) == nheads, "dt_bias.size(0) must equal nheads");
    FLASHINFER_CHECK(bias.size(1) == dim, "dt_bias.size(1) must equal dim");
    FLASHINFER_CHECK(bias.stride(0) == 1, "dt_bias.stride(0) must be 1, got ", bias.stride(0));
    FLASHINFER_CHECK(bias.stride(1) == 0, "dt_bias.stride(1) must be 0 (broadcasted), got ",
                     bias.stride(1));
  }

  if (z.has_value()) {
    auto& z_tensor = z.value();
    CHECK_CUDA(z_tensor);
    CHECK_DIM(3, z_tensor);  // z: {batch, nheads, dim}
    FLASHINFER_CHECK(z_tensor.size(0) == batch, "z.size(0) must equal batch");
    FLASHINFER_CHECK(z_tensor.size(1) == nheads, "z.size(1) must equal nheads");
    FLASHINFER_CHECK(z_tensor.size(2) == dim, "z.size(2) must equal dim");
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(z_tensor);
    FLASHINFER_CHECK(z_tensor.stride(1) == dim, "z.stride(1) must equal dim, got ",
                     z_tensor.stride(1), " expected ", z_tensor.size(2));
  }

  if (state_batch_indices) {
    CHECK_DIM(1, (*state_batch_indices));
    FLASHINFER_CHECK(state_batch_indices.value().size(0) == batch,
                     "state_batch_indices.shape must be (", batch, ")");
  }

  SelectiveStateUpdateParams p;

  // copy dimensions
  p.batch = batch;
  p.nheads = nheads;
  p.dim = dim;
  p.dstate = dstate;
  p.ngroups = ngroups;
  p.state_cache_size = state_cache_size;
  p.dt_softplus = dt_softplus;
  p.pad_slot_id = pad_slot_id;

  // Copy strides
  p.x_stride_batch = x.stride(0);
  p.dt_stride_batch = dt.stride(0);
  p.B_stride_batch = B.stride(0);
  p.C_stride_batch = C.stride(0);
  p.out_stride_batch = output.stride(0);
  if (state_batch_indices) p.state_batch_indices = state_batch_indices.value().data_ptr();

  // Copy pointers
  p.state = state.data_ptr();
  p.x = x.data_ptr();
  p.dt = dt.data_ptr();
  p.output = output.data_ptr();
  if (dt_bias) {
    p.dt_bias = dt_bias.value().data_ptr();
  }
  if (z) {
    p.z = z.value().data_ptr();
    p.z_stride_batch = z.value().stride(0);
  }
  p.A = A.data_ptr();
  p.B = B.data_ptr();
  p.C = C.data_ptr();
  p.D = D.data_ptr();

  // Set device and get stream
  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  const cudaStream_t stream = get_stream(state.device());

  // Dispatch based on dtype combination
  DLDataType state_dtype = state.dtype();
  DLDataType input_dtype = x.dtype();
  DLDataType weight_dtype = dt.dtype();
  DLDataType matrixA_dtype = A.dtype();

  int64_t state_dtype_code = encode_dlpack_dtype(state_dtype);
  int64_t input_dtype_code = encode_dlpack_dtype(input_dtype);
  int64_t weight_dtype_code = encode_dlpack_dtype(weight_dtype);
  int64_t matrixA_dtype_code = encode_dlpack_dtype(matrixA_dtype);

  // Pack all dtype codes into a single value for switching
  auto dtype_key =
      std::make_tuple(state_dtype_code, input_dtype_code, weight_dtype_code, matrixA_dtype_code);

  if (dtype_key == std::make_tuple(/*state*/ bfloat16_code, /*input */ bfloat16_code,
                                   /*weight */ bfloat16_code, /*matrixA */ float32_code)) {
    using state_t = nv_bfloat16;
    using input_t = nv_bfloat16;
    using weight_t = nv_bfloat16;
    using matrixA_t = float;
    invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
  } else if (dtype_key == std::make_tuple(/*state*/ float16_code, /*input */ bfloat16_code,
                                          /*weight */ bfloat16_code, /*matrixA */ float32_code)) {
    using state_t = half;
    using input_t = nv_bfloat16;
    using weight_t = nv_bfloat16;
    using matrixA_t = float;
    invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
  } else if (dtype_key == std::make_tuple(/*state*/ float32_code, /*input */ bfloat16_code,
                                          /*weight */ bfloat16_code, /*matrixA */ float32_code)) {
    using state_t = float;
    using input_t = nv_bfloat16;
    using weight_t = nv_bfloat16;
    using matrixA_t = float;
    invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
  } else if (dtype_key == std::make_tuple(/*state*/ bfloat16_code, /*input */ bfloat16_code,
                                          /*weight */ float32_code, /*matrixA */ float32_code)) {
    using state_t = nv_bfloat16;
    using input_t = nv_bfloat16;
    using weight_t = float;
    using matrixA_t = float;
    invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
  } else if (dtype_key == std::make_tuple(/*state*/ float16_code, /*input */ bfloat16_code,
                                          /*weight */ float32_code, /*matrixA */ float32_code)) {
    using state_t = half;
    using input_t = nv_bfloat16;
    using weight_t = float;
    using matrixA_t = float;
    invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
  } else if (dtype_key == std::make_tuple(/*state*/ float32_code, /*input */ bfloat16_code,
                                          /*weight */ float32_code, /*matrixA */ float32_code)) {
    using state_t = float;
    using input_t = nv_bfloat16;
    using weight_t = float;
    using matrixA_t = float;
    invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
  } else {
    // Default case: unsupported dtype combination
    TVM_FFI_ICHECK(false)
        << "Unsupported dtype combination for selective_state_update: "
        << "state_dtype=" << state_dtype.code << ":" << state_dtype.bits << ", "
        << "input_dtype=" << input_dtype.code << ":" << input_dtype.bits << ", "
        << "weight_dtype=" << weight_dtype.code << ":" << weight_dtype.bits << ", "
        << "matrixA_dtype=" << matrixA_dtype.code << ":" << matrixA_dtype.bits
        << ". Supported combos include:\n"
        << "  (state=bfloat16, input=bfloat16, weight=bfloat16, matrixA=float32)\n"
        << "  (state=float16, input=bfloat16, weight=bfloat16, matrixA=float32)\n"
        << "  (state=float32, input=bfloat16, weight=bfloat16, matrixA=float32)\n"
        << "  (state=bfloat16, input=bfloat16, weight=float32, matrixA=float32)\n"
        << "  (state=float16, input=bfloat16, weight=float32, matrixA=float32)\n"
        << "  (state=float32, input=bfloat16, weight=float32, matrixA=float32)";
  }
}

}  // namespace flashinfer::mamba
