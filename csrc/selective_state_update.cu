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
// clang-format off
// config.inc MUST come before the header: it defines DIM, DSTATE, NTOKENS_MTP
// constexprs that the header's function templates rely on. Reordering breaks compilation.
// NOTE: the .inc file is generated from the jinja templates
#include "selective_state_update_config.inc"
#include <flashinfer/mamba/selective_state_update.cuh>
// clang-format on
#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

namespace flashinfer::mamba {

static inline void validate_state_tensor(TensorView const& state) {
  CHECK_CUDA(state);
  CHECK_DIM(4, state);  // state: {state_cache_size, nheads, dim, dstate}
  // Check that dimensions 1, 2, 3 are contiguous (batch dimension can be non-contiguous)
  auto strides = state.strides();
  auto sizes = state.sizes();
  FLASHINFER_CHECK(strides[3] == 1, "state dimension 3 (dstate) must have stride 1");
  FLASHINFER_CHECK(strides[2] == sizes[3],
                   "state dimension 2 (dim) must be contiguous with dimension 3");
  FLASHINFER_CHECK(strides[1] == sizes[2] * sizes[3],
                   "state dimension 1 (nheads) must be contiguous with dimension 2");
}

inline void validate_D_tensor(TensorView const& D, int64_t nheads, int64_t dim) {
  CHECK_CUDA(D);
  CHECK_DIM(2, D);  // D: {nheads, dim}
  FLASHINFER_CHECK(D.size(0) == nheads, "D.size(0) must equal nheads");
  FLASHINFER_CHECK(D.size(1) == dim, "D.size(1) must equal dim");
  FLASHINFER_CHECK(D.stride(0) == 1, "D.stride(0) must be 1, got ", D.stride(0));
  FLASHINFER_CHECK(D.stride(1) == 0, "D.stride(1) must be 0 (broadcasted), got ", D.stride(1));
}

inline void validate_A_tensor(TensorView const& A, int64_t nheads, int64_t dim, int64_t dstate) {
  CHECK_CUDA(A);
  CHECK_DIM(3, A);  // A: {nheads, dim, dstate}
  FLASHINFER_CHECK(A.size(0) == nheads, "A.size(0) must equal nheads");
  FLASHINFER_CHECK(A.size(1) == dim, "A.size(1) must equal dim");
  FLASHINFER_CHECK(A.size(2) == dstate, "A.size(2) must equal dstate");
  FLASHINFER_CHECK(A.stride(0) == 1, "A.stride(0) must be 1, got ", A.stride(0));
  FLASHINFER_CHECK(A.stride(1) == 0, "A.stride(1) must be 0 (broadcasted), got ", A.stride(1));
  FLASHINFER_CHECK(A.stride(2) == 0, "A.stride(2) must be 0 (broadcasted), got ", A.stride(2));
}

inline void validate_dt_bias_tensor(Optional<TensorView> const& dt_bias, int64_t nheads,
                                    int64_t dim) {
  if (!dt_bias.has_value()) return;
  auto const& bias = dt_bias.value();
  CHECK_CUDA(bias);
  CHECK_DIM(2, bias);  // dt_bias: {nheads, dim}
  FLASHINFER_CHECK(bias.size(0) == nheads, "dt_bias.size(0) must equal nheads");
  FLASHINFER_CHECK(bias.size(1) == dim, "dt_bias.size(1) must equal dim");
  FLASHINFER_CHECK(bias.stride(0) == 1, "dt_bias.stride(0) must be 1, got ", bias.stride(0));
  FLASHINFER_CHECK(bias.stride(1) == 0, "dt_bias.stride(1) must be 0 (broadcasted), got ",
                   bias.stride(1));
}

inline void validate_state_batch_indices(Optional<TensorView> const& state_batch_indices,
                                         int64_t batch) {
  if (!state_batch_indices.has_value()) return;
  CHECK_DIM(1, (*state_batch_indices));
  CHECK_CONTIGUOUS((*state_batch_indices));
  FLASHINFER_CHECK(state_batch_indices.value().size(0) == batch,
                   "state_batch_indices.shape must be (", batch, ")");
}

inline void validate_intermediate_state_indices(
    Optional<TensorView> const& intermediate_state_indices, int64_t batch) {
  if (!intermediate_state_indices.has_value()) return;
  CHECK_CUDA(intermediate_state_indices.value());
  CHECK_DIM(1, intermediate_state_indices.value());
  CHECK_CONTIGUOUS(intermediate_state_indices.value());
  FLASHINFER_CHECK(intermediate_state_indices.value().size(0) == batch,
                   "intermediate_state_indices.shape must be (", batch, ")");
}

inline void validate_intermediate_states_buffer(
    Optional<TensorView> const& intermediate_states_buffer) {
  if (!intermediate_states_buffer.has_value()) return;
  CHECK_CUDA(intermediate_states_buffer.value());
  CHECK_CONTIGUOUS(intermediate_states_buffer.value());
}

// Validates dtype consistency across tensors
inline void validate_dtype_consistency(
    TensorView const& state, TensorView const& dt, TensorView const& D, TensorView const& x,
    TensorView const& B, TensorView const& C, Optional<TensorView> const& dt_bias,
    Optional<TensorView> const& z = Optional<TensorView>(),
    Optional<TensorView> const& out = Optional<TensorView>(),
    Optional<TensorView> const& intermediate_states_buffer = Optional<TensorView>()) {
  auto state_dtype = state.dtype();
  auto weight_dtype = dt.dtype();
  auto input_dtype = x.dtype();
  FLASHINFER_CHECK(D.dtype() == weight_dtype, "D must have the same dtype as dt");
  FLASHINFER_CHECK(B.dtype() == input_dtype, "B must have the same dtype as x");
  FLASHINFER_CHECK(C.dtype() == input_dtype, "C must have the same dtype as x");
  if (dt_bias.has_value()) {
    FLASHINFER_CHECK(dt_bias.value().dtype() == weight_dtype,
                     "dt_bias must have the same dtype as dt");
  }
  if (z.has_value()) {
    FLASHINFER_CHECK(z.value().dtype() == input_dtype, "z must have the same dtype as x");
  }
  if (out.has_value()) {
    FLASHINFER_CHECK(out.value().dtype() == input_dtype, "out must have the same dtype as x");
  }
  if (intermediate_states_buffer.has_value()) {
    FLASHINFER_CHECK(intermediate_states_buffer.value().dtype() == state_dtype,
                     "intermediate_states_buffer must have the same dtype as state");
  }
}

void run_selective_state_update_stp(TensorView const& state, TensorView const& x,
                                    TensorView const& dt, TensorView const& A, TensorView const& B,
                                    TensorView const& C, TensorView const& D,
                                    Optional<TensorView> z, Optional<TensorView> dt_bias,
                                    bool dt_softplus, Optional<TensorView> state_batch_indices,
                                    int64_t pad_slot_id, Optional<TensorView> out,
                                    bool disable_state_update, int64_t algorithm) {
  // Extract dimensions from input tensors
  auto const batch = x.size(0);
  auto const state_cache_size = state.size(0);
  auto const nheads = state.size(1);
  auto const dim = state.size(2);
  auto const dstate = state.size(3);
  auto const ngroups = B.size(1);

  FLASHINFER_CHECK(state_cache_size >= batch, "state.size(0) must be >= x.size(0)");
  FLASHINFER_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");

  // Check x shape and strides
  CHECK_CUDA(x);
  CHECK_DIM(3, x);
  FLASHINFER_CHECK(x.size(1) == nheads, "x.size(1) must equal nheads");
  FLASHINFER_CHECK(x.size(2) == dim, "x.size(2) must equal dim");
  CHECK_LAST_DIM_CONTIGUOUS(x);
  FLASHINFER_CHECK(x.stride(1) == dim, "x.stride(1) must equal dim, got ", x.stride(1),
                   " expected ", dim);

  // Check dt shape and strides
  CHECK_CUDA(dt);
  CHECK_DIM(3, dt);  // dt: {batch, nheads, dim}
  FLASHINFER_CHECK(dt.size(0) == batch, "dt.size(0) must equal batch");
  FLASHINFER_CHECK(dt.size(1) == nheads, "dt.size(1) must equal nheads");
  FLASHINFER_CHECK(dt.size(2) == dim, "dt.size(2) must equal dim");
  FLASHINFER_CHECK(dt.stride(1) == 1, "dt.stride(1) must be 1, got ", dt.stride(1));
  FLASHINFER_CHECK(dt.stride(2) == 0, "dt.stride(2) must be 0 (broadcasted), got ", dt.stride(2));

  // Validate common tensors using helper functions
  validate_state_tensor(state);
  validate_D_tensor(D, nheads, dim);
  validate_A_tensor(A, nheads, dim, dstate);
  validate_dt_bias_tensor(dt_bias, nheads, dim);
  validate_state_batch_indices(state_batch_indices, batch);

  // Check B shape and strides
  CHECK_CUDA(B);
  CHECK_DIM(3, B);  // B: {batch, ngroups, dstate}
  FLASHINFER_CHECK(B.size(0) == batch, "B.size(0) must equal batch");
  FLASHINFER_CHECK(B.size(1) == ngroups, "B.size(1) must equal ngroups");
  FLASHINFER_CHECK(B.size(2) == dstate, "B.size(2) must equal dstate");
  CHECK_LAST_DIM_CONTIGUOUS(B);
  FLASHINFER_CHECK(B.stride(1) == dstate, "B.stride(1) must equal dstate, got ", B.stride(1),
                   " expected ", dstate);

  // Check C shape and strides
  CHECK_CUDA(C);
  CHECK_DIM(3, C);  // C: {batch, ngroups, dstate}
  FLASHINFER_CHECK(C.size(0) == batch, "C.size(0) must equal batch");
  FLASHINFER_CHECK(C.size(1) == ngroups, "C.size(1) must equal ngroups");
  FLASHINFER_CHECK(C.size(2) == dstate, "C.size(2) must equal dstate");
  CHECK_LAST_DIM_CONTIGUOUS(C);
  FLASHINFER_CHECK(C.stride(1) == dstate, "C.stride(1) must equal dstate, got ", C.stride(1),
                   " expected ", dstate);

  // Optional z check
  if (z.has_value()) {
    auto& z_tensor = z.value();
    CHECK_CUDA(z_tensor);
    CHECK_DIM(3, z_tensor);  // z: {batch, nheads, dim}
    FLASHINFER_CHECK(z_tensor.size(0) == batch, "z.size(0) must equal batch");
    FLASHINFER_CHECK(z_tensor.size(1) == nheads, "z.size(1) must equal nheads");
    FLASHINFER_CHECK(z_tensor.size(2) == dim, "z.size(2) must equal dim");
    CHECK_LAST_DIM_CONTIGUOUS(z_tensor);
    FLASHINFER_CHECK(z_tensor.stride(1) == dim, "z.stride(1) must equal dim, got ",
                     z_tensor.stride(1), " expected ", dim);
  }

  // Check output tensor if provided
  if (out.has_value()) {
    auto& output = out.value();
    CHECK_CUDA(output);
    CHECK_CONTIGUOUS(output);
    CHECK_DIM(3, output);
    FLASHINFER_CHECK(output.size(0) == batch, "out.size(0) must equal batch");
    FLASHINFER_CHECK(output.size(1) == nheads, "out.size(1) must equal nheads");
    FLASHINFER_CHECK(output.size(2) == dim, "out.size(2) must equal dim");
    CHECK_LAST_DIM_CONTIGUOUS(output);
    FLASHINFER_CHECK(output.stride(1) == dim, "out.stride(1) must equal dim");
  }

  // Validate dtype consistency
  validate_dtype_consistency(state, dt, D, x, B, C, dt_bias, z, out);

  // Initialize params struct
  SelectiveStateUpdateParams p;

  // Copy dimensions
  p.batch = batch;
  p.nheads = nheads;
  p.dim = dim;
  p.dstate = dstate;
  p.ngroups = ngroups;
  p.state_cache_size = state_cache_size;
  p.dt_softplus = dt_softplus;
  p.pad_slot_id = pad_slot_id;
  p.update_state = !disable_state_update;

  // Copy strides
  p.x_stride_batch = x.stride(0);
  p.dt_stride_batch = dt.stride(0);
  p.B_stride_batch = B.stride(0);
  p.C_stride_batch = C.stride(0);
  if (out.has_value()) {
    p.out_stride_batch = out.value().stride(0);
  } else {
    p.out_stride_batch = 0;
  }
  p.state_stride_batch = state.stride(0);
  if (state_batch_indices.has_value()) {
    p.state_batch_indices = const_cast<void*>(state_batch_indices.value().data_ptr());
  }

  // Copy pointers
  p.state = const_cast<void*>(state.data_ptr());
  p.x = const_cast<void*>(x.data_ptr());
  p.dt = const_cast<void*>(dt.data_ptr());
  if (out.has_value()) {
    p.output = out.value().data_ptr();
  } else {
    p.output = nullptr;
  }
  if (dt_bias.has_value()) {
    p.dt_bias = const_cast<void*>(dt_bias.value().data_ptr());
  }
  if (z.has_value()) {
    p.z = const_cast<void*>(z.value().data_ptr());
    p.z_stride_batch = z.value().stride(0);
  }
  p.A = const_cast<void*>(A.data_ptr());
  p.B = const_cast<void*>(B.data_ptr());
  p.C = const_cast<void*>(C.data_ptr());
  p.D = const_cast<void*>(D.data_ptr());

  // Set device and get stream
  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  const cudaStream_t stream = get_stream(state.device());

  auto algo = static_cast<SSUAlgorithm>(algorithm);
  invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t, stateIndex_t>(p, algo, stream);
}

void run_selective_state_update_mtp(
    TensorView const& state, TensorView const& x, TensorView const& dt, TensorView const& A,
    TensorView const& B, TensorView const& C, TensorView const& D, Optional<TensorView> z,
    Optional<TensorView> dt_bias, bool dt_softplus, Optional<TensorView> state_batch_indices,
    int64_t pad_slot_id, Optional<TensorView> out, bool disable_state_update,
    Optional<TensorView> intermediate_states_buffer,
    Optional<TensorView> intermediate_state_indices, int64_t cache_steps, int64_t algorithm) {
  // Extract dimensions from input tensors
  auto const batch = x.size(0);
  auto const ntokens_mtp = x.size(1);
  auto const state_cache_size = state.size(0);
  auto const nheads = state.size(1);
  auto const dim = state.size(2);
  auto const dstate = state.size(3);
  auto const ngroups = B.size(2);

  FLASHINFER_CHECK(state_cache_size >= batch, "state.size(0) must be >= x.size(0)");
  FLASHINFER_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");

  // Check x shape and strides
  CHECK_CUDA(x);
  CHECK_DIM(4, x);
  FLASHINFER_CHECK(x.size(2) == nheads, "x.size(2) must equal nheads");
  FLASHINFER_CHECK(x.size(3) == dim, "x.size(3) must equal dim");
  CHECK_LAST_DIM_CONTIGUOUS(x);
  FLASHINFER_CHECK(x.stride(2) == dim, "x.stride(2) must equal dim, got ", x.stride(2),
                   " expected ", dim);

  // Check dt shape and strides
  CHECK_CUDA(dt);
  CHECK_DIM(4, dt);  // dt: {batch, ntokens_mtp, nheads, dim}
  FLASHINFER_CHECK(dt.size(0) == batch, "dt.size(0) must equal batch =", batch);
  FLASHINFER_CHECK(dt.size(1) == ntokens_mtp, "dt.size(1) must equal ntokens_mtp =", ntokens_mtp);
  FLASHINFER_CHECK(dt.size(2) == nheads, "dt.size(2) must equal nheads");
  FLASHINFER_CHECK(dt.size(3) == dim, "dt.size(3) must equal dim");
  FLASHINFER_CHECK(dt.stride(2) == 1, "dt.stride(2) must be 1, got ", dt.stride(2));
  FLASHINFER_CHECK(dt.stride(3) == 0, "dt.stride(3) must be 0 (broadcasted), got ", dt.stride(3));

  // Validate common tensors using helper functions
  validate_state_tensor(state);
  validate_D_tensor(D, nheads, dim);
  validate_A_tensor(A, nheads, dim, dstate);
  validate_dt_bias_tensor(dt_bias, nheads, dim);
  validate_state_batch_indices(state_batch_indices, batch);

  // Check B shape and strides
  CHECK_CUDA(B);
  CHECK_DIM(4, B);  // B: {batch, ntokens_mtp, ngroups, dstate}
  FLASHINFER_CHECK(B.size(0) == batch, "B.size(0) must equal batch =", batch);
  FLASHINFER_CHECK(B.size(1) == ntokens_mtp, "B.size(1) must equal ntokens_mtp =", ntokens_mtp);
  FLASHINFER_CHECK(B.size(2) == ngroups, "B.size(2) must equal ngroups =", ngroups);
  FLASHINFER_CHECK(B.size(3) == dstate, "B.size(3) must equal dstate =", dstate);
  CHECK_LAST_DIM_CONTIGUOUS(B);
  FLASHINFER_CHECK(B.stride(2) == dstate, "B.stride(2) must equal dstate, got ", B.stride(2),
                   " expected ", dstate);

  // Check C shape and strides
  CHECK_CUDA(C);
  CHECK_DIM(4, C);  // C: {batch, ntokens_mtp, ngroups, dstate}
  FLASHINFER_CHECK(C.size(0) == batch, "C.size(0) must equal batch");
  FLASHINFER_CHECK(C.size(1) == ntokens_mtp, "C.size(1) must equal ntokens_mtp =", ntokens_mtp);
  FLASHINFER_CHECK(C.size(2) == ngroups, "C.size(2) must equal ngroups");
  FLASHINFER_CHECK(C.size(3) == dstate, "C.size(3) must equal dstate");
  CHECK_LAST_DIM_CONTIGUOUS(C);
  FLASHINFER_CHECK(C.stride(2) == dstate, "C.stride(2) must equal dstate, got ", C.stride(2),
                   " expected ", dstate);

  // Optional z check
  if (z.has_value()) {
    auto& z_tensor = z.value();
    CHECK_CUDA(z_tensor);
    CHECK_DIM(4, z_tensor);  // z: {batch, ntokens_mtp, nheads, dim}
    FLASHINFER_CHECK(z_tensor.size(0) == batch, "z.size(0) must equal batch");
    FLASHINFER_CHECK(z_tensor.size(1) == ntokens_mtp, "z.size(1) must equal ntokens_mtp");
    FLASHINFER_CHECK(z_tensor.size(2) == nheads, "z.size(2) must equal nheads");
    FLASHINFER_CHECK(z_tensor.size(3) == dim, "z.size(3) must equal dim");
    CHECK_LAST_DIM_CONTIGUOUS(z_tensor);
    FLASHINFER_CHECK(z_tensor.stride(2) == dim, "z.stride(2) must equal dim, got ",
                     z_tensor.stride(2), " expected ", dim);
  }

  // Check output tensor if provided
  if (out.has_value()) {
    auto& output = out.value();
    CHECK_CUDA(output);
    CHECK_DIM(4, output);
    FLASHINFER_CHECK(output.size(0) == batch, "out.size(0) must equal batch = ", batch);
    FLASHINFER_CHECK(output.size(1) == ntokens_mtp,
                     "out.size(1) must equal ntokens_mtp = ", ntokens_mtp);
    FLASHINFER_CHECK(output.size(2) == nheads, "out.size(2) must equal nheads = ", nheads);
    FLASHINFER_CHECK(output.size(3) == dim, "out.size(3) must equal dim = ", dim);
    CHECK_LAST_DIM_CONTIGUOUS(output);
    FLASHINFER_CHECK(output.stride(2) == dim, "out.stride(2) = ", output.stride(2),
                     " must equal dim = ", dim);
  }

  // Validate dtype consistency
  validate_dtype_consistency(state, dt, D, x, B, C, dt_bias, z, out, intermediate_states_buffer);
  validate_intermediate_state_indices(intermediate_state_indices, batch);
  validate_intermediate_states_buffer(intermediate_states_buffer);

  // Validate that state_batch_indices and intermediate_state_indices have the same dtype
  if (state_batch_indices.has_value() && intermediate_state_indices.has_value()) {
    DLDataType state_batch_idx_dtype = state_batch_indices.value().dtype();
    DLDataType intermediate_idx_dtype = intermediate_state_indices.value().dtype();
    FLASHINFER_CHECK(state_batch_idx_dtype.code == intermediate_idx_dtype.code &&
                         state_batch_idx_dtype.bits == intermediate_idx_dtype.bits,
                     "state_batch_indices and intermediate_state_indices must have the same dtype");
  }

  // Validate cache_steps is non-negative
  FLASHINFER_CHECK(cache_steps >= 0, "cache_steps must be non-negative, got ", cache_steps);

  // Initialize MTP params struct
  mtp::SelectiveStateMTPParams p;

  // Copy dimensions (inherited from base)
  p.batch = batch;
  p.nheads = nheads;
  p.dim = dim;
  p.dstate = dstate;
  p.ngroups = ngroups;
  p.state_cache_size = state_cache_size;
  p.dt_softplus = dt_softplus;
  p.pad_slot_id = pad_slot_id;

  // MTP-specific dimensions
  p.ntokens_mtp = ntokens_mtp;
  p.cache_steps = static_cast<uint64_t>(cache_steps);
  p.update_state = !disable_state_update;

  // Copy batch strides (inherited)
  p.x_stride_batch = x.stride(0);
  p.dt_stride_batch = dt.stride(0);
  p.B_stride_batch = B.stride(0);
  p.C_stride_batch = C.stride(0);
  if (out.has_value()) {
    p.out_stride_batch = out.value().stride(0);
  } else {
    p.out_stride_batch = 0;
  }
  p.state_stride_batch = state.stride(0);

  // Copy MTP strides
  p.x_stride_mtp = x.stride(1);
  p.dt_stride_mtp = dt.stride(1);
  p.B_stride_mtp = B.stride(1);
  p.C_stride_mtp = C.stride(1);
  if (out.has_value()) {
    p.out_stride_mtp = out.value().stride(1);
  } else {
    p.out_stride_mtp = 0;
  }

  if (state_batch_indices.has_value()) {
    p.state_batch_indices = const_cast<void*>(state_batch_indices.value().data_ptr());
  }

  if (intermediate_states_buffer.has_value()) {
    p.intermediate_states = const_cast<void*>(intermediate_states_buffer.value().data_ptr());
    p.intermediate_state_stride_batch = intermediate_states_buffer.value().stride(0);
  }

  if (intermediate_state_indices.has_value()) {
    p.intermediate_state_indices = const_cast<void*>(intermediate_state_indices.value().data_ptr());
  }

  // Copy pointers
  p.state = const_cast<void*>(state.data_ptr());
  p.x = const_cast<void*>(x.data_ptr());
  p.dt = const_cast<void*>(dt.data_ptr());
  if (out.has_value()) {
    p.output = out.value().data_ptr();
  } else {
    p.output = nullptr;
  }
  if (dt_bias.has_value()) {
    p.dt_bias = const_cast<void*>(dt_bias.value().data_ptr());
  }
  if (z.has_value()) {
    p.z = const_cast<void*>(z.value().data_ptr());
    p.z_stride_batch = z.value().stride(0);
    p.z_stride_mtp = z.value().stride(1);
  }
  p.A = const_cast<void*>(A.data_ptr());
  p.B = const_cast<void*>(B.data_ptr());
  p.C = const_cast<void*>(C.data_ptr());
  p.D = const_cast<void*>(D.data_ptr());

  // Set device and get stream
  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  const cudaStream_t stream = get_stream(state.device());

  auto algo = static_cast<SSUAlgorithm>(algorithm);
  mtp::invokeSelectiveStateUpdateMTP<input_t, weight_t, matrixA_t, state_t, stateIndex_t>(p, algo,
                                                                                          stream);
}

// =============================================================================
// Generic dispatcher - routes to single-token or multi-token based on x.dim()
// =============================================================================
void selective_state_update(TensorView state, TensorView x, TensorView dt, TensorView A,
                            TensorView B, TensorView C, TensorView D, Optional<TensorView> z,
                            Optional<TensorView> dt_bias, bool dt_softplus,
                            Optional<TensorView> state_batch_indices, int64_t pad_slot_id,
                            TensorView output, bool disable_state_update,
                            Optional<TensorView> intermediate_states_buffer,
                            Optional<TensorView> intermediate_state_indices, int64_t cache_steps,
                            int64_t algorithm) {
  if (x.dim() == 3) {
    run_selective_state_update_stp(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus,
                                   state_batch_indices, pad_slot_id, output, disable_state_update,
                                   algorithm);
  } else if (x.dim() == 4) {
    run_selective_state_update_mtp(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus,
                                   state_batch_indices, pad_slot_id, output, disable_state_update,
                                   intermediate_states_buffer, intermediate_state_indices,
                                   cache_steps, algorithm);
  } else {
    FLASHINFER_CHECK(false,
                     "x must have 3 dimensions (single-token) or 4 dimensions (multi-token), got ",
                     x.dim());
  }
}

}  // namespace flashinfer::mamba
