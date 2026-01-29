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
#include <sstream>

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
  CHECK_INPUT_TYPE(state_batch_indices.value(), dl_int32);
}

inline void validate_intermediate_state_indices(
    Optional<TensorView> const& intermediate_state_indices, int64_t batch) {
  if (!intermediate_state_indices.has_value()) return;
  CHECK_CUDA(intermediate_state_indices.value());
  CHECK_DIM(1, intermediate_state_indices.value());
  CHECK_CONTIGUOUS(intermediate_state_indices.value());
  FLASHINFER_CHECK(intermediate_state_indices.value().size(0) == batch,
                   "intermediate_state_indices.shape must be (", batch, ")");
  CHECK_INPUT_TYPE(intermediate_state_indices.value(), dl_int32);
}

inline void validate_intermediate_states_buffer(
    Optional<TensorView> const& intermediate_states_buffer) {
  if (!intermediate_states_buffer.has_value()) return;
  CHECK_CUDA(intermediate_states_buffer.value());
  CHECK_CONTIGUOUS(intermediate_states_buffer.value());
}

// Validates dtype consistency across tensors
inline void validate_dtypes(
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

// Helper to convert dtype code to string for error messages
inline const char* dtype_code_to_string(int64_t code) {
  if (code == bfloat16_code) return "bfloat16";
  if (code == float16_code) return "float16";
  if (code == float32_code) return "float32";
  return "unknown";
}

// Type traits to map dtype codes to C++ types
template <int64_t code>
struct DTypeToType;

template <>
struct DTypeToType<bfloat16_code> {
  using type = nv_bfloat16;
};
template <>
struct DTypeToType<float16_code> {
  using type = half;
};
template <>
struct DTypeToType<float32_code> {
  using type = float;
};

// Allowed dtype combinations: {state_code, input_code, weight_code, matrixA_code}
constexpr std::tuple<int64_t, int64_t, int64_t, int64_t> allowed_dtype_combos[] = {
    {bfloat16_code, bfloat16_code, bfloat16_code, float32_code},
    {float16_code, bfloat16_code, bfloat16_code, float32_code},
    {float32_code, bfloat16_code, bfloat16_code, float32_code},
    {bfloat16_code, bfloat16_code, float32_code, float32_code},
    {float16_code, bfloat16_code, float32_code, float32_code},
    {float32_code, bfloat16_code, float32_code, float32_code},
};

// Helper to dispatch to the right template instantiation
template <int64_t state_code, int64_t input_code, int64_t weight_code, int64_t matrixA_code>
void dispatchCombo(SelectiveStateUpdateParams& p, cudaStream_t stream) {
  using state_t = typename DTypeToType<state_code>::type;
  using input_t = typename DTypeToType<input_code>::type;
  using weight_t = typename DTypeToType<weight_code>::type;
  using matrixA_t = typename DTypeToType<matrixA_code>::type;
  invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
}

void run_selective_state_update_stp(TensorView const& state, TensorView const& x,
                                    TensorView const& dt, TensorView const& A, TensorView const& B,
                                    TensorView const& C, TensorView const& D,
                                    Optional<TensorView> z, Optional<TensorView> dt_bias,
                                    bool dt_softplus, Optional<TensorView> state_batch_indices,
                                    int64_t pad_slot_id, Optional<TensorView> out,
                                    bool disable_state_update) {
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
  validate_dtypes(state, dt, D, x, B, C, dt_bias, z, out);

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

  // Dispatch based on dtype combination
  DLDataType state_dtype = state.dtype();
  DLDataType input_dtype = x.dtype();
  DLDataType weight_dtype = dt.dtype();
  DLDataType matrixA_dtype = A.dtype();

  int64_t state_dtype_code = encode_dlpack_dtype(state_dtype);
  int64_t input_dtype_code = encode_dlpack_dtype(input_dtype);
  int64_t weight_dtype_code = encode_dlpack_dtype(weight_dtype);
  int64_t matrixA_dtype_code = encode_dlpack_dtype(matrixA_dtype);

  // Dispatch kernel based on dtype combination
  auto dtype_key =
      std::make_tuple(state_dtype_code, input_dtype_code, weight_dtype_code, matrixA_dtype_code);

  // Compile-time recursive dispatcher using Y-combinator pattern for lambda self-recursion
  auto tryDispatch = [&](const auto& key, auto idx, auto& self) -> bool {
    constexpr size_t I = decltype(idx)::value;
    if constexpr (I < std::size(allowed_dtype_combos)) {
      constexpr auto combo = allowed_dtype_combos[I];
      if (key == combo) {
        constexpr auto s = std::get<0>(combo);
        constexpr auto i = std::get<1>(combo);
        constexpr auto w = std::get<2>(combo);
        constexpr auto m = std::get<3>(combo);
        dispatchCombo<s, i, w, m>(p, stream);
        return true;
      }
      return self(key, std::integral_constant<size_t, I + 1>{}, self);
    }
    return false;
  };

  // Dispatch using compile-time type traits
  if (!tryDispatch(dtype_key, std::integral_constant<size_t, 0>{}, tryDispatch)) {
    // Unsupported dtype combination - build error message dynamically
    std::ostringstream error_msg;
    error_msg << "Unsupported dtype combination for selective_state_update: "
              << "state_dtype=" << state_dtype.code << ":" << state_dtype.bits << ", "
              << "input_dtype=" << input_dtype.code << ":" << input_dtype.bits << ", "
              << "weight_dtype=" << weight_dtype.code << ":" << weight_dtype.bits << ", "
              << "matrixA_dtype=" << matrixA_dtype.code << ":" << matrixA_dtype.bits
              << ". Supported combos include:\n";
    for (const auto& combo : allowed_dtype_combos) {
      error_msg << "  (state=" << dtype_code_to_string(std::get<0>(combo))
                << ", input=" << dtype_code_to_string(std::get<1>(combo))
                << ", weight=" << dtype_code_to_string(std::get<2>(combo))
                << ", matrixA=" << dtype_code_to_string(std::get<3>(combo)) << ")\n";
    }
    TVM_FFI_ICHECK(false) << error_msg.str();
  }
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
                            Optional<TensorView> intermediate_state_indices, int64_t cache_steps) {
  if (x.dim() == 3) {
    run_selective_state_update_stp(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus,
                                   state_batch_indices, pad_slot_id, output, disable_state_update);
  } else if (x.dim() == 4) {
    FLASHINFER_CHECK(false, "x must have 3 dimensions (single-token), got ", x.dim());
    // run_selective_state_update_mtp(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus,
    //                                state_batch_indices, pad_slot_id, output,
    //                                disable_state_update, intermediate_states_buffer,
    //                                intermediate_state_indices, cache_steps);
  } else {
    FLASHINFER_CHECK(false,
                     "x must have 3 dimensions (single-token) or 4 dimensions (multi-token), got ",
                     x.dim());
  }
}

// Old function - commented out for reference
// void selective_state_update_old(TensorView state, TensorView x, TensorView dt, TensorView
// output,
//                             TensorView A, TensorView B, TensorView C, TensorView D,
//                             Optional<TensorView> z, Optional<TensorView> dt_bias, bool
//                             dt_softplus, Optional<TensorView> state_batch_indices, int64_t
//                             pad_slot_id) {
//   auto const batch = x.size(0);
//   auto const state_cache_size = state.size(0);
//   auto const nheads = state.size(1);
//   auto const dim = state.size(2);
//   auto const dstate = state.size(3);
//   auto const ngroups = B.size(1);

//   FLASHINFER_CHECK(state_cache_size >= batch, "state.size(0) must be >= x.size(0)");

//   FLASHINFER_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");

//   // Check x shape and strides
//   CHECK_DIM(3, x);
//   FLASHINFER_CHECK(x.size(1) == nheads, "x.size(1) must equal nheads");
//   FLASHINFER_CHECK(x.size(2) == dim, "x.size(2) must equal dim");
//   CHECK_LAST_DIM_CONTIGUOUS_INPUT(x);
//   FLASHINFER_CHECK(x.stride(1) == dim, "x.stride(1) must equal dim, got ", x.stride(1),
//                    " expected ", x.size(2));

//   // Check output shape and strides
//   CHECK_DIM(3, output);
//   CHECK_LAST_DIM_CONTIGUOUS(output);
//   FLASHINFER_CHECK(output.size(1) == nheads, "output.size(1) must equal nheads");
//   FLASHINFER_CHECK(output.size(2) == dim, "output.size(2) must equal dim");
//   FLASHINFER_CHECK(output.stride(1) == dim, "output.stride(1) must equal dim");

//   // Check dt shape and strides
//   CHECK_CUDA(dt);
//   CHECK_DIM(3, dt);  // dt: {batch, nheads, dim}
//   FLASHINFER_CHECK(dt.size(1) == nheads, "dt.size(1) must equal nheads");
//   FLASHINFER_CHECK(dt.size(2) == dim, "dt.size(2) must equal dim");
//   FLASHINFER_CHECK(dt.stride(1) == 1, "dt.stride(1) must be 1, got ", dt.stride(1));
//   FLASHINFER_CHECK(dt.stride(2) == 0, "dt.stride(2) must be 0 (broadcasted), got ",
//   dt.stride(2));

//   // Check state - fully contiguous
//   CHECK_INPUT(state);   // CUDA + fully contiguous (uses TVM FFI)
//   CHECK_DIM(4, state);  // state: {state_cache_size, nheads, dim, dstate}

//   // Check B shape and strides
//   CHECK_CUDA(B);
//   CHECK_DIM(3, B);  // B: {batch, B.size(1), dstate}
//   FLASHINFER_CHECK(B.size(0) == batch, "B.size(0) must equal batch");
//   FLASHINFER_CHECK(B.size(1) == ngroups, "B.size(1) must equal ngroups");
//   FLASHINFER_CHECK(B.size(2) == dstate, "B.size(2) must equal dstate");
//   CHECK_LAST_DIM_CONTIGUOUS(B);  // stride(2) == 1
//   FLASHINFER_CHECK(B.stride(1) == B.size(2), "B.stride(1) must equal dstate, got ", B.stride(1),
//                    " expected ", B.size(2));

//   // Check C shape and strides
//   CHECK_CUDA(C);
//   CHECK_LAST_DIM_CONTIGUOUS(C);  // stride(2) == 1
//   CHECK_DIM(3, C);               // C: {batch, C.size(1), dstate}
//   FLASHINFER_CHECK(C.stride(1) == C.size(2), "C.stride(1) must equal dstate, got ", C.stride(1),
//                    " expected ", C.size(2));
//   FLASHINFER_CHECK(C.size(0) == batch, "C.size(0) must equal batch");
//   FLASHINFER_CHECK(C.size(1) == ngroups, "C.size(1) must equal ngroups");
//   FLASHINFER_CHECK(C.size(2) == dstate, "C.size(2) must equal dstate");

//   // Check D - specific stride patterns indicating broadcasting
//   CHECK_CUDA(D);
//   CHECK_DIM(2, D);  // D: {nheads, dim}
//   FLASHINFER_CHECK(D.size(0) == nheads, "D.size(0) must equal nheads");
//   FLASHINFER_CHECK(D.size(1) == dim, "D.size(1) must equal dim");
//   FLASHINFER_CHECK(D.stride(0) == 1, "D.stride(0) must be 1, got ", D.stride(0));
//   FLASHINFER_CHECK(D.stride(1) == 0, "D.stride(1) must be 0 (broadcasted), got ", D.stride(1));

//   // Check A - specific stride patterns indicating broadcasting
//   CHECK_CUDA(A);
//   CHECK_DIM(3, A);  // A: {nheads, dim, dstate}
//   FLASHINFER_CHECK(A.size(0) == nheads, "A.size(0) must equal nheads");
//   FLASHINFER_CHECK(A.size(1) == dim, "A.size(1) must equal dim");
//   FLASHINFER_CHECK(A.size(2) == dstate, "A.size(2) must equal dstate");
//   FLASHINFER_CHECK(A.stride(1) == 0, "A.stride(1) must be 0 (broadcasted), got ", A.stride(1));
//   FLASHINFER_CHECK(A.stride(2) == 0, "A.stride(2) must be 0 (broadcasted), got ", A.stride(2));

//   // Optional dt_bias check
//   if (dt_bias.has_value()) {
//     auto& bias = dt_bias.value();
//     CHECK_CUDA(bias);
//     CHECK_DIM(2, bias);  // dt_bias: {nheads, dim}
//     FLASHINFER_CHECK(bias.size(0) == nheads, "dt_bias.size(0) must equal nheads");
//     FLASHINFER_CHECK(bias.size(1) == dim, "dt_bias.size(1) must equal dim");
//     FLASHINFER_CHECK(bias.stride(0) == 1, "dt_bias.stride(0) must be 1, got ", bias.stride(0));
//     FLASHINFER_CHECK(bias.stride(1) == 0, "dt_bias.stride(1) must be 0 (broadcasted), got ",
//                      bias.stride(1));
//   }

//   if (z.has_value()) {
//     auto& z_tensor = z.value();
//     CHECK_CUDA(z_tensor);
//     CHECK_DIM(3, z_tensor);  // z: {batch, nheads, dim}
//     FLASHINFER_CHECK(z_tensor.size(0) == batch, "z.size(0) must equal batch");
//     FLASHINFER_CHECK(z_tensor.size(1) == nheads, "z.size(1) must equal nheads");
//     FLASHINFER_CHECK(z_tensor.size(2) == dim, "z.size(2) must equal dim");
//     CHECK_LAST_DIM_CONTIGUOUS_INPUT(z_tensor);
//     FLASHINFER_CHECK(z_tensor.stride(1) == dim, "z.stride(1) must equal dim, got ",
//                      z_tensor.stride(1), " expected ", z_tensor.size(2));
//   }

//   if (state_batch_indices) {
//     CHECK_DIM(1, (*state_batch_indices));
//     FLASHINFER_CHECK(state_batch_indices.value().size(0) == batch,
//                      "state_batch_indices.shape must be (", batch, ")");
//   }

//   SelectiveStateUpdateParams p;

//   // copy dimensions
//   p.batch = batch;
//   p.nheads = nheads;
//   p.dim = dim;
//   p.dstate = dstate;
//   p.ngroups = ngroups;
//   p.state_cache_size = state_cache_size;
//   p.dt_softplus = dt_softplus;
//   p.pad_slot_id = pad_slot_id;

//   // Copy strides
//   p.x_stride_batch = x.stride(0);
//   p.dt_stride_batch = dt.stride(0);
//   p.B_stride_batch = B.stride(0);
//   p.C_stride_batch = C.stride(0);
//   p.out_stride_batch = output.stride(0);
//   if (state_batch_indices) p.state_batch_indices = state_batch_indices.value().data_ptr();

//   // Copy pointers
//   p.state = state.data_ptr();
//   p.x = x.data_ptr();
//   p.dt = dt.data_ptr();
//   p.output = output.data_ptr();
//   if (dt_bias) {
//     p.dt_bias = dt_bias.value().data_ptr();
//   }
//   if (z) {
//     p.z = z.value().data_ptr();
//     p.z_stride_batch = z.value().stride(0);
//   }
//   p.A = A.data_ptr();
//   p.B = B.data_ptr();
//   p.C = C.data_ptr();
//   p.D = D.data_ptr();

//   // Set device and get stream
//   ffi::CUDADeviceGuard device_guard(state.device().device_id);
//   const cudaStream_t stream = get_stream(state.device());

//   // Dispatch based on dtype combination
//   DLDataType state_dtype = state.dtype();
//   DLDataType input_dtype = x.dtype();
//   DLDataType weight_dtype = dt.dtype();
//   DLDataType matrixA_dtype = A.dtype();

//   int64_t state_dtype_code = encode_dlpack_dtype(state_dtype);
//   int64_t input_dtype_code = encode_dlpack_dtype(input_dtype);
//   int64_t weight_dtype_code = encode_dlpack_dtype(weight_dtype);
//   int64_t matrixA_dtype_code = encode_dlpack_dtype(matrixA_dtype);

//   // Pack all dtype codes into a single value for switching
//   auto dtype_key =
//       std::make_tuple(state_dtype_code, input_dtype_code, weight_dtype_code, matrixA_dtype_code);

//   if (dtype_key == std::make_tuple(bfloat16_code, bfloat16_code, bfloat16_code, float32_code)) {
//     using state_t = nv_bfloat16;
//     using input_t = nv_bfloat16;
//     using weight_t = nv_bfloat16;
//     using matrixA_t = float;
//     invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
//   } else if (dtype_key ==
//              std::make_tuple(float16_code, bfloat16_code, bfloat16_code, float32_code)) {
//     using state_t = half;
//     using input_t = nv_bfloat16;
//     using weight_t = nv_bfloat16;
//     using matrixA_t = float;
//     invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
//   } else if (dtype_key ==
//              std::make_tuple(float32_code, bfloat16_code, bfloat16_code, float32_code)) {
//     using state_t = float;
//     using input_t = nv_bfloat16;
//     using weight_t = nv_bfloat16;
//     using matrixA_t = float;
//     invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
//   } else if (dtype_key ==
//              std::make_tuple(bfloat16_code, bfloat16_code, float32_code, float32_code)) {
//     using state_t = nv_bfloat16;
//     using input_t = nv_bfloat16;
//     using weight_t = float;
//     using matrixA_t = float;
//     invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
//   } else if (dtype_key ==
//              std::make_tuple(float16_code, bfloat16_code, float32_code, float32_code)) {
//     using state_t = half;
//     using input_t = nv_bfloat16;
//     using weight_t = float;
//     using matrixA_t = float;
//     invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
//   } else if (dtype_key ==
//              std::make_tuple(float32_code, bfloat16_code, float32_code, float32_code)) {
//     using state_t = float;
//     using input_t = nv_bfloat16;
//     using weight_t = float;
//     using matrixA_t = float;
//     invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t>(p, stream);
//   } else {
//     // Default case: unsupported dtype combination
//     TVM_FFI_ICHECK(false)
//         << "Unsupported dtype combination for selective_state_update: "
//         << "state_dtype=" << state_dtype.code << ":" << state_dtype.bits << ", "
//         << "input_dtype=" << input_dtype.code << ":" << input_dtype.bits << ", "
//         << "weight_dtype=" << weight_dtype.code << ":" << weight_dtype.bits << ", "
//         << "matrixA_dtype=" << matrixA_dtype.code << ":" << matrixA_dtype.bits
//         << ". Supported combos include:\n"
//         << "  (state=bfloat16, input=bfloat16, weight=bfloat16, matrixA=float32)\n"
//         << "  (state=float16, input=bfloat16, weight=bfloat16, matrixA=float32)\n"
//         << "  (state=float32, input=bfloat16, weight=bfloat16, matrixA=float32)\n"
//         << "  (state=bfloat16, input=bfloat16, weight=float32, matrixA=float32)\n"
//         << "  (state=float16, input=bfloat16, weight=float32, matrixA=float32)\n"
//         << "  (state=float32, input=bfloat16, weight=float32, matrixA=float32)";
//   }
// }

}  // namespace flashinfer::mamba
