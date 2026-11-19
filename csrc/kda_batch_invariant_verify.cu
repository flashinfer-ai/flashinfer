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
#include <flashinfer/kda/batch_invariant_verify.cuh>
#include <initializer_list>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

namespace {

template <typename T>
T* ptr(TensorView t) {
  return static_cast<T*>(t.data_ptr());
}

void check(std::initializer_list<TensorView> tensors, DLDataTypeCode code, int bits,
           TensorView on) {
  for (TensorView t : tensors) {
    TVM_FFI_ICHECK(t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1)
        << "a verify buffer has an unexpected dtype";
    CHECK_CONTIGUOUS(t);
    CHECK_DEVICE(t, on);
  }
}

// the kernel moves these 16 bytes at a time
void check_aligned(std::initializer_list<TensorView> tensors) {
  for (TensorView t : tensors) {
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0)
        << "a tensor read 16 bytes at a time does not start at such a boundary";
  }
}

}  // namespace

// has_accepted == 0: every pending verify is kept in full; n == 0: settle only
void kda_batch_invariant_verify_settle(TensorView slots, TensorView accepted, int64_t has_accepted,
                                       int64_t n, TensorView state, TensorView state_bf16,
                                       TensorView cnt, TensorView prefix, TensorView k_inv_rows,
                                       TensorView k_inv_f32_rows, TensorView rhs_rows,
                                       TensorView update_rows, TensorView l_rows, TensorView stamp,
                                       TensorView meta, TensorView v_prefix, TensorView prefix_rows,
                                       TensorView v_state, TensorView v_k, TensorView v_kf,
                                       TensorView v_r, TensorView v_u, TensorView v_l) {
  namespace bv = kda::bi_verify;
  CHECK_DIM(4, state);
  TVM_FFI_ICHECK(state.size(2) == bv::D && state.size(3) == bv::D);
  const int64_t num_slots = state.size(0), num_heads = state.size(1), batch = slots.numel();
  const int64_t sh = num_slots * num_heads, rows = sh * bv::CH * bv::D;
  TVM_FFI_ICHECK(batch > 0 && n >= 0 && n <= bv::CH) << "a verify is at most one chunk of rows";
  TVM_FFI_ICHECK(cnt.numel() == sh && meta.numel() == sh * bv::META);
  TVM_FFI_ICHECK(prefix.numel() == sh * bv::D && v_prefix.numel() == sh * bv::D);
  TVM_FFI_ICHECK(prefix_rows.numel() % (sh * bv::D) == 0);
  const int64_t cap = prefix_rows.numel() / (sh * bv::D);
  TVM_FFI_ICHECK(cap >= 1 && n <= cap) << "more rows per sequence than the verify record holds";
  TVM_FFI_ICHECK(stamp.numel() == sh * bv::STAMP && v_state.numel() == state.numel() &&
                 state_bf16.numel() == state.numel());
  TVM_FFI_ICHECK(k_inv_rows.numel() == rows && k_inv_f32_rows.numel() == rows &&
                 rhs_rows.numel() == rows && update_rows.numel() == rows && v_k.numel() == rows &&
                 v_kf.numel() == rows && v_r.numel() == rows && v_u.numel() == rows);
  TVM_FFI_ICHECK(l_rows.numel() == sh * bv::CH * bv::CH && v_l.numel() == l_rows.numel());
  TVM_FFI_ICHECK(has_accepted == 0 || accepted.numel() == batch);
  check({state, prefix, k_inv_f32_rows, stamp, v_prefix, prefix_rows, v_state, v_kf}, kDLFloat, 32,
        state);
  check({state_bf16, k_inv_rows, rhs_rows, update_rows, l_rows, v_k, v_r, v_u, v_l}, kDLBfloat, 16,
        state);
  check({cnt, meta, slots, accepted}, kDLInt, 32, state);
  check_aligned({state, state_bf16, k_inv_rows, k_inv_f32_rows, rhs_rows, update_rows, l_rows,
                 v_state, v_k, v_kf, v_r, v_u, v_l});

  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  cudaError_t status = bv::Settle(
      ptr<const int>(slots), has_accepted ? ptr<const int>(accepted) : nullptr, static_cast<int>(n),
      ptr<float>(state), ptr<uint16_t>(state_bf16), ptr<int>(cnt), ptr<float>(prefix),
      ptr<uint16_t>(k_inv_rows), ptr<float>(k_inv_f32_rows), ptr<uint16_t>(rhs_rows),
      ptr<uint16_t>(update_rows), ptr<uint16_t>(l_rows), ptr<float>(stamp), ptr<int>(meta),
      ptr<float>(v_prefix), ptr<const float>(prefix_rows), ptr<float>(v_state), ptr<uint16_t>(v_k),
      ptr<float>(v_kf), ptr<uint16_t>(v_r), ptr<uint16_t>(v_u), ptr<uint16_t>(v_l),
      static_cast<int>(cap), static_cast<int>(batch), static_cast<int>(num_heads),
      static_cast<int>(num_slots), get_stream(state.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "kda_batch_invariant_verify_settle failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kda_batch_invariant_verify_settle, kda_batch_invariant_verify_settle);
