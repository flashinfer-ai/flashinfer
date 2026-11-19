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

// both kernels move these 16 bytes at a time
void check_aligned(std::initializer_list<TensorView> tensors) {
  for (TensorView t : tensors) {
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0)
        << "a tensor read 16 bytes at a time does not start at such a boundary";
  }
}

// the shapes of the class's side state and of its verify record, for a pool [slots, heads, D, D]
void check_side(TensorView state, TensorView cnt, TensorView prefix, TensorView k_rows,
                TensorView kf_rows, TensorView r_rows, TensorView u_rows, TensorView l_rows,
                TensorView stamp, TensorView meta, TensorView v_prefix, TensorView v_state,
                TensorView v_k, TensorView v_kf, TensorView v_r, TensorView v_u, TensorView v_l) {
  namespace bv = kda::bi_verify;
  CHECK_DIM(4, state);
  TVM_FFI_ICHECK(state.size(2) == bv::D && state.size(3) == bv::D);
  const int64_t sh = state.size(0) * state.size(1), rows = sh * bv::CH * bv::D;
  TVM_FFI_ICHECK(cnt.numel() == sh && meta.numel() == sh * bv::META);
  TVM_FFI_ICHECK(prefix.numel() == sh * bv::D && v_prefix.numel() == sh * bv::D);
  TVM_FFI_ICHECK(stamp.numel() == sh * bv::STAMP && v_state.numel() == state.numel());
  TVM_FFI_ICHECK(k_rows.numel() == rows && kf_rows.numel() == rows && r_rows.numel() == rows &&
                 u_rows.numel() == rows && v_k.numel() == rows && v_kf.numel() == rows &&
                 v_r.numel() == rows && v_u.numel() == rows);
  TVM_FFI_ICHECK(l_rows.numel() == sh * bv::CH * bv::CH && v_l.numel() == l_rows.numel());
  check({state, prefix, kf_rows, stamp, v_prefix, v_state, v_kf}, kDLFloat, 32, state);
  check({k_rows, r_rows, u_rows, l_rows, v_k, v_r, v_u, v_l}, kDLBfloat, 16, state);
  check({cnt, meta}, kDLInt, 32, state);
  check_aligned({state, prefix, k_rows, kf_rows, r_rows, u_rows, l_rows, v_prefix, v_state, v_k,
                 v_kf, v_r, v_u, v_l});
}

}  // namespace

void kda_batch_invariant_verify_snapshot(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta, TensorView slots,
    TensorView state, TensorView cnt, TensorView prefix, TensorView k_inv_rows,
    TensorView k_inv_f32_rows, TensorView rhs_rows, TensorView update_rows, TensorView l_rows,
    TensorView stamp, TensorView meta, TensorView v_prefix, TensorView v_state, TensorView v_k,
    TensorView v_kf, TensorView v_r, TensorView v_u, TensorView v_l, TensorView t_q, TensorView t_k,
    TensorView t_v, TensorView t_g, TensorView t_beta) {
  namespace bv = kda::bi_verify;
  check_side(state, cnt, prefix, k_inv_rows, k_inv_f32_rows, rhs_rows, update_rows, l_rows, stamp,
             meta, v_prefix, v_state, v_k, v_kf, v_r, v_u, v_l);
  CHECK_DIM(3, q);
  CHECK_DIM(3, t_q);
  const int64_t num_slots = state.size(0), num_heads = state.size(1), batch = slots.numel();
  TVM_FFI_ICHECK(batch > 0 && q.size(0) % batch == 0) << "the rows are not n per sequence";
  const int64_t n = q.size(0) / batch;
  TVM_FFI_ICHECK(q.size(1) == num_heads && q.size(2) == bv::D);
  TVM_FFI_ICHECK(k.numel() == q.numel() && v.numel() == q.numel() && g.numel() == q.numel());
  TVM_FFI_ICHECK(beta.numel() == q.size(0) * num_heads);
  TVM_FFI_ICHECK(t_q.size(0) % num_slots == 0 && t_q.size(1) == num_heads && t_q.size(2) == bv::D);
  const int64_t cap = t_q.size(0) / num_slots;
  TVM_FFI_ICHECK(n >= 1 && n <= cap) << "more rows per sequence than the verify record holds";
  TVM_FFI_ICHECK(t_k.numel() == t_q.numel() && t_v.numel() == t_q.numel() &&
                 t_g.numel() == t_q.numel() && t_beta.numel() == t_q.size(0) * num_heads);
  check({q, k, v, g, beta, t_q, t_k, t_v, t_g, t_beta}, kDLBfloat, 16, state);
  check({slots}, kDLInt, 32, state);
  check_aligned({q, k, v, g, t_q, t_k, t_v, t_g});

  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  cudaError_t status = bv::Snapshot(
      ptr<const uint16_t>(q), ptr<const uint16_t>(k), ptr<const uint16_t>(v),
      ptr<const uint16_t>(g), ptr<const uint16_t>(beta), ptr<const int>(slots), static_cast<int>(n),
      ptr<const float>(state), ptr<const int>(cnt), ptr<const float>(prefix),
      ptr<const uint16_t>(k_inv_rows), ptr<const float>(k_inv_f32_rows),
      ptr<const uint16_t>(rhs_rows), ptr<const uint16_t>(update_rows), ptr<const uint16_t>(l_rows),
      ptr<const float>(stamp), ptr<int>(meta), ptr<float>(v_prefix), ptr<float>(v_state),
      ptr<uint16_t>(v_k), ptr<float>(v_kf), ptr<uint16_t>(v_r), ptr<uint16_t>(v_u),
      ptr<uint16_t>(v_l), ptr<uint16_t>(t_q), ptr<uint16_t>(t_k), ptr<uint16_t>(t_v),
      ptr<uint16_t>(t_g), ptr<uint16_t>(t_beta), static_cast<int>(cap), static_cast<int>(batch),
      static_cast<int>(num_heads), static_cast<int>(num_slots), get_stream(state.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "kda_batch_invariant_verify_snapshot failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kda_batch_invariant_verify_snapshot,
                              kda_batch_invariant_verify_snapshot);

// has_accepted == 0: every sequence keeps all its rows (`accepted` is not read)
void kda_batch_invariant_verify_accept(TensorView slots, TensorView accepted, int64_t has_accepted,
                                       TensorView state, TensorView state_bf16, TensorView cnt,
                                       TensorView prefix, TensorView k_inv_rows,
                                       TensorView k_inv_f32_rows, TensorView rhs_rows,
                                       TensorView update_rows, TensorView l_rows, TensorView stamp,
                                       TensorView meta, TensorView v_prefix, TensorView v_state,
                                       TensorView v_k, TensorView v_kf, TensorView v_r,
                                       TensorView v_u, TensorView v_l, TensorView starts,
                                       TensorView lens, int64_t cap) {
  namespace bv = kda::bi_verify;
  check_side(state, cnt, prefix, k_inv_rows, k_inv_f32_rows, rhs_rows, update_rows, l_rows, stamp,
             meta, v_prefix, v_state, v_k, v_kf, v_r, v_u, v_l);
  const int64_t num_slots = state.size(0), num_heads = state.size(1), batch = slots.numel();
  TVM_FFI_ICHECK(batch > 0 && cap >= 1 && num_slots * cap <= INT32_MAX);
  TVM_FFI_ICHECK(state_bf16.numel() == state.numel());
  TVM_FFI_ICHECK(starts.numel() == batch && lens.numel() == batch);
  TVM_FFI_ICHECK(has_accepted == 0 || accepted.numel() == batch);
  check({state_bf16}, kDLBfloat, 16, state);
  check({slots, accepted, starts, lens}, kDLInt, 32, state);
  check_aligned({state_bf16});

  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  cudaError_t status = bv::Accept(
      ptr<const int>(slots), has_accepted ? ptr<const int>(accepted) : nullptr, ptr<float>(state),
      ptr<uint16_t>(state_bf16), ptr<int>(cnt), ptr<float>(prefix), ptr<uint16_t>(k_inv_rows),
      ptr<float>(k_inv_f32_rows), ptr<uint16_t>(rhs_rows), ptr<uint16_t>(update_rows),
      ptr<uint16_t>(l_rows), ptr<float>(stamp), ptr<int>(meta), ptr<const float>(v_prefix),
      ptr<const float>(v_state), ptr<const uint16_t>(v_k), ptr<const float>(v_kf),
      ptr<const uint16_t>(v_r), ptr<const uint16_t>(v_u), ptr<const uint16_t>(v_l),
      ptr<int>(starts), ptr<int>(lens), static_cast<int>(cap), static_cast<int>(batch),
      static_cast<int>(num_heads), static_cast<int>(num_slots), get_stream(state.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "kda_batch_invariant_verify_accept failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kda_batch_invariant_verify_accept, kda_batch_invariant_verify_accept);
