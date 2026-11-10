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
#include <flashinfer/kda/batch_invariant_decode.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

namespace {

template <typename T>
T* ptr(TensorView t) {
  return static_cast<T*>(t.data_ptr());
}

void check_dtype(TensorView t, DLDataTypeCode code, int bits, const char* name) {
  TVM_FFI_ICHECK(t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1)
      << name << " has an unexpected dtype";
}

}  // namespace

void kda_batch_invariant_decode_step(TensorView q, TensorView k, TensorView v, TensorView g,
                                     TensorView beta, TensorView slots, TensorView out,
                                     TensorView state, TensorView state_bf16, TensorView cnt,
                                     TensorView prefix, TensorView k_inv_rows,
                                     TensorView k_inv_f32_rows, TensorView rhs_rows,
                                     TensorView update_rows, TensorView l_rows, TensorView stamp,
                                     TensorView a_log, TensorView dt_bias, double scale,
                                     double half_gate_scale_log2) {
  namespace bi = kda::bi_decode;
  CHECK_DIM(3, q);
  CHECK_DIM(4, state);
  const int64_t batch = q.size(0), num_heads = q.size(1), num_slots = state.size(0);
  TVM_FFI_ICHECK(q.size(2) == bi::D) << "head dim must be 128";
  TVM_FFI_ICHECK(k.numel() == q.numel() && v.numel() == q.numel() && g.numel() == q.numel() &&
                 out.numel() == q.numel());
  TVM_FFI_ICHECK(beta.numel() == batch * num_heads);
  TVM_FFI_ICHECK_EQ(slots.numel(), batch);
  TVM_FFI_ICHECK(state.size(1) == num_heads && state.size(2) == bi::D && state.size(3) == bi::D);
  const int64_t sh = num_slots * num_heads;
  TVM_FFI_ICHECK(state_bf16.numel() == state.numel() && cnt.numel() == sh);
  TVM_FFI_ICHECK(prefix.numel() == sh * bi::D && stamp.numel() == sh * bi::STAMP);
  const int64_t rows = sh * bi::CH * bi::D;
  TVM_FFI_ICHECK(k_inv_rows.numel() == rows && k_inv_f32_rows.numel() == rows &&
                 rhs_rows.numel() == rows && update_rows.numel() == rows);
  TVM_FFI_ICHECK(l_rows.numel() == sh * bi::CH * bi::CH);
  TVM_FFI_ICHECK(a_log.numel() == num_heads && dt_bias.numel() == num_heads * bi::D);
  for (TensorView t :
       {q, k, v, g, beta, out, state_bf16, k_inv_rows, rhs_rows, update_rows, l_rows}) {
    check_dtype(t, kDLBfloat, 16, "a bf16 tensor");
    CHECK_CONTIGUOUS(t);
  }
  for (TensorView t : {state, prefix, k_inv_f32_rows, stamp, a_log, dt_bias}) {
    check_dtype(t, kDLFloat, 32, "an fp32 tensor");
    CHECK_CONTIGUOUS(t);
  }
  for (TensorView t : {slots, cnt}) {
    check_dtype(t, kDLInt, 32, "an int32 tensor");
    CHECK_CONTIGUOUS(t);
  }
  CHECK_DEVICE(q, state);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  cudaError_t status = bi::DecodeStep(
      ptr<const uint16_t>(q), ptr<const uint16_t>(k), ptr<const uint16_t>(v),
      ptr<const uint16_t>(g), ptr<const uint16_t>(beta), ptr<const int>(slots), ptr<uint16_t>(out),
      ptr<float>(state), ptr<uint16_t>(state_bf16), ptr<int>(cnt), ptr<float>(prefix),
      ptr<uint16_t>(k_inv_rows), ptr<float>(k_inv_f32_rows), ptr<uint16_t>(rhs_rows),
      ptr<uint16_t>(update_rows), ptr<uint16_t>(l_rows), ptr<float>(stamp), ptr<const float>(a_log),
      ptr<const float>(dt_bias), static_cast<float>(scale),
      static_cast<float>(half_gate_scale_log2), static_cast<int>(batch),
      static_cast<int>(num_heads), stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "kda_batch_invariant_decode_step failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kda_batch_invariant_decode_step, kda_batch_invariant_decode_step);

void kda_batch_invariant_begin(TensorView slots, TensorView state, TensorView state_bf16,
                               TensorView cnt, TensorView stamp) {
  namespace bi = kda::bi_decode;
  CHECK_DIM(4, state);
  const int64_t num_heads = state.size(1), sh = state.size(0) * num_heads;
  TVM_FFI_ICHECK(state.size(2) == bi::D && state.size(3) == bi::D);
  TVM_FFI_ICHECK(state_bf16.numel() == state.numel() && cnt.numel() == sh &&
                 stamp.numel() == sh * bi::STAMP);
  check_dtype(state, kDLFloat, 32, "state");
  check_dtype(stamp, kDLFloat, 32, "stamp");
  check_dtype(state_bf16, kDLBfloat, 16, "state_bf16");
  check_dtype(slots, kDLInt, 32, "slots");
  check_dtype(cnt, kDLInt, 32, "cnt");
  for (TensorView t : {slots, state, state_bf16, cnt, stamp}) {
    CHECK_CONTIGUOUS(t);
  }
  CHECK_DEVICE(slots, state);

  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  cudaError_t status =
      bi::Begin(ptr<const int>(slots), ptr<const float>(state), ptr<uint16_t>(state_bf16),
                ptr<int>(cnt), ptr<float>(stamp), static_cast<int>(slots.numel()),
                static_cast<int>(num_heads), get_stream(state.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "kda_batch_invariant_begin failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kda_batch_invariant_begin, kda_batch_invariant_begin);
