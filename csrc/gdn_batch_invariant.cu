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
#include <flashinfer/gdn/batch_invariant_decode.cuh>

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

void gdn_batch_invariant_decode_step(TensorView q, TensorView k, TensorView v, TensorView g,
                                     TensorView beta, TensorView slots, TensorView out,
                                     TensorView state, TensorView state_bf16, TensorView cnt,
                                     TensorView ring_l, TensorView ring_b, TensorView kn,
                                     TensorView m_rows, TensorView x_rows, TensorView vks_rows,
                                     TensorView nv_rows, TensorView hist_q, TensorView hist_k,
                                     TensorView hist_v, TensorView hist_g, TensorView hist_beta,
                                     double scale) {
  namespace bi = gdn::bi_decode;
  CHECK_DIM(3, q);
  CHECK_DIM(3, v);
  CHECK_DIM(4, state);
  const int64_t batch = q.size(0), num_q_heads = q.size(1), num_v_heads = v.size(1);
  TVM_FFI_ICHECK(q.size(2) == bi::D && v.size(2) == bi::D) << "head dim must be 128";
  TVM_FFI_ICHECK(num_q_heads > 0 && num_v_heads % num_q_heads == 0)
      << "the v-head count must be a multiple of the q-head count";
  TVM_FFI_ICHECK(k.numel() == q.numel() && v.size(0) == batch && out.numel() == v.numel());
  TVM_FFI_ICHECK(g.numel() == batch * num_v_heads && beta.numel() == g.numel());
  TVM_FFI_ICHECK_EQ(slots.numel(), batch);
  TVM_FFI_ICHECK(state.size(1) == num_v_heads && state.size(2) == bi::D && state.size(3) == bi::D);
  // the side buffers hold `capacity` slots; the state pool may be a view of fewer
  const int64_t capacity = cnt.numel() / num_v_heads, num_slots = state.size(0);
  const int64_t sh = capacity * num_v_heads;
  TVM_FFI_ICHECK(num_slots <= capacity && cnt.numel() == sh);
  TVM_FFI_ICHECK(state_bf16.numel() == sh * bi::D * bi::D);
  TVM_FFI_ICHECK(ring_l.numel() == sh * bi::CH && ring_b.numel() == sh * bi::CH);
  TVM_FFI_ICHECK(m_rows.numel() == sh * bi::CH * bi::CH && x_rows.numel() == m_rows.numel());
  TVM_FFI_ICHECK(vks_rows.numel() == sh * bi::CH * bi::D && nv_rows.numel() == vks_rows.numel());
  const int64_t hq_elems = capacity * bi::CH * num_q_heads * bi::D;
  TVM_FFI_ICHECK(kn.numel() == hq_elems && hist_q.numel() == hq_elems &&
                 hist_k.numel() == hq_elems);
  TVM_FFI_ICHECK(hist_v.numel() == vks_rows.numel());
  TVM_FFI_ICHECK(hist_g.numel() == sh * bi::CH && hist_beta.numel() == sh * bi::CH);
  for (TensorView t :
       {q, k, v, out, state_bf16, kn, m_rows, x_rows, vks_rows, nv_rows, hist_q, hist_k, hist_v}) {
    check_dtype(t, kDLBfloat, 16, "a bf16 tensor");
    CHECK_CONTIGUOUS(t);
    CHECK_DEVICE(t, state);
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0)
        << "bf16 tensors must be 16-byte aligned";
  }
  for (TensorView t : {g, beta, state, ring_l, ring_b, hist_g, hist_beta}) {
    check_dtype(t, kDLFloat, 32, "an fp32 tensor");
    CHECK_CONTIGUOUS(t);
    CHECK_DEVICE(t, state);
  }
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0)
      << "the state pool must be 16-byte aligned";
  for (TensorView t : {slots, cnt}) {
    check_dtype(t, kDLInt, 32, "an int32 tensor");
    CHECK_CONTIGUOUS(t);
    CHECK_DEVICE(t, state);
  }

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  cudaError_t status = bi::DecodeStep(
      ptr<const uint16_t>(q), ptr<const uint16_t>(k), ptr<const uint16_t>(v), ptr<const float>(g),
      ptr<const float>(beta), ptr<const int>(slots), ptr<uint16_t>(out), ptr<float>(state),
      ptr<uint16_t>(state_bf16), ptr<int>(cnt), ptr<float>(ring_l), ptr<float>(ring_b),
      ptr<uint16_t>(kn), ptr<uint16_t>(m_rows), ptr<uint16_t>(x_rows), ptr<uint16_t>(vks_rows),
      ptr<uint16_t>(nv_rows), ptr<uint16_t>(hist_q), ptr<uint16_t>(hist_k), ptr<uint16_t>(hist_v),
      ptr<float>(hist_g), ptr<float>(hist_beta), static_cast<float>(scale), static_cast<int>(batch),
      static_cast<int>(num_q_heads), static_cast<int>(num_v_heads), static_cast<int>(num_slots),
      stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "gdn_batch_invariant_decode_step failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_batch_invariant_decode_step, gdn_batch_invariant_decode_step);
