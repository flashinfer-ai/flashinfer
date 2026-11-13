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
#include <cstdint>
#include <flashinfer/kda/batch_invariant_class_s.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

namespace {

namespace bi = kda::bi_class_s;

template <typename T>
T* ptr(TensorView t) {
  return static_cast<T*>(t.data_ptr());
}

void check_dtype(TensorView t, DLDataTypeCode code, int bits, const char* name) {
  TVM_FFI_ICHECK(t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1)
      << name << " has an unexpected dtype";
}

// Shapes and dtypes shared by the two entries; returns the head count.
int64_t check_common(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                     TensorView out, TensorView a_log, TensorView dt_bias, TensorView pool,
                     TensorView idx, TensorView key_rows, TensorView u_rows, TensorView prefix_rows,
                     TensorView cnt, TensorView stamp) {
  CHECK_DIM(3, q);
  CHECK_DIM(4, pool);
  const int64_t rows = q.size(0), num_heads = q.size(1), num_slots = pool.size(0);
  TVM_FFI_ICHECK(q.size(2) == bi::D) << "head dim must be 128";
  TVM_FFI_ICHECK(k.numel() == q.numel() && v.numel() == q.numel() && g.numel() == q.numel() &&
                 out.numel() == q.numel());
  TVM_FFI_ICHECK(beta.numel() == rows * num_heads);
  TVM_FFI_ICHECK(pool.size(1) == num_heads && pool.size(2) == bi::D && pool.size(3) == bi::D);
  const int64_t sh = num_slots * num_heads, cached = sh * bi::C * bi::D;
  TVM_FFI_ICHECK(key_rows.numel() == cached && u_rows.numel() == cached &&
                 prefix_rows.numel() == cached);
  TVM_FFI_ICHECK(cnt.numel() == sh && stamp.numel() == sh * bi::STAMP);
  TVM_FFI_ICHECK(a_log.numel() == num_heads && dt_bias.numel() == num_heads * bi::D);
  for (TensorView t : {q, k, v, g, beta, out, pool, key_rows}) {
    check_dtype(t, kDLBfloat, 16, "a bf16 tensor");
    CHECK_CONTIGUOUS(t);
  }
  for (TensorView t : {u_rows, prefix_rows, a_log, dt_bias}) {
    check_dtype(t, kDLFloat, 32, "an fp32 tensor");
    CHECK_CONTIGUOUS(t);
  }
  for (TensorView t : {idx, cnt}) {
    check_dtype(t, kDLInt, 32, "an int32 tensor");
    CHECK_CONTIGUOUS(t);
  }
  check_dtype(stamp, kDLInt, 64, "stamp");
  CHECK_CONTIGUOUS(stamp);
  // the kernels read and write these in 16-byte units
  for (TensorView t : {q, k, v, g, out, pool, key_rows, u_rows, prefix_rows, dt_bias, stamp}) {
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0)
        << "a tensor is not 16-byte aligned";
  }
  CHECK_DEVICE(q, pool);
  CHECK_DEVICE(idx, pool);
  return num_heads;
}

// lower_bound * log2(e) / 2: the kernels' gate scale
float half_gate_scale_log2(double lower_bound) {
  return static_cast<float>(lower_bound * 1.4426950408889634 * 0.5);
}

}  // namespace

void kda_batch_invariant_class_s_decode_step(TensorView q, TensorView k, TensorView v, TensorView g,
                                             TensorView beta, TensorView a_log, TensorView dt_bias,
                                             TensorView pool, TensorView idx, TensorView out,
                                             TensorView key_rows, TensorView u_rows,
                                             TensorView prefix_rows, TensorView cnt,
                                             TensorView stamp, double scale, double lower_bound) {
  const int64_t num_heads = check_common(q, k, v, g, beta, out, a_log, dt_bias, pool, idx, key_rows,
                                         u_rows, prefix_rows, cnt, stamp);
  const int64_t batch = q.size(0);
  TVM_FFI_ICHECK_EQ(idx.numel(), batch);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaError_t status = bi::DecodeStep(
      ptr<const uint16_t>(q), ptr<const uint16_t>(k), ptr<const uint16_t>(v),
      ptr<const uint16_t>(g), ptr<const uint16_t>(beta), ptr<const float>(a_log),
      ptr<const float>(dt_bias), ptr<uint16_t>(pool), ptr<const int>(idx), ptr<uint16_t>(out),
      ptr<uint16_t>(key_rows), ptr<float>(u_rows), ptr<float>(prefix_rows), ptr<int>(cnt),
      ptr<uint64_t>(stamp), static_cast<float>(scale), half_gate_scale_log2(lower_bound),
      static_cast<int>(batch), static_cast<int>(num_heads), static_cast<int>(pool.size(0)),
      get_stream(q.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "kda_batch_invariant_class_s_decode_step failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kda_batch_invariant_class_s_decode_step,
                              kda_batch_invariant_class_s_decode_step);

// Sequence b appends packed rows [cu[b], cu[b + 1]) to slot idx[b].
void kda_batch_invariant_class_s_prefill(TensorView q, TensorView k, TensorView v, TensorView g,
                                         TensorView beta, TensorView a_log, TensorView dt_bias,
                                         TensorView pool, TensorView idx, TensorView cu,
                                         TensorView out, TensorView key_rows, TensorView u_rows,
                                         TensorView prefix_rows, TensorView cnt, TensorView stamp,
                                         double scale, double lower_bound, bool fresh) {
  const int64_t num_heads = check_common(q, k, v, g, beta, out, a_log, dt_bias, pool, idx, key_rows,
                                         u_rows, prefix_rows, cnt, stamp);
  const int64_t batch = idx.numel();
  check_dtype(cu, kDLInt, 32, "cu");
  CHECK_CONTIGUOUS(cu);
  CHECK_DEVICE(cu, pool);
  TVM_FFI_ICHECK_EQ(cu.numel(), batch + 1);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaError_t status = bi::Prefill(
      ptr<const uint16_t>(q), ptr<const uint16_t>(k), ptr<const uint16_t>(v),
      ptr<const uint16_t>(g), ptr<const uint16_t>(beta), ptr<const float>(a_log),
      ptr<const float>(dt_bias), ptr<uint16_t>(pool), ptr<const int>(idx), ptr<const int>(cu),
      ptr<uint16_t>(out), ptr<uint16_t>(key_rows), ptr<float>(u_rows), ptr<float>(prefix_rows),
      ptr<int>(cnt), ptr<uint64_t>(stamp), static_cast<float>(scale),
      half_gate_scale_log2(lower_bound), static_cast<int>(batch), static_cast<int>(num_heads),
      static_cast<int>(pool.size(0)), fresh, get_stream(q.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "kda_batch_invariant_class_s_prefill failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kda_batch_invariant_class_s_prefill,
                              kda_batch_invariant_class_s_prefill);
