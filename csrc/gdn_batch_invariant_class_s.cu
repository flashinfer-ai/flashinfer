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
#include <flashinfer/gdn/batch_invariant_class_s.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

namespace {

namespace bs = gdn::bi_class_s;

template <typename T>
T* ptr(TensorView t) {
  return static_cast<T*>(t.data_ptr());
}

void check_dtype(TensorView t, DLDataTypeCode code, int bits, const char* name) {
  TVM_FFI_ICHECK(t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1)
      << name << " has an unexpected dtype";
}

struct Geometry {
  int64_t tokens, num_q_heads, num_v_heads, num_slots;
};

Geometry check(TensorView q, TensorView k, TensorView v, TensorView ga, TensorView gb,
               TensorView A_log, TensorView dt_bias, TensorView slots, TensorView out,
               TensorView state, TensorView hist_k, TensorView hist_u, TensorView hist_l,
               TensorView cnt) {
  CHECK_DIM(3, q);
  CHECK_DIM(3, v);
  CHECK_DIM(4, state);
  const int64_t tokens = q.size(0), H = q.size(1), HV = v.size(1);
  TVM_FFI_ICHECK(q.size(2) == bs::D && v.size(2) == bs::D) << "head dim must be 128";
  TVM_FFI_ICHECK(H > 0 && HV % H == 0) << "the v-head count must be a multiple of the q-head count";
  TVM_FFI_ICHECK(k.numel() == q.numel() && v.size(0) == tokens && out.numel() == v.numel());
  TVM_FFI_ICHECK(ga.numel() == tokens * HV && gb.numel() == ga.numel());
  TVM_FFI_ICHECK(A_log.numel() == HV && dt_bias.numel() == HV);
  TVM_FFI_ICHECK(state.size(1) == HV && state.size(2) == bs::D && state.size(3) == bs::D);
  // the side buffers hold `capacity` slots; the state pool may be a view of fewer
  const int64_t capacity = cnt.numel() / (HV * 8);
  TVM_FFI_ICHECK(state.size(0) <= capacity && cnt.numel() == capacity * HV * 8);
  TVM_FFI_ICHECK(hist_k.numel() == capacity * 2 * H * bs::C * bs::D);
  TVM_FFI_ICHECK(hist_u.numel() == capacity * 2 * HV * bs::C * bs::D);
  TVM_FFI_ICHECK(hist_l.numel() == capacity * 2 * HV * bs::C);
  for (TensorView t : {q, k, v, ga, gb, out, state, hist_k}) {
    check_dtype(t, kDLBfloat, 16, "a bf16 tensor");
    CHECK_CONTIGUOUS(t);
    CHECK_DEVICE(t, state);
  }
  for (TensorView t : {q, k, v, state, hist_k}) {
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0)
        << "bf16 rows must be 16-byte aligned";
  }
  for (TensorView t : {A_log, dt_bias, hist_u, hist_l}) {
    check_dtype(t, kDLFloat, 32, "an fp32 tensor");
    CHECK_CONTIGUOUS(t);
    CHECK_DEVICE(t, state);
  }
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(hist_u.data_ptr()) % 16 == 0);
  for (TensorView t : {slots, cnt}) {
    check_dtype(t, kDLInt, 32, "an int32 tensor");
    CHECK_CONTIGUOUS(t);
    CHECK_DEVICE(t, state);
  }
  return {tokens, H, HV, state.size(0)};
}

}  // namespace

// One token per slot: slots[i] takes row i; early picks the placement for few blocks per SM.
void gdn_batch_invariant_class_s_decode(TensorView q, TensorView k, TensorView v, TensorView ga,
                                        TensorView gb, TensorView A_log, TensorView dt_bias,
                                        TensorView slots, TensorView out, TensorView state,
                                        TensorView hist_k, TensorView hist_u, TensorView hist_l,
                                        TensorView cnt, double scale, int64_t early) {
  const Geometry g =
      check(q, k, v, ga, gb, A_log, dt_bias, slots, out, state, hist_k, hist_u, hist_l, cnt);
  TVM_FFI_ICHECK(slots.numel() == g.tokens) << "a step takes one token per slot";
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaError_t status = bs::LaunchDecode(
      ptr<const uint16_t>(q), ptr<const uint16_t>(k), ptr<const uint16_t>(v),
      ptr<const uint16_t>(ga), ptr<const uint16_t>(gb), ptr<const float>(A_log),
      ptr<const float>(dt_bias), ptr<uint16_t>(state), ptr<const int>(slots), ptr<uint16_t>(out),
      ptr<uint16_t>(hist_k), ptr<float>(hist_u), ptr<float>(hist_l), ptr<int>(cnt),
      static_cast<float>(scale), static_cast<int>(slots.numel()), static_cast<int>(g.num_q_heads),
      static_cast<int>(g.num_v_heads), static_cast<int>(g.num_slots), static_cast<int>(early),
      get_stream(q.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "the gdn class S decode kernel failed: " << cudaGetErrorString(status);
}

// Rows [cu[i], cu[i + 1]) go to slots[i]; prefix (may be empty) counts the tokens before them.
void gdn_batch_invariant_class_s_prefill(TensorView q, TensorView k, TensorView v, TensorView ga,
                                         TensorView gb, TensorView A_log, TensorView dt_bias,
                                         TensorView slots, TensorView cu, TensorView prefix,
                                         TensorView out, TensorView state, TensorView hist_k,
                                         TensorView hist_u, TensorView hist_l, TensorView cnt,
                                         double scale, int64_t num_tiles, int64_t max_tokens,
                                         int64_t reset) {
  const Geometry g =
      check(q, k, v, ga, gb, A_log, dt_bias, slots, out, state, hist_k, hist_u, hist_l, cnt);
  const int64_t batch = slots.numel();
  check_dtype(cu, kDLInt, 32, "cu_seqlens");
  CHECK_CONTIGUOUS(cu);
  CHECK_DEVICE(cu, state);
  TVM_FFI_ICHECK(cu.numel() == batch + 1);
  TVM_FFI_ICHECK(num_tiles == 1 || num_tiles == 2 || num_tiles == 4 || num_tiles == 8);
  TVM_FFI_ICHECK(max_tokens >= 1);
  const int* prefix_ptr = nullptr;
  if (prefix.numel() != 0) {
    check_dtype(prefix, kDLInt, 32, "prefix_lens");
    CHECK_CONTIGUOUS(prefix);
    CHECK_DEVICE(prefix, state);
    TVM_FFI_ICHECK(prefix.numel() == batch);
    prefix_ptr = ptr<const int>(prefix);
  }
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaError_t status = bs::LaunchPrefill(
      ptr<const uint16_t>(q), ptr<const uint16_t>(k), ptr<const uint16_t>(v),
      ptr<const uint16_t>(ga), ptr<const uint16_t>(gb), ptr<const float>(A_log),
      ptr<const float>(dt_bias), ptr<uint16_t>(state), ptr<const int>(slots), ptr<const int>(cu),
      prefix_ptr, ptr<uint16_t>(out), ptr<uint16_t>(hist_k), ptr<float>(hist_u), ptr<float>(hist_l),
      ptr<int>(cnt), static_cast<float>(scale), static_cast<int>(batch),
      static_cast<int>(g.num_q_heads), static_cast<int>(g.num_v_heads),
      static_cast<int>(g.num_slots), static_cast<int>(num_tiles), static_cast<int>(max_tokens),
      static_cast<int>(reset), static_cast<int>(g.tokens), get_stream(q.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "the gdn class S prefill kernel failed: " << cudaGetErrorString(status);
}

// The long-run placement of the prefill: one CTA per (sequence, v-head, V-tile).
void gdn_batch_invariant_class_s_prefill_chain(TensorView q, TensorView k, TensorView v,
                                               TensorView ga, TensorView gb, TensorView A_log,
                                               TensorView dt_bias, TensorView slots, TensorView cu,
                                               TensorView prefix, TensorView out, TensorView state,
                                               TensorView hist_k, TensorView hist_u,
                                               TensorView hist_l, TensorView cnt, double scale,
                                               int64_t num_tiles, int64_t reset) {
  const Geometry g =
      check(q, k, v, ga, gb, A_log, dt_bias, slots, out, state, hist_k, hist_u, hist_l, cnt);
  const int64_t batch = slots.numel();
  check_dtype(cu, kDLInt, 32, "cu_seqlens");
  CHECK_CONTIGUOUS(cu);
  CHECK_DEVICE(cu, state);
  TVM_FFI_ICHECK(cu.numel() == batch + 1);
  TVM_FFI_ICHECK(num_tiles == 1 || num_tiles == 2 || num_tiles == 4);
  const int* prefix_ptr = nullptr;
  if (prefix.numel() != 0) {
    check_dtype(prefix, kDLInt, 32, "prefix_lens");
    CHECK_CONTIGUOUS(prefix);
    CHECK_DEVICE(prefix, state);
    TVM_FFI_ICHECK(prefix.numel() == batch);
    prefix_ptr = ptr<const int>(prefix);
  }
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaError_t status = bs::LaunchPrefillChain(
      ptr<const uint16_t>(q), ptr<const uint16_t>(k), ptr<const uint16_t>(v),
      ptr<const uint16_t>(ga), ptr<const uint16_t>(gb), ptr<const float>(A_log),
      ptr<const float>(dt_bias), ptr<uint16_t>(state), ptr<const int>(slots), ptr<const int>(cu),
      prefix_ptr, ptr<uint16_t>(out), ptr<uint16_t>(hist_k), ptr<float>(hist_u), ptr<float>(hist_l),
      ptr<int>(cnt), static_cast<float>(scale), static_cast<int>(batch),
      static_cast<int>(g.num_q_heads), static_cast<int>(g.num_v_heads),
      static_cast<int>(g.num_slots), static_cast<int>(num_tiles), static_cast<int>(reset),
      static_cast<int>(g.tokens), get_stream(q.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "the gdn class S chain prefill failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_batch_invariant_class_s_decode,
                              gdn_batch_invariant_class_s_decode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_batch_invariant_class_s_prefill,
                              gdn_batch_invariant_class_s_prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_batch_invariant_class_s_prefill_chain,
                              gdn_batch_invariant_class_s_prefill_chain);
