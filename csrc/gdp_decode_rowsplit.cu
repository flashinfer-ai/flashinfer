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
// tvm-ffi binding of the GDP row-split decode step (flashinfer/gdp_decode_rowsplit.py).
// One module per (token count, Householder count, lanes per row), compiled in through
// -DFI_GDP_NUM_TOKENS / -DFI_GDP_NUM_HOUSEHOLDER / -DFI_GDP_LANES; the CTA tile
// (state rows per CTA, rows per lane) is chosen per call among the instances below.
#include <flashinfer/gdp/decode_rowsplit.cuh>

#include "tvm_ffi_utils.h"

#if !defined(FI_GDP_NUM_TOKENS) || !defined(FI_GDP_NUM_HOUSEHOLDER) || !defined(FI_GDP_LANES)
#error "build with -DFI_GDP_NUM_TOKENS, -DFI_GDP_NUM_HOUSEHOLDER and -DFI_GDP_LANES"
#endif

namespace {

namespace rs = flashinfer::gdp::rowsplit;

template <typename T>
T* Ptr(const TensorView& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}

void CheckTensor(const TensorView& t, const TensorView& ref, DLDataType dtype, int ndim,
                 const char* name) {
  TVM_FFI_ICHECK_EQ(t.device().device_type, kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK_EQ(t.device().device_id, ref.device().device_id)
      << name << " must live on the device of q";
  TVM_FFI_ICHECK(t.dtype() == dtype) << name << " has an unexpected dtype";
  TVM_FFI_ICHECK_EQ(t.ndim(), ndim) << name << " must be a " << ndim << "-D tensor";
  // A dimension of size 1 is never stepped over, so its stride is free (torch keeps
  // e.g. the column stride of a [B, T] table's [:, -1] slice for B == 1).
  TVM_FFI_ICHECK(t.numel() == 0 || t.size(ndim - 1) == 1 || t.stride(ndim - 1) == 1)
      << name << " must be dense in its last dimension";
}

bool Aligned(const TensorView& t, int64_t elements) {
  if (reinterpret_cast<uintptr_t>(Ptr<char>(t)) % (elements * (t.dtype().bits / 8)) != 0) {
    return false;
  }
  for (int d = 0; d + 1 < t.ndim(); ++d) {
    if (t.size(d) > 1 && t.stride(d) % elements != 0) return false;
  }
  return true;
}

template <int kRows, int kRowsPerLane>
bool TryLaunch(int64_t rows, int64_t rows_per_lane, const rs::Params& params, int64_t batch,
               cudaStream_t stream, cudaError_t* status) {
  using Traits = rs::KernelTraits<FI_GDP_NUM_TOKENS, FI_GDP_NUM_HOUSEHOLDER, kRows, kRowsPerLane,
                                  FI_GDP_LANES>;
  if (rows != kRows || rows_per_lane != kRowsPerLane) return false;
  if constexpr (Traits::kSupported) {
    *status = rs::LaunchGdpDecodeRowSplit<FI_GDP_NUM_TOKENS, FI_GDP_NUM_HOUSEHOLDER, kRows,
                                          kRowsPerLane, FI_GDP_LANES>(params, batch, stream);
    return true;
  }
  return false;
}

}  // namespace

// q [B, T, H, 128], k [B, T, n, H, 128], v [B, T, n, H, 64], a [B, T, H], b [B, T, n, H]
// (BF16); A_log, dt_bias [H] (FP32); state [pool_size, H, 64, 128] (FP32); read_slots,
// write_slots [B] and scatter_slots [B, T] (int32); out [B, T, H, 64] (BF16).
void gdp_decode_rowsplit(TensorView q, TensorView k, TensorView v, TensorView a, TensorView b,
                         TensorView A_log, TensorView dt_bias, TensorView state,
                         TensorView read_slots, TensorView write_slots, TensorView scatter_slots,
                         TensorView out, double scale, int64_t use_scatter,
                         int64_t disable_state_update, int64_t rows_per_cta,
                         int64_t rows_per_lane) {
  constexpr int T = FI_GDP_NUM_TOKENS;
  constexpr int NH = FI_GDP_NUM_HOUSEHOLDER;
  const DLDataType bf16{kDLBfloat, 16, 1}, f32{kDLFloat, 32, 1}, i32{kDLInt, 32, 1};
  CheckTensor(q, q, bf16, 4, "q");
  CheckTensor(k, q, bf16, 5, "k");
  CheckTensor(v, q, bf16, 5, "v");
  CheckTensor(a, q, bf16, 3, "a");
  CheckTensor(b, q, bf16, 4, "b");
  CheckTensor(A_log, q, f32, 1, "A_log");
  CheckTensor(dt_bias, q, f32, 1, "dt_bias");
  CheckTensor(state, q, f32, 4, "state");
  CheckTensor(read_slots, q, i32, 1, "read_slots");
  CheckTensor(write_slots, q, i32, 1, "write_slots");
  CheckTensor(scatter_slots, q, i32, 2, "scatter_slots");
  CheckTensor(out, q, bf16, 4, "out");

  const int64_t B = q.size(0), H = q.size(2);
  TVM_FFI_ICHECK(q.size(1) == T && q.size(3) == rs::kHeadK) << "q must be [B, " << T << ", H, 128]";
  TVM_FFI_ICHECK(k.size(0) == B && k.size(1) == T && k.size(2) == NH && k.size(3) == H &&
                 k.size(4) == rs::kHeadK)
      << "k must be [B, T, " << NH << ", H, 128]";
  TVM_FFI_ICHECK(v.size(0) == B && v.size(1) == T && v.size(2) == NH && v.size(3) == H &&
                 v.size(4) == rs::kHeadV)
      << "v must be [B, T, " << NH << ", H, 64]";
  TVM_FFI_ICHECK(a.size(0) == B && a.size(1) == T && a.size(2) == H) << "a must be [B, T, H]";
  TVM_FFI_ICHECK(b.size(0) == B && b.size(1) == T && b.size(2) == NH && b.size(3) == H)
      << "b must be [B, T, n, H]";
  TVM_FFI_ICHECK(A_log.size(0) == H && dt_bias.size(0) == H) << "A_log / dt_bias must be [H]";
  TVM_FFI_ICHECK(state.size(0) >= 1 && state.size(1) == H && state.size(2) == rs::kHeadV &&
                 state.size(3) == rs::kHeadK)
      << "state must be [pool_size, H, 64, 128]";
  TVM_FFI_ICHECK(read_slots.size(0) == B && write_slots.size(0) == B);
  TVM_FFI_ICHECK(!use_scatter || (scatter_slots.size(0) == B && scatter_slots.size(1) == T))
      << "scatter_slots must be [B, T]";
  TVM_FFI_ICHECK(out.size(0) == B && out.size(1) == T && out.size(2) == H &&
                 out.size(3) == rs::kHeadV)
      << "out must be [B, T, H, 64]";
  // q / k are read in 8-byte pieces and the state rows in 16-byte pieces.
  TVM_FFI_ICHECK(Aligned(q, 4) && Aligned(k, 4)) << "q and k must be 8-byte aligned";
  TVM_FFI_ICHECK(Aligned(state, 4) && state.stride(2) >= rs::kHeadK)
      << "state must be 16-byte aligned with dense rows";
  if (B == 0) return;

  rs::Params params;
  params.q = Ptr<const uint16_t>(q);
  params.k = Ptr<const uint16_t>(k);
  params.v = Ptr<const uint16_t>(v);
  params.a = Ptr<const uint16_t>(a);
  params.b = Ptr<const uint16_t>(b);
  params.A_log = Ptr<const float>(A_log);
  params.dt_bias = Ptr<const float>(dt_bias);
  params.state = Ptr<float>(state);
  params.read_slots = Ptr<const int32_t>(read_slots);
  params.write_slots = Ptr<const int32_t>(write_slots);
  params.scatter_slots = Ptr<const int32_t>(scatter_slots);
  params.out = Ptr<uint16_t>(out);
  params.pool_size = state.size(0);
  params.heads = static_cast<int>(H);
  params.scale = static_cast<float>(scale);
  params.use_scatter = use_scatter ? 1 : 0;
  params.disable_state_update = disable_state_update ? 1 : 0;
  for (int d = 0; d < 3; ++d) {
    params.q_stride[d] = q.stride(d);
    params.a_stride[d] = a.stride(d);
    params.state_stride[d] = state.stride(d);
    params.out_stride[d] = out.stride(d);
  }
  for (int d = 0; d < 4; ++d) {
    params.k_stride[d] = k.stride(d);
    params.v_stride[d] = v.stride(d);
    params.b_stride[d] = b.stride(d);
  }
  params.scatter_stride[0] = scatter_slots.stride(0);
  params.scatter_stride[1] = scatter_slots.stride(1);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  cudaError_t status = cudaSuccess;
  const bool launched =
#if FI_GDP_LANES == 8
      TryLaunch<8, 1>(rows_per_cta, rows_per_lane, params, B, stream, &status) ||
      TryLaunch<16, 1>(rows_per_cta, rows_per_lane, params, B, stream, &status) ||
      TryLaunch<32, 1>(rows_per_cta, rows_per_lane, params, B, stream, &status) ||
      TryLaunch<16, 2>(rows_per_cta, rows_per_lane, params, B, stream, &status) ||
      TryLaunch<64, 4>(rows_per_cta, rows_per_lane, params, B, stream, &status);
#else
      TryLaunch<8, 1>(rows_per_cta, rows_per_lane, params, B, stream, &status) ||
      TryLaunch<16, 1>(rows_per_cta, rows_per_lane, params, B, stream, &status) ||
      TryLaunch<32, 2>(rows_per_cta, rows_per_lane, params, B, stream, &status) ||
      TryLaunch<64, 2>(rows_per_cta, rows_per_lane, params, B, stream, &status);
#endif
  TVM_FFI_ICHECK(launched) << "no row-split instance with " << rows_per_cta << " rows per CTA and "
                           << rows_per_lane << " rows per lane for T = " << T << ", n = " << NH
                           << ", " << FI_GDP_LANES << " lanes per row";
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "gdp_decode_rowsplit launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdp_decode_rowsplit, gdp_decode_rowsplit);
