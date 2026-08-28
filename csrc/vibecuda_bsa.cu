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
// TVM-FFI binding for the VibeCUDA SM100/SM103 block-sparse attention
// forward (see include/flashinfer/vibecuda/bsa_fwd.cuh for the kernels and
// the raw host entry, and flashinfer/vibecuda_bsa.py for the public API).
#include <flashinfer/vibecuda/bsa_fwd.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer::vibecuda;

namespace {

inline void check_contiguous_3d(const TensorView& t, const char* name) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 3) << name << " must be a 3D tensor";
  TVM_FFI_ICHECK_EQ(t.stride(2), 1) << name << " must be contiguous in the last dim";
  TVM_FFI_ICHECK_EQ(t.stride(1), t.size(2)) << name << " must be contiguous";
  TVM_FFI_ICHECK_EQ(t.stride(0), t.size(1) * t.size(2)) << name << " must be contiguous";
}

}  // namespace

void vibecuda_bsa_fwd(TensorView out, TensorView lse, TensorView q, TensorView k,
                      TensorView v, TensorView block_mask, TensorView workspace,
                      int64_t block_size, double sm_scale, int64_t split_g,
                      bool return_lse) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(block_mask);
  CHECK_DEVICE(out, q);
  CHECK_DEVICE(k, v);

  check_contiguous_3d(q, "q");
  check_contiguous_3d(k, "k");
  check_contiguous_3d(v, "v");

  const int64_t M = q.size(0);
  const int64_t HQ = q.size(1);
  const int64_t D = q.size(2);
  const int64_t N = k.size(0);
  const int64_t HKV = k.size(1);

  TVM_FFI_ICHECK(D == 64 || D == 96 || D == 128)
      << "vibecuda_bsa_fwd requires head_dim in {64, 96, 128} (got " << D << ")";
  TVM_FFI_ICHECK(block_size % 64 == 0)
      << "vibecuda_bsa_fwd requires block_size to be a multiple of 64 (got " << block_size
      << ")";
  TVM_FFI_ICHECK(split_g >= 1 && split_g <= 16)
      << "vibecuda_bsa_fwd requires 1 <= split_g <= 16 (got " << split_g << ")";

  const int64_t q_dtype = encode_dlpack_dtype(q.dtype());
  TVM_FFI_ICHECK(q_dtype == bfloat16_code || q_dtype == float16_code)
      << "vibecuda_bsa_fwd requires bfloat16 or float16 q";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(k.dtype()), q_dtype)
      << "k must have the same dtype as q";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(v.dtype()), q_dtype)
      << "v must have the same dtype as q";
  TVM_FFI_ICHECK(k.size(2) == D && v.size(0) == N && v.size(1) == HKV && v.size(2) == D)
      << "k/v must have matching [N, num_kv_heads, head_dim] shapes";
  TVM_FFI_ICHECK(HQ >= HKV && HQ % HKV == 0)
      << "num_qo_heads (" << HQ << ") must be a multiple of num_kv_heads (" << HKV
      << ") for the GQA head mapping";

  check_contiguous_3d(block_mask, "block_mask");
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(block_mask.dtype()), encode_dlpack_dtype(dl_bool))
      << "block_mask must have dtype bool";
  const int64_t MB = (M + block_size - 1) / block_size;
  const int64_t NB = (N + block_size - 1) / block_size;
  TVM_FFI_ICHECK(block_mask.size(0) == HQ && block_mask.size(1) == MB &&
                 block_mask.size(2) == NB)
      << "block_mask must have shape (num_qo_heads=" << HQ << ", ceil(M/block_size)=" << MB
      << ", ceil(N/block_size)=" << NB << "), got (" << block_mask.size(0) << ", "
      << block_mask.size(1) << ", " << block_mask.size(2) << ")";

  check_contiguous_3d(out, "out");
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out.dtype()), q_dtype) << "out must have q's dtype";
  TVM_FFI_ICHECK(out.size(0) == M && out.size(1) == HQ && out.size(2) == D)
      << "out must have shape (" << M << ", " << HQ << ", " << D << ")";

  const bool is_bf16 = (q_dtype == bfloat16_code);
  float* lse_ptr = nullptr;
  if (return_lse) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(lse.dtype()), float32_code)
        << "lse must have dtype float32";
    TVM_FFI_ICHECK_EQ(lse.ndim(), 2) << "lse must be a 2D tensor";
    TVM_FFI_ICHECK(lse.size(0) == M && lse.size(1) == HQ)
        << "lse must have shape (" << M << ", " << HQ << ")";
    TVM_FFI_ICHECK_EQ(lse.stride(1), 1) << "lse must be contiguous in the last dim";
    TVM_FFI_ICHECK_EQ(lse.stride(0), HQ) << "lse must be contiguous";
    lse_ptr = static_cast<float*>(lse.data_ptr());
  }

  const int64_t rows_pad = ((M + 63) / 64) * 64;
  float* ows_ptr = nullptr;
  if (split_g > 1) {
    const int64_t required = split_g * rows_pad * HQ * (D + 4);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(workspace.dtype()), float32_code)
        << "workspace must have dtype float32";
    TVM_FFI_ICHECK(workspace.numel() >= required)
        << "split workspace too small: need at least " << required << " float32 elements "
        << "(split_g=" << split_g << ", rows_pad=" << rows_pad << ", num_qo_heads=" << HQ
        << ", head_dim+4=" << (D + 4) << "), got " << workspace.numel();
    TVM_FFI_ICHECK_EQ(workspace.stride(workspace.ndim() - 1), 1)
        << "workspace must be contiguous";
    ows_ptr = static_cast<float*>(workspace.data_ptr());
  }

  TVM_FFI_ICHECK(sm_scale > 0.0) << "sm_scale must be positive (got " << sm_scale << ")";

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  const cudaError_t status = VibeCUDABSAFwdRaw(
      out.data_ptr(), lse_ptr, q.data_ptr(), k.data_ptr(), v.data_ptr(),
      static_cast<const bool*>(block_mask.data_ptr()), ows_ptr, (int)M, (int)N,
      (int)HQ, (int)HKV, (int)D, (int)block_size, return_lse, (int)split_g, is_bf16,
      (float)sm_scale, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "vibecuda_bsa_fwd failed with error " << cudaGetErrorString(status);
}
