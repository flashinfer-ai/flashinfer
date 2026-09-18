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

// VibeCUDA MSA backend binding: one `run` entry that reproduces the validated
// level-3 dispatcher over the three kernel translation units in this directory
// (warp-specialized UMMA/TMEM g16 prefill, block-bucketed UMMA g4 prefill,
// and the general per-token / packed-pair fallback). The g16 UMMA route is
// flat dense-KV only and dates back to the gate-validated level-3 pipeline.
// All outputs and scratch buffers are caller-allocated (see
// flashinfer/msa_ops/_vibecuda_sm100.py).

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>

#include "msa_vibecuda_common.h"
#include "tvm_ffi_utils.h"

namespace flashinfer::msa_vibecuda {

using tvm::ffi::TensorView;

namespace {

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckContiguous(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& t, const char* name, int code, int bits, int lanes) {
  DLDataType d = t.dtype();
  TVM_FFI_CHECK((int)d.code == code && (int)d.bits == bits && (int)d.lanes == lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << code << ", bits=" << bits
      << ", lanes=" << lanes << "), got (code=" << (int)d.code << ", bits=" << (int)d.bits
      << ", lanes=" << (int)d.lanes << ")";
}

inline void CheckSameCudaDevice(const TensorView& t, const TensorView& reference, const char* name,
                                const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name
      << ": got cuda:" << t.device().device_id << " versus cuda:" << reference.device().device_id;
}

inline void CheckDevice(int32_t device_id) {
  int major = 0;
  int minor = 0;
  cudaError_t status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaDeviceGetAttribute(major) failed: " << cudaGetErrorString(status);
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaDeviceGetAttribute(minor) failed: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(major == 10 && (minor == 0 || minor == 3), RuntimeError)
      << "the VibeCUDA MSA backend supports compute capability 10.0 or 10.3, got " << major << "."
      << minor;
}

// Workspace size mirrors (ints only); kept in sync with
// umma_g4_forward's table math in msa_vibecuda_g4.cu. The float half is
// slots * 66 with slots = total_q * num_q_heads * topk.
inline int64_t G4WorkspaceInts(int64_t nbuckets, int64_t hn, int64_t topk, int64_t rows_bound,
                               int64_t tiles_bound) {
  return nbuckets * 2 + hn + (nbuckets + 1) * 2 + nbuckets * 5 + tiles_bound + hn * topk +
         rows_bound + 1 + 4;
}

}  // namespace

// q/k/v arrive dense (bf16/fp16) or with k/v pre-viewed as uint8 when the
// caller's KV is fp8-e4m3; kv_kind says which (0 bf16, 1 fp16, 2 fp8).
// Dummy 1-element tensors are accepted for every scratch argument the
// selected route does not consume.
void Run(TensorView arg_q, TensorView arg_k, TensorView arg_v, TensorView arg_out,
         TensorView arg_q2k, TensorView arg_cu_q, TensorView arg_cu_k, TensorView arg_page_table,
         TensorView arg_seqused_k, TensorView arg_ws_int, TensorView arg_ws_float,
         int64_t arg_kv_kind, int64_t arg_seqlen_q, int64_t arg_causal, int64_t arg_ws_int_need,
         int64_t arg_ws_float_need, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError) << "cuda_stream must be non-negative";
  ffi::CUDADeviceGuard device_guard(arg_q.device().device_id);
  CheckDevice(arg_q.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));

  CheckCudaTensor(arg_q, "q");
  CheckContiguous(arg_q, "q");
  TVM_FFI_CHECK(arg_q.ndim() == 3 && arg_q.size(2) == 128, ValueError)
      << "q must be (total_q, num_q_heads, 128)";
  // bf16 (code 4) or fp16 (code 2, 16 bits)
  const bool is_bf16 = arg_q.dtype().code == 4;
  CheckDtype(arg_q, "q", is_bf16 ? 4 : 2, 16, 1);

  CheckCudaTensor(arg_k, "k");
  CheckSameCudaDevice(arg_k, arg_q, "k", "q");
  CheckContiguous(arg_k, "k");
  CheckCudaTensor(arg_v, "v");
  CheckSameCudaDevice(arg_v, arg_q, "v", "q");
  CheckContiguous(arg_v, "v");
  TVM_FFI_CHECK(arg_k.ndim() == arg_v.ndim() && (arg_k.ndim() == 3 || arg_k.ndim() == 4),
                ValueError)
      << "k/v must both be flat 3D or paged 4D tensors";
  for (int i = 0; i < arg_k.ndim(); ++i) {
    TVM_FFI_CHECK(arg_k.size(i) == arg_v.size(i), ValueError) << "k/v shape mismatch";
  }
  const int64_t kv_kind = arg_kv_kind;
  TVM_FFI_CHECK(kv_kind == 0 || kv_kind == 1 || kv_kind == 2, ValueError)
      << "kv_kind must be 0 (bf16), 1 (fp16), or 2 (fp8)";
  if (kv_kind == 2) {
    // fp8 K/V is handed over as uint8 views of the original fp8 tensors.
    CheckDtype(arg_k, "k", 1, 8, 1);
    CheckDtype(arg_v, "v", 1, 8, 1);
    TVM_FFI_CHECK(is_bf16, ValueError) << "fp8 KV requires bf16 Q";
  } else {
    CheckDtype(arg_k, "k", kv_kind == 0 ? 4 : 2, 16, 1);
    CheckDtype(arg_v, "v", kv_kind == 0 ? 4 : 2, 16, 1);
    TVM_FFI_CHECK((kv_kind == 0) == is_bf16, ValueError) << "q/k dtype mismatch";
  }

  CheckCudaTensor(arg_out, "out");
  CheckSameCudaDevice(arg_out, arg_q, "out", "q");
  CheckContiguous(arg_out, "out");
  CheckDtype(arg_out, "out", is_bf16 ? 4 : 2, 16, 1);
  TVM_FFI_CHECK(arg_out.ndim() == 3 && arg_out.size(0) == arg_q.size(0) &&
                    arg_out.size(1) == arg_q.size(1) && arg_out.size(2) == 128,
                ValueError)
      << "out must match q's (total_q, num_q_heads, 128)";

  CheckCudaTensor(arg_q2k, "q2k_indices");
  CheckSameCudaDevice(arg_q2k, arg_q, "q2k_indices", "q");
  CheckContiguous(arg_q2k, "q2k_indices");
  CheckDtype(arg_q2k, "q2k_indices", 0, 32, 1);
  CheckCudaTensor(arg_cu_q, "cu_seqlens_q");
  CheckSameCudaDevice(arg_cu_q, arg_q, "cu_seqlens_q", "q");
  CheckContiguous(arg_cu_q, "cu_seqlens_q");
  CheckDtype(arg_cu_q, "cu_seqlens_q", 0, 32, 1);
  CheckCudaTensor(arg_cu_k, "cu_seqlens_k");
  CheckSameCudaDevice(arg_cu_k, arg_q, "cu_seqlens_k", "q");
  CheckContiguous(arg_cu_k, "cu_seqlens_k");
  CheckDtype(arg_cu_k, "cu_seqlens_k", 0, 32, 1);

  const int total_q = (int)arg_q.size(0);
  const int num_q_heads = (int)arg_q.size(1);
  const int num_kv_heads = (int)arg_k.size(1);
  const int topk = (int)arg_q2k.size(2);
  const int nbatch = (int)arg_cu_q.size(0) - 1;
  TVM_FFI_CHECK(nbatch >= 1, ValueError) << "cu_seqlens_q must have at least two entries";
  TVM_FFI_CHECK(num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, ValueError)
      << "num_q_heads must be a positive multiple of num_kv_heads";
  const int group = num_q_heads / num_kv_heads;
  TVM_FFI_CHECK(group >= 1 && group <= 16, ValueError) << "group size must be in [1, 16]";
  TVM_FFI_CHECK(
      arg_q2k.ndim() == 3 && arg_q2k.size(0) == num_kv_heads && arg_q2k.size(1) == total_q,
      ValueError)
      << "q2k_indices must be (num_kv_heads, total_q, topk)";
  TVM_FFI_CHECK(arg_cu_k.size(0) == nbatch + 1, ValueError)
      << "cu_seqlens_k must have batch + 1 entries";

  const bool paged = arg_page_table.ndim() == 2;
  TVM_FFI_CHECK(paged == (arg_k.ndim() == 4), ValueError)
      << "the paged/flat page-table and KV layouts must agree";
  int64_t num_pages = paged ? arg_k.size(0) : 0;
  int64_t max_pages = paged ? arg_page_table.size(1) : (arg_k.size(0) + 127) / 128;
  long pt_stride = 0;
  const int* page_table_ptr = nullptr;
  const int* seqused_ptr = nullptr;
  if (paged) {
    CheckCudaTensor(arg_page_table, "page_table");
    CheckSameCudaDevice(arg_page_table, arg_q, "page_table", "q");
    CheckContiguous(arg_page_table, "page_table");
    CheckDtype(arg_page_table, "page_table", 0, 32, 1);
    TVM_FFI_CHECK(arg_page_table.size(0) == nbatch, ValueError)
        << "page_table must have batch rows";
    CheckCudaTensor(arg_seqused_k, "seqused_k");
    CheckSameCudaDevice(arg_seqused_k, arg_q, "seqused_k", "q");
    CheckContiguous(arg_seqused_k, "seqused_k");
    CheckDtype(arg_seqused_k, "seqused_k", 0, 32, 1);
    TVM_FFI_CHECK(arg_seqused_k.ndim() == 1 && arg_seqused_k.size(0) == nbatch, ValueError)
        << "seqused_k must have batch entries";
    pt_stride = arg_page_table.stride(0);
    page_table_ptr = (const int*)arg_page_table.data_ptr();
    seqused_ptr = (const int*)arg_seqused_k.data_ptr();
  }

  const bool causal = arg_causal != 0;
  const int seqlen_q = (int)arg_seqlen_q;

  const int kv_dtype_code = (int)kv_kind;
  ::msa_vibecuda::CoreParams p;
  p.q = arg_q.data_ptr();
  p.q2k = (const int*)arg_q2k.data_ptr();
  p.cu_q = (const int*)arg_cu_q.data_ptr();
  p.cu_k = (const int*)arg_cu_k.data_ptr();
  p.page_table = page_table_ptr;
  p.out = arg_out.data_ptr();
  p.q_tok = arg_q.stride(0);
  p.q_head = arg_q.stride(1);
  p.o_tok = arg_out.stride(0);
  p.o_head = arg_out.stride(1);
  p.q2k_h = arg_q2k.stride(0);
  p.q2k_n = arg_q2k.stride(1);
  p.pt_stride = pt_stride;
  p.total_q = total_q;
  p.num_q_heads = num_q_heads;
  p.num_kv_heads = num_kv_heads;
  p.group = group;
  p.topk = topk;
  p.nbatch = nbatch;
  p.seqlen_q = seqlen_q;
  p.causal = causal ? 1 : 0;
  p.pack_T = 1;
  p.ws_next = nullptr;
  p.ws_total = 0;
  p.ws_ntiles = 0;
  const float scale = 1.0f / sqrtf(128.0f);
  p.scale_log2e = scale * 1.4426950408889634f;

  // Route order mirrors the accepted level-3 dispatcher exactly:
  //   1. warp-specialized UMMA g16 prefill (flat, dense bf16/fp16 KV),
  //   2. block-bucketed UMMA g4 prefill (paged, group==4),
  //   3. general per-token / packed pair fallback.
  const bool g16_ok =
      msa_umma_g16::umma_g16_eligible(group, seqlen_q, topk, (int)kv_kind, paged, true);
  if (g16_ok) {
    msa_umma_g16::umma_g16_forward(arg_q.data_ptr(), is_bf16, arg_k.data_ptr(), arg_v.data_ptr(),
                                   (const int*)arg_q2k.data_ptr(), (const int*)arg_cu_q.data_ptr(),
                                   (const int*)arg_cu_k.data_ptr(), arg_out.data_ptr(), total_q,
                                   (int)arg_k.size(0), num_q_heads, num_kv_heads, topk, nbatch,
                                   causal, stream);
    return;
  }
  if (msa_umma_g4::umma_g4_eligible(group, paged, kv_dtype_code, topk, nbatch, (int)max_pages,
                                    num_kv_heads, total_q)) {
    const int64_t hn = (int64_t)num_kv_heads * total_q;
    const int64_t rows = (int64_t)total_q * num_q_heads;
    const int64_t slots = rows * topk;
    const int64_t nbuckets = (int64_t)num_kv_heads * nbatch * max_pages;
    const int64_t rows_bound = hn * topk + nbuckets * 32;
    const int64_t tiles_bound = hn * topk / 32 + nbuckets;
    const int64_t need_i = G4WorkspaceInts(nbuckets, hn, topk, rows_bound, tiles_bound);
    const int64_t need_f = slots * 64 + slots * 2;
    TVM_FFI_CHECK(need_i <= arg_ws_int_need, ValueError)
        << "g4 int workspace too small: need " << need_i << ", plan returned " << arg_ws_int_need;
    TVM_FFI_CHECK(need_f <= arg_ws_float_need, ValueError)
        << "g4 float workspace too small: need " << need_f << ", plan returned "
        << arg_ws_float_need;
    CheckCudaTensor(arg_ws_int, "ws_int");
    CheckSameCudaDevice(arg_ws_int, arg_q, "ws_int", "q");
    CheckContiguous(arg_ws_int, "ws_int");
    CheckDtype(arg_ws_int, "ws_int", 0, 32, 1);
    CheckCudaTensor(arg_ws_float, "ws_float");
    CheckSameCudaDevice(arg_ws_float, arg_q, "ws_float", "q");
    CheckContiguous(arg_ws_float, "ws_float");
    CheckDtype(arg_ws_float, "ws_float", 2, 32, 1);
    msa_umma_g4::umma_g4_forward(arg_q.data_ptr(), is_bf16, arg_k.data_ptr(), arg_v.data_ptr(),
                                 (const int*)arg_q2k.data_ptr(), (const int*)arg_cu_q.data_ptr(),
                                 (const int*)arg_cu_k.data_ptr(), page_table_ptr,
                                 arg_out.data_ptr(), total_q, num_q_heads, num_kv_heads, topk,
                                 nbatch, (int)num_pages, (int)max_pages, pt_stride, p.q_tok,
                                 p.q_head, p.o_tok, p.o_head, (int*)arg_ws_int.data_ptr(),
                                 (float*)arg_ws_float.data_ptr(), seqlen_q, causal, stream);
    return;
  }
  TVM_FFI_CHECK(topk <= 36, ValueError)
      << "the VibeCUDA MSA general route supports topk <= 36; larger topk requires "
         "the eligible GQA-16 prefill route";
  ::msa_vibecuda::KvLayout kv;
  kv.d0 = arg_k.size(0);
  kv.d1 = num_kv_heads;
  kv.s0 = arg_k.stride(0);
  kv.s1 = arg_k.stride(1);
  kv.s2 = paged ? arg_k.stride(2) : 0;
  msa_vibecuda_core::core_forward(p, kv, arg_k.data_ptr(), arg_v.data_ptr(), is_bf16, kv_dtype_code,
                                  paged, stream);
}

}  // namespace flashinfer::msa_vibecuda

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::msa_vibecuda::Run);
