// SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team
// SPDX-License-Identifier: Apache-2.0
//
// Fused split-KV combine for Minimax Sparse Attention (SM120/SM121).
//
// Reduces per-KV-block partial outputs into the final attention output with
// LSE weighting:
//
//   out[q, h, :] = sum_s w_s * o_partial[s, q, h, :] / sum_s w_s
//   w_s          = exp2(lse2[s, q, h] - max_s lse2[s, q, h])
//
// where lse2 is the log2-domain LSE written by the KV-major forward kernel
// and s ranges over the query's valid split slots [0, split_counts[q, hkv]).
// Slots >= count hold uninitialized memory and are never read.
//
// Optionally writes the combined natural-log LSE.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

constexpr int kMaxTopK = 256;  // slot id is packed into 8 bits upstream

template <typename T>
__global__ void sparse_combine_kernel(T const* __restrict__ o_partial,  // [topk, total_q, Hq, d]
                                      float const* __restrict__ lse2,   // [topk, total_q, Hq]
                                      int const* __restrict__ split_counts,  // [total_q, Hkv]
                                      T* __restrict__ out,                   // [total_q, Hq, d]
                                      float* __restrict__ lse_out,  // [total_q, Hq] or nullptr
                                      float out_scale,  // e.g. NVFP4 V global dequant scale
                                      int total_q, int Hq, int group_size, int topk, int head_dim) {
  int q = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;

  __shared__ float s_w[kMaxTopK];
  __shared__ float s_max, s_denom;

  int count = split_counts[(size_t)q * (Hq / group_size) + h / group_size];
  if (count > topk) count = topk;

  size_t qh = (size_t)q * Hq + h;
  size_t slot_stride = (size_t)total_q * Hq;

  if (count <= 0) {
    for (int c = tid; c < head_dim; c += blockDim.x) out[qh * head_dim + c] = T(0.0f);
    if (lse_out != nullptr && tid == 0) lse_out[qh] = -CUDART_INF_F;
    return;
  }

  // slot weights (single warp is enough; count <= 256)
  if (tid == 0) {
    float m = -CUDART_INF_F;
    for (int s = 0; s < count; ++s) {
      float v = lse2[s * slot_stride + qh];
      s_w[s] = v;
      m = fmaxf(m, v);
    }
    float denom = 0.0f;
    for (int s = 0; s < count; ++s) {
      float w = exp2f(s_w[s] - m);  // m == -inf -> exp2(nan) handled below
      if (!isfinite(m)) w = 0.0f;
      s_w[s] = w;
      denom += w;
    }
    s_max = m;
    s_denom = denom;
    if (lse_out != nullptr) {
      // natural-log LSE: (max + log2(denom)) * ln(2)
      lse_out[qh] =
          (denom > 0.0f && isfinite(m)) ? (m + log2f(denom)) * 0.6931471805599453f : -CUDART_INF_F;
    }
  }
  __syncthreads();

  float denom = s_denom;
  float inv = (denom > 0.0f) ? 1.0f / denom : 0.0f;

  for (int c = tid; c < head_dim; c += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < count; ++s) {
      float w = s_w[s];
      if (w > 0.0f) acc += w * float(o_partial[(s * slot_stride + qh) * head_dim + c]);
    }
    out[qh * head_dim + c] = T(acc * inv * out_scale);
  }
}

}  // anonymous namespace

void sparse_combine(TensorView o_partial, TensorView lse_partial, TensorView split_counts,
                    TensorView out, Optional<TensorView> lse_out, int64_t group_size,
                    double out_scale) {
  CHECK_INPUT(o_partial);
  CHECK_INPUT(lse_partial);
  CHECK_INPUT(split_counts);
  CHECK_INPUT(out);
  CHECK_DIM(4, o_partial);
  CHECK_DIM(3, lse_partial);
  CHECK_DIM(2, split_counts);
  CHECK_DIM(3, out);
  TVM_FFI_ICHECK(encode_dlpack_dtype(lse_partial.dtype()) == float32_code)
      << "lse_partial must be float32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(split_counts.dtype()) == int32_code)
      << "split_counts must be int32";

  int topk = (int)o_partial.size(0);
  int total_q = (int)o_partial.size(1);
  int Hq = (int)o_partial.size(2);
  int d = (int)o_partial.size(3);
  TVM_FFI_ICHECK(topk <= kMaxTopK) << "topk must be <= " << kMaxTopK;
  TVM_FFI_ICHECK(out.size(0) == total_q && out.size(1) == Hq && out.size(2) == d);
  TVM_FFI_ICHECK(lse_partial.size(0) == topk && lse_partial.size(1) == total_q &&
                 lse_partial.size(2) == Hq);
  TVM_FFI_ICHECK(split_counts.size(0) == total_q);
  int Hkv = (int)split_counts.size(1);
  TVM_FFI_ICHECK(Hq == Hkv * (int)group_size) << "group_size mismatch";

  float* lse_out_ptr = nullptr;
  if (lse_out.has_value()) {
    CHECK_INPUT(lse_out.value());
    CHECK_DIM(2, lse_out.value());
    TVM_FFI_ICHECK(encode_dlpack_dtype(lse_out.value().dtype()) == float32_code);
    TVM_FFI_ICHECK(lse_out.value().size(0) == total_q && lse_out.value().size(1) == Hq);
    lse_out_ptr = static_cast<float*>(lse_out.value().data_ptr());
  }

  cudaStream_t stream = get_current_stream();
  dim3 grid(total_q, Hq);
  int threads = std::min(128, d);

  auto dtype_code = encode_dlpack_dtype(o_partial.dtype());
  TVM_FFI_ICHECK(encode_dlpack_dtype(out.dtype()) == dtype_code)
      << "out dtype must match o_partial";
  if (dtype_code == bfloat16_code) {
    sparse_combine_kernel<__nv_bfloat16>
        <<<grid, threads, 0, stream>>>(static_cast<const __nv_bfloat16*>(o_partial.data_ptr()),
                                       static_cast<const float*>(lse_partial.data_ptr()),
                                       static_cast<const int*>(split_counts.data_ptr()),
                                       static_cast<__nv_bfloat16*>(out.data_ptr()), lse_out_ptr,
                                       (float)out_scale, total_q, Hq, (int)group_size, topk, d);
  } else if (dtype_code == float16_code) {
    sparse_combine_kernel<__half><<<grid, threads, 0, stream>>>(
        static_cast<const __half*>(o_partial.data_ptr()),
        static_cast<const float*>(lse_partial.data_ptr()),
        static_cast<const int*>(split_counts.data_ptr()), static_cast<__half*>(out.data_ptr()),
        lse_out_ptr, (float)out_scale, total_q, Hq, (int)group_size, topk, d);
  } else {
    TVM_FFI_ICHECK(false) << "o_partial must be bf16 or fp16";
  }
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sparse_combine launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_combine, sparse_combine);
