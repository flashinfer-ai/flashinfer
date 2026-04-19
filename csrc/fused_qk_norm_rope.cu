/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <climits>
#include <flashinfer/fused_qk_norm_rope.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

/*!
 * \brief TVM-FFI entry point for the fused QK RMSNorm + RoPE op.
 *
 * Applies, in a single kernel launch and in place on ``q`` and ``k``:
 *   1. (optional, gated by ``is_qk_norm``) per-head RMSNorm with ``q_weight`` /
 *      ``k_weight`` along the last dimension;
 *   2. RoPE with on-the-fly cos/sin from ``pos_ids`` and ``rope_theta``, plus
 *      an optional YaRN frequency correction (``yarn_factor != 1``).
 *
 * \param q            [nnz, num_q_heads, head_dim], last dim contiguous.
 * \param k            [nnz, num_kv_heads, head_dim], last dim contiguous.
 * \param q_weight     [head_dim] RMSNorm weights for Q (ignored when ``is_qk_norm`` is false).
 * \param k_weight     [head_dim] RMSNorm weights for K (ignored when ``is_qk_norm`` is false).
 * \param pos_ids      [nnz] int32 position ids.
 * \param rotary_dim   Number of leading dims to apply RoPE to (``<= head_dim``, even).
 * \param interleave   If true, RoPE rotates (x[2i], x[2i+1]) (GPT-J style). If false,
 *                     RoPE rotates (x[i], x[i + rotary_dim/2]) (Neox / Llama style).
 * \param is_qk_norm   When false, skip RMSNorm and just apply RoPE.
 * \param yarn_*       YaRN scaling parameters. Use ``yarn_factor = 1`` and
 *                     ``yarn_attention_factor = 1`` to disable YaRN.
 */
void fused_qk_norm_rope(TensorView q, TensorView k, TensorView q_weight, TensorView k_weight,
                        TensorView pos_ids, int64_t rotary_dim, double eps, double rope_theta,
                        bool interleave, double yarn_factor, double yarn_low, double yarn_high,
                        double yarn_attention_factor, bool is_qk_norm) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_weight);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_weight);
  CHECK_INPUT(pos_ids);

  CHECK_DEVICE(q, k);
  CHECK_DEVICE(q, q_weight);
  CHECK_DEVICE(q, k_weight);
  CHECK_DEVICE(q, pos_ids);

  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  CHECK_DIM(1, q_weight);
  CHECK_DIM(1, k_weight);
  CHECK_DIM(1, pos_ids);

  TVM_FFI_ICHECK_EQ(q.dtype(), k.dtype()) << "q and k must share the same dtype";
  TVM_FFI_ICHECK_EQ(q.dtype(), q_weight.dtype()) << "q and q_weight must share the same dtype";
  TVM_FFI_ICHECK_EQ(q.dtype(), k_weight.dtype()) << "q and k_weight must share the same dtype";
  CHECK_INPUT_TYPE(pos_ids, dl_int32);

  int64_t const nnz = q.size(0);
  int64_t const num_q_heads = q.size(1);
  int64_t const num_kv_heads = k.size(1);
  int64_t const head_dim = q.size(2);

  TVM_FFI_ICHECK_EQ(k.size(0), nnz) << "q and k must have matching token count";
  TVM_FFI_ICHECK_EQ(k.size(2), head_dim) << "q and k must have matching head_dim";
  TVM_FFI_ICHECK_EQ(q_weight.size(0), head_dim) << "q_weight must have shape [head_dim]";
  TVM_FFI_ICHECK_EQ(k_weight.size(0), head_dim) << "k_weight must have shape [head_dim]";
  TVM_FFI_ICHECK_EQ(pos_ids.size(0), nnz) << "pos_ids must have shape [nnz]";

  // FusedQKNormRopeKernel indexes tokens/heads with int32 throughout (to keep
  // register pressure low). Ensure the batch sizes fit before we cast, so we
  // fail fast with a clear message instead of silently wrapping.
  TVM_FFI_ICHECK_LE(nnz, static_cast<int64_t>(INT_MAX))
      << "nnz (= q.size(0)) must fit in int32, got " << nnz;
  TVM_FFI_ICHECK_LE(num_q_heads + num_kv_heads, static_cast<int64_t>(INT_MAX))
      << "num_q_heads + num_kv_heads must fit in int32, got " << (num_q_heads + num_kv_heads);

  // head_dim must be one of the template-specialized values and a multiple of 64.
  TVM_FFI_ICHECK(head_dim == 64 || head_dim == 128 || head_dim == 256)
      << "fused_qk_norm_rope only supports head_dim in {64, 128, 256}, got " << head_dim;

  TVM_FFI_ICHECK_GT(rotary_dim, 0) << "rotary_dim must be positive";
  TVM_FFI_ICHECK_LE(rotary_dim, head_dim) << "rotary_dim must be <= head_dim";
  TVM_FFI_ICHECK_EQ(rotary_dim % 2, 0) << "rotary_dim must be even";

  int64_t const num_elems_per_thread = head_dim / 32;  // = 2, 4, or 8

  // For partial-rope Neox style, `pair_offset = (rotary_dim / 2) / num_elems_per_thread`
  // drives the `__shfl_xor_sync` that swaps the two halves. The xor-shuffle only
  // produces the desired pairing when `pair_offset` is a power of 2.
  if (!interleave) {
    TVM_FFI_ICHECK_EQ(rotary_dim % (2 * num_elems_per_thread), 0)
        << "rotary_dim must be a multiple of " << (2 * num_elems_per_thread)
        << " (2 * head_dim / 32) when interleave=False (Neox style)";
    int64_t const pair_offset = rotary_dim / (2 * num_elems_per_thread);
    TVM_FFI_ICHECK(pair_offset > 0 && (pair_offset & (pair_offset - 1)) == 0)
        << "rotary_dim / (2 * head_dim / 32) must be a power of 2 when interleave=False, got "
        << pair_offset;
  }

  if (yarn_factor == 1.0) {
    TVM_FFI_ICHECK_EQ(yarn_attention_factor, 1.0)
        << "yarn_attention_factor must be 1.0 when yarn_factor == 1.0 (YaRN disabled)";
  }

  if (nnz == 0) {
    return;
  }

  size_t const q_stride_n = q.stride(0);
  size_t const q_stride_h = q.stride(1);
  size_t const k_stride_n = k.stride(0);
  size_t const k_stride_h = k.stride(1);

  // FusedQKNormRopeLauncher issues per-lane vectorized loads of width
  // `num_elems_per_thread * sizeof(DType)` bytes (uint32_t / uint2 / uint4 for
  // head_dim = 64 / 128 / 256 respectively). The tensor base pointer is
  // tensor-aligned and the per-lane offset (lane_id * num_elems_per_thread) is
  // a multiple of num_elems_per_thread, so the full pointer
  // `base + token_idx * stride_n + head_idx * stride_h + lane_id * num_elems_per_thread`
  // is aligned iff each non-trivial stride is a multiple of num_elems_per_thread.
  // Dimensions of size 1 (or broadcast stride 0) contribute no offset and are
  // exempt. Validate here to fail fast with a clean diagnostic instead of
  // hitting a misaligned-address error inside the kernel.
  if (nnz > 1) {
    TVM_FFI_ICHECK_EQ(q_stride_n % num_elems_per_thread, 0)
        << "FusedQKNormRopeLauncher: q.stride(0) must be a multiple of head_dim/32 = "
        << num_elems_per_thread
        << " to satisfy vectorized load alignment, got q.stride(0) = " << q_stride_n;
    TVM_FFI_ICHECK_EQ(k_stride_n % num_elems_per_thread, 0)
        << "FusedQKNormRopeLauncher: k.stride(0) must be a multiple of head_dim/32 = "
        << num_elems_per_thread
        << " to satisfy vectorized load alignment, got k.stride(0) = " << k_stride_n;
  }
  if (num_q_heads > 1) {
    TVM_FFI_ICHECK_EQ(q_stride_h % num_elems_per_thread, 0)
        << "FusedQKNormRopeLauncher: q.stride(1) must be a multiple of head_dim/32 = "
        << num_elems_per_thread
        << " to satisfy vectorized load alignment, got q.stride(1) = " << q_stride_h;
  }
  if (num_kv_heads > 1) {
    TVM_FFI_ICHECK_EQ(k_stride_h % num_elems_per_thread, 0)
        << "FusedQKNormRopeLauncher: k.stride(1) must be a multiple of head_dim/32 = "
        << num_elems_per_thread
        << " to satisfy vectorized load alignment, got k.stride(1) = " << k_stride_h;
  }

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaStream_t const stream = get_stream(q.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    cudaError_t status = FusedQKNormRopeLauncher<c_type>(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(q_weight.data_ptr()), static_cast<c_type*>(k_weight.data_ptr()),
        static_cast<int32_t*>(pos_ids.data_ptr()), static_cast<int>(nnz),
        static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
        static_cast<int>(rotary_dim), static_cast<float>(eps), static_cast<float>(rope_theta),
        interleave, static_cast<float>(yarn_factor), static_cast<float>(yarn_low),
        static_cast<float>(yarn_high), static_cast<float>(yarn_attention_factor), is_qk_norm,
        q_stride_n, q_stride_h, k_stride_n, k_stride_h, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "FusedQKNormRopeLauncher failed: " << cudaGetErrorString(status);
    return true;
  });
}
