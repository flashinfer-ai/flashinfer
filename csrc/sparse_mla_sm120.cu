// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Sparse-MLA SM120 paged attention orchestrator (prefill-only).
//
// Decode for both DSV3_2 and DSV4 routes through the standalone
// SparseMlaSm120DecodeDsv3_2 / SparseMlaSm120DecodeDsv4 entry points from
// Python (see flashinfer/sparse_mla_sm120.py). This entry point handles
// prefill dispatch for both model types (with optional dual cache for DSV4).

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120 {

// Forward declaration (defined in sparse_mla_sm120_prefill.cu).
bool sparse_mla_prefill_dispatch(ModelType mt, int num_heads, int topk, int page_block_size,
                                 int topk_extra, int extra_page_block_size, const bf16* Q,
                                 const uint8_t* KV_cache, const int32_t* indices,
                                 const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                                 bf16* output, float* out_lse, float sm_scale, int num_tokens,
                                 size_t stride_kv_block, size_t extra_stride_kv_block,
                                 const float* attn_sink, const int* topk_length,
                                 const int* extra_topk_length, cudaStream_t stream);

namespace {

inline ModelType resolve_model_type(int d_qk, int64_t model_type) {
  constexpr int64_t kAuto = -1;
  if (d_qk == 576) {
    if (model_type == kAuto) return ModelType::DSV3_2;
    const auto mt = static_cast<ModelType>(model_type);
    TVM_FFI_ICHECK(mt == ModelType::DSV3_2 || mt == ModelType::GLM_NSA)
        << "d_qk=576 supports model_type auto, DSV3_2, or GLM_NSA; got " << model_type;
    return mt;
  }
  if (d_qk == 512) {
    const auto mt = static_cast<ModelType>(
        model_type == kAuto ? static_cast<int64_t>(ModelType::DSV4) : model_type);
    TVM_FFI_ICHECK(mt == ModelType::DSV4)
        << "d_qk=512 supports only model_type auto or DSV4; got " << model_type;
    return mt;
  }
  TVM_FFI_ICHECK(false) << "Unsupported d_qk=" << d_qk
                        << "; expected 576 (DSV3_2/GLM_NSA) or 512 (DSV4)";
  return ModelType::DSV4;
}

inline int bytes_per_token(ModelType mt) {
  switch (mt) {
    case ModelType::DSV3_2:
    case ModelType::GLM_NSA:
      return 656;
    case ModelType::DSV4:
      return 584;
  }
  TVM_FFI_ICHECK(false) << "Unsupported sparse MLA model type";
  return 0;
}

struct PagedKVLayout {
  int page_block_size;
  size_t stride_kv_block;
};

inline PagedKVLayout parse_paged_kv_layout(const TensorView& kv, int bpt, const char* name) {
  const size_t elem_bytes = static_cast<size_t>(kv.dtype().bits / 8);
  if (kv.ndim() == 2) {
    const size_t block_bytes = static_cast<size_t>(kv.size(1)) * elem_bytes;
    TVM_FFI_ICHECK_EQ(block_bytes % static_cast<size_t>(bpt), 0)
        << name << " 2D block width " << block_bytes
        << " is not divisible by bytes_per_token=" << bpt;
    return {static_cast<int>(block_bytes / static_cast<size_t>(bpt)), block_bytes};
  }
  if (kv.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(kv.size(-1), bpt)
        << name << " 3D form must be [num_pages, page_block_size, " << bpt << "]";
    return {static_cast<int>(kv.size(1)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  TVM_FFI_ICHECK_EQ(kv.ndim(), 4) << name << " must be 2D [num_pages, page_bytes], 3D "
                                  << "[num_pages, page_block_size, bytes_per_token], HND "
                                  << "[num_pages, 1, page_block_size, bytes_per_token], or NHD "
                                  << "[num_pages, page_block_size, 1, bytes_per_token]";
  TVM_FFI_ICHECK_EQ(kv.size(-1), bpt)
      << name << " last dim must be bytes_per_token=" << bpt << ", got " << kv.size(-1);
  if (kv.size(1) == 1) {
    return {static_cast<int>(kv.size(2)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  if (kv.size(2) == 1) {
    return {static_cast<int>(kv.size(1)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  TVM_FFI_ICHECK(false) << name << " 4D form must have singleton KV-head axis at dim 1 "
                        << "(HND) or dim 2 (NHD)";
  return {0, 0};
}

inline int check_dense_indices_2d_or_s_q_3d(const TensorView& idx, const char* name,
                                            int num_tokens) {
  TVM_FFI_ICHECK(idx.ndim() == 2 || idx.ndim() == 3)
      << name << " must be [num_tokens, topk] or [num_tokens, 1, topk]; got ndim=" << idx.ndim();
  const int width = static_cast<int>(idx.size(-1));
  TVM_FFI_ICHECK_EQ(idx.size(0), num_tokens) << name << " leading dimension must match num_tokens";
  if (idx.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(idx.size(1), 1) << name << " middle dimension must be singleton";
  }
  TVM_FFI_ICHECK_EQ(idx.stride(-1), 1) << name << " last dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(idx.stride(0), width)
      << name << " must be dense row-major over the token dimension";
  return width;
}

}  // namespace

void SparseMlaSm120PagedAttention(
    TensorView q,         // [num_tokens, num_heads, d_qk] bf16
    TensorView kv_cache,  // packed paged FP8; HND/NHD/3D/2D forms accepted
    TensorView indices,   // [num_tokens, topk] or [num_tokens, 1, topk] int32 (-1 = skip)
    TensorView output,    // [num_tokens, num_heads, d_v] bf16 — in-place
    TensorView out_lse,   // [num_tokens, num_heads] f32 — in-place
    double sm_scale, int64_t model_type,
    Optional<TensorView> topk_length,        // [num_tokens] int32, optional
    Optional<TensorView> attn_sink,          // [num_heads] f32, optional
    Optional<TensorView> extra_kv_cache,     // optional dual cache
    Optional<TensorView> extra_indices,      // optional dual cache indices
    Optional<TensorView> extra_topk_length)  // [num_tokens] int32, optional
{
  // ── Input validation ───────────────────────────────────────────────
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  // kv_cache: CUDA + last-dim contiguous only; padded block stride is OK.
  CHECK_CUDA(kv_cache);
  CHECK_LAST_DIM_CONTIGUOUS(kv_cache);
  CHECK_INPUT_TYPE(kv_cache, dl_uint8);
  CHECK_INPUT_AND_TYPE(indices, dl_int32);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(out_lse, dl_float32);

  CHECK_DIM(3, q);

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int d_qk = static_cast<int>(q.size(2));
  const int topk = check_dense_indices_2d_or_s_q_3d(indices, "indices", num_tokens);
  const ModelType mt = resolve_model_type(d_qk, model_type);
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, bytes_per_token(mt), "kv_cache");
  const int page_block_size = kv_layout.page_block_size;

  TVM_FFI_ICHECK_GT(num_heads, 0);
  TVM_FFI_ICHECK_LE(num_heads, 128);
  TVM_FFI_ICHECK_GT(topk, 0);
  TVM_FFI_ICHECK_GT(page_block_size, 0);
  TVM_FFI_ICHECK_EQ(output.ndim(), 3) << "output must be [num_tokens, num_heads, 512]";
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(output.size(2), 512) << "SM120 sparse-MLA requires d_v == 512";
  TVM_FFI_ICHECK_EQ(out_lse.ndim(), 2) << "out_lse must be [num_tokens, num_heads]";
  TVM_FFI_ICHECK_EQ(out_lse.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(out_lse.size(1), num_heads);

  if (topk_length.has_value()) {
    const auto& tl = topk_length.value();
    CHECK_INPUT_AND_TYPE(tl, dl_int32);
    TVM_FFI_ICHECK_EQ(tl.ndim(), 1) << "topk_length must be [num_tokens]";
    TVM_FFI_ICHECK_EQ(tl.size(0), num_tokens);
  }

  // attn_sink (per-head bias added pre-softmax).
  const float* attn_sink_ptr = nullptr;
  if (attn_sink.has_value()) {
    const auto& s = attn_sink.value();
    CHECK_INPUT_AND_TYPE(s, dl_float32);
    TVM_FFI_ICHECK_EQ(s.ndim(), 1);
    TVM_FFI_ICHECK_EQ(s.size(0), num_heads) << "attn_sink must be [num_heads]";
    attn_sink_ptr = static_cast<const float*>(s.data_ptr());
  }

  // Optional dual-cache extras.
  const uint8_t* extra_kv_ptr = nullptr;
  const int32_t* extra_idx_ptr = nullptr;
  int extra_page_block_size = 0;
  size_t extra_stride_kv_block = 0;
  int extra_topk = 0;
  TVM_FFI_ICHECK(!extra_indices.has_value() || extra_kv_cache.has_value())
      << "extra_indices requires extra_kv_cache";
  TVM_FFI_ICHECK(!extra_topk_length.has_value() || extra_kv_cache.has_value())
      << "extra_topk_length requires extra_kv_cache";
  if (extra_kv_cache.has_value()) {
    TVM_FFI_ICHECK(extra_indices.has_value()) << "extra_kv_cache requires extra_indices";
    const auto& ekv = extra_kv_cache.value();
    const auto& eidx = extra_indices.value();
    // Same relaxation as the main kv_cache: padded block stride is OK.
    CHECK_CUDA(ekv);
    CHECK_LAST_DIM_CONTIGUOUS(ekv);
    CHECK_INPUT_TYPE(ekv, dl_uint8);
    CHECK_INPUT_AND_TYPE(eidx, dl_int32);
    const PagedKVLayout extra_layout =
        parse_paged_kv_layout(ekv, bytes_per_token(mt), "extra_kv_cache");
    extra_page_block_size = extra_layout.page_block_size;
    extra_stride_kv_block = extra_layout.stride_kv_block;
    extra_topk = check_dense_indices_2d_or_s_q_3d(eidx, "extra_indices", num_tokens);
    TVM_FFI_ICHECK_GT(extra_page_block_size, 0);
    TVM_FFI_ICHECK_GT(extra_topk, 0);
    extra_kv_ptr = static_cast<const uint8_t*>(ekv.data_ptr());
    extra_idx_ptr = static_cast<const int32_t*>(eidx.data_ptr());
    if (extra_topk_length.has_value()) {
      const auto& etl = extra_topk_length.value();
      CHECK_INPUT_AND_TYPE(etl, dl_int32);
      TVM_FFI_ICHECK_EQ(etl.ndim(), 1) << "extra_topk_length must be [num_tokens]";
      TVM_FFI_ICHECK_EQ(etl.size(0), num_tokens);
    }
  }

  const int* tl_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const int* etl_ptr = extra_topk_length.has_value()
                           ? static_cast<const int*>(extra_topk_length.value().data_ptr())
                           : nullptr;

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  const auto Q_ptr = static_cast<const bf16*>(q.data_ptr());
  const auto KV_ptr = static_cast<const uint8_t*>(kv_cache.data_ptr());
  const auto idx_ptr = static_cast<const int32_t*>(indices.data_ptr());
  const auto O_ptr = static_cast<bf16*>(output.data_ptr());
  const auto LSE_ptr = static_cast<float*>(out_lse.data_ptr());

  // Decode (num_tokens <= 64) is dispatched by Python directly through the
  // standalone decode-dsv3_2 / decode-dsv4 entry points. The orchestrator
  // only handles prefill.
  TVM_FFI_ICHECK_GT(num_tokens, 64)
      << "Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2 "
         "or sparse_mla_sm120_decode_dsv4; got num_tokens="
      << num_tokens;

  const bool ok = sparse_mla_prefill_dispatch(
      mt, num_heads, topk, page_block_size, extra_topk, extra_page_block_size, Q_ptr, KV_ptr,
      idx_ptr, extra_kv_ptr, extra_idx_ptr, O_ptr, LSE_ptr, static_cast<float>(sm_scale),
      num_tokens, kv_layout.stride_kv_block, extra_stride_kv_block, attn_sink_ptr, tl_ptr, etl_ptr,
      stream);
  TVM_FFI_ICHECK(ok) << "Unsupported sparse-MLA prefill configuration: "
                     << "model="
                     << (mt == ModelType::DSV3_2 ? "DSV3_2"
                                                 : (mt == ModelType::GLM_NSA ? "GLM_NSA" : "DSV4"))
                     << " num_heads=" << num_heads << " topk=" << topk
                     << " page_block_size=" << page_block_size << " topk_extra=" << extra_topk
                     << " extra_page_block_size=" << extra_page_block_size;
}

}  // namespace flashinfer::sparse_mla_sm120
