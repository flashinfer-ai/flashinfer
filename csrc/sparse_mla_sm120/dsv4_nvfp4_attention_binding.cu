// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0

#include "../tvm_ffi_utils.h"
#include "attention_descriptor.h"
#include "attention_dispatch.h"
#include "attention_validation.h"
#include "dsv4_nvfp4_validation.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120::nvfp4 {

namespace {

void check_output_alignment(const TensorView& output, const execution::ExecutionPlan& plan) {
  if (plan.merge == execution::Merge::Stage1) return;
  check_attention_alignment(output, plan.merge == execution::Merge::Direct ? 4 : 16, "output");
}

void attention(TensorView q, TensorView kv_cache, TensorView indices, Optional<TensorView> mid_out,
               Optional<TensorView> mid_lse, TensorView output, TensorView out_lse,
               int64_t num_splits, double sm_scale, Optional<TensorView> topk_length,
               Optional<TensorView> attn_sink, Optional<TensorView> extra_kv_cache,
               Optional<TensorView> extra_indices, Optional<TensorView> extra_topk_length,
               int64_t cpb, bool stage1_only, bool prefill,
               const execution::ExecutionPlan* prepared = nullptr, double lse_scale = 1.0) {
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  check_attention_alignment(q, alignof(__nv_bfloat162), "q");
  CHECK_CUDA(kv_cache);
  CHECK_INPUT_AND_TYPE(indices, dl_int32);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(out_lse, dl_float32);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q.size(2), 512);
  TVM_FFI_ICHECK(q.IsContiguous()) << "q must be contiguous";
  TVM_FFI_ICHECK_EQ(indices.ndim(), 2);
  TVM_FFI_ICHECK(indices.IsContiguous());
  TVM_FFI_ICHECK(output.IsContiguous());
  TVM_FFI_ICHECK(out_lse.IsContiguous());
  CHECK_DEVICE(q, kv_cache);
  CHECK_DEVICE(q, indices);
  CHECK_DEVICE(q, output);
  CHECK_DEVICE(q, out_lse);
  TVM_FFI_ICHECK_LE(q.size(0), INT_MAX) << "NVFP4 tokens exceed integer range";
  TVM_FFI_ICHECK(has_instance(q.size(1), indices.size(1))) << "unsupported NVFP4 heads/topk";
  const int tokens = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(1));
  TVM_FFI_ICHECK_EQ(indices.size(0), tokens);
  TVM_FFI_ICHECK_EQ(output.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.size(0), tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), heads);
  TVM_FFI_ICHECK_EQ(output.size(2), 512);
  TVM_FFI_ICHECK_EQ(out_lse.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out_lse.size(0), tokens);
  TVM_FFI_ICHECK_EQ(out_lse.size(1), heads);
  const bool dual = extra_kv_cache.has_value();
  TVM_FFI_ICHECK_EQ(dual, extra_indices.has_value())
      << "extra_kv_cache and extra_indices must be provided together";
  TVM_FFI_ICHECK(!extra_topk_length.has_value() || dual)
      << "extra_topk_length requires an extra cache";
  const auto layout = parse_nvfp4_paged_layout(kv_cache);
  TVM_FFI_ICHECK_EQ(layout.page_size, execution::FixedPageSize)
      << "NVFP4 attention supports page_size=64";
  Dsv4Nvfp4AttentionParams p{};
  p.q = static_cast<const bf16*>(q.data_ptr());
  p.cache = static_cast<const uint8_t*>(kv_cache.data_ptr());
  p.indices = static_cast<const int32_t*>(indices.data_ptr());
  p.output = static_cast<bf16*>(output.data_ptr());
  p.out_lse = static_cast<float*>(out_lse.data_ptr());
  p.num_tokens = tokens;
  p.sm_scale = static_cast<float>(sm_scale);
  p.lse_scale = static_cast<float>(lse_scale);
  p.page_stride_bytes = layout.page_stride_bytes;
  auto length_pointer = [&](Optional<TensorView> value, const char* name) -> const int* {
    if (!value.has_value()) return nullptr;
    const auto& length = value.value();
    CHECK_CUDA(length);
    CHECK_INPUT_TYPE(length, dl_int32);
    CHECK_DEVICE(q, length);
    TVM_FFI_ICHECK_EQ(length.ndim(), 1);
    TVM_FFI_ICHECK_EQ(length.size(0), tokens);
    TVM_FFI_ICHECK(length.IsContiguous()) << name << " must be contiguous";
    return static_cast<const int*>(length.data_ptr());
  };
  p.topk_length = length_pointer(topk_length, "topk_length");
  if (dual) {
    const auto& cache = extra_kv_cache.value();
    const auto& idx = extra_indices.value();
    CHECK_CUDA(cache);
    CHECK_INPUT_TYPE(cache, dl_uint8);
    CHECK_INPUT_AND_TYPE(idx, dl_int32);
    CHECK_DEVICE(q, cache);
    CHECK_DEVICE(q, idx);
    TVM_FFI_ICHECK_EQ(idx.ndim(), 2);
    TVM_FFI_ICHECK(idx.IsContiguous());
    TVM_FFI_ICHECK_EQ(idx.size(0), tokens);
    TVM_FFI_ICHECK_GT(idx.size(1), 0);
    TVM_FFI_ICHECK_LE(idx.size(1), MaxExtraTopK) << "NVFP4 extra_topk exceeds integer chunk range";
    p.extra_topk = static_cast<int>(idx.size(1));
    const auto extra_layout = parse_nvfp4_paged_layout(cache);
    TVM_FFI_ICHECK(
        execution::visit_extra_page(extra_layout.page_size, [](auto) { return true; }).supported)
        << "NVFP4 extra cache page_size must be 2 or 64";
    p.extra_cache = static_cast<const uint8_t*>(cache.data_ptr());
    p.extra_indices = static_cast<const int32_t*>(idx.data_ptr());
    p.extra_page_size = extra_layout.page_size;
    p.extra_page_stride_bytes = extra_layout.page_stride_bytes;
    p.extra_topk_length = length_pointer(extra_topk_length, "extra_topk_length");
  }
  if (attn_sink.has_value()) {
    const auto& sink = attn_sink.value();
    CHECK_CUDA(sink);
    CHECK_INPUT_TYPE(sink, dl_float32);
    CHECK_DEVICE(q, sink);
    TVM_FFI_ICHECK_EQ(sink.ndim(), 1);
    TVM_FFI_ICHECK_EQ(sink.size(0), heads);
    TVM_FFI_ICHECK(sink.IsContiguous()) << "attn_sink must be contiguous";
    p.attn_sink = static_cast<const float*>(sink.data_ptr());
  }
  if (!prefill && (prepared == nullptr || prepared->partial_bytes != 0)) {
    TVM_FFI_ICHECK(mid_out.has_value() && mid_lse.has_value());
    const auto& mo = mid_out.value();
    const auto& ml = mid_lse.value();
    CHECK_INPUT_AND_TYPE(mo, dl_bfloat16);
    CHECK_INPUT_AND_TYPE(ml, dl_float32);
    CHECK_DEVICE(q, mo);
    CHECK_DEVICE(q, ml);
    TVM_FFI_ICHECK(mo.IsContiguous()) << "mid_out must be contiguous";
    TVM_FFI_ICHECK(ml.IsContiguous()) << "mid_lse must be contiguous";
    TVM_FFI_ICHECK_GT(num_splits, 0);
    TVM_FFI_ICHECK_EQ(num_splits, (topk + 63) / 64 + (p.extra_topk + 63) / 64);
    TVM_FFI_ICHECK_GE(cpb, 0);
    TVM_FFI_ICHECK_LE(cpb, num_splits);
    if (prepared != nullptr) {
      TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(mo.data_ptr()) % 16, 0)
          << "DSV4 NVFP4 partial workspace must be 16B-aligned";
      TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(ml.data_ptr()) % 16, 0)
          << "DSV4 NVFP4 LSE workspace must be 16B-aligned";
      TVM_FFI_ICHECK_GE(size_t(mo.numel()) * sizeof(bf16), prepared->partial_bytes)
          << "DSV4 NVFP4 execution plan partial capacity mismatch";
      TVM_FFI_ICHECK_GE(size_t(ml.numel()) * sizeof(float), prepared->lse_bytes)
          << "DSV4 NVFP4 execution plan LSE capacity mismatch";
    } else {
      TVM_FFI_ICHECK_EQ(mo.ndim(), 4);
      TVM_FFI_ICHECK_EQ(ml.ndim(), 3);
      TVM_FFI_ICHECK_EQ(mo.size(0), tokens);
      TVM_FFI_ICHECK_EQ(mo.size(1), heads);
      TVM_FFI_ICHECK_EQ(mo.size(2), num_splits);
      TVM_FFI_ICHECK_EQ(mo.size(3), 512);
      TVM_FFI_ICHECK_EQ(ml.size(0), tokens);
      TVM_FFI_ICHECK_EQ(ml.size(1), heads);
      TVM_FFI_ICHECK_EQ(ml.size(2), num_splits);
    }
    p.mid_out = static_cast<bf16*>(mo.data_ptr());
    p.mid_lse = static_cast<float*>(ml.data_ptr());
  }
  if (tokens == 0) return;
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const auto stream = get_stream(q.device());
  if (prepared != nullptr) {
    const auto& m = prepared->metadata;
    TVM_FFI_ICHECK(
        m.tokens == tokens && m.heads == heads && m.topk == topk && m.extra_topk == p.extra_topk &&
        m.page_stride_bytes == p.page_stride_bytes &&
        m.extra_page_stride_bytes == p.extra_page_stride_bytes &&
        m.extra_page_size == p.extra_page_size && m.has_lengths == topk_length.has_value() &&
        m.has_extra_lengths == extra_topk_length.has_value() && m.has_sink == attn_sink.has_value())
        << "DSV4 NVFP4 execution plan tensor metadata mismatch";
    check_output_alignment(output, *prepared);
    const auto status = dispatch_attention(p, *prepared, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "NVFP4 execute_attention: " << cudaGetErrorString(status);
    return;
  }
  int sm_count = 0, max_shared = 0;
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, q.device().device_id),
      cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                           q.device().device_id),
                    cudaSuccess);
  // cpb narrows to resolve_dsv4_nvfp4's int parameter; reject values that
  // would wrap instead of silently selecting a different chunking.
  TVM_FFI_ICHECK(cpb >= 0 && cpb <= INT_MAX) << "NVFP4 cpb is out of int range: " << cpb;
  auto plan = execution::resolve_dsv4_nvfp4(
      tokens, heads, topk, p.extra_topk, layout.page_size, p.extra_page_size, p.page_stride_bytes,
      p.extra_page_stride_bytes, cpb, sm_count, max_shared, prefill, stage1_only);
  plan.metadata.has_lengths = topk_length.has_value();
  plan.metadata.has_extra_lengths = extra_topk_length.has_value();
  plan.metadata.has_sink = attn_sink.has_value();
  check_output_alignment(output, plan);
  const auto status = dispatch_attention(p, plan, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "NVFP4 dispatch_attention: " << cudaGetErrorString(status);
}

}  // namespace

void SparseMlaSm120NVFP4Decode(TensorView q, TensorView kv_cache, TensorView indices,
                               TensorView mid_out, TensorView mid_lse, TensorView output,
                               TensorView out_lse, int64_t num_splits, double sm_scale,
                               Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                               Optional<TensorView> extra_kv_cache,
                               Optional<TensorView> extra_indices,
                               Optional<TensorView> extra_topk_length,
                               int64_t chunks_per_block_override, bool stage1_only,
                               double lse_scale) {
  attention(q, kv_cache, indices, mid_out, mid_lse, output, out_lse, num_splits, sm_scale,
            topk_length, attn_sink, extra_kv_cache, extra_indices, extra_topk_length,
            chunks_per_block_override, stage1_only, false, nullptr, lse_scale);
}

void SparseMlaSm120NVFP4Prefill(TensorView q, TensorView kv_cache, TensorView indices,
                                TensorView output, TensorView out_lse, double sm_scale,
                                Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                                Optional<TensorView> extra_kv_cache,
                                Optional<TensorView> extra_indices,
                                Optional<TensorView> extra_topk_length, double lse_scale) {
  attention(q, kv_cache, indices, Optional<TensorView>(), Optional<TensorView>(), output, out_lse,
            0, sm_scale, topk_length, attn_sink, extra_kv_cache, extra_indices, extra_topk_length,
            0, false, true, nullptr, lse_scale);
}

void ExecuteAttentionPlan(ffi::Module descriptor, TensorView q, TensorView cache,
                          TensorView indices, Optional<TensorView> mid, Optional<TensorView> mlse,
                          TensorView output, TensorView lse, double scale,
                          Optional<TensorView> lengths, Optional<TensorView> sink,
                          Optional<TensorView> extra_cache, Optional<TensorView> extra_indices,
                          Optional<TensorView> extra_lengths, double lse_scale) {
  const auto& plan = execution::unpack_plan(descriptor, true);
  TVM_FFI_ICHECK(q.ndim() == 3 && q.size(0) == plan.metadata.tokens &&
                 q.size(1) == plan.metadata.heads)
      << "DSV4 NVFP4 execution plan query mismatch";
  attention(q, cache, indices, mid, mlse, output, lse, plan.chunk_capacity, scale, lengths, sink,
            extra_cache, extra_indices, extra_lengths, plan.cpb,
            plan.merge == execution::Merge::Stage1, plan.metadata.variant == 1, &plan, lse_scale);
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv4_nvfp4_inspect_metadata,
                              flashinfer::sparse_mla_sm120::inspect_attention_metadata<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv4_nvfp4_execute_attention,
                              flashinfer::sparse_mla_sm120::nvfp4::ExecuteAttentionPlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_nvfp4_decode,
                              flashinfer::sparse_mla_sm120::nvfp4::SparseMlaSm120NVFP4Decode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_nvfp4_prefill,
                              flashinfer::sparse_mla_sm120::nvfp4::SparseMlaSm120NVFP4Prefill);
