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

#include <cuda_runtime.h>

#include <flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/decode_dsv4_nvfp4_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/streaming_dsv4_nvfp4_kernel.cuh>

#include "sparse_mla_sm120_nvfp4_common.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120::nvfp4 {

#define NVFP4_CUDA_CHECK(call)                                            \
  do {                                                                    \
    const cudaError_t status = (call);                                    \
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status); \
  } while (0)

namespace {

template <int NUM_HEADS, int TOPK, int PAGE_SIZE, bool DUAL_CACHE>
void launch_decode(const bf16* q, const uint8_t* cache, const int32_t* indices, bf16* mid_out,
                   float* mid_lse, bf16* output, float* out_lse, const int* topk_length,
                   const float* attn_sink, const uint8_t* extra_cache, const int32_t* extra_indices,
                   const int* extra_topk_length, int extra_topk, int extra_page_size,
                   size_t extra_page_stride, int num_tokens, int num_splits,
                   int chunks_per_block_override, float sm_scale, size_t page_stride,
                   bool stage1_only, cudaStream_t stream) {
  constexpr bool CAN_GROUP_HEADS = NUM_HEADS >= STREAMING_HEADS_PER_CTA;
  constexpr int GROUPED_H_BLOCKS =
      (NUM_HEADS + STREAMING_HEADS_PER_CTA - 1) / STREAMING_HEADS_PER_CTA;
  constexpr int UNGROUPED_H_BLOCKS = (NUM_HEADS + HPB - 1) / HPB;
  // Grouping amortizes the local V conversion only when both the candidate
  // range and the grouped grid are large enough. Keep the lower-overhead
  // 16-head kernel for short attention and the small-batch tail.
  const bool use_grouped = CAN_GROUP_HEADS && num_splits >= 8 && num_tokens * GROUPED_H_BLOCKS >= 8;
  const int h_blocks = use_grouped ? GROUPED_H_BLOCKS : UNGROUPED_H_BLOCKS;

  int chunks_per_block = chunks_per_block_override;
  if (chunks_per_block < 1 || chunks_per_block > num_splits) {
    int sm_count = 0;
    int device = 0;
    NVFP4_CUDA_CHECK(cudaGetDevice(&device));
    NVFP4_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    const int per_token_head = num_tokens * h_blocks;
    // Repeating Q quantization and the CTA-local V conversion for additional
    // split waves is materially more expensive than leaving a fraction of the
    // final wave empty. Select the fullest grid in the minimum feasible number
    // of waves; representative B=8/16/32,H=64 supported shapes select the
    // CPB that keeps roughly 80--96 CTAs resident in one wave.
    const int target_waves = (per_token_head + sm_count - 1) / sm_count;
    chunks_per_block = 1;
    float best_gap = static_cast<float>(target_waves) + 1.f;
    for (int cpb = 1; cpb <= num_splits; ++cpb) {
      const int effective_splits = (num_splits + cpb - 1) / cpb;
      const int active = per_token_head * effective_splits;
      const int ceil_waves = (active + sm_count - 1) / sm_count;
      if (ceil_waves != target_waves) continue;
      const float gap = static_cast<float>(ceil_waves) -
                        static_cast<float>(active) / static_cast<float>(sm_count);
      if (gap < best_gap - 1e-6f || (gap < best_gap + 1e-6f && cpb > chunks_per_block)) {
        best_gap = gap;
        chunks_per_block = cpb;
      }
    }
  }

  // Only ceil(num_chunks / chunks_per_block) split CTAs perform work. Pack
  // those partials densely at the front of the caller-provided scratch so the
  // merge neither launches empty CTAs nor scans sentinel-only split slots.
  const int active_splits = (num_splits + chunks_per_block - 1) / chunks_per_block;
  const bool write_direct = !stage1_only && active_splits == 1;
  if constexpr (CAN_GROUP_HEADS) {
    if (use_grouped) {
      constexpr size_t DYN_SMEM_BYTES = StreamingNVFP4Smem::SIZE;
      auto grouped_kernel =
          sparse_mla_streaming_dsv4_nvfp4_kernel<NUM_HEADS, TOPK, PAGE_SIZE, DUAL_CACHE>;
      NVFP4_CUDA_CHECK(cudaFuncSetAttribute(
          grouped_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));
      grouped_kernel<<<dim3(num_tokens, GROUPED_H_BLOCKS, active_splits),
                       dim3(STREAMING_BLOCK_THREADS), DYN_SMEM_BYTES, stream>>>(
          q, cache, indices, output, out_lse, mid_out, mid_lse, attn_sink, topk_length, extra_cache,
          extra_indices, extra_topk_length, extra_topk, extra_page_size, extra_page_stride,
          num_tokens, active_splits, chunks_per_block, sm_scale, page_stride, write_direct);
    }
  }
  if (!use_grouped) {
    constexpr size_t DYN_SMEM_BYTES = DecodeNVFP4Smem<ModelType::DSV4>::SIZE;
    auto kernel = sparse_mla_decode_dsv4_nvfp4_kernel<ModelType::DSV4, NUM_HEADS, TOPK, PAGE_SIZE,
                                                      DUAL_CACHE>;
    NVFP4_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));
    kernel<<<dim3(num_tokens, UNGROUPED_H_BLOCKS, active_splits), dim3(DECODE_BLOCK_THREADS),
             DYN_SMEM_BYTES, stream>>>(
        q, cache, indices, mid_out, mid_lse, output, out_lse, attn_sink, topk_length, extra_cache,
        extra_indices, extra_topk_length, extra_topk, extra_page_size, extra_page_stride,
        num_tokens, active_splits, chunks_per_block, sm_scale, page_stride, write_direct);
  }
  NVFP4_CUDA_CHECK(cudaGetLastError());

  if (stage1_only || write_direct) return;

  if (active_splits == 2) {
    constexpr int MERGE_H_BLOCKS = (NUM_HEADS + HPB - 1) / HPB;
    auto merge2_kernel = sparse_mla_decode_dsv4_nvfp4_merge2_kernel<NUM_HEADS>;
    merge2_kernel<<<dim3(num_tokens, MERGE_H_BLOCKS), dim3(DECODE_MERGE2_THREADS), 0, stream>>>(
        mid_out, mid_lse, output, out_lse, attn_sink, num_tokens);
    NVFP4_CUDA_CHECK(cudaGetLastError());
    return;
  }

  constexpr int MERGE_THREADS = 64;
  constexpr int DIMS_PER_THREAD = 512 / MERGE_THREADS;
  auto merge_kernel =
      sparse_mla_decode_dsv4_merge_kernel<NUM_HEADS, 512, MERGE_THREADS, DIMS_PER_THREAD>;
  const size_t merge_smem = static_cast<size_t>(active_splits) * sizeof(float);
  merge_kernel<<<dim3(num_tokens, NUM_HEADS), dim3(MERGE_THREADS), merge_smem, stream>>>(
      mid_out, mid_lse, output, out_lse, attn_sink, num_tokens, active_splits, NUM_HEADS, NUM_HEADS,
      NUM_HEADS);
  NVFP4_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void SparseMlaSm120NVFP4Decode(TensorView q, TensorView kv_cache, TensorView indices,
                               TensorView mid_out, TensorView mid_lse, TensorView output,
                               TensorView out_lse, int64_t num_splits, double sm_scale,
                               Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                               Optional<TensorView> extra_kv_cache,
                               Optional<TensorView> extra_indices,
                               Optional<TensorView> extra_topk_length,
                               int64_t chunks_per_block_override, bool stage1_only) {
  CHECK_CUDA(q);
  CHECK_CUDA(kv_cache);
  CHECK_CUDA(indices);
  CHECK_CUDA(mid_out);
  CHECK_CUDA(mid_lse);
  CHECK_CUDA(output);
  CHECK_CUDA(out_lse);
  TVM_FFI_ICHECK_EQ(q.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q.size(2), 512);
  TVM_FFI_ICHECK(q.IsContiguous()) << "q must be contiguous";
  TVM_FFI_ICHECK_EQ(indices.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(indices.ndim(), 2);
  TVM_FFI_ICHECK(indices.IsContiguous());
  TVM_FFI_ICHECK_EQ(mid_out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(mid_lse.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(out_lse.dtype(), dl_float32);
  TVM_FFI_ICHECK(mid_out.IsContiguous()) << "mid_out must be contiguous";
  TVM_FFI_ICHECK(mid_lse.IsContiguous()) << "mid_lse must be contiguous";
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(out_lse.IsContiguous()) << "out_lse must be contiguous";
  TVM_FFI_ICHECK_GT(num_splits, 0);

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(1));
  TVM_FFI_ICHECK_EQ(indices.size(0), num_tokens);
  TVM_FFI_ICHECK_GT(topk, 0);
  TVM_FFI_ICHECK_EQ(mid_out.ndim(), 4);
  TVM_FFI_ICHECK_EQ(mid_lse.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(output.size(2), 512);
  TVM_FFI_ICHECK_EQ(out_lse.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out_lse.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(out_lse.size(1), num_heads);
  TVM_FFI_ICHECK_GE(chunks_per_block_override, 0);
  TVM_FFI_ICHECK_LE(chunks_per_block_override, num_splits);

  CHECK_DEVICE(q, kv_cache);
  CHECK_DEVICE(q, indices);
  CHECK_DEVICE(q, mid_out);
  CHECK_DEVICE(q, mid_lse);
  CHECK_DEVICE(q, output);
  CHECK_DEVICE(q, out_lse);
  const bool has_extra = extra_kv_cache.has_value();
  TVM_FFI_ICHECK_EQ(has_extra, extra_indices.has_value())
      << "extra_kv_cache and extra_indices must be provided together";
  TVM_FFI_ICHECK(!extra_topk_length.has_value() || has_extra)
      << "extra_topk_length requires an extra cache";

  int extra_topk = 0;
  PagedLayout extra_layout{0, 0, 0};
  const uint8_t* extra_cache_ptr = nullptr;
  const int32_t* extra_indices_ptr = nullptr;
  const int* extra_topk_length_ptr = nullptr;
  if (has_extra) {
    const auto& extra_cache = extra_kv_cache.value();
    const auto& extra_idx = extra_indices.value();
    CHECK_CUDA(extra_cache);
    CHECK_INPUT_TYPE(extra_cache, dl_uint8);
    CHECK_INPUT_AND_TYPE(extra_idx, dl_int32);
    CHECK_DEVICE(q, extra_cache);
    CHECK_DEVICE(q, extra_idx);
    TVM_FFI_ICHECK_EQ(extra_idx.ndim(), 2);
    TVM_FFI_ICHECK(extra_idx.IsContiguous());
    TVM_FFI_ICHECK_EQ(extra_idx.size(0), num_tokens);
    extra_topk = static_cast<int>(extra_idx.size(1));
    TVM_FFI_ICHECK_GT(extra_topk, 0);
    extra_layout = parse_nvfp4_paged_layout(extra_cache);
    TVM_FFI_ICHECK(extra_layout.page_size == 2 || extra_layout.page_size == 64)
        << "NVFP4 extra cache page_size must be 2 or 64";
    extra_cache_ptr = static_cast<const uint8_t*>(extra_cache.data_ptr());
    extra_indices_ptr = static_cast<const int32_t*>(extra_idx.data_ptr());
    if (extra_topk_length.has_value()) {
      const auto& length = extra_topk_length.value();
      CHECK_INPUT_AND_TYPE(length, dl_int32);
      CHECK_DEVICE(q, length);
      TVM_FFI_ICHECK_EQ(length.ndim(), 1);
      TVM_FFI_ICHECK_EQ(length.size(0), num_tokens);
      extra_topk_length_ptr = static_cast<const int*>(length.data_ptr());
    }
  }
  const int expected_splits = (topk + DECODE_CAND_WINDOW - 1) / DECODE_CAND_WINDOW +
                              (extra_topk + DECODE_CAND_WINDOW - 1) / DECODE_CAND_WINDOW;
  TVM_FFI_ICHECK_EQ(num_splits, expected_splits);
  const PagedLayout layout = parse_nvfp4_paged_layout(kv_cache);
  TVM_FFI_ICHECK_EQ(layout.page_size, 64) << "initial NVFP4 decode supports page_size=64";
  TVM_FFI_ICHECK_EQ(mid_out.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(mid_out.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(mid_out.size(2), num_splits);
  TVM_FFI_ICHECK_EQ(mid_out.size(3), 512);
  TVM_FFI_ICHECK_EQ(mid_lse.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(mid_lse.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(mid_lse.size(2), num_splits);

  if (topk_length.has_value()) {
    const auto& length = topk_length.value();
    CHECK_INPUT_AND_TYPE(length, dl_int32);
    CHECK_DEVICE(q, length);
    TVM_FFI_ICHECK_EQ(length.ndim(), 1);
    TVM_FFI_ICHECK_EQ(length.size(0), num_tokens);
  }
  if (attn_sink.has_value()) {
    const auto& sink = attn_sink.value();
    CHECK_INPUT_AND_TYPE(sink, dl_float32);
    CHECK_DEVICE(q, sink);
    TVM_FFI_ICHECK_EQ(sink.ndim(), 1);
    TVM_FFI_ICHECK_EQ(sink.size(0), num_heads);
  }

  if (num_tokens == 0) return;

  const int* topk_length_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaStream_t stream = get_stream(q.device());
#define DISPATCH_NVFP4_DECODE(H, K)                                                                \
  if (num_heads == (H) && topk == (K)) {                                                           \
    if (has_extra) {                                                                               \
      launch_decode<(H), (K), 64, true>(                                                           \
          static_cast<const bf16*>(q.data_ptr()),                                                  \
          static_cast<const uint8_t*>(kv_cache.data_ptr()),                                        \
          static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()), \
          static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),          \
          static_cast<float*>(out_lse.data_ptr()), topk_length_ptr, attn_sink_ptr,                 \
          extra_cache_ptr, extra_indices_ptr, extra_topk_length_ptr, extra_topk,                   \
          extra_layout.page_size, extra_layout.page_stride_bytes, num_tokens,                      \
          static_cast<int>(num_splits), static_cast<int>(chunks_per_block_override),               \
          static_cast<float>(sm_scale), layout.page_stride_bytes, stage1_only, stream);            \
    } else {                                                                                       \
      launch_decode<(H), (K), 64, false>(                                                          \
          static_cast<const bf16*>(q.data_ptr()),                                                  \
          static_cast<const uint8_t*>(kv_cache.data_ptr()),                                        \
          static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()), \
          static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),          \
          static_cast<float*>(out_lse.data_ptr()), topk_length_ptr, attn_sink_ptr, nullptr,        \
          nullptr, nullptr, 0, 0, 0, num_tokens, static_cast<int>(num_splits),                     \
          static_cast<int>(chunks_per_block_override), static_cast<float>(sm_scale),               \
          layout.page_stride_bytes, stage1_only, stream);                                          \
    }                                                                                              \
    return;                                                                                        \
  }
  DISPATCH_NVFP4_DECODE(16, 128)
  DISPATCH_NVFP4_DECODE(16, 512)
  DISPATCH_NVFP4_DECODE(32, 128)
  DISPATCH_NVFP4_DECODE(32, 512)
  DISPATCH_NVFP4_DECODE(64, 128)
  DISPATCH_NVFP4_DECODE(64, 512)
  DISPATCH_NVFP4_DECODE(128, 128)
  DISPATCH_NVFP4_DECODE(128, 512)
#undef DISPATCH_NVFP4_DECODE
  TVM_FFI_ICHECK(false) << "unsupported initial NVFP4 decode shape: heads=" << num_heads
                        << ", topk=" << topk;
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_nvfp4_decode,
                              flashinfer::sparse_mla_sm120::nvfp4::SparseMlaSm120NVFP4Decode);

#undef NVFP4_CUDA_CHECK
