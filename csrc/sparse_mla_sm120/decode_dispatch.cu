// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <flashinfer/attention/sparse_mla_sm120/execution/kernel_launch.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/cute_prefill/prefill_fp8.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv4_fp8_splitkv_decode.cuh>

#include "attention_descriptor.h"
#include "attention_dispatch.h"

namespace flashinfer::sparse_mla_sm120 {

cudaError_t dispatch_decode(const execution::AttentionParams& p,
                            const execution::ExecutionPlan& plan, cudaStream_t stream) {
  const bool compatibility = is_dsv41(static_cast<ModelType>(plan.metadata.model)) &&
                             plan.implementation == execution::Implementation::Ordinary;
  if (!compatibility && plan.implementation != execution::Implementation::Dsv4Fp8SplitKVDecode)
    return dispatch_legacy_decode(p, plan, stream);
  const auto& m = plan.metadata;
  dsv4_fp8_splitkv_decode::Params a{};
  a.q = p.q;
  a.kv = p.kv;
  a.extra_kv = p.extra_kv;
  a.indices = p.indices;
  a.extra_indices = p.extra_indices;
  a.lengths = p.topk_length;
  a.extra_lengths = p.extra_topk_length;
  a.positions = p.mega.positions;
  a.cos_sin = p.mega.cos_sin;
  a.sink = p.attn_sink;
  a.output = p.mega.output;
  a.scales = p.mega.scales;
  a.bf16_output = p.output;
  a.lse = p.out_lse;
  a.tokens = m.tokens;
  a.heads = m.heads;
  a.topk = m.topk;
  a.extra_topk = m.extra_topk;
  a.kv_stride = m.row_stride_bytes;
  a.extra_stride = m.extra_row_stride_bytes;
  a.page_stride = m.page_stride_bytes;
  a.extra_page_stride = m.extra_page_stride_bytes;
  a.page_size = m.page_size;
  a.extra_page_size = m.extra_page_size;
  a.page_div = flashinfer::uint_fastdiv(uint32_t(m.page_size));
  a.extra_page_div = flashinfer::uint_fastdiv(uint32_t(m.extra_topk ? m.extra_page_size : 1));
  a.sm_scale = p.sm_scale;
  a.lse_stride = m.lse_stride;
  a.lse_scale = p.lse_scale;
  a.partial = p.mid_out;
  a.partial_lse = p.mid_lse;
  a.splits = plan.active_splits;
  a.partial_heads = plan.scratch_heads;
  a.partial_stride =
      plan.partial_bytes ? (compatibility ? p.allocated_splits : plan.scratch_split_stride) : 0;
  a.chunks_per_block = compatibility ? plan.cpb : 0;
  a.indices_stride = m.indices_stride;
  a.extra_indices_stride = m.extra_indices_stride;
  auto launch = [&](auto groups, auto fused, auto split, auto inline_scale, auto partial) {
    constexpr int W = decltype(groups)::value;
    constexpr bool PartialOnly = decltype(partial)::value;
    constexpr bool Mega = decltype(fused)::value;
    auto kernel = dsv4_fp8_splitkv_decode::decode<W, Mega, decltype(split)::value,
                                                  decltype(inline_scale)::value, PartialOnly>;
    static thread_local bool configured[32] = {};
    auto status = configure_dynamic_smem_per_device(
        kernel, sizeof(dsv4_fp8_splitkv_decode::Shared<W>), configured);
    if (status != cudaSuccess) return status;
    static thread_local bool carveout_set[32] = {};
    int device = 0;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    const bool cacheable = device >= 0 && device < 32;
    if (!cacheable || !carveout_set[device]) {
      error = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
      if (error != cudaSuccess) return error;
      if (cacheable) carveout_set[device] = true;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(m.tokens, (m.heads + W * 8 - 1) / (W * 8), a.splits);
    config.blockDim = dim3((W * 4 + 1) * 32);
    config.dynamicSmemBytes = sizeof(dsv4_fp8_splitkv_decode::Shared<W>);
    config.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 1, unsigned(a.splits)};
    if (!PartialOnly && a.splits > 1) {
      config.attrs = &attr;
      config.numAttrs = 1;
    }
    auto result = cudaLaunchKernelEx(&config, kernel, a);
    if (result != cudaSuccess) return result;
    if constexpr (PartialOnly) {
      if (a.partial_stride > 0)
        dsv4_fp8_splitkv_decode::merge<decltype(split)::value, Mega>
            <<<dim3(m.tokens, m.heads), 64, 0, stream>>>(a);
      return cudaGetLastError();
    }
    return cudaSuccess;
  };
  auto heads = [&](auto fused, auto inline_scale, auto partial) {
    if (compatibility) {
      if (m.heads <= 8)
        return a.splits == 8 ? launch(cute::Int<1>{}, fused, cute::Int<8>{}, inline_scale, partial)
                             : launch(cute::Int<1>{}, fused, cute::Int<0>{}, inline_scale, partial);
      return a.splits == 8 ? launch(cute::Int<2>{}, fused, cute::Int<8>{}, inline_scale, partial)
                           : launch(cute::Int<2>{}, fused, cute::Int<0>{}, inline_scale, partial);
    }
    if (m.heads == 8)
      return a.splits == 8 ? launch(cute::Int<1>{}, fused, cute::Int<8>{}, inline_scale, partial)
                           : launch(cute::Int<1>{}, fused, cute::Int<0>{}, inline_scale, partial);
    if (m.heads <= 64)
      return a.splits == 8 ? launch(cute::Int<2>{}, fused, cute::Int<8>{}, inline_scale, partial)
                           : launch(cute::Int<2>{}, fused, cute::Int<0>{}, inline_scale, partial);
    return cudaErrorInvalidValue;
  };
  auto layout = [&](auto inlined) {
    if (!p.mega.output) return heads(std::false_type{}, inlined, std::true_type{});
    if (compatibility) return heads(std::true_type{}, inlined, std::true_type{});
    return heads(std::true_type{}, inlined, std::false_type{});
  };
  return m.model == int(ModelType::DSV4_1_INLINE) ? layout(std::true_type{})
                                                  : layout(std::false_type{});
}

namespace execution {
namespace {

int splitkv_head_groups(int heads) { return heads <= 8 ? 1 : 2; }

size_t splitkv_shared_bytes(int groups) {
  switch (groups) {
    case 1:
      return sizeof(dsv4_fp8_splitkv_decode::Shared<1>);
    default:
      return sizeof(dsv4_fp8_splitkv_decode::Shared<2>);
  }
}

Implementation get_implementation(const AttentionMetadata& m, int64_t mode, int64_t sm_count) {
  if (mode == 1) return Implementation::Dsv4Fp8SplitKVDecode;
  if (mode == 2) return Implementation::Dsv4Fp8Decode;

  const int groups = splitkv_head_groups(m.heads);
  const int64_t splitkv_tasks = int64_t(m.tokens) * ((m.heads + groups * 8 - 1) / (groups * 8));
  const int64_t decode_tasks = int64_t(m.tokens) * ((m.heads + 31) / 32);
  const int64_t splitkv_capacity = sm_count * (groups == 2 ? 1 : 2);
  const int64_t decode_waves = (decode_tasks + sm_count - 1) / sm_count;
  const int64_t splitkv_waves = (splitkv_tasks + splitkv_capacity - 1) / splitkv_capacity;
  // Admit the pipelined backbone at half an SM wave, then compare complete
  // waves. Its 32-head CTA has one resident block/SM; the split-KV CTA has two
  // except for the 64-candidate cooperative tile. The measured unsplit wave-cost ratio is 4/3.
  // This accounts for the backbone's second-wave step instead of assuming
  // that more queries always favor it. There is no runtime GPU timing/readback.
  if (decode_tasks * 2 >= sm_count && decode_waves * 3 <= splitkv_waves * 4)
    return Implementation::Dsv4Fp8Decode;
  return Implementation::Dsv4Fp8SplitKVDecode;
}

}  // namespace

void set_fp8_decode_resources(ExecutionPlan& p) {
  const int groups = p.metadata.heads <= 8 ? 1 : 2;
  p.block_threads = (groups * 4 + 1) * 32;
  p.shared_bytes = splitkv_shared_bytes(groups);
}

// Runs the existing metadata validation first; this resolver only changes scheduling.
ffi::Module resolve_descriptor(ffi::Array<int64_t> values, int64_t mode, int64_t sm_count,
                               int64_t max_shared) {
  auto m = unpack_metadata(values);
  TVM_FFI_ICHECK(mode >= 0 && mode <= 2 && sm_count > 0 && sm_count <= INT_MAX && max_shared > 0);
  m.variant = 0;
  auto p = resolve_attention(m, NumericRoute::FP8, 1, {int(sm_count), size_t(max_shared)});
  const bool footer = m.model == int(ModelType::DSV4_1);
  TVM_FFI_ICHECK(
      is_dsv41(static_cast<ModelType>(m.model)) && !m.extra_fp4 &&
      (m.heads == 8 || m.heads == 16 || m.heads == 32 || m.heads == 64) &&
      (footer || m.page_stride_bytes % m.row_stride_bytes == 0) &&
      m.indices_stride == size_t(m.topk) &&
      (!m.extra_topk || ((footer || m.extra_page_stride_bytes % m.extra_row_stride_bytes == 0) &&
                         m.extra_indices_stride == size_t(m.extra_topk))))
      << "DSv4 FP8 decode requires footer pages or flat inline FP8 KV, contiguous indices and "
         "H=8/16/32/64";
  p.implementation = get_implementation(m, mode, sm_count);
  p.scratch_heads = m.heads;
  p.merge = Merge::Direct;  // Mega finalizes in-kernel; BF16 decode uses the same merge separately.
  p.cpb = 0;
  if (p.implementation == Implementation::Dsv4Fp8Decode) {
    p.metadata.variant = 1;
    p.head_blocks = (m.heads + 31) / 32;
    p.active_splits = 1;
    p.block_threads = 384;
    p.shared_bytes = sizeof(cute_prefill::integrated::Shared);
    p.partial_bytes = 0;
    p.lse_bytes = 0;
  } else {
    const int groups = splitkv_head_groups(m.heads);
    p.metadata.variant = 0;
    p.head_blocks = (m.heads + groups * 8 - 1) / (groups * 8);
    const int64_t tasks = int64_t(m.tokens) * p.head_blocks;
    const int64_t capacity = sm_count * (groups == 2 ? 1 : 2);
    const int tile = groups == 2 ? 64 : 32;
    const int tiles = (m.topk + tile - 1) / tile + (m.extra_topk + tile - 1) / tile;
    // Portable clusters have at most eight CTAs. Cap the power-of-two choice
    // by live tile capacity; actual device lengths can still make chunks empty.
    int splits = 1;
    while (splits < 8 && tasks * splits * 2 <= capacity) splits *= 2;
    p.active_splits = std::max(1, std::min(splits, tiles));
    p.block_threads = (groups * 4 + 1) * 32;
    p.shared_bytes = splitkv_shared_bytes(groups);
    const size_t slots = p.active_splits > 1 ? size_t(m.tokens) * m.heads * p.active_splits : 0;
    p.partial_bytes = slots * 512 * sizeof(bf16);
    p.lse_bytes = slots * sizeof(float);
  }
  p.chunk_capacity = p.scratch_split_stride = p.active_splits;
  TVM_FFI_ICHECK_LE(p.shared_bytes, max_shared);
  return pack_plan(p);
}
}  // namespace execution
}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(resolve_decode_attention,
                              flashinfer::sparse_mla_sm120::execution::resolve_descriptor);
