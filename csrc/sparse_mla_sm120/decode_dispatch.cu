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
  if (plan.implementation != execution::Implementation::Dsv4Fp8SplitKVDecode)
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
  auto launch = [&](auto warps, auto fused, auto split, auto inline_scale) {
    constexpr int W = decltype(warps)::value;
    constexpr bool Mega = decltype(fused)::value;
    auto kernel = dsv4_fp8_splitkv_decode::decode<W, Mega, decltype(split)::value,
                                                  decltype(inline_scale)::value>;
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
    config.gridDim = dim3(m.tokens, plan.head_blocks, a.splits);
    config.blockDim = dim3((W + 1) * 32);
    config.dynamicSmemBytes = sizeof(dsv4_fp8_splitkv_decode::Shared<W>);
    config.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 1, unsigned(a.splits)};
    if (a.splits > 1) {
      config.attrs = &attr;
      config.numAttrs = 1;
    }
    return cudaLaunchKernelEx(&config, kernel, a);
  };
  auto heads = [&](auto fused, auto inline_scale) {
    if (m.heads == 8) return launch(cute::Int<1>{}, fused, cute::Int<0>{}, inline_scale);
    if (m.heads == 16) return launch(cute::Int<2>{}, fused, cute::Int<0>{}, inline_scale);
    if (m.heads == 64) {
      return a.splits == 8 ? launch(cute::Int<3>{}, fused, cute::Int<8>{}, inline_scale)
                           : launch(cute::Int<3>{}, fused, cute::Int<0>{}, inline_scale);
    }
    return launch(cute::Int<4>{}, fused, cute::Int<0>{}, inline_scale);
  };
  if (m.model == int(ModelType::DSV4_1_INLINE))
    return p.mega.output ? heads(std::true_type{}, std::true_type{})
                         : heads(std::false_type{}, std::true_type{});
  return p.mega.output ? heads(std::true_type{}, std::false_type{})
                       : heads(std::false_type{}, std::false_type{});
}

namespace execution {
namespace {

int splitkv_compute_warps(int heads) { return heads == 64 ? 3 : std::min(heads / 8, 4); }

size_t splitkv_shared_bytes(int warps) {
  switch (warps) {
    case 1:
      return sizeof(dsv4_fp8_splitkv_decode::Shared<1>);
    case 2:
      return sizeof(dsv4_fp8_splitkv_decode::Shared<2>);
    case 3:
      return sizeof(dsv4_fp8_splitkv_decode::Shared<3>);
    default:
      return sizeof(dsv4_fp8_splitkv_decode::Shared<4>);
  }
}

Implementation get_implementation(const AttentionMetadata& m, int64_t mode, int64_t sm_count) {
  if (mode == 1) return Implementation::Dsv4Fp8SplitKVDecode;
  if (mode == 2) return Implementation::Dsv4Fp8Decode;

  const int warps = splitkv_compute_warps(m.heads);
  const int64_t splitkv_tasks = int64_t(m.tokens) * ((m.heads + warps * 8 - 1) / (warps * 8));
  const int64_t decode_tasks = int64_t(m.tokens) * ((m.heads + 31) / 32);
  const int64_t splitkv_capacity = sm_count * (warps == 4 ? 1 : 2);
  const int64_t decode_waves = (decode_tasks + sm_count - 1) / sm_count;
  const int64_t splitkv_waves = (splitkv_tasks + splitkv_capacity - 1) / splitkv_capacity;
  // Admit the pipelined backbone at half an SM wave, then compare complete
  // waves. Its 32-head CTA has one resident block/SM; the split-KV CTA has two
  // except for four compute warps. The measured unsplit wave-cost ratio is 4/3.
  // This accounts for the backbone's second-wave step instead of assuming
  // that more queries always favor it. There is no runtime GPU timing/readback.
  if (decode_tasks * 2 >= sm_count && decode_waves * 3 <= splitkv_waves * 4)
    return Implementation::Dsv4Fp8Decode;
  return Implementation::Dsv4Fp8SplitKVDecode;
}

}  // namespace

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
      (footer || m.page_stride_bytes == size_t(m.page_size) * m.row_stride_bytes) &&
      m.indices_stride == size_t(m.topk) &&
      (!m.extra_topk || ((footer || m.extra_page_stride_bytes ==
                                        size_t(m.extra_page_size) * m.extra_row_stride_bytes) &&
                         m.extra_indices_stride == size_t(m.extra_topk))))
      << "DSv4 FP8 decode requires footer pages or flat inline FP8 KV, contiguous indices and "
         "H=8/16/32/64";
  p.implementation = get_implementation(m, mode, sm_count);
  p.scratch_heads = m.heads;
  p.merge = Merge::Direct;  // Neither implementation launches a separate merge.
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
    const int warps = splitkv_compute_warps(m.heads);
    p.metadata.variant = 0;
    p.head_blocks = (m.heads + warps * 8 - 1) / (warps * 8);
    const int64_t tasks = int64_t(m.tokens) * p.head_blocks;
    const int64_t capacity = sm_count * (warps == 4 ? 1 : 2);
    const int tiles = (m.topk + 31) / 32 + (m.extra_topk + 31) / 32;
    // Portable clusters have at most eight CTAs. Cap the power-of-two choice
    // by live tile capacity; actual device lengths can still make chunks empty.
    int splits = 1;
    while (splits < 8 && tasks * splits * 2 <= capacity) splits *= 2;
    p.active_splits = std::max(1, std::min(splits, tiles));
    p.block_threads = (warps + 1) * 32;
    p.shared_bytes = splitkv_shared_bytes(warps);
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
