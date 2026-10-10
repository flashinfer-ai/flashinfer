// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <flashinfer/attention/sparse_mla_sm120/execution/merge.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv32_fp8/decode.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_sg.cuh>

#include "attention_validation.h"

namespace flashinfer::sparse_mla_sm120 {

constexpr ModelType GLM_NVFP4 = ModelType::GLM_NSA_NVFP4;

static void check_cuda(cudaError_t status) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

template <int HEADS>
void launch_glm_nvfp4(TensorView q, TensorView cache, TensorView indices, TensorView output,
                      TensorView lse, TensorView scale, double sm_scale, int64_t cpb,
                      ffi::Optional<TensorView> lengths, ffi::Optional<TensorView> sink,
                      ffi::Optional<TensorView> mid, ffi::Optional<TensorView> mid_lse) {
  const auto stream = get_stream(q.device());
  const auto* q_ptr = static_cast<const bf16*>(q.data_ptr());
  const auto* kv_ptr = static_cast<const uint8_t*>(cache.data_ptr());
  const auto* ix_ptr = static_cast<const int32_t*>(indices.data_ptr());
  auto* out_ptr = static_cast<bf16*>(output.data_ptr());
  auto* lse_ptr = static_cast<float*>(lse.data_ptr());
  const auto* scale_ptr = static_cast<const float*>(scale.data_ptr());
  const auto* len_ptr =
      lengths.has_value() ? static_cast<const int*>(lengths.value().data_ptr()) : nullptr;
  const auto* sink_ptr =
      sink.has_value() ? static_cast<const float*>(sink.value().data_ptr()) : nullptr;
  const int tokens = q.size(0), topk = indices.size(1);
  if (cpb > 0) {
    using Cfg = Dsv32DecodeConfig<GLM_NVFP4, HEADS>;
    const int chunks = (topk + Cfg::BI - 1) / Cfg::BI;
    TVM_FFI_ICHECK_LE(cpb, chunks);
    const int splits = (chunks + cpb - 1) / cpb;
    TVM_FFI_ICHECK(mid.has_value() && mid_lse.has_value()) << "decode requires FP32 scratch";
    CHECK_INPUT_AND_TYPE(mid.value(), dl_float32);
    CHECK_INPUT_AND_TYPE(mid_lse.value(), dl_float32);
    check_attention_device(q, mid.value(), "mid_out");
    check_attention_device(q, mid_lse.value(), "mid_lse");
    check_attention_alignment(mid.value(), 16, "mid_out");
    check_decode_scratch_capacity(q, mid.value(), mid_lse.value(), splits);
    auto* mid_ptr = static_cast<float*>(mid.value().data_ptr());
    auto* mlse_ptr = static_cast<float*>(mid_lse.value().data_ptr());
    constexpr int shared = Dsv32DecodeSmem<GLM_NVFP4, HEADS>::LAUNCH_BYTES;
    auto kernel = cpb == 1 ? sparse_mla_decode_dsv3_2_kernel<GLM_NVFP4, HEADS, true>
                           : sparse_mla_decode_dsv3_2_kernel<GLM_NVFP4, HEADS>;
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    kernel<<<dim3(tokens, (HEADS + 15) / 16, splits), Cfg::BLOCK_THREADS, shared, stream>>>(
        q_ptr, kv_ptr, ix_ptr, mid_ptr, mlse_ptr, len_ptr, tokens, HEADS, topk, splits, cpb,
        sm_scale, cache.stride(0), indices.stride(0), 416, flashinfer::uint_fastdiv(64), scale_ptr);
    check_cuda(cudaGetLastError());
    auto merge = sparse_mla_decode_dsv4_merge_kernel<HEADS, 512, 64, 8, float>;
    merge<<<dim3(tokens, HEADS), 64, splits * sizeof(float), stream>>>(
        mid_ptr, mlse_ptr, out_ptr, lse_ptr, sink_ptr, tokens, splits, HEADS, HEADS, lse.stride(0),
        1.f);
  } else {
    TVM_FFI_ICHECK_EQ(cpb, 0) << "prefill uses chunks_per_block=0";
    TVM_FFI_ICHECK(indices.IsContiguous() && topk % 64 == 0)
        << "prefill requires contiguous indices and topk divisible by 64";
    using L = SmemLayout<GLM_NVFP4, QkComputeMode::FP8>;
    auto kernel = sparse_mla_prefill_kernel<GLM_NVFP4, QkComputeMode::FP8, HEADS, 64>;
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, L::TOTAL));
    PrefillColdParams cold{};
    cold.sm_scale = sm_scale;
    cold.num_tokens = tokens;
    cold.kv_stride_bytes = 416;
    cold.out_lse_stride_elems = lse.stride(0);
    cold.topk = topk;
    cold.topk_length = len_ptr;
    cold.attn_sink = sink_ptr;
    cold.kv_global_scale = scale_ptr;
    kernel<<<tokens*((HEADS + 15) / 16), PrefillTileCfg<GLM_NVFP4>::BASE_BLOCK_THREADS, L::TOTAL,
             stream>>>(q_ptr, kv_ptr, ix_ptr, sink_ptr, out_ptr, lse_ptr, cold);
  }
  check_cuda(cudaGetLastError());
}

void SparseMlaGlmNvfp4(TensorView q, TensorView cache, TensorView indices, TensorView output,
                       TensorView lse, TensorView scale, double sm_scale, int64_t cpb,
                       ffi::Optional<TensorView> lengths, ffi::Optional<TensorView> sink,
                       ffi::Optional<TensorView> mid, ffi::Optional<TensorView> mid_lse) {
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_DIM(3, q);
  TVM_FFI_ICHECK(q.size(0) > 0 && q.size(2) == 576) << "expected BF16 [T, H, 576] queries";
  const int tokens = q.size(0), heads = q.size(1);
  check_attention_alignment(q, 16, "q");
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_DIM(3, output);
  TVM_FFI_ICHECK(output.size(0) == tokens && output.size(1) == heads && output.size(2) == 512);
  check_attention_alignment(output, 16, "output");
  CHECK_INPUT_TYPE(indices, dl_int32);
  CHECK_DIM(2, indices);
  TVM_FFI_ICHECK(indices.size(0) == tokens && indices.size(1) > 0 && indices.stride(1) == 1 &&
                 indices.stride(0) >= indices.size(1));
  CHECK_INPUT_TYPE(lse, dl_float32);
  CHECK_DIM(2, lse);
  TVM_FFI_ICHECK(lse.size(0) == tokens && lse.size(1) == heads && lse.stride(1) == 1 &&
                 lse.stride(0) >= heads);
  CHECK_INPUT_AND_TYPE(scale, dl_float32);
  TVM_FFI_ICHECK_EQ(scale.numel(), 1);
  TVM_FFI_ICHECK(cache.IsContiguous()) << "NVFP4 cache must contain packed 416-byte rows";
  auto layout = parse_paged_kv_layout(cache, 416, true, "kv_cache");
  TVM_FFI_ICHECK(layout.page_block_size == 64 && layout.stride_kv_row == 416 && cache.size(0) > 0);
  for (const auto& item : {cache, indices, output, lse, scale})
    check_attention_device(q, item, "tensor");
  check_attention_vector(q, lengths, tokens, dl_int32, "topk_length");
  check_attention_vector(q, sink, heads, dl_float32, "attn_sink");
  ffi::CUDADeviceGuard guard(q.device().device_id);
  int major = 0;
  check_cuda(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.device().device_id));
  TVM_FFI_ICHECK_EQ(major, 12) << "GLM NVFP4 sparse MLA requires SM120/SM121";
#define HEAD_CASE(H)                                                                              \
  case H:                                                                                         \
    launch_glm_nvfp4<H>(q, cache, indices, output, lse, scale, sm_scale, cpb, lengths, sink, mid, \
                        mid_lse);                                                                 \
    break
  switch (heads) {
    HEAD_CASE(8);
    HEAD_CASE(16);
    HEAD_CASE(32);
    HEAD_CASE(64);
    HEAD_CASE(128);
    default:
      TVM_FFI_ICHECK(false) << "supported query heads: 8, 16, 32, 64, 128";
  }
#undef HEAD_CASE
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_glm_nvfp4,
                              flashinfer::sparse_mla_sm120::SparseMlaGlmNvfp4);
