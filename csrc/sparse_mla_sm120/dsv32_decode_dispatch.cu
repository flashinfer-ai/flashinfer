// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// V32 (DSv3.2) decode — forked from decode-dsv4. Warp-spec (1 IO + 8 math =
// 288 threads), TMA bulk gather of FP8 INLINE 656 B/token KV cache, double
// buffered with per-buffer mbarrier pairs, static grid (num_tokens × HBLOCKS
// × num_splits). Reuses decode-dsv4's merge kernel for split combine.
//
// Supports the V32-family dispatch grid: dedicated instantiations at
//   num_heads ∈ {8, 16, 32, 64, 128}
// plus one runtime-H instantiation (any num_heads <= 128 off the grid) and
// GLM53_NOPE dedicated 8/16/32/64 + runtime-H. topk is a runtime argument — one
// instantiation serves every indices-row width.

#include <cuda_runtime.h>

#include <flashinfer/attention/sparse_mla_sm120/execution/merge.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv32_fp8/decode.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

#include "../tvm_ffi_utils.h"
#include "attention_dispatch.h"

namespace flashinfer::sparse_mla_sm120 {

#define DSV32_CUDA_CHECK(call)                                                             \
  do {                                                                                      \
    cudaError_t e = (call);                                                                 \
    TVM_FFI_ICHECK_EQ(e, cudaSuccess) << "sparse-MLA " #call ": " << cudaGetErrorString(e); \
  } while (0)

template <ModelType MT, int NUM_HEADS>
static bool launch_decode_dsv3_2_impl(int num_heads, int topk, const bf16* Q,
                                      const uint8_t* KV_cache, const int32_t* indices,
                                      bf16* mid_out, float* mid_lse, const int* topk_length,
                                      bf16* output, float* out_lse, const float* attn_sink,
                                      int num_tokens, int num_splits, int chunks_per_block_override,
                                      float sm_scale, size_t stride_kv_block,
                                      size_t stride_indices_token, int stride_kv_row,
                                      size_t stride_out_lse, cudaStream_t stream) {
  using KV = KVCacheTraits<MT>;
  static_assert(KV::D_QK == 576 || (MT == ModelType::GLM53_NOPE && KV::D_QK == 512));
  // NUM_HEADS == 0 is the runtime-head-count instantiation: num_heads (<= 128)
  // comes from the argument and the mid scratch is HPB-aligned.
  const int h_blocks = (NUM_HEADS == 0) ? (num_heads + HPB - 1) / HPB : (NUM_HEADS + HPB - 1) / HPB;
  const int q_heads = (NUM_HEADS == 0) ? num_heads : NUM_HEADS;

  // Dynamic smem layout (must match kernels/dsv32_fp8/decode.cuh exactly).
  //   sm_q_rope    HPB * D_ROPE * 2B               = 2 KB
  //   sm_q_fp8     HPB * Q_NOPE_STRIDE             = 8.25 KB
  //   sm_q_sc      HPB * NUM_SCALES * 4B           = 256 B
  //   sm_kv_fp8    2 * BI * KV_SMEM_STRIDE         = 66 KB (NoPE + INLINE scales)
  //   sm_kv_rope   2 * BI * D_ROPE * 2B            = 16 KB
  //   mbar pad+barriers                            = 48 B
  //   sm_reduce    2 * N_WARPS * HPB * 4           = 1 KB
  //   sm_w_head_sc N_V_CHUNKS * HPB * 4            = 256 B
  //   sm_w_fp8 ×2  2 * HPB * (BI + 16)             = 2.5 KB
  // Plus static sm_p_full HPB * BI * 2B            = 2 KB
  // Grand total ≈ 98 KB (under 99 KB sm120a carveout, 1 block/SM).
  constexpr int DYN_SMEM_BYTES = Dsv32DecodeSmem<MT>::LAUNCH_BYTES;

  auto kernel = sparse_mla_decode_dsv3_2_kernel<MT, NUM_HEADS, execution::FixedPageSize>;
  DSV32_CUDA_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  const int chunks_per_block = chunks_per_block_override;

  // Launch the full Python-allocated num_splits grid. Inactive splits return
  // early after marking LSE = -1e30f — keeps mid_out/mid_lse stride aligned
  // with wrapper allocation.
  dim3 grid1(num_tokens, h_blocks, num_splits);
  dim3 block1(DSV32_BLOCK_THREADS);
  kernel<<<grid1, block1, DYN_SMEM_BYTES, stream>>>(
      Q, KV_cache, indices, mid_out, mid_lse, topk_length, num_tokens, q_heads, topk, num_splits,
      chunks_per_block, sm_scale, stride_kv_block, stride_indices_token, stride_kv_row);
  DSV32_CUDA_CHECK(cudaGetLastError());

  // Stage 2: reuse decode-dsv4 merge kernel (D_V=512 identical for both).
  constexpr int MERGE_BLOCK_THREADS = 64;
  constexpr int MERGE_DIMS_PER_THREAD = KV::D_V / MERGE_BLOCK_THREADS;
  auto merge_kernel = sparse_mla_decode_dsv4_merge_kernel<NUM_HEADS, KV::D_V, MERGE_BLOCK_THREADS,
                                                          MERGE_DIMS_PER_THREAD>;
  dim3 grid2(num_tokens, q_heads);
  dim3 block2(MERGE_BLOCK_THREADS);
  const size_t merge_smem_bytes = (size_t)num_splits * sizeof(float);
  merge_kernel<<<grid2, block2, merge_smem_bytes, stream>>>(
      mid_out, mid_lse, output, out_lse, attn_sink, num_tokens, num_splits, q_heads, h_blocks * HPB,
      stride_out_lse);
  DSV32_CUDA_CHECK(cudaGetLastError());
  return true;
}

template <ModelType MT>
cudaError_t dispatch_dsv32_heads(const execution::AttentionParams& p,
                                 const execution::ExecutionPlan& plan, cudaStream_t stream) {
  return execution::visit_decode_heads<MT>(plan.specialized_heads, [&](auto head) {
    launch_decode_dsv3_2_impl<MT, decltype(head)::value>(
        p.num_heads, p.topk, p.q, p.kv, p.indices, p.mid_out, p.mid_lse, p.topk_length, p.output,
        p.out_lse, p.attn_sink, p.num_tokens, p.allocated_splits, plan.cpb, p.sm_scale,
        p.page_stride_bytes, p.indices_stride_elems, plan.metadata.row_stride_bytes,
        p.out_lse_stride_elems, stream);
    return cudaSuccess;
  });
}

cudaError_t dispatch_dsv32(const execution::AttentionParams& params,
                           const execution::ExecutionPlan& plan, cudaStream_t stream) {
  switch (static_cast<ModelType>(plan.metadata.model)) {
    case ModelType::DSV3_2:
      return dispatch_dsv32_heads<ModelType::DSV3_2>(params, plan, stream);
    case ModelType::GLM_NSA:
      return dispatch_dsv32_heads<ModelType::GLM_NSA>(params, plan, stream);
    case ModelType::GLM53_NOPE:
      return dispatch_dsv32_heads<ModelType::GLM53_NOPE>(params, plan, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace flashinfer::sparse_mla_sm120
