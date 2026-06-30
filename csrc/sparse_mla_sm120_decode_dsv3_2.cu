// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// V32 (DSv3.2) decode — forked from decode-dsv4. Warp-spec (1 IO + 8 math =
// 288 threads), TMA bulk gather of FP8 INLINE 656 B/token KV cache, double
// buffered with per-buffer mbarrier pairs, static grid (num_tokens × HBLOCKS
// × num_splits). Reuses decode-dsv4's merge kernel for split combine.
//
// Supports the full V32-family dispatch grid:
//   num_heads ∈ {8, 16, 32, 64, 128}
//   topk      ∈ {128, 512, 1024, 2048}
//   pbs       = 64
// = 20 instantiations.

#include <cuda_runtime.h>

#include <cstdio>
#include <flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh>  // merge kernel
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

namespace flashinfer::sparse_mla_sm120 {

#define DSV3_2_CUDA_CHECK(call)                                             \
  do {                                                                      \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
      printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      return false;                                                         \
    }                                                                       \
  } while (0)

template <ModelType MT, int NUM_HEADS, int TOPK>
static bool launch_decode_dsv3_2_impl(const bf16* Q, const uint8_t* KV_cache,
                                      const int32_t* indices, bf16* mid_out, float* mid_lse,
                                      const int* topk_length, bf16* output, float* out_lse,
                                      const float* attn_sink, int num_tokens, int num_splits,
                                      int chunks_per_block_override, float sm_scale,
                                      size_t stride_kv_block, cudaStream_t stream) {
  using KV = KVCacheTraits<MT>;
  static_assert(KV::D_QK == 576);
  constexpr int H_BLOCKS = (NUM_HEADS + HPB - 1) / HPB;

  // Dynamic smem layout (must match decode_dsv3_2_kernel.cuh exactly).
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
  constexpr int N_V_CHUNKS_LAUNCH = KV::D_NOPE / KV::QUANT_TILE;  // 4
  constexpr int DYN_SMEM_BYTES =
      HPB * KV::D_ROPE * (int)sizeof(bf16)                    // sm_q_rope
      + HPB * KV::Q_NOPE_STRIDE                               // sm_q_fp8
      + HPB * KV::NUM_SCALES * (int)sizeof(float)             // sm_q_sc
      + DSV3_2_KV_BUF_COUNT * DSV3_2_BI * KV::KV_SMEM_STRIDE  // sm_kv_fp8 ×2 (NoPE + INLINE scales)
      + DSV3_2_KV_BUF_COUNT * DSV3_2_BI * KV::D_ROPE * (int)sizeof(bf16)  // sm_kv_rope ×2
      + 16                                                                // mbar align pad
      + 4 * (int)sizeof(uint64_t)                                         // mbar_full + mbar_empty
      + 2 * DSV3_2_N_WARPS * HPB * (int)sizeof(float)                     // sm_reduce
      + N_V_CHUNKS_LAUNCH * HPB * (int)sizeof(float)                      // sm_w_head_sc
      + 2 * HPB * (DSV3_2_BI + 16);                                       // sm_w_fp8 ×2

  auto kernel = sparse_mla_decode_dsv3_2_kernel<MT, NUM_HEADS, TOPK, 64>;
  DSV3_2_CUDA_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  // chunks_per_block heuristic identical to decode-dsv4: among cpb candidates
  // with at most CEIL_WAVES_MAX integer waves, minimize the last-wave tail
  // gap; ties broken by largest cpb (fewer launched blocks contending on L2).
  int chunks_per_block;
  if (chunks_per_block_override >= 1 && chunks_per_block_override <= num_splits) {
    chunks_per_block = chunks_per_block_override;
  } else {
    int sm_count = 0;
    int device = 0;
    DSV3_2_CUDA_CHECK(cudaGetDevice(&device));
    DSV3_2_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    if (sm_count <= 0) sm_count = 188;
    constexpr int CEIL_WAVES_MAX = 3;
    const int per_token_head = num_tokens * H_BLOCKS;
    chunks_per_block = 1;
    float best_gap = 2.0f;
    for (int cpb = 1; cpb <= num_splits; ++cpb) {
      const int eff = (num_splits + cpb - 1) / cpb;
      const int active = per_token_head * eff;
      const int ceil_w = (active + sm_count - 1) / sm_count;
      if (ceil_w > CEIL_WAVES_MAX) continue;
      const float waves = (float)active / (float)sm_count;
      const float gap = (float)ceil_w - waves;
      if (gap < best_gap - 1e-6f || (gap < best_gap + 1e-6f && cpb > chunks_per_block)) {
        best_gap = gap;
        chunks_per_block = cpb;
      }
    }
  }

  // Launch the full Python-allocated num_splits grid. Inactive splits return
  // early after marking LSE = -inf — keeps mid_out/mid_lse stride aligned with
  // wrapper allocation.
  dim3 grid1(num_tokens, H_BLOCKS, num_splits);
  dim3 block1(DSV3_2_BLOCK_THREADS);
  kernel<<<grid1, block1, DYN_SMEM_BYTES, stream>>>(Q, KV_cache, indices, mid_out, mid_lse,
                                                    topk_length, num_tokens, num_splits,
                                                    chunks_per_block, sm_scale, stride_kv_block);
  DSV3_2_CUDA_CHECK(cudaGetLastError());

  // Stage 2: reuse decode-dsv4 merge kernel (D_V=512 identical for both).
  constexpr int MERGE_BLOCK_THREADS = 64;
  constexpr int MERGE_DIMS_PER_THREAD = KV::D_V / MERGE_BLOCK_THREADS;
  auto merge_kernel = sparse_mla_decode_dsv4_merge_kernel<NUM_HEADS, KV::D_V, MERGE_BLOCK_THREADS,
                                                          MERGE_DIMS_PER_THREAD>;
  dim3 grid2(num_tokens, NUM_HEADS);
  dim3 block2(MERGE_BLOCK_THREADS);
  const size_t merge_smem_bytes = (size_t)num_splits * sizeof(float);
  merge_kernel<<<grid2, block2, merge_smem_bytes, stream>>>(mid_out, mid_lse, output, out_lse,
                                                            attn_sink, num_tokens, num_splits);
  DSV3_2_CUDA_CHECK(cudaGetLastError());
  return true;
}

// Public surface: V32-family (DSv3.2 / GLM_NSA) decode.
// Returns false if (num_heads, topk) is outside the dispatch envelope.
bool launch_sparse_mla_decode_dsv3_2(ModelType mt, int num_heads, int topk, int num_tokens,
                                     int num_splits, const bf16* Q, const uint8_t* KV_cache,
                                     const int32_t* indices, bf16* mid_out, float* mid_lse,
                                     bf16* output, float* out_lse, const int* topk_length,
                                     const float* attn_sink, int chunks_per_block_override,
                                     float sm_scale, size_t stride_kv_block, cudaStream_t stream) {
  if (num_splits <= 0) return false;
#define DSV3_2_DISPATCH_MT(MT_VALUE, H, K)                                                     \
  if (num_heads == (H) && topk == (K)) {                                                       \
    return launch_decode_dsv3_2_impl<MT_VALUE, (H), (K)>(                                      \
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,       \
        num_tokens, num_splits, chunks_per_block_override, sm_scale, stride_kv_block, stream); \
  }
#define DSV3_2_DISPATCH(H, K)                      \
  do {                                             \
    if (mt == ModelType::DSV3_2) {                 \
      DSV3_2_DISPATCH_MT(ModelType::DSV3_2, H, K)  \
    } else if (mt == ModelType::GLM_NSA) {         \
      DSV3_2_DISPATCH_MT(ModelType::GLM_NSA, H, K) \
    }                                              \
  } while (0);
  DSV3_2_DISPATCH(8, 128)
  DSV3_2_DISPATCH(8, 512)
  DSV3_2_DISPATCH(8, 1024)
  DSV3_2_DISPATCH(8, 2048)
  DSV3_2_DISPATCH(16, 128)
  DSV3_2_DISPATCH(16, 512)
  DSV3_2_DISPATCH(16, 1024)
  DSV3_2_DISPATCH(16, 2048)
  DSV3_2_DISPATCH(32, 128)
  DSV3_2_DISPATCH(32, 512)
  DSV3_2_DISPATCH(32, 1024)
  DSV3_2_DISPATCH(32, 2048)
  DSV3_2_DISPATCH(64, 128)
  DSV3_2_DISPATCH(64, 512)
  DSV3_2_DISPATCH(64, 1024)
  DSV3_2_DISPATCH(64, 2048)
  DSV3_2_DISPATCH(128, 128)
  DSV3_2_DISPATCH(128, 512)
  DSV3_2_DISPATCH(128, 1024)
  DSV3_2_DISPATCH(128, 2048)
#undef DSV3_2_DISPATCH
#undef DSV3_2_DISPATCH_MT
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
