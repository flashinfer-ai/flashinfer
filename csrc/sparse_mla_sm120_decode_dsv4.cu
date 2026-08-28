// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>

#include <cstdio>
#include <flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

namespace flashinfer::sparse_mla_sm120 {

#define CUDA_CHECK_BOOL(call)                                               \
  do {                                                                      \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
      printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      return false;                                                         \
    }                                                                       \
  } while (0)

template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
static bool launch_decode_dsv4_impl(
    int num_heads, const bf16* Q, const uint8_t* KV_cache, const int32_t* indices, bf16* mid_out,
    float* mid_lse, const int* topk_length, bf16* output, float* out_lse, const float* attn_sink,
    const uint8_t* extra_KV_cache, const int32_t* extra_indices, const int* extra_topk_length,
    int extra_topk, int pbs_extra, size_t stride_extra_kv_block, int num_tokens, int num_splits,
    int chunks_per_block_override, float sm_scale, size_t stride_kv_block,
    size_t stride_indices_token, size_t stride_extra_indices_token, cudaStream_t stream) {
  using KV = KVCacheTraits<MT>;
  using Cfg = DecodeTileCfg<MT>;
  // Ceiling div so NUM_HEADS < HPB (small-TP configs, e.g. h=8) still get a
  // tile. NUM_HEADS == 0 is the runtime-head-count instantiation: num_heads
  // (<= 128) is taken from the argument, and the mid scratch is HPB-aligned
  // (h_blocks * HPB rows per token). The kernel internally clamps Q loads and
  // merge reads to the valid rows.
  const int h_blocks = (NUM_HEADS == 0) ? (num_heads + HPB - 1) / HPB : (NUM_HEADS + HPB - 1) / HPB;
  const int q_heads = (NUM_HEADS == 0) ? num_heads : NUM_HEADS;

  // Stage 1: decode-dsv4 (A1.2) partial-output kernel.
  // Dynamic smem layout (FP8 XV, double-buffered KV). Measured on sm_120
  // against a 101376 B per-block opt-in cap:
  //
  //   term                                        DSV4       DOTS3_SWA
  //                                            (BI=64,W=8)  (BI=32,W=4)
  //   sm_q_rope    HPB * D_ROPE * 2B               2048         2048
  //   sm_q_fp8     HPB * Q_NOPE_STRIDE             7424        16640
  //   sm_q_sc      HPB * NUM_SCALES * 4B            448          512
  //   sm_kv_fp8    2 * BI * KV_SMEM_STRIDE        59392        66560
  //   sm_kv_sc     2 * BI * SCALE_BYTES_PER_TOKEN  1024          512
  //   sm_kv_rope   2 * BI * D_ROPE * 2B           16384         8192
  //   mbar + pad                                     48           48
  //   sm_reduce    2 * N_WARPS * HPB * 4           1024          512
  //   sm_w_head_sc N_V_CHUNKS * HPB * 4             448          512
  //   sm_w_fp8 x2  2 * HPB * (BI + 16)             2560         1536
  //   dynamic total                               90800        97072
  // Static smem (kernel-side), sm_p_full = HPB * BI * 2B:
  //   DSV4 2048 B; DOTS3_SWA 0 (V_HAS_ROPE=false makes the bf16 P dead).
  //   grand total                                 92848        97072
  //
  // DOTS3_SWA leaves ~4.2 KB spare. BI=64 for it needs 173872 B and the driver
  // rejects the opt-in outright. Both configs run 1 block/SM.
  constexpr int N_V_CHUNKS_LAUNCH = KV::D_NOPE / KV::QUANT_TILE;  // DSV4 7, DOTS3_SWA 8
  constexpr int DYN_SMEM_BYTES =
      HPB * KV::D_ROPE * (int)sizeof(bf16)                            // sm_q_rope
      + HPB * KV::Q_NOPE_STRIDE                                       // sm_q_fp8
      + HPB * KV::NUM_SCALES * (int)sizeof(float)                     // sm_q_sc
      + Cfg::KV_BUF_COUNT * Cfg::BI * KV::KV_SMEM_STRIDE              // sm_kv_fp8 ×2
      + Cfg::KV_BUF_COUNT * Cfg::BI * KV::SCALE_BYTES_PER_TOKEN       // sm_kv_sc ×2
      + Cfg::KV_BUF_COUNT * Cfg::BI * KV::D_ROPE * (int)sizeof(bf16)  // sm_kv_rope ×2
      + 16                                                            // mbar align pad
      + 4 * (int)sizeof(uint64_t)                                     // mbar_full+empty
      + 2 * Cfg::N_WARPS * HPB * (int)sizeof(float)                   // sm_reduce
      + N_V_CHUNKS_LAUNCH * HPB * (int)sizeof(float)                  // sm_w_head_sc
      + 2 * HPB * (Cfg::BI + 16);                                     // sm_w_fp8 ×2 (vc parity)

  auto kernel = sparse_mla_decode_dsv4_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  CUDA_CHECK_BOOL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  // chunks_per_block heuristic: among cpb candidates with at most
  // CEIL_WAVES_MAX integer waves, minimize the last-wave tail gap
  // (ceil(waves) - waves). On ties, prefer the largest cpb so fewer
  // launched blocks contend on L2. The ceil_w cap rules out cpb values
  // whose fractional gap looks small but require many integer waves.
  // chunks_per_block_override: calibrated-model or explicit caller choice;
  // the heuristic is the fallback when no override is given.
  int chunks_per_block;
  if (chunks_per_block_override >= 1 && chunks_per_block_override <= num_splits) {
    chunks_per_block = chunks_per_block_override;
  } else {
    int sm_count = 0;
    int device = 0;
    CUDA_CHECK_BOOL(cudaGetDevice(&device));
    CUDA_CHECK_BOOL(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    if (sm_count <= 0) {
      printf("CUDA %s:%d invalid SM count %d\n", __FILE__, __LINE__, sm_count);
      return false;
    }
    constexpr int CEIL_WAVES_MAX = 3;
    const int per_token_head = num_tokens * h_blocks;
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
  // Launch the FULL Python-allocated num_splits grid blocks; inactive splits
  // (chunk_lo >= num_chunks_total) return early after marking LSE = -1e30f,
  // which is cheap. This keeps the mid_out/mid_lse stride matching Python's
  // allocation without extra coordination.
  dim3 grid1(num_tokens, h_blocks, num_splits);
  dim3 block1(Cfg::BLOCK_THREADS);
  kernel<<<grid1, block1, DYN_SMEM_BYTES, stream>>>(
      Q, KV_cache, indices, mid_out, mid_lse, topk_length, extra_KV_cache, extra_indices,
      extra_topk_length, extra_topk, pbs_extra, stride_extra_kv_block, num_tokens, q_heads,
      num_splits, chunks_per_block, sm_scale, stride_kv_block, stride_indices_token,
      stride_extra_indices_token);
  CUDA_CHECK_BOOL(cudaGetLastError());

  // Stage 2: merge splits → final output + LSE.
  // Grid: (num_tokens, NUM_HEADS). One block (BLOCK_THREADS=64) covers the
  // full D_V via uint4 vec loads: D_V=512 -> 8 bf16/thread, D_V=1024 (DOTS3_SWA)
  // -> 16, i.e. 2 uint4 per thread. Both keep DIMS_PER_THREAD % 8 == 0, which
  // the merge kernel static_asserts.
  // For h=128/T=16 this is 2048 blocks vs the prior 8192 (4× fewer).
  constexpr int MERGE_BLOCK_THREADS = 64;
  constexpr int MERGE_DIMS_PER_THREAD = KV::D_V / MERGE_BLOCK_THREADS;
  auto merge_kernel = sparse_mla_decode_dsv4_merge_kernel<NUM_HEADS, KV::D_V, MERGE_BLOCK_THREADS,
                                                          MERGE_DIMS_PER_THREAD>;
  dim3 grid2(num_tokens, q_heads);
  dim3 block2(MERGE_BLOCK_THREADS);
  const size_t merge_smem_bytes = (size_t)num_splits * sizeof(float);
  merge_kernel<<<grid2, block2, merge_smem_bytes, stream>>>(mid_out, mid_lse, output, out_lse,
                                                            attn_sink, num_tokens, num_splits,
                                                            q_heads, h_blocks * HPB);
  CUDA_CHECK_BOOL(cudaGetLastError());
  return true;
}

// Public surface — explicit instantiation switch.
// page_block_size=64 only. DSV4: TOPK ∈ {128, 192, 256, 512, 1024}.
// TOPK=192 covers the padded DeepSeek-V4-Flash-0731 DSpark K=5 shape (128 SWA
// + 5 active draft entries), while TOPK=256 covers wider DSpark
// configurations.
// Head counts: the production grid {8, 16, 32, 64, 128} keeps dedicated
// instantiations — measured 0.9-2.5% faster than the runtime-H kernel on hot
// shapes (compile-time head strides); every other num_heads in [1, 128] falls
// back to one runtime-H instantiation per topk (NUM_HEADS == 0), which pads
// the head tile with zero-Q rows and HPB-aligns the mid scratch.
bool launch_sparse_mla_decode_dsv4(
    ModelType mt, int num_heads, int topk, int page_block_size, int num_tokens, int num_splits,
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices, bf16* mid_out, float* mid_lse,
    bf16* output, float* out_lse, const int* topk_length, const float* attn_sink,
    const uint8_t* extra_KV_cache, const int32_t* extra_indices, const int* extra_topk_length,
    int extra_topk, int pbs_extra, size_t stride_extra_kv_block, int chunks_per_block_override,
    float sm_scale, size_t stride_kv_block, size_t stride_indices_token,
    size_t stride_extra_indices_token, cudaStream_t stream) {
  if (mt != ModelType::DSV4 && mt != ModelType::DOTS3_SWA) return false;
  if (page_block_size != 64) return false;
  if (num_splits <= 0) return false;
  if (num_heads < 1 || num_heads > 128) return false;
#define DECODE_DISPATCH(MT_, H, K)                                                          \
  if (mt == (MT_) && num_heads == (H) && topk == (K)) {                                     \
    return launch_decode_dsv4_impl<(MT_), (H), (K), 64>(                                    \
        num_heads, Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse,    \
        attn_sink, extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra, \
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \
        stride_kv_block, stride_indices_token, stride_extra_indices_token, stream);         \
  }
// Runtime-H fallback: any num_heads in [1, 128] whose exact count no
// dedicated instantiation above claimed (they return first).
#define DECODE_DISPATCH_RT(MT_, K)                                                          \
  if (mt == (MT_) && topk == (K)) {                                                         \
    return launch_decode_dsv4_impl<(MT_), 0, (K), 64>(                                      \
        num_heads, Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse,    \
        attn_sink, extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra, \
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \
        stride_kv_block, stride_indices_token, stride_extra_indices_token, stream);         \
  }
#define DSV4_DISPATCH(H, K) DECODE_DISPATCH(ModelType::DSV4, (H), (K))
#define DSV4_DISPATCH_RT(K) DECODE_DISPATCH_RT(ModelType::DSV4, (K))
#define DSV4_DISPATCH_ROW(H) \
  DSV4_DISPATCH(H, 128)      \
  DSV4_DISPATCH(H, 192)      \
  DSV4_DISPATCH(H, 256)      \
  DSV4_DISPATCH(H, 512)      \
  DSV4_DISPATCH(H, 1024)
  DSV4_DISPATCH_ROW(8)
  DSV4_DISPATCH_ROW(16)
  DSV4_DISPATCH_ROW(32)
  DSV4_DISPATCH_ROW(64)
  DSV4_DISPATCH_ROW(128)
#undef DSV4_DISPATCH_ROW
  DSV4_DISPATCH_RT(128)
  DSV4_DISPATCH_RT(192)
  DSV4_DISPATCH_RT(256)
  DSV4_DISPATCH_RT(512)
  DSV4_DISPATCH_RT(1024)
  // DOTS3_SWA sliding-window decode: 64 heads, a 513-token window carried as an
  // index set. TOPK=576 is the tightest fit: it covers 513, divides BI=32
  // (18 chunks) and the 64-wide split granularity (9 splits). The 1024 entry
  // the DSV4 grid happens to offer would nearly double the split-K scratch
  // (mid_out is [tokens, heads, num_splits, d_v]) for no extra coverage.
  // topk_length is optional for DOTS3_SWA: DecodeTileCfg<DOTS3_SWA>::WINDOW caps
  // the per-token candidate count inside the kernel, so omitting it costs
  // nothing beyond the window itself. Unused slots must still carry -1, which
  // the QK mask turns into -inf.
  // Head counts cover the TP shards of a 64-head layer (TP4 -> 16) with
  // dedicated instantiations; any other count up to 128 rides the runtime-H
  // fallback.
  DECODE_DISPATCH(ModelType::DOTS3_SWA, 8, 576)
  DECODE_DISPATCH(ModelType::DOTS3_SWA, 16, 576)
  DECODE_DISPATCH(ModelType::DOTS3_SWA, 32, 576)
  DECODE_DISPATCH(ModelType::DOTS3_SWA, 64, 576)
  DECODE_DISPATCH_RT(ModelType::DOTS3_SWA, 576)
#undef DSV4_DISPATCH_RT
#undef DSV4_DISPATCH
#undef DECODE_DISPATCH_RT
#undef DECODE_DISPATCH
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
