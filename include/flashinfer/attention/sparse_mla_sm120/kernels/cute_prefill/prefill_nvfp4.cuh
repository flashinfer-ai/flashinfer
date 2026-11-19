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

#pragma once

#include <cutlass/arch/barrier.h>

#include "../../common/lse.cuh"
#include "../../common/zero_row.cuh"
#include "../../compute/nvfp4_vt.cuh"
#include "../dsv4_nvfp4/resources.cuh"
#include "nvfp4_gather.cuh"
#include "nvfp4_mma.cuh"
#include "nvfp4_q_stage.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

// CTA tile: one query token, H=min(NUM_HEADS,64) heads, N=64 selected
// candidates, D=448 non-RoPE + 64 RoPE features. Main-cache tiles precede
// extra-cache tiles; the final tile of each cache may be partial.
//
// Role assignment (warp IDs below are local to each 4-warp WG):
//   WG0 + WG1: math. Let w=4*wg+warp, in [0,8).
//     QK: warp w owns candidates [8*w,8*w+8), all H heads and all D.
//         It traverses H in 16-head groups; Q is staged once in shared
//         memory and loaded into MMA fragments as needed, not kept in regs.
//     Softmax: these same warps reduce max/sum across their candidate slices
//         through shared scratch, update online statistics and rescale O.
//         They publish BF16 P[H,64], then quantize P in 16-candidate groups.
//     PV: ownership switches from candidates to output features. Warp w owns
//         non-RoPE groups g=w+8*s (g<28), each covering [16*g,16*g+16),
//         plus RoPE features [448+8*w,448+8*w+8). Every warp consumes all
//         64 candidates and retains its O[H,owned_features] in FP32 regs.
//   WG2 + WG3: IO. WG2 warps 0/1 gather raw KV[64,512] and scales.
//     All eight IO warps transpose and requantize the non-RoPE V operand
//     from feature-axis cache scales to candidate-axis PV scales.
//
// Handoffs: two raw-KV slots feed both QK and the BF16 PV tail; one Vt slot
// feeds non-RoPE PV. IO publishes raw KV after async gather completion and
// Vt after cooperative conversion. Math releases Vt after non-RoPE PV and
// raw KV only after the RoPE PV tail. QK/softmax/PV are sequential phases
// of the same math warps, not independent producer/consumer roles.
//
// Mega=false accepts rotated Q and writes BF16 context. Mega=true applies
// Q RoPE during staging, then inverse output RoPE and FP8 quantization in
// the epilogue. Neither specialization splits the selection across CTAs.

__device__ __forceinline__ float2 prefill_sink_normalization(float lse, float sink) {
  if (sink == INFINITY) return make_float2(0.f, INFINITY);
  if (sink == -INFINITY) return make_float2(1.f, lse);
  const float peak = fmaxf(lse, sink);
  const float mass = lse > -1e29f ? exp2f(lse - peak) : 0.f;
  const float total = mass + exp2f(sink - peak);
  return make_float2(total > 0.f ? mass / total : 0.f,
                     total > 0.f ? log2f(total) + peak : -INFINITY);
}

template <int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, bool DUAL_CACHE, bool Mega>
__global__ void __launch_bounds__(STREAMING_BLOCK_THREADS, 1) sparse_mla_prefill_dsv4_nvfp4_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ indices, bf16* __restrict__ output, float* __restrict__ out_lse,
    const float* __restrict__ attn_sink, const int* __restrict__ topk_length_ptr,
    const uint8_t* __restrict__ extra_kv_cache, const int32_t* __restrict__ extra_indices,
    const int* __restrict__ extra_topk_length_ptr, int extra_topk, int extra_page_block_size,
    size_t extra_page_stride_bytes, int num_tokens, float sm_scale, int page_block_size,
    size_t page_stride_bytes, float lse_scale, SparseMlaMegaParams mega = {}) {
  static_assert(NUM_HEADS <= STREAMING_HEADS_PER_CTA || NUM_HEADS % STREAMING_HEADS_PER_CTA == 0);
  static_assert(NUM_HEADS % HPB == 0,
                "streaming kernel processes full HPB head groups; thinner heads use the decode "
                "kernel");
  constexpr int VALID_HEAD_GROUPS =
      NUM_HEADS < STREAMING_HEADS_PER_CTA ? (NUM_HEADS + HPB - 1) / HPB : STREAMING_HEAD_GROUPS;
  constexpr int HEADS_PER_CTA = VALID_HEAD_GROUPS * HPB;
  constexpr int D_NOPE = DSV4NVFP4Cache::D_NOPE;
  constexpr int D_ROPE = DSV4NVFP4Cache::D_ROPE;
  constexpr int D_QK = DSV4NVFP4Cache::D_QK;
  constexpr int D_V = DSV4NVFP4Cache::D_V;
  constexpr int HEAD_BLOCKS = NUM_HEADS / HEADS_PER_CTA;
  constexpr int NUM_K64_TILES = D_NOPE / 64;
  constexpr int PV_SCALE_GROUPS = D_NOPE / SF_VEC_SIZE;
  constexpr int PV_GROUPS_PER_WARP = (PV_SCALE_GROUPS + STREAMING_N_WARPS - 1) / STREAMING_N_WARPS;
  constexpr int PV_N8_TILES_PER_GROUP = SF_VEC_SIZE / 8;
  constexpr int ROPE_DIMS_PER_WARP = D_ROPE / STREAMING_N_WARPS;
  constexpr int ROPE_N_TILES = ROPE_DIMS_PER_WARP / 8;
  constexpr int ROPE_K_ITERS = STREAMING_CAND_WINDOW / 16;
  using namespace cute;
  using Head = Int<HPB>;
  using HeadGroup = Int<VALID_HEAD_GROUPS>;
  using Candidate = Int<STREAMING_CAND_WINDOW>;
  using MathWarp = Int<STREAMING_N_WARPS>;
  using PStride = Int<STREAMING_P_STRIDE>;
  using HeadGroupStride = Int<HPB * STREAMING_P_STRIDE>;
  using Feature = Int<D_V>;
  const int token_idx = blockIdx.x;
  const int head_block = blockIdx.y;
  if (token_idx >= num_tokens || head_block >= HEAD_BLOCKS) return;
  const int h_start = head_block * HEADS_PER_CTA;
  int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + token_idx) : TOPK;
  topk_len = max(0, min(topk_len, TOPK));
  const int num_main_chunks = (topk_len + STREAMING_CAND_WINDOW - 1) / STREAMING_CAND_WINDOW;
  int extra_topk_len = 0;
  if constexpr (DUAL_CACHE) {
    extra_topk_len = extra_topk_length_ptr ? __ldg(extra_topk_length_ptr + token_idx) : extra_topk;
    extra_topk_len = max(0, min(extra_topk_len, extra_topk));
  }
  const int num_extra_chunks = (extra_topk_len + STREAMING_CAND_WINDOW - 1) / STREAMING_CAND_WINDOW;
  const int num_chunks = num_main_chunks + num_extra_chunks;
  constexpr int chunk_lo = 0;
  const int chunk_hi = num_chunks;
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const bool is_io = warp_id >= STREAMING_N_WARPS;

  extern __shared__ __align__(16) char smem_raw[];
  auto sm = StreamingNVFP4Smem::init(smem_raw);
  auto gIndices = make_tensor(
      make_gmem_ptr(indices),
      make_layout(make_shape(Int<TOPK>{}, num_tokens), make_stride(_1{}, size_t(TOPK))));
  auto gIndexRow = gIndices(_, token_idx);
  auto gLse = make_tensor(
      make_gmem_ptr(out_lse),
      make_layout(make_shape(Int<NUM_HEADS>{}, num_tokens), make_stride(_1{}, size_t(NUM_HEADS))));
  auto gO = make_tensor(make_gmem_ptr(output),
                        make_layout(make_shape(Int<D_V>{}, Int<NUM_HEADS>{}, num_tokens),
                                    make_stride(_1{}, Int<D_V>{}, size_t(NUM_HEADS) * D_V)));
  auto gQ = make_tensor(make_gmem_ptr(q),
                        make_layout(make_shape(Int<D_QK>{}, Int<NUM_HEADS>{}, num_tokens),
                                    make_stride(_1{}, Int<D_QK>{}, size_t(NUM_HEADS) * D_QK)));
  auto extra_index_row = [&] {
    auto gExtra = make_tensor(
        make_gmem_ptr(extra_indices),
        make_layout(make_shape(extra_topk, num_tokens), make_stride(_1{}, size_t(extra_topk))));
    return gExtra(_, token_idx);
  };

  // Phase 0 — Initialize handoffs and handle an empty selection.
  // Raw KV: two slots. One transactional arrival publishes each gathered tile;
  // every math thread releases it after its last read. V-transpose slots use
  // all IO threads as producers and all math threads as consumers.
  using Barrier = cutlass::arch::ClusterBarrier;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int slot = 0; slot < STREAMING_KV_BUF_COUNT; ++slot) {
      Barrier::init(sm.mbar_full(slot), 1);
      Barrier::init(sm.mbar_empty(slot), STREAMING_MATH_THREADS);
    }
#pragma unroll
    for (int slot = 0; slot < STREAMING_VT_PIPE_STAGES; ++slot) {
      Barrier::init(sm.mbar_vt_full(slot), STREAMING_IO_WARPS * 32);
      Barrier::init(sm.mbar_vt_empty(slot), STREAMING_MATH_THREADS);
    }
  }
  __syncthreads();

  if (chunk_lo >= num_chunks) {
    if (!is_io) {
      for (int i = threadIdx.x; i < HEADS_PER_CTA * D_V; i += STREAMING_MATH_THREADS) {
        const int head = i / D_V;
        const int dim = i - head * D_V;
        if constexpr (Mega) {
          const size_t offset = gO.layout()(dim, h_start + head, token_idx);
          mega.output[offset] = 0;
          if ((dim & 31) == 0) mega.scales[offset / 32] = 105;
        } else {
          gO(dim, h_start + head, token_idx) = __float2bfloat16(0.f);
        }
      }
      if (threadIdx.x < HEADS_PER_CTA) {
        const int h = h_start + threadIdx.x;
        gLse(h, token_idx) =
            scale_output_lse(attn_sink ? __ldg(attn_sink + h) * LOG2E : -INFINITY, lse_scale);
      }
    }
    return;
  }

  auto issue_gather = [&](int chunk, int buf) {
    cute_gather_tile<PAGE_BLOCK_SIZE, DUAL_CACHE, STREAMING_KV_SMEM_STRIDE,
                     STREAMING_GATHER_WARPS * 32>(
        kv_cache, raw_pointer_cast(gIndexRow.data()), chunk, topk_len, false, page_block_size,
        page_stride_bytes, (warp_id - STREAMING_N_WARPS) * 32 + lane, sm.kv_fp4(buf),
        sm.kv_rope(buf), sm.kv_sc(buf), sm.mbar_full(buf), num_main_chunks, extra_topk_len,
        extra_kv_cache, DUAL_CACHE ? raw_pointer_cast(extra_index_row().data()) : nullptr,
        extra_page_block_size, extra_page_stride_bytes);
  };

  // Phase 1 — IO role: WG2/WG3. Only WG2 warps 0/1 issue the gather;
  // all eight IO warps prepare Vt. Each slot is acquired before overwrite.
  // Raw KV may advance into the second slot while math consumes the first.
  if (is_io) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(STREAMING_IO_MAX_REGS));
    // Double-buffer raw KV so gathering the next tile can overlap PV.
    // The consumer releases each transposed V slot after PV completes.
    uint32_t raw_free_phase = 1;
    uint32_t raw_ready_phase = 0;
    uint32_t vt_free_phase[STREAMING_VT_PIPE_STAGES] = {};
#pragma unroll
    for (int stage = 0; stage < STREAMING_VT_PIPE_STAGES; ++stage) vt_free_phase[stage] = 1;
    int raw_slot = 0;
    for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
      const int buf = (chunk - chunk_lo) % STREAMING_KV_BUF_COUNT;
      Barrier::wait(sm.mbar_empty(raw_slot), raw_free_phase);
      if (threadIdx.x < STREAMING_MATH_THREADS + STREAMING_GATHER_WARPS * 32)
        issue_gather(chunk, buf);
      Barrier::wait(sm.mbar_full(raw_slot), raw_ready_phase);
      const int vt_slot = chunk % STREAMING_VT_PIPE_STAGES;
      Barrier::wait(sm.mbar_vt_empty(vt_slot), vt_free_phase[vt_slot]);
      prepare_nvfp4_vt_from_smem<STREAMING_IO_WARPS * 32, STREAMING_KV_SMEM_STRIDE,
                                 STREAMING_MATH_THREADS>(sm.kv_fp4(buf), sm.kv_sc(buf),
                                                         sm.vt_data(vt_slot), sm.vt_sc(vt_slot));
      cutlass::arch::NamedBarrier::sync(
          STREAMING_IO_WARPS * 32,
          static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::VT_PRODUCER));
      Barrier::arrive(sm.mbar_vt_full(vt_slot));
      vt_free_phase[vt_slot] ^= 1;
      if (++raw_slot == STREAMING_KV_BUF_COUNT) {
        raw_slot = 0;
        raw_free_phase ^= 1;
        raw_ready_phase ^= 1;
      }
    }
    return;
  }

  // Phase 2 — Math prologue: WG0/WG1 stage Q once, including optional RoPE.
  // Reserve registers for FP32 O and online max/sum across every KV tile.
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(STREAMING_MATH_MAX_REGS));

  const int gid = lane >> 2;
  const int tid = lane & 3;
#pragma unroll
  for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
    auto gQHead = gQ(_, h_start + group * HPB, token_idx);
    const bf16* q_base = raw_pointer_cast(gQHead.data());
    cute_quantize_prefill_q_nvfp4_to_smem<STREAMING_MATH_THREADS, STREAMING_Q_FP4_STRIDE,
                                          STREAMING_Q_SCALE_STRIDE>(
        sm.q_fp4(group), sm.q_sc(group), sm.q_rope(group), q_base, HPB,
        (Mega ? mega.cos_sin + size_t(mega.positions[token_idx]) * 64 : nullptr));
  }

  float acc_nope[STREAMING_HEAD_GROUPS][PV_GROUPS_PER_WARP][PV_N8_TILES_PER_GROUP][4] = {0};
  float acc_rope[STREAMING_HEAD_GROUPS][ROPE_N_TILES][4] = {0};
  // Each lane owns two heads (gid and gid+8) in every head group.
  auto lane_head_shape = Shape<_2, HeadGroup>{};
  auto rMax = make_tensor<float>(lane_head_shape);
  auto rSum = make_tensor<float>(lane_head_shape);
  // MMA C stores two adjacent feature values for each of those heads.
  auto rONope = make_tensor(
      &acc_nope[0][0][0][0],
      make_layout(Shape<_2, _2, Int<PV_N8_TILES_PER_GROUP>, Int<PV_GROUPS_PER_WARP>, HeadGroup>{}));
  auto rORope =
      make_tensor(&acc_rope[0][0][0], make_layout(Shape<_2, _2, Int<ROPE_N_TILES>, HeadGroup>{}));
  auto sReduce = make_tensor(make_smem_ptr(sm.reduce_scratch()),
                             make_layout(Shape<Head, MathWarp, HeadGroup>{}));
  auto sP = make_tensor(
      make_smem_ptr(sm.p_full(0)),
      make_layout(Shape<Head, Candidate, HeadGroup>{}, Stride<PStride, _1, HeadGroupStride>{}));
#pragma unroll
  for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
    rMax(0, group) = rMax(1, group) = -1e30f;
    rSum(0, group) = rSum(1, group) = 0.f;
  }

  const auto copy_rope_a = cute::make_tiled_copy_A(
      cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, uint16_t>{}, cute::make_tiled_mma(RopeMma{}));
  uint32_t raw_ready_phase = 0;
  uint32_t vt_ready_phase[STREAMING_VT_PIPE_STAGES] = {};
  int raw_consumer_slot = 0;
  for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
    const int buf = (chunk - chunk_lo) % STREAMING_KV_BUF_COUNT;
    int section_chunk = chunk;
    int section_len = topk_len;
    const int32_t* section_indices = raw_pointer_cast(gIndexRow.data());
    if constexpr (DUAL_CACHE) {
      if (chunk >= num_main_chunks) {
        section_chunk = chunk - num_main_chunks;
        section_len = extra_topk_len;
        section_indices = raw_pointer_cast(extra_index_row().data());
      }
    }
    const int chunk_start = section_chunk * STREAMING_CAND_WINDOW;
    const int chunk_end = min(chunk_start + STREAMING_CAND_WINDOW, section_len);
    Barrier::wait(sm.mbar_full(raw_consumer_slot), raw_ready_phase);
    uint8_t* sm_kv_fp4 = sm.kv_fp4(buf);
    uint8_t* sm_kv_sc = sm.kv_sc(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);
    const int warp_first_cand = warp_id * STREAMING_ENTRIES_PER_WARP;

    // Phase 3 — QK: each math warp computes S[H,8] for its candidates,
    // accumulating the seven FP4 K=64 steps and four BF16 RoPE K=16 steps.
    float qk[STREAMING_HEAD_GROUPS][STREAMING_QK_N_TILES][4] = {0};
    // Lane-local score coordinates: (candidate in pair, head in pair, head group).
    static_assert(STREAMING_QK_N_TILES == 1);
    auto lane_score_shape = Shape<_2, _2, HeadGroup>{};
    auto rScore = make_tensor(&qk[0][0][0], make_layout(lane_score_shape));
    cute_qk_nvfp4<VALID_HEAD_GROUPS, STREAMING_Q_FP4_STRIDE, STREAMING_Q_SCALE_STRIDE,
                  STREAMING_KV_SMEM_STRIDE, STREAMING_SCALE_BYTES_PER_TOKEN, NUM_K64_TILES>(
        qk, sm.q_fp4(0), sm.q_sc(0), sm_kv_fp4, sm_kv_sc, warp_first_cand, lane);

#pragma unroll
    for (int ks = 0; ks < D_ROPE / 16; ++ks) {
      auto sK = make_tensor(make_smem_ptr(sm_kv_rope), make_layout(Shape<Candidate, Int<D_ROPE>>{},
                                                                   Stride<Int<D_ROPE>, _1>{}));
      const int candidate = warp_first_cand + gid;
      const int feature = ks * 16 + tid * 2;
      auto rB = cute::make_tensor<uint32_t>(cute::make_layout(cute::_2{}));
      rB(0) = *reinterpret_cast<const uint32_t*>(&sK(candidate, feature));
      rB(1) = *reinterpret_cast<const uint32_t*>(&sK(candidate, feature + 8));
#pragma unroll
      for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
        auto rA = cute::make_tensor<uint32_t>(cute::make_layout(cute::_4{}));
        // Partition the swizzled [head,RoPE feature] tile directly.
        auto sQ =
            cute::make_tensor(cute::make_smem_ptr(sm.q_rope(group)), CutePrefillQRopeLayout{});
        auto sQTile = cute::local_tile(sQ, cute::Shape<cute::_16, cute::_16>{},
                                       cute::make_coord(cute::_0{}, ks));
        copy_mma_operand<uint16_t, 8>(copy_rope_a, cute::recast<uint16_t>(sQTile), rA, lane);
        auto rC = group_modes<0, 2>(rScore(_, _, group));
        cute::MMA_Atom<RopeMma>{}.call(rA, rB, rC);
      }
    }

    // Phase 4 — Mask and online softmax. Tail/negative indices contribute
    // zero probability. Reduce across all eight candidate-owning warps
    // before rescaling the persistent O and publishing this tile's P.
    auto rWarpMax = make_tensor<float>(lane_head_shape);
    auto rWarpSum = make_tensor<float>(lane_head_shape);
    auto rP = make_tensor<float>(lane_score_shape);
    auto rTileMax = make_tensor<float>(lane_head_shape);
    const int c0 = warp_first_cand + tid * 2;
    const int c1 = c0 + 1;
    const int abs_c0 = chunk_start + c0;
    const int abs_c1 = chunk_start + c1;
    const int idx0 = abs_c0 < section_len ? section_indices[abs_c0] : -1;
    const int idx1 = abs_c1 < section_len ? section_indices[abs_c1] : -1;
    const bool valid_c0 = abs_c0 < chunk_end && idx0 >= 0;
    const bool valid_c1 = abs_c1 < chunk_end && idx1 >= 0;
    const float qk_scale = sm_scale * LOG2E;
#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      rScore(0, 0, group) = valid_c0 ? rScore(0, 0, group) * qk_scale : -1e30f;
      rScore(0, 1, group) = valid_c0 ? rScore(0, 1, group) * qk_scale : -1e30f;
      rScore(1, 0, group) = valid_c1 ? rScore(1, 0, group) * qk_scale : -1e30f;
      rScore(1, 1, group) = valid_c1 ? rScore(1, 1, group) * qk_scale : -1e30f;

      rWarpMax(0, group) = fmaxf(rScore(0, 0, group), rScore(1, 0, group));
      rWarpMax(1, group) = fmaxf(rScore(0, 1, group), rScore(1, 1, group));
#pragma unroll
      for (int s = 2; s >= 1; s >>= 1) {
        rWarpMax(0, group) =
            fmaxf(rWarpMax(0, group), __shfl_xor_sync(0xffffffff, rWarpMax(0, group), s));
        rWarpMax(1, group) =
            fmaxf(rWarpMax(1, group), __shfl_xor_sync(0xffffffff, rWarpMax(1, group), s));
      }
      rP(0, 0, group) = valid_c0 ? exp2f(rScore(0, 0, group) - rWarpMax(0, group)) : 0.f;
      rP(1, 0, group) = valid_c1 ? exp2f(rScore(1, 0, group) - rWarpMax(0, group)) : 0.f;
      rP(0, 1, group) = valid_c0 ? exp2f(rScore(0, 1, group) - rWarpMax(1, group)) : 0.f;
      rP(1, 1, group) = valid_c1 ? exp2f(rScore(1, 1, group) - rWarpMax(1, group)) : 0.f;
      rWarpSum(0, group) = rP(0, 0, group) + rP(1, 0, group);
      rWarpSum(1, group) = rP(0, 1, group) + rP(1, 1, group);
#pragma unroll
      for (int s = 2; s >= 1; s >>= 1) {
        rWarpSum(0, group) += __shfl_xor_sync(0xffffffff, rWarpSum(0, group), s);
        rWarpSum(1, group) += __shfl_xor_sync(0xffffffff, rWarpSum(1, group), s);
      }
      if (tid == 0) {
        sReduce(gid, warp_id, group) = rWarpMax(0, group);
        sReduce(gid + 8, warp_id, group) = rWarpMax(1, group);
      }
    }
    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));

    if (threadIdx.x < HEADS_PER_CTA) {
      const int group = threadIdx.x / HPB;
      const int head = threadIdx.x % HPB;
      float block_max = -1e30f;
#pragma unroll
      for (int w = 0; w < STREAMING_N_WARPS; ++w)
        block_max = fmaxf(block_max, sReduce(head, w, group));
      sReduce(head, 0, group) = block_max;
    }
    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      rTileMax(0, group) = sReduce(gid, 0, group);
      rTileMax(1, group) = sReduce(gid + 8, 0, group);
    }
    // All warps must retain block max in registers before warp 0's slots are
    // recycled for the rescaled local sums.
    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      if (tid == 0) {
        sReduce(gid, warp_id, group) =
            rWarpSum(0, group) * exp2f(rWarpMax(0, group) - rTileMax(0, group));
        sReduce(gid + 8, warp_id, group) =
            rWarpSum(1, group) * exp2f(rWarpMax(1, group) - rTileMax(1, group));
      }
    }
    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));

    if (threadIdx.x < HEADS_PER_CTA) {
      const int group = threadIdx.x / HPB;
      const int head = threadIdx.x % HPB;
      float block_sum = 0.f;
#pragma unroll
      for (int w = 0; w < STREAMING_N_WARPS; ++w) block_sum += sReduce(head, w, group);
      sReduce(head, 0, group) = block_sum;
    }
    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      const float block_max0 = rTileMax(0, group);
      const float block_max1 = rTileMax(1, group);
      const float block_sum0 = sReduce(gid, 0, group);
      const float block_sum1 = sReduce(gid + 8, 0, group);
      const float new_max0 = fmaxf(rMax(0, group), block_max0);
      const float new_max1 = fmaxf(rMax(1, group), block_max1);
      const float alpha0 = rMax(0, group) > -1e29f ? exp2f(rMax(0, group) - new_max0) : 0.f;
      const float alpha1 = rMax(1, group) > -1e29f ? exp2f(rMax(1, group) - new_max1) : 0.f;
      const float block_rescale0 = exp2f(block_max0 - new_max0);
      const float block_rescale1 = exp2f(block_max1 - new_max1);
      const float warp_rescale0 = exp2f(rWarpMax(0, group) - new_max0);
      const float warp_rescale1 = exp2f(rWarpMax(1, group) - new_max1);

      if (chunk > chunk_lo) {
#pragma unroll
        for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
#pragma unroll
          for (int nt = 0; nt < PV_N8_TILES_PER_GROUP; ++nt) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              rONope(i, 0, nt, slot, group) *= alpha0;
              rONope(i, 1, nt, slot, group) *= alpha1;
            }
          }
        }
#pragma unroll
        for (int nt = 0; nt < ROPE_N_TILES; ++nt) {
          rORope(0, 0, nt, group) *= alpha0;
          rORope(1, 0, nt, group) *= alpha0;
          rORope(0, 1, nt, group) *= alpha1;
          rORope(1, 1, nt, group) *= alpha1;
        }
        rSum(0, group) = rSum(0, group) * alpha0 + block_sum0 * block_rescale0;
        rSum(1, group) = rSum(1, group) * alpha1 + block_sum1 * block_rescale1;
      } else {
        rSum(0, group) = block_sum0 * block_rescale0;
        rSum(1, group) = block_sum1 * block_rescale1;
      }
      rMax(0, group) = new_max0;
      rMax(1, group) = new_max1;

      const float w0 = rP(0, 0, group) * warp_rescale0;
      const float w1 = rP(1, 0, group) * warp_rescale0;
      const float w2 = rP(0, 1, group) * warp_rescale1;
      const float w3 = rP(1, 1, group) * warp_rescale1;
      sP(gid, c0, group) = __float2bfloat16(w0);
      sP(gid, c1, group) = __float2bfloat16(w1);
      sP(gid + 8, c0, group) = __float2bfloat16(w2);
      sP(gid + 8, c1, group) = __float2bfloat16(w3);
    }

    // Math warps form P while IO warps transpose and requantize the
    // 64-candidate V tile in shared memory.
    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));

    // Phase 5 — PV: quantize P along candidates, then reuse each head group
    // across the feature-split warps. Keep BF16 P for the RoPE-value tail.
    using CandidateGroup = Int<NVFP4_VT_SCALE_GROUPS>;
    auto sPQuant =
        make_tensor(make_smem_ptr(sm.p_fp4(0)),
                    make_layout(Shape<Head, CandidateGroup, HeadGroup>{},
                                Stride<Int<STREAMING_W_PACKED_STRIDE>, Int<FP4_PACKED_PER_GROUP>,
                                       Int<HPB * STREAMING_W_PACKED_STRIDE>>{}));
    auto sPScale = make_tensor(make_smem_ptr(sm.w_sc(0)),
                               make_layout(Shape<CandidateGroup, Head, HeadGroup>{}));
    auto quant_tasks = make_layout(Shape<CandidateGroup, Head, HeadGroup>{});
    for (int task = threadIdx.x; task < size(quant_tasks); task += STREAMING_MATH_THREADS) {
      auto [candidate_group, head, group] = quant_tasks.get_flat_coord(task);
      cute_quantize_prefill_group16(&sP(head, candidate_group * SF_VEC_SIZE, group),
                                    &sPQuant(head, candidate_group, group),
                                    &sPScale(candidate_group, head, group));
    }
    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));
    const int vt_slot = chunk % STREAMING_VT_PIPE_STAGES;
    Barrier::wait(sm.mbar_vt_full(vt_slot), vt_ready_phase[vt_slot]);

    cute_pv_nvfp4<VALID_HEAD_GROUPS, STREAMING_W_PACKED_STRIDE, STREAMING_N_WARPS>(
        acc_nope, sm.p_fp4(0), sm.w_sc(0), sm.vt_data(vt_slot), sm.vt_sc(vt_slot), warp_id, lane);

    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));
    Barrier::arrive(sm.mbar_vt_empty(vt_slot));
    vt_ready_phase[vt_slot] ^= 1;

    {
      using namespace cute;
      using Bf16 = cute::bfloat16_t;
      const auto mma = make_tiled_mma(RopeMma{});
      const auto thr_mma = mma.get_slice(lane);
      const auto copy_p = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Bf16>{}, mma);
      const auto copy_v = make_tiled_copy_B(Copy_Atom<UniversalCopy<Bf16>, Bf16>{}, mma);
      const auto thr_copy_p = copy_p.get_slice(lane);
      const auto thr_copy_v = copy_v.get_slice(lane);

      // A=(head, candidate, head group), B=(feature, candidate),
      // C=(head, feature). V stays candidate-major in the raw KV slot.
      auto sV = recast<Bf16>(make_tensor(
          make_smem_ptr(sm_kv_rope), make_layout(Shape<Int<D_ROPE>, Int<STREAMING_CAND_WINDOW>>{},
                                                 Stride<_1, Int<D_ROPE>>{})));
      auto sVWarp = local_tile(sV, Shape<Int<ROPE_DIMS_PER_WARP>, Int<STREAMING_CAND_WINDOW>>{},
                               make_coord(warp_id, _0{}));
      auto tPsP = thr_copy_p.partition_S(recast<Bf16>(sP));
      auto tVsV = thr_copy_v.partition_S(sVWarp);

      // Allocate one K=16 fragment per head group so the same V fragment
      // can feed all groups before advancing to the next candidate tile.
      auto p_shape = shape(thr_mma.partition_A(make_identity_tensor(Shape<_16, _16>{})));
      auto tPrP = make_tensor<Bf16>(append(p_shape, Int<VALID_HEAD_GROUPS>{}));
      auto tVrV =
          thr_mma.partition_fragment_B(make_identity_tensor(Shape<Int<ROPE_DIMS_PER_WARP>, _16>{}));
      auto tPrP_copy = thr_copy_p.retile_D(tPrP);
      auto tVrV_copy = thr_copy_v.retile_D(tVrV);
      auto c_shape =
          shape(thr_mma.partition_C(make_identity_tensor(Shape<_16, Int<ROPE_DIMS_PER_WARP>>{})));
      auto tOrO =
          make_tensor(rORope.data(), make_layout(append(c_shape, Int<VALID_HEAD_GROUPS>{})));

#pragma unroll
      for (int k = 0; k < ROPE_K_ITERS; ++k) {
        copy(copy_p, tPsP(_, _, k, _), tPrP_copy(_, _, _0{}, _));
        copy(copy_v, tVsV(_, _, k), tVrV_copy(_, _, _0{}));
#pragma unroll
        for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
          gemm(mma, tPrP(_, _, _, group), tVrV, tOrO(_, _, _, group));
        }
      }
    }

    cutlass::arch::NamedBarrier::sync(
        STREAMING_MATH_THREADS,
        static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));
    Barrier::arrive(sm.mbar_empty(raw_consumer_slot));
    if (++raw_consumer_slot == STREAMING_KV_BUF_COUNT) {
      raw_consumer_slot = 0;
      raw_ready_phase ^= 1;
    }
  }

  // Phase 6 — Epilogue. WG0 warp 0 writes LSE and sink normalization;
  // all math warps normalize their O fragments. Mega reuses dead raw-KV
  // storage for one BF16 [16,512] context group before inverse RoPE/casting.
  auto sOutputScale = make_tensor(make_smem_ptr(reinterpret_cast<float*>(sm.p_full(0))),
                                  make_layout(Shape<Head, HeadGroup>{}));
  if (warp_id == 0 && tid == 0) {
#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      constexpr float empty_lse = -INFINITY;
      float lse0 = rSum(0, group) > 0.f ? log2f(rSum(0, group)) + rMax(0, group) : empty_lse;
      float lse1 = rSum(1, group) > 0.f ? log2f(rSum(1, group)) + rMax(1, group) : empty_lse;
      float output_scale0 = 1.f;
      float output_scale1 = 1.f;
      const int h0 = h_start + group * HPB + gid;
      if (attn_sink != nullptr) {
        const auto norm0 = prefill_sink_normalization(lse0, __ldg(attn_sink + h0) * LOG2E);
        const auto norm1 = prefill_sink_normalization(lse1, __ldg(attn_sink + h0 + 8) * LOG2E);
        output_scale0 = norm0.x;
        output_scale1 = norm1.x;
        lse0 = norm0.y;
        lse1 = norm1.y;
      }
      sOutputScale(gid, group) = output_scale0;
      sOutputScale(gid + 8, group) = output_scale1;
      gLse(h0, token_idx) = scale_output_lse(lse0, lse_scale);
      gLse(h0 + 8, token_idx) = scale_output_lse(lse1, lse_scale);
    }
  }
  cutlass::arch::NamedBarrier::sync(
      STREAMING_MATH_THREADS,
      static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::MATH));

#pragma unroll
  for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
    const float inv_sum0 = rSum(0, group) > 0.f ? sOutputScale(gid, group) / rSum(0, group) : 0.f;
    const float inv_sum1 =
        rSum(1, group) > 0.f ? sOutputScale(gid + 8, group) / rSum(1, group) : 0.f;
    const int group_h_start = h_start + group * HPB;
    auto gHead = gO(_, group_h_start, token_idx);
    bf16* destination = raw_pointer_cast(gHead.data());
    // Producers have finished the final VT tile; raw KV is no longer live.
    static_assert(2 * StreamingNVFP4Smem::SMEM_KV_FP4_BUF >= HPB * D_V * sizeof(bf16));
    if constexpr (Mega) destination = reinterpret_cast<bf16*>(sm.kv_fp4(0));
    auto mO = make_tensor(destination, make_layout(Shape<Head, Feature>{}, Stride<Feature, _1>{}));
#pragma unroll
    for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
      const int scale_group = slot * STREAMING_N_WARPS + warp_id;
      if (scale_group >= PV_SCALE_GROUPS) continue;
#pragma unroll
      for (int nt = 0; nt < PV_N8_TILES_PER_GROUP; ++nt) {
        const int d0 = scale_group * SF_VEC_SIZE + nt * 8 + tid * 2;
        const __nv_bfloat162 lo = __floats2bfloat162_rn(rONope(0, 0, nt, slot, group) * inv_sum0,
                                                        rONope(1, 0, nt, slot, group) * inv_sum0);
        const __nv_bfloat162 hi = __floats2bfloat162_rn(rONope(0, 1, nt, slot, group) * inv_sum1,
                                                        rONope(1, 1, nt, slot, group) * inv_sum1);
        *reinterpret_cast<__nv_bfloat162*>(&mO(gid, d0)) = lo;
        *reinterpret_cast<__nv_bfloat162*>(&mO(gid + 8, d0)) = hi;
      }
    }
#pragma unroll
    for (int nt = 0; nt < ROPE_N_TILES; ++nt) {
      const int d0 = D_NOPE + warp_id * ROPE_DIMS_PER_WARP + nt * 8 + tid * 2;
      const __nv_bfloat162 lo = __floats2bfloat162_rn(rORope(0, 0, nt, group) * inv_sum0,
                                                      rORope(1, 0, nt, group) * inv_sum0);
      const __nv_bfloat162 hi = __floats2bfloat162_rn(rORope(0, 1, nt, group) * inv_sum1,
                                                      rORope(1, 1, nt, group) * inv_sum1);
      *reinterpret_cast<__nv_bfloat162*>(&mO(gid, d0)) = lo;
      *reinterpret_cast<__nv_bfloat162*>(&mO(gid + 8, d0)) = hi;
    }
    if constexpr (Mega) {
      cutlass::arch::NamedBarrier::sync(
          STREAMING_MATH_THREADS,
          static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::Q_STAGE));
      for (int i = threadIdx.x; i < HPB * D_V / 8; i += STREAMING_MATH_THREADS) {
        const int head = i / (D_V / 8);
        const int feature = (i % (D_V / 8)) * 8;
        mega_store8(*reinterpret_cast<const uint4*>(&mO(head, feature)),
                    gO.layout()(feature, group_h_start + head, token_idx), token_idx, mega);
      }
      // Release epilogue scratch before the next head group reuses raw KV.
      cutlass::arch::NamedBarrier::sync(
          STREAMING_MATH_THREADS,
          static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::Q_STAGE));
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
