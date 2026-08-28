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

#pragma once

#include "arch/barrier.cuh"
#include "arch/common.cuh"
#include "arch/cp_async.cuh"
#include "arch/ldmatrix_sm120.cuh"
#include "arch/mma_sm120.cuh"
#include "arch/stmatrix_sm120.cuh"
#include "common/d2_load_b.cuh"
#include "common/fp8_quant.cuh"
#include "common/kv_cache_io.cuh"
#include "common/online_softmax.cuh"
#include "common/q_rope.cuh"
#include "common/scale_mma.cuh"
#include "common/smem_layout.cuh"
#include "common/xv_rope_mma.cuh"
#include "model/kv_cache_traits.cuh"
#include "model/scale_convert.cuh"

// Cold (launch-invariant) parameters shared by every prefill kernel family
// (SG/MG/dual in prefill_mg_kernel.cuh, swapAB in prefill_swapab_kernel.cuh);
// passed by grid constant.
struct PrefillColdParams {
  float sm_scale;
  int num_tokens;
  size_t stride_kv_block;
  // Dual-cache only (sparse_mla_prefill_mg_dual_kernel); ignored elsewhere.
  size_t stride_kv_block_extra;
  int topk_extra;          // dual-cache only. Runtime topk_extra so callers can
                           // pass any cdiv(max_model_len, compress_ratio) value
                           // without per-bound template instantiations.
  const float* attn_sink;  // [NUM_HEADS] float32, natural log domain. nullptr = disabled.
  const int* topk_length;  // [num_tokens] int32, nullptr = uniform TOPK.
  const int*
      topk_length_extra;  // [num_tokens] int32, dual-cache only. nullptr = uniform topk_extra.
};

// ============================================================================
// SG prefill tile configuration, per model type.
//
// The DeepSeek-family models run the namespace-scope 64/8/4 tile. DOTS3_SWA's
// 1024-wide nope doubles every KV-derived buffer: at BI=64 the KV double buffer
// alone is 64 * 1040 * 2 = 133120 B against sm120's 101376 B per-block opt-in
// cap, before Q (19 KB) or W_FP8. BI=32 brings the whole layout to ~93 KB.
//
// The two warp counts are independent, and for DOTS3_SWA they have to be:
//
//   QK_WARPS  = BI / 8. The QK m16n8k32 MMA gives each warp exactly one n-tile
//               of 8 candidates (selected by `gid = lane >> 2`), so the
//               candidate-parallel warp count is fixed by BI. BI has a floor of
//               32 — the FP8 XV MMA consumes k=32 entries per step and
//               d2_load_b_fp8 reads entry_base+16+tid*4+3.
//   MATH_WARPS = how the D_V output columns are split. `acc_o` is
//               ACC_TILES * 4 = D_V / (8 * MATH_WARPS) * 4 floats per thread —
//               note HPB does not appear, since it is the MMA's M and every
//               lane holds 4 of the 128 outputs either way. MATH_WARPS is the
//               only lever on accumulator registers.
//
// Tying the two together is what hurts DOTS3_SWA: BI=32 forces 4 warps, and at
// 4 warps D_V=1024 puts 128 floats of accumulator in every thread. Splitting
// them lets 4 warps do QK while 8 share the XV MMA and the epilogue, halving
// `acc_o` to 64 and halving XV wall time. The split is nearly free because the
// XV MMA sub-loop already reads only smem (`w_fp8`, `w_head_sc_all`, `kv_smem`)
// and is already separated from the register-dependent W-quantize by a barrier.
//
// Going to 12 warps also re-enables setmaxnreg: at 384 threads nvcc's baseline
// is capped near 65536/384 = 170, so the math warps have something real to
// claim from the IO warps. At 256 threads it could already allocate 255 on its
// own and a `.inc` to 232 would have been an invalid decrease.
//
// Measured on RTX PRO 6000 (sm120), DOTS3_SWA prefill, coupled 4/4 -> split
// 4 QK / 8 XV: spill 240 B -> 152 B, smem unchanged at 93584 B, and 2-6% faster
// across (num_heads, num_tokens) in {16,64} x {256,1024}. The gain is well
// under the 2x the XV split alone implies, because the kernel is bound on the
// KV gather (each token pulls window * 1152 B, and NUM_HEADS > 16 re-reads it
// once per 16-head CTA) rather than on the XV MMA.
//
// MG is deliberately not parameterized. Its per-group Q and KV buffers put a
// D_NOPE=1024 model over the cap at any BI that still satisfies the k=32 floor,
// so DOTS3_SWA is SG-only and dispatches by CTA replication for NUM_HEADS > 16.
template <ModelType MT>
struct PrefillTilePrimary {
  static constexpr int CAND_WINDOW = BI;           // 64
  static constexpr int MATH_WARPS = N_MATH_WARPS;  // 8 == CAND_WINDOW/8, no split
  static constexpr int IO_WARPS = N_IO_WARPS;      // 4
  static constexpr bool REG_REALLOC = true;
  // A top-k candidate list is a scattered selection out of a long context that
  // no other CTA is likely to want, so the gather marks it evict-first to leave
  // the rest of L2 alone.
  static constexpr bool L2_EVICT_FIRST = true;
  // Sliding-window bound on a token's candidate count; 0 for models whose
  // candidate list is a genuine top-k with no positional bound.
  static constexpr int WINDOW = 0;
};

template <>
struct PrefillTilePrimary<ModelType::DOTS3_SWA> {
  // 1024-wide nope: at BI=64 the KV double buffer alone is 64 * 1040 * 2 =
  // 133120 B against sm120's 101376 B per-block opt-in cap, before Q (19 KB) or
  // W_FP8. BI=32 brings the whole layout to 93584 B.
  static constexpr int CAND_WINDOW = 32;  // -> QK_WARPS = 4
  static constexpr int MATH_WARPS = 8;    // XV/epilogue split, decoupled from QK
  static constexpr int IO_WARPS = 4;
  static constexpr bool REG_REALLOC = true;
  // Here the candidate list *is* a sliding window: consecutive query tokens
  // share all but one of their 513 entries, and one CTA serves one query token,
  // so the gathered rows are the hottest data in the kernel and the unique
  // working set is only (num_tokens + window) * 1152 B. Evict-first would throw
  // away exactly what the next CTA is about to ask for.
  static constexpr bool L2_EVICT_FIRST = false;
  static constexpr int WINDOW = 513;  // the family's sliding-window size
};

template <ModelType MT>
struct PrefillTileCfg {
  using P = PrefillTilePrimary<MT>;
  static constexpr int MATH_WARPS = P::MATH_WARPS;
  static constexpr int IO_WARPS = P::IO_WARPS;
  static constexpr bool REG_REALLOC = P::REG_REALLOC;
  static constexpr bool L2_EVICT_FIRST = P::L2_EVICT_FIRST;

  static constexpr int N_TOTAL_WARPS = MATH_WARPS + IO_WARPS;  // 12 for both
  static constexpr int BLOCK_THREADS = N_TOTAL_WARPS * 32;     // 384
  static constexpr int MATH_THREADS = MATH_WARPS * 32;         // 256
  static constexpr int IO_THREADS = IO_WARPS * 32;             // 128
  static constexpr int ENTRIES_PER_WARP = 8;                   // pinned by the QK MMA
  static constexpr int BI = P::CAND_WINDOW;                    // DS 64,  DOTS3_SWA 32
  static constexpr int QK_WARPS = BI / ENTRIES_PER_WARP;       // DS 8,   DOTS3_SWA 4
  static constexpr int QK_THREADS = QK_WARPS * 32;             // DS 256, DOTS3_SWA 128
  // When set, warps [QK_WARPS, MATH_WARPS) skip QK/softmax/W-quantize and join
  // only for the XV MMA and the epilogue.
  static constexpr bool SPLIT_QK_XV = MATH_WARPS != QK_WARPS;

  // Registers the math warps claim after the IO warps drop to 32. The ceiling
  // is (65536 - IO_THREADS * 32) / MATH_THREADS = 240 at 8+4 warps; 232 leaves
  // a warpgroup-granularity margin and is what every config has used.
  static constexpr int MATH_MAXNREG = 232;

  static constexpr int WINDOW = P::WINDOW;
  static constexpr bool HAS_WINDOW = WINDOW > 0;

  static_assert(BI >= 32, "FP8 XV consumes k=32 entries per step; BI < 32 reads past the tile");
  static_assert(BI % 32 == 0, "BI must be a whole number of FP8 XV k-steps");
  static_assert(MATH_WARPS >= QK_WARPS,
                "the XV split may widen the QK warp set but never narrow it");
  static_assert(MATH_WARPS % 4 == 0 && IO_WARPS % 4 == 0,
                "setmaxnreg is warpgroup-aligned; both halves must be whole warpgroups");
  static_assert(QK_WARPS % 4 == 0,
                "the QK warps must be a whole warpgroup prefix of the math warps");
  static_assert(!REG_REALLOC || MATH_MAXNREG * MATH_THREADS + 32 * IO_THREADS <= 65536,
                "redistributed register budget exceeds the per-SM file");
};

template <ModelType MT, int PAGE_BLOCK_SIZE>
__device__ __forceinline__ const uint8_t* prefill_kv_entry_base(
    const uint8_t* __restrict__ kv_global, int idx, size_t stride_kv_block) {
  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  idx = (idx >= 0) ? idx : 0;
  // Addressing mode follows the scale layout, not V_HAS_ROPE: an inline-scale
  // model (DSV3_2 / GLM_NSA / GLM53_NOPE) is a flat token array, a footer-scale
  // model (DSV4 / DOTS3_SWA) is paged with the footer after the block's data.
  // This matches io_bulk_gather_tile. Keying it on V_HAS_ROPE happened to agree
  // for the three DeepSeek-family models and disagrees for DOTS3_SWA, which is
  // footer-scaled with no rope in V.
  if constexpr (!KV::SCALE_IN_KV_SMEM) {
    const int bi = idx / PAGE_BLOCK_SIZE;
    const int li = idx % PAGE_BLOCK_SIZE;
    return kv_global + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;
  } else {
    return kv_global + (size_t)idx * IO::IO_STRIDE;
  }
}
