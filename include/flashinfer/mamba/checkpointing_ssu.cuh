/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_MAMBA_CHECKPOINTING_SSU_CUH_
#define FLASHINFER_MAMBA_CHECKPOINTING_SSU_CUH_

#include <cstdint>

namespace flashinfer::mamba::checkpointing {

struct CheckpointingSsuParams {
  uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{};
  uint32_t state_cache_size{};
  uint32_t npredicted{};
  uint32_t max_window{};
  int32_t pad_slot_id{-1};

  // v12 §59: per-head DIM split factor.  Must be one of {1, 2, 4}.  The host
  // launcher dispatches to a kernel template specialized on this value; the
  // kernel cross-checks via assert(params.d_split == D_SPLIT).
  int32_t d_split{1};

  bool dt_softplus{false};

  // Note: Programmatic Dependent Launch is JIT-stamped via the `ENABLE_PDL`
  // constexpr (see checkpointing_ssu_customize_config.jinja).  Each .so has
  // its PDL mode baked in; no runtime field needed.

  // ── Tensor pointers ──
  void* __restrict__ state{nullptr};    // (state_cache_size, nheads, dim, dstate)
  void* __restrict__ x{nullptr};        // (batch, NPREDICTED, nheads, dim)
  void* __restrict__ dt{nullptr};       // (batch, NPREDICTED, nheads, dim) tie_hdim
  void* __restrict__ A{nullptr};        // (nheads, dim, dstate) tie_hdim
  void* __restrict__ B{nullptr};        // (batch, NPREDICTED, ngroups, dstate)
  void* __restrict__ C{nullptr};        // (batch, NPREDICTED, ngroups, dstate)
  void* __restrict__ D{nullptr};        // (nheads, dim), optional
  void* __restrict__ z{nullptr};        // (batch, NPREDICTED, nheads, dim), optional
  void* __restrict__ dt_bias{nullptr};  // (nheads, dim) tie_hdim, optional
  void* __restrict__ output{nullptr};   // (batch, NPREDICTED, nheads, dim)

  // ── Ring-buffer cache tensors (ReplaySSM contract, 2026-07-10) ──
  // Single-buffered, head-major.  The physical ring length RING_BUFFER_LEN is
  // RUNTIME-implicit (`ring_buffer_len` below = the tensors' row count, never
  // a JIT key); the LOGICAL replay window is the compile-time MAX_WINDOW =
  // RING_BUFFER_LEN - NPREDICTED (flush rule pnat + 2T > RING_BUFFER_LEN ⇔
  // pnat + T > MAX_WINDOW).  ring_start[slot] is the oldest live row; live
  // rows are (ring_start + j) % RING_BUFFER_LEN for j in [0, pnat); appends
  // land at (ring_start + pnat + i) % RING_BUFFER_LEN.  The HOST owns all
  // bookkeeping (start advance on flush, pnat update) — kernels only read
  // ring_start/pnat and append.  No decay is cached: replay decays are
  // recomputed from dt_cache.
  void* __restrict__ x_cache{nullptr};     // (state_cache_size, nheads, RING_BUFFER_LEN, dim)
  void* __restrict__ B_cache{nullptr};     // (state_cache_size, ngroups, RING_BUFFER_LEN, dstate)
  void* __restrict__ dt_cache{nullptr};    // (state_cache_size, nheads, RING_BUFFER_LEN) f32
  void* __restrict__ ring_start{nullptr};  // (state_cache_size,) int32
  void* __restrict__ prev_num_accepted{nullptr};  // (state_cache_size,) int32
  int32_t ring_buffer_len{};                      // = MAX_WINDOW + NPREDICTED (host-asserted)

  // ── Index tensors ──
  void* __restrict__ state_batch_indices{nullptr};  // (batch,) optional

  // ── Varlen (v20): packed inputs ──
  // When non-null, `x/dt/B/C/z/out` are laid out as
  // `(1, total_tokens, nheads, dim)` / `(1, total_tokens, ngroups, dstate)`
  // and `cu_seqlens[i]` gives the token-axis base of sequence i.
  // `seq_len_i = cu_seqlens[i+1] - cu_seqlens[i]`.  Kernel dispatch on
  // `cu_seqlens != nullptr` selects a `VARLEN=true` template.
  //
  // The `*_stride_seq` fields below already encode the outer iteration
  // stride for both modes — the wrapper sets them to:
  //   non-varlen:  `tensor.stride(0)`  (per-batch)
  //   varlen   :  `tensor.stride(1)`  (per-token, since sequences are packed
  //                                    into a single batch of total_tokens)
  // so the kernel uses one formula `seq * *_stride_seq` regardless of mode.
  void* __restrict__ cu_seqlens{nullptr};  // (batch+1,) int32, optional

  // ── Block-scale decode factors for quantized state ──
  void* __restrict__ state_scale{nullptr};  // float32: (state_cache_size, nheads, dim)

  // ── Philox PRNG seed for stochastic rounding ──
  const int64_t* rand_seed{nullptr};

  // ── Strides ──
  // state: (state_cache_size, nheads, dim, dstate) — inner 3 dims contiguous
  int64_t state_stride_seq{};

  // For the six batch-side tensors (x, dt, B, C, out, z), `*_stride_seq`
  // is the outer iteration stride — per-batch in non-varlen, per-token in
  // varlen.  `*_stride_token` is the inner per-row (T-axis) stride, same
  // in both modes.

  // x: (batch, NPREDICTED, nheads, dim) [non-varlen] / (1, total_tokens, nheads, dim) [varlen]
  int64_t x_stride_seq{};
  int64_t x_stride_token{};

  // dt: (batch, NPREDICTED, nheads, dim) — tie_hdim (stride_dim=0)
  int64_t dt_stride_seq{};
  int64_t dt_stride_token{};

  // B: (batch, NPREDICTED, ngroups, dstate)
  int64_t B_stride_seq{};
  int64_t B_stride_token{};

  // C: (batch, NPREDICTED, ngroups, dstate)
  int64_t C_stride_seq{};
  int64_t C_stride_token{};

  // output: (batch, NPREDICTED, nheads, dim)
  int64_t out_stride_seq{};
  int64_t out_stride_token{};

  // z: (batch, NPREDICTED, nheads, dim)
  int64_t z_stride_seq{};
  int64_t z_stride_token{};

  // x_cache: (state_cache_size, nheads, RING_BUFFER_LEN, dim) — row (dim) contiguous,
  // one head's RING_BUFFER_LEN rows contiguous
  int64_t x_cache_stride_seq{};
  int64_t x_cache_stride_head{};
  int64_t x_cache_stride_pos{};

  // B_cache: (state_cache_size, ngroups, RING_BUFFER_LEN, dstate)
  int64_t B_cache_stride_seq{};
  int64_t B_cache_stride_group{};
  int64_t B_cache_stride_pos{};

  // dt_cache: (state_cache_size, nheads, RING_BUFFER_LEN) — pos contiguous
  int64_t dt_cache_stride_seq{};
  int64_t dt_cache_stride_head{};

  // state_scale: (state_cache_size, nheads, dim)
  int64_t state_scale_stride_seq{};

  // ── Two-kernel split scratch (both non-null ⇒ precompute+main path) ──
  // Appended at the very end of the struct on purpose: keeps every field
  // above at its original offset, so the monolithic kernel's struct layout
  // (and cache-line access pattern) is byte-for-byte unchanged.
  // Caller-provided (graph-safe, like `output` — no in-wrapper allocation).
  // Precompute writes them per-head; main reads them.  Both null ⇒ monolithic.
  //
  // cb_scaled is bf16 (= input_t) in FRAGMENT-NATIVE layout
  // [batch, nheads, lane(0..31), reg(0..7)] = matmul-4's fragA
  // (mma.m16n8k16 A operand): the precompute STG.128s each lane's 8-bf16
  // fragment, the main LDG.128s it straight into fragA — no smem / LDSM /
  // swizzle on either side.  512 B/head.
  void* __restrict__ cb_scaled{nullptr};   // bf16 (batch, nheads, 32, 8), fragA-native
  void* __restrict__ cumAdt_vec{nullptr};  // f32 (batch, nheads, NPREDICTED_PAD_MMA_M) — raw cumAdt
  // CB_old (C6): old-token CB for the NO-WRITE path only.  Same fragA-native
  // layout as cb_scaled; the precompute writes it only when !must_checkpoint
  // (the write path folds old tokens into state via the replay instead), and
  // the main does OUT.3 = cb_old @ old_x.  Caller-provided when two-kernel.
  void* __restrict__ cb_old{nullptr};  // bf16 (batch, nheads, 32, 8), fragA-native
};

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_CHECKPOINTING_SSU_CUH_
