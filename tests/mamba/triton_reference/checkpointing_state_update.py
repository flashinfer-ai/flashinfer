# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py
# SPDX-FileCopyrightText: Copyright contributors to the sglang project
#
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py

import torch
import triton
import triton.language as tl

from .softplus import softplus

PAD_SLOT_ID = -1


def _get_sm_version() -> int:
    """SM version as a single integer (e.g., 100 for sm_100a, 89 for sm_89).

    Local equivalent of `tensorrt_llm._utils.get_sm_version` so that this
    reference has no trtllm dependency.
    """
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


# Backwards-compat alias for the trtllm-style import name.
get_sm_version = _get_sm_version

MIN_REPLAY_TILE_SIZE = 16

# Work-item layout for replay_work_items tensor (int32, shape (batch, 4)).
# Each row carries: [position_in_decode_batch, cache_slot, pnat, active_buf_idx].
_REPLAY_WORK_POSITION_IN_DECODE_BATCH = tl.constexpr(0)
_REPLAY_WORK_CACHE_SLOT = tl.constexpr(1)
_REPLAY_WORK_PNAT = tl.constexpr(2)
_REPLAY_WORK_CACHE_BUF_IDX = tl.constexpr(3)
_REPLAY_WORK_ITEM_WIDTH = tl.constexpr(4)

_TMA_ALLOCATOR_SET = False


def _ensure_tma_allocator() -> None:
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    def _alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_alloc_fn)
    _TMA_ALLOCATOR_SET = True


@triton.jit
def _stochastic_round_fp16x2(x: tl.tensor, rand: tl.tensor) -> tl.tensor:
    """Stochastic rounding: fp32 pair → fp16x2 using Philox random bits.

    Uses PTX cvt.rs.f16x2.f32 which rounds each fp32 value to fp16 using
    the random bits to break ties, avoiding systematic rounding bias that
    accumulates over many decode steps with fp16 state.

    Adapted from flashinfer (Apache-2.0, vLLM/mamba lineage).
    """
    return tl.inline_asm_elementwise(
        asm="""{
        cvt.rs.f16x2.f32 $0, $2, $1, $3;
        }""",
        constraints=("=r,r,r,r,r"),
        args=(x, rand),
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _stochastic_round_fp8x4_e4m3(x: tl.tensor, rand: tl.tensor) -> tl.tensor:
    """Stochastic rounding: fp32 quad → fp8 e4m3 using Philox random bits.

    Uses PTX cvt.rs.satfinite.e4m3x4.f32 which combines stochastic rounding
    and saturating cast in a single op (output is final fp8, no separate
    clamp needed).  The reversed source-register order {$4,$3,$2,$1} is
    load-bearing — PTX packs leftmost source into the high byte but Triton's
    pack=4 is little-endian, so the natural {$1,$2,$3,$4} order would
    silently shuffle every group of 4 contiguous outputs.

    Requires SM_100a+ (Blackwell B200).  Caller must gate at the wrapper
    level — this kernel does not check.

    Adapted from vLLM PR #40012 (Apache-2.0).
    """
    return tl.inline_asm_elementwise(
        asm="cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;",
        constraints="=r,r,r,r,r,r,r,r,r",
        args=(x, rand),
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _bitrev32(x: tl.tensor) -> tl.tensor:
    return tl.inline_asm_elementwise(
        asm="brev.b32 $0, $1;",
        constraints="=r,r",
        args=(x,),
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _stochastic_round_int8_packed(
    x: tl.tensor, rand: tl.tensor, offs_n: tl.tensor
) -> tl.tensor:
    """Stochastic rounding for int8 using one random uint32 per 4 values via _bitrev32."""
    low = rand & 0x0000FFFF
    high = (rand >> 16) & 0x0000FFFF
    low_rev = _bitrev32(low) >> 16
    high_rev = _bitrev32(high) >> 16
    rand_pos = offs_n & 3
    rand16 = tl.where(
        rand_pos == 0,
        low,
        tl.where(rand_pos == 1, low_rev, tl.where(rand_pos == 2, high, high_rev)),
    )
    rand01 = rand16.to(tl.float32) * (1.0 / float(1 << 16))
    return tl.extra.cuda.libdevice.floor(x + rand01)


@triton.jit
def _stochastic_round_int16_packed(
    x: tl.tensor, rand: tl.tensor, offs_n: tl.tensor
) -> tl.tensor:
    """Stochastic rounding for int16 using one random uint32 per 2 values via _bitrev32."""
    rand_bits = tl.where((offs_n & 1) == 0, rand, _bitrev32(rand))
    rand01 = (rand_bits & 0x00FFFFFF).to(tl.float32) * (1.0 / float(1 << 24))
    return tl.extra.cuda.libdevice.floor(x + rand01)


# Precompute kernel: CB_scaled, decay_vec.  Writes new cache (old_B,
# old_dt, old_dA_cumsum) to the WRITE buffer slot for next step's replay.
# Grid: (batch, nheads // HEADS_PER_BLOCK).


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)}
)
@triton.jit()
def _checkpointing_precompute_kernel(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Varlen: cu_seqlens[0..N] (int32) — when non-None, IS_VARLEN=True and
    # x/dt/B/C/out are packed (1, total_tokens, ...).  Outer offsets switch
    # from `pid_b * stride_*_batch` to `bos * stride_*_T`; T-axis predicates
    # tighten from `offs_t < T` to `offs_t < seq_len`.  Cache-side tensors
    # (old_*, state, cache_buf_idx, prev_num_accepted_tokens) remain
    # indexed by `cache_batch_idx` — varlen is purely a batch-side layout
    # change.
    cu_seqlens_ptr,
    # Output pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    # Cache pointers (both buffers reachable via stride_*_dbuf).  This
    # kernel writes to either the active (= cache_buf_idx) or inactive
    # (= 1 - cache_buf_idx) buffer depending on WRITE_CHECKPOINT — see
    # comment block at top of kernel body.
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    # Double-buffer index (per cache slot) — selects this step's "active"
    # buffer (= where the historical inputs for this step live).
    cache_buf_idx_ptr,
    # Per-request accepted-tokens count (already-cached old tokens at
    # [0, PNAT) of the active buffer; new tokens this step go after them
    # on no-checkpoint steps).
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides: (cache, 2, T, ngroups, dstate)
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    # Checkpointing flag — selects target buffer + offset for new-token
    # cache writes.  See "Cache write semantics" block below.
    WRITE_CHECKPOINT: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_hg = tl.program_id(axis=1)  # head-group index
    first_head = pid_hg * HEADS_PER_BLOCK

    # Varlen vs non-varlen unified: `seq` is the outer iteration index.
    #   non-varlen: seq = pid_b (per-batch).  wrapper passes
    #               stride_*_batch = X.stride(0)         (per-batch).
    #   varlen   : seq = cu_seqlens[pid_b] (per-token).  wrapper passes
    #               stride_*_batch = X.stride(1)         (per-token).
    # `seq * stride_*_batch` then lands on the start of this sequence's
    # row in both modes.  Cache-side accesses (old_*, state, etc.) stay
    # indexed by cache_batch_idx — varlen is purely a batch-side layout.
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        if seq_len <= 0:
            return
        seq = bos
    else:
        seq = pid_b.to(tl.int64)
        seq_len = T

    # Resolve cache index for writes
    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    # Signal main kernel to start (internal PDL).  Main's replay phase
    # reads only from the READ buffer (written by the PREVIOUS step) —
    # safe even if conv1d and this kernel are still running.  Main's
    # gdc_wait() gates the output phase, which reads conv1d outputs
    # (x, C) and this kernel's outputs (cb_scaled, decay_vec).
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()

    # --- Cache write semantics ---
    # cache_buf_idx names this step's "active" buffer — the one with the
    # historical inputs at [0, PNAT).  The other buffer is "staging".
    #
    # Where do we write new tokens this step?
    #   WRITE_CHECKPOINT=False (no overflow): append to ACTIVE buffer at
    #       offset [PNAT : PNAT+T).  Caller does NOT flip cache_buf_idx
    #       afterward; PNAT_next = PNAT + accepted.  [0, PNAT) preserved.
    #   WRITE_CHECKPOINT=True (would overflow): write to STAGING buffer at
    #       [0, T).  Caller flips cache_buf_idx afterward; next step's
    #       active = the one we just wrote.  PNAT_next = accepted.  Old
    #       data in the previous active buffer is folded into state via
    #       the replay update and discarded.  This matches today's replay
    #       kernel behavior exactly.
    buf_active = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    if WRITE_CHECKPOINT:
        write_buf = 1 - buf_active
        write_offset = 0
    else:
        write_buf = buf_active
        write_offset = prev_num_accepted_tokens

    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    # T-axis predicate: per-sequence seq_len in varlen (= T constexpr in
    # non-varlen — compiler folds).  Precompute only writes / reloads from
    # cache at positions it just wrote (the seq_len new tokens), so it
    # never needs a separate cache-side mask — the main kernel does.
    t_mask = offs_t < seq_len
    n_mask = offs_n < dstate

    # Causal mask is shared across all heads (depends only on offs_t)
    causal_mask = offs_t[:, None] >= offs_t[None, :]
    valid_mask = causal_mask & t_mask[:, None] & t_mask[None, :]

    # --- Loop 1: compute per-head dt/dA_cumsum/decay BEFORE gdc_wait ---
    # These only depend on dt (from in_proj, not conv1d) and parameters (A, dt_bias).
    # Store to cache; will reload after the wait for CB scaling.
    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local

        dt_base = dt_ptr + seq * stride_dt_batch + head_idx * stride_dt_head
        dt = tl.load(dt_base + offs_t * stride_dt_T, mask=t_mask, other=0.0).to(
            tl.float32
        )
        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + head_idx * stride_dt_bias_head).to(
                tl.float32
            )
            dt = dt + dt_bias
        if DT_SOFTPLUS:
            dt = softplus(dt)

        A = tl.load(A_ptr + head_idx * stride_A_head).to(tl.float32)
        dA_cumsum = tl.cumsum(A * dt, axis=0)
        decay_vec = tl.exp(dA_cumsum)

        # Cross-step continuity for old_dA_cumsum: when appending to active_buf
        # at PNAT > 0, add the previous tail value so the buffer holds one
        # continuous cumsum across back-to-back no-write steps.
        # Write path (write_offset = 0, fresh staging buffer) starts from 0.
        if not WRITE_CHECKPOINT:
            if prev_num_accepted_tokens > 0:
                prev_total = tl.load(
                    old_dA_cumsum_ptr
                    + cache_batch_idx * stride_old_dA_cumsum_cache
                    + write_buf * stride_old_dA_cumsum_dbuf
                    + head_idx * stride_old_dA_cumsum_head
                    + (prev_num_accepted_tokens - 1) * stride_old_dA_cumsum_T
                ).to(tl.float32)
                dA_cumsum = dA_cumsum + prev_total

        # Store dt, dA_cumsum to cache at [write_offset : write_offset+T) of
        # write_buf (selected by WRITE_CHECKPOINT — see top of kernel).
        old_dt_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + write_buf * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        tl.store(
            old_dt_base + (write_offset + offs_t) * stride_old_dt_T,
            dt,
            mask=t_mask,
        )

        old_dA_cumsum_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + write_buf * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        tl.store(
            old_dA_cumsum_base + (write_offset + offs_t) * stride_old_dA_cumsum_T,
            dA_cumsum,
            mask=t_mask,
        )

        # decay_vec is per-call scratch (not cached); always write at offs_t.
        decay_vec_base = (
            decay_vec_ptr + pid_b * stride_dv_batch + head_idx * stride_dv_head
        )
        tl.store(decay_vec_base + offs_t * stride_dv_t, decay_vec, mask=t_mask)

    # --- Wait for upstream kernel (external PDL) before loading B and C ---
    # All dt processing above is independent of conv1d outputs.
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # --- Load C and B once for the group (shared across HEADS_PER_BLOCK heads) ---
    group_idx = first_head // nheads_ngroups_ratio
    C_base = C_ptr + seq * stride_C_batch + group_idx * stride_C_group
    B_base = B_ptr + seq * stride_B_batch + group_idx * stride_B_group

    C_all = tl.load(
        C_base + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    B_all = tl.load(
        B_base + offs_t[:, None] * stride_B_T + offs_n[None, :] * stride_B_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    # Compute raw CB once — shared across all heads in this block
    raw_CB = tl.dot(C_all.to(tl.bfloat16), tl.trans(B_all).to(tl.bfloat16))

    # Store B to cache at [write_offset : write_offset+T) of write_buf.
    if first_head % nheads_ngroups_ratio == 0:
        old_B_base = (
            old_B_ptr
            + cache_batch_idx * stride_old_B_cache
            + write_buf * stride_old_B_dbuf
            + group_idx * stride_old_B_group
        )
        tl.store(
            old_B_base
            + (write_offset + offs_t)[:, None] * stride_old_B_T
            + offs_n[None, :] * stride_old_B_dstate,
            B_all,
            mask=t_mask[:, None] & n_mask[None, :],
        )

    # --- Loop 2: reload per-head dA_cumsum/dt from cache, scale CB ---
    # Reload from where loop 1 stored: write_buf at [write_offset, write_offset+T).
    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local

        # Reload dt and dA_cumsum from cache (just written in loop 1)
        old_dt_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + write_buf * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        dt = tl.load(
            old_dt_base + (write_offset + offs_t) * stride_old_dt_T,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)

        old_dA_cumsum_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + write_buf * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        dA_cumsum = tl.load(
            old_dA_cumsum_base + (write_offset + offs_t) * stride_old_dA_cumsum_T,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)

        # Scale raw_CB with per-head decay and dt
        decay_matrix = tl.exp(dA_cumsum[:, None] - dA_cumsum[None, :])
        CB_scaled = tl.where(valid_mask, raw_CB * decay_matrix * dt[None, :], 0.0)

        cb_scaled_base = (
            cb_scaled_ptr + pid_b * stride_cb_batch + head_idx * stride_cb_head
        )
        tl.store(
            cb_scaled_base
            + offs_t[:, None] * stride_cb_t
            + offs_t[None, :] * stride_cb_j,
            CB_scaled,
            mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T),
        )


# ── New precompute kernels (double-buffered old_x / rectangle path) ───────────


@triton.jit()
def _replay_precompute_impl(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Output pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    # Cache pointers
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    cache_buf_idx_ptr,
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    # Dimensions
    T: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    # Runtime (not constexpr): allows dynamic dispatch kernel to pass per-slot
    # needs_write without inlining two specializations.
    write_checkpoint,
    # Varlen: cu_seqlens[0..N] (int32) — only read when IS_VARLEN; Triton DCEs
    # the load otherwise.  Varlen is purely a batch-side layout: x/dt/B/C are
    # packed (1, total_tokens, ...) and the outer offset switches to
    # bos * stride_*_T.  Cache (old_*) indexing is unchanged.
    cu_seqlens_ptr,
    IS_VARLEN: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_hg = tl.program_id(axis=1)
    first_head = pid_hg * HEADS_PER_BLOCK

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    # Varlen seq resolution (batch-side only).  bos = packed-token offset of
    # this sequence; seq_len = its new-token count.  Non-varlen folds to the
    # constexpr T and per-batch offsets.
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        if seq_len <= 0:
            return
        dt_seq_off = bos * stride_dt_T
        B_seq_off = bos * stride_B_T
        C_seq_off = bos * stride_C_T
    else:
        seq_len = T
        dt_seq_off = pid_b * stride_dt_batch
        B_seq_off = pid_b * stride_B_batch
        C_seq_off = pid_b * stride_C_batch

    buf_active = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    if write_checkpoint:
        write_buf = 1 - buf_active
        write_offset = 0
    else:
        write_buf = buf_active
        write_offset = prev_num_accepted_tokens

    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    t_mask = offs_t < seq_len
    n_mask = offs_n < dstate

    causal_mask = offs_t[:, None] >= offs_t[None, :]
    valid_mask = causal_mask & t_mask[:, None] & t_mask[None, :]

    # Vectorized pre-wait phase: load dt (H, T) and compute scale_combo (H, T, T)
    # that stays live in registers across gdc_wait.
    offs_h = tl.arange(0, HEADS_PER_BLOCK)
    heads_block = first_head + offs_h  # (H,)

    dt_addrs = (
        dt_ptr
        + dt_seq_off
        + heads_block[:, None] * stride_dt_head
        + offs_t[None, :] * stride_dt_T
    )
    dt = tl.load(dt_addrs, mask=t_mask[None, :], other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + heads_block * stride_dt_bias_head).to(
            tl.float32
        )
        dt = dt + dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = softplus(dt)

    A = tl.load(A_ptr + heads_block * stride_A_head).to(tl.float32)  # (H,)
    dA_cumsum = tl.cumsum(A[:, None] * dt, axis=1)  # (H, T)
    decay_vec = tl.exp(dA_cumsum)  # (H, T)

    # Cross-step continuity: add tail from previous step when appending.
    if write_checkpoint or prev_num_accepted_tokens == 0:
        prev_total = tl.zeros((HEADS_PER_BLOCK,), dtype=tl.float32)
    else:
        last_cumsum_ptrs = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + buf_active * stride_old_dA_cumsum_dbuf
            + heads_block * stride_old_dA_cumsum_head
            + (prev_num_accepted_tokens - 1) * stride_old_dA_cumsum_T
        )
        prev_total = tl.load(last_cumsum_ptrs).to(tl.float32)

    tl.store(
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + write_buf * stride_old_dt_dbuf
        + heads_block[:, None] * stride_old_dt_head
        + (write_offset + offs_t)[None, :] * stride_old_dt_T,
        dt,
        mask=t_mask[None, :],
    )
    tl.store(
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + write_buf * stride_old_dA_cumsum_dbuf
        + heads_block[:, None] * stride_old_dA_cumsum_head
        + (write_offset + offs_t)[None, :] * stride_old_dA_cumsum_T,
        dA_cumsum + prev_total[:, None],
        mask=t_mask[None, :],
    )
    tl.store(
        decay_vec_ptr
        + pid_b * stride_dv_batch
        + heads_block[:, None] * stride_dv_head
        + offs_t[None, :] * stride_dv_t,
        decay_vec,
        mask=t_mask[None, :],
    )

    # scale_combo (H, T, T) stays live across gdc_wait.
    decay_matrix = tl.exp(dA_cumsum[:, :, None] - dA_cumsum[:, None, :])  # (H, T, T)
    scale_combo = decay_matrix * dt[:, None, :]  # (H, T, T)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    group_idx = first_head // nheads_ngroups_ratio
    C_tile = tl.load(
        C_ptr
        + C_seq_off
        + group_idx * stride_C_group
        + offs_t[:, None] * stride_C_T
        + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    B_tile = tl.load(
        B_ptr
        + B_seq_off
        + group_idx * stride_B_group
        + offs_t[:, None] * stride_B_T
        + offs_n[None, :] * stride_B_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    raw_CB = tl.dot(C_tile.to(tl.bfloat16), tl.trans(B_tile).to(tl.bfloat16))

    if first_head % nheads_ngroups_ratio == 0:
        tl.store(
            old_B_ptr
            + cache_batch_idx * stride_old_B_cache
            + write_buf * stride_old_B_dbuf
            + group_idx * stride_old_B_group
            + (write_offset + offs_t)[:, None] * stride_old_B_T
            + offs_n[None, :] * stride_old_B_dstate,
            B_tile,
            mask=t_mask[:, None] & n_mask[None, :],
        )

    CB_scaled_block = tl.where(
        valid_mask[None, :, :],
        raw_CB[None, :, :] * scale_combo,
        0.0,
    )  # (H, T, T)
    tl.store(
        cb_scaled_ptr
        + pid_b * stride_cb_batch
        + heads_block[:, None, None] * stride_cb_head
        + offs_t[None, :, None] * stride_cb_t
        + offs_t[None, None, :] * stride_cb_j,
        CB_scaled_block,
        mask=(offs_t[None, :, None] < BLOCK_SIZE_T)
        & (offs_t[None, None, :] < BLOCK_SIZE_T),
    )


@triton.jit()
def _rectangle_precompute_impl(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Output pointers
    cb_scaled_ptr,  # (batch, nheads, BLOCK_SIZE_T, BLOCK_SIZE_K)
    decay_vec_ptr,  # (batch, nheads, BLOCK_SIZE_T)
    # Cache pointers
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    cache_buf_idx_ptr,
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    USE_GATHER_FOR_NEW_TOKENS: tl.constexpr,
    # Varlen: see _replay_precompute_impl — batch-side packing only.
    cu_seqlens_ptr,
    IS_VARLEN: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_hg = tl.program_id(axis=1)
    first_head = pid_hg * HEADS_PER_BLOCK

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        if seq_len <= 0:
            return
        dt_seq_off = bos * stride_dt_T
        B_seq_off = bos * stride_B_T
        C_seq_off = bos * stride_C_T
    else:
        seq_len = T
        dt_seq_off = pid_b * stride_dt_batch
        B_seq_off = pid_b * stride_B_batch
        C_seq_off = pid_b * stride_C_batch

    # Nowrite-only: always append to active buffer.
    buf_active = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    write_buf = buf_active
    write_offset = prev_num_accepted_tokens

    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_window = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    t_mask = offs_t < seq_len
    n_mask = offs_n < dstate

    is_history_position = offs_window < prev_num_accepted_tokens
    safe_history_idx = tl.where(is_history_position, offs_window, 0)
    step_idx_from_window = offs_window - prev_num_accepted_tokens
    is_step_position = (step_idx_from_window >= 0) & (step_idx_from_window < seq_len)
    safe_step_idx = tl.where(is_step_position, step_idx_from_window, 0)

    offs_h = tl.arange(0, HEADS_PER_BLOCK)
    heads_block = first_head + offs_h

    # Load this step's dt, compute continuous dA_cumsum (H, T)
    dt_new = tl.load(
        dt_ptr
        + dt_seq_off
        + heads_block[:, None] * stride_dt_head
        + offs_t[None, :] * stride_dt_T,
        mask=t_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias_h = tl.load(dt_bias_ptr + heads_block * stride_dt_bias_head).to(
            tl.float32
        )
        dt_new = dt_new + dt_bias_h[:, None]
    if DT_SOFTPLUS:
        dt_new = softplus(dt_new)

    A_h = tl.load(A_ptr + heads_block * stride_A_head).to(tl.float32)
    dA_cumsum_step = tl.cumsum(A_h[:, None] * dt_new, axis=1)

    if prev_num_accepted_tokens == 0:
        dA_cumsum_prefix = tl.zeros((HEADS_PER_BLOCK,), dtype=tl.float32)
    else:
        dA_cumsum_prefix = tl.load(
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + buf_active * stride_old_dA_cumsum_dbuf
            + heads_block * stride_old_dA_cumsum_head
            + (prev_num_accepted_tokens - 1) * stride_old_dA_cumsum_T,
        ).to(tl.float32)

    tl.store(
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + write_buf * stride_old_dt_dbuf
        + heads_block[:, None] * stride_old_dt_head
        + (write_offset + offs_t)[None, :] * stride_old_dt_T,
        dt_new,
        mask=t_mask[None, :],
    )
    dA_cumsum_new = dA_cumsum_step + dA_cumsum_prefix[:, None]  # (H, T) continuous
    tl.store(
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + write_buf * stride_old_dA_cumsum_dbuf
        + heads_block[:, None] * stride_old_dA_cumsum_head
        + (write_offset + offs_t)[None, :] * stride_old_dA_cumsum_T,
        dA_cumsum_new,
        mask=t_mask[None, :],
    )

    # Load historical dt/dA_cumsum for combo_block before gdc_wait.
    group_idx = first_head // nheads_ngroups_ratio
    history_mask_h = is_history_position[None, :]

    history_B = tl.load(
        old_B_ptr
        + cache_batch_idx * stride_old_B_cache
        + buf_active * stride_old_B_dbuf
        + group_idx * stride_old_B_group
        + safe_history_idx[:, None] * stride_old_B_T
        + offs_n[None, :] * stride_old_B_dstate,
        mask=is_history_position[:, None] & n_mask[None, :],
        other=0.0,
    )

    old_dt_read_h = (
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + buf_active * stride_old_dt_dbuf
        + heads_block * stride_old_dt_head
    )
    old_dA_cumsum_read_h = (
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + buf_active * stride_old_dA_cumsum_dbuf
        + heads_block * stride_old_dA_cumsum_head
    )
    history_dt = tl.load(
        old_dt_read_h[:, None] + safe_history_idx[None, :] * stride_old_dt_T,
        mask=history_mask_h,
        other=0.0,
    ).to(tl.float32)
    history_dA_cumsum = tl.load(
        old_dA_cumsum_read_h[:, None]
        + safe_history_idx[None, :] * stride_old_dA_cumsum_T,
        mask=history_mask_h,
        other=0.0,
    ).to(tl.float32)

    # Remap new-step dt/dA_cumsum to window positions
    if USE_GATHER_FOR_NEW_TOKENS:
        step_gather_idx = tl.broadcast_to(
            safe_step_idx[None, :], (HEADS_PER_BLOCK, BLOCK_SIZE_K)
        )
        step_dt_in_window = tl.where(
            is_step_position[None, :],
            tl.gather(dt_new, step_gather_idx, axis=1),
            0.0,
        )
        step_dA_cumsum_in_window = tl.where(
            is_step_position[None, :],
            tl.gather(dA_cumsum_new, step_gather_idx, axis=1),
            0.0,
        )
    else:
        step_in_window_selector = (
            offs_t[:, None] == (offs_window[None, :] - prev_num_accepted_tokens)
        ) & is_step_position[None, :]
        step_dt_in_window = tl.sum(
            tl.where(step_in_window_selector[None, :, :], dt_new[:, :, None], 0.0),
            axis=1,
        )
        step_dA_cumsum_in_window = tl.sum(
            tl.where(
                step_in_window_selector[None, :, :], dA_cumsum_new[:, :, None], 0.0
            ),
            axis=1,
        )

    # decay_vec[t] = exp(continuous_cumsum[t])
    decay_vec_block = tl.exp(dA_cumsum_new)  # (H, T)
    tl.store(
        decay_vec_ptr
        + pid_b * stride_dv_batch
        + heads_block[:, None] * stride_dv_head
        + offs_t[None, :] * stride_dv_t,
        decay_vec_block,
        mask=t_mask[None, :],
    )

    # combo_block[h, t, j] = dt[j] * exp(cumsum[t] - cumsum[j])
    dt_factor_window = tl.where(
        is_history_position[None, :], history_dt, step_dt_in_window
    )
    neg_dA_cumsum_window = tl.where(
        is_history_position[None, :], -history_dA_cumsum, -step_dA_cumsum_in_window
    )
    exp_diff = tl.exp(neg_dA_cumsum_window[:, None, :] + dA_cumsum_new[:, :, None])
    combo_block = dt_factor_window[:, None, :] * exp_diff  # (H, T, window)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    C_tile = tl.load(
        C_ptr
        + C_seq_off
        + group_idx * stride_C_group
        + offs_t[:, None] * stride_C_T
        + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    step_B = tl.load(
        B_ptr
        + B_seq_off
        + group_idx * stride_B_group
        + offs_t[:, None] * stride_B_T
        + offs_n[None, :] * stride_B_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    step_B_in_window = tl.load(
        B_ptr
        + B_seq_off
        + group_idx * stride_B_group
        + safe_step_idx[:, None] * stride_B_T
        + offs_n[None, :] * stride_B_dstate,
        mask=is_step_position[:, None] & n_mask[None, :],
        other=0.0,
    )
    B_window = history_B + step_B_in_window
    raw_rect_CB = tl.dot(C_tile.to(tl.bfloat16), tl.trans(B_window).to(tl.bfloat16))

    if first_head % nheads_ngroups_ratio == 0:
        tl.store(
            old_B_ptr
            + cache_batch_idx * stride_old_B_cache
            + write_buf * stride_old_B_dbuf
            + group_idx * stride_old_B_group
            + (write_offset + offs_t)[:, None] * stride_old_B_T
            + offs_n[None, :] * stride_old_B_dstate,
            step_B,
            mask=t_mask[:, None] & n_mask[None, :],
        )

    # Causal mask: history positions + causal step positions
    t_idx_2d = offs_t[:, None]
    window_idx_2d = offs_window[None, :]
    is_history_2d = window_idx_2d < prev_num_accepted_tokens
    step_idx_from_window_2d = window_idx_2d - prev_num_accepted_tokens
    is_step_causal_2d = (
        (step_idx_from_window_2d >= 0)
        & (step_idx_from_window_2d < seq_len)
        & (step_idx_from_window_2d <= t_idx_2d)
    )
    causal_combined = (is_history_2d | is_step_causal_2d) & t_mask[:, None]

    rect_CB_scaled = tl.where(
        causal_combined[None, :, :],
        raw_rect_CB[None, :, :] * combo_block,
        0.0,
    )  # (H, T, window)
    tl.store(
        cb_scaled_ptr
        + pid_b * stride_cb_batch
        + heads_block[:, None, None] * stride_cb_head
        + offs_t[None, :, None] * stride_cb_t
        + offs_window[None, None, :] * stride_cb_j,
        rect_CB_scaled,
        mask=(offs_t[None, :, None] < BLOCK_SIZE_T)
        & (offs_window[None, None, :] < BLOCK_SIZE_K),
    )


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_CACHE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: max(
            triton.next_power_of_2(args["T"]), MIN_REPLAY_TILE_SIZE
        )
    }
)
@triton.heuristics(
    {
        "BLOCK_SIZE_K": lambda args: max(
            triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]),
            MIN_REPLAY_TILE_SIZE,
        )
    }
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.jit()
def _dynamic_precompute_kernel(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Output pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    # Cache pointers
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    cache_buf_idx_ptr,
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    RECTANGLE: tl.constexpr,
    RECTANGLE_USE_GATHER: tl.constexpr,
    # Varlen: cu_seqlens[0..N]; IS_VARLEN derived by the heuristic above.
    cu_seqlens_ptr,
    IS_VARLEN: tl.constexpr,
):
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()

    pid_b = tl.program_id(axis=0)
    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    pnat_local = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    # Varlen: per-slot seq_len gates needs_write; non-varlen folds to T.
    if IS_VARLEN:
        _vbos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        _veos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len_local = (_veos - _vbos).to(tl.int32)
    else:
        seq_len_local = T
    needs_write = pnat_local + seq_len_local > MAX_REPLAY_BUFFER_LENGTH

    if needs_write or not RECTANGLE:
        _replay_precompute_impl(
            dt_ptr,
            dt_bias_ptr,
            A_ptr,
            B_ptr,
            C_ptr,
            cb_scaled_ptr,
            decay_vec_ptr,
            old_B_ptr,
            old_dt_ptr,
            old_dA_cumsum_ptr,
            cache_buf_idx_ptr,
            prev_num_accepted_tokens_ptr,
            state_batch_indices_ptr,
            T,
            dstate,
            nheads_ngroups_ratio,
            stride_dt_batch,
            stride_dt_T,
            stride_dt_head,
            stride_dt_bias_head,
            stride_A_head,
            stride_B_batch,
            stride_B_T,
            stride_B_group,
            stride_B_dstate,
            stride_C_batch,
            stride_C_T,
            stride_C_group,
            stride_C_dstate,
            stride_cb_batch,
            stride_cb_head,
            stride_cb_t,
            stride_cb_j,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_t,
            stride_old_B_cache,
            stride_old_B_dbuf,
            stride_old_B_T,
            stride_old_B_group,
            stride_old_B_dstate,
            stride_old_dt_cache,
            stride_old_dt_dbuf,
            stride_old_dt_head,
            stride_old_dt_T,
            stride_old_dA_cumsum_cache,
            stride_old_dA_cumsum_dbuf,
            stride_old_dA_cumsum_head,
            stride_old_dA_cumsum_T,
            DT_SOFTPLUS,
            HAS_DT_BIAS,
            HAS_CACHE_BATCH_INDICES,
            BLOCK_SIZE_DSTATE,
            BLOCK_SIZE_T,
            LAUNCH_WITH_PDL,
            HEADS_PER_BLOCK,
            needs_write,
            cu_seqlens_ptr,
            IS_VARLEN,
        )
    else:
        _rectangle_precompute_impl(
            dt_ptr,
            dt_bias_ptr,
            A_ptr,
            B_ptr,
            C_ptr,
            cb_scaled_ptr,
            decay_vec_ptr,
            old_B_ptr,
            old_dt_ptr,
            old_dA_cumsum_ptr,
            cache_buf_idx_ptr,
            prev_num_accepted_tokens_ptr,
            state_batch_indices_ptr,
            T,
            MAX_REPLAY_BUFFER_LENGTH,
            dstate,
            nheads_ngroups_ratio,
            stride_dt_batch,
            stride_dt_T,
            stride_dt_head,
            stride_dt_bias_head,
            stride_A_head,
            stride_B_batch,
            stride_B_T,
            stride_B_group,
            stride_B_dstate,
            stride_C_batch,
            stride_C_T,
            stride_C_group,
            stride_C_dstate,
            stride_cb_batch,
            stride_cb_head,
            stride_cb_t,
            stride_cb_j,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_t,
            stride_old_B_cache,
            stride_old_B_dbuf,
            stride_old_B_T,
            stride_old_B_group,
            stride_old_B_dstate,
            stride_old_dt_cache,
            stride_old_dt_dbuf,
            stride_old_dt_head,
            stride_old_dt_T,
            stride_old_dA_cumsum_cache,
            stride_old_dA_cumsum_dbuf,
            stride_old_dA_cumsum_head,
            stride_old_dA_cumsum_T,
            DT_SOFTPLUS,
            HAS_DT_BIAS,
            HAS_CACHE_BATCH_INDICES,
            BLOCK_SIZE_DSTATE,
            BLOCK_SIZE_T,
            BLOCK_SIZE_K,
            LAUNCH_WITH_PDL,
            HEADS_PER_BLOCK,
            RECTANGLE_USE_GATHER,
            cu_seqlens_ptr,
            IS_VARLEN,
        )


# Main kernel: tl.dot replay + precomputed CB output.
# Grid: (cdiv(dim, M), batch, nheads).


@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_CACHE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics({"USE_RS_ROUNDING": lambda args: args["rand_seed_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)}
)
@triton.jit()
def _checkpointing_main_kernel(
    # Pointers
    state_ptr,
    # Per-(cache, head, dim) decode scale, fp32, only consulted when QUANT_MAX>0.
    # Layout (cache, nheads, dim) — broadcast over dstate at load/store.
    state_scales_ptr,
    # Cache READ pointers (read-buffer from previous step)
    old_x_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    # Cache WRITE pointer (write-buffer for old_x only; B/dt/dA_cumsum written by precompute)
    prev_num_accepted_tokens_ptr,
    cache_buf_idx_ptr,
    # New input pointers
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    # Precomputed pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    state_batch_indices_ptr,
    # Varlen: cu_seqlens[0..N] (int32) — when non-None, IS_VARLEN=True and
    # x/C/z/out are packed (1, total_tokens, ...).  See precompute kernel
    # for the full rationale.
    cu_seqlens_ptr,
    # Stochastic rounding
    rand_seed_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides: (cache, nheads, dim) — only used when QUANT_MAX>0
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides: (cache, T, nheads, dim) — single-buffered
    stride_old_x_cache,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # old_B strides: (cache, 2, T, ngroups, dstate)
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    # State quantization: 0.0 means non-quantized (fp16/bf16/fp32); >0 means
    # quantized (int8=127, int16=32767, fp8_e4m3fn=448).  Single in-kernel
    # switch for the dequant-on-load and encode-on-store paths.  Wrapper sets
    # this from state.dtype; kernel-entry static_assert below pins the
    # invariant that it must coincide with int8/int16/float8e4nv state dtype.
    QUANT_MAX: tl.constexpr,
    # Checkpointing flags
    WRITE_CHECKPOINT: tl.constexpr,  # When True: quantize+write post-replay state to HBM (checkpoint step).
    # When False: skip state write entirely (non-checkpoint step).
    RECTANGLE: tl.constexpr,  # Reserved for the rectangle non-checkpoint optimization path.
    # Currently asserted False at the wrapper; kernel takes the
    # replay-style code path unconditionally.
):
    # Compile-time invariant: QUANT_MAX > 0 must coincide with a quantized
    # state dtype (int8 / int16 / float8e4nv) and only those.  Cheap
    # insurance against a wrapper bug that desynchronizes the two.
    tl.static_assert(
        (QUANT_MAX > 0.0)
        == (
            (state_ptr.dtype.element_ty == tl.int8)
            or (state_ptr.dtype.element_ty == tl.int16)
            or (state_ptr.dtype.element_ty == tl.float8e4nv)
        ),
        "QUANT_MAX > 0.0 must coincide with int8 / int16 / float8e4nv state dtype.",
    )

    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # See precompute kernel for the unified-`seq` rationale.
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        if seq_len <= 0:
            return
        seq = bos
    else:
        seq = pid_b.to(tl.int64)
        seq_len = T

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    # Active buffer (= cache_buf_idx) holds the historical inputs for this
    # step at [0, PNAT).  The replay phase reads from there.  The new-tokens
    # write target depends on WRITE_CHECKPOINT — see Cache write semantics
    # block in the precompute kernel for the full rationale.
    active_buf = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    if WRITE_CHECKPOINT:
        write_buf = 1 - active_buf  # noqa: F841 — old_x is single-buffered (no use here)
        write_offset = 0
    else:
        write_buf = active_buf  # noqa: F841 — old_x is single-buffered (no use here)
        write_offset = prev_num_accepted_tokens

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate
    # Two T-axis predicates:
    #   t_mask     — batch-side (new tokens this call): runtime seq_len in
    #                varlen, constexpr T in non-varlen (compiler folds).
    #   cache_t_mask — cache-side (replay reads of up to prev_k items):
    #                  always the constexpr T bound.  Per-index gating is
    #                  via `tl.where(offs_t < prev_k, coeff, 0)` below.
    t_mask = offs_t < seq_len
    cache_t_mask = offs_t < T

    # Load state
    state_ptr += cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    state_ptrs = (
        state_ptr
        + offs_m[:, None] * stride_state_dim
        + offs_n[None, :] * stride_state_dstate
    )
    state_mask = m_mask[:, None] & n_mask[None, :]
    state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    # Dequantize on load (per-(head, dim) decode scale, broadcast over dstate).
    # Only consulted when QUANT_MAX>0 — non-quantized paths skip entirely.
    if QUANT_MAX > 0.0:
        state_scales_base = (
            state_scales_ptr
            + cache_batch_idx * stride_state_scales_cache
            + pid_h * stride_state_scales_head
        )
        decode_scale = tl.load(
            state_scales_base + offs_m * stride_state_scales_dim,
            mask=m_mask,
            other=1.0,
        ).to(tl.float32)
        state = state * decode_scale[:, None]

    # Phase 1: Replay via tl.dot fast-forward (reads from active_buf)
    group_idx = pid_h // nheads_ngroups_ratio

    # Load precomputed dt and dA_cumsum from READ buffer
    old_dt_base = (
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + active_buf * stride_old_dt_dbuf
        + pid_h * stride_old_dt_head
    )
    old_dt_all = tl.load(
        old_dt_base + offs_t * stride_old_dt_T, mask=cache_t_mask, other=0.0
    ).to(tl.float32)

    old_dA_cumsum_base = (
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + active_buf * stride_old_dA_cumsum_dbuf
        + pid_h * stride_old_dA_cumsum_head
    )
    old_dA_cumsum_all = tl.load(
        old_dA_cumsum_base + offs_t * stride_old_dA_cumsum_T,
        mask=cache_t_mask,
        other=0.0,
    ).to(tl.float32)

    # Load dA_cumsum at prev_k-1 directly via pointer math (avoids masked reduction).
    # Clamp to [0, T-1] defensively — out-of-contract PNAT > T would read OOB.
    prev_k_idx = tl.minimum(tl.maximum(prev_num_accepted_tokens - 1, 0), T - 1)
    total_dA_cumsum = tl.load(
        old_dA_cumsum_base + prev_k_idx * stride_old_dA_cumsum_T
    ).to(tl.float32)

    # Step 0 invariant: PNAT=0 means `state` is already last step's state (not
    # two back).  coeff is all-zero (offs_t < 0), total_decay is 1.0, so the
    # replay leaves `state` unchanged — cache contents don't matter on step 0.
    coeff = tl.exp(total_dA_cumsum - old_dA_cumsum_all) * old_dt_all
    coeff = tl.where(offs_t < prev_num_accepted_tokens, coeff, 0.0)

    # Load old_x: (BLOCK_SIZE_T, BLOCK_SIZE_M) — single-buffered
    old_x_base = (
        old_x_ptr + cache_batch_idx * stride_old_x_cache + pid_h * stride_old_x_head
    )
    old_x_all = tl.load(
        old_x_base
        + offs_t[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        mask=cache_t_mask[:, None] & m_mask[None, :],
        other=0.0,
    )

    # Load old_B from READ buffer: (BLOCK_SIZE_T, BLOCK_SIZE_DSTATE)
    old_B_base = (
        old_B_ptr
        + cache_batch_idx * stride_old_B_cache
        + active_buf * stride_old_B_dbuf
        + group_idx * stride_old_B_group
    )
    old_B_all = tl.load(
        old_B_base
        + offs_t[:, None] * stride_old_B_T
        + offs_n[None, :] * stride_old_B_dstate,
        mask=cache_t_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Scale B by coefficients
    dB_scaled = coeff[:, None] * old_B_all

    # Apply total decay to initial state FIRST, then add contributions
    total_decay = tl.where(prev_num_accepted_tokens > 0, tl.exp(total_dA_cumsum), 1.0)
    state *= total_decay

    # tl.dot fast-forward: old_x^T @ dB_scaled → (M, dstate)
    state += tl.dot(tl.trans(old_x_all).to(tl.bfloat16), dB_scaled.to(tl.bfloat16))

    # Write post-replay state — only on checkpoint steps.  When
    # WRITE_CHECKPOINT is False, the replay computed `state` is local-only and
    # discarded; skipping the HBM store + Philox path is the main performance
    # win of replay-style checkpointing on the common (non-checkpoint) step.
    if WRITE_CHECKPOINT:
        if USE_RS_ROUNDING:
            # Generate (M, dstate) random tensor for stochastic rounding.
            # Used by fp16 / int8 / int16 / fp8 SR paths below.  Quarter-sized
            # randint4x calls produce 4 u32s each; we interleave them into the
            # full (M, dstate) tensor — 4x fewer PRNG rounds vs per-element.
            rand_seed = tl.load(rand_seed_ptr)
            base_rand = cache_batch_idx * stride_state_batch + pid_h * stride_state_head
            offs_n_q = tl.arange(0, BLOCK_SIZE_DSTATE // 4)
            rand_offsets_q = (
                base_rand
                + offs_m[:, None] * stride_state_dim
                + offs_n_q[None, :] * (stride_state_dstate * 4)
            )  # (M, dstate//4)
            if PHILOX_ROUNDS > 0:
                r0, r1, r2, r3 = tl.randint4x(rand_seed, rand_offsets_q, PHILOX_ROUNDS)
            else:
                r0, r1, r2, r3 = tl.randint4x(rand_seed, rand_offsets_q)
            # Interleave 4 quarter-sized tensors → full (M, dstate) random tensor
            r01 = tl.join(r0, r1)  # (M, dstate//4, 2)
            r23 = tl.join(r2, r3)  # (M, dstate//4, 2)
            r0123 = tl.join(r01, r23)  # (M, dstate//4, 2, 2)
            rand = tl.reshape(r0123, (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE))

        if QUANT_MAX > 0.0:
            # Quantized state path: int8 / int16 / fp8_e4m3fn (RN or SR).
            # 1) Per-(head, dim) channel scale via amax over dstate.
            amax = tl.max(tl.abs(state), axis=1)  # (M,)
            encode_scale = tl.where(amax == 0.0, 1.0, QUANT_MAX / amax)  # (M,)
            decode_scale = 1.0 / encode_scale  # (M,)
            # 2) Store decode_scale (1/encode) so reads do a single multiply.
            state_scales_ptrs = (
                state_scales_ptr
                + cache_batch_idx * stride_state_scales_cache
                + pid_h * stride_state_scales_head
                + offs_m * stride_state_scales_dim
            )
            tl.store(state_scales_ptrs, decode_scale, mask=m_mask)
            # 3) Scale state into quant range — into a NEW variable so the
            # downstream output phase still sees the dequantized fp32 state.
            state_q = state * encode_scale[:, None]
            # 4) Round per dtype.  Order matters: handle fp8 SR first (PTX
            # combines round + saturating cast in one op, output is final fp8
            # so we store and finish on that branch).  Other branches share
            # the clamp + cast tail below.
            if USE_RS_ROUNDING and (state_ptrs.dtype.element_ty == tl.float8e4nv):
                # fp8_e4m3fn + SR — PTX cvt.rs.satfinite.e4m3x4.f32.  Output
                # is final fp8 (saturate included); store directly.
                tl.store(
                    state_ptrs,
                    _stochastic_round_fp8x4_e4m3(state_q, rand),
                    mask=state_mask,
                )
            else:
                if USE_RS_ROUNDING:
                    # int8 / int16 + SR — uniform-noise + floor.
                    # (fp8 SR was handled by the early branch above.)
                    tl.static_assert(
                        (state_ptrs.dtype.element_ty == tl.int8)
                        or (state_ptrs.dtype.element_ty == tl.int16),
                        "Quantized SR fall-through expects int8 or int16; "
                        "fp8 SR is handled by the prior branch.",
                    )
                    rand01 = (rand & 0x00FFFFFF).to(tl.float32) * (1.0 / float(1 << 24))
                    state_q = tl.extra.cuda.libdevice.floor(state_q + rand01)
                elif state_ptrs.dtype.element_ty != tl.float8e4nv:
                    # int8 / int16 + RN — explicit round before clamp.
                    # fp8 + RN deliberately skips this — explicit round() would
                    # destroy fp8 sub-integer precision; native cast at store
                    # does RN at the fp8 grid resolution.
                    tl.static_assert(
                        (state_ptrs.dtype.element_ty == tl.int8)
                        or (state_ptrs.dtype.element_ty == tl.int16),
                        "Quantized RN with explicit round() expects int8 or int16.",
                    )
                    state_q = tl.extra.cuda.libdevice.round(state_q)
                # Clamp + cast tail: int8/int16 (RN+SR) and fp8 RN.
                # fp8 RN reaches here without prior round() — .to(float8e4nv)
                # does native RN at the fp8 grid.
                state_q = tl.minimum(tl.maximum(state_q, -QUANT_MAX), QUANT_MAX)
                tl.store(
                    state_ptrs,
                    state_q.to(state_ptrs.dtype.element_ty),
                    mask=state_mask,
                )
        elif USE_RS_ROUNDING:
            # Non-quantized + SR: only fp16 (bf16 has no PTX SR cast; fp32
            # doesn't need rounding).
            tl.static_assert(
                state_ptrs.dtype.element_ty == tl.float16,
                "Non-quantized SR only supports fp16 state.",
            )
            tl.store(state_ptrs, _stochastic_round_fp16x2(state, rand), mask=state_mask)
        else:
            # Non-quantized + RN: fp16 / bf16 / fp32 native cast.
            tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=state_mask)

    # Phase 2: Output using precomputed CB_scaled and decay_vec.
    x_ptr += seq * stride_x_batch + pid_h * stride_x_head
    C_ptr += seq * stride_C_batch + group_idx * stride_C_group
    if HAS_Z:
        z_ptr += seq * stride_z_batch + pid_h * stride_z_head
    out_ptr += seq * stride_out_batch + pid_h * stride_out_head

    if HAS_D:
        D = tl.load(
            D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)

    # Wait for precompute kernel (PDL) before reading its outputs.
    # With chained PDL (conv1d → precompute → main), gdc_wait() ensures
    # precompute has completed — which transitively ensures conv1d has
    # completed (precompute waited on conv1d via its own gdc_wait).
    # All loads below (x, C from conv1d; CB_scaled, decay_vec from precompute)
    # are safe after this point.
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # Load conv1d outputs: C_all and x_all
    C_all = tl.load(
        C_ptr + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    x_all = tl.load(
        x_ptr + offs_t[:, None] * stride_x_T + offs_m[None, :] * stride_x_dim,
        mask=t_mask[:, None] & m_mask[None, :],
        other=0.0,
    )
    # Store new x to old_x cache at [write_offset : write_offset+T).
    # old_x is single-buffered: write goes to the active buffer regardless;
    # replay already read positions [0, PNAT) so write_offset = PNAT (no
    # overlap) on no-checkpoint steps.  On checkpoint steps write_offset = 0
    # (cache reset; old data folded into state via replay update).
    tl.store(
        old_x_base
        + (write_offset + offs_t)[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        x_all,
        mask=t_mask[:, None] & m_mask[None, :],
    )
    x_all = x_all.to(tl.float32)

    # Load precomputed CB_scaled and decay_vec
    cb_scaled_base = cb_scaled_ptr + pid_b * stride_cb_batch + pid_h * stride_cb_head
    CB_scaled = tl.load(
        cb_scaled_base + offs_t[:, None] * stride_cb_t + offs_t[None, :] * stride_cb_j,
        mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T),
        other=0.0,
    ).to(tl.float32)

    decay_vec_base = decay_vec_ptr + pid_b * stride_dv_batch + pid_h * stride_dv_head
    decay_vec = tl.load(
        decay_vec_base + offs_t * stride_dv_t, mask=t_mask, other=0.0
    ).to(tl.float32)

    # init_out = C_all @ state^T * decay_vec
    init_out = (
        tl.dot(C_all.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16))
        * decay_vec[:, None]
    )

    # cb_out = CB_scaled @ x_all
    cb_out = tl.dot(CB_scaled.to(tl.bfloat16), x_all.to(tl.bfloat16))

    out_all = init_out + cb_out

    if HAS_D:
        out_all = out_all + x_all * D[None, :]

    if HAS_Z:
        for t in range(T):
            z_t = tl.load(
                z_ptr + t * stride_z_T + offs_m * stride_z_dim, mask=m_mask, other=0.0
            ).to(tl.float32)
            out_t = tl.sum(tl.where((offs_t == t)[:, None], out_all, 0.0), axis=0)
            out_t = out_t * z_t * tl.sigmoid(z_t)
            tl.store(
                out_ptr + t * stride_out_T + offs_m * stride_out_dim, out_t, mask=m_mask
            )
    else:
        out_all_ptrs = (
            out_ptr + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
        )
        tl.store(out_all_ptrs, out_all, mask=t_mask[:, None] & m_mask[None, :])


# ── New main kernels (double-buffered old_x / rectangle / persistent) ─────────


@triton.jit()
def _persistent_main_impl(
    # Per-work-unit indices resolved by the persistent wrapper
    pid_m,
    pid_b,
    pid_h,
    cache_batch_idx,
    active_buf,
    prev_num_accepted_tokens,
    # Pointers
    state_ptr,
    state_scales_ptr,
    old_x_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    rand_seed_ptr,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides (double-buffered: cache, dbuf, T, head, dim)
    stride_old_x_cache,
    stride_old_x_dbuf,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_WINDOW: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    WRITE_CHECKPOINT: tl.constexpr,
    # IS_DYNAMIC: when True (persistent_dynamic), is_write is resolved per-slot
    # from PNAT at runtime.  When False (persistent_main), is_write is the
    # WRITE_CHECKPOINT constexpr.  Ported from the standalone reference to fix
    # the dropped runtime write-gating: dynamic + RECTANGLE=False previously
    # passed WRITE_CHECKPOINT=False to every slot, so checkpoint state was never
    # written (replay still ran, so output looked correct).
    IS_DYNAMIC: tl.constexpr = False,
    # WRITE_CHECKPOINT_IS_CONSTEXPR: force is_write to the WRITE_CHECKPOINT
    # constexpr even in dynamic mode.  The rectangle write-arm sets this after
    # the outer kernel has already narrowed the runtime branch to write slots,
    # so the no-write codegen DCEs.
    WRITE_CHECKPOINT_IS_CONSTEXPR: tl.constexpr = False,
    # Varlen: x/C/out are packed (1, total_tokens, ...); see precompute.  Only
    # the batch-side outer offset changes; cache indexing is unaffected.
    cu_seqlens_ptr=None,
    IS_VARLEN: tl.constexpr = False,
):
    tl.static_assert(
        (QUANT_MAX > 0.0)
        == (
            (state_ptr.dtype.element_ty == tl.int8)
            or (state_ptr.dtype.element_ty == tl.int16)
            or (state_ptr.dtype.element_ty == tl.float8e4nv)
        ),
        "QUANT_MAX > 0.0 must coincide with int8 / int16 / float8e4nv state dtype.",
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_window = tl.arange(0, BLOCK_SIZE_WINDOW)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate

    # Varlen: batch-side packing only — bos is this sequence's packed-token
    # offset, seq_len its new-token count.  Non-varlen folds to constexpr T and
    # per-batch offsets.  Cache (state/old_*) indexing is unaffected.
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        x_seq_off = bos * stride_x_T
        C_seq_off = bos * stride_C_T
        out_seq_off = bos * stride_out_T
        z_seq_off = bos * stride_z_T
    else:
        seq_len = T
        x_seq_off = pid_b * stride_x_batch
        C_seq_off = pid_b * stride_C_batch
        out_seq_off = pid_b * stride_out_batch
        z_seq_off = pid_b * stride_z_batch
    t_mask = offs_t < seq_len
    cache_window_mask = offs_window < MAX_REPLAY_BUFFER_LENGTH

    # Resolve is_write (ported from the standalone reference):
    #   - WRITE_CHECKPOINT_IS_CONSTEXPR: caller already narrowed to write slots
    #     → use the WRITE_CHECKPOINT constexpr (no-write codegen DCEs).
    #   - IS_DYNAMIC (persistent_dynamic, RECTANGLE=False caller): per-slot
    #     runtime branch on PNAT.  Both write/no-write codegen live in one body.
    #   - else (persistent_main): WRITE_CHECKPOINT constexpr from caller.
    if WRITE_CHECKPOINT_IS_CONSTEXPR:
        is_write: tl.constexpr = WRITE_CHECKPOINT
    elif IS_DYNAMIC:
        is_write = (prev_num_accepted_tokens + seq_len) > MAX_REPLAY_BUFFER_LENGTH
    else:
        is_write = WRITE_CHECKPOINT
    if is_write:
        write_offset = 0
        write_buf = 1 - active_buf
    else:
        write_offset = prev_num_accepted_tokens
        write_buf = active_buf

    # Load state
    state_ptr_local = (
        state_ptr + cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    )
    state_ptrs = (
        state_ptr_local
        + offs_m[:, None] * stride_state_dim
        + offs_n[None, :] * stride_state_dstate
    )
    state_mask = m_mask[:, None] & n_mask[None, :]
    state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    if QUANT_MAX > 0.0:
        decode_scale = tl.load(
            state_scales_ptr
            + cache_batch_idx * stride_state_scales_cache
            + pid_h * stride_state_scales_head
            + offs_m * stride_state_scales_dim,
            mask=m_mask,
            other=1.0,
        ).to(tl.float32)
        state = state * decode_scale[:, None]

    # Phase 1: Replay via tl.dot (double-buffered old_x, reads from active_buf)
    group_idx = pid_h // nheads_ngroups_ratio

    old_dt_base = (
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + active_buf * stride_old_dt_dbuf
        + pid_h * stride_old_dt_head
    )
    old_dA_cumsum_base = (
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + active_buf * stride_old_dA_cumsum_dbuf
        + pid_h * stride_old_dA_cumsum_head
    )

    old_dt_all = tl.load(
        old_dt_base + offs_window * stride_old_dt_T,
        mask=cache_window_mask,
        other=0.0,
    ).to(tl.float32)
    old_dA_cumsum_all = tl.load(
        old_dA_cumsum_base + offs_window * stride_old_dA_cumsum_T,
        mask=cache_window_mask,
        other=0.0,
    ).to(tl.float32)

    prev_k_idx = tl.minimum(
        tl.maximum(prev_num_accepted_tokens - 1, 0), MAX_REPLAY_BUFFER_LENGTH - 1
    )
    total_dA_cumsum = tl.load(
        old_dA_cumsum_base + prev_k_idx * stride_old_dA_cumsum_T
    ).to(tl.float32)

    coeff = tl.exp(total_dA_cumsum - old_dA_cumsum_all) * old_dt_all
    coeff = tl.where(offs_window < prev_num_accepted_tokens, coeff, 0.0)

    # Load old_x from double-buffered cache (active_buf at [0, PNAT))
    old_x_base = (
        old_x_ptr
        + cache_batch_idx * stride_old_x_cache
        + active_buf * stride_old_x_dbuf
        + pid_h * stride_old_x_head
    )
    old_x_all = tl.load(
        old_x_base
        + offs_window[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        mask=cache_window_mask[:, None] & m_mask[None, :],
        other=0.0,
    )

    old_B_base = (
        old_B_ptr
        + cache_batch_idx * stride_old_B_cache
        + active_buf * stride_old_B_dbuf
        + group_idx * stride_old_B_group
    )
    old_B_all = tl.load(
        old_B_base
        + offs_window[:, None] * stride_old_B_T
        + offs_n[None, :] * stride_old_B_dstate,
        mask=cache_window_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    dB_scaled = coeff[:, None] * old_B_all
    total_decay = tl.where(prev_num_accepted_tokens > 0, tl.exp(total_dA_cumsum), 1.0)
    state *= total_decay
    state += tl.dot(tl.trans(old_x_all).to(tl.bfloat16), dB_scaled.to(tl.bfloat16))

    # Phase 3: Write post-replay state (checkpoint step only)
    if is_write:
        if USE_RS_ROUNDING:
            rand_seed = tl.load(rand_seed_ptr + cache_batch_idx)
            base_rand = cache_batch_idx * stride_state_batch + pid_h * stride_state_head
            offs_n_q = tl.arange(0, BLOCK_SIZE_DSTATE // 4)
            rand_offsets_q = (
                base_rand
                + offs_m[:, None] * stride_state_dim
                + offs_n_q[None, :] * (stride_state_dstate * 4)
            )
            if PHILOX_ROUNDS > 0:
                r0, r1, r2, r3 = tl.randint4x(rand_seed, rand_offsets_q, PHILOX_ROUNDS)
            else:
                r0, r1, r2, r3 = tl.randint4x(rand_seed, rand_offsets_q)
            r01 = tl.join(r0, r1)
            r23 = tl.join(r2, r3)
            r0123 = tl.join(r01, r23)
            rand = tl.reshape(r0123, (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE))

        if QUANT_MAX > 0.0:
            amax = tl.max(tl.abs(state), axis=1)
            encode_scale = tl.where(amax == 0.0, 1.0, QUANT_MAX / amax)
            decode_scale = 1.0 / encode_scale
            tl.store(
                state_scales_ptr
                + cache_batch_idx * stride_state_scales_cache
                + pid_h * stride_state_scales_head
                + offs_m * stride_state_scales_dim,
                decode_scale,
                mask=m_mask,
            )
            state_q = state * encode_scale[:, None]
            if USE_RS_ROUNDING and (state_ptrs.dtype.element_ty == tl.float8e4nv):
                tl.store(
                    state_ptrs,
                    _stochastic_round_fp8x4_e4m3(state_q, rand),
                    mask=state_mask,
                )
            else:
                if USE_RS_ROUNDING:
                    tl.static_assert(
                        (state_ptrs.dtype.element_ty == tl.int8)
                        or (state_ptrs.dtype.element_ty == tl.int16),
                        "Quantized SR fall-through expects int8 or int16.",
                    )
                    state_q = (
                        _stochastic_round_int8_packed(
                            state_q, rand.to(tl.uint32), offs_n[None, :]
                        )
                        if state_ptrs.dtype.element_ty == tl.int8
                        else _stochastic_round_int16_packed(
                            state_q, rand.to(tl.uint32), offs_n[None, :]
                        )
                    )
                elif state_ptrs.dtype.element_ty != tl.float8e4nv:
                    tl.static_assert(
                        (state_ptrs.dtype.element_ty == tl.int8)
                        or (state_ptrs.dtype.element_ty == tl.int16),
                        "Quantized RN expects int8 or int16.",
                    )
                    state_q = tl.extra.cuda.libdevice.round(state_q)
                state_q = tl.minimum(tl.maximum(state_q, -QUANT_MAX), QUANT_MAX)
                tl.store(
                    state_ptrs, state_q.to(state_ptrs.dtype.element_ty), mask=state_mask
                )
        elif USE_RS_ROUNDING:
            tl.static_assert(
                state_ptrs.dtype.element_ty == tl.float16,
                "Non-quantized SR only supports fp16 state.",
            )
            tl.store(state_ptrs, _stochastic_round_fp16x2(state, rand), mask=state_mask)
        else:
            tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=state_mask)

    # Phase 2: Output
    x_ptr_local = x_ptr + x_seq_off + pid_h * stride_x_head
    C_ptr_local = C_ptr + C_seq_off + group_idx * stride_C_group
    out_ptr_local = out_ptr + out_seq_off + pid_h * stride_out_head

    # Write new x to double-buffered old_x at write position
    x_all = tl.load(
        x_ptr_local + offs_t[:, None] * stride_x_T + offs_m[None, :] * stride_x_dim,
        mask=t_mask[:, None] & m_mask[None, :],
        other=0.0,
    )
    # write_buf = (1 - active_buf) on write (staging), active_buf on no-write
    # (append) — resolved above alongside is_write.
    old_x_write_base = (
        old_x_ptr
        + cache_batch_idx * stride_old_x_cache
        + write_buf * stride_old_x_dbuf
        + pid_h * stride_old_x_head
    )
    tl.store(
        old_x_write_base
        + (write_offset + offs_t)[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        x_all,
        mask=t_mask[:, None] & m_mask[None, :],
    )
    x_all = x_all.to(tl.float32)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    C_all = tl.load(
        C_ptr_local + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    CB_scaled = tl.load(
        cb_scaled_ptr
        + pid_b * stride_cb_batch
        + pid_h * stride_cb_head
        + offs_t[:, None] * stride_cb_t
        + offs_t[None, :] * stride_cb_j,
        mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T),
        other=0.0,
    ).to(tl.float32)
    decay_vec = tl.load(
        decay_vec_ptr
        + pid_b * stride_dv_batch
        + pid_h * stride_dv_head
        + offs_t * stride_dv_t,
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)

    init_out = (
        tl.dot(C_all.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16))
        * decay_vec[:, None]
    )
    cb_out = tl.dot(CB_scaled.to(tl.bfloat16), x_all.to(tl.bfloat16))
    out_all = init_out + cb_out

    if HAS_D:
        D = tl.load(
            D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        out_all = out_all + x_all * D[None, :]

    if HAS_Z:
        z_ptr_local = z_ptr + z_seq_off + pid_h * stride_z_head
        for t in range(T):
            z_t = tl.load(
                z_ptr_local + t * stride_z_T + offs_m * stride_z_dim,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)
            out_t = tl.sum(tl.where((offs_t == t)[:, None], out_all, 0.0), axis=0)
            out_t = out_t * z_t * tl.sigmoid(z_t)
            # Varlen: guard the per-t store so t >= seq_len doesn't overwrite the
            # next packed sequence's output (folds to always-true non-varlen).
            tl.store(
                out_ptr_local + t * stride_out_T + offs_m * stride_out_dim,
                out_t,
                mask=m_mask & (t < seq_len),
            )
    else:
        tl.store(
            out_ptr_local
            + offs_t[:, None] * stride_out_T
            + offs_m[None, :] * stride_out_dim,
            out_all,
            mask=t_mask[:, None] & m_mask[None, :],
        )


@triton.jit()
def _persistent_rectangle_impl(
    # Per-work-unit indices resolved by the persistent wrapper
    pid_m,
    pid_b,
    pid_h,
    cache_batch_idx,
    active_buf,
    prev_num_accepted_tokens,
    # Pointers
    state_ptr,
    state_scales_ptr,
    old_x_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides (double-buffered)
    stride_old_x_cache,
    stride_old_x_dbuf,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides (T × window)
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    # Varlen: x/C/out packed (1, total_tokens, ...); batch-side offsets only.
    cu_seqlens_ptr=None,
    IS_VARLEN: tl.constexpr = False,
):
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_window = tl.arange(0, BLOCK_SIZE_K)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        x_seq_off = bos * stride_x_T
        C_seq_off = bos * stride_C_T
        out_seq_off = bos * stride_out_T
        z_seq_off = bos * stride_z_T
    else:
        seq_len = T
        x_seq_off = pid_b * stride_x_batch
        C_seq_off = pid_b * stride_C_batch
        out_seq_off = pid_b * stride_out_batch
        z_seq_off = pid_b * stride_z_batch
    t_mask = offs_t < seq_len

    # Window index masks
    is_history = offs_window < prev_num_accepted_tokens
    safe_hist_idx = tl.where(is_history, offs_window, 0)
    step_idx_fw = offs_window - prev_num_accepted_tokens
    is_step = (step_idx_fw >= 0) & (step_idx_fw < seq_len)
    safe_step_idx = tl.where(is_step, step_idx_fw, 0)

    # Load state (no replay on this path)
    state_ptr_local = (
        state_ptr + cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    )
    state_ptrs = (
        state_ptr_local
        + offs_m[:, None] * stride_state_dim
        + offs_n[None, :] * stride_state_dstate
    )
    state_mask = m_mask[:, None] & n_mask[None, :]
    state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    if QUANT_MAX > 0.0:
        decode_scale = tl.load(
            state_scales_ptr
            + cache_batch_idx * stride_state_scales_cache
            + pid_h * stride_state_scales_head
            + offs_m * stride_state_scales_dim,
            mask=m_mask,
            other=1.0,
        ).to(tl.float32)
        state = state * decode_scale[:, None]

    group_idx = pid_h // nheads_ngroups_ratio

    # Load history x before gdc_wait (doesn't depend on conv1d outputs)
    old_x_read_base = (
        old_x_ptr
        + cache_batch_idx * stride_old_x_cache
        + active_buf * stride_old_x_dbuf
        + pid_h * stride_old_x_head
    )
    history_x = tl.load(
        old_x_read_base
        + safe_hist_idx[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        mask=is_history[:, None] & m_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # Load C, step x
    C_tile = tl.load(
        C_ptr
        + C_seq_off
        + group_idx * stride_C_group
        + offs_t[:, None] * stride_C_T
        + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    x_ptr_local = x_ptr + x_seq_off + pid_h * stride_x_head
    step_x_in_window = tl.load(
        x_ptr_local
        + safe_step_idx[:, None] * stride_x_T
        + offs_m[None, :] * stride_x_dim,
        mask=is_step[:, None] & m_mask[None, :],
        other=0.0,
    )
    # Write new x tokens to active buffer (no-write mode: same buffer, no flip)
    tl.store(
        old_x_read_base
        + offs_window[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        step_x_in_window,
        mask=is_step[:, None] & m_mask[None, :],
    )

    x_window = history_x + step_x_in_window.to(tl.float32)

    if HAS_D or HAS_Z:
        step_in_window_selector = offs_t[:, None] == (
            offs_window[None, :] - prev_num_accepted_tokens
        )
        step_x = tl.dot(
            step_in_window_selector.to(tl.bfloat16), step_x_in_window.to(tl.bfloat16)
        )
    else:
        step_x = x_window  # unused placeholder

    CB_scaled = tl.load(
        cb_scaled_ptr
        + pid_b * stride_cb_batch
        + pid_h * stride_cb_head
        + offs_t[:, None] * stride_cb_t
        + offs_window[None, :] * stride_cb_j,
        mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_window[None, :] < BLOCK_SIZE_K),
        other=0.0,
    ).to(tl.float32)
    decay_vec = tl.load(
        decay_vec_ptr
        + pid_b * stride_dv_batch
        + pid_h * stride_dv_head
        + offs_t * stride_dv_t,
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)

    state_out = (
        tl.dot(C_tile.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16))
        * decay_vec[:, None]
    )
    if QUANT_MAX > 0.0:
        state_out = state_out * decode_scale[None, :]

    token_out = tl.dot(CB_scaled.to(tl.bfloat16), x_window.to(tl.bfloat16))
    output_tile = state_out + token_out

    if HAS_D:
        D = tl.load(
            D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        output_tile = output_tile + step_x.to(tl.float32) * D[None, :]

    out_ptr_local = out_ptr + out_seq_off + pid_h * stride_out_head
    if HAS_Z:
        z_ptr_local = z_ptr + z_seq_off + pid_h * stride_z_head
        z_tile = tl.load(
            z_ptr_local + offs_t[:, None] * stride_z_T + offs_m[None, :] * stride_z_dim,
            mask=t_mask[:, None] & m_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gated_output = output_tile * z_tile * tl.sigmoid(z_tile)
        tl.store(
            out_ptr_local
            + offs_t[:, None] * stride_out_T
            + offs_m[None, :] * stride_out_dim,
            gated_output,
            mask=t_mask[:, None] & m_mask[None, :],
        )
    else:
        tl.store(
            out_ptr_local
            + offs_t[:, None] * stride_out_T
            + offs_m[None, :] * stride_out_dim,
            output_tile,
            mask=t_mask[:, None] & m_mask[None, :],
        )


@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_CACHE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics({"USE_RS_ROUNDING": lambda args: args["rand_seed_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: max(
            triton.next_power_of_2(args["T"]), MIN_REPLAY_TILE_SIZE
        )
    }
)
@triton.heuristics(
    {
        "BLOCK_SIZE_WINDOW": lambda args: max(
            triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]),
            MIN_REPLAY_TILE_SIZE,
        )
    }
)
@triton.heuristics(
    {
        "BLOCK_SIZE_K": lambda args: max(
            triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]),
            MIN_REPLAY_TILE_SIZE,
        )
    }
)
@triton.heuristics(
    {"NUM_PID_M_BLOCKS": lambda args: triton.cdiv(args["dim"], args["BLOCK_SIZE_M"])}
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.jit()
def _persistent_main_kernel(
    # Pointers
    state_ptr,
    state_scales_ptr,
    old_x_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    rand_seed_ptr,
    state_batch_indices_ptr,
    cache_buf_idx_ptr,
    prev_num_accepted_tokens_ptr,
    n_writes_ptr,
    replay_work_items_ptr,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    batch_total,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides
    stride_old_x_cache,
    stride_old_x_dbuf,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_WINDOW: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_PID_M_BLOCKS: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    WRITE_CHECKPOINT: tl.constexpr,
    IS_DYNAMIC: tl.constexpr,
    RECTANGLE: tl.constexpr,
    NUM_PERSISTENT: tl.constexpr,
    # Varlen: cu_seqlens[0..N]; IS_VARLEN derived by the heuristic above.
    cu_seqlens_ptr,
    IS_VARLEN: tl.constexpr,
):
    # n_writes is only used when IS_DYNAMIC=False; Triton DCEs the load when IS_DYNAMIC=True.
    n_writes = tl.load(n_writes_ptr)

    if IS_DYNAMIC:
        slot_lo = 0
        slot_hi = batch_total
    else:
        if WRITE_CHECKPOINT:
            slot_lo = 0
            slot_hi = n_writes
        else:
            slot_lo = n_writes
            slot_hi = batch_total
    n_slots_local = slot_hi - slot_lo

    pid = tl.program_id(axis=0)
    total_work = n_slots_local * NUM_PID_M_BLOCKS * nheads

    for tile_id in tl.range(pid, total_work, NUM_PERSISTENT):
        pid_m = tile_id % NUM_PID_M_BLOCKS
        pid_b_local = (tile_id // NUM_PID_M_BLOCKS) % n_slots_local
        pid_h = tile_id // (NUM_PID_M_BLOCKS * n_slots_local)
        work_item_idx = pid_b_local + slot_lo

        if IS_DYNAMIC:
            pid_b = work_item_idx
            if HAS_CACHE_BATCH_INDICES:
                cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
            else:
                cache_batch_idx = pid_b.to(tl.int64)
            active_buf = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
            pnat = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
        else:
            work_item_base = (
                replay_work_items_ptr + work_item_idx * _REPLAY_WORK_ITEM_WIDTH
            )
            pid_b = tl.load(work_item_base + _REPLAY_WORK_POSITION_IN_DECODE_BATCH)
            cache_batch_idx = tl.load(work_item_base + _REPLAY_WORK_CACHE_SLOT).to(
                tl.int64
            )
            pnat = tl.load(work_item_base + _REPLAY_WORK_PNAT)
            active_buf = tl.load(work_item_base + _REPLAY_WORK_CACHE_BUF_IDX).to(
                tl.int32
            )

        # Varlen: per-work-item seq_len for the write decision (folds to T).
        if IS_VARLEN:
            _kbos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
            _keos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
            seq_len_w = (_keos - _kbos).to(tl.int32)
        else:
            seq_len_w = T

        if RECTANGLE:
            if IS_DYNAMIC:
                is_w = (pnat + seq_len_w) > MAX_REPLAY_BUFFER_LENGTH
            else:
                is_w = WRITE_CHECKPOINT
            if is_w:
                _persistent_main_impl(
                    pid_m,
                    pid_b,
                    pid_h,
                    cache_batch_idx,
                    active_buf,
                    pnat,
                    state_ptr,
                    state_scales_ptr,
                    old_x_ptr,
                    old_B_ptr,
                    old_dt_ptr,
                    old_dA_cumsum_ptr,
                    x_ptr,
                    C_ptr,
                    D_ptr,
                    z_ptr,
                    out_ptr,
                    cb_scaled_ptr,
                    decay_vec_ptr,
                    rand_seed_ptr,
                    T,
                    MAX_REPLAY_BUFFER_LENGTH,
                    dim,
                    dstate,
                    nheads_ngroups_ratio,
                    stride_state_batch,
                    stride_state_head,
                    stride_state_dim,
                    stride_state_dstate,
                    stride_state_scales_cache,
                    stride_state_scales_head,
                    stride_state_scales_dim,
                    stride_old_x_cache,
                    stride_old_x_dbuf,
                    stride_old_x_T,
                    stride_old_x_head,
                    stride_old_x_dim,
                    stride_old_B_cache,
                    stride_old_B_dbuf,
                    stride_old_B_T,
                    stride_old_B_group,
                    stride_old_B_dstate,
                    stride_old_dt_cache,
                    stride_old_dt_dbuf,
                    stride_old_dt_head,
                    stride_old_dt_T,
                    stride_old_dA_cumsum_cache,
                    stride_old_dA_cumsum_dbuf,
                    stride_old_dA_cumsum_head,
                    stride_old_dA_cumsum_T,
                    stride_x_batch,
                    stride_x_T,
                    stride_x_head,
                    stride_x_dim,
                    stride_C_batch,
                    stride_C_T,
                    stride_C_group,
                    stride_C_dstate,
                    stride_D_head,
                    stride_D_dim,
                    stride_z_batch,
                    stride_z_T,
                    stride_z_head,
                    stride_z_dim,
                    stride_out_batch,
                    stride_out_T,
                    stride_out_head,
                    stride_out_dim,
                    stride_cb_batch,
                    stride_cb_head,
                    stride_cb_t,
                    stride_cb_j,
                    stride_dv_batch,
                    stride_dv_head,
                    stride_dv_t,
                    BLOCK_SIZE_M,
                    HAS_D,
                    HAS_Z,
                    HAS_CACHE_BATCH_INDICES,
                    BLOCK_SIZE_DSTATE,
                    BLOCK_SIZE_T,
                    BLOCK_SIZE_WINDOW,
                    LAUNCH_WITH_PDL,
                    USE_RS_ROUNDING,
                    PHILOX_ROUNDS,
                    QUANT_MAX,
                    True,  # WRITE_CHECKPOINT
                    IS_DYNAMIC,
                    True,  # WRITE_CHECKPOINT_IS_CONSTEXPR: write arm pre-narrowed
                    cu_seqlens_ptr,
                    IS_VARLEN,
                )
            else:
                _persistent_rectangle_impl(
                    pid_m,
                    pid_b,
                    pid_h,
                    cache_batch_idx,
                    active_buf,
                    pnat,
                    state_ptr,
                    state_scales_ptr,
                    old_x_ptr,
                    x_ptr,
                    C_ptr,
                    D_ptr,
                    z_ptr,
                    out_ptr,
                    cb_scaled_ptr,
                    decay_vec_ptr,
                    T,
                    MAX_REPLAY_BUFFER_LENGTH,
                    dim,
                    dstate,
                    nheads_ngroups_ratio,
                    stride_state_batch,
                    stride_state_head,
                    stride_state_dim,
                    stride_state_dstate,
                    stride_state_scales_cache,
                    stride_state_scales_head,
                    stride_state_scales_dim,
                    stride_old_x_cache,
                    stride_old_x_dbuf,
                    stride_old_x_T,
                    stride_old_x_head,
                    stride_old_x_dim,
                    stride_x_batch,
                    stride_x_T,
                    stride_x_head,
                    stride_x_dim,
                    stride_C_batch,
                    stride_C_T,
                    stride_C_group,
                    stride_C_dstate,
                    stride_D_head,
                    stride_D_dim,
                    stride_z_batch,
                    stride_z_T,
                    stride_z_head,
                    stride_z_dim,
                    stride_out_batch,
                    stride_out_T,
                    stride_out_head,
                    stride_out_dim,
                    stride_cb_batch,
                    stride_cb_head,
                    stride_cb_t,
                    stride_cb_j,
                    stride_dv_batch,
                    stride_dv_head,
                    stride_dv_t,
                    BLOCK_SIZE_M,
                    HAS_D,
                    HAS_Z,
                    HAS_CACHE_BATCH_INDICES,
                    BLOCK_SIZE_DSTATE,
                    BLOCK_SIZE_T,
                    BLOCK_SIZE_K,
                    LAUNCH_WITH_PDL,
                    QUANT_MAX,
                    cu_seqlens_ptr,
                    IS_VARLEN,
                )
        else:
            _persistent_main_impl(
                pid_m,
                pid_b,
                pid_h,
                cache_batch_idx,
                active_buf,
                pnat,
                state_ptr,
                state_scales_ptr,
                old_x_ptr,
                old_B_ptr,
                old_dt_ptr,
                old_dA_cumsum_ptr,
                x_ptr,
                C_ptr,
                D_ptr,
                z_ptr,
                out_ptr,
                cb_scaled_ptr,
                decay_vec_ptr,
                rand_seed_ptr,
                T,
                MAX_REPLAY_BUFFER_LENGTH,
                dim,
                dstate,
                nheads_ngroups_ratio,
                stride_state_batch,
                stride_state_head,
                stride_state_dim,
                stride_state_dstate,
                stride_state_scales_cache,
                stride_state_scales_head,
                stride_state_scales_dim,
                stride_old_x_cache,
                stride_old_x_dbuf,
                stride_old_x_T,
                stride_old_x_head,
                stride_old_x_dim,
                stride_old_B_cache,
                stride_old_B_dbuf,
                stride_old_B_T,
                stride_old_B_group,
                stride_old_B_dstate,
                stride_old_dt_cache,
                stride_old_dt_dbuf,
                stride_old_dt_head,
                stride_old_dt_T,
                stride_old_dA_cumsum_cache,
                stride_old_dA_cumsum_dbuf,
                stride_old_dA_cumsum_head,
                stride_old_dA_cumsum_T,
                stride_x_batch,
                stride_x_T,
                stride_x_head,
                stride_x_dim,
                stride_C_batch,
                stride_C_T,
                stride_C_group,
                stride_C_dstate,
                stride_D_head,
                stride_D_dim,
                stride_z_batch,
                stride_z_T,
                stride_z_head,
                stride_z_dim,
                stride_out_batch,
                stride_out_T,
                stride_out_head,
                stride_out_dim,
                stride_cb_batch,
                stride_cb_head,
                stride_cb_t,
                stride_cb_j,
                stride_dv_batch,
                stride_dv_head,
                stride_dv_t,
                BLOCK_SIZE_M,
                HAS_D,
                HAS_Z,
                HAS_CACHE_BATCH_INDICES,
                BLOCK_SIZE_DSTATE,
                BLOCK_SIZE_T,
                BLOCK_SIZE_WINDOW,
                LAUNCH_WITH_PDL,
                USE_RS_ROUNDING,
                PHILOX_ROUNDS,
                QUANT_MAX,
                WRITE_CHECKPOINT,
                IS_DYNAMIC,
                False,  # WRITE_CHECKPOINT_IS_CONSTEXPR: dynamic resolves at runtime
                cu_seqlens_ptr,
                IS_VARLEN,
            )


# Python wrapper


_QUANT_MAX_BY_DTYPE = {
    torch.int8: 127.0,
    torch.int16: 32767.0,
    torch.float8_e4m3fn: 448.0,
}


def checkpointing_state_update(
    state: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt: torch.Tensor,
    old_dA_cumsum: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    rand_seed: torch.Tensor | None = None,
    philox_rounds: int = 10,
    state_scales: torch.Tensor | None = None,
    launch_with_pdl=False,
    use_internal_pdl=True,
    write_checkpoint: bool = True,
    rectangle: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    _block_size_m: int | None = None,
    _num_warps: int | None = None,
    _num_stages: int | None = None,
    _precompute_num_warps: int | None = None,
    _precompute_num_stages: int | None = None,
    _heads_per_block: int | None = None,
    _maxnreg: int | None = None,
    _num_ctas: int | None = None,
):
    """
    Replay SSM state update with precomputed CB and tl.dot fast-forward.

    Two-kernel architecture:
      1. Precompute kernel: computes CB_scaled and decay_vec from B, C, dt, A.
         Writes processed dt/dA_cumsum/B to double-buffered cache for next step.
      2. Main kernel: replays old tokens via tl.dot fast-forward on cached data,
         then computes output using precomputed CB_scaled and new x/C inputs.

    PDL (Programmatic Dependent Launch) chain:
      conv1d → (external PDL) → precompute → (internal PDL) → main
      External PDL: precompute starts while conv1d is running; gdc_wait()
        in precompute blocks until conv1d completes before loading B/C.
      Internal PDL: main starts while precompute is running; main's replay
        phase uses only cached data from the previous step.  gdc_wait() in
        main blocks until precompute completes before loading conv1d outputs
        (x, C) and precompute outputs (CB_scaled, decay_vec).

    Uses double-buffered cache tensors.  cache_buf_idx[slot] indicates which
    buffer (0 or 1) to READ from for replay.  The WRITE buffer is 1 - read.
    Caller must flip cache_buf_idx[slot] after each call.

    Arguments:
        state: (cache, nheads, dim, dstate) in-place.  After the call, contains
            the state after replaying prev_num_accepted_tokens old tokens.
        old_x: (cache, T, nheads, dim) bf16 — old x cache (single-buffered).
        old_B: (cache, 2, T, ngroups, dstate) bf16 — double-buffered old B cache.
        old_dt: (cache, 2, nheads, T) fp32 — double-buffered processed dt.
        old_dA_cumsum: (cache, 2, nheads, T) fp32 — double-buffered cumulative A*dt.
        cache_buf_idx: (cache,) int32 — which buffer to read (0 or 1).
        prev_num_accepted_tokens: (cache,) int32.
        x: (batch, T, nheads, dim) new token inputs.
        dt: (batch, T, nheads, dim) with stride(-1)==0 (tie_hdim).
        A: (nheads, dim, dstate) with stride(-1)==0, stride(-2)==0 (tie_hdim).
        B: (batch, T, ngroups, dstate).
        C: (batch, T, ngroups, dstate).
        out: (batch, T, nheads, dim) preallocated output.
        D: (nheads, dim) optional feed-through parameter.
        z: (batch, T, nheads, dim) optional silu gate.
        dt_bias: (nheads, dim) optional, with stride(-1)==0 (tie_hdim).
        state_batch_indices: (batch,) optional cache slot mapping.
        rand_seed: optional single-element int64 CUDA tensor for Philox PRNG seed.
            When provided, state is stochastically rounded on store.  Supported
            for state.dtype in (fp16, int8, int16, fp8_e4m3fn); other dtypes
            silently use deterministic rounding.  fp16+SR and fp8+SR both
            require sm_100a (Blackwell B200+) — wrapper asserts this loudly.
        philox_rounds: number of Philox PRNG rounds (default 10).
        state_scales: required when state.dtype in (int8, int16, fp8_e4m3fn).
            Shape (cache_size, nheads, dim), fp32.  Per-(head, dim) channel
            decode scale (= 1 / encode_scale).  The kernel writes scales on
            checkpoint steps and reads them on load (broadcast over dstate).
            Ignored for non-quantized state dtypes.
        launch_with_pdl: enable external PDL (conv1d → precompute chain).
            Defaults False; caller opts in when the upstream chain is PDL-safe.
            Ignored on hardware that doesn't support PDL (sm < 90).
        use_internal_pdl: enable internal PDL (precompute → main overlap).
            Defaults True; override for testing only.
            Ignored on hardware that doesn't support PDL (sm < 90).

        _-prefixed kwargs (_block_size_m, _num_warps, _num_stages,
        _precompute_num_warps, _precompute_num_stages, _heads_per_block,
        _maxnreg, _num_ctas) are benchmark-only overrides; production callers
        should leave them None to use the heuristic-tuned defaults.
    """
    # PDL needs sm >= 90.
    if get_sm_version() < 90:
        launch_with_pdl = False
        use_internal_pdl = False

    # Constexpr modes:
    #   write_checkpoint=True,  rectangle=False  → checkpoint step (default).
    #   write_checkpoint=False, rectangle=False  → non-checkpoint step (skip state HBM write).
    #   write_checkpoint=False, rectangle=True   → reserved for the rectangle non-checkpoint
    #                                               optimization path (not wired yet).
    #   write_checkpoint=True,  rectangle=True   → not supported.
    if rectangle:
        raise NotImplementedError(
            "RECTANGLE path is not wired yet; pass rectangle=False."
        )
    assert not (write_checkpoint and rectangle), (
        "WRITE_CHECKPOINT and RECTANGLE are mutually exclusive."
    )

    # --- Hardware support gates ---
    # fp8 e4m3fn (any rounding mode) needs SM 89+ for the fp32↔e4m3 cvt PTX
    # instructions (Ada Lovelace introduced them; Hopper/Blackwell carry them).
    if state.dtype == torch.float8_e4m3fn:
        assert get_sm_version() >= 89, (
            "fp8_e4m3fn state requires SM 89+ (Ada Lovelace / Hopper / Blackwell) "
            f"for fp32↔fp8 cvt PTX instructions; current SM is {get_sm_version()}."
        )

    # PTX cvt.rs.* (stochastic rounding) family lands on Blackwell only.
    # Wrapper fails loud; framework decides fall-back (e.g. drop SR, use RN).
    # int8 / int16 SR uses pure-Triton libdevice.floor + uniform noise — no
    # PTX SR instruction needed, runs anywhere.
    if rand_seed is not None:
        if state.dtype == torch.float16:
            assert get_sm_version() >= 100, (
                "fp16 stochastic rounding (PTX cvt.rs.f16x2.f32) requires "
                f"sm_100a (Blackwell B200+); current SM is {get_sm_version()}."
            )
        elif state.dtype == torch.float8_e4m3fn:
            assert get_sm_version() >= 100, (
                "fp8 stochastic rounding (PTX cvt.rs.satfinite.e4m3x4.f32) "
                f"requires sm_100a (Blackwell B200+); current SM is {get_sm_version()}."
            )

    # --- Unsqueeze inputs to canonical shapes ---
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None:
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if z.dim() == 3:
            z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if out.dim() == 3:
        out = out.unsqueeze(1)

    cache_size, nheads, dim, dstate = state.shape
    # Varlen: inputs are packed (1, total_tokens, nheads, dim) and
    # cu_seqlens[0..N] partitions them into N sequences.  T (the JIT key,
    # = max_seqlen) is the upper bound on every cu_seqlens[i+1] - cu_seqlens[i].
    # In non-varlen mode batch comes from x.shape[0], T from x.shape[1].
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert x.dim() == 4 and x.shape[0] == 1, (
            f"varlen: x must be (1, total_tokens, nheads, dim), got shape "
            f"{tuple(x.shape)}"
        )
        assert cu_seqlens.dim() == 1 and cu_seqlens.dtype == torch.int32, (
            f"cu_seqlens must be 1D int32, got shape {tuple(cu_seqlens.shape)} "
            f"dtype {cu_seqlens.dtype}"
        )
        assert cu_seqlens.is_cuda, "cu_seqlens must be a CUDA tensor"
        batch = cu_seqlens.shape[0] - 1
        T = max_seqlen if max_seqlen is not None else old_x.shape[1]
        # In varlen mode the wrapper's runtime shape check on x/dt/B/C is
        # against (1, total_tokens, ...); the strides we hand the kernels
        # are the per-token strides (stride(1)).  The kernels switch outer
        # offset to `bos * stride_*_T` under IS_VARLEN.
    else:
        assert max_seqlen is None, (
            "max_seqlen is only valid with cu_seqlens (varlen mode)"
        )
        batch, T, _, _ = x.shape
    ngroups = B.shape[2]
    assert nheads % ngroups == 0

    # --- Quantization plumbing ---
    # QUANT_MAX > 0 ⇔ state is int8 / int16 / fp8_e4m3fn.  Kernel-entry
    # static_assert on the Triton side mirrors this invariant.
    quant_max = _QUANT_MAX_BY_DTYPE.get(state.dtype, 0.0)
    is_quantized = quant_max > 0.0
    if is_quantized:
        assert state_scales is not None, (
            f"state.dtype={state.dtype} requires state_scales tensor "
            "(shape (cache_size, nheads, dim), fp32)."
        )
        assert state_scales.shape == (cache_size, nheads, dim), (
            f"state_scales shape mismatch: expected {(cache_size, nheads, dim)}, "
            f"got {state_scales.shape}."
        )
        assert state_scales.dtype == torch.float32, (
            f"state_scales must be fp32, got {state_scales.dtype}."
        )
        assert state_scales.device == state.device

    # Cache T-axis = MAX_WINDOW (the replay buffer capacity).  For the
    # placeholder degenerate case max_window = T (every step is a checkpoint
    # step).  For real replay-style checkpointing, max_window > T and
    # `prev_num_accepted_tokens` can be 0..max_window.
    max_window = old_x.shape[1]
    assert max_window >= T, f"T={T} exceeds cache max_window={max_window}"
    # Replay-style code path uses BLOCK_SIZE_T = max(np2(T), 16) for the
    # combined T-axis (T_new tile size) and reuses it for window loads.  Until
    # the heuristic is generalized to track max_window separately, require
    # max_window to fit within that tile.
    block_size_t = max(triton.next_power_of_2(T), 16)
    assert max_window <= block_size_t, (
        f"max_window={max_window} exceeds BLOCK_SIZE_T={block_size_t} "
        f"derived from T={T}; extend the heuristic to include max_window."
    )

    if is_varlen:
        # Packed (1, total_tokens, ...) — total_tokens already validated by
        # cu_seqlens; the kernel slices per-sequence via bos.  Inner dims
        # (nheads, dim) and (ngroups, dstate) match the non-varlen layout.
        assert x.shape[2:] == (nheads, dim)
        assert dt.shape == x.shape
        assert B.shape[2:] == (ngroups, dstate)
        assert C.shape == B.shape
    else:
        assert x.shape == (batch, T, nheads, dim)
        assert dt.shape == x.shape
        assert B.shape == (batch, T, ngroups, dstate)
        assert C.shape == B.shape
    assert A.shape == (nheads, dim, dstate)
    assert old_x.shape == (cache_size, max_window, nheads, dim)
    assert old_B.shape == (cache_size, 2, max_window, ngroups, dstate)
    assert old_dt.shape == (cache_size, 2, nheads, max_window)
    assert old_dA_cumsum.shape == (cache_size, 2, nheads, max_window)
    assert cache_buf_idx.shape == (cache_size,)
    assert prev_num_accepted_tokens.shape == (cache_size,)

    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and (dt_bias is None or dt_bias.stride(-1) == 0)
    )
    assert tie_hdim

    device = x.device
    BLOCK_SIZE_T = max(triton.next_power_of_2(T), 16)

    # Allocate precomputed intermediates (per-call, not cached)
    cb_scaled = torch.empty(
        batch, nheads, BLOCK_SIZE_T, BLOCK_SIZE_T, device=device, dtype=torch.float32
    )
    decay_vec = torch.empty(
        batch, nheads, BLOCK_SIZE_T, device=device, dtype=torch.float32
    )

    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))
        if z is not None
        else (0, 0, 0, 0)
    )

    # Kernel tuning: BLOCK_SIZE_M, num_warps, HEADS_PER_BLOCK, precompute_num_warps.
    # Dtype-aware heuristic from B200 sweeps (batch 1-512, T=6/32, TP=8, conv1d +
    # chained PDL).  Keyed on total_heads, BLOCK_SIZE_T, and state dtype; 16-bit
    # states prefer different tiles from fp32 due to lower bandwidth.  Philox
    # gets its own branch — stochastic rounding shifts compute toward CUDA cores,
    # so small-batch configs want more warps to hide the extra work.
    total_heads = batch * nheads
    heads_per_group = nheads // ngroups
    state_is_16bit = state.dtype in (torch.float16, torch.bfloat16)
    use_philox = rand_seed is not None
    if BLOCK_SIZE_T <= 16:
        if use_philox and state_is_16bit:
            # Philox: more warps at small batch to hide CUDA core work.
            # At large batch, converges to non-Philox fp16 config.
            if total_heads <= 16:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    4,
                    4,
                    4,
                    1,
                )
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    8,
                    1,
                    2,
                    1,
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    4,
                    2,
                    1,
                )
        elif state_is_16bit:
            if total_heads <= 16:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    4,
                    4,
                    1,
                )
            elif total_heads <= 64:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    8,
                    1,
                    2,
                    1,
                )
            elif total_heads <= 256:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    16,
                    2,
                    2,
                    1,
                )
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    1,
                    1,
                    min(2, heads_per_group),
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    4,
                    2,
                    1,
                )
        else:  # fp32 state (no Philox — fp32 doesn't need stochastic rounding)
            if total_heads <= 32:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    8,
                    1,
                    4,
                    1,
                )
            elif total_heads <= 64:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    8,
                    1,
                    2,
                    1,
                )
            elif total_heads <= 128:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    8,
                    2,
                    2,
                    1,
                )
            elif total_heads <= 256:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    16,
                    1,
                    2,
                    1,
                )
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    64,
                    2,
                    2,
                    min(2, heads_per_group),
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    4,
                    2,
                    1,
                )
    else:  # T > 16
        if state_is_16bit:
            if total_heads <= 128:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    16,
                    2,
                    4,
                    1,
                )
            elif total_heads <= 256:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    16,
                    1,
                    4,
                    min(2, heads_per_group),
                )
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    1,
                    1,
                    min(4, heads_per_group),
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    1,
                    4,
                    min(2, heads_per_group),
                )
        else:  # fp32 state
            if total_heads <= 128:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    16,
                    2,
                    4,
                    1,
                )
            elif total_heads <= 256:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    2,
                    4,
                    min(2, heads_per_group),
                )
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    64,
                    2,
                    2,
                    min(4, heads_per_group),
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    64,
                    2,
                    4,
                    min(2, heads_per_group),
                )
    if _block_size_m is not None:
        BLOCK_SIZE_M = _block_size_m
    if _num_warps is not None:
        num_warps = _num_warps
    if _heads_per_block is not None:
        heads_per_block = _heads_per_block
    if _precompute_num_warps is not None:
        precompute_num_warps = _precompute_num_warps

    HAS_CACHE_BATCH_INDICES = state_batch_indices is not None

    # Outer iteration stride for the batch-side tensors (dt, B, C, x, z, out).
    # The kernel reads `seq * stride_*_batch` where seq is either pid_b
    # (non-varlen) or cu_seqlens[pid_b] (varlen).  In varlen we therefore
    # need the per-TOKEN stride here (stride(1)), not the per-batch
    # stride (stride(0)) — outer iteration walks tokens, not batches.
    if is_varlen:
        dt_outer_stride = dt.stride(1)
        B_outer_stride = B.stride(1)
        C_outer_stride = C.stride(1)
        x_outer_stride = x.stride(1)
        z_outer_stride = z.stride(1) if z is not None else 0
        out_outer_stride = out.stride(1)
    else:
        dt_outer_stride = dt.stride(0)
        B_outer_stride = B.stride(0)
        C_outer_stride = C.stride(0)
        x_outer_stride = x.stride(0)
        z_outer_stride = z.stride(0) if z is not None else 0
        out_outer_stride = out.stride(0)

    with torch.cuda.device(device.index):
        # --- Precompute kernel ---
        assert nheads % heads_per_block == 0, (
            f"nheads ({nheads}) must be divisible by heads_per_block ({heads_per_block})"
        )
        assert heads_per_block <= heads_per_group, (
            f"heads_per_block ({heads_per_block}) must not cross group boundary ({heads_per_group})"
        )
        _checkpointing_precompute_kernel[(batch, nheads // heads_per_block)](
            dt,
            dt_bias,
            A,
            B,
            C,
            cu_seqlens,
            cb_scaled,
            decay_vec,
            old_B,
            old_dt,
            old_dA_cumsum,
            cache_buf_idx,
            prev_num_accepted_tokens,
            state_batch_indices,
            pad_slot_id,
            T,
            dstate,
            nheads // ngroups,
            # dt strides (outer = batch if non-varlen, per-token if varlen)
            dt_outer_stride,
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            # B strides
            B_outer_stride,
            B.stride(1),
            B.stride(2),
            B.stride(3),
            # C strides
            C_outer_stride,
            C.stride(1),
            C.stride(2),
            C.stride(3),
            # cb_scaled strides
            cb_scaled.stride(0),
            cb_scaled.stride(1),
            cb_scaled.stride(2),
            cb_scaled.stride(3),
            # decay_vec strides
            decay_vec.stride(0),
            decay_vec.stride(1),
            decay_vec.stride(2),
            # old_B strides
            old_B.stride(0),
            old_B.stride(1),
            old_B.stride(2),
            old_B.stride(3),
            old_B.stride(4),
            # old_dt strides
            old_dt.stride(0),
            old_dt.stride(1),
            old_dt.stride(2),
            old_dt.stride(3),
            # old_dA_cumsum strides
            old_dA_cumsum.stride(0),
            old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2),
            old_dA_cumsum.stride(3),
            dt_softplus,
            HAS_CACHE_BATCH_INDICES=HAS_CACHE_BATCH_INDICES,
            LAUNCH_WITH_PDL=launch_with_pdl,
            LAUNCH_DEPENDENT_KERNELS=use_internal_pdl,
            HEADS_PER_BLOCK=heads_per_block,
            WRITE_CHECKPOINT=write_checkpoint,
            num_warps=precompute_num_warps,
            **(
                {"num_stages": _precompute_num_stages} if _precompute_num_stages else {}
            ),
            launch_pdl=launch_with_pdl,
        )

        # --- Main kernel ---
        def grid(META):
            return (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)

        # state_scales pointer + strides: real tensor when quantized, otherwise
        # zero-strided dummy (kernel never reads it because QUANT_MAX==0.0).
        if is_quantized:
            state_scales_arg = state_scales
            state_scales_strides = (
                state_scales.stride(0),
                state_scales.stride(1),
                state_scales.stride(2),
            )
        else:
            state_scales_arg = state  # any valid ptr — gated by QUANT_MAX==0
            state_scales_strides = (0, 0, 0)

        _checkpointing_main_kernel[grid](
            state,
            state_scales_arg,
            old_x,
            old_B,
            old_dt,
            old_dA_cumsum,
            prev_num_accepted_tokens,
            cache_buf_idx,
            x,
            C,
            D,
            z,
            out,
            cb_scaled,
            decay_vec,
            state_batch_indices,
            cu_seqlens,
            rand_seed,
            pad_slot_id,
            T,
            dim,
            dstate,
            nheads // ngroups,
            # state strides
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            # state_scales strides (cache, head, dim)
            state_scales_strides[0],
            state_scales_strides[1],
            state_scales_strides[2],
            # old_x strides (single-buffered: cache, T, nheads, dim)
            old_x.stride(0),
            old_x.stride(1),
            old_x.stride(2),
            old_x.stride(3),
            # old_B strides
            old_B.stride(0),
            old_B.stride(1),
            old_B.stride(2),
            old_B.stride(3),
            old_B.stride(4),
            # old_dt strides
            old_dt.stride(0),
            old_dt.stride(1),
            old_dt.stride(2),
            old_dt.stride(3),
            # old_dA_cumsum strides
            old_dA_cumsum.stride(0),
            old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2),
            old_dA_cumsum.stride(3),
            # x strides (outer = batch if non-varlen, per-token if varlen)
            x_outer_stride,
            x.stride(1),
            x.stride(2),
            x.stride(3),
            # C strides
            C_outer_stride,
            C.stride(1),
            C.stride(2),
            C.stride(3),
            # D strides
            *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
            # z strides
            z_outer_stride,
            z_strides[1],
            z_strides[2],
            z_strides[3],
            # out strides
            out_outer_stride,
            out.stride(1),
            out.stride(2),
            out.stride(3),
            # cb_scaled strides
            cb_scaled.stride(0),
            cb_scaled.stride(1),
            cb_scaled.stride(2),
            cb_scaled.stride(3),
            # decay_vec strides
            decay_vec.stride(0),
            decay_vec.stride(1),
            decay_vec.stride(2),
            BLOCK_SIZE_M,
            LAUNCH_WITH_PDL=use_internal_pdl,
            PHILOX_ROUNDS=philox_rounds if rand_seed is not None else 0,
            QUANT_MAX=quant_max,
            WRITE_CHECKPOINT=write_checkpoint,
            RECTANGLE=rectangle,
            num_warps=num_warps,
            **({"num_stages": _num_stages} if _num_stages else {}),
            **({"num_ctas": _num_ctas} if _num_ctas else {}),
            **({"maxnreg": _maxnreg} if _maxnreg else {}),
            launch_pdl=use_internal_pdl,
        )


# ── Default tuning table (simplified — persistent_dynamic mode only) ──────────

# Maps (split_knob_name) → (pm_write_key, pm_nowrite_key) for bridging between
# persistent_dynamic (unified) and persistent_main (split) knob namespaces.
_PD_TO_PM_SPLIT_MAP = {
    "_block_size_m": ("_block_size_m_write", "_block_size_m_nowrite"),
    "_num_warps": ("_num_warps_write", "_num_warps_nowrite"),
    "_num_stages": ("_num_stages_write", "_num_stages_nowrite"),
    "_cta_per_sm": ("_cta_per_sm_write", "_cta_per_sm_nowrite"),
    "_num_loop_stages": ("_num_loop_stages_write", "_num_loop_stages_nowrite"),
}

# Tuning table: (dtype_str, sr_str) → [(batch_threshold, mode, knobs), ...]
# Each entry covers batch <= threshold.  Last entry is the fallback for large batches.
# We support persistent_dynamic only for now (IS_DYNAMIC=True).
_DEFAULT_TUNING: dict = {
    ("fp32", "RN"): [
        (
            1024,
            "persistent_dynamic",
            {
                "_block_size_m": 16,
                "_cta_per_sm": 4,
                "_flatten": False,
                "_heads_per_block": 1,
                "_num_loop_stages": 2,
                "_num_stages": 2,
                "_num_warps": 4,
                "_precompute_num_warps": 4,
                "_use_tma_rect_load": False,
                "_use_tma_replay_nowrite_load": False,
                "_use_tma_replay_write_load": False,
                "_use_tma_replay_write_store": False,
                "_warp_specialize": False,
                "rectangle_for_nowrite": False,
                "nowrite_first": False,
            },
        ),
    ],
    ("fp16", "RN"): [
        (
            1024,
            "persistent_dynamic",
            {
                "_block_size_m": 32,
                "_cta_per_sm": 4,
                "_flatten": False,
                "_heads_per_block": 1,
                "_num_loop_stages": 2,
                "_num_stages": 2,
                "_num_warps": 4,
                "_precompute_num_warps": 4,
                "_use_tma_rect_load": False,
                "_use_tma_replay_nowrite_load": False,
                "_use_tma_replay_write_load": False,
                "_use_tma_replay_write_store": False,
                "_warp_specialize": False,
                "rectangle_for_nowrite": False,
                "nowrite_first": False,
            },
        ),
    ],
    ("bf16", "RN"): [
        (
            1024,
            "persistent_dynamic",
            {
                "_block_size_m": 32,
                "_cta_per_sm": 4,
                "_flatten": False,
                "_heads_per_block": 1,
                "_num_loop_stages": 2,
                "_num_stages": 2,
                "_num_warps": 4,
                "_precompute_num_warps": 4,
                "_use_tma_rect_load": False,
                "_use_tma_replay_nowrite_load": False,
                "_use_tma_replay_write_load": False,
                "_use_tma_replay_write_store": False,
                "_warp_specialize": False,
                "rectangle_for_nowrite": False,
                "nowrite_first": False,
            },
        ),
    ],
    ("int8", "RN"): [
        (
            1024,
            "persistent_dynamic",
            {
                "_block_size_m": 32,
                "_cta_per_sm": 4,
                "_flatten": False,
                "_heads_per_block": 1,
                "_num_loop_stages": 2,
                "_num_stages": 2,
                "_num_warps": 4,
                "_precompute_num_warps": 4,
                "_use_tma_rect_load": False,
                "_use_tma_replay_nowrite_load": False,
                "_use_tma_replay_write_load": False,
                "_use_tma_replay_write_store": False,
                "_warp_specialize": False,
                "rectangle_for_nowrite": False,
                "nowrite_first": False,
            },
        ),
    ],
    ("int16", "RN"): [
        (
            1024,
            "persistent_dynamic",
            {
                "_block_size_m": 32,
                "_cta_per_sm": 4,
                "_flatten": False,
                "_heads_per_block": 1,
                "_num_loop_stages": 2,
                "_num_stages": 2,
                "_num_warps": 4,
                "_precompute_num_warps": 4,
                "_use_tma_rect_load": False,
                "_use_tma_replay_nowrite_load": False,
                "_use_tma_replay_write_load": False,
                "_use_tma_replay_write_store": False,
                "_warp_specialize": False,
                "rectangle_for_nowrite": False,
                "nowrite_first": False,
            },
        ),
    ],
    ("fp8", "RN"): [
        (
            1024,
            "persistent_dynamic",
            {
                "_block_size_m": 32,
                "_cta_per_sm": 4,
                "_flatten": False,
                "_heads_per_block": 1,
                "_num_loop_stages": 2,
                "_num_stages": 2,
                "_num_warps": 4,
                "_precompute_num_warps": 4,
                "_use_tma_rect_load": False,
                "_use_tma_replay_nowrite_load": False,
                "_use_tma_replay_write_load": False,
                "_use_tma_replay_write_store": False,
                "_warp_specialize": False,
                "rectangle_for_nowrite": False,
                "nowrite_first": False,
            },
        ),
    ],
    # SR variants — copy of RN for now (SR adds Philox, same tile shape is fine)
    ("fp16", "SR"): [
        (
            1024,
            "persistent_dynamic",
            {
                "_block_size_m": 32,
                "_cta_per_sm": 4,
                "_flatten": False,
                "_heads_per_block": 1,
                "_num_loop_stages": 2,
                "_num_stages": 2,
                "_num_warps": 4,
                "_precompute_num_warps": 4,
                "_use_tma_rect_load": False,
                "_use_tma_replay_nowrite_load": False,
                "_use_tma_replay_write_load": False,
                "_use_tma_replay_write_store": False,
                "_warp_specialize": False,
                "rectangle_for_nowrite": False,
                "nowrite_first": False,
            },
        ),
    ],
    ("int8", "SR"): [
        (
            1024,
            "persistent_dynamic",
            {
                "_block_size_m": 32,
                "_cta_per_sm": 4,
                "_flatten": False,
                "_heads_per_block": 1,
                "_num_loop_stages": 2,
                "_num_stages": 2,
                "_num_warps": 4,
                "_precompute_num_warps": 4,
                "_use_tma_rect_load": False,
                "_use_tma_replay_nowrite_load": False,
                "_use_tma_replay_write_load": False,
                "_use_tma_replay_write_store": False,
                "_warp_specialize": False,
                "rectangle_for_nowrite": False,
                "nowrite_first": False,
            },
        ),
    ],
}


def _bridge_tuning_knobs(knobs: dict, from_mode: str, to_mode: str) -> dict:
    """Convert a tuning dict between persistent_dynamic ↔ persistent_main knob namespaces."""
    out = dict(knobs)
    if from_mode == "persistent_dynamic" and to_mode == "persistent_main":
        for unsplit, (pm_w, pm_nw) in _PD_TO_PM_SPLIT_MAP.items():
            if unsplit in out:
                out.setdefault(pm_w, out[unsplit])
                out.setdefault(pm_nw, out[unsplit])
                del out[unsplit]
    elif from_mode == "persistent_main" and to_mode == "persistent_dynamic":
        for unsplit, (pm_w, pm_nw) in _PD_TO_PM_SPLIT_MAP.items():
            if pm_nw in out:
                out.setdefault(unsplit, out[pm_nw])
                out.pop(pm_w, None)
                out.pop(pm_nw, None)
    return out


def _resolve_tuning(batch: int, nheads: int, dt_str: str, sr_str: str):
    """Look up the default mode + knobs for this (eff_batch, dt, sr) cell.

    Returns (mode, knobs_dict) or None if the table has no entry covering this
    dtype/sr after fallbacks.
    """
    eff_b = batch * max(1, nheads)
    valid_dtypes = {"fp32", "fp16", "bf16", "int8", "int16", "fp8"}
    if dt_str not in valid_dtypes:
        raise ValueError(
            f"replay_selective_state_update: unsupported state dtype {dt_str!r}; "
            f"expected one of {sorted(valid_dtypes)}"
        )
    keys_to_try = [(dt_str, sr_str)]
    if sr_str == "RN":
        keys_to_try.append((dt_str, "SR"))
    if dt_str in ("bf16", "int16"):
        keys_to_try.append(("fp16", "SR"))
    elif dt_str == "fp8":
        keys_to_try.append(("int8", "SR"))
    entries = None
    for k in keys_to_try:
        if k in _DEFAULT_TUNING:
            entries = _DEFAULT_TUNING[k]
            break
    if entries is None:
        return None
    for thresh, mode, knobs in entries:
        if eff_b <= thresh:
            return mode, dict(knobs)
    thresh, mode, knobs = entries[-1]
    return mode, dict(knobs)


def replay_selective_state_update(
    state: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt: torch.Tensor,
    old_dA_cumsum: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    rand_seed: torch.Tensor | None = None,
    philox_rounds: int = 10,
    state_scales: torch.Tensor | None = None,
    launch_with_pdl: bool = False,
    use_internal_pdl: bool = True,
    write_checkpoint: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    rectangle_for_nowrite: bool = False,
    mode: str | None = None,
    _block_size_m: int | None = None,
    _num_warps: int | None = None,
    _num_stages: int | None = None,
    _precompute_num_warps: int | None = None,
    _precompute_num_stages: int | None = None,
    _heads_per_block: int | None = None,
    _maxnreg: int | None = None,
    _num_ctas: int | None = None,
    _cta_per_sm: int | None = None,
    _num_loop_stages: int | None = None,
    _flatten: bool | None = None,
    _warp_specialize: bool | None = None,
):
    """
    Replay SSM state update with new persistent-kernel architecture.

    When old_x is 4-D (single-buffered legacy shape), routes to
    checkpointing_state_update for backward compatibility.

    When old_x is 5-D (double-buffered: cache, 2, max_window, nheads, dim),
    uses the new persistent IS_DYNAMIC kernel path.
    """
    # ── Backward compat: single-buffered old_x → old kernel path ──────────────
    if old_x.dim() == 4:
        return checkpointing_state_update(
            state=state,
            old_x=old_x,
            old_B=old_B,
            old_dt=old_dt,
            old_dA_cumsum=old_dA_cumsum,
            cache_buf_idx=cache_buf_idx,
            prev_num_accepted_tokens=prev_num_accepted_tokens,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            out=out,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            pad_slot_id=pad_slot_id,
            rand_seed=rand_seed,
            philox_rounds=philox_rounds,
            state_scales=state_scales,
            launch_with_pdl=launch_with_pdl,
            use_internal_pdl=use_internal_pdl,
            write_checkpoint=write_checkpoint,
            rectangle=rectangle_for_nowrite,
        )

    # ── New double-buffered path ───────────────────────────────────────────────
    assert old_x.dim() == 5, (
        f"old_x must be 4-D (legacy) or 5-D (double-buffered), got {old_x.dim()}-D"
    )

    sm_version = _get_sm_version()
    if sm_version < 90:
        launch_with_pdl = False
        use_internal_pdl = False

    if state.dtype == torch.float8_e4m3fn:
        assert sm_version >= 89, (
            "fp8_e4m3fn state requires SM 89+ for fp32↔fp8 cvt PTX instructions; "
            f"current SM is {sm_version}."
        )
    if rand_seed is not None:
        if state.dtype == torch.float16:
            assert sm_version >= 100, (
                "fp16 stochastic rounding (PTX cvt.rs.f16x2.f32) requires "
                f"sm_100a (Blackwell B200+); current SM is {sm_version}."
            )
        elif state.dtype == torch.float8_e4m3fn:
            assert sm_version >= 100, (
                "fp8 stochastic rounding requires sm_100a (Blackwell B200+); "
                f"current SM is {sm_version}."
            )
        else:
            assert state.dtype in (torch.int8, torch.int16), (
                "stochastic rounding is supported only for state.dtype in "
                f"(fp16, int8, int16, fp8_e4m3fn), got {state.dtype}."
            )

    # Unsqueeze to canonical (batch, T, nheads, dim) shapes
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 3:
        out = out.unsqueeze(1)

    cache_size, nheads, dim, dstate = state.shape
    # Varlen: x/dt/B/C/out are packed (1, total_tokens, ...); cu_seqlens[0..N]
    # partitions them into N sequences.  batch = N, T = max_seqlen.  The cache
    # stays per-slot (double-buffered) — varlen is a batch-side layout only.
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert x.dim() == 4 and x.shape[0] == 1, (
            f"varlen: x must be (1, total_tokens, nheads, dim), got {tuple(x.shape)}"
        )
        assert cu_seqlens.dim() == 1 and cu_seqlens.dtype == torch.int32, (
            f"cu_seqlens must be 1D int32, got shape {tuple(cu_seqlens.shape)} "
            f"dtype {cu_seqlens.dtype}"
        )
        assert cu_seqlens.is_cuda, "cu_seqlens must be a CUDA tensor"
        batch = cu_seqlens.shape[0] - 1
        T = max_seqlen if max_seqlen is not None else old_x.shape[2]
    else:
        assert max_seqlen is None, (
            "max_seqlen is only valid with cu_seqlens (varlen mode)"
        )
        batch, T, _, _ = x.shape
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, (
        f"nheads ({nheads}) must be divisible by ngroups ({ngroups})"
    )

    max_window = old_x.shape[2]
    assert max_window >= T, f"T={T} exceeds cache max_window={max_window}"
    assert old_x.shape == (cache_size, 2, max_window, nheads, dim), (
        f"old_x shape mismatch: expected {(cache_size, 2, max_window, nheads, dim)}, "
        f"got {tuple(old_x.shape)}"
    )
    assert old_B.shape == (cache_size, 2, max_window, ngroups, dstate), (
        f"old_B shape mismatch: expected {(cache_size, 2, max_window, ngroups, dstate)}, "
        f"got {tuple(old_B.shape)}"
    )
    assert old_dt.shape == (cache_size, 2, nheads, max_window), (
        f"old_dt shape mismatch: expected {(cache_size, 2, nheads, max_window)}, "
        f"got {tuple(old_dt.shape)}"
    )
    assert old_dA_cumsum.shape == (cache_size, 2, nheads, max_window), (
        f"old_dA_cumsum shape mismatch: expected {(cache_size, 2, nheads, max_window)}, "
        f"got {tuple(old_dA_cumsum.shape)}"
    )
    assert cache_buf_idx.shape == (cache_size,), (
        f"cache_buf_idx shape mismatch: expected ({cache_size},), got {tuple(cache_buf_idx.shape)}"
    )
    assert cache_buf_idx.dtype == torch.int32, (
        f"cache_buf_idx must be int32, got {cache_buf_idx.dtype}"
    )
    assert prev_num_accepted_tokens.shape == (cache_size,), (
        f"prev_num_accepted_tokens shape mismatch: expected ({cache_size},), "
        f"got {tuple(prev_num_accepted_tokens.shape)}"
    )
    assert prev_num_accepted_tokens.dtype == torch.int32, (
        f"prev_num_accepted_tokens must be int32, got {prev_num_accepted_tokens.dtype}"
    )

    quant_max = _QUANT_MAX_BY_DTYPE.get(state.dtype, 0.0)
    is_quantized = quant_max > 0.0
    if is_quantized:
        assert state_scales is not None, (
            f"state.dtype={state.dtype} requires state_scales tensor "
            "(shape (cache_size, nheads, dim), fp32)."
        )
        assert state_scales.shape == (cache_size, nheads, dim), (
            f"state_scales shape mismatch: expected {(cache_size, nheads, dim)}, "
            f"got {tuple(state_scales.shape)}"
        )
        assert state_scales.dtype == torch.float32, (
            f"state_scales must be fp32, got {state_scales.dtype}"
        )

    if rand_seed is not None:
        assert rand_seed.dtype == torch.int64, (
            f"rand_seed dtype must be int64, got {rand_seed.dtype}"
        )
        if rand_seed.shape[0] == 1 and cache_size > 1:
            rand_seed = rand_seed.expand(cache_size).contiguous()
        assert rand_seed.shape[0] >= cache_size, (
            f"rand_seed must have length 1 or >= cache_size ({cache_size}); "
            f"got shape {tuple(rand_seed.shape)}"
        )

    # Default-tuning lookup
    _dt_str = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.float8_e4m3fn: "fp8",
    }.get(state.dtype, str(state.dtype))
    _sr_str = "SR" if rand_seed is not None else "RN"
    _table_entry = _resolve_tuning(batch, nheads, _dt_str, _sr_str)
    if _table_entry is None:
        raise ValueError(
            "replay_selective_state_update has no default tuning for "
            f"state dtype {_dt_str!r} with rounding mode {_sr_str!r}."
        )
    _table_mode, _table_knobs = _table_entry
    if mode is None:
        mode = _table_mode
    # Only persistent_dynamic supported here; persistent_main requires sorted work_items.
    assert mode == "persistent_dynamic", (
        f"Only 'persistent_dynamic' mode is supported in this simplified wrapper; "
        f"got mode={mode!r}"
    )

    # Fill None-valued kwargs from table
    _block_size_m = (
        _block_size_m
        if _block_size_m is not None
        else _table_knobs.get("_block_size_m")
    )
    _num_warps = (
        _num_warps if _num_warps is not None else _table_knobs.get("_num_warps")
    )
    _num_stages = (
        _num_stages if _num_stages is not None else _table_knobs.get("_num_stages")
    )
    _heads_per_block = (
        _heads_per_block
        if _heads_per_block is not None
        else _table_knobs.get("_heads_per_block")
    )
    _precompute_num_warps = (
        _precompute_num_warps
        if _precompute_num_warps is not None
        else _table_knobs.get("_precompute_num_warps")
    )
    _cta_per_sm = (
        _cta_per_sm if _cta_per_sm is not None else _table_knobs.get("_cta_per_sm", 4)
    )
    _num_loop_stages = (
        _num_loop_stages
        if _num_loop_stages is not None
        else _table_knobs.get("_num_loop_stages", 2)
    )
    _flatten = _flatten if _flatten is not None else _table_knobs.get("_flatten", False)
    if rectangle_for_nowrite is False and "rectangle_for_nowrite" in _table_knobs:
        rectangle_for_nowrite = bool(_table_knobs["rectangle_for_nowrite"])

    assert _block_size_m is not None, (
        "replay_selective_state_update requires _block_size_m tuning"
    )
    assert _num_warps is not None, (
        "replay_selective_state_update requires _num_warps tuning"
    )
    assert _heads_per_block is not None, (
        "replay_selective_state_update requires _heads_per_block tuning"
    )
    assert _precompute_num_warps is not None, (
        "replay_selective_state_update requires _precompute_num_warps tuning"
    )

    device = x.device
    BLOCK_SIZE_M = _block_size_m
    BLOCK_SIZE_T = max(triton.next_power_of_2(T), MIN_REPLAY_TILE_SIZE)
    BLOCK_SIZE_K = max(triton.next_power_of_2(max_window), MIN_REPLAY_TILE_SIZE)
    rectangle_use_gather = BLOCK_SIZE_T == BLOCK_SIZE_K

    heads_per_group = nheads // ngroups
    heads_per_block = int(_heads_per_block)
    heads_per_block = min(heads_per_block, heads_per_group)
    while (
        heads_per_group % heads_per_block != 0
        or heads_per_block & (heads_per_block - 1) != 0
    ):
        heads_per_block -= 1
    heads_per_block = max(heads_per_block, 1)

    # Precomputed intermediates
    cb_scaled = torch.empty(
        batch, nheads, BLOCK_SIZE_T, BLOCK_SIZE_K, device=device, dtype=torch.float32
    )
    decay_vec = torch.empty(
        batch, nheads, BLOCK_SIZE_T, device=device, dtype=torch.float32
    )

    # Dummy tensors for IS_DYNAMIC=True path (kernel DCEs these loads)
    n_writes = torch.zeros(1, dtype=torch.int32, device=device)
    replay_work_items = torch.zeros(
        batch, _REPLAY_WORK_ITEM_WIDTH, dtype=torch.int32, device=device
    )

    HAS_CACHE_BATCH_INDICES = state_batch_indices is not None

    # state_scales plumbing
    if is_quantized:
        state_scales_arg = state_scales
        state_scales_strides = (
            state_scales.stride(0),
            state_scales.stride(1),
            state_scales.stride(2),
        )
    else:
        state_scales_arg = state  # dummy pointer, gated by QUANT_MAX==0
        state_scales_strides = (0, 0, 0)

    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))
        if z is not None
        else (0, 0, 0, 0)
    )
    d_strides = (D.stride(0), D.stride(1)) if D is not None else (0, 0)

    _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_persistent = (_cta_per_sm if _cta_per_sm else 1) * _num_sms
    precomp_grid = (batch, nheads // heads_per_block)
    _num_pid_m = (dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    total_work_launch = max(1, batch * _num_pid_m * nheads)
    grid = (min(num_persistent, total_work_launch),)

    with torch.cuda.device(device.index):
        # Step 1: Precompute CB_scaled and decay_vec
        _dynamic_precompute_kernel[precomp_grid](
            dt,
            dt_bias,
            A,
            B,
            C,
            cb_scaled,
            decay_vec,
            old_B,
            old_dt,
            old_dA_cumsum,
            cache_buf_idx,
            prev_num_accepted_tokens,
            state_batch_indices,
            T,
            max_window,
            dstate,
            nheads // ngroups,
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            cb_scaled.stride(0),
            cb_scaled.stride(1),
            cb_scaled.stride(2),
            cb_scaled.stride(3),
            decay_vec.stride(0),
            decay_vec.stride(1),
            decay_vec.stride(2),
            old_B.stride(0),
            old_B.stride(1),
            old_B.stride(2),
            old_B.stride(3),
            old_B.stride(4),
            old_dt.stride(0),
            old_dt.stride(1),
            old_dt.stride(2),
            old_dt.stride(3),
            old_dA_cumsum.stride(0),
            old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2),
            old_dA_cumsum.stride(3),
            dt_softplus,
            HAS_CACHE_BATCH_INDICES=HAS_CACHE_BATCH_INDICES,
            LAUNCH_WITH_PDL=launch_with_pdl,
            LAUNCH_DEPENDENT_KERNELS=use_internal_pdl,
            HEADS_PER_BLOCK=heads_per_block,
            RECTANGLE=rectangle_for_nowrite,
            RECTANGLE_USE_GATHER=rectangle_use_gather,
            cu_seqlens_ptr=cu_seqlens,
            num_warps=_precompute_num_warps,
            **(
                {"num_stages": _precompute_num_stages} if _precompute_num_stages else {}
            ),
            launch_pdl=launch_with_pdl,
        )

        # Step 2: Persistent main kernel (IS_DYNAMIC=True — per-slot runtime dispatch)
        _persistent_main_kernel[grid](
            state,
            state_scales_arg,
            old_x,
            old_B,
            old_dt,
            old_dA_cumsum,
            x,
            C,
            D,
            z,
            out,
            cb_scaled,
            decay_vec,
            rand_seed,
            state_batch_indices,
            cache_buf_idx,
            prev_num_accepted_tokens,
            n_writes,
            replay_work_items,
            T,
            max_window,
            batch,
            dim,
            dstate,
            nheads,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            state_scales_strides[0],
            state_scales_strides[1],
            state_scales_strides[2],
            old_x.stride(0),
            old_x.stride(1),
            old_x.stride(2),
            old_x.stride(3),
            old_x.stride(4),
            old_B.stride(0),
            old_B.stride(1),
            old_B.stride(2),
            old_B.stride(3),
            old_B.stride(4),
            old_dt.stride(0),
            old_dt.stride(1),
            old_dt.stride(2),
            old_dt.stride(3),
            old_dA_cumsum.stride(0),
            old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2),
            old_dA_cumsum.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            d_strides[0],
            d_strides[1],
            z_strides[0],
            z_strides[1],
            z_strides[2],
            z_strides[3],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            cb_scaled.stride(0),
            cb_scaled.stride(1),
            cb_scaled.stride(2),
            cb_scaled.stride(3),
            decay_vec.stride(0),
            decay_vec.stride(1),
            decay_vec.stride(2),
            BLOCK_SIZE_M,
            LAUNCH_WITH_PDL=use_internal_pdl,
            PHILOX_ROUNDS=philox_rounds if rand_seed is not None else 0,
            QUANT_MAX=quant_max,
            WRITE_CHECKPOINT=False,  # unused when IS_DYNAMIC=True
            IS_DYNAMIC=True,
            RECTANGLE=rectangle_for_nowrite,
            cu_seqlens_ptr=cu_seqlens,
            NUM_PERSISTENT=num_persistent,
            num_warps=_num_warps,
            **({"num_stages": _num_stages} if _num_stages else {}),
            **({"num_ctas": _num_ctas} if _num_ctas else {}),
            **({"maxnreg": _maxnreg} if _maxnreg else {}),
            launch_pdl=use_internal_pdl,
        )
