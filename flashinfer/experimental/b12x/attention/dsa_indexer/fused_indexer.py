"""Unified FUSED indexer kernel: score q·k AND select top-k in one launch.

Design (see plan): a row-cooperative persistent kernel where each CTA streams a
slice of one query row's K, scores q·k via ``mxfp8`` MMA into shared memory
(m = heads / MQA), folds each scored sub-tile into an in-SMEM running top-k via
the ``tiled_topk`` radix (coarse-8-bit-histogram + byte-refine), and CTAs merge
in turn. Logits never reach global memory — peak per-row working set is
O(topk), not O(K). contiguous K/V vs paged K/V is a constexpr K-loader
specialization; the score assembly and top-k are identical (always m = heads).

This module is the SCAFFOLD: launch config, dispatch gates, scratch trio, and
the kernel class shell. The fused score→select kernel body lands per the phased
plan (Phase 1 = paged; Phase 2 = CONTIGUOUS_MLA loader; Phase 3 = topk 512 /
paged_output). The m=heads score helpers are reused from
``b12x/attention/indexer/kernel.py`` (``_compute_mxfp8_tile_partials`` et al.)
and the radix select / virtual-index loaders from
``b12x/attention/indexer/tiled_topk.py``; nothing is reinvented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import os

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import (
    KernelCompileSpec,
    launch as b12x_launch,
    tensor_compile_fact,
)
from b12x._lib.intrinsics import shared_ptr_to_u32
from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from b12x._lib.utils import current_cuda_stream

# Kept in lock-step with the source kernels we fuse.
from b12x.attention.dsa_indexer._tuning import (
    FUSED_MERGE_AUTO,
    FUSED_MERGE_CHOICES,
    FUSED_MERGE_COOPERATIVE,
    FUSED_MERGE_SERIAL,
)
from b12x.attention.dsa_indexer.persistent_topk import (
    _RADIX,
    _RADIX_THRESHOLD,
)
from b12x.attention.dsa_indexer.tiled_topk import (
    _SCAN_UNROLL,
    _SUPPORTED_TOPK,
    _THREADS_PER_CTA as _RADIX_THREADS,
    _convert_to_uint8,
    _convert_to_uint32,
    _exact_overflow_fallback,
    _smem_ld,
    _smem_red_add,
    _smem_st,
    _smem_xadd,
)
from b12x._lib.intrinsics import ld_shared_f32
from b12x.attention.dsa_indexer.kernel import (
    _to_kernel_tensor,
    _stream_issue_k_page_cp_async,
    _INDEX_HEAD_DIM,
    _PAGE_SIZE,
    _PAGED_Q_HEAD_TILE,
    _PAGED_THREADS_PER_CTA,
    _PAGED_TOKENS_PER_GROUP,
    _PAGED_WARPS_PER_CTA,
    _WARP_THREADS,
    _compute_mxfp8_tile_partials,
    _compute_mxfp8_tile_partials_qldm,
    _load_index_k_page_scalar,
    _num_q_head_tiles,
    _permuted_offset_128b,
    _reduce_column_pair_sum,
    _repack_k_page_to_permuted,
    _smem_addr_from_b128_offset,
)
from b12x._lib.intrinsics import (
    atomic_add_global_i32,
    atomic_add_global_u64,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_acquire_i32,
    ld_global_cg_v4_u32,
    ld_global_nc_f32,
    ld_global_nc_u32,
    ld_global_v4_u32,
    red_add_global_i32,
    red_add_global_release_i32,
    spin_wait_global_ge_i32,
    ldmatrix_m8n8x4_b16,
    ld_shared_v4_u32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    st_shared_u8,
    st_shared_v4_u32,
    threadfence,
)

# Triage kill-switch: B12X_INDEXER_DIRECT_K=0 forces every variant back to
# the staged pipeline (which also restores the v2 cache keys), so serving can
# A/B the direct-L2 score in place without a checkout change.
_DIRECT_K_SCORE_ENABLED = os.environ.get("B12X_INDEXER_DIRECT_K", "1").lower() not in {
    "0",
    "false",
    "no",
    "",
}

_THREADS_PER_CTA = 1024
# Per-CTA K-slice cap (cooperative split granularity). Matches persistent_topk's
# chunking knob; the fused kernel streams this slice through MMA sub-tiles rather
# than materializing it.
_MAX_CHUNK_ELEMENTS = 8192

# Constexpr K/V-layout specialization (the ONLY contiguous-vs-paged divergence in the
# fused kernel — everything downstream of the SMEM staging tile is shared).
KV_LAYOUT_CONTIGUOUS_MLA = 0
KV_LAYOUT_PAGED = 1

# Route metadata is capture-static; live seqlen is deliberately not a policy input,
# so vLLM graph replay cannot switch backends. The generic policy retains its
# measured small-row limit. C4 uses top-k 512 with either 64 heads (DeepSeek V4)
# or 32 heads (GLM-5.3-Flash), while GLM-5.2 uses 32 heads and top-k 2048.
# L2-flushed DSV4F measurements select fused through B16 for C4 on both
# RTX-class SM120 and GB10/SM121.
# Prefill remains packed-contiguous and never reaches this decode-only resolver.
FUSED_MAX_ROWS = 6
_C4_B12X_FUSED_MAX_ROWS = 16
# GLM re-measured after the two-level tiled fold landed: fused keeps rows 1-16
# (wins/ties at 8K, dominant at 32K-131K, e.g. r1@131K 41us vs 90us tiled) but
# loses rows 32-64 everywhere (r64@131K 1541us vs 1043us tiled, r64@8K 100 vs
# 78) -- the 2-CTAs-per-group merge and one-resident-CTA occupancy starve it.
_GLM32_FUSED_MAX_ROWS = 16
FUSED_MIN_WIDTH = 20480  # (capacity-sizing only; no longer a routing gate)


@cute.jit
def _load_flat_k_tile_scalar(
    k_quant_flat: cute.Tensor,  # [k_rows, 128] uint8
    abs_base: Int32,
    valid_rows: Int32,
    s_k_page_stage: cute.Tensor,
    lane_linear: Int32,
):
    """CONTIGUOUS_MLA K-load: stage a 64-row tile of contiguous K starting at abs_base.

    Rows >= valid_rows are zero-filled (so the MMA contributes 0 for them — masked
    out anyway by the staging-write valid_slots guard).
    """
    linear = lane_linear
    total = Int32(_PAGE_SIZE * _INDEX_HEAD_DIM)
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM)
        col = linear - row * Int32(_INDEX_HEAD_DIM)
        b = cutlass.Uint8(0)
        if row < valid_rows:
            b = k_quant_flat[abs_base + row, col]
        s_k_page_stage[row, col, Int32(0)] = b
        linear += Int32(_PAGED_THREADS_PER_CTA)


# --- wide (all-thread) page loaders ------------------------------------------
# The per-page K load dominates the fused per-page cost. The kernel.py scalar
# loaders stride by _PAGED_THREADS_PER_CTA (128), so a 8 KB page is 64 sequential
# coalesced rounds. These widen the stride to n_threads (up to 1024) so the page
# loads in ~8 rounds. Same coalesced byte pattern (consecutive tx -> consecutive
# col), just 8x more threads in flight to hide latency.
@cute.jit
def _load_index_k_page_wide(
    k_quant_bytes: cute.Tensor,  # [pages, 64, 128] uint8
    page_id: Int32,
    s_k_page_stage: cute.Tensor,
    tx: Int32,
    n_threads: Int32,
):
    linear = tx
    total = Int32(_PAGE_SIZE * _INDEX_HEAD_DIM)
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM)
        col = linear - row * Int32(_INDEX_HEAD_DIM)
        s_k_page_stage[row, col, Int32(0)] = k_quant_bytes[page_id, row, col]
        linear += n_threads


@cute.jit
def _load_flat_k_tile_wide(
    k_quant_flat: cute.Tensor,  # [k_rows, 128] uint8
    abs_base: Int32,
    valid_rows: Int32,
    s_k_page_stage: cute.Tensor,
    tx: Int32,
    n_threads: Int32,
):
    linear = tx
    total = Int32(_PAGE_SIZE * _INDEX_HEAD_DIM)
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM)
        col = linear - row * Int32(_INDEX_HEAD_DIM)
        b = cutlass.Uint8(0)
        if row < valid_rows:
            b = k_quant_flat[abs_base + row, col]
        s_k_page_stage[row, col, Int32(0)] = b
        linear += n_threads


@cute.jit
def _load_permute_k_page_g2s(
    k_quant_bytes: cute.Tensor,  # [pages, 64, 128] uint8 (16B-aligned base)
    page_id: Int32,
    page_stride_bytes: Int64,  # dim-0 byte stride; 8192 for contiguous K, 8448 for
    # the packed paged cache (64*128 quant + 64*4 scales / page)
    k_perm_base_addr: Int32,
    tx: Int32,
    n_threads: Int32,
):
    """paged fused global->shared load + permute: read 16-byte K granules straight
    from global into the ldmatrix-permuted SMEM layout. Replaces the scalar
    linear load + separate SMEM->SMEM repack pass (and its barrier) with one
    vectorized pass. Granule addresses are 16B-aligned (page*page_stride_bytes +
    row*128 + vec*16); page_stride_bytes MUST be the real dim-0 stride of
    k_quant_bytes (the packed paged cache interleaves per-page scales, so its page
    stride is 8448, not the contiguous 8192) — using a hardcoded stride reads the
    wrong page for the packed layout. It stays 16B-aligned (8448 = 16*528).
    """
    page_elem_base = Int64(page_id) * page_stride_bytes
    linear = tx
    total = Int32(_PAGE_SIZE * (_INDEX_HEAD_DIM // 16))
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM // 16)
        vec_idx = linear - row * Int32(_INDEX_HEAD_DIM // 16)
        g_addr = get_ptr_as_int64(
            k_quant_bytes,
            page_elem_base
            + Int64(row) * Int64(_INDEX_HEAD_DIM)
            + Int64(vec_idx) * Int64(16),
        )
        v0, v1, v2, v3 = ld_global_v4_u32(g_addr)
        dst_addr = _smem_addr_from_b128_offset(
            k_perm_base_addr,
            _permuted_offset_128b(row, vec_idx, Int32(_INDEX_HEAD_DIM // 16)),
        )
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += n_threads


@cute.jit
def _repack_k_page_wide(
    k_linear_base_addr: Int32,
    k_perm_base_addr: Int32,
    tx: Int32,
    n_threads: Int32,
):
    linear = tx
    total = Int32(_PAGE_SIZE * (_INDEX_HEAD_DIM // 16))
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM // 16)
        vec_idx = linear - row * Int32(_INDEX_HEAD_DIM // 16)
        src_addr = k_linear_base_addr + Int32(row * _INDEX_HEAD_DIM + vec_idx * 16)
        dst_addr = _smem_addr_from_b128_offset(
            k_perm_base_addr,
            _permuted_offset_128b(row, vec_idx, Int32(_INDEX_HEAD_DIM // 16)),
        )
        v0, v1, v2, v3 = ld_shared_v4_u32(src_addr)
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += n_threads


@dataclass(frozen=True)
class _FusedLaunchConfig:
    chunk_size: int
    ctas_per_group: int
    num_groups: int
    total_ctas: int


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _resolve_fused_launch_config(
    num_rows: int, width: int, device: torch.device
) -> _FusedLaunchConfig:
    """Row-cooperative split — cloned from persistent_topk._resolve_launch_config.

    One group owns one query row; ``ctas_per_group`` CTAs split the row's K of
    ``width`` tokens; groups persist grid-stride over rows.
    """
    max_chunk = max(_MAX_CHUNK_ELEMENTS, _THREADS_PER_CTA)
    ctas_per_group = max(1, (width + max_chunk - 1) // max_chunk)
    chunk_size = min(
        max_chunk, _align_up((width + ctas_per_group - 1) // ctas_per_group, 1)
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_groups = min(max(num_rows, 1), max(1, num_sms // ctas_per_group))
    total_ctas = max(1, num_groups * ctas_per_group)
    return _FusedLaunchConfig(
        chunk_size=chunk_size,
        ctas_per_group=ctas_per_group,
        num_groups=num_groups,
        total_ctas=total_ctas,
    )


# --- scratch sizing (per-group combine state + inter-CTA carry slab) ----------
# Combine state per group: arrival + output counters + a small relay slot. The
# inter-CTA carry slab holds each CTA's published running top-k (value f32 +
# index i32) for the next CTA in the relay to fold.
_STATE_WORDS_PER_GROUP = 4  # arrival_counter, output_counter, relay_phase, pad

# Cross-CTA merge auto-switch (ctas_per_group>1): a group whose live row K length
# seq_len <= merge_threshold uses the serial last-CTA reduction (no grid
# barriers); larger rows use the cooperative grid-barrier radix. Both arms are
# exact, so the threshold only trades perf. Planned launches resolve the
# threshold through _resolve_default_merge_threshold; this value is the
# constructor default for direct kernel constructions.
_LAST_CTA_MERGE_MAX = 49152
# Sentinel "never coop" threshold: larger than any live seq_len, so the per-group
# auto-switch always takes the serial last-CTA arm (single-CTA groups, and
# groups that cannot be co-resident).
_FORCE_LAST_CTA = 1 << 30
# K stages of the paged score pipeline (one 64-token page each).
_K_STAGES = 3

# Per-group cross-CTA merge state (int32 words): the cooperative radix
# selects the pivot over three digits of the order-preserving 32-bit key
# (12, 12, and 8 bits), each round accumulating the group's digit histogram
# into its own buffer, followed by the scalars both merge arms share. The
# arrival counter backs the grid barrier, the output counter reserves winner
# slots (and the pack-slab offset on the last-CTA arm), the tie counter
# reserves slots at the pivot, and the total/cleanup words carry the group's
# candidate count and its end-of-launch departure count.
# A 12-bit round publishes both a 256-bin coarse histogram (top 8 bits of
# the digit) and a 4096-bin fine histogram, so the pivot scan reads the
# coarse buffer and then only the 16 fine bins under the coarse pivot.
_FUSED_WIDE_RADIX = 4096
_FUSED_COARSE0 = 0
_FUSED_FINE0 = _FUSED_COARSE0 + _RADIX
_FUSED_COARSE1 = _FUSED_FINE0 + _FUSED_WIDE_RADIX
_FUSED_FINE1 = _FUSED_COARSE1 + _RADIX
_FUSED_COARSE2 = _FUSED_FINE1 + _FUSED_WIDE_RADIX
_FUSED_STATE_ARRIVAL = _FUSED_COARSE2 + _RADIX
_FUSED_STATE_TOTAL = _FUSED_STATE_ARRIVAL + 1
# Output (winner) and tie counters are adjacent and 8-byte aligned so one
# 64-bit fetch-add reserves both slot ranges.
_FUSED_STATE_OUTPUT = _FUSED_STATE_ARRIVAL + 2
_FUSED_STATE_TIES = _FUSED_STATE_ARRIVAL + 3
_FUSED_STATE_CLEANUP = _FUSED_STATE_ARRIVAL + 4
_COOP_STATE_WORDS = _FUSED_STATE_ARRIVAL + 8
assert (_FUSED_STATE_OUTPUT % 2 == 0) and (_COOP_STATE_WORDS % 2 == 0)
assert _FUSED_WIDE_RADIX == 4 * _RADIX_THREADS


@cute.jit
def _fused_state_offset(group_id: Int32, word: Int32) -> Int32:
    return group_id * Int32(_COOP_STATE_WORDS) + word


@cute.jit
def _fused_state_ptr(state: cute.Tensor, group_id: Int32, word: Int32):
    return get_ptr_as_int64(state, _fused_state_offset(group_id, word))


@cute.jit
def _fused_group_barrier(
    state: cute.Tensor, group_id: Int32, phase: Int32, ctas_per_group: Int32, tx: Int32
) -> Int32:
    """Grid barrier over the group's CTAs on the arrival counter; returns the next phase."""
    arrival_ptr = _fused_state_ptr(state, group_id, Int32(_FUSED_STATE_ARRIVAL))
    if tx == Int32(0):
        red_add_global_release_i32(arrival_ptr, Int32(1))
        spin_wait_global_ge_i32(arrival_ptr, (phase + Int32(1)) * ctas_per_group)
    cute.arch.sync_threads()
    return phase + Int32(1)


@cute.jit
def _page_id_or_invalid(
    real_page_table: cute.Tensor, q_idx: Int32, col: Int32, page_end: Int32
) -> Int32:
    """Page id of page-table column ``col`` of row ``q_idx``, or -1 past the slice."""
    pid = Int32(-1)
    if col < page_end:
        if col < Int32(real_page_table.shape[1]):
            pid = Int32(real_page_table[q_idx, col])
    return pid


@cute.jit
def _k_stage_addr(stage: Int32, stage0: Int32, stage1: Int32, stage2: Int32) -> Int32:
    """Shared-memory base of K ring stage ``stage`` (0, 1, or 2)."""
    addr = stage0
    if stage == Int32(1):
        addr = stage1
    if stage == Int32(2):
        addr = stage2
    return addr


@cute.jit
def _coop_publish_wide(
    merge_state: cute.Tensor,
    group_id: Int32,
    hist_off: cutlass.Constexpr[int],
    hist_addr: Int32,
    tx: Int32,
):
    """Add this CTA's 4096-bin local histogram (thread ``t`` owns bins
    ``4t .. 4t+3``) into the group's global buffer and clear it."""
    c0, c1, c2, c3 = ld_shared_v4_u32(hist_addr + tx * Int32(16))
    base = _fused_state_ptr(merge_state, group_id, Int32(hist_off) + tx * Int32(4))
    if c0 != Uint32(0):
        red_add_global_i32(base, Int32(c0))
    if c1 != Uint32(0):
        red_add_global_i32(base + Int64(4), Int32(c1))
    if c2 != Uint32(0):
        red_add_global_i32(base + Int64(8), Int32(c2))
    if c3 != Uint32(0):
        red_add_global_i32(base + Int64(12), Int32(c3))
    st_shared_v4_u32(
        hist_addr + tx * Int32(16), Uint32(0), Uint32(0), Uint32(0), Uint32(0)
    )


@cute.jit
def _coop_wide_round(
    merge_state: cute.Tensor,
    group_id: Int32,
    coarse_off: cutlass.Constexpr[int],
    fine_off: cutlass.Constexpr[int],
    s_c0_values: cute.Tensor,
    carry_count: Int32,
    key_mask: Uint32,
    key_prefix: Uint32,
    digit_shift: cutlass.Constexpr[int],
    wide_hist_addr: Int32,
    coarse_hist_addr: Int32,
    s_hist0: cute.Tensor,
    remaining_k: Int32,
    barrier_phase: Int32,
    ctas_pg: Int32,
    tx: Int32,
    lane: Int32,
    warp_idx: Int32,
    s_coop_scalars: cute.Tensor,
    read_total: cutlass.Constexpr[bool],
) -> Int32:
    """One 12-bit cooperative radix round; returns the next barrier phase.

    Every candidate whose key matches ``key_prefix`` under ``key_mask``
    contributes the 12-bit digit at ``digit_shift`` to the CTA's fine
    (4096-bin, in the K ring) and coarse (256-bin, in ``s_hist0``) local
    histograms, both of which must be zero on entry and are left zero. The
    local histograms are published to the group buffers at ``fine_off`` and
    ``coarse_off``, the grid barrier is crossed, and warp 0 resolves the
    pivot into ``s_coop_scalars`` (readable after the trailing CTA barrier).
    """
    i = Int32(tx)
    while i < carry_count:
        key = _convert_to_uint32(Float32(s_c0_values[i]))
        if (key & key_mask) == key_prefix:
            digit = Int32((key >> Uint32(digit_shift)) & Uint32(0xFFF))
            _smem_red_add(wide_hist_addr, digit, Int32(1))
            _smem_red_add(coarse_hist_addr, digit >> Int32(4), Int32(1))
        i += Int32(_RADIX_THREADS)
    cute.arch.sync_threads()
    _coop_publish_wide(merge_state, group_id, fine_off, wide_hist_addr, tx)
    if tx < Int32(_RADIX):
        c = Int32(s_hist0[tx])
        if c > Int32(0):
            red_add_global_i32(
                _fused_state_ptr(merge_state, group_id, Int32(coarse_off) + tx), c
            )
        s_hist0[tx] = Int32(0)
    phase = _fused_group_barrier(merge_state, group_id, barrier_phase, ctas_pg, tx)
    if warp_idx == Int32(0):
        _coop_scan_round12(
            merge_state,
            group_id,
            coarse_off,
            fine_off,
            remaining_k,
            lane,
            s_coop_scalars,
            read_total,
        )
    cute.arch.sync_threads()
    return phase


@cute.jit
def _coop_narrow_round(
    merge_state: cute.Tensor,
    group_id: Int32,
    coarse_off: cutlass.Constexpr[int],
    s_c0_values: cute.Tensor,
    carry_count: Int32,
    key_mask: Uint32,
    key_prefix: Uint32,
    coarse_hist_addr: Int32,
    s_hist0: cute.Tensor,
    remaining_k: Int32,
    barrier_phase: Int32,
    ctas_pg: Int32,
    tx: Int32,
    lane: Int32,
    warp_idx: Int32,
    s_coop_scalars: cute.Tensor,
) -> Int32:
    """The final 8-bit cooperative radix round over the low key byte.

    Same contract as :func:`_coop_wide_round` with a 256-bin histogram only.
    """
    if tx < Int32(_RADIX):
        s_hist0[tx] = Int32(0)
    cute.arch.sync_threads()
    i = Int32(tx)
    while i < carry_count:
        key = _convert_to_uint32(Float32(s_c0_values[i]))
        if (key & key_mask) == key_prefix:
            _smem_red_add(coarse_hist_addr, Int32(key & Uint32(0xFF)), Int32(1))
        i += Int32(_RADIX_THREADS)
    cute.arch.sync_threads()
    if tx < Int32(_RADIX):
        c = Int32(s_hist0[tx])
        if c > Int32(0):
            red_add_global_i32(
                _fused_state_ptr(merge_state, group_id, Int32(coarse_off) + tx), c
            )
        s_hist0[tx] = Int32(0)
    phase = _fused_group_barrier(merge_state, group_id, barrier_phase, ctas_pg, tx)
    if warp_idx == Int32(0):
        _coop_scan_round(
            merge_state, group_id, coarse_off, remaining_k, lane, s_coop_scalars, False
        )
    cute.arch.sync_threads()
    return phase


@cute.jit
def _coop_scan_round12(
    merge_state: cute.Tensor,
    group_id: Int32,
    coarse_off: cutlass.Constexpr[int],
    fine_off: cutlass.Constexpr[int],
    remaining_k: Int32,
    lane: Int32,
    s_coop_scalars: cute.Tensor,
    read_total: cutlass.Constexpr[bool],
):
    """Warp 0: resolve a 12-bit pivot digit from the coarse and fine histograms.

    The 256-bin coarse histogram (lane ``l`` owns bins ``8l .. 8l+7``) is
    suffix-scanned for the coarse pivot ``c`` and the count above it; lanes
    0-3 then read the 16 fine bins under ``c`` and refine. Both are read
    through L2 so other CTAs' reductions are visible. ``s_coop_scalars``
    receives the 12-bit pivot bin, the count strictly above it, and the
    bin's own count; with ``read_total`` it also receives the group's
    candidate total.
    """
    base = _fused_state_ptr(
        merge_state, group_id, Int32(coarse_off) + lane * Int32(8)
    )
    h0, h1, h2, h3 = ld_global_cg_v4_u32(base)
    h4, h5, h6, h7 = ld_global_cg_v4_u32(base + Int64(16))
    s7 = Int32(h7)
    s6 = s7 + Int32(h6)
    s5 = s6 + Int32(h5)
    s4 = s5 + Int32(h4)
    s3 = s4 + Int32(h3)
    s2 = s3 + Int32(h2)
    s1 = s2 + Int32(h1)
    s0 = s1 + Int32(h0)
    above = s0
    for d in (1, 2, 4, 8, 16):
        other = cute.arch.shuffle_sync_down(above, offset=d)
        if lane + Int32(d) < Int32(_WARP_THREADS):
            above = above + other
    above = above - s0
    suffix = [s0 + above, s1 + above, s2 + above, s3 + above,
              s4 + above, s5 + above, s6 + above, s7 + above, above]
    coarse_bin = Int32(-1)
    coarse_gt = Int32(-1)
    for j in cutlass.range_constexpr(8):
        if (suffix[j] >= remaining_k) & (suffix[j + 1] < remaining_k):
            coarse_bin = lane * Int32(8) + Int32(j)
            coarse_gt = suffix[j + 1]
    for d in (16, 8, 4, 2, 1):
        coarse_bin = _max_i32(
            coarse_bin, cute.arch.shuffle_sync_bfly(coarse_bin, offset=d)
        )
        coarse_gt = _max_i32(
            coarse_gt, cute.arch.shuffle_sync_bfly(coarse_gt, offset=d)
        )
    f0 = Uint32(0)
    f1 = Uint32(0)
    f2 = Uint32(0)
    f3 = Uint32(0)
    if lane < Int32(4):
        f0, f1, f2, f3 = ld_global_cg_v4_u32(
            _fused_state_ptr(
                merge_state,
                group_id,
                Int32(fine_off) + coarse_bin * Int32(16) + lane * Int32(4),
            )
        )
    t3 = Int32(f3)
    t2 = t3 + Int32(f2)
    t1 = t2 + Int32(f1)
    t0 = t1 + Int32(f0)
    fine_above = t0
    for d in (1, 2):
        other = cute.arch.shuffle_sync_down(fine_above, offset=d)
        if lane + Int32(d) < Int32(4):
            fine_above = fine_above + other
    fine_above = fine_above - t0 + coarse_gt
    fsuffix = [t0 + fine_above, t1 + fine_above, t2 + fine_above, t3 + fine_above, fine_above]
    if lane < Int32(4):
        for j in cutlass.range_constexpr(4):
            if (fsuffix[j] >= remaining_k) & (fsuffix[j + 1] < remaining_k):
                s_coop_scalars[0] = Uint32(coarse_bin * Int32(16) + lane * Int32(4) + Int32(j))
                s_coop_scalars[1] = Uint32(fsuffix[j + 1])
                s_coop_scalars[2] = Uint32(fsuffix[j] - fsuffix[j + 1])
    if cutlass.const_expr(read_total):
        if lane == Int32(0):
            s_coop_scalars[3] = Uint32(
                ld_global_acquire_i32(
                    _fused_state_ptr(merge_state, group_id, Int32(_FUSED_STATE_TOTAL))
                )
            )


@cute.jit
def _max_i32(a: Int32, b: Int32) -> Int32:
    r = a
    if b > a:
        r = b
    return r


@cute.jit
def _coop_scan_round(
    merge_state: cute.Tensor,
    group_id: Int32,
    hist_off: cutlass.Constexpr[int],
    remaining_k: Int32,
    lane: Int32,
    s_coop_scalars: cute.Tensor,
    read_total: cutlass.Constexpr[bool],
):
    """Warp 0: suffix-scan a 256-bin group histogram and publish the pivot.

    Lane ``l`` owns bins ``8l .. 8l+7`` of the histogram at word offset
    ``hist_off``, read through L2 so other CTAs' reductions are visible.
    ``s_coop_scalars`` receives the pivot bin (the highest bin whose suffix
    count reaches ``remaining_k``), the count strictly above it, and the
    bin's own count; with ``read_total`` it also receives the group's
    candidate total.
    """
    base = _fused_state_ptr(
        merge_state, group_id, Int32(hist_off) + lane * Int32(8)
    )
    h0, h1, h2, h3 = ld_global_cg_v4_u32(base)
    h4, h5, h6, h7 = ld_global_cg_v4_u32(base + Int64(16))
    s7 = Int32(h7)
    s6 = s7 + Int32(h6)
    s5 = s6 + Int32(h5)
    s4 = s5 + Int32(h4)
    s3 = s4 + Int32(h3)
    s2 = s3 + Int32(h2)
    s1 = s2 + Int32(h1)
    s0 = s1 + Int32(h0)
    above = s0
    for d in (1, 2, 4, 8, 16):
        other = cute.arch.shuffle_sync_down(above, offset=d)
        if lane + Int32(d) < Int32(_WARP_THREADS):
            above = above + other
    above = above - s0
    suffix = [s0 + above, s1 + above, s2 + above, s3 + above,
              s4 + above, s5 + above, s6 + above, s7 + above, above]
    for j in cutlass.range_constexpr(8):
        if (suffix[j] >= remaining_k) & (suffix[j + 1] < remaining_k):
            s_coop_scalars[0] = Uint32(lane * Int32(8) + Int32(j))
            s_coop_scalars[1] = Uint32(suffix[j + 1])
            s_coop_scalars[2] = Uint32(suffix[j] - suffix[j + 1])
    if cutlass.const_expr(read_total):
        if lane == Int32(0):
            s_coop_scalars[3] = Uint32(
                ld_global_acquire_i32(
                    _fused_state_ptr(merge_state, group_id, Int32(_FUSED_STATE_TOTAL))
                )
            )


@cute.jit
def _load_q_bytes_g2s_v4(
    q_bytes: cute.Tensor,
    q_idx: Int32,
    q_row_stride_bytes: Int64,
    real_q_bytes: cutlass.Constexpr[int],
    padded_q_bytes: cutlass.Constexpr[int],
    q_smem_base_addr: Int32,
    tx: Int32,
    n_threads: Int32,
):
    """Vectorized 16-byte query staging for the common contiguous-head layout."""
    linear_vec = tx
    total_vecs = Int32(int(padded_q_bytes) // 16)
    real_vecs = Int32(int(real_q_bytes) // 16)
    row_base = Int64(q_idx) * q_row_stride_bytes
    while linear_vec < total_vecs:
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if linear_vec < real_vecs:
            q_addr = get_ptr_as_int64(q_bytes, row_base + Int64(linear_vec) * Int64(16))
            v0, v1, v2, v3 = ld_global_v4_u32(q_addr)
        st_shared_v4_u32(q_smem_base_addr + linear_vec * Int32(16), v0, v1, v2, v3)
        linear_vec += n_threads


@cute.jit
def _score_tokens_direct_k(
    q_perm_base_addr: Int32,
    s_w: cute.Tensor,
    num_heads: Int32,
    k_quant_bytes: cute.Tensor,
    k_byte_off: Int64,
    lane: Int32,
    num_q_head_tiles: cutlass.Constexpr[int],
):
    """Score 8 tokens against every head tile with K read straight from L2.

    Byte-identical to the smem-staged score core: each lane loads exactly the
    bytes ldmatrix.m8n8x4.b16 would hand it (reg m, lane l = token l//4,
    byte-quad l%4 of K half m), preserving the raw-fragment convention the
    MMA relies on without staging K through shared memory. k_byte_off is the
    lane's base byte offset (page + token-row + quad). After the butterfly
    reduction every lane holds the two column totals for columns
    2*(lane%4) and 2*(lane%4)+1, summed over all head tiles.
    """
    # num_heads is unreferenced: s_w is zero-padded to the padded head count.
    q_row = lane & Int32(15)
    q_half = lane >> Int32(4)
    k0_0 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off))
    k1_0 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(16)))
    k0_1 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(32)))
    k1_1 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(48)))
    k0_2 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(64)))
    k1_2 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(80)))
    k0_3 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(96)))
    k1_3 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(112)))
    total0 = Float32(0.0)
    total1 = Float32(0.0)
    qgrp = lane // Int32(4)
    for tile in cutlass.range_constexpr(num_q_head_tiles):
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        q_tile_base = q_perm_base_addr + Int32(
            tile * _PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM
        )
        for step in cutlass.range_constexpr(_INDEX_HEAD_DIM // 32):
            q_addr = _smem_addr_from_b128_offset(
                q_tile_base,
                _permuted_offset_128b(
                    q_row, Int32(2 * step) + q_half, Int32(_INDEX_HEAD_DIM // 16)
                ),
            )
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_addr)
            if cutlass.const_expr(step == 0):
                bk0 = k0_0
                bk1 = k1_0
            elif cutlass.const_expr(step == 1):
                bk0 = k0_1
                bk1 = k1_1
            elif cutlass.const_expr(step == 2):
                bk0 = k0_2
                bk1 = k1_2
            else:
                bk0 = k0_3
                bk1 = k1_3
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc0,
                acc1,
                acc2,
                acc3,
                a0,
                a1,
                a2,
                a3,
                bk0,
                bk1,
                Uint32(0x7F7F7F7F),
                Uint32(0x7F7F7F7F),
            )
            acc0 = d0
            acc1 = d1
            acc2 = d2
            acc3 = d3
        head0 = Int32(tile * _PAGED_Q_HEAD_TILE) + qgrp
        head1 = head0 + Int32(8)
        w0 = Float32(s_w[head0])
        w1 = Float32(s_w[head1])
        p0 = Float32(fmax_f32(acc0, Float32(0.0)) * w0)
        p0 = Float32(p0 + fmax_f32(acc2, Float32(0.0)) * w1)
        p1 = Float32(fmax_f32(acc1, Float32(0.0)) * w0)
        p1 = Float32(p1 + fmax_f32(acc3, Float32(0.0)) * w1)
        total0 = Float32(total0 + _reduce_column_pair_sum(p0))
        total1 = Float32(total1 + _reduce_column_pair_sum(p1))
    return total0, total1


def _fused_indexer_state_nbytes(launch: _FusedLaunchConfig) -> int:
    return launch.num_groups * _STATE_WORDS_PER_GROUP * 4


def _fused_indexer_carry_nbytes(launch: _FusedLaunchConfig, topk: int) -> int:
    # value f32 + index i32 = 8 bytes per (group, cta, topk) slot.
    return launch.num_groups * launch.ctas_per_group * int(topk) * 8


def fused_indexer_workspace_nbytes(
    num_rows: int,
    width: int,
    topk: int,
    *,
    device: torch.device | str | None = None,
) -> int:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        # CPU planning fallback: assume 1 CTA/group worst case.
        return max(int(num_rows), 1) * (_STATE_WORDS_PER_GROUP * 4 + int(topk) * 8)
    launch = _resolve_fused_launch_config(int(num_rows), int(width), device)
    return _fused_indexer_state_nbytes(launch) + _fused_indexer_carry_nbytes(
        launch, int(topk)
    )


def _fused_indexer_capacity_nbytes(
    max_rows: int, max_width: int, topk: int, *, device: torch.device
) -> int:
    max_rows = max(int(max_rows), 1)
    max_width = max(int(max_width), 1)
    if device.type != "cuda":
        return fused_indexer_workspace_nbytes(max_rows, max_width, topk, device=device)
    # Worst case over the strides that change ctas_per_group / num_groups.
    candidate_widths = {1, max_width}
    if max_width > FUSED_MIN_WIDTH:
        candidate_widths.add(FUSED_MIN_WIDTH + 1)
    return max(
        fused_indexer_workspace_nbytes(max_rows, w, topk, device=device)
        for w in candidate_widths
    )


@dataclass(frozen=True, kw_only=True)
class B12XFusedIndexerScratchCaps:
    device: torch.device | str
    max_rows: int
    max_width: int
    topk: int

    def __post_init__(self) -> None:
        if int(self.topk) not in _SUPPORTED_TOPK:
            raise ValueError(
                f"fused indexer supports topk {_SUPPORTED_TOPK}, got {self.topk}"
            )


# --- dispatch gates -----------------------------------------------------------
def supports_fused_indexer(
    *,
    topk: int,
    num_rows: int,
    width: int,
) -> bool:
    """Whether the fused kernel is applicable (capability gate)."""
    if int(topk) not in _SUPPORTED_TOPK:
        return False
    if int(width) <= 0 or int(num_rows) <= 0:
        return False
    return True


def resolve_fused_indexer_path(
    *,
    topk: int,
    num_rows: int,
    width: int,
    num_heads: int | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Route to the fused kernel only for small decode batches (rows <= N).

    Row count, head count, top-k, and width here are all capture-time workspace
    metadata; live seqlen is deliberately absent, so the selected route is stable
    across vLLM CUDA-graph replays. The general small-decode gate remains six
    rows. C4's 64-head and 32-head top-k-512 shapes use fused through B16 on
    SM120 and SM121. GLM-5.2's 32-head/top-k-2048 shape uses fused through B16
    and the streamed tiled route beyond. Prefill is selected before this
    decode-only resolver and remains packed-contiguous.
    """
    if not supports_fused_indexer(topk=topk, num_rows=num_rows, width=width):
        return False
    if int(topk) == 512 and num_heads is not None and int(num_heads) in {32, 64}:
        # C4 (DSV4 heads=64, GLM-5.3-Flash heads=32): both RTX-class SM120 and
        # GB10/SM121 use the same fixed-page scorer and top-k specialization.
        return (
            compute_capability in {(12, 0), (12, 1)}
            and int(num_rows) <= _C4_B12X_FUSED_MAX_ROWS
        )
    if int(topk) == 2048 and num_heads is not None and int(num_heads) == 32:
        return int(num_rows) <= _GLM32_FUSED_MAX_ROWS
    return int(num_rows) <= FUSED_MAX_ROWS


def fused_indexer_scratch_max_rows(
    *,
    topk: int,
    num_heads: int,
    compute_capability: tuple[int, int] | None = None,
) -> int:
    """Return the fixed row capacity needed by fused decode scratch."""

    if (
        int(topk) == 512
        and int(num_heads) in {32, 64}
        and compute_capability in {(12, 0), (12, 1)}
    ):
        return _C4_B12X_FUSED_MAX_ROWS
    if int(topk) == 2048 and int(num_heads) == 32:
        return _GLM32_FUSED_MAX_ROWS
    return FUSED_MAX_ROWS


def _resolve_default_ctas_per_group(
    *,
    num_rows: int,
    max_pages: int,
    device: torch.device,
) -> int:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    # Target a single CTA wave (rows * ctas_per_group ~ num_sms): more CTAs
    # only add last-CTA merge work + a second wave. Never split below 1 page.
    return max(1, min(int(max_pages), num_sms // max(1, int(num_rows))))


def _resolve_default_merge_threshold(
    *,
    ctas_per_group: int,
    num_heads: int,
    topk: int,
) -> int:
    """Live-length threshold above which a multi-CTA group merges cooperatively.

    The cooperative arm publishes each radix round's histogram with one grid
    barrier and places its output without another; measured under graph
    replay on sm120 (top-k 512 with 64 heads and top-k 2048 with 32 heads,
    1 to 8 rows, 4k to 64k live tokens) it won or tied the serial last-CTA
    arm at every point, so every co-resident group merges cooperatively at
    any live length. Single-CTA groups have no cross-CTA merge.
    """
    del num_heads, topk
    if int(ctas_per_group) <= 1:
        return _FORCE_LAST_CTA
    return 0


def resolve_fused_merge_threshold(
    fused_merge: str,
    *,
    ctas_per_group: int,
    num_heads: int,
    topk: int,
) -> int:
    """Live-length threshold implementing the policy's ``fused_merge`` choice.

    A group whose live row length exceeds the threshold merges cooperatively;
    otherwise it takes the serial last-CTA arm. Single-CTA groups have no
    cross-CTA merge. ``auto`` applies the planner default
    (:func:`_resolve_default_merge_threshold`).
    """
    if fused_merge not in FUSED_MERGE_CHOICES:
        raise ValueError(
            f"unsupported fused_merge {fused_merge!r}; expected one of "
            f"{FUSED_MERGE_CHOICES}"
        )
    if int(ctas_per_group) <= 1 or fused_merge == FUSED_MERGE_SERIAL:
        return _FORCE_LAST_CTA
    if fused_merge == FUSED_MERGE_COOPERATIVE:
        return 0
    return _resolve_default_merge_threshold(
        ctas_per_group=int(ctas_per_group),
        num_heads=int(num_heads),
        topk=int(topk),
    )


def fused_indexer_decode_warmup_rows(
    *,
    topk: int,
    num_heads: int,
    max_pages: int,
    device: torch.device,
    fused_merge: str = FUSED_MERGE_AUTO,
) -> tuple[int, ...]:
    """Return one safe launch row count per compiled decode policy.

    The default fused planner specializes ``ctas_per_group`` and its merge
    threshold by row count. Each returned row count selects a distinct policy
    while preserving the planner's one-machine-wave launch bound. Callers can
    launch these shapes before graph capture to populate every runtime variant.
    ``fused_merge`` must match the policy choice the serving plan applies.
    """
    props = torch.cuda.get_device_properties(device)
    compute_capability = (
        (int(props.major), int(props.minor))
        if hasattr(props, "major") and hasattr(props, "minor")
        else None
    )
    max_rows = fused_indexer_scratch_max_rows(
        topk=int(topk),
        num_heads=int(num_heads),
        compute_capability=compute_capability,
    )

    width = int(max_pages) * _PAGE_SIZE
    rows_by_policy: list[int] = []
    seen_policies: set[tuple[int, int]] = set()
    for rows in range(1, max_rows + 1):
        if not resolve_fused_indexer_path(
            topk=int(topk),
            num_rows=rows,
            width=width,
            num_heads=int(num_heads),
            compute_capability=compute_capability,
        ):
            continue
        ctas_per_group = _resolve_default_ctas_per_group(
            num_rows=rows,
            max_pages=int(max_pages),
            device=device,
        )
        merge_threshold = resolve_fused_merge_threshold(
            fused_merge,
            ctas_per_group=ctas_per_group,
            num_heads=int(num_heads),
            topk=int(topk),
        )
        policy = (ctas_per_group, merge_threshold)
        if policy not in seen_policies:
            seen_policies.add(policy)
            rows_by_policy.append(rows)
    return tuple(rows_by_policy)


# --- scratch plan trio (layout-agnostic: sizes only) --------------------------
@dataclass(frozen=True)
class B12XFusedIndexerScratchPlan:
    caps: B12XFusedIndexerScratchCaps
    workspace_nbytes: int
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def workspace_view(
        self,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Materialize the int32 workspace view from a bound arena slice."""
        arena = scratch_tensor(scratch, self._scratch_specs, owner="fused indexer")
        return arena[: self.workspace_nbytes].view(torch.int32)


def plan_fused_indexer_scratch(
    caps: B12XFusedIndexerScratchCaps,
) -> B12XFusedIndexerScratchPlan:
    device = torch.device(caps.device)
    workspace_nbytes = _fused_indexer_capacity_nbytes(
        caps.max_rows, caps.max_width, caps.topk, device=device
    )
    return B12XFusedIndexerScratchPlan(
        caps=caps,
        workspace_nbytes=workspace_nbytes,
        _scratch_specs=(
            scratch_buffer_spec(
                "fused_indexer.state",
                nbytes=workspace_nbytes,
                device=device,
            ),
        ),
    )


# --- fused kernel SharedStorage -----------------------------------------------
# 1024-thread CTA. Score phase: warps 0-3 (tx < _PAGED_THREADS_PER_CTA) run the
# scorer MMA over a K-sub-tile into the logit staging (s_stage_*); all 1024
# threads then run the tiled_topk radix folding {staging ∪ carry} -> carry.
# Layout = scorer fields (q/weights/k_page/perm/scales/partial_logits) + radix
# scratch (hist0/1, out_idx, counters, cand0/1) + the SMEM running-topk carry
# (double-buffered value+gindex) + the per-sub-tile logit staging (value+gindex).
# Batching: score directly into an over-sized carry (topk + slack); run the radix
# only to "trim" carry back to topk when it would overflow -> one radix per
# ~slack/page pages instead of one per page (the dominant cost). Slack is a
# perf/SMEM knob.
# The fused scorer is synchronization/instruction limited rather than HBM
# limited on SM120 (the 64-head DSV4 B32/128K production launch reaches only
# ~45% DRAM throughput).  Spend the otherwise-idle one-block-per-SM shared-memory
# headroom on a larger DSV4 carry, batching fifty-two 64-token pages between
# exact radix trims.  DSV4 serving graphs always have a large static context
# capacity, so do not create capacity-dependent variants that are unreachable
# in production.  Other contracts retain the smaller footprint, notably GLM's
# top-k-2048 specialization, which cannot fit the larger carry.
# 448 (not 512): the deferred-reduce pipeline's third K stage + partials/scales
# rings cost ~10KB of SMEM; topk-2048 (GLM) needs 64 fewer carry entries to fit
# the 101376B SM120 carveout. One trim per ~7 pages instead of 8.
_BATCH_SLACK = 448
_DSV4_BATCH_SLACK = 2816


def _resolve_fused_batch_slack(
    *,
    kv_layout: int,
    num_heads: int,
    topk: int,
) -> int:
    """Select the capture-static carry batching size for one fused variant."""

    if int(kv_layout) == KV_LAYOUT_PAGED and int(num_heads) == 64 and int(topk) == 512:
        return _DSV4_BATCH_SLACK
    return _BATCH_SLACK


@lru_cache(maxsize=64)
def _fused_indexer_shared_storage_cls(
    padded_q_heads: int,
    tokens_per_work: int,
    num_q_head_tiles: int,
    topk: int,
    carry_cap: int,
    cands: int,
):
    class SharedStorage:
        pass

    def _u8_page():
        return cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, _PAGE_SIZE * _INDEX_HEAD_DIM], 1024
        ]

    def _i32(n):
        return cute.struct.Align[cute.struct.MemRange[cutlass.Int32, int(n)], 128]

    def _f32(n):
        return cute.struct.Align[cute.struct.MemRange[cutlass.Float32, int(n)], 128]

    annotations = {
        # --- scorer fields (mirror _paged_indexer_shared_storage_cls) ---
        "mbar_ptr_k": cute.struct.MemRange[cutlass.Int64, 1],
        "q_bytes": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(padded_q_heads) * _INDEX_HEAD_DIM],
            16,
        ],
        "weights": _f32(int(padded_q_heads)),
        "k_page": _u8_page(),
        "k_page_perm": _u8_page(),
        "k_page3": _u8_page(),
        "scales": _f32(4 * _PAGE_SIZE),
        # Second per-page scale slot: the paged branch ping-pongs K between
        # k_page_perm (stage 0) and k_page (stage 1, otherwise unused there),
        # so the scale staging needs a matching second buffer.
        "partial_logits": _f32(int(tokens_per_work) * int(num_q_head_tiles)),
        "partial_logits2": _f32(int(tokens_per_work) * int(num_q_head_tiles)),
        # --- radix scratch (mirror tiled_topk SharedStorage) ---
        "hist0": _i32(384),
        "hist1": _i32(384),
        "out_idx": _i32(int(topk)),
        "counter": _i32(1),
        "thr_id": _i32(1),
        "ni0": _i32(1),
        "ni1": _i32(1),
        "last_rem": _i32(1),
        # relay[0]=pack offset, relay[1]=arrival-old (last-CTA broadcast scalars)
        "relay": _i32(2),
        # cooperative-merge SMEM: suffix-scan of the global histogram + per-CTA
        # scalars (prefix/remaining_k/pivot_bin/rem) + counters (gt-count/base).
        "coop_suffix": _i32(int(_RADIX)),
        "coop_scalars": cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, 4], 128],
        "coop_counters": _i32(2),
        "cand0": _i32(cands),
        "cand1": _i32(cands),
        # --- running accumulator: scored tokens append here; over-sized to topk+slack ---
        "carry0_values": _f32(int(carry_cap)),
        "carry0_gindex": _i32(int(carry_cap)),
        # --- trim output (exactly topk) ---
        "carry1_values": _f32(int(topk)),
        "carry1_gindex": _i32(int(topk)),
    }
    SharedStorage.__annotations__ = annotations
    return cute.struct(SharedStorage)


@cute.jit
def _fused_radix_select(
    vin_v: cute.Tensor,
    vin_g: cute.Tensor,
    vout_v: cute.Tensor,
    vout_g: cute.Tensor,
    n_total: Int32,
    topk_static: Int32,
    cands: cutlass.Constexpr[int],
    tx: Int32,
    s_hist0: cute.Tensor,
    s_hist1: cute.Tensor,
    s_out: cute.Tensor,
    s_cand0: cute.Tensor,
    s_cand1: cute.Tensor,
    h0: Int32,
    ctr: Int32,
    thr: Int32,
    ni0: Int32,
    ni1: Int32,
    lr: Int32,
):
    """Exact radix top-k of vin[0:n_total] -> vout[0:topk_static] (single source).

    Assumes n_total > topk_static (callers only trim when over capacity). Reuses the
    tiled_topk coarse-8-bit-histogram + byte-refine. All 1024 threads must call this.
    """
    topk = topk_static
    if tx < Int32(257):
        s_hist0[tx] = Int32(0)
    cute.arch.sync_threads()
    # Single contiguous source: read vin_v[idx] directly (no virtual-index branch)
    # and unroll the full-length scan by _SCAN_UNROLL for memory-level parallelism,
    # mirroring tiled_topk's select (this is the rows=1 last-CTA merge hot loop).
    idx_base = Int32(tx)
    full_scan_limit = n_total - Int32((_SCAN_UNROLL - 1) * _RADIX_THREADS)
    while idx_base < full_scan_limit:
        for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
            sidx = idx_base + Int32(scan_u * _RADIX_THREADS)
            bin8 = _convert_to_uint8(Float32(vin_v[sidx]))
            _smem_red_add(h0, Int32(bin8), Int32(1))
        idx_base += Int32(_RADIX_THREADS * _SCAN_UNROLL)
    while idx_base < n_total:
        bin8 = _convert_to_uint8(Float32(vin_v[idx_base]))
        _smem_red_add(h0, Int32(bin8), Int32(1))
        idx_base += Int32(_RADIX_THREADS)
    cute.arch.sync_threads()
    for stage in cutlass.range_constexpr(8):
        j = Int32(1 << stage)
        if tx < Int32(256):
            if (stage & 1) == 0:
                value = Int32(s_hist0[tx])
                if tx < Int32(256) - j:
                    value = value + Int32(s_hist0[tx + j])
                s_hist1[tx] = value
            else:
                value = Int32(s_hist1[tx])
                if tx < Int32(256) - j:
                    value = value + Int32(s_hist1[tx + j])
                s_hist0[tx] = value
        cute.arch.sync_threads()
    if tx < Int32(256):
        val_tx = Int32(s_hist0[tx])
        val_tx1 = Int32(s_hist0[tx + Int32(1)])
        if val_tx > topk:
            if val_tx1 <= topk:
                _smem_st(thr, Int32(0), Int32(tx))
                _smem_st(ni0, Int32(0), Int32(0))
                _smem_st(ctr, Int32(0), Int32(0))
    cute.arch.sync_threads()
    threshold_bin = _smem_ld(thr, Int32(0))
    topk = topk - Int32(s_hist0[threshold_bin + Int32(1)])
    # Captured below (refine-setup) = true #candidates in the coarse threshold bin; used
    # to detect candidate-buffer overflow and fall back to an exact re-scan radix.
    bin_count = Int32(0)

    if topk == Int32(0):
        idx_base = Int32(tx)
        while idx_base < n_total:
            bin8 = _convert_to_uint8(Float32(vin_v[idx_base]))
            if Int32(bin8) > threshold_bin:
                pos = _smem_xadd(ctr, Int32(0), Int32(1))
                s_out[pos] = idx_base
            idx_base += Int32(_RADIX_THREADS)

    if topk != Int32(0):
        cute.arch.sync_threads()
        if tx < Int32(257):
            s_hist0[tx] = Int32(0)
        cute.arch.sync_threads()
        idx_base = Int32(tx)
        full_scan_limit = n_total - Int32((_SCAN_UNROLL - 1) * _RADIX_THREADS)
        while idx_base < full_scan_limit:
            for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
                sidx = idx_base + Int32(scan_u * _RADIX_THREADS)
                raw_input = Float32(vin_v[sidx])
                bin8 = _convert_to_uint8(raw_input)
                if Int32(bin8) > threshold_bin:
                    pos = _smem_xadd(ctr, Int32(0), Int32(1))
                    s_out[pos] = sidx
                else:
                    if Int32(bin8) == threshold_bin:
                        cand_pos = _smem_xadd(ni0, Int32(0), Int32(1))
                        if cand_pos < Int32(cands):
                            s_cand0[cand_pos] = sidx
                            key32 = _convert_to_uint32(raw_input)
                            sub_bin = (key32 >> Uint32(24)) & Uint32(0xFF)
                            _smem_red_add(h0, Int32(sub_bin), Int32(1))
            idx_base += Int32(_RADIX_THREADS * _SCAN_UNROLL)
        while idx_base < n_total:
            raw_input = Float32(vin_v[idx_base])
            bin8 = _convert_to_uint8(raw_input)
            if Int32(bin8) > threshold_bin:
                pos = _smem_xadd(ctr, Int32(0), Int32(1))
                s_out[pos] = idx_base
            else:
                if Int32(bin8) == threshold_bin:
                    cand_pos = _smem_xadd(ni0, Int32(0), Int32(1))
                    if cand_pos < Int32(cands):
                        s_cand0[cand_pos] = idx_base
                        key32 = _convert_to_uint32(raw_input)
                        sub_bin = (key32 >> Uint32(24)) & Uint32(0xFF)
                        _smem_red_add(h0, Int32(sub_bin), Int32(1))
            idx_base += Int32(_RADIX_THREADS)
        cute.arch.sync_threads()
        # ni0 now holds the FULL threshold-bin candidate count: the xadd ran for every
        # bin-matching key, even past `cands`. Capture it to detect refine overflow.
        bin_count = Int32(_smem_ld(ni0, Int32(0)))

        for round_idx in cutlass.range_constexpr(4):
            if topk != Int32(-1):
                r_idx_is_0 = (round_idx % 2) == 0
                r_idx_next_is_0 = not r_idx_is_0
                raw_num_input = (
                    _smem_ld(ni0, Int32(0))
                    if cutlass.const_expr(r_idx_is_0)
                    else _smem_ld(ni1, Int32(0))
                )
                num_input = (
                    raw_num_input if raw_num_input < Int32(cands) else Int32(cands)
                )
                for stage in cutlass.range_constexpr(8):
                    j = Int32(1 << stage)
                    if tx < Int32(256):
                        if (stage & 1) == 0:
                            value = Int32(s_hist0[tx])
                            if tx < Int32(256) - j:
                                value = value + Int32(s_hist0[tx + j])
                            s_hist1[tx] = value
                        else:
                            value = Int32(s_hist1[tx])
                            if tx < Int32(256) - j:
                                value = value + Int32(s_hist1[tx + j])
                            s_hist0[tx] = value
                    cute.arch.sync_threads()
                if tx < Int32(256):
                    val_tx = Int32(s_hist0[tx])
                    val_tx1 = Int32(s_hist0[tx + Int32(1)])
                    if val_tx > topk:
                        if val_tx1 <= topk:
                            _smem_st(thr, Int32(0), Int32(tx))
                            if cutlass.const_expr(r_idx_next_is_0):
                                _smem_st(ni0, Int32(0), Int32(0))
                            else:
                                _smem_st(ni1, Int32(0), Int32(0))
                            _smem_st(lr, Int32(0), topk - val_tx1)
                cute.arch.sync_threads()
                sub_threshold = _smem_ld(thr, Int32(0))
                topk = topk - Int32(s_hist0[sub_threshold + Int32(1)])

                if topk == Int32(0):
                    i = Int32(tx)
                    while i < num_input:
                        c_idx = (
                            Int32(s_cand0[i])
                            if cutlass.const_expr(r_idx_is_0)
                            else Int32(s_cand1[i])
                        )
                        offset = Int32(24 - round_idx * 8)
                        raw_val = Float32(vin_v[c_idx])
                        key32 = _convert_to_uint32(raw_val)
                        bin = (key32 >> Uint32(offset)) & Uint32(0xFF)
                        if Int32(bin) > sub_threshold:
                            pos = _smem_xadd(ctr, Int32(0), Int32(1))
                            s_out[pos] = c_idx
                        i += Int32(_RADIX_THREADS)
                    topk = Int32(-1)

                if topk != Int32(-1):
                    cute.arch.sync_threads()
                    if tx < Int32(257):
                        s_hist0[tx] = Int32(0)
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < num_input:
                        c_idx = (
                            Int32(s_cand0[i])
                            if cutlass.const_expr(r_idx_is_0)
                            else Int32(s_cand1[i])
                        )
                        raw_val = Float32(vin_v[c_idx])
                        offset = Int32(24 - round_idx * 8)
                        key32 = _convert_to_uint32(raw_val)
                        bin = (key32 >> Uint32(offset)) & Uint32(0xFF)
                        if Int32(bin) > sub_threshold:
                            pos = _smem_xadd(ctr, Int32(0), Int32(1))
                            s_out[pos] = c_idx
                        else:
                            if Int32(bin) == sub_threshold:
                                if cutlass.const_expr(round_idx == 3):
                                    old_rem = _smem_xadd(lr, Int32(0), Int32(-1))
                                    if old_rem > Int32(0):
                                        s_out[topk_static - old_rem] = c_idx
                                else:
                                    cand_pos = (
                                        _smem_xadd(ni0, Int32(0), Int32(1))
                                        if cutlass.const_expr(r_idx_next_is_0)
                                        else _smem_xadd(ni1, Int32(0), Int32(1))
                                    )
                                    if cand_pos < Int32(cands):
                                        if cutlass.const_expr(r_idx_next_is_0):
                                            s_cand0[cand_pos] = c_idx
                                        else:
                                            s_cand1[cand_pos] = c_idx
                                        sub_bin = (
                                            key32 >> Uint32(24 - (round_idx + 1) * 8)
                                        ) & Uint32(0xFF)
                                        _smem_red_add(h0, Int32(sub_bin), Int32(1))
                        i += Int32(_RADIX_THREADS)
                    cute.arch.sync_threads()

    # ---- exact overflow fallback ----
    # The buffered coarse+refine above drops winners when more than `cands`
    # candidates share the coarse threshold bucket (clustered scores; the
    # cross-CTA merge can pack up to ctas_per_group*topk candidates). Redo the
    # selection exactly by re-scanning vin -- see _exact_overflow_fallback. The
    # single contiguous vin_v source uses the flat=True load path.
    if bin_count > Int32(cands):
        _exact_overflow_fallback(
            tx,
            n_total,
            topk_static,
            s_hist0,
            s_hist1,
            s_out,
            h0,
            ctr,
            thr,
            ni0,
            ni1,
            lr,
            flat=True,
            flat_values=vin_v,
            input_tensor=vin_v,
            carry_values=vin_v,
            row_base=Int32(0),
            row_start=Int32(0),
            carry_base=Int32(0),
            chunk_len=Int32(0),
            block_q=1,
            block_k=1,
            is_tiled=False,
            is_first=True,
        )
    cute.arch.sync_threads()
    i = Int32(tx)
    while i < topk_static:
        sel = Int32(s_out[i])
        vout_v[i] = Float32(vin_v[sel])
        vout_g[i] = Int32(vin_g[sel])
        i += Int32(_RADIX_THREADS)


class DSAFusedIndexerKernel:
    """Fused DSA indexer: score a query row's K slice and select its top-k.

    One 1024-thread CTA per (row, slice). ``ctas_per_group`` CTAs split a
    row's pages; each scores its slice and the group merges across CTAs.

    Score phase (paged): 64-token pages stream through a 3-stage cp.async
    ring one page ahead of the MMA; the score warps run the mxfp8 MMA
    against the ldmatrix-permuted Q staged in shared memory and a page's
    reduce and carry append are deferred behind the next page's barrier.
    The 4-head-tile paged contract at 3 rows or more instead scores 8
    tokens per warp straight from L2 (``direct_k_score``). Scored tokens
    append to an over-sized carry accumulator that a radix select trims
    back to ``topk`` when it would overflow, so the trim runs once per
    batch of pages rather than per page.

    Merge phase (``ctas_per_group > 1``): a group whose live length exceeds
    ``merge_threshold`` selects the final top-k cooperatively with three
    grid-barrier radix rounds (12, 12, and 8 key bits) over the CTAs'
    carries; otherwise every CTA packs its carry into a global slab and the
    last CTA to arrive radix-selects the row. Both arms self-reset their
    group state for the next launch or graph replay.
    """

    def __init__(
        self,
        *,
        num_heads_static: int,
        topk: int,
        kv_layout: int = KV_LAYOUT_PAGED,
        paged_output: bool = False,
        ctas_per_group: int = 1,
        num_sms: int = 0,
        merge_threshold: int = _LAST_CTA_MERGE_MAX,
        k_quant_page_stride: int = _PAGE_SIZE * _INDEX_HEAD_DIM,
        k_scales_row_stride: int = _PAGE_SIZE,
        max_seq_capacity: int = 1 << 30,
        vectorized_q_load: bool = False,
        q_row_stride_bytes: int = 0,
    ):
        self.num_heads_static = int(num_heads_static)
        self.topk = int(topk)
        self.kv_layout = int(kv_layout)
        self.ctas_per_group = max(1, int(ctas_per_group))
        self.merge_threshold = int(merge_threshold)
        self.max_seq_capacity = max(0, int(max_seq_capacity))
        self.vectorized_q_load = bool(vectorized_q_load)
        self.q_row_stride_bytes = int(q_row_stride_bytes)
        # dim-0 byte stride of k_quant_bytes for the PAGED wide load. Defaults to
        # the contiguous 8192 (64*128); the packed paged cache interleaves per-page
        # scales so its real page stride is 8448 -- run_fused_paged_indexer passes
        # k_quant_bytes.stride(0) so the load reads the correct page.
        self.k_quant_page_stride = int(k_quant_page_stride)
        # dim-0 ELEMENT stride of the (pages, 64) f32 scales view: the packed
        # cache interleaves per-page scales, while standalone scale tensors are
        # normally contiguous.  Keep this independent from the quant-page byte
        # stride so both layouts obey their actual tensor contracts.
        self.k_scales_row_stride = int(k_scales_row_stride)
        if self.kv_layout not in (KV_LAYOUT_CONTIGUOUS_MLA, KV_LAYOUT_PAGED):
            raise ValueError(f"bad kv_layout {self.kv_layout}")
        self.paged_output = bool(paged_output)
        if self.paged_output and self.kv_layout != KV_LAYOUT_PAGED:
            raise ValueError("physical-slot output requires the paged K/V layout")
        if self.topk not in _SUPPORTED_TOPK:
            raise ValueError(
                f"fused indexer supports topk {_SUPPORTED_TOPK}, got {self.topk}"
            )
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                f"fused indexer supports 1/2/4 head tiles, got {self.num_q_head_tiles}"
            )
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        # All-warp scoring: spread the page's 64 tokens across as many warps as the
        # head-tiling leaves free, scoring the whole page in ONE pass instead of the
        # 4-warp / page_splits-deep sequential loop. token_groups covers the page
        # (8 tokens/group => 8 groups) bounded by warps available after head-tiling.
        # heads=16 (tiles=1): 8 warps; heads=32 (tiles=2): 16; heads=64 (tiles=4): 32.
        total_warps = _RADIX_THREADS // _WARP_THREADS
        self.token_groups = min(
            _PAGE_SIZE // _PAGED_TOKENS_PER_GROUP,  # 8 => one pass over the page
            total_warps // self.num_q_head_tiles,  # warps free after head-tiling
        )
        self.tokens_per_work = _PAGED_TOKENS_PER_GROUP * self.token_groups
        self.page_splits = _PAGE_SIZE // self.tokens_per_work
        # Warp-owns-tokens direct-L2 score: no K staging, no page barriers.
        # Gated to the 4-head-tile paged decode contract (DSV4 64-head) at
        # rows >= 3 (ctas_per_group <= 16): those rows re-read an L2-resident
        # K where 16B fragment loads are free. rows 1-2 stream mostly-cold K
        # from DRAM, where the staged pipeline's contiguous 128B page loads
        # win (measured +25% at rows=1, +3.5% at rows=2); they keep the
        # historical path.
        self.direct_k_score = (
            _DIRECT_K_SCORE_ENABLED
            and self.kv_layout == KV_LAYOUT_PAGED
            and self.num_q_head_tiles == 4
            and self.ctas_per_group <= 16
        )
        self.score_warps = self.token_groups * self.num_q_head_tiles
        self.score_threads = self.score_warps * _WARP_THREADS
        # Over-sized accumulator: scored tokens append until topk+slack, then
        # trim.  Keep the large measured slack specific to the paged DSV4
        # contract so top-k-2048 and contiguous contracts retain their existing
        # shared-memory residency.
        batch_slack = _resolve_fused_batch_slack(
            kv_layout=self.kv_layout,
            num_heads=self.num_heads_static,
            topk=self.topk,
        )
        self.carry_cap = int(self.topk) + int(batch_slack)
        # The trim radix runs over <= carry_cap elements, so its threshold-bin
        # candidate set is bounded by carry_cap -> size cand0/cand1 to that.
        self.cands = self.carry_cap
        # In-kernel cross-CTA merge (ctas_per_group > 1): each CTA atomically
        # packs its REAL local candidates into a per-group global slab of
        # capacity ctas_per_group * topk; the last-arriving CTA radix-selects the
        # final top-k over the packed reals (no host merge launch).
        self.merge_in_kernel = self.ctas_per_group > 1
        # Cooperative grid-barrier merge (_fused_group_barrier) spins until every one of
        # the group's ctas_per_group CTAs has arrived, so they must all be co-resident
        # at once. This kernel launches one CTA per SM (1024 threads + min_blocks_per_mp
        # =1), so a group larger than the SM count can never be fully resident and the
        # barrier deadlocks the device. When that happens, drop to the serial last-CTA
        # arm, which packs to a global slab and exits (no barrier) and therefore
        # tolerates arbitrary grid oversubscription. num_sms <= 0 means "unknown" and
        # applies no guard (the ctas=1 FLAT path and direct constructions).
        self.num_sms = int(num_sms)
        self.coop_co_resident = self.num_sms <= 0 or self.ctas_per_group <= self.num_sms
        self.coop_merge_possible = (
            self.merge_in_kernel
            and self.max_seq_capacity > self.merge_threshold
            and self.coop_co_resident
        )
        self.pack_cap = self.ctas_per_group * int(self.topk)
        # Both cross-CTA merge arms are compiled and chosen at runtime per group by
        # seq_len vs merge_threshold (see the merge dispatch). Both index the
        # per-group _COOP_STATE_WORDS state block, and the last-CTA arm also needs
        # the global pack slab, so size state at _COOP_STATE_WORDS whenever the
        # in-kernel merge is active.
        self.state_words_per_group = _COOP_STATE_WORDS if self.merge_in_kernel else 2

    def _get_shared_storage_cls(self):
        return _fused_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
            self.topk,
            self.carry_cap,
            self.cands,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        out_indices: cute.Tensor,
        out_values: cute.Tensor,
        pack_values: cute.Tensor,
        pack_indices: cute.Tensor,
        merge_state: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_bytes,
            weights,
            k_quant_bytes,
            k_scales,
            real_page_table,
            seqlens_per_query,
            k_start,
            k_end,
            out_indices,
            out_values,
            pack_values,
            pack_indices,
            merge_state,
        ).launch(
            grid=(q_bytes.shape[0] * self.ctas_per_group, 1, 1),
            block=[_RADIX_THREADS, 1, 1],
            # The 1024-thread radix CTA is architecturally limited to one
            # resident block on SM120 (1536 threads/SM).  The cooperative
            # planner also caps the default launch to one machine-wide wave.
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        out_indices: cute.Tensor,
        out_values: cute.Tensor,
        pack_values: cute.Tensor,
        pack_indices: cute.Tensor,
        merge_state: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        bid = Int32(bid)
        # Relay: ctas_per_group CTAs cooperate on one row (group). Each owns a
        # page-slice and emits a LOCAL top-k into candidate[cta_in_group, group, :];
        # the host merges the per-CTA partials. ctas_per_group==1 => single-CTA.
        ctas_pg = Int32(self.ctas_per_group)
        group_id = bid // ctas_pg
        cta_in_group = bid - group_id * ctas_pg
        q_idx = group_id
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)
        topk_static = Int32(self.topk)
        num_heads = Int32(self.num_heads_static)
        # All-warp scoring: warps [0, score_warps) run the MMA + reduce (the page
        # is covered in one pass). K-load / repack stay at _PAGED_THREADS_PER_CTA.
        score_threads = Int32(self.score_threads)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._get_shared_storage_cls())

        s_w = storage.weights.get_tensor(
            cute.make_layout((self.padded_q_heads,), stride=(1,))
        )
        q_smem_base_addr = shared_ptr_to_u32(storage.q_bytes.data_ptr())
        k_page_base_addr = shared_ptr_to_u32(storage.k_page.data_ptr())
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        k_page3_base_addr = shared_ptr_to_u32(storage.k_page3.data_ptr())
        s_k_page_stage = storage.k_page.get_tensor(
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM, 1),
                stride=(_INDEX_HEAD_DIM, 1, _PAGE_SIZE * _INDEX_HEAD_DIM),
            )
        )
        s_scale = storage.scales.get_tensor(
            cute.make_layout((_PAGE_SIZE,), stride=(1,))
        )
        s_partial_logits = storage.partial_logits.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        s_partial_logits2 = storage.partial_logits2.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        s_c0_values = storage.carry0_values.get_tensor(
            cute.make_layout((self.carry_cap,), stride=(1,))
        )
        s_c0_gindex = storage.carry0_gindex.get_tensor(
            cute.make_layout((self.carry_cap,), stride=(1,))
        )
        s_c1_values = storage.carry1_values.get_tensor(
            cute.make_layout((self.topk,), stride=(1,))
        )
        s_c1_gindex = storage.carry1_gindex.get_tensor(
            cute.make_layout((self.topk,), stride=(1,))
        )
        s_hist0 = storage.hist0.get_tensor(cute.make_layout((384,), stride=(1,)))
        s_hist1 = storage.hist1.get_tensor(cute.make_layout((384,), stride=(1,)))
        s_out = storage.out_idx.get_tensor(cute.make_layout((self.topk,), stride=(1,)))
        s_cand0 = storage.cand0.get_tensor(cute.make_layout((self.cands,), stride=(1,)))
        s_cand1 = storage.cand1.get_tensor(cute.make_layout((self.cands,), stride=(1,)))
        s_relay = storage.relay.get_tensor(cute.make_layout((2,), stride=(1,)))
        s_coop_suffix = storage.coop_suffix.get_tensor(
            cute.make_layout((_RADIX,), stride=(1,))
        )
        s_coop_scalars = storage.coop_scalars.get_tensor(
            cute.make_layout((4,), stride=(1,))
        )
        s_coop_counters = storage.coop_counters.get_tensor(
            cute.make_layout((2,), stride=(1,))
        )
        # coop local histogram reuses hist0 (>=256); its u32 base address for atomics:
        coop_hist_addr = shared_ptr_to_u32(storage.hist0.data_ptr())
        coop_ctr_addr = shared_ptr_to_u32(storage.coop_counters.data_ptr())
        h0 = shared_ptr_to_u32(storage.hist0.data_ptr())
        ctr = shared_ptr_to_u32(storage.counter.data_ptr())
        thr = shared_ptr_to_u32(storage.thr_id.data_ptr())
        ni0 = shared_ptr_to_u32(storage.ni0.data_ptr())
        ni1 = shared_ptr_to_u32(storage.ni1.data_ptr())
        lr = shared_ptr_to_u32(storage.last_rem.data_ptr())

        # Q vectors and head weights do not depend on the row length, so their
        # loads are issued before the seq_len -> page-table -> page-fetch
        # chain and complete while it resolves; they are stored to shared
        # memory once the first pages are in flight. Thread ``t`` owns Q
        # vector ``t`` (16 bytes of one head) in the ldmatrix-permuted
        # per-head-tile layout and weight ``t``.
        q_v0 = Uint32(0)
        q_v1 = Uint32(0)
        q_v2 = Uint32(0)
        q_v3 = Uint32(0)
        q_dst_addr = Int32(0)
        q_owner = Int32(0)
        if cutlass.const_expr(self.vectorized_q_load):
            if tx < Int32(self.padded_q_heads * (_INDEX_HEAD_DIM // 16)):
                q_owner = Int32(1)
                q_head = tx // Int32(_INDEX_HEAD_DIM // 16)
                q_vec = tx - q_head * Int32(_INDEX_HEAD_DIM // 16)
                if q_head < num_heads:
                    q_v0, q_v1, q_v2, q_v3 = ld_global_v4_u32(
                        get_ptr_as_int64(
                            q_bytes,
                            Int64(q_idx) * Int64(self.q_row_stride_bytes)
                            + Int64(q_head) * Int64(_INDEX_HEAD_DIM)
                            + Int64(q_vec) * Int64(16),
                        )
                    )
                q_tile = q_head // Int32(_PAGED_Q_HEAD_TILE)
                q_dst_addr = _smem_addr_from_b128_offset(
                    q_smem_base_addr + q_tile * Int32(_PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM),
                    _permuted_offset_128b(
                        q_head - q_tile * Int32(_PAGED_Q_HEAD_TILE),
                        q_vec,
                        Int32(_INDEX_HEAD_DIM // 16),
                    ),
                )
        w_val = Float32(0.0)
        if tx < Int32(self.padded_q_heads):
            if tx < num_heads:
                w_val = Float32(weights[q_idx, tx])

        # PAGED: valid k = [0, seqlen) via page table. CONTIGUOUS_MLA: contiguous k =
        # [k_start, k_end) with abs_start = k_start (no page table).
        abs_start = Int32(0)
        if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
            seq_len = Int32(seqlens_per_query[q_idx])
        else:
            abs_start = Int32(k_start[q_idx])
            seq_len = Int32(k_end[q_idx]) - abs_start
        if seq_len < Int32(0):
            seq_len = Int32(0)
        total_pages = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)
        # This CTA's page-slice of the row (balanced split across ctas_pg CTAs).
        pages_per_cta = (total_pages + ctas_pg - Int32(1)) // ctas_pg
        page_start = cta_in_group * pages_per_cta
        page_end = page_start + pages_per_cta
        if page_end > total_pages:
            page_end = total_pages
        if page_start > total_pages:
            page_start = total_pages

        # Page ids of the slice are read two pages ahead of their use (pid_cur
        # and pid_p1 hold pages p and p+1 at the top of iteration p) so the
        # page-table load never sits between a landed page and the next fetch
        # issue. The first page is issued before Q staging so its fetch
        # overlaps the Q and weight staging.
        scales_base_addr = shared_ptr_to_u32(storage.scales.data_ptr())
        pid_cur = Int32(-1)
        pid_p1 = Int32(-1)
        pid_p2 = Int32(-1)
        if cutlass.const_expr(
            self.kv_layout == KV_LAYOUT_PAGED and not self.direct_k_score
        ):
            pid_cur = _page_id_or_invalid(real_page_table, q_idx, page_start, page_end)
            pid_p1 = _page_id_or_invalid(
                real_page_table, q_idx, page_start + Int32(1), page_end
            )
            if pid_cur >= Int32(0):
                _stream_issue_k_page_cp_async(
                    k_quant_bytes,
                    k_scales,
                    pid_cur,
                    k_page_perm_base_addr,
                    scales_base_addr,
                    Int32(0),
                    tx,
                    k_quant_page_stride=int(self.k_quant_page_stride),
                    k_scales_row_stride=int(self.k_scales_row_stride),
                    issue_threads=_RADIX_THREADS,
                )
            cute.arch.cp_async_commit_group()

        # Stage q + weights. Q lands straight in the ldmatrix-permuted
        # per-head-tile layout so the score core consumes raw fragments on
        # BOTH operands (no byte packs, no 16b->8b fragment swizzles); heads
        # past num_heads stage zeros.
        if cutlass.const_expr(self.vectorized_q_load):
            if q_owner != Int32(0):
                st_shared_v4_u32(q_dst_addr, q_v0, q_v1, q_v2, q_v3)
        else:
            # Preserve arbitrary tensor head/row strides.  Each thread moves
            # individual bytes directly into the same XOR-permuted granules as
            # the vectorized loader; this is the correctness fallback for views
            # whose 128-byte head rows are not 16-byte vector-load compatible.
            q_linear = tx
            total_q_bytes = Int32(self.padded_q_heads * _INDEX_HEAD_DIM)
            while q_linear < total_q_bytes:
                head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
                vec_idx = col_idx // Int32(16)
                byte_idx = col_idx - vec_idx * Int32(16)
                tile_idx = head_idx // Int32(_PAGED_Q_HEAD_TILE)
                row_in_tile = head_idx - tile_idx * Int32(_PAGED_Q_HEAD_TILE)
                tile_base = q_smem_base_addr + tile_idx * Int32(
                    _PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM
                )
                dst_addr = (
                    _smem_addr_from_b128_offset(
                        tile_base,
                        _permuted_offset_128b(
                            row_in_tile, vec_idx, Int32(_INDEX_HEAD_DIM // 16)
                        ),
                    )
                    + byte_idx
                )
                q_byte = cutlass.Uint8(0)
                if head_idx < num_heads:
                    q_byte = q_bytes[q_idx, head_idx, col_idx]
                st_shared_u8(dst_addr, q_byte)
                q_linear += Int32(_RADIX_THREADS)
        if tx < Int32(self.padded_q_heads):
            s_w[tx] = w_val
        cute.arch.sync_threads()

        head_tile_slot = warp_idx % Int32(self.num_q_head_tiles)
        token_group = warp_idx // Int32(self.num_q_head_tiles)

        if cutlass.const_expr(self.direct_k_score):
            # Warp-owns-tokens score: every warp streams 8 consecutive tokens
            # per round with K fragments read straight from L2 (no staging,
            # no producer pipeline, no per-page barrier). Slot bookkeeping is
            # uniform register arithmetic, so different rounds write disjoint
            # carry regions and the only CTA rendezvous are trims.
            carry_count = Int32(0)
            span_first = page_start * Int32(_PAGE_SIZE)
            span_end_tok = page_end * Int32(_PAGE_SIZE)
            if span_end_tok > seq_len:
                span_end_tok = seq_len
            span = span_end_tok - span_first
            if span < Int32(0):
                span = Int32(0)
            lane4 = lane % Int32(4)
            round_base = Int32(0)
            while round_base < span:
                round_valid = span - round_base
                if round_valid > Int32(256):
                    round_valid = Int32(256)
                if carry_count + Int32(256) > Int32(self.carry_cap):
                    cute.arch.sync_threads()
                    _fused_radix_select(
                        s_c0_values,
                        s_c0_gindex,
                        s_c1_values,
                        s_c1_gindex,
                        carry_count,
                        topk_static,
                        self.cands,
                        tx,
                        s_hist0,
                        s_hist1,
                        s_out,
                        s_cand0,
                        s_cand1,
                        h0,
                        ctr,
                        thr,
                        ni0,
                        ni1,
                        lr,
                    )
                    cute.arch.sync_threads()
                    ti = Int32(tx)
                    while ti < topk_static:
                        s_c0_values[ti] = Float32(s_c1_values[ti])
                        s_c0_gindex[ti] = Int32(s_c1_gindex[ti])
                        ti += Int32(_RADIX_THREADS)
                    cute.arch.sync_threads()
                    carry_count = topk_static
                my_rel = round_base + warp_idx * Int32(8)
                pid = Int32(-1)
                tip0 = Int32(0)
                if my_rel < span:
                    my_page = my_rel // Int32(_PAGE_SIZE)
                    tip0 = my_rel - my_page * Int32(_PAGE_SIZE)
                    pcol = page_start + my_page
                    if pcol < Int32(real_page_table.shape[1]):
                        pid = Int32(real_page_table[q_idx, pcol])
                t0 = Float32(0.0)
                t1 = Float32(0.0)
                if pid >= Int32(0):
                    k_off = Int64(pid) * Int64(self.k_quant_page_stride) + Int64(
                        (tip0 + lane // Int32(4)) * Int32(_INDEX_HEAD_DIM)
                        + lane4 * Int32(4)
                    )
                    t0, t1 = _score_tokens_direct_k(
                        q_smem_base_addr,
                        s_w,
                        num_heads,
                        k_quant_bytes,
                        k_off,
                        lane,
                        self.num_q_head_tiles,
                    )
                if lane < Int32(4):
                    for cc in cutlass.range_constexpr(2):
                        col = Int32(2) * lane4 + Int32(cc)
                        tok_rel = my_rel + col
                        if tok_rel < span:
                            val = Float32(-3.4028235e38)
                            gidx = Int32(-1)
                            if pid >= Int32(0):
                                if cutlass.const_expr(cc == 0):
                                    tsel = t0
                                else:
                                    tsel = t1
                                scale_addr = get_ptr_as_int64(
                                    k_scales,
                                    Int64(pid) * Int64(self.k_scales_row_stride)
                                    + Int64(tip0 + col),
                                )
                                val = Float32(tsel * ld_global_nc_f32(scale_addr))
                                if cutlass.const_expr(self.paged_output):
                                    gidx = pid * Int32(_PAGE_SIZE) + tip0 + col
                                else:
                                    gidx = span_first + tok_rel
                            s_c0_values[carry_count + tok_rel - round_base] = val
                            s_c0_gindex[carry_count + tok_rel - round_base] = gidx
                carry_count = carry_count + round_valid
                round_base += Int32(256)
            cute.arch.sync_threads()
            if carry_count > topk_static:
                _fused_radix_select(
                    s_c0_values,
                    s_c0_gindex,
                    s_c1_values,
                    s_c1_gindex,
                    carry_count,
                    topk_static,
                    self.cands,
                    tx,
                    s_hist0,
                    s_hist1,
                    s_out,
                    s_cand0,
                    s_cand1,
                    h0,
                    ctr,
                    thr,
                    ni0,
                    ni1,
                    lr,
                )
                cute.arch.sync_threads()
                ti = Int32(tx)
                while ti < topk_static:
                    s_c0_values[ti] = Float32(s_c1_values[ti])
                    s_c0_gindex[ti] = Int32(s_c1_gindex[ti])
                    ti += Int32(_RADIX_THREADS)
                cute.arch.sync_threads()
                carry_count = topk_static
        else:
            carry_count = Int32(0)
            # Paged K pipeline: page 0 of the slice was issued before Q
            # staging; iteration p issues page p+1 into the K stage page p-2
            # used (the 3-stage ring holds p-1, p, and p+1), then waits for
            # page p. Deferred-reduce flow (page_splits == 1): page p's
            # reduce and carry append run at iteration p+1 behind the
            # pipeline barrier, so one barrier per page orders both the K
            # stage and the partials handoff, and page p-1's scales stay live
            # for it (4-slot scales ring).
            k_stage = Int32(0)
            sc_slot = Int32(0)
            cur_scale_addr = scales_base_addr
            prev_do = Int32(0)
            prev_valid = Int32(0)
            prev_out_base = Int32(0)
            prev_scale_addr = scales_base_addr
            prev_pl2 = Int32(0)
            cur_pl2 = Int32(0)
            page_col = page_start
            while page_col < page_end:
                page_base = page_col * Int32(_PAGE_SIZE)
                valid_slots = seq_len - page_base
                if valid_slots > Int32(_PAGE_SIZE):
                    valid_slots = Int32(_PAGE_SIZE)
                n_new = Int32(0)
                do_page = Int32(0)
                page_id = Int32(0)
                if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
                    page_id = pid_cur
                    if (page_id >= Int32(0)) & (valid_slots > Int32(0)):
                        do_page = Int32(1)
                    # Read the id of page p+2 now; it is issued at iteration p+1.
                    pid_p2 = _page_id_or_invalid(
                        real_page_table, q_idx, page_col + Int32(2), page_end
                    )
                else:
                    if valid_slots > Int32(0):
                        do_page = Int32(1)

                cur_k_base = _k_stage_addr(
                    k_stage, k_page_perm_base_addr, k_page_base_addr, k_page3_base_addr
                )
                cur_scale_base = scales_base_addr
                if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
                    nxt_valid = Int32(0)
                    if (page_col + Int32(1) < page_end) & (pid_p1 >= Int32(0)):
                        nxt_valid = Int32(1)
                    next_stage = k_stage + Int32(1)
                    if next_stage == Int32(_K_STAGES):
                        next_stage = Int32(0)
                    nsc = sc_slot + Int32(1)
                    if nsc == Int32(4):
                        nsc = Int32(0)
                    if nxt_valid != Int32(0):
                        _stream_issue_k_page_cp_async(
                            k_quant_bytes,
                            k_scales,
                            pid_p1,
                            _k_stage_addr(
                                next_stage,
                                k_page_perm_base_addr,
                                k_page_base_addr,
                                k_page3_base_addr,
                            ),
                            scales_base_addr + nsc * Int32(_PAGE_SIZE * 4),
                            Int32(0),
                            tx,
                            k_quant_page_stride=int(self.k_quant_page_stride),
                            k_scales_row_stride=int(self.k_scales_row_stride),
                            issue_threads=_RADIX_THREADS,
                        )
                    cute.arch.cp_async_commit_group()
                    # Page p has landed once every group but the newest has
                    # completed; the barrier then orders page p-1's partials
                    # for the deferred reduce.
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()
                    cur_scale_base = cur_scale_addr
                    if cutlass.const_expr(self.page_splits == 1):
                        # Reduce + append the PREVIOUS page: its partials (all
                        # head tiles) and its scales are ordered by the pipeline
                        # barrier above; the score of the current page then runs
                        # into the other partials buffer with no trailing barrier.
                        if prev_do != Int32(0):
                            if tx < score_threads:
                                if (head_tile_slot == Int32(0)) & (
                                    lane < Int32(_PAGED_TOKENS_PER_GROUP)
                                ):
                                    slot_idx = (
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                                        + lane
                                    )
                                    if slot_idx < prev_valid:
                                        logit = Float32(0.0)
                                        h_i = Int32(0)
                                        if prev_pl2 == Int32(0):
                                            while h_i < Int32(self.num_q_head_tiles):
                                                logit = Float32(
                                                    logit
                                                    + s_partial_logits[slot_idx, h_i]
                                                )
                                                h_i += Int32(1)
                                        else:
                                            while h_i < Int32(self.num_q_head_tiles):
                                                logit = Float32(
                                                    logit
                                                    + s_partial_logits2[slot_idx, h_i]
                                                )
                                                h_i += Int32(1)
                                        s_c0_values[carry_count + slot_idx] = Float32(
                                            logit
                                            * ld_shared_f32(
                                                prev_scale_addr + slot_idx * Int32(4)
                                            )
                                        )
                                        s_c0_gindex[carry_count + slot_idx] = (
                                            prev_out_base + slot_idx
                                        )
                            carry_count = carry_count + prev_valid
                            prev_do = Int32(0)
                if do_page != Int32(0):
                    n_new = valid_slots
                    if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
                        pass
                    else:
                        # CONTIGUOUS_MLA: masked wide scalar load into linear staging, then repack.
                        _load_flat_k_tile_wide(
                            k_quant_bytes,
                            abs_start + page_base,
                            valid_slots,
                            s_k_page_stage,
                            tx,
                            Int32(_RADIX_THREADS),
                        )
                        scale_idx = tx
                        while scale_idx < Int32(_PAGE_SIZE):
                            sv = Float32(0.0)
                            if scale_idx < valid_slots:
                                sv = Float32(
                                    k_scales[abs_start + page_base + scale_idx]
                                )
                            s_scale[scale_idx] = sv
                            scale_idx += Int32(_RADIX_THREADS)
                        cute.arch.sync_threads()
                        _repack_k_page_wide(
                            k_page_base_addr,
                            k_page_perm_base_addr,
                            tx,
                            Int32(_RADIX_THREADS),
                        )
                        cute.arch.sync_threads()

                    if cutlass.const_expr(
                        self.kv_layout == KV_LAYOUT_PAGED and self.page_splits == 1
                    ):
                        tb_defer = token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                        if tx < score_threads:
                            if tb_defer < valid_slots:
                                htb_defer = head_tile_slot * Int32(_PAGED_Q_HEAD_TILE)
                                if cur_pl2 == Int32(0):
                                    _compute_mxfp8_tile_partials_qldm(
                                        q_smem_base_addr,
                                        s_w,
                                        num_heads,
                                        cur_k_base,
                                        tb_defer,
                                        htb_defer,
                                        head_tile_slot,
                                        lane,
                                        s_partial_logits,
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                        head_tile_slot,
                                    )
                                else:
                                    _compute_mxfp8_tile_partials_qldm(
                                        q_smem_base_addr,
                                        s_w,
                                        num_heads,
                                        cur_k_base,
                                        tb_defer,
                                        htb_defer,
                                        head_tile_slot,
                                        lane,
                                        s_partial_logits2,
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                        head_tile_slot,
                                    )
                        prev_do = Int32(1)
                        prev_valid = valid_slots
                        prev_scale_addr = cur_scale_addr
                        prev_pl2 = cur_pl2
                        cur_pl2 = cur_pl2 ^ Int32(1)
                        prev_out_base = abs_start + page_base
                        if cutlass.const_expr(self.paged_output):
                            prev_out_base = page_id * Int32(_PAGE_SIZE)
                    else:
                        split_idx = Int32(0)
                        while split_idx < Int32(self.page_splits):
                            # No partial-logits pre-zero: every group with token_base <
                            # valid_slots runs the MMA (writing all its token columns for
                            # all head tiles), and the reduce only reads slot_idx <
                            # valid_slots -> it never consumes a stale (unwritten) partial.
                            token_base = split_idx * Int32(
                                self.tokens_per_work
                            ) + token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                            if tx < score_threads:
                                if token_base < valid_slots:
                                    head_tile_base = head_tile_slot * Int32(
                                        _PAGED_Q_HEAD_TILE
                                    )
                                    _compute_mxfp8_tile_partials_qldm(
                                        q_smem_base_addr,
                                        s_w,
                                        num_heads,
                                        cur_k_base,
                                        token_base,
                                        head_tile_base,
                                        head_tile_slot,
                                        lane,
                                        s_partial_logits,
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                        head_tile_slot,
                                    )
                            cute.arch.sync_threads()
                            if tx < score_threads:
                                if (head_tile_slot == Int32(0)) & (
                                    lane < Int32(_PAGED_TOKENS_PER_GROUP)
                                ):
                                    slot_idx = token_base + lane
                                    if slot_idx < valid_slots:
                                        logit = Float32(0.0)
                                        partial_row = (
                                            token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                                            + lane
                                        )
                                        h_i = Int32(0)
                                        while h_i < Int32(self.num_q_head_tiles):
                                            logit = Float32(
                                                logit
                                                + s_partial_logits[partial_row, h_i]
                                            )
                                            h_i += Int32(1)
                                        s_c0_values[carry_count + slot_idx] = Float32(
                                            logit
                                            * ld_shared_f32(
                                                cur_scale_base + slot_idx * Int32(4)
                                            )
                                        )
                                        output_idx = abs_start + page_base + slot_idx
                                        if cutlass.const_expr(
                                            self.kv_layout == KV_LAYOUT_PAGED
                                            and self.paged_output
                                        ):
                                            output_idx = (
                                                page_id * Int32(_PAGE_SIZE) + slot_idx
                                            )
                                        s_c0_gindex[carry_count + slot_idx] = output_idx
                            # page_splits==1 (always for 1/2/4 head tiles): no post-reduce
                            # barrier — the next page's load barrier (or the trim's leading
                            # sync, or the pre-output sync) orders this reduce before any
                            # later writer of s_partial_logits / reader of the carry. Multi-
                            # split would need the barrier between splits, so keep it then.
                            if cutlass.const_expr(self.page_splits > 1):
                                cute.arch.sync_threads()
                            split_idx += Int32(1)
                # ---- accumulate + conditional trim ----
                is_last = page_col == (page_end - Int32(1))
                need_trim = Int32(0) != Int32(0)
                if cutlass.const_expr(
                    self.kv_layout == KV_LAYOUT_PAGED and self.page_splits == 1
                ):
                    # Deferred flow: the count already advanced at the deferred
                    # append; the current page appends next iteration (or in the
                    # post-loop tail), so only capacity pressure trims here -- the
                    # final trim runs after the loop.
                    need_trim = carry_count + Int32(_PAGE_SIZE) > Int32(self.carry_cap)
                else:
                    # This page's n_new scored tokens were written into carry0[carry_count:].
                    if do_page != Int32(0):
                        carry_count = carry_count + n_new
                    # Trim carry0 back to topk only when the next page would overflow the
                    # over-sized accumulator, or at the very end. This runs the radix once
                    # per ~(_BATCH_SLACK/_PAGE_SIZE) pages instead of every page.
                    need_trim = (
                        carry_count + Int32(_PAGE_SIZE) > Int32(self.carry_cap)
                    ) | (is_last & (carry_count > topk_static))
                if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
                    k_stage = k_stage + Int32(1)
                    if k_stage == Int32(_K_STAGES):
                        k_stage = Int32(0)
                    pid_cur = pid_p1
                    pid_p1 = pid_p2
                    sc_slot = sc_slot + Int32(1)
                    if sc_slot == Int32(4):
                        sc_slot = Int32(0)
                    cur_scale_addr = scales_base_addr + sc_slot * Int32(_PAGE_SIZE * 4)
                if need_trim:
                    cute.arch.sync_threads()
                    _fused_radix_select(
                        s_c0_values,
                        s_c0_gindex,
                        s_c1_values,
                        s_c1_gindex,
                        carry_count,
                        topk_static,
                        self.cands,
                        tx,
                        s_hist0,
                        s_hist1,
                        s_out,
                        s_cand0,
                        s_cand1,
                        h0,
                        ctr,
                        thr,
                        ni0,
                        ni1,
                        lr,
                    )
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < topk_static:
                        s_c0_values[i] = Float32(s_c1_values[i])
                        s_c0_gindex[i] = Int32(s_c1_gindex[i])
                        i += Int32(_RADIX_THREADS)
                    cute.arch.sync_threads()
                    carry_count = topk_static
                page_col += Int32(1)

            cute.arch.sync_threads()
            if cutlass.const_expr(
                self.kv_layout == KV_LAYOUT_PAGED and self.page_splits == 1
            ):
                # Drain the deferred pipeline: the last page's partials are ordered
                # by the barrier above; append it, then run the final trim the
                # in-loop path no longer performs.
                if prev_do != Int32(0):
                    if tx < score_threads:
                        if (head_tile_slot == Int32(0)) & (
                            lane < Int32(_PAGED_TOKENS_PER_GROUP)
                        ):
                            slot_idx = (
                                token_group * Int32(_PAGED_TOKENS_PER_GROUP) + lane
                            )
                            if slot_idx < prev_valid:
                                logit = Float32(0.0)
                                h_i = Int32(0)
                                if prev_pl2 == Int32(0):
                                    while h_i < Int32(self.num_q_head_tiles):
                                        logit = Float32(
                                            logit + s_partial_logits[slot_idx, h_i]
                                        )
                                        h_i += Int32(1)
                                else:
                                    while h_i < Int32(self.num_q_head_tiles):
                                        logit = Float32(
                                            logit + s_partial_logits2[slot_idx, h_i]
                                        )
                                        h_i += Int32(1)
                                s_c0_values[carry_count + slot_idx] = Float32(
                                    logit
                                    * ld_shared_f32(
                                        prev_scale_addr + slot_idx * Int32(4)
                                    )
                                )
                                s_c0_gindex[carry_count + slot_idx] = (
                                    prev_out_base + slot_idx
                                )
                    carry_count = carry_count + prev_valid
                if carry_count > topk_static:
                    cute.arch.sync_threads()
                    _fused_radix_select(
                        s_c0_values,
                        s_c0_gindex,
                        s_c1_values,
                        s_c1_gindex,
                        carry_count,
                        topk_static,
                        self.cands,
                        tx,
                        s_hist0,
                        s_hist1,
                        s_out,
                        s_cand0,
                        s_cand1,
                        h0,
                        ctr,
                        thr,
                        ni0,
                        ni1,
                        lr,
                    )
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < topk_static:
                        s_c0_values[i] = Float32(s_c1_values[i])
                        s_c0_gindex[i] = Int32(s_c1_gindex[i])
                        i += Int32(_RADIX_THREADS)
                    cute.arch.sync_threads()
                    carry_count = topk_static
        if cutlass.const_expr(not self.merge_in_kernel):
            # ---- single CTA per row: carry0 is already the final top-k ----
            i = Int32(tx)
            while i < topk_static:
                is_valid = i < carry_count
                out_values[group_id, i] = (
                    Float32(s_c0_values[i]) if is_valid else Float32(float("-inf"))
                )
                out_indices[group_id, i] = (
                    Int32(s_c0_gindex[i]) if is_valid else Int32(-1)
                )
                i += Int32(_RADIX_THREADS)
        else:
            # ---- in-kernel cross-CTA merge: runtime-selected strategy ----
            # All CTAs in a group share the row's live K length (seq_len, from
            # metadata), so they pick the SAME arm with no extra barrier:
            #   seq_len >  merge_threshold -> cooperative grid-barrier radix over
            #     the SMEM carries (parallel; amortizes the barriers at long K);
            #   seq_len <= merge_threshold -> serial last-CTA reduction over a
            #     packed slab (no grid barriers; cheaper for few-row/short-K).
            # Both arms index the per-group _COOP_STATE_WORDS state block via
            # _fused_state_offset, so groups taking different arms never collide.
            ctas_pg = Int32(self.ctas_per_group)
            barrier_phase = Int32(0)
            cap = Int32(self.pack_cap)
            slab_base = group_id * cap
            pack_v_row = cute.make_tensor(
                pack_values.iterator + slab_base,
                cute.make_layout((self.pack_cap,), stride=(1,)),
            )
            pack_i_row = cute.make_tensor(
                pack_indices.iterator + slab_base,
                cute.make_layout((self.pack_cap,), stride=(1,)),
            )
            # Pre-declare every variable assigned inside the dynamic arms below;
            # cute forbids a None->typed transition inside a dynamic `if`.
            i = Int32(0)
            base = Int32(0)
            off = Int32(0)
            total = Int32(0)
            key = Uint32(0)
            mask = Uint32(0)
            prefix = Uint32(0)
            shift = Uint32(0)
            pivot_bin = Int32(0)
            count_gt = Int32(0)
            digit = Int32(0)
            bin_count = Int32(0)
            winners = Int32(0)
            remaining_k = Int32(0)
            done = Int32(0)
            local_gt = Int32(0)
            local_eq = Int32(0)
            gt_base = Int32(0)
            eq_base = Int32(0)
            pos = Int32(0)
            c = Int32(0)
            coop_possible = Int32(1 if self.coop_merge_possible else 0)
            if (seq_len > Int32(self.merge_threshold)) & (coop_possible != Int32(0)):
                # ---- cooperative grid-barrier radix merge ----
                # Round 0 histograms the top 12 key bits of every candidate,
                # round 1 the next 12 bits of the candidates matching the
                # round-0 pivot, and round 2 the low byte of those matching
                # the round-1 pivot. Each round accumulates the group's
                # histogram into its own global buffer, so one grid barrier
                # per round publishes it, and every CTA then reads the buffer
                # back and derives the same pivot digit. The round-0 barrier
                # also publishes the group's candidate total. Winners above
                # the final pivot prefix and the tie fill at the prefix are
                # placed through two group counters whose bases follow from
                # the histogram, so no barrier separates them, and the
                # refinement stops once the pivot bin holds exactly the
                # remaining slots. The last CTA to leave zeroes the state
                # block. The 4096-bin local histogram lives in the K ring,
                # which is idle once scoring is done.
                if tx == Int32(0):
                    red_add_global_i32(
                        _fused_state_ptr(merge_state, group_id, Int32(_FUSED_STATE_TOTAL)),
                        carry_count,
                    )
                cute.arch.cp_async_wait_group(0)
                wide_hist_addr = k_page_base_addr
                st_shared_v4_u32(
                    wide_hist_addr + tx * Int32(16),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
                if tx < Int32(_RADIX):
                    s_hist0[tx] = Int32(0)
                cute.arch.sync_threads()
                barrier_phase = _coop_wide_round(
                    merge_state,
                    group_id,
                    _FUSED_COARSE0,
                    _FUSED_FINE0,
                    s_c0_values,
                    carry_count,
                    Uint32(0),
                    Uint32(0),
                    20,
                    wide_hist_addr,
                    coop_hist_addr,
                    s_hist0,
                    topk_static,
                    barrier_phase,
                    ctas_pg,
                    tx,
                    lane,
                    warp_idx,
                    s_coop_scalars,
                    True,
                )
                total = Int32(s_coop_scalars[3])
                if total <= topk_static:
                    # Every candidate survives: pack at an atomic base; CTA 0 pads.
                    if tx == Int32(0):
                        s_coop_counters[1] = atomic_add_global_i32(
                            _fused_state_ptr(
                                merge_state, group_id, Int32(_FUSED_STATE_OUTPUT)
                            ),
                            carry_count,
                        )
                    cute.arch.sync_threads()
                    base = Int32(s_coop_counters[1])
                    i = Int32(tx)
                    while i < carry_count:
                        out_values[group_id, base + i] = Float32(s_c0_values[i])
                        out_indices[group_id, base + i] = Int32(s_c0_gindex[i])
                        i += Int32(_RADIX_THREADS)
                    if cta_in_group == Int32(0):
                        i = total + Int32(tx)
                        while i < topk_static:
                            out_values[group_id, i] = Float32(float("-inf"))
                            out_indices[group_id, i] = Int32(-1)
                            i += Int32(_RADIX_THREADS)
                else:
                    pivot_bin = Int32(s_coop_scalars[0])
                    count_gt = Int32(s_coop_scalars[1])
                    bin_count = Int32(s_coop_scalars[2])
                    prefix = Uint32(pivot_bin) << Uint32(20)
                    mask = Uint32(0xFFF00000)
                    winners = count_gt
                    remaining_k = topk_static - count_gt
                    done = Int32(0)
                    if bin_count == remaining_k:
                        done = Int32(1)
                    if done == Int32(0):
                        barrier_phase = _coop_wide_round(
                            merge_state,
                            group_id,
                            _FUSED_COARSE1,
                            _FUSED_FINE1,
                            s_c0_values,
                            carry_count,
                            mask,
                            prefix,
                            8,
                            wide_hist_addr,
                            coop_hist_addr,
                            s_hist0,
                            remaining_k,
                            barrier_phase,
                            ctas_pg,
                            tx,
                            lane,
                            warp_idx,
                            s_coop_scalars,
                            False,
                        )
                        pivot_bin = Int32(s_coop_scalars[0])
                        count_gt = Int32(s_coop_scalars[1])
                        bin_count = Int32(s_coop_scalars[2])
                        prefix = prefix | (Uint32(pivot_bin) << Uint32(8))
                        mask = Uint32(0xFFFFFF00)
                        winners = winners + count_gt
                        remaining_k = remaining_k - count_gt
                        if bin_count == remaining_k:
                            done = Int32(1)
                    if done == Int32(0):
                        barrier_phase = _coop_narrow_round(
                            merge_state,
                            group_id,
                            _FUSED_COARSE2,
                            s_c0_values,
                            carry_count,
                            mask,
                            prefix,
                            coop_hist_addr,
                            s_hist0,
                            remaining_k,
                            barrier_phase,
                            ctas_pg,
                            tx,
                            lane,
                            warp_idx,
                            s_coop_scalars,
                        )
                        pivot_bin = Int32(s_coop_scalars[0])
                        count_gt = Int32(s_coop_scalars[1])
                        bin_count = Int32(s_coop_scalars[2])
                        prefix = prefix | Uint32(pivot_bin)
                        mask = Uint32(0xFFFFFFFF)
                        winners = winners + count_gt
                        remaining_k = remaining_k - count_gt
                    # Count this CTA's winners and pivot ties, then reserve
                    # their group slots: winners fill [0, winners), ties fill
                    # [winners, topk) in arrival order.
                    if tx == Int32(0):
                        s_coop_counters[0] = Int32(0)
                        s_coop_counters[1] = Int32(0)
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < carry_count:
                        key = _convert_to_uint32(Float32(s_c0_values[i])) & mask
                        if key > prefix:
                            _smem_xadd(coop_ctr_addr, Int32(0), Int32(1))
                        else:
                            if key == prefix:
                                _smem_xadd(coop_ctr_addr, Int32(1), Int32(1))
                        i += Int32(_RADIX_THREADS)
                    cute.arch.sync_threads()
                    if tx == Int32(0):
                        local_gt = Int32(s_coop_counters[0])
                        local_eq = Int32(s_coop_counters[1])
                        # One 64-bit fetch-add reserves the winner slots (low
                        # word) and the tie slots (high word).
                        reserved = atomic_add_global_u64(
                            _fused_state_ptr(
                                merge_state, group_id, Int32(_FUSED_STATE_OUTPUT)
                            ),
                            Int64(local_gt) + (Int64(local_eq) << Int64(32)),
                        )
                        s_coop_scalars[0] = Uint32(
                            (reserved & Int64(0xFFFFFFFF)).to(Int32)
                        )
                        s_coop_scalars[1] = Uint32(
                            winners + (reserved >> Int64(32)).to(Int32)
                        )
                        s_coop_counters[0] = Int32(0)
                        s_coop_counters[1] = Int32(0)
                    cute.arch.sync_threads()
                    gt_base = Int32(s_coop_scalars[0])
                    eq_base = Int32(s_coop_scalars[1])
                    i = Int32(tx)
                    while i < carry_count:
                        key = _convert_to_uint32(Float32(s_c0_values[i])) & mask
                        if key > prefix:
                            pos = gt_base + _smem_xadd(coop_ctr_addr, Int32(0), Int32(1))
                            out_values[group_id, pos] = Float32(s_c0_values[i])
                            out_indices[group_id, pos] = Int32(s_c0_gindex[i])
                        else:
                            if key == prefix:
                                pos = eq_base + _smem_xadd(
                                    coop_ctr_addr, Int32(1), Int32(1)
                                )
                                if pos < topk_static:
                                    out_values[group_id, pos] = Float32(s_c0_values[i])
                                    out_indices[group_id, pos] = Int32(s_c0_gindex[i])
                        i += Int32(_RADIX_THREADS)
                # Departure: once its output is placed, every CTA publishes
                # that it is done with the group state, and the last one
                # zeroes the whole block (round histograms and counters) so
                # the next launch or graph replay starts clean. The fence
                # orders this CTA's reservation and histogram traffic before
                # its departure becomes visible, so the zeroing can never
                # overtake a pending reservation.
                cute.arch.sync_threads()
                if tx == Int32(0):
                    threadfence()
                    s_relay[0] = atomic_add_global_i32(
                        _fused_state_ptr(merge_state, group_id, Int32(_FUSED_STATE_CLEANUP)),
                        Int32(1),
                    )
                cute.arch.sync_threads()
                if Int32(s_relay[0]) == (ctas_pg - Int32(1)):
                    i = Int32(tx)
                    while i < Int32(_COOP_STATE_WORDS):
                        merge_state[_fused_state_offset(group_id, i)] = Int32(0)
                        i += Int32(_RADIX_THREADS)
            else:
                # ---- in-kernel cross-CTA merge (relay) ----
                # Each CTA atomically packs its carry_count REAL candidates into the
                # per-group global slab; the last-arriving CTA radix-selects the final
                # top-k over the packed reals and writes the row (no host merge launch).
                # 1. reserve a contiguous slab slot for this CTA's real candidates.
                #    woff/arrival live in this group's coop-state block (via
                #    _FUSED_STATE_OUTPUT / _FUSED_STATE_ARRIVAL), so a group on
                #    this arm never collides with a sibling group on the coop arm.
                if tx == Int32(0):
                    woff_ptr = _fused_state_ptr(
                        merge_state, group_id, Int32(_FUSED_STATE_OUTPUT)
                    )
                    s_relay[0] = atomic_add_global_i32(woff_ptr, carry_count)
                cute.arch.sync_threads()
                off = Int32(s_relay[0])
                # 2. publish carry0[0:carry_count] into slab[slab_base+off : +count]
                i = Int32(tx)
                while i < carry_count:
                    pack_values[slab_base + off + i] = Float32(s_c0_values[i])
                    pack_indices[slab_base + off + i] = Int32(s_c0_gindex[i])
                    i += Int32(_RADIX_THREADS)
                cute.arch.sync_threads()
                # 3. release the writes, then bump arrival; old==ctas-1 => I am last
                if tx == Int32(0):
                    threadfence()
                    arr_ptr = _fused_state_ptr(
                        merge_state, group_id, Int32(_FUSED_STATE_ARRIVAL)
                    )
                    s_relay[1] = atomic_add_global_i32(arr_ptr, Int32(1))
                cute.arch.sync_threads()
                if Int32(s_relay[1]) == (ctas_pg - Int32(1)):
                    threadfence()  # acquire: observe every producer's packed writes
                    total = Int32(
                        merge_state[
                            _fused_state_offset(group_id, Int32(_FUSED_STATE_OUTPUT))
                        ]
                    )
                    if total > topk_static:
                        _fused_radix_select(
                            pack_v_row,
                            pack_i_row,
                            s_c1_values,
                            s_c1_gindex,
                            total,
                            topk_static,
                            self.cands,
                            tx,
                            s_hist0,
                            s_hist1,
                            s_out,
                            s_cand0,
                            s_cand1,
                            h0,
                            ctr,
                            thr,
                            ni0,
                            ni1,
                            lr,
                        )
                        cute.arch.sync_threads()
                        i = Int32(tx)
                        while i < topk_static:
                            out_values[group_id, i] = Float32(s_c1_values[i])
                            out_indices[group_id, i] = Int32(s_c1_gindex[i])
                            i += Int32(_RADIX_THREADS)
                    else:
                        i = Int32(tx)
                        while i < topk_static:
                            if i < total:
                                out_values[group_id, i] = Float32(pack_v_row[i])
                                out_indices[group_id, i] = Int32(pack_i_row[i])
                            else:
                                out_values[group_id, i] = Float32(float("-inf"))
                                out_indices[group_id, i] = Int32(-1)
                            i += Int32(_RADIX_THREADS)
                    # reset the group's counters so the next launch / graph replay starts clean
                    cute.arch.sync_threads()
                    if tx == Int32(0):
                        merge_state[
                            _fused_state_offset(group_id, Int32(_FUSED_STATE_OUTPUT))
                        ] = Int32(0)
                        merge_state[
                            _fused_state_offset(group_id, Int32(_FUSED_STATE_ARRIVAL))
                        ] = Int32(0)


@lru_cache(maxsize=64)
def _build_fused_indexer_kernel(
    kv_layout: int,
    num_heads_static: int,
    topk: int,
    paged_output: bool,
    ctas_per_group: int,
    num_sms: int = 0,
    merge_threshold: int = _LAST_CTA_MERGE_MAX,
    k_quant_page_stride: int = _PAGE_SIZE * _INDEX_HEAD_DIM,
    k_scales_row_stride: int = _PAGE_SIZE,
    max_seq_capacity: int = 1 << 30,
    vectorized_q_load: bool = False,
    q_row_stride_bytes: int = 0,
):
    return DSAFusedIndexerKernel(
        num_heads_static=num_heads_static,
        topk=topk,
        kv_layout=kv_layout,
        paged_output=paged_output,
        ctas_per_group=ctas_per_group,
        num_sms=num_sms,
        merge_threshold=merge_threshold,
        k_quant_page_stride=k_quant_page_stride,
        k_scales_row_stride=k_scales_row_stride,
        max_seq_capacity=max_seq_capacity,
        vectorized_q_load=vectorized_q_load,
        q_row_stride_bytes=q_row_stride_bytes,
    )


def _arg_sig(*tensors: torch.Tensor):
    return tuple((tuple(t.shape), str(t.dtype)) for t in tensors)


def _fused_indexer_tensor_key(name: str, tensor: torch.Tensor) -> tuple[object, ...]:
    dynamic_row_names = {
        "q",
        "w",
        "pt",
        "sl",
        "kstart",
        "kend",
        "oi",
        "ov",
        "pv",
        "pi",
        "st",
    }
    dynamic_dims = (0,) if name in dynamic_row_names and int(tensor.ndim) >= 1 else ()
    return tensor_compile_fact(name, tensor, dynamic_dims=dynamic_dims)


def _launch_fused(kernel, cute_args, key_tensors, policy, *, launcher=None):
    """Graph-safe launch via b12x_launch (compile-once + replayable run_compiled).

    Unlike a bare cute.compile()+call, the b12x_launch path is CUDA-graph-capturable
    (the existing indexer kernels rely on it). Row-bearing key tensors use dynamic
    row dimensions; policy distinguishes the constexpr variant.  A prepared owner
    supplies the already-resolved launcher; this branch must never consult the
    compiler cache during serving replay.
    """
    # The variant string names the traced kernel body: the direct-K score
    # path and the cross-CTA merge arm (cooperative grid barrier, or the
    # serial last-CTA arm for a group that oversubscribes the SMs, see
    # coop_co_resident) are distinct traces. Its version suffix is bumped
    # whenever the traced body changes, so a cached cubin is never reused
    # for a different kernel.
    if kernel.direct_k_score:
        # direct-K needs ctas_per_group <= 16, co-resident on every real SM12x part,
        # so this keeps its historical key. The suffix keeps the key honest (no trace
        # collision) if a <16-SM device ever oversubscribes a direct-K build.
        variant = (
            "fused_indexer_v10_directk64"
            if kernel.coop_co_resident
            else "fused_indexer_v10_directk64_oversub_serial"
        )
    elif kernel.coop_co_resident:
        variant = "fused_indexer_v8_coop"
    else:
        variant = "fused_indexer_v8_oversub_serial"
    cache_key = tuple(_fused_indexer_tensor_key(name, t) for name, t in key_tensors) + (
        (variant,) + tuple(policy),
    )
    labels = tuple(name for name, _ in key_tensors) + ("policy",)
    compile_spec = KernelCompileSpec.from_key(
        "attention.indexer.fused_indexer", 1, cache_key, labels=labels
    )
    if launcher is not None:
        from b12x._lib.compiler import run_compiled

        return run_compiled(launcher, cute_args)
    return b12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=cute_args,
        runtime_args=cute_args,
    )


def _alloc_merge_scratch(rows: int, topk: int, ctas_per_group: int, dev):
    """Per-group merge state + pack slab for the in-kernel cross-CTA merge.

    ctas_per_group == 1 takes no merge path -> minimal dummies. Otherwise the
    runtime auto-switch can take either arm per group, so allocate BOTH: the
    per-group state at _COOP_STATE_WORDS int32 (cooperative histogram/barrier/
    output/total counters, which the last-CTA arm reuses for its woff/arrival
    words) and the global pack slab (rows * ctas_per_group * topk) the last-CTA
    arm packs into. State is zeroed each call; under CUDA-graph capture the memset
    is captured and replays clean (the per-launch reset).
    """
    if ctas_per_group <= 1:
        pack_v = torch.empty((1,), dtype=torch.float32, device=dev)
        pack_i = torch.empty((1,), dtype=torch.int32, device=dev)
        state = torch.zeros((2,), dtype=torch.int32, device=dev)
    else:
        cap = rows * ctas_per_group * int(topk)
        pack_v = torch.empty((cap,), dtype=torch.float32, device=dev)
        pack_i = torch.empty((cap,), dtype=torch.int32, device=dev)
        state = torch.zeros((rows * _COOP_STATE_WORDS,), dtype=torch.int32, device=dev)
    return pack_v, pack_i, state


def fused_indexer_scratch_capacity(
    max_rows: int, topk: int, num_sms: int
) -> tuple[int, int]:
    """Frugal FIXED capacity for the fused cross-CTA merge scratch -- for a workspace to
    allocate ONCE and reuse, never grow.

    The merge packs each cooperating CTA's local top-k (already trimmed to `topk`) into a
    per-group slab slot. With ctas_per_group = num_sms // rows, the total packed
    candidates = rows * ctas_per_group * topk <= num_sms * topk for ANY rows in
    [1, num_sms]. So the pack slab is bounded by num_sms * topk -- INDEPENDENT of batch
    size AND sequence length (the candidate count is capped by per-CTA top-k trimming,
    not seq_len). This is the whole point: unlike a per-seq gather buffer, the fused merge
    scratch never scales with context/max_model_len, so the frugal constant cap is exact.

    Returns (pack_elems, state_words): pack_values and pack_indices each need pack_elems
    (f32 / i32); merge_state needs state_words int32.
    """
    pack_elems = max(1, int(num_sms)) * int(topk)
    state_words = max(1, int(max_rows)) * _COOP_STATE_WORDS
    return pack_elems, state_words




def _can_vectorize_q_load(q_bytes: torch.Tensor) -> bool:
    if hasattr(q_bytes, "fake_mode") or q_bytes.is_meta:
        # Preparation represents the declared alignment as a storage offset.
        address = q_bytes.storage_offset() * q_bytes.element_size()
    else:
        address = int(q_bytes.data_ptr())
    return (
        address % 16 == 0
        and int(q_bytes.stride(2)) == 1
        and int(q_bytes.stride(1)) == _INDEX_HEAD_DIM
        and int(q_bytes.stride(0)) % 16 == 0
    )


def run_fused_paged_indexer(
    *,
    q_bytes: torch.Tensor,  # uint8 view of fp8 q, [rows, heads, 128]
    weights: torch.Tensor,  # f32, [rows, heads]
    k_quant_bytes: torch.Tensor,  # uint8, [pages, 64, 128]
    k_scales: torch.Tensor,  # f32, [pages, 64]
    real_page_table: torch.Tensor,  # i32, [rows, max_pages]
    seqlens: torch.Tensor,  # i32, [rows]
    num_heads: int,
    topk: int,
    out_indices: torch.Tensor | None = None,
    out_values: torch.Tensor | None = None,
    ctas_per_group: int | None = None,
    merge_threshold: int | None = None,
    pack_values: torch.Tensor | None = None,
    pack_indices: torch.Tensor | None = None,
    merge_state: torch.Tensor | None = None,
    merge_state_preinitialized: bool = False,
    output_physical_slots: bool = False,
    launcher=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paged fused indexer. ctas_per_group>1 splits the row's K across cooperating CTAs.
    Returns (indices, values).

    pack_values/pack_indices/merge_state: optional caller-owned (workspace) scratch for
    serving. Size them with fused_indexer_scratch_capacity(max_rows, topk, num_sms) ONCE
    (fixed capacity, never grows -- the merge scratch is seq-independent). When omitted,
    scratch is allocated per call (benchmarks/tests). A caller-owned merge_state must
    either be zero-initialized once and pass merge_state_preinitialized=True, or it is
    zeroed here for backwards compatibility. Both merge arms self-reset every scalar
    they carry across launches, so a preinitialized state needs no replay-time memset.

    merge_threshold drives the per-group runtime cross-CTA merge auto-switch: a row
    whose live seq_len <= merge_threshold uses the serial last-CTA reduction (no grid
    barriers; faster for few-row/short-K); larger rows use the cooperative
    grid-barrier radix. 0 forces cooperative; a very large value forces last-CTA.
    A group that oversubscribes the SMs (ctas_per_group > num_sms) always takes the
    serial last-CTA arm regardless of merge_threshold: the cooperative grid barrier
    needs every CTA of the group co-resident (1 CTA/SM), impossible past num_sms."""
    rows = int(q_bytes.shape[0])
    dev = q_bytes.device
    max_pages = int(real_page_table.shape[1])
    if ctas_per_group is None:
        ctas_per_group = _resolve_default_ctas_per_group(
            num_rows=rows,
            max_pages=max_pages,
            device=dev,
        )
    ctas_per_group = max(1, int(ctas_per_group))
    if merge_threshold is None:
        merge_threshold = _resolve_default_merge_threshold(
            ctas_per_group=ctas_per_group,
            num_heads=int(num_heads),
            topk=int(topk),
        )

    out_i = (
        torch.empty((rows, topk), dtype=torch.int32, device=dev)
        if out_indices is None
        else out_indices
    )
    out_v = (
        torch.empty((rows, topk), dtype=torch.float32, device=dev)
        if out_values is None
        else out_values
    )
    if pack_values is not None and pack_indices is not None and merge_state is not None:
        # Workspace-owned fixed-capacity scratch. Capacity is validated against
        # fused_indexer_scratch_capacity by the workspace at allocation time.
        pack_need = rows * ctas_per_group * int(topk)
        state_need = rows * _COOP_STATE_WORDS
        if pack_values.numel() < pack_need or pack_indices.numel() < pack_need:
            raise ValueError(
                f"fused indexer pack scratch too small: need {pack_need}, have "
                f"{min(pack_values.numel(), pack_indices.numel())} "
                f"(size via fused_indexer_scratch_capacity)"
            )
        if merge_state.numel() < state_need:
            raise ValueError(
                f"fused indexer merge_state too small: need {state_need}, have "
                f"{merge_state.numel()} (size via fused_indexer_scratch_capacity)"
            )
        pack_v = pack_values.narrow(0, 0, pack_need)
        pack_i = pack_indices.narrow(0, 0, pack_need)
        state = merge_state.narrow(0, 0, state_need)
        if not bool(merge_state_preinitialized):
            state.zero_()
    else:
        pack_v, pack_i, state = _alloc_merge_scratch(rows, topk, ctas_per_group, dev)
    # Real dim-0 byte stride of the K-quant tensor: the packed paged cache interleaves
    # per-page scales (stride 8448), a plain [pages,64,128] view is contiguous (8192).
    # The wide g2s load must use this, not a hardcoded stride, to read the right page.
    k_quant_page_stride = int(k_quant_bytes.stride(0))
    k_scales_row_stride = int(k_scales.stride(0))
    vectorized_q_load = _can_vectorize_q_load(q_bytes)
    q_row_stride_bytes = int(q_bytes.stride(0)) if vectorized_q_load else 0
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    kernel = _build_fused_indexer_kernel(
        KV_LAYOUT_PAGED,
        int(num_heads),
        int(topk),
        bool(output_physical_slots),
        ctas_per_group,
        num_sms=int(num_sms),
        merge_threshold=int(merge_threshold),
        k_quant_page_stride=k_quant_page_stride,
        k_scales_row_stride=k_scales_row_stride,
        max_seq_capacity=max_pages * _PAGE_SIZE,
        vectorized_q_load=vectorized_q_load,
        q_row_stride_bytes=q_row_stride_bytes,
    )
    # k_start/k_end are constexpr-dead in the PAGED kernel. Reuse the existing
    # int32 row metadata instead of capturing a pointless zero-fill kernel.
    unused_k_bounds = seqlens
    args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(weights, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(real_page_table, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(unused_k_bounds, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(unused_k_bounds, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_i, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_v, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(pack_v, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(pack_i, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(state, cutlass.Int32, assumed_align=4),
        current_cuda_stream(),
    )
    key_tensors = [
        ("q", q_bytes),
        ("w", weights),
        ("kq", k_quant_bytes),
        ("ks", k_scales),
        ("pt", real_page_table),
        ("sl", seqlens),
        ("kstart", unused_k_bounds),
        ("kend", unused_k_bounds),
        ("oi", out_i),
        ("ov", out_v),
        ("pv", pack_v),
        ("pi", pack_i),
        ("st", state),
    ]
    _launch_fused(
        kernel,
        args,
        key_tensors,
        (
            KV_LAYOUT_PAGED,
            int(num_heads),
            int(topk),
            bool(output_physical_slots),
            int(ctas_per_group),
            int(merge_threshold),
            k_quant_page_stride,
            max_pages * _PAGE_SIZE,
            k_scales_row_stride,
            bool(vectorized_q_load),
            q_row_stride_bytes,
        ),
        launcher=launcher,
    )
    return out_i, out_v


def run_fused_indexer_mla(
    *,
    q_bytes: torch.Tensor,  # uint8 view of fp8 q, [rows, heads, 128]
    weights: torch.Tensor,  # f32, [rows, heads]
    k_quant_bytes: torch.Tensor,  # uint8, [k_rows, 128] (contiguous MLA K)
    k_scales: torch.Tensor,  # f32, [k_rows]
    k_start: torch.Tensor,  # i32, [rows]
    k_end: torch.Tensor,  # i32, [rows]
    num_heads: int,
    topk: int,
    out_indices: torch.Tensor | None = None,
    out_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v1 FLAT/MLA fused indexer over contiguous K with per-row [k_start, k_end). ctas=1."""
    rows = int(q_bytes.shape[0])
    dev = q_bytes.device
    out_i = (
        torch.empty((rows, topk), dtype=torch.int32, device=dev)
        if out_indices is None
        else out_indices
    )
    out_v = (
        torch.empty((rows, topk), dtype=torch.float32, device=dev)
        if out_values is None
        else out_values
    )
    pack_v, pack_i, state = _alloc_merge_scratch(rows, topk, 1, dev)
    vectorized_q_load = _can_vectorize_q_load(q_bytes)
    q_row_stride_bytes = int(q_bytes.stride(0)) if vectorized_q_load else 0
    kernel = _build_fused_indexer_kernel(
        KV_LAYOUT_CONTIGUOUS_MLA,
        int(num_heads),
        int(topk),
        False,
        1,
        vectorized_q_load=vectorized_q_load,
        q_row_stride_bytes=q_row_stride_bytes,
    )
    # page table / seqlens unused for FLAT; pass minimal dummies.
    dummy_pt = torch.zeros((rows, 1), dtype=torch.int32, device=dev)
    dummy_sl = torch.zeros((rows,), dtype=torch.int32, device=dev)
    args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(weights, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(dummy_pt, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(dummy_sl, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_start, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_end, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_i, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_v, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(pack_v, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(pack_i, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(state, cutlass.Int32, assumed_align=4),
        current_cuda_stream(),
    )
    key_tensors = [
        ("q", q_bytes),
        ("w", weights),
        ("kq", k_quant_bytes),
        ("ks", k_scales),
        ("pt", dummy_pt),
        ("sl", dummy_sl),
        ("kstart", k_start),
        ("kend", k_end),
        ("oi", out_i),
        ("ov", out_v),
        ("pv", pack_v),
        ("pi", pack_i),
        ("st", state),
    ]
    _launch_fused(
        kernel,
        args,
        key_tensors,
        (
            KV_LAYOUT_CONTIGUOUS_MLA,
            int(num_heads),
            int(topk),
            1,
            bool(vectorized_q_load),
            q_row_stride_bytes,
        ),
    )
    return out_i, out_v
