# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight admission for the Qwen 3.8 Flash-Next gated attention block: the geometry, the device / DSL gates, the version probe.

Imports torch, FlashInfer's own probes and the Qwen 3.5 package's light gates only -- never the kernels, the CuTe DSL or the
vendored cuDNN Frontend tree -- so a serving stack's init-time dispatch question (:func:`qwen_3_8_block_support`) costs nothing
when the answer is no.  The three forms -- ``decode`` (the in-block indexer's DECODE step), ``extend`` and ``prefill`` (its EXTEND
step: a uniform chunk behind a cached prefix; ``prefill`` = the same step at an empty prefix) -- are ONE vendored declaration form
(``QsaSpec(index_source="indexer")`` over the pools) and are served together: the decode envelope's decline table and the indexer
facts are READ off the vendored serving envelope once every lighter gate has passed, and the indexer rules this module mirrors from
that tree are re-read against it then (:func:`check_vendored_rules`).

ONE geometry gate, :meth:`Qwen38BlockGeometry.unsupported_reason`, serves both callers -- the probe and ``api.prepare`` (through
:func:`require_served`) -- so a geometry the probe advertises is one the forms' declaration admits, and a geometry it declines is
a typed ``NotImplementedError`` at plan time before any vendored declaration is built (the geometry gate itself runs before any
vendored import; the projection-route gate after it, :func:`proj_route_reason`, asks the vendored tile rule's host arithmetic and
declines the TP 2 / TP 4 shards).  One FlashInfer-side rows bound remains, on EVERY form, at this pin -- the launch-grid bound of ONE
run, :data:`EXTEND_ROWS_GRID_MAX` rows (:func:`extend_rows_grid_reason`): the vendored indexer's query compress -- the DECODE step's and
the EXTEND step's share the launch -- runs with the run's row count as a CUDA grid dimension, so a longer run is declined typed before any
vendored code.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TypeAlias

import torch

from ..qwen_3_5_block import support as _s35
from ..qwen_3_5_block.support import Qwen35BlockSource, cutedsl_version, device_cc

FEATURE = "Qwen 3.8 Flash-Next gated attention block (bf16)"
SUPPORTED_CC: Tuple[int, int] = _s35.SUPPORTED_CC
HEAD_DIM = _s35.HEAD_DIM
CUTEDSL_REQUIRED = _s35.CUTEDSL_REQUIRED
#: the forms this package serves (``ALL_FORMS``' order): ``prefill`` / ``extend`` = the in-block indexer's EXTEND step over the pools (a
#: uniform ``[batch, T]`` chunk behind a cached prefix; ``prefill`` is the same vendored form at an empty prefix, ``seq_lens[b] == T``),
#: ``decode`` = its DECODE step -- ONE vendored declaration form (``QsaSpec(index_source="indexer")`` with ``kv_source="paged"``), so the
#: three are served together by the one geometry gate (the 4c8e24355 pin; before it the two prefill forms declined by name)
FORMS: Tuple[str, ...] = ("prefill", "extend", "decode")
ALL_FORMS: Tuple[str, ...] = ("prefill", "extend", "decode")
#: the forms that run the EXTEND step (``index_select_form="extend"`` pinned on the vendored declaration)
EXTEND_FORMS: Tuple[str, ...] = ("prefill", "extend")
PAGE_SIZES: Tuple[int, ...] = _s35.PAGE_SIZES
PAGE_SIZE_ALIGN = _s35.PAGE_SIZE_ALIGN
#: the cache spellings: FlashInfer's NHD / HND (5-D or a (k, v) pair) and the serving stacks' UNIFIED row ``[num_pages, page_size, H_kv, 2 * D]``
KV_LAYOUTS: Tuple[str, ...] = ("NHD", "HND", "UNIFIED")
DTYPES: Tuple[torch.dtype, ...] = (torch.bfloat16,)
ROPE_DIMS: Tuple[int, ...] = _s35.ROPE_DIMS
_PROJ_TILE = _s35._PROJ_TILE

# The indexer rules this module mirrors from the vendored tree (``gated_attention_block.qsa_select_paged`` / ``qsa_select`` /
# ``QsaSpec.validate`` / the envelope), so the init-time probe stays free of the kernels and the DSL; :func:`check_vendored_rules`
# re-reads every one of them off ``DecodeEnvelope`` (``index_*`` fields) wherever the tree is imported, and ``test_admission.py``
# plants a drift per constant -- a re-vendor that moves a rule fails loudly instead of advertising what the tree declines.
INDEX_HEAD_DIM = 128  # the head dim the paged scorer packs
INDEX_KV_HEADS: Tuple[int, ...] = (
    1,
)  # one raw key head (the checkpoint's own validator)
INDEX_BLOCK_SIZES: Tuple[int, ...] = (
    4,
)  # tokens per selectable block: the sparse loader's one gather granule
INDEX_TOP_K_RANGE: Tuple[int, int, int] = (
    4,
    512,
    4,
)  # (min, max, align) blocks per row
INDEX_QUERY_ROWS_MAX = 64  # S_q x next_pow2(index_heads) padded MMA rows per sequence the scorer keeps resident
INDEX_STATE_ROWS_EXTRA = 3  # the raw-key ring needs cap >= S_q + 3 rows per block (a completing block pools the three positions below the step)
INDEX_PAGE_SIZES: Tuple[int, ...] = (
    PAGE_SIZES  # the pages the paged scorer addresses (a power-of-two count of compressed rows)
)
# The in-block indexer's STEP FORMS over the pools and the EXTEND step's rows, mirrored from the vendored tree's DSL-free modules
# (``gated_attention_block.api`` constants, ``gated_attention_block.cache_write_contract``) and the envelope's ``index_select_forms`` /
# ``index_extend_ring_rows_min`` / ``index_extend_select_budget_bytes`` / ``index_compress_roundings`` rows (the 4c8e24355 pin): the
# vocabulary of the vendored ``index_select_form`` declaration knob (``"auto"`` = the DECODE step inside the decode scorer's query-row
# envelope ``q_len x next_pow2(index_heads) <= INDEX_QUERY_ROWS_MAX`` and the EXTEND step past it; the two pins) -- this package pins
# ``"extend"`` on the ``extend`` / ``prefill`` forms and leaves the decode form at ``"auto"`` (the light gate confines it to the decode step);
# the EXTEND step's ring floor (``cap >= block_size - 1 = 3`` rows per ring block: the straddling block reads at most three positions below
# the chunk; the DECODE step's ``cap >= q_len + 3`` does not apply) -- ONE spelling with ``INDEX_STATE_ROWS_EXTRA`` (asserted by the tripwire);
# the select-window budget (the fp32 compact-logits scratch ONE scorer launch may write: ``ceil(batch x T / R)`` windows of ``R = min(batch x T,
# max(16, budget // (n_blocks_max x 4)))`` rows -- a module constant published in the envelope, never a knob); and the rounding conventions of
# ``Qwen38BlockGeometry.index_compress_rounding`` (``"single"`` the reference chain, ``"hf"`` the HF / vLLM-prepare rounding points).
INDEX_SELECT_FORM_AUTO = "auto"
INDEX_SELECT_FORM_DECODE = "decode"
INDEX_SELECT_FORM_EXTEND = "extend"
INDEX_SELECT_FORMS: Tuple[str, ...] = ("auto", "decode", "extend")
INDEX_EXTEND_RING_ROWS_MIN = 3  # the EXTEND step's ring floor: cap >= block_size - 1 rows per ring block (vLLM's cap 4 satisfies it)
INDEX_EXTEND_SELECT_BUDGET_BYTES = (
    512 << 20
)  # the select-window budget (bytes of fp32 logits scratch per scorer launch)
INDEX_COMPRESS_ROUNDINGS: Tuple[str, ...] = ("single", "hf")
INDEX_COMPRESS_ROUNDING_DEFAULT = "single"
#: the validated decode capture envelope: the band slab GEMM runs the small-M route's 128-row swap-AB tile at ``batch x q_len <= 128``
#: rows -- the vendored ``kernels/proj_gemm.SMALL_M_MAX`` (re-read by the tripwire).  Past 128 rows the slab takes the GEMM engine's own
#: baseline tile for its shape on the SAME JIT-only route since the 6b21be0e0 pin (the projection driver's large-M rule, re-read by the
#: tripwire too: ``band_slab_jit_tile_past_small_m``), so no FlashInfer-side PROJECTION rows bound exists on any form -- the ``extend`` /
#: ``prefill`` forms take any ``batch x T`` the vendored block serves up to the launch-grid bound below (the previous pin's
#: ``EXTEND_ROWS_MAX`` / ``extend_proj_slab_rows`` row is retired).
DECODE_CAPTURE_ROWS_MAX = 128
#: the FlashInfer-side bound of ONE run's rows (``batch_size x q_len_per_req``) on EVERY form at this pin: the vendored indexer's query
#: compress (the DECODE step's and the EXTEND step's execute in ``gated_attention_block/api.py`` -> ``kernels/qsa_compress.run_qsa_compress`` over the
#: ``[rows, index_heads, index_head_dim]`` view of the band) launches with the run's ROW COUNT as the grid's y dimension (one grid row per
#: query token, the heads along x), and CUDA caps a grid's y at 65,535 -- a run of more rows would pass every other gate and fail at launch.
#: Not a projection fact (the band slab names a tile at every row count) and not an envelope row (never among :data:`DECLINE_KEYS`, which
#: the tripwire reads off the envelope): the compressor's grid is flattened in cuDNN Frontend first, then a re-vendor lifts this bound.
#: The remedy: several uniform chunks of at most this many rows per run (extend / prefill), several decode steps, or a smaller batch.
EXTEND_ROWS_GRID_MAX = 65535
#: the FlashInfer-side decline row of that bound (the key keeps its first name; the row is EVERY form's): published under each form
#: record's ``declines`` beside the vendored rows (:func:`extend_rows_grid_text`); ``api.prepare`` raises it typed, with the run's numbers,
#: before any vendored code
EXTEND_GRID_ROWS_KEY = "indexer_extend_grid_rows"
#: the FlashInfer-side decline (not an envelope row): a tensor-parallel shard whose dense projection is off the capture-safe
#: small-M route of the vendored projection GEMM on the device (the probe lists it under this key; ``prepare`` raises it typed)
PROJ_SLAB_ROUTE_KEY = "proj_slab_route"
#: the compressed-key cache dtypes this entry serves on EVERY form (``Qwen38BlockGeometry.index_cache_dtype``; ``None`` = bf16): the
#: vendored tree's e4m3 arms -- the compress kernels' e4m3 store (the prefill / extend writers and the decode step), the paged scorer's
#: e4m3 read arm (decode), the extend step's gather + MXFP8 scorer arm -- are served since the 4c8e24355 pin (one cache, one chain: a
#: decode step reads what a chunk wrote).  The FlashInfer-side ``indexer_e4m3_cache`` decline row of the previous pins is retired.
INDEX_CACHE_DTYPES_SERVED: Tuple[torch.dtype, ...] = (
    torch.bfloat16,
    torch.float8_e4m3fn,
)
#: the decline keys of the README's Errors table every re-vendor must keep publishing under ``decode.declines``
DECLINE_KEYS: Tuple[str, ...] = (
    "indexer_f16",
    "indexer_head_dim",
    "indexer_thd",
    "indexer_query_rows",
    "indexer_page_size",
    "indexer_e4m3_pool",
    "indexer_shared_list",
    "shared_list_rows",
    "fuse_prepare_rope_tables_unfused",
    # the EXTEND step's rows (the 4c8e24355 pin): the texts the vendored declaration raises on the extend / prefill forms
    "indexer_extend_fuse_prepare",
    "indexer_extend_ring_slots",
    "indexer_extend_rope_tables_only",
    "indexer_extend_ring_rows",
    "indexer_extend_chunk_rows",
)
INDEX_LIST_FORMS: Tuple[Optional[str], ...] = (None, "per_row", "shared")
# The sparse core's BODY knob, mirrored from the vendored tree (``sdpa.fwd.sparse_tile_rules``, the DSL-free home of the ``tile_form``
# vocabulary, the two bodies' ``kernel_template`` names and the placement rule since the 977757763 pin -- ``sdpa.fwd.heuristics``
# re-exports them) and re-read against it by :func:`check_vendored_rules`.  The block-sparse decode step on cc 10.7 has two bodies: the
# tcgen05 core (``sparse_d256_f16``) and, since the 7301fad21 pin, the LEAN decode tile (``sparse_lean_d256_f16``: ``mma.sync`` +
# ``cp.async``, one CTA of ``128 x N_MT`` threads per (sequence, KV head, chunk) -- 128 threads per m16 tile, the step's rows riding
# inside the CTA --, no TMEM / TMA / cluster).  ``"auto"`` (this entry's default) takes the lean tile on cc 10.7 when ``batch x H_kv <=
# SPARSE_TILE_LEAN_MAX_UNITS`` CTAs per chunk (the bound's unit since the 977757763 pin: the lean grid's own, so the step width is NOT in
# the bound) at every served ``q_len`` 1..4 where the body admits the step -- under the in-block indexer the list is per row, which the
# lean body serves at ``q_len_per_req == 1`` only, so for this entry that is ``batch <= 2`` at ``q_len_per_req 1`` on two KV heads -- and
# the tcgen05 core everywhere else; a performance-only knob: the same function under either body within the tree's fixed-order-sum
# budget, never bitwise across the bodies.
SPARSE_TILE_FORM_AUTO = "auto"
SPARSE_TILE_FORMS: Tuple[str, ...] = ("auto", "lean", "tcgen05")
SPARSE_TILE_TCGEN05 = "sparse_d256_f16"  # the tcgen05 core's kernel_template (the readback of ``plan.sparse_tile`` on that body)
SPARSE_TILE_LEAN = "sparse_lean_d256_f16"  # the lean decode tile's kernel_template
SPARSE_TILE_LEAN_MAX_UNITS = 4  # the AUTO placement bound on cc 10.7, in CTAs per chunk (``batch x H_kv``: B <= 2 on two KV heads, at every served q_len)
# The bound's UNIT as behaviour, mirrored from the vendored placement rule and re-read by :func:`check_vendored_rules`: what
# ``sparse_decode_lean_tile_selected`` answers on cc 10.7 at ``top_k`` 512 over two KV heads per ``(batch, q_len)`` cell -- True iff
# ``batch x 2 <= 4``, whatever the step width.  A tree that counted ``batch x H_kv x q_len`` (the 7301fad21 pin) reads the (1, 3), (1, 4)
# and (2, 4) cells False, so the tripwire fires on the previous unit.  The rule's reading, not this entry's route: under the in-block
# indexer's per-row list the lean body admits ``q_len_per_req == 1`` only (:func:`sparse_tile` reads the route cell by cell).
SPARSE_TILE_LEAN_RULE_CELLS: Tuple[Tuple[int, int, bool], ...] = (
    (1, 1, True),
    (1, 3, True),
    (1, 4, True),
    (2, 1, True),
    (2, 4, True),
    (3, 1, False),
    (4, 1, False),
    (4, 4, False),
)


def next_pow2(n: int) -> int:
    """The smallest power of two >= ``n`` (``padded_heads`` of the paged scorer: the head group is padded to it)."""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def rope_dim_reason(rope_dim: int, head_dim: int = HEAD_DIM) -> Optional[str]:
    """The Qwen 3.5 package's rope_dim rule (the same two stage-(1) renderings), worded for this block."""
    r = _s35.rope_dim_reason(rope_dim, head_dim)
    return None if r is None else r.replace("the Qwen 3.5 block", "the Qwen 3.8 block")


def device_reason(device) -> Optional[str]:
    r = _s35.device_reason(device)
    return None if r is None else r.replace("the Qwen 3.5 block", "the Qwen 3.8 block")


def dtype_reason(dtype: torch.dtype) -> Optional[str]:
    return _s35.dtype_reason(dtype)


def cutedsl_reason() -> Optional[str]:
    r = _s35.cutedsl_reason()
    return None if r is None else r.replace("Qwen 3.5 block", "Qwen 3.8 block")


def top_k_reason(top_k: int) -> Optional[str]:
    lo, hi, align = INDEX_TOP_K_RANGE
    if not (lo <= int(top_k) <= hi) or int(top_k) % align:
        return (
            f"top_k must be a multiple of {align} in [{lo}, {hi}] (blocks per row: the sparse core's per-row index staging, copied in "
            f"16-byte units), got top_k={top_k}"
        )
    return None


def _sm_count(device) -> Optional[int]:
    """The SM count of ``device`` (the projection route's one-wave arithmetic), or None without a CUDA device (the CPU probe never
    reaches the route gate: the device gate declines first)."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        return int(torch.cuda.get_device_properties(index).multi_processor_count)
    except Exception:  # noqa: BLE001 -- a device the driver does not describe: no route verdict (the device gate speaks)
        return None


def proj_route_reason(
    geometry, device, dtype: torch.dtype = torch.bfloat16
) -> Optional[str]:
    """Why the block's two dense projections at this geometry cannot take the capture-safe route on ``device``, or None.

    The vendored projection GEMM serves a CAPTURED decode step on its small-M route only (the graph route creates a cuDNN handle at the
    first execute per stream, which cannot happen inside a CUDA-graph capture), and that route names a tile for an N that 256 divides,
    or that 128 divides where its 64-row weight tiles overflow one wave of the part's SMs -- the vendored rule's own arithmetic
    (``kernels.proj_gemm.small_m_tile_config``) is asked here, at ``m = 1``, with the device's SM count.  The TP 1 Flash-Next slab
    (N 13952 = 109 x 128, 218 64-row tiles) is on the route on every served part; the TP 2 / TP 4 shard slabs (N 7296 / 4224) are not.
    Imports the vendored tree (host arithmetic only; called after the device gate passed)."""
    sm = _sm_count(device)
    if sm is None:
        return None
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block.kernels.proj_gemm import (
        small_m_tile_config,
    )

    d = int(geometry.head_dim)
    for label, n, k in (
        (
            "the fused QKV + gate + indexer-band slab",
            int(geometry.n_qkvg),
            int(geometry.d_model),
        ),
        ("the out projection", int(geometry.d_model), int(geometry.num_qo_heads) * d),
    ):
        if small_m_tile_config(1, n, k, dtype=dtype, sm_count=sm) is None:
            return (
                f"{label} [rows, {k}] x [{k}, {n}] (N={n}) is off the capture-safe small-M projection route on this {sm}-SM part: the "
                "route names a tile for an N that 256 divides, or that 128 divides where its 64-row weight tiles overflow one wave "
                "(the TP 1 slab, N 13952); any other N resolves to the graph route, whose first execute creates a cuDNN handle and "
                "cannot live in a CUDA graph. The TP 1 (24 / 2) geometry is served; this tensor-parallel shard is not -- run the block "
                "at TP 1, or keep the serving stack's own attention for the shard"
            )
    return None


def extend_rows_grid_text() -> str:
    """The probe's text of the :data:`EXTEND_GRID_ROWS_KEY` row (no batch, no chunk): the bound and the remedy."""
    return (
        f"one run of any form holds at most {EXTEND_ROWS_GRID_MAX} rows (batch_size x q_len_per_req) at this pin: the vendored "
        "indexer's query compress (the decode step's and the extend step's) launches with the run's row count as a CUDA grid "
        f"dimension, which CUDA caps at {EXTEND_ROWS_GRID_MAX} -- run a longer chunk as several uniform chunks of at most that many "
        "rows per run (runs grouped by their length), a wider decode step as several decode steps, or a smaller batch"
    )


def extend_rows_grid_reason(
    geometry, form: str, batch_size: int, q_len_per_req: int
) -> Optional[str]:
    """Why one ``form`` run of ``batch_size x q_len_per_req`` rows is past the compressor's launch-grid bound at this pin, or None.

    The vendored indexer's query compress -- the decode step's and the extend step's share the launch -- runs with the run's row count as
    the grid's y dimension (one grid row per query token, the geometry's ``index_heads`` along x), which CUDA caps at
    :data:`EXTEND_ROWS_GRID_MAX` = 65,535 -- so a run of more rows would pass every other gate and fail at launch.  The text names the
    bound, the rows and the remedy (several uniform chunks -- several decode steps on the decode form -- of at most that many rows per run
    at this batch, or a smaller batch).  Device-free and free of vendored imports: a light gate (``api.prepare`` raises it typed before
    any vendored code on every form; the probe publishes the row's generic text, :func:`extend_rows_grid_text`)."""
    rows = int(batch_size) * int(q_len_per_req)
    if rows <= EXTEND_ROWS_GRID_MAX:
        return None
    per_seq = EXTEND_ROWS_GRID_MAX // int(batch_size)
    return (
        f"batch_size={batch_size} x q_len_per_req={q_len_per_req} = {rows} rows in one {form} run exceed {EXTEND_ROWS_GRID_MAX} (the "
        f"{EXTEND_GRID_ROWS_KEY} row: at this pin the vendored indexer's query compress launches with the run's row count as a CUDA grid "
        f"dimension -- one grid row per query token, the {int(geometry.index_heads)} indexer heads along the other --, which CUDA caps at "
        f"{EXTEND_ROWS_GRID_MAX}) -- run it as several "
        + ("decode steps" if form == "decode" else "uniform chunks")
        + f" of at most {EXTEND_ROWS_GRID_MAX} rows per run "
        + (
            f"(q_len_per_req <= {per_seq} at this batch)"
            if per_seq >= 1
            else "(this batch alone exceeds the bound)"
        )
        + ", or a smaller batch"
    )


@dataclass(frozen=True)
class Qwen38BlockGeometry:
    """The QSA sub-layer's shape at THIS tensor-parallel rank: the Qwen 3.5 fields first (Qwen3.8-Flash-Next at TP 1: ``d_model=2560,
    num_qo_heads=24, num_kv_heads=2``; ``12 / 1`` at TP 2, ``6 / 1`` at TP 4 -- the indexer is replicated on every rank), then the indexer
    appended: ``index_heads x index_head_dim`` query heads over ``index_kv_heads`` raw key head, ``block_size``-token blocks, ``top_k``
    blocks per row (the 2048-token budget = 512 blocks of 4), the indexer norms' epsilon, the list form (``None`` -> ``"per_row"``,
    the decode indexer's function; ``"shared"`` is declined by name), the indexer cache's dtype (``None`` = the activation dtype, bf16;
    ``torch.float8_e4m3fn`` = the e4m3 compressed-key cache, served on every form since the 4c8e24355 pin) and, appended last, the
    indexer writers' rounding convention ``index_compress_rounding`` (:data:`INDEX_COMPRESS_ROUNDINGS`: ``"single"``, the reference --
    fp32 through the pooled mean, the RMSNorm and the RoPE, one rounding at the store --, or ``"hf"``, the HF / vLLM-prepare ROUNDING POINTS
    -- the pooled key rounded before its norm, the normed row before its rotation, the rotation's three operations each rounded -- for a
    stack whose cache mixes rows another writer rounded that way; NOT bitwise that writer: the fp32 reduction orders differ); a frozen
    field, so it is part of every plan's key and a compile key of every indexer writer of the plan (the chunk / prefill compress, the decode
    step's recipes).  The constructor checks the FORM of the fields; :meth:`unsupported_reason` is the kernels' envelope (``head_dim`` 256,
    ``d_model`` a multiple of the projection tile, ``rope_dim`` in :data:`ROPE_DIMS`, the indexer rows mirrored from the vendored tree)."""

    d_model: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int = HEAD_DIM
    rope_dim: int = 64
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    sm_scale: Optional[float] = None
    # appended (the QSA indexer; Qwen 3.8 Flash-Next: 4 x 128 over 1 raw key head, 4-token blocks, a 2048-token budget = 512 blocks)
    index_heads: int = 4
    index_kv_heads: int = 1
    index_head_dim: int = 128
    index_norm_eps: float = 1e-6
    block_size: int = 4
    top_k: int = 512
    index_list_form: Optional[str] = (
        None  # None -> "per_row" (the served form at decode); "per_row" explicit; "shared" declined by name
    )
    index_cache_dtype: Optional[torch.dtype] = (
        None  # None = the activation dtype (bf16); torch.float8_e4m3fn = the e4m3 compressed-key cache (served on every form)
    )
    # appended (the 4c8e24355 pin): the indexer writers' rounding convention -- "single" (the reference) or "hf" (the HF / vLLM rounding points)
    index_compress_rounding: str = INDEX_COMPRESS_ROUNDING_DEFAULT

    def __post_init__(self):
        for name in (
            "d_model",
            "num_qo_heads",
            "num_kv_heads",
            "head_dim",
            "rope_dim",
            "index_heads",
            "index_kv_heads",
            "index_head_dim",
            "block_size",
            "top_k",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(
                    f"Qwen38BlockGeometry.{name} must be a positive int, got {v!r}"
                )
        if self.num_qo_heads % self.num_kv_heads:
            raise ValueError(
                f"Qwen38BlockGeometry: num_qo_heads={self.num_qo_heads} must be a multiple of num_kv_heads={self.num_kv_heads} (GQA groups)"
            )
        if self.rope_dim % 2 or self.rope_dim > self.head_dim:
            raise ValueError(
                f"Qwen38BlockGeometry: rope_dim={self.rope_dim} must be even and at most head_dim={self.head_dim}"
            )
        if not self.qk_norm_eps > 0.0:
            raise ValueError(
                f"Qwen38BlockGeometry: qk_norm_eps must be > 0, got {self.qk_norm_eps}"
            )
        if not self.index_norm_eps > 0.0:
            raise ValueError(
                f"Qwen38BlockGeometry: index_norm_eps must be > 0, got {self.index_norm_eps}"
            )
        if self.sm_scale is not None and not (float(self.sm_scale) > 0.0):
            raise ValueError(
                f"Qwen38BlockGeometry: sm_scale must be positive or None, got {self.sm_scale}"
            )
        if self.index_list_form not in INDEX_LIST_FORMS:
            raise ValueError(
                f"Qwen38BlockGeometry: index_list_form must be None (-> 'per_row'), 'per_row' or 'shared', got {self.index_list_form!r}"
            )
        if self.index_cache_dtype is not None and not isinstance(
            self.index_cache_dtype, torch.dtype
        ):
            raise ValueError(
                f"Qwen38BlockGeometry: index_cache_dtype must be None or a torch.dtype, got {self.index_cache_dtype!r}"
            )
        if self.index_compress_rounding not in INDEX_COMPRESS_ROUNDINGS:
            raise ValueError(
                f"Qwen38BlockGeometry: index_compress_rounding must be one of {INDEX_COMPRESS_ROUNDINGS} ('single' = the reference chain, "
                f"one rounding at the store; 'hf' = the HF / vLLM-prepare rounding points), got {self.index_compress_rounding!r}"
            )

    @property
    def group(self) -> int:
        """Query heads per KV head."""
        return self.num_qo_heads // self.num_kv_heads

    @property
    def n_qkvg_dense(self) -> int:
        """Columns of the four dense bands of ``W_qkvg`` (``Q | GATE | K | V``)."""
        return (2 * self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def index_band_cols(self) -> int:
        """Columns of the fifth (indexer) band: ``(index_heads + index_kv_heads) * index_head_dim`` (640 at the defaults)."""
        return (self.index_heads + self.index_kv_heads) * self.index_head_dim

    @property
    def n_qkvg(self) -> int:
        """Columns of the fused ``W_qkvg`` with the indexer band (``Q | GATE | K | V | INDEX``): 13952 at 24 / 2 over 2560."""
        return self.n_qkvg_dense + self.index_band_cols

    @property
    def scale(self) -> float:
        return (
            float(self.sm_scale)
            if self.sm_scale is not None
            else float(self.head_dim) ** -0.5
        )

    @property
    def resolved_index_list_form(self) -> str:
        """What ``index_list_form`` resolves to on the decode form: ``"per_row"`` for ``None``."""
        return "per_row" if self.index_list_form is None else self.index_list_form

    @property
    def resolved_index_cache_dtype(self) -> torch.dtype:
        """The paged compressed-key cache's dtype: ``index_cache_dtype``, or bf16 (the activation dtype) for ``None``."""
        return (
            torch.bfloat16 if self.index_cache_dtype is None else self.index_cache_dtype
        )

    @property
    def decode_max_q_len(self) -> int:
        """Query rows per sequence the decode indexer's scorer serves at this head count: ``INDEX_QUERY_ROWS_MAX // next_pow2(index_heads)``
        (16 at 4 heads); a longer step is a typed decline."""
        return INDEX_QUERY_ROWS_MAX // next_pow2(self.index_heads)

    def unsupported_reason(self) -> Optional[str]:
        """Why the kernels do not serve this geometry (``None`` = served): the Qwen 3.5 rows, then the indexer rows mirrored from the
        vendored tree; independent of the device."""
        if self.head_dim != HEAD_DIM:
            return f"the Qwen 3.8 block kernels serve head_dim={HEAD_DIM} only (the Rubin d256 flavor); got head_dim={self.head_dim}"
        if self.d_model % _PROJ_TILE:
            return f"d_model must be a multiple of {_PROJ_TILE} (the forced projection GEMM tile); got d_model={self.d_model}"
        rope = rope_dim_reason(self.rope_dim, self.head_dim)
        if rope is not None:
            return rope
        if self.index_head_dim != INDEX_HEAD_DIM:
            return (
                f"the in-block indexer's paged scorer packs {INDEX_HEAD_DIM}-wide heads; got index_head_dim={self.index_head_dim} "
                "(another head dim has no scorer)"
            )
        if self.index_kv_heads not in INDEX_KV_HEADS:
            return f"the in-block indexer takes one raw key head (index_kv_heads in {INDEX_KV_HEADS}); got index_kv_heads={self.index_kv_heads}"
        if self.block_size not in INDEX_BLOCK_SIZES:
            return (
                f"block_size must be in {INDEX_BLOCK_SIZES} (the sparse core gathers one 4-token block per transaction; another block size has "
                f"no loader); got block_size={self.block_size}"
            )
        tk = top_k_reason(self.top_k)
        if tk is not None:
            return tk
        if (
            self.index_cache_dtype is not None
            and self.index_cache_dtype not in INDEX_CACHE_DTYPES_SERVED
        ):
            return (
                f"index_cache_dtype={self.index_cache_dtype}: the paged compressed-key cache is served in bf16 (index_cache_dtype=None) or "
                "torch.float8_e4m3fn (the vendored compress kernels' e4m3 store arm, the paged scorer's and the extend gather's e4m3 read arms) "
                "-- declare one of those"
            )
        if self.index_list_form == "shared":
            return (
                "index_list_form='shared': the decode indexer selects per row (one list per query row, the serving stacks' function); "
                "declare index_list_form='per_row' or None"
            )
        return None

    def frost(self):
        """The vendored cuDNN Frontend geometry with its ``QsaSpec`` (imports the vendored tree)."""
        from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
            GatedAttentionBlockGeometry,
            QsaSpec,
        )

        return GatedAttentionBlockGeometry(
            d_model=self.d_model,
            h_q=self.num_qo_heads,
            h_kv=self.num_kv_heads,
            d_head=self.head_dim,
            rope_dim=self.rope_dim,
            qk_norm_eps=self.qk_norm_eps,
            attn_scale=self.sm_scale,
            is_causal=True,
            causal_bottom_right=True,
            qk_norm=self.qk_norm,
            qsa=QsaSpec(
                block_size=self.block_size,
                top_k=self.top_k,
                index_source="indexer",
                index_band=True,
                index_heads=self.index_heads,
                index_kv_heads=self.index_kv_heads,
                index_head_dim=self.index_head_dim,
                index_norm_eps=self.index_norm_eps,
                index_cache_dtype=self.index_cache_dtype,
                index_list_form=self.index_list_form,
                index_compress_rounding=self.index_compress_rounding,
            ),
        )


#: the version probe's record: the SAME vendored tree as the Qwen 3.5 package's (one drop, one manifest), so the record type is its
Qwen38BlockSource: TypeAlias = Qwen35BlockSource


def source() -> Qwen38BlockSource:
    """The version probe (cached in the Qwen 3.5 package): the vendored cuDNN Frontend commit / version / manifest digest."""
    return _s35.source()


@dataclass(frozen=True)
class Qwen38BlockFormSupport:
    form: str
    served: bool
    reason: Optional[str] = None
    #: every typed decline the form can raise at plan time, by name, with its message (the FE decline table, verbatim)
    declines: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Qwen38BlockSupport:
    """The static answer to "can this process run the Qwen 3.8 block for this geometry / dtype on this device" -- read once at init by
    a serving stack's dispatch predicate instead of catching exceptions.  CPU-constructible; never raises for an unsupported device.
    The Qwen 3.5 record's fields first, the indexer facts appended -- every appended value READ off the vendored envelope's
    ``decode.index_*`` rows (``()`` / ``0`` where the tree was not read), never a second literal."""

    served: bool
    reason: Optional[str]
    device_cc: Optional[Tuple[int, int]]
    cutedsl_version: Optional[str]
    cutedsl_ok: bool
    forms: Tuple[str, ...]
    prefill: Qwen38BlockFormSupport
    extend: Qwen38BlockFormSupport
    decode: Qwen38BlockFormSupport
    source: Qwen38BlockSource
    cutedsl_required: str = CUTEDSL_REQUIRED
    page_sizes: Tuple[int, ...] = PAGE_SIZES
    kv_layouts: Tuple[str, ...] = KV_LAYOUTS
    #: query rows per sequence the decode form serves (the scorer's envelope at the geometry's head count; 16 at 4 heads)
    decode_max_q_len: int = 0
    #: ``batch x q_len`` rows a captured step may hold (the capture-safe projection route)
    decode_capture_rows_max: int = DECODE_CAPTURE_ROWS_MAX
    dtypes: Tuple[torch.dtype, ...] = DTYPES
    #: the pool dtypes the decode form serves at this geometry (bf16 under the indexer; the e4m3 pool is a typed decline until the sparse core's dequant arm)
    kv_cache_dtypes: Tuple[torch.dtype, ...] = ()
    rope_dims: Tuple[int, ...] = ROPE_DIMS
    # appended: the indexer rows of the vendored envelope
    index_sources: Tuple[str, ...] = ()
    index_list_forms: Tuple[str, ...] = ()
    index_list_form_default: str = ""
    index_cache_dtypes: Tuple[torch.dtype, ...] = ()
    index_head_dims: Tuple[int, ...] = ()
    index_kv_heads: Tuple[int, ...] = ()
    index_block_sizes: Tuple[int, ...] = ()
    index_top_k_range: Tuple[int, int, int] = (0, 0, 0)
    index_query_rows_max: int = 0
    index_state_rows_extra: int = 0
    index_page_sizes: Tuple[int, ...] = ()
    fuse_prepare_default: bool = False
    pdl_default: bool = False
    # appended (the lean decode tile): the sparse core's bodies this part serves by ``kernel_template`` (the tcgen05 core everywhere the form
    # is served; the lean tile too on cc 10.7) and whether ``sparse_tile_form="auto"`` consults the lean placement rule on this part
    # (``True`` on cc 10.7: the measured part); the per-cell reading (which body a ``[batch, q_len]`` step takes) is :func:`sparse_tile`
    sparse_tiles: Tuple[str, ...] = ()
    sparse_tile_lean_default: bool = False
    # appended (the 4c8e24355 pin; the in-block indexer's EXTEND step): the step forms the vendored declaration knob takes, the EXTEND
    # step's ring floor (``cap >= 3``), its select-window budget (bytes) and the indexer writers' rounding conventions -- READ off the
    # envelope (``()`` / ``0`` where the tree was not read)
    index_select_forms: Tuple[str, ...] = ()
    index_extend_ring_rows_min: int = 0
    index_extend_select_budget_bytes: int = 0
    index_compress_roundings: Tuple[str, ...] = ()
    # appended (this pin's one FlashInfer-side rows bound): the rows ONE run of ANY form may hold (``batch_size x q_len_per_req``) -- the
    # vendored indexer's query compress (the decode step's and the extend step's) launches with the run's row count as a CUDA grid
    # dimension (``EXTEND_ROWS_GRID_MAX``; the ``indexer_extend_grid_rows`` row of every form record); lifted by a re-vendor once the
    # compressor's grid is flattened
    extend_rows_grid_max: int = EXTEND_ROWS_GRID_MAX

    def form(self, name: str) -> Qwen38BlockFormSupport:
        if name not in ALL_FORMS:
            raise ValueError(f"form must be one of {ALL_FORMS}, got {name!r}")
        return getattr(self, name)


def _frost_decode_table(
    geometry: Qwen38BlockGeometry, dtype: torch.dtype, device, kv_cache_dtype
) -> Dict[str, Any]:
    """The vendored tree's own decode envelope (imports the vendored tree: called after the light gates passed)."""
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
        GatedAttentionBlockFwd,
    )

    check_vendored_rules()  # the mirrored indexer rules against the tree's: a drift is a loud refusal, never a false verdict
    env = GatedAttentionBlockFwd.serving_envelope(
        geometry.frost(), dtype, torch.device(device), device_cc=SUPPORTED_CC
    )
    d = env.decode
    served, reason = bool(d.served), d.reason
    if (
        served
        and kv_cache_dtype is not None
        and kv_cache_dtype not in d.kv_cache_dtypes
    ):
        served = False
        reason = d.declines.get(
            "indexer_e4m3_pool",
            f"kv_cache_dtype={kv_cache_dtype} is not served under the in-block indexer (served: {d.kv_cache_dtypes})",
        )
    declines = dict(d.declines)
    route = proj_route_reason(geometry, device, dtype)
    if (
        route is not None
    ):  # the FlashInfer-side row: the shard's slab off the capture-safe projection route on this device
        declines[PROJ_SLAB_ROUTE_KEY] = route
        if served:
            served, reason = False, route
    return dict(
        served=served,
        reason=reason,
        declines=declines,
        kv_cache_dtypes=tuple(d.kv_cache_dtypes),
        index_sources=tuple(d.index_sources),
        index_list_forms=tuple(d.index_list_forms),
        index_list_form_default=str(d.index_list_form_default),
        index_cache_dtypes=tuple(
            d.index_cache_dtypes
        ),  # bf16 and e4m3: served by the tree and by this entry on every form
        index_head_dims=tuple(int(x) for x in d.index_head_dims),
        index_kv_heads=tuple(int(x) for x in d.index_kv_heads),
        index_block_sizes=tuple(int(x) for x in d.index_block_sizes),
        index_top_k_range=tuple(int(x) for x in d.index_top_k_range),
        index_query_rows_max=int(d.index_query_rows_max),
        index_state_rows_extra=int(d.index_state_rows_extra),
        index_page_sizes=tuple(int(x) for x in d.index_page_sizes),
        fuse_prepare_default=bool(d.fuse_prepare_default),
        pdl_default=bool(d.pdl_default),
        sparse_tiles=tuple(str(x) for x in d.sparse_tiles),
        sparse_tile_lean_default=bool(d.sparse_tile_lean_default),
        index_select_forms=tuple(str(x) for x in d.index_select_forms),
        index_extend_ring_rows_min=int(d.index_extend_ring_rows_min),
        index_extend_select_budget_bytes=int(d.index_extend_select_budget_bytes),
        index_compress_roundings=tuple(str(x) for x in d.index_compress_roundings),
    )


def sparse_tile(
    device,
    geometry: Qwen38BlockGeometry,
    *,
    batch_size: int,
    q_len_per_req: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    sparse_tile_form: str = SPARSE_TILE_FORM_AUTO,
    decode_split_kv: Optional[int] = None,
    kv_cache_dtype: Optional[torch.dtype] = None,
    device_cc_override: Optional[Tuple[int, int]] = None,
) -> Optional[str]:
    """Which BODY of the sparse core a ``[batch_size, q_len_per_req]`` decode step at ``geometry`` takes on ``device`` under
    ``sparse_tile_form`` -- :data:`SPARSE_TILE_LEAN` / :data:`SPARSE_TILE_TCGEN05` (the ``kernel_template`` ``plan.sparse_tile`` reads
    back), or ``None`` where no body of the sparse core runs (e4m3 pools) or where the explicit body does not admit the step (the plan
    is then the typed decline).  READ off the vendored envelope's own per-cell rule (``DecodeEnvelope.sparse_tile``: one copy of the
    placement arithmetic, never re-spelled here); imports the vendored tree.  ``device_cc_override`` (appended) stands in for the
    device's compute capability for a device-free reading (the host tests ask the cc 10.7 rule from a CPU host)."""
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
        GatedAttentionBlockFwd,
    )

    if sparse_tile_form not in SPARSE_TILE_FORMS:
        raise ValueError(
            f"sparse_tile_form must be one of {SPARSE_TILE_FORMS}, got {sparse_tile_form!r}"
        )
    check_vendored_rules()
    cc = device_cc_override if device_cc_override is not None else device_cc(device)
    env = GatedAttentionBlockFwd.serving_envelope(
        geometry.frost(), dtype, torch.device(device), device_cc=cc
    )
    return env.decode.sparse_tile(
        int(q_len_per_req),
        int(batch_size),
        sparse_tile_form=sparse_tile_form,
        decode_split_kv=decode_split_kv,
        kv_cache_dtype=kv_cache_dtype,
    )


def qwen_3_8_block_support(
    device,
    geometry: Qwen38BlockGeometry,
    *,
    dtype: torch.dtype = torch.bfloat16,
    kv_cache_dtype: Optional[torch.dtype] = None,
) -> Qwen38BlockSupport:
    """See :func:`flashinfer.qwen_3_8_block.qwen_3_8_block_support` (the public, decorated entry)."""
    if not isinstance(geometry, Qwen38BlockGeometry):
        raise TypeError(
            f"geometry must be a Qwen38BlockGeometry, got {type(geometry).__name__}"
        )
    cc = device_cc(device)
    version = cutedsl_version()
    dsl_reason = cutedsl_reason()
    common = (
        geometry.unsupported_reason()
        or dtype_reason(dtype)
        or device_reason(device)
        or dsl_reason
    )
    table: Dict[str, Any] = {}
    if common is not None:
        decode = Qwen38BlockFormSupport("decode", False, common)
    else:
        try:
            table = _frost_decode_table(geometry, dtype, device, kv_cache_dtype)
        except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
            table = dict(
                served=False,
                reason=f"the vendored cuDNN Frontend tree could not answer: {type(exc).__name__}: {exc}",
                declines={},
            )
        declines = dict(table["declines"])
        if "index_sources" in table:
            # the ONE FlashInfer-side row of every form record where the tree was read: the launch-grid bound of one run at this pin
            # (EXTEND_GRID_ROWS_KEY: 65,535 rows -- the decode step's and the extend step's query compress share the launch; a per-run
            # fact, never the verdict)
            declines[EXTEND_GRID_ROWS_KEY] = extend_rows_grid_text()
        decode = Qwen38BlockFormSupport(
            "decode", bool(table["served"]), table["reason"], declines
        )
    # the extend / prefill forms are the SAME vendored declaration form (the EXTEND step of the in-block indexer over the pools): served
    # iff the decode envelope is, with the ONE decline table (the five indexer_extend_* rows and the launch-grid row among its keys).  No
    # projection rows bound since the 6b21be0e0 pin (the band slab rides the JIT-only route at every row count: the tripwire re-reads the
    # rule)
    extend_declines = dict(decode.declines)
    prefill = Qwen38BlockFormSupport(
        "prefill", decode.served, decode.reason, extend_declines
    )
    extend = Qwen38BlockFormSupport(
        "extend", decode.served, decode.reason, extend_declines
    )
    served_forms = FORMS if decode.served else ()
    appended = {
        k: table[k]
        for k in (
            "kv_cache_dtypes",
            "index_sources",
            "index_list_forms",
            "index_list_form_default",
            "index_cache_dtypes",
            "index_head_dims",
            "index_kv_heads",
            "index_block_sizes",
            "index_top_k_range",
            "index_query_rows_max",
            "index_state_rows_extra",
            "index_page_sizes",
            "fuse_prepare_default",
            "pdl_default",
            "sparse_tiles",
            "sparse_tile_lean_default",
            "index_select_forms",
            "index_extend_ring_rows_min",
            "index_extend_select_budget_bytes",
            "index_compress_roundings",
        )
        if k in table
    }
    return Qwen38BlockSupport(
        served=bool(served_forms),
        reason=common
        if common is not None
        else (None if served_forms else decode.reason),
        device_cc=cc,
        cutedsl_version=version,
        cutedsl_ok=dsl_reason is None,
        forms=served_forms,
        prefill=prefill,
        extend=extend,
        decode=decode,
        source=source(),
        decode_max_q_len=geometry.decode_max_q_len,
        **appended,
    )


@functools.lru_cache(maxsize=1)
def check_vendored_rules() -> Dict[str, Any]:
    """Re-read the indexer rules this module mirrors from the vendored tree (imports it) and refuse a drift.

    The Qwen 3.5 package's tripwire runs first (the shared stage-(1) rules); then the decode envelope of the 24 / 2 Flash-Next geometry
    at bf16 with ``device_cc=(10, 7)`` -- device-free -- yields the indexer rows every mirrored constant is compared with
    (``index_head_dims``, ``index_kv_heads``, ``index_block_sizes``, ``index_top_k_range``, ``index_query_rows_max``,
    ``index_state_rows_extra``, ``index_page_sizes``), the scorer's row rule as BEHAVIOUR (``validate_params`` over q_len 1 .. 20 at
    4 heads against ``decode_max_q_len``), the capture-safe row bound (``kernels.proj_gemm.SMALL_M_MAX``), the projection route's verdicts
    (``small_m_tile_config``: the 24 / 2 slab on it, the shard slabs off it), the lift of the extend forms' rows bound as BEHAVIOUR (the
    driver's ``_auto_tile_name`` names a JIT tile for the 24 / 2 slab PAST the small-M route -- 512 and 2048 rows at every served SM count,
    the large-M rule of the 6b21be0e0 pin; a tree without it would put a chunk past 128 rows on the graph route the vendored root refuses),
    the sparse core's body knob (the ``tile_form`` vocabulary,
    the two bodies' ``kernel_template`` names, the AUTO placement bound, its UNIT as the placement rule's reading over ``(batch, q_len)``
    cells -- CTAs per chunk, the step width outside the bound -- and the envelope's cc 10.7 reading: both bodies served, the lean rule
    consulted), the EXTEND step's rows (the ``index_select_form`` vocabulary, the ring floor -- ONE spelling with ``index_state_rows_extra``
    --, the select-window budget and the rounding conventions, each read off the envelope AND off the tree's DSL-free module constants) and,
    behaviourally, every
    decline key the README's Errors table names present in ``decode.declines``.  Once per process (the probe's vendored read,
    ``api.prepare``); returns the tree's constants.  Raises ``RuntimeError`` naming every drifted fact."""
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
        INDEX_COMPRESS_ROUNDING_DEFAULT as _TREE_ROUNDING_DEFAULT,
    )
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
        INDEX_COMPRESS_ROUNDINGS as _TREE_ROUNDINGS,
    )
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
        QSA_EXTEND_SELECT_BUDGET_BYTES as _TREE_BUDGET,
    )
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
        QSA_INDEX_SELECT_FORMS as _TREE_SELECT_FORMS,
    )
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
        GatedAttentionBlockFwd,
    )
    from ..qwen_3_5_block._cudnn_frost.QSA import (  # the paged scorer's DSL-free host (``gated_attention_block/qsa_select_paged.py`` before the pinned commit)
        selection_paged as qsa_select_paged,
    )
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block.kernels import (
        proj_gemm,
    )
    from ..qwen_3_5_block._cudnn_frost.sdpa.fwd import (  # the DSL-free home of the tile_form vocabulary and the placement rule (the heuristics re-export them)
        sparse_tile_rules as _rules,
    )

    _s35.check_vendored_rules()
    g = Qwen38BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    env = GatedAttentionBlockFwd.serving_envelope(
        g.frost(), torch.bfloat16, torch.device("cpu"), device_cc=SUPPORTED_CC
    ).decode
    facts: Dict[str, Any] = {
        "index_head_dims": tuple(int(x) for x in env.index_head_dims),
        "index_kv_heads": tuple(int(x) for x in env.index_kv_heads),
        "index_block_sizes": tuple(int(x) for x in env.index_block_sizes),
        "index_top_k_range": tuple(int(x) for x in env.index_top_k_range),
        "index_query_rows_max": int(env.index_query_rows_max),
        "index_state_rows_extra": int(env.index_state_rows_extra),
        "index_page_sizes": tuple(int(x) for x in env.index_page_sizes),
        "decode_capture_rows_max": int(proj_gemm.SMALL_M_MAX),
        # the extend forms' rows bound LIFTED (the 6b21be0e0 pin): the band slab (N 13952, K 2560) names a JIT tile past the small-M route
        # through the driver's own resolver, at every served SM count -- the premise of serving any chunk length on the extend / prefill forms
        "band_slab_jit_tile_past_small_m": tuple(
            proj_gemm._auto_tile_name(
                m,
                g.n_qkvg,
                g.d_model,
                dtype=torch.bfloat16,
                a_major="k",
                b_major="k",
                block_scale=False,
                alpha=False,
                sm_count=sm,
            )
            is not None
            for m in (512, 2048)
            for sm in (148, 204, 212)
        ),
        # the sparse core's body knob (the lean decode tile): the vocabulary, the two bodies' kernel_template names, the AUTO placement
        # bound, its unit as the rule's own reading over (batch, q_len) cells (cc 10.7, the Flash-Next top_k, two KV heads), and the
        # envelope's reading of the cc 10.7 part (both bodies served, the AUTO pin consulting the lean rule there)
        "sparse_tile_forms": tuple(_rules.SPARSE_TILE_FORMS),
        "sparse_tile_names": (
            str(_rules.SPARSE_TILE_TCGEN05),
            str(_rules.SPARSE_TILE_LEAN),
        ),
        "sparse_tile_lean_max_units": int(_rules.LEAN_SPARSE_MAX_UNITS),
        "sparse_tile_lean_rule_cells": tuple(
            (
                b,
                q,
                bool(
                    _rules.sparse_decode_lean_tile_selected(
                        batch=b,
                        h_kv=g.num_kv_heads,
                        s_q=q,
                        top_k=g.top_k,
                        device_cc=SUPPORTED_CC,
                    )
                ),
            )
            for b, q, _ in SPARSE_TILE_LEAN_RULE_CELLS
        ),
        "sparse_tiles_cc107": tuple(str(x) for x in env.sparse_tiles),
        "sparse_tile_lean_default_cc107": bool(env.sparse_tile_lean_default),
        # the EXTEND step's rows (the 4c8e24355 pin): the envelope's four fields and the tree's DSL-free constants behind them -- one spelling
        "index_select_forms": tuple(str(x) for x in env.index_select_forms),
        "index_select_forms_tree": tuple(str(x) for x in _TREE_SELECT_FORMS),
        "index_extend_ring_rows_min": int(env.index_extend_ring_rows_min),
        "index_extend_ring_rows_min_is_the_ring_extra": int(
            env.index_extend_ring_rows_min
        )
        == int(env.index_state_rows_extra),
        "index_extend_select_budget_bytes": int(env.index_extend_select_budget_bytes),
        "index_extend_select_budget_bytes_tree": int(_TREE_BUDGET),
        "index_compress_roundings": tuple(str(x) for x in env.index_compress_roundings),
        "index_compress_roundings_tree": tuple(str(x) for x in _TREE_ROUNDINGS),
        "index_compress_rounding_default_tree": str(_TREE_ROUNDING_DEFAULT),
    }
    mirrored: Dict[str, Any] = {
        "index_head_dims": (INDEX_HEAD_DIM,),
        "index_kv_heads": INDEX_KV_HEADS,
        "index_block_sizes": INDEX_BLOCK_SIZES,
        "index_top_k_range": INDEX_TOP_K_RANGE,
        "index_query_rows_max": INDEX_QUERY_ROWS_MAX,
        "index_state_rows_extra": INDEX_STATE_ROWS_EXTRA,
        "index_page_sizes": INDEX_PAGE_SIZES,
        "decode_capture_rows_max": DECODE_CAPTURE_ROWS_MAX,
        "band_slab_jit_tile_past_small_m": (True,) * 6,
        "sparse_tile_forms": SPARSE_TILE_FORMS,
        "sparse_tile_names": (SPARSE_TILE_TCGEN05, SPARSE_TILE_LEAN),
        "sparse_tile_lean_max_units": SPARSE_TILE_LEAN_MAX_UNITS,
        "sparse_tile_lean_rule_cells": SPARSE_TILE_LEAN_RULE_CELLS,
        "sparse_tiles_cc107": (SPARSE_TILE_TCGEN05, SPARSE_TILE_LEAN),
        "sparse_tile_lean_default_cc107": True,
        "index_select_forms": INDEX_SELECT_FORMS,
        "index_select_forms_tree": INDEX_SELECT_FORMS,
        "index_extend_ring_rows_min": INDEX_EXTEND_RING_ROWS_MIN,
        "index_extend_ring_rows_min_is_the_ring_extra": INDEX_EXTEND_RING_ROWS_MIN
        == INDEX_STATE_ROWS_EXTRA,
        "index_extend_select_budget_bytes": INDEX_EXTEND_SELECT_BUDGET_BYTES,
        "index_extend_select_budget_bytes_tree": INDEX_EXTEND_SELECT_BUDGET_BYTES,
        "index_compress_roundings": INDEX_COMPRESS_ROUNDINGS,
        "index_compress_roundings_tree": INDEX_COMPRESS_ROUNDINGS,
        "index_compress_rounding_default_tree": INDEX_COMPRESS_ROUNDING_DEFAULT,
    }
    drift = [
        f"{k}: support.py has {mirrored[k]!r}, the vendored tree {facts[k]!r}"
        for k in facts
        if mirrored[k] != facts[k]
    ]
    # the scorer's row rule as the stage applies it: q_len x next_pow2(index_heads) <= the envelope's row cap, for q_len 1 .. 20
    for q_len in range(1, 21):
        try:
            qsa_select_paged.validate_params(
                h_i=g.index_heads, s_q=q_len, page_size=PAGE_SIZES[0], units_per_warp=1
            )
            tree = True
        except ValueError:
            tree = False
        if tree != (q_len <= g.decode_max_q_len):
            drift.append(
                f"q_len={q_len} at {g.index_heads} heads: the paged scorer {'takes' if tree else 'declines'} it, support.py reads the other way"
            )
    missing = [k for k in DECLINE_KEYS if k not in env.declines]
    if missing:
        drift.append(
            f"decode.declines lacks the row(s) {missing} the README's Errors table names"
        )
    # the projection route's verdicts the probe rests on (the vendored rule's own arithmetic at m = 1): the 24 / 2 slab ON the
    # small-M route at every served part's SM count, the TP 2 / TP 4 shard slabs OFF it (the README's TP note)
    for sm in (148, 204, 212):
        if (
            proj_gemm.small_m_tile_config(
                1, g.n_qkvg, g.d_model, dtype=torch.bfloat16, sm_count=sm
            )
            is None
        ):
            drift.append(
                f"the 24 / 2 slab (N {g.n_qkvg}) is off the small-M projection route at m=1 on a {sm}-SM part"
            )
    for n in (7296, 4224):
        if (
            proj_gemm.small_m_tile_config(
                1, n, g.d_model, dtype=torch.bfloat16, sm_count=204
            )
            is not None
        ):
            drift.append(
                f"the shard slab N {n} is ON the small-M projection route at m=1 on a 204-SM part: the README's TP note is stale"
            )
    if drift:
        raise RuntimeError(
            "qwen_3_8_block: the indexer rules support.py mirrors differ from the vendored cuDNN Frontend tree's; the gate "
            "(Qwen38BlockGeometry.unsupported_reason) must be re-derived with the re-vendored tree: "
            + "; ".join(drift)
        )
    return facts


def require_served(
    device, geometry: Qwen38BlockGeometry, dtype: torch.dtype, form: str
) -> None:
    """The plan-time gate: a typed ``NotImplementedError`` naming the form, the device, the DSL version, the dtype or the geometry."""
    if form not in ALL_FORMS:
        raise ValueError(f"form must be one of {ALL_FORMS}, got {form!r}")
    # every form of ALL_FORMS is served by the one gate since the 4c8e24355 pin (FORMS == ALL_FORMS; no second check)
    reason = (
        geometry.unsupported_reason()
        or dtype_reason(dtype)
        or device_reason(device)
        or cutedsl_reason()
        or proj_route_reason(geometry, device, dtype)
    )
    if reason is not None:
        raise NotImplementedError(f"qwen_3_8_block ({form}): {reason}")
