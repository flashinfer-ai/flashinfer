# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The adapter: FlashInfer arguments -> the vendored cuDNN Frontend block's DECODE declaration with the in-block indexer; plans,
workspaces and the decode form's run.

A :class:`Qwen38BlockPlan` is one compiled declaration of the vendored ``GatedAttentionBlockFwd`` for one shape bucket (the batch,
the query rows per sequence, the page size and the pools' layout, the block table's row stride, the side caches' FORM -- the
compressed-key cache's rows per page, the ring's capacity --, the input-contract flags, the knobs); ``prepare`` builds it outside
CUDA-graph capture (it JIT-compiles the kernels and sizes the workspace), ``run`` launches it -- the FIRST ``run`` after ``prepare``
may already be captured (no plan constant, no allocation, no host read on the execute path).  The functional entry point keeps a
small LRU of plans keyed on exactly the facts a plan compiles in (the Qwen 3.5 package's rules: a plan a CUDA graph was captured
over outlives its eviction and is released by ``clear_plan_cache`` only).  The vendored tree is the Qwen 3.5 package's
(``..qwen_3_5_block._cudnn_frost``) and so are the cache roots (``configure_vendored_tree``) and the operand helpers.

The sparse core's BODY knob (``sparse_tile_form``, appended with the 7301fad21 pin; performance-only): ``"auto"`` (the default) lets
the vendored tree pick the body -- on cc 10.7 the LEAN decode tile at ``batch x H_kv <= 4`` CTAs per chunk (the bound's unit since the
977757763 pin: the step's rows ride inside the CTA, so ``q_len`` is not in it) at every served ``q_len`` 1..4 the body admits -- under the
in-block indexer the list is per row, which the lean tile serves at ``q_len_per_req == 1`` only, so ``batch <= 2`` at ``q_len 1`` on two KV
heads --, the tcgen05 core everywhere else --, ``"lean"`` /
``"tcgen05"`` pin a body (an explicit body the step does not admit is the tree's typed decline).  The body a plan compiled reads back
in ``plan.sparse_tile`` (its ``kernel_template``) and ``plan.declaration``; it is a plan-cache key.
"""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..qwen_3_5_block import api as _api35
from ..qwen_3_5_block.api import (
    _ALIGN_BYTES,
    _INT_DTYPES,
    _capturing,
    _check_not_capturing,
    _device_of,
    _on,
    _tensor,
    configure_vendored_tree,
)
from . import support as _support
from .support import (
    DECODE_CAPTURE_ROWS_MAX,
    FORMS,
    INDEX_QUERY_ROWS_MAX,
    INDEX_STATE_ROWS_EXTRA,
    PAGE_SIZE_ALIGN,
    PAGE_SIZES,
    SPARSE_TILE_FORM_AUTO,
    SPARSE_TILE_FORMS,
    Qwen38BlockGeometry,
    next_pow2,
)
from .weights import Qwen38BlockWeights

KvCache = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
PositionTables = Tuple[torch.Tensor, torch.Tensor]
_PKG = "qwen_3_8_block"
_LAYOUTS_FI = ("NHD", "HND")
_LAYOUT_UNIFIED = "UNIFIED"

__all__ = [
    "KvCache",
    "PositionTables",
    "Qwen38BlockPlan",
    "cached_plan",
    "clear_plan_cache",
    "configure_vendored_tree",
    "decode",
    "kv_views",
    "kv_views_unified",
    "prepare",
    "workspace_size",
]


def _frost_block():
    from ..qwen_3_5_block._cudnn_frost.gated_attention_block import (
        GatedAttentionBlockFwd,
    )

    return GatedAttentionBlockFwd


def _rebrand(exc: BaseException) -> BaseException:
    """The Qwen 3.5 package's operand helpers name their package in every message; this package's name takes its place."""
    text = str(exc).replace("qwen_3_5_block:", f"{_PKG}:")
    return type(exc)(text)


def _rows(x, counts, width, name, dtype, device):
    try:
        return _api35._rows(x, counts, width, name, dtype, device)
    except ValueError as exc:
        raise _rebrand(exc) from None


def _int_vector(x, counts, name, device, dtypes=(torch.int32,)):
    try:
        return _api35._int_vector(x, counts, name, device, dtypes=dtypes)
    except ValueError as exc:
        raise _rebrand(exc) from None


def _block_table(x, batch, min_cols, row_stride, device, name="block_table"):
    try:
        return _api35._block_table(x, batch, min_cols, row_stride, device, name=name)
    except ValueError as exc:
        raise _rebrand(exc) from None


def _slots(x, counts, device, name="slot_mapping"):
    return _int_vector(x, counts, name, device, dtypes=_INT_DTYPES)


# --------------------------------------------------------------------------- operand form checks (host-side, typed, no device read)


def _check_weights(
    weights,
    w_o,
    geometry: Qwen38BlockGeometry,
    dtype: torch.dtype,
    device: torch.device,
) -> Qwen38BlockWeights:
    """The five weight tensors by FORM: ``w_qkvg [n_qkvg (with the indexer band), d_model]``, the two QK-norm weights ``[head_dim]`` (iff
    ``qk_norm``), the two indexer norms ``[index_head_dim]`` (always), ``w_o [d_model, H_q x head_dim]``; contiguous, the plan's dtype and device."""
    if not isinstance(weights, Qwen38BlockWeights):
        raise ValueError(
            f"{_PKG}: weights must be the Qwen38BlockWeights of qwen_3_8_block_pack_weights, got {type(weights).__name__}"
        )
    w = _tensor(weights.w_qkvg, "weights.w_qkvg")
    _on(w, device, "weights.w_qkvg")
    if (
        w.dtype != dtype
        or tuple(w.shape) != (geometry.n_qkvg, geometry.d_model)
        or not w.is_contiguous()
    ):
        raise ValueError(
            f"{_PKG}: weights.w_qkvg must be a contiguous {dtype} [{geometry.n_qkvg}, {geometry.d_model}] (Q | GATE | K | V | INDEX bands: "
            f"{geometry.n_qkvg_dense} dense columns + the {geometry.index_band_cols}-column indexer band), got {w.dtype} {tuple(w.shape)}"
        )
    for name, t, width, required in (
        ("weights.w_q_norm", weights.w_q_norm, geometry.head_dim, geometry.qk_norm),
        ("weights.w_k_norm", weights.w_k_norm, geometry.head_dim, geometry.qk_norm),
        ("weights.w_iq_norm", weights.w_iq_norm, geometry.index_head_dim, True),
        ("weights.w_ik_norm", weights.w_ik_norm, geometry.index_head_dim, True),
    ):
        if required:
            t = _tensor(t, name)
            _on(t, device, name)
            if t.dtype != dtype or tuple(t.shape) != (width,) or not t.is_contiguous():
                raise ValueError(
                    f"{_PKG}: {name} must be a contiguous {dtype} [{width}] vector, got {t.dtype} {tuple(t.shape)}"
                )
        elif t is not None:
            raise ValueError(
                f"{_PKG}: {name} must be None under geometry.qk_norm=False"
            )
    o = _tensor(w_o, "w_o")
    _on(o, device, "w_o")
    want = (geometry.d_model, geometry.num_qo_heads * geometry.head_dim)
    if o.dtype != dtype or tuple(o.shape) != want or not o.is_contiguous():
        raise ValueError(
            f"{_PKG}: w_o must be a contiguous {dtype} {list(want)} (the o_proj weight), got {o.dtype} {tuple(o.shape)}"
        )
    return weights


def _check_aligned(t: torch.Tensor, name: str) -> None:
    if int(t.data_ptr()) % _ALIGN_BYTES:
        raise ValueError(
            f"{_PKG}: {name} must start at a 16-byte aligned address (the kernels move its rows in 16-byte vectors), got data_ptr {t.data_ptr():#x}"
        )


def _check_index_kv_cache(
    x, num_pages: int, page_size: int, geometry: Qwen38BlockGeometry, dtype, device
) -> torch.Tensor:
    """The paged compressed-key cache ``[num_pages, page_size / block_size, 1, index_head_dim]`` (one compressed row per block; its page
    axis the MAIN block table's), the plan's dtype and device, a 16-byte base; the strides are the stack's (the kernels bind them)."""
    x = _tensor(x, "index_kv_cache")
    _on(x, device, "index_kv_cache")
    rows = page_size // geometry.block_size
    want = (num_pages, rows, 1, geometry.index_head_dim)
    if x.dim() != 4 or tuple(int(s) for s in x.shape) != want:
        raise ValueError(
            f"{_PKG}: index_kv_cache must be [num_pages={num_pages}, page_size / block_size={rows}, 1, {geometry.index_head_dim}] (the paged "
            f"compressed-key cache on the main block table: one row per {geometry.block_size}-token block), got {tuple(x.shape)}"
        )
    if x.dtype != dtype:
        raise ValueError(
            f"{_PKG}: index_kv_cache must be {dtype} (the indexer cache dtype of this plan), got {x.dtype}"
        )
    if x.stride(3) != 1:
        raise ValueError(
            f"{_PKG}: index_kv_cache: the head dim must be contiguous, got strides {tuple(x.stride())}"
        )
    _check_aligned(x, "index_kv_cache")
    return x


def _check_index_state_cache(
    x, q_len: int, geometry: Qwen38BlockGeometry, dtype, device, *, ring_slots: bool
) -> torch.Tensor:
    """The raw-key ring ``[num_ring_blocks >= 1, cap, 1, index_head_dim]`` with ``cap >= q_len + 3`` (a completing block pools the three
    positions below the step while the step writes ``q_len`` rows), the plan's dtype and device, a 16-byte base; its blocks contiguous
    under the ring-slots arm (the stack's flat slots ``block x cap + pos % cap`` address it so)."""
    x = _tensor(x, "index_state_cache")
    _on(x, device, "index_state_cache")
    need = q_len + INDEX_STATE_ROWS_EXTRA
    if (
        x.dim() != 4
        or int(x.shape[0]) < 1
        or int(x.shape[2]) != 1
        or int(x.shape[3]) != geometry.index_head_dim
    ):
        raise ValueError(
            f"{_PKG}: index_state_cache must be [num_ring_blocks >= 1, cap, 1, {geometry.index_head_dim}] (the raw-key ring; block 0 the null "
            f"block of a padded entry), got {tuple(x.shape)}"
        )
    if int(x.shape[1]) < need:
        raise ValueError(
            f"{_PKG}: index_state_cache holds cap={int(x.shape[1])} rows per block; the raw-key ring needs at least q_len_per_req + "
            f"{INDEX_STATE_ROWS_EXTRA} = {need} rows per block (a completing block pools the {INDEX_STATE_ROWS_EXTRA} positions below the step while "
            f"the step writes {q_len} rows) -- a serving stack's ring of whole {geometry.block_size}-token groups covering 1 + num_speculative_tokens "
            "rows satisfies it"
        )
    if x.dtype != dtype:
        raise ValueError(
            f"{_PKG}: index_state_cache must be {dtype} (the raw keys as the projection emits them: a bit copy, never quantized), got {x.dtype}"
        )
    if x.stride(3) != 1:
        raise ValueError(
            f"{_PKG}: index_state_cache: the head dim must be contiguous, got strides {tuple(x.stride())}"
        )
    if ring_slots and int(x.stride(0)) != int(x.shape[1]) * int(x.stride(1)):
        raise ValueError(
            f"{_PKG}: index_state_cache: ring_slots=True names flat ring rows (block x cap + pos % cap), which needs the ring's blocks contiguous "
            f"(stride[0] == cap x stride[1]); got shape {tuple(x.shape)} strides {tuple(x.stride())}"
        )
    _check_aligned(x, "index_state_cache")
    return x


def _check_index_state_block_table(x, batch: int, device) -> torch.Tensor:
    x = _tensor(x, "index_state_block_table")
    _on(x, device, "index_state_block_table")
    if (
        x.dtype != torch.int32
        or x.dim() != 2
        or tuple(int(s) for s in x.shape) != (batch, 1)
        or x.stride(1) != 1
    ):
        raise ValueError(
            f"{_PKG}: index_state_block_table must be an int32 [{batch}, 1] table (the ring block per sequence, 0 for a padded entry; any row "
            f"stride; a wider stack table is handed as its leading-row slice [:batch]), got {x.dtype} {tuple(x.shape)} strides {tuple(x.stride())}"
        )
    return x


def _check_position_tables(
    tables, geometry: Qwen38BlockGeometry, n_pos: int, dtype, device
) -> PositionTables:
    """``(cos_pos, sin_pos)`` ``[1 or batch, >= max_pages x page_size, rope_dim]``: the position-indexed tables of every position the block
    table can address (the ceiling by form), the plan's dtype and device, contiguous, 16-byte bases."""
    if not (isinstance(tables, (tuple, list)) and len(tables) == 2):
        raise ValueError(
            f"{_PKG}: position_tables must be the (cos_pos, sin_pos) pair of qwen_3_8_block_rope_position_tables, got {type(tables).__name__}"
        )
    out = []
    for name, t in (("cos_pos", tables[0]), ("sin_pos", tables[1])):
        t = _tensor(t, name)
        _on(t, device, name)
        if t.dim() != 3 or int(t.shape[0]) < 1 or int(t.shape[2]) != geometry.rope_dim:
            raise ValueError(
                f"{_PKG}: {name} must be [1 or batch, >= {n_pos}, rope_dim={geometry.rope_dim}] (duplicated halves per position), got {tuple(t.shape)}"
            )
        if int(t.shape[1]) < n_pos:
            raise ValueError(
                f"{_PKG}: {name} holds {int(t.shape[1])} positions; the plan's block table reaches max_pages x page_size = {n_pos} positions and the "
                "block refuses a shorter table by form (a completing block's start lies below the step) -- form the tables with "
                f"qwen_3_8_block_rope_position_tables(cos_sin_cache, num_positions={n_pos}) or more"
            )
        if t.dtype != dtype:
            raise ValueError(
                f"{_PKG}: {name} must be {dtype} (the plan's dtype), got {t.dtype}"
            )
        if not t.is_contiguous():
            raise ValueError(
                f"{_PKG}: {name} must be contiguous, got strides {tuple(t.stride())}"
            )
        _check_aligned(t, name)
        out.append(t)
    return out[0], out[1]


# --------------------------------------------------------------------------- the paged KV cache: FlashInfer layouts + the unified row -> the block's views


def kv_views_unified(
    kv_cache: torch.Tensor,
    *,
    geometry: Qwen38BlockGeometry,
    page_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The block's ``[num_pages, num_kv_heads, page_size, head_dim]`` K and V views of a serving stack's UNIFIED paged cache -- strides
    only, no copy.

    ``kv_cache`` is the ``[num_pages, page_size, H_kv, 2 x D]`` tensor whose row holds K at ``[..., :D]`` and V at ``[..., D:]`` (the
    Qwen 3.8 Flash-Next owner's main cache in vLLM; any strides the stack's layout planner chose).  Typed ``ValueError`` naming the
    operand on any other form; the pool contract the kernels bind (``D`` contiguous, every other stride a 16-byte multiple, 16-byte-aligned
    bases) is checked here and again by the vendored block."""
    kv = _tensor(kv_cache, "kv_cache")
    h, p, d = geometry.num_kv_heads, int(page_size), geometry.head_dim
    if kv.dim() != 4 or tuple(int(s) for s in kv.shape[1:]) != (p, h, 2 * d):
        raise ValueError(
            f"{_PKG}: a UNIFIED kv_cache is [num_pages, page_size, H_kv, 2 x D] = [*, {p}, {h}, {2 * d}] (K at [..., :{d}], V at [..., {d}:]), got {tuple(kv.shape)}"
        )
    if kv.device.type != "cuda":
        raise ValueError(f"{_PKG}: kv_cache must be a CUDA tensor, got {kv.device}")
    if kv.dtype != dtype:
        raise ValueError(
            f"{_PKG}: kv_cache must be {dtype} (the activation dtype: the pools hold bf16 K / V), got {kv.dtype}"
        )
    t = kv.transpose(1, 2)  # [num_pages, H_kv, page_size, 2 D]
    views = []
    for name, view in (("kv_cache (K)", t[..., :d]), ("kv_cache (V)", t[..., d:])):
        if view.stride(3) != 1:
            raise ValueError(
                f"{_PKG}: {name}: the head dim must be contiguous, got strides {tuple(kv.stride())}"
            )
        es = view.element_size()
        for ax in range(3):
            if (int(view.stride(ax)) * es) % _ALIGN_BYTES:
                raise ValueError(
                    f"{_PKG}: {name}: every stride but the head dim's must be a 16-byte multiple, got strides {tuple(kv.stride())} ({es}-byte elements)"
                )
        if view.data_ptr() % _ALIGN_BYTES:
            raise ValueError(
                f"{_PKG}: {name} must be 16-byte aligned, got data_ptr {view.data_ptr():#x}"
            )
        views.append(view)
    return views[0], views[1]


def kv_views(
    kv_cache: KvCache,
    kv_layout: str,
    *,
    geometry: Qwen38BlockGeometry,
    page_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The block's K / V views of a FlashInfer ``"NHD"`` / ``"HND"`` cache (5-D or a ``(k, v)`` pair; the Qwen 3.5 package's rule) or of a
    ``"UNIFIED"`` row (:func:`kv_views_unified`)."""
    layout = str(kv_layout).upper()
    if layout == _LAYOUT_UNIFIED:
        return kv_views_unified(
            kv_cache, geometry=geometry, page_size=page_size, dtype=dtype
        )
    if layout not in _LAYOUTS_FI:
        raise ValueError(
            f"{_PKG}: kv_layout must be 'NHD', 'HND' or 'UNIFIED', got {kv_layout!r}"
        )
    try:
        return _api35.kv_views(
            kv_cache, layout, geometry=geometry, page_size=page_size, dtype=dtype
        )
    except ValueError as exc:
        raise _rebrand(exc) from None


def _page_size_of(kv_cache: KvCache, kv_layout: str, geometry) -> int:
    """The page size a cache buffer carries in its shape (the functional entry point reads it there)."""
    layout = str(kv_layout).upper()
    if layout == _LAYOUT_UNIFIED:
        t = _tensor(kv_cache, "kv_cache")
        if t.dim() != 4:
            raise ValueError(
                f"{_PKG}: a UNIFIED kv_cache is [num_pages, page_size, H_kv, 2 x D], got shape {tuple(t.shape)}"
            )
        return int(t.shape[1])
    try:
        return _api35._page_size_of(kv_cache, layout, geometry)
    except ValueError as exc:
        raise _rebrand(exc) from None


# --------------------------------------------------------------------------- the plan


@dataclass(eq=False)
class Qwen38BlockPlan:
    """One compiled declaration of the block for one decode shape bucket; see the module docstring.  Built by :func:`prepare`.
    The Qwen 3.5 plan's fields first, the decode indexer's appended."""

    form: str
    geometry: Qwen38BlockGeometry
    dtype: torch.dtype
    device: torch.device
    kv_layout: str
    page_size: int
    num_pages: int
    batch_size: int  #: sequences
    q_len: int  #: query rows per sequence
    max_pages: int  #: the table columns the plan addresses
    table_row_stride: int  #: the block table's row stride, compiled in
    max_seq_len: Optional[int]
    decode_split_kv: Optional[int]
    workspace_size: int
    declaration: Dict[str, Any]
    _block: Any = field(repr=False)
    _pool_shape: Tuple[int, ...] = field(repr=False)
    _k_strides: Tuple[int, ...] = field(repr=False)
    _v_strides: Tuple[int, ...] = field(repr=False)
    _workspace: Optional[torch.Tensor] = field(default=None, repr=False)
    live_rows_multiple: int = 1
    _captured: bool = field(default=False, repr=False)
    # appended: the decode indexer's facts
    index_cache_dtype: torch.dtype = (
        torch.bfloat16
    )  #: the paged compressed-key cache's dtype
    ring_cap: int = 0  #: the raw-key ring's rows per block the plan was declared with (a run may bind a ring with at least q_len + 3)
    ring_slots: bool = False  #: run(index_state_slot_mapping=) names the ring WRITE rows (the stack's own slots)
    rope_position_tables_only: bool = True  #: every rotation reads the position tables; rope_cos / rope_sin are None at run
    fuse_prepare: bool = True  #: the resolved prepare-fusion knob (one launch vs the four-launch twin; bitwise)
    use_pdl: bool = False  #: the resolved programmatic-dependent-launch switch
    kv_cache_dtype: torch.dtype = (
        torch.bfloat16
    )  #: the pools' dtype (bf16; the e4m3 pool is a typed decline today)
    _position_tables: Optional[PositionTables] = field(default=None, repr=False)
    # appended (the lean decode tile): the sparse core's body this plan compiled -- the resolved form ("lean" / "tcgen05") and its
    # kernel_template (support.SPARSE_TILE_LEAN / SPARSE_TILE_TCGEN05), pinned by the vendored block against the kernel that compiled
    sparse_tile_form: Optional[str] = None
    sparse_tile: Optional[str] = None

    @property
    def num_tokens(self) -> int:
        """Token rows a run binds: ``batch_size * q_len``."""
        return self.batch_size * self.q_len

    @property
    def position_tables(self) -> Optional[PositionTables]:
        """The ``(cos_pos, sin_pos)`` tables the plan was prepared with: what a run binds when handed no pair of its own (the functional
        entry hands its call's pair to every run)."""
        return self._position_tables

    def _run_position_tables(self, tables: Optional[PositionTables]) -> PositionTables:
        """The pair THIS run binds: the caller's, checked against the plan (``[1 or batch, >= max_pages x page_size, rope_dim]``, the
        plan's dtype and device -- the functional entry hands its call's pair here on every call, cache hit or not, so a bucket never
        rotates with the pair another call compiled in), else the pair the plan was prepared with."""
        if tables is None:
            return self._position_tables
        return _check_position_tables(
            tables,
            self.geometry,
            self.max_pages * self.page_size,
            self.dtype,
            self.device,
        )

    def supports(self, **call) -> bool:
        """Whether a call with these keyword tensors (the ``run`` keywords) fits this plan: the host checks of ``run`` without a launch."""
        try:
            self._run_position_tables(call.pop("position_tables", None))
            self._bind(**call)
        except (ValueError, TypeError):
            return False
        return True

    def _pools(self, kv_cache: KvCache) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = kv_views(
            kv_cache,
            self.kv_layout,
            geometry=self.geometry,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
        )
        if (
            tuple(k.shape) != self._pool_shape
            or tuple(k.stride()) != self._k_strides
            or tuple(v.stride()) != self._v_strides
        ):
            raise ValueError(
                f"{_PKG}: kv_cache does not match the plan's pools (plan: {self._pool_shape} K strides {self._k_strides} V strides {self._v_strides}; "
                f"got {tuple(k.shape)} K strides {tuple(k.stride())} V strides {tuple(v.stride())}); another page count or layout is another plan"
            )
        return k, v

    def _workspace_for(self, workspace_buffer: Optional[torch.Tensor]) -> torch.Tensor:
        if workspace_buffer is None:
            if self._workspace is None:
                raise RuntimeError(
                    f"{_PKG}: this plan owns no workspace; pass workspace_buffer"
                )
            return self._workspace
        ws = _tensor(workspace_buffer, "workspace_buffer")
        _on(ws, self.device, "workspace_buffer")
        if (
            ws.dtype != torch.uint8
            or ws.dim() != 1
            or not ws.is_contiguous()
            or int(ws.numel()) < self.workspace_size
        ):
            raise ValueError(
                f"{_PKG}: workspace_buffer must be a contiguous uint8 buffer of at least plan.workspace_size = {self.workspace_size} bytes"
            )
        return ws

    def _bind(
        self,
        *,
        hidden_states=None,
        rope_cos=None,
        rope_sin=None,
        kv_cache=None,
        slot_mapping=None,
        block_table=None,
        seq_lens=None,
        query_start_loc=None,
        num_actual_tokens=None,
        out=None,
        index_kv_cache=None,
        index_state_cache=None,
        index_state_block_table=None,
        index_state_slot_mapping=None,
        k_scale=None,
        v_scale=None,
        block_ids_out=None,
        block_lens_out=None,
    ) -> Dict[str, Any]:
        """The host-side form checks of one call, as the arguments the block's ``execute`` takes."""
        g, dev, dt = self.geometry, self.device, self.dtype
        dm, rope = g.d_model, g.rope_dim
        b, s = self.batch_size, self.q_len
        t = b * s
        bound: Dict[str, Any] = {}
        if query_start_loc is not None or num_actual_tokens is not None:
            raise ValueError(
                f"{_PKG}: the decode form takes no query_start_loc / num_actual_tokens (a uniform [batch, {s}] step)"
            )
        if k_scale is not None or v_scale is not None:
            raise ValueError(
                f"{_PKG}: k_scale / v_scale are the e4m3 pools' descales; this plan's pools are {self.kv_cache_dtype} (the e4m3 pool under the "
                "in-block indexer is a typed decline until the sparse core's in-kernel dequant arm lands) -- pass None"
            )
        if kv_cache is not None:
            bound["k_cache"], bound["v_cache"] = self._pools(kv_cache)
        if hidden_states is not None:
            h_rows, _, rank = _rows(hidden_states, (t,), dm, "hidden_states", dt, dev)
            bound["h"] = h_rows.view(b, s, dm)
            bound["out"] = (
                _rows(out, (t,), dm, "out", dt, dev)[0].view(b, s, dm)
                if out is not None
                else None
            )
            bound["rows"], bound["rank"] = t, rank
        if self.rope_position_tables_only:
            if rope_cos is not None or rope_sin is not None:
                raise ValueError(
                    f"{_PKG}: rope_cos / rope_sin are refused on a plan with rope_position_tables_only=True: every rotation of the step reads the "
                    "plan's position tables at the row's device-derived position -- pass None (or plan with rope_position_tables_only=False)"
                )
            bound["cos"] = bound["sin"] = None
        elif hidden_states is not None:
            if rope_cos is None or rope_sin is None:
                raise ValueError(
                    f"{_PKG}: rope_cos / rope_sin (the per-token tables [batch x q_len, rope_dim]) are required on a plan with "
                    "rope_position_tables_only=False"
                )
            bound["cos"] = _rows(rope_cos, (t,), rope, "rope_cos", dt, dev)[0].view(
                b, s, rope
            )
            bound["sin"] = _rows(rope_sin, (t,), rope, "rope_sin", dt, dev)[0].view(
                b, s, rope
            )
        if slot_mapping is not None:
            bound["slot_mapping"] = _slots(slot_mapping, (t,), dev)
        if seq_lens is not None:
            bound["kv_lens"] = _int_vector(seq_lens, (b,), "seq_lens", dev)
        if block_table is not None:
            bound["block_table"] = _block_table(
                block_table, b, self.max_pages, self.table_row_stride, dev
            )
        if index_kv_cache is not None:
            bound["index_kv_cache"] = _check_index_kv_cache(
                index_kv_cache,
                self.num_pages,
                self.page_size,
                g,
                self.index_cache_dtype,
                dev,
            )
        if index_state_cache is not None:
            bound["index_state_cache"] = _check_index_state_cache(
                index_state_cache, s, g, dt, dev, ring_slots=self.ring_slots
            )
        if index_state_block_table is not None:
            bound["index_state_block_table"] = _check_index_state_block_table(
                index_state_block_table, b, dev
            )
        if self.ring_slots:
            if index_state_slot_mapping is None:
                if slot_mapping is not None:
                    raise ValueError(
                        f"{_PKG}: this plan was prepared with ring_slots=True: run needs index_state_slot_mapping [batch x q_len] (the stack's ring "
                        "slots block x cap + pos % cap, -1 = no write) of slot_mapping's dtype -- no silent derived fallback"
                    )
            else:
                x = _int_vector(
                    index_state_slot_mapping,
                    (t,),
                    "index_state_slot_mapping",
                    dev,
                    dtypes=_INT_DTYPES,
                )
                if slot_mapping is not None and x.dtype != bound["slot_mapping"].dtype:
                    raise ValueError(
                        f"{_PKG}: index_state_slot_mapping is {x.dtype} but slot_mapping is {bound['slot_mapping'].dtype}: the ring slots share the slot "
                        "mapping's dtype (one artifact per slot dtype) -- hand both in one dtype, never a cast"
                    )
                bound["index_state_slot_mapping"] = x
        elif index_state_slot_mapping is not None:
            raise ValueError(
                f"{_PKG}: index_state_slot_mapping is given, but this plan was prepared with ring_slots=False (the ring rows derive from "
                "index_state_block_table and seq_lens in the kernel; the flag is a compile key) -- plan with ring_slots=True, or pass None"
            )
        for name, x, shape2, shape3 in (
            ("block_ids_out", block_ids_out, (t, g.top_k), (b, s, g.top_k)),
            ("block_lens_out", block_lens_out, (t,), (b, s)),
        ):
            if x is None:
                continue
            x = _tensor(x, name)
            _on(x, dev, name)
            if (
                x.dtype != torch.int32
                or tuple(int(v) for v in x.shape) not in (shape2, shape3)
                or not x.is_contiguous()
            ):
                raise ValueError(
                    f"{_PKG}: {name} must be a contiguous int32 {list(shape2)} (or {list(shape3)}) buffer, got {x.dtype} {tuple(x.shape)}"
                )
            bound[name] = x
        return bound

    def run(
        self,
        hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        weights: Qwen38BlockWeights,
        w_o: torch.Tensor,
        *,
        kv_cache: KvCache,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
        index_kv_cache: torch.Tensor,
        index_state_cache: torch.Tensor,
        index_state_block_table: torch.Tensor,
        index_state_slot_mapping: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        block_ids_out: Optional[torch.Tensor] = None,
        block_lens_out: Optional[torch.Tensor] = None,
        position_tables: Optional[PositionTables] = None,
    ) -> torch.Tensor:
        """Launch the plan on the current CUDA stream (capturable from the FIRST run; see the README's capture rules).  Returns ``out``.
        ``position_tables`` binds that ``(cos_pos, sin_pos)`` pair for THIS run (checked against the plan: the form, the reach, the
        dtype, the device); ``None`` binds the pair the plan was prepared with."""
        g, dev, dt = self.geometry, self.device, self.dtype
        if not self._captured and _capturing():
            self._captured = True  # a graph bakes this plan's workspace: the plan LRU must not free it at eviction
        _check_weights(weights, w_o, g, dt, dev)
        b = self._bind(
            hidden_states=hidden_states,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_table=block_table,
            seq_lens=seq_lens,
            out=out,
            index_kv_cache=index_kv_cache,
            index_state_cache=index_state_cache,
            index_state_block_table=index_state_block_table,
            index_state_slot_mapping=index_state_slot_mapping,
            k_scale=k_scale,
            v_scale=v_scale,
            block_ids_out=block_ids_out,
            block_lens_out=block_lens_out,
        )
        ws = self._workspace_for(workspace_buffer)
        cos_pos, sin_pos = self._run_position_tables(position_tables)
        o = (
            b["out"]
            if b["out"] is not None
            else torch.empty(
                self.batch_size, self.q_len, g.d_model, dtype=dt, device=dev
            )
        )
        self._block.execute(
            b["h"],
            weights.w_qkvg,
            weights.w_q_norm,
            weights.w_k_norm,
            b["cos"],
            b["sin"],
            w_o,
            o,
            ws,
            k_cache=b["k_cache"],
            v_cache=b["v_cache"],
            block_table=b["block_table"],
            kv_lens=b["kv_lens"],
            slot_mapping=b["slot_mapping"],
            w_iq_norm=weights.w_iq_norm,
            w_ik_norm=weights.w_ik_norm,
            block_ids_out=b.get("block_ids_out"),
            block_lens_out=b.get("block_lens_out"),
            index_kv_cache=b["index_kv_cache"],
            index_state_cache=b["index_state_cache"],
            index_state_block_table=b["index_state_block_table"],
            index_state_slot_mapping=b.get("index_state_slot_mapping"),
            cos_pos=cos_pos,
            sin_pos=sin_pos,
        )
        if out is not None:
            return out
        return o if b["rank"] == 3 else o.view(b["rows"], g.d_model)


def _light_decode_gates(
    geometry: Qwen38BlockGeometry,
    page_size: int,
    q_len: int,
    kv_cache_dtype,
    use_pdl,
    sparse_tile_form: str = SPARSE_TILE_FORM_AUTO,
) -> None:
    """The decode form's typed declines that need no vendored code (the mirrored constants): the page sizes the paged scorer addresses,
    the scorer's query-row envelope, the e4m3 pool, the sparse core's body vocabulary.  The vendored declaration repeats them with the
    envelope's own texts."""
    if page_size not in PAGE_SIZES:
        raise NotImplementedError(
            f"{_PKG} (decode): the paged scorer splits a block id into its page and row by a shift, so page_size must be one of {PAGE_SIZES} "
            f"(a power-of-two count of {geometry.block_size}-token compressed rows per page); got page_size={page_size}"
        )
    rows = q_len * next_pow2(geometry.index_heads)
    if rows > INDEX_QUERY_ROWS_MAX:
        raise NotImplementedError(
            f"{_PKG} (decode): q_len_per_req={q_len} x next_pow2(index_heads={geometry.index_heads}) = {rows} query rows per sequence is past "
            f"the decode scorer's envelope ({INDEX_QUERY_ROWS_MAX}: four 16-row MMA tiles) -- at most q_len_per_req={geometry.decode_max_q_len} "
            "at this head count"
        )
    if kv_cache_dtype is not None and kv_cache_dtype not in (torch.bfloat16,):
        raise NotImplementedError(
            f"{_PKG} (decode): kv_cache_dtype={kv_cache_dtype}: the block-sparse decode step with the in-block indexer serves bf16 pools; the "
            "e4m3 pool waits for the sparse core's in-kernel dequant arm (a later cuDNN Frontend PR) -- declare kv_cache_dtype=None"
        )
    if use_pdl is not None and not isinstance(use_pdl, bool):
        raise TypeError(
            f"{_PKG}: use_pdl must be None (the vendored default) or a bool, got {type(use_pdl).__name__}"
        )
    if (
        not isinstance(sparse_tile_form, str)
        or sparse_tile_form not in SPARSE_TILE_FORMS
    ):
        raise ValueError(
            f"{_PKG}: sparse_tile_form must be one of {SPARSE_TILE_FORMS} ('auto' = the vendored placement rule: the lean decode tile on cc 10.7 "
            f"at batch x H_kv <= 4 CTAs per chunk where the body admits the step, the tcgen05 core elsewhere; 'lean' / 'tcgen05' explicit), got {sparse_tile_form!r}"
        )


def prepare(
    form: str,
    *,
    geometry: Qwen38BlockGeometry,
    device,
    weights: Qwen38BlockWeights,
    w_o: torch.Tensor,
    kv_cache: KvCache,
    page_size: int,
    batch_size: int,
    dtype: torch.dtype = torch.bfloat16,
    kv_layout: str = "NHD",
    q_len_per_req: int = 1,
    max_pages: Optional[int] = None,
    block_table: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
    decode_split_kv: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    # appended (the Qwen 3.8 decode form)
    index_kv_cache: Optional[torch.Tensor] = None,
    index_state_cache: Optional[torch.Tensor] = None,
    index_state_block_table: Optional[torch.Tensor] = None,
    position_tables: Optional[PositionTables] = None,
    rope_position_tables_only: bool = True,
    ring_slots: bool = False,
    fuse_prepare: Optional[bool] = None,
    use_pdl: Optional[bool] = None,
    kv_cache_dtype: Optional[torch.dtype] = None,
    _declaration_overrides: Optional[Dict[str, Any]] = None,
    sparse_tile_form: str = SPARSE_TILE_FORM_AUTO,
) -> Qwen38BlockPlan:
    """Declare, check and compile the block for one decode shape bucket; see :func:`flashinfer.qwen_3_8_block.prepare_qwen_3_8_block_bf16`.

    ``_declaration_overrides`` is a test hook, not part of the API: extra keywords for the vendored declaration; the public entry points
    never pass it."""
    if form not in _support.ALL_FORMS:
        raise ValueError(
            f"{_PKG}: form must be one of {_support.ALL_FORMS}, got {form!r}"
        )
    if not isinstance(geometry, Qwen38BlockGeometry):
        raise TypeError(
            f"{_PKG}: geometry must be a Qwen38BlockGeometry, got {type(geometry).__name__}"
        )
    _support.require_served(device, geometry, dtype, form)
    assert form in FORMS
    dev = _device_of(device)
    _check_not_capturing("prepare")
    caches = configure_vendored_tree()
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or page_size <= 0
        or page_size % PAGE_SIZE_ALIGN
    ):
        raise ValueError(
            f"{_PKG}: page_size must be a positive multiple of {PAGE_SIZE_ALIGN} (validated: {PAGE_SIZES}), got {page_size!r}"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(
            f"{_PKG}: batch_size must be a positive int, got {batch_size!r}"
        )
    q = q_len_per_req
    if isinstance(q, bool) or not isinstance(q, int) or q <= 0:
        raise ValueError(f"{_PKG}: q_len_per_req must be a positive int, got {q!r}")
    for nm, val in (
        ("rope_position_tables_only", rope_position_tables_only),
        ("ring_slots", ring_slots),
    ):
        if not isinstance(val, bool):
            raise TypeError(f"{_PKG}: {nm} must be a bool, got {type(val).__name__}")
    if fuse_prepare is not None and not isinstance(fuse_prepare, bool):
        raise TypeError(
            f"{_PKG}: fuse_prepare must be None (the vendored measured default) or a bool, got {type(fuse_prepare).__name__}"
        )
    if decode_split_kv is not None and (
        isinstance(decode_split_kv, bool)
        or not isinstance(decode_split_kv, int)
        or decode_split_kv < 1
    ):
        raise ValueError(
            f"{_PKG}: decode_split_kv must be None or an int >= 1, got {decode_split_kv!r}"
        )
    _light_decode_gates(
        geometry, page_size, q, kv_cache_dtype, use_pdl, sparse_tile_form
    )
    pool_dtype = dtype if kv_cache_dtype is None else kv_cache_dtype
    _check_weights(weights, w_o, geometry, dtype, dev)
    k, v = kv_views(
        kv_cache, kv_layout, geometry=geometry, page_size=page_size, dtype=pool_dtype
    )
    if k.device != dev:
        raise ValueError(f"{_PKG}: kv_cache lives on {k.device}, the plan on {dev}")
    num_pages = int(k.shape[0])
    # the block table sample: the stack's own (its row stride compiles in; the sparse core binds the WHOLE table, so the sample must be
    # contiguous -- a leading-row slice of a stack table at its full width is) or a contiguous [batch, max_pages] stand-in
    if block_table is not None:
        bt = _tensor(block_table, "block_table")
        _on(bt, dev, "block_table")
        if (
            bt.dtype != torch.int32
            or bt.dim() != 2
            or int(bt.shape[0]) < batch_size
            or bt.stride(1) != 1
        ):
            raise ValueError(
                f"{_PKG}: block_table must be an int32 [>= {batch_size}, max_pages] table with a contiguous page axis, got {bt.dtype} {tuple(bt.shape)}"
            )
        pages = int(bt.shape[1]) if max_pages is None else int(max_pages)
        if pages <= 0 or pages > int(bt.shape[1]):
            raise ValueError(
                f"{_PKG}: max_pages must be in [1, {int(bt.shape[1])}] (the table's columns), got {max_pages!r}"
            )
        if pages != int(bt.shape[1]):
            raise ValueError(
                f"{_PKG}: max_pages={pages} differs from block_table's {int(bt.shape[1])} columns: the block-sparse core binds the WHOLE table (a "
                "column slice is not contiguous) -- plan with max_pages=None (the table's width) or a table of that width"
            )
        table_sample = bt[:batch_size]
    else:
        if max_pages is None:
            raise ValueError(
                f"{_PKG}: pass max_pages (the table columns a step may address) or the block_table the steps will bind"
            )
        pages = int(max_pages)
        if pages <= 0:
            raise ValueError(f"{_PKG}: max_pages must be positive, got {max_pages!r}")
        table_sample = torch.zeros(batch_size, pages, dtype=torch.int32, device=dev)
    row_stride = int(table_sample.stride(0))
    if max_seq_len is not None and (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or not (1 <= max_seq_len <= pages * page_size)
    ):
        raise ValueError(
            f"{_PKG}: max_seq_len must be an int in [1, max_pages * page_size = {pages * page_size}], got {max_seq_len!r}"
        )
    # the side caches (REQUIRED) and the ring block table (a zeros sample when None: form only)
    if index_kv_cache is None or index_state_cache is None:
        raise ValueError(
            f"{_PKG}: the decode form needs index_kv_cache (the paged compressed-key cache [num_pages, page_size / {geometry.block_size}, 1, "
            f"{geometry.index_head_dim}] on the main block table) and index_state_cache (the raw-key ring [num_ring_blocks, cap >= q_len_per_req + "
            f"{INDEX_STATE_ROWS_EXTRA}, 1, {geometry.index_head_dim}]) at prepare: their form compiles in"
        )
    icache = _check_index_kv_cache(
        index_kv_cache, num_pages, page_size, geometry, dtype, dev
    )
    ring = _check_index_state_cache(
        index_state_cache, q, geometry, dtype, dev, ring_slots=ring_slots
    )
    if index_state_block_table is not None:
        # the documented sample is [>= batch_size, 1]: a wider stack table's leading rows (the block_table sample's rule above); the
        # run binds exactly [batch, 1]
        rt = _tensor(index_state_block_table, "index_state_block_table")
        ring_table = _check_index_state_block_table(
            rt[:batch_size] if rt.dim() == 2 else rt, batch_size, dev
        )
    else:
        ring_table = torch.zeros(batch_size, 1, dtype=torch.int32, device=dev)
    if position_tables is None:
        raise ValueError(
            f"{_PKG}: the decode form needs position_tables = (cos_pos, sin_pos) at prepare (qwen_3_8_block_rope_position_tables(cos_sin_cache, "
            f"num_positions >= max_pages x page_size = {pages * page_size})): the block rotates the completing block's compressed key at the block "
            "START, up to three positions below the step, which per-token rows do not hold"
        )
    cos_pos, sin_pos = _check_position_tables(
        position_tables, geometry, pages * page_size, dtype, dev
    )
    g = geometry.frost()
    dm, rope = geometry.d_model, geometry.rope_dim
    Fwd = _frost_block()
    # The geometry gate require_served ran mirrors the vendored tree's rules; with the tree imported, re-read them -- a re-vendor that
    # moved a rule fails here, loudly, never as a plan the gate and the tree read differently.
    _support.check_vendored_rules()

    def mk(*shape):
        return torch.empty(*shape, dtype=dtype, device=dev)

    # the measured declaration: the unfused projection (the band's slab GEMM on the small-M route), the gate fused (the core's epilogue
    # unsplit, the split combine otherwise), the prepare fusion / PDL at the vendored defaults unless asked
    knobs: Dict[str, Any] = dict(fuse_norm_rope=False, fuse_gate=True)
    if _declaration_overrides:
        knobs.update(_declaration_overrides)
    tables = (
        (None, None)
        if rope_position_tables_only
        else (mk(batch_size, q, rope), mk(batch_size, q, rope))
    )
    blk = Fwd(
        mk(batch_size, q, dm),
        weights.w_qkvg,
        weights.w_q_norm,
        weights.w_k_norm,
        tables[0],
        tables[1],
        w_o,
        mk(batch_size, q, dm),
        g,
        paged_kv_page_size=page_size,
        kv_source="paged",
        sample_k_cache=k,
        sample_v_cache=v,
        sample_block_table=table_sample,
        max_kv_len=max_seq_len,
        decode_split_kv=decode_split_kv,
        fuse_prepare=fuse_prepare,
        use_pdl=use_pdl,
        kv_cache_dtype=kv_cache_dtype,
        sample_index_kv_cache=icache,
        sample_index_state_cache=ring,
        sample_index_state_block_table=ring_table,
        index_state_slot_mapping_present=ring_slots,
        rope_position_tables_only=rope_position_tables_only,
        sparse_tile_form=sparse_tile_form,
        **knobs,
    )
    # The projection GEMMs on the capture-safe JIT route by construction (the band slab's 128-row tile at batch x q_len <= 128); the
    # hook reaches into the vendored block -- a re-vendor that renames these attributes must fail here, loudly, not drop the route.
    missing = [
        a
        for a in (
            "_proj",
            "_out_proj",
            "_stages",
            "_indexer_decode",
            "sparse_tile_form",
            "sparse_tile",
        )
        if not hasattr(blk, a)
    ]
    if missing:
        raise RuntimeError(
            f"{_PKG}: the vendored GatedAttentionBlockFwd no longer carries {missing}; the adapter's capture-safe hook (api.prepare) must be "
            "updated with the re-vendored tree"
        )
    for st in (blk._proj, blk._out_proj):
        if st is not None:
            st.capture_safe = True
    blk.check_support()
    blk.compile()
    size = int(blk.get_workspace_size())
    if workspace_buffer is not None:
        ws = _tensor(workspace_buffer, "workspace_buffer")
        _on(ws, dev, "workspace_buffer")
        if (
            ws.dtype != torch.uint8
            or ws.dim() != 1
            or not ws.is_contiguous()
            or int(ws.numel()) < size
        ):
            raise ValueError(
                f"{_PKG}: workspace_buffer must be a contiguous uint8 buffer of at least plan.workspace_size = {size} bytes (got "
                f"{int(ws.numel()) if isinstance(ws, torch.Tensor) else 'n/a'})"
            )
        owned = None
    else:
        owned = torch.zeros(size, dtype=torch.uint8, device=dev)
    ixd = blk._indexer_decode
    out_proj_tile = getattr(
        getattr(blk._out_proj, "_plan", None), "tile_config_name", None
    )
    proj_tile = getattr(getattr(blk._proj, "_plan", None), "tile_config_name", None)
    split = int(getattr(blk, "decode_split_kv", decode_split_kv or 1) or 1)
    fused = bool(blk.fuse_prepare)
    # the launch count of one execute by the stage list: the slab GEMM, the prepare (one launch fused; norm + RoPE and the write-through
    # unfused), the indexer stage's own launches (2 fused / 4 unfused), the sparse tile (+ the combine under a split), the out projection
    # (+ its split-K reducer where the small-row tile splits K); the profiler census is the pin, this is the statement
    launches = (
        1
        + (1 if fused else 2)
        + int(ixd.launches_per_execute())
        + 1
        + (1 if split > 1 else 0)
        + 1
        + (1 if (out_proj_tile and "splitK" in str(out_proj_tile)) else 0)
    )
    declaration = dict(
        form=form,
        page_size=page_size,
        kv_layout=str(kv_layout).upper(),
        num_pages=num_pages,
        batch_size=batch_size,
        rows=q,
        max_pages=pages,
        table_row_stride=row_stride,
        max_seq_len=max_seq_len,
        stages=[type(s).__name__ for s in blk._stages],
        gate_fused=bool(getattr(blk, "gate_fused", False)),
        cache_write_fused=bool(getattr(blk, "cache_write_fused", False)),
        fuse_norm_rope_effective=bool(
            getattr(
                blk, "fuse_norm_rope_effective", getattr(blk, "fuse_norm_rope", False)
            )
        ),
        cache_write_in_norm_rope=bool(getattr(blk, "cache_write_in_norm_rope", False)),
        use_pdl=bool(getattr(blk, "use_pdl", False)),
        use_pdl_requested=getattr(blk, "use_pdl_requested", None),
        use_pdl_effective=bool(getattr(blk, "use_pdl_effective", False)),
        pdl_stages=list(getattr(blk, "pdl_stages", ())),
        decode_split_kv=getattr(blk, "decode_split_kv", None),
        decode_route=None,
        out_proj_tile=out_proj_tile,
        proj_tile=proj_tile,
        live_rows_multiple=1,
        caches=caches,
        workspace_size=size,
        # appended: the decode indexer's readbacks
        index_list_form=getattr(blk, "qsa_list_form", None),
        index_cache_dtype=dtype,
        kv_cache_dtype=pool_dtype,
        fuse_prepare=fused,
        fuse_prepare_requested=getattr(blk, "fuse_prepare_requested", fuse_prepare),
        ring_slots=ring_slots,
        rope_position_tables_only=rope_position_tables_only,
        ring_cap=int(ring.shape[1]),
        n_blocks_max=int(ixd.n_blocks_max),
        topk_knobs=tuple(ixd.topk_knobs) if ixd.topk_knobs is not None else None,
        indexer_launches=int(ixd.launches_per_execute()),
        launches_per_execute=launches,
        capture_rows_max=DECODE_CAPTURE_ROWS_MAX,
        capture_safe_rows=bool(batch_size * q <= DECODE_CAPTURE_ROWS_MAX),
        position_table_rows=int(cos_pos.shape[1]),
        # appended: the sparse core's body (the lean decode tile beside the tcgen05 core): the declaration, the resolved body, its
        # kernel_template (pinned by the vendored block against the compiled kernel) and the lean tile's sub-tiles per CTA (None on the core)
        sparse_tile_form_requested=getattr(
            blk, "sparse_tile_form_requested", sparse_tile_form
        ),
        sparse_tile_form=getattr(blk, "sparse_tile_form", None),
        sparse_tile=getattr(blk, "sparse_tile", None),
        sparse_sub_tiles_per_cta=getattr(blk, "sparse_sub_tiles_per_cta", None),
    )
    return Qwen38BlockPlan(
        form=form,
        geometry=geometry,
        dtype=dtype,
        device=dev,
        kv_layout=str(kv_layout).upper(),
        page_size=page_size,
        num_pages=num_pages,
        batch_size=batch_size,
        q_len=q,
        max_pages=pages,
        table_row_stride=row_stride,
        max_seq_len=max_seq_len,
        decode_split_kv=getattr(blk, "decode_split_kv", decode_split_kv),
        workspace_size=size,
        declaration=declaration,
        _block=blk,
        _pool_shape=tuple(int(x) for x in k.shape),
        _k_strides=tuple(int(x) for x in k.stride()),
        _v_strides=tuple(int(x) for x in v.stride()),
        _workspace=owned,
        index_cache_dtype=dtype,
        ring_cap=int(ring.shape[1]),
        ring_slots=ring_slots,
        rope_position_tables_only=rope_position_tables_only,
        fuse_prepare=fused,
        use_pdl=bool(getattr(blk, "use_pdl", False)),
        kv_cache_dtype=pool_dtype,
        _position_tables=(cos_pos, sin_pos),
        sparse_tile_form=getattr(blk, "sparse_tile_form", None),
        sparse_tile=getattr(blk, "sparse_tile", None),
    )


def workspace_size(
    form: str,
    *,
    geometry: Qwen38BlockGeometry,
    device,
    page_size: int,
    batch_size: int,
    q_len_per_req: int = 1,
    max_pages: int,
    dtype: torch.dtype = torch.bfloat16,
    decode_split_kv: Optional[int] = None,
    fuse_prepare: Optional[bool] = None,
    ring_cap: Optional[int] = None,
    sparse_tile_form: str = SPARSE_TILE_FORM_AUTO,
) -> int:
    """The workspace bytes a plan of this shape bucket needs, with no plan kept -- what a serving stack sizes its one shared
    workspace with at init (before its memory profiling run, when the caches do not exist yet): the vendored block is declared on
    stand-in samples of the bucket's FORM (one page, a zeros table of ``max_pages`` columns, a ring of ``ring_cap`` rows), checked,
    COMPILED (the carve includes the projection engines' scratch, which is the compiled plan's; the kernels come from the in-process
    memo or the compiled-plan cache after the first process) and its workspace carve read.  The carve depends on the bucket
    (``batch_size x q_len_per_req`` rows, ``max_pages x page_size / block_size`` compressed rows, the split), never on the caches' page
    count or strides, so the number equals the compiled plan's (``test_decode.py`` pins it).  ``sparse_tile_form`` (appended) must be the
    plan's: the carve is the BODY's (the lean tile's split writes a wider partial slab than the core's).  Same typed declines as
    :func:`prepare`."""
    if form not in _support.ALL_FORMS:
        raise ValueError(
            f"{_PKG}: form must be one of {_support.ALL_FORMS}, got {form!r}"
        )
    if not isinstance(geometry, Qwen38BlockGeometry):
        raise TypeError(
            f"{_PKG}: geometry must be a Qwen38BlockGeometry, got {type(geometry).__name__}"
        )
    _support.require_served(device, geometry, dtype, form)
    dev = _device_of(device)
    _check_not_capturing("workspace_size")
    configure_vendored_tree()
    q = int(q_len_per_req)
    if isinstance(q_len_per_req, bool) or q <= 0:
        raise ValueError(
            f"{_PKG}: q_len_per_req must be a positive int, got {q_len_per_req!r}"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(
            f"{_PKG}: batch_size must be a positive int, got {batch_size!r}"
        )
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages <= 0:
        raise ValueError(
            f"{_PKG}: max_pages must be a positive int (the block table's columns), got {max_pages!r}"
        )
    _light_decode_gates(geometry, page_size, q, None, None, sparse_tile_form)
    cap = int(ring_cap) if ring_cap is not None else q + INDEX_STATE_ROWS_EXTRA
    if cap < q + INDEX_STATE_ROWS_EXTRA:
        raise ValueError(
            f"{_PKG}: ring_cap={cap} is below q_len_per_req + {INDEX_STATE_ROWS_EXTRA} = {q + INDEX_STATE_ROWS_EXTRA}"
        )
    g = geometry.frost()
    dm, d, h_kv = geometry.d_model, geometry.head_dim, geometry.num_kv_heads
    Fwd = _frost_block()
    _support.check_vendored_rules()

    def mk(*shape):
        return torch.empty(*shape, dtype=dtype, device=dev)

    blk = Fwd(
        mk(batch_size, q, dm),
        mk(geometry.n_qkvg, dm),
        mk(d) if geometry.qk_norm else None,
        mk(d) if geometry.qk_norm else None,
        None,
        None,
        mk(dm, geometry.num_qo_heads * d),
        mk(batch_size, q, dm),
        g,
        paged_kv_page_size=page_size,
        kv_source="paged",
        sample_k_cache=mk(1, h_kv, page_size, d),
        sample_v_cache=mk(1, h_kv, page_size, d),
        sample_block_table=torch.zeros(
            batch_size, int(max_pages), dtype=torch.int32, device=dev
        ),
        decode_split_kv=decode_split_kv,
        fuse_prepare=fuse_prepare,
        fuse_norm_rope=False,
        fuse_gate=True,
        sample_index_kv_cache=mk(
            1, page_size // geometry.block_size, 1, geometry.index_head_dim
        ),
        sample_index_state_cache=mk(1, cap, 1, geometry.index_head_dim),
        sample_index_state_block_table=torch.zeros(
            batch_size, 1, dtype=torch.int32, device=dev
        ),
        rope_position_tables_only=True,
        sparse_tile_form=sparse_tile_form,
    )
    for st in (blk._proj, blk._out_proj):
        if st is not None:
            st.capture_safe = True
    blk.check_support()
    blk.compile()
    return int(blk.get_workspace_size())


# --------------------------------------------------------------------------- the plan LRU behind the functional entry point
#
# The Qwen 3.5 package's ownership rules, one LRU per package: a plan owns its workspace, so evicting one frees device memory -- unless
# a CUDA graph was captured over it: such a plan is retained at its eviction and lives until ``clear_plan_cache()``; a retained bucket
# asked for again is revived into the LRU.  A bucket WITHOUT a cached plan cannot be built under capture, so the lookup refuses it there.

_PLANS: "collections.OrderedDict[tuple, Qwen38BlockPlan]" = collections.OrderedDict()
_RETAINED: "list[tuple[tuple, Qwen38BlockPlan]]" = []
_PLANS_LOCK = threading.Lock()
PLAN_CACHE_SIZE = 32


def clear_plan_cache() -> None:
    """Drop every plan of the functional entry point -- the LRU's and the retained ones captured graphs ran -- and with them their
    workspaces.  The ONLY release of a retained plan: call it once no CUDA graph captured over a functional-entry plan can replay any
    more (after ``graph.reset()``, at model teardown), never while one may; under capture it is refused."""
    if _capturing():
        raise RuntimeError(
            f"{_PKG}: clear_plan_cache() releases the workspaces captured graphs replay into and cannot run under CUDA-graph capture"
        )
    with _PLANS_LOCK:
        _PLANS.clear()
        _RETAINED.clear()


def _insert_plan(key: tuple, plan: Qwen38BlockPlan) -> Qwen38BlockPlan:
    cached = _PLANS.get(key)
    if cached is not None and cached is not plan:
        if plan._captured:
            _RETAINED.append((key, plan))
        plan = cached
    _PLANS[key] = plan
    _PLANS.move_to_end(key)
    while len(_PLANS) > PLAN_CACHE_SIZE:
        k, evicted = _PLANS.popitem(last=False)
        if evicted._captured:
            _RETAINED.append((k, evicted))
    return plan


def _revive_plan(key: tuple) -> Optional[Qwen38BlockPlan]:
    for i, (k, plan) in enumerate(_RETAINED):
        if k == key:
            del _RETAINED[i]
            return _insert_plan(key, plan)
    return None


def cached_plan(
    form: str,
    *,
    geometry,
    device,
    weights,
    w_o,
    kv_cache,
    kv_layout,
    page_size,
    batch_size,
    dtype=torch.bfloat16,
    **kw,
) -> Qwen38BlockPlan:
    """The plan of the functional entry point: an LRU keyed on exactly what a plan compiles in -- the Qwen 3.5 key (the form, the
    geometry with its indexer fields, the dtype, the device, the pools' layout / shape / strides, the batch, the query rows, the table's
    columns and row stride, the KV bound, the split) plus the side caches' FORM (shapes, dtypes), the two input-contract flags, the knobs
    (the sparse core's body knob included: two bodies are two compiled plans) and the pools' dtype.  NOT in the key: the side caches' strides (symbolic in the kernels: a stack may re-allocate them with other
    strides without a new plan) and the position tables (the functional entry hands THIS call's pair to every run, checked against the
    plan's form there; the pair the first call compiled in is only the plan's default for an explicit run that hands none).

    Under CUDA-graph capture only a bucket already cached is served (its plan is then retained past eviction); a cold bucket raises a
    typed ``RuntimeError`` naming the remedy -- one eager call outside the capture warms it."""
    dev = _device_of(device)
    pool_dtype = dtype if kw.get("kv_cache_dtype") is None else kw["kv_cache_dtype"]
    k, v = kv_views(
        kv_cache, kv_layout, geometry=geometry, page_size=page_size, dtype=pool_dtype
    )
    bt = kw.get("block_table")
    icache, ring = kw.get("index_kv_cache"), kw.get("index_state_cache")
    key = (
        form,
        geometry,
        dtype,
        dev.index,
        str(kv_layout).upper(),
        page_size,
        tuple(int(x) for x in k.shape),
        tuple(int(x) for x in k.stride()),
        tuple(int(x) for x in v.stride()),
        batch_size,
        kw.get("q_len_per_req", 1),
        kw.get("max_pages"),
        None if bt is None else (int(bt.shape[1]), int(bt.stride(0))),
        kw.get("max_seq_len"),
        kw.get("decode_split_kv"),
        # appended
        None
        if not isinstance(icache, torch.Tensor)
        else (tuple(int(x) for x in icache.shape), icache.dtype),
        None
        if not isinstance(ring, torch.Tensor)
        else (tuple(int(x) for x in ring.shape), ring.dtype),
        bool(kw.get("ring_slots", False)),
        bool(kw.get("rope_position_tables_only", True)),
        kw.get("fuse_prepare"),
        kw.get("use_pdl"),
        pool_dtype,
        kw.get("sparse_tile_form", SPARSE_TILE_FORM_AUTO),
    )
    capturing = _capturing()
    with _PLANS_LOCK:
        plan = _PLANS.get(key)
        if plan is not None:
            _PLANS.move_to_end(key)
        else:
            plan = _revive_plan(key)
        if plan is not None:
            if capturing:
                plan._captured = True
            return plan
    if capturing:
        raise RuntimeError(
            f"{_PKG}: this shape bucket has no cached plan, and CUDA-graph capture cannot build one (a plan compiles kernels and allocates "
            "its workspace); warm the bucket outside the capture -- one eager call of the same entry point -- or run an explicit plan "
            "(prepare_qwen_3_8_block_bf16 / Qwen38BlockWrapper), then capture"
        )
    plan = prepare(
        form,
        geometry=geometry,
        device=dev,
        weights=weights,
        w_o=w_o,
        kv_cache=kv_cache,
        page_size=page_size,
        batch_size=batch_size,
        dtype=dtype,
        kv_layout=kv_layout,
        **kw,
    )
    with _PLANS_LOCK:
        return _insert_plan(key, plan)


def decode(
    hidden_states,
    rope_cos,
    rope_sin,
    weights,
    w_o,
    *,
    geometry,
    kv_cache,
    slot_mapping,
    block_table,
    seq_lens,
    index_kv_cache,
    index_state_cache,
    index_state_block_table,
    index_state_slot_mapping=None,
    position_tables=None,
    q_len_per_req=1,
    max_seq_len=None,
    decode_split_kv=None,
    kv_layout="NHD",
    out=None,
    workspace_buffer=None,
    plan=None,
    block_ids_out=None,
    block_lens_out=None,
    sparse_tile_form=SPARSE_TILE_FORM_AUTO,
):
    """The functional decode entry: the bucket's cached plan (the LRU; a cold bucket under capture refused) or the given one.  The
    arms follow the arguments: ``rope_cos / rope_sin`` None = the position-table arm (``position_tables`` then REQUIRED), given = the
    per-token arm (the position tables still required for the block starts); ``index_state_slot_mapping`` given = the ring-slots arm;
    ``sparse_tile_form`` (appended) is the sparse core's body knob of the bucket's plan (a plan-cache key; ignored with ``plan``)."""
    if plan is None:
        if position_tables is None:
            raise ValueError(
                f"{_PKG}: the functional decode entry binds THIS call's position_tables = (cos_pos, sin_pos) (qwen_3_8_block_rope_position_tables("
                "cos_sin_cache, num_positions >= max_pages x page_size)) -- the bucket's cached plan never substitutes the pair another call "
                "compiled in; a plan of your own (prepare_qwen_3_8_block_bf16 / Qwen38BlockWrapper) holds one"
            )
        rows, _, _ = _api35._rows_of(hidden_states)
        q = int(q_len_per_req)
        if q <= 0 or rows % q:
            raise ValueError(
                f"{_PKG}: hidden_states carries {rows} rows, not a multiple of q_len_per_req={q_len_per_req}"
            )
        page_size = _page_size_of(kv_cache, kv_layout, geometry)
        plan = cached_plan(
            "decode",
            geometry=geometry,
            device=_tensor(hidden_states, "hidden_states").device,
            weights=weights,
            w_o=w_o,
            kv_cache=kv_cache,
            kv_layout=kv_layout,
            page_size=page_size,
            batch_size=rows // q,
            q_len_per_req=q,
            block_table=block_table,
            max_seq_len=max_seq_len,
            decode_split_kv=decode_split_kv,
            index_kv_cache=index_kv_cache,
            index_state_cache=index_state_cache,
            index_state_block_table=index_state_block_table,
            position_tables=position_tables,
            rope_position_tables_only=rope_cos is None and rope_sin is None,
            ring_slots=index_state_slot_mapping is not None,
            sparse_tile_form=sparse_tile_form,
        )
    return plan.run(
        hidden_states,
        rope_cos,
        rope_sin,
        weights,
        w_o,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        block_table=block_table,
        seq_lens=seq_lens,
        out=out,
        workspace_buffer=workspace_buffer,
        index_kv_cache=index_kv_cache,
        index_state_cache=index_state_cache,
        index_state_block_table=index_state_block_table,
        index_state_slot_mapping=index_state_slot_mapping,
        block_ids_out=block_ids_out,
        block_lens_out=block_lens_out,
        position_tables=position_tables,
    )
