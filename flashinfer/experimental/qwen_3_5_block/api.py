# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The adapter: FlashInfer arguments -> vendored cuDNN Frontend block declarations; plans, workspaces and the three forms' runs.

A :class:`Qwen35BlockPlan` is one compiled declaration of the vendored ``GatedAttentionBlockFwd`` for one shape bucket
(the form, the batch / token capacities, the page size and the pools' layout, the block table's row stride, the decode
query length); ``prepare`` builds it outside CUDA-graph capture (it JIT-compiles the kernels and allocates the
workspace), ``run`` launches it -- the first ``run`` after ``prepare`` may already be captured.  The functional entry
points keep a small LRU of plans keyed on exactly the facts a plan compiles in, so a serving stack's shape buckets map
onto plans without any host-side bookkeeping of its own; a plan a CUDA graph was captured over outlives its eviction (the
graph baked its workspace) and is released by ``clear_plan_cache`` only.
"""

from __future__ import annotations

import collections
import functools
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, Union

import torch

from . import support as _support
from .support import (
    DECODE_MAX_ROWS,
    FORMS,
    PAGE_SIZE_ALIGN,
    PAGE_SIZES,
    Qwen35BlockGeometry,
)
from .weights import Qwen35BlockWeights

KvCache = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
_ALIGN_BYTES = 16
_INT_DTYPES = (torch.int32, torch.int64)

__all__ = [
    "KvCache",
    "Qwen35BlockPlan",
    "cached_plan",
    "clear_plan_cache",
    "configure_vendored_tree",
    "decode",
    "extend",
    "kv_views",
    "prefill",
    "prepare",
]


def _frost_block():
    from ._cudnn_frost.gated_attention_block import GatedAttentionBlockFwd

    return GatedAttentionBlockFwd


# --------------------------------------------------------------------------- the vendored tree in a FlashInfer process

# The vendored tree's two on-disk caches and the variables it reads for them (documented in the README, "Caches and environment").
_ENV_COMPILED_CACHE = "CUDNN_FRONTEND_COMPILED_CACHE"
_ENV_GEMM_SOURCE_CACHE = "CUDNN_FRONTEND_GEMM_KERNEL_CACHE"
_VENDORED_BANNER = "%s is an experimental API"
# The vendored classes log their own experimental banner on a logger named after the class; FlashInfer's ExperimentalWarning
# is the one opt-in signal of this API, so exactly that record is dropped.
_VENDORED_BANNER_LOGGERS = ("GatedAttentionBlockFwd", "SdpaFwdDslSm100")


class _DropVendoredBanner(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.msg != _VENDORED_BANNER


@functools.lru_cache(maxsize=1)
def configure_vendored_tree() -> Dict[str, Any]:
    """Once per process, before the first plan: point the vendored cuDNN Frontend tree's two on-disk caches under FlashInfer's
    cache root and drop its duplicate experimental banner.  Returns the cache facts a plan records in its ``declaration``.

    The compiled-plan cache (``compile_cached``: the exported kernel objects, reloaded in milliseconds at the next start-up) goes
    to ``FLASHINFER_CACHE_DIR/qwen_3_5_block/compiled_plans`` and the rendered GEMM sources to ``.../gemm_sources`` -- unless the
    caller already chose a location through the tree's own variables (``CUDNN_FRONTEND_COMPILED_CACHE``,
    ``CUDNN_FRONTEND_GEMM_KERNEL_CACHE``), which win.  ``CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`` turns the first cache off; the
    second is where the JIT writes the source it imports and cannot be turned off.  Nothing is written here: the directories
    are created by the first compile.
    """
    from flashinfer.jit.env import FLASHINFER_CACHE_DIR

    from ._cudnn_frost.frost import compiled_cache

    root = FLASHINFER_CACHE_DIR / "qwen_3_5_block"
    compiled_plans = os.environ.get(_ENV_COMPILED_CACHE)
    if not compiled_plans:
        compiled_plans = str(root / "compiled_plans")
        compiled_cache.set_cache_dir(compiled_plans)
    gemm_sources = os.environ.get(_ENV_GEMM_SOURCE_CACHE)
    if not gemm_sources:
        # the tree reads the variable at every compile (no setter exists for this cache); set only when the caller left it unset
        gemm_sources = str(root / "gemm_sources")
        os.environ[_ENV_GEMM_SOURCE_CACHE] = gemm_sources
    for name in _VENDORED_BANNER_LOGGERS:
        logging.getLogger(name).addFilter(_DropVendoredBanner())
    return {
        "compiled_plans": compiled_plans,
        "compiled_plans_enabled": bool(compiled_cache.enabled()),
        "gemm_sources": gemm_sources,
    }


def _capturing() -> bool:
    """Whether the current stream is in CUDA-graph capture: one driver query, no synchronization (False without a CUDA runtime)."""
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _check_not_capturing(what: str) -> None:
    if _capturing():
        raise RuntimeError(
            f"qwen_3_5_block: {what} compiles kernels and allocates and cannot run under CUDA-graph capture; "
            "plan outside the capture (one plan per shape bucket), then capture run()"
        )


def _device_of(device) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise NotImplementedError(
            f"qwen_3_5_block: the kernels run on a CUDA device (SM107); got {dev}"
        )
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


# --------------------------------------------------------------------------- operand form checks (host-side, typed, no device read)


def _tensor(x, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ValueError(
            f"qwen_3_5_block: {name} must be a torch.Tensor, got {type(x).__name__}"
        )
    return x


def _on(x: torch.Tensor, device: torch.device, name: str) -> None:
    if x.device != device:
        raise ValueError(
            f"qwen_3_5_block: {name} must live on {device} (the plan's device), got {x.device}"
        )


def _rows(
    x,
    counts: Sequence[int],
    width: int,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int]:
    """A token matrix ``[rows, width]`` (or ``[b, s, width]`` with ``b * s`` rows) as a contiguous 2-D view: ``(view, rows, rank)``."""
    x = _tensor(x, name)
    _on(x, device, name)
    if x.dtype != dtype:
        raise ValueError(
            f"qwen_3_5_block: {name} must be {dtype} (the plan's dtype), got {x.dtype}"
        )
    if x.dim() == 3:
        rows, w = int(x.shape[0]) * int(x.shape[1]), int(x.shape[2])
    elif x.dim() == 2:
        rows, w = int(x.shape[0]), int(x.shape[1])
    else:
        raise ValueError(
            f"qwen_3_5_block: {name} must be [num_tokens, {width}] (or [batch, seq, {width}]), got shape {tuple(x.shape)}"
        )
    if w != width:
        raise ValueError(
            f"qwen_3_5_block: {name} must have {width} columns, got {w} (shape {tuple(x.shape)})"
        )
    if rows not in counts:
        want = " or ".join(str(c) for c in counts)
        raise ValueError(
            f"qwen_3_5_block: {name} must carry {want} token rows for this plan, got {rows} (shape {tuple(x.shape)})"
        )
    if not x.is_contiguous():
        raise ValueError(
            f"qwen_3_5_block: {name} must be a contiguous token matrix (rows of {width}), got strides {tuple(x.stride())}"
        )
    return x.view(rows, w), rows, x.dim()


def _int_vector(
    x, counts: Sequence[int], name: str, device: torch.device, dtypes=(torch.int32,)
) -> torch.Tensor:
    x = _tensor(x, name)
    _on(x, device, name)
    if x.dtype not in dtypes:
        raise ValueError(
            f"qwen_3_5_block: {name} must be {' / '.join(str(d) for d in dtypes)}, got {x.dtype}"
        )
    if x.dim() != 1 or int(x.numel()) not in counts:
        want = " or ".join(str(c) for c in counts)
        raise ValueError(
            f"qwen_3_5_block: {name} must be a 1-D tensor of {want} entries for this plan, got shape {tuple(x.shape)}"
        )
    if not x.is_contiguous():
        raise ValueError(
            f"qwen_3_5_block: {name} must be contiguous, got strides {tuple(x.stride())}"
        )
    return x


def _block_table(
    x,
    batch: int,
    min_cols: int,
    row_stride: Optional[int],
    device: torch.device,
    name: str = "block_table",
) -> torch.Tensor:
    x = _tensor(x, name)
    _on(x, device, name)
    if x.dtype != torch.int32:
        raise ValueError(
            f"qwen_3_5_block: {name} must be int32 (the dense page table of the serving stack), got {x.dtype}"
        )
    if x.dim() != 2 or int(x.shape[0]) != batch or int(x.shape[1]) < min_cols:
        raise ValueError(
            f"qwen_3_5_block: {name} must be [{batch}, >= {min_cols}] (batch rows, at least max_pages columns) for this plan, got shape {tuple(x.shape)}"
        )
    if x.stride(1) != 1:
        raise ValueError(
            f"qwen_3_5_block: {name}'s page axis must be contiguous, got strides {tuple(x.stride())}"
        )
    if row_stride is not None and int(x.stride(0)) != row_stride:
        raise ValueError(
            f"qwen_3_5_block: {name} row stride {int(x.stride(0))} differs from the plan's {row_stride} (the row stride is compiled in; "
            "plan with the table the step will bind, or re-plan)"
        )
    return x


def _slots(x, counts: Sequence[int], device: torch.device) -> torch.Tensor:
    return _int_vector(x, counts, "slot_mapping", device, dtypes=_INT_DTYPES)


def _check_weights(
    weights,
    w_o,
    geometry: Qwen35BlockGeometry,
    dtype: torch.dtype,
    device: torch.device,
) -> Qwen35BlockWeights:
    if not isinstance(weights, Qwen35BlockWeights):
        raise ValueError(
            f"qwen_3_5_block: weights must be the Qwen35BlockWeights of qwen_3_5_block_pack_weights, got {type(weights).__name__}"
        )
    w = _tensor(weights.w_qkvg, "weights.w_qkvg")
    _on(w, device, "weights.w_qkvg")
    if (
        w.dtype != dtype
        or tuple(w.shape) != (geometry.n_qkvg, geometry.d_model)
        or not w.is_contiguous()
    ):
        raise ValueError(
            f"qwen_3_5_block: weights.w_qkvg must be a contiguous {dtype} [{geometry.n_qkvg}, {geometry.d_model}] (Q | GATE | K | V bands), "
            f"got {w.dtype} {tuple(w.shape)}"
        )
    for name, t in (
        ("weights.w_q_norm", weights.w_q_norm),
        ("weights.w_k_norm", weights.w_k_norm),
    ):
        if geometry.qk_norm:
            t = _tensor(t, name)
            _on(t, device, name)
            if (
                t.dtype != dtype
                or tuple(t.shape) != (geometry.head_dim,)
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"qwen_3_5_block: {name} must be a contiguous {dtype} [{geometry.head_dim}] vector, got {t.dtype} {tuple(t.shape)}"
                )
        elif t is not None:
            raise ValueError(
                f"qwen_3_5_block: {name} must be None under geometry.qk_norm=False"
            )
    o = _tensor(w_o, "w_o")
    _on(o, device, "w_o")
    want = (geometry.d_model, geometry.num_qo_heads * geometry.head_dim)
    if o.dtype != dtype or tuple(o.shape) != want or not o.is_contiguous():
        raise ValueError(
            f"qwen_3_5_block: w_o must be a contiguous {dtype} {list(want)} (the o_proj weight), got {o.dtype} {tuple(o.shape)}"
        )
    return weights


# --------------------------------------------------------------------------- the paged KV cache: FlashInfer layouts -> the block's views


class KvGeometry(Protocol):
    """What :func:`kv_views` reads of a block geometry: the KV head count and the head dim.  A structural type, so the sibling
    packages' geometries (the Qwen 3.8 block's among them) are accepted without inheriting from :class:`Qwen35BlockGeometry`."""

    @property
    def num_kv_heads(self) -> int: ...

    @property
    def head_dim(self) -> int: ...


def kv_views(
    kv_cache: KvCache,
    kv_layout: str,
    *,
    geometry: KvGeometry,
    page_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The block's ``[num_pages, num_kv_heads, page_size, head_dim]`` K and V views of a FlashInfer paged cache -- strides only, no copy.

    ``kv_cache`` is FlashInfer's 5-D ``[num_pages, 2, page_size, H_kv, D]`` (``kv_layout="NHD"``) or ``[num_pages, 2, H_kv, page_size, D]``
    (``"HND"``) buffer, or a ``(k_cache, v_cache)`` pair of 4-D tensors in the same layouts.  Typed ``ValueError`` naming the operand on
    any other form; the pool contract the kernels bind (``D`` contiguous, every other stride a 16-byte multiple, a 16-byte-aligned base)
    is checked here and again by the vendored block."""
    layout = str(kv_layout).upper()
    if layout not in ("NHD", "HND"):
        raise ValueError(
            f"qwen_3_5_block: kv_layout must be 'NHD' or 'HND', got {kv_layout!r}"
        )
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) != 2:
            raise ValueError(
                "qwen_3_5_block: kv_cache must be a 5-D [num_pages, 2, ...] buffer or a (k_cache, v_cache) pair"
            )
        k, v = (
            _tensor(kv_cache[0], "kv_cache[0] (K)"),
            _tensor(kv_cache[1], "kv_cache[1] (V)"),
        )
    else:
        kv = _tensor(kv_cache, "kv_cache")
        if kv.dim() != 5 or int(kv.shape[1]) != 2:
            raise ValueError(
                f"qwen_3_5_block: a 5-D kv_cache is [num_pages, 2, page_size, H_kv, D] (NHD) or [num_pages, 2, H_kv, page_size, D] (HND), got {tuple(kv.shape)}"
            )
        k, v = kv[:, 0], kv[:, 1]
    h, p, d = geometry.num_kv_heads, int(page_size), geometry.head_dim
    views = []
    for name, t in (("kv_cache (K)", k), ("kv_cache (V)", v)):
        if t.dim() != 4:
            raise ValueError(
                f"qwen_3_5_block: {name} must be 4-D, got shape {tuple(t.shape)}"
            )
        if t.device.type != "cuda":
            raise ValueError(
                f"qwen_3_5_block: {name} must be a CUDA tensor, got {t.device}"
            )
        if t.dtype != dtype:
            raise ValueError(
                f"qwen_3_5_block: {name} must be {dtype} (the activation dtype: the pools hold bf16 K / V), got {t.dtype}"
            )
        want = (p, h, d) if layout == "NHD" else (h, p, d)
        if tuple(int(s) for s in t.shape[1:]) != want:
            spelled = (
                "[num_pages, page_size, H_kv, D]"
                if layout == "NHD"
                else "[num_pages, H_kv, page_size, D]"
            )
            raise ValueError(
                f"qwen_3_5_block: {name} must be {spelled} = [*, {', '.join(map(str, want))}] under kv_layout={layout}, got {tuple(t.shape)}"
            )
        view = t.permute(0, 2, 1, 3) if layout == "NHD" else t
        if view.stride(3) != 1:
            raise ValueError(
                f"qwen_3_5_block: {name}: the head dim must be contiguous, got strides {tuple(t.stride())}"
            )
        es = view.element_size()
        for ax in range(3):
            if (int(view.stride(ax)) * es) % _ALIGN_BYTES:
                raise ValueError(
                    f"qwen_3_5_block: {name}: every stride but the head dim's must be a 16-byte multiple, got strides {tuple(t.stride())} ({es}-byte elements)"
                )
        if view.data_ptr() % _ALIGN_BYTES:
            raise ValueError(
                f"qwen_3_5_block: {name} must be 16-byte aligned, got data_ptr {view.data_ptr():#x}"
            )
        views.append(view)
    if int(views[0].shape[0]) != int(views[1].shape[0]):
        raise ValueError(
            f"qwen_3_5_block: the K and V pools must hold the same number of pages, got {views[0].shape[0]} and {views[1].shape[0]}"
        )
    return views[0], views[1]


# --------------------------------------------------------------------------- the plan


@dataclass(eq=False)
class Qwen35BlockPlan:
    """One compiled declaration of the block for one shape bucket; see the module docstring.  Built by :func:`prepare`."""

    form: str
    geometry: Qwen35BlockGeometry
    dtype: torch.dtype
    device: torch.device
    kv_layout: str
    page_size: int
    num_pages: int
    batch_size: (
        int  #: prefill: sequences; extend: the sequence capacity; decode: sequences
    )
    q_len: int  #: prefill: the sequence length; extend: the token capacity; decode: query rows per sequence
    max_pages: int  #: paged forms: the table columns the plan addresses (0 on prefill)
    table_row_stride: (
        int  #: paged forms: the block table's row stride, compiled in (0 on prefill)
    )
    max_seq_len: Optional[int]
    decode_split_kv: Optional[int]
    workspace_size: int
    declaration: Dict[str, Any]
    _block: Any = field(repr=False)
    _pool_shape: Tuple[int, ...] = field(repr=False)
    _k_strides: Tuple[int, ...] = field(repr=False)
    _v_strides: Tuple[int, ...] = field(repr=False)
    _workspace: Optional[torch.Tensor] = field(default=None, repr=False)
    #: extend: every launched token-row count (the capacity, a ``num_actual_tokens``) is a multiple of this (1 = any count).  On a
    #: plan of at most ``SMALL_CAPACITY`` tokens the out projection runs the small-capacity GEMM tile, whose token rows are the
    #: kernel's epilogue-vector axis (see :func:`_out_projection_row_granularity`).
    live_rows_multiple: int = 1
    #: set once a CUDA graph is captured over this plan (the graph baked the workspace's address): the functional entry points'
    #: LRU then retains the plan past its eviction instead of freeing it -- see ``cached_plan`` / ``clear_plan_cache``
    _captured: bool = field(default=False, repr=False)

    @property
    def num_tokens(self) -> int:
        """Token rows a run binds: ``batch_size * seq_len`` (prefill), the token capacity (extend), ``batch_size * q_len`` (decode)."""
        return self.q_len if self.form == "extend" else self.batch_size * self.q_len

    def supports(self, **call) -> bool:
        """Whether a call with these keyword tensors (the ``run`` keywords) fits this plan: the host checks of ``run`` without a launch."""
        try:
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
            dtype=self.dtype,
        )
        if (
            tuple(k.shape) != self._pool_shape
            or tuple(k.stride()) != self._k_strides
            or tuple(v.stride()) != self._v_strides
        ):
            raise ValueError(
                f"qwen_3_5_block: kv_cache does not match the plan's pools (plan: {self._pool_shape} K strides {self._k_strides} V strides {self._v_strides}; "
                f"got {tuple(k.shape)} K strides {tuple(k.stride())} V strides {tuple(v.stride())}); another page count or layout is another plan"
            )
        return k, v

    def _workspace_for(self, workspace_buffer: Optional[torch.Tensor]) -> torch.Tensor:
        if workspace_buffer is None:
            if self._workspace is None:
                raise RuntimeError(
                    "qwen_3_5_block: this plan owns no workspace; pass workspace_buffer"
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
                f"qwen_3_5_block: workspace_buffer must be a contiguous uint8 buffer of at least plan.workspace_size = {self.workspace_size} bytes"
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
    ) -> Dict[str, Any]:
        """The host-side form checks of one call, as the arguments the block's ``execute`` takes."""
        g, dev, dt = self.geometry, self.device, self.dtype
        dm, rope = g.d_model, g.rope_dim
        bound: Dict[str, Any] = {}
        if kv_cache is not None:
            bound["k_cache"], bound["v_cache"] = self._pools(kv_cache)
        if self.form == "extend":
            cap = self.q_len
            live = None
            if num_actual_tokens is not None:
                if (
                    isinstance(num_actual_tokens, bool)
                    or not isinstance(num_actual_tokens, int)
                    or not (1 <= num_actual_tokens <= cap)
                ):
                    raise ValueError(
                        f"qwen_3_5_block: num_actual_tokens must be an int in [1, {cap}] (the plan's token capacity), got {num_actual_tokens!r}"
                    )
                live = int(num_actual_tokens)
                if live % self.live_rows_multiple:
                    raise ValueError(
                        f"qwen_3_5_block: num_actual_tokens={live} must be a multiple of {self.live_rows_multiple} on this plan "
                        f"(token_capacity={cap} <= {SMALL_CAPACITY}: the out projection runs the small-capacity GEMM tile "
                        f"{self.declaration.get('out_proj_tile')}, whose token rows are its epilogue-vector axis); hand "
                        f"num_actual_tokens=None with the token tensors at the capacity, round the live count up to a multiple of "
                        f"{self.live_rows_multiple} with pad rows (finite values, slot -1), or plan with token_capacity > {SMALL_CAPACITY}"
                    )
            counts = (cap,) if live is None else (cap, live)
            if hidden_states is not None:
                bound["h"], rows, _ = _rows(
                    hidden_states, counts, dm, "hidden_states", dt, dev
                )
                bound["cos"] = _rows(rope_cos, (rows,), rope, "rope_cos", dt, dev)[0]
                bound["sin"] = _rows(rope_sin, (rows,), rope, "rope_sin", dt, dev)[0]
                bound["out"] = (
                    _rows(out, (rows,), dm, "out", dt, dev)[0]
                    if out is not None
                    else None
                )
                bound["rows"] = rows
            if slot_mapping is not None:
                bound["slot_mapping"] = _slots(slot_mapping, counts, dev)
            if query_start_loc is not None:
                bound["seq_lens"] = _int_vector(
                    query_start_loc, (self.batch_size + 1,), "query_start_loc", dev
                )
            if seq_lens is not None:
                bound["kv_lens"] = _int_vector(
                    seq_lens, (self.batch_size,), "seq_lens", dev
                )
            if block_table is not None:
                bound["block_table"] = _block_table(
                    block_table,
                    self.batch_size,
                    self.max_pages,
                    self.table_row_stride,
                    dev,
                )
            bound["num_tokens_live"] = live
            return bound
        if query_start_loc is not None or num_actual_tokens is not None:
            raise ValueError(
                f"qwen_3_5_block: the {self.form} form takes no query_start_loc / num_actual_tokens (a uniform [batch, {self.q_len}] step)"
            )
        b, s = self.batch_size, self.q_len
        t = b * s
        if hidden_states is not None:
            h2, _, rank = _rows(hidden_states, (t,), dm, "hidden_states", dt, dev)
            bound["h"] = h2.view(b, s, dm)
            bound["cos"] = _rows(rope_cos, (t,), rope, "rope_cos", dt, dev)[0].view(
                b, s, rope
            )
            bound["sin"] = _rows(rope_sin, (t,), rope, "rope_sin", dt, dev)[0].view(
                b, s, rope
            )
            bound["out"] = (
                _rows(out, (t,), dm, "out", dt, dev)[0].view(b, s, dm)
                if out is not None
                else None
            )
            bound["rows"], bound["rank"] = t, rank
        if slot_mapping is not None:
            bound["slot_mapping"] = _slots(slot_mapping, (t,), dev)
        if self.form == "prefill":
            if block_table is not None or seq_lens is not None:
                raise ValueError(
                    "qwen_3_5_block: the prefill form attends over the step's own tokens and takes no block_table / seq_lens (use the extend form over a cached prefix)"
                )
            return bound
        if seq_lens is not None:
            bound["kv_lens"] = _int_vector(seq_lens, (b,), "seq_lens", dev)
        if block_table is not None:
            bound["block_table"] = _block_table(
                block_table, b, self.max_pages, self.table_row_stride, dev
            )
        return bound

    def run(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        weights: Qwen35BlockWeights,
        w_o: torch.Tensor,
        *,
        kv_cache: KvCache,
        slot_mapping: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        num_actual_tokens: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Launch the plan on the current CUDA stream (capturable; see the README's capture rules).  Returns ``out``."""
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
            query_start_loc=query_start_loc,
            num_actual_tokens=num_actual_tokens,
            out=out,
        )
        ws = self._workspace_for(workspace_buffer)
        common = (
            weights.w_qkvg,
            weights.w_q_norm,
            weights.w_k_norm,
            b["cos"],
            b["sin"],
            w_o,
        )
        if self.form == "extend":
            for name in ("seq_lens", "kv_lens", "block_table"):
                if name not in b:
                    spelled = {
                        "seq_lens": "query_start_loc",
                        "kv_lens": "seq_lens",
                        "block_table": "block_table",
                    }[name]
                    raise ValueError(
                        f"qwen_3_5_block: the extend form requires {spelled}"
                    )
            o = (
                b["out"]
                if b["out"] is not None
                else torch.empty(b["rows"], g.d_model, dtype=dt, device=dev)
            )
            self._block.execute(
                b["h"],
                *common,
                o,
                ws,
                seq_lens=b["seq_lens"],
                k_cache=b["k_cache"],
                v_cache=b["v_cache"],
                block_table=b["block_table"],
                kv_lens=b["kv_lens"],
                slot_mapping=b["slot_mapping"],
                num_tokens_live=b["num_tokens_live"],
            )
            return o
        o = (
            b["out"]
            if b["out"] is not None
            else torch.empty(
                self.batch_size, self.q_len, g.d_model, dtype=dt, device=dev
            )
        )
        if self.form == "prefill":
            self._block.execute(
                b["h"],
                *common,
                o,
                ws,
                k_cache=b["k_cache"],
                v_cache=b["v_cache"],
                slot_mapping=b["slot_mapping"],
            )
        else:
            for name, spelled in (
                ("kv_lens", "seq_lens"),
                ("block_table", "block_table"),
            ):
                if name not in b:
                    raise ValueError(
                        f"qwen_3_5_block: the decode form requires {spelled}"
                    )
            self._block.execute(
                b["h"],
                *common,
                o,
                ws,
                k_cache=b["k_cache"],
                v_cache=b["v_cache"],
                block_table=b["block_table"],
                kv_lens=b["kv_lens"],
                slot_mapping=b["slot_mapping"],
            )
        if out is not None:
            return out
        return o if b["rank"] == 3 else o.view(b["rows"], g.d_model)


def prepare(
    form: str,
    *,
    geometry: Qwen35BlockGeometry,
    device,
    weights: Qwen35BlockWeights,
    w_o: torch.Tensor,
    kv_cache: KvCache,
    page_size: int,
    batch_size: int,
    dtype: torch.dtype = torch.bfloat16,
    kv_layout: str = "NHD",
    seq_len: Optional[int] = None,
    token_capacity: Optional[int] = None,
    q_len_per_req: int = 1,
    max_pages: Optional[int] = None,
    block_table: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
    max_query_len: Optional[int] = None,
    decode_split_kv: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    _declaration_overrides: Optional[Dict[str, Any]] = None,
) -> Qwen35BlockPlan:
    """Declare, check and compile the block for one shape bucket; see :func:`flashinfer.qwen_3_5_block.prepare_qwen_3_5_block_bf16`.

    ``_declaration_overrides`` is a test hook, not part of the API: extra keywords for the vendored declaration (for instance
    ``{"fuse_cache_write": False}`` to run the write-through as its own launch); the public entry points never pass it."""
    if form not in FORMS:
        raise ValueError(f"qwen_3_5_block: form must be one of {FORMS}, got {form!r}")
    if not isinstance(geometry, Qwen35BlockGeometry):
        raise TypeError(
            f"qwen_3_5_block: geometry must be a Qwen35BlockGeometry, got {type(geometry).__name__}"
        )
    _support.require_served(device, geometry, dtype, form)
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
            f"qwen_3_5_block: page_size must be a positive multiple of {PAGE_SIZE_ALIGN} (validated: {PAGE_SIZES}), got {page_size!r}"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(
            f"qwen_3_5_block: batch_size must be a positive int, got {batch_size!r}"
        )
    _check_weights(weights, w_o, geometry, dtype, dev)
    k, v = kv_views(
        kv_cache, kv_layout, geometry=geometry, page_size=page_size, dtype=dtype
    )
    if k.device != dev:
        raise ValueError(
            f"qwen_3_5_block: kv_cache lives on {k.device}, the plan on {dev}"
        )
    g = geometry.frost()
    dm, rope = geometry.d_model, geometry.rope_dim
    Fwd = _frost_block()
    # The geometry gate require_served ran mirrors the vendored tree's rules; with the tree imported, re-read them -- a re-vendor that
    # moved a rule fails here, loudly, like the capture-safe hook below, never as a plan the gate and the tree read differently.
    _support.check_vendored_rules()

    def mk(*shape):
        return torch.empty(*shape, dtype=dtype, device=dev)

    knobs: Dict[str, Any] = dict(fuse_norm_rope=True, fuse_gate=True)
    if _declaration_overrides:
        knobs.update(_declaration_overrides)
    table_sample = None
    pages = 0
    row_stride = 0
    if form != "prefill":
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
                    f"qwen_3_5_block: block_table must be an int32 [>= {batch_size}, max_pages] table with a contiguous page axis, got {bt.dtype} {tuple(bt.shape)}"
                )
            pages = int(bt.shape[1]) if max_pages is None else int(max_pages)
            if pages <= 0 or pages > int(bt.shape[1]):
                raise ValueError(
                    f"qwen_3_5_block: max_pages must be in [1, {int(bt.shape[1])}] (the table's columns), got {max_pages!r}"
                )
            table_sample = bt[:batch_size, :pages]
        else:
            if max_pages is None:
                raise ValueError(
                    "qwen_3_5_block: pass max_pages (the table columns a step may address) or the block_table the steps will bind"
                )
            pages = int(max_pages)
            if pages <= 0:
                raise ValueError(
                    f"qwen_3_5_block: max_pages must be positive, got {max_pages!r}"
                )
            table_sample = torch.zeros(batch_size, pages, dtype=torch.int32, device=dev)
        row_stride = int(table_sample.stride(0))
        if max_seq_len is not None and (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, int)
            or not (1 <= max_seq_len <= pages * page_size)
        ):
            raise ValueError(
                f"qwen_3_5_block: max_seq_len must be an int in [1, max_pages * page_size = {pages * page_size}], got {max_seq_len!r}"
            )
    if form == "prefill":
        if (
            seq_len is None
            or isinstance(seq_len, bool)
            or not isinstance(seq_len, int)
            or seq_len <= 0
        ):
            raise ValueError(
                f"qwen_3_5_block: the prefill form needs seq_len (the uniform sequence length of the step), got {seq_len!r}"
            )
        if (
            block_table is not None
            or max_pages is not None
            or max_seq_len is not None
            or decode_split_kv is not None
            or token_capacity is not None
        ):
            raise ValueError(
                "qwen_3_5_block: the prefill form takes seq_len only (no block_table / max_pages / max_seq_len / decode_split_kv / token_capacity)"
            )
        rows = seq_len
        blk = Fwd(
            mk(batch_size, rows, dm),
            weights.w_qkvg,
            weights.w_q_norm,
            weights.w_k_norm,
            mk(batch_size, rows, rope),
            mk(batch_size, rows, rope),
            w_o,
            mk(batch_size, rows, dm),
            g,
            paged_kv_page_size=page_size,
            **knobs,
        )
    elif form == "extend":
        if (
            token_capacity is None
            or isinstance(token_capacity, bool)
            or not isinstance(token_capacity, int)
            or token_capacity < 2
        ):
            raise ValueError(
                f"qwen_3_5_block: the extend form needs token_capacity (the step's token capacity T_max >= 2), got {token_capacity!r}"
            )
        if decode_split_kv is not None or seq_len is not None or q_len_per_req != 1:
            raise ValueError(
                "qwen_3_5_block: the extend form takes token_capacity (and max_query_len), not seq_len / q_len_per_req / decode_split_kv"
            )
        rows = token_capacity
        mql = rows if max_query_len is None else int(max_query_len)
        if not (2 <= mql <= rows) or batch_size * mql < rows:
            raise ValueError(
                f"qwen_3_5_block: max_query_len must satisfy 2 <= max_query_len <= token_capacity and batch_size * max_query_len >= token_capacity, got {mql}"
            )
        blk = Fwd(
            mk(rows, dm),
            weights.w_qkvg,
            weights.w_q_norm,
            weights.w_k_norm,
            mk(rows, rope),
            mk(rows, rope),
            w_o,
            mk(rows, dm),
            g,
            thd=True,
            num_sequences=batch_size,
            max_seq_len=mql,
            cu_seqlens=True,
            token_capacity=rows,
            paged_kv_page_size=page_size,
            kv_source="paged",
            sample_k_cache=k,
            sample_v_cache=v,
            sample_block_table=table_sample,
            **knobs,
        )
    else:
        q = q_len_per_req
        if isinstance(q, bool) or not isinstance(q, int) or q <= 0:
            raise ValueError(
                f"qwen_3_5_block: q_len_per_req must be a positive int, got {q!r}"
            )
        if q > DECODE_MAX_ROWS:
            raise NotImplementedError(
                f"qwen_3_5_block (decode): q_len_per_req={q} is past the decode envelope ({DECODE_MAX_ROWS} rows per sequence on the Rubin d256 decode tile; "
                f"{_support.DECODE_MAX_Q_LEN} claimed); a longer step is the extend form's business"
            )
        if seq_len is not None or token_capacity is not None:
            raise ValueError(
                "qwen_3_5_block: the decode form takes q_len_per_req, not seq_len / token_capacity"
            )
        if decode_split_kv is not None and (
            isinstance(decode_split_kv, bool)
            or not isinstance(decode_split_kv, int)
            or decode_split_kv < 1
        ):
            raise ValueError(
                f"qwen_3_5_block: decode_split_kv must be None or an int >= 1, got {decode_split_kv!r}"
            )
        rows = q
        blk = Fwd(
            mk(batch_size, q, dm),
            weights.w_qkvg,
            weights.w_q_norm,
            weights.w_k_norm,
            mk(batch_size, q, rope),
            mk(batch_size, q, rope),
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
            **knobs,
        )
    # The projection GEMMs on the capture-safe JIT route by construction (the paged forms set it themselves; the dense prefill
    # form does not yet -- a cuDNN Frontend follow-up -- and the vendored tree carries no cuDNN backend to query).  The hook
    # reaches into the vendored block; a re-vendor that renames these attributes must fail here, loudly, not drop the route.
    missing = [a for a in ("_proj", "_out_proj", "_stages") if not hasattr(blk, a)]
    if missing:
        raise RuntimeError(
            f"qwen_3_5_block: the vendored GatedAttentionBlockFwd no longer carries {missing}; the adapter's capture-safe hook "
            "(api.prepare) must be updated with the re-vendored tree"
        )
    for st in (blk._proj, blk._out_proj):
        if st is not None:
            st.capture_safe = True
    blk.check_support()
    blk.compile()
    size = int(blk.get_workspace_size())
    out_proj_tile, live_multiple = None, 1
    if form == "extend":
        out_proj_tile, live_multiple = _out_projection_row_granularity(blk, dtype)
        # By construction the template picks the widest vector that divides the declared capacity, so this never fires; it is the
        # tripwire for a re-vendor that changes the rule.
        if rows % live_multiple:
            raise ValueError(
                f"qwen_3_5_block: token_capacity={rows} must be a multiple of {live_multiple} on a plan of at most {SMALL_CAPACITY} tokens "
                f"(the out projection runs the small-capacity GEMM tile {out_proj_tile}, whose token rows are its epilogue-vector axis); "
                f"pick a multiple of {live_multiple} or a token_capacity above {SMALL_CAPACITY}"
            )
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
                f"qwen_3_5_block: workspace_buffer must be a contiguous uint8 buffer of at least plan.workspace_size = {size} bytes (got {int(ws.numel()) if isinstance(ws, torch.Tensor) else 'n/a'})"
            )
        if form == "extend":
            ws[
                :size
            ].zero_()  # the bounded packed form's pad-row contract: an unwritten O row reads 0 (zeroed once, outside capture)
        owned = None
    else:
        owned = torch.zeros(size, dtype=torch.uint8, device=dev)
    route = getattr(blk, "decode_route", None)
    declaration = dict(
        form=form,
        page_size=page_size,
        kv_layout=str(kv_layout).upper(),
        num_pages=int(k.shape[0]),
        batch_size=batch_size,
        rows=rows,
        max_pages=pages,
        table_row_stride=row_stride,
        max_seq_len=max_seq_len,
        stages=[type(s).__name__ for s in blk._stages],
        gate_fused=bool(getattr(blk, "gate_fused", False)),
        cache_write_fused=bool(getattr(blk, "cache_write_fused", False)),
        # what runs for stage (1): the fused norm + RoPE projection fork, or -- at decode row counts (B x q_len <= 128), the
        # vendored declaration's own lowering -- the small-M slab GEMM and the norm + RoPE pass, with the paged write-through
        # folded into that launch below 128 rows (`cache_write_in_norm_rope`) and the separate cache write past them
        fuse_norm_rope_effective=bool(
            getattr(
                blk, "fuse_norm_rope_effective", getattr(blk, "fuse_norm_rope", False)
            )
        ),
        # the vendored defaults of the decode step, read back (the block passes neither knob; both are performance-only: the
        # same bytes land in `out` and in the pools either way) -- where the lowered chain's write-through runs, and programmatic
        # dependent launch: `use_pdl` the resolved switch (`None` resolves True on the decode tile's routes on cc 10.7, False
        # everywhere else), the declaration as given, the stages launching with the attribute, whether every switched stage does
        cache_write_in_norm_rope=bool(getattr(blk, "cache_write_in_norm_rope", False)),
        use_pdl=bool(getattr(blk, "use_pdl", False)),
        use_pdl_requested=getattr(blk, "use_pdl_requested", None),
        use_pdl_effective=bool(getattr(blk, "use_pdl_effective", False)),
        pdl_stages=list(getattr(blk, "pdl_stages", ())),
        decode_split_kv=getattr(blk, "decode_split_kv", None),
        decode_route=None
        if route is None
        else {
            k_: getattr(route, k_, None)
            for k_ in (
                "s_q",
                "group",
                "tile",
                "n_q",
                "token_units",
                "pack_gqa",
                "claimed",
                # appended with the Rubin decode levers: KV streams per (sequence, KV head), the route policy, the width cap
                "kv_streams",
                "policy",
                "q_tile_max",
            )
        },
        out_proj_tile=out_proj_tile,
        live_rows_multiple=live_multiple,
        caches=caches,
        workspace_size=size,
    )
    return Qwen35BlockPlan(
        form=form,
        geometry=geometry,
        dtype=dtype,
        device=dev,
        kv_layout=str(kv_layout).upper(),
        page_size=page_size,
        num_pages=int(k.shape[0]),
        batch_size=batch_size,
        q_len=rows,
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
        live_rows_multiple=live_multiple,
    )


# The token capacity up to which the vendored out projection takes the small-capacity (swap-AB) GEMM route; the vendored
# ``kernels/proj_gemm.py`` ``SMALL_M_MAX``.  Read from the tree when available so the two cannot drift.
def _small_capacity() -> int:
    try:
        from ._cudnn_frost.gated_attention_block.kernels import proj_gemm

        return int(proj_gemm.SMALL_M_MAX)
    except Exception:  # noqa: BLE001 - the number is informational in error messages
        return 128


SMALL_CAPACITY = _small_capacity()


def _out_projection_row_granularity(
    blk, dtype: torch.dtype
) -> Tuple[Optional[str], int]:
    """``(the out-projection tile that runs, the multiple every launched token-row count must satisfy)`` on the bounded form.

    Under a swap-AB tile (the small-capacity route, ``token_capacity <= SMALL_CAPACITY``) the token rows are the GEMM's N, which the
    kernel template declares symbolic with the epilogue vector's divisibility (``vec_bytes_epi // itemsize`` elements); the fused QKV
    projection and the forced 256-wide out-projection tile take the rows as M, which carries no such rule.  Read from the compiled
    plan, never recomputed: the tile depends on the shape and on the device's SM count.
    """
    plan = getattr(getattr(blk, "_out_proj", None), "_plan", None)
    jit = getattr(plan, "jit", None)
    name = getattr(plan, "tile_config_name", None)
    config = getattr(jit, "config", None)
    if jit is None or config is None or not getattr(config, "swap_ab", False):
        return name, 1
    itemsize = torch.empty((), dtype=dtype).element_size()
    try:
        from ._cudnn_frost.gemm.frost.sm100 import compiler as gemm_compiler

        vec_bytes = int(gemm_compiler._epi_vec_bytes(jit.chain, config))
    except Exception:  # noqa: BLE001 - a re-vendored compiler without the helper: the catalog's widest half-precision vector
        vec_bytes = 64
    return name, max(1, vec_bytes // itemsize)


# --------------------------------------------------------------------------- the plan LRU behind the functional entry points
#
# A functional entry point takes its plan from ``_PLANS``, an LRU of ``PLAN_CACHE_SIZE`` shape buckets.  A plan owns its workspace,
# so evicting one frees device memory -- unless a CUDA graph was captured over it: the graph baked the workspace's address and the
# block's buffers, and a replay after the free would write into memory the allocator has since handed to something else, silently.
# Such a plan is marked ``_captured`` (under the lock, at the lookup the capture goes through) and moves to ``_RETAINED`` at its
# eviction instead, where it lives until ``clear_plan_cache()``; a retained bucket asked for again is revived into the LRU (the same
# plan and workspace: no second compile, no second workspace per bucket).  Invariant: a captured plan is in ``_PLANS`` or
# ``_RETAINED`` until ``clear_plan_cache()``.  A bucket WITHOUT a cached plan cannot be built under capture (``prepare`` compiles
# and allocates), so the lookup refuses it there, typed, instead of letting ``prepare`` fail deeper.

_PLANS: "collections.OrderedDict[tuple, Qwen35BlockPlan]" = collections.OrderedDict()
_RETAINED: "list[tuple[tuple, Qwen35BlockPlan]]" = []
_PLANS_LOCK = threading.Lock()
PLAN_CACHE_SIZE = 32


def clear_plan_cache() -> None:
    """Drop every plan of the functional entry points -- the LRU's and the retained ones captured graphs ran -- and with them
    their workspaces.  The ONLY release of a retained plan: call it once no CUDA graph captured over a functional-entry plan can
    replay any more (after ``graph.reset()``, at model teardown), never while one may; under capture it is refused."""
    if _capturing():
        raise RuntimeError(
            "qwen_3_5_block: clear_plan_cache() releases the workspaces captured graphs replay into and cannot run under CUDA-graph capture"
        )
    with _PLANS_LOCK:
        _PLANS.clear()
        _RETAINED.clear()


def _insert_plan(key: tuple, plan: Qwen35BlockPlan) -> Qwen35BlockPlan:
    """Under ``_PLANS_LOCK``: ``plan`` into the LRU at its fresh end, then evict past the bound -- a captured plan into ``_RETAINED``.
    Two threads that filled one bucket at once: the first writer's plan stays (a graph may already run it), the other is kept
    only when captured; the plan the bucket holds is returned."""
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


def _revive_plan(key: tuple) -> Optional[Qwen35BlockPlan]:
    """Under ``_PLANS_LOCK``: the retained plan of ``key`` back into the LRU (and out of ``_RETAINED``), or None."""
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
) -> Qwen35BlockPlan:
    """The plan of the functional entry points: an LRU keyed on exactly what a plan compiles in.

    Under CUDA-graph capture only a bucket already cached is served (its plan is then retained past eviction, see above); a cold
    bucket raises a typed ``RuntimeError`` naming the remedy -- one eager call outside the capture warms it."""
    dev = _device_of(device)
    k, v = kv_views(
        kv_cache, kv_layout, geometry=geometry, page_size=page_size, dtype=dtype
    )
    bt = kw.get("block_table")
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
        kw.get("seq_len"),
        kw.get("token_capacity"),
        kw.get("q_len_per_req", 1),
        kw.get("max_pages"),
        None if bt is None else (int(bt.shape[1]), int(bt.stride(0))),
        kw.get("max_seq_len"),
        kw.get("max_query_len"),
        kw.get("decode_split_kv"),
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
                plan._captured = True  # marked under the lock: an eviction racing this capture retains the plan
            return plan
    if capturing:
        raise RuntimeError(
            "qwen_3_5_block: this shape bucket has no cached plan, and CUDA-graph capture cannot build one (a plan compiles kernels "
            "and allocates its workspace); warm the bucket outside the capture -- one eager call of the same entry point -- or run "
            "an explicit plan (prepare_qwen_3_5_block_bf16 / Qwen35BlockWrapper), then capture"
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


def _rows_of(hidden_states, name="hidden_states") -> Tuple[int, int, int]:
    """``(rows, batch, seq)`` of a token matrix (``batch = 1, seq = rows`` for a 2-D one)."""
    x = _tensor(hidden_states, name)
    if x.dim() == 3:
        return int(x.shape[0]) * int(x.shape[1]), int(x.shape[0]), int(x.shape[1])
    if x.dim() == 2:
        return int(x.shape[0]), 1, int(x.shape[0])
    raise ValueError(
        f"qwen_3_5_block: {name} must be [num_tokens, d_model] or [batch, seq, d_model], got shape {tuple(x.shape)}"
    )


def prefill(
    hidden_states,
    rope_cos,
    rope_sin,
    weights,
    w_o,
    *,
    geometry,
    kv_cache,
    slot_mapping,
    kv_layout="NHD",
    out=None,
    workspace_buffer=None,
    plan=None,
):
    if plan is None:
        _, b, s = _rows_of(hidden_states)
        page_size = _page_size_of(kv_cache, kv_layout, geometry)
        plan = cached_plan(
            "prefill",
            geometry=geometry,
            device=_tensor(hidden_states, "hidden_states").device,
            weights=weights,
            w_o=w_o,
            kv_cache=kv_cache,
            kv_layout=kv_layout,
            page_size=page_size,
            batch_size=b,
            seq_len=s,
        )
    return plan.run(
        hidden_states,
        rope_cos,
        rope_sin,
        weights,
        w_o,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        out=out,
        workspace_buffer=workspace_buffer,
    )


def extend(
    hidden_states,
    rope_cos,
    rope_sin,
    weights,
    w_o,
    *,
    geometry,
    query_start_loc,
    kv_cache,
    slot_mapping,
    block_table,
    seq_lens,
    token_capacity=None,
    max_query_len=None,
    num_actual_tokens=None,
    kv_layout="NHD",
    out=None,
    workspace_buffer=None,
    plan=None,
):
    if plan is None:
        rows, _, _ = _rows_of(hidden_states)
        cap = rows if token_capacity is None else int(token_capacity)
        b = int(_tensor(seq_lens, "seq_lens").numel())
        page_size = _page_size_of(kv_cache, kv_layout, geometry)
        plan = cached_plan(
            "extend",
            geometry=geometry,
            device=_tensor(hidden_states, "hidden_states").device,
            weights=weights,
            w_o=w_o,
            kv_cache=kv_cache,
            kv_layout=kv_layout,
            page_size=page_size,
            batch_size=b,
            token_capacity=cap,
            max_query_len=max_query_len,
            block_table=block_table,
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
        query_start_loc=query_start_loc,
        num_actual_tokens=num_actual_tokens,
        out=out,
        workspace_buffer=workspace_buffer,
    )


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
    q_len_per_req=1,
    max_seq_len=None,
    decode_split_kv=None,
    kv_layout="NHD",
    out=None,
    workspace_buffer=None,
    plan=None,
):
    if plan is None:
        rows, _, _ = _rows_of(hidden_states)
        q = int(q_len_per_req)
        if q <= 0 or rows % q:
            raise ValueError(
                f"qwen_3_5_block: hidden_states carries {rows} rows, not a multiple of q_len_per_req={q_len_per_req}"
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
    )


def _page_size_of(
    kv_cache: KvCache, kv_layout: str, geometry: Qwen35BlockGeometry
) -> int:
    """The page size a FlashInfer cache buffer carries in its shape (the functional entry points read it there)."""
    layout = str(kv_layout).upper()
    t = (
        kv_cache[0]
        if isinstance(kv_cache, (tuple, list))
        else _tensor(kv_cache, "kv_cache")
    )
    if isinstance(kv_cache, (tuple, list)):
        t = _tensor(t, "kv_cache[0] (K)")
        if t.dim() != 4:
            raise ValueError(
                f"qwen_3_5_block: kv_cache[0] (K) must be 4-D, got shape {tuple(t.shape)}"
            )
        return int(t.shape[1]) if layout == "NHD" else int(t.shape[2])
    if t.dim() != 5:
        raise ValueError(
            f"qwen_3_5_block: a 5-D kv_cache is [num_pages, 2, ...], got shape {tuple(t.shape)}"
        )
    return int(t.shape[2]) if layout == "NHD" else int(t.shape[3])
