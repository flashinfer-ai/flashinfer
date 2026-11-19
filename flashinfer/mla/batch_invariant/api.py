# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry of the sparse MLA classes: what ``trtllm_batch_decode_with_kv_cache_mla`` runs for
``sparse_mla_top_k > 0`` under ``batch_invariant=True``.

One row per query token. A row reads its own ``sparse_indices`` row (physical cache rows, compacted and ascending,
``-1`` after the live entries) and its own query rows; nothing else enters its output. Every unsupported argument
raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back. A call is two launches
at most and reads nothing on the host; it can be captured in a CUDA graph, and calls on several streams do not share state.
"""

from __future__ import annotations

import functools
import math
from typing import List, Optional, Sequence

import torch

from ...batch_invariant.support import compute_capability, unsupported
from . import spec
from .config import SparseMlaConfig, default_config

MLA_ARCHS = ((10, 0), (10, 3), (10, 7))
_WS_ROW = spec.D_LATENT + 4
_E4M3 = getattr(torch, "float8_e4m3fn", None)
_LOG2E = 1.0 / math.log(2.0)

__all__ = [
    "MLA_ARCHS",
    "check_device",
    "class_of",
    "ex2_device",
    "run_decode",
    "sparse_mla",
]


@functools.cache
def _module():
    from ...jit.mla_batch_invariant_sparse import gen_mla_batch_invariant_sparse_module

    return gen_mla_batch_invariant_sparse_module().build_and_load()


@functools.cache
def _sm_count(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def _fmt(t: torch.Tensor) -> Optional[str]:
    if t.dtype == torch.bfloat16:
        return "bf16"
    if t.dtype == torch.uint8 or (_E4M3 is not None and t.dtype == _E4M3):
        return "e4m3"
    return None


def class_of(query: torch.Tensor) -> str:
    """The class a query of this dtype and width runs (``mla.sparse.<fmt>.d512[r64]``)."""
    fmt = _fmt(query) or "bf16"
    rope = int(query.shape[-1]) - spec.D_LATENT
    return spec.class_name(fmt, rope if rope in spec.ROPE_DIMS else 64)


def check_device(device: torch.device, name: str) -> None:
    if device.type != "cuda" or compute_capability(device) not in MLA_ARCHS:
        raise unsupported(name, f"the device {device}", "SM100 / SM103 / SM107")


def sparse_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_lens: Optional[torch.Tensor] = None,
    *,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    config: Optional[SparseMlaConfig] = None,
) -> torch.Tensor:
    """``query`` ``[rows, heads, 576 | 512]`` and ``kv_cache`` ``[..., 576 | 512]`` of one dtype (bf16, or e4m3 as
    ``float8_e4m3fn`` / ``uint8``), ``sparse_indices`` ``[rows, width]`` int32, ``sparse_lens`` ``[rows]`` int32 live
    counts (``None``: an entry is live when it is not negative). Returns ``[rows, heads, 512]`` bf16. ``config``
    overrides the dispatcher with another member of the space (same bits)."""
    name = class_of(query)
    check_device(query.device, name)
    fmt = _fmt(query)
    if fmt is None or _fmt(kv_cache) != fmt:
        raise unsupported(
            name,
            f"query {query.dtype} with cache {kv_cache.dtype}",
            "bf16 or e4m3, the same for both",
        )
    if query.ndim != 3 or int(query.shape[-1]) not in (spec.d_qk(0), spec.d_qk(64)):
        raise unsupported(
            name, f"a query of shape {tuple(query.shape)}", "[rows, heads, 576 | 512]"
        )
    rows, heads, D = (int(x) for x in query.shape)
    if not 1 <= heads <= spec.MAX_HEADS:
        raise unsupported(name, f"{heads} heads", f"1 to {spec.MAX_HEADS} heads")
    if int(kv_cache.shape[-1]) != D:
        raise unsupported(
            name, f"cache rows of {kv_cache.shape[-1]} values", f"{D}, the query width"
        )
    if sparse_indices.ndim != 2 or int(sparse_indices.shape[0]) != rows:
        raise unsupported(
            name,
            f"an index list of shape {tuple(sparse_indices.shape)}",
            "[rows, width]",
        )
    width = int(sparse_indices.shape[1])
    if width < 1:
        raise unsupported(name, "an empty index list", "width >= 1")
    tensors = [("query", query), ("kv_cache", kv_cache), ("index list", sparse_indices)]
    if sparse_indices.dtype != torch.int32:
        raise unsupported(name, f"indices of {sparse_indices.dtype}", "int32")
    if sparse_lens is not None:
        if sparse_lens.dtype != torch.int32 or sparse_lens.numel() != rows:
            raise unsupported(name, "live counts", "one int32 per row")
        sparse_lens = sparse_lens.reshape(rows)
        tensors.append(("live counts", sparse_lens))
    if out is None:
        out = torch.empty(
            rows, heads, spec.D_LATENT, dtype=torch.bfloat16, device=query.device
        )
    elif out.dtype != torch.bfloat16 or tuple(out.shape) != (
        rows,
        heads,
        spec.D_LATENT,
    ):
        raise unsupported(
            name, f"out {tuple(out.shape)} {out.dtype}", "[rows, heads, 512] bf16"
        )
    tensors.append(("out", out))
    for label, t in tensors[:-1]:
        lo, hi = t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()
        if lo < out.data_ptr() + out.numel() * 2 and out.data_ptr() < hi:
            raise unsupported(name, f"out overlapping the {label}", "separate storage")
    for label, t in tensors:
        if t.device != query.device:
            raise unsupported(name, f"{label} on {t.device}", "one device")
        if not t.is_contiguous():
            raise unsupported(name, f"a non-contiguous {label}", "contiguous tensors")
        if label in ("query", "kv_cache", "out") and t.data_ptr() % 16:
            raise unsupported(name, f"a {label} off a 16-byte boundary", "aligned rows")
    c1, c2 = float(bmm1_scale), float(bmm2_scale)
    if not (
        0.0 < c1 < math.inf
        and abs(c1) * _LOG2E <= spec.FLT_MAX
        and spec.scale_log2(c1) > 0.0
        and math.isfinite(c2)
    ):
        raise unsupported(
            name,
            f"bmm1_scale {c1}, bmm2_scale {c2}",
            "a positive finite bmm1_scale, a finite bmm2_scale",
        )
    if rows == 0:
        return out
    if config is None:
        index = query.device.index
        config = default_config(
            rows,
            heads,
            width,
            _sm_count(torch.cuda.current_device() if index is None else index),
        )
    elif not config.well_formed():
        raise unsupported(name, f"the configuration {config}", "a member of the space")
    tiles = spec.n_tiles(width)
    per = 1 if config.fold == "inkernel" else tiles
    # per call: the kernels read the partial rows back, so a buffer shared between calls would couple them
    ws = torch.empty(
        rows * heads * per * _WS_ROW, dtype=torch.float32, device=query.device
    )
    q = query.view(torch.uint8) if fmt == "e4m3" else query
    kv = kv_cache.view(torch.uint8) if fmt == "e4m3" else kv_cache
    _module().mla_batch_invariant_sparse(
        q,
        kv.reshape(-1, D),
        sparse_indices,
        sparse_lens if sparse_lens is not None else sparse_indices.new_empty(0),
        ws,
        out,
        spec.scale_log2(c1),
        spec.scale_out(c2),
        config.head_block,
        config.gather_warps,
        config.tiles_per_unit,
        config.pv_pass,
        config.mma_split,
    )
    return out


def run_decode(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    sparse_mla_top_k: int,
    out: Optional[torch.Tensor],
    bmm1_scale,
    bmm2_scale,
    backend: str,
    cum_seq_lens_q: Optional[torch.Tensor],
    sparse_mla_top_k_lens: Optional[torch.Tensor],
    refused: dict,
) -> torch.Tensor:
    """``trtllm_batch_decode_with_kv_cache_mla(..., batch_invariant=True)``: the argument translation and the
    refusals. ``refused`` maps the name of every argument the class does not take to whether the caller set it."""
    name = spec.class_name(
        _fmt(query) or "bf16",
        qk_rope_head_dim if qk_rope_head_dim in spec.ROPE_DIMS else 64,
    )
    if backend != "auto":
        raise unsupported(name, f"backend={backend!r}", 'backend="auto"')
    if sparse_mla_top_k <= 0:
        raise unsupported(
            name, "dense MLA (sparse_mla_top_k == 0)", "sparse_mla_top_k > 0"
        )
    check_device(query.device, name)
    for arg, was_set in refused.items():
        if was_set:
            raise unsupported(name, arg, "the argument left at its default")
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
        raise unsupported(name, "tensor scales", "float bmm1_scale and bmm2_scale")
    if kv_lora_rank != spec.D_LATENT or qk_rope_head_dim not in spec.ROPE_DIMS:
        raise unsupported(
            name,
            f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}",
            "kv_lora_rank 512 with qk_rope_head_dim 64 or 0",
        )
    D = spec.d_qk(qk_rope_head_dim)
    if query.ndim not in (3, 4) or int(query.shape[-1]) != D:
        raise unsupported(
            name,
            f"a query of shape {tuple(query.shape)}",
            f"[batch, q_len, heads, {D}] or [total_q, heads, {D}] with cum_seq_lens_q",
        )
    if (query.ndim == 3) != (cum_seq_lens_q is not None):
        raise unsupported(
            name,
            "a query rank that does not match cum_seq_lens_q",
            "a 4-d query, or a 3-d query with cum_seq_lens_q",
        )
    rows = int(query.shape[0]) * (int(query.shape[1]) if query.ndim == 4 else 1)
    heads = int(query.shape[-2])
    if int(block_tables.shape[-1]) != sparse_mla_top_k or block_tables.numel() != (
        rows * sparse_mla_top_k
    ):
        raise unsupported(
            name,
            f"block_tables of shape {tuple(block_tables.shape)}",
            "one row of sparse_mla_top_k physical token indices per query token",
        )
    if cum_seq_lens_q is not None and cum_seq_lens_q.device != query.device:
        raise unsupported(name, "cum_seq_lens_q", "offsets on the query's device")
    lens = sparse_mla_top_k_lens
    if lens is None and seq_lens is not None:
        # the trtllm-gen RoPE convention: a request's live count is min(seq_len, top_k)
        if (
            seq_lens.dtype != torch.int32
            or seq_lens.device != query.device
            or (query.ndim == 4 and seq_lens.numel() != int(query.shape[0]))
        ):
            raise unsupported(
                name, "seq_lens", "one int32 per request on the query's device"
            )
        per_req = seq_lens.reshape(-1).clamp(max=sparse_mla_top_k)
        if query.ndim == 4:
            lens = per_req.repeat_interleave(int(query.shape[1]))
        else:
            if (
                cum_seq_lens_q.numel() != per_req.numel() + 1
                or cum_seq_lens_q.device != query.device
            ):
                raise unsupported(
                    name, "cum_seq_lens_q", "batch + 1 offsets on the query's device"
                )
            # the offsets are trusted: a token past the last offset takes the last request's count
            token = torch.arange(rows, device=query.device, dtype=cum_seq_lens_q.dtype)
            req = torch.searchsorted(cum_seq_lens_q[1:].contiguous(), token, right=True)
            lens = per_req[req.clamp_(max=per_req.numel() - 1)]
        lens = lens.contiguous()
    shape = (*query.shape[:-1], spec.D_LATENT)
    if out is not None and tuple(out.shape) != shape:
        raise unsupported(name, f"out of shape {tuple(out.shape)}", f"{shape}")
    if not (query.is_contiguous() and block_tables.is_contiguous()):
        raise unsupported(
            name, "a non-contiguous query or index list", "contiguous tensors"
        )
    flat = sparse_mla(
        query.view(rows, heads, D),
        kv_cache,
        block_tables.view(rows, sparse_mla_top_k),
        lens,
        bmm1_scale=float(bmm1_scale),
        bmm2_scale=float(bmm2_scale),
        out=None if out is None else out.view(rows, heads, spec.D_LATENT),
    )
    return flat.view(shape) if out is None else out


def ex2_device(
    xs: Sequence[float], device: Optional[torch.device] = None
) -> List[float]:
    """``ex2.approx.ftz.f32`` of every binary32 argument, on the device (the exact CPU model's table)."""
    if not len(xs):
        return []
    x = torch.tensor(list(xs), dtype=torch.float32, device=device or "cuda")
    y = torch.empty_like(x)
    _module().mla_batch_invariant_sparse_ex2(x, y)
    return y.cpu().tolist()
