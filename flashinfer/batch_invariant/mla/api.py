# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The API delegate of the sparse MLA classes: what ``trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k > 0,
backend="auto", batch_invariant=True)`` runs. It maps the call onto a plan of :mod:`.engine` and refuses every argument
the classes do not take."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from ..support import unsupported
from . import engine, spec
from .engine import clear_plans  # noqa: F401

#: Full-width list lengths per (device, rows, top_k). Never evicted: a captured CUDA graph reads them.
_full: Dict[tuple, torch.Tensor] = {}


_STABLE_API = "trtllm_batch_decode_with_kv_cache_mla"


def _used(value) -> bool:
    return not (
        value is None
        or value is False
        or (isinstance(value, (int, float)) and value == 0)
    )


def _family_name(q_dtype, qk_rope_head_dim) -> str:
    try:
        return spec.class_name(q_dtype, spec.LATENT_DIM + qk_rope_head_dim)
    except (ValueError, TypeError):
        return spec.FAMILY


_DEFAULT_LIMIT = "the class's own inputs"
_LIMITS = {
    "sinks": "no attention sinks",
    "skip_softmax_threshold_scale_factor": "standard attention",
    "enable_dcp": "one rank (no decode context parallelism; the cross-rank merge is outside the classes)",
    "cp_world": "one rank (no decode context parallelism; the cross-rank merge is outside the classes)",
    "cp_rank": "one rank (no decode context parallelism; the cross-rank merge is outside the classes)",
    "causal_seqlens_kv_global": "one rank (no decode context parallelism)",
    "use_fp16_softmax": "the fp32 softmax of the pinned kernel",
    "multi_ctas_kv_counter_buffer": "no trtllm-gen counter buffer (the KV list is never split)",
    "uses_shared_paged_kv_idx=False": "the shared K / V index layout",
    "cute_dsl_impl": 'cute_dsl_impl="auto" (the class is the kernel)',
    "kv_scale_format": 'kv_scale_format="auto" (native rows carry no inline scales)',
}


def sparse_mla_batch_invariant(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    sparse_mla_top_k: int,
    out: Optional[torch.Tensor],
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    backend: str,
    lse: Optional[torch.Tensor],
    return_lse: bool,
    return_lse_base: Optional[str],
    cum_seq_lens_q: Optional[torch.Tensor],
    sparse_mla_top_k_lens: Optional[torch.Tensor],
    validate: bool,
    refused: tuple,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """``trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k > 0, backend="auto", batch_invariant=True)`` on
    SM100 / SM103 / SM107: every query token is one row of the class's kernel.

    ``block_tables`` holds each token's selected storage rows (``page * page_size + offset``), live entries first and
    ``-1`` after them; ``sparse_mla_top_k_lens`` gives the live counts and defaults to the full width (a ``-1`` tail is
    masked and gives the same bits as the shorter live count). ``bmm1_scale`` and ``bmm2_scale`` are the softmax and
    output scales. ``seq_lens`` and ``max_seq_len`` are not read: a row sees its list and nothing else. ``refused``
    holds ``(name, value)`` pairs of the arguments the classes do not take."""

    def name() -> str:
        return _family_name(query.dtype, qk_rope_head_dim)

    if backend != "auto":
        raise unsupported(
            name(),
            f'backend="{backend}" with batch_invariant=True',
            'backend="auto" (the class is the kernel)',
        )
    if not isinstance(sparse_mla_top_k, int) or sparse_mla_top_k <= 0:
        raise unsupported(
            name(),
            f"sparse_mla_top_k={sparse_mla_top_k!r}",
            "sparse MLA (sparse_mla_top_k > 0)",
        )
    if kv_lora_rank != spec.LATENT_DIM:
        raise unsupported(name(), f"kv_lora_rank={kv_lora_rank!r}", "kv_lora_rank=512")
    want_lse = return_lse or lse is not None
    if want_lse and return_lse_base != "basee":
        raise unsupported(
            name(),
            f"LSE output with return_lse_base={return_lse_base!r}",
            'natural-log LSE (return_lse_base="basee")',
        )
    for arg, value in refused:
        if _used(value):
            raise unsupported(
                name(), f"{arg}={value!r}", _LIMITS.get(arg, _DEFAULT_LIMIT)
            )

    if bmm1_scale is None or bmm2_scale is None:
        raise unsupported(
            name(),
            f"bmm1_scale={bmm1_scale!r}, bmm2_scale={bmm2_scale!r}",
            "explicit softmax and output scales",
        )

    shape = query.shape
    ndim = len(shape)
    if ndim == 4:
        rows = shape[0] * shape[1]
    elif ndim == 3 and cum_seq_lens_q is not None:
        rows = shape[0]
    else:
        raise unsupported(
            name(),
            f"a query of shape {tuple(shape)}",
            "[batch, q_len, heads, head_dim], or [total_q, heads, head_dim] with cum_seq_lens_q",
        )
    heads = shape[-2]
    device = query.device
    plan = engine.planned(
        device,
        heads,
        query.dtype,
        qk_rope_head_dim,
        isinstance(bmm1_scale, torch.Tensor),
        isinstance(bmm2_scale, torch.Tensor),
    )
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{_STABLE_API}(batch_invariant=True): run one call of this shape before CUDA graph capture"
            )
        plan = engine.plan(
            device,
            heads,
            q_dtype=query.dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=bmm1_scale,
            output_scale=bmm2_scale,
        )
    if (
        block_tables.ndim not in (2, 3)
        or block_tables.shape[-1] != sparse_mla_top_k
        or block_tables.numel() != rows * sparse_mla_top_k
    ):
        raise unsupported(
            plan.class_name,
            f"block_tables of shape {tuple(block_tables.shape)} for {rows} query tokens",
            "one [.., sparse_mla_top_k] row of selected storage rows per query token",
        )
    if kv_cache.ndim == 4:
        if kv_cache.shape[1] != 1:
            raise unsupported(
                plan.class_name, "a KV cache with several KV heads", "one KV head"
            )
    if kv_cache.ndim != 2:
        if not kv_cache.is_contiguous():
            raise unsupported(
                plan.class_name,
                "a KV cache with padded pages",
                "contiguous pages (block_tables address page * page_size + offset)",
            )
        kv_cache = kv_cache.view(-1, kv_cache.shape[-1])
    if sparse_mla_top_k_lens is None:
        lengths = _full.get((device, rows, sparse_mla_top_k))
        if lengths is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"{_STABLE_API}(batch_invariant=True): run one call of this shape before CUDA graph capture"
                )
            lengths = torch.full(
                (rows,), sparse_mla_top_k, dtype=torch.int32, device=device
            )
            _full[(device, rows, sparse_mla_top_k)] = lengths
    else:
        lengths = sparse_mla_top_k_lens.reshape(rows)
    latent = spec.LATENT_DIM
    if out is None:
        out = torch.empty((*shape[:-1], latent), dtype=torch.bfloat16, device=device)
    elif out.shape[:-1] != shape[:-1]:
        raise unsupported(
            plan.class_name,
            f"out of shape {tuple(out.shape)}",
            f"bf16 {(*shape[:-1], latent)}",
        )
    if want_lse:
        if lse is None:
            lse = torch.empty((rows, heads), dtype=torch.float32, device=device)
        elif lse.numel() != rows * heads:
            raise unsupported(
                plan.class_name,
                f"lse of shape {tuple(lse.shape)}",
                f"fp32 with {rows} x {heads} values",
            )
    query3 = query if ndim == 3 else query.view(rows, heads, shape[-1])
    indices = (
        block_tables
        if block_tables.ndim == 2
        else block_tables.reshape(rows, sparse_mla_top_k)
    )
    out3 = out if ndim == 3 else out.view(rows, heads, latent)
    lse2 = lse.view(rows, heads) if want_lse else None
    plan.check(query3, kv_cache, indices, lengths, out3, lse2, validate)
    plan.launch(
        query3,
        kv_cache,
        indices,
        lengths,
        out3,
        lse2,
        plan.scale(bmm1_scale, 0, "bmm1_scale"),
        plan.scale(bmm2_scale, 1, "bmm2_scale"),
    )
    return (out, lse) if return_lse else out
