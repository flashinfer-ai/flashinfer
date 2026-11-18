# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The API delegate of the sparse MLA classes: the refusals of ``batch_invariant=True`` on the ``prims_ts`` sparse
MLA wrapper, and the route of ``trtllm_batch_decode_with_kv_cache_mla(..., batch_invariant=True)`` to that wrapper."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from ..support import check_device, unsupported
from . import spec

_STABLE_API = "trtllm_batch_decode_with_kv_cache_mla"
#: Plans per call shape. Never evicted: a captured CUDA graph reads its plan's buffers.
_plans: Dict[tuple, tuple] = {}


def clear_plans() -> None:
    """Drop every cached plan of the stable function. CUDA graphs captured through it must not be replayed afterwards."""
    _plans.clear()


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


def check_plan(
    device,
    *,
    num_qo_heads,
    q_dtype,
    qk_rope_head_dim,
    softmax_scale,
    refused,
) -> str:
    """Refuse every plan input outside the classes; return the class name. ``refused`` maps the names of arguments
    the classes do not take to their values (``None`` / ``False`` / ``0`` mean unused)."""
    name = _family_name(q_dtype, qk_rope_head_dim)
    check_device(torch.device(device), name, spec.MLA_ARCHS)
    if q_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise unsupported(name, f"q_dtype {q_dtype}", "bf16 or e4m3 Q / KV")
    if qk_rope_head_dim not in (0, spec.ROPE_DIM):
        raise unsupported(
            name,
            f"qk_rope_head_dim={qk_rope_head_dim!r}",
            "0 (512-wide) or 64 (576-wide)",
        )
    if softmax_scale is None:
        raise unsupported(
            name,
            "the default softmax_scale ((512 + qk_rope_head_dim) ** -0.5)",
            "an explicit softmax_scale (the scale is an input of the row term)",
        )
    for arg, value in refused.items():
        default = not isinstance(value, torch.Tensor) and (arg, value) in _PLAN_DEFAULTS
        if _used(value) and not default:
            raise unsupported(
                name, f"{arg}={value!r}", _LIMITS.get(arg, _DEFAULT_LIMIT)
            )
    if isinstance(num_qo_heads, int) and num_qo_heads > spec.MAX_HEADS:
        raise unsupported(name, f"{num_qo_heads} query heads", "1 to 128 heads")
    return name


_DEFAULT_LIMIT = "the class's own inputs"
_PLAN_DEFAULTS = (
    ("kv_cache_format", "native"),
    ("extra_kv_cache_format", "native"),
    ("kv_cache_scale", 1.0),
)
_LIMITS = {
    "max_extra_topk": "one KV pool (no extra source list)",
    "extra_kv_cache_format": "one native KV pool",
    "kv_cache_format": 'kv_cache_format="native" (bf16 or e4m3 rows; the quantized record formats are not in the classes)',
    "use_attention_sinks": "no attention sinks",
    "output_quant_block_size": "bf16 output (no fused output quantization)",
    "tuning": "the pinned softmax rescale (threshold 0)",
    "kv_cache_scale": "native caches (no per-tensor cache scale)",
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


def _override(scale, planned):
    """``None`` when a call's scale is the plan's own (the run then skips re-normalizing it)."""
    if scale is planned:
        return None
    if isinstance(scale, torch.Tensor) or scale != planned:
        return scale
    return None


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
    refused: Optional[dict],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """``trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k > 0, backend="auto", batch_invariant=True)`` on
    SM100 / SM103 / SM107: every query token is one row of the pinned ``prims_ts`` sparse MLA kernel.

    ``block_tables`` holds each token's selected storage rows (``page * page_size + offset``), live entries first and
    ``-1`` after them; ``sparse_mla_top_k_lens`` gives the live counts and defaults to the full width (a ``-1`` tail is
    masked and gives the same bits as the shorter live count). ``bmm1_scale`` and ``bmm2_scale`` are the softmax and
    output scales. ``seq_lens`` and ``max_seq_len`` are not read: a row sees its list and nothing else."""

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
    if kv_cache.dtype != query.dtype:
        raise unsupported(
            name(),
            f"a {kv_cache.dtype} KV cache with a {query.dtype} query",
            "bf16 or e4m3 rows in the query's dtype",
        )
    want_lse = return_lse or lse is not None
    if want_lse and return_lse_base != "basee":
        raise unsupported(
            name(),
            f"LSE output with return_lse_base={return_lse_base!r}",
            'natural-log LSE (return_lse_base="basee")',
        )
    if refused:
        for arg, value in refused.items():
            if _used(value):
                raise unsupported(
                    name(), f"{arg}={value!r}", _LIMITS.get(arg, _DEFAULT_LIMIT)
                )
    if not isinstance(bmm1_scale, torch.Tensor) and not (
        isinstance(bmm1_scale, (int, float)) and bmm1_scale > 0
    ):
        raise unsupported(
            name(), f"bmm1_scale={bmm1_scale!r}", "a positive softmax scale"
        )

    head_dim = spec.LATENT_DIM + qk_rope_head_dim
    shape = query.shape
    prefix: Tuple[int, ...]
    if len(shape) == 4:
        prefix = (shape[0], shape[1])
        rows = shape[0] * shape[1]
    elif len(shape) == 3 and cum_seq_lens_q is not None:
        prefix = (shape[0],)
        rows = shape[0]
    else:
        raise unsupported(
            name(),
            f"a query of shape {tuple(shape)}",
            "[batch, q_len, heads, head_dim], or [total_q, heads, head_dim] with cum_seq_lens_q",
        )
    heads = shape[-2]
    if shape[-1] != head_dim:
        raise unsupported(
            name(), f"a query of width {shape[-1]}", f"width {head_dim} for this layout"
        )
    if (
        block_tables.ndim not in (2, 3)
        or block_tables.shape[-1] != sparse_mla_top_k
        or block_tables.numel() != rows * sparse_mla_top_k
    ):
        raise unsupported(
            name(),
            f"block_tables of shape {tuple(block_tables.shape)} for {rows} query tokens",
            "one [.., sparse_mla_top_k] row of selected storage rows per query token",
        )
    indices = (
        block_tables
        if block_tables.ndim == 2
        else block_tables.reshape(rows, sparse_mla_top_k)
    )
    if kv_cache.ndim == 4:
        if kv_cache.shape[1] != 1:
            raise unsupported(name(), "a KV cache with several KV heads", "one KV head")
        kv_cache = kv_cache.squeeze(1)

    key = (
        query.device,
        rows,
        heads,
        sparse_mla_top_k,
        query.dtype,
        qk_rope_head_dim,
        isinstance(bmm1_scale, torch.Tensor),
        isinstance(bmm2_scale, torch.Tensor),
    )
    plan = _plans.get(key)
    if plan is None:
        check_device(query.device, name(), spec.MLA_ARCHS)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{_STABLE_API}(batch_invariant=True): run one call of this shape before CUDA graph capture"
            )
        from ...api_logging import warn_experimental_backend_once
        from ...attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

        warn_experimental_backend_once(
            _STABLE_API, "prims_ts (batch-invariant sparse MLA)"
        )
        wrapper = BatchSparseMLADecodePagedTSWrapper()
        wrapper.plan(
            query.device,
            rows,
            heads,
            max_topk=sparse_mla_top_k,
            q_dtype=query.dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            batch_invariant=True,
            softmax_scale=bmm1_scale,
            output_scale=bmm2_scale,
        )
        full = torch.full(
            (rows,), sparse_mla_top_k, dtype=torch.int32, device=query.device
        )
        plan = (wrapper, full, bmm1_scale, bmm2_scale)
        _plans[key] = plan
    wrapper, full, planned_softmax, planned_output = plan

    if sparse_mla_top_k_lens is None:
        lengths = full
    elif sparse_mla_top_k_lens.ndim == 1:
        lengths = sparse_mla_top_k_lens
    else:
        lengths = sparse_mla_top_k_lens.reshape(rows)

    if out is None:
        out = torch.empty(
            (*prefix, heads, spec.LATENT_DIM), dtype=torch.bfloat16, device=query.device
        )
    elif out.shape != (*prefix, heads, spec.LATENT_DIM) or out.dtype != torch.bfloat16:
        raise unsupported(
            name(),
            f"out of shape {tuple(out.shape)} and dtype {out.dtype}",
            f"bf16 {(*prefix, heads, spec.LATENT_DIM)}",
        )
    if want_lse:
        if lse is None:
            lse = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
        elif lse.numel() != rows * heads or lse.dtype != torch.float32:
            raise unsupported(
                name(),
                f"lse of shape {tuple(lse.shape)}",
                f"fp32 with {rows} x {heads} values",
            )
    wrapper.run(
        query
        if len(shape) == 4 and shape[1] == 1
        else query.view(rows, 1, heads, head_dim),
        kv_cache,
        indices,
        lengths,
        softmax_scale=_override(bmm1_scale, planned_softmax),
        output_scale=_override(bmm2_scale, planned_output),
        out=out
        if len(shape) == 4 and shape[1] == 1
        else out.view(rows, 1, heads, spec.LATENT_DIM),
        lse=lse.view(rows, 1, heads) if want_lse else None,
        validate=validate,
    )
    return (out, lse) if return_lse else out
