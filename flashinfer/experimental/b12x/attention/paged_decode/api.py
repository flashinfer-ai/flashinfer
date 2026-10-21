"""Public surface for :mod:`b12x.attention.paged_decode`."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ..._lib.gating import default_is_supported
from . import META
from ._preparation import Binding, make_plan
from ._tuning import PagedDecodeConfig, PagedDecodeQuery, validate_query

_COMPONENT = "attention.paged_decode"


@dataclass(frozen=True, kw_only=True)
class Caps:
    """One attention layer family's decode/verify contract.

    ``max_q_per_req`` bounds the query rows of any request in a batch (1 for
    plain decode, 1 + draft tokens for speculative verification); requests may
    carry any mix of lengths up to it.  ``window_left`` bounds visible keys on
    the left (-1: none).  FP8 KV dequant scales are one value per layer
    (``descale_layout="tensor"``), per request (``"request"``) or per request
    and KV head (``"head"``).  ``causal=False`` lets every query row see the whole
    sequence (speculative drafters that attend across their query block).
    """

    device: torch.device | str
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    max_batch: int
    max_q_per_req: int = 8
    kv_dtype: torch.dtype = torch.bfloat16
    window_left: int = -1
    causal: bool = True
    has_sinks: bool = False
    descale_layout: str = "tensor"


def plan(
    caps: Caps, *, invocation=None, override: PagedDecodeConfig | None = None
) -> Plan:
    """Declare a layer family; ``PreparationSession`` compiles and admits it."""
    return make_plan(caps, invocation=invocation, override=override)


def bind(
    plan: Plan,
    *,
    scratch,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    attention_sink_bias: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> Binding:
    """Bind caller tensors to a prepared plan (capture-safe: no allocation).

    Shapes: ``q [T, Hq, Dqk]`` (may be a strided view), ``output [T, Hq, Dvo]``,
    ``k_cache [pages, page_size, Hkv, Dqk]`` and ``v_cache [pages, page_size,
    Hkv, Dvo]`` views, ``page_table [B, W]``,
    ``cache_seqlens [B]`` (KV length including the new rows), ``cu_seqlens_q
    [B + 1]``; FP8 caches take float32 ``k_descale``/``v_descale`` of shape
    ``[1]``, ``[B]`` or ``[B, Hkv]`` per ``Caps.descale_layout``.  ``scratch`` is the
    caller-owned buffers of ``plan.scratch_specs()``.

    All tensors must be on the plan's device with unit inner strides.  Q and
    cache addresses and outer strides must be 16-byte aligned.  Cache outer
    strides must be positive, and each page stride must be divisible by the
    selected stage-row count times ``cache.stride(1)``.  Both caches must have the
    same page count.  Split scratch must be contiguous; output and the FP32
    output partial buffer require 16-byte aligned addresses.
    """
    state = require_prepared(plan, _COMPONENT, q.device)
    return state.bind(
        plan=plan,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        attention_sink_bias=attention_sink_bias,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
    )


def run(binding: Binding) -> torch.Tensor:
    """Launch the prepared forward (and split merge); returns ``output``."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be paged_decode.Binding")
    return require_prepared(binding.plan, _COMPONENT, binding.q.device).run(binding)


def write_kv(
    plan: Plan,
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Append new K/V rows to the layer family's paged caches.

    ``key [T, Hkv, Dqk]`` and ``value [T, Hkv, Dvo]`` are BF16 (strided rows
    allowed); ``k_cache``/``v_cache`` are the same views ``bind`` takes;
    ``slot_mapping [T]`` is int64 ``page * page_size + offset`` with negative
    entries skipped (padded rows).  FP8 caches store ``e4m3(x / scale)`` with
    float32 ``k_scale``/``v_scale`` of one element.  Capture-safe.
    """
    require_prepared(plan, _COMPONENT, key.device).write_kv(
        key=key,
        value=value,
        k_cache=k_cache,
        v_cache=v_cache,
        slot_mapping=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


def supports(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    kv_dtype: torch.dtype = torch.bfloat16,
    max_q_per_req: int = 8,
    window_left: int = -1,
    causal: bool = True,
    has_sinks: bool = False,
    descale_layout: str = "tensor",
) -> bool:
    """Whether a layer family's geometry is within this op's contract."""
    try:
        validate_query(
            PagedDecodeQuery(
                kv_dtype=str(kv_dtype).removeprefix("torch."),
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(num_kv_heads),
                head_dim_qk=int(head_dim_qk),
                head_dim_vo=int(head_dim_vo),
                page_size=int(page_size),
                max_batch=1,
                max_q_per_req=int(max_q_per_req),
                window_left=int(window_left),
                causal=bool(causal),
                has_sinks=bool(has_sinks),
                descale_layout=str(descale_layout),
                sm_count=1,
            )
        )
    except (TypeError, ValueError):
        return False
    return True


def clear_caches() -> None:
    """Compiled programs live on prepared plans; nothing is cached globally."""


__all__ = [
    "Binding",
    "Caps",
    "PagedDecodeConfig",
    "PagedDecodeQuery",
    "Plan",
    "bind",
    "clear_caches",
    "is_supported",
    "plan",
    "run",
    "supports",
    "write_kv",
]
