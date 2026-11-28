# SPDX-License-Identifier: Apache-2.0
"""Cached three-axis MRoPE trace and portable numerical reference."""

from typing import Any, cast

import torch
from ..template import Const, Scalar, Tensor, TraceTemplate, Var


def _mrope_round(x, dtype):
    """Round an FP64 result without an FP32 double-rounding tie.

    Torch's double->half/bfloat16 conversion can go through float. Compare its
    two neighboring representable values in FP64 and break ties to an even
    significand. This reference helper is deliberately not performance code.
    """
    rounded = x.to(dtype)
    if dtype == torch.float32:
        return rounded
    best = rounded
    distance = (x - best.double()).abs()
    for direction in (-float("inf"), float("inf")):
        neighbor = torch.nextafter(rounded, torch.full_like(rounded, direction))
        other_distance = (x - neighbor.double()).abs()
        even = (neighbor.view(torch.int16) & 1) == 0
        choose = (other_distance < distance) | ((other_distance == distance) & even)
        best = torch.where(choose, neighbor, best)
        distance = torch.where(choose, other_distance, distance)
    # Infinity is the rounding destination at the overflow midpoint even though
    # the largest finite value has a smaller absolute distance to finite x.
    largest = torch.full((), torch.finfo(dtype).max, dtype=dtype, device=x.device)
    previous = torch.nextafter(largest, torch.zeros_like(largest))
    threshold = largest.double() + (largest.double() - previous.double()) / 2
    overflow = torch.copysign(torch.full_like(rounded, float("inf")), rounded)
    return torch.where(x.abs() >= threshold, overflow, best)


@torch.no_grad()
def _mrope_reference(
    positions,
    query,
    key,
    head_size,
    cos_sin_cache,
    mrope_section,
    mrope_interleaved=False,
):
    """Finite-value reference with explicit cache-precision product/FMA rounding."""
    q = query.reshape(query.shape[0], query.shape[1] // head_size, head_size)
    k = key.reshape(key.shape[0], key.shape[1] // head_size, head_size)
    cache, sections = cos_sin_cache, mrope_section
    axis_interleaved = mrope_interleaved
    r = cache.shape[1]
    half = r // 2
    all_cos, all_sin = cache[positions.long()].chunk(2, dim=-1)
    if axis_interleaved:
        channels = torch.arange(half, device=q.device)
        height = (channels % 3 == 1) & (channels < 3 * sections[1])
        width = (channels % 3 == 2) & (channels < 3 * sections[2])
        cos = torch.where(height, all_cos[1], all_cos[0])
        cos = torch.where(width, all_cos[2], cos)
        sin = torch.where(height, all_sin[1], all_sin[0])
        sin = torch.where(width, all_sin[2], sin)
    else:
        cos = torch.cat(
            [part[i] for i, part in enumerate(all_cos.split(sections, dim=-1))], dim=-1
        )
        sin = torch.cat(
            [part[i] for i, part in enumerate(all_sin.split(sections, dim=-1))], dim=-1
        )
    cos, sin = cos[:, None], sin[:, None]
    result = []
    for x in (q, k):
        y = x.clone()
        a = x[..., :half]
        b = x[..., half:r]
        a, b = a.to(cache.dtype), b.to(cache.dtype)
        # Emulate the two FMAs independently, using wider arithmetic before the
        # cache-type rounding. Test inputs are bounded finite FP16/BF16 values.
        bs, bc = b * sin, b * cos
        first = _mrope_round(a.double() * cos.double() - bs.double(), cache.dtype)
        second = _mrope_round(a.double() * sin.double() + bc.double(), cache.dtype)
        y[..., :half], y[..., half:r] = first, second
        result.append(y)
    return result[0].reshape_as(query), result[1].reshape_as(key)


_mrope_reference._trace_reference_dependencies = (_mrope_round,)


def _mrope_init(
    *,
    nnz,
    max_seq_len=256,
    num_q_heads_x_head_size=2048,
    num_k_heads_x_head_size=1024,
    head_size=128,
    rotary_dim=128,
    section_t=16,
    section_h=24,
    section_w=24,
    mrope_interleaved=0,
    query_bf16=1,
    cache_fp32=1,
    position_int64=1,
    device="cuda",
    seed=0,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16 if query_bf16 else torch.float16
    # Realistic padded QKV rows also exercise native strides in generated replay.
    pitch = num_q_heads_x_head_size + 2 * num_k_heads_x_head_size
    storage = torch.randn(nnz, pitch, dtype=dtype, device=device)
    query = storage[:, :num_q_heads_x_head_size]
    key = storage[
        :, num_q_heads_x_head_size : num_q_heads_x_head_size + num_k_heads_x_head_size
    ]
    return dict(
        positions=torch.randint(
            max_seq_len,
            (3, nnz),
            device=device,
            dtype=torch.int64 if position_int64 else torch.int32,
        ),
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=torch.rand(
            max_seq_len,
            rotary_dim,
            device=device,
            dtype=torch.float32 if cache_fp32 else dtype,
        )
        * 2
        - 1,
        mrope_section=(section_t, section_h, section_w),
        mrope_interleaved=bool(mrope_interleaved),
    )


cast(Any, _mrope_init)._trace_init_const_axes = (
    "num_q_heads_x_head_size",
    "num_k_heads_x_head_size",
    "head_size",
    "rotary_dim",
    "section_t",
    "section_h",
    "section_w",
    "mrope_interleaved",
    "query_bf16",
    "cache_fp32",
    "position_int64",
)


def _section(kw, index):
    sections = kw.get("mrope_section")
    return None if sections is None else sections[index]


apply_mrope_with_cos_sin_cache_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="mrope_cos_sin_cache_inplace",
    description="In-place cached three-axis MRoPE with NeoX pairing and cache-type arithmetic.",
    axes={
        "nnz": Var(),
        "max_seq_len": Var(),
        "num_axes": Const(value=3, abbrev=""),
        "num_q_heads_x_head_size": Const(abbrev="q"),
        "num_k_heads_x_head_size": Const(abbrev="k"),
        "head_size": Const(abbrev="d"),
        "rotary_dim": Const(abbrev="r"),
        "section_t": Const(abbrev="t", extract=lambda kw: _section(kw, 0)),
        "section_h": Const(abbrev="h", extract=lambda kw: _section(kw, 1)),
        "section_w": Const(abbrev="w", extract=lambda kw: _section(kw, 2)),
        "mrope_interleaved": Const(
            abbrev="i", extract=lambda kw: int(kw.get("mrope_interleaved", False))
        ),
        "query_bf16": Const(
            abbrev="b", extract=lambda kw: int(kw["query"].dtype == torch.bfloat16)
        ),
        "cache_fp32": Const(
            abbrev="f",
            extract=lambda kw: int(kw["cos_sin_cache"].dtype == torch.float32),
        ),
        "position_int64": Const(
            abbrev="p", extract=lambda kw: int(kw["positions"].dtype == torch.int64)
        ),
    },
    inputs={
        "positions": Tensor(["num_axes", "nnz"]),
        "query": Tensor(["nnz", "num_q_heads_x_head_size"]),
        "key": Tensor(["nnz", "num_k_heads_x_head_size"]),
        "head_size": Scalar("int32"),
        "cos_sin_cache": Tensor(["max_seq_len", "rotary_dim"]),
        # Section counts and layout are bound Const axes in the exported init.
        "mrope_interleaved": Scalar("int32", optional=True),
    },
    outputs={
        "query": Tensor(["nnz", "num_q_heads_x_head_size"], dtype_from="query"),
        "key": Tensor(["nnz", "num_k_heads_x_head_size"], dtype_from="key"),
    },
    reference=_mrope_reference,
    init=_mrope_init,
)
