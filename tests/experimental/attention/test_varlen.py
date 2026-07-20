from __future__ import annotations

import math
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_sm12x as require_sm120


def _require_contiguous_backend() -> torch.device:
    device = require_sm120()
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    return device


def _run_attention_with_plan(
    plan,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    attention_sink_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.experimental.sm12x.attention._shared.contiguous import (
        sm12x_attention_forward,
        plan_attention_scratch,
    )

    scratch_plan = plan_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        attention_sink_bias=attention_sink_bias,
    )
    return sm12x_attention_forward(binding=binding)


def _run_varlen_attention_with_plan(
    plan,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: Optional[bool] = None,
    window_size: Optional[Tuple[int, int]] = None,
    softmax_scale: Optional[float] = None,
    attention_sink_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.experimental.sm12x.attention._shared.contiguous import (
        sm12x_varlen_attention_forward,
        plan_varlen_attention_scratch,
    )

    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        window_size=window_size,
        softmax_scale=softmax_scale,
        attention_sink_bias=attention_sink_bias,
    )
    return sm12x_varlen_attention_forward(binding=binding)


def _vision_reference_attention_segment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sinks: Optional[torch.Tensor],
    num_kv_groups: int,
    softmax_scale: float,
    window_size: Tuple[int, int],
) -> torch.Tensor:
    output_dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()

    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

    q_len = q.shape[0]
    k_len = k.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q, k) * softmax_scale

    left, right = window_size
    if left != -1 or right != -1:
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
        q_aligned = q_pos + k_len - q_len
        keep = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
        if left != -1:
            keep &= k_pos >= q_aligned - left
        if right != -1:
            keep &= k_pos <= q_aligned + right
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))

    if sinks is not None:
        sink_logits = sinks.to(device=q.device, dtype=torch.float32).view(
            q.shape[1], 1, 1
        )
        scores = torch.cat([scores, sink_logits.expand(q.shape[1], q_len, 1)], dim=-1)
        attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32)[..., :k_len]
    else:
        attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32)

    return torch.einsum("hqk,khd->qhd", attn_probs, v).to(dtype=output_dtype)


def _vision_torch_ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[int, int] = (-1, -1),
    s_aux: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be packed rank-3 tensors")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("q head count must be divisible by kv head count")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    offsets = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    outputs = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=False):
        outputs.append(
            _vision_reference_attention_segment(
                q[start:end],
                k[start:end],
                v[start:end],
                sinks=s_aux,
                num_kv_groups=q.shape[1] // k.shape[1],
                softmax_scale=float(softmax_scale),
                window_size=window_size,
            )
        )
    if not outputs:
        return torch.empty_like(q)
    return torch.cat(outputs, dim=0)


def _pack_rank4_segments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 tensor, got rank {x.ndim}")
    batch, seqlen = int(x.shape[0]), int(x.shape[1])
    cu_seqlens = torch.arange(
        0,
        (batch + 1) * seqlen,
        seqlen,
        dtype=torch.int32,
        device=x.device,
    )
    return x.reshape(batch * seqlen, *x.shape[2:]).contiguous(), cu_seqlens


def _contiguous_ref_from_rank4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_size: Tuple[int, int],
    s_aux: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    q_packed, cu_seqlens = _pack_rank4_segments(q)
    k_packed, k_cu_seqlens = _pack_rank4_segments(k)
    v_packed, v_cu_seqlens = _pack_rank4_segments(v)
    if not torch.equal(cu_seqlens, k_cu_seqlens) or not torch.equal(
        cu_seqlens, v_cu_seqlens
    ):
        raise ValueError(
            "torch_ref vision attention expects matching packed segment lengths"
        )
    out = _vision_torch_ref_attention(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens,
        softmax_scale=softmax_scale,
        window_size=window_size,
        s_aux=s_aux,
    )
    return out.reshape(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])


def _make_gqa_inputs(
    shape: tuple[int, int, int, int],
    *,
    kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch, seqlen, q_heads, head_dim = shape
    q = (
        torch.randn(
            batch,
            seqlen,
            q_heads,
            head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        / 4
    )
    k = (
        torch.randn(
            batch,
            seqlen,
            kv_heads,
            head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        / 4
    )
    v = (
        torch.randn(
            batch,
            seqlen,
            kv_heads,
            head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        / 4
    )
    return q.contiguous(), k.contiguous(), v.contiguous()


def _make_varlen_gqa_inputs(
    lengths: tuple[int, ...],
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total = sum(lengths)
    q = (
        torch.randn(
            total, q_heads, head_dim, generator=generator, device=device, dtype=dtype
        )
        / 4
    )
    k = (
        torch.randn(
            total, kv_heads, head_dim, generator=generator, device=device, dtype=dtype
        )
        / 4
    )
    v = (
        torch.randn(
            total, kv_heads, head_dim, generator=generator, device=device, dtype=dtype
        )
        / 4
    )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    return q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def test_sglang_torch_ref_handles_local_window_gqa_and_sinks_on_cpu() -> None:
    q, k, v = _make_gqa_inputs(
        (2, 7, 4, 16),
        kv_heads=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        seed=11,
    )
    sinks = torch.linspace(-0.5, 0.25, q.shape[2])

    out = _contiguous_ref_from_rank4(
        q,
        k,
        v,
        window_size=(2, 1),
        s_aux=sinks,
    )
    out_without_sinks = _contiguous_ref_from_rank4(
        q,
        k,
        v,
        window_size=(2, 1),
    )

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, out_without_sinks)


@pytest.mark.parametrize(
    ("causal", "window_size"),
    [
        (True, (-1, 0)),
        (False, (-1, -1)),
        (False, (8, 8)),
    ],
)
@torch.inference_mode()
def test_contiguous_attention_matches_sglang_torch_ref(
    causal: bool,
    window_size: Tuple[int, int],
) -> None:
    device = _require_contiguous_backend()
    from flashinfer.experimental.sm12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_attention_plan,
    )

    clear_attention_caches()
    q, k, v = _make_gqa_inputs(
        (1, 48, 4, 64),
        kv_heads=2,
        dtype=torch.bfloat16,
        device=device,
        seed=23 if causal else 29,
    )

    plan = create_attention_plan(q, k, v, causal=causal, window_size=window_size)
    out, _lse = _run_attention_with_plan(
        plan,
        q,
        k,
        v,
    )
    torch.cuda.synchronize()

    ref = _contiguous_ref_from_rank4(
        q,
        k,
        v,
        window_size=window_size,
    )

    assert (out - ref).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref) >= 0.9999


@torch.inference_mode()
def test_varlen_contiguous_attention_matches_sglang_torch_ref_swa_gqa_and_sinks() -> (
    None
):
    device = _require_contiguous_backend()
    from flashinfer.experimental.sm12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
    )

    clear_attention_caches()
    lengths = (5, 17, 9)
    q, k, v, cu_seqlens = _make_varlen_gqa_inputs(
        lengths,
        q_heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        device=device,
        seed=37,
    )
    window_size = (4, 3)
    max_seqlen = max(lengths)
    sinks = torch.linspace(
        -0.25,
        0.5,
        q.shape[1],
        dtype=torch.float32,
        device=device,
    )

    plan = create_varlen_attention_plan(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
        window_size=window_size,
        attention_sink_bias=sinks,
    )
    out, _lse = _run_varlen_attention_with_plan(
        plan,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
        window_size=window_size,
        attention_sink_bias=sinks,
    )
    torch.cuda.synchronize()

    ref = _vision_torch_ref_attention(
        q,
        k,
        v,
        cu_seqlens,
        window_size=window_size,
        s_aux=sinks,
    )

    assert (out - ref).abs().max().item() <= 0.035
    assert _cosine_similarity(out, ref) >= 0.9998
