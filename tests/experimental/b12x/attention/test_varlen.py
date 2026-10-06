from __future__ import annotations

import math
import gc
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x as require_sm120


def _require_contiguous_backend() -> torch.device:
    device = require_sm120()
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    return device


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim,value_dim", [(128, 128), (192, 128), (256, 256)])
def test_varlen_capacity_reuses_program_for_live_lengths(
    monkeypatch, causal, head_dim, value_dim,
):
    """One prepared capacity serves changing rows and batches, including graphs."""
    from b12x.attention import varlen
    from b12x.attention._shared.contiguous import api as native
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    device = _require_contiguous_backend()
    torch.manual_seed(441)
    qcap = torch.empty((257, 3, head_dim), device=device, dtype=torch.bfloat16)
    device = qcap.device
    kcap = torch.empty((1025, 3, head_dim), device=device, dtype=torch.bfloat16)
    vcap = torch.empty((1025, 3, value_dim), device=device, dtype=torch.bfloat16)
    cucap = torch.zeros(5, device=device, dtype=torch.int32)
    declaration = varlen.plan(qcap, kcap, vcap, cucap, cucap,
                              max_seqlen_q=257, max_seqlen_k=1025, causal=causal)

    def prepare(state):
        spec, = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(scratch=scratch, q=qcap, k=kcap, v=vcap,
                             cu_seqlens_q=cucap, cu_seqlens_k=cucap)
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch,))

    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        session.prepare((declaration.request(name="varlen-capacity", prepare_call=prepare),))
        state = require_prepared(declaration, "attention.varlen", device)
        program = state.plan.compiled
        spec, = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

        def no_compile(*args, **kwargs):
            raise AssertionError("Live lengths must reuse the prepared program")

        monkeypatch.setattr(native, "_compile_varlen_attention", no_compile)
        for q_lengths, k_lengths in [([1], [7]), ([37, 91], [257, 303]),
                                      ([0, 129, 128], [13, 501, 511])]:
            q = torch.randn((sum(q_lengths), 3, head_dim), device=device, dtype=qcap.dtype)
            k = torch.randn((sum(k_lengths), 3, head_dim), device=device, dtype=qcap.dtype)
            v = torch.randn((sum(k_lengths), 3, value_dim), device=device, dtype=qcap.dtype)
            cq = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()],
                              device=device, dtype=torch.int32)
            ck = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()],
                              device=device, dtype=torch.int32)
            binding = varlen.bind(declaration, scratch=scratch, q=q, k=k, v=v,
                                  cu_seqlens_q=cq, cu_seqlens_k=ck,
                                  max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths))
            assert binding.binding.plan.compiled is program
            for _ in range(3):
                varlen.run(binding)
            torch.cuda.synchronize()
            gc.collect()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                out, lse = varlen.run(binding)
            for _ in range(2):
                q.normal_()
                v.normal_()
                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                qstart = kstart = 0
                for qlen, klen in zip(q_lengths, k_lengths):
                    if qlen:
                        scores = torch.einsum("qhd,khd->hqk", q[qstart:qstart+qlen].double(),
                                              k[kstart:kstart+klen].double()) / math.sqrt(head_dim)
                        if causal:
                            mask = (torch.arange(klen, device=device)[None, :] >
                                    torch.arange(qlen, device=device)[:, None] + klen - qlen)
                            scores.masked_fill_(mask[None], -torch.inf)
                        truth = torch.einsum("hqk,khd->qhd", scores.softmax(-1),
                                             v[kstart:kstart+klen].double())
                        torch.testing.assert_close(out[qstart:qstart+qlen].float(), truth.float(),
                                                   atol=0.02, rtol=0.01)
                        torch.testing.assert_close(lse[:, qstart:qstart+qlen].double(),
                                                   scores.logsumexp(-1), atol=0.003, rtol=0.001)
                    qstart += qlen
                    kstart += klen
                assert torch.isfinite(out).all() and torch.count_nonzero(out)
            graph.reset()


def _run_attention_with_plan(
    declaration,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    attention_sink_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    def prepare_call(state):
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(
            scratch=scratch, q=q, k=k, v=v,
            softmax_scale=softmax_scale, attention_sink_bias=attention_sink_bias,
        )
        return PreparedCall(run=lambda: state.run(binding))

    with PreparationSession(device=q.device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(
            name="batched", prepare_call=prepare_call,
        ),))
        plan = declaration
        state = require_prepared(plan, "attention.varlen", q.device)
        binding = varlen.bind_batched(
            plan, scratch=torch.empty(
                state.scratch_plan.scratch_specs()[0].shape,
                dtype=state.scratch_plan.scratch_specs()[0].dtype, device=q.device,
            ), q=q, k=k, v=v, softmax_scale=softmax_scale,
            attention_sink_bias=attention_sink_bias,
        )
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output, lse = varlen.run_batched(binding)
        output.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(q.device)
        return output, lse


def _run_varlen_attention_with_plan(
    declaration,
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
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    def prepare_call(state):
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(
            scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu_seqlens,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, causal=causal,
            window_size=window_size, softmax_scale=softmax_scale,
            attention_sink_bias=attention_sink_bias,
        )
        return PreparedCall(run=lambda: state.run(binding))

    with PreparationSession(device=q.device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(
            name="varlen", prepare_call=prepare_call,
        ),))
        plan = declaration
        state = require_prepared(plan, "attention.varlen", q.device)
        (spec,) = state.scratch_plan.scratch_specs()
        binding = varlen.bind(
            plan,
            scratch=torch.empty(spec.shape, dtype=spec.dtype, device=spec.device),
            q=q, k=k, v=v, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k, causal=causal, window_size=window_size,
            softmax_scale=softmax_scale, attention_sink_bias=attention_sink_bias,
        )
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output, lse = varlen.run(binding)
        output.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(q.device)
        return output, lse


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
    from b12x.attention import varlen

    q, k, v = _make_gqa_inputs(
        (1, 48, 4, 64),
        kv_heads=2,
        dtype=torch.bfloat16,
        device=device,
        seed=23 if causal else 29,
    )

    declaration = varlen.plan_batched(q, k, v, causal=causal, window_size=window_size)
    out, _lse = _run_attention_with_plan(
        declaration,
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
    from b12x.attention import varlen
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

    declaration = varlen.plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen, causal=False, window_size=window_size,
        attention_sink_bias=sinks,
    )
    out, _lse = _run_varlen_attention_with_plan(
        declaration,
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


@torch.inference_mode()
def test_unequal_value_prefill_qk256_v128_causal_window_ragged() -> None:
    from b12x.attention import varlen

    device = _require_contiguous_backend()
    torch.manual_seed(20260813)
    lengths = [65, 514]
    total = sum(lengths)
    heads = 8
    q = (torch.randn(total, heads, 256, dtype=torch.float32, device=device) * 0.1).to(
        torch.bfloat16
    )
    k = (torch.randn(total, heads, 256, dtype=torch.float32, device=device) * 0.1).to(
        torch.bfloat16
    )
    v = (torch.randn(total, heads, 128, dtype=torch.float32, device=device) * 0.1).to(
        torch.bfloat16
    )
    cu = torch.tensor([0, 65, total], dtype=torch.int32, device=device)
    declaration = varlen.plan(
        q, k, v, cu, max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
        causal=True, window_size=(512, 0),
    )
    actual, actual_lse = _run_varlen_attention_with_plan(
        declaration, q, k, v, cu, max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths), softmax_scale=1.0 / 16.0,
        causal=True, window_size=(512, 0),
    )

    expected = torch.empty_like(actual)
    begin = 0
    for length in lengths:
        q_seq = q[begin : begin + length].float().transpose(0, 1)
        k_seq = k[begin : begin + length].float().transpose(0, 1)
        v_seq = v[begin : begin + length].float().transpose(0, 1)
        scores = torch.matmul(q_seq, k_seq.transpose(-1, -2)) / 16.0
        positions = torch.arange(length, device=device)
        mask = (positions[None, :] <= positions[:, None]) & (
            positions[None, :] >= positions[:, None] - 512
        )
        scores.masked_fill_(~mask, float("-inf"))
        expected[begin : begin + length] = (
            torch.matmul(torch.softmax(scores, dim=-1), v_seq)
            .transpose(0, 1)
            .to(torch.bfloat16)
        )
        begin += length

    torch.cuda.synchronize()
    assert actual.shape == (total, heads, 128)
    assert bool(torch.isfinite(actual).all().item())
    assert bool(torch.isfinite(actual_lse).all().item())
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
