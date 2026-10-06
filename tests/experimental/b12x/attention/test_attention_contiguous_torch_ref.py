from __future__ import annotations

import math

import pytest
import torch

from b12x.testing.reference.helpers import require_b12x


def _device() -> torch.device:
    device = require_b12x()
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    return device


def _reference(q, k, v, *, cu=None, causal=False, window=(-1, -1), sinks=None):
    if cu is None:
        q = q.reshape(-1, *q.shape[-2:])
        k = k.reshape(-1, *k.shape[-2:])
        v = v.reshape(-1, *v.shape[-2:])
        cu = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
    output = torch.empty((q.shape[0], q.shape[1], v.shape[-1]), device=q.device, dtype=q.dtype)
    offsets = cu.tolist()
    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
        q_seq, k_seq, v_seq = q[begin:end].float(), k[begin:end].float(), v[begin:end].float()
        if q_seq.shape[1] != k_seq.shape[1]:
            groups = q_seq.shape[1] // k_seq.shape[1]
            k_seq = k_seq.repeat_interleave(groups, dim=1)
            v_seq = v_seq.repeat_interleave(groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_seq, k_seq) * (q.shape[-1] ** -0.5)
        positions = torch.arange(end - begin, device=q.device)
        keep = torch.ones((end - begin, end - begin), dtype=torch.bool, device=q.device)
        if causal:
            keep &= positions[None, :] <= positions[:, None]
        if window[0] != -1:
            keep &= positions[None, :] >= positions[:, None] - window[0]
        if window[1] != -1:
            keep &= positions[None, :] <= positions[:, None] + window[1]
        scores.masked_fill_(~keep.unsqueeze(0), float("-inf"))
        if sinks is not None:
            scores = torch.cat((scores, sinks.float().view(-1, 1, 1).expand(-1, end - begin, 1)), dim=-1)
            probs = torch.softmax(scores, dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1)
        output[begin:end] = torch.einsum("hqk,khd->qhd", probs, v_seq).to(q.dtype)
    return output


def _prepared_batched(q, k, v, *, causal, window, sinks=None):
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    declaration = varlen.plan_batched(q, k, v, causal=causal, window_size=window, attention_sink_bias=sinks)

    def prepare_call(state):
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(scratch=scratch, q=q, k=k, v=v, attention_sink_bias=sinks)
        return PreparedCall(run=lambda: state.run(binding))

    with PreparationSession(device=q.device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="batched", prepare_call=prepare_call),))
        state = require_prepared(declaration, "attention.varlen")
        (spec,) = state.scratch_plan.scratch_specs()
        binding = varlen.bind_batched(declaration, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=spec.device), q=q, k=k, v=v, attention_sink_bias=sinks)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output, lse = varlen.run_batched(binding)
        output.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(q.device)
        return output, lse


def _prepared_varlen(q, k, v, cu, *, max_len, causal, window, sinks=None):
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    declaration = varlen.plan(q, k, v, cu, max_seqlen_q=max_len, max_seqlen_k=max_len, causal=causal, window_size=window, attention_sink_bias=sinks)

    def prepare_call(state):
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu, max_seqlen_q=max_len, max_seqlen_k=max_len, causal=causal, window_size=window, attention_sink_bias=sinks)
        return PreparedCall(run=lambda: state.run(binding))

    with PreparationSession(device=q.device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="varlen", prepare_call=prepare_call),))
        state = require_prepared(declaration, "attention.varlen")
        (spec,) = state.scratch_plan.scratch_specs()
        binding = varlen.bind(declaration, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=spec.device), q=q, k=k, v=v, cu_seqlens_q=cu, max_seqlen_q=max_len, max_seqlen_k=max_len, causal=causal, window_size=window, attention_sink_bias=sinks)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output, lse = varlen.run(binding)
        output.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(q.device)
        return output, lse


@pytest.mark.parametrize("causal,window", [(True, (-1, 0)), (False, (8, 8))])
def test_prepared_batched_gqa_matches_reference_under_graph(causal, window):
    device = _device()
    torch.manual_seed(23)
    q = torch.randn((1, 48, 4, 64), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 48, 2, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn((1, 48, 2, 64), device=device, dtype=torch.bfloat16)
    actual, lse = _prepared_batched(q, k, v, causal=causal, window=window)
    expected = _reference(q, k, v, causal=causal, window=window).reshape_as(actual)
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_prepared_varlen_gqa_sinks_and_unequal_value_dimension():
    device = _device()
    torch.manual_seed(37)
    lengths = (65, 97)
    total = sum(lengths)
    q = torch.randn((total, 8, 256), device=device, dtype=torch.bfloat16)
    k = torch.randn((total, 2, 256), device=device, dtype=torch.bfloat16)
    v = torch.randn((total, 2, 128), device=device, dtype=torch.bfloat16)
    cu = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), device=device, dtype=torch.int32)
    sinks = torch.linspace(-0.25, 0.5, q.shape[1], device=device, dtype=torch.float32)
    actual, lse = _prepared_varlen(
        q, k, v, cu, max_len=max(lengths), causal=True, window=(64, 0), sinks=sinks,
    )
    expected = _reference(q, k, v, cu=cu, causal=True, window=(64, 0), sinks=sinks)
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_prepared_batched_copies_live_noncontiguous_sink_values():
    device = _device()
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    torch.manual_seed(41)
    q = torch.randn((1, 32, 4, 64), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 32, 2, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn((1, 32, 2, 64), device=device, dtype=torch.bfloat16)
    sink_source = torch.linspace(-0.5, 0.5, 8, device=device, dtype=torch.bfloat16)[::2]
    initial_sink = sink_source.clone()
    declaration = varlen.plan_batched(q, k, v, causal=False, attention_sink_bias=sink_source)

    def prepare_call(state):
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(scratch=scratch, q=q, k=k, v=v, attention_sink_bias=sink_source)
        return PreparedCall(run=lambda: state.run(binding))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(
            name="live-sink", prepare_call=prepare_call,
        ),))
        plan = declaration
        state = require_prepared(plan, "attention.varlen")
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = varlen.bind_batched(
            plan, scratch=scratch, q=q, k=k, v=v, attention_sink_bias=sink_source,
        )
        first = varlen.run_batched(binding)[0].clone()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                second = varlen.run_batched(binding)[0]
            sink_source.add_(0.75)
            graph.replay()
            torch.cuda.synchronize(q.device)
            torch.testing.assert_close(
                first, _reference(q, k, v, sinks=initial_sink).reshape_as(first),
                rtol=3e-2, atol=3e-2,
            )
            torch.testing.assert_close(
                second, _reference(q, k, v, sinks=sink_source).reshape_as(second),
                rtol=3e-2, atol=3e-2,
            )
        finally:
            graph.reset()
