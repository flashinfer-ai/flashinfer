"""Inductor graph coverage for the packed MUSA SSD Triton pipeline."""

import pytest
import torch

from flashinfer.mamba.ssd_combined import ssd_combined_fwd_varlen


def _metadata(device):
    return (
        torch.tensor([0, 4, 8], device=device, dtype=torch.int32),
        torch.tensor([0, 4, 8], device=device, dtype=torch.int32),
        torch.tensor([0, 1], device=device, dtype=torch.int32),
        torch.tensor([0, 1], device=device, dtype=torch.int32),
    )


def _inputs(seed):
    torch.manual_seed(seed)
    device = torch.device("musa")
    heads, dim, dstate, groups = 4, 8, 16, 2
    x = torch.randn(8, heads, dim, device=device, dtype=torch.bfloat16)
    dt = torch.randn(8, heads, device=device, dtype=torch.float32) * 0.1
    A = -torch.rand(heads, device=device, dtype=torch.float32) - 1
    B = torch.randn(8, groups, dstate, device=device, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    D = torch.randn(heads, dim, device=device, dtype=torch.float32)
    z = torch.randn_like(x)
    dt_bias = torch.randn(heads, device=device, dtype=torch.float32)
    cu, cu_chunks, last, sequence_ids = _metadata(device)
    out = torch.empty_like(x)
    initial = torch.randn(2, heads, dim, dstate, device=device, dtype=torch.float16)
    return x, dt, A, B, C, D, z, dt_bias, cu, cu_chunks, last, sequence_ids, out, initial


def _step(
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    dt_bias,
    cu,
    cu_chunks,
    last,
    sequence_ids,
    out,
    initial,
):
    return ssd_combined_fwd_varlen(
        x,
        dt,
        A,
        B,
        C,
        4,
        cu,
        cu_chunks,
        last,
        sequence_ids,
        out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial,
        state_dtype=initial.dtype,
    )


@pytest.mark.skipif(
    not hasattr(torch.version, "musa") or torch.version.musa is None,
    reason="MUSA Inductor test",
)
def test_packed_ssd_inductor_fullgraph():
    eager_inputs = _inputs(2301)
    eager_args = tuple(t.clone() if torch.is_tensor(t) else t for t in eager_inputs)
    eager = _step(*eager_args)
    eager_out = eager_args[-2].clone()

    compiled = torch.compile(_step, backend="inductor", fullgraph=True, dynamic=False)
    inputs = _inputs(2301)
    actual = compiled(*inputs)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, eager, rtol=0.05, atol=0.05)
    torch.testing.assert_close(inputs[-2], eager_out, rtol=0.05, atol=0.05)

    replay_inputs = _inputs(2302)
    replay_eager_args = tuple(
        t.clone() if torch.is_tensor(t) else t for t in replay_inputs
    )
    replay_eager = _step(*replay_eager_args)
    replay_eager_out = replay_eager_args[-2].clone()
    replay_actual = compiled(*replay_inputs)
    torch.musa.synchronize()
    torch.testing.assert_close(replay_actual, replay_eager, rtol=0.05, atol=0.05)
    torch.testing.assert_close(replay_inputs[-2], replay_eager_out, rtol=0.05, atol=0.05)

    explanation = torch._dynamo.explain(_step)(*_inputs(2303))
    assert explanation.graph_break_count == 0
