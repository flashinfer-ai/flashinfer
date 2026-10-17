"""Mamba recurrence, packed state ownership, and frozen graph replay."""

from contextlib import contextmanager

import pytest
import torch

from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import mamba_ssd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def inputs(
    lengths,
    *,
    heads=8,
    groups=2,
    dim=64,
    dstate=128,
    state_dtype=torch.bfloat16,
    parameter_dtype=torch.float32,
    padding=0,
):
    torch.manual_seed(13)
    tokens, seqs = sum(lengths), len(lengths)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    def packed(width, stride):
        return rand(tokens, stride)[:, :width]

    x = packed(heads * dim, heads * dim + padding).view(tokens, heads, dim)
    B = packed(groups * dstate, groups * dstate + padding).view(tokens, groups, dstate)
    C = packed(groups * dstate, groups * dstate + padding).view(tokens, groups, dstate)
    dt = rand(tokens, heads, dtype=parameter_dtype)
    A = -torch.rand(heads, device="cuda") - 1
    D, bias = (
        rand(heads, dtype=parameter_dtype),
        (rand(heads, dtype=parameter_dtype) - 4),
    )
    state = rand(2 * seqs + 1, heads * dim * dstate + padding, dtype=state_dtype)
    state = state[:, : heads * dim * dstate].view(2 * seqs + 1, heads, dim, dstate)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    src = torch.arange(1, 1 + seqs, device="cuda", dtype=torch.int64)
    dst = src + seqs
    fresh = torch.zeros(seqs, device="cuda", dtype=torch.bool)
    ns = torch.tensor([seqs], device="cuda", dtype=torch.int32)
    nt = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    return (
        x,
        dt,
        A,
        B,
        C,
        D,
        bias,
        state,
        cu,
        src,
        dst,
        fresh,
        ns,
        nt,
    ), torch.full_like(x, -31)


def reference(args, out):
    x, dt, A, B, C, D, bias, state, cu, src, dst, fresh, ns, nt = args
    result, states = out.clone(), state.clone()
    groups = torch.arange(x.shape[1], device=x.device) // (x.shape[1] // B.shape[1])
    for request in range(int(ns)):
        lo, hi = int(cu[request]), int(cu[request + 1])
        if int(src[request]) < 0 or int(dst[request]) < 0 or hi == lo:
            continue
        acc = (
            torch.zeros_like(state[0], dtype=torch.float32)
            if fresh[request]
            else state[src[request]].float()
        )
        for t in range(lo, hi):
            step = torch.nn.functional.softplus(dt[t].float() + bias.float())
            decay = torch.exp(A * step)
            acc = (
                acc * decay[:, None, None]
                + x[t].float()[:, :, None]
                * (step[:, None] * B[t, groups].float())[:, None, :]
            )
            result[t] = (acc * C[t, groups].float()[:, None, :]).sum(-1) + D.float()[
                :, None
            ] * x[t].float()
        states[dst[request]] = acc
    return result, states


@contextmanager
def prepared(args, out, override=None):
    plan = mamba_ssd.plan(
        mamba_ssd.query_from_call(*args, out=out), device=out.device, override=override
    )
    workspace = torch.empty(
        plan.memory_requirements().scratch_nbytes, dtype=torch.uint8, device=out.device
    )
    with PreparationSession(
        device=out.device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="mamba",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(*args, out=out, workspace=workspace)
                    ),
                ),
            )
        )
        session.freeze()
        yield plan, workspace


def assert_numerics(actual, expected):
    assert torch.isfinite(actual).all()
    error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)
    assert error < 0.008, error.item()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.025, atol=0.06)


@pytest.mark.parametrize(
    "lengths", [(1,), (15,), (16,), (17,), (128,), (129,), (0, 7, 33), (1, 127, 0, 257)]
)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrence_and_distinct_state_slots(lengths, state_dtype):
    args, out = inputs(lengths, state_dtype=state_dtype, padding=16)
    expected, states = reference(args, out)
    initial = args[7].clone()
    with prepared(args, out) as (plan, workspace):
        mamba_ssd.run(*args, out=out, plan=plan, workspace=workspace)
        torch.cuda.synchronize()
        assert_numerics(out, expected)
        assert_numerics(args[7], states)
        torch.testing.assert_close(
            args[7][: len(lengths) + 1], initial[: len(lengths) + 1], rtol=0, atol=0
        )


@pytest.mark.parametrize("dim,dstate", [(64, 64), (128, 128)])
@pytest.mark.parametrize("tokens", [35, 257])
def test_geometry_and_bf16_parameters(dim, dstate, tokens):
    args, out = inputs(
        (tokens,), dim=dim, dstate=dstate, parameter_dtype=torch.bfloat16
    )
    expected, states = reference(args, out)
    with prepared(args, out) as (plan, workspace):
        mamba_ssd.run(*args, out=out, plan=plan, workspace=workspace)
        assert_numerics(out, expected)
        assert_numerics(args[7], states)


@pytest.mark.parametrize(
    "warps,elements,rows,d,n",
    [
        (4, 4, 64, 64, 128),
        (8, 16, 64, 64, 128),
        (16, 8, 64, 64, 128),
        (16, 16, 128, 64, 64),
        (8, 8, 128, 64, 128),
        (16, 8, 128, 128, 128),
    ],
)
def test_launch_configurations(warps, elements, rows, d, n):
    args, out = inputs((7, 129, 0, 257), dim=d, dstate=n)
    expected, states = reference(args, out)
    with prepared(args, out, mamba_ssd.MambaConfig(warps, elements, rows)) as (
        plan,
        workspace,
    ):
        mamba_ssd.run(*args, out=out, plan=plan, workspace=workspace)
        assert_numerics(out, expected)
        assert_numerics(args[7], states)


@pytest.mark.parametrize("capacity", [33, 4096])
def test_graph_live_counts_fresh_slots_and_poisoned_padding(capacity):
    args, out = inputs((17, capacity - 17, 0))
    state, cu, src, dst, fresh, ns, nt = args[7:]
    initial = state.clone()
    with prepared(args, out) as (plan, workspace):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mamba_ssd.run(*args, out=out, plan=plan, workspace=workspace)
        address = out.data_ptr()
        for lengths in ((17, capacity - 17), (1,), (), (7, 3, 0)):
            n, t = len(lengths), sum(lengths)
            state.copy_(initial)
            fresh.fill_(True)
            state[src] = float("nan")
            cu.fill_(-123)
            cu[: n + 1].copy_(
                torch.tensor(
                    [0, *torch.tensor(lengths, dtype=torch.int32).cumsum(0).tolist()],
                    device="cuda",
                    dtype=torch.int32,
                )
            )
            ns.fill_(n)
            nt.fill_(t)
            out.fill_(-31)
            expected, states = reference(args, out)
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before and out.data_ptr() == address
            assert_numerics(out, expected)
            torch.testing.assert_close(out[t:], expected[t:], rtol=0, atol=0)
            for r in range(n):
                if lengths[r]:
                    assert_numerics(state[dst[r]], states[dst[r]])
        graph.reset()
