# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GPU tests of ``chunk_gated_delta_rule(backend="cake_gdn", use_affine_split=True)``.

The split route is held to an FP64 per-token recurrence: its relative RMS
error on the output and the final state may exceed the unsplit Cake kernel's
by at most 10 % on realistic gates and 20 % on near-unit gates.  A call whose
longest sequence gets one segment is bitwise equal to the unsplit kernel.
Graph capture, replay with changed lengths, plan reuse and streams are
checked against eager calls.  SM103 only.
"""

from __future__ import annotations

import math

import pytest
import torch

from flashinfer import gdn_prefill as prefill_mod
from flashinfer.gdn_prefill import chunk_gated_delta_rule

D = 128
SCALE = 1.0 / math.sqrt(D)
HEADS = [(2, 4), (4, 8), (8, 16), (16, 32)]
GATE = {"real": 1.10, "stress": 1.20}


@pytest.fixture(autouse=True)
def _require_sm103():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("the Cake GDN affine-split route ships for SM103 only")
    if prefill_mod._chunk_gated_delta_rule_gdn_affine_split_sm100 is None:
        pytest.skip("the Cake GDN affine-split backend is not installed")


def _split_module():
    from flashinfer.gdn_kernels.blackwell import cake_gdn_affine_split

    return cake_gdn_affine_split


def _sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _problem(hq, hv, t, kind="real", seed=0):
    """Q/K L2-normalized; ``real``: Mamba-style decays, ``stress``: decays in
    [0.999, 1] with beta near 1 and a unit-scale initial state."""

    gen = torch.Generator(device="cuda").manual_seed(
        1234
        + 1000003 * seed
        + 7919 * hq
        + 104729 * hv
        + t
        + (17 if kind == "stress" else 0)
    )

    def rn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.float32, generator=gen)

    q = torch.nn.functional.normalize(rn(t, hq, D), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(rn(t, hq, D), dim=-1).to(torch.bfloat16)
    v = rn(t, hv, D).to(torch.bfloat16)
    if kind == "real":
        a = torch.empty(hv, device="cuda").uniform_(1.0, 16.0, generator=gen)
        dt = torch.exp(
            torch.empty(hv, device="cuda").uniform_(
                math.log(1e-3), math.log(1e-1), generator=gen
            )
        )
        dt_bias = dt + torch.log(-torch.expm1(-dt))
        g = torch.exp(-a * torch.nn.functional.softplus(rn(t, hv) + dt_bias))
        beta = torch.sigmoid(rn(t, hv))
        h0 = 0.05 * rn(1, hv, D, D)
    else:
        g = 1.0 - 0.001 * torch.rand(t, hv, device="cuda", generator=gen)
        beta = (0.9 + 0.1 * torch.rand(t, hv, device="cuda", generator=gen)).clamp_(
            max=0.99999
        )
        h0 = rn(1, hv, D, D)
    return dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        h0=h0.contiguous(),
        hq=hq,
        hv=hv,
        t=t,
    )


def _fp64_reference(pr, start=0, end=None):
    """Per-token FP64 recurrence over the tokens [start, end); state [H, V, K]."""

    end = pr["t"] if end is None else end
    hv, hq = pr["hv"], pr["hq"]
    rep = hv // hq
    q = pr["q"][start:end].double().repeat_interleave(rep, dim=1)
    k = pr["k"][start:end].double().repeat_interleave(rep, dim=1)
    v = pr["v"][start:end].double()
    g = pr["g"][start:end].double()
    beta = pr["beta"][start:end].double()
    state = pr["h0"][0].double().transpose(-1, -2).contiguous()  # [H, K, V]
    out = torch.empty(end - start, hv, D, dtype=torch.float64, device="cuda")
    for i in range(end - start):
        state = g[i].view(-1, 1, 1) * state
        old_v = torch.einsum("hk,hkv->hv", k[i], state)
        delta = beta[i].view(-1, 1) * (v[i] - old_v)
        state = state + k[i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[i] = SCALE * torch.einsum("hk,hkv->hv", q[i], state)
    return out, state.transpose(-1, -2).unsqueeze(0)


def _call(pr, *, cu=None, output=None, output_state=None, **kwargs):
    t, hv = pr["t"], pr["hv"]
    if cu is None:
        cu = torch.tensor([0, t], dtype=torch.int32, device="cuda")
    if output is None:
        output = torch.empty(t, hv, D, dtype=torch.bfloat16, device="cuda")
    if output_state is None:
        output_state = torch.empty(1, hv, D, D, dtype=torch.float32, device="cuda")
    kwargs.setdefault("initial_state", pr["h0"])
    return chunk_gated_delta_rule(
        pr["q"],
        pr["k"],
        pr["v"],
        g=pr["g"],
        beta=pr["beta"],
        scale=SCALE,
        output_final_state=True,
        cu_seqlens=cu,
        output=output,
        output_state=output_state,
        backend="cake_gdn",
        **kwargs,
    )


def _rel_rms(x, ref):
    x, ref = x.double(), ref.double()
    return float(torch.linalg.vector_norm(x - ref) / torch.linalg.vector_norm(ref))


def _p_bound(hv, t, split_count=None):
    return (
        _split_module()
        .plan_split(
            num_heads=hv, max_seqlen=t, sm_count=_sm_count(), split_count=split_count
        )
        .p_bound
    )


def _zero_words():
    split = _split_module()
    entries = list(split._EAGER.values()) + list(split._PINNED.values())
    return all(
        int(e.flags.count_nonzero()) == 0 and int(e.ticket.count_nonzero()) == 0
        for e in entries
    )


# --- accuracy against FP64 and the unsplit kernel ------------------------------


def _accuracy_case(hq, hv, t, kind):
    pr = _problem(hq, hv, t, kind)
    ref_o, ref_s = _fp64_reference(pr)
    o_split, s_split = _call(pr, use_affine_split=True)
    o_base, s_base = _call(pr, use_cp=False)
    torch.cuda.synchronize()
    for name, x, base, ref in (
        ("output", o_split, o_base, ref_o),
        ("state", s_split, s_base, ref_s),
    ):
        assert torch.isfinite(x).all(), name
        err, err_base = _rel_rms(x, ref), _rel_rms(base, ref)
        assert err <= GATE[kind] * err_base, (name, err, err_base)
        assert err < 1e-2, (name, err)
    return pr, o_split, s_split, o_base, s_base


@pytest.mark.parametrize("hq,hv", HEADS)
@pytest.mark.parametrize("t", [1024, 1152, 2048, 4096])
@pytest.mark.parametrize("kind", ["real", "stress"])
def test_split_accuracy_against_fp64(hq, hv, t, kind):
    _accuracy_case(hq, hv, t, kind)


@pytest.mark.long_running
@pytest.mark.parametrize("hq,hv", HEADS)
@pytest.mark.parametrize("t", [3072, 8192])
@pytest.mark.parametrize("kind", ["real", "stress"])
def test_split_accuracy_against_fp64_long(hq, hv, t, kind):
    _accuracy_case(hq, hv, t, kind)


@pytest.mark.parametrize("hq,hv", [(2, 4), (4, 8)])
@pytest.mark.parametrize("split_count", [2, 3, 8])
def test_forced_split_count_accuracy(hq, hv, split_count):
    pr = _problem(hq, hv, 1000, "real")  # a ragged tail chunk
    ref_o, ref_s = _fp64_reference(pr)
    o_split, s_split = _call(pr, use_affine_split=True, _affine_split_count=split_count)
    o_base, s_base = _call(pr, use_cp=False)
    for x, base, ref in ((o_split, o_base, ref_o), (s_split, s_base, ref_s)):
        assert _rel_rms(x, ref) <= GATE["real"] * _rel_rms(base, ref)


def test_split_is_run_to_run_deterministic():
    pr = _problem(4, 8, 4096, "real")
    o1, s1 = _call(pr, use_affine_split=True)
    o1, s1 = o1.clone(), s1.clone()
    o2, s2 = _call(pr, use_affine_split=True)
    assert torch.equal(o1, o2) and torch.equal(s1, s2)


# --- one segment: the unsplit kernel, bitwise ----------------------------------


@pytest.mark.parametrize(
    "hq,hv,t,split_count",
    [
        (2, 4, 512, None),
        (2, 4, 1024, None),
        (4, 8, 384, None),
        (4, 8, 1024, None),
        (8, 16, 1024, None),
        (16, 32, 4096, None),
        (4, 8, 4096, 1),
        (2, 4, 2048, 1),
    ],
)
def test_one_segment_is_bitwise_the_unsplit_kernel(hq, hv, t, split_count):
    assert _p_bound(hv, t, split_count) == 1
    pr = _problem(hq, hv, t, "real")
    o_split, s_split = _call(pr, use_affine_split=True, _affine_split_count=split_count)
    o_base, s_base = _call(pr, use_cp=False)
    assert torch.equal(o_split, o_base)
    assert torch.equal(s_split, s_base)


# --- kernel routes ------------------------------------------------------------


def _kernel_names(fn):
    from tests.cake.test_helpers.cake_profiler import cuda_kernel_names

    return cuda_kernel_names(fn)


def test_split_launches_pass1_then_pass2_only():
    pr = _problem(4, 8, 4096, "real")
    assert _p_bound(8, 4096) > 1
    _call(pr, use_affine_split=True)  # build and warm
    names = _kernel_names(lambda: _call(pr, use_affine_split=True))
    assert len(names) == 2, names
    assert all("cake_gdn_affine_split" in name for name in names), names


@pytest.mark.parametrize("use_cp", ["auto", False])
def test_default_false_keeps_the_route_and_the_bits(use_cp):
    pr = _problem(4, 8, 4096, "real")
    o_default, s_default = _call(pr, use_cp=use_cp)
    o_default, s_default = o_default.clone(), s_default.clone()
    o_false, s_false = _call(pr, use_cp=use_cp, use_affine_split=False)
    assert torch.equal(o_default, o_false) and torch.equal(s_default, s_false)
    default_names = _kernel_names(lambda: _call(pr, use_cp=use_cp))
    false_names = _kernel_names(
        lambda: _call(pr, use_cp=use_cp, use_affine_split=False)
    )
    assert default_names == false_names
    assert not any("affine_split" in name for name in default_names)


# --- variable lengths within one sequence -------------------------------------


@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("start,end", [(0, 4096), (0, 2500), (640, 4096), (0, 0)])
def test_live_bounds_and_untouched_rows(cu_dtype, start, end):
    pr = _problem(4, 8, 4096, "real")
    cu = torch.tensor([start, end], dtype=cu_dtype, device="cuda")
    output = torch.full((4096, 8, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    state = torch.full((1, 8, D, D), float("nan"), dtype=torch.float32, device="cuda")
    _call(pr, cu=cu, output=output, output_state=state, use_affine_split=True)
    torch.cuda.synchronize()
    assert torch.isnan(output[:start]).all() and torch.isnan(output[end:]).all()
    if end == start:
        assert torch.equal(state, pr["h0"])
        return
    ref_o, ref_s = _fp64_reference(pr, start, end)
    sub = {n: pr[n][start:end].contiguous() for n in ("q", "k", "v", "g", "beta")}
    base = dict(pr, **sub, t=end - start)
    o_base, s_base = _call(base, use_cp=False)
    assert _rel_rms(output[start:end], ref_o) <= GATE["real"] * _rel_rms(o_base, ref_o)
    assert _rel_rms(state, ref_s) <= GATE["real"] * _rel_rms(s_base, ref_s)


def test_no_initial_state_unit_gates_and_no_final_state():
    pr = _problem(4, 8, 2048, "real")
    pr0 = dict(pr, h0=torch.zeros_like(pr["h0"]))
    ones = torch.ones_like(pr["g"])
    for g, beta in ((None, pr["beta"]), (pr["g"], None), (None, None)):
        case = dict(
            pr0, g=ones if g is None else g, beta=ones if beta is None else beta
        )
        ref_o, ref_s = _fp64_reference(case)
        out = torch.empty(2048, 8, D, dtype=torch.bfloat16, device="cuda")
        o, s = chunk_gated_delta_rule(
            pr["q"],
            pr["k"],
            pr["v"],
            g=g,
            beta=beta,
            scale=SCALE,
            output_final_state=True,
            cu_seqlens=torch.tensor([0, 2048], dtype=torch.int32, device="cuda"),
            output=out,
            backend="cake_gdn",
            use_affine_split=True,
        )
        # The unsplit kernel on the materialized equivalents (zero state, unit gates).
        o_base, s_base = _call(case, use_cp=False)
        assert _rel_rms(o, ref_o) <= GATE["real"] * _rel_rms(o_base, ref_o)
        assert _rel_rms(s, ref_s) <= GATE["real"] * _rel_rms(s_base, ref_s)
        only = chunk_gated_delta_rule(
            pr["q"],
            pr["k"],
            pr["v"],
            g=g,
            beta=beta,
            scale=SCALE,
            output_final_state=False,
            cu_seqlens=torch.tensor([0, 2048], dtype=torch.int32, device="cuda"),
            backend="cake_gdn",
            use_affine_split=True,
        )
        assert torch.equal(only, o)


def test_in_place_state_and_qk_l2norm():
    pr = _problem(4, 8, 2048, "real")
    state = pr["h0"].clone()
    o_inplace, s_inplace = _call(
        pr, initial_state=state, output_state=state, use_affine_split=True
    )
    o_ref, s_ref = _call(pr, use_affine_split=True)
    assert s_inplace.data_ptr() == state.data_ptr()
    assert torch.equal(o_inplace, o_ref) and torch.equal(s_inplace, s_ref)
    raw = dict(
        pr,
        q=(pr["q"].float() * 3).to(torch.bfloat16),
        k=(pr["k"].float() * 2).to(torch.bfloat16),
    )
    o_norm, s_norm = _call(raw, use_affine_split=True, use_qk_l2norm_in_kernel=True)
    o_base, s_base = _call(raw, use_cp=False, use_qk_l2norm_in_kernel=True)
    ref_o, ref_s = _fp64_reference(
        dict(
            raw,
            q=torch.nn.functional.normalize(raw["q"].float(), dim=-1).to(
                torch.bfloat16
            ),
            k=torch.nn.functional.normalize(raw["k"].float(), dim=-1).to(
                torch.bfloat16
            ),
        )
    )
    assert _rel_rms(o_norm, ref_o) <= GATE["real"] * _rel_rms(o_base, ref_o) + 1e-3
    assert _rel_rms(s_norm, ref_s) <= GATE["real"] * _rel_rms(s_base, ref_s) + 1e-3


# --- plans, streams and CUDA graphs -------------------------------------------


def test_plan_entry_is_reused_and_words_return_to_zero():
    split = _split_module()
    split.clear_affine_split_plans()
    pr = _problem(2, 4, 4096, "real")
    _call(pr, use_affine_split=True)
    (entry,) = split._EAGER.values()
    for _ in range(100):
        _call(pr, use_affine_split=True)
    torch.cuda.synchronize()
    assert list(split._EAGER.values()) == [entry]
    assert _zero_words()


def test_capture_requires_a_warm_plan():
    split = _split_module()
    split.clear_affine_split_plans()
    pr = _problem(4, 8, 4096, "real")
    cu = torch.tensor([0, 4096], dtype=torch.int32, device="cuda")
    out = torch.empty(4096, 8, D, dtype=torch.bfloat16, device="cuda")
    state = torch.empty(1, 8, D, D, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="must be warmed before CUDA graph capture"),
        torch.cuda.graph(graph, stream=stream),
    ):
        _call(pr, cu=cu, output=out, output_state=state, use_affine_split=True)
    assert not split._EAGER and not split._PINNED


@pytest.mark.parametrize("hq,hv", [(2, 4), (4, 8)])
def test_graph_replay_with_new_contents_and_lengths(hq, hv):
    split = _split_module()
    split.clear_affine_split_plans()
    t = 4096
    static = _problem(hq, hv, t, "real", seed=0)
    cu = torch.tensor([0, t], dtype=torch.int32, device="cuda")
    out = torch.empty(t, hv, D, dtype=torch.bfloat16, device="cuda")
    state = torch.empty(1, hv, D, D, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(static, cu=cu, output=out, output_state=state, use_affine_split=True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _call(static, cu=cu, output=out, output_state=state, use_affine_split=True)
    assert len(split._PINNED) == 1 and not split._EAGER
    # Lengths that move the segment count from the cap down to 1, an offset
    # start and an empty sequence.
    bounds = [
        (0, t),
        (0, 3000),
        (0, 1500),
        (0, 700),
        (0, 64),
        (0, 1),
        (512, t),
        (100, 100),
    ]
    for replay in range(100):
        fresh = _problem(hq, hv, t, "real", seed=1 + replay % 5)
        a, b = bounds[replay % len(bounds)]
        for name in ("q", "k", "v", "g", "beta", "h0"):
            static[name].copy_(fresh[name])
        cu.copy_(torch.tensor([a, b], dtype=torch.int32))
        out.fill_(float("nan"))
        state.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _zero_words(), replay
        o_eager = torch.full_like(out, float("nan"))
        s_eager = torch.full_like(state, float("nan"))
        _call(
            fresh,
            cu=cu.clone(),
            output=o_eager,
            output_state=s_eager,
            use_affine_split=True,
        )
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), o_eager.view(torch.int16)), (
            replay,
            a,
            b,
        )
        assert torch.equal(state.view(torch.int32), s_eager.view(torch.int32)), (
            replay,
            a,
            b,
        )
    # Eager traffic over more geometries than the LRU holds, then replay on
    # another stream: the pinned workspaces stay alive.
    for tt in (1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664):
        _call(_problem(hq, hv, tt, "real"), use_affine_split=True)
    assert len(split._EAGER) == 8
    cu.copy_(torch.tensor([0, t], dtype=torch.int32))
    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        graph.replay()
    other.synchronize()
    o_eager, s_eager = _call(static, use_affine_split=True)
    assert torch.equal(out, o_eager) and torch.equal(state, s_eager)
    assert _zero_words()
    split.clear_affine_split_plans()


def test_two_streams_concurrently_in_eager():
    split = _split_module()
    split.clear_affine_split_plans()
    problems = [_problem(4, 8, 4096, "real", seed=s) for s in (1, 2)]
    expected = [
        tuple(x.clone() for x in _call(pr, use_affine_split=True)) for pr in problems
    ]
    streams = [torch.cuda.Stream() for _ in problems]
    results = []
    torch.cuda.synchronize()
    for _ in range(10):
        for pr, stream in zip(problems, streams, strict=True):
            with torch.cuda.stream(stream):
                results.append(_call(pr, use_affine_split=True))
    torch.cuda.synchronize()
    for i, (o, s) in enumerate(results[-2:]):
        assert torch.equal(o, expected[i][0]) and torch.equal(s, expected[i][1])
    assert _zero_words()
    split.clear_affine_split_plans()


def test_one_capture_repeats_a_call_of_one_plan():
    """Two same-shape calls in one capture (two layers) share the capture's
    entry; a second capture of the same shapes owns another one."""

    split = _split_module()
    split.clear_affine_split_plans()
    t, hv = 4096, 8
    layers = [_problem(4, hv, t, "real", seed=s) for s in (1, 2)]
    cu = torch.tensor([0, t], dtype=torch.int32, device="cuda")
    outs = [torch.empty(t, hv, D, dtype=torch.bfloat16, device="cuda") for _ in layers]
    states = [
        torch.empty(1, hv, D, D, dtype=torch.float32, device="cuda") for _ in layers
    ]

    def run_layers():
        for pr, o, st in zip(layers, outs, states, strict=True):
            _call(pr, cu=cu, output=o, output_state=st, use_affine_split=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_layers()
    stream.synchronize()
    graphs = []
    for _ in range(2):
        with torch.cuda.stream(stream):
            _call(layers[0], cu=cu, use_affine_split=True)  # warm a fresh entry
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run_layers()
        graphs.append(graph)
        assert len(split._PINNED) == len(graphs) and not split._EAGER
    for graph in graphs:
        for o, st in zip(outs, states, strict=True):
            o.fill_(float("nan"))
            st.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _zero_words()
        for pr, o, st in zip(layers, outs, states, strict=True):
            o_eager, s_eager = _call(pr, use_affine_split=True)
            assert torch.equal(o.view(torch.int16), o_eager.view(torch.int16))
            assert torch.equal(st.view(torch.int32), s_eager.view(torch.int32))
    split.clear_affine_split_plans()


# --- rejections on the device -------------------------------------------------


def _unaligned_state():
    return torch.zeros(8 * D * D + 1, device="cuda")[1:].view(1, 8, D, D)


@pytest.mark.parametrize(
    "change,reason",
    [
        (dict(heads=6), "requires Hv in (4, 8, 16, 32), Hk == Hq and Hv % Hq == 0"),
        (dict(cu=[0, 1000, 2048]), "supports one sequence (num_seqs == 1)"),
        (
            dict(cu=[0, 7, 2048, 9], cu_step=2),
            "requires contiguous int32 or int64 cu_seqlens",
        ),
        (dict(h0="bf16"), "requires FP32 states with contiguous [H, V, K] rows"),
        (
            dict(h0="unaligned"),
            "requires 16-byte aligned state pointers and slot strides",
        ),
        (dict(g="strided"), "requires contiguous FP32 g/beta [tokens, heads]"),
        (
            dict(q="f16"),
            "requires contiguous BF16 q/k/v/output of head size 128 on one device",
        ),
    ],
)
def test_out_of_scope_calls_raise_before_any_launch(change, reason):
    import re

    hv = change.get("heads", 8)
    pr = _problem(4, 8, 2048, "real")
    q, k = pr["q"], pr["k"]
    v = (
        pr["v"]
        if hv == 8
        else torch.randn(2048, hv, D, device="cuda").to(torch.bfloat16)
    )
    g, beta = (
        pr["g"][:, :1].expand(2048, hv).contiguous(),
        pr["beta"][:, :1].expand(2048, hv).contiguous(),
    )
    h0 = torch.zeros(1, hv, D, D, device="cuda")
    if change.get("h0") == "bf16":
        h0 = h0.to(torch.bfloat16)
    elif change.get("h0") == "unaligned":
        h0 = _unaligned_state()
    if change.get("g") == "strided":
        g = torch.zeros(hv, 2048, device="cuda").t()
    if change.get("q") == "f16":
        q, k, v = q.half(), k.half(), v.half()
    cu = torch.tensor(change.get("cu", [0, 2048]), dtype=torch.int32, device="cuda")
    cu = cu[:: change.get("cu_step", 1)]
    output = torch.full(
        (2048, max(4, hv), D), float("nan"), dtype=q.dtype, device="cuda"
    )
    with pytest.raises(ValueError, match=re.escape(f"use_affine_split=True: {reason}")):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=SCALE,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            output=output,
            backend="cake_gdn",
            use_affine_split=True,
        )
    torch.cuda.synchronize()
    assert torch.isnan(output).all()


@pytest.mark.parametrize(
    "cu_cpu,match",
    [
        ([0, 1], "spanning all input tokens"),
        ([0, 2048, 4096], "cu_seqlens_cpu must have num_seqs \\+ 1 = 2 entries"),
    ],
)
def test_host_offsets_are_checked_on_the_split_route(cu_cpu, match):
    pr = _problem(4, 8, 4096, "real")
    assert _p_bound(8, 4096) > 1
    output = torch.full((4096, 8, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match=match):
        _call(pr, output=output, cu_seqlens_cpu=cu_cpu, use_affine_split=True)
    torch.cuda.synchronize()
    assert torch.isnan(output).all()
    o_cpu, s_cpu = _call(pr, cu_seqlens_cpu=[0, 4096], use_affine_split=True)
    o_cpu, s_cpu = o_cpu.clone(), s_cpu.clone()
    o_dev, s_dev = _call(pr, use_affine_split=True)
    assert torch.equal(o_cpu, o_dev) and torch.equal(s_cpu, s_dev)


def test_scope_is_checked_before_the_gates_are_materialized():
    """A log-domain gate is materialized by a launch; an out-of-scope state
    must be rejected before it."""

    pr = _problem(4, 8, 2048, "real")
    log_g = torch.log(pr["g"].clamp_min(1e-30))
    h0 = torch.zeros(1, 8, D, D, dtype=torch.bfloat16, device="cuda")

    def bad_call():
        with pytest.raises(
            ValueError,
            match=r"use_affine_split=True: requires FP32 states with contiguous",
        ):
            chunk_gated_delta_rule(
                pr["q"],
                pr["k"],
                pr["v"],
                g=log_g,
                beta=pr["beta"],
                scale=SCALE,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=torch.tensor([0, 2048], dtype=torch.int32, device="cuda"),
                backend="cake_gdn",
                gate_domain="log",
                use_affine_split=True,
            )

    assert _kernel_names(bad_call) == []
