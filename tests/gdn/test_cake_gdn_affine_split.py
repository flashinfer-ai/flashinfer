# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GPU tests of ``chunk_gated_delta_rule(backend="cake_gdn", use_affine_split=True)``.

The split route is held to an FP64 per-token recurrence: its relative RMS
error on the output and the final state may exceed the unsplit Cake kernel's
by at most 10 % on realistic gates and 20 % on near-unit gates, per sequence
of a batch.  A call whose longest sequence gets one segment is bitwise equal
to the unsplit kernel.  A sequence's bits do not depend on the batch around
it.  Graph capture, replay with changed lengths, plan reuse and streams are
checked against eager calls.  SM100, SM103 and SM107.
"""

from __future__ import annotations

import math

import pytest
import torch

from flashinfer import gdn_prefill as prefill_mod
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.jit.cake_gdn import CakeGDNUnsupportedError

D = 128
SCALE = 1.0 / math.sqrt(D)
HEADS = [(2, 4), (4, 8), (8, 16), (16, 32)]
GATE = {"real": 1.10, "stress": 1.20}


_TARGETS = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}


@pytest.fixture(autouse=True)
def _require_target():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() not in _TARGETS:
        pytest.skip("the Cake GDN affine-split route ships for SM100/SM103/SM107")
    if prefill_mod._chunk_gated_delta_rule_gdn_affine_split_sm100 is None:
        pytest.skip("the Cake GDN affine-split backend is not installed")


def _split_module():
    from flashinfer.gdn_kernels.blackwell import cake_gdn_affine_split

    return cake_gdn_affine_split


def _sm_count() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _require_qk_norm():
    """``use_qk_l2norm_in_kernel=True`` needs the Q/K normalization kernel
    (CUDA 13 and a CuTe DSL architecture)."""

    from flashinfer.cute_dsl.availability import is_cute_dsl_arch_supported

    cuda_major = int((torch.version.cuda or "0").split(".")[0])
    if not (
        cuda_major >= 13
        and is_cute_dsl_arch_supported(*torch.cuda.get_device_capability())
    ):
        pytest.skip("use_qk_l2norm_in_kernel needs the Q/K normalization kernel")


def _target() -> str:
    return _TARGETS[torch.cuda.get_device_capability()]


def _active_clusters() -> int:
    """The CTA count up to which the unsplit kernel resolves its DV-split
    variant on this device (``2 * num_seqs * num_v_heads`` at most this)."""

    return prefill_mod._cake_gdn._ARCH_ACTIVE_CLUSTERS[_target()]


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
            num_heads=hv,
            max_seqlen=t,
            sm_count=_sm_count(),
            target=_target(),
            split_count=split_count,
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


@pytest.mark.parametrize("with_h0", [True, False], ids=["h0", "no_h0"])
@pytest.mark.parametrize(
    "hq,hv,t,split_count",
    [
        (2, 4, 512, None),
        (2, 4, 960, None),
        (4, 8, 384, None),
        (4, 8, 960, None),
        (8, 16, 1024, None),
        (8, 16, 1984, None),
        (16, 32, 4096, None),
        (16, 32, 8192, None),
        (4, 8, 4096, 1),
        (2, 4, 2048, 1),
        (8, 16, 1984, 1),
        (16, 32, 8192, 1),
    ],
)
def test_one_segment_is_bitwise_the_unsplit_kernel(hq, hv, t, split_count, with_h0):
    if _p_bound(hv, t, split_count) != 1:
        pytest.skip(f"{hv} heads at {t} tokens take several segments on this device")
    pr = _problem(hq, hv, t, "real")
    h0 = pr["h0"] if with_h0 else None
    o_split, s_split = _call(
        pr, initial_state=h0, use_affine_split=True, _affine_split_count=split_count
    )
    o_split, s_split = o_split.clone(), s_split.clone()
    try:
        o_base, s_base = _call(pr, initial_state=h0, use_cp=False)
    except CakeGDNUnsupportedError:
        # The unsplit route has no kernel without an initial state for these
        # heads on this device; the split route serves the call as from zeros.
        assert not with_h0
        o_base, s_base = _call(
            pr, initial_state=torch.zeros_like(pr["h0"]), use_cp=False
        )
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


def test_one_sequence_of_one_segment_runs_the_unsplit_kernel():
    """One sequence in one segment from an initial state runs the unsplit
    kernel (its DV-split variant); without an initial state it runs the
    first kernel alone."""

    pr = _problem(4, 8, 512, "real")
    assert _p_bound(8, 512) == 1
    _call(pr, use_affine_split=True)
    names = _kernel_names(lambda: _call(pr, use_affine_split=True))
    assert names == _kernel_names(lambda: _call(pr, use_cp=False)), names
    assert not any("affine_split" in name for name in names), names
    _call(pr, initial_state=None, use_affine_split=True)
    names = _kernel_names(lambda: _call(pr, initial_state=None, use_affine_split=True))
    assert len(names) == 1 and "cake_gdn_affine_split" in names[0], names


@pytest.mark.parametrize(
    "hq,hv,n",
    [(8, 16, 5), (8, 16, 6), (8, 16, 7), (16, 32, 2), (16, 32, 3), (16, 32, 4)],
)
def test_batch_of_one_segment_uses_the_unsplit_kernel_only_where_bitwise(hq, hv, n):
    """A batch of one-segment sequences from initial states runs the unsplit
    kernel while it resolves the DV-split variant (2 N Hv at most 148 on SM100,
    160 on SM103 and 212 on SM107), else the first kernel alone; either way
    each sequence holds the bits it has alone."""

    lens = [960] * n  # one segment on every target
    pr = _batch(hq, hv, lens)
    _call_batch(pr, use_affine_split=True, cu_seqlens_cpu=pr["cu"])
    names = _kernel_names(
        lambda: _call_batch(pr, use_affine_split=True, cu_seqlens_cpu=pr["cu"])
    )
    unsplit = 2 * n * hv <= _active_clusters()
    assert any("affine_split" in name for name in names) != unsplit, names
    o, s = _call_batch(pr, use_affine_split=True, cu_seqlens_cpu=pr["cu"])
    o, s = o.clone(), s.clone()
    for i in (0, n - 1):
        o_alone, s_alone = _call(_sequence(pr, i), use_affine_split=True)
        a, b = pr["cu"][i], pr["cu"][i + 1]
        assert torch.equal(o_alone, o[a:b]) and torch.equal(s_alone[0], s[i]), i


@pytest.mark.parametrize("hq,hv", [(1, 4), (1, 32), (2, 32)])
def test_one_segment_batch_of_any_head_group(hq, hv):
    """A one-segment batch from initial states runs for every admitted head
    group: through the unsplit kernel where it ships the record, else
    through the first kernel alone. Each sequence holds the bits it has
    alone, and the unsplit kernel's accuracy where it runs."""

    pr = _batch(hq, hv, [512, 512])
    o, s = (
        x.clone()
        for x in _call_batch(pr, use_affine_split=True, cu_seqlens_cpu=pr["cu"])
    )
    for i in range(2):
        seq = _sequence(pr, i)
        o_alone, s_alone = _call(seq, use_affine_split=True)
        a, b = pr["cu"][i], pr["cu"][i + 1]
        assert torch.equal(o_alone, o[a:b]) and torch.equal(s_alone[0], s[i]), i
        ref_o, ref_s = _fp64_reference(seq, 0, seq["t"])
        try:
            o_base, s_base = _call(seq, use_cp=False)
        except CakeGDNUnsupportedError:
            # No unsplit record for this head group. Observed on B200, GB300
            # and VR200: at most 0.0039 (output) and 0.0032 (state), as for
            # the head groups the unsplit kernel serves.
            assert _rel_rms(o[a:b], ref_o) <= 1e-2 and _rel_rms(s[i], ref_s) <= 1e-2
            continue
        assert _rel_rms(o[a:b], ref_o) <= GATE["real"] * _rel_rms(o_base, ref_o)
        assert _rel_rms(s[i], ref_s) <= GATE["real"] * _rel_rms(s_base, ref_s)


def test_one_segment_from_initial_states_without_final_states():
    """From initial states and without final states, a batch of one-segment
    sequences runs the first kernel alone, as the unsplit kernel ships no
    such variant, with the output of the call that keeps the states."""

    pr = _batch(8, 16, [480, 480])  # one segment on every target, unhinted
    cu = torch.tensor(pr["cu"], dtype=torch.int32, device="cuda")

    def only():
        return chunk_gated_delta_rule(
            pr["q"],
            pr["k"],
            pr["v"],
            g=pr["g"],
            beta=pr["beta"],
            scale=SCALE,
            initial_state=pr["h0"],
            output_final_state=False,
            cu_seqlens=cu,
            backend="cake_gdn",
            use_affine_split=True,
        )

    o = only().clone()
    names = _kernel_names(only)
    assert len(names) == 1 and "cake_gdn_affine_split" in names[0], names
    o_ref, _ = _call_batch(pr, use_affine_split=True)
    assert torch.equal(o, o_ref)


def test_one_sequence_state_of_any_slot_stride():
    """A one-sequence state is contiguous whatever its slot stride (a size-1
    dimension, as after a DLPack round trip), so it runs with the bits of
    the dense state."""

    pr = _problem(4, 8, 2048, "real")
    o_ref, s_ref = (x.clone() for x in _call(pr, use_affine_split=True))
    h0 = pr["h0"].clone().as_strided((1, 8, D, D), (1, D * D, D, 1))
    state = torch.empty(8 * D * D, device="cuda").as_strided(
        (1, 8, D, D), (1, D * D, D, 1)
    )
    assert h0.is_contiguous() and h0.stride(0) == 1
    o, s = _call(pr, initial_state=h0, output_state=state, use_affine_split=True)
    assert torch.equal(o, o_ref) and torch.equal(s, s_ref)


def test_batch_of_empty_sequences_keeps_the_initial_states():
    pr = _batch(4, 8, [0, 0, 0])
    assert pr["t"] == 0
    o, s = _call_batch(pr, use_affine_split=True)
    assert o.shape == (0, 8, D) and torch.equal(s, pr["h0"])
    o, s = _call_batch(pr, initial_state=None, use_affine_split=True)
    assert not s.any()


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


def test_in_place_state():
    pr = _problem(4, 8, 2048, "real")
    state = pr["h0"].clone()
    o_inplace, s_inplace = _call(
        pr, initial_state=state, output_state=state, use_affine_split=True
    )
    o_ref, s_ref = _call(pr, use_affine_split=True)
    assert s_inplace.data_ptr() == state.data_ptr()
    assert torch.equal(o_inplace, o_ref) and torch.equal(s_inplace, s_ref)


def test_qk_l2norm():
    _require_qk_norm()
    pr = _problem(4, 8, 2048, "real")
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
        (
            dict(cu=[0, 1000, 2048]),
            "requires contiguous FP32 states [N, H, V, K]",
        ),
        (
            dict(cu=[0, 1000, 2048], h0="expanded"),
            "requires contiguous FP32 states [N, H, V, K]",
        ),
        (
            dict(cu=[0, 1000, 2048], h0="padded"),
            "requires contiguous FP32 states [N, H, V, K]",
        ),
        (
            dict(cu=[0, 7, 2048, 9], cu_step=2),
            "requires contiguous int32 or int64 cu_seqlens",
        ),
        (dict(h0="bf16"), "requires contiguous FP32 states [N, H, V, K]"),
        (
            dict(h0="unaligned"),
            "requires 16-byte aligned state pointers",
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
    elif change.get("h0") == "expanded":
        h0 = h0.expand(2, hv, D, D)
    elif change.get("h0") == "padded":
        # Aligned sequence slots that do not overlap, 4 floats apart.
        slots = torch.zeros(2, hv * D * D + 4, device="cuda")
        h0 = slots[:, : hv * D * D].view(2, hv, D, D)
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
            match=r"use_affine_split=True: requires contiguous FP32 states",
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


# --- batches of several sequences ----------------------------------------------


def _batch(hq, hv, lens, kind="real", seed=0):
    """One problem per sequence (``_problem``), packed: ``cu`` and the initial
    states ``h0 [N, H, V, K]``."""

    parts = [
        _problem(hq, hv, max(n, 1), kind, seed=seed + 7 * i) for i, n in enumerate(lens)
    ]
    cat = {
        name: torch.cat([p[name][:n] for p, n in zip(parts, lens, strict=True)])
        for name in ("q", "k", "v", "g", "beta")
    }
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    h0 = torch.cat([p["h0"] for p in parts]).contiguous()
    return dict(cat, h0=h0, hq=hq, hv=hv, t=cu[-1], cu=cu)


def _call_batch(pr, *, output=None, output_state=None, **kwargs):
    cu = torch.tensor(pr["cu"], dtype=torch.int32, device="cuda")
    n = len(pr["cu"]) - 1
    if output_state is None:
        output_state = torch.empty(
            n, pr["hv"], D, D, dtype=torch.float32, device="cuda"
        )
    return _call(pr, cu=cu, output=output, output_state=output_state, **kwargs)


def _sequence(pr, n):
    """Sequence ``n`` of a batch as a problem of its own."""

    a, b = pr["cu"][n], pr["cu"][n + 1]
    sub = {name: pr[name][a:b].contiguous() for name in ("q", "k", "v", "g", "beta")}
    return dict(pr, **sub, h0=pr["h0"][n : n + 1].contiguous(), t=b - a, cu=[0, b - a])


BATCHES = [
    pytest.param([4096, 1000, 0, 2500], id="varlen4"),
    pytest.param([2048, 2048], id="2x2048"),
    pytest.param([8192] * 4, id="4x8192", marks=pytest.mark.long_running),
]


@pytest.mark.parametrize("hq,hv", HEADS)
@pytest.mark.parametrize("lens", BATCHES)
@pytest.mark.parametrize("kind", ["real", "stress"])
def test_batch_accuracy_against_fp64(hq, hv, lens, kind):
    """Per sequence, the output rows and final state against the FP64
    recurrence, gated on the unsplit kernel's error over the same batch."""

    pr = _batch(hq, hv, lens, kind)
    o_split, s_split = _call_batch(pr, use_affine_split=True)
    o_split, s_split = o_split.clone(), s_split.clone()
    o_base, s_base = _call_batch(pr, use_cp=False)
    torch.cuda.synchronize()
    for n in range(len(lens)):
        a, b = pr["cu"][n], pr["cu"][n + 1]
        if a == b:
            assert torch.equal(s_split[n], pr["h0"][n])
            continue
        ref_o, ref_s = _fp64_reference(_sequence(pr, n))
        for name, x, base, ref in (
            ("output", o_split[a:b], o_base[a:b], ref_o),
            ("state", s_split[n : n + 1], s_base[n : n + 1], ref_s),
        ):
            assert torch.isfinite(x).all(), (n, name)
            err, err_base = _rel_rms(x, ref), _rel_rms(base, ref)
            assert err <= GATE[kind] * err_base, (n, name, err, err_base)
            assert err < 1e-2, (n, name, err)


INVARIANCE = [
    # (Hq, Hv, lens): at most 960 tokens (one segment), 961 and longer
    # (split), an empty sequence.
    (2, 4, [900, 4096, 512, 0, 960, 961, 1024, 2048]),
    (4, 8, [961, 960, 3000, 1089, 64, 1]),
    (8, 16, [1984, 2048, 300, 4096]),
    (16, 32, [4096, 100, 2000]),
]


@pytest.mark.parametrize("hq,hv,lens", INVARIANCE)
def test_batch_invariance_is_bitwise(hq, hv, lens):
    """Each sequence alone holds the bits it has inside the batch, with and
    without the ``max_seqlen`` / ``cu_seqlens_cpu`` hints and in another order."""

    pr = _batch(hq, hv, lens)
    longest = max(lens)
    runs = {}
    for name, kwargs in (
        ("plain", {}),
        ("max_seqlen", dict(max_seqlen=longest)),
        ("cu_seqlens_cpu", dict(cu_seqlens_cpu=pr["cu"])),
    ):
        o, s = _call_batch(pr, use_affine_split=True, **kwargs)
        runs[name] = (o.clone(), s.clone())
    order = list(reversed(range(len(lens))))
    rev = _batch(hq, hv, [lens[i] for i in order])
    for name in ("q", "k", "v", "g", "beta"):
        rev[name] = torch.cat(
            [pr[name][pr["cu"][i] : pr["cu"][i + 1]] for i in order]
        ).contiguous()
    rev["h0"] = pr["h0"][order].contiguous()
    o_rev, s_rev = _call_batch(rev, use_affine_split=True)
    for n in range(len(lens)):
        a, b = pr["cu"][n], pr["cu"][n + 1]
        r = order.index(n)
        ra, rb = rev["cu"][r], rev["cu"][r + 1]
        o_ref, s_ref = runs["plain"]
        for name, (o, s) in runs.items():
            assert torch.equal(o[a:b], o_ref[a:b]) and torch.equal(s[n], s_ref[n]), (
                name,
                n,
            )
        assert torch.equal(o_rev[ra:rb], o_ref[a:b]), ("reversed", n)
        assert torch.equal(s_rev[r], s_ref[n]), ("reversed", n)
        if a == b:
            continue
        o_alone, s_alone = _call(_sequence(pr, n), use_affine_split=True)
        assert torch.equal(o_alone, o_ref[a:b]), ("alone", n, lens[n])
        assert torch.equal(s_alone[0], s_ref[n]), ("alone", n, lens[n])


@pytest.mark.parametrize("hq,hv", HEADS)
def test_batch_of_one_segment_runs_pass1_alone(hq, hv):
    """A batch whose longest sequence gets one segment, without initial
    states, launches the first kernel alone, at a cap of one segment, over no
    split workspace."""

    split = _split_module()
    split.clear_affine_split_plans()
    pr = _batch(hq, hv, [960, 700, 0, 900])
    kwargs = dict(use_affine_split=True, cu_seqlens_cpu=pr["cu"], initial_state=None)
    _call_batch(pr, **kwargs)
    names = _kernel_names(lambda: _call_batch(pr, **kwargs))
    assert len(names) == 1 and "cake_gdn_affine_split" in names[0], names
    (entry,) = split._EAGER.values()
    assert entry.p_cap == 1 and entry.pass2 is None
    assert entry.o_loc.numel() == entry.c.numel() == entry.bm.numel() == 1
    split.clear_affine_split_plans()


def test_batch_launch_cap_follows_the_bound():
    """The cap is the segment count of the longest sequence: the exact one of
    ``cu_seqlens_cpu``, else the hint, else the token count."""

    split = _split_module()
    pr = _batch(4, 8, [1152, 2048, 300, 0])
    for kwargs, cap in (
        (dict(cu_seqlens_cpu=pr["cu"]), 4),
        (dict(max_seqlen=2048), 4),
        (dict(max_seqlen=1200), None),
        ({}, _p_bound(8, 3500)),
    ):
        split.clear_affine_split_plans()
        if cap is None:
            with pytest.raises(ValueError, match="smaller than the longest"):
                _call_batch(
                    pr,
                    use_affine_split=True,
                    cu_seqlens_cpu=pr["cu"],
                    **kwargs,
                )
            continue
        _call_batch(pr, use_affine_split=True, **kwargs)
        (entry,) = split._EAGER.values()
        assert entry.p_cap == cap, (kwargs, entry.p_cap)
    split.clear_affine_split_plans()


def test_batch_call_forms_keep_the_bits():
    """int64 ``cu_seqlens``, ``cu_seqlens_cpu`` as a tensor, an in-place
    state, ``initial_state=None`` (against explicit zeros), unit gates
    (against ones) and ``output_final_state=False`` hold the bits of the
    plain call over a batch."""

    pr = _batch(4, 8, [4096, 1000, 0, 2500])
    n = 4
    o_ref, s_ref = (x.clone() for x in _call_batch(pr, use_affine_split=True))
    cu64 = torch.tensor(pr["cu"], dtype=torch.int64, device="cuda")
    o, s = _call(
        pr,
        cu=cu64,
        output_state=torch.empty_like(s_ref),
        use_affine_split=True,
        cu_seqlens_cpu=torch.tensor(pr["cu"]),
    )
    assert torch.equal(o, o_ref) and torch.equal(s, s_ref)
    state = pr["h0"].clone()
    o, s = _call_batch(
        pr, initial_state=state, output_state=state, use_affine_split=True
    )
    assert s.data_ptr() == state.data_ptr()
    assert torch.equal(o, o_ref) and torch.equal(s, s_ref)
    zero = dict(pr, h0=torch.zeros_like(pr["h0"]))
    o_zero, s_zero = (x.clone() for x in _call_batch(zero, use_affine_split=True))
    o, s = _call_batch(pr, initial_state=None, use_affine_split=True)
    assert torch.equal(o, o_zero) and torch.equal(s, s_zero)
    ones = dict(pr, g=torch.ones_like(pr["g"]), beta=torch.ones_like(pr["beta"]))
    o_ones, s_ones = (x.clone() for x in _call_batch(ones, use_affine_split=True))
    cu = torch.tensor(pr["cu"], dtype=torch.int32, device="cuda")
    o, s = chunk_gated_delta_rule(
        pr["q"],
        pr["k"],
        pr["v"],
        g=None,
        beta=None,
        scale=SCALE,
        initial_state=pr["h0"],
        output_final_state=True,
        cu_seqlens=cu,
        backend="cake_gdn",
        use_affine_split=True,
    )
    assert s.shape == (n, 8, D, D)
    assert torch.equal(o, o_ones) and torch.equal(s, s_ones)
    only = chunk_gated_delta_rule(
        pr["q"],
        pr["k"],
        pr["v"],
        g=pr["g"],
        beta=pr["beta"],
        scale=SCALE,
        initial_state=pr["h0"],
        output_final_state=False,
        cu_seqlens=cu,
        backend="cake_gdn",
        use_affine_split=True,
    )
    assert torch.equal(only, o_ref)


def test_batch_qk_l2norm_keeps_the_bits():
    """The Q/K L2 norm holds the bits of normalized inputs over a batch."""

    _require_qk_norm()
    pr = _batch(4, 8, [4096, 1000, 0, 2500])
    raw = dict(
        pr,
        q=(pr["q"].float() * 3).to(torch.bfloat16),
        k=(pr["k"].float() * 2).to(torch.bfloat16),
    )
    o_norm, s_norm = (
        x.clone()
        for x in _call_batch(raw, use_affine_split=True, use_qk_l2norm_in_kernel=True)
    )
    from flashinfer.gdn_kernels.qk_l2norm import normalize_qk

    qn, kn = normalize_qk(raw["q"], raw["k"])
    o, s = _call_batch(dict(raw, q=qn, k=kn), use_affine_split=True)
    assert torch.equal(o, o_norm) and torch.equal(s, s_norm)


def test_batch_over_the_tile_limit_raises_before_any_launch():
    split = _split_module()
    t, n = 8192, 1100  # 4 heads, P = 8: 1100 * 4 * 30 pass-1 tiles
    assert split.pass1_grid(n, 4, 8, _sm_count())[0] > split.MAX_PASS1_TILES
    pr = _problem(2, 4, t, "real")
    cu = torch.linspace(0, t, n + 1, device="cuda").round().to(torch.int32)
    output = torch.full((t, 4, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="pass-1 tiles"):
        _call(
            pr,
            cu=cu,
            output=output,
            output_state=torch.empty(n, 4, D, D, device="cuda"),
            initial_state=None,
            use_affine_split=True,
        )
    torch.cuda.synchronize()
    assert torch.isnan(output).all()


@pytest.mark.parametrize("hq,hv", [(2, 4), (4, 8), (8, 16)])
def test_batch_graph_replay_with_new_lengths(hq, hv):
    """N = 4 fixed: a captured call replayed over lengths that move every
    segment count (empty and one-token sequences included) equals the eager
    call, and leaves the flags and the ticket at zero."""

    split = _split_module()
    split.clear_affine_split_plans()
    t = 8192
    static = _batch(hq, hv, [2048] * 4, seed=0)
    cu = torch.tensor(static["cu"], dtype=torch.int32, device="cuda")
    out = torch.empty(t, hv, D, dtype=torch.bfloat16, device="cuda")
    state = torch.empty(4, hv, D, D, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="must be warmed"),
        torch.cuda.graph(graph, stream=stream),
    ):
        _call(static, cu=cu, output=out, output_state=state, use_affine_split=True)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(static, cu=cu, output=out, output_state=state, use_affine_split=True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _call(static, cu=cu, output=out, output_state=state, use_affine_split=True)
    lengths = [
        [2048] * 4,
        [8192, 0, 0, 0],
        [1, 8000, 64, 127],
        [1000, 1000, 1000, 1000],
        [0, 0, 4096, 4000],
        [3000, 961, 960, 2000],
    ]
    for replay in range(30):
        lens = lengths[replay % len(lengths)]
        fresh = _batch(hq, hv, [8192], seed=1 + replay % 4)
        for name in ("q", "k", "v", "g", "beta"):
            static[name].copy_(fresh[name])
        static["h0"].copy_(_batch(hq, hv, [1] * 4, seed=replay)["h0"])
        offsets = [0]
        for n in lens:
            offsets.append(offsets[-1] + n)
        cu.copy_(torch.tensor(offsets, dtype=torch.int32))
        out.fill_(float("nan"))
        state.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _zero_words(), replay
        # Rows past the last offset stay untouched (NaN), as in eager calls.
        assert torch.isfinite(out[: offsets[-1]]).all(), (replay, lens)
        assert torch.isnan(out[offsets[-1] :]).all(), (replay, lens)
        assert torch.isfinite(state).all(), (replay, lens)
        for i, n in enumerate(lens):
            if n == 0:
                assert torch.equal(state[i], static["h0"][i]), (replay, lens, i)
        o_eager = torch.full_like(out, float("nan"))
        s_eager = torch.full_like(state, float("nan"))
        _call(
            static,
            cu=cu.clone(),
            output=o_eager,
            output_state=s_eager,
            use_affine_split=True,
        )
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), o_eager.view(torch.int16)), (
            replay,
            lens,
        )
        assert torch.equal(state.view(torch.int32), s_eager.view(torch.int32)), (
            replay,
            lens,
        )
    split.clear_affine_split_plans()
