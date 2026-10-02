"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Contract tests of the experimental Cake DSA indexer top-k backend
# (flashinfer-ai/flashinfer#5676).  The host-side tests (registry, gate
# policy, validation, binding, the reference itself) run without a GPU; the
# operator tests need a compute capability 10.0 / 10.3 / 10.7 device with a
# registered generated program and skip otherwise.  Opt-in heavy variant:
# ``FLASHINFER_CAKE_DSA_INDEXER_FULL=1`` runs the recorded packed key geometry
# through the FP64 reference (minutes on B200).

import contextlib
import os
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_dsa_indexer import cake_backend, cake_jit
from flashinfer.experimental.cake_dsa_indexer.cake_backend import (
    ABI_CONTRACT,
    CONTRACT_SCALARS,
    CONTRACT_TENSORS,
    GATE_POLICY_FIELDS,
    HEAD_DIM,
    MAX_TOP_K,
    NUM_HEADS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    ZERO_SIGN_POLICIES,
    GatePolicy,
    bind_stage,
    dsa_indexer_workspace_size,
    finalize_stage_for,
    generated_program_available,
    grid_dims,
    prepare_dsa_indexer_topk,
    record_for,
    record_gate_policy,
    record_zero_sign_policy,
    validate_dsa_indexer_inputs,
    visible_key_count,
    visible_key_counts,
)
from tests.test_helpers import cake_dsa_indexer_reference as ref

FULL = os.environ.get("FLASHINFER_CAKE_DSA_INDEXER_FULL", "0") not in (
    "",
    "0",
    "false",
    "no",
)
# The candidate-gate constants of the first generated program (the registry
# record carries the authoritative values; these drive the host-only tests).
EXAMPLE_POLICY = GatePolicy(
    queries_per_cta=4,
    candidate_entry_bytes=8,
    candidate_multiplier=4,
    candidate_slack=128,
    tile_keys=128,
    check_period_max=32,
    check_period_cap_divisor=512,
    sample_tiles_max=32,
    sample_shift_permille=250,
    finalize_small_max_top_k=2048,
)
P1_CU_SEQLENS_Q = (0, 1777, 3532, 5655, 7765, 9888, 11998, 14121, 16231)
P1_CU_SEQLENS_K = (0, 1777, 58619, 60742, 128665, 130788, 198711, 200834, 268757)
P2_T = 16172


def _lengths(cu):
    return [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)]


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated DSA indexer program not registered for this device")


@contextlib.contextmanager
def _quiet_experimental():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


def _zero_sign_policy() -> str:
    _, record = record_for(torch.device("cuda"))
    return record_zero_sign_policy(record)


def _run(inputs: ref.IndexerInputs, **extra):
    out = cake_backend.dsa_indexer_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        **extra,
    )
    assert isinstance(out, tuple) and len(out) == 2
    return out


def _judge_ok(result, reference, **kw) -> ref.Judgement:
    if kw.get("exact_bits") and "zero_sign_policy" not in kw:
        kw["zero_sign_policy"] = _zero_sign_policy()
    verdict = ref.judge(result, reference, **kw)
    assert verdict.passed, str(verdict)
    return verdict


def _independent_membership_check(
    inputs: ref.IndexerInputs, indices: torch.Tensor
) -> None:
    """Visibility and segment membership from the scalar rule in Python (no tensors shared with the reference)."""
    ids = indices.cpu().tolist()
    K = inputs.top_k
    offsets = inputs.offsets()
    starts = inputs.query_starts()
    for s in range(inputs.num_segments):
        for u in range(inputs.seg_q_len[s]):
            visible = visible_key_count(
                offsets[s], u, inputs.ratio, inputs.seg_k_len[s]
            )
            n = min(K, visible)
            row = ids[starts[s] + u]
            assert all(0 <= j < visible for j in row[:n]), (s, u, row[:8])
            assert all(j == -1 for j in row[n:]), (s, u)
            assert row[:n] == sorted(set(row[:n])), (s, u)


# ---------------------------------------------------------------------------
# Host-only: registry, policy, validation, binding
# ---------------------------------------------------------------------------


def test_public_api_is_marked_experimental():
    from flashinfer.dsa_indexer import dsa_indexer_topk

    assert dsa_indexer_topk.is_experimental


def test_registry_records_are_well_formed():
    if not cake_jit.MODULES:
        # placeholder registry: every architecture reports the missing program by name
        assert cake_jit.registered_archs() == ()
        for arch in cake_jit.ARCHES:
            with pytest.raises(NotImplementedError, match="5676"):
                cake_jit.select_module(arch)
        return
    seen = set()
    for name, record in cake_jit.MODULES.items():
        assert record["arch"] in cake_jit.ARCHES and record["arch"] not in seen
        seen.add(record["arch"])
        assert record["abi"] == ABI_CONTRACT
        stages = cake_jit.registered_stages(name)
        assert "scan" in stages and any(s in stages for s in cake_jit.FINALIZE_STAGES)
        policy = record_gate_policy(record)
        assert set(GATE_POLICY_FIELDS) == set(record["gate_policy"])
        assert policy.candidate_capacity(2048) >= 2048 + policy.candidate_slack
        assert record_zero_sign_policy(record) in ZERO_SIGN_POLICIES
        root = cake_jit.__file__.rsplit("/", 1)[0] + "/csrc"
        for stage in stages:
            physical = record[stage]
            for key in (
                "module",
                "sources",
                "compile_flags",
                "ffi_entry",
                "arg_plan",
                "closure_sha256",
                "grid",
                "launch",
            ):
                assert key in physical, (name, stage, key)
            assert len(physical["grid"]) == 3
            assert all(
                os.path.isfile(os.path.join(root, src)) for src in physical["sources"]
            ), (name, stage)
            for kind, arg in physical["arg_plan"]:
                assert kind in (
                    "buffer",
                    "tma_buffer",
                    "parameter",
                    "grid",
                    "workspace",
                ), (stage, kind, arg)
                if kind != "grid":
                    key = cake_backend.CONTRACT_ALIASES.get(arg, arg)
                    assert (
                        key in CONTRACT_TENSORS
                        or key in CONTRACT_SCALARS
                        or key in ("num_queries", "num_keys")
                    ), (stage, arg)
        assert finalize_stage_for(stages, policy, 2048) in stages
        assert finalize_stage_for(stages, policy, MAX_TOP_K) in stages


def test_gate_policy_capacity_period_and_workspace():
    p = EXAMPLE_POLICY
    assert p.candidate_capacity(2048) == 8192
    assert p.candidate_capacity(1) == 256  # max(4, 1 + 128) rounded up to a tile
    assert p.candidate_capacity(4096) == 16384
    assert p.candidate_capacity(2048, multiplier=2) == 4096
    assert p.check_period(2048, 8192) == 16
    assert p.check_period_limit(2048, 8192) == 32
    assert p.check_period(1, 256) == 1
    wider = GatePolicy(**{**asdict(p), "check_period_cap_divisor": 1024})
    assert (
        wider.check_period(2048, 8192) == 8
    )  # an architecture with a wider default period
    assert wider.check_period(4096, 16384) == 16
    assert p.workspace_bytes(2048, 148) == 148 * 4 * 8192 * 8
    assert (
        dsa_indexer_workspace_size(2048, grid_ctas=148, policy=p) == 148 * 4 * 8192 * 8
    )
    assert (
        dsa_indexer_workspace_size(256, grid_ctas=212, policy=p) == 212 * 4 * 1024 * 8
    )
    with pytest.raises(ValueError):
        dsa_indexer_workspace_size(0, grid_ctas=1, policy=p)
    with pytest.raises(ValueError):
        dsa_indexer_workspace_size(MAX_TOP_K + 1, grid_ctas=1, policy=p)
    with pytest.raises(ValueError):
        GatePolicy.from_record({"gate_policy": {"queries_per_cta": 4}})


def test_finalize_stage_selection():
    p = EXAMPLE_POLICY
    both = ("scan", "finalize", "finalize_small")
    assert finalize_stage_for(both, p, 2048) == "finalize_small"
    assert finalize_stage_for(both, p, 2049) == "finalize"
    assert finalize_stage_for(both, p, 64, "finalize") == "finalize"
    assert finalize_stage_for(("scan", "finalize"), p, 64) == "finalize"
    with pytest.raises(ValueError):
        finalize_stage_for(both, p, 4096, "finalize_small")
    with pytest.raises(NotImplementedError):
        finalize_stage_for(("scan", "finalize"), p, 64, "finalize_small")
    with pytest.raises(NotImplementedError):
        finalize_stage_for(("scan",), p, 64)


@pytest.mark.parametrize(
    "spec",
    [
        dict(seg_q_len=[5, 3], seg_k_len=[9, 3], ratio=1, offsets=None),
        dict(seg_q_len=[4, 6], seg_k_len=[10, 2], ratio=1, offsets=[-3, 4]),
        dict(seg_q_len=[7, 1, 0], seg_k_len=[4, 1, 5], ratio=2, offsets=None),
        dict(seg_q_len=[9], seg_k_len=[6], ratio=3, offsets=[-7]),
    ],
)
def test_visible_key_counts_match_scalar_rule(spec):
    counts = visible_key_counts(
        spec["seg_q_len"],
        spec["seg_k_len"],
        ratio=spec["ratio"],
        q_causal_offsets=spec["offsets"],
    ).tolist()
    offsets = cake_backend.effective_offsets(
        spec["seg_q_len"], spec["seg_k_len"], spec["offsets"], spec["ratio"]
    )
    expected = [
        visible_key_count(offsets[s], u, spec["ratio"], spec["seg_k_len"][s])
        for s in range(len(spec["seg_q_len"]))
        for u in range(spec["seg_q_len"][s])
    ]
    assert counts == expected
    helper = ref.visible_rows(
        ref.build_inputs(
            torch.zeros(
                (sum(spec["seg_q_len"]), NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16
            ),
            torch.zeros((sum(spec["seg_k_len"]), HEAD_DIM), dtype=torch.bfloat16),
            torch.zeros((sum(spec["seg_q_len"]), NUM_HEADS)),
            spec["seg_q_len"],
            spec["seg_k_len"],
            top_k=4,
            softmax_scale=1.0,
            q_causal_offsets=spec["offsets"],
            ratio=spec["ratio"],
        )
    ).tolist()
    assert helper == expected


def _host_inputs(total_q=6, total_k=10, segments=(0, 2, 6), key_segments=(0, 3, 10)):
    q = torch.zeros((total_q, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    k = torch.zeros((total_k, HEAD_DIM), dtype=torch.bfloat16)
    w = torch.zeros((total_q, NUM_HEADS), dtype=torch.float32)
    cu_q = torch.tensor(segments, dtype=torch.int32)
    cu_k = torch.tensor(key_segments, dtype=torch.int32)
    return SimpleNamespace(
        q=q, k=k, w=w, cu_q=cu_q, cu_k=cu_k, offsets=None, top_k=4, scale=0.5, ratio=1
    )


def _validate(h):
    return validate_dsa_indexer_inputs(
        h.q, h.k, h.w, h.cu_q, h.cu_k, h.offsets, h.top_k, h.scale, h.ratio
    )


def test_validate_reaches_the_device_rule_for_host_tensors():
    # shapes, dtypes, strides and scalars pass; host tensors fail only the device rule
    with pytest.raises(ValueError, match="CUDA"):
        _validate(_host_inputs())
    h = _host_inputs()
    h.k = torch.zeros((10, ref.TRAINER_KEY_ROW_STRIDE), dtype=torch.bfloat16)[
        :, :HEAD_DIM
    ]  # trainer view
    with pytest.raises(ValueError, match="CUDA"):
        _validate(h)
    h = _host_inputs()
    h.offsets = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="CUDA"):
        _validate(h)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda h: setattr(h, "q", h.q[:, :, :64]), r"q must be"),
        (lambda h: setattr(h, "q", h.q.float()), "bfloat16"),
        (
            lambda h: setattr(h, "q", h.q.transpose(0, 1).contiguous().transpose(0, 1)),
            "contiguous",
        ),
        (lambda h: setattr(h, "k", h.k[:, :64]), r"k must be"),
        (lambda h: setattr(h, "k", h.k.float()), "bfloat16"),
        (
            lambda h: setattr(
                h, "k", torch.zeros((10, 130), dtype=torch.bfloat16)[:, :128]
            ),
            "row stride",
        ),
        (
            lambda h: setattr(h, "k", h.k.t().contiguous().t()),
            "unit inner stride|row stride",
        ),
        (lambda h: setattr(h, "w", h.w[:, :16]), r"w must be"),
        (lambda h: setattr(h, "w", h.w.to(torch.bfloat16)), "float32"),
        (lambda h: setattr(h, "cu_q", h.cu_q.to(torch.int64)), "int32"),
        (lambda h: setattr(h, "cu_k", h.cu_k[:-1]), "S \\+ 1"),
        (
            lambda h: setattr(h, "offsets", torch.zeros(3, dtype=torch.int64)),
            "q_causal_offsets must be",
        ),
        (lambda h: setattr(h, "offsets", torch.zeros(2, dtype=torch.int32)), "int64"),
        (lambda h: setattr(h, "top_k", 0), "top_k"),
        (lambda h: setattr(h, "top_k", MAX_TOP_K + 1), "top_k"),
        (lambda h: setattr(h, "top_k", 2.0), "top_k"),
        (lambda h: setattr(h, "ratio", 0), "ratio"),
        (lambda h: setattr(h, "scale", 0.0), "softmax_scale"),
        (lambda h: setattr(h, "scale", float("nan")), "softmax_scale"),
    ],
)
def test_validate_rejects(mutate, match):
    h = _host_inputs()
    mutate(h)
    with pytest.raises(ValueError, match=match):
        _validate(h)


def test_grid_dims():
    scalars = {"num_queries": 1000, "num_keys": 5000}
    assert grid_dims(["sms", 1, 1], scalars, 148) == (148, 1, 1)
    assert grid_dims(["sms*2", 1, 1], scalars, 148) == (296, 1, 1)
    assert grid_dims(["num_queries", 1, 1], scalars, 148) == (1000, 1, 1)
    assert grid_dims(["num_queries/256", 2, 1], scalars, 148) == (4, 2, 1)
    assert grid_dims(["num_keys*3/512", 1, 1], scalars, 148) == (30, 1, 1)
    assert grid_dims([0, 1, 1], scalars, 148) == (1, 1, 1)
    with pytest.raises(ValueError):
        grid_dims(["sms", 1], scalars, 148)


def _fake_record(arg_plan):
    return {
        "arch": "sm_100a",
        "abi": ABI_CONTRACT,
        "stages": ["scan", "finalize_small"],
        "gate_policy": dict(
            zip(GATE_POLICY_FIELDS, (4, 8, 4, 128, 128, 32, 32, 250, 2048), strict=True)
        ),
        "numerics": {"zero_sign_policy": "positive_accumulator"},
        "scan": {
            "module": "fake_scan",
            "sources": [],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": arg_plan,
            "closure_sha256": "0" * 64,
            "workspace_bytes": 0,
            "grid": ["sms", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
    }


def test_bind_stage_orders_arguments_and_fails_closed(monkeypatch):
    calls = []
    fake_module = SimpleNamespace(run=lambda *args: calls.append(args))
    monkeypatch.setattr(
        cake_backend, "load_cake_dsa_indexer_module", lambda name, stage: fake_module
    )
    plan = [
        ["tma_buffer", "K"],
        ["buffer", "indices"],  # alias of Indices
        ["parameter", "topk"],  # alias of top_k
        ["parameter", "softmax_scale"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ]
    monkeypatch.setitem(cake_jit.MODULES, "fake", _fake_record(plan))
    values = {
        "K": "k-view",
        "Indices": "idx",
        "top_k": 7,
        "softmax_scale": 0.25,
        "Q": "unused",
    }
    launch = bind_stage("fake", "scan", values, (148, 1, 1))
    assert launch.arguments == ("k-view", "idx", 7, 0.25, 148, 1, 1)
    assert launch.grid == (148, 1, 1)
    launch()
    assert calls == [launch.arguments]
    monkeypatch.setitem(
        cake_jit.MODULES, "fake", _fake_record(plan + [["parameter", "mystery_knob"]])
    )
    with pytest.raises(KeyError, match="mystery_knob"):
        bind_stage("fake", "scan", values, (148, 1, 1))
    monkeypatch.setitem(cake_jit.MODULES, "fake", _fake_record([["buffer", "Cand"]]))
    with pytest.raises(KeyError, match="Cand"):
        bind_stage("fake", "scan", {"Cand": None}, (1, 1, 1))


# ---------------------------------------------------------------------------
# Host-only: the reference itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed,top_k,ratio,offsets",
    [(0, 5, 1, None), (1, 1, 1, [2, -1]), (2, 3, 2, None), (3, 20, 1, None)],
)
def test_reference_matches_bruteforce(seed, top_k, ratio, offsets):
    inputs = ref.make_random_inputs(
        [9, 6],
        [11, 7],
        top_k=top_k,
        seed=seed,
        device="cpu",
        ratio=ratio,
        q_causal_offsets=offsets,
        softmax_scale=0.3,
    )
    result = ref.select_reference(inputs, order="sequential")
    ids, scores = ref.bruteforce_rows(inputs)
    assert result.indices.tolist() == ids
    got = result.scores_fp64.tolist()
    for row_got, row_exp in zip(got, scores, strict=True):
        for a, b in zip(row_got, row_exp, strict=True):
            assert a == b or abs(a - b) <= 1e-12 * max(1.0, abs(b)), (a, b)
    assert not ref.structure_failures(inputs, result.indices, result.scores)
    assert ref.judge((result.indices, result.scores), result, exact_bits=True).passed


def test_reference_sequential_order_is_partition_invariant():
    inputs = ref.make_random_inputs([40, 30], [70, 50], top_k=16, seed=5, device="cpu")
    a = ref.select_reference(inputs, order="sequential", query_chunk=7, key_chunk=9)
    b = ref.select_reference(inputs, order="sequential", query_chunk=64, key_chunk=4096)
    assert ref.same_bits((a.indices, a.scores), (b.indices, b.scores))
    assert torch.equal(a.scores_fp64, b.scores_fp64)


@pytest.mark.parametrize("name", list(ref.EXACT_CASES))
def test_exact_cases_match_closed_form(name):
    inputs = ref.make_exact_case(name, "cpu")
    result = ref.select_reference(inputs, order="sequential")
    score = ref.exact_closed_form(inputs)
    ids = result.indices.tolist()
    s32 = result.scores.tolist()
    s64 = result.scores_fp64.tolist()
    for t in range(inputs.num_queries):
        for c, j in enumerate(ids[t]):
            if j < 0:
                break
            expected = score(t, j)
            assert s64[t][c] == expected and s32[t][c] == expected, (name, t, c, j)
    if name == "signed_zeros":
        bits = result.scores.view(torch.int32)
        zero = (result.scores == 0) & (result.indices >= 0)
        assert bool(zero.any(dim=1).all()), (
            "zero-score keys must reach the selector boundary on every row"
        )
        # the program's positive-accumulator reduction yields +0.0 for every zero score; an IEEE sequential
        # head sum would give -0.0 on the all-negative-weight (even) rows, so the bit check is discriminating
        assert bool((bits[zero] == 0).all())
        even = torch.arange(inputs.num_queries) % 2 == 0
        assert bool(zero[even].any()) and bool(zero[~even].any())
        with pytest.raises(NotImplementedError):
            ref.judge(
                (result.indices, result.scores),
                result,
                exact_bits=True,
                zero_sign_policy="ieee_sum",
            )


# the scaled geometry of the exact cases, small enough for the sequential CPU reference
_SCALED_EXACT = dict(num_queries=24, num_keys=1024)


@pytest.mark.parametrize("name", list(ref.EXACT_CASES))
def test_scaled_exact_cases_match_closed_form(name):
    inputs = ref.make_exact_case(name, "cpu", **_SCALED_EXACT)
    assert (inputs.seg_q_len, inputs.seg_k_len, inputs.offsets()) == (
        [24],
        [1024],
        [1000],
    )
    assert 256 <= inputs.top_k <= 257
    result = ref.select_reference(inputs, order="sequential")
    score = ref.exact_closed_form(inputs)
    ids = result.indices.tolist()
    s32 = result.scores.tolist()
    s64 = result.scores_fp64.tolist()
    visible = result.visible.tolist()
    for t in range(inputs.num_queries):
        selected = ids[t][: int(result.selected_count[t])]
        assert len(selected) == inputs.top_k and all(j >= 0 for j in selected)
        chosen = [score(t, j) for j in selected]
        for c, (j, expected) in enumerate(zip(selected, chosen, strict=True)):
            assert s64[t][c] == expected and s32[t][c] == expected, (name, t, c, j)
        rest = [score(t, j) for j in range(visible[t]) if j not in set(selected)]
        if name == "distinct":
            assert len(set(chosen)) == len(chosen)
            if visible[t] == 1024:
                assert len({score(t, j) for j in range(1024)}) == 1024
        elif name == "cutoff_ties":
            if visible[t] == 1024:
                assert min(chosen) == max(rest), "the cut must fall inside a tie group"
        elif name == "all_equal":
            assert len(set(chosen)) == 1 and set(rest) == set(chosen)
        elif name == "negative":
            assert all((v < 0) if t % 3 == 0 else (v > 0) for v in chosen)
        elif name == "few_winners_large_tie":
            base = score(t, 0)
            winners = {w for w in (3, 232, 511, 512, 1023) if w < visible[t]}
            assert {j for j in selected if score(t, j) != base} == winners
            assert all(v == base for v in rest)
    if name == "signed_zeros":
        bits = result.scores.view(torch.int32)
        zero = (result.scores == 0) & (result.indices >= 0)
        assert bool(zero.any(dim=1).all()), (
            "zero-score keys must reach the selector boundary on every row"
        )
        assert bool((bits[zero] == 0).all())
        even = torch.arange(inputs.num_queries) % 2 == 0
        assert bool(zero[even].any()) and bool(zero[~even].any())
        assert bool(((result.scores > 0) & (result.indices >= 0))[~even].any())
        assert not bool(((result.scores > 0) & (result.indices >= 0))[even].any())


def test_scaled_exact_cases_reject_unusable_geometries():
    with pytest.raises(ValueError):
        ref.make_exact_case("distinct", "cpu", num_queries=8)
    with pytest.raises(ValueError):
        ref.make_exact_case("distinct", "cpu", num_queries=9, num_keys=8)
    with pytest.raises(ValueError):
        ref.make_exact_case("distinct", "cpu", num_queries=8, num_keys=65536)
    with pytest.raises(
        ValueError
    ):  # the first row would see too few keys for the zero boundary
        ref.make_exact_case("signed_zeros", "cpu", num_queries=1024, num_keys=1024)


@pytest.mark.parametrize("name", list(ref.PACKED_GEOMETRIES))
def test_packed_geometry_extra_segments_keep_the_declared_segments(name):
    base = ref.make_packed_inputs(name, "cpu")
    inputs = ref.make_packed_inputs(
        name, "cpu", extra_segments=2, extra_segment_queries=8, extra_segment_keys=64
    )
    n = base.num_segments
    assert inputs.seg_q_len == base.seg_q_len + [8, 8]
    assert inputs.seg_k_len == base.seg_k_len + [64, 64]
    assert (inputs.top_k, inputs.ratio, inputs.softmax_scale) == (
        base.top_k,
        base.ratio,
        base.softmax_scale,
    )
    assert inputs.offsets()[:n] == base.offsets()
    assert inputs.offsets()[n:] == [base.ratio * 64 - 8] * 2
    assert (inputs.q_causal_offsets is None) == (
        base.q_causal_offsets is None and base.ratio == 1
    )
    visible = ref.visible_rows(inputs, device="cpu").tolist()
    assert visible[: base.num_queries] == ref.visible_rows(base, device="cpu").tolist()
    extra = visible[base.num_queries :]
    assert extra[7] == 64 and extra[15] == 64 and min(extra) >= 56


def test_judge_rejects_a_clearly_better_lacking_key_and_bad_bits():
    inputs = ref.make_exact_case("distinct", "cpu")
    reference = ref.select_reference(inputs, order="sequential")
    ids, scores = reference.indices.clone(), reference.scores.clone()
    assert ref.judge((ids, scores), reference, exact_bits=True).passed
    # swap the weakest selected key of row 0 for an unselected, lower-scoring visible key
    row = 0
    n = int(reference.selected_count[row])
    chosen = set(ids[row, :n].tolist())
    visible = int(reference.visible[row])
    unselected = [j for j in range(visible) if j not in chosen]
    assert unselected
    worst_slot = min(range(n), key=lambda c: (float(scores[row, c]), int(ids[row, c])))
    ids[row, worst_slot] = unselected[0]
    scores[row, worst_slot] = ref.score_pairs(
        inputs, torch.tensor([row]), torch.tensor([unselected[0]])
    )[1][0]
    order = torch.argsort(
        torch.where(ids[row] >= 0, ids[row], torch.full_like(ids[row], 2**31 - 1))
    )
    ids[row], scores[row] = ids[row][order], scores[row][order]
    verdict = ref.judge((ids, scores), reference)
    assert not verdict.passed and any(
        "clearly better" in f for f in verdict.failures
    ), str(verdict)
    # a wrong score bit pattern on a correct id
    ids, scores = reference.indices.clone(), reference.scores.clone()
    scores[1, 0] = scores[1, 0] + 1.0
    verdict = ref.judge((ids, scores), reference)
    assert not verdict.passed and any(
        "outside the bound" in f for f in verdict.failures
    ), str(verdict)
    # a structural error: a duplicate id
    ids, scores = reference.indices.clone(), reference.scores.clone()
    ids[2, 1] = ids[2, 0]
    assert ref.structure_failures(inputs, ids, scores)


# ---------------------------------------------------------------------------
# Operator: exact semantics, causality, accuracy, near ties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(ref.EXACT_CASES))
def test_exact_semantics(name):
    _require_program()
    inputs = ref.make_exact_case(name, "cuda")
    reference = ref.select_reference(inputs, order="sequential")
    verdict = _judge_ok(_run(inputs), reference, exact_bits=True)
    assert verdict.rows_identical == inputs.num_queries


def test_signed_zero_bits_follow_the_documented_policy():
    _require_program()
    inputs = ref.make_exact_case("signed_zeros", "cuda")
    indices, scores = _run(inputs)
    reference = ref.select_reference(inputs, order="sequential")
    assert torch.equal(
        indices, reference.indices
    )  # +0.0 == -0.0 for ranking: the id rule decides inside the zero group
    policy = _zero_sign_policy()
    zero = (scores == 0) & (indices >= 0)
    assert bool(zero.any())
    bits = scores.view(torch.int32)
    if policy != "positive_accumulator":
        pytest.fail(f"the reference has no model for zero-sign policy {policy!r}")
    assert bool((bits[zero] == 0).all()), (
        "the chains start from +0.0 with a fixed FMA order: every zero score is +0.0"
    )
    assert torch.equal(bits[zero], reference.scores.view(torch.int32)[zero])


@pytest.mark.parametrize("name", list(ref.PACKED_GEOMETRIES))
def test_packed_causality(name):
    _require_program()
    inputs = ref.make_packed_inputs(name, "cuda")
    reference = ref.select_reference(inputs)
    out = _run(inputs)
    _independent_membership_check(inputs, out[0])
    _judge_ok(out, reference)


@pytest.mark.parametrize("seed,peaked", [(1, False), (2, False), (3, True)])
def test_random_accuracy_within_bound(seed, peaked):
    _require_program()
    inputs = ref.make_random_inputs(
        [1100, 900, 1000],
        [1100, 20000, 1000],
        top_k=2048,
        seed=seed,
        device="cuda",
        peaked=peaked,
    )
    reference = ref.select_reference(inputs)
    verdict = _judge_ok(_run(inputs), reference)
    assert (
        verdict.max_fp32_error_over_bound <= 1.0
        and verdict.max_fp64_error_over_bound <= 1.0
    )


def test_near_tie_duplicated_keys_prefer_larger_ids():
    """Replicated keys give bit-identical scores under any deterministic scheme: the larger id must win."""
    _require_program()
    n_distinct, reps, lq, K = 300, 8, 48, 2048
    base = ref.make_random_inputs([lq], [n_distinct], top_k=K, seed=21, device="cuda")
    lk = n_distinct * reps
    group = (
        torch.randperm(
            lk, generator=torch.Generator(device="cuda").manual_seed(5), device="cuda"
        )
        % n_distinct
    )
    k = base.k[group].contiguous()
    inputs = ref.build_inputs(
        base.q,
        k,
        base.w,
        [lq],
        [lk],
        top_k=K,
        softmax_scale=base.softmax_scale,
        label="near_tie_duplicates",
    )
    reference = ref.select_reference(inputs, order="sequential")
    out = _run(inputs)
    _judge_ok(out, reference)
    ids = out[0].cpu()
    group_cpu = group.cpu()
    visible = ref.visible_rows(inputs, device="cpu")
    for t in range(inputs.num_queries):
        n = min(K, int(visible[t]))
        sel = ids[t, :n]
        sel_groups = group_cpu[sel]
        for g in sel_groups.unique().tolist():
            members = (group_cpu[: int(visible[t])] == g).nonzero().squeeze(1)
            chosen = sel[sel_groups == g]
            if (
                chosen.numel() < members.numel()
            ):  # the cutoff splits this replica group: the largest ids are chosen
                assert torch.equal(
                    chosen.sort().values, members.sort().values[-chosen.numel() :]
                ), (t, g)


def test_near_tie_perturbed_keys_disagree_only_within_uncertainty():
    _require_program()
    inputs = ref.make_random_inputs([64], [6000], top_k=2048, seed=22, device="cuda")
    k = inputs.k
    src = torch.arange(0, 3000, device="cuda")
    k[src + 3000] = k[src]
    col = k[src + 3000, 0].contiguous()
    k[src + 3000, 0] = (col.view(torch.int16) + 1).view(
        torch.bfloat16
    )  # one BF16 ulp on one dimension
    reference = ref.select_reference(inputs)
    verdict = _judge_ok(_run(inputs), reference)
    assert (
        verdict.rows_identical + verdict.rows_within_uncertainty == inputs.num_queries
    )


# ---------------------------------------------------------------------------
# Operator: dynamic shapes, repeatability, partition independence
# ---------------------------------------------------------------------------


def _recorded_inputs(name: str):
    if name == "P1":
        seg_q, seg_k = _lengths(P1_CU_SEQLENS_Q), _lengths(P1_CU_SEQLENS_K)
    else:  # the P1 pattern scaled to the second recorded token count (its boundaries were not recorded)
        seg_q = _lengths(P1_CU_SEQLENS_Q)
        seg_q = [max(1, round(v * P2_T / sum(seg_q))) for v in seg_q]
        seg_q[-1] += P2_T - sum(seg_q)
        seg_k = list(seg_q)
    if not FULL:
        seg_k = list(
            seg_q
        )  # keys per segment = queries per segment keeps the FP64 reference to seconds
    return ref.make_random_inputs(
        seg_q, seg_k, top_k=2048, seed=7571, device="cuda", label=name
    )


@pytest.mark.parametrize("name", ["P1", "P2"])
def test_recorded_token_counts(name):
    _require_program()
    inputs = _recorded_inputs(name)
    assert inputs.num_queries in (16231, P2_T)
    reference = ref.select_reference(inputs)
    _judge_ok(_run(inputs), reference)


_BOUNDARY = [
    (5, lk, 32)
    for lk in (
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
    )
]
_BOUNDARY += [
    (n, n, 64)
    for n in (1, 2, 3, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025)
]
_BOUNDARY += [
    (16, 5000, k)
    for k in (
        1,
        2,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        2047,
        2048,
        2049,
        4095,
        4096,
    )
]


@pytest.mark.parametrize("lq,lk,top_k", _BOUNDARY)
def test_tile_boundary_neighbours(lq, lk, top_k):
    _require_program()
    inputs = ref.make_boundary_inputs(lq, lk, top_k=top_k, device="cuda")
    _judge_ok(_run(inputs), ref.select_reference(inputs))


def test_changing_shapes_in_sequence():
    _require_program()
    sequence = [
        ([37], [1000], 64),
        ([2049], [2049], 2048),
        ([3, 700, 1], [3, 9000, 1], 512),
        ([16231 // 8] * 2, [16231 // 8, 20000], 2048),
        ([129], [129], 1),
    ]
    for i, (lq, lk, k) in enumerate(sequence):
        inputs = ref.make_random_inputs(lq, lk, top_k=k, seed=30 + i, device="cuda")
        _judge_ok(_run(inputs), ref.select_reference(inputs))


def test_repeatability_bitwise():
    _require_program()
    inputs = ref.make_random_inputs(
        [700, 300], [700, 9000], top_k=2048, seed=40, device="cuda", peaked=True
    )
    a, b, c = _run(inputs), _run(inputs), _run(inputs)
    assert ref.same_bits(a, b) and ref.same_bits(a, c)


PARTITION_KNOBS = (
    {"grid_ctas": 37},
    {"candidate_multiplier": 2},
    {"check_period": 1},
    {"sample_tiles_max": 0},
    {"sample_shift_permille": -500},
    {"finalize_stage": "finalize"},
)


@pytest.mark.parametrize(
    "knobs", PARTITION_KNOBS, ids=lambda d: "_".join(f"{k}={v}" for k, v in d.items())
)
def test_partition_knobs_do_not_change_results(knobs):
    _require_program()
    inputs = ref.make_random_inputs(
        [1500, 600], [1500, 30000], top_k=2048, seed=41, device="cuda"
    )
    base = _run(inputs)
    _judge_ok(base, ref.select_reference(inputs))
    other = _run(inputs, **knobs)
    assert ref.same_bits(base, other), f"partition knob {knobs} changed the result"


# ---------------------------------------------------------------------------
# Operator: robustness, graph capture, workspace, public entry
# ---------------------------------------------------------------------------


def test_strided_k_view_matches_contiguous_bitwise():
    _require_program()
    seg_q = [max(1, v // 16) for v in _lengths(P1_CU_SEQLENS_Q)]
    seg_k = [max(1, v // 16) for v in _lengths(P1_CU_SEQLENS_K)]
    strided = ref.make_random_inputs(
        seg_q,
        seg_k,
        top_k=256,
        seed=7701,
        device="cuda",
        k_row_stride=ref.TRAINER_KEY_ROW_STRIDE,
    )
    assert (
        strided.k.stride(0) == ref.TRAINER_KEY_ROW_STRIDE
        and not strided.k.is_contiguous()
    )
    contiguous = ref.build_inputs(
        strided.q,
        strided.k.contiguous(),
        strided.w,
        seg_q,
        seg_k,
        top_k=256,
        softmax_scale=strided.softmax_scale,
    )
    a, b = _run(strided), _run(contiguous)
    assert ref.same_bits(a, b)
    _judge_ok(a, ref.select_reference(contiguous))


def test_nonfinite_inputs_return_with_intact_structure():
    _require_program()
    inputs = ref.make_nonfinite_inputs("cuda")
    indices, scores = _run(inputs)
    torch.cuda.synchronize()
    assert (
        tuple(indices.shape) == (inputs.num_queries, inputs.top_k)
        and indices.dtype == torch.int32
    )
    assert (
        tuple(scores.shape) == (inputs.num_queries, inputs.top_k)
        and scores.dtype == torch.float32
    )
    reference = ref.select_reference(inputs)
    _judge_ok((indices, scores), reference, rows=ref.clean_rows(inputs))
    visible = ref.visible_rows(inputs)
    for t in ref.dirty_rows(inputs).tolist():
        row = indices[t].tolist()
        valid = [j for j in row if j >= 0]
        assert len(valid) == len(set(valid)) and all(
            j < int(visible[t]) for j in valid
        ), t
        assert valid == sorted(valid), t


def test_prepared_runner_graph_capture_and_replay():
    _require_program()
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=50, device="cuda"
    )
    eager = _run(inputs)
    runner = prepare_dsa_indexer_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            runner()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner()
    graph.replay()
    torch.cuda.synchronize()
    assert ref.same_bits(captured, eager)
    fresh = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=51, device="cuda"
    )
    inputs.q.copy_(fresh.q)
    inputs.k.copy_(fresh.k)
    inputs.w.copy_(fresh.w)
    graph.replay()
    torch.cuda.synchronize()
    _judge_ok((captured[0].clone(), captured[1].clone()), ref.select_reference(inputs))


def test_workspace_bound_is_respected():
    _require_program()
    inputs = ref.make_random_inputs(
        [900, 300], [900, 7000], top_k=2048, seed=60, device="cuda"
    )
    device = torch.device("cuda", torch.cuda.current_device())
    nbytes = dsa_indexer_workspace_size(inputs.top_k, device)
    _, record = record_for(device)
    policy = record_gate_policy(record)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert nbytes == policy.workspace_bytes(inputs.top_k, sms)
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    indices = torch.empty(
        (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=device
    )
    scores = torch.empty(
        (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=device
    )
    runner = prepare_dsa_indexer_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        workspace_buffer=workspace,
        indices=indices,
        scores=scores,
    )
    assert runner.workspace_bytes == nbytes
    runner()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    out = runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert (
        after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0
    ), "the prepared launch allocated"
    assert (
        out[0].data_ptr() == indices.data_ptr()
        and out[1].data_ptr() == scores.data_ptr()
    )
    _judge_ok(out, ref.select_reference(inputs))
    with pytest.raises(ValueError, match="workspace_buffer needs"):
        prepare_dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            workspace_buffer=workspace[: nbytes - 8],
        )


def test_empty_rows_and_no_keys_are_all_padding():
    _require_program()
    device = torch.device("cuda")
    q = torch.zeros((0, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    k = torch.zeros((5, HEAD_DIM), dtype=torch.bfloat16, device=device)
    w = torch.zeros((0, NUM_HEADS), dtype=torch.float32, device=device)
    cu = torch.tensor([0, 0], dtype=torch.int32, device=device)
    indices, scores = cake_backend.dsa_indexer_topk(
        q, k, w, cu, torch.tensor([0, 5], dtype=torch.int32, device=device), top_k=8
    )
    assert tuple(indices.shape) == (0, 8) and tuple(scores.shape) == (0, 8)
    inputs = ref.make_random_inputs([3, 4], [0, 0], top_k=8, seed=1, device="cuda")
    indices, scores = _run(inputs)
    assert bool((indices == -1).all()) and bool(torch.isneginf(scores).all())
    _judge_ok((indices, scores), ref.select_reference(inputs))


def test_public_entry_point_matches_backend_and_warns_once():
    _require_program()
    from flashinfer.dsa_indexer import dsa_indexer_topk, dsa_indexer_topk_workspace_size

    inputs = ref.make_random_inputs(
        [200, 300], [200, 2500], top_k=128, seed=70, device="cuda"
    )
    expected = _run(inputs)
    with pytest.warns(ExperimentalWarning):
        got = dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
        )
    assert ref.same_bits(got, expected)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        again = dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            max_seqlen_q=max(inputs.seg_q_len),
            max_seqlen_k=max(inputs.seg_k_len),
        )
    assert ref.same_bits(again, expected)
    assert dsa_indexer_topk_workspace_size(
        128, inputs.device
    ) == dsa_indexer_workspace_size(128, inputs.device)
    with pytest.raises(ValueError, match="backend"), _quiet_experimental():
        dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            backend="other",
        )
