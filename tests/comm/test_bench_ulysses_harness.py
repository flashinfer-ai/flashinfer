# flashinfer: pure-CPU tests for the Ulysses benchmark harness methodology —
# fail-closed compare gate, enforced run minimums, order rotation, stats.

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_ulysses_a2a.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_ulysses_a2a", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _payload(bench, label, commit, impls, ws=8, p50=1.0):
    meta = {
        "schema": bench.SCHEMA_VERSION,
        "harness_sha": "abc123def456",
        "label": label,
        "commit": commit,
        "world_size": ws,
        "workload": bench.default_workload(ws),
        "unit": "3x scatter_heads + 1x gather_heads (+sdpa for e2e_attn), ms",
        "repeats": 5,
        "iters": 30,
        "warmup": 5,
        "sample_reduction": "max across ranks per iteration",
        "order_policy": "rotation per repeat (every impl visits every position)",
        "torch": "2.x",
        "device": "H20",
        "package_dirty": False,
    }
    results = {
        name: {
            unit: {"p50": p50, "p95": p50, "mean": p50, "std": 0.0, "n": 150}
            for unit in bench.UNITS
        }
        for name in impls
    }
    return {"meta": meta, "results": results}


BASE_IMPLS = ["raw", "nccl_ref"]
NEW_IMPLS = ["raw", "communicator", "communicator_nccl", "nccl_ref"]


def test_compare_pass_and_gate(bench):
    base = _payload(bench, "baseline", "c83e4204", BASE_IMPLS, p50=1.0)
    new = _payload(bench, "new", "deadbeef", NEW_IMPLS, p50=1.02)
    lines, failed = bench.compare_payloads(base, new, 3.0)
    assert not failed and lines
    # 5% regression on a gated pair must fail
    new_bad = _payload(bench, "new", "deadbeef", NEW_IMPLS, p50=1.05)
    _lines, failed = bench.compare_payloads(base, new_bad, 3.0)
    assert failed


def test_compare_rejects_self_comparison(bench):
    base = _payload(bench, "baseline", "c83e4204", NEW_IMPLS)
    same_label = _payload(bench, "baseline", "deadbeef", NEW_IMPLS)
    with pytest.raises(ValueError, match="labels are identical"):
        bench.compare_payloads(base, same_label, 3.0)
    same_commit = _payload(bench, "new", "c83e4204", NEW_IMPLS)
    with pytest.raises(ValueError, match="commits are identical"):
        bench.compare_payloads(base, same_commit, 3.0)


def test_compare_rejects_missing_gate_pairs(bench):
    base = _payload(bench, "baseline", "c83e4204", BASE_IMPLS)
    # no communicator in new: the raw->communicator gate cannot be vacuous
    new = _payload(bench, "new", "deadbeef", ["raw", "nccl_ref"])
    with pytest.raises(ValueError, match="fail-closed.*communicator missing"):
        bench.compare_payloads(base, new, 3.0)
    # empty results must not pass either
    empty = _payload(bench, "new", "deadbeef", [])
    with pytest.raises(ValueError, match="fail-closed"):
        bench.compare_payloads(base, empty, 3.0)


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema", "other-schema"),
        ("harness_sha", "fffffffffff0"),
        ("world_size", 4),
        ("repeats", 7),
        ("iters", 60),
        ("warmup", 10),
        ("torch", "1.x"),
        ("device", "A100"),
        ("sample_reduction", "rank0 only"),
        ("order_policy", "fixed"),
        ("unit", "something else"),
    ],
)
def test_compare_rejects_meta_mismatch(bench, field, value):
    base = _payload(bench, "baseline", "c83e4204", BASE_IMPLS)
    new = _payload(bench, "new", "deadbeef", NEW_IMPLS)
    new["meta"][field] = value
    if field in ("repeats", "iters"):
        # keep the artifact internally consistent (n == repeats*iters) so the
        # cross-artifact meta mismatch is what fires, not the n sanity check
        n = new["meta"]["repeats"] * new["meta"]["iters"]
        for units in new["results"].values():
            for st in units.values():
                st["n"] = n
    with pytest.raises(ValueError, match=f"meta field '{field}'"):
        bench.compare_payloads(base, new, 3.0)


def test_compare_rejects_workload_mismatch(bench):
    base = _payload(bench, "baseline", "c83e4204", BASE_IMPLS)
    new = _payload(bench, "new", "deadbeef", NEW_IMPLS)
    new["meta"]["workload"] = dict(new["meta"]["workload"], H=48)
    with pytest.raises(ValueError, match="meta field 'workload'"):
        bench.compare_payloads(base, new, 3.0)


def test_run_args_minimums(bench):
    ok = dict(
        world_size=8,
        impls=["raw"],
        repeats=5,
        iters=30,
        warmup=5,
        device_count=8,
        available=lambda n: True,
    )
    bench.validate_run_args(**ok)
    with pytest.raises(ValueError, match="repeats"):
        bench.validate_run_args(**{**ok, "repeats": 4})
    with pytest.raises(ValueError, match="iters"):
        bench.validate_run_args(**{**ok, "iters": 29})
    with pytest.raises(ValueError, match="warmup"):
        bench.validate_run_args(**{**ok, "warmup": 0})
    with pytest.raises(ValueError, match="world_size"):
        bench.validate_run_args(**{**ok, "world_size": 3})
    with pytest.raises(ValueError, match="GPUs"):
        bench.validate_run_args(**{**ok, "device_count": 4})
    with pytest.raises(ValueError, match="at least one"):
        bench.validate_run_args(**{**ok, "impls": []})
    with pytest.raises(ValueError, match="unknown implementation"):
        bench.validate_run_args(**{**ok, "impls": ["magic"]})
    with pytest.raises(ValueError, match="unavailable"):
        bench.validate_run_args(
            **{**ok, "impls": ["communicator"], "available": lambda n: False}
        )


def test_rotation_covers_all_positions(bench):
    impls = ["a", "b", "c", "d"]
    seen = {name: set() for name in impls}
    for rep in range(5):
        order = bench.rotation_order(impls, rep)
        assert sorted(order) == sorted(impls)
        for pos, name in enumerate(order):
            seen[name].add(pos)
    for name, positions in seen.items():
        assert positions == set(range(len(impls))), (
            f"{name} never visited positions {set(range(len(impls))) - positions}"
        )


def test_stats_median_is_conventional(bench):
    # even sample count: conventional median averages the middle pair
    st = bench._stats([1.0, 2.0, 3.0, 4.0])
    assert st["p50"] == 2.5
    assert st["n"] == 4


def test_compare_rejects_bad_data_and_provenance(bench):
    base = _payload(bench, "baseline", "c83e4204", BASE_IMPLS)
    good = _payload(bench, "new", "deadbeef", NEW_IMPLS)

    dirty = copy.deepcopy(good)
    dirty["meta"]["package_dirty"] = True
    with pytest.raises(ValueError, match="dirty/unknown package provenance"):
        bench.compare_payloads(base, dirty, 3.0)

    unknown = copy.deepcopy(good)
    unknown["meta"]["commit"] = "unknown"
    with pytest.raises(ValueError, match="dirty/unknown package provenance"):
        bench.compare_payloads(base, unknown, 3.0)

    for bad_p50 in (float("nan"), float("inf"), 0.0, -1.0):
        bad = copy.deepcopy(good)
        bad["results"]["raw"]["a2a"]["p50"] = bad_p50
        with pytest.raises(ValueError, match="invalid p50"):
            bench.compare_payloads(base, bad, 3.0)

    short = copy.deepcopy(good)
    short["results"]["raw"]["a2a"]["n"] = 149
    with pytest.raises(ValueError, match="truncated or padded"):
        bench.compare_payloads(base, short, 3.0)


def test_payload_copy_is_not_mutated(bench):
    base = _payload(bench, "baseline", "c83e4204", BASE_IMPLS)
    new = _payload(bench, "new", "deadbeef", NEW_IMPLS)
    base_copy = copy.deepcopy(base)
    bench.compare_payloads(base, new, 3.0)
    assert base == base_copy
