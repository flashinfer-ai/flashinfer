"""Unit tests for flashinfer.trace_apply (no GPU required).

Covers the apply-time core: schema round-trip, index keying by fi_api with the
full concrete axis vector (const+var), multi-definition dispatch, lookup
filters, axis-extraction parity with the live TraceTemplate, and the
Python/Triton solution loader.
"""

from __future__ import annotations

import torch

from flashinfer.trace_apply.index import build_index
from flashinfer.trace_apply.lookup import lookup
from flashinfer.trace_apply.schema import Definition, Solution, TraceRecord
from flashinfer.trace_apply.source import Trace
from flashinfer.trace_apply.axes import (
    bind_namespace,
    build_candidate_kwargs,
    build_extractor_maps,
    extract_from_namespace,
)
from flashinfer.trace_apply import loader

FI_API = "flashinfer.norm.rmsnorm"


def _defn(name: str, hidden: int) -> dict:
    return {
        "name": name,
        "op_type": "rmsnorm",
        "tags": [f"fi_api:{FI_API}", "status:verified"],
        "axes": {
            "batch_size": {"type": "var"},
            "hidden_size": {"type": "const", "value": hidden},
        },
        "inputs": {
            "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
            "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
        },
        "outputs": {
            "output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
        },
    }


def _sol(name: str, defn: str, author: str = "alice", entry: str = "main.py::run") -> dict:
    return {
        "name": name,
        "definition": defn,
        "author": author,
        "spec": {
            "language": "python",
            "target_hardware": ["NVIDIA B200"],
            "entry_point": entry,
        },
        "sources": [{"path": "main.py", "content": "def run(*a, **k):\n    return 'CANDIDATE'\n"}],
    }


def _rec(defn: str, sol: str, batch: int, status: str = "PASSED", latency: float = 0.1, hw: str = "NVIDIA B200") -> dict:
    return {
        "definition": defn,
        "solution": sol,
        "workload": {"axes": {"batch_size": batch}, "uuid": f"{defn}-{batch}"},
        "evaluation": {
            "status": status,
            "environment": {"hardware": hw, "libs": {"torch": "2.11.0"}},
            "performance": {"latency_ms": latency, "reference_latency_ms": 1.0},
            "correctness": {"max_relative_error": 1e-3},
        },
    }


def _trace(defn_dicts, sol_dicts, rec_dicts) -> Trace:
    definitions = {}
    for d in defn_dicts:
        defn = Definition.from_dict(d)
        definitions[defn.name] = defn
    solutions = {}
    for s in sol_dicts:
        sol = Solution.from_dict(s)
        solutions[(sol.definition, sol.name)] = sol
    records = [TraceRecord.from_dict(r) for r in rec_dicts]
    return Trace(paths=None, definitions=definitions, solutions=solutions, records=records)


# ---------------------------------------------------------------------------


def test_schema_roundtrip():
    defn = Definition.from_dict(_defn("rmsnorm_h7168", 7168))
    assert defn.fi_api() == FI_API
    assert defn.const_axes() == {"hidden_size": 7168}
    assert defn.var_axis_names() == ["batch_size"]

    sol = Solution.from_dict(_sol("triton_v1", "rmsnorm_h7168"))
    assert sol.spec.language == "python"
    assert sol.spec.entry_point == "main.py::run"
    assert sol.sources[0].path == "main.py"

    rec = TraceRecord.from_dict(_rec("rmsnorm_h7168", "triton_v1", 32))
    assert rec.status == "PASSED"
    assert rec.workload.axes == {"batch_size": 32}
    assert rec.environment.hardware == "NVIDIA B200"
    assert abs(rec.performance.latency_ms - 0.1) < 1e-9


def test_index_keys_by_fi_api_with_full_axes():
    # Two const-specialized definitions for the SAME api must coexist.
    trace = _trace(
        [_defn("rmsnorm_h1536", 1536), _defn("rmsnorm_h7168", 7168)],
        [_sol("s1536", "rmsnorm_h1536"), _sol("s7168", "rmsnorm_h7168")],
        [_rec("rmsnorm_h1536", "s1536", 32), _rec("rmsnorm_h7168", "s7168", 32)],
    )
    idx = build_index(trace)
    assert idx.has_candidates_for(FI_API, "sm100")
    # full axis vector = const(hidden) + var(batch)
    c = lookup(idx, FI_API, {"hidden_size": 1536, "batch_size": 32}, "sm100")
    assert c is not None and c.solution.name == "s1536"
    c = lookup(idx, FI_API, {"hidden_size": 7168, "batch_size": 32}, "sm100")
    assert c is not None and c.solution.name == "s7168"
    # wrong hidden → miss (this is the multi-def/const-baking bug guard)
    assert lookup(idx, FI_API, {"hidden_size": 9999, "batch_size": 32}, "sm100") is None
    # partial axes (missing const) → miss
    assert lookup(idx, FI_API, {"batch_size": 32}, "sm100") is None


def test_lookup_filters_status_and_sm():
    trace = _trace(
        [_defn("rmsnorm_h7168", 7168)],
        [_sol("s7168", "rmsnorm_h7168")],
        [_rec("rmsnorm_h7168", "s7168", 32, status="FAILED")],
    )
    idx = build_index(trace)
    # FAILED status filtered out
    assert lookup(idx, FI_API, {"hidden_size": 7168, "batch_size": 32}, "sm100") is None

    trace2 = _trace(
        [_defn("rmsnorm_h7168", 7168)],
        [_sol("s7168", "rmsnorm_h7168")],
        [_rec("rmsnorm_h7168", "s7168", 32)],
    )
    idx2 = build_index(trace2)
    # solution targets B200 (sm100); a request on sm90 must miss (no key for sm90)
    assert lookup(idx2, FI_API, {"hidden_size": 7168, "batch_size": 32}, "sm90") is None
    assert lookup(idx2, FI_API, {"hidden_size": 7168, "batch_size": 32}, "sm100") is not None


def test_latency_ranking_picks_fastest():
    trace = _trace(
        [_defn("rmsnorm_h7168", 7168)],
        [_sol("slow", "rmsnorm_h7168"), _sol("fast", "rmsnorm_h7168")],
        [
            _rec("rmsnorm_h7168", "slow", 32, latency=0.9),
            _rec("rmsnorm_h7168", "fast", 32, latency=0.1),
        ],
    )
    idx = build_index(trace)
    c = lookup(idx, FI_API, {"hidden_size": 7168, "batch_size": 32}, "sm100")
    assert c is not None and c.solution.name == "fast"


def test_axis_extraction_parity_with_live_template():
    # Reuse the live rmsnorm template; extracted axes must match the shapes,
    # for both positional and keyword call forms.
    import sys

    import flashinfer.trace_apply.install  # noqa: F401 — ensure submodule loaded

    install_mod = sys.modules["flashinfer.trace_apply.install"]
    reg = install_mod._registry_by_fi_api()
    assert FI_API in reg
    original, templates = reg[FI_API]
    emaps = build_extractor_maps(templates)

    inp = torch.empty(32, 7168)
    w = torch.empty(7168)
    expected = {"batch_size": 32, "hidden_size": 7168}

    def axes_of(args, kwargs):
        return extract_from_namespace(emaps, bind_namespace(original, args, kwargs))

    assert axes_of((inp, w), {}) == expected  # positional
    assert axes_of((), {"input": inp, "weight": w}) == expected  # keyword
    assert axes_of((inp,), {"weight": w}) == expected  # mixed

    # build_candidate_kwargs maps the template inputs (json keys) to values:
    # rmsnorm's 'hidden_states' input has param='input'.
    ns = bind_namespace(original, (inp, w), {})
    ck = build_candidate_kwargs(templates[0], ns)
    assert set(ck) == {"hidden_states", "weight"}
    assert ck["hidden_states"] is inp and ck["weight"] is w


def test_stateful_plan_run_namespace_and_candidate_kwargs():
    # gqa_paged_decode: q + paged_kv_cache come from run(); kv_indptr/kv_indices/
    # sm_scale come from plan(). Verify the stateful namespace builder merges
    # them, axis extraction sees ALL axes (incl plan-derived len_indptr/
    # num_kv_indices), and the candidate kwargs are reconstructed by json key.
    import sys

    import flashinfer.trace_apply.install  # noqa: F401
    from flashinfer.trace_apply.axes import stash_plan_kwargs
    from flashinfer.trace_apply.stateful import adapter_for

    install_mod = sys.modules["flashinfer.trace_apply.install"]
    DECODE = "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run"
    reg = install_mod._registry_by_fi_api()
    assert DECODE in reg
    run_original, templates = reg[DECODE]
    tmpl = templates[0]
    adapter = adapter_for(DECODE)
    assert adapter is not None

    emaps = build_extractor_maps(templates)
    build_ns = install_mod._make_stateful_namespace(run_original, tmpl, adapter)

    class _W:  # stand-in wrapper instance
        pass

    self_obj = _W()
    # Simulate plan(indptr, indices, ..., sm_scale=...) having stashed its args.
    kv_indptr = torch.zeros(2, dtype=torch.int32)
    kv_indices = torch.zeros(3, dtype=torch.int32)
    stash_plan_kwargs(id(self_obj), {"indptr": kv_indptr, "indices": kv_indices, "sm_scale": 0.125})

    q = torch.empty(1, 32, 128)
    k_cache = torch.empty(10, 8, 8, 128)
    v_cache = torch.empty(10, 8, 8, 128)
    ns = build_ns((self_obj, q, (k_cache, v_cache)), {})

    axes = extract_from_namespace(emaps, ns)
    assert axes == {
        "batch_size": 1,
        "num_qo_heads": 32,
        "head_dim": 128,
        "num_pages": 10,
        "page_size": 8,
        "num_kv_heads": 8,
        "len_indptr": 2,
        "num_kv_indices": 3,
    }

    ck = build_candidate_kwargs(tmpl, ns)
    assert ck["q"] is q
    assert ck["k_cache"] is k_cache and ck["v_cache"] is v_cache
    assert ck["kv_indptr"] is kv_indptr and ck["kv_indices"] is kv_indices
    assert ck["sm_scale"] == 0.125


def test_python_solution_loader(tmp_path, monkeypatch):
    # Loader caches under ~/.cache; redirect HOME to tmp so the test is hermetic.
    monkeypatch.setenv("HOME", str(tmp_path))
    sol = Solution.from_dict(
        {
            "name": "addone",
            "definition": "dummy",
            "author": "bob",
            "spec": {"language": "python", "target_hardware": ["NVIDIA B200"], "entry_point": "main.py::run"},
            "sources": [{"path": "main.py", "content": "def run(x):\n    return x + 1\n"}],
        }
    )
    fn = loader.load(sol)
    out = fn(torch.tensor([1.0, 2.0]))
    assert torch.allclose(out, torch.tensor([2.0, 3.0]))
