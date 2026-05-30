"""Unit tests for flashinfer.trace_apply (no GPU required).

Covers the apply-time core: the first-class Solution schema, routing by
definition identity ``(fi_api, const-axes, input-dtypes)``, dtype disambiguation,
arch safety by compute capability, axis-extraction parity with the live
TraceTemplate, stateful plan/run state recovery, output adaptation, and the
Python-family solution loader.
"""

from __future__ import annotations

import torch

from flashinfer.trace.solution import Solution
from flashinfer.trace_apply import adapt
from flashinfer.trace_apply.apply import (
    bind_namespace,
    build_extractor_maps,
    extract_axes,
    _registry_by_fi_api,
    _stateful_namespace_builder,
)
from flashinfer.trace_apply.config import Definition, TraceConfig
from flashinfer.trace_apply.loaders import load as load_solution
from flashinfer.trace_apply.routing import build_index, lookup

FI_API = "flashinfer.norm.rmsnorm"

# rmsnorm inputs are 'hidden_states' (param 'input') and 'weight'.
def _dt(dtype: str) -> frozenset:
    return frozenset({("hidden_states", dtype), ("weight", dtype)})


def _defn(name: str, hidden: int, dtype: str = "bfloat16") -> dict:
    return {
        "name": name,
        "op_type": "rmsnorm",
        "tags": [f"fi_api:{FI_API}"],
        "axes": {
            "batch_size": {"type": "var"},
            "hidden_size": {"type": "const", "value": hidden},
        },
        "inputs": {
            "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": dtype},
            "weight": {"shape": ["hidden_size"], "dtype": dtype},
        },
        "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": dtype}},
    }


def _sol(name: str, defn: str, author: str = "alice", hw=("sm100",)) -> dict:
    return {
        "name": name,
        "definition": defn,
        "author": author,
        "spec": {
            "language": "python",
            "target_hardware": list(hw),
            "entry_point": "main.py::run",
        },
        "sources": [{"path": "main.py", "content": "def run(*a, **k):\n    return 'CAND'\n"}],
    }


def _config(defn_dicts, sol_dicts) -> TraceConfig:
    definitions = {}
    for d in defn_dicts:
        defn = Definition.from_dict(d)
        definitions[defn.name] = defn
    solutions = {}
    for s in sol_dicts:
        sol = Solution.from_dict(s)
        solutions[(sol.definition, sol.name)] = sol
    return TraceConfig(definitions=definitions, solutions=solutions)


# ---------------------------------------------------------------------------


def test_schema_roundtrip():
    defn = Definition.from_dict(_defn("rmsnorm_h7168", 7168))
    assert defn.fi_api() == FI_API
    assert defn.const_axes() == {"hidden_size": 7168}
    assert defn.input_dtypes() == _dt("bfloat16")

    sol = Solution.from_dict(_sol("triton_v1", "rmsnorm_h7168"))
    assert sol.spec.language == "python"
    assert sol.spec.is_python_family
    assert sol.spec.entry_point == "main.py::run"
    assert sol.sources[0].path == "main.py"
    # JSON round-trip
    assert Solution.from_dict(sol.to_dict()) == sol


def test_index_routes_by_const_and_dtype_key():
    config = _config(
        [_defn("rmsnorm_h1536", 1536), _defn("rmsnorm_h7168", 7168)],
        [_sol("s1536", "rmsnorm_h1536"), _sol("s7168", "rmsnorm_h7168")],
    )
    idx = build_index(config)
    assert idx.has_candidates_for(FI_API)
    assert idx.const_names(FI_API) == {"hidden_size"}
    c = lookup(idx, FI_API, {"hidden_size": 1536}, _dt("bfloat16"), "sm100")
    assert c is not None and c.solution.name == "s1536"
    c = lookup(idx, FI_API, {"hidden_size": 7168}, _dt("bfloat16"), "sm100")
    assert c is not None and c.solution.name == "s7168"
    # unknown const-shape → miss
    assert lookup(idx, FI_API, {"hidden_size": 9999}, _dt("bfloat16"), "sm100") is None


def test_dtype_in_routing_key():
    # Same shape, different dtype → distinct definitions/solutions, routed apart.
    config = _config(
        [_defn("rms_fp16", 4096, "float16"), _defn("rms_bf16", 4096, "bfloat16")],
        [_sol("s_fp16", "rms_fp16"), _sol("s_bf16", "rms_bf16")],
    )
    idx = build_index(config)
    c = lookup(idx, FI_API, {"hidden_size": 4096}, _dt("float16"), "sm100")
    assert c is not None and c.solution.name == "s_fp16"
    c = lookup(idx, FI_API, {"hidden_size": 4096}, _dt("bfloat16"), "sm100")
    assert c is not None and c.solution.name == "s_bf16"
    # a dtype with no matching solution → clean miss (the "guard" behavior)
    assert lookup(idx, FI_API, {"hidden_size": 4096}, _dt("float32"), "sm100") is None


def test_arch_safety_by_compute_capability():
    # sm-tagged solution: must match the running SM; no GPU-SKU name table.
    config = _config([_defn("rmsnorm_h7168", 7168)], [_sol("s", "rmsnorm_h7168", hw=("sm100",))])
    idx = build_index(config)
    assert lookup(idx, FI_API, {"hidden_size": 7168}, _dt("bfloat16"), "sm90") is None
    assert lookup(idx, FI_API, {"hidden_size": 7168}, _dt("bfloat16"), "sm100") is not None


def test_arch_no_constraint_when_no_sm_token():
    # A solution that lists only a non-sm token ("cuda") imposes no arch gate.
    config = _config([_defn("rmsnorm_h7168", 7168)], [_sol("s", "rmsnorm_h7168", hw=("cuda",))])
    idx = build_index(config)
    assert lookup(idx, FI_API, {"hidden_size": 7168}, _dt("bfloat16"), "sm90") is not None


def test_one_solution_per_definition_keeps_first():
    config = _config(
        [_defn("rmsnorm_h7168", 7168)],
        [_sol("first", "rmsnorm_h7168"), _sol("second", "rmsnorm_h7168")],
    )
    idx = build_index(config)
    c = lookup(idx, FI_API, {"hidden_size": 7168}, _dt("bfloat16"), "sm100")
    assert c is not None and c.solution.name == "first"


def test_axis_extraction_parity_with_live_template():
    reg = _registry_by_fi_api()
    assert FI_API in reg
    original, templates = reg[FI_API]
    emaps = build_extractor_maps(templates)

    inp = torch.empty(32, 7168)
    w = torch.empty(7168)
    expected = {"batch_size": 32, "hidden_size": 7168}

    def axes_of(args, kwargs):
        return extract_axes(emaps, bind_namespace(original, args, kwargs))

    assert axes_of((inp, w), {}) == expected
    assert axes_of((), {"input": inp, "weight": w}) == expected
    assert axes_of((inp,), {"weight": w}) == expected

    ns = bind_namespace(original, (inp, w), {})
    ck = adapt.build_candidate_kwargs(templates[0], ns)
    assert set(ck) == {"hidden_states", "weight"}
    assert ck["hidden_states"] is inp and ck["weight"] is w
    # dtype extraction matches the definition signature for required tensors.
    dt = adapt.extract_input_dtypes(templates[0], ns)
    assert dt == _dt("float32")


def test_stateful_plan_run_namespace_and_candidate_kwargs():
    from flashinfer.trace_apply.plan_state import adapter_for, stash_plan_kwargs

    DECODE = "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run"
    reg = _registry_by_fi_api()
    assert DECODE in reg
    run_original, templates = reg[DECODE]
    tmpl = templates[0]
    adapter = adapter_for(DECODE)
    assert adapter is not None

    emaps = build_extractor_maps(templates)
    build_ns = _stateful_namespace_builder(run_original, tmpl, adapter)

    class _W:
        pass

    self_obj = _W()
    kv_indptr = torch.zeros(2, dtype=torch.int32)
    kv_indices = torch.zeros(3, dtype=torch.int32)
    stash_plan_kwargs(self_obj, {"indptr": kv_indptr, "indices": kv_indices, "sm_scale": 0.125})

    q = torch.empty(1, 32, 128)
    k_cache = torch.empty(10, 8, 8, 128)
    v_cache = torch.empty(10, 8, 8, 128)
    ns = build_ns((self_obj, q, (k_cache, v_cache)), {})

    axes = extract_axes(emaps, ns)
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

    ck = adapt.build_candidate_kwargs(tmpl, ns)
    assert ck["q"] is q
    assert ck["k_cache"] is k_cache and ck["v_cache"] is v_cache
    assert ck["kv_indptr"] is kv_indptr and ck["kv_indices"] is kv_indices
    assert ck["sm_scale"] == 0.125


def _live_template(fi_api):
    return _registry_by_fi_api()[fi_api][1][0]


def test_output_adapt_value_returning_with_out():
    tmpl = _live_template("flashinfer.norm.rmsnorm")
    x = torch.randn(4, 8)
    w = torch.ones(8)
    dests = adapt.output_dests("flashinfer.norm.rmsnorm")

    def sol(hidden_states, weight):
        return hidden_states + 1.0

    out = adapt.adapt_and_call(template=tmpl, fn=sol, namespace={"input": x, "weight": w},
                               dps=False, is_inplace=False, dests=dests)
    assert torch.allclose(out, x + 1.0)

    buf = torch.empty_like(x)
    out2 = adapt.adapt_and_call(template=tmpl, fn=sol, namespace={"input": x, "weight": w, "out": buf},
                                dps=False, is_inplace=False, dests=dests)
    assert out2 is buf and torch.allclose(buf, x + 1.0)


def test_output_adapt_inplace_writeback():
    FA = "flashinfer.norm.fused_add_rmsnorm"
    tmpl = _live_template(FA)
    inp = torch.zeros(4, 8)
    res = torch.zeros(4, 8)
    w = torch.ones(8)
    new_out = torch.full((4, 8), 2.0)
    new_res = torch.full((4, 8), 3.0)

    def sol(hidden_states, residual, weight):
        return new_out, new_res

    ret = adapt.adapt_and_call(template=tmpl, fn=sol,
                               namespace={"input": inp, "residual": res, "weight": w},
                               dps=False, is_inplace=True, dests=adapt.output_dests(FA))
    assert ret is None
    assert torch.allclose(inp, new_out)
    assert torch.allclose(res, new_res)


def test_output_adapt_dps_solution():
    tmpl = _live_template("flashinfer.norm.rmsnorm")
    x = torch.randn(4, 8)
    w = torch.ones(8)
    buf = torch.empty_like(x)

    def dps_sol(hidden_states, weight, output):
        output.copy_(hidden_states * 2.0)

    out = adapt.adapt_and_call(template=tmpl, fn=dps_sol, namespace={"input": x, "weight": w, "out": buf},
                               dps=True, is_inplace=False, dests=adapt.output_dests("flashinfer.norm.rmsnorm"))
    assert out is buf and torch.allclose(buf, x * 2.0)


def test_output_arity_gated_by_return_lse():
    import pytest

    DEC = "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run"
    tmpl = _live_template(DEC)
    dests, act = adapt.output_dests(DEC), adapt.output_activation(DEC)
    assert act == {"lse": "return_lse"}
    o, lse = torch.zeros(2, 4), torch.ones(2)

    def sol_full(**kw):
        return o, lse

    r = adapt.adapt_and_call(template=tmpl, fn=sol_full, namespace={"return_lse": False},
                             dps=False, is_inplace=False, dests=dests, activation=act)
    assert r is o
    r = adapt.adapt_and_call(template=tmpl, fn=sol_full, namespace={"return_lse": True},
                             dps=False, is_inplace=False, dests=dests, activation=act)
    assert isinstance(r, tuple) and r[0] is o and r[1] is lse

    with pytest.raises(RuntimeError):
        adapt.adapt_and_call(template=tmpl, fn=lambda **kw: o, namespace={"return_lse": True},
                             dps=False, is_inplace=False, dests=dests, activation=act)

    def dps1(output, **kw):
        output.fill_(7.0)

    ob = torch.empty(2, 4)
    r = adapt.adapt_and_call(template=tmpl, fn=dps1, namespace={"return_lse": False, "out": ob},
                             dps=True, is_inplace=False, dests=dests, activation=act)
    assert r is ob and torch.allclose(ob, torch.full_like(ob, 7.0))

    def dps2(output, lse, **kw):
        output.fill_(7.0)
        lse.fill_(1.0)

    ob2, lb2 = torch.empty(2, 4), torch.empty(2)
    r = adapt.adapt_and_call(template=tmpl, fn=dps2,
                             namespace={"return_lse": True, "out": ob2, "lse": lb2},
                             dps=True, is_inplace=False, dests=dests, activation=act)
    assert isinstance(r, tuple) and r[0] is ob2 and r[1] is lb2


def test_python_solution_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sol = Solution.from_dict(
        {
            "name": "addone", "definition": "dummy", "author": "bob",
            "spec": {"language": "python", "target_hardware": ["sm100"], "entry_point": "main.py::run"},
            "sources": [{"path": "main.py", "content": "def run(x):\n    return x + 1\n"}],
        }
    )
    fn = load_solution(sol)
    out = fn(torch.tensor([1.0, 2.0]))
    assert torch.allclose(out, torch.tensor([2.0, 3.0]))


def test_loader_rejects_path_traversal(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("HOME", str(tmp_path))
    # SourceFile itself rejects abs/.. paths at construction.
    for bad in ("../evil.py", "/etc/evil.py", "a/../../evil.py"):
        with pytest.raises(ValueError):
            Solution.from_dict(
                {
                    "name": "x", "definition": "d", "author": "a",
                    "spec": {"language": "python", "target_hardware": ["sm100"], "entry_point": "main.py::run"},
                    "sources": [{"path": "main.py", "content": "x=1\n"}, {"path": bad, "content": "x=1\n"}],
                }
            )


def test_solution_hash_is_source_order_independent():
    base = {"name": "s", "definition": "d", "author": "a",
            "spec": {"language": "cuda", "target_hardware": ["sm100"], "entry_point": "main.cu::run"}}
    a = Solution.from_dict({**base, "sources": [
        {"path": "main.cu", "content": "A"}, {"path": "k.cuh", "content": "B"}]})
    b = Solution.from_dict({**base, "sources": [
        {"path": "k.cuh", "content": "B"}, {"path": "main.cu", "content": "A"}]})
    assert a.hash() == b.hash()


def test_plan_stash_weakref_cleanup():
    import gc
    import weakref

    from flashinfer.trace_apply.plan_state import fetch_plan_kwargs, stash_plan_kwargs

    class _W:
        pass

    w = _W()
    stash_plan_kwargs(w, {"indptr": 1})
    assert fetch_plan_kwargs(w) == {"indptr": 1}
    ref = weakref.ref(w)
    del w
    gc.collect()
    assert ref() is None
