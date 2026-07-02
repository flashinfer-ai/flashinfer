"""Tests for flashinfer.trace_apply (no GPU required).

Two layers, one file:

* **Unit** — the first-class Solution schema, definition-name computation (the
  basis of name-routing), axis-extraction parity with the live TraceTemplate,
  stateful plan/run state recovery, output adaptation, and the Python solution
  loader.
* **Integration** — ``enable_apply({definition_name: callable_or_Solution})``
  end to end: register a solution by definition name and assert the real public
  ``flashinfer.norm.rmsnorm`` is routed to it (name-routing) rather than the GPU
  kernel. Pure-Python callables run on CPU, so no GPU is required.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
import flashinfer.norm as fnorm
import flashinfer.trace_apply as ta
from flashinfer.trace.solution import Solution
from flashinfer.trace_apply import adapt
from flashinfer.trace_apply.apply import (
    bind_namespace,
    build_extractor_maps,
    extract_axes,
    _derive_output_dests,
    _registry_by_fi_api,
    _stateful_namespace_builder,
)
from flashinfer.trace_apply.config import Definition
from flashinfer.trace_apply.loaders import load as load_solution

FI_API = "flashinfer.norm.rmsnorm"
SENTINEL = 4242.0


def _dt(dtype: str) -> frozenset:
    # rmsnorm inputs are 'hidden_states' (param 'input') and 'weight'.
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
        "sources": [
            {"path": "main.py", "content": "def run(*a, **k):\n    return 'CAND'\n"}
        ],
    }


def _live_template(fi_api):
    return _registry_by_fi_api()[fi_api][1][0]


def _torch_rmsnorm(hidden_states, weight, eps=1e-6):
    x = hidden_states.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(hidden_states.dtype)


def _ref_rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
        x.dtype
    )


@pytest.fixture(autouse=True)
def _reset_trace_apply():
    """Apply is process-global; ensure every test starts and ends disabled."""
    ta.disable_apply()
    yield
    ta.disable_apply()


# ===========================================================================
# Unit tests
# ===========================================================================


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
    assert Solution.from_dict(sol.to_dict()) == sol


def test_definition_name_from_template():
    # Name-routing relies on this: the live template recomputes the definition
    # name from const axes (var axes are excluded).
    tmpl = _live_template(FI_API)
    assert tmpl.definition_name({"batch_size": 8, "hidden_size": 16}) == "rmsnorm_h16"
    assert (
        tmpl.definition_name({"batch_size": 3, "hidden_size": 7168}) == "rmsnorm_h7168"
    )


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
    dt = adapt.extract_input_dtypes(templates[0], ns)
    assert dt == _dt("float32")


def test_stateful_plan_run_namespace_and_candidate_kwargs():
    from flashinfer.trace_apply.plan_capture import adapter_for, stash_plan_kwargs

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
    stash_plan_kwargs(
        self_obj, {"indptr": kv_indptr, "indices": kv_indices, "sm_scale": 0.125}
    )

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


def test_output_adapt_value_returning_returns_value():
    tmpl = _live_template(FI_API)
    x = torch.randn(4, 8)
    w = torch.ones(8)
    dests = adapt.output_dests(tmpl)
    assert dests == {}  # value-returning outputs declare no destination

    def sol(hidden_states, weight):
        return hidden_states + 1.0

    out = adapt.adapt_and_call(
        template=tmpl,
        fn=sol,
        namespace={"input": x, "weight": w},
        dps=False,
        is_inplace=False,
        dests=dests,
    )
    assert torch.allclose(out, x + 1.0)


def test_derive_output_dests_auto_binds_out_and_lse():
    # The trace declares no output param for value-returning APIs; the out=/lse=
    # buffers are auto-derived from the live API signature at enable time.
    reg = _registry_by_fi_api()

    rms_orig = reg[FI_API][0]
    assert adapt.output_dests(_live_template(FI_API)) == {}  # nothing in the trace
    assert _derive_output_dests(_live_template(FI_API), rms_orig) == {"output": "out"}

    DEC = "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run"
    dec_orig = reg[DEC][0]
    assert _derive_output_dests(_live_template(DEC), dec_orig) == {
        "output": "out",
        "lse": "lse",
    }

    # In-place bindings come from the trace param and are preserved (fused_add has
    # no out= param, so nothing is auto-added).
    FA = "flashinfer.norm.fused_add_rmsnorm"
    fa_orig = reg[FA][0]
    assert _derive_output_dests(_live_template(FA), fa_orig) == {
        "output": "input",
        "residual": "residual",
    }


def test_output_adapt_honors_caller_out_buffer():
    # With dests={"output": "out"} (as auto-derived), a value-returning solution +
    # a caller-provided out= buffer → result copied into the buffer and returned.
    tmpl = _live_template(FI_API)
    x = torch.randn(4, 8)
    w = torch.ones(8)
    buf = torch.empty(4, 8)

    def sol(hidden_states, weight):
        return hidden_states + 1.0

    out = adapt.adapt_and_call(
        template=tmpl,
        fn=sol,
        namespace={"input": x, "weight": w, "out": buf},
        dps=False,
        is_inplace=False,
        dests={"output": "out"},
    )
    assert out is buf  # returns the caller's buffer, like the real kernel
    assert torch.allclose(buf, x + 1.0)


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

    dests = adapt.output_dests(tmpl)
    assert dests == {
        "output": "input",
        "residual": "residual",
    }  # in-place, from template
    ret = adapt.adapt_and_call(
        template=tmpl,
        fn=sol,
        namespace={"input": inp, "residual": res, "weight": w},
        dps=False,
        is_inplace=True,
        dests=dests,
    )
    assert ret is None
    assert torch.allclose(inp, new_out)
    assert torch.allclose(res, new_res)


def test_output_adapt_dps_solution():
    tmpl = _live_template(FI_API)
    x = torch.randn(4, 8)
    w = torch.ones(8)

    def dps_sol(hidden_states, weight, output):
        output.copy_(hidden_states * 2.0)

    out = adapt.adapt_and_call(
        template=tmpl,
        fn=dps_sol,
        namespace={"input": x, "weight": w},
        dps=True,
        is_inplace=False,
        dests=adapt.output_dests(tmpl),
    )
    assert torch.allclose(out, x * 2.0)


def test_output_arity_passthrough():
    DEC = "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run"
    tmpl = _live_template(DEC)
    dests = adapt.output_dests(tmpl)
    assert dests == {}  # value-returning: no output destination binding
    o, lse = torch.zeros(2, 4), torch.ones(2)

    r = adapt.adapt_and_call(
        template=tmpl,
        fn=lambda **kw: o,
        namespace={},
        dps=False,
        is_inplace=False,
        dests=dests,
    )
    assert r is o
    r = adapt.adapt_and_call(
        template=tmpl,
        fn=lambda **kw: (o, lse),
        namespace={},
        dps=False,
        is_inplace=False,
        dests=dests,
    )
    assert isinstance(r, tuple) and r[0] is o and r[1] is lse


def test_python_solution_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sol = Solution.from_dict(
        {
            "name": "addone",
            "definition": "dummy",
            "author": "bob",
            "spec": {
                "language": "python",
                "target_hardware": ["sm100"],
                "entry_point": "main.py::run",
            },
            "sources": [
                {"path": "main.py", "content": "def run(x):\n    return x + 1\n"}
            ],
        }
    )
    fn = load_solution(sol)
    out = fn(torch.tensor([1.0, 2.0]))
    assert torch.allclose(out, torch.tensor([2.0, 3.0]))


def test_loader_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for bad in ("../evil.py", "/etc/evil.py", "a/../../evil.py"):
        with pytest.raises(ValueError):
            Solution.from_dict(
                {
                    "name": "x",
                    "definition": "d",
                    "author": "a",
                    "spec": {
                        "language": "python",
                        "target_hardware": ["sm100"],
                        "entry_point": "main.py::run",
                    },
                    "sources": [
                        {"path": "main.py", "content": "x=1\n"},
                        {"path": bad, "content": "x=1\n"},
                    ],
                }
            )


def test_solution_hash_is_source_order_independent():
    base = {
        "name": "s",
        "definition": "d",
        "author": "a",
        "spec": {
            "language": "cuda",
            "target_hardware": ["sm100"],
            "entry_point": "main.cu::run",
        },
    }
    a = Solution.from_dict(
        {
            **base,
            "sources": [
                {"path": "main.cu", "content": "A"},
                {"path": "k.cuh", "content": "B"},
            ],
        }
    )
    b = Solution.from_dict(
        {
            **base,
            "sources": [
                {"path": "k.cuh", "content": "B"},
                {"path": "main.cu", "content": "A"},
            ],
        }
    )
    assert a.hash() == b.hash()


def test_plan_stash_weakref_cleanup():
    import gc
    import weakref

    from flashinfer.trace_apply.plan_capture import fetch_plan_kwargs, stash_plan_kwargs

    class _W:
        pass

    w = _W()
    stash_plan_kwargs(w, {"indptr": 1})
    assert fetch_plan_kwargs(w) == {"indptr": 1}
    ref = weakref.ref(w)
    del w
    gc.collect()
    assert ref() is None


# ===========================================================================
# Integration tests (enable_apply, name-routing, no GPU)
# ===========================================================================


def test_enable_apply_dispatches_to_callable():
    calls = []

    def solution(hidden_states, weight):
        calls.append((hidden_states, weight))
        return torch.full_like(hidden_states, SENTINEL)

    n = ta.enable_apply({"rmsnorm_h16": solution})
    assert ta.is_enabled() and n >= 1

    w = torch.ones(16)
    out = flashinfer.norm.rmsnorm(torch.ones(8, 16), w)
    assert torch.allclose(out, torch.full_like(out, SENTINEL)), "candidate did not run"
    # a different var axis (batch) still routes to the same solution
    out_b = flashinfer.norm.rmsnorm(torch.ones(3, 16), w)
    assert torch.allclose(out_b, torch.full_like(out_b, SENTINEL))
    # the top-level alias (flashinfer.rmsnorm) is patched too
    out2 = flashinfer.rmsnorm(torch.ones(8, 16), w)
    assert torch.allclose(out2, torch.full_like(out2, SENTINEL))

    assert ta.stats().get(FI_API, {}).get("hit", 0) >= 1
    assert calls  # the solution actually executed


def test_enable_apply_real_kernel_numeric():
    """A genuine torch RMSNorm callable; assert the substituted output matches."""
    ta.enable_apply({"rmsnorm_h16": _torch_rmsnorm})
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    w = torch.randn(16)
    out = flashinfer.norm.rmsnorm(x, w)
    assert torch.allclose(out, _ref_rmsnorm(x, w), rtol=1e-4, atol=1e-4)
    assert ta.stats().get(FI_API, {}).get("hit", 0) >= 1


def test_enable_apply_honors_out_kwarg():
    # End-to-end: a caller passing out= gets the substituted result written into
    # the buffer (auto-derived out= binding), matching the real kernel.
    ta.enable_apply({"rmsnorm_h16": _torch_rmsnorm})
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    w = torch.randn(16)
    buf = torch.empty(8, 16)
    out = flashinfer.norm.rmsnorm(x, w, out=buf)
    assert out is buf  # returns the caller's buffer
    assert torch.allclose(buf, _ref_rmsnorm(x, w), rtol=1e-4, atol=1e-4)


def test_enable_apply_solution_object(tmp_path, monkeypatch):
    """A first-class Solution (loaded via the loader), not a raw callable."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # hermetic loader cache
    src = (
        "import torch\n"
        "def run(hidden_states, weight, eps=1e-6):\n"
        "    x = hidden_states.float()\n"
        "    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)\n"
        "    return (y * weight.float()).to(hidden_states.dtype)\n"
    )
    sol = Solution.from_dict(
        {
            "name": "torch_rms",
            "definition": "rmsnorm_h16",
            "author": "tester",
            "spec": {
                "language": "python",
                "target_hardware": ["sm100"],
                "entry_point": "main.py::run",
            },
            "sources": [{"path": "main.py", "content": src}],
        }
    )
    ta.enable_apply({"rmsnorm_h16": sol})
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    w = torch.randn(16)
    out = flashinfer.norm.rmsnorm(x, w)
    assert torch.allclose(out, _ref_rmsnorm(x, w), rtol=1e-4, atol=1e-4)
    assert ta.stats().get(FI_API, {}).get("hit", 0) >= 1


def test_fallback_on_miss_calls_original(monkeypatch):
    """A shape with no registered name must fall through to the original API."""

    # Stand a CPU stub in for the GPU kernel so the fallback path is observable
    # without a GPU; enable_apply wraps the *current* attribute as the original.
    def cpu_original(input, weight, *args, **kwargs):
        return input - 1.0

    monkeypatch.setattr(fnorm, "rmsnorm", cpu_original)

    def sol(hidden_states, weight):
        return torch.full_like(hidden_states, SENTINEL)

    ta.enable_apply({"rmsnorm_h16": sol})
    hit = flashinfer.norm.rmsnorm(
        torch.ones(4, 16), torch.ones(16)
    )  # "rmsnorm_h16" → solution
    assert torch.allclose(hit, torch.full_like(hit, SENTINEL))
    miss = flashinfer.norm.rmsnorm(
        torch.ones(4, 8), torch.ones(8)
    )  # "rmsnorm_h8" → original
    assert torch.allclose(miss, torch.zeros_like(miss))  # 1 - 1 == 0
    st = ta.stats().get(FI_API, {})
    assert st.get("hit", 0) >= 1 and st.get("fallback_no_candidate", 0) >= 1


def test_strict_mode_matched_solution_error_raises():
    """A matched solution that raises must propagate (not silently fall back)."""

    def boom(hidden_states, weight):
        raise ValueError("boom")

    ta.enable_apply({"rmsnorm_h16": boom})
    with pytest.raises(ValueError, match="boom"):
        flashinfer.norm.rmsnorm(torch.ones(8, 16), torch.ones(16))
    assert ta.stats().get(FI_API, {}).get("error", 0) >= 1


def test_enable_overrides_previous_then_disable_reverts():
    def s1(hidden_states, weight):
        return torch.full_like(hidden_states, SENTINEL)

    def s2(hidden_states, weight):
        return torch.full_like(hidden_states, 1234.0)

    w = torch.ones(16)
    ta.enable_apply({"rmsnorm_h16": s1})
    out1 = flashinfer.norm.rmsnorm(torch.ones(2, 16), w)
    assert torch.allclose(out1, torch.full_like(out1, SENTINEL))

    ta.enable_apply({"rmsnorm_h16": s2})  # re-enable overrides
    out2 = flashinfer.norm.rmsnorm(torch.ones(2, 16), w)
    assert torch.allclose(out2, torch.full_like(out2, 1234.0))

    ta.disable_apply()
    assert not ta.is_enabled()
    assert getattr(flashinfer.norm.rmsnorm, "_trace_apply", False) is False


def test_solutions_positional_and_keyword_equivalent():
    def sol(hidden_states, weight):
        return torch.full_like(hidden_states, SENTINEL)

    n1 = ta.enable_apply({"rmsnorm_h16": sol})  # positional dict
    ta.disable_apply()
    n2 = ta.enable_apply(solutions={"rmsnorm_h16": sol})  # keyword
    assert n1 == n2 and n1 >= 1
    out = flashinfer.norm.rmsnorm(torch.ones(8, 16), torch.ones(16))
    assert torch.allclose(out, torch.full_like(out, SENTINEL))


def test_enable_apply_empty_is_noop():
    assert ta.enable_apply({}) == 0
    assert not ta.is_enabled()
