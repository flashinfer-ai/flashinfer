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

import importlib

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
def _reset_trace_apply(monkeypatch: pytest.MonkeyPatch):
    """Reset process-global Apply state without requiring a CUDA device."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ta.disable_apply()
    yield
    ta.disable_apply()


# ===========================================================================
# Unit tests
# ===========================================================================


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_definition_name_from_template():
    # Name-routing relies on this: the live template recomputes the definition
    # name from const axes (var axes are excluded).
    tmpl = _live_template(FI_API)
    assert tmpl.definition_name({"batch_size": 8, "hidden_size": 16}) == "rmsnorm_h16"
    assert (
        tmpl.definition_name({"batch_size": 3, "hidden_size": 7168}) == "rmsnorm_h7168"
    )


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_mla_stateful_adapter_recovers_moved_wrapper_plan_state():
    from flashinfer.trace_apply.plan_capture import adapter_for

    moved = "flashinfer.mla._batch_mla._wrapper.BatchMLAPagedAttentionWrapper.run"
    import flashinfer.mla  # noqa: F401  # Registers the moved wrapper trace.

    reg = _registry_by_fi_api()
    assert moved in reg
    run_original, templates = reg[moved]
    template = templates[0]
    adapter = adapter_for(moved)
    assert adapter is not None

    class _Wrapper:
        _kv_indptr_buf = torch.tensor([0, 1, 2], dtype=torch.int32)
        _kv_indices_buf = torch.tensor([3, 4], dtype=torch.int32)
        _sm_scale = 0.125

    q_nope = torch.empty(2, 4, 8, dtype=torch.bfloat16)
    q_pe = torch.empty(2, 4, 3, dtype=torch.bfloat16)
    ckv_cache = torch.empty(5, 7, 8, dtype=torch.bfloat16)
    kpe_cache = torch.empty(5, 7, 3, dtype=torch.bfloat16)
    build_ns = _stateful_namespace_builder(run_original, template, adapter)
    namespace = build_ns((_Wrapper(), q_nope, q_pe, ckv_cache, kpe_cache), {})

    candidate_kwargs = adapt.build_candidate_kwargs(template, namespace)
    assert candidate_kwargs["kv_indptr"].tolist() == [0, 1, 2]
    assert candidate_kwargs["kv_indices"].tolist() == [3, 4]
    assert candidate_kwargs["sm_scale"] == 0.125


@pytest.mark.cpu_only
def test_mla_trace_apply_accepts_historical_and_moved_wrapper_keys():
    from flashinfer.trace_apply.plan_capture import adapter_for

    historical = "flashinfer.mla._core.BatchMLAPagedAttentionWrapper.run"
    moved = "flashinfer.mla._batch_mla._wrapper.BatchMLAPagedAttentionWrapper.run"
    import flashinfer.mla  # noqa: F401  # Registers the MLA wrapper trace.

    reg = _registry_by_fi_api()
    assert historical in reg
    assert moved in reg
    assert adapter_for(historical) is not None
    assert adapter_for(moved) is not None


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


SSD_FI_API = "flashinfer.mamba.ssd_combined.ssd_combined_fwd"


def _ssd_combined_kwargs(*, slots: bool, checkpoint_rows: int) -> dict:
    """CPU arguments of one ``ssd_combined_fwd`` call with a checkpoint pool:
    batched with dense states, or packed varlen with an initial and a final
    pool (five rows) addressed through slot vectors."""
    seqlen, nheads = 16, 2
    state = (nheads, 64, 128)
    batch = 1 if slots else 2
    kwargs = {
        "x": torch.randn(batch, seqlen, nheads, 64),
        "dt": torch.randn(batch, seqlen, nheads),
        "A": torch.randn(nheads),
        "B": torch.randn(batch, seqlen, nheads, 128),
        "C": torch.randn(batch, seqlen, nheads, 128),
        "checkpoint_token_indices": torch.tensor([3, 11], dtype=torch.int32),
        "checkpoint_state_slots": torch.tensor([1, 0], dtype=torch.int32),
        "checkpoint_states": torch.randn(checkpoint_rows, *state),
    }
    if slots:
        kwargs.update(
            cu_seqlens=torch.tensor([0, 8, seqlen], dtype=torch.int32),
            initial_states=torch.randn(7, *state),
            initial_state_slots=torch.tensor([4, 0], dtype=torch.int32),
            final_states=torch.randn(5, *state),
            final_state_slots=torch.tensor([1, 3], dtype=torch.int32),
        )
    return kwargs


def _ssd_combined_apply(kwargs: dict):
    """The selected template, namespace and output destinations of one
    ``ssd_combined_fwd`` call, built as the Apply wrapper builds them."""
    importlib.import_module("flashinfer.mamba.ssd_combined")
    from flashinfer.trace.templates.mamba import ssd_combined_trace_dispatch

    original = _registry_by_fi_api()[SSD_FI_API][0]
    namespace = bind_namespace(original, (), kwargs)
    template = ssd_combined_trace_dispatch(**namespace)
    return template, namespace, _derive_output_dests(template, original)


def _ssd_combined_definition_name(kwargs: dict) -> str:
    """The definition name the Apply wrapper computes for one call."""
    template, namespace, _ = _ssd_combined_apply(kwargs)
    axes = extract_axes(build_extractor_maps([template]), namespace)
    return template.definition_name(axes)


def _ssd_combined_state_kwargs(*, slots: bool, dtype: torch.dtype) -> dict:
    """``_ssd_combined_kwargs`` with every state tensor in ``dtype``."""
    kwargs = _ssd_combined_kwargs(slots=slots, checkpoint_rows=5)
    for key in ("initial_states", "final_states", "checkpoint_states"):
        if key in kwargs:
            kwargs[key] = kwargs[key].to(dtype)
    return kwargs


def _ssd_combined_seq_idx_kwargs(dtype: torch.dtype) -> dict:
    """A slot call in the ``seq_idx`` form (two sequences of eight tokens in
    one chunk) instead of the ``cu_seqlens`` form, with f32 state pools."""
    kwargs = _ssd_combined_state_kwargs(slots=True, dtype=torch.float32)
    del kwargs["cu_seqlens"]
    kwargs.update(
        seq_idx=torch.tensor([[0] * 8 + [1] * 8], dtype=dtype),
        chunk_indices=torch.tensor([0, 0], dtype=torch.int32),
        chunk_offsets=torch.tensor([0, 8], dtype=torch.int32),
    )
    return kwargs


def _ssd_combined_tag_solution(calls: list, tag: str):
    """A value-returning solution that records ``tag`` when called."""

    def solution(**inputs):
        calls.append(tag)
        return inputs["x"], inputs["final_states"]

    return solution


@pytest.mark.cpu_only
def test_ssd_combined_apply_routes_slot_pools_by_state_dtype(monkeypatch, tmp_path):
    """Slot calls of identical dimensions with f16 and f32 state pools have
    different definition names, so each reaches only its own solution and
    traces to its own JSON file. The dense names carry no state dtype."""
    import functools

    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    f16_kwargs = _ssd_combined_state_kwargs(slots=True, dtype=torch.float16)
    f32_kwargs = _ssd_combined_state_kwargs(slots=True, dtype=torch.float32)
    # The cu_seqlens form runs the int64 seq_idx runner.
    f16_name = "ssd_combined_varlen_slots_f16_sidx_i64_d_none_final_h2_d64_g2_s128"
    f32_name = "ssd_combined_varlen_slots_f32_sidx_i64_d_none_final_h2_d64_g2_s128"
    for kwargs, name in ((f16_kwargs, f16_name), (f32_kwargs, f32_name)):
        assert _ssd_combined_definition_name(kwargs) == name
        definition = module.ssd_combined_fwd.fi_trace(save_dir=tmp_path, **kwargs)
        assert definition["name"] == name
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{f16_name}.json",
        f"{f32_name}.json",
    ]
    for dtype in (torch.bfloat16, torch.float32):
        dense_kwargs = _ssd_combined_state_kwargs(slots=False, dtype=dtype)
        assert (
            _ssd_combined_definition_name(dense_kwargs)
            == "ssd_combined_batched_d_none_final_h2_d64_g2_s128"
        )

    calls = []

    @functools.wraps(module.ssd_combined_fwd)
    def cpu_original(*args, **kwargs):
        calls.append("original")
        return kwargs["x"], kwargs["final_states"]

    def solution_for(tag):
        return _ssd_combined_tag_solution(calls, tag)

    monkeypatch.setattr(module, "ssd_combined_fwd", cpu_original)
    assert ta.enable_apply(
        {f16_name: solution_for("f16"), f32_name: solution_for("f32")}
    )
    module.ssd_combined_fwd(**f32_kwargs)
    module.ssd_combined_fwd(**f16_kwargs)
    assert calls == ["f32", "f16"]
    assert ta.stats().get(SSD_FI_API, {}).get("hit", 0) == 2

    # Only the f16 name registered: the f32 call reaches the original.
    calls.clear()
    ta.enable_apply({f16_name: solution_for("f16")})
    module.ssd_combined_fwd(**f32_kwargs)
    module.ssd_combined_fwd(**f16_kwargs)
    assert calls == ["original", "f16"]
    assert ta.stats().get(SSD_FI_API, {}).get("fallback_no_candidate", 0) == 1


@pytest.mark.cpu_only
def test_ssd_combined_apply_routes_slot_calls_by_seq_idx_dtype(monkeypatch, tmp_path):
    """A runner accepts only the seq_idx dtype it was built for, so slot calls
    with int32 and int64 seq_idx have different definition names (the
    cu_seqlens form takes the int64 name), each definition records its
    seq_idx dtype, and each call reaches only its own solution. The dense
    varlen names carry no seq_idx dtype."""
    import functools

    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    from flashinfer.trace.templates.mamba import ssd_combined_trace_dispatch

    i32_kwargs = _ssd_combined_seq_idx_kwargs(torch.int32)
    i64_kwargs = _ssd_combined_seq_idx_kwargs(torch.int64)
    cu_seqlens_kwargs = _ssd_combined_state_kwargs(slots=True, dtype=torch.float32)
    i32_name = "ssd_combined_varlen_slots_f32_sidx_i32_d_none_final_h2_d64_g2_s128"
    i64_name = "ssd_combined_varlen_slots_f32_sidx_i64_d_none_final_h2_d64_g2_s128"
    for kwargs, name, seq_idx_dtype in (
        (i32_kwargs, i32_name, "int32"),
        (i64_kwargs, i64_name, "int64"),
        (cu_seqlens_kwargs, i64_name, "int64"),
    ):
        assert _ssd_combined_definition_name(kwargs) == name
        definition = module.ssd_combined_fwd.fi_trace(save_dir=tmp_path, **kwargs)
        assert definition["name"] == name
        assert definition["inputs"]["seq_idx"]["dtype"] == seq_idx_dtype
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{i32_name}.json",
        f"{i64_name}.json",
    ]
    with pytest.raises(ValueError, match="seq_idx dtype"):
        ssd_combined_trace_dispatch(**_ssd_combined_seq_idx_kwargs(torch.int16))
    for dtype in (torch.int32, torch.int64):
        dense_kwargs = _ssd_combined_state_kwargs(slots=False, dtype=torch.float32)
        dense_kwargs["seq_idx"] = torch.zeros(2, 16, dtype=dtype)
        assert (
            _ssd_combined_definition_name(dense_kwargs)
            == "ssd_combined_varlen_d_none_final_h2_d64_g2_s128"
        )

    calls = []

    @functools.wraps(module.ssd_combined_fwd)
    def cpu_original(*args, **kwargs):
        calls.append("original")
        return kwargs["x"], kwargs["final_states"]

    monkeypatch.setattr(module, "ssd_combined_fwd", cpu_original)
    assert ta.enable_apply(
        {
            i32_name: _ssd_combined_tag_solution(calls, "i32"),
            i64_name: _ssd_combined_tag_solution(calls, "i64"),
        }
    )
    module.ssd_combined_fwd(**i64_kwargs)
    module.ssd_combined_fwd(**i32_kwargs)
    module.ssd_combined_fwd(**cu_seqlens_kwargs)
    assert calls == ["i64", "i32", "i64"]
    assert ta.stats().get(SSD_FI_API, {}).get("hit", 0) == 3


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("name", "dtype"),
    (
        ("final_states", torch.float32),
        ("checkpoint_states", torch.float32),
        ("initial_states", torch.bfloat16),
    ),
)
def test_ssd_combined_apply_rejects_mixed_slot_state_dtypes(monkeypatch, name, dtype):
    """A slot call whose state tensors do not share the state dtype reaches
    the original API (which raises before any write), never the solution of
    the definition its first state tensor names."""
    import functools

    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    from flashinfer.trace.templates.mamba import ssd_combined_trace_dispatch

    f16_kwargs = _ssd_combined_state_kwargs(slots=True, dtype=torch.float16)
    f16_name = "ssd_combined_varlen_slots_f16_sidx_i64_d_none_final_h2_d64_g2_s128"
    assert _ssd_combined_definition_name(f16_kwargs) == f16_name
    mixed = dict(f16_kwargs)
    mixed[name] = mixed[name].to(dtype)
    # An explicit state_dtype that disagrees with consistent pools.
    explicit = dict(f16_kwargs, state_dtype=torch.float32)
    for kwargs in (mixed, explicit):
        with pytest.raises(ValueError, match="doesn't match state_dtype"):
            ssd_combined_trace_dispatch(**kwargs)

    calls = []

    @functools.wraps(module.ssd_combined_fwd)
    def cpu_original(*args, **kwargs):
        calls.append("original")
        raise ValueError("state dtype mismatch")

    monkeypatch.setattr(module, "ssd_combined_fwd", cpu_original)
    assert ta.enable_apply({f16_name: _ssd_combined_tag_solution(calls, "f16")})
    for kwargs in (mixed, explicit):
        with pytest.raises(ValueError, match="state dtype mismatch"):
            module.ssd_combined_fwd(**kwargs)
    assert calls == ["original", "original"]
    assert ta.stats().get(SSD_FI_API, {}).get("hit", 0) == 0
    module.ssd_combined_fwd(**f16_kwargs)
    assert calls == ["original", "original", "f16"]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("slots_name", "pool_name"),
    (
        ("initial_state_slots", "initial_states"),
        ("final_state_slots", "final_states"),
    ),
)
def test_ssd_combined_apply_rejects_slot_vector_without_pool(
    monkeypatch, slots_name, pool_name
):
    """A slot call without the pool of one of its slot vectors reaches the
    original API (which raises before any write), never the solution of the
    definition the complete call names, though its other pool still gives
    the same state dtype."""
    import functools

    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    from flashinfer.trace.templates.mamba import ssd_combined_trace_dispatch

    complete = _ssd_combined_state_kwargs(slots=True, dtype=torch.float16)
    f16_name = "ssd_combined_varlen_slots_f16_sidx_i64_d_none_final_h2_d64_g2_s128"
    assert _ssd_combined_definition_name(complete) == f16_name
    missing = dict(complete)
    del missing[pool_name]
    with pytest.raises(ValueError, match=f"{slots_name} requires {pool_name}"):
        ssd_combined_trace_dispatch(**missing)

    calls = []

    @functools.wraps(module.ssd_combined_fwd)
    def cpu_original(*args, **kwargs):
        calls.append("original")
        raise ValueError("slot vector without its state pool")

    monkeypatch.setattr(module, "ssd_combined_fwd", cpu_original)
    assert ta.enable_apply({f16_name: _ssd_combined_tag_solution(calls, "f16")})
    with pytest.raises(ValueError, match="without its state pool"):
        module.ssd_combined_fwd(**missing)
    assert calls == ["original"]
    assert ta.stats().get(SSD_FI_API, {}).get("hit", 0) == 0
    module.ssd_combined_fwd(**complete)
    assert calls == ["original", "f16"]


@pytest.mark.cpu_only
def test_ssd_combined_bool_and_uint8_flags_share_one_slot_contract(
    monkeypatch, tmp_path
):
    """The API reads bool and uint8 ``initial_state_flags`` as the same bytes.
    Both trace to one slot definition that records uint8 flags, and Apply
    hands its solution uint8 flags: a view of the caller's bool flags, or the
    caller's uint8 flags unchanged. The original API keeps the caller's
    tensor."""
    import functools

    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    f32_name = "ssd_combined_varlen_slots_f32_sidx_i64_d_none_final_h2_d64_g2_s128"
    f16_name = "ssd_combined_varlen_slots_f16_sidx_i64_d_none_final_h2_d64_g2_s128"
    bool_flags = torch.tensor([True, False])
    uint8_flags = torch.tensor([1, 0], dtype=torch.uint8)
    bool_kwargs = _ssd_combined_state_kwargs(slots=True, dtype=torch.float32)
    bool_kwargs["initial_state_flags"] = bool_flags
    uint8_kwargs = dict(bool_kwargs, initial_state_flags=uint8_flags)
    for kwargs in (bool_kwargs, uint8_kwargs):
        assert _ssd_combined_definition_name(kwargs) == f32_name
        definition = module.ssd_combined_fwd.fi_trace(save_dir=tmp_path, **kwargs)
        assert definition["name"] == f32_name
        assert definition["inputs"]["initial_state_flags"]["dtype"] == "uint8"
    assert [path.name for path in tmp_path.iterdir()] == [f"{f32_name}.json"]
    assert bool_kwargs["initial_state_flags"] is bool_flags

    seen = []

    def solution(**inputs):
        seen.append(inputs["initial_state_flags"])
        return inputs["x"], inputs["final_states"]

    @functools.wraps(module.ssd_combined_fwd)
    def cpu_original(*args, **kwargs):
        seen.append(kwargs["initial_state_flags"])
        return kwargs["x"], kwargs["final_states"]

    monkeypatch.setattr(module, "ssd_combined_fwd", cpu_original)
    assert ta.enable_apply({f32_name: solution})
    module.ssd_combined_fwd(**bool_kwargs)
    module.ssd_combined_fwd(**uint8_kwargs)
    assert ta.stats().get(SSD_FI_API, {}).get("hit", 0) == 2
    bool_seen, uint8_seen = seen
    assert bool_seen.dtype == torch.uint8
    assert bool_seen.data_ptr() == bool_flags.data_ptr()
    assert bool_seen.tolist() == [1, 0]
    assert uint8_seen is uint8_flags

    # No solution for this name: the original API gets the caller's bool flags.
    seen.clear()
    ta.enable_apply({f16_name: solution})
    module.ssd_combined_fwd(**bool_kwargs)
    assert len(seen) == 1 and seen[0] is bool_flags


@pytest.mark.cpu_only
def test_ssd_combined_templates_keep_main_output_order():
    """The positional destination order of every SSDCombined schema is the
    dense order of the base revision: out, checkpoint_states, seq_chunk_cumsum
    (varlen), final_states. The API's ``(out, final_states)`` tuple binds by
    name through ``return_outputs``."""
    from flashinfer.trace.templates.mamba import ssd_combined_trace_dispatch

    for template in ssd_combined_trace_dispatch.templates:
        prefix = template.name_prefix
        varlen = prefix.startswith("ssd_combined_varlen")
        final = not prefix.endswith("_no_final")
        assert adapt._tensor_output_keys(template) == [
            "out",
            "checkpoint_states",
            *(["seq_chunk_cumsum"] if varlen else []),
            *(["final_states"] if final else []),
        ], prefix
        assert template.return_outputs == ("out", "final_states"), prefix
        if final:
            assert template.outputs["final_states"].param == "final_states"
    # The dense schemas of the base revision, pinned.
    dense = {
        template.name_prefix: adapt._tensor_output_keys(template)
        for template in ssd_combined_trace_dispatch.templates
        if "slots_" not in template.name_prefix
    }
    for layout in ("none", "vector", "matrix"):
        assert dense[f"ssd_combined_batched_d_{layout}_no_final"] == [
            "out",
            "checkpoint_states",
        ]
        assert dense[f"ssd_combined_batched_d_{layout}_final"] == [
            "out",
            "checkpoint_states",
            "final_states",
        ]
        assert dense[f"ssd_combined_varlen_d_{layout}_no_final"] == [
            "out",
            "checkpoint_states",
            "seq_chunk_cumsum",
        ]
        assert dense[f"ssd_combined_varlen_d_{layout}_final"] == [
            "out",
            "checkpoint_states",
            "seq_chunk_cumsum",
            "final_states",
        ]
    assert len(dense) == 12


@pytest.mark.cpu_only
@pytest.mark.parametrize("checkpoint_rows", (5, 7), ids=("same_shape", "seven_rows"))
def test_ssd_combined_value_returning_slot_solution_keeps_checkpoints(
    checkpoint_rows,
):
    # The returned final pool goes into the caller's final pool, never into the
    # checkpoint pool (silently overwritten when the shapes match, a failed
    # copy after the solution ran otherwise).
    kwargs = _ssd_combined_kwargs(slots=True, checkpoint_rows=checkpoint_rows)
    template, namespace, dests = _ssd_combined_apply(kwargs)
    # Apply routes by definition name, which carries the pools' state dtype.
    assert template.name_prefix.startswith("ssd_combined_varlen_slots_f32_")
    # The pools are required inputs.
    assert {("initial_states", "float32"), ("final_states", "float32")} <= set(
        adapt.extract_input_dtypes(template, namespace)
    )
    pool, checkpoints = kwargs["final_states"], kwargs["checkpoint_states"]
    checkpoints_before = checkpoints.clone()
    out_value = torch.randn_like(kwargs["x"])
    new_pool = torch.randn_like(pool)

    def sol(**inputs):
        assert inputs["final_states"] is pool
        assert inputs["checkpoint_states"] is checkpoints
        return out_value, new_pool

    out, final = adapt.adapt_and_call(
        template=template,
        fn=sol,
        namespace=namespace,
        dps=False,
        is_inplace=False,
        dests=dests,
    )
    assert out is out_value
    assert final is pool and torch.equal(pool, new_pool)
    assert torch.equal(checkpoints, checkpoints_before)


@pytest.mark.cpu_only
def test_ssd_combined_value_returning_dense_solution_keeps_checkpoints():
    # A dense call: the returned final states go to the caller's final_states
    # when given (else they are returned as is), never into the checkpoint
    # buffer (of the same shape here) or the caller's out buffer.
    kwargs = _ssd_combined_kwargs(slots=False, checkpoint_rows=2)
    kwargs["out"] = torch.empty_like(kwargs["x"])
    out_value = torch.randn_like(kwargs["x"])
    final_value = torch.randn_like(kwargs["checkpoint_states"])
    checkpoints = kwargs["checkpoint_states"]
    checkpoints_before = checkpoints.clone()

    def call(kwargs):
        template, namespace, dests = _ssd_combined_apply(kwargs)
        assert template.name_prefix.startswith("ssd_combined_batched_")
        assert dests == {
            "out": "out",
            "final_states": "final_states",
            "checkpoint_states": "checkpoint_states",
        }
        return adapt.adapt_and_call(
            template=template,
            fn=lambda **inputs: (out_value, final_value),
            namespace=namespace,
            dps=False,
            is_inplace=False,
            dests=dests,
        )

    out, final = call(kwargs)
    assert out is kwargs["out"] and torch.equal(out, out_value)
    assert final is final_value
    assert torch.equal(checkpoints, checkpoints_before)
    kwargs["final_states"] = torch.empty_like(final_value)
    out, final = call(kwargs)
    assert final is kwargs["final_states"] and torch.equal(final, final_value)
    assert torch.equal(checkpoints, checkpoints_before)


@pytest.mark.cpu_only
def test_ssd_combined_positional_dps_slot_solution_writes_the_final_pool():
    # A destination-passing solution receives out, the checkpoint pool and
    # then the caller's final pool as the final_states destination; the
    # call returns the API's (out, final_states).
    kwargs = _ssd_combined_kwargs(slots=True, checkpoint_rows=7)
    template, namespace, dests = _ssd_combined_apply(kwargs)
    pool, checkpoints = kwargs["final_states"], kwargs["checkpoint_states"]
    checkpoints_before = checkpoints.clone()
    num_inputs = len(template.inputs)
    destinations = []

    def sol(*args):
        destinations.extend(args[num_inputs:])
        args[num_inputs + 2].fill_(1.0)

    result = adapt.adapt_and_call(
        template=template,
        fn=sol,
        namespace=namespace,
        dps=True,
        is_inplace=False,
        dests=dests,
        positional=True,
    )
    out_buffer, checkpoint_buffer, final_buffer = destinations
    assert out_buffer.shape == kwargs["x"].shape
    assert final_buffer is pool and checkpoint_buffer is checkpoints
    assert len(result) == 2
    assert result[0] is out_buffer and result[1] is pool
    assert bool((pool == 1.0).all())
    assert torch.equal(checkpoints, checkpoints_before)


@pytest.mark.cpu_only
def test_ssd_combined_positional_dps_dense_solution_keeps_main_positions():
    # A dense destination-passing solution built against the base revision's
    # definition receives out, checkpoint_states and final_states at the
    # same positions; each write lands in the matching caller buffer and the
    # call returns the API's (out, final_states).
    kwargs = _ssd_combined_kwargs(slots=False, checkpoint_rows=2)
    kwargs["out"] = torch.empty_like(kwargs["x"])
    # Same shape as the checkpoint buffer, so a swap would not raise.
    kwargs["final_states"] = torch.zeros_like(kwargs["checkpoint_states"])
    template, namespace, dests = _ssd_combined_apply(kwargs)
    assert template.name_prefix == "ssd_combined_batched_d_none_final"
    num_inputs = len(template.inputs)
    destinations = []

    def sol(*args):
        destinations.extend(args[num_inputs:])
        out, checkpoint_states, final_states = args[num_inputs:]
        out.fill_(1.0)
        checkpoint_states.fill_(2.0)
        final_states.fill_(3.0)

    result = adapt.adapt_and_call(
        template=template,
        fn=sol,
        namespace=namespace,
        dps=True,
        is_inplace=False,
        dests=dests,
        positional=True,
    )
    out_buffer, checkpoint_buffer, final_buffer = destinations
    assert out_buffer is kwargs["out"]
    assert checkpoint_buffer is kwargs["checkpoint_states"]
    assert final_buffer is kwargs["final_states"]
    assert bool((kwargs["out"] == 1.0).all())
    assert bool((kwargs["checkpoint_states"] == 2.0).all())
    assert bool((kwargs["final_states"] == 3.0).all())
    assert len(result) == 2
    assert result[0] is kwargs["out"] and result[1] is kwargs["final_states"]


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "family,bound,scale_types,scalar_tensor",
    [
        ("decode", False, (True, False), True),
        ("mla", False, (False, True), False),
        ("block_sparse", False, (True, True), True),
        ("paged_block_sparse", False, (True, False), False),
        ("block_sparse", True, (False, True), True),
        ("paged_block_sparse", True, (True, True), False),
    ],
)
def test_prims_ts_device_scale_routes(family, bound, scale_types, scalar_tensor):
    from tests.trace.test_prims_ts_global_scales import _trace_case

    def make_scales(scalar, host_value):
        return tuple(
            torch.empty(() if scalar else (1,), device="meta", dtype=torch.float32)
            if tensor
            else host_value
            for tensor in scale_types
        )

    scales = make_scales(scalar_tensor, 0.25)
    api, kwargs, names = _trace_case(family, bound, scales)
    definition = flashinfer.fi_trace(api, **kwargs)
    calls = []
    query = kwargs.get("q", kwargs.get("query"))
    expected_output = torch.full_like(query, SENTINEL)

    def solution(**inputs):
        calls.append(inputs)
        return expected_output

    ta.enable_apply({definition["name"]: solution})
    # Resolve the public callable after enable_apply installs the replacement.
    target = (
        api.__self__.run
        if bound
        else getattr(importlib.import_module(api.__module__), api.__name__)
    )

    def check_call(arguments, expected_scales):
        assert target(**arguments) is expected_output
        received = calls[-1]
        for name, expected in zip(names, expected_scales, strict=True):
            actual = received[name]
            if isinstance(expected, torch.Tensor):
                # Meta operands cannot be read back. Both accepted input ranks
                # reach the solution in the traced one-element tensor ABI.
                assert actual.shape == (1,)
                assert actual is expected or actual._base is expected
            else:
                assert actual == expected

    check_call(kwargs, scales)
    if bound:
        overrides = make_scales(not scalar_tensor, 0.75)
        check_call({**kwargs, **dict(zip(names, overrides, strict=True))}, overrides)
        check_call(kwargs, scales)
    fi_api = next(
        tag.removeprefix("fi_api:")
        for tag in definition["tags"]
        if tag.startswith("fi_api:")
    )
    assert ta.stats()[fi_api]["hit"] == (3 if bound else 1)


def _assert_prims_ts_scale_fallback(
    api, kwargs, definition, error_type, expected_error
):
    # Establish the real API's rejection without requiring a GPU or replacing
    # its implementation. Apply must preserve that fallback, not run a solution.
    with pytest.raises(error_type, match=expected_error):
        api(**kwargs)
    calls = []

    def solution(**inputs):
        calls.append(inputs)
        return kwargs["q"]

    ta.enable_apply({definition["name"]: solution})
    target = (
        api.__self__.run
        if hasattr(api, "__self__")
        else getattr(importlib.import_module(api.__module__), api.__name__)
    )
    with pytest.raises(error_type, match=expected_error):
        target(**kwargs)
    assert not calls


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "bound,scale_types,scale_index,invalid,error_type",
    [
        pytest.param(False, (False, False), 0, 0.0, ValueError, id="host-zero-ff"),
        pytest.param(True, (False, False), 1, True, TypeError, id="host-bool-ff-bound"),
        pytest.param(
            False, (True, False), 1, float("inf"), ValueError, id="host-inf-tf"
        ),
        pytest.param(
            True, (False, True), 0, 1e-50, ValueError, id="host-underflow-ft-bound"
        ),
        pytest.param(
            False,
            (True, False),
            0,
            ((2,), torch.float32, "cuda:0"),
            ValueError,
            id="shape-tf",
        ),
        pytest.param(
            True,
            (False, True),
            1,
            ((1,), torch.float16, "cuda:0"),
            ValueError,
            id="dtype-ft-bound",
        ),
        pytest.param(
            False,
            (True, False),
            0,
            ((1,), torch.float32, "cpu"),
            ValueError,
            id="device-tf",
        ),
        pytest.param(
            True,
            (False, True),
            1,
            ((1,), torch.float32, "cuda:1"),
            ValueError,
            id="device-ft-bound",
        ),
    ],
)
def test_prims_ts_invalid_scale_falls_back(
    bound, scale_types, scale_index, invalid, error_type
):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from tests.trace.test_prims_ts_global_scales import _trace_case

    with FakeTensorMode():
        scales = tuple(
            torch.empty(1, device="cuda:0") if tensor else 0.25
            for tensor in scale_types
        )
        if isinstance(invalid, tuple):
            shape, dtype, device = invalid
            invalid = torch.empty(shape, dtype=dtype, device=device)
    api, kwargs, names = _trace_case(
        "block_sparse" if bound else "decode", bound, scales
    )
    if bound:
        state = api.__self__._plan_state
        state.device = torch.device("cuda:0")
        state.use_kv_valid_bits = False
        state.dummy_kv_valid_bits = None
        kwargs.update(validate=False, out=kwargs["q"])
        expected_error = names[scale_index]
    else:
        with FakeTensorMode():
            kwargs["q"] = torch.empty(
                kwargs["q"].shape, dtype=kwargs["q"].dtype, device="cuda:0"
            )
        expected_error = "block_tables must be a CUDA tensor"
    definition = flashinfer.fi_trace(api, **kwargs)
    kwargs[names[scale_index]] = invalid
    _assert_prims_ts_scale_fallback(api, kwargs, definition, error_type, expected_error)


@pytest.mark.cpu_only
def test_enable_apply_real_kernel_numeric():
    """A genuine torch RMSNorm callable; assert the substituted output matches."""
    ta.enable_apply({"rmsnorm_h16": _torch_rmsnorm})
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    w = torch.randn(16)
    out = flashinfer.norm.rmsnorm(x, w)
    assert torch.allclose(out, _ref_rmsnorm(x, w), rtol=1e-4, atol=1e-4)
    assert ta.stats().get(FI_API, {}).get("hit", 0) >= 1


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_strict_mode_matched_solution_error_raises():
    """A matched solution that raises must propagate (not silently fall back)."""

    def boom(hidden_states, weight):
        raise ValueError("boom")

    ta.enable_apply({"rmsnorm_h16": boom})
    with pytest.raises(ValueError, match="boom"):
        flashinfer.norm.rmsnorm(torch.ones(8, 16), torch.ones(16))
    assert ta.stats().get(FI_API, {}).get("error", 0) >= 1


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_solutions_positional_and_keyword_equivalent():
    def sol(hidden_states, weight):
        return torch.full_like(hidden_states, SENTINEL)

    n1 = ta.enable_apply({"rmsnorm_h16": sol})  # positional dict
    ta.disable_apply()
    n2 = ta.enable_apply(solutions={"rmsnorm_h16": sol})  # keyword
    assert n1 == n2 and n1 >= 1
    out = flashinfer.norm.rmsnorm(torch.ones(8, 16), torch.ones(16))
    assert torch.allclose(out, torch.full_like(out, SENTINEL))


@pytest.mark.cpu_only
def test_enable_apply_empty_is_noop():
    assert ta.enable_apply({}) == 0
    assert not ta.is_enabled()


@pytest.mark.cpu_only
def test_multi_template_apply_uses_only_selected_metadata(monkeypatch):
    """A runtime-selected template owns extraction, plan state, and outputs."""

    from flashinfer.api_logging import _TRACE_DISPATCHERS
    import flashinfer.trace_apply.apply as apply_module
    from flashinfer.trace.template import Const, Tensor, TraceTemplate
    from flashinfer.trace_apply import plan_capture
    from flashinfer.trace_apply.plan_capture import StatefulAdapter

    def make_template(name, state_param, *, destination=True):
        output = (
            Tensor(["width"], dtype_from="state", param="target")
            if destination
            else Tensor(["width"], dtype_from="state")
        )
        return TraceTemplate(
            op_type="synthetic_multi_template",
            name_prefix=name,
            axes={"width": Const(abbrev="w")},
            inputs={"state": Tensor(["width"], param=state_param)},
            outputs={"output": output},
        )

    other = make_template("other", "other_state")
    broken = make_template("broken", "broken_state")
    selected = make_template("selected", "selected_state")
    no_destination = make_template(
        "no_destination", "no_destination_state", destination=False
    )
    unknown = make_template("unknown", "unknown_state")

    def fail_to_build_extractors():
        raise ValueError("invalid template")

    monkeypatch.setattr(broken, "_build_axis_extractors", fail_to_build_extractors)
    templates = [other, broken, selected, no_destination]
    extractor_maps = build_extractor_maps(templates)
    state = torch.empty(4)
    assert extractor_maps[1] is None
    assert extract_axes(extractor_maps, {"selected_state": state}) == {"width": 4}

    class Owner:
        def plan(self):
            return None

        def run(self, target, other_state=None) -> None:
            del other_state
            target.fill_(-1)

    original = Owner.run
    modes = {
        "broken": broken,
        "selected": selected,
        "no_destination": no_destination,
        "unknown": unknown,
    }

    def dispatch(*, self, save_dir=None, name=None, **_kwargs):
        del save_dir, name
        return modes[self.mode]

    fi_api = "tests.synthetic_multi_template.Owner.run"
    adapter = StatefulAdapter(plan_attr="plan", self_attrs={"state": "_state"})
    monkeypatch.setattr(
        apply_module,
        "_registry_by_fi_api",
        lambda: {fi_api: (original, templates)},
    )
    monkeypatch.setattr(apply_module, "_resolve_target", lambda _fi_api: (Owner, "run"))
    monkeypatch.setattr(apply_module, "_build_alias_map", lambda: {})
    monkeypatch.setitem(_TRACE_DISPATCHERS, original, dispatch)
    monkeypatch.setattr(plan_capture, "is_stateful", lambda name: name == fi_api)
    monkeypatch.setattr(plan_capture, "adapter_for", lambda _name: adapter)

    def should_not_run(**_kwargs):
        raise AssertionError("unselected solution was called")

    def selected_solution(*, state):
        return torch.ones_like(state)

    assert (
        ta.enable_apply(
            {
                "broken_w4": should_not_run,
                "broken": should_not_run,
                "selected_w4": selected_solution,
                "no_destination_w4": should_not_run,
                "unknown_w4": should_not_run,
            }
        )
        == 1
    )

    owner = Owner()
    owner._state = state
    target = torch.zeros_like(state)
    owner.mode = "selected"
    assert owner.run(target, other_state=torch.empty(7)) is None
    torch.testing.assert_close(target, torch.ones_like(target))

    for mode in ("broken", "no_destination", "unknown"):
        owner.mode = mode
        target.zero_()
        assert owner.run(target) is None
        torch.testing.assert_close(target, -torch.ones_like(target))


# ===========================================================================
# Live MoE APIs: the per-call template dispatcher picks S=0 vs S>0
# ===========================================================================

MOE_FP8_API = "flashinfer.fused_moe.core.trtllm_fp8_block_scale_moe"
MOE_FP4_API = "flashinfer.fused_moe.core.trtllm_fp4_block_scale_moe"
_DEEPSEEK_V3 = 2


def _fp8_moe_kwargs(num_shared, *, T=8, E=32, H=256, I=128):
    rows, bs = E + num_shared, 128
    return dict(
        routing_logits=torch.randn(T, E, dtype=torch.float32),
        routing_bias=torch.zeros(E, dtype=torch.bfloat16),
        hidden_states=torch.zeros(T, H, dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.ones(H // bs, T, dtype=torch.float32),
        gemm1_weights=torch.zeros(rows, 2 * I, H, dtype=torch.float8_e4m3fn),
        gemm1_weights_scale=torch.ones(rows, 2 * I // bs, H // bs, dtype=torch.float32),
        gemm2_weights=torch.zeros(rows, H, I, dtype=torch.float8_e4m3fn),
        gemm2_weights_scale=torch.ones(rows, H // bs, I // bs, dtype=torch.float32),
        num_experts=E,
        top_k=8,
        n_group=8,
        topk_group=4,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=2.5,
        routing_method_type=_DEEPSEEK_V3,
        num_fused_shared_experts=num_shared,
    )


def _fp4_moe_kwargs(num_shared, *, T=2, E=8, H=256, I=128):
    rows = E + num_shared
    ones = torch.ones(rows, dtype=torch.float32)
    return dict(
        routing_logits=torch.randn(T, E, dtype=torch.bfloat16),
        routing_bias=torch.zeros(E, dtype=torch.bfloat16),
        hidden_states=torch.zeros(T, H // 2, dtype=torch.uint8),
        hidden_states_scale=torch.ones(T, H // 16, dtype=torch.float8_e4m3fn),
        gemm1_weights=torch.zeros(rows, 2 * I, H // 2, dtype=torch.uint8),
        gemm1_weights_scale=torch.ones(rows, 2 * I, H // 16, dtype=torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=ones,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=torch.zeros(rows, H, I // 2, dtype=torch.uint8),
        gemm2_weights_scale=torch.ones(rows, H, I // 16, dtype=torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=ones,
        output1_scale_gate_scalar=ones,
        output2_scale_scalar=ones,
        num_experts=E,
        top_k=2,
        n_group=2,
        topk_group=1,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=2.5,
        routing_method_type=_DEEPSEEK_V3,
        num_fused_shared_experts=num_shared,
    )


def _moe_definition_name(fi_api, call_kwargs):
    """The name the wrapper computes: dispatcher-selected template + its axes."""
    from flashinfer.api_logging import _TRACE_DISPATCHERS

    original, _templates = _registry_by_fi_api()[fi_api]
    namespace = bind_namespace(original, (), call_kwargs)
    template = _TRACE_DISPATCHERS[original](save_dir=None, name=None, **namespace)
    axes = extract_axes(build_extractor_maps([template]), namespace)
    return template.definition_name(axes)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "fi_api, make_kwargs",
    [(MOE_FP8_API, _fp8_moe_kwargs), (MOE_FP4_API, _fp4_moe_kwargs)],
    ids=["fp8", "fp4"],
)
def test_moe_apply_dispatches_shared_experts_sibling(monkeypatch, fi_api, make_kwargs):
    """S>0 and S=0 calls of one API resolve to different definitions, and a
    name registered for only one of them falls back to the original for the other."""
    import functools

    import flashinfer.fused_moe.core as core

    attr = fi_api.rsplit(".", 1)[1]
    shared_name = _moe_definition_name(fi_api, make_kwargs(1))
    routed_name = _moe_definition_name(fi_api, make_kwargs(0))
    assert "shared_experts" in shared_name and "_s1_" in shared_name
    assert "shared_experts" not in routed_name

    calls = []

    @functools.wraps(getattr(core, attr))
    def cpu_original(*args, **kwargs):
        calls.append("original")
        return torch.zeros(1)

    def shared_solution(**kwargs):
        calls.append("shared")
        return torch.zeros(1)

    def routed_solution(**kwargs):
        calls.append("routed")
        return torch.zeros(1)

    monkeypatch.setattr(core, attr, cpu_original)
    assert ta.enable_apply({shared_name: shared_solution, routed_name: routed_solution})
    getattr(core, attr)(**make_kwargs(1))
    getattr(core, attr)(**make_kwargs(0))
    assert calls == ["shared", "routed"]
    assert ta.stats().get(fi_api, {}).get("hit", 0) == 2

    # Only the S>0 name registered: S=0 must reach the original, not the sibling.
    calls.clear()
    ta.enable_apply({shared_name: shared_solution})
    getattr(core, attr)(**make_kwargs(0))
    getattr(core, attr)(**make_kwargs(1))
    assert calls == ["original", "shared"]
    assert ta.stats().get(fi_api, {}).get("fallback_no_candidate", 0) >= 1
