"""An in-process compile never inherits the artifacts that compile planning left behind."""

from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from b12x._lib import compile_plan, program_cache
from b12x._lib.compile_plan import DeferredCuTeKernel, DeferredTritonKernel, ProgramKey


def test_factory_cache_is_job_local_and_normalizes_mapping_order():
    from b12x.preparation import FrozenMapping

    built = []

    @program_cache.program_cache(scope="preparation")
    def factory(payload, ordinal):
        value = SimpleNamespace(payload=payload, ordinal=ordinal)
        built.append(value)
        return value

    first, second = (program_cache.PreparationProgramCache() for _ in range(2))
    with first.activate():
        value = factory({"shape": (3, 8), "mode": "a16"}, 0)
        assert factory(FrozenMapping({"mode": "a16", "shape": (3, 8)}), 0) is value
        with second.activate():
            assert factory({"shape": (3, 8), "mode": "a16"}, 0) is not value
        assert factory({"shape": (3, 8), "mode": "a16"}, 0) is value
        assert factory({"shape": (5, 8), "mode": "a16"}, 0) is not value
        assert factory({"shape": (3, 8), "mode": "a16"}, 1) is not value
    assert len(built) == 4
    first.clear()
    second.clear()
    assert factory({}, 0) is not factory({}, 0)
    assert factory.cache_info().currsize == 0


def test_factory_cache_rejects_tensor_keys_and_does_not_cache_errors():
    import torch

    attempts = []

    @program_cache.program_cache(scope="preparation")
    def factory(payload):
        attempts.append(payload)
        raise ValueError("invalid declaration")

    scope = program_cache.PreparationProgramCache()
    with scope.activate():
        with pytest.raises(TypeError, match="requires metadata"):
            factory(torch.empty(0))
        for _ in range(2):
            with pytest.raises(ValueError, match="invalid declaration"):
                factory({"rows": 3})
    assert len(attempts) == 2
    assert scope.clear() == 0


def test_factory_hit_loads_programs_into_each_owner_and_clear_releases_bundle():
    key = ProgramKey("cute", "9" * 64, "shared")

    @program_cache.program_cache(scope="preparation")
    def factory():
        return compile_plan.CompiledCuTeProgram(lambda: 17, key)

    scope = program_cache.PreparationProgramCache()
    with scope.activate():
        with compile_plan.retain_compiled_programs() as first:
            value = compile_plan.load_programs(factory())
        with compile_plan.retain_compiled_programs() as second:
            assert compile_plan.load_programs(factory()) is value
    reference = weakref.ref(value)
    del value
    scope.clear()
    first.owners.clear()
    assert reference()() == 17
    second.owners.clear()
    assert reference() is None


def test_factory_scope_does_not_leak_through_global_cache_registry():
    import gc

    @program_cache.program_cache(scope="preparation")
    def factory():
        return compile_plan.CompiledCuTeProgram(
            lambda: None, ProgramKey("cute", "8" * 64)
        )

    scope = program_cache.PreparationProgramCache()
    with scope.activate():
        reference = weakref.ref(factory())
    del scope
    gc.collect()
    assert reference() is None


def test_compile_factory_inventory_has_explicit_reuse_boundaries():
    import ast
    import re

    root = Path(__file__).resolve().parents[4]
    references = set()
    sources = {}
    for path in (root / "flashinfer/experimental/b12x").rglob("*.py"):
        tree = ast.parse(path.read_text())
        sources[path] = tree
        references.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"b12x\.[\w.]+:_?compile_\w+", node.value)
        )
    exceptions = {
        "b12x.norm.mhc._preparation:compile_mhc": "validates before scoped _compile_mhc",
        "b12x.gemm._shared.wo_mxfp8:compile_wo_quantizers": "registered executable bundle cache",
    }
    assert exceptions.keys() <= references
    assert len(references) >= 53
    for reference in references - exceptions.keys():
        module, name = reference.split(":")
        tree = sources[
            root / "flashinfer/experimental" / (module.replace(".", "/") + ".py")
        ]
        factory = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "program_cache"
            and any(
                keyword.arg == "scope"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "preparation"
                for keyword in decorator.keywords
            )
            for decorator in factory.decorator_list
        ), reference


@pytest.mark.parametrize("rows", [1, 3, 32, 33])
@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_compressed_mla_materialization_shares_launches_not_scratch_plans(
    monkeypatch, rows, mode
):
    from b12x.attention.compressed_sparse_mla import _preparation as mla
    from b12x.preparation import DetectedDevice, DeviceIdentity

    device = DetectedDevice(
        0, DeviceIdentity("nvidia", (12, 0), 188, "synthetic SM120")
    )
    caps = mla.B12XCompressedSparseMLAScratchCaps(
        device="cuda:0",
        num_q_heads=16,
        max_q_rows=rows,
        max_width=64,
        swa_width=64,
        indexed_width=0,
        mode=mode,
    )
    invocation = mla.invocation_from_descriptors(
        q=dict(
            shape=(rows, 16, 512), stride=(8192, 512, 1), alignment=16, dtype="bfloat16"
        ),
        swa_cache=dict(
            shape=(32, 40960), stride=(40960, 1), alignment=16, dtype="uint8"
        ),
        output_mode="provided",
    )
    plans = [mla.plan(caps, invocation=invocation) for _ in range(2)]
    config = mla.TUNING.configure(plans[0].query, device=device.identity).default
    built = []
    program = compile_plan.CompiledCuTeProgram(
        lambda: None, ProgramKey("cute", "5" * 64)
    )

    def lower(*_args):
        built.append(1)
        return SimpleNamespace(
            grid_programs=(program,), merge_program=None, lse_program=None
        )

    monkeypatch.setattr(mla, "_lower_launch", lower)
    monkeypatch.setattr(mla, "_lower_indexed_mapper", lambda *_args: None)
    scope = program_cache.PreparationProgramCache()
    with scope.activate(), compile_plan.retain_compiled_programs():
        states = [
            plan._materialize(SimpleNamespace(config=config), device) for plan in plans
        ]
    scope.clear()
    assert built == [1]
    assert states[0] is not states[1]
    assert states[0].scratch_plan is not states[1].scratch_plan
    assert states[0].launch is states[1].launch
    assert compile_plan.program_keys(states[0]) == program.__b12x_programs__


def test_evict_planning_artifacts_drops_unresolved_deferred_programs_only(monkeypatch):
    """Planning eviction drops only unresolved deferred compiler artifacts."""
    planned = DeferredCuTeKernel(ProgramKey("cute", "a" * 64, "k"), memory_key=("m",))
    resolved = DeferredCuTeKernel(ProgramKey("cute", "b" * 64, "k"), memory_key=("n",))
    resolved._resolved = object()
    carrier = SimpleNamespace()
    object.__setattr__(carrier, "__b12x_programs__", (planned.__b12x_programs__[0],))
    object.__setattr__(carrier, "__b12x_dependencies__", (planned,))
    memo = {
        "planned": planned,
        "carrier": carrier,
        "resolved": resolved,
        "compiled": object(),
    }
    mirror = {"planned": 1, "carrier": 2, "resolved": 3}
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [(memo, (mirror,), None)])
    triton_planned = DeferredTritonKernel(
        ProgramKey("triton", "c" * 64, "t"), source=SimpleNamespace(name="t")
    )
    triton_resolved = DeferredTritonKernel(
        ProgramKey("triton", "d" * 64, "t"), source=SimpleNamespace(name="t")
    )
    triton_resolved._resolved = object()
    kernel_cache = {"k1": triton_planned, "k2": triton_resolved}
    key_cache = {"sig1": "k1", "sig2": "k2"}
    jit = type(
        "Jit", (), {"device_caches": {0: (kernel_cache, key_cache, None, None)}}
    )()
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", {jit})

    removed = compile_plan.evict_planning_artifacts([planned.__b12x_programs__[0]])

    assert removed == 2
    assert set(memo) == {"resolved", "compiled"}
    assert set(mirror) == {"resolved"}
    assert set(kernel_cache) == {"k1", "k2"} and set(key_cache) == {"sig1", "sig2"}

    assert compile_plan.evict_planning_artifacts() == 1
    assert set(kernel_cache) == {"k2"} and set(key_cache) == {"sig2"}


def test_program_keys_skip_torch_scalars_inside_launch_records():
    """Program discovery ignores Torch metadata alongside compiled kernels."""
    import torch
    from typing import NamedTuple

    class Launch(NamedTuple):
        compiled: object
        dtype: torch.dtype
        table: torch.Tensor

    kernel = DeferredTritonKernel(
        ProgramKey("triton", "e" * 64, "t"), source=SimpleNamespace(name="t")
    )
    assert (
        compile_plan.program_keys(Launch(kernel, torch.int32, torch.zeros(1)))
        == kernel.__b12x_programs__
    )


def test_program_keys_accept_plain_functions_without_triton(monkeypatch):
    """Host closures without retained programs do not import optional Triton."""
    import builtins

    original_import = builtins.__import__

    def reject_triton(name, *args, **kwargs):
        """Fail if program-key discovery imports the optional Triton package."""
        if name == "triton.compiler.compiler":
            raise AssertionError("program_keys imported Triton for a plain function")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_triton)
    assert compile_plan.program_keys(lambda: None) == ()


def test_first_use_evicts_deferred_launchers_from_decorated_kernel_memos(monkeypatch):
    """Deferred launchers are evicted while resolved launcher memo entries remain."""
    planned = DeferredCuTeKernel(
        ProgramKey("cute", "f" * 64, "planned"), memory_key=("p",)
    )
    resolved = DeferredCuTeKernel(
        ProgramKey("cute", "1" * 64, "resolved"), memory_key=("r",)
    )
    resolved._resolved = object()
    calls = []

    @program_cache.program_cache
    def factory(name):
        calls.append(name)
        kernel = planned if name == "planned" else resolved
        return compile_plan.attach_programs(lambda: None, kernel)

    monkeypatch.setattr(program_cache, "_CACHES", {factory})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())
    deferred_launcher = factory("planned")
    resident_launcher = factory("resolved")

    assert compile_plan.evict_planning_artifacts(planned.__b12x_programs__) == 1
    assert factory("resolved") is resident_launcher
    assert factory("planned") is not deferred_launcher
    assert calls == ["planned", "resolved", "planned"]


def test_mhc_program_bundle_reuse_obeys_deferred_and_resident_reclamation(monkeypatch):
    """MHC program bundles preserve resolved entries across planning eviction."""
    from b12x.norm.mhc import _preparation as mhc
    from b12x.preparation import FrozenMapping

    key = ProgramKey("cute", "2" * 64, "mhc")
    built = []

    def build(*_args):
        program = DeferredCuTeKernel(key, memory_key=("mhc",))
        built.append(program)
        return {"partial": program}

    mhc._compile_mhc.cache_clear()
    monkeypatch.setattr(mhc._compile_mhc, "_function", build)
    monkeypatch.setattr(
        mhc, "_codegen_snapshot", lambda: FrozenMapping({"constant": 1})
    )
    monkeypatch.setattr(program_cache, "_CACHES", {mhc._compile_mhc})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())
    payload = {"codegen": {"constant": 1}}
    with program_cache.PreparationProgramCache().activate():
        planned = mhc.compile_mhc(payload, {}, {}, 0)
        assert mhc.compile_mhc(dict(payload), {}, {}, 0) is planned
        assert compile_plan.program_keys(planned) == (key,)
        assert compile_plan.evict_planning_artifacts((key,)) == 1
        resident = mhc.compile_mhc(payload, {}, {}, 0)
        assert resident is not planned
        resident["partial"]._resolved = object()
        assert compile_plan.evict_planning_artifacts((key,)) == 0
        assert mhc.compile_mhc(payload, {}, {}, 0) is resident
        assert program_cache.evict_unretained(frozenset({key})) == 0
        assert program_cache.evict_unretained(frozenset()) == 1
        assert len(built) == 2


def test_mxfp4_cache_releases_losing_programs_and_keeps_selected_program(monkeypatch):
    from b12x.attention.dsa_indexer import mxfp4

    winner = ProgramKey("cute", "3" * 64, "attention.indexer.mxfp4")
    loser = ProgramKey("cute", "4" * 64, "attention.indexer.mxfp4")
    keys = iter((winner, loser))
    monkeypatch.setattr(
        mxfp4,
        "b12x_compile",
        lambda *_a, **_k: compile_plan.CompiledCuTeProgram(lambda: None, next(keys)),
    )
    monkeypatch.setattr(mxfp4, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(mxfp4, "make_ptr", lambda *_a, **_k: None)
    monkeypatch.setattr(program_cache, "_CACHES", {mxfp4._compile})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    mxfp4._compile.cache_clear()
    try:
        selected = mxfp4._compile("quantize", (False, 64), 0)
        assert compile_plan.program_keys(selected) == (winner,)
        compile_plan.load_programs(selected)
        selected_ref = weakref.ref(selected.raw)
        del selected
        losing_ref = weakref.ref(mxfp4._compile("quantize", (True, 64), 0).raw)
        assert program_cache.evict_unretained({winner}) == 1
        assert losing_ref() is None
        assert mxfp4._compile("quantize", (False, 64), 0).raw is selected_ref()
        assert program_cache.evict_unretained(set()) == 1
        assert selected_ref() is None
    finally:
        mxfp4._compile.cache_clear()


def test_mhc_program_bundle_hit_still_validates_codegen_snapshot(monkeypatch):
    import pytest
    from b12x.norm.mhc import _preparation as mhc
    from b12x.preparation import FrozenMapping

    mhc._compile_mhc.cache_clear()
    monkeypatch.setattr(mhc._compile_mhc, "_function", lambda *_args: {})
    monkeypatch.setattr(
        mhc, "_codegen_snapshot", lambda: FrozenMapping({"constant": 1})
    )
    with program_cache.PreparationProgramCache().activate():
        payload = {"codegen": {"constant": 1}}
        mhc.compile_mhc(payload, {}, {}, 0)
        monkeypatch.setattr(
            mhc, "_codegen_snapshot", lambda: FrozenMapping({"constant": 2})
        )
        with pytest.raises(ValueError, match="code-generation snapshot"):
            mhc.compile_mhc(payload, {}, {}, 0)
