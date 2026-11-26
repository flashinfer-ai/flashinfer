"""Preparation jobs and sessions only evict programs they introduced or own."""

from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import pytest

from b12x._lib import compile_plan, program_cache
from b12x._lib.compile_plan import (
    CompiledCuTeProgram,
    DeferredCuTeKernel,
    DeferredTritonKernel,
    ProgramBundle,
    ProgramKey,
)
from b12x.preparation import PreparedCall, session as session_module
from b12x.preparation.session import prepare_default
from ..conftest import require_b12x
from .test_defaults import Config, contract
from .test_session import _deterministic_timer, declaration, session


class _FakeTritonKernel:
    def __init__(self, program_key: ProgramKey):
        self.__b12x_programs__ = (program_key,)


def _make_triton_jit(program_key: ProgramKey, cache_key: str = "k0"):
    kernel = _FakeTritonKernel(program_key)
    kernel_cache = {cache_key: kernel}
    key_cache = {"sig0": cache_key}
    jit = type(
        "Jit", (), {"device_caches": {0: (kernel_cache, key_cache, None, None)}}
    )()
    return jit, kernel, kernel_cache, key_cache


def _plan_with_program(
    factory, *, name="op", tuning=None, benchmark=None, fail=False, compile_jobs=None
):
    overrides = {
        "_materialize": lambda selection, _device: compile_plan.load_programs(
            factory(selection.config)
        ),
    }
    if compile_jobs is not None:
        overrides["_compile_jobs"] = compile_jobs
    plan = replace(declaration(tuning=tuning), **overrides)

    def prepare_call(state):
        if fail:
            raise RuntimeError("prepare_call failed")
        return PreparedCall(run=state)

    return plan.request(
        name=name,
        prepare_call=prepare_call,
        benchmark_call=benchmark,
    )


def test_prepare_default_preserves_preexisting_unplanned_programs(monkeypatch):
    """First-use prepare_default leaves pre-existing unplanned kernels intact."""
    monkeypatch.setattr(session_module, "_LAZY_SESSIONS", {})

    unplanned_proc_key = ProgramKey("cute", "a" * 64, "unplanned.process")
    unplanned_map_key = ProgramKey("cute", "b" * 64, "unplanned.mapping")
    unplanned_triton_key = ProgramKey("triton", "c" * 64, "unplanned.triton")
    late_unplanned_key = ProgramKey("cute", "d" * 64, "unplanned.between_calls")
    first_plan_key = ProgramKey("cute", "e" * 64, "lazy.first")
    second_plan_key = ProgramKey("cute", "f" * 64, "lazy.second")

    @program_cache.program_cache
    def proc_cache(tag):
        keys = {
            "unplanned": unplanned_proc_key,
            "late": late_unplanned_key,
            "first": first_plan_key,
            "second": second_plan_key,
        }
        return CompiledCuTeProgram(lambda tag=tag: tag, keys[tag])

    mapping_cache = {}
    mirror_cache = {}
    triton_jit, triton_kernel, kernel_cache, key_cache = _make_triton_jit(
        unplanned_triton_key
    )

    monkeypatch.setattr(program_cache, "_CACHES", {proc_cache})
    monkeypatch.setattr(
        program_cache, "_MAPPING_CACHES", [(mapping_cache, (mirror_cache,), None)]
    )
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", {triton_jit})

    unplanned_proc = proc_cache("unplanned")
    unplanned_map = CompiledCuTeProgram(lambda: "map", unplanned_map_key)
    mapping_cache["m0"] = unplanned_map
    mirror_cache["m0"] = 1

    proc_ref = weakref.ref(unplanned_proc)
    map_ref = weakref.ref(unplanned_map)
    triton_ref = weakref.ref(triton_kernel)
    del unplanned_proc, unplanned_map, triton_kernel

    req_first = _plan_with_program(lambda _cfg: proc_cache("first"), name="first")
    prepare_default(req_first)
    gc.collect()

    assert proc_ref() is not None
    assert map_ref() is not None
    assert triton_ref() is not None
    assert "m0" in mapping_cache and "m0" in mirror_cache
    assert "k0" in kernel_cache and "sig0" in key_cache

    # An unplanned kernel loaded between two prepare_default calls on the reused
    # lazy session must also survive the second first-use preparation.
    late_unplanned = proc_cache("late")
    late_ref = weakref.ref(late_unplanned)
    del late_unplanned

    req_second = _plan_with_program(lambda _cfg: proc_cache("second"), name="second")
    prepare_default(req_second)
    gc.collect()

    assert proc_ref() is not None
    assert map_ref() is not None
    assert triton_ref() is not None
    assert late_ref() is not None


def test_autotune_job_evicts_losing_candidates_but_preserves_unplanned_programs(
    tmp_path, monkeypatch
):
    """Autotuning reclaims its own losing candidates without touching unplanned kernels."""
    _deterministic_timer(monkeypatch)

    unplanned_key = ProgramKey("cute", "1" * 64, "unplanned.moe")
    candidate_keys = {
        1: ProgramKey("cute", "2" * 64, "cand.1"),
        2: ProgramKey("cute", "3" * 64, "cand.2"),
        4: ProgramKey("cute", "4" * 64, "cand.4"),
        7: ProgramKey("cute", "40" * 32, "cand.default"),
    }

    @program_cache.program_cache
    def cache(kind):
        if kind == "unplanned":
            return CompiledCuTeProgram(lambda: "moe", unplanned_key)
        return CompiledCuTeProgram(lambda width=kind: width * 3, candidate_keys[kind])

    monkeypatch.setattr(program_cache, "_CACHES", {cache})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())
    monkeypatch.setattr(session_module, "compiled_program_available", lambda _p: True)
    monkeypatch.setattr(
        session_module,
        "describe_compilation",
        lambda job: SimpleNamespace(programs=(job,)),
    )

    unplanned = cache("unplanned")
    unplanned_ref = weakref.ref(unplanned)
    del unplanned

    refs = {}

    def materialize(config):
        prog = cache(config.width)
        refs[config.width] = weakref.ref(prog)
        return prog

    req = _plan_with_program(
        materialize,
        name="tuned",
        tuning=contract(),
        benchmark=lambda state: PreparedCall(run=state, produce=lambda: None),
        compile_jobs=lambda config, _device: (candidate_keys[config.width],),
    )

    with session(tmp_path) as engine:
        result = engine.prepare((req,))
        assert result.benchmarked_candidates == 3
        winner_width = result.selections["tuned"].config.width
        gc.collect()

        assert unplanned_ref() is not None
        assert refs[winner_width]() is not None
        for width, ref in refs.items():
            if width != winner_width:
                assert ref() is None

    gc.collect()
    assert unplanned_ref() is not None
    assert refs[winner_width]() is None


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_cancelled_job_only_evicts_programs_it_created(
    tmp_path, monkeypatch, cancel
):
    """A failed or cancelled job rolls back its own programs without evicting others."""
    _deterministic_timer(monkeypatch)

    unplanned_key = ProgramKey("cute", "5" * 64, "unplanned")
    existing_plan_key = ProgramKey("cute", "6" * 64, "existing.plan")
    failed_job_key = ProgramKey("cute", "7" * 64, "failed.job")
    failed_job_key_2 = ProgramKey("cute", "70" * 32, "failed.job.2")

    @program_cache.program_cache
    def cache(tag):
        keys = {
            "unplanned": unplanned_key,
            "existing": existing_plan_key,
            "failed": failed_job_key,
            "failed2": failed_job_key_2,
        }
        if tag in ("failed", "failed2"):
            width = 1 if tag == "failed" else 2
            return CompiledCuTeProgram(lambda width=width: width * 3, keys[tag])
        return CompiledCuTeProgram(lambda tag=tag: tag, keys[tag])

    monkeypatch.setattr(program_cache, "_CACHES", {cache})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())
    monkeypatch.setattr(session_module, "compiled_program_available", lambda _p: True)
    monkeypatch.setattr(
        session_module,
        "describe_compilation",
        lambda job: SimpleNamespace(programs=(job,)),
    )

    unplanned = cache("unplanned")
    unplanned_ref = weakref.ref(unplanned)
    del unplanned

    created_refs = []

    def make_failed(cfg):
        tag = "failed2" if cfg.width == 2 else "failed"
        prog = cache(tag)
        created_refs.append(weakref.ref(prog))
        return prog

    monkeypatch.setattr(session_module, "_ADVANCE_SECONDS", 0.0)
    with session(tmp_path) as engine:
        existing_req = _plan_with_program(lambda _cfg: cache("existing"), name="keep")
        engine.prepare((existing_req,), autotune=False)
        existing_ref = weakref.ref(existing_req.plan.prepared.state)

        if cancel:
            # Advance an autotuning job until it loads candidate programs inside its
            # advance() scope and yields mid-race, then cancel before completion.
            cand_keys = {1: failed_job_key, 2: failed_job_key_2, 7: failed_job_key}
            cancel_req = _plan_with_program(
                make_failed,
                name="boom",
                tuning=contract(values=(1, 2)),
                benchmark=lambda state: PreparedCall(run=state, produce=lambda: None),
                compile_jobs=lambda cfg, _device: (cand_keys[cfg.width],),
            )
            job = engine.begin((cancel_req,))
            while not created_refs:
                progress = job.advance()
                assert not progress.done
            job.close()
        else:
            failing_req = _plan_with_program(make_failed, name="boom", fail=True)
            job = engine.begin((failing_req,))
            with pytest.raises(RuntimeError, match="prepare_call failed"):
                while not job.advance().done:
                    pass
        # Failed/cancelled job's programs must be evicted from the cache immediately.
        assert cache.cache_info().currsize == 2
        del job
        gc.collect()

        assert unplanned_ref() is not None
        assert existing_ref() is not None
        assert created_refs and all(ref() is None for ref in created_refs)

    gc.collect()
    assert unplanned_ref() is not None
    assert existing_ref() is None


def test_release_and_session_close_preserve_unplanned_and_shared_external_programs(
    tmp_path, monkeypatch
):
    """Releasing plans or closing a session only evicts session-owned programs."""
    unplanned_before_key = ProgramKey("cute", "8" * 64, "unplanned.before")
    unplanned_after_key = ProgramKey("cute", "9" * 64, "unplanned.after")
    shared_external_key = ProgramKey("cute", "a1" * 32, "unplanned.shared")
    owned_plan_key = ProgramKey("cute", "b2" * 32, "session.owned")

    @program_cache.program_cache
    def cache(tag):
        keys = {
            "before": unplanned_before_key,
            "after": unplanned_after_key,
            "shared": shared_external_key,
            "owned": owned_plan_key,
        }
        return CompiledCuTeProgram(lambda tag=tag: tag, keys[tag])

    monkeypatch.setattr(program_cache, "_CACHES", {cache})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())

    before_ref = weakref.ref(cache("before"))
    shared_ref = weakref.ref(cache("shared"))

    with session(tmp_path, autotune=False) as engine:
        owned_req = _plan_with_program(lambda _cfg: cache("owned"), name="owned")
        shared_req = _plan_with_program(lambda _cfg: cache("shared"), name="shared")
        engine.prepare((owned_req, shared_req))
        owned_ref = weakref.ref(owned_req.plan.prepared.state)

        # Unplanned kernel warmed after prepare() must survive release() and close().
        after_ref = weakref.ref(cache("after"))

        engine.release(owned_req.plan)
        gc.collect()
        assert owned_ref() is None
        assert before_ref() is not None
        assert after_ref() is not None
        assert shared_ref() is not None

    gc.collect()
    assert before_ref() is not None
    assert after_ref() is not None
    assert shared_ref() is not None


def test_upgrade_plan_evicts_replaced_default_while_preserving_unplanned_programs(
    tmp_path, monkeypatch
):
    """Upgrading a default plan to a tuned winner still evicts the replaced default."""
    _deterministic_timer(monkeypatch)

    unplanned_key = ProgramKey("cute", "c3" * 32, "unplanned")
    default_key = ProgramKey("cute", "d4" * 32, "default.7")
    cand_keys = {
        1: ProgramKey("cute", "e5" * 32, "cand.1"),
        2: ProgramKey("cute", "f6" * 32, "cand.2"),
        4: ProgramKey("cute", "07" * 32, "cand.4"),
        7: default_key,
    }

    @program_cache.program_cache
    def cache(kind):
        if kind == "unplanned":
            return CompiledCuTeProgram(lambda: "u", unplanned_key)
        return CompiledCuTeProgram(lambda width=kind: width * 3, cand_keys[kind])

    monkeypatch.setattr(program_cache, "_CACHES", {cache})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())

    monkeypatch.setattr(session_module, "compiled_program_available", lambda _p: True)
    monkeypatch.setattr(
        session_module,
        "describe_compilation",
        lambda job: SimpleNamespace(programs=(job,)),
    )
    unplanned_ref = weakref.ref(cache("unplanned"))
    plan = replace(
        declaration(tuning=contract()),
        _compile_jobs=lambda config, _device: (cand_keys[config.width],),
        _materialize=lambda selection, _device: compile_plan.load_programs(
            cache(selection.config.width)
        ),
    )

    with session(tmp_path, autotune=False) as engine:
        req = plan.request(
            name="upgradeable",
            prepare_call=lambda state: PreparedCall(run=state),
            benchmark_call=lambda state: PreparedCall(run=state, produce=lambda: None),
        )
        engine.prepare((req,))
        assert plan.selection.config == Config(7)
        default_ref = weakref.ref(plan.prepared.state)

        engine.prepare((req,), autotune=True)
        assert plan.selection.source == "tuned"
        assert plan.selection.config == Config(2)
        winner_ref = weakref.ref(plan.prepared.state)
        gc.collect()

        assert default_ref() is None
        assert winner_ref() is not None
        assert unplanned_ref() is not None


def test_triton_kernel_missing_from_resident_programs_is_still_preserved(monkeypatch):
    """A cached Triton kernel absent from _RESIDENT_PROGRAMS still survives prepare_default."""
    monkeypatch.setattr(session_module, "_LAZY_SESSIONS", {})

    triton_key = ProgramKey("triton", "18" * 32, "prehook.triton")
    plan_key = ProgramKey("cute", "29" * 32, "lazy.plan")
    triton_jit, triton_kernel, kernel_cache, key_cache = _make_triton_jit(triton_key)
    assert triton_key not in compile_plan._RESIDENT_PROGRAMS

    @program_cache.program_cache
    def cache():
        return CompiledCuTeProgram(lambda: 1, plan_key)

    monkeypatch.setattr(program_cache, "_CACHES", {cache})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", {triton_jit})

    triton_ref = weakref.ref(triton_kernel)
    del triton_kernel

    prepare_default(_plan_with_program(lambda _cfg: cache(), name="lazy"))
    gc.collect()

    assert triton_ref() is not None
    assert "k0" in kernel_cache and "sig0" in key_cache


def test_cached_program_keys_and_candidate_scoped_eviction(monkeypatch):
    """cached_program_keys ignores unresolved deferred artifacts; candidates scopes eviction."""
    live_key = ProgramKey("cute", "3a" * 32, "live")
    cand_key = ProgramKey("cute", "4b" * 32, "candidate")
    other_key = ProgramKey("cute", "5c" * 32, "other")
    deferred_cute = DeferredCuTeKernel(
        ProgramKey("cute", "6d" * 32, "deferred.cute"), memory_key=("d",)
    )
    deferred_triton = DeferredTritonKernel(
        ProgramKey("triton", "7e" * 32, "deferred.triton"),
        source=SimpleNamespace(name="deferred.triton"),
    )

    mapping = {
        "live": CompiledCuTeProgram(lambda: 1, live_key),
        "cand": CompiledCuTeProgram(lambda: 2, cand_key),
        "other": CompiledCuTeProgram(lambda: 3, other_key),
        "bundle_live_other": ProgramBundle((live_key, other_key)),
        "bundle_live_cand": ProgramBundle((live_key, cand_key)),
        "deferred_cute": deferred_cute,
        "deferred_triton": deferred_triton,
    }
    monkeypatch.setattr(program_cache, "_CACHES", set())
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [(mapping, (), None)])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())

    keys = program_cache.cached_program_keys()
    assert {live_key, cand_key, other_key} <= keys
    assert deferred_cute.__b12x_programs__[0] not in keys
    assert deferred_triton.__b12x_programs__[0] not in keys

    # Evicting only {cand_key} while keeping {live_key} drops cand and bundle_live_cand,
    # leaving other and bundle_live_other untouched.
    removed = program_cache.evict_unretained({live_key}, candidates={cand_key})
    assert removed == 2
    assert "cand" not in mapping
    assert "bundle_live_cand" not in mapping
    assert "live" in mapping
    assert "other" in mapping
    assert "bundle_live_other" in mapping

    # If candidates are already covered by keep, nothing is evicted.
    assert program_cache.evict_unretained({live_key}, candidates={live_key}) == 0
    assert "other" in mapping


def test_job_completion_and_cancellation_evict_unresolved_planning_artifacts(
    tmp_path, monkeypatch
):
    """Unresolved deferred planning artifacts are evicted while unplanned kernels stay."""
    unplanned_key = ProgramKey("cute", "81" * 32, "unplanned.compiled")
    plan_key = ProgramKey("cute", "82" * 32, "plan.compiled")
    deferred_cute_key = ProgramKey("cute", "83" * 32, "deferred.cute")
    deferred_triton_key = ProgramKey("triton", "84" * 32, "deferred.triton")

    @program_cache.program_cache
    def cache(tag):
        keys = {"unplanned": unplanned_key, "plan": plan_key}
        return CompiledCuTeProgram(lambda tag=tag: tag, keys[tag])

    mapping = {}
    triton_kernel_cache = {}
    triton_key_cache = {}
    triton_jit = type(
        "Jit",
        (),
        {"device_caches": {0: (triton_kernel_cache, triton_key_cache, None, None)}},
    )()

    monkeypatch.setattr(program_cache, "_CACHES", {cache})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [(mapping, (), None)])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", {triton_jit})

    unplanned_ref = weakref.ref(cache("unplanned"))

    def plant_deferred():
        mapping["planned"] = DeferredCuTeKernel(deferred_cute_key, memory_key=("p",))
        triton_kernel_cache["tk"] = DeferredTritonKernel(
            deferred_triton_key, source=SimpleNamespace(name="deferred.triton")
        )
        triton_key_cache["sig"] = "tk"

    def materialize(_cfg):
        plant_deferred()
        return cache("plan")

    # Normal job completion evicts unresolved deferred artifacts, keeps unplanned kernel.
    with session(tmp_path, autotune=False) as engine:
        engine.prepare((_plan_with_program(materialize, name="ok"),))
        assert "planned" not in mapping
        assert "tk" not in triton_kernel_cache and "sig" not in triton_key_cache
        gc.collect()
        assert unplanned_ref() is not None

        # Cancelled job also evicts unresolved deferred artifacts.
        job = engine.begin((_plan_with_program(lambda _cfg: cache("plan"), name="c"),))
        plant_deferred()
        job.close()
        assert "planned" not in mapping
        assert "tk" not in triton_kernel_cache and "sig" not in triton_key_cache
        gc.collect()
        assert unplanned_ref() is not None


def test_graph_replay_of_unplanned_kernels_survives_first_use_preparation():
    """SM12x GPU test: captured unplanned CuTe & Triton kernels survive prepare_default."""
    device = require_b12x()
    import torch
    from b12x.gemm import bf16_gemv
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _w4a16_topk_sum_launch_flat,
        pack_topk_routes_by_expert,
    )

    m, topk, hidden_size, num_experts = 4, 2, 128, 8
    _, packed_routes, route_blocks = route_pack_capacity(
        m * topk, 16, num_experts, topk=topk
    )

    def route_workspace():
        # CUDA graph capture requires a preallocated route-packing workspace.
        def buffer(elements):
            return torch.empty(elements, device=device, dtype=torch.int32)

        return dict(
            packed_route_indices=buffer(packed_routes),
            block_expert_ids=buffer(route_blocks),
            packed_route_count=buffer(1),
            expert_offsets=buffer(num_experts + 1),
            expert_counts=buffer(num_experts),
        )

    fc2_out = torch.randn(m * topk, hidden_size, device=device, dtype=torch.bfloat16)
    sum_out = torch.empty(m, hidden_size, device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(
        0, num_experts, (m, topk), device=device, dtype=torch.int32
    )
    topk_weights = torch.rand(m, topk, device=device, dtype=torch.float32)

    # Warm unplanned CuTe (W4A16TopKSumKernel) and Triton (pack_topk_routes) kernels
    # outside any preparation plan or retain_compiled_programs() scope.
    stream_int = torch.cuda.current_stream(device).cuda_stream
    _w4a16_topk_sum_launch_flat(
        fc2_out,
        sum_out,
        m,
        topk,
        hidden_size,
        "bf16",
        stream_int,
        topk_weights=topk_weights,
        route_expert_ids=topk_ids,
    )
    packed_ref = pack_topk_routes_by_expert(
        topk_ids, block_size=16, num_experts=num_experts, **route_workspace()
    )
    torch.cuda.synchronize(device)
    expected_sum = sum_out.clone()

    graph_workspace = route_workspace()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            stream_int = torch.cuda.current_stream(device).cuda_stream
            _w4a16_topk_sum_launch_flat(
                fc2_out,
                sum_out,
                m,
                topk,
                hidden_size,
                "bf16",
                stream_int,
                topk_weights=topk_weights,
                route_expert_ids=topk_ids,
            )
            packed_graph = pack_topk_routes_by_expert(
                topk_ids,
                block_size=16,
                num_experts=num_experts,
                **graph_workspace,
            )

        # Trigger an unprepared first-use preparation on an unrelated b12x op.
        gemv_x = torch.randn(2, 2048, device=device, dtype=torch.bfloat16)
        gemv_w = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
        gemv_plan = bf16_gemv.plan(bf16_gemv.query_from_call(gemv_x, gemv_w))
        prepare_default(
            gemv_plan.request(
                name="gemv",
                prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run(gemv_x, gemv_w)
                ),
            )
        )
        gemv_out = bf16_gemv.mm(gemv_x, gemv_w, plan=gemv_plan)
        torch.cuda.synchronize(device)
        assert torch.isfinite(gemv_out).all()

        # Force collection of any evicted executables before graph replay.
        gc.collect()

        sum_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)

        torch.testing.assert_close(sum_out, expected_sum, rtol=0, atol=0)
        for got, want in zip(packed_graph, packed_ref, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    finally:
        graph.reset()


def test_prehook_native_triton_specialization_survives_later_preparation(
    tmp_path, monkeypatch
):
    """A native Triton specialization cached before hook installation is not evicted."""
    import triton
    from triton.compiler import CompiledKernel

    class _FakeCompiledTritonKernel(CompiledKernel):
        def __init__(self, program_key: ProgramKey):
            self.__b12x_programs__ = (program_key,)
            self.module = object()

        def __call__(self):
            return None

        def __del__(self):
            pass

    prehook_key = ProgramKey("triton", "91" * 32, "prehook.native.spec1")
    job_key = ProgramKey("triton", "92" * 32, "job.native.spec2")

    @triton.jit
    def _native_kernel(x):
        pass

    # Mark the JIT as a native b12x kernel and pre-populate a specialization before
    # _NATIVE_JITS ever sees this JITFunction (simulating compilation before hook install).
    _native_kernel.fn.__module__ = "b12x.test_prehook"
    prehook_compiled = _FakeCompiledTritonKernel(prehook_key)
    kernel_cache = {"k_prehook": prehook_compiled}
    key_cache = {"sig_prehook": "k_prehook"}
    _native_kernel.device_caches[0] = (kernel_cache, key_cache, None, None)

    monkeypatch.setattr(program_cache, "_CACHES", set())
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())

    prehook_ref = weakref.ref(prehook_compiled)
    del prehook_compiled

    # Hook installation registers JITs that already exist, so their pre-hook
    # specializations are counted even though run_observed never saw them.
    assert prehook_key not in compile_plan.triton_cached_program_keys()
    compile_plan._discover_native_triton_jits()
    assert prehook_key in compile_plan.triton_cached_program_keys()

    job_compiled_refs = []

    def make_plan(_cfg):
        # During preparation, the JIT is observed and adds a second specialization.
        compile_plan._NATIVE_JITS.add(_native_kernel)
        job_compiled = _FakeCompiledTritonKernel(job_key)
        kernel_cache["k_job"] = job_compiled
        key_cache["sig_job"] = "k_job"
        job_compiled_refs.append(weakref.ref(job_compiled))
        return job_compiled

    with session(tmp_path, autotune=False) as engine:
        req = _plan_with_program(make_plan, name="prehook_triton_op")
        engine.prepare((req,))
        assert prehook_ref() is not None
        assert "k_prehook" in kernel_cache and "sig_prehook" in key_cache
        assert "k_job" in kernel_cache

        engine.release(req.plan)
        gc.collect()
        assert prehook_ref() is not None
        assert "k_prehook" in kernel_cache and "sig_prehook" in key_cache
        assert "k_job" not in kernel_cache and "sig_job" not in key_cache

    gc.collect()
    assert prehook_ref() is not None
    assert "k_prehook" in kernel_cache and "sig_prehook" in key_cache
    assert job_compiled_refs and job_compiled_refs[0]() is None


def test_nested_retain_compiled_programs_keeps_caller_container_intact(
    tmp_path, monkeypatch
):
    """Preparation inside caller-owned retain_compiled_programs does not clear outer state."""
    outer_key = ProgramKey("cute", "a1" * 32, "outer.retained")
    plan_key = ProgramKey("cute", "a2" * 32, "nested.plan")
    fail_key = ProgramKey("cute", "a3" * 32, "nested.fail")

    @program_cache.program_cache
    def cache(tag):
        keys = {"outer": outer_key, "plan": plan_key, "fail": fail_key}
        return CompiledCuTeProgram(lambda tag=tag: tag, keys[tag])

    monkeypatch.setattr(program_cache, "_CACHES", {cache})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())

    with compile_plan.retain_compiled_programs() as outer:
        outer_prog = cache("outer")
        outer_prog()
        outer_id = id(outer_prog)
        outer_ref = weakref.ref(outer_prog)
        del outer_prog
        assert outer_id in outer.owners

        with session(tmp_path, autotune=False) as engine:
            # 1. Successful prepare + release must not touch caller's outer container.
            req = _plan_with_program(lambda _cfg: cache("plan"), name="nested_ok")
            engine.prepare((req,))
            assert req.plan.prepared.retained is not outer
            assert outer_id in outer.owners

            engine.release(req.plan)
            gc.collect()
            assert outer_id in outer.owners
            assert outer_key in compile_plan.retained_program_keys()
            assert outer_ref() is not None

            # 2. Failed _instantiate error cleanup must not clear caller's outer container.
            failing = _plan_with_program(
                lambda _cfg: cache("fail"), name="nested_fail", fail=True
            )
            with pytest.raises(RuntimeError, match="prepare_call failed"):
                engine.prepare((failing,))
            assert outer_id in outer.owners
            assert outer_key in compile_plan.retained_program_keys()
            assert outer_ref() is not None

    # Even after exiting outer retention, outer_prog was external to the session and survives.
    gc.collect()
    assert outer_ref() is not None


@pytest.mark.parametrize("cancel", [False, True])
def test_external_program_loaded_between_begin_and_advance_survives(
    tmp_path, monkeypatch, cancel
):
    """Unrelated programs loaded after begin() or between advances are never evicted."""
    _deterministic_timer(monkeypatch)

    ext_before_advance_key = ProgramKey("cute", "b1" * 32, "ext.before.advance")
    ext_between_advances_key = ProgramKey("cute", "b2" * 32, "ext.between.advances")
    ext_before_cancel_key = ProgramKey("cute", "b3" * 32, "ext.before.cancel")
    winner_key = ProgramKey("cute", "b4" * 32, "job.winner")
    loser_key = ProgramKey("cute", "b5" * 32, "job.loser")
    default_key = ProgramKey("cute", "b6" * 32, "job.default")

    @program_cache.program_cache
    def cache(tag):
        keys = {
            "ext_before_advance": ext_before_advance_key,
            "ext_between_advances": ext_between_advances_key,
            "ext_before_cancel": ext_before_cancel_key,
            "winner": winner_key,
            "loser": loser_key,
            "default": default_key,
        }
        widths = {"loser": 1, "winner": 2, "default": 7}
        if tag in widths:
            width = widths[tag]
            return CompiledCuTeProgram(lambda width=width: width * 3, keys[tag])
        return CompiledCuTeProgram(lambda tag=tag: tag, keys[tag])

    monkeypatch.setattr(program_cache, "_CACHES", {cache})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())
    monkeypatch.setattr(session_module, "compiled_program_available", lambda _p: True)
    monkeypatch.setattr(
        session_module,
        "describe_compilation",
        lambda job: SimpleNamespace(programs=(job,)),
    )

    refs = {}
    cand_keys = {1: loser_key, 2: winner_key, 7: default_key}

    def factory(cfg):
        tag = {1: "loser", 2: "winner", 7: "default"}[cfg.width]
        prog = cache(tag)
        refs[tag] = weakref.ref(prog)
        return prog

    req = _plan_with_program(
        factory,
        name="coop_op",
        tuning=contract(values=(1, 2)),
        benchmark=lambda state: PreparedCall(run=state, produce=lambda: None),
        compile_jobs=lambda cfg, _device: (cand_keys[cfg.width],),
    )

    monkeypatch.setattr(session_module, "_ADVANCE_SECONDS", 0.0)
    with session(tmp_path) as engine:
        job = engine.begin((req,))

        # Caller loads an unrelated kernel after begin() and before the first advance().
        ext1 = cache("ext_before_advance")
        ext1_ref = weakref.ref(ext1)
        del ext1

        progress = job.advance()
        assert not progress.done

        # Caller loads another unrelated kernel between cooperative advances.
        ext2 = cache("ext_between_advances")
        ext2_ref = weakref.ref(ext2)
        del ext2

        if cancel:
            # Advance until at least one job candidate is loaded, then load an external
            # kernel right before calling job.close().
            while not refs:
                progress = job.advance()
                assert not progress.done
            ext3 = cache("ext_before_cancel")
            ext3_ref = weakref.ref(ext3)
            del ext3
            job.close()
            del job
            gc.collect()
            assert ext1_ref() is not None
            assert ext2_ref() is not None
            assert ext3_ref() is not None
            # Programs loaded by the cancelled job itself were reclaimed.
            for ref in refs.values():
                assert ref() is None
        else:
            while not job.advance().done:
                pass
            del job
            gc.collect()
            assert ext1_ref() is not None
            assert ext2_ref() is not None
            assert refs["winner"]() is not None
            assert refs["loser"]() is None

    gc.collect()
    assert ext1_ref() is not None
    assert ext2_ref() is not None
    if cancel:
        assert ext3_ref() is not None
    else:
        assert refs["winner"]() is None


def test_in_process_compile_evicts_active_job_factory_memo(monkeypatch):
    """compile_workers=0: the job's own factory memo must not hand back planning output."""
    planned_key = ProgramKey("cute", "c" * 64, "planned")
    resolved_key = ProgramKey("cute", "d" * 64, "resolved")
    calls = []

    @program_cache.program_cache(scope="preparation")
    def factory(name):
        calls.append(name)
        kernel = DeferredCuTeKernel(
            planned_key if name == "planned" else resolved_key, memory_key=(name,)
        )
        if name == "resolved":
            kernel._resolved = object()
        return compile_plan.attach_programs(lambda: None, kernel)

    monkeypatch.setattr(program_cache, "_CACHES", weakref.WeakSet())
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())
    scope = program_cache.PreparationProgramCache()
    with scope.activate():
        planned = factory("planned")
        resolved = factory("resolved")
        assert compile_plan.evict_planning_artifacts((planned_key,)) == 1
        assert factory("resolved") is resolved
        assert factory("planned") is not planned
    assert calls == ["planned", "resolved", "planned"]
    # Outside the job, eviction does not reach into its memo.
    with scope.activate():
        relowered = factory("planned")
    assert compile_plan.evict_planning_artifacts((planned_key,)) == 0
    with scope.activate():
        assert factory("planned") is relowered
    scope.clear()
