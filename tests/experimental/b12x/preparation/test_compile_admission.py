"""A stop drops only undispatched optional work, never promoted requirements."""
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from b12x._lib import compile_pool
from b12x._lib.compile_plan import ProgramKey
from b12x._lib.compile_pool import CompilationPlan, CompileJob, CompilePool


class Activity(list):
    def get_lock(self):
        return nullcontext()


class Workers:
    def __init__(self, **kwargs):
        self._pool = (SimpleNamespace(pid=1, exitcode=None),)
        self.calls = []

    def apply_async(self, function, args, callback, error_callback):
        self.calls.append((args[1], callback, error_callback))

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


def test_stop_preserves_promoted_shared_required_programs(monkeypatch):
    workers = Workers()
    inherited_visibility = []
    pool_arguments = []

    def create_pool(**kwargs):
        inherited_visibility.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        pool_arguments.append(kwargs)
        return workers

    context = SimpleNamespace(Array=lambda kind, values: Activity(values), Pool=create_pool)
    monkeypatch.setattr(compile_pool.multiprocessing, "get_context", lambda kind: context)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    pool = CompilePool(
        device_ordinal=0,
        compute_capability=(12, 0),
        device_uuid="synthetic-device",
        product_name="synthetic-gpu",
        sm_count=1,
        max_shared_memory_per_block=1024,
        max_shared_memory_per_multiprocessor=2048,
        workers=2,
    )
    assert inherited_visibility == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
    assert pool_arguments[0]["initargs"][5:7] == (1024, 2048)
    plans = tuple(CompilationPlan(
        CompileJob.create("integration.producer:compile", index), (ProgramKey("cute", str(index)),),
    ) for index in range(10))
    pool.submit_plans(plans, required=False)
    assert len(workers.calls) == 2
    required = pool.submit_plans((plans[7],), required=True)
    pool.cancel_optional()
    assert not pool.ready(required)
    for index in range(2):
        programs, completed, failed = workers.calls[index]
        completed((1, 1, 0, programs))
    assert len(workers.calls) == 3
    programs, completed, failed = workers.calls[2]
    assert programs == plans[7].programs
    assert not pool.ready(required)
    completed((1, 1, 0, programs))
    assert pool.ready(required)
    assert not pool.pending
    summary = pool.summary()
    assert (summary.cute_programs, summary.triton_programs) == (3, 0)
    assert summary.cute_compilations == 3
    pool.close()


def test_summary_counts_unique_dispatched_programs_across_overlapping_plans(monkeypatch):
    workers = Workers()
    context = SimpleNamespace(Array=lambda kind, values: Activity(values), Pool=lambda **kwargs: workers)
    monkeypatch.setattr(compile_pool.multiprocessing, "get_context", lambda kind: context)
    pool = CompilePool(
        device_ordinal=0,
        compute_capability=(12, 0),
        device_uuid="synthetic-device",
        product_name="synthetic-gpu",
        sm_count=1,
        max_shared_memory_per_block=1024,
        max_shared_memory_per_multiprocessor=2048,
        workers=1,
    )
    cute = ProgramKey("cute", "shared")
    triton = ProgramKey("triton", "shared")
    another = ProgramKey("cute", "another")
    first = CompilationPlan(CompileJob.create("integration.producer:compile", 1), (cute, triton))
    second = CompilationPlan(CompileJob.create("integration.producer:compile", 2), (cute, another))
    pool.submit_plans((first, second, first))
    summary = pool.summary()
    assert (summary.jobs, summary.requested_jobs) == (1, 3)
    assert (summary.cute_programs, summary.triton_programs) == (1, 1)
    assert (summary.cute_compilations, summary.triton_compilations) == (0, 0)
    programs, complete, _ = workers.calls[0]
    complete((1, 1, 1, programs))
    summary = pool.summary()
    assert (summary.cute_programs, summary.triton_programs) == (2, 1)
    assert (summary.cute_compilations, summary.triton_compilations) == (1, 1)
    programs, complete, _ = workers.calls[1]
    complete((1, 1, 0, programs))
    pool.submit_plans((second, first))
    summary = pool.summary()
    assert (summary.jobs, summary.requested_jobs) == (2, 5)
    assert (summary.cute_programs, summary.triton_programs) == (2, 1)
    assert (summary.cute_compilations, summary.triton_compilations) == (2, 1)
    assert not pool.pending
    pool.close()


def test_failed_job_cannot_become_ready(monkeypatch):
    workers = Workers()
    context = SimpleNamespace(Array=lambda kind, values: Activity(values), Pool=lambda **kwargs: workers)
    monkeypatch.setattr(compile_pool.multiprocessing, "get_context", lambda kind: context)
    pool = CompilePool(
        device_ordinal=0,
        compute_capability=(12, 0),
        device_uuid="synthetic-device",
        product_name="synthetic-gpu",
        sm_count=1,
        max_shared_memory_per_block=1024,
        max_shared_memory_per_multiprocessor=2048,
        workers=1,
    )
    plan = CompilationPlan(CompileJob.create("integration.producer:compile"), (ProgramKey("cute", "failed"),))
    required = pool.submit_plans((plan,))
    workers.calls[0][2](ValueError("compiler rejected specialization"))
    with pytest.raises(ValueError, match="rejected specialization"):
        pool.ready(required)
    pool.close(terminate=True)


def test_offline_cutlass_device_attributes_are_explicit_and_fail_closed():
    max_smem_per_mp = object()
    cuda_helpers = SimpleNamespace(
        cuda=SimpleNamespace(
            CUdevice_attribute=SimpleNamespace(
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR=(
                    max_smem_per_mp
                )
            )
        ),
        get_device_attribute=lambda attribute: 0,
    )
    compile_pool._configure_offline_cutlass_device_attributes(cuda_helpers, 2048)
    assert cuda_helpers.get_device_attribute(max_smem_per_mp) == 2048
    with pytest.raises(RuntimeError, match="does not provide CUDA device attribute"):
        cuda_helpers.get_device_attribute(object())


def test_loading_explicit_dependencies_keeps_capture_free_of_loader_access(monkeypatch):
    from b12x._lib import compiler
    from b12x._lib.compile_plan import (
        CompiledCuTeProgram, DeferredCuTeKernel, attach_programs, load_programs,
    )
    from b12x._lib.runtime_control import kernel_resolution_guard

    calls = []
    program = ProgramKey("cute", "prepared-load-dependency")

    def execute(value):
        calls.append(value)
        return value + 1

    resident = CompiledCuTeProgram(execute, program)
    deferred = DeferredCuTeKernel(program, ("dependency",))
    monkeypatch.setattr(compiler, "_memory_cache_get", lambda key: resident)

    def invoke(value):
        return deferred(value)

    attach_programs(invoke, deferred)
    load_programs({"selected": invoke})
    assert calls == []

    def forbidden_lookup(key):
        raise AssertionError("prepared execution consulted the loader cache")

    monkeypatch.setattr(compiler, "_memory_cache_get", forbidden_lookup)
    with kernel_resolution_guard("loaded dependency must remain executable"):
        assert invoke(41) == 42


def test_program_key_metadata_is_not_an_executable_dependency():
    from b12x._lib.compile_plan import ProgramBundle, load_programs

    with pytest.raises(TypeError):
        load_programs(ProgramBundle((ProgramKey("cute", "metadata-only"),)))
