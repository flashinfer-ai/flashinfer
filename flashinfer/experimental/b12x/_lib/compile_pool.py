"""Compile production program-key groups in device-pinned spawned workers."""

from __future__ import annotations

import importlib
import multiprocessing
import os
import pickle
import time
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from multiprocessing.pool import Pool
from threading import Condition, Lock
from typing import Any, Iterable

from .compile_plan import (
    ProgramKey, observe_programs, plan_compilations, program_keys, record_program,
    serialized_compilations,
)


def _validate_metadata(value):
    if value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, Enum):
        _validate_metadata(value.value)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _validate_metadata(getattr(value, item.name))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_metadata(key)
            _validate_metadata(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_metadata(item)
        return
    import torch
    if isinstance(value, (torch.dtype, torch.device)):
        return
    raise TypeError(f"compile jobs accept metadata only, not {type(value).__name__}")

@dataclass(frozen=True, kw_only=True)
class CompileJob:
    """An importable metadata factory, not itself a compiled-program identity."""

    factory: str
    args: tuple[object, ...] = ()
    kwargs: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        module, separator, name = self.factory.partition(":")
        if (not separator or not module or not name or "<locals>" in name
                or any(not part.isidentifier() for part in (*module.split("."), *name.split(".")))):
            raise ValueError("compile factories require an importable module:attribute")
        _validate_metadata(self.args)
        _validate_metadata(self.kwargs)
        if any(not isinstance(key, str) for key, value in self.kwargs):
            raise TypeError("compile keyword names must be strings")

    @classmethod
    def create(cls, factory: str, *args: object, **kwargs: object):
        # __post_init__ also validates direct construction and rejects tensors.
        return cls(factory=factory, args=tuple(args), kwargs=tuple(sorted(kwargs.items())))


def _factory(reference):
    module, _, attribute = reference.partition(":")
    value = importlib.import_module(module)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"compile factory is not callable: {reference}")
    return value


@dataclass(frozen=True)
class CompilationPlan:
    job: CompileJob
    programs: tuple[ProgramKey, ...]


def describe_compilation(job: CompileJob) -> CompilationPlan:
    """Run only the factory's metadata construction, without lowering or launch."""
    with plan_compilations() as observed:
        value = _factory(job.factory)(*job.args, **dict(job.kwargs))
        returned = program_keys(value)
    if not returned:
        raise ValueError(f"compile factory must return its program carriers: {job.factory}")
    if not observed <= set(returned):
        raise ValueError(f"compile factory discarded required programs: {job.factory}")
    return CompilationPlan(job, tuple(sorted(set(returned), key=lambda p: (p.dialect, p.key))))


def compile_in_process(plans: Iterable[CompilationPlan]) -> None:
    """Run each plan's factory in this process so its programs reach the compile cache.

    This is how a plan compiles on first use without compiler workers. The
    deferred programs that planning left in the kernel memos and in Triton's
    per-function caches are evicted first, so the factories lower their
    programs instead of being handed a planning artifact; the caller checks
    artifact availability afterwards as it does for pool jobs.
    """
    from .compile_plan import evict_planning_artifacts
    plans = tuple(plans)
    evict_planning_artifacts(program for plan in plans for program in plan.programs)
    for plan in plans:
        job = plan.job
        with serialized_compilations():
            _factory(job.factory)(*job.args, **dict(job.kwargs))


_WORKER_TRITON_COMPILES = 0
_WORKER_RUNTIME_LIBRARIES = ()
_SPAWN_ENVIRONMENT_LOCK = Lock()


def _configure_offline_cutlass_device_attributes(
    cuda_helpers: Any,
    max_shared_memory_per_multiprocessor: int,
):
    max_smem_per_mp_attribute = getattr(
        cuda_helpers.cuda.CUdevice_attribute,
        "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
    )

    def get_device_attribute(attribute):
        if attribute == max_smem_per_mp_attribute:
            return max_shared_memory_per_multiprocessor
        raise RuntimeError(
            "compiler target metadata does not provide CUDA device attribute "
            f"{attribute!r}"
        )

    cuda_helpers.get_device_attribute = get_device_attribute


@contextmanager
def _offline_compiler_spawn_environment():
    """Hide CUDA devices from children before their Python interpreter starts."""
    with _SPAWN_ENVIRONMENT_LOCK:
        missing = object()
        previous = os.environ.get("CUDA_VISIBLE_DEVICES", missing)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            yield
        finally:
            if previous is missing:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _initialize_worker(
    device_ordinal: int,
    compute_capability: tuple[int, int],
    device_uuid: str,
    product_name: str,
    sm_count: int,
    max_shared_memory_per_block: int,
    max_shared_memory_per_multiprocessor: int,
    activity: Any,
):
    # Compiler children receive a complete target description from the parent.
    # Hiding devices before importing CUDA-aware libraries prevents an
    # accidental runtime or driver call from retaining a device context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from triton import knobs
    from . import compiler

    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("compiler worker inherited an initialized CUDA runtime")
    compiler._configure_offline_compile_target(
        device_ordinal, compute_capability, sm_count,
    )

    def target_ordinal(device=None):
        if device is None:
            return device_ordinal
        if type(device) is int:
            ordinal = device
        else:
            resolved = torch.device(device)
            if resolved.type != "cuda":
                raise ValueError(f"compiler target must be CUDA, got {resolved}")
            ordinal = device_ordinal if resolved.index is None else resolved.index
        if ordinal != device_ordinal:
            raise ValueError(
                f"compiler target ordinal changed from {device_ordinal} to {ordinal}"
            )
        return ordinal

    @contextmanager
    def device_scope(device):
        target_ordinal(device)
        yield

    class OfflineStream:
        cuda_stream = 0
        device = torch.device("cuda", device_ordinal)

        def synchronize(self):
            raise RuntimeError("compiler workers cannot synchronize CUDA")

    properties = type("OfflineCudaDeviceProperties", (), {
        "major": compute_capability[0],
        "minor": compute_capability[1],
        "multi_processor_count": sm_count,
        "shared_memory_per_block": max_shared_memory_per_block,
        "shared_memory_per_block_optin": max_shared_memory_per_block,
        "shared_memory_per_multiprocessor": max_shared_memory_per_multiprocessor,
        "name": product_name,
        "uuid": device_uuid,
    })()
    offline_stream = OfflineStream()

    def get_device_properties(device=None):
        target_ordinal(device)
        return properties

    def get_device_capability(device=None):
        target_ordinal(device)
        return compute_capability

    def current_stream(device=None):
        target_ordinal(device)
        return offline_stream

    def reject_cuda_initialization(*_args, **_kwargs):
        raise RuntimeError("compiler workers cannot initialize CUDA")

    # Compile factories use these calls only to describe their target. Keep
    # those operations pure and fail closed on every execution-side entrypoint.
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: max(1, device_ordinal + 1)
    torch.cuda.current_device = lambda: device_ordinal
    torch.cuda.set_device = target_ordinal
    torch.cuda.device = device_scope
    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties
    torch.cuda.get_device_name = lambda device=None: (
        get_device_properties(device).name
    )
    torch.cuda.current_stream = current_stream
    torch.cuda.default_stream = current_stream
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.synchronize = reject_cuda_initialization
    torch.cuda.init = reject_cuda_initialization
    torch.cuda._lazy_init = reject_cuda_initialization

    from torch._subclasses.fake_tensor import FakeTensor
    from torch.utils._mode_utils import no_dispatch

    def has_advanced_index(value):
        if isinstance(value, (FakeTensor, list)):
            return True
        if isinstance(value, tuple):
            return any(has_advanced_index(item) for item in value)
        return False

    def meta_index(value):
        if isinstance(value, FakeTensor):
            return torch.empty_strided(
                tuple(value.shape), tuple(value.stride()),
                dtype=value.dtype, device="meta",
            ).as_strided(
                tuple(value.shape), tuple(value.stride()), value.storage_offset()
            )
        if isinstance(value, tuple):
            return tuple(meta_index(item) for item in value)
        if isinstance(value, list):
            return [meta_index(item) for item in value]
        return value

    original_getitem = torch.Tensor.__getitem__
    original_setitem = torch.Tensor.__setitem__

    def fake_geometry(tensor, index):
        with no_dispatch():
            return original_getitem(meta_index(tensor), meta_index(index))

    def fake_getitem(tensor, index):
        if not isinstance(tensor, FakeTensor):
            return original_getitem(tensor, index)
        geometry = fake_geometry(tensor, index)
        if has_advanced_index(index):
            return torch.empty_strided(
                tuple(geometry.shape), tuple(geometry.stride()),
                dtype=tensor.dtype, device=tensor.device,
            )
        return tensor.as_strided(
            tuple(geometry.shape), tuple(geometry.stride()),
            geometry.storage_offset(),
        )

    def fake_setitem(tensor, index, value):
        if not isinstance(tensor, FakeTensor):
            return original_setitem(tensor, index, value)
        with no_dispatch():
            meta = meta_index(tensor)
            index = meta_index(index)
            value = meta_index(value)
            original_setitem(meta, index, value)
            # Meta index_put_ skips broadcasting checks. Basic assignment to
            # the selected region validates them, including leading ones.
            destination = original_getitem(meta, index)
            original_setitem(destination, Ellipsis, value)

    torch.Tensor.__getitem__ = fake_getitem
    torch.Tensor.__setitem__ = fake_setitem

    # CuTe's PIC object writer resolves the host launch shim while exporting.
    # Loading the shim supplies symbols only; CUDA remains invisible and any
    # execution-side Torch entrypoint above still fails closed.
    import ctypes
    from cutlass.runtime import find_runtime_libraries

    global _WORKER_RUNTIME_LIBRARIES
    _WORKER_RUNTIME_LIBRARIES = tuple(
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        for path in find_runtime_libraries(enable_tvm_ffi=False)
    )

    major, minor = compute_capability
    arch_suffix = "a" if major >= 9 else ""
    from cutlass.cutlass_dsl import CuTeDSL
    CuTeDSL._get_dsl().envar.arch = f"sm_{major}{minor}{arch_suffix}"

    from cutlass.base_dsl.runtime import cuda as cuda_helpers
    _configure_offline_cutlass_device_attributes(
        cuda_helpers, max_shared_memory_per_multiprocessor
    )

    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver
    triton_target = GPUTarget("cuda", major * 10 + minor, 32)

    class OfflineTritonDriver:
        def get_current_device(self):
            return device_ordinal

        def get_device_capability(self, device):
            target_ordinal(device)
            return compute_capability

        def get_current_target(self):
            return triton_target

        def get_current_stream(self, device):
            target_ordinal(device)
            return 0

    driver.set_active(OfflineTritonDriver())
    if torch.cuda.is_initialized():
        raise RuntimeError("compiler worker initialized CUDA during setup")
    original = compiler._call_cute_compile

    def compile_counted(*args, **kwargs):
        with activity.get_lock():
            activity[0] += 1
            activity[1] = max(activity[1], activity[0])
        try:
            return original(*args, **kwargs)
        finally:
            with activity.get_lock():
                activity[0] -= 1

    compiler._call_cute_compile = compile_counted
    previous = knobs.compilation.listener

    def listener(*args, **kwargs):
        global _WORKER_TRITON_COMPILES
        metadata = kwargs.get("metadata", args[1] if len(args) > 1 else {})
        record_program(ProgramKey("triton", metadata["hash"], metadata.get("name", "")))
        if not kwargs.get("cache_hit", args[4] if len(args) > 4 else False):
            _WORKER_TRITON_COMPILES += 1
        if previous is not None:
            previous(*args, **kwargs)

    knobs.compilation.listener = listener


def _run_job(payload: bytes, expected: tuple[ProgramKey, ...]):
    import torch
    from .compiler import compile_cache_info

    job = pickle.loads(payload)
    before_cute = int(compile_cache_info()["compile_misses"])
    before_triton = _WORKER_TRITON_COMPILES
    try:
        with observe_programs() as observed, serialized_compilations():
            result = _factory(job.factory)(*job.args, **dict(job.kwargs))
            actual = set(program_keys(result))
        if not observed <= actual or actual != set(expected):
            def descriptions(keys):
                return sorted(f"{key.dialect}:{key.name} ({key.key})" for key in keys)
            raise RuntimeError(
                f"factory program keys changed between planning and compilation: {job.factory}; "
                f"planned-only {descriptions(set(expected) - actual)}, "
                f"compiled-only {descriptions(actual - set(expected))}, "
                f"observed {descriptions(observed - actual)} outside the carriers; args {job.args!r:.600}"
            )
        if torch.cuda.is_initialized():
            raise RuntimeError("compiler worker initialized CUDA while compiling artifacts")
    except Exception as error:
        # Many compiler-specific exception constructors are not pickle-safe.
        raise RuntimeError(f"{job.factory}: {type(error).__name__}: {error}") from error
    return (os.getpid(), int(compile_cache_info()["compile_misses"]) - before_cute,
            _WORKER_TRITON_COMPILES - before_triton, expected)


@dataclass(frozen=True, kw_only=True)
class CompilationSummary:
    jobs: int
    requested_jobs: int
    cute_programs: int
    triton_programs: int
    workers_used: int
    cute_compilations: int
    triton_compilations: int
    peak_parallel_cute_compilations: int
    elapsed_seconds: float


class CompilePool:
    """Deduplicate exact program keys, then pipeline ready complete plan sets."""

    def __init__(
        self,
        *,
        device_ordinal: int,
        compute_capability: tuple[int, int],
        device_uuid: str,
        product_name: str,
        sm_count: int,
        max_shared_memory_per_block: int,
        max_shared_memory_per_multiprocessor: int,
        workers: int,
    ):
        if type(device_ordinal) is not int or device_ordinal < 0:
            raise ValueError("compiler pool needs a nonnegative visible device ordinal")
        if type(workers) is not int or workers <= 0:
            raise ValueError("compiler worker count must be positive")
        compute_capability = tuple(compute_capability)
        if (
            len(compute_capability) != 2
            or any(type(value) is not int or value < 0 for value in compute_capability)
        ):
            raise ValueError("compiler pool needs a two-part compute capability")
        device_uuid = str(device_uuid).strip()
        if not device_uuid:
            raise ValueError("compiler pool needs a device UUID")
        product_name = str(product_name).strip()
        if not product_name:
            raise ValueError("compiler pool needs a device product name")
        if type(sm_count) is not int or sm_count <= 0:
            raise ValueError("compiler pool needs a positive SM count")
        if (
            type(max_shared_memory_per_block) is not int
            or max_shared_memory_per_block <= 0
        ):
            raise ValueError("compiler pool needs a positive shared-memory limit")
        if (
            type(max_shared_memory_per_multiprocessor) is not int
            or max_shared_memory_per_multiprocessor <= 0
        ):
            raise ValueError(
                "compiler pool needs a positive multiprocessor shared-memory limit"
            )
        self._started = time.monotonic()
        self._condition = Condition()
        self._plans: dict[bytes, CompilationPlan] = {}
        self._submitted: set[ProgramKey] = set()
        self._cute_programs = self._triton_programs = 0
        self._completed: set[ProgramKey] = set()
        self._required: list[CompilationPlan] = []
        self._optional: list[CompilationPlan] = []
        self._inflight = 0
        self._limit = workers
        self._optional_cancelled = False
        self._errors: list[BaseException] = []
        self._pids = set()
        self._cute_count = self._triton_count = self._requested_jobs = self._jobs = 0
        context = multiprocessing.get_context("spawn")
        self._activity = context.Array("q", (0, 0))
        with _offline_compiler_spawn_environment():
            self._pool: Pool | None = context.Pool(
                processes=workers, initializer=_initialize_worker,
                initargs=(
                    device_ordinal,
                    compute_capability,
                    device_uuid,
                    product_name,
                    sm_count,
                    max_shared_memory_per_block,
                    max_shared_memory_per_multiprocessor,
                    self._activity,
                ),
            )
        self._workers = tuple(self._pool._pool)

    def _finish(self, result):
        pid, cute, triton, programs = result
        with self._condition:
            self._pids.add(pid)
            self._cute_count += cute
            self._triton_count += triton
            self._completed.update(programs)
            self._inflight -= 1
            self._dispatch()
            self._condition.notify_all()

    def _fail(self, error):
        with self._condition:
            self._errors.append(error)
            self._inflight -= 1
            self._condition.notify_all()

    def plan(self, jobs: Iterable[CompileJob]) -> tuple[CompilationPlan, ...]:
        result = []
        for job in jobs:
            payload = pickle.dumps(job, protocol=5)
            if payload not in self._plans:
                self._plans[payload] = describe_compilation(job)
            result.append(self._plans[payload])
        return tuple(result)

    def _dispatch(self):
        # Called under the Condition. Pool.apply_async's queue is bounded here,
        # not merely by the number of worker processes consuming that queue.
        while self._pool is not None and self._inflight < self._limit and not self._errors:
            queue = self._required if self._required else self._optional
            if not queue:
                break
            plan = queue.pop(0)
            added = set(plan.programs) - self._submitted
            if not added:
                continue
            self._submitted.update(added)
            self._cute_programs += sum(p.dialect == "cute" for p in added)
            self._triton_programs += sum(p.dialect == "triton" for p in added)
            self._jobs += 1
            self._inflight += 1
            self._pool.apply_async(
                _run_job, (pickle.dumps(plan.job, protocol=5), plan.programs),
                callback=self._finish, error_callback=self._fail,
            )

    def submit_plans(self, plans: Iterable[CompilationPlan], *, required=True) -> tuple[ProgramKey, ...]:
        programs = set()
        with self._condition:
            if self._pool is None:
                raise RuntimeError("compiler pool is closed")
            self._raise_errors()
            for plan in plans:
                self._requested_jobs += 1
                programs.update(plan.programs)
                if not required and self._optional_cancelled:
                    continue
                missing = set(plan.programs) - self._submitted
                if not missing:
                    continue
                if required:
                    promoted = [queued for queued in self._optional if missing.intersection(queued.programs)]
                    self._optional = [queued for queued in self._optional if queued not in promoted]
                    self._required.extend(promoted)
                queued_programs = {
                    program for queued in (*self._required, *self._optional)
                    for program in queued.programs
                }
                if missing - queued_programs:
                    (self._required if required else self._optional).append(plan)
            self._dispatch()
        return tuple(sorted(programs, key=lambda p: (p.dialect, p.key)))

    def submit(self, jobs: Iterable[CompileJob], *, required=True) -> tuple[ProgramKey, ...]:
        return self.submit_plans(self.plan(jobs), required=required)

    def cancel_optional(self):
        with self._condition:
            self._optional_cancelled = True
            self._optional.clear()
            self._condition.notify_all()

    def wake(self):
        with self._condition:
            self._condition.notify_all()

    def _raise_errors(self):
        if self._errors:
            raise self._errors[0]
        exited = [(worker.pid, worker.exitcode) for worker in self._workers if worker.exitcode is not None]
        if exited:
            raise RuntimeError(f"preparation compiler worker exited: {exited}")

    @property
    def pending(self):
        with self._condition:
            self._raise_errors()
            return bool(self._required or self._optional or self._inflight)

    def wait_for_progress(self, timeout=0.05):
        with self._condition:
            self._raise_errors()
            if self._inflight:
                self._condition.wait(timeout=min(0.05, max(0.0, timeout)))
            self._raise_errors()

    def ready(self, batch: tuple[ProgramKey, ...]) -> bool:
        with self._condition:
            self._raise_errors()
            return all(program in self._completed for program in batch)

    def wait_any(self, batches: Iterable[tuple[ProgramKey, ...]], *, deadline=None):
        batches = tuple(batches)
        with self._condition:
            while True:
                exited = [(worker.pid, worker.exitcode) for worker in self._workers if worker.exitcode is not None]
                if exited:
                    raise RuntimeError(f"startup compiler worker exited: {exited}")
                if self._errors:
                    raise self._errors[0]
                if any(all(program in self._completed for program in batch) for batch in batches):
                    return
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("explicit compilation deadline exceeded")
                self._condition.wait(timeout=.1 if remaining is None else min(.1, remaining))

    def wait(self, batch, *, deadline=None):
        self.wait_any((batch,), deadline=deadline)

    @property
    def active_compilations(self):
        with self._activity.get_lock():
            return int(self._activity[0])

    @property
    def programs(self):
        return frozenset(self._submitted)

    def summary(self):
        with self._condition:
            return CompilationSummary(
                jobs=self._jobs, requested_jobs=self._requested_jobs,
                cute_programs=self._cute_programs,
                triton_programs=self._triton_programs,
                workers_used=len(self._pids), cute_compilations=self._cute_count,
                triton_compilations=self._triton_count,
                peak_parallel_cute_compilations=int(self._activity[1]),
                elapsed_seconds=time.monotonic() - self._started,
            )

    def compile(self, jobs, *, deadline=None):
        required = self.submit(jobs)
        self.wait(required, deadline=deadline)
        return self.summary()

    def close(self, *, terminate=False):
        if self._pool is None:
            return
        try:
            if not terminate:
                while self.pending:
                    self.wait_for_progress()
        except BaseException:
            pool, self._pool = self._pool, None
            pool.terminate()
            pool.join()
            raise
        pool, self._pool = self._pool, None
        pool.terminate() if terminate else pool.close()
        pool.join()

    def __enter__(self):
        return self

    def __exit__(self, kind, _value, _traceback):
        self.close(terminate=kind is not None)


__all__ = ["CompilationPlan", "CompilationSummary", "CompileJob", "CompilePool", "describe_compilation"]
