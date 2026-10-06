"""One serial preparation lifecycle with cooperative compilation and tuning.

The session fills each plan's prepared slot in place. Plans that are already
prepared are skipped, so a later batch is incremental. Candidate races are
timed in bounded batches; the running best candidate is carried into every
later batch so each candidate is compared head to head with it.
"""
from __future__ import annotations

import logging
import gc
import math
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from b12x._lib.compile_plan import (
    compiled_program_available, observe_programs, retain_compiled_programs,
)
from b12x._lib.compile_pool import CompilePool, describe_compilation, compile_in_process
from b12x._lib.program_cache import PreparationProgramCache, evict_unretained
from b12x._lib.runtime_control import KernelResolutionFrozenError, kernel_resolution_guard
from ._cache import SelectionCache, cache_identity, digest
from ._measurement import DEFAULT_SAMPLES, SURVIVOR_ROUNDS
from ._timing import PreparationTiming
from .device import DetectedDevice, detect_device
from .types import (
    CollectiveRequirement, FrozenMapping, MemoryRequirements, Plan,
    PreparationProgress, PreparationRequest, PreparationResult, PreparedCall,
    Selection, TuningCacheRequirement, TuningRequirement, _CompositePlan, _Prepared, _close_all,
    _plan_scope, _prime, _set_lazy_preparer, call_scope,
)

logger = logging.getLogger("b12x.preparation")


class _CandidateLaunchRejected(Exception):
    """A trial could not launch, but its resources were safely released."""


def _cooperative_launch_rejected(error):
    from cutlass.base_dsl.common import DSLCudaRuntimeError

    return isinstance(error, DSLCudaRuntimeError) and error.error_code == 720


@dataclass
class _Obligation:
    request: PreparationRequest
    configuration: object
    requests: tuple = ()
    candidates: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    selection: Selection | None = None
    key: str | None = None
    programs: dict = field(default_factory=dict)
    compiled: dict = field(default_factory=dict)
    compile_assignments: dict = field(default_factory=dict)
    cache_pending: bool = False
    ready: bool = False
    planned_candidates: int = 0


@dataclass(frozen=True)
class _TuningBatch:
    contributions: tuple[TuningRequirement, ...]


class _CallGuard:
    """Run a prepared call's restore and close callbacks at most once each."""

    def __init__(self, call):
        if not isinstance(call, PreparedCall):
            raise TypeError("call factory must return PreparedCall")
        self.call = call
        self._restored = self._closed = False

    def restore(self):
        if not self._restored:
            self._restored = True
            if self.call.restore is not None:
                with call_scope():
                    self.call.restore()

    def close(self):
        if not self._closed:
            self._closed = True
            if self.call.close is not None:
                self.call.close()

    def finish(self):
        try:
            _close_all((self.restore, self.close))
        finally:
            self.call = None


@dataclass
class _Trial:
    index: int
    assignment: object
    config: object
    call: PreparedCall
    guard: _CallGuard
    retained: object
    state: object
    resident: int
    latency_us: float | None = None

    def close(self):
        try:
            self.guard.finish()
        finally:
            self.call = self.state = self.retained = None


# Local metadata and GPU steps share one time slice. Collective readiness and
# pending compilation always return control before more work is admitted.
_ADVANCE_SECONDS = 0.1

# Cross-rank priming assumes co-scheduled ranks: a collective authorization is
# consumed in the round after the exchange, and the local work before the launch
# (autotune rounds) is unbounded, so the barrier converts the shared
# authorization into a launch lockstep. Generous by default: it only ever waits
# during preparation, and must cover the slowest rank's pre-launch round.
def _barrier_timeout_seconds():
    """Return the finite positive collective-barrier deadline from the environment."""
    try:
        timeout = float(os.environ.get("B12X_COLLECTIVE_BARRIER_TIMEOUT", "120"))
    except ValueError as error:
        raise ValueError("B12X_COLLECTIVE_BARRIER_TIMEOUT must be a finite positive number of seconds") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("B12X_COLLECTIVE_BARRIER_TIMEOUT must be a finite positive number of seconds")
    return timeout


class CollectiveBarrierTimeout(RuntimeError):
    """A collective barrier callback exceeded its deadline."""

    def __init__(self, key: str, ranks: tuple[int, ...], arrived: tuple[int, ...] | None = None, *, timeout: float = 120.0):
        """Record verified arrivals when the barrier backend supplies them."""
        if arrived is None:
            message = f"collective barrier callback timed out for {key!r} after {timeout:g}s"
        else:
            waiting = tuple(sorted(set(ranks) - set(arrived)))
            message = (
                f"collective barrier timed out for {key!r} after "
                f"{timeout:g}s: ranks {waiting} never entered "
                f"(arrived: {tuple(sorted(arrived))})"
            )
        super().__init__(message)
        self.key = key
        self.ranks = ranks
        self.arrived = None if arrived is None else tuple(sorted(arrived))
        self.timeout = timeout


def _declaration_key(plan):
    """Return the immutable declaration identity used to coalesce requests."""
    if isinstance(plan, _CompositePlan):
        return (
            plan.component_id, plan.composite_semantic_version, plan.capacity_metadata,
            tuple((count, _declaration_key(child)) for count, child in plan.variants.items()),
        )
    contract = plan.contract
    return (
        contract.component_id, contract.query_schema_version, contract.config_schema_version,
        contract.semantic_version, contract.candidate_contract_version,
        FrozenMapping(contract.encode_query(plan.query)), plan.invocation,
        None if plan.override is None else contract.config_payload(plan.override),
        plan._device,
    )


def _topological(requests):
    by_name = {}
    for request in requests:
        if not isinstance(request, PreparationRequest):
            raise TypeError("preparation batches contain PreparationRequest values")
        if request.name in by_name:
            raise ValueError(f"duplicate preparation name {request.name!r}")
        by_name[request.name] = request
    missing = {name for request in requests for name in request.dependencies} - by_name.keys()
    if missing:
        raise ValueError(f"unknown preparation dependencies: {sorted(missing)}")
    ordered, visiting, visited = [], set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("preparation dependencies contain a cycle")
        if name in visited:
            return
        visiting.add(name)
        for dependency in by_name[name].dependencies:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(by_name[name])

    for name in by_name:
        visit(name)
    return ordered


def _child_requests(request):
    plan = request.plan
    return {
        count: child.request(
            name=f"{request.name}/m{count}",
            prepare_call=request.prepare_call[count],
            benchmark_call=None if request.benchmark_call is None else request.benchmark_call[count],
            dependencies=request.dependencies, collective=request.collective,
            retain_benchmark_call=request.retain_benchmark_call,
        )
        for count, child in plan.variants.items()
    }


def _expand_requests(requests):
    requests = list(_topological(requests))
    expanded, composites = [], {}
    for request in requests:
        if request.plan.shared and (request.dependencies or request.collective is not None):
            raise ValueError(f"shared plan {request.name!r} cannot declare dependencies or collectives")
        if isinstance(request.plan, _CompositePlan):
            children = _child_requests(request)
            composites[request.name] = (request, children)
            expanded.extend(children.values())
        else:
            expanded.append(request)

    from dataclasses import replace

    normalized = []
    for request in expanded:
        dependencies = []
        for name in request.dependencies:
            if name in composites:
                dependencies.extend(child.name for child in composites[name][1].values())
            else:
                dependencies.append(name)
        normalized.append(replace(request, dependencies=tuple(dict.fromkeys(dependencies))))
    return _topological(normalized), composites


def _coalesce_requests(requests):
    """Group declaration-equivalent work while preserving its runtime bindings."""
    groups, keys = {}, {}
    for request in requests:
        key = (
            _declaration_key(request.plan),
            tuple(dict.fromkeys(keys[name] for name in request.dependencies)),
            None if request.collective is None else request.collective.ranks,
        )
        keys[request.name] = key
        groups.setdefault(key, []).append(request)
    return tuple(tuple(group) for group in groups.values())


def _plans_of(request):
    """Return a request plan and every child plan of a composite request."""
    plan = request.plan
    if isinstance(plan, _CompositePlan):
        return (plan, *plan.variants.values())
    return (plan,)


def _default_compile_workers(device):
    identity = device.identity
    spark = (
        identity is not None
        and identity.vendor == "nvidia"
        and identity.product_name in {"gb10", "nvidia gb10"}
    )
    return int(os.environ.get("B12X_COMPILE_WORKERS", "4" if spark else "8"))


class PreparationSession:
    def __init__(
        self, *, device=None, autotune=True, cache_dir=None, namespace=None,
        compile_workers=None, rounds=SURVIVOR_ROUNDS, samples=DEFAULT_SAMPLES, cache_only=False,
        race_batch=32, race_budget=None, collective_barrier=None,
    ):
        """Initialize planning resources and an optional collective entry barrier.

        ``collective_barrier(key, ranks)`` runs in a daemon thread and must be
        safe to call there. A timeout does not cancel that callback; another
        preparation job cannot begin until the callback exits.
        """
        self.device = device if isinstance(device, DetectedDevice) else detect_device(device)
        if compile_workers is None:
            compile_workers = _default_compile_workers(self.device)
        for name, value in (
            ("compile_workers", compile_workers), ("rounds", rounds), ("samples", samples),
            ("race_batch", race_batch),
        ):
            minimum = 0 if name == "compile_workers" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if race_budget is not None and (type(race_budget) is not int or race_budget <= 0):
            raise ValueError("race_budget must be a positive byte count or None")
        if type(autotune) is not bool or type(cache_only) is not bool:
            raise TypeError("autotune and cache_only must be boolean")
        self.autotune, self.cache_only = autotune, cache_only
        self.compile_workers, self.rounds, self.samples = compile_workers, rounds, samples
        self.race_batch, self.race_budget = race_batch, race_budget
        self.namespace = FrozenMapping(namespace or {})
        if collective_barrier is not None and not callable(collective_barrier):
            raise TypeError("collective_barrier must be callable")
        self.collective_barrier = collective_barrier
        self._barrier_timeout = (
            _barrier_timeout_seconds() if collective_barrier is not None else 120.0
        )
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._cache = None
        self._stop = threading.Event()
        if os.environ.get("B12X_AUTOTUNE", "1") == "0":
            self._stop.set()
        self._thread = threading.get_ident()
        self._pool = None
        self._job = None
        self._pending_collective_barrier = None
        self._plans = []
        self._shared = {}
        self._guard = None
        self._tuning_rank = 0
        self._tuning_ranks = (0,)
        self._tuning_cache_synchronized = False
        self.state = "OPEN"

    def configure_compile_workers(self, workers: int | None = None) -> None:
        """Change compiler concurrency between preparation jobs."""
        self._check_thread()
        if self._job is not None or self._pool is not None:
            raise RuntimeError("compiler concurrency can only change between jobs")
        if workers is None:
            workers = _default_compile_workers(self.device)
        if type(workers) is not int or workers < 0:
            raise ValueError("compile_workers must be an integer of at least 0")
        self.compile_workers = workers

    def configure_tuning_shard(self, rank: int, ranks: tuple[int, ...]) -> None:
        """Assign this process a disjoint share of non-collective candidates."""
        self._check_thread()
        if self._job is not None:
            raise RuntimeError("cannot change tuning ranks during preparation")
        ranks = tuple(ranks)
        if (
            type(rank) is not int
            or rank < 0
            or not ranks
            or any(type(item) is not int or item < 0 for item in ranks)
            or ranks != tuple(sorted(set(ranks)))
            or rank not in ranks
        ):
            raise ValueError(
                "tuning ranks must be sorted, unique, and include this rank"
            )
        self._tuning_rank = rank
        self._tuning_ranks = ranks
        self._tuning_cache_synchronized = False

    def _check_thread(self):
        """Reject use from another thread or after the session closes."""
        if threading.get_ident() != self._thread:
            raise RuntimeError("only cancel_tuning may run on another thread")
        if self.state == "CLOSED":
            raise RuntimeError("preparation session is closed")

    def _selection_cache(self):
        """Return the session selection cache, creating it for CUDA devices."""
        if self._cache is None:
            if self.device.ordinal is None:
                raise RuntimeError("selection caching requires a CUDA device")
            root = self.cache_dir
            if root is None:
                from b12x._lib.compiler import _cute_compile_cache_dir
                root = _cute_compile_cache_dir() / "preparation"
            self._cache = SelectionCache(root, cache_identity(self.namespace.to_dict(), self.device.ordinal))
        return self._cache

    def _gpu_scope(self):
        """Return the CUDA device context for preparation operations."""
        if self.device.ordinal is None:
            return nullcontext()
        import torch
        return torch.cuda.device(self.device.ordinal)

    def _synchronize(self):
        """Synchronize the preparation device when it is CUDA-backed."""
        if self.device.ordinal is not None:
            import torch
            torch.cuda.synchronize(self.device.ordinal)

    def _allocated(self):
        """Return bytes currently allocated on the preparation device."""
        if self.device.ordinal is None:
            return 0
        import torch
        return torch.cuda.memory_allocated(self.device.ordinal)

    def _race_budget(self):
        """Return the configured or device-derived autotuning memory budget."""
        if self.race_budget is not None:
            return self.race_budget
        if self.device.ordinal is None:
            return None
        import torch
        free, _ = torch.cuda.mem_get_info(self.device.ordinal)
        return max(int(free) // 2, 1)

    def begin(self, requests, *, autotune=None):
        """Start one preparation job after the prior collective barrier drains."""
        self._check_thread()
        if self._job is not None:
            raise RuntimeError("one preparation job may be active")
        pending = self._pending_collective_barrier
        if pending is not None:
            if not pending.is_set():
                raise RuntimeError("previous collective barrier is still running")
            self._pending_collective_barrier = None
        if autotune is not None and type(autotune) is not bool:
            raise TypeError("job autotune override must be boolean or None")
        requests = tuple(requests)
        _topological(requests)
        if self.state == "FROZEN":
            for request in requests:
                if any(plan.prepared is None for plan in _plans_of(request)):
                    raise KernelResolutionFrozenError("new preparation obligation after session freeze")
        job = PreparationJob(
            self,
            requests,
            autotune=self.autotune if autotune is None else autotune,
        )
        self._job = job
        if self.state != "FROZEN":
            self.state = "PREPARING"
        return job

    def prepare(
        self, requests, *, coordinator=None, progress=None, autotune=None
    ):
        requests = tuple(requests)
        if coordinator is None and any(request.collective is not None for request in requests):
            raise ValueError("collective preparation requires a coordinator")
        job = self.begin(requests, autotune=autotune)
        authorization = {}
        while True:
            state = job.advance(**authorization)
            if progress is not None:
                progress(state)
            if state.done:
                return job.result()
            if state.ready_cache is not None or state.ready_tuning:
                if coordinator is None:
                    raise ValueError("sharded preparation requires a coordinator")
                key = "cache" if state.ready_cache is not None else "tuning"
                authorization = {key: coordinator(state)}
            else:
                authorization = {"collective_key": coordinator(state) if coordinator is not None else None}
            if state.pending_compilation and self._pool is not None:
                self._pool.wait_for_progress()

    def candidate_memory_envelope(self, requests):
        """Return the largest scratch envelope across every legal config."""
        self._check_thread()
        if self._job is not None:
            raise RuntimeError("cannot inspect memory while preparation is active")
        expanded, _ = _expand_requests(tuple(requests))
        requirements = []
        for requests in _coalesce_requests(expanded):
            plan = requests[0].plan
            configuration = plan.contract.configure(
                plan.query, device=self.device.identity, override=plan.override,
            )
            configs = ((configuration.pinned,) if configuration.pinned is not None else (
                configuration.default,
                *(config for _, config in plan.contract.iterate(configuration)),
            ))
            plans, shared = [], False
            for request in requests:
                candidate = request.plan
                if candidate in plans or (candidate.shared and shared):
                    continue
                plans.append(candidate)
                shared |= candidate.shared
            seen = set()
            for config in configs:
                payload = plan.contract.config_payload(config)
                if payload in seen:
                    continue
                seen.add(payload)
                for candidate in plans:
                    with _plan_scope(candidate):
                        requirements.append(candidate._memory_requirements(config, self.device))
        return MemoryRequirements.sequential(requirements)

    def cancel_tuning(self):
        self._stop.set()
        pool = self._pool
        if pool is not None:
            pool.wake()

    def _compiler(self):
        if self._pool is None:
            if (
                self.device.ordinal is None
                or self.device.identity is None
                or self.device.uuid is None
                or self.device.max_shared_memory_per_block is None
                or self.device.max_shared_memory_per_multiprocessor is None
            ):
                raise RuntimeError("kernel compilation requires a CUDA device")
            self._pool = CompilePool(
                device_ordinal=self.device.ordinal,
                compute_capability=self.device.identity.compute_capability,
                device_uuid=self.device.uuid,
                product_name=self.device.identity.product_name,
                sm_count=self.device.identity.sm_count,
                max_shared_memory_per_block=self.device.max_shared_memory_per_block,
                max_shared_memory_per_multiprocessor=(
                    self.device.max_shared_memory_per_multiprocessor
                ),
                workers=self.compile_workers,
            )
        return self._pool

    def _drain(self):
        if self._pool is not None:
            if self._stop.is_set():
                pool, self._pool = self._pool, None
                summary = pool.summary()
                pool.close(terminate=True)
                return summary
            while self._pool.pending:
                yield "compile"
            summary = self._pool.summary()
            self._pool.close()
            self._pool = None
            return summary
        return None

    def _release_payload(self, plan, prepared):
        if plan in prepared.users:
            prepared.users.remove(plan)
        if plan._prepared is prepared:
            plan._clear()
        if plan in self._plans and plan._prepared is None:
            self._plans.remove(plan)
        if plan.shared:
            key = _declaration_key(plan)
            if self._shared.get(key) is plan and plan.prepared is not prepared:
                if plan.prepared is None:
                    replacement = next((user for user in prepared.users if user.shared), None)
                    if replacement is None:
                        self._shared.pop(key)
                    else:
                        self._shared[key] = replacement
        if prepared.users or prepared.closed:
            return
        prepared.closed = True
        closers, prepared.closers = prepared.closers, ()
        try:
            _close_all(closers)
        finally:
            prepared.retained = None
            prepared.state = None
            prepared.owners = ()

    def release(self, plan):
        """Drop a plan's prepared state; destroy graphs that replay it first."""
        self.release_many((plan,))

    def release_many(self, plans) -> None:
        """Release a batch after graph teardown, then reclaim unused programs once."""
        self._check_thread()
        if self._job is not None:
            raise RuntimeError("cannot release resources during preparation")
        pending = {}

        def collect(plan):
            if plan in pending:
                return
            pending[plan] = plan._prepared
            if isinstance(plan, _CompositePlan):
                for child in plan.variants.values():
                    collect(child)

        for plan in plans:
            collect(plan)
        closers = [lambda plan=plan, prepared=prepared: self._release_payload(plan, prepared)
                   for plan, prepared in pending.items() if prepared is not None]
        if not closers:
            return
        try:
            _close_all((self._synchronize, *closers))
        finally:
            _close_all((self._reclaim_programs,))

    def _reclaim_programs(self):
        timing = PreparationTiming("cleanup", rank=self._tuning_rank)
        keep = frozenset(program for plan in self._plans if plan.prepared is not None
                         for program in plan.prepared.programs)
        with timing.span("program_reclamation"):
            removed = evict_unretained(keep)
            gc.collect()
        timing.record("complete", evicted_entries=removed)

    @contextmanager
    def capture(self):
        """Refuse kernel resolution for the duration of a CUDA graph capture."""
        self._check_thread()
        if self._job is not None or self._pool is not None:
            raise RuntimeError("capture requires drained preparation")
        with kernel_resolution_guard("prepared CUDA graph capture"):
            yield

    def freeze(self):
        """Forbid later kernel resolution after all preparation work drains."""
        self._check_thread()
        if self._job is not None or self._pool is not None:
            raise RuntimeError("freeze requires drained preparation")
        if self._guard is None:
            self._guard = kernel_resolution_guard("prepared serving session")
            self._guard.__enter__()
        self.state = "FROZEN"

    def close(self):
        """Release preparation resources and mark the session closed."""
        if self.state == "CLOSED":
            return
        self._check_thread()
        synchronize = bool(self._plans) or self._job is not None
        closers = []
        if self._job is not None:
            closers.append(self._job.close)
        if self._pool is not None:
            pool, self._pool = self._pool, None
            pool.cancel_optional()
            closers.append(pool.close)
        for plan in reversed(tuple(self._plans)):
            prepared = plan._prepared
            if prepared is not None:
                closers.append(lambda plan=plan, prepared=prepared: self._release_payload(plan, prepared))
        if self._guard is not None:
            guard, self._guard = self._guard, None
            closers.append(lambda: guard.__exit__(None, None, None))
        self.state = "CLOSED"
        try:
            _close_all(((self._synchronize,) if synchronize else ()) + tuple(closers))
        finally:
            self._plans.clear()
            self._shared.clear()
            _close_all((self._reclaim_programs,))

    def __enter__(self):
        """Enter the session after validating the owning thread."""
        self._check_thread()
        return self

    def __exit__(self, kind, value, traceback):
        """Close session resources and preserve a body exception if cleanup fails."""
        try:
            self.close()
        except BaseException as error:
            if value is None:
                raise
            value.add_note(f"session cleanup failed: {error!r}")


class PreparationJob:
    def __init__(self, session, requests, *, autotune):
        """Initialize state for one bounded sequence of preparation requests."""
        self.session, self.requests = session, requests
        self.autotune = autotune
        self._steps = self._run()
        self._blocked = None
        self._result = None
        self._error = None
        self._closed = False
        self._temporaries = []
        self._new_plans = []
        self._benchmark_closers = []
        self._cache_hits = self._benchmarked = self._overlaps = 0
        self._all_programs = set()
        self._dependency_programs = {}
        self._coverage = {}
        self._benchmark_calls = {}
        self._plans_by_name = {}
        self._started = time.monotonic()
        self._timing = PreparationTiming("job", rank=session._tuning_rank)
        self._program_cache = PreparationProgramCache(self._timing)
        self._last_advance_end = None
        self._phase = "planning"
        self._active_request = None
        self._total_requests = len(requests)
        self._completed_requests = 0
        self._selection_counts = {}
        self._candidate_count = self._candidates_prepared = 0
        self._global_candidate_count = 0
        self._total_candidates = None
        self._candidate_sharded = False
        self._batch_index = self._batch_candidates = 0
        self._completed_rounds = self._total_rounds = self._active_count = 0
        self._latest_round_us = ()
        self._race_eviction = None
        self._compilations = 0
        self._stack_limit = None
        self._direct_ready = False
        if session.device.ordinal is not None:
            from b12x._lib.program_cache import stack_limit_bytes
            with session._gpu_scope():
                self._stack_limit = stack_limit_bytes()

    def _progress(
        self,
        running,
        pending_compilation,
        ready_collectives,
        done,
        ready_tuning=(),
        ready_cache=None,
    ):
        """Build a preparation progress snapshot from the live job state."""
        active_compilations = 0
        if self.session._pool is not None:
            summary = self.session._pool.summary()
            self._compilations = summary.cute_compilations + summary.triton_compilations
            active_compilations = self.session._pool.active_compilations
        phase = (
            "ready" if done else "waiting for ranks" if ready_collectives or ready_tuning or ready_cache
            else "compiling" if pending_compilation else self._phase
        )
        request = self._active_request
        return PreparationProgress(
            running, pending_compilation, ready_collectives, done,
            phase=phase, component_id="" if request is None else request.plan.component_id,
            request_name="" if request is None else request.name,
            completed_requests=self._completed_requests, total_requests=self._total_requests,
            candidate_count=self._candidate_count, candidates_prepared=self._candidates_prepared,
            measured_candidates=self._benchmarked, completed_rounds=self._completed_rounds,
            total_rounds=self._total_rounds if phase == "autotuning" else 0,
            active_count=self._active_count,
            latest_round_us=self._latest_round_us, cache_hits=self._cache_hits,
            compilations=self._compilations, active_compilations=active_compilations,
            elapsed_seconds=time.monotonic() - self._started,
            tuning_stopped=self._warmup_only and self.session._stop.is_set(),
            ready_tuning=tuple(ready_tuning),
            candidate_sharded=self._candidate_sharded,
            batch_index=self._batch_index, batch_candidates=self._batch_candidates,
            tuning_rank=self.session._tuning_rank,
            total_candidates=self._total_candidates, global_candidate_count=self._global_candidate_count,
            selection_counts=tuple(sorted(self._selection_counts.items())),
            ready_cache=ready_cache,
        )

    def _collective_barrier(self, requirement: CollectiveRequirement) -> None:
        """Convert a shared collective authorization into a launch lockstep.

        All participant ranks were authorized for the same requirement in the
        same exchange round, but the local work before any rank launches is
        unbounded (autotune races). The embedder-supplied barrier waits for
        every participant rank to enter before this rank proceeds to the
        launch, so the kernel-side spin window only covers launch jitter.
        """
        barrier = self.session.collective_barrier
        completed = threading.Event()
        errors = []

        def wait_for_peers():
            """Run the embedder barrier and publish its terminal state."""
            try:
                barrier(requirement.key, requirement.ranks)
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        try:
            thread = threading.Thread(target=wait_for_peers, daemon=True)
            self.session._pending_collective_barrier = completed
            try:
                thread.start()
            except BaseException:
                self.session._pending_collective_barrier = None
                raise
            if not completed.wait(self.session._barrier_timeout):
                raise CollectiveBarrierTimeout(
                    requirement.key, requirement.ranks, timeout=self.session._barrier_timeout,
                )
            if errors:
                raise errors[0]
        except CollectiveBarrierTimeout:
            raise
        except Exception as error:
            raise RuntimeError(
                f"collective barrier for {requirement.key!r} failed: {error}"
            ) from error

    def advance(self, *, collective_key=None, tuning=None, cache=None):
        """Advance preparation until work completes or reaches a readiness boundary."""
        started = time.perf_counter()
        if self._last_advance_end is not None:
            self._timing.add("between_advances", started - self._last_advance_end)
        try:
            with self._program_cache.activate():
                return self._advance(collective_key=collective_key, tuning=tuning, cache=cache)
        finally:
            if self._result is not None or self._closed:
                self._program_cache.clear()
            self._timing.add("advance", time.perf_counter() - started)
            self._timing.record(
                "complete" if self._result is not None else "progress",
                periodic=self._result is None and self._error is None,
                phase=self._phase, failed=self._error is not None,
                request=None if self._active_request is None else self._active_request.name,
                completed_requests=self._completed_requests,
            )
            self._last_advance_end = time.perf_counter()

    def _advance(self, *, collective_key=None, tuning=None, cache=None):
        """Run work until it reaches a readiness boundary or time slice."""
        self.session._check_thread()
        if self._error is not None:
            raise self._error
        if self._result is not None:
            return self._progress(False, False, (), True)
        if self._closed:
            raise RuntimeError("preparation job is closed")
        if self.session._stop.is_set() and self.session._pool is not None:
            pool, self.session._pool = self.session._pool, None
            pool.cancel_optional()
            pool.close(terminate=True)
        send_value = None
        if isinstance(self._blocked, TuningCacheRequirement):
            if cache is None:
                return self._progress(False, False, (), False, ready_cache=self._blocked)
            snapshots = tuple(cache)
            if not snapshots and self.session._stop.is_set():
                pass
            elif len(snapshots) != len(self._blocked.ranks) or any(
                not isinstance(item, TuningCacheRequirement) or item.ranks != self._blocked.ranks
                for item in snapshots
            ):
                raise ValueError("tuning cache agreement does not match participating ranks")
            self._blocked = None
            send_value = snapshots
        elif cache is not None:
            raise ValueError("tuning cache agreement does not match ready work")
        elif isinstance(self._blocked, CollectiveRequirement):
            if collective_key != self._blocked.key:
                return self._progress(False, False, (self._blocked,), False)
            if self.session.collective_barrier is not None:
                try:
                    self._collective_barrier(self._blocked)
                except BaseException as error:
                    self._error = error
                    try:
                        self.close()
                    except BaseException as cleanup:
                        error.add_note(f"job cleanup failed: {cleanup!r}")
                    raise
            self._blocked = None
        elif isinstance(self._blocked, _TuningBatch):
            required = self._blocked.contributions
            if tuning is None:
                return self._progress(False, False, (), False, required)
            winners = (tuning,) if isinstance(tuning, TuningRequirement) else tuple(tuning)
            expected = {item.key: item.ranks for item in required}
            if not winners and self.session._stop.is_set():
                pass
            elif (len(winners) != len(expected) or any(
                not isinstance(item, TuningRequirement)
                or expected.get(item.key) != item.ranks
                or item.assignment is None
                for item in winners
            ) or len({item.key for item in winners}) != len(winners)):
                raise ValueError("tuning consolidation does not match pending races")
            self._blocked = None
            send_value = winners
        elif isinstance(self._blocked, TuningRequirement):
            if tuning is None:
                return self._progress(False, False, (), False, (self._blocked,))
            if (
                not isinstance(tuning, TuningRequirement)
                or tuning.key != self._blocked.key
                or tuning.ranks != self._blocked.ranks
                or tuning.assignment is None
            ):
                raise ValueError("tuning authorization does not match ready work")
            self._blocked = None
            send_value = tuning
        elif collective_key is not None:
            raise ValueError("collective authorization does not match ready work")
        elif tuning is not None:
            raise ValueError("tuning authorization does not match ready work")
        try:
            deadline = time.monotonic() + _ADVANCE_SECONDS
            while True:
                started = time.perf_counter()
                try:
                    signal = self._steps.send(send_value)
                finally:
                    self._timing.add(self._phase, time.perf_counter() - started)
                send_value = None
                if isinstance(signal, TuningCacheRequirement):
                    self._blocked = signal
                    return self._progress(False, False, (), False, ready_cache=signal)
                if isinstance(signal, CollectiveRequirement):
                    self._blocked = signal
                    return self._progress(False, False, (signal,), False)
                if isinstance(signal, _TuningBatch):
                    self._blocked = signal
                    return self._progress(False, False, (), False, signal.contributions)
                if isinstance(signal, TuningRequirement):
                    self._blocked = signal
                    return self._progress(False, False, (), False, (signal,))
                if signal == "compile":
                    return self._progress(False, True, (), False)
                if time.monotonic() >= deadline:
                    break
        except StopIteration as finished:
            self._result = finished.value
            self.session._job = None
            if self.session.state != "FROZEN":
                self.session.state = "READY"
            return self._progress(False, False, (), True)
        except BaseException as error:
            self._error = error
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"job cleanup failed: {cleanup!r}")
            raise
        return self._progress(True, False, (), False)

    def result(self):
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("preparation is not complete")
        return self._result

    def close(self):
        if self._closed:
            return
        self._closed = True
        closers = [self._steps.close, *reversed(self._temporaries)]
        self._temporaries.clear()
        if self._result is None:
            closers.extend(reversed(self._benchmark_closers))
            self._benchmark_closers.clear()
            for plan in reversed(self._new_plans):
                prepared = plan._prepared
                if prepared is not None:
                    closers.append(
                        lambda plan=plan, prepared=prepared: self.session._release_payload(plan, prepared)
                    )
        if self.session._pool is not None:
            pool, self.session._pool = self.session._pool, None
            pool.cancel_optional()
            closers.append(lambda: pool.close(terminate=self._error is not None))
        self.session._job = None
        try:
            _close_all(closers)
        finally:
            self._program_cache.clear()
            if self._result is None:
                _close_all((self.session._reclaim_programs,))

    def _choice_key(self, obligation, selections):
        request, configuration = obligation.request, obligation.configuration
        dependencies = []
        for name in request.dependencies:
            if name not in selections:
                return None
            selection, contract = selections[name]
            dependencies.append({
                "component": selection.component_id, "query": selection.query.to_dict(),
                "config": contract.config_payload(selection.config).to_dict(),
                "semantic_version": contract.semantic_version,
            })
        contract = request.plan.contract
        return digest({
            "component": contract.component_id,
            "query_schema": contract.query_schema_version,
            "config_schema": contract.config_schema_version,
            "semantic_version": contract.semantic_version,
            "candidate_contract_version": contract.candidate_contract_version,
            "query": configuration.encoded_query.to_dict(),
            "invocation": request.plan.invocation.to_dict(),
            "pin": None if configuration.pinned is None else contract.config_payload(configuration.pinned).to_dict(),
            "dependencies": dependencies,
        })

    def _selection(self, obligation, config, source, assignment=None):
        return Selection(
            obligation.request.plan.component_id, obligation.configuration.encoded_query,
            config, source, assignment,
        )

    def _lookup(self, obligation, selections):
        if obligation.selection is not None:
            return
        obligation.key = self._choice_key(obligation, selections)
        if obligation.key is None:
            return
        with self._timing.span("cache_lookup"):
            cache = self.session._selection_cache()
            record = cache.get(obligation.key)
        if record is None:
            return
        configuration, contract = obligation.configuration, obligation.request.plan.contract
        assignment = FrozenMapping(record["assignment"])
        configuration.space.validate(assignment)
        config = contract._lower(configuration.query, configuration.device, assignment)
        if contract.config_payload(config) != record["config"]:
            raise ValueError("cached assignment no longer lowers to its saved config")
        obligation.selection = self._selection(obligation, config, "cached", assignment)
        obligation.coverage = dict(record["coverage"])
        self._cache_hits += 1

    def _alias_shared(self, obligation):
        """Install an equal shared declaration's payload; True when aliased."""
        plan = obligation.request.plan
        if not plan.shared:
            return False
        owner = self.session._shared.get(_declaration_key(plan))
        if owner is None or owner is plan or owner.prepared is None:
            return False
        prepared = owner.prepared
        if prepared.selection.source == "default" and self.autotune and not self.session._stop.is_set():
            return False
        previous = plan._prepared
        prepared.users.append(plan)
        plan._install(prepared)
        if previous is not None and previous is not prepared:
            self.session._release_payload(plan, previous)
        if plan not in self.session._plans:
            self.session._plans.append(plan)
        self._new_plans.append(plan)
        obligation.selection = prepared.selection
        obligation.ready = True
        return True

    @property
    def _warmup_only(self):
        return not self.session.cache_only and (not self.autotune or self.session._stop.is_set())

    def _direct_compilation(self):
        if self._direct_ready:
            return
        if self.session._pool is not None:
            pool, self.session._pool = self.session._pool, None
            pool.close(terminate=True)
        from b12x._lib.compile_plan import evict_planning_artifacts
        evict_planning_artifacts()
        self._direct_ready = True

    def _configure(self, groups):
        obligations, settled, enumerations, requirements = [], {}, {}, []
        for requests in groups:
            request = next((item for item in requests if item.plan.prepared is not None and (
                not self.autotune or self.session._stop.is_set()
                or item.plan.selection.source != "default"
            )), None)
            if request is None:
                request = next((item for item in requests if item.benchmark_call is not None), requests[0])
            requests = (request, *(item for item in requests if item is not request))
            self._active_request = request
            self._candidate_count = self._candidates_prepared = 0
            self._candidate_sharded = False
            plan = request.plan
            caps_device = plan._device
            if caps_device is None:
                caps_device = getattr(plan.query, "device", None)
            if caps_device is not None:
                import torch
                resolved = torch.device(caps_device)
                if resolved.type != "cuda" or resolved.index != self.session.device.ordinal:
                    raise ValueError("declaration and session devices differ")
            configuration = plan.contract.configure(
                plan.query, device=self.session.device.identity, override=plan.override,
                search=not self._warmup_only,
            )
            obligation = _Obligation(request, configuration, requests=requests)
            obligations.append(obligation)
            prepared = plan.prepared
            if (
                prepared is not None
                and (
                    not self.autotune
                    or self.session._stop.is_set()
                    or prepared.selection.source != "default"
                )
            ):
                obligation.selection = prepared.selection
                obligation.ready = True
            elif self._alias_shared(obligation):
                pass
            elif configuration.pinned is not None:
                obligation.selection = self._selection(obligation, configuration.pinned, "override")
            elif self._warmup_only:
                obligation.selection = self._selection(obligation, configuration.default, "default")
            else:
                single_product = all(len(knob.values) == 1 for knob in configuration.space.knobs)
                if not single_product:
                    self._lookup(obligation, settled)
                if (obligation.selection is None and not single_product
                        and not self.session.cache_only
                        and (not self.autotune or self.session._stop.is_set())):
                    obligation.selection = self._selection(obligation, configuration.default, "default")
                if obligation.selection is None:
                    declaration = _declaration_key(plan)
                    enumerated = enumerations.get(declaration)
                    if enumerated is not None:
                        candidates, coverage = enumerated
                        obligation.candidates.extend(candidates)
                        complete = True
                    else:
                        iterator = plan.contract.iterate(configuration)
                        while not iterator.done:
                            if (not single_product and self.session._stop.is_set()
                                    and not self.session.cache_only):
                                break
                            try:
                                candidate = iterator.step()
                            except StopIteration:
                                break
                            if candidate is not None:
                                obligation.candidates.append(candidate)
                            if self.session.cache_only and len(obligation.candidates) == 2:
                                break
                        complete = iterator.done
                        coverage = {
                            "cartesian_count": iterator.cartesian_count,
                            "legal_count": iterator.legal_count,
                            "effective_count": iterator.effective_count,
                            "measured_count": 0,
                        }
                        # Equal declarations enumerate to the same candidates; an
                        # interrupted enumeration describes the stop, not the space.
                        if complete:
                            enumerations[declaration] = (
                                tuple(obligation.candidates), coverage,
                            )
                    self._global_candidate_count = self._candidate_count = len(obligation.candidates)
                    if complete and not obligation.candidates:
                        raise ValueError(f"no eligible configurations for {plan.component_id}")
                    if complete and len(obligation.candidates) == 1:
                        assignment, config = obligation.candidates[0]
                        obligation.selection = self._selection(obligation, config, "fixed", assignment)
                    elif self.session._stop.is_set() and not self.session.cache_only:
                        obligation.selection = self._selection(obligation, configuration.default, "default")
                    elif self.session.cache_only and self._choice_key(obligation, settled) is not None:
                        raise LookupError(f"no completed selection for {request.name}")
                    if obligation.selection is None:
                        obligation.coverage = dict(coverage)
            for item in requests:
                if obligation.selection is not None:
                    settled[item.name] = (obligation.selection, item.plan.contract)
                if (item.plan.prepared is not None and item.plan.selection == obligation.selection
                        or item is not request and item.plan.shared and plan.shared):
                    continue
                config = (
                    configuration.default if obligation.selection is None
                    else obligation.selection.config
                )
                with _plan_scope(item.plan):
                    requirements.append(item.plan._memory_requirements(config, self.session.device))
            yield "metadata"
        MemoryRequirements.sequential(requirements)
        rank_index = self.session._tuning_ranks.index(self.session._tuning_rank)
        rank_count = len(self.session._tuning_ranks)
        for obligation in obligations:
            if obligation.selection is None:
                obligation.planned_candidates = len(obligation.candidates[rank_index::rank_count])
        self._total_candidates = sum(item.planned_candidates for item in obligations)
        return obligations

    def _compile(self, obligation, config, *, required):
        contract, plan = obligation.request.plan.contract, obligation.request.plan
        payload = contract.config_payload(config)
        if payload not in obligation.compiled:
            compiled = []
            for job in plan._compile_jobs(config, self.session.device):
                if self._warmup_only:
                    return frozenset()
                with self.session._gpu_scope():
                    with self._timing.span("compile_plan"):
                        description = describe_compilation(job)
                compiled.append(description)
                yield "metadata"
            obligation.compiled[payload] = tuple(compiled)
            obligation.programs[payload] = frozenset(p for item in compiled for p in item.programs)
        programs = obligation.programs[payload]
        self._all_programs.update(programs)
        missing = [item for item in obligation.compiled[payload] if any(
            not compiled_program_available(program) for program in item.programs
        )]
        if missing:
            if self._warmup_only:
                return programs
            if self.session.cache_only:
                raise LookupError(f"missing required compiled artifact for {obligation.request.name}")
            if self.session.compile_workers == 0:
                # No compiler workers: compile in this process, the way a
                # kernel compiles on first use without startup preparation.
                with self.session._gpu_scope():
                    compile_in_process(missing)
            else:
                pool = self.session._compiler()
                pool.submit_plans(missing, required=required)
        return programs

    def _wait_programs(self, programs):
        while not all(compiled_program_available(program) for program in programs):
            if self._warmup_only:
                return
            pool = self.session._pool
            if pool is None:
                raise LookupError("required compiler artifacts are unavailable")
            if pool.ready(tuple(programs)):
                raise RuntimeError("compiler completed without publishing required artifacts")
            yield "compile"
        if self.session._pool is not None:
            # Surface a factory failure even if another job published shared code.
            self.session._pool.pending

    def _instantiate(self, request, selection, factory, expected, *, synchronize=True, compile_directly=False, reject_unlaunchable=False):
        import torch

        from ._measurement import no_compilation
        compilation = nullcontext() if compile_directly else no_compilation()
        with _plan_scope(request.plan), self.session._gpu_scope(), compilation, retain_compiled_programs() as retained, observe_programs() as used:
            # Prepared resources outlive the caller's profiling scope and must
            # remain mutable during later serving warmup and graph replay.
            try:
                with torch.inference_mode(False), torch.no_grad():
                    with self._timing.span("materialize"):
                        state = request.plan._materialize(selection, self.session.device)
                    with self._timing.span("bind"):
                        call = factory(state)
                    guard = _CallGuard(call)
                    self._temporaries.append(guard.finish)
                rejected = None
                try:
                    with self._timing.span("prime"):
                        _prime(call)
                except Exception as error:
                    if not reject_unlaunchable or not _cooperative_launch_rejected(error):
                        raise
                    rejected = error
                if rejected is not None:
                    # Drain preceding launches before restoring scratch. Cleanup
                    # runs outside the handler so its failures remain fatal.
                    self.session._synchronize()
                    guard.finish()
                    self._temporaries.remove(guard.finish)
                    raise _CandidateLaunchRejected(
                        "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (720)"
                    ) from rejected
                if synchronize:
                    self.session._synchronize()
            except _CandidateLaunchRejected:
                raise
            except Exception as error:
                message = (
                    f"{request.name} failed to prepare with configuration "
                    f"{selection.config!r} ({selection.source}): {error}"
                )
                try:
                    wrapped = type(error)(message)
                except Exception:
                    wrapped = RuntimeError(message)
                raise wrapped from error
        if compile_directly:
            expected.update(used)
        if used - expected:
            raise RuntimeError(f"preparation used undeclared programs for {request.name}")
        return state, call, guard, retained

    def _call(self, obligation, selection, factory, *, request=None, synchronize=True, reject_unlaunchable=False):
        request = obligation.request if request is None else request
        plan = request.plan
        expected = obligation.programs[plan.contract.config_payload(selection.config)]
        expected = expected | frozenset(
            program for name in request.dependencies
            for program in self._dependency_programs[name]
        )
        with self._timing.span("materialize_prime"):
            return self._instantiate(
                request, selection, factory, expected, synchronize=synchronize,
                reject_unlaunchable=reject_unlaunchable,
            )

    def _trial(self, obligation, index, assignment, config):
        if self.session._stop.is_set():
            return None
        request = obligation.request
        programs = obligation.programs[request.plan.contract.config_payload(config)]
        while not all(compiled_program_available(program) for program in programs):
            if self.session._stop.is_set():
                return None
            yield "compile"
            if self.session._pool is not None:
                self.session._pool.pending
        selection = self._selection(obligation, config, "tuned", assignment)
        with self._timing.span("memory_accounting"):
            before = self.session._allocated()
        state, call, guard, retained = self._call(
            obligation, selection, request.benchmark_call, synchronize=False,
            reject_unlaunchable=True,
        )
        with self._timing.span("memory_accounting"):
            resident = max(0, self.session._allocated() - before)
        self._candidates_prepared += 1
        yield "gpu"
        return _Trial(index, assignment, config, call, guard, retained, state, resident)

    def _observe(self, race):
        self._completed_rounds = race.completed_rounds
        self._total_rounds = race.planned_rounds
        self._latest_round_us = race.latest_round_us
        self._active_count = race.active_count

    def _release_race_resources(self):
        self._race_eviction = None

    def _measure(self, trials, *, champion):
        from ._measurement import _l2_flush_fn, prepare_race_steps, measure_race_steps
        calls = [trial.call for trial in trials]
        race = None
        steps = measuring = None
        try:
            self._phase = "calibrating"
            if self._race_eviction is None and self.session.device.ordinal is not None:
                import torch
                with self.session._gpu_scope():
                    self._race_eviction = _l2_flush_fn(
                        torch.device("cuda", self.session.device.ordinal), enabled=True,
                    )
            steps = prepare_race_steps(
                calls, device_ordinal=self.session.device.ordinal, samples=self.session.samples,
                primed=True, eviction=self._race_eviction,
            )
            while not self.session._stop.is_set():
                try:
                    next(steps)
                except StopIteration as finished:
                    race = finished.value
                    break
                yield "gpu"
            if race is None:
                return None
            self._phase = "autotuning"
            measuring = measure_race_steps(
                race, device_ordinal=self.session.device.ordinal, rounds=self.session.rounds,
                compilation_active=lambda: self.session._pool is not None and self.session._pool.active_compilations,
                eliminate=True, champion=champion,
            )
            while not self.session._stop.is_set():
                try:
                    next(measuring)
                    self._observe(race)
                except StopIteration as finished:
                    self._observe(race)
                    return finished.value
                yield "gpu"
            return None
        finally:
            closers = []
            for generator in (steps, measuring):
                if generator is not None:
                    closers.append(generator.close)
            if race is not None:
                closers.append(race.close)
            with self._timing.span("race_cleanup"):
                _close_all(closers)

    def _race(self, obligation):
        request = obligation.request
        self._phase = "preparing candidates"
        rank_index = self.session._tuning_ranks.index(self.session._tuning_rank)
        rank_count = len(self.session._tuning_ranks)
        indexed_candidates = tuple(enumerate(obligation.candidates))
        local_candidates = indexed_candidates[rank_index::rank_count]
        self._candidate_count = len(local_candidates)
        self._candidate_sharded = rank_count > 1
        self._candidates_prepared = self._completed_rounds = self._active_count = 0
        self._total_rounds = 0
        self._latest_round_us = ()
        if request.benchmark_call is None:
            raise ValueError(f"multi-candidate preparation requires a representative benchmark: {request.name}")
        required_programs = ()
        if rank_count == 1:
            required_programs = yield from self._compile(
                obligation, obligation.configuration.default, required=True
            )
        for _, (assignment, config) in local_candidates:
            if self.session._stop.is_set():
                return None
            yield from self._compile(obligation, config, required=False)
            if self.session._stop.is_set():
                return None
            compile_assignment = obligation.configuration.space.compile_assignment(assignment)
            actual = obligation.programs[request.plan.contract.config_payload(config)]
            previous = obligation.compile_assignments.setdefault(compile_assignment, actual)
            if previous != actual:
                raise ValueError(f"runtime parameter changed actual compile keys for {request.name}")
        yield from self._wait_programs(required_programs)
        if not local_candidates:
            return TuningRequirement(
                obligation.key, self.session._tuning_ranks, None, None, None,
            )
        budget = self.session._race_budget()
        pending = list(local_candidates)
        champion = carried = None
        live = []
        rejected_count = 0
        try:
            while pending or carried is not None:
                if self.session._stop.is_set():
                    return None
                self._phase = "preparing candidates"
                self._batch_index += 1
                self._batch_candidates = 0
                self._completed_rounds = self._total_rounds = self._active_count = 0
                self._latest_round_us = ()
                batch, resident = [], 0 if champion is None else champion.resident
                if carried is not None:
                    batch.append(carried)
                    resident += carried.resident
                    carried = None
                while pending and len(batch) < self.session.race_batch:
                    index, (assignment, config) = pending.pop(0)
                    try:
                        trial = yield from self._trial(obligation, index, assignment, config)
                    except _CandidateLaunchRejected as error:
                        rejected_count += 1
                        logger.warning(
                            "Skipping %s candidate %d on rank %d: %r: %s",
                            request.name, index, self.session._tuning_rank, config, error,
                        )
                        self._timing.record(
                            "candidate_rejected", request=request.name,
                            candidate_index=index, reason=str(error),
                        )
                        continue
                    if trial is None:
                        return None
                    live.append(trial)
                    batch.append(trial)
                    resident += trial.resident
                    if budget is not None and resident > budget and len(batch) > 1:
                        carried = batch.pop()
                        break
                if not batch:
                    continue
                trials = ([] if champion is None else [champion]) + batch
                self._batch_candidates = len(trials)
                self._timing.record(
                    "batch_begin", request=request.name, batch=self._batch_index,
                    candidate_indices=[trial.index for trial in trials],
                    carried_champion=None if champion is None else champion.index,
                    local_candidates=len(local_candidates), remaining=len(pending),
                )
                measurement = yield from self._measure(trials, champion=champion is not None)
                if measurement is None:
                    return None
                # Global enumeration order breaks exact timing ties without
                # adding a second policy to the distributed selection.
                best = min(
                    range(len(trials)),
                    key=lambda position: (measurement.latencies_us[position], trials[position].index),
                )
                self._benchmarked += len(batch)
                self._overlaps += measurement.overlapped_samples
                with self._timing.span("trial_release"):
                    for position, trial in enumerate(trials):
                        if position != best:
                            trial.close()
                            live.remove(trial)
                champion = trials[best]
                champion.latency_us = float(measurement.latencies_us[best])
                self._timing.record(
                    "batch_end", request=request.name, batch=self._batch_index,
                    candidate_indices=[trial.index for trial in trials],
                    latencies_us=list(measurement.latencies_us),
                    winner=champion.index,
                )
            obligation.coverage["measured_count"] = len(local_candidates) - rejected_count
            if champion is None:
                if rank_count > 1:
                    return TuningRequirement(
                        obligation.key, self.session._tuning_ranks, None, None, None,
                        rejected_count=rejected_count,
                    )
                raise RuntimeError(
                    f"no launchable candidates for {request.name}: "
                    f"all {rejected_count} candidates rejected with "
                    "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (720)"
                )
            assignment, config = champion.assignment, champion.config
            candidate_index, latency_us = champion.index, champion.latency_us
            if rank_count > 1:
                return TuningRequirement(
                    obligation.key, self.session._tuning_ranks,
                    assignment, latency_us, candidate_index,
                    rejected_count=rejected_count,
                    cute_programs=tuple(
                        program.key for program in obligation.programs[
                            request.plan.contract.config_payload(config)
                        ] if program.dialect == "cute"
                    ),
                )
            selection = self._selection(obligation, config, "tuned", assignment)
            obligation.cache_pending = rejected_count == 0
            return selection
        finally:
            closers = [self.session._synchronize]
            closers.extend(trial.close for trial in live)
            if carried is not None and carried not in live:
                closers.append(carried.close)
            closers.append(self._release_race_resources)
            _close_all(closers)

    def _publish(self, request, selection, state, call, retained, programs, variants=None):
        plan = request.plan
        prepared = _Prepared(
            state=state, selection=selection, programs=programs, retained=retained,
            owners=() if call is None else tuple(call.owners),
            closers=() if call is None or call.close is None else (call.close,),
            device=self.session.device, variants=variants,
        )
        prepared.users.append(plan)
        previous = plan._prepared
        plan._install(prepared)
        if plan not in self.session._plans:
            self.session._plans.append(plan)
        if plan.shared:
            self.session._shared[_declaration_key(plan)] = plan
        self._new_plans.append(plan)
        self._plans_by_name[request.name] = plan
        if previous is not None and not previous.closed:
            self.session._release_payload(plan, previous)
        return prepared

    def _retain_benchmark(self, request, selection, expected):
        compile_directly = self._warmup_only and self.session.state != "FROZEN"
        if compile_directly:
            self._direct_compilation()
            expected = set(expected)
        state, call, guard, retained = self._instantiate(
            request, selection, request.benchmark_call, expected,
            compile_directly=compile_directly,
        )
        self._all_programs.update(expected)
        self._temporaries.remove(guard.finish)
        holder = (state, retained)
        self._benchmark_closers.append(lambda holder=holder: guard.finish())
        self._benchmark_calls[request.name] = call
        yield "gpu"

    def _install_obligation(self, obligation, selections):
        request, plan = obligation.request, obligation.request.plan
        selection = obligation.selection
        self._active_request = request
        if obligation.ready:
            programs = plan.prepared.programs
        else:
            programs = frozenset()
            if not self._warmup_only:
                programs = yield from self._compile(obligation, selection.config, required=True)
                yield from self._wait_programs(programs)
            if self._warmup_only:
                configuration = obligation.configuration
                source = "override" if configuration.pinned is not None else "default"
                selection = self._selection(obligation, configuration.default, source)
                obligation.selection = selection
                obligation.cache_pending = False
                obligation.coverage = {}
                programs = frozenset()
                self._direct_compilation()
            if obligation.cache_pending:
                with self._timing.span("cache_save"):
                    self.session._selection_cache().save(
                        obligation.key,
                        assignment=selection.assignment,
                        config=plan.contract.config_payload(selection.config),
                        coverage=obligation.coverage,
                        programs=programs,
                    )
        for item in obligation.requests:
            self._active_request = item
            item_plan = item.plan
            prepared = item_plan.prepared
            dependencies = frozenset(
                program for name in item.dependencies for program in self._dependency_programs[name]
            )
            if self._warmup_only:
                configuration = obligation.configuration
                source = "override" if configuration.pinned is not None else "default"
                selection = (prepared.selection if prepared is not None else
                             self._selection(obligation, configuration.default, source))
            if prepared is None or prepared.selection != selection:
                alias = _Obligation(item, obligation.configuration)
                if not self._alias_shared(alias):
                    if item.collective is not None:
                        yield item.collective
                    if self._warmup_only:
                        configuration = obligation.configuration
                        source = "override" if configuration.pinned is not None else "default"
                        selection = (prepared.selection if prepared is not None else
                                     self._selection(obligation, configuration.default, source))
                    if prepared is None or prepared.selection != selection:
                        self._phase = "warming heuristics" if self._warmup_only else "priming"
                        if self._warmup_only:
                            self._direct_compilation()
                            with _plan_scope(item_plan):
                                item_plan._memory_requirements(selection.config, self.session.device)
                            expected = set(dependencies)
                            with self._timing.span("materialize_prime"):
                                state, call, guard, retained = self._instantiate(
                                    item, selection, item.prepare_call, expected, compile_directly=True,
                                )
                            expected = frozenset(expected)
                            self._all_programs.update(expected)
                        else:
                            expected = programs | dependencies
                            state, call, guard, retained = self._call(
                                obligation, selection, item.prepare_call, request=item,
                            )
                        self._publish(item, selection, state, call, retained, expected)
                        self._temporaries.remove(guard.finish)
                        guard.restore()
                        del call, guard
                        yield "gpu"
            selection = item_plan.selection
            expected = item_plan.prepared.programs | dependencies
            selections[item.name] = (selection, item_plan.contract)
            self._dependency_programs[item.name] = expected
            self._plans_by_name[item.name] = item_plan
            self._coverage[item.name] = obligation.coverage
            if item.retain_benchmark_call:
                if item.collective is not None:
                    yield item.collective
                yield from self._retain_benchmark(item, selection, expected)
        self._timing.record(
            "request_end", request=request.name, source=selection.source,
            coverage=obligation.coverage,
        )
        source = plan.selection.source
        self._selection_counts[source] = self._selection_counts.get(source, 0) + 1
        self._completed_requests += 1

    def _consolidate(self, pending, selections):
        if not pending:
            return
        contributions = tuple(item[1] for item in pending)
        self._phase = "consolidating"
        winners = yield _TuningBatch(contributions)
        by_key = {winner.key: winner for winner in winners}
        for obligation, contribution in pending:
            configuration = obligation.configuration
            if self.session._stop.is_set():
                obligation.selection = self._selection(obligation, configuration.default, "default")
            else:
                winner = by_key[contribution.key]
                configuration.space.validate(winner.assignment)
                config = obligation.request.plan.contract._lower(
                    configuration.query, configuration.device, winner.assignment,
                )
                obligation.selection = self._selection(obligation, config, "tuned", winner.assignment)
                obligation.coverage["measured_count"] = len(obligation.candidates) - winner.rejected_count
                obligation.cache_pending = winner.rejected_count == 0
            self._timing.record("request_resume", request=obligation.request.name)
            yield from self._install_obligation(obligation, selections)
        pending.clear()

    def _run(self):
        if not self.requests:
            return PreparationResult(plans={})
        if (self.session.state != "FROZEN" and not self._warmup_only
                and len(self.session._tuning_ranks) > 1
                and not self.session._tuning_cache_synchronized):
            cache = self.session._selection_cache()
            snapshots = yield TuningCacheRequirement(
                self.session._tuning_ranks, cache.identity, cache.records,
            )
            if snapshots:
                cache.reconcile(snapshots)
                self.session._tuning_cache_synchronized = True
        if self.session.state == "FROZEN":
            requests, composites = self._expand()
            groups = _coalesce_requests(requests)
            self._total_requests = len(groups)
            for request in requests:
                plan = request.plan
                self._plans_by_name[request.name] = plan
                if request.retain_benchmark_call:
                    if request.collective is not None:
                        yield request.collective
                    yield from self._retain_benchmark(request, plan.selection, plan.prepared.programs)
            for name, (request, children) in composites.items():
                self._plans_by_name[name] = request.plan
            for group in groups:
                source = group[0].plan.selection.source
                self._selection_counts[source] = self._selection_counts.get(source, 0) + 1
            self._completed_requests = self._total_requests
            result = PreparationResult(
                plans=self._plans_by_name, benchmark_calls=self._benchmark_calls,
                benchmark_closers=self._benchmark_closers,
            )
            self._benchmark_closers = []
            closers, self._temporaries = self._temporaries, []
            _close_all(reversed(closers))
            return result
        requests, composites = self._expand()
        groups = _coalesce_requests(requests)
        self._total_requests = len(groups)
        obligations = yield from self._configure(groups)
        dependencies = {name for request in requests for name in request.dependencies}
        terminal_collectives = [
            obligation for obligation in obligations
            if obligation.request.collective is not None
            and not any(item.name in dependencies for item in obligation.requests)
        ]
        terminal_ids = {id(item) for item in terminal_collectives}
        obligations = [item for item in obligations if id(item) not in terminal_ids] + terminal_collectives
        selections = {}
        pending = []
        pending_names = set()
        for obligation in obligations:
            request = obligation.request
            plan = request.plan
            if pending_names.intersection(request.dependencies) or (pending and request.collective is not None):
                yield from self._consolidate(pending, selections)
                pending_names.clear()
            self._active_request = request
            self._phase = "selecting"
            self._batch_index = self._batch_candidates = 0
            self._candidate_sharded = False
            self._timing.record(
                "request_begin", request=request.name, component=plan.component_id,
                candidates=len(obligation.candidates), ready=obligation.ready,
                query=obligation.configuration.encoded_query.to_dict(),
                bindings=len(obligation.requests),
            )
            self._global_candidate_count = self._candidate_count = len(obligation.candidates)
            self._candidates_prepared = self._completed_rounds = self._active_count = 0
            self._total_rounds = 0
            self._latest_round_us = ()
            if obligation.ready:
                selection = obligation.selection
                programs = plan.prepared.programs
                obligation.programs[plan.contract.config_payload(selection.config)] = programs
            else:
                if self._warmup_only:
                    configuration = obligation.configuration
                    source = "override" if configuration.pinned is not None else "default"
                    obligation.selection = self._selection(obligation, configuration.default, source)
                if obligation.selection is None:
                    self._lookup(obligation, selections)
                if obligation.selection is not None and obligation.planned_candidates:
                    self._total_candidates -= obligation.planned_candidates
                    obligation.planned_candidates = 0
                if obligation.selection is None:
                    if self.session.cache_only:
                        raise LookupError(f"no completed selection for {request.name}")
                    if not self.session._stop.is_set() and self.autotune:
                        if request.collective is not None:
                            raise ValueError("collective declarations must be fixed or explicitly pinned")
                        result = yield from self._race(obligation)
                        if isinstance(result, TuningRequirement):
                            pending.append((obligation, result))
                            pending_names.update(item.name for item in obligation.requests)
                            self._timing.record("local_race_end", request=request.name)
                            continue
                        obligation.selection = result
                    if obligation.selection is None:
                        obligation.selection = self._selection(obligation, obligation.configuration.default, "default")
            yield from self._install_obligation(obligation, selections)
        yield from self._consolidate(pending, selections)
        for name, (request, children) in composites.items():
            plan = request.plan
            child_plans = {count: child.plan for count, child in children.items()}
            if plan.prepared is not None and all(
                child.prepared is not None for child in child_plans.values()
            ) and plan.prepared.variants is not None and all(
                plan.prepared.variants[count] is child for count, child in child_plans.items()
            ) and not any(child in self._new_plans for child in child_plans.values()):
                self._plans_by_name[name] = plan
                continue
            states = {count: child.prepared.state for count, child in child_plans.items()}
            programs = frozenset(
                program for child in child_plans.values() for program in child.prepared.programs
            )
            state = plan._assemble(states, self.session.device)
            self._publish(request, None, state, None, None, programs, variants=plan.variants)
            yield "metadata"
        self._phase = "finishing"
        summary = yield from self.session._drain()
        if summary is not None:
            self._compilations = summary.cute_compilations + summary.triton_compilations
        from b12x._lib.program_cache import evict_unretained
        keep = frozenset(
            program for plan in self.session._plans if plan.prepared is not None
            for program in plan.prepared.programs
        )
        self.session._synchronize()
        self._program_cache.clear()
        evict_unretained(keep)
        result = PreparationResult(
            plans=self._plans_by_name, coverage=self._coverage,
            benchmark_calls=self._benchmark_calls, benchmark_closers=self._benchmark_closers,
            cache_hits=self._cache_hits, benchmarked_candidates=self._benchmarked,
            elapsed_seconds=time.monotonic() - self._started, compilation=summary,
            overlapped_benchmark_samples=self._overlaps,
            program_counts={dialect: sum(program.dialect == dialect for program in self._all_programs)
                            for dialect in ("cute", "triton")},
        )
        self._benchmark_closers = []
        closers, self._temporaries = self._temporaries, []
        _close_all(reversed(closers))
        del closers
        if self._benchmarked:
            # Losing candidates were launched: their executables are now
            # unreferenced and their local memory growth is undone.
            from b12x._lib.program_cache import reclaim_device_memory
            self.session._synchronize()
            with self.session._gpu_scope():
                reclaim_device_memory(keep, stack_limit=self._stack_limit)
        return result

    def _expand(self):
        return _expand_requests(self.requests)


_LAZY_SESSIONS: dict[object, PreparationSession] = {}


def prepare_default(request):
    """Prepare one request with its default configuration, outside any driver.

    Allowed while no session is frozen and no stream is capturing. The
    request's own prepare call primes the state, so families whose state
    depends on the caller's storage (a KV cache, a pool) materialize it for
    that storage. Returns the plan's prepared payload.
    """
    from b12x._lib.runtime_control import kernel_resolution_frozen

    plan = request.plan
    name = plan.component_id
    if kernel_resolution_frozen():
        raise RuntimeError(f"{name} plan is not prepared and kernel resolution is frozen")
    try:
        import torch
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        capturing = False
    if capturing:
        raise RuntimeError(
            f"{name} plan {request.name!r} is not prepared; a shape used under CUDA "
            "graph capture must be declared and prepared before the capture"
        )
    first = plan if isinstance(plan, Plan) else next(iter(plan.variants.values()))
    device = first._device if first._device is not None else getattr(first.query, "device", None)
    detected = detect_device(device)
    session = _LAZY_SESSIONS.get(detected.ordinal)
    if session is None or session.state == "CLOSED" or session._job is not None:
        # Compiles in this process: a first-use default is a JIT kernel, not a
        # startup pass with compiler workers.
        session = PreparationSession(device=detected, autotune=False, compile_workers=0)
        _LAZY_SESSIONS[detected.ordinal] = session
    session.prepare((request,), autotune=False)
    return plan._prepared


@lru_cache(maxsize=256)
def _warn_unprepared_declaration(key, component_id, query_type, dimensions):
    shape = f" ({dimensions})" if dimensions else ""
    logger.warning(
        "%s: %s%s was not prepared before its first use; preparing its default "
        "configuration. Repeated matching declarations are logged at DEBUG.",
        component_id, query_type, shape,
    )


def _prepare_default(plan):
    """Materialize an unprepared plan with its default configuration, without priming.

    This is the path a plan takes when it is bound or run without having been
    prepared: the heuristic configuration, its scratch, and kernels that
    compile on first use. Warnings identify distinct declarations; layer-local
    plan handles and complete queries remain available at DEBUG.
    """
    if isinstance(plan, _CompositePlan):
        query_type, query = type(plan).__name__, plan.capacity_metadata
    else:
        query_type = type(plan.query).__name__
        query = plan.contract.encode_query(plan.query)
    fields = [
        (name, value) for name, value in query.items()
        if type(value) in (int, float, str)
        or isinstance(value, (tuple, list)) and len(value) <= 8
        and all(type(item) is int for item in value)
    ]
    fields.sort(key=lambda item: not item[0].startswith(("max_", "planned_")))
    dimensions = ", ".join(f"{name}={value}" for name, value in fields[:8])
    if len(fields) > 8:
        dimensions += ", ..."
    _warn_unprepared_declaration(
        _declaration_key(plan), plan.component_id, query_type, dimensions,
    )
    logger.debug(
        "%s: unprepared plan %s#%s: %r",
        plan.component_id, query_type, plan.handle, query,
    )

    def noop(state):
        return PreparedCall(run=lambda: None)

    if isinstance(plan, _CompositePlan):
        request = plan.request(
            name=f"default:{plan.handle}", prepare_calls={count: noop for count in plan.token_counts},
        )
    else:
        request = plan.request(name=f"default:{plan.handle}", prepare_call=noop)
    return prepare_default(request)


_set_lazy_preparer(_prepare_default)
