"""Immutable declarations, prepared state, and preparation metadata.

A ``Plan`` is a declaration: contract, query, invocation metadata, and the
component callbacks that compile, size, and materialize one configuration.
A ``PreparationSession`` fills the plan's prepared slot in place. Families
bind and run from the plan; nothing else carries executable state.
"""

from __future__ import annotations

import itertools
import math
import sys
import weakref
from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

if TYPE_CHECKING:
    from b12x._lib.compile_pool import CompileJob
    from b12x._lib.scratch import ScratchBufferSpec
    from .device import DetectedDevice
    from .tuning import TuningContract

ConfigT = TypeVar("ConfigT")
_SCALAR_TYPES = (str, int, float, bool, type(None))

_CURRENT_PLAN: ContextVar[object] = ContextVar("b12x_preparation_plan")


def _owned_tensor_nbytes(tensors):
    """Count only explicitly owned tensor storage, deduplicating aliased views."""
    storages = {}
    for tensor in tensors:
        if tensor is not None:
            storage = tensor.untyped_storage()
            storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def current_plan():
    """The plan whose memory, materialize, or priming callback is running.

    Plans hash by identity, so this value is a valid allocation-owner key for
    ``PersistentMemory``. It is never a serving lookup and never part of a
    selection-cache key.
    """
    try:
        return _CURRENT_PLAN.get()
    except LookupError:
        raise RuntimeError(
            "the current plan is available only inside preparation callbacks"
        ) from None


def current_prepared_state():
    """The current plan's installed state, or ``None`` while it is unprepared."""
    prepared = current_plan()._prepared
    return None if prepared is None or prepared.closed else prepared.state


@contextmanager
def _plan_scope(plan):
    token = _CURRENT_PLAN.set(plan)
    try:
        yield
    finally:
        _CURRENT_PLAN.reset(token)


def _normalized_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _freeze_value(value: object, *, field: str) -> object:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item, field=f"{field}[]") for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"metadata field {field!r} requires an explicit finite codec")
    if isinstance(value, _SCALAR_TYPES):
        return value
    raise TypeError(
        f"metadata field {field!r} must contain JSON-compatible values, "
        f"got {type(value).__name__}"
    )


def _thaw_value(value: object) -> object:
    if isinstance(value, FrozenMapping):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


@dataclass(frozen=True, init=False)
class FrozenMapping(Mapping[str, object]):
    """A recursively immutable, hashable mapping for preparation metadata."""

    _items: tuple[tuple[str, object], ...]

    def __init__(self, values: Mapping[str, object] | None = None) -> None:
        items: list[tuple[str, object]] = []
        for key, value in (values or {}).items():
            if not isinstance(key, str) or not key:
                raise ValueError("metadata mapping keys must be non-empty strings")
            items.append((key, _freeze_value(value, field=key)))
        object.__setattr__(self, "_items", tuple(sorted(items)))

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, object]:
        return {key: _thaw_value(value) for key, value in self._items}


@dataclass(frozen=True)
class DeviceIdentity:
    """Portable hardware identity used by preparation specialization."""

    vendor: str
    compute_capability: tuple[int, int]
    sm_count: int
    product_name: str

    def __post_init__(self) -> None:
        vendor = _normalized_name(self.vendor)
        product_name = _normalized_name(self.product_name)
        if not vendor or not product_name:
            raise ValueError("vendor and product_name must be non-empty")
        capability = tuple(self.compute_capability)
        if len(capability) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in capability
        ):
            raise ValueError("compute_capability must contain two nonnegative integers")
        if not isinstance(self.sm_count, int) or isinstance(self.sm_count, bool):
            raise TypeError("sm_count must be an integer")
        if self.sm_count <= 0:
            raise ValueError("sm_count must be positive")
        object.__setattr__(self, "vendor", vendor)
        object.__setattr__(self, "compute_capability", capability)
        object.__setattr__(self, "product_name", product_name)


@dataclass(frozen=True)
class PersistentMemory:
    key: Hashable
    required_nbytes: int
    resident_nbytes: int = 0

    def __post_init__(self):
        hash(self.key)
        for size in (self.required_nbytes, self.resident_nbytes):
            if type(size) is not int or size < 0:
                raise ValueError("persistent byte counts must be nonnegative integers")


@dataclass(frozen=True)
class MemoryRequirements:
    scratch: tuple[ScratchBufferSpec, ...] = ()
    persistent: tuple[PersistentMemory, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "scratch", tuple(self.scratch))
        object.__setattr__(self, "persistent", tuple(self.persistent))

    @property
    def scratch_nbytes(self):
        return sum((spec.nbytes + 255) // 256 * 256 for spec in self.scratch)

    @property
    def pending_persistent_nbytes(self):
        return sum(
            max(0, entry.required_nbytes - entry.resident_nbytes)
            for entry in self._persistent_by_key().values()
        )

    def _persistent_by_key(self):
        result = {}
        for entry in self.persistent:
            previous = result.get(entry.key)
            if previous is not None:
                if previous.resident_nbytes != entry.resident_nbytes:
                    raise ValueError("conflicting residency for shared allocation")
                if previous.required_nbytes >= entry.required_nbytes:
                    continue
            result[entry.key] = entry
        return result

    @classmethod
    def sequential(cls, requirements):
        """Calls reuse scratch sequentially; persistent allocation keys are shared."""
        scratch, size, persistent = (), (0, 0), []
        for requirement in requirements:
            if not isinstance(requirement, cls):
                raise TypeError("memory callback must return MemoryRequirements")
            candidate_size = (
                requirement.scratch_nbytes,
                sum(spec.nbytes for spec in requirement.scratch),
            )
            if candidate_size > size:
                scratch, size = requirement.scratch, candidate_size
            persistent.extend(requirement.persistent)
        combined = cls(scratch=scratch, persistent=tuple(persistent))
        return cls(
            scratch=scratch, persistent=tuple(combined._persistent_by_key().values())
        )


@dataclass(frozen=True)
class Selection(Generic[ConfigT]):
    component_id: str
    query: FrozenMapping
    config: ConfigT
    source: Literal["override", "cached", "tuned", "default", "fixed"]
    assignment: FrozenMapping | None = None


@dataclass(frozen=True)
class CollectiveRequirement:
    key: str
    ranks: tuple[int, ...]

    def __post_init__(self):
        ranks = tuple(self.ranks)
        if (
            not self.key
            or not ranks
            or any(type(rank) is not int or rank < 0 for rank in ranks)
        ):
            raise ValueError("collectives require a key and nonnegative global ranks")
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("collective ranks must be sorted and unique")
        object.__setattr__(self, "ranks", ranks)


@dataclass(frozen=True)
class TuningCacheRequirement:
    """Rank-local completed selections to reconcile before sharding candidates."""

    ranks: tuple[int, ...]
    identity: FrozenMapping
    records: FrozenMapping

    def __post_init__(self):
        ranks = tuple(self.ranks)
        if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ValueError("tuning cache agreement requires nonnegative ranks")
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("tuning cache ranks must be sorted and unique")
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "identity", FrozenMapping(self.identity))
        object.__setattr__(self, "records", FrozenMapping(self.records))


@dataclass(frozen=True)
class TuningRequirement:
    """One rank's winner from its disjoint share of a tuning race."""

    key: str
    ranks: tuple[int, ...]
    assignment: FrozenMapping | None
    latency_us: float | None
    candidate_index: int | None
    rejected_count: int = 0
    cute_programs: tuple[str, ...] = ()

    def __post_init__(self):
        ranks = tuple(self.ranks)
        if (
            not self.key
            or not ranks
            or any(type(rank) is not int or rank < 0 for rank in ranks)
        ):
            raise ValueError("tuning requirements need a key and nonnegative ranks")
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("tuning ranks must be sorted and unique")
        if type(self.rejected_count) is not int or self.rejected_count < 0:
            raise ValueError("rejected candidate count must be a nonnegative integer")
        empty = self.assignment is None
        programs = tuple(self.cute_programs)
        if (empty and programs) or any(
            not isinstance(key, str)
            or len(key) != 64
            or any(char not in "0123456789abcdef" for char in key)
            for key in programs
        ):
            raise ValueError("winner programs must be CuTe cache keys")
        object.__setattr__(self, "cute_programs", tuple(sorted(set(programs))))
        if empty != (self.latency_us is None) or empty != (
            self.candidate_index is None
        ):
            raise ValueError("a tuning contribution must be either complete or empty")
        if not empty:
            object.__setattr__(self, "assignment", FrozenMapping(self.assignment))
            if (
                type(self.latency_us) not in (int, float)
                or not math.isfinite(self.latency_us)
                or self.latency_us < 0
            ):
                raise ValueError("tuning latency must be finite and nonnegative")
            if type(self.candidate_index) is not int or self.candidate_index < 0:
                raise ValueError("tuning candidate index must be nonnegative")
        object.__setattr__(self, "ranks", ranks)


@dataclass(frozen=True)
class PreparationProgress:
    running: bool
    pending_compilation: bool
    ready_collectives: tuple[CollectiveRequirement, ...]
    done: bool
    phase: str = "planning"
    component_id: str = ""
    request_name: str = ""
    completed_requests: int = 0
    total_requests: int = 0
    candidate_count: int = 0
    candidates_prepared: int = 0
    measured_candidates: int = 0
    completed_rounds: int = 0
    total_rounds: int = 0
    active_count: int = 0
    latest_round_us: tuple[float, ...] = ()
    cache_hits: int = 0
    compilations: int = 0
    active_compilations: int = 0
    elapsed_seconds: float = 0.0
    tuning_stopped: bool = False
    ready_tuning: tuple[TuningRequirement, ...] = ()
    candidate_sharded: bool = False
    batch_index: int = 0
    batch_candidates: int = 0
    tuning_rank: int = 0
    total_candidates: int | None = None
    global_candidate_count: int = 0
    selection_counts: tuple[tuple[str, int], ...] = ()
    ready_cache: TuningCacheRequirement | None = None


@dataclass(kw_only=True)
class PreparedCall:
    """Prepared operation; timed calls enqueue work on the current CUDA stream.

    A benchmark call must return without waiting for device work or using
    unjoined side streams. Reset and activation production run before its gate.
    ``capture_safe`` describes caller capture support, not the timing method.
    """

    run: Callable[[], object]
    output: object = None
    produce: Callable[[], None] | None = None
    reset: Callable[[], None] | None = None
    restore: Callable[[], None] | None = None
    owners: tuple[object, ...] = ()
    close: Callable[[], None] | None = None
    capture_safe: bool = True
    benchmark_producers: tuple[Callable[[], None], ...] = ()

    def invoke(self):
        result = self.run()
        if result is not None:
            self.output = result
        return result


def call_scope():
    """The mode prepared calls execute in.

    Callers' pools and activations are often inference tensors, which may be
    updated in place only inside inference mode. Materialized resources are
    created outside it so they stay mutable everywhere; the calls that write
    them run inside it, as serving does.
    """
    import torch

    return torch.inference_mode()


def _prime(call: PreparedCall):
    with call_scope():
        if call.reset is not None:
            call.reset()
        if call.produce is not None:
            call.produce()
        return call.invoke()


def _close_all(closers):
    primary = None
    for closer in closers:
        if closer is not None:
            try:
                closer()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note(f"additional cleanup failure: {error!r}")
    if primary is not None:
        active = sys.exception()
        if active is not None and not isinstance(active, GeneratorExit):
            active.add_note(f"cleanup failure: {primary!r}")
        else:
            raise primary


@dataclass(eq=False)
class _Prepared:
    """The payload of a plan's prepared slot.

    ``state`` is the component's private materialized state. ``programs`` is
    the closure of compiled programs the state may launch, including those of
    its dependencies. ``retained`` keeps those programs resident for graphs.
    ``users`` lists every plan aliasing this payload; the last user to release
    it runs ``closers``.
    """

    state: object
    selection: Selection | None
    programs: frozenset
    retained: object
    owners: tuple
    closers: tuple
    device: DetectedDevice
    variants: Mapping[int, Plan] | None = None
    scratch: tuple[ScratchBufferSpec, ...] | None = None
    users: list = field(default_factory=list)
    closed: bool = False


_HANDLES = itertools.count(1)
_PLANS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _register_handle(plan):
    handle = next(_HANDLES)
    object.__setattr__(plan, "_handle", handle)
    _PLANS[handle] = plan


def plan_from_handle(handle: int):
    """Resolve a live plan from the integer handle carried by b12x custom ops."""
    plan = _PLANS.get(handle)
    if plan is None:
        raise RuntimeError(f"plan handle {handle} does not name a live plan")
    return plan


class _PreparedSlot:
    """Prepared-slot behavior shared by scalar and composite plans."""

    @property
    def handle(self) -> int:
        """Stable integer identity for custom-op arguments."""
        return self._handle

    @property
    def prepared(self):
        """The installed payload, or ``None`` while unprepared or released."""
        prepared = self._prepared
        return None if prepared is None or prepared.closed else prepared

    @property
    def selection(self):
        prepared = self.prepared
        return None if prepared is None else prepared.selection

    def _install(self, prepared):
        previous = self._prepared
        object.__setattr__(self, "_prepared", prepared)
        try:
            if prepared.scratch is None:
                prepared.scratch = self.memory_requirements().scratch
        except BaseException:
            object.__setattr__(self, "_prepared", previous)
            raise

    def _clear(self):
        object.__setattr__(self, "_prepared", None)

    def scratch_specs(self):
        prepared = self.prepared
        if prepared is not None:
            return prepared.scratch
        return self.memory_requirements().scratch

    def __getstate__(self):
        """Pickle the declaration only.

        Compiler caches serialize guard values, which can include a plan closed
        over by compiled code. The prepared payload holds device programs and
        is never serialized; the copy keeps the handle number but is not
        registered, so it never resolves through ``plan_from_handle``.
        """
        state = dict(self.__dict__)
        state["_prepared"] = None
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            object.__setattr__(self, key, value)


@dataclass(frozen=True, kw_only=True, eq=False)
class Plan(_PreparedSlot, Generic[ConfigT]):
    contract: TuningContract
    query: object
    _compile_jobs: Callable[[ConfigT, DetectedDevice], tuple[CompileJob, ...]]
    _memory_requirements: Callable[[ConfigT, DetectedDevice], MemoryRequirements]
    _materialize: Callable[[Selection[ConfigT], DetectedDevice], object]
    invocation: FrozenMapping = FrozenMapping()
    override: ConfigT | None = None
    dependencies: tuple[str, ...] = ()
    _device: object | None = None
    shared: bool = False
    _prepared: _Prepared | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _handle: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "invocation", FrozenMapping(self.invocation))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if type(self.shared) is not bool:
            raise TypeError("shared must be boolean")
        _register_handle(self)

    @property
    def component_id(self):
        return self.contract.component_id

    def request(
        self,
        *,
        name,
        prepare_call,
        benchmark_call=None,
        dependencies=(),
        collective=None,
        retain_benchmark_call=False,
    ):
        return PreparationRequest(
            name=name,
            plan=self,
            prepare_call=prepare_call,
            benchmark_call=benchmark_call,
            dependencies=tuple(dict.fromkeys((*self.dependencies, *dependencies))),
            collective=collective,
            retain_benchmark_call=retain_benchmark_call,
        )

    def memory_requirements(self):
        """Memory of the prepared configuration, else of the default one."""
        from .device import detect_device

        prepared = self.prepared
        if prepared is not None:
            config, device = prepared.selection.config, prepared.device
        else:
            device = detect_device(
                self._device
                if self._device is not None
                else getattr(self.query, "device", None)
            )
            configuration = self.contract.configure(
                self.query, device=device.identity, override=self.override
            )
            config = (
                configuration.default
                if configuration.pinned is None
                else configuration.pinned
            )
        with _plan_scope(self):
            return self._memory_requirements(config, device)


@dataclass(frozen=True, kw_only=True, eq=False)
class _CompositePlan(_PreparedSlot):
    """An exact-M capacity family: one child plan per planned token count."""

    component_id: str
    capacity_metadata: FrozenMapping
    variants: Mapping[int, Plan]
    _assemble: Callable[[Mapping[int, object], DetectedDevice], object]
    composite_semantic_version: int = 1
    dependencies: tuple[str, ...] = ()
    shared: bool = False
    _prepared: _Prepared | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _handle: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self):
        variants = dict(self.variants)
        counts = tuple(variants)
        if not counts or any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError("composite counts must be positive integers")
        if counts != tuple(sorted(set(counts))):
            raise ValueError("composite counts must be sorted and unique")
        if any(not isinstance(child, Plan) for child in variants.values()):
            raise TypeError("composite variants must be scalar plans")
        if (
            type(self.composite_semantic_version) is not int
            or self.composite_semantic_version <= 0
        ):
            raise ValueError("composite semantic version must be positive")
        if type(self.shared) is not bool:
            raise TypeError("shared must be boolean")
        object.__setattr__(self, "variants", MappingProxyType(variants))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(
            self, "capacity_metadata", FrozenMapping(self.capacity_metadata)
        )
        _register_handle(self)

    @property
    def token_counts(self) -> tuple[int, ...]:
        return tuple(self.variants)

    def request(
        self,
        *,
        name,
        prepare_calls,
        benchmark_calls=None,
        dependencies=(),
        collective=None,
        retain_benchmark_call=False,
    ):
        if set(prepare_calls) != set(self.token_counts):
            raise ValueError(
                "composite preparation calls must cover exact planned counts"
            )
        if benchmark_calls is not None and set(benchmark_calls) != set(
            self.token_counts
        ):
            raise ValueError(
                "composite benchmark calls must cover exact planned counts"
            )
        return PreparationRequest(
            name=name,
            plan=self,
            prepare_call=MappingProxyType(dict(prepare_calls)),
            benchmark_call=None
            if benchmark_calls is None
            else MappingProxyType(dict(benchmark_calls)),
            dependencies=tuple(dict.fromkeys((*self.dependencies, *dependencies))),
            collective=collective,
            retain_benchmark_call=retain_benchmark_call,
        )

    def memory_requirements(self):
        return MemoryRequirements.sequential(
            child.memory_requirements() for child in self.variants.values()
        )


@dataclass(frozen=True, kw_only=True)
class PreparationRequest:
    name: str
    plan: Plan | _CompositePlan
    prepare_call: (
        Callable[[object], PreparedCall]
        | Mapping[int, Callable[[object], PreparedCall]]
    )
    benchmark_call: (
        Callable[[object], PreparedCall]
        | Mapping[int, Callable[[object], PreparedCall]]
        | None
    ) = None
    dependencies: tuple[str, ...] = ()
    collective: CollectiveRequirement | None = None
    retain_benchmark_call: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("requests require a nonempty name")
        if not isinstance(self.plan, (Plan, _CompositePlan)):
            raise TypeError("requests require a declarative Plan")
        if type(self.retain_benchmark_call) is not bool:
            raise TypeError("retain_benchmark_call must be boolean")
        if self.retain_benchmark_call and self.benchmark_call is None:
            raise ValueError("benchmark retention requires a benchmark call factory")
        if isinstance(self.plan, Plan):
            if not callable(self.prepare_call) or (
                self.benchmark_call is not None and not callable(self.benchmark_call)
            ):
                raise TypeError("scalar call factories must be callable")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))


_LAZY_PREPARER: Callable[[object], _Prepared] | None = None


def _set_lazy_preparer(preparer):
    global _LAZY_PREPARER
    _LAZY_PREPARER = preparer


def require_prepared(plan, component_id, device=None):
    """Return the plan's prepared state for ``component_id``.

    An unprepared plan is materialized with its default configuration on
    first use, with a warning; after ``session.freeze()`` or during CUDA graph
    capture that is an error.
    """
    if not isinstance(plan, (Plan, _CompositePlan)):
        raise TypeError(f"{component_id} requires a Plan, got {type(plan).__name__}")
    if plan.component_id != component_id:
        raise ValueError(f"plan belongs to {plan.component_id}, not {component_id}")
    prepared = plan._prepared
    if prepared is None:
        if _LAZY_PREPARER is None:
            raise RuntimeError(f"{component_id} plan is not prepared")
        prepared = _LAZY_PREPARER(plan)
    if prepared.closed:
        raise RuntimeError(f"{component_id} plan resources have been released")
    if device is not None:
        import torch

        actual = torch.device(device)
        if actual.type != "cuda" or actual.index != prepared.device.ordinal:
            raise ValueError("plan and tensor device differ")
    return prepared.state


class PreparationResult:
    """Report of one preparation batch; the prepared state lives on the plans."""

    def __init__(
        self,
        *,
        plans,
        coverage=None,
        benchmark_calls=None,
        benchmark_closers=(),
        cache_hits=0,
        benchmarked_candidates=0,
        elapsed_seconds=0.0,
        compilation=None,
        parent_cute_compilations=0,
        parent_triton_compilations=0,
        overlapped_benchmark_samples=0,
        program_counts=None,
    ):
        self.plans = MappingProxyType(dict(plans))
        self.selections = MappingProxyType(
            {
                name: plan.selection
                for name, plan in self.plans.items()
                if plan.selection is not None
            }
        )
        self.coverage = MappingProxyType(
            {
                name: MappingProxyType(dict(value))
                for name, value in (coverage or {}).items()
            }
        )
        self.benchmark_calls = MappingProxyType(dict(benchmark_calls or {}))
        self.cache_hits = cache_hits
        self.benchmarked_candidates = benchmarked_candidates
        self.elapsed_seconds = elapsed_seconds
        self.compilation = compilation
        self.parent_cute_compilations = parent_cute_compilations
        self.parent_triton_compilations = parent_triton_compilations
        self.overlapped_benchmark_samples = overlapped_benchmark_samples
        self.program_counts = MappingProxyType(dict(program_counts or {}))
        self._benchmark_closers = tuple(benchmark_closers)
        self.closed = False

    def close(self):
        """Release retained benchmark trials; prepared plans are unaffected."""
        if not self.closed:
            self.closed = True
            closers, self._benchmark_closers = self._benchmark_closers, ()
            _close_all(closers)

    def __enter__(self):
        if self.closed:
            raise RuntimeError("preparation result is closed")
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as error:
            if value is None:
                raise
            value.add_note(f"preparation cleanup failed: {error!r}")
