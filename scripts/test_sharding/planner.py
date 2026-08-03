from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence, TypedDict, TypeVar

from .estimates import EstimateBook
from .models import ALGORITHM_VERSION, Batch, CollectedNode, Plan, PlanningOptions, Unit


class CapacityMetrics(TypedDict):
    configured_workers_by_shard: dict[str, int]
    estimated_shard_load_ms: dict[str, int]
    estimated_worker_load_ms: dict[str, list[int]]
    estimated_makespan_ms: int
    total_estimated_load_ms: int
    total_estimated_overhead_ms: int
    estimated_overhead_by_source_ms: dict[str, int]
    deadline_seconds: int
    required_workers_by_shard: dict[str, int | None]
    required_total_worker_slots: int | None


@dataclass(frozen=True)
class _AtomicItem:
    id: str
    nodes: tuple[CollectedNode, ...]
    estimated_ms: int


@dataclass(frozen=True)
class _UnitGroup:
    id: str
    units: tuple[Unit, ...]
    estimated_ms: int


_MIN_WORK_TO_OVERHEAD_RATIO = 15
_MAX_BATCHES_PER_TARGET_UNIT = 4  # to reduce pytest overhead
_MIN_AFFINITY_IMPROVEMENT_PERCENT = 5


def _stable_id(kind: str, nodeids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(kind.encode())
    digest.update(b"\0")
    digest.update(ALGORITHM_VERSION.encode())
    for nodeid in sorted(nodeids, key=lambda value: value.encode("utf-8")):
        digest.update(b"\0")
        digest.update(nodeid.encode("utf-8"))
    return f"{kind}-{digest.hexdigest()[:16]}"


T = TypeVar("T")


def _lpt_bins(
    items: Sequence[T],
    count: int,
    duration: Callable[[T], int],
    stable_id: Callable[[T], str],
) -> list[list[T]]:
    bins: list[list[T]] = [[] for _ in range(count)]
    loads = [(0, index) for index in range(count)]
    heapq.heapify(loads)
    ordered = sorted(
        items,
        key=lambda item: (-duration(item), stable_id(item).encode("utf-8")),
    )
    for item in ordered:
        load, index = heapq.heappop(loads)
        bins[index].append(item)
        heapq.heappush(loads, (load + duration(item), index))
    return bins


def _pack_with_soft_target(
    items: Sequence[T],
    target_ms: int,
    duration: Callable[[T], int],
    stable_id: Callable[[T], str],
) -> list[list[T]]:
    if not items:
        return []
    oversized = [item for item in items if duration(item) > target_ms]
    regular = [item for item in items if duration(item) <= target_ms]
    result = [
        [item] for item in sorted(oversized, key=lambda item: stable_id(item).encode())
    ]
    if not regular:
        return result
    count = max(1, math.ceil(sum(duration(item) for item in regular) / target_ms))
    attempts: dict[int, list[list[T]]] = {}

    def try_count(candidate: int) -> list[list[T]]:
        if candidate not in attempts:
            attempts[candidate] = _lpt_bins(regular, candidate, duration, stable_id)
        return attempts[candidate]

    def fits(candidate: int) -> bool:
        return all(
            sum(duration(item) for item in group) <= target_ms
            for group in try_count(candidate)
        )

    if fits(count):
        return result + try_count(count)

    lower = count
    upper = min(len(regular), max(lower + 1, lower * 2))
    while upper < len(regular) and not fits(upper):
        lower = upper
        upper = min(len(regular), upper * 2)
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if fits(middle):
            upper = middle
        else:
            lower = middle
    return result + try_count(upper)


def _unit_source(unit: Unit) -> str:
    sources = {batch.source_file for batch in unit.batches}
    if len(sources) != 1:
        raise ValueError(f"unit {unit.id} crosses source files")
    return sources.pop()


def source_affine_unit_bins(
    units: Sequence[Unit],
    count: int,
) -> list[list[Unit]]:
    """Balance units, splitting sources only for a material makespan improvement."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not units:
        return [[] for _ in range(count)]
    by_source: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        by_source[_unit_source(unit)].append(unit)
    whole_groups = [
        _UnitGroup(
            id=source,
            units=tuple(
                sorted(by_source[source], key=lambda unit: unit.id.encode("utf-8"))
            ),
            estimated_ms=sum(unit.estimated_ms for unit in by_source[source]),
        )
        for source in sorted(by_source, key=lambda value: value.encode("utf-8"))
    ]
    whole_bins = _lpt_bins(
        whole_groups,
        count,
        lambda group: group.estimated_ms,
        lambda group: group.id,
    )
    fair_share = max(
        1,
        math.ceil(sum(unit.estimated_ms for unit in units) / count),
    )
    split_groups: list[_UnitGroup] = []
    for source in sorted(by_source, key=lambda value: value.encode("utf-8")):
        source_units = by_source[source]
        source_load = sum(unit.estimated_ms for unit in source_units)
        chunk_count = min(count, max(1, math.ceil(source_load / fair_share)))
        chunks = _lpt_bins(
            source_units,
            chunk_count,
            lambda unit: unit.estimated_ms,
            lambda unit: unit.id,
        )
        for index, chunk in enumerate(chunks):
            if not chunk:
                continue
            ordered = tuple(sorted(chunk, key=lambda unit: unit.id.encode("utf-8")))
            split_groups.append(
                _UnitGroup(
                    id=f"{source}\0{index:04d}",
                    units=ordered,
                    estimated_ms=sum(unit.estimated_ms for unit in ordered),
                )
            )
    split_bins = _lpt_bins(
        split_groups,
        count,
        lambda group: group.estimated_ms,
        lambda group: group.id,
    )
    whole_makespan = max(
        (sum(group.estimated_ms for group in group_bin) for group_bin in whole_bins),
        default=0,
    )
    split_makespan = max(
        (sum(group.estimated_ms for group in group_bin) for group_bin in split_bins),
        default=0,
    )
    materially_better = split_makespan * 100 <= whole_makespan * (
        100 - _MIN_AFFINITY_IMPROVEMENT_PERCENT
    )
    group_bins = split_bins if materially_better else whole_bins
    return [
        [unit for group in group_bin for unit in group.units]
        for group_bin in group_bins
    ]


def _atomic_items(
    nodes: Sequence[CollectedNode], estimates: dict[str, int]
) -> list[_AtomicItem]:
    groups: dict[str, list[CollectedNode]] = defaultdict(list)
    for node in nodes:
        group_key = (
            f"group:{node.source_file}:{node.shard_group}"
            if node.shard_group is not None
            else f"node:{node.nodeid}"
        )
        groups[group_key].append(node)
    items = []
    for key, members in groups.items():
        ordered = tuple(sorted(members, key=lambda item: item.order))
        items.append(
            _AtomicItem(
                id=key,
                nodes=ordered,
                estimated_ms=sum(estimates[node.nodeid] for node in ordered),
            )
        )
    return items


def build_plan(
    collected_nodes: Sequence[CollectedNode],
    estimate_book: EstimateBook,
    options: PlanningOptions,
) -> Plan:
    nodes = tuple(sorted(collected_nodes, key=lambda node: node.order))
    if len({node.nodeid for node in nodes}) != len(nodes):
        raise ValueError("collection contains duplicate node IDs")
    coverage = estimate_book.coverage(node.nodeid for node in nodes)
    duration_ms: dict[str, int] = {}
    fallback_counts: Counter[str] = Counter()
    fallback_sources: dict[str, str] = {}
    for node in nodes:
        lookup = estimate_book.lookup(
            node.nodeid,
            options.profile,
            options.unknown_case_seconds,
            coverage=coverage,
        )
        duration_ms[node.nodeid] = max(1, round(lookup.seconds * 1000))
        fallback_counts[lookup.source] += 1
        if lookup.source != "exact-current-profile":
            fallback_sources[node.nodeid] = lookup.source

    source_nodes: dict[str, list[CollectedNode]] = defaultdict(list)
    for node in nodes:
        source_nodes[node.source_file].append(node)

    batches: list[Batch] = []
    solo_sources = {node.source_file for node in nodes if node.solo}
    for source_file in sorted(source_nodes, key=lambda value: value.encode("utf-8")):
        overhead_ms = estimate_book.overhead_ms(source_file, options.profile)
        baseline_capacity = max(1, options.checkpoint_seconds * 1000 - overhead_ms)
        unit_capacity = max(1, options.target_unit_seconds * 1000 - overhead_ms)
        overhead_aware_capacity = min(
            unit_capacity,
            _MIN_WORK_TO_OVERHEAD_RATIO * overhead_ms,
        )
        item_capacity = max(baseline_capacity, overhead_aware_capacity)
        if source_file in solo_sources:
            batch_nodes = tuple(
                sorted(source_nodes[source_file], key=lambda node: node.order)
            )
            nodeids = tuple(node.nodeid for node in batch_nodes)
            item_ms = sum(duration_ms[nodeid] for nodeid in nodeids)
            batches.append(
                Batch(
                    id=_stable_id("batch", nodeids),
                    source_file=source_file,
                    nodeids=nodeids,
                    estimated_ms=overhead_ms + item_ms,
                    overhead_ms=overhead_ms,
                    oversized=item_ms > item_capacity,
                )
            )
            continue
        atomic = _atomic_items(source_nodes[source_file], duration_ms)
        item_bins = _pack_with_soft_target(
            atomic,
            item_capacity,
            lambda item: item.estimated_ms,
            lambda item: item.id,
        )
        target_unit_count = max(
            1,
            math.ceil(sum(item.estimated_ms for item in atomic) / unit_capacity),
        )
        process_count_cap = target_unit_count * _MAX_BATCHES_PER_TARGET_UNIT
        if len(item_bins) > process_count_cap:
            item_bins = _lpt_bins(
                atomic,
                min(process_count_cap, len(atomic)),
                lambda item: item.estimated_ms,
                lambda item: item.id,
            )
        for item_bin in item_bins:
            batch_nodes = tuple(
                sorted(
                    (node for item in item_bin for node in item.nodes),
                    key=lambda node: node.order,
                )
            )
            nodeids = tuple(node.nodeid for node in batch_nodes)
            item_ms = sum(duration_ms[nodeid] for nodeid in nodeids)
            batches.append(
                Batch(
                    id=_stable_id("batch", nodeids),
                    source_file=source_file,
                    nodeids=nodeids,
                    estimated_ms=overhead_ms + item_ms,
                    overhead_ms=overhead_ms,
                    oversized=item_ms > item_capacity,
                )
            )

    batches_by_source: dict[str, list[Batch]] = defaultdict(list)
    for batch in batches:
        batches_by_source[batch.source_file].append(batch)
    unit_bins: list[list[Batch]] = []
    for source_file in sorted(
        batches_by_source, key=lambda value: value.encode("utf-8")
    ):
        source_batches = batches_by_source[source_file]
        if source_file in solo_sources:
            unit_bins.extend(
                [batch]
                for batch in sorted(
                    source_batches, key=lambda batch: batch.id.encode("utf-8")
                )
            )
            continue
        unit_bins.extend(
            _pack_with_soft_target(
                source_batches,
                options.target_unit_seconds * 1000,
                lambda batch: batch.estimated_ms,
                lambda batch: batch.id,
            )
        )
    units = []
    for batch_bin in unit_bins:
        ordered_batches = tuple(sorted(batch_bin, key=lambda batch: batch.id.encode()))
        unit_nodeids = [nodeid for batch in ordered_batches for nodeid in batch.nodeids]
        estimate = sum(batch.estimated_ms for batch in ordered_batches)
        units.append(
            Unit(
                id=_stable_id("unit", unit_nodeids),
                batches=ordered_batches,
                estimated_ms=estimate,
                oversized=estimate > options.target_unit_seconds * 1000,
            )
        )

    shard_bins = source_affine_unit_bins(units, options.shard_count)
    shard_for_unit = {
        unit.id: shard_index
        for shard_index, shard_units in enumerate(shard_bins)
        for unit in shard_units
    }
    assigned = tuple(
        replace(unit, shard_index=shard_for_unit[unit.id])
        for unit in sorted(units, key=lambda item: item.id.encode("utf-8"))
    )
    plan = Plan(
        options,
        nodes,
        assigned,
        dict(fallback_counts),
        fallback_sources,
    )
    errors = validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    return plan


def _estimated_worker_loads(
    units: Sequence[Unit],
    workers: int,
    solo_sources: frozenset[str],
) -> list[int]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    solo_units = [unit for unit in units if _unit_is_solo(unit, solo_sources)]
    regular_units = [unit for unit in units if not _unit_is_solo(unit, solo_sources)]
    # Workers start with source-affine queues but steal whole units after
    # draining their own queue, so LPT is the appropriate steady-state model.
    bins = _lpt_bins(
        regular_units,
        workers,
        lambda unit: unit.estimated_ms,
        lambda unit: unit.id,
    )
    exclusive_ms = sum(unit.estimated_ms for unit in solo_units)
    return [
        exclusive_ms + sum(unit.estimated_ms for unit in worker_units)
        for worker_units in bins
    ]


def _unit_is_solo(unit: Unit, solo_sources: frozenset[str]) -> bool:
    return any(batch.source_file in solo_sources for batch in unit.batches)


def capacity_metrics(
    plan: Plan,
    workers_by_shard: Mapping[int, int] | None = None,
    *,
    deadline_seconds: int = 0,
) -> CapacityMetrics:
    """Return deterministic load, makespan, and deadline-capacity estimates."""

    configured = workers_by_shard or {}
    shard_loads: dict[str, int] = {}
    worker_loads: dict[str, list[int]] = {}
    makespans: list[int] = []
    required_by_shard: dict[str, int | None] = {}
    deadline_ms = deadline_seconds * 1000
    solo_sources = frozenset(node.source_file for node in plan.nodes if node.solo)
    for shard_index in range(plan.options.shard_count):
        units = [unit for unit in plan.units if unit.shard_index == shard_index]
        load = sum(unit.estimated_ms for unit in units)
        shard_loads[str(shard_index)] = load
        workers = max(1, int(configured.get(shard_index, 1)))
        loads = _estimated_worker_loads(units, workers, solo_sources)
        worker_loads[str(shard_index)] = loads
        makespans.append(max(loads, default=0))
        if deadline_ms <= 0:
            required_by_shard[str(shard_index)] = None
        elif not units:
            required_by_shard[str(shard_index)] = 0
        else:
            regular_count = sum(not _unit_is_solo(unit, solo_sources) for unit in units)
            required_by_shard[str(shard_index)] = next(
                (
                    candidate
                    for candidate in range(1, max(1, regular_count) + 1)
                    if max(
                        _estimated_worker_loads(units, candidate, solo_sources),
                        default=0,
                    )
                    <= deadline_ms
                ),
                None,
            )
    required_values = list(required_by_shard.values())
    required_total = (
        sum(value for value in required_values if value is not None)
        if deadline_ms > 0 and all(value is not None for value in required_values)
        else None
    )
    overhead_by_source: dict[str, int] = defaultdict(int)
    for unit in plan.units:
        for batch in unit.batches:
            overhead_by_source[batch.source_file] += batch.overhead_ms
    return {
        "configured_workers_by_shard": {
            str(index): max(1, int(configured.get(index, 1)))
            for index in range(plan.options.shard_count)
        },
        "estimated_shard_load_ms": shard_loads,
        "estimated_worker_load_ms": worker_loads,
        "estimated_makespan_ms": max(makespans, default=0),
        "total_estimated_load_ms": sum(shard_loads.values()),
        "total_estimated_overhead_ms": sum(
            batch.overhead_ms for unit in plan.units for batch in unit.batches
        ),
        "estimated_overhead_by_source_ms": dict(sorted(overhead_by_source.items())),
        "deadline_seconds": deadline_seconds,
        "required_workers_by_shard": required_by_shard,
        "required_total_worker_slots": required_total,
    }


def validate_plan(plan: Plan) -> list[str]:
    errors: list[str] = []
    expected = [node.nodeid for node in plan.nodes]
    assigned = [
        nodeid
        for unit in plan.units
        for batch in unit.batches
        for nodeid in batch.nodeids
    ]
    duplicates = sorted(
        nodeid for nodeid, count in Counter(assigned).items() if count > 1
    )
    missing = sorted(set(expected) - set(assigned))
    extra = sorted(set(assigned) - set(expected))
    if duplicates:
        errors.append(f"duplicate nodes: {duplicates[:5]}")
    if missing:
        errors.append(f"missing nodes: {missing[:5]}")
    if extra:
        errors.append(f"unexpected nodes: {extra[:5]}")
    sources_by_nodeid = {node.nodeid: node.source_file for node in plan.nodes}
    for unit in plan.units:
        if not 0 <= unit.shard_index < plan.options.shard_count:
            errors.append(f"unit {unit.id} has invalid shard {unit.shard_index}")
        unit_sources = {batch.source_file for batch in unit.batches}
        if len(unit_sources) > 1:
            errors.append(f"unit {unit.id} crosses source files")
        for batch in unit.batches:
            sources = {
                sources_by_nodeid.get(nodeid, batch.source_file)
                for nodeid in batch.nodeids
                if nodeid in sources_by_nodeid
            } or {batch.source_file}
            if sources != {batch.source_file}:
                errors.append(f"batch {batch.id} crosses source files")
    return errors
