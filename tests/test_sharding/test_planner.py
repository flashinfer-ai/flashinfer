from __future__ import annotations

import json
from pathlib import Path

from scripts.test_sharding.estimates import (
    DurationEstimate,
    DurationLookup,
    EstimateBook,
    OverheadEstimate,
)
from scripts.test_sharding.models import CollectedNode, Plan, PlanningOptions
from scripts.test_sharding.planner import build_plan, capacity_metrics, validate_plan


def _node(
    nodeid: str,
    order: int,
    *,
    shard_group: str | None = None,
) -> CollectedNode:
    return CollectedNode.from_nodeid(nodeid, order=order, shard_group=shard_group)


def test_plan_is_reproducible_and_covers_each_node_once(tmp_path: Path) -> None:
    nodes = [
        _node("tests/a/test_alpha.py::test_a[0]", 0),
        _node("tests/a/test_alpha.py::test_a[1]", 1),
        _node("tests/a/test_alpha.py::test_b", 2),
        _node("tests/b/test_beta.py::TestBeta::test_c", 3),
    ]
    estimates = EstimateBook(
        [
            DurationEstimate("b100-cu13", node.nodeid, seconds, 1)
            for node, seconds in zip(nodes, [4.0, 3.0, 2.0, 1.0], strict=True)
        ]
    )
    options = PlanningOptions(
        profile="b100-cu13",
        checkpoint_seconds=5,
        target_unit_seconds=7,
        unknown_case_seconds=5,
        shard_count=2,
    )

    first = build_plan(nodes, estimates, options)
    second = build_plan(list(reversed(nodes)), estimates, options)

    assert first.to_dict() == second.to_dict()
    assert validate_plan(first) == []
    assert sorted(
        nodeid
        for unit in first.units
        for batch in unit.batches
        for nodeid in batch.nodeids
    ) == sorted(node.nodeid for node in nodes)
    assert json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":")
    ) == json.dumps(second.to_dict(), sort_keys=True, separators=(",", ":"))


def test_shard_group_is_atomic_even_when_it_exceeds_checkpoint() -> None:
    nodes = [
        _node("tests/test_grouped.py::test_grouped[0]", 0, shard_group="compile"),
        _node("tests/test_grouped.py::test_grouped[1]", 1, shard_group="compile"),
        _node("tests/test_grouped.py::test_other", 2),
    ]
    estimates = EstimateBook(
        [DurationEstimate("profile", node.nodeid, 4.0, 1) for node in nodes]
    )
    plan = build_plan(
        nodes,
        estimates,
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=5,
            target_unit_seconds=10,
            unknown_case_seconds=5,
            shard_count=1,
        ),
    )

    grouped_batches = [
        batch
        for unit in plan.units
        for batch in unit.batches
        if "tests/test_grouped.py::test_grouped[0]" in batch.nodeids
    ]
    assert len(grouped_batches) == 1
    assert grouped_batches[0].nodeids == (
        "tests/test_grouped.py::test_grouped[0]",
        "tests/test_grouped.py::test_grouped[1]",
    )
    assert grouped_batches[0].oversized is True


def test_unknown_duration_uses_documented_fallback_order_and_floor() -> None:
    book = EstimateBook(
        [
            DurationEstimate("current", "tests/test_x.py::test_known[a]", 1.0, 2),
            DurationEstimate("current", "tests/test_x.py::test_known[b]", 9.0, 2),
            DurationEstimate("other", "tests/test_x.py::test_new", 7.0, 3),
        ]
    )

    exact_other = book.lookup(
        "tests/test_x.py::test_new", "current", unknown_floor_seconds=5
    )
    same_function = book.lookup(
        "tests/test_x.py::test_known[c]", "current", unknown_floor_seconds=5
    )
    suite_history = book.lookup(
        "tests/new/test_none.py::test_none", "current", unknown_floor_seconds=5
    )
    no_history = EstimateBook().lookup(
        "tests/new/test_none.py::test_none", "current", unknown_floor_seconds=5
    )

    assert exact_other.seconds == 7.0
    assert exact_other.source == "exact-other-profile"
    assert same_function.seconds == 9.0
    assert same_function.source == "function-current-profile"
    assert suite_history.seconds == 9.0
    assert suite_history.source == "suite-current-profile"
    assert no_history.seconds == 5.0
    assert no_history.source == "unknown-floor"


def test_unknown_profile_prefers_b300_duration_fallbacks() -> None:
    known_node = "tests/test_x.py::test_case[known]"
    book = EstimateBook(
        [
            DurationEstimate("sm103-cuda12", known_node, 4.0, 1),
            DurationEstimate("sm103-cuda13", known_node, 6.0, 1),
            DurationEstimate("unrelated-profile", known_node, 100.0, 1),
        ]
    )

    exact = book.lookup(known_node, "new-profile", unknown_floor_seconds=5)
    same_function = book.lookup(
        "tests/test_x.py::test_case[new]",
        "new-profile",
        unknown_floor_seconds=5,
    )

    assert exact == DurationLookup(6.0, "exact-other-profile")
    assert same_function == DurationLookup(6.0, "function-other-profile")


def test_unknown_profile_prefers_b300_overhead_fallbacks() -> None:
    source_file = "tests/test_x.py"
    book = EstimateBook(
        overheads=[
            OverheadEstimate("sm103-cuda12", source_file, 3.0, 2.0, 1),
            OverheadEstimate("sm103-cuda13", source_file, 5.0, 3.0, 1),
            OverheadEstimate("unrelated-profile", source_file, 50.0, 50.0, 1),
        ]
    )

    assert book.overhead_ms(source_file, "new-profile") == 8000


def test_missing_source_overhead_uses_overall_median() -> None:
    book = EstimateBook(
        overheads=[
            OverheadEstimate("current", "tests/test_a.py", 1.0, 2.0, 1),
            OverheadEstimate("current", "tests/test_b.py", 2.0, 5.0, 10),
            OverheadEstimate("current", "tests/test_c.py", 8.0, 12.0, 100),
            OverheadEstimate("other", "tests/test_d.py", 50.0, 50.0, 1),
        ]
    )

    assert book.overhead_ms("tests/test_new.py", "current") == 13500


def test_missing_profile_overhead_uses_overall_median() -> None:
    book = EstimateBook(
        overheads=[
            OverheadEstimate("sm103-cuda12", "tests/test_a.py", 1.0, 3.0, 1),
            OverheadEstimate("sm103-cuda13", "tests/test_b.py", 4.0, 6.0, 1),
            OverheadEstimate("unrelated-profile", "tests/test_c.py", 50.0, 50.0, 1),
        ]
    )

    assert book.overhead_ms("tests/test_new.py", "new-profile") == 10000


def test_b300_preference_only_applies_when_target_profile_is_absent() -> None:
    requested_node = "tests/test_x.py::test_case[requested]"
    book = EstimateBook(
        [
            DurationEstimate(
                "target-profile",
                "tests/test_x.py::test_case[existing]",
                1.0,
                1,
            ),
            DurationEstimate("sm103-cuda13", requested_node, 6.0, 1),
            DurationEstimate("unrelated-profile", requested_node, 100.0, 1),
        ]
    )

    lookup = book.lookup(
        requested_node,
        "target-profile",
        unknown_floor_seconds=5,
    )

    assert lookup == DurationLookup(100.0, "exact-other-profile")


def test_unknown_profile_uses_available_history_when_b300_is_absent() -> None:
    requested_node = "tests/test_x.py::test_case[requested]"
    book = EstimateBook([DurationEstimate("available-profile", requested_node, 7.0, 1)])

    lookup = book.lookup(
        requested_node,
        "new-profile",
        unknown_floor_seconds=5,
    )

    assert lookup == DurationLookup(7.0, "exact-other-profile")


def test_plan_records_each_fallback_source_and_worker_aware_capacity() -> None:
    nodes = [
        _node("tests/test_x.py::test_known[a]", 0),
        _node("tests/test_x.py::test_new", 1),
    ]
    book = EstimateBook([DurationEstimate("profile", nodes[0].nodeid, 4.0, 1)])
    plan = build_plan(
        nodes,
        book,
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=5,
            target_unit_seconds=5,
            unknown_case_seconds=2,
            shard_count=1,
        ),
    )

    assert plan.fallback_sources == {
        "tests/test_x.py::test_new": "source-current-profile"
    }
    one_worker = capacity_metrics(plan, {0: 1}, deadline_seconds=5)
    two_workers = capacity_metrics(plan, {0: 2}, deadline_seconds=5)
    assert one_worker["total_estimated_load_ms"] == 8000
    assert two_workers["estimated_makespan_ms"] < one_worker["estimated_makespan_ms"]
    assert one_worker["required_workers_by_shard"] == {"0": 2}


def test_plan_serialization_stores_each_nodeid_once() -> None:
    nodes = [
        _node("tests/test_x.py::test_known[a-very-long-parameter]", 0),
        _node("tests/test_x.py::test_new[another-very-long-parameter]", 1),
    ]
    book = EstimateBook([DurationEstimate("profile", nodes[0].nodeid, 1.0, 1)])
    plan = build_plan(
        nodes,
        book,
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=5,
            target_unit_seconds=5,
            unknown_case_seconds=2,
            shard_count=1,
        ),
    )

    serialized = plan.to_dict()
    encoded = json.dumps(serialized, sort_keys=True, separators=(",", ":"))

    assert Plan.from_dict(serialized) == plan
    assert serialized["schema_version"] == 2
    assert serialized["nodeids"] == [node.nodeid for node in nodes]
    assert "nodes" not in serialized
    assert "batch_ids" not in encoded
    assert all(encoded.count(node.nodeid) == 1 for node in nodes)


def test_plan_reader_accepts_schema_one_manifests() -> None:
    nodes = [_node("tests/test_x.py::test_case", 0)]
    plan = build_plan(
        nodes,
        EstimateBook([DurationEstimate("profile", nodes[0].nodeid, 1.0, 1)]),
        PlanningOptions(profile="profile"),
    )
    legacy = {
        "schema_version": 1,
        "algorithm_version": "lpt-ms-v2",
        "options": plan.options.to_dict(),
        "nodes": [node.to_dict() for node in plan.nodes],
        "units": [unit.to_dict() for unit in plan.units],
        "fallback_counts": plan.fallback_counts,
        "fallback_sources": plan.fallback_sources,
    }

    assert Plan.from_dict(legacy) == plan
