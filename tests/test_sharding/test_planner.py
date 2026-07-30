from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.test_sharding import planner
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
    solo: bool = False,
) -> CollectedNode:
    return CollectedNode.from_nodeid(
        nodeid,
        order=order,
        shard_group=shard_group,
        solo=solo,
    )


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


def test_solo_source_is_one_batch_and_one_logical_unit() -> None:
    nodes = [
        _node("tests/test_solo.py::test_case[0]", 0, solo=True),
        _node("tests/test_solo.py::test_case[1]", 1, solo=True),
        _node("tests/test_regular.py::test_case", 2),
    ]
    plan = build_plan(
        nodes,
        EstimateBook(
            [DurationEstimate("profile", node.nodeid, 4.0, 1) for node in nodes]
        ),
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=5,
            target_unit_seconds=20,
            shard_count=1,
        ),
    )

    solo_units = [
        unit
        for unit in plan.units
        if any(batch.source_file == "tests/test_solo.py" for batch in unit.batches)
    ]
    assert len(solo_units) == 1
    assert len(solo_units[0].batches) == 1
    assert solo_units[0].batches[0].nodeids == (
        "tests/test_solo.py::test_case[0]",
        "tests/test_solo.py::test_case[1]",
    )
    assert Plan.from_dict(plan.to_dict()) == plan


def test_capacity_metrics_account_for_solo_exclusivity() -> None:
    nodes = [
        _node("tests/test_solo.py::test_case", 0, solo=True),
        _node("tests/test_regular_a.py::test_case", 1),
        _node("tests/test_regular_b.py::test_case", 2),
    ]
    plan = build_plan(
        nodes,
        EstimateBook(
            [
                DurationEstimate("profile", nodes[0].nodeid, 10.0, 1),
                DurationEstimate("profile", nodes[1].nodeid, 8.0, 1),
                DurationEstimate("profile", nodes[2].nodeid, 8.0, 1),
            ]
        ),
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=5,
            target_unit_seconds=5,
            shard_count=1,
        ),
    )

    metrics = capacity_metrics(plan, {0: 2}, deadline_seconds=18)

    assert metrics["total_estimated_load_ms"] == 26000
    assert metrics["estimated_worker_load_ms"] == {"0": [18000, 18000]}
    assert metrics["estimated_makespan_ms"] == 18000
    assert metrics["required_workers_by_shard"] == {"0": 2}
    assert capacity_metrics(plan, deadline_seconds=17)["required_workers_by_shard"] == {
        "0": None
    }


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
    assert suite_history.seconds == 5.0
    assert suite_history.source == "suite-mean-current-profile"
    assert no_history.seconds == 5.0
    assert no_history.source == "unknown-floor"


def test_sparse_function_and_source_history_uses_profile_mean() -> None:
    durations = [
        DurationEstimate("current", "tests/test_x.py::test_case[a]", 1.0, 1),
        DurationEstimate("current", "tests/test_x.py::test_case[b]", 9.0, 1),
        *[
            DurationEstimate(
                "current",
                f"tests/other/test_case_{index}.py::test_case",
                1.0,
                1,
            )
            for index in range(8)
        ],
    ]
    book = EstimateBook(durations)
    collected = [
        "tests/test_x.py::test_case[a]",
        "tests/test_x.py::test_case[b]",
        *[f"tests/test_x.py::test_case[{index}]" for index in range(8)],
        *[f"tests/other/test_case_{index}.py::test_case" for index in range(10)],
    ]

    lookup = book.lookup(
        "tests/test_x.py::test_case[missing]",
        "current",
        unknown_floor_seconds=1,
        coverage=book.coverage(collected),
    )

    assert lookup == DurationLookup(1.8, "suite-mean-current-profile")


def test_well_covered_function_history_keeps_conservative_p90() -> None:
    book = EstimateBook(
        [
            DurationEstimate(
                "current",
                f"tests/test_x.py::test_case[{index}]",
                float(index + 1),
                1,
            )
            for index in range(9)
        ]
    )

    lookup = book.lookup(
        "tests/test_x.py::test_case[missing]",
        "current",
        unknown_floor_seconds=1,
        coverage=book.coverage(
            [
                *[f"tests/test_x.py::test_case[{index}]" for index in range(9)],
                "tests/test_x.py::test_case[missing]",
            ]
        ),
    )

    assert lookup == DurationLookup(9.0, "function-current-profile")


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
        "tests/test_x.py::test_new": "suite-mean-current-profile"
    }
    one_worker = capacity_metrics(plan, {0: 1}, deadline_seconds=5)
    two_workers = capacity_metrics(plan, {0: 2}, deadline_seconds=5)
    assert one_worker["total_estimated_load_ms"] == 8000
    assert two_workers["estimated_makespan_ms"] < one_worker["estimated_makespan_ms"]
    assert one_worker["required_workers_by_shard"] == {"0": 2}


def test_high_overhead_source_expands_checkpoint_to_avoid_process_churn() -> None:
    nodes = [
        _node(f"tests/test_expensive.py::test_case[{index}]", index)
        for index in range(100)
    ]
    plan = build_plan(
        nodes,
        EstimateBook(
            [DurationEstimate("profile", node.nodeid, 1.0, 1) for node in nodes],
            [
                OverheadEstimate(
                    "profile",
                    "tests/test_expensive.py",
                    process_startup_seconds=5.0,
                    source_warmup_seconds=5.0,
                    sample_count=1,
                )
            ],
        ),
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=20,
            target_unit_seconds=1000,
            shard_count=1,
        ),
    )

    batches = [batch for unit in plan.units for batch in unit.batches]
    assert len(batches) == 1
    assert batches[0].estimated_ms == 110_000


def test_source_process_count_is_capped_when_overhead_is_unknown() -> None:
    nodes = [
        _node(f"tests/test_many.py::test_case[{index}]", index) for index in range(100)
    ]
    plan = build_plan(
        nodes,
        EstimateBook(
            [DurationEstimate("profile", node.nodeid, 1.0, 1) for node in nodes]
        ),
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=1,
            target_unit_seconds=100,
            shard_count=1,
        ),
    )

    batches = [batch for unit in plan.units for batch in unit.batches]
    assert len(batches) == planner._MAX_BATCHES_PER_TARGET_UNIT
    assert all(len(batch.nodeids) == 25 for batch in batches)


def test_source_affinity_avoids_split_without_material_makespan_gain() -> None:
    durations = {
        "tests/test_c.py": (4.0, 4.0),
        "tests/test_b.py": (2.0, 2.0),
        "tests/test_a.py": (1.0, 1.0),
    }
    nodes = []
    estimates = []
    for source, seconds_values in durations.items():
        for seconds in seconds_values:
            node = _node(
                f"{source}::test_case[{len(nodes)}]",
                len(nodes),
            )
            nodes.append(node)
            estimates.append(DurationEstimate("profile", node.nodeid, seconds, 1))
    plan = build_plan(
        nodes,
        EstimateBook(estimates),
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=1,
            target_unit_seconds=4,
            shard_count=2,
        ),
    )

    c_shards = {
        unit.shard_index
        for unit in plan.units
        if unit.batches[0].source_file == "tests/test_c.py"
    }
    assert len(c_shards) == 1
    assert max(capacity_metrics(plan)["estimated_shard_load_ms"].values()) == 8000


def test_source_affinity_keeps_balanced_sources_on_one_external_shard() -> None:
    nodes = [
        *[_node(f"tests/test_a.py::test_case[{index}]", index) for index in range(4)],
        _node("tests/test_b.py::test_case", 4),
        _node("tests/test_c.py::test_case", 5),
    ]
    estimates = EstimateBook(
        [
            DurationEstimate(
                "profile",
                node.nodeid,
                6.0 if node.source_file == "tests/test_a.py" else 12.0,
                1,
            )
            for node in nodes
        ]
    )

    plan = build_plan(
        nodes,
        estimates,
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=5,
            target_unit_seconds=6,
            shard_count=2,
        ),
    )

    shards_by_source: dict[str, set[int]] = {}
    for unit in plan.units:
        sources = {batch.source_file for batch in unit.batches}
        assert len(sources) == 1
        shards_by_source.setdefault(sources.pop(), set()).add(unit.shard_index)
    assert shards_by_source["tests/test_a.py"] in ({0}, {1})
    shard_loads = capacity_metrics(plan)["estimated_shard_load_ms"]
    assert sorted(shard_loads.values()) == [24_000, 24_000]


def test_source_affinity_splits_source_that_exceeds_one_shard_share() -> None:
    nodes = [
        *[_node(f"tests/test_a.py::test_case[{index}]", index) for index in range(6)],
        _node("tests/test_b.py::test_case", 6),
        _node("tests/test_c.py::test_case", 7),
    ]
    plan = build_plan(
        nodes,
        EstimateBook(
            [
                DurationEstimate(
                    "profile",
                    node.nodeid,
                    6.0 if node.source_file == "tests/test_a.py" else 12.0,
                    1,
                )
                for node in nodes
            ]
        ),
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=5,
            target_unit_seconds=6,
            shard_count=2,
        ),
    )

    a_shards = {
        unit.shard_index
        for unit in plan.units
        if unit.batches[0].source_file == "tests/test_a.py"
    }
    assert a_shards == {0, 1}


def test_soft_target_search_does_not_retry_every_intermediate_bin_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = planner._lpt_bins

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, "_lpt_bins", counted)

    bins = planner._pack_with_soft_target(
        list(range(64)),
        100,
        lambda _item: 51,
        str,
    )

    assert len(bins) == 64
    assert calls <= 8


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
    assert serialized["schema_version"] == 3
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
