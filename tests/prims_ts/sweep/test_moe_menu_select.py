# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import asdict, replace

import pytest

from flashinfer.prims_ts.moe.tools.sweep.menu_config import AutotuneCost
from flashinfer.prims_ts.moe.tools.sweep.menu_select import (
    DeploymentBuckets,
    JointCandidate,
    JointShape,
    autotune_cost,
    deployment_buckets_from_target,
    global_best_times,
    menu_metrics,
    select_joint_menu,
)
from flashinfer.prims_ts.moe.tools.sweep.portfolio import TimingValue


def _candidate(candidate_id, fc, tile_n):
    return JointCandidate(
        candidate_id=candidate_id,
        fc=fc,
        activation_class="gated" if fc == "fc1" else None,
        tile_n=tile_n,
        options={"tile_n": tile_n},
    )


def _shape(shape_id, deployment):
    return JointShape(
        shape_id=shape_id,
        deployment=deployment,
        model=deployment,
        activation_class="gated",
        tp=1,
        ep=1,
        tokens=8,
        routing="uniform",
        selected_tile_ns=(8, 16),
        weight=1.0,
    )


def _inputs():
    candidates = {
        "a8": _candidate("a8", "fc1", 8),
        "b8": _candidate("b8", "fc2", 8),
        "a16": _candidate("a16", "fc1", 16),
        "a16_twin": _candidate("a16_twin", "fc1", 16),
        "b16": _candidate("b16", "fc2", 16),
    }
    shapes = {
        "s1": _shape("s1", "model-a"),
        "s2": _shape("s2", "model-b"),
    }
    values = {
        "a8": (10.0, 0.1),
        "b8": (10.0, 0.1),
        "a16": (5.0, 0.1),
        "a16_twin": (5.05, 0.1),
        "b16": (5.0, 0.1),
    }
    timings = {
        candidate_id: {
            shape_id: TimingValue.measured(time, spread) for shape_id in shapes
        }
        for candidate_id, (time, spread) in values.items()
    }
    return candidates, shapes, timings


def test_joint_time_adds_fc1_and_fc2_and_floor_opens_default_tile():
    candidates, shapes, timings = _inputs()
    best = global_best_times(candidates, shapes, timings)
    assert best == {"s1": 10.0, "s2": 10.0}

    floor = {"a8", "b8"}
    metrics = menu_metrics(
        floor,
        candidates=candidates,
        shapes=shapes,
        timings=timings,
        global_best=best,
        cost=AutotuneCost(),
    )
    assert metrics.geomean_regret == 2.0
    assert metrics.covered_shapes == 2


def test_greedy_can_open_new_tile_as_pair_and_drops_noisy_twin():
    candidates, shapes, timings = _inputs()
    selection = select_joint_menu(
        candidates,
        shapes,
        timings,
        cost=AutotuneCost(seconds_per_pair=1.0, seconds_per_compile=2.0),
    )

    assert selection.floor_size == 2
    assert selection.selected_ids[:2] == ("a8", "b8")
    assert {"a16", "b16"}.issubset(selection.selected_ids)
    assert "a16_twin" not in selection.selected_ids
    assert selection.order[-1].metrics.geomean_regret == 1.0
    assert set(selection.leave_one_model_out) == {"model-a", "model-b"}


def test_greedy_rejects_improvement_within_measurement_noise():
    candidates, shapes, timings = _inputs()
    candidates["b16_twin"] = _candidate("b16_twin", "fc2", 16)
    timings["b16_twin"] = {
        "s1": TimingValue.measured(6.0, 0.1),
        "s2": TimingValue.measured(4.95, 0.1),
    }
    selection = select_joint_menu(
        candidates,
        shapes,
        timings,
        cost=AutotuneCost(seconds_per_pair=1.0, seconds_per_compile=2.0),
    )

    assert {"a16", "b16"}.issubset(selection.selected_ids)
    assert "b16_twin" not in selection.selected_ids


def test_lomo_groups_multiple_deployments_of_the_same_model():
    candidates, shapes, timings = _inputs()
    shapes = {
        shape_id: replace(shape, model="shared-model")
        for shape_id, shape in shapes.items()
    }

    selection = select_joint_menu(candidates, shapes, timings)

    assert selection.leave_one_model_out == {}


def test_floor_rejects_candidates_without_joint_measurements():
    candidates, shapes, timings = _inputs()
    timings["a8"] = {shape_id: TimingValue.not_applicable() for shape_id in shapes}

    with pytest.raises(ValueError, match="no measured floor pair"):
        select_joint_menu(candidates, shapes, timings)


def test_floor_skips_group_without_any_measurement():
    candidates, shapes, timings = _inputs()
    shapes["dark"] = replace(_shape("dark", "model-c"), activation_class="elementwise")

    selection = select_joint_menu(candidates, shapes, timings)

    assert selection.floor_size == 2
    final = selection.order[-1].metrics
    assert (final.covered_shapes, final.total_shapes) == (2, 2)
    assert set(selection.leave_one_model_out) == {"model-a", "model-b", "model-c"}


def test_extra_tile_floor_keeps_pair_that_never_wins():
    candidates, shapes, timings = _inputs()
    shapes = {
        shape_id: replace(shape, selected_tile_ns=(8, 16, 32))
        for shape_id, shape in shapes.items()
    }
    candidates["a32"] = _candidate("a32", "fc1", 32)
    candidates["b32"] = _candidate("b32", "fc2", 32)
    timings["a32"] = {shape_id: TimingValue.measured(20.0, 0.1) for shape_id in shapes}
    timings["b32"] = {shape_id: TimingValue.measured(20.0, 0.1) for shape_id in shapes}

    selection = select_joint_menu(
        candidates,
        shapes,
        timings,
        floor_tile_ns=(32,),
    )

    assert selection.floor_size == 4
    assert selection.selected_ids[:4] == ("a8", "b8", "a32", "b32")
    assert selection.order[2].reason == "floor:gated:32"
    assert selection.order[3].reason == "floor:gated:32"
    assert selection.floor_tile_ns == (32,)


def test_extra_tile_floor_scores_only_plausible_shapes():
    candidates = {
        "a8": _candidate("a8", "fc1", 8),
        "b8": _candidate("b8", "fc2", 8),
        "a32": _candidate("a32", "fc1", 32),
        "b32_near": _candidate("b32_near", "fc2", 32),
        "b32_far": _candidate("b32_far", "fc2", 32),
    }
    shapes = {
        "near": replace(_shape("near", "model-a"), selected_tile_ns=(8, 32)),
        "far": replace(_shape("far", "model-a"), selected_tile_ns=(8, 32)),
    }
    values = {
        "a8": {"near": 10.0, "far": 10.0},
        "b8": {"near": 10.0, "far": 10.0},
        "a32": {"near": 10.0, "far": 10.0},
        "b32_near": {"near": 11.0, "far": 100.0},
        "b32_far": {"near": 15.0, "far": 20.0},
    }
    timings = {
        candidate_id: {
            shape_id: TimingValue.measured(time, 0.1)
            for shape_id, time in per_shape.items()
        }
        for candidate_id, per_shape in values.items()
    }

    selection = select_joint_menu(
        candidates,
        shapes,
        timings,
        floor_tile_ns=(32,),
        floor_plausible_ratio=1.25,
        compute_lomo=False,
    )

    assert selection.selected_ids[:4] == ("a8", "b8", "a32", "b32_near")


def test_extra_tile_floor_skips_unreachable_and_deduplicates_base_floor():
    candidates, shapes, timings = _inputs()

    unreachable = select_joint_menu(
        candidates,
        shapes,
        timings,
        floor_tile_ns=(32,),
        compute_lomo=False,
    )
    duplicate = select_joint_menu(
        candidates,
        shapes,
        timings,
        floor_tile_ns=(8, 8),
        compute_lomo=False,
    )

    assert unreachable.floor_size == 2
    assert duplicate.floor_size == 2
    assert duplicate.floor_tile_ns == (8,)


def test_extra_tile_floor_lomo_skips_tile_absent_from_training_subset():
    candidates, shapes, timings = _inputs()
    shapes["s2"] = replace(shapes["s2"], selected_tile_ns=(8, 16, 32))
    candidates["a32"] = _candidate("a32", "fc1", 32)
    candidates["b32"] = _candidate("b32", "fc2", 32)
    timings["a32"] = {"s2": TimingValue.measured(4.0, 0.1)}
    timings["b32"] = {"s2": TimingValue.measured(4.0, 0.1)}

    selection = select_joint_menu(
        candidates,
        shapes,
        timings,
        floor_tile_ns=(32,),
    )

    assert selection.floor_size == 4
    assert set(selection.leave_one_model_out) == {"model-a", "model-b"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"floor_tile_ns": (0,)}, "floor_tile_ns values must be positive"),
        (
            {"floor_plausible_ratio": 0.99},
            "floor_plausible_ratio must be at least 1",
        ),
    ],
)
def test_extra_tile_floor_validates_options(kwargs, message):
    candidates, shapes, timings = _inputs()

    with pytest.raises(ValueError, match=message):
        select_joint_menu(candidates, shapes, timings, **kwargs)


def test_autotune_cost_counts_cross_product_per_bucket():
    candidates, shapes, _ = _inputs()
    result = autotune_cost(
        {"a8", "a16", "b8", "b16"},
        candidates=candidates,
        deployments=(
            DeploymentBuckets(
                deployment="model-a",
                model="model-a",
                activation_class="gated",
                tp=1,
                ep=1,
                weight=1.0,
                windows=(((8, 16), 1),),
            ),
            DeploymentBuckets(
                deployment="model-b",
                model="model-b",
                activation_class="gated",
                tp=1,
                ep=1,
                weight=1.0,
                windows=(((8, 16), 1),),
            ),
        ),
        cost=AutotuneCost(seconds_per_pair=0.5, seconds_per_compile=2.0),
    )
    # Each deployment compiles four kernels and profiles one pair at each tile,
    # plus the runner's default tactic once in the bucket.
    assert result.runtime_kernels == 4
    assert result.deployment_kernels_mean == 4
    assert result.deployment_pairs_mean == 3
    assert result.deployment_pairs_max == 3
    assert result.deployment_seconds_mean == 9.5


def test_autotune_cost_counts_twins_and_deduplicates_explicit_twin():
    candidates, _, _ = _inputs()
    candidates["a8"] = replace(
        candidates["a8"], options={"tile_n": 8, "tile_scheduler": "persistent"}
    )
    candidates["b8"] = replace(
        candidates["b8"], options={"tile_n": 8, "tile_scheduler": "persistent"}
    )
    candidates["a8_explicit_twin"] = replace(
        candidates["a8"],
        candidate_id="a8_explicit_twin",
        options={
            "tile_n": 8,
            "tile_scheduler": "persistent",
            "use_clc_fast_drain": True,
            "use_work_throttle": True,
        },
    )
    deployment = DeploymentBuckets(
        deployment="model-a",
        model="model-a",
        activation_class="gated",
        tp=1,
        ep=1,
        weight=1.0,
        windows=(((8,), 3),),
    )
    result = autotune_cost(
        {"a8", "a8_explicit_twin", "b8"},
        candidates=candidates,
        deployments=(deployment,),
        cost=AutotuneCost(),
    )
    assert result.runtime_kernels == 4
    assert result.deployment_kernels_mean == 4
    assert result.deployment_pairs_mean == 15


def test_autotune_cost_matches_cross_class_twin_collapse_from_emitter():
    candidates = {
        "elementwise_base": JointCandidate(
            candidate_id="elementwise_base",
            fc="fc1",
            activation_class="elementwise",
            tile_n=8,
            options={"tile_n": 8, "tile_scheduler": "persistent"},
        ),
        "gated_twin": JointCandidate(
            candidate_id="gated_twin",
            fc="fc1",
            activation_class="gated",
            tile_n=8,
            options={
                "tile_n": 8,
                "tile_scheduler": "persistent",
                "use_clc_fast_drain": True,
                "use_work_throttle": True,
            },
        ),
        "fc2": _candidate("fc2", "fc2", 8),
    }
    result = autotune_cost(
        set(candidates),
        candidates=candidates,
        deployments=(
            DeploymentBuckets(
                deployment="gated",
                model="gated",
                activation_class="gated",
                tp=1,
                ep=1,
                weight=1.0,
                windows=(((8,), 1),),
            ),
        ),
        cost=AutotuneCost(),
    )
    # Emission keeps the element-wise base, transfers the gated class to it,
    # and lets the runtime add the twin: two gated FC1 kernels x one FC2.
    assert result.deployment_kernels_mean == 3
    assert result.deployment_pairs_mean == 3


def test_autotune_cost_ignores_unreachable_tiles():
    candidates, _, _ = _inputs()
    result = autotune_cost(
        set(candidates),
        candidates=candidates,
        deployments=(
            DeploymentBuckets(
                deployment="model-a",
                model="model-a",
                activation_class="gated",
                tp=1,
                ep=1,
                weight=1.0,
                windows=(((8,), 1),),
            ),
        ),
        cost=AutotuneCost(seconds_per_pair=1.0, seconds_per_compile=1.0),
    )
    assert result.runtime_kernels == 4
    assert result.deployment_kernels_mean == 2
    assert result.deployment_pairs_mean == 2
    assert result.deployment_seconds_mean == 4


def test_autotune_cost_uses_deployment_weights_for_mean_not_max():
    candidates, _, _ = _inputs()
    result = autotune_cost(
        {"a8", "a16", "b8", "b16"},
        candidates=candidates,
        deployments=(
            DeploymentBuckets(
                deployment="small",
                model="small",
                activation_class="gated",
                tp=1,
                ep=1,
                weight=1.0,
                windows=(((8,), 1),),
            ),
            DeploymentBuckets(
                deployment="wide",
                model="wide",
                activation_class="gated",
                tp=1,
                ep=1,
                weight=3.0,
                windows=(((8, 16), 1),),
            ),
        ),
        cost=AutotuneCost(seconds_per_pair=1.0, seconds_per_compile=1.0),
    )
    assert result.deployment_kernels_mean == 3.5
    assert result.deployment_pairs_mean == 2.75
    assert result.deployment_pairs_max == 3
    assert result.deployment_seconds_mean == 6.25
    assert result.deployment_seconds_max == 7


def test_target_buckets_use_runtime_hybrid_buckets_and_tile_windows(tmp_path):
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets
    from flashinfer.prims_ts.moe.config_mapper import _selected_tile_ns
    from flashinfer.prims_ts.moe.tools.sweep.menu_config import load_menu_target
    from flashinfer.prims_ts.moe.tools.sweep.shapes import SUPPORTED_NVFP4_TILE_N

    target_path = tmp_path / "target.json"
    target_path.write_text(
        """
        {
          "schema_version": 1,
          "sm_arch": 107,
          "dtype": "NvFP4xNvFP4",
          "routings": ["uniform"],
          "deployments": [{
            "model": "kimi-k3-nvfp4",
            "tp_ep": [{
              "tp_ep": "4:4",
              "tokens": [32, 512],
              "tune_max_num_tokens": 8192
            }]
          }]
        }
        """
    )
    target = load_menu_target(target_path)
    deployments = deployment_buckets_from_target(asdict(target))
    assert len(deployments) == 1
    buckets = get_hybrid_num_tokens_buckets(8192, 1)
    assert sum(count for _window, count in deployments[0].windows) == len(buckets)
    expected = {
        _selected_tile_ns(
            num_tokens=bucket,
            top_k=16,
            num_local_experts=224,
            num_experts=896,
            supported_tiles=SUPPORTED_NVFP4_TILE_N,
        )
        for bucket in buckets
    }
    assert {window for window, _count in deployments[0].windows} == expected


def test_lomo_cost_uses_only_held_out_deployments():
    candidates, shapes, timings = _inputs()
    deployments = (
        DeploymentBuckets(
            deployment="model-a",
            model="model-a",
            activation_class="gated",
            tp=1,
            ep=1,
            weight=1.0,
            windows=(((8,), 1),),
        ),
        DeploymentBuckets(
            deployment="model-b",
            model="model-b",
            activation_class="gated",
            tp=1,
            ep=1,
            weight=1.0,
            windows=(((8, 16), 1),),
        ),
    )
    selection = select_joint_menu(
        candidates,
        shapes,
        timings,
        cost=AutotuneCost(seconds_per_pair=1.0, seconds_per_compile=1.0),
        deployments=deployments,
    )
    held_a = selection.leave_one_model_out["model-a"]["held_out_curve"][-1]
    held_b = selection.leave_one_model_out["model-b"]["held_out_curve"][-1]
    assert held_a["deployment_pairs_max"] == 2
    assert held_b["deployment_pairs_max"] == 3
