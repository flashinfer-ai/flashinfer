# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import json
import math
from pathlib import Path

import pytest

from flashinfer.prims_ts.moe.tools.sweep.menu_config import (
    activation_class,
    build_menu_shapes,
    load_menu_target,
)


def _write_target(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sm_arch": 107,
                "dtype": "NvFP4xNvFP4",
                "routings": ["uniform", "balanced_round_robin"],
                "deployments": [
                    {
                        "model": "deepseek-v4-pro-nvfp4",
                        "weight": 2.0,
                        "tp_ep": [
                            {
                                "tp_ep": "4:4",
                                "tokens": [8, 16],
                                "token_weights": [1.0, 3.0],
                            },
                            {
                                "tp_ep": [8, 8],
                                "tokens": [8],
                                "weight": 3.0,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_target_weights_are_normalized_per_deployment(tmp_path):
    target_path = tmp_path / "target.json"
    _write_target(target_path)
    target = load_menu_target(target_path)
    shapes = build_menu_shapes(target)

    assert math.isclose(sum(shape.weight for shape in shapes), 2.0)
    by_origin = {
        (origin.tp, origin.ep, origin.global_num_tokens): origin.weight
        for shape in shapes
        for origin in shape.origins
    }
    # Routing gets half of each logical point's weight.
    assert math.isclose(by_origin[4, 4, 8], 2.0 * 0.25 * 0.25 * 0.5)
    assert math.isclose(by_origin[4, 4, 16], 2.0 * 0.25 * 0.75 * 0.5)
    assert math.isclose(by_origin[8, 8, 8], 2.0 * 0.75 * 0.5)


def test_target_rejects_mixed_quant_modes(tmp_path):
    target_path = tmp_path / "target.json"
    _write_target(target_path)
    value = json.loads(target_path.read_text())
    value["dtype"] = "MxFP8xMxFP8"
    target_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="all deployments"):
        load_menu_target(target_path)


def test_tune_max_num_tokens_is_optional_and_covers_target_tokens(tmp_path):
    target_path = tmp_path / "target.json"
    _write_target(target_path)
    value = json.loads(target_path.read_text())
    points = value["deployments"][0]["tp_ep"]
    points[0]["tune_max_num_tokens"] = 8192
    target_path.write_text(json.dumps(value))
    target = load_menu_target(target_path)
    assert target.deployments[0].parallelism[0].tune_max_num_tokens == 8192
    assert target.deployments[0].parallelism[1].tune_max_num_tokens is None

    points[0]["tune_max_num_tokens"] = 8
    target_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="at least the largest target token"):
        load_menu_target(target_path)


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("swiglu", "gated"),
        ("situ", "gated"),
        ("relu2", "elementwise"),
        ("silu", "elementwise"),
    ],
)
def test_activation_classes(activation, expected):
    assert activation_class(activation) == expected


def test_checked_in_sm107_target_loads():
    path = (
        Path(__file__).parents[3]
        / "flashinfer/prims_ts/moe/tools/sweep/menus/sm107_nvfp4.json"
    )
    target = load_menu_target(path)
    assert target.sm_arch == 107
    assert target.dtype == "NvFP4xNvFP4"
    assert {deployment.activation for deployment in target.deployments} >= {
        "Swiglu",
        "Situ",
        "Relu2",
    }
