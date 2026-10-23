# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU row-contract tests, independent of CuTe and GPU allocation."""

import dataclasses

import pytest

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_abi import (
    LeanFc12Config,
    LeanFc12Inputs,
    build_lean_row_plan,
)


def test_separate_data_scale_padding_and_empty_experts():
    config = LeanFc12Config(4, 256, 128, 256, 384)
    plan = build_lean_row_plan((0, 1, 129, 0), config)
    assert plan.expert_data_row_offsets == (0, 0, 64, 256, 256)
    assert plan.expert_scale_row_offsets == (0, 0, 128, 384, 384)
    assert plan.expert_token_sizes == (0, 1, 129, 0)


def test_existing_split_linear_scale_rows_can_share_data_padding():
    config = LeanFc12Config(3, 128, 256, 256, 256, scale_row_alignment=64)
    plan = build_lean_row_plan((65, 0, 1), config)
    assert plan.expert_data_row_offsets == plan.expert_scale_row_offsets
    assert plan.expert_data_row_offsets == (0, 128, 128, 192)


def test_all_empty_experts_need_no_rows():
    plan = build_lean_row_plan((0, 0), LeanFc12Config(2, 128, 128, 0, 0))
    assert plan.expert_data_row_offsets == plan.expert_scale_row_offsets == (0, 0, 0)


@pytest.mark.parametrize("counts", [(1,), (-1, 2), (1.0, 2), (True, 2), (129, 128)])
def test_invalid_counts_or_capacity(counts):
    with pytest.raises(ValueError):
        build_lean_row_plan(counts, LeanFc12Config(2, 128, 128, 256, 256))


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden", 64),
        ("intermediate", 0),
        ("local_num_experts", 0),
        ("scale_row_capacity", 64),
        ("data_row_capacity", -64),
        ("data_row_alignment", 0),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(LeanFc12Config(2, 128, 128, 256, 256), **{field: value})


def test_lean_abi_excludes_communication_and_weighting():
    fields = {field.name for field in dataclasses.fields(LeanFc12Inputs)}
    assert not fields.intersection(
        {"route_ids", "route_scores", "world_size", "rank", "peer_mapper"}
    )
    assert {
        "expert_data_row_offsets",
        "expert_scale_row_offsets",
        "expert_token_sizes",
    } <= fields
