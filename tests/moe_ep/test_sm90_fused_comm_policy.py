# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host eligibility for shared BF16 stores and dispatch-count publication."""

import pytest

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.src.moe_hopper_fp8.fused_comm_policy import (
    FusedCommOptimizations,
    resolve_fused_comm_optimizations,
)


def resolve(**overrides):
    args = dict(
        mma_tiler_mnk=(256, 64, 128),
        cluster_shape_mnk=(2, 1, 1),
        static_expert_shape=(96, 6144, 7168),
        swap_ab=True,
        fused_token_comm=True,
        bf16_output=True,
    )
    args.update(overrides)
    return resolve_fused_comm_optimizations(**args)


@pytest.mark.parametrize("hidden", [512, 2048, 7168])
@pytest.mark.parametrize("tile_k", [128, 256])
def test_output_layout_does_not_depend_on_input_precision(hidden, tile_k):
    assert resolve(
        mma_tiler_mnk=(256, 64, tile_k),
        static_expert_shape=(96, 6144, hidden),
    ) == FusedCommOptimizations(True, True)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(swap_ab=False),
        dict(fused_token_comm=False),
        dict(dedup_dispatch=True),
    ],
)
def test_unqualified_protocol_retains_original_paths(overrides):
    assert resolve(**overrides) == FusedCommOptimizations()


@pytest.mark.parametrize(
    "overrides",
    [
        dict(bf16_output=False),
        dict(combine_format="fp8"),
        dict(token_back_by_dispatch=True),
        dict(fc2_in_kernel_topk_reduce=True),
        dict(mma_tiler_mnk=(256, 32, 128)),
        dict(mma_tiler_mnk=(128, 64, 128)),
        dict(static_expert_shape=None),
        dict(static_expert_shape=(96, 6144, 768)),
    ],
)
def test_unqualified_consumer_keeps_scalar_stores(overrides):
    assert resolve(**overrides) == FusedCommOptimizations(False, True)


@pytest.mark.parametrize(
    "pair,skip", [(False, False), (True, False), (False, True), (True, True)]
)
def test_independent_controls(pair, skip):
    result = resolve(paired_stores=pair, skip_zero_counts=skip)
    assert result == FusedCommOptimizations(pair, skip)


@pytest.mark.parametrize("name", ["paired_stores", "skip_zero_counts"])
def test_diagnostic_controls_reject_non_boolean(name):
    with pytest.raises(ValueError, match="must be a bool"):
        resolve(**{name: 1})
