# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused static/host tests for the autonomous device epilogue."""

import inspect

import pytest

cutlass = pytest.importorskip("cutlass")

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.device_epilogue import (  # noqa: E402
    DeviceEpilogueConfig,
    PersistentMxfp8Mxfp4Epilogue,
    route_metadata_host,
)


@pytest.mark.parametrize("warps", [4, 8])
def test_device_epilogue_geometry(warps) -> None:
    config = DeviceEpilogueConfig(epilogue_warps=warps)
    assert config.token_groups == 4
    assert config.fc1_output_features_per_tile == 64
    assert config.fc1_active_warps == warps
    assert config.fc2_active_warps == warps
    assert config.accumulator_tmem_columns == 256


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cta_tile_features": 256}, "128-row"),
        ({"cta_tile_tokens": 48}, "requires N64, N128, or N192"),
        ({"accumulator_stages": 1}, "exactly two"),
        ({"epilogue_warps": 6}, "four or eight"),
        ({"epilogue_warps": 8, "barrier_id": 3}, "must differ"),
        ({"epilogue_warps": 8, "barrier_id": 4}, "must differ"),
        ({"gate_up_clamp": -1.0}, "non-negative"),
    ],
)
def test_device_epilogue_rejects_unsupported_geometry(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        DeviceEpilogueConfig(**kwargs)


def test_route_metadata_wire_format() -> None:
    source_rank = 0x1234
    source_token = 0x89ABCDEF
    source_topk = 0x0056
    packed = (((source_rank << 16) | source_topk) << 32) | source_token
    assert route_metadata_host(packed) == (
        source_rank,
        source_token,
        source_topk,
    )


def test_device_interface_has_no_routing_score() -> None:
    for method_name in ("run", "_run_fc1_task_tile", "_run_fc2_task_tile"):
        parameters = inspect.signature(
            getattr(PersistentMxfp8Mxfp4Epilogue, method_name)
        ).parameters
        assert "route_score" not in parameters
        assert "topk_score" not in parameters
        assert "routing_weight" not in parameters


def test_device_epilogue_owns_pipeline_and_publication() -> None:
    source = inspect.getsource(PersistentMxfp8Mxfp4Epilogue.run)
    assert "consumer_wait" in source
    assert "fence_view_async_tmem_load" in source
    assert "consumer_release" in source
    assert "fence_acq_rel_gpu" in source
    assert "_red_add_release_gpu_s32" in source


def test_device_epilogue_reuses_numerical_primitives() -> None:
    source = inspect.getsource(PersistentMxfp8Mxfp4Epilogue)
    assert "Fc1Epilogue" in source
    assert "_clamp_and_swiglu(" in source
    assert "stage_native_bf16(" in source
    assert "store_staged_vector(" in source


def test_device_epilogue_exchanges_paired_native_feature_warps() -> None:
    source = inspect.getsource(PersistentMxfp8Mxfp4Epilogue)
    assert "_load_fc1_native_half" in source
    assert "epilogue_exchange" in source
    assert "pair_idx" in source
