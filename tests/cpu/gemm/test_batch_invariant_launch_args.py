# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant GEMM records' runtime launch arguments, without a device: the persistent grid, the swizzle size and
the raster order reach the kernel as arguments, so an invalid forced value is a typed error on the host, not a launch
failure, and a forced grid overrides the device's co-resident count."""

import dataclasses

import pytest

from flashinfer.gemm.batch_invariant.blocks.compile import launch_args, object_record
from flashinfer.gemm.batch_invariant.generator import Knobs

pytestmark = pytest.mark.cpu_only

KNOBS = Knobs((128, 128), (1, 2), False, True, "n", 2)


def test_launch_args_are_the_record_values_with_the_device_grid():
    assert launch_args(KNOBS, 148) == (148, 2, 0)
    assert launch_args(
        dataclasses.replace(KNOBS, raster_along="m", swizzle_size=1), 212
    ) == (212, 1, 1)


def test_forced_grid_overrides_the_device_count():
    assert launch_args(dataclasses.replace(KNOBS, grid_clusters=8), 148) == (8, 2, 0)


@pytest.mark.parametrize("grid", [-1, -148])
def test_negative_forced_grid_is_rejected_on_the_host(grid):
    with pytest.raises(ValueError, match="persistent grid"):
        launch_args(dataclasses.replace(KNOBS, grid_clusters=grid), 148)


def test_device_count_of_zero_is_rejected_on_the_host():
    with pytest.raises(ValueError, match="persistent grid"):
        launch_args(KNOBS, 0)


@pytest.mark.parametrize(
    "field, value", [("swizzle_size", 0), ("swizzle_size", 3), ("raster_along", "k")]
)
def test_invalid_swizzle_or_raster_is_rejected_on_the_host(field, value):
    cfg = dataclasses.replace(KNOBS, **{field: value})
    assert not cfg.well_formed()  # the record predicate rejects them too
    with pytest.raises(ValueError, match="swizzle"):
        launch_args(cfg, 148)


def test_object_record_drops_the_runtime_values_only():
    rec = object_record(dataclasses.replace(KNOBS, grid_clusters=8))
    assert (rec.raster_along, rec.swizzle_size, rec.grid_clusters) == ("m", 1, 0)
    assert (rec.tile_m, rec.tile_n, rec.cluster_shape_mn) == (128, 128, (1, 2))
