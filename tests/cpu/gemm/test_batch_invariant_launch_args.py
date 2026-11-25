# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host validation and identity of configuration records shared by MoE and the formal GEMM model."""

import dataclasses
import re

import pytest

from flashinfer.gemm.batch_invariant.generator import Knobs

pytestmark = pytest.mark.cpu_only

KNOBS = Knobs((128, 128), (1, 2), False, True, "n", 2)


@pytest.mark.parametrize(
    "field, value", [("swizzle_size", 0), ("swizzle_size", 3), ("raster_along", "k")]
)
def test_invalid_swizzle_or_raster_is_rejected_on_the_host(field, value):
    cfg = dataclasses.replace(KNOBS, **{field: value})
    assert not cfg.well_formed()  # the record predicate rejects them too


PLACEMENT_FIELDS = {"raster_along", "swizzle_size", "grid_clusters"}


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, tuple):
        return (value[0] * 2, value[1])
    if isinstance(value, str):
        return "n" if value == "m" else "m"
    return value + 1


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(Knobs)])
def test_every_compile_time_knob_is_in_the_object_identity(field):
    """``describe_object()`` is ``describe()`` minus the placement tokens: a change to any other knob changes the object's
    name (two kernels never share an on-disk object), a change to a placement does not."""
    cfg = dataclasses.replace(KNOBS, grid_clusters=3)
    new = dataclasses.replace(cfg, **{field: _changed(getattr(cfg, field))})
    assert new.describe() != cfg.describe()
    if field in PLACEMENT_FIELDS:
        assert new.describe_object() == cfg.describe_object()
    else:
        assert new.describe_object() != cfg.describe_object()
    assert re.search(r"_r[mn]_sw\d|_g\d", new.describe_object()) is None, (
        new.describe_object()
    )
