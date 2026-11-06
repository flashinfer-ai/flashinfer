"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""The schedule name of a mixed-length split launch (CPU; no GPU needed).

The split names the program of each part.  A part that falls back from a
refused cooperative launch switches its own ``schedule``; the split's name must
follow it.  The parts here are plain stand-ins and run on the caller stream
only, so the split builds no CUDA stream or event.
"""

from types import SimpleNamespace

import pytest

import flashinfer.cake_kda_tf32_runtime as runtime

pytestmark = pytest.mark.cpu_only

OWNER_HELPER = "fused_m64_independent_dvsplit_fp32_state_owner2helper"
PAIR_SHARE = "fused_m64_independent_dvsplit_fp32_state_pair_share"
SHORT = "fused_m64_independent_dvsplit_fp32_state"


def _part(schedule):
    return SimpleNamespace(
        schedule=schedule,
        route="fused_m64",
        gate_kind="kda",
        gpu_arch="sm_100a",
        sm_count=148,
    )


def test_split_schedule_names_the_current_part_programs():
    long_part, short_part = _part(OWNER_HELPER), _part(SHORT)
    split = runtime._MixLenSplitLaunch(
        [(long_part, 0, (0, 0)), (short_part, 0, (1, 3))], "cpu", "rng"
    )
    assert split._streams == {} and split._fork is None
    assert split.split_mode == "rng"
    assert split.schedule == f"mixlen_rng[{OWNER_HELPER}@0-0+{SHORT}@1-3]"

    # A refused cooperative launch switches the part to its fallback program.
    long_part.schedule = PAIR_SHARE
    assert split.schedule == f"mixlen_rng[{PAIR_SHARE}@0-0+{SHORT}@1-3]"
    assert "schedule" not in vars(split)
