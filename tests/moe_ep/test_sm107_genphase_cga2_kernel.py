# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the admitted two-CTA N64 geometry, including both SFB tile halves."""

import pytest
import torch


@pytest.mark.arch_rubin
@pytest.mark.parametrize("capacity", [1, 17, 65, 129])
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize(
    "schedule,work_id",
    [
        (("grouped", None), "grid_stride"),
        (("phase_interleave", None), "atomic_counter"),
    ],
)
def test_genphase_n64_cga2(monkeypatch, capacity, early, schedule, work_id):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    from tests.moe_ep.test_sm107_kernel_boundaries import _run_case

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run_case(
        "mxfp8_e4m3",
        capacity,
        8,
        dict(
            kernel_variant="genphase",
            mma_tiler_mnk=(256, 64, 128),
            cluster_shape_mn=(2, 1),
            fc2_use_bulk=True,
            fc2_tma_stages=1,
            epi_flag_batches=(1, 4),
            schedule_policy=schedule,
            work_id_mode=work_id,
            apply_routing_weights_before_fc2=early,
            reduce_topk_in_kernel=False,
        ),
    )
