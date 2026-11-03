# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise retained legacy finalization while internode GIN is disabled."""

import os

import pytest
import torch

from .test_moe_ep_nvfp4_cutedsl_mega_multirank import _launcher_ranks, _run_mega_layer

pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_blackwell, pytest.mark.node_exclusive]


@pytest.mark.parametrize("live,per_token_scale", [(17, False), (64, True)])
def test_legacy_nvfp4_off_persistent_finalization(live, per_token_scale):
    from flashinfer.moe_ep import FinalizeMode

    rank, world = _launcher_ranks()
    if world != 4:
        pytest.skip("requires four ranks in one NVLink domain")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    live_count = torch.full((1,), live, dtype=torch.int32, device="cuda")
    _run_mega_layer(
        rank,
        world,
        quantize_input=True,
        num_tokens=live,
        max_tokens=64,
        per_token_scale=per_token_scale,
        config_extra=dict(
            finalize_mode=FinalizeMode.PERSISTENT,
            num_valid_tokens_tensor=live_count,
        ),
    )
