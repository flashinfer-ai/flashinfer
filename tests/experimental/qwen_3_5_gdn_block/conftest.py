# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the Qwen 3.5 GDN block tests: the SM100 gate the FROST GPU modules request."""

import pytest
import torch


@pytest.fixture
def frost_device():
    """The current CUDA device when it is SM100 (cc 10.0) and the FROST backend serves it -- read off the API's own
    probe, so the skip names the exact reason."""
    from flashinfer.qwen_3_5_gdn_block import (
        Qwen35GdnGeometry,
        qwen_3_5_gdn_block_support,
    )

    from tests.experimental.qwen_3_5_gdn_block.reference import QWEN38_27B

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device (SM100)")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("requires sm_100 (cc 10.0)")
    sup = qwen_3_5_gdn_block_support(device, Qwen35GdnGeometry(**QWEN38_27B), "decode")
    if not sup.served:
        pytest.skip(f"the FROST GDN block is not served here: {sup.reason}")
    return device
