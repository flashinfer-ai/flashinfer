# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the Qwen 3.5 gated attention block tests: the exact-arch gate the GPU modules request."""

import pytest
import torch


@pytest.fixture
def rubin():
    """Skip unless the device is SM107 (cc 10.7) and the installed CuTe DSL serves it -- read off the API's own probe, so the skip
    names the exact reason (device, DSL version, missing Rubin target)."""
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry, qwen_3_5_block_support

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device (SM107)")
    if torch.cuda.get_device_capability(0) != (10, 7):
        pytest.skip("requires sm_107 (Rubin)")
    sup = qwen_3_5_block_support(
        torch.device("cuda", 0),
        Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2),
    )
    if not sup.served:
        pytest.skip(f"the Qwen 3.5 block is not served here: {sup.reason}")
    yield sup
