# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""cuTile paged-decode coverage kept outside the generic backend matrix."""

import pytest
import torch

from flashinfer.cutile.cutile_common import is_cuda_tile_available
from tests.attention.test_batch_decode_kernels import (
    _run_batch_decode_with_paged_kv_cache_case,
)

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

pytestmark = pytest.mark.solo


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048, 16384])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_batch_decode_cutile(batch_size, kv_len, page_size, num_qo_heads, head_dim):
    """cuTile paged decode must match the single-decode reference."""
    _run_batch_decode_with_paged_kv_cache_case(
        "cutile",
        batch_size,
        kv_len,
        page_size,
        4,
        num_qo_heads,
        head_dim,
        "NHD",
        "NONE",
        0.0,
        True,
        torch.float16,
        torch.float16,
        True,
    )
