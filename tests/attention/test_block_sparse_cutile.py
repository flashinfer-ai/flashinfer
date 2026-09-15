# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""cuTile block-sparse coverage kept outside the generic backend matrix."""

import pytest

from flashinfer.cutile.cutile_common import is_cuda_tile_available
from tests.attention.test_block_sparse import _run_block_sparse_attention_case

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

pytestmark = pytest.mark.solo

_CUTILE_CASES = [
    (R, C, M, N, num_qo_heads, num_kv_heads, head_dim)
    for R in (1, 4, 16, 128)
    for C in (16, 128)
    for M in (64, 128, 256)
    for N in (64, 128, 256)
    for num_qo_heads in (1, 4, 16)
    for num_kv_heads in (1, 4, 16)
    for head_dim in (128, 256)
    if num_qo_heads % num_kv_heads == 0 and M % R == 0 and N % C == 0
]


@pytest.mark.parametrize("R,C,M,N,num_qo_heads,num_kv_heads,head_dim", _CUTILE_CASES)
def test_block_sparse_cutile(R, C, M, N, num_qo_heads, num_kv_heads, head_dim):
    """cuTile block-sparse attention must match the dense reference."""
    _run_block_sparse_attention_case(
        "cutile",
        R,
        C,
        M,
        N,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        False,
    )
