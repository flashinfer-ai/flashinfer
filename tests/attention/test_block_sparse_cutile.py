# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""cuTile block-sparse coverage kept outside the generic backend matrix."""

import pytest

from flashinfer.cutile.cutile_common import is_cuda_tile_available
from tests.attention.test_block_sparse import _run_block_sparse_attention_case
from tests.test_helpers.parametrize import pairwise_product_cases, parametrize_product

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

from flashinfer.attention.kernels.cutile import (  # noqa: E402
    fmha_prefill_bsr_cutile as _prefill_bsr_cutile,
)



@pytest.fixture(autouse=True, scope="module")
def _single_cutile_prefill_config():
    """Check one config per case instead of autotuning (keeps CI time bounded)."""
    saved = _prefill_bsr_cutile._AUTOTUNE_DISABLED
    _prefill_bsr_cutile._AUTOTUNE_DISABLED = True
    _prefill_bsr_cutile._prefill_paged_lpt_tune_cache.clear()
    yield
    _prefill_bsr_cutile._AUTOTUNE_DISABLED = saved
    # Do not leak single-config tuning results into later tests in this process.
    _prefill_bsr_cutile._prefill_paged_lpt_tune_cache.clear()


# Keep dependent dimensions coupled so --full preserves the original 660 cases.
@parametrize_product(
    {
        "R,M": [(R, M) for R in (1, 4, 16, 128) for M in (64, 128, 256) if M % R == 0],
        "C,N": [(C, N) for C in (16, 128) for N in (64, 128, 256) if N % C == 0],
        "num_qo_heads,num_kv_heads": [
            (q, kv) for q in (1, 4, 16) for kv in (1, 4, 16) if q % kv == 0
        ],
        "head_dim": [128, 256],
    },
    regular=pairwise_product_cases,
)
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
