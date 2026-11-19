# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The band of ``chunk_gated_delta_rule_paged(backend="auto")``.

``_paged_auto_picks_cake`` reads host values only, so its boundaries are
pinned here without a GPU: the parallel-state and length bounds of every tier
of each capability, and the ``max_seqlen`` hint against the mean estimate.
"""

import pytest
import torch

import flashinfer.gdn_kernels.blackwell as blackwell
from flashinfer.gdn_paged import (
    _CAKE_GDN_PAGED_AUTO_BANDS,
    _cake_cp_paged_rejection_reason,
    _paged_auto_picks_cake,
)
from flashinfer.gdn_prefill import _cake_gdn_cp_band_max_seqlen

SM100, SM103, SM107 = (10, 0), (10, 3), (10, 7)


def _picks(capability, num_seqs, num_sab_heads, seqlen_bound, io_dtype=torch.bfloat16):
    return _paged_auto_picks_cake(
        capability=capability,
        num_seqs=num_seqs,
        num_sab_heads=num_sab_heads,
        seqlen_bound=seqlen_bound,
        io_dtype=io_dtype,
    )


@pytest.mark.cpu_only
def test_band_constants():
    for dtype in (torch.bfloat16, torch.float16):
        for capability in (SM100, SM103):
            assert _CAKE_GDN_PAGED_AUTO_BANDS[(capability, dtype)] == (
                (12, 2048),
                (32, 4096),
            )
        assert _CAKE_GDN_PAGED_AUTO_BANDS[(SM107, dtype)] == (
            (12, 2048),
            (32, 4096),
            (48, 6144),
        )


# (parallel states, length bound, picks on SM100/SM103, picks on SM107)
_TABLE = [
    # Tier (12, 2048) on every capability.
    (12, 2048, True, True),
    (12, 2047, False, False),
    (13, 2048, False, False),
    (1, 2048, True, True),
    (1, 2047, False, False),
    # Tier (32, 4096) on every capability.
    (13, 4096, True, True),
    (32, 4096, True, True),
    (32, 4095, False, False),
    (33, 4096, False, False),
    # Tier (48, 6144) on SM107 only.
    (33, 6144, False, True),
    (48, 6144, False, True),
    (48, 6143, False, False),
    (49, 6144, False, False),
    (49, 65536, False, False),
    (32, 65536, True, True),
    # No sequence or no tokens.
    (0, 4096, False, False),
    (4, 0, False, False),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("io_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capability", [SM100, SM103, SM107])
@pytest.mark.parametrize(
    ("parallel_states", "seqlen_bound", "picks_sm100", "picks_sm107"), _TABLE
)
def test_band(
    capability, parallel_states, seqlen_bound, picks_sm100, picks_sm107, io_dtype
):
    picks = picks_sm107 if capability == SM107 else picks_sm100
    # The band reads N * Hs only: N sequences of one head and one sequence of N heads.
    assert _picks(capability, parallel_states, 1, seqlen_bound, io_dtype) is picks
    assert _picks(capability, 1, parallel_states, seqlen_bound, io_dtype) is picks


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "capability", [(9, 0), (10, 1), (11, 0), (12, 0), (12, 1), (8, 0)]
)
def test_other_capabilities_keep_flashinfer(capability):
    assert not _picks(capability, 1, 4, 65536)


@pytest.mark.cpu_only
@pytest.mark.parametrize("capability", [SM100, SM103, SM107])
@pytest.mark.parametrize("io_dtype", [torch.float32, torch.float8_e4m3fn])
def test_other_io_dtypes_keep_flashinfer(capability, io_dtype):
    assert not _picks(capability, 1, 4, 65536, io_dtype)


@pytest.mark.cpu_only
@pytest.mark.parametrize("capability", [SM100, SM103, SM107])
@pytest.mark.parametrize(
    ("max_seqlen", "total", "num_seqs", "picks"),
    [
        # Without a hint: ceil(total / num_seqs); 2 x 8 = 16 states, tier (32, 4096).
        (None, 8192, 2, True),
        (None, 8191, 2, True),
        (None, 8190, 2, False),
        # A balanced batch of short sequences stays out though its total is long.
        (None, 2 * 2048, 2, False),
        # The hint replaces the estimate in both directions.
        (4096, 2 * 2048, 2, True),
        (4095, 2 * 8192, 2, False),
        (2048, 16384, 2, False),
    ],
)
def test_hint_or_mean_estimate(capability, max_seqlen, total, num_seqs, picks):
    bound = _cake_gdn_cp_band_max_seqlen(max_seqlen, total, num_seqs)
    assert _picks(capability, num_seqs, 8, bound) is picks


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("max_seqlen", "total", "picks"),
    [(None, 4096, True), (None, 4094, False), (2048, 2048, True), (2047, 8192, False)],
)
def test_hint_or_mean_estimate_short_tier(max_seqlen, total, picks):
    # 2 x 4 = 8 states: tier (12, 2048).
    bound = _cake_gdn_cp_band_max_seqlen(max_seqlen, total, 2)
    assert _picks(SM103, 2, 4, bound) is picks


@pytest.mark.cpu_only
def test_rejection_gate(monkeypatch):
    """The gate names the missing Cake programs and otherwise forwards the
    paged route's own check, which rejects CPU tensors."""
    q = torch.zeros(64, 2, 128, dtype=torch.bfloat16)
    pool = torch.zeros(2, 2, 128, 128)
    if blackwell._gdn_cp_paged_rejection_reason is not None:
        assert _cake_cp_paged_rejection_reason(q, q, q, pool, 64) == (
            "requires CUDA tensors"
        )
    monkeypatch.setattr(blackwell, "_chunk_gated_delta_rule_gdn_cp_paged_sm100", None)
    assert _cake_cp_paged_rejection_reason(q, q, q, pool, 64) == (
        "needs the Cake GDN CP programs"
    )
