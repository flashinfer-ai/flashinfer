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

import numpy as np
import pytest
import torch

import flashinfer.gdn_kernels.blackwell as blackwell
import flashinfer.gdn_paged as gdn_paged
from flashinfer.gdn_paged import (
    _CAKE_GDN_PAGED_AUTO_BANDS,
    _cake_cp_paged_rejection_reason,
    _chunk_gated_delta_rule_paged_cake_cp,
    _paged_auto_picks_cake,
    _paged_auto_uses_cake,
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
        assert _CAKE_GDN_PAGED_AUTO_BANDS[(SM100, dtype)] == (
            (12, 2048),
            (24, 4096),
            (32, 6144),
        )
        assert _CAKE_GDN_PAGED_AUTO_BANDS[(SM103, dtype)] == ((12, 2048), (32, 4096))
        assert _CAKE_GDN_PAGED_AUTO_BANDS[(SM107, dtype)] == (
            (12, 2048),
            (32, 4096),
            (48, 6144),
        )


# (parallel states, length bound, picks on SM100, on SM103, on SM107)
_TABLE = [
    # Tier (12, 2048) on every capability.
    (12, 2048, True, True, True),
    (12, 2047, False, False, False),
    (13, 2048, False, False, False),
    (1, 2048, True, True, True),
    (1, 2047, False, False, False),
    # Tier (24, 4096) on SM100; (32, 4096) covers it on SM103 and SM107.
    (13, 4096, True, True, True),
    (24, 4096, True, True, True),
    (24, 4095, False, False, False),
    (25, 4096, False, True, True),
    # Tier (32, 6144) on SM100.
    (25, 6144, True, True, True),
    (32, 6144, True, True, True),
    (32, 6143, False, True, True),
    (33, 6144, False, False, True),
    # Tier (32, 4096) on SM103 and SM107.
    (32, 4096, False, True, True),
    (32, 4095, False, False, False),
    (33, 4096, False, False, False),
    # Tier (48, 6144) on SM107 only.
    (48, 6144, False, False, True),
    (48, 6143, False, False, False),
    (49, 6144, False, False, False),
    (49, 65536, False, False, False),
    (32, 65536, True, True, True),
    # No sequence or no tokens.
    (0, 4096, False, False, False),
    (4, 0, False, False, False),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("io_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("parallel_states", "seqlen_bound", "sm100", "sm103", "sm107"), _TABLE
)
def test_band(parallel_states, seqlen_bound, sm100, sm103, sm107, io_dtype):
    for capability, picks in ((SM100, sm100), (SM103, sm103), (SM107, sm107)):
        # The band reads N * Hs only: N sequences of one head, one of N heads.
        for num_seqs, heads in ((parallel_states, 1), (1, parallel_states)):
            got = _picks(capability, num_seqs, heads, seqlen_bound, io_dtype)
            assert got is picks, (capability, num_seqs, heads, seqlen_bound)


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


class _FakeCudaTensor:
    """Host metadata of a CUDA tensor; the auto choice reads nothing else."""

    def __init__(self, *shape, dtype=torch.bfloat16):
        self.shape, self.dtype, self.device = shape, dtype, torch.device("cuda")

    def size(self, dim):
        return self.shape[dim]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("aligned_prefixes", "max_seqlen", "picks"),
    [
        # 2 x 4 = 8 states and 2 x 2048 new tokens: inside the band.
        (True, None, True),
        (False, None, False),
        (True, 2048, True),
        (True, 4096, True),
        (False, 4096, False),
        # A hint below ceil(total / num_seqs), which the Cake route rejects.
        (True, 2047, False),
        (True, 0, False),
        # Any integer type is a hint; bool and float are not.
        (True, np.int64(2048), True),
        (True, np.int32(4096), True),
        (True, np.int64(2047), False),
        (False, np.int64(4096), False),
        (True, True, False),
        (True, False, False),
        (True, 2048.0, False),
        (True, np.float64(4096), False),
    ],
)
def test_auto_needs_aligned_prefixes_and_a_valid_hint(
    monkeypatch, aligned_prefixes, max_seqlen, picks
):
    monkeypatch.setattr(gdn_paged, "get_compute_capability", lambda device: SM103)
    monkeypatch.setattr(gdn_paged, "_cake_cp_paged_rejection_reason", lambda *a: None)
    q = _FakeCudaTensor(4096, 4, 128)
    v = _FakeCudaTensor(4096, 4, 128)
    pool = _FakeCudaTensor(8, 4, 128, 128, dtype=torch.float32)
    got = _paged_auto_uses_cake(
        q,
        q,
        v,
        pool,
        None,
        num_seqs=2,
        page_size=1024,
        max_seqlen=max_seqlen,
        aligned_prefixes=aligned_prefixes,
    )
    assert got is picks


@pytest.mark.cpu_only
def test_auto_falls_back_when_the_route_rejects(monkeypatch):
    monkeypatch.setattr(gdn_paged, "get_compute_capability", lambda device: SM103)
    monkeypatch.setattr(
        gdn_paged, "_cake_cp_paged_rejection_reason", lambda *a: "rejected"
    )
    q = _FakeCudaTensor(4096, 4, 128)
    assert not _paged_auto_uses_cake(
        q,
        q,
        q,
        q,
        None,
        num_seqs=2,
        page_size=1024,
        max_seqlen=None,
        aligned_prefixes=True,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_seqlen", [True, False, 2048.0, np.float64(2048), "2048"])
def test_cake_gdn_rejects_a_non_integer_hint(max_seqlen):
    """backend="cake_gdn" raises a ValueError for a hint that is not an integer,
    before it touches any tensor."""
    q = torch.zeros(64, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="max_seqlen must be an integer"):
        _chunk_gated_delta_rule_paged_cake_cp(
            q,
            q,
            q,
            None,
            None,
            None,
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32),
            state_pool=torch.zeros(2, 2, 128, 128),
            page_indices=torch.zeros(1, dtype=torch.int32),
            page_size=64,
            slots=None,
            use_qk_l2norm_in_kernel=False,
            output=None,
            gate_domain="linear",
            use_gate_in_kernel=False,
            A_log=None,
            dt_bias=None,
            beta_is_logit=False,
            max_seqlen=max_seqlen,
        )
