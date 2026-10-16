"""CPU tests of the SM100/SM103 ``cake_gdn`` ``use_cp="auto"`` band rule.

The band is evaluated on the caller's ``max_seqlen`` hint; without a hint the host estimates the longest sequence as
``ceil(total_seq_len / num_seqs)`` so a balanced batch of short sequences (``4 x 2048``) stays on the non-CP route the
Cake dispatcher selects for it, instead of entering the band on the packed length.
"""

import pytest
import torch

from flashinfer.gdn_prefill import (
    _CAKE_GDN_CP_SM100_BAND_MIN_MAX_SEQLEN,
    _CAKE_GDN_CP_SM100_BAND_WORK_FACTOR,
    _cake_gdn_cp_band_max_seqlen,
    _cake_gdn_cp_sm100_band,
)


def _band(
    *,
    num_seqs,
    hq,
    hv,
    max_seqlen,
    num_sms=148,
    capability=(10, 0),
    io_dtype=torch.bfloat16,
):
    state = torch.empty(0, dtype=torch.float32)
    return _cake_gdn_cp_sm100_band(
        device_capability=capability,
        num_seqs=num_seqs,
        num_sab_heads=max(hq, hv),
        max_seqlen=max_seqlen,
        num_sms=num_sms,
        io_dtype=io_dtype,
        initial_state=state,
        output_state=None,
        output_final_state=True,
        g=torch.empty(0),
        beta=torch.empty(0),
        checkpoint_every_n_tokens=0,
        state_indices=None,
    )


def test_band_constants():
    assert _CAKE_GDN_CP_SM100_BAND_WORK_FACTOR == 4
    assert _CAKE_GDN_CP_SM100_BAND_MIN_MAX_SEQLEN == 4096


@pytest.mark.parametrize(
    ("max_seqlen", "total", "num_seqs", "expected"),
    [
        (16384, 16384, 1, 16384),
        (2048, 8192, 4, 2048),
        (None, 16384, 1, 16384),
        (None, 8192, 4, 2048),
        (None, 16383, 16, 1024),
        (None, 8256, 2, 4128),
    ],
)
def test_band_bound_is_the_hint_or_the_balanced_estimate(
    max_seqlen, total, num_seqs, expected
):
    assert _cake_gdn_cp_band_max_seqlen(max_seqlen, total, num_seqs) == expected


@pytest.mark.parametrize("num_sms", [148, 152])
def test_balanced_short_batches_stay_outside_the_band_without_a_hint(num_sms):
    # 4 x 2048 with Ho=8 (4*4*8 = 128 < 148): the packed length 8192 would enter the band, the estimate 2048 does not.
    assert not _band(
        num_seqs=4,
        hq=4,
        hv=8,
        max_seqlen=_cake_gdn_cp_band_max_seqlen(None, 8192, 4),
        num_sms=num_sms,
    )
    assert not _band(
        num_seqs=8,
        hq=2,
        hv=4,
        max_seqlen=_cake_gdn_cp_band_max_seqlen(None, 8192, 8),
        num_sms=num_sms,
    )
    assert _band(
        num_seqs=1,
        hq=8,
        hv=16,
        max_seqlen=_cake_gdn_cp_band_max_seqlen(None, 16384, 1),
        num_sms=num_sms,
    )
    assert _band(
        num_seqs=2,
        hq=4,
        hv=8,
        max_seqlen=_cake_gdn_cp_band_max_seqlen(None, 16384, 2),
        num_sms=num_sms,
    )


@pytest.mark.parametrize("num_sms", [148, 152, 160])
def test_band_membership_of_the_contract_rows(num_sms):
    # 17 band rows per architecture of the 66-row qwen_tp_fp32state contract: 1x16384 and 1x4096 for every geometry,
    # 2x8192 only while 4 * 2 * Ho < SMs (Ho <= 16); the verdicts are identical for 148, 152 and 160 SMs.
    geometries = ((8, 16), (4, 8), (2, 4), (8, 32), (4, 16), (2, 8))
    members = 0
    for hq, hv in geometries:
        for num_seqs, max_seqlen in (
            (1, 16384),
            (1, 4096),
            (2, 8192),
            (4, 2048),
            (8, 1024),
            (16, 512),
            (32, 512),
            (64, 256),
            (1, 2048),
            (4, 512),
        ):
            verdict = _band(
                num_seqs=num_seqs, hq=hq, hv=hv, max_seqlen=max_seqlen, num_sms=num_sms
            )
            expected = max_seqlen >= 4096 and 4 * num_seqs * max(hq, hv) < num_sms
            assert verdict is expected, (hq, hv, num_seqs, max_seqlen, num_sms)
            members += int(verdict)
    assert members == 17


def test_band_requires_sm100_or_sm103_and_fp32_state():
    assert not _band(num_seqs=1, hq=8, hv=16, max_seqlen=16384, capability=(10, 7))
    assert not _band(num_seqs=1, hq=8, hv=16, max_seqlen=16384, capability=(9, 0))
    assert _band(
        num_seqs=1, hq=8, hv=16, max_seqlen=16384, capability=(10, 3), num_sms=152
    )
    assert not _band(num_seqs=1, hq=8, hv=16, max_seqlen=16384, io_dtype=torch.float32)
