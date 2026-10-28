from __future__ import annotations

import hashlib

import torch

from b12x._lib.quant.lut_e4m3 import (
    LUT_E4M3_DIRECT_TABLE_ENTRIES,
    lut_e4m3_direct_table_cpu,
    lut_e4m3_value_table_cpu,
)


def test_lut_e4m3_direct_table_is_finite_and_bijective_by_rank() -> None:
    direct = lut_e4m3_direct_table_cpu()
    assert direct.dtype == torch.uint8
    assert direct.shape == (LUT_E4M3_DIRECT_TABLE_ENTRIES,)
    values = lut_e4m3_value_table_cpu()
    expected_histogram = torch.bincount(values.to(torch.int64), minlength=256) * 16
    for rate_index in range(3):
        labels = direct[rate_index << 16 : (rate_index + 1) << 16]
        assert not bool(torch.any(labels == 0x80))
        assert not bool(torch.any((labels & 0x7F) == 0x7F))
        torch.testing.assert_close(
            torch.bincount(labels.to(torch.int64), minlength=256),
            expected_histogram,
            rtol=0,
            atol=0,
        )


def test_lut_e4m3_tables_have_frozen_identity() -> None:
    values = lut_e4m3_value_table_cpu()
    assert values.dtype == torch.uint8
    assert values.shape == (1 << 12,)
    assert (
        hashlib.sha256(values.numpy().tobytes()).hexdigest()
        == "cca11fe5744c9c93a34f4217f342fbc0f74ecc8a007c076582424a505fc9da5e"
    )
    expected_sha256 = {
        2: "62027916386245a84c86156a0a08b6cf07e41548871af0e56fb40780558f6293",
        3: "afe7b3633e7d243b00b379b18ec4dca573722b3727cafef47fcb6470d7e7e6c9",
        4: "5a9620f0c4d8f0a60d0b6fbea921dcecbd193e6febf31043aaa6403c20389c2f",
    }
    direct = lut_e4m3_direct_table_cpu()
    for rate_index, bits in enumerate((2, 3, 4)):
        labels = direct[rate_index << 16 : (rate_index + 1) << 16]
        assert (
            hashlib.sha256(labels.numpy().tobytes()).hexdigest()
            == expected_sha256[bits]
        )
