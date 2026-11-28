"""FP8 split policy must reflect the actual device and active sparse work."""

import pytest

pytest.importorskip("cutlass")

from flashinfer.cute_dsl.sparse.sm100_blk64.dispatch_helpers import (
    sm100_blk64_auto_fp8_kv_splits,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "topk,heads,sq,sms,expected",
    [
        (900, 4, 8192, 148, 1),
        (256, 4, 64, 148, 16),
        (64, 4, 64, 148, 8),
        (16, 4, 64, 148, 1),
        (128, 4, 512, 148, 4),
        (128, 4, 512, 80, 2),
        (128, 4, 512, 32, 1),
        (0, 4, 64, 148, 1),
    ],
)
def test_split_policy(topk, heads, sq, sms, expected):
    assert sm100_blk64_auto_fp8_kv_splits(topk, heads, sq, num_sms=sms) == expected


@pytest.mark.cpu_only
def test_batch_parallelism_avoids_unnecessary_splits():
    assert sm100_blk64_auto_fp8_kv_splits(256, 4, 64, batch=40, num_sms=148) == 1
