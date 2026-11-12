"""CPU tests for the hybrid token-bucket mapper used by the unified MoE runners.

``map_to_hybrid_bucket`` sizes per-bucket workspaces and picks the autotune and
cuTile compile bucket, so it must agree with ``get_hybrid_num_tokens_buckets``
and never return a bucket above ``tune_max_num_tokens`` (#9413).
"""

import pytest

from flashinfer.fused_moe.runners import _cutile_bucket_compile_token_counts
from flashinfer.fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    make_hybrid_bucket_mapper,
    map_to_hybrid_bucket,
)

# Powers of two, ceilings inside each spacing phase, and phase boundaries.
_CEILINGS = (1, 2, 3, 5, 64, 65, 100, 200, 256, 257, 300, 1000, 2048, 2100, 3000)
_CEILINGS += (4096, 5000, 8192, 10000)


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_num_tokens", _CEILINGS)
def test_every_token_count_maps_to_a_generated_bucket(max_num_tokens):
    buckets = set(get_hybrid_num_tokens_buckets(max_num_tokens))
    mapper = make_hybrid_bucket_mapper(max_num_tokens)
    for x in range(1, max_num_tokens + 1):
        bucket = map_to_hybrid_bucket(x, max_num_tokens)
        assert x <= bucket <= max_num_tokens, (x, bucket)
        assert bucket in buckets, (x, bucket)
        assert mapper(x) == bucket


@pytest.mark.cpu_only
def test_small_non_power_of_two_ceiling_clamps_to_ceiling():
    assert get_hybrid_num_tokens_buckets(100)[-2:] == (64, 100)
    assert map_to_hybrid_bucket(64, 100) == 64
    for x in (65, 70, 99, 100):
        assert map_to_hybrid_bucket(x, 100) == 100


@pytest.mark.cpu_only
def test_cutile_compile_bucket_accepts_clamped_bucket():
    # The cuTile serving path compiles map_to_hybrid_bucket(num_tokens, ceiling);
    # it used to receive 128 here and reject it as above the ceiling.
    bucket = map_to_hybrid_bucket(70, 100)
    counts = _cutile_bucket_compile_token_counts(
        bucket,
        tune_max_num_tokens=100,
        num_experts=8,
        top_k=2,
        hidden_size=256,
        intermediate_size=512,
        block_size=64,
    )
    assert counts
    assert all(65 <= count <= 100 for count in counts)
