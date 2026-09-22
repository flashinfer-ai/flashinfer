# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.experimental.prims_ts_sparse_mla.policy import (
    _SparseMlaTuning,
    select_sparse_mla_profile,
    validate_selection,
)


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize(
    "rows,heads,capacity,query_length,sm_count",
    [
        (1, 8, 256, 1, 152),
        (4, 24, 384, 1, 132),
        (16, 128, 2176, 4, 152),
        (32, 48, 640, 8, 152),
        (64, 96, 1152, 4, 132),
        (128, 192, 8192, 8, 152),
        (1024, 128, 2176, 8, 152),
        (8192, 64, 2176, 8192, 152),
    ],
)
def test_policy_is_host_only_and_eligible(
    monkeypatch, dtype, rows, heads, capacity, query_length, sm_count
):
    def forbidden(*args, **kwargs):
        raise AssertionError("policy must not allocate or query a GPU")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    choice = select_sparse_mla_profile(
        rows=rows,
        heads=heads,
        query_length=query_length,
        capacity=capacity,
        dtype=dtype,
        sm_count=sm_count,
    )
    validate_selection(choice, dtype=dtype)
    assert choice.reason
    assert 1 <= choice.splits <= 128
    if dtype == "fp8":
        assert not choice.tuning.defer_max_update


def test_policy_rejects_unsupported_topology_before_compilation():
    with pytest.raises(ValueError, match="does not support split-KV"):
        select_sparse_mla_profile(
            rows=4,
            heads=16,
            query_length=1,
            capacity=512,
            dtype="fp8",
            sm_count=152,
            forced=_SparseMlaTuning(
                family="swap", tile_size_q=16, split_kv=2, scheduler="static"
            ),
        )
    with pytest.raises(ValueError, match="shared-memory"):
        select_sparse_mla_profile(
            rows=4,
            heads=64,
            query_length=1,
            capacity=512,
            dtype="bf16",
            sm_count=152,
            forced=_SparseMlaTuning(
                family="keep", tile_size_q=64, reuse_kv=True, reuse_kv_stages=20
            ),
        )
