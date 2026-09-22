# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Guard random selection, physical placement, and poisoned-row coverage."""

import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[2] / "benchmarks/sparse_mla_fixtures.py"
_spec = importlib.util.spec_from_file_location("sparse_mla_fixtures", _path)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_random_sparse_fixture(variable, dtype):
    args = dict(
        batch=3,
        heads=16,
        queries=3,
        dtype=dtype,
        swa_page=256,
        compressed_page=1,
        seed=2026,
        compressed_topk=128,
        swa_cache_tokens=512,
        compressed_cache_tokens=2048,
        variable_cache_lengths=True,
        variable_topk_lengths=variable,
        device="cpu",
    )
    first = _fixtures.flashmla_fixture(**args)
    second = _fixtures.flashmla_fixture(**args)
    for key in first:
        if isinstance(first[key], torch.Tensor):
            torch.testing.assert_close(
                first[key].float(), second[key].float(), rtol=0, atol=0, equal_nan=True
            )
    for cache_name, index_name, length_name in (
        ("swa", "si", "sl"),
        ("compressed", "ci", "cl"),
    ):
        cache = first[cache_name].float().reshape(-1, 512)
        indices = first[index_name].reshape(9, -1)
        lengths = first[length_name].reshape(9)
        tail = torch.arange(indices.shape[-1])[None, :] >= lengths[:, None]
        assert (indices[tail] == -1).all()
        used = torch.zeros(cache.shape[0], dtype=torch.bool)
        for row, length in zip(indices, lengths, strict=True):
            valid = row[:length][row[:length] >= 0].long()
            assert valid.unique().numel() == valid.numel()
            assert torch.isfinite(cache[valid]).all()
            used[valid] = True
        assert (~used).any()
        assert torch.isnan(cache[~used]).all()
    ci = first["ci"][0, 0]
    assert (ci[1:] - ci[:-1]).abs().max() > 128
    assert not torch.equal(first["ci"][:, 0], first["ci"][:, 1])
    assert torch.equal(first["combined"][:, :128].reshape_as(first["si"]), first["si"])
    assert torch.equal(first["combined_lengths"], (first["sl"] + first["cl"]).flatten())


def test_short_sources_preserve_fixed_swa_segment_and_trim_compressed_padding():
    fixture = _fixtures.flashmla_fixture(
        2,
        16,
        4,
        torch.bfloat16,
        256,
        1,
        2026,
        compressed_topk=128,
        swa_cache_tokens=8,
        compressed_cache_tokens=5,
        variable_cache_lengths=False,
        device="cpu",
    )
    assert (fixture["sl"] == 128).all()
    assert torch.equal(
        (fixture["si"] >= 0).sum(-1), torch.tensor([[5, 6, 7, 8]]).expand(2, 4)
    )
    assert (fixture["cl"] == 5).all()
    assert (fixture["combined_lengths"] == 133).all()
    assert (fixture["ci"][..., 5:] == -1).all()
