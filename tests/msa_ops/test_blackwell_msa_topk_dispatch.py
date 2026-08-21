"""CPU-only dispatch checks for Blackwell MSA top-k selection."""

import importlib

import pytest
import torch


_topk_module = importlib.import_module("flashinfer.msa_ops.sparse_topk_select")


def test_sm100_sm103_reject_per_token_bounds_before_backend_dispatch(monkeypatch):
    monkeypatch.setattr(_topk_module, "is_blackwell_msa_device", lambda _device: True)

    def backend_must_not_run(*_args, **_kwargs):
        raise AssertionError("backend must not be called")

    monkeypatch.setattr(
        _topk_module, "blackwell_msa_topk_select", backend_must_not_run
    )

    max_score = torch.empty((1, 16, 2), dtype=torch.float32)
    num_valid_pages = torch.ones(2, dtype=torch.int32)
    with pytest.raises(NotImplementedError, match=r"SM120/SM121.*scalar"):
        _topk_module.msa_topk_select(
            max_score, 16, num_valid_pages=num_valid_pages
        )


def test_sm100_sm103_scalar_bound_still_reaches_backend(monkeypatch):
    monkeypatch.setattr(_topk_module, "is_blackwell_msa_device", lambda _device: True)
    sentinel = torch.empty((2, 1, 16), dtype=torch.int32)
    received = {}

    def backend(max_score, topk, **kwargs):
        received.update(max_score=max_score, topk=topk, **kwargs)
        return sentinel

    monkeypatch.setattr(_topk_module, "blackwell_msa_topk_select", backend)

    max_score = torch.empty((1, 16, 2), dtype=torch.float32)
    result = _topk_module.msa_topk_select(max_score, 16, num_valid_pages=12)

    assert result is sentinel
    assert received["max_score"] is max_score
    assert received["topk"] == 16
    assert received["num_valid_pages"] == 12
