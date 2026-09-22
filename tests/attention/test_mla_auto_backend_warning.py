"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

import warnings
from unittest.mock import patch

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper


WARN_TAG = "not Blackwell-native"


def _fresh_state():
    BatchMLAPagedAttentionWrapper._blackwell_auto_fallback_warned = False


def _make(buf, backend):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BatchMLAPagedAttentionWrapper(buf, backend=backend)
        return [str(x.message) for x in w if WARN_TAG in str(x.message)]


@pytest.fixture
def buf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    _fresh_state()
    return torch.empty(8 * 1024 * 1024, dtype=torch.int8, device="cuda")


@patch(
    "flashinfer.mla._batch_mla._wrapper._get_compute_capability", return_value=(10, 0)
)
def test_auto_warns_once_on_blackwell(_cc, buf):
    assert len(_make(buf, "auto")) == 1
    assert len(_make(buf, "auto")) == 0  # one-time


@patch(
    "flashinfer.mla._batch_mla._wrapper._get_compute_capability", return_value=(10, 0)
)
def test_explicit_backend_does_not_warn(_cc, buf):
    assert _make(buf, "fa2") == []
    assert _make(buf, "cutlass") == []


@patch(
    "flashinfer.mla._batch_mla._wrapper._get_compute_capability", return_value=(9, 0)
)
def test_no_warn_on_hopper(_cc, buf):
    assert _make(buf, "auto") == []


@pytest.mark.parametrize(
    ("capability", "expected_backend", "unexpected_backend"),
    [
        ((10, 0), "backend='cutile'", "backend='cutlass'"),
        ((11, 0), "backend='cutlass'", "backend='cutile'"),
    ],
)
def test_auto_warning_recommends_an_architecture_supported_backend(
    monkeypatch, capability, expected_backend, unexpected_backend
):
    """The fallback warning must not recommend a backend that rejects the GPU."""
    monkeypatch.setattr(
        "flashinfer.mla._batch_mla._wrapper._get_compute_capability",
        lambda _device: capability,
    )
    _fresh_state()

    with pytest.warns(UserWarning, match=WARN_TAG) as caught:
        BatchMLAPagedAttentionWrapper._maybe_warn_blackwell_auto_fallback(
            torch.device("cuda"), "fa2"
        )

    message = str(caught[0].message)
    assert expected_backend in message
    assert unexpected_backend not in message
