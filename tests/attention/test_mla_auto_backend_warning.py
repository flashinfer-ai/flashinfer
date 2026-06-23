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


@patch("flashinfer.mla._core.get_compute_capability", return_value=(10, 0))
def test_auto_warns_once_on_blackwell(_cc, buf):
    assert len(_make(buf, "auto")) == 1
    assert len(_make(buf, "auto")) == 0  # one-time


@patch("flashinfer.mla._core.get_compute_capability", return_value=(10, 0))
def test_explicit_backend_does_not_warn(_cc, buf):
    assert _make(buf, "fa2") == []
    assert _make(buf, "cutlass") == []


@patch("flashinfer.mla._core.get_compute_capability", return_value=(9, 0))
def test_no_warn_on_hopper(_cc, buf):
    assert _make(buf, "auto") == []
