"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

import warnings

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper
from flashinfer.mla._batch_mla._auto_policy import _BatchMLAPagedAttentionAutoBackend


WARN_TAG = "not Blackwell-native"


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    monkeypatch.setattr(
        _BatchMLAPagedAttentionAutoBackend, "_blackwell_auto_fallback_warned", False
    )


@pytest.mark.parametrize("backend", ["auto", "fa2", "cutlass"])
def test_constructor_does_not_emit_backend_selection_warning(backend):
    workspace = torch.empty(1, dtype=torch.uint8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BatchMLAPagedAttentionWrapper(workspace, backend=backend)
        BatchMLAPagedAttentionWrapper(workspace, backend=backend)
    assert not [warning for warning in caught if WARN_TAG in str(warning.message)]


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
        "flashinfer.mla._batch_mla._auto_policy._get_compute_capability",
        lambda _device: capability,
    )

    with pytest.warns(UserWarning, match=WARN_TAG) as caught:
        _BatchMLAPagedAttentionAutoBackend._maybe_warn_blackwell_auto_fallback(
            torch.device("cuda"), "fa2"
        )

    message = str(caught[0].message)
    assert expected_backend in message
    assert unexpected_backend not in message
