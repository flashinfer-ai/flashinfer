"""Require CUDA at the pytest boundary of a declared GPU CI lane."""

import os

import pytest


def pytest_sessionstart():
    """Fail a GPU lane before unavailable CUDA can turn its tests into skips."""
    lane = os.environ.get("CI_GPU_NAME")
    if not lane:
        return

    import torch

    if not torch.cuda.is_available():
        raise pytest.UsageError(
            f"GPU CI lane {lane!r} requires an available CUDA device; "
            "check the driver/runtime and CUDA_VISIBLE_DEVICES"
        )
