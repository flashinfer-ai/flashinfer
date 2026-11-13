"""CPU selection tests for checkpointing SSU's PDL device gate."""

import importlib

import pytest
import torch


def _module():
    """Return the checkpointing SSU Python module."""
    return importlib.import_module("flashinfer.mamba.checkpointing_ssu")


def _inputs():
    """Build minimal CPU inputs for tactic-selection tests."""
    state = torch.empty(1, dtype=torch.bfloat16)
    x = torch.empty(1, dtype=torch.bfloat16)
    inputs = [None] * 22
    inputs[0] = state
    inputs[1] = x
    for index in (19, 20, 21):
        inputs[index] = torch.empty(1)
    return inputs


@pytest.mark.cpu_only
def test_two_kernel_requires_pdl_capable_device(monkeypatch):
    """Reject two-kernel selection when the device lacks PDL."""
    module = _module()
    monkeypatch.setattr(module, "device_support_pdl", lambda device: False)

    assert not module.CheckpointingSSURunner._two_kernel_supported(_inputs())


@pytest.mark.cpu_only
def test_two_kernel_available_with_pdl_and_supported_layout(monkeypatch):
    """Allow two-kernel selection for PDL devices with supported layouts."""
    module = _module()
    monkeypatch.setattr(module, "device_support_pdl", lambda device: True)

    assert module.CheckpointingSSURunner._two_kernel_supported(_inputs())
