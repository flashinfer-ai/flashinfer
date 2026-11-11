"""CPU selection and JIT-config tests for checkpointing SSU's PDL gates."""

import importlib

import pytest
import torch


def _module():
    return importlib.import_module("flashinfer.mamba.checkpointing_ssu")


def _inputs():
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
    module = _module()
    monkeypatch.setattr(module, "device_support_pdl", lambda device: False)

    assert not module.CheckpointingSSURunner._two_kernel_supported(_inputs())


@pytest.mark.cpu_only
def test_two_kernel_available_with_pdl_and_supported_layout(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "device_support_pdl", lambda device: True)

    assert module.CheckpointingSSURunner._two_kernel_supported(_inputs())


@pytest.mark.cpu_only
def test_monolith_jit_config_disables_two_kernel(tmp_path, monkeypatch):
    jit = importlib.import_module("flashinfer.jit.mamba.checkpointing_ssu")
    monkeypatch.setattr(jit.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    monkeypatch.setattr(jit, "_arch_flags", lambda: [])

    spec = jit.gen_checkpointing_ssu_module(
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.int32,
        None,
        64,
        128,
        4,
        16,
        16,
        1,
        enable_two_kernel=False,
    )

    assert spec.name.endswith("_mono")
    config = (tmp_path / spec.name / "checkpointing_ssu_config.inc").read_text()
    assert "constexpr bool ENABLE_TWO_KERNEL = false;" in config
