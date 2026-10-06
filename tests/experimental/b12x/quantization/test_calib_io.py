"""Safetensors-only calibration IO — pickle must be rejected everywhere."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from b12x.quantization.mxfp6.calib_io import load_hidden_states, save_hidden_states


def test_hidden_states_roundtrip(tmp_path: Path) -> None:
    x = torch.randn(64, 128, dtype=torch.bfloat16)
    p = tmp_path / "l20_hidden.safetensors"
    save_hidden_states(x, p, metadata={"layer": "20"})
    back = load_hidden_states(p)
    assert back.dtype == torch.bfloat16
    assert torch.equal(back, x)


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".pkl", ".bin"])
def test_save_rejects_pickle_paths(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(ValueError, match="safetensors"):
        save_hidden_states(torch.zeros(2, 32), tmp_path / f"x{suffix}")


@pytest.mark.parametrize("suffix", [".pt", ".pth"])
def test_load_rejects_pickle_paths(tmp_path: Path, suffix: str) -> None:
    bad = tmp_path / f"x{suffix}"
    bad.write_bytes(b"junk")
    with pytest.raises(ValueError, match="safetensors"):
        load_hidden_states(bad)


def test_save_requires_safetensors_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safetensors"):
        save_hidden_states(torch.zeros(2, 32), tmp_path / "x.tensor")
