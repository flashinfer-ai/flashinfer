"""Safetensors-only IO for calibration hidden states.

b12x policy: NO pickle artifacts, ever. ``torch.save`` / ``.pt`` paths are
rejected outright. Hidden-state captures are single-tensor safetensors files
with the tensor under the key ``"hidden_states"``.
"""
from __future__ import annotations

import pathlib

import torch

_PICKLE_SUFFIXES = (".pt", ".pth", ".pkl", ".pickle", ".bin")
HIDDEN_STATES_KEY = "hidden_states"


def _require_safetensors_path(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    if p.suffix.lower() in _PICKLE_SUFFIXES:
        raise ValueError(
            f"refusing pickle path {p} (suffix {p.suffix}). b12x artifacts are "
            "safetensors-only; use a *.safetensors path."
        )
    if p.suffix.lower() != ".safetensors":
        raise ValueError(
            f"expected a *.safetensors path, got {p} — e.g. "
            f"{p.with_suffix('.safetensors')}"
        )
    return p


def save_hidden_states(
    x: torch.Tensor,
    path: str | pathlib.Path,
    *,
    metadata: dict[str, str] | None = None,
) -> pathlib.Path:
    """Write a ``(tokens, K)`` capture to a single-tensor safetensors file."""
    from safetensors.torch import save_file

    p = _require_safetensors_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {HIDDEN_STATES_KEY: x.detach().contiguous().cpu()},
        str(p),
        metadata={k: str(v) for k, v in (metadata or {}).items()} or None,
    )
    return p


def load_hidden_states(
    path: str | pathlib.Path,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load a hidden-state capture written by :func:`save_hidden_states`.

    Falls back to the first tensor in the file if ``hidden_states`` is absent
    (hand-rolled captures), still safetensors-only.
    """
    from safetensors import safe_open

    p = _require_safetensors_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"hidden-state capture not found: {p}")
    with safe_open(str(p), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        key = HIDDEN_STATES_KEY if HIDDEN_STATES_KEY in keys else keys[0]
        x = f.get_tensor(key)
    return x.to(device)
