"""Fixture for setting ``torch.backends.cuda.matmul`` flags within one test."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cuda_matmul_flags():
    """Return a setter for ``torch.backends.cuda.matmul`` flags; restores them at teardown.

    Usage: ``cuda_matmul_flags(allow_tf32=False)``.

    ``monkeypatch.setattr`` cannot be used for these flags. ``cuBLASModule``
    serves them from C state through ``__getattr__``/``__setattr__``, so they are
    never in the instance ``__dict__``. Since pytest 9.2 monkeypatch restores
    such an attribute with ``delattr``, which ``cuBLASModule`` does not support,
    and teardown fails with ``AttributeError``.
    """
    matmul = torch.backends.cuda.matmul
    saved: dict[str, object] = {}

    def set_flags(**flags: object) -> None:
        for name, value in flags.items():
            saved.setdefault(name, getattr(matmul, name))
            setattr(matmul, name, value)

    yield set_flags
    for name, value in reversed(saved.items()):
        setattr(matmul, name, value)
