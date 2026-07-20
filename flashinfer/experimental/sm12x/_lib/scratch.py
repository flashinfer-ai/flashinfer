# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/cute/scratch.py @ 9f2eb830 (2026-05-27) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Shared caller-owned scratch plan helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class ScratchBufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @property
    def nbytes(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= int(dim)
        return numel * torch.empty((), dtype=self.dtype).element_size()


def scratch_buffer_spec(
    name: str,
    *,
    nbytes: int,
    device: torch.device,
) -> ScratchBufferSpec:
    return ScratchBufferSpec(
        name=name,
        shape=(max(int(nbytes), 1),),
        dtype=torch.uint8,
        device=device,
    )


def scratch_tensor(
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    specs: tuple[ScratchBufferSpec, ...],
    *,
    owner: str,
) -> torch.Tensor:
    if len(specs) != 1:
        raise RuntimeError(
            f"{owner} scratch plans currently expect exactly one scratch buffer"
        )
    spec = specs[0]
    if isinstance(scratch, torch.Tensor):
        tensor = scratch
    elif isinstance(scratch, Mapping):
        if spec.name not in scratch:
            raise KeyError(f"scratch mapping is missing {spec.name!r}")
        tensor = scratch[spec.name]
    else:
        if len(scratch) != 1:
            raise ValueError(
                f"scratch sequence must contain exactly one tensor, got {len(scratch)}"
            )
        tensor = scratch[0]
    if tensor.dtype != spec.dtype:
        raise TypeError(
            f"{spec.name} scratch must have dtype {spec.dtype}, got {tensor.dtype}"
        )
    if tensor.device != spec.device:
        raise ValueError(
            f"{spec.name} scratch device {tensor.device} does not match {spec.device}"
        )
    if int(tensor.numel()) < int(spec.shape[0]):
        raise ValueError(
            f"{spec.name} scratch has {int(tensor.numel())} bytes, requires {int(spec.shape[0])}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{spec.name} scratch must be contiguous")
    return tensor.reshape(-1)


__all__ = [
    "ScratchBufferSpec",
    "scratch_buffer_spec",
    "scratch_tensor",
]
