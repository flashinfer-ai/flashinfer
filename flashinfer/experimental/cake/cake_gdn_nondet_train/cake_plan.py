"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The host plan of one call that the forward and the backward share: the one-wave piece table of the sequences
(:func:`pieces`), cached with the device's SM count per sequence-length table (:func:`host_plan`), and the
placeholder that keeps a saved list's arity where a call saves no tensor (:func:`placeholder`).

A long sequence's scan splits into pieces of at most ``L`` chunks: the smallest ``L`` whose first wave fits one CTA
per SM, ``sum_d max(P_d - 1, 1) * HV <= sm_count`` with ``P_d = ceil(n_d / L)`` pieces of sequence ``d`` (``n_d``
chunks).  The first wave runs the last piece and the ``P_d - 2`` middle pieces of every split sequence (a sequence of
one piece runs whole there), the second wave the other ``P_d - 1`` pieces, which fit too.  The table depends on the
chunk counts, ``HV`` and the SM count of the launching device alone: no occupancy query, no tensor contents and no
fixed virtual SM count, so a device with more SMs gets shorter pieces.  Whether a call runs the split or the serial
scan is each scan's priced choice; both read the same table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch

from . import cake_backend, cake_jit

__all__ = [
    "HostPlan",
    "Pieces",
    "chunk_counts",
    "host_plan",
    "piece_length",
    "pieces",
    "placeholder",
]

CHUNK: int = int(cake_jit.CHUNK)


def chunk_counts(seq_lens: Sequence[int]) -> tuple[int, ...]:
    """The chunk count of each sequence of ``seq_lens`` tokens (``ceil(n / CHUNK)``, 0 for an empty sequence)."""
    return tuple(-(-int(n) // CHUNK) for n in seq_lens)


def piece_length(chunks: Sequence[int], num_v_heads: int, sm_count: int) -> int | None:
    """The piece length ``L`` (chunks) of the one-wave split over sequences of ``chunks`` chunks: the smallest ``L``
    with ``sum_d max(ceil(n_d / L) - 1, 1) * num_v_heads <= sm_count``.  ``None`` where no ``L`` splits a sequence:
    more sequences x value heads than SMs, or every sequence one piece at the smallest fitting ``L``."""
    chunks = [int(n) for n in chunks]
    hv, sms = int(num_v_heads), int(sm_count)
    if not chunks or hv <= 0 or len(chunks) * hv > sms or max(chunks) < 2:
        return None

    def ctas(length: int) -> int:
        return sum(max(-(-n // length) - 1, 1) for n in chunks) * hv

    lo, hi = 1, max(chunks)  # ctas(max(chunks)) = N x HV fits
    while lo < hi:
        mid = (lo + hi) // 2
        if ctas(mid) <= sms:
            hi = mid
        else:
            lo = mid + 1
    return lo if max(chunks) > lo else None


@dataclass(frozen=True)
class Pieces:
    """The one-wave piece table of one call.  ``length`` is ``L`` (0: no split, every sequence one piece), ``counts``
    the pieces ``P_d`` per sequence, ``first_wave`` and ``second_wave`` the CTAs of the two waves at one CTA per
    (piece, value head)."""

    length: int
    counts: tuple[int, ...]
    first_wave: int
    second_wave: int

    @property
    def split(self) -> bool:
        return self.length > 0


@lru_cache(maxsize=1024)
def _pieces(chunks: tuple[int, ...], num_v_heads: int, sm_count: int) -> Pieces:
    length = piece_length(chunks, num_v_heads, sm_count)
    if length is None:
        ones = tuple(1 for _ in chunks)
        return Pieces(0, ones, len(chunks) * num_v_heads, 0)
    counts = tuple(max(-(-n // length), 1) for n in chunks)
    return Pieces(
        length,
        counts,
        sum(max(p - 1, 1) for p in counts) * num_v_heads,
        sum(p - 1 for p in counts) * num_v_heads,
    )


def pieces(seq_lens: Sequence[int], num_v_heads: int, sm_count: int) -> Pieces:
    """The one-wave piece table of sequences of ``seq_lens`` tokens with ``num_v_heads`` value heads on a device of
    ``sm_count`` SMs (cached per chunk-count table)."""
    return _pieces(chunk_counts(seq_lens), int(num_v_heads), int(sm_count))


@dataclass(frozen=True)
class HostPlan:
    """The host plan entry of one sequence-length table on one device: the chunk counts, the value heads, the
    device's architecture and SM count, and the piece table both scans read."""

    chunks: tuple[int, ...]
    num_v_heads: int
    arch: str
    sm_count: int
    pieces: Pieces


@lru_cache(maxsize=1024)
def host_plan(
    seq_lens: tuple[int, ...], num_v_heads: int, device_index: int
) -> HostPlan:
    """The host plan of sequences of ``seq_lens`` tokens (the host copy of ``cu_seqlens`` as lengths) with
    ``num_v_heads`` value heads on CUDA device ``device_index``, cached per (lengths, value heads, device)."""
    sm_count = int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    chunks = chunk_counts(seq_lens)
    return HostPlan(
        chunks,
        int(num_v_heads),
        cake_backend.device_arch(device_index),
        sm_count,
        _pieces(chunks, int(num_v_heads), sm_count),
    )


def placeholder(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """A tensor of ``shape``, ``dtype`` and ``device`` without storage of its own (one element, every stride 0): it
    stands in a saved list for an entry the call does not save, so the list keeps its arity.  Consumers decide from
    the plan, never by reading a placeholder."""
    shape = tuple(int(s) for s in shape)
    if not shape:
        return torch.empty((), dtype=dtype, device=device)
    return torch.empty(1, dtype=dtype, device=device).expand(shape)
