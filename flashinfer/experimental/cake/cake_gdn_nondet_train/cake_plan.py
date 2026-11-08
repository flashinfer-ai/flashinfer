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
(:func:`pieces`), cached with the device's SM count per sequence-length table (:func:`host_plan`), the frozen tape
level of each architecture (:data:`TAPE_LEVELS`, :func:`tape_state`), the backward forms that a measurement fixes for a
device and head shape (:data:`ADJOINT_FORMS`, :func:`adjoint_form`), the shape class of the forward's one-wave split
(:data:`FORWARD_SPLIT_MIN_PIECE`, :func:`forward_split_length`), the M-share flag of a piece table
(:data:`M_SHARE_COSTS`, :func:`m_share`), and the placeholder that keeps a saved list's arity where a call saves no
tensor (:func:`placeholder`).

A long sequence's scan splits into pieces of at most ``L`` chunks: the smallest ``L`` whose first wave fits one CTA
per SM, ``sum_d max(P_d - 1, 1) * HV <= sm_count`` with ``P_d = ceil(n_d / L)`` pieces of sequence ``d`` (``n_d``
chunks).  The first wave runs the last piece and the ``P_d - 2`` middle pieces of every split sequence (a sequence of
one piece runs whole there), the second wave the other ``P_d - 1`` pieces, which fit too.  The table depends on the
chunk counts, ``HV`` and the SM count of the launching device alone: no occupancy query, no tensor contents and no
fixed virtual SM count, so a device with more SMs gets shorter pieces.  Whether a call runs the split or the serial
scan is each scan's priced choice; both read the same table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

import torch

from . import cake_backend, cake_jit

__all__ = [
    "ADJOINT_FORMS",
    "FORWARD_SPLIT_MIN_PIECE",
    "LEVEL_STATE",
    "STATE_NAMES",
    "TAPE_LEVELS",
    "HostPlan",
    "Pieces",
    "chunk_counts",
    "forward_split_length",
    "host_plan",
    "piece_length",
    "pieces",
    "adjoint_form",
    "M_SHARE_COSTS",
    "M_SHARE_MIN_SAVING",
    "m_share",
    "m_share_saving",
    "placeholder",
    "tape_state",
]

CHUNK: int = int(cake_jit.CHUNK)

# The forward state a call can keep for the backward, in the saved list's order.
STATE_NAMES: tuple[str, ...] = ("w", "u", "h", "v_new")
# The state each tape level keeps.  ``T_full`` keeps the WY factor ``w`` and the scan's ``h`` / ``v_new``, everything
# the backward reads: the WY output ``u`` is read only to rebuild ``h`` / ``v_new``, so its bytes are freed when the
# forward returns.
LEVEL_STATE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"T_full": ("w", "h", "v_new")}
)
# The tape level of each architecture, frozen: one level per architecture for every row, so the backward of a call and
# its accuracy record read the same tape.
TAPE_LEVELS: Mapping[str, str] = MappingProxyType(
    {"sm_100a": "T_full", "sm_103a": "T_full", "sm_107a": "T_full"}
)

# The backward form of a call where a measurement on the device fixes it instead of the split's pricing, by
# (architecture, SM count, H, HV).  ``"nosplit"``: the planner's tables (no one-wave split of the adjoint walk; a group
# of three keeps its pre-scan R-form table) with the pre-scans' warm-up cut, which skips the steps whose decay has
# fallen below the threshold: the split's price does not depend on the gates, the planner's time does.  An entry is
# the form whose worst time relative to the best FROST arm, over the FI-bench and log-spaced gates, is the lower one:
# sm_100a on GB200 (152 SMs) at H 1 / HV 3, measured at the pair rows (three sequences of about 11k tokens).  From
# H 2 / HV 6 on the split stays: its worst ratio is the lower one there, set by the long H 2 / HV 6 packs.
ADJOINT_FORMS: Mapping[tuple[str, int, int, int], str] = MappingProxyType(
    {("sm_100a", 152, 1, 3): "nosplit"}
)


def adjoint_form(
    arch: str, sm_count: int, num_heads: int, num_v_heads: int
) -> str | None:
    """The form :data:`ADJOINT_FORMS` fixes for a call of ``num_heads`` / ``num_v_heads`` heads on a device of
    ``arch`` with ``sm_count`` SMs, ``None`` where the pricing decides."""
    return ADJOINT_FORMS.get((arch, int(sm_count), int(num_heads), int(num_v_heads)))


# M-share: where a call's forward runs the split forward and its backward the one-wave split of the adjoint walk on the
# same piece table, the taped forward keeps its segment state, whose M half holds each middle piece's map as the forward
# applies it (``x -> M x + B``), and the backward's first wave walks the middle pieces from zero for their start adjoint
# alone (the dvl32 walk with OUT, no tapes, no seed) beside the last pieces' walks, instead of the summary that walks the
# adjoint's map from the identity as well; the stitch then composes the end adjoints with ``M^T``, the adjoint of the
# forward's map (its ``transpose_m`` build).  The first wave's costs, by (architecture, SM count), measured on the device
# at the gating rows: the step (us) of the last pieces' taped walk beside the middle pieces (``walk``), of the summary
# (``summary``) and of the walk from zero (``gonly``), and the time (us) each first wave takes beyond its longer launch
# (``corun_summary``, ``corun_gonly``).  sm_100a on B200 (148 SMs); a device without an entry keeps the summary.
M_SHARE_COSTS: Mapping[tuple[str, int], Mapping[str, float]] = MappingProxyType(
    {
        ("sm_100a", 148): MappingProxyType(
            {
                "walk": 2.236,
                "summary": 2.679,
                "gonly": 1.930,
                "corun_summary": 2.3,
                "corun_gonly": 17.1,
            }
        ),
    }
)
# the smallest priced saving (us) of the backward's first wave that takes M-share
M_SHARE_MIN_SAVING = 15.0


def _wave1_pieces(chunks: Sequence[int], length: int) -> tuple[int, int]:
    """The longest last piece (or one-piece sequence) and the longest middle piece, in chunks, of sequences of ``chunks``
    chunks cut into ``ceil(n / length)`` pieces balanced to within one chunk (piece ``s`` of ``P`` ends at chunk
    ``(s + 1) n // P``)."""
    last = middle = 0
    for n in (int(c) for c in chunks):
        count = max(-(-n // length), 1)
        last = max(last, n - ((count - 1) * n) // count)
        for piece in range(1, count - 1):
            middle = max(middle, ((piece + 1) * n) // count - (piece * n) // count)
    return last, middle


def m_share_saving(
    chunks: Sequence[int], num_v_heads: int, arch: str, sm_count: int
) -> float | None:
    """The priced saving (us) of M-share in the backward's first wave over sequences of ``chunks`` chunks with
    ``num_v_heads`` value heads on a device of ``arch`` with ``sm_count`` SMs (:data:`M_SHARE_COSTS`): the first wave
    with the summary, ``max(last walk, middle summary) + corun_summary``, less the first wave with the walks from zero,
    ``max(last walk, middle gonly) + corun_gonly``.  ``None`` without a cost entry, without a split or without a middle
    piece."""
    costs = M_SHARE_COSTS.get((arch, int(sm_count)))
    if costs is None:
        return None
    length = piece_length(chunks, num_v_heads, sm_count)
    if length is None:
        return None
    last, middle = _wave1_pieces(chunks, length)
    if middle == 0:
        return None
    walk = last * costs["walk"]
    with_summary = max(walk, middle * costs["summary"]) + costs["corun_summary"]
    with_gonly = max(walk, middle * costs["gonly"]) + costs["corun_gonly"]
    return with_summary - with_gonly


def m_share(chunks: Sequence[int], num_v_heads: int, arch: str, sm_count: int) -> bool:
    """Whether a call over sequences of ``chunks`` chunks takes M-share (:data:`M_SHARE_COSTS`): its priced saving
    reaches :data:`M_SHARE_MIN_SAVING`.  The flag of the piece table: the forward keeps the maps and the backward reads
    them only where both run their split on that table (the chain checks)."""
    saving = m_share_saving(chunks, num_v_heads, arch, sm_count)
    return saving is not None and saving >= M_SHARE_MIN_SAVING


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


# The shape class of the forward's one-wave split, per architecture: the shortest piece (chunks) of the host piece
# table (:func:`pieces`) on which a call's forward runs the split forward instead of the prep forward's split scan.
# The split's fixed costs (the front, the middle pieces' summary, the stitch and a second wave) pay back on long pieces
# only.  Measured on B200 (148 SMs) and GB300 (152; fwd+bwd and checkpoint, the larger ratio to the best FROST arm
# over the FI-bench and log-spaced gates), the split lowers that ratio at the long rows (two to six sequences of 11k..131k tokens
# at HV 6: pieces of 132..158 chunks) and at the pair row of HV 24 (three sequences of about 11k tokens: 63 chunks) and
# raises it at the pair rows of HV 12 and 6 (38 and 20 chunks; HV 3 has 11).  The class reads the host piece table
# alone (the sequence lengths, HV and the SM count), never tensor contents.
FORWARD_SPLIT_MIN_PIECE: Mapping[str, int] = MappingProxyType(
    {"sm_100a": 48, "sm_103a": 48}
)


def forward_split_length(arch: str, table: Pieces) -> int | None:
    """The piece length of the forward's one-wave split of a call on ``arch`` with the host piece table ``table``:
    ``table.length`` where the table splits and its pieces reach the architecture's :data:`FORWARD_SPLIT_MIN_PIECE`,
    ``None`` elsewhere (the prep forward's split scan)."""
    if not table.split or table.length < FORWARD_SPLIT_MIN_PIECE.get(arch, 1 << 30):
        return None
    return table.length


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


def tape_state(arch: str, save_state: bool) -> tuple[str, ...]:
    """The state tensors (of :data:`STATE_NAMES`) the saved list of a call on ``arch`` carries: those of the
    architecture's tape level (:data:`TAPE_LEVELS`) under ``save_state``, none without it (the backward rebuilds the
    state).  The others stand in the list as placeholders (:func:`placeholder`)."""
    return LEVEL_STATE[TAPE_LEVELS[arch]] if save_state else ()


# the one-element tensor every placeholder of a (device, dtype) expands, made at its first placeholder
_PLACEHOLDER_BASES: dict[tuple[torch.device | str, torch.dtype], torch.Tensor] = {}


def placeholder(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """A tensor of ``shape``, ``dtype`` and ``device`` without storage of its own (one element, every stride 0): it
    stands in a saved list for an entry the call does not save, so the list keeps its arity.  Consumers decide from
    the plan, never by reading a placeholder.  Every placeholder of a (device, dtype) is a view of the same element,
    made once, so a call allocates nothing for it (the host, not the GPU, sets the forward's time at short rows)."""
    base = _PLACEHOLDER_BASES.get((device, dtype))
    if base is None:
        base = _PLACEHOLDER_BASES[(device, dtype)] = torch.empty(
            1, dtype=dtype, device=device
        )
    shape = tuple(int(s) for s in shape)
    return base.expand(shape) if shape else base.view(())
