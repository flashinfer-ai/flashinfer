# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""State series of the stateless call: ``chunk_gated_delta_rule(..., batch_invariant=True, train_states=<buffer>)``.

The same launch as the stateless call, with the state every 64-token chunk of every sequence starts from stored in
``train_states``, a caller-owned fp32 buffer ``[slots, HV, 128, 128]`` with at least :func:`training_state_slots`
rows. It is the one term a backward cannot rebuild from the inputs of one chunk without re-running the state scan.
The row layout is the one cuDNN's GDN backward takes as ``state_checkpoints`` at a 64-token cadence: per sequence,
in order, the zero state its first chunk starts from, then the state after each 64 tokens short of its end. A state
is ``[HV, 128 (value), 128 (key)]``, one per v head. No pool is read or written and the rows are unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from ..support import unsupported
from .pool import _POOLS, _pinned, pool_key
from .spec import CHUNK, CLASS, HEAD_DIM


def training_state_slots(seq_lens: Sequence[int]) -> int:
    """Rows ``train_states`` needs for sequences of these lengths: one per chunk, ``(n - 1) // 64 + 1`` for a
    sequence of ``n`` tokens."""
    return max(1, sum((int(n) - 1) // CHUNK + 1 for n in seq_lens))


def training_state_rows(seq_lens: Sequence[int]) -> List[List[int]]:
    """Per sequence, its ``train_states`` rows in order: row ``j`` holds the state its chunk ``j`` starts from, the
    zero state for ``j = 0`` and the state after its first ``64 * j`` tokens otherwise."""
    rows, used = [], 0
    for n in seq_lens:
        chunks = (int(n) - 1) // CHUNK + 1
        rows.append(list(range(used, used + chunks)))
        used += chunks
    return rows


def _checkpoint_rows(
    offs: Sequence[int],
) -> Tuple[List[int], List[int], List[int]]:
    """The kernel's checkpoint entries, one per whole chunk of a sequence (-1 skips the one at a sequence's end),
    their per-sequence starts, and the rows of the zero states."""
    rows: List[int] = []
    starts = [0]
    lead: List[int] = []
    used = 0
    for a, b in zip(offs, offs[1:], strict=False):
        n = b - a
        kept = (n - 1) // CHUNK
        lead.append(used)
        rows += list(range(used + 1, used + 1 + kept)) + [-1] * (n // CHUNK - kept)
        used += kept + 1
        starts.append(len(rows))
    return rows, starts, lead


def check_train_states(
    states: torch.Tensor, HV: int, device: torch.device, offs: Sequence[int]
) -> None:
    need = training_state_slots([b - a for a, b in zip(offs, offs[1:], strict=False)])
    if (
        not isinstance(states, torch.Tensor)
        or states.dtype != torch.float32
        or states.ndim != 4
        or tuple(states.shape[1:]) != (HV, HEAD_DIM, HEAD_DIM)
        or not states.is_contiguous()
    ):
        raise unsupported(
            CLASS,
            f"train_states {getattr(states, 'shape', type(states).__name__)} {getattr(states, 'dtype', '')}",
            f"a contiguous fp32 [slots, {HV}, {HEAD_DIM}, {HEAD_DIM}] buffer",
        )
    if states.device != device:
        raise unsupported(CLASS, "train_states on another device", "the inputs' device")
    if states.requires_grad:
        raise unsupported(CLASS, "train_states that requires grad", "a plain buffer")
    if states.is_inference() and not torch.is_inference_mode_enabled():
        raise unsupported(
            CLASS,
            "train_states allocated under inference_mode in a call outside it",
            "a buffer allocated in the mode of the call",
        )
    if pool_key(states) in _POOLS:
        raise unsupported(
            CLASS,
            "train_states that is a state pool of the class",
            "a buffer of its own",
        )
    if int(states.shape[0]) < need:
        raise unsupported(
            CLASS,
            f"train_states of {int(states.shape[0])} slots for a call that stores {need} states",
            "training_state_slots(seq_lens) slots",
        )


def zero_state_prefill_with_states(
    rows: Tuple[torch.Tensor, ...],
    cu: Optional[torch.Tensor],
    offs: List[int],
    states: torch.Tensor,
    scale: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """The class's term for whole sequences from the zero state, the state each chunk starts from stored in ``states``."""
    dev = out.device
    N = len(offs) - 1
    ck_rows, ck_starts, lead = _checkpoint_rows(offs)
    meta = torch.tensor(
        [-1] * N + (list(offs) if cu is None else []) + ck_starts + ck_rows,
        dtype=torch.int32,
        device=dev,
    )
    skip = meta[:N]
    pos = N
    if cu is None:
        cu = meta[pos : pos + N + 1]
        pos += N + 1
    checkpoints = None
    if any(r >= 0 for r in ck_rows):
        checkpoints = (meta[pos + N + 1 :], meta[pos : pos + N + 1])
    _pinned(
        rows,
        cu,
        list(offs),
        states,
        skip,
        skip,
        scale,
        output=out,
        checkpoints=checkpoints,
    )
    states.index_fill_(0, torch.tensor(lead, device=dev), 0.0)
    return out
