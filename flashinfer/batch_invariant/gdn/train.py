# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Boundary states of the stateless call: ``chunk_gated_delta_rule(..., batch_invariant=True, train_states=<buffer>)``.

The same launch as the stateless call, with every sequence's state at its 64-token boundaries stored in
``train_states``, a caller-owned fp32 buffer ``[slots, HV, 128, 128]`` with at least :func:`training_state_slots`
rows. They are the one term a backward cannot rebuild from the inputs of one chunk: with the state a chunk starts
from, the chunk's terms are a function of its own 64 rows; without it a backward re-runs the state scan. A stored state
is ``[HV, 128 (value), 128 (key)]``, one per v head. No pool is read or written and the rows are unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from ..support import unsupported
from .pool import _POOLS, _pinned, pool_key
from .spec import CHUNK, CLASS, HEAD_DIM


def training_state_slots(seq_lens: Sequence[int]) -> int:
    """Rows ``train_states`` needs for sequences of these lengths: one per 64-token boundary inside a sequence, and
    at least one."""
    return max(1, sum((int(n) - 1) // CHUNK for n in seq_lens))


def training_state_rows(seq_lens: Sequence[int]) -> List[List[int]]:
    """Per sequence, the ``train_states`` rows of its boundary states in order: row ``j`` holds the state after the
    sequence's first ``64 * (j + 1)`` tokens, which is the state its chunk ``j + 1`` starts from."""
    rows, used = [], 0
    for n in seq_lens:
        kept = (int(n) - 1) // CHUNK
        rows.append(list(range(used, used + kept)))
        used += kept
    return rows


def _checkpoint_rows(offs: Sequence[int]) -> Tuple[List[int], List[int]]:
    """The kernel's checkpoint entries, one per whole chunk of a sequence; -1 skips the one at a sequence's end."""
    rows: List[int] = []
    starts = [0]
    used = 0
    for a, b in zip(offs, offs[1:], strict=False):
        n = b - a
        kept = (n - 1) // CHUNK
        rows += list(range(used, used + kept)) + [-1] * (n // CHUNK - kept)
        used += kept
        starts.append(len(rows))
    return rows, starts


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
            f"train_states of {int(states.shape[0])} slots for a call that stores {need} boundary states",
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
    """The class's term for whole sequences from the zero state, their 64-token boundary states stored in ``states``."""
    dev = out.device
    N = len(offs) - 1
    ck_rows, ck_starts = _checkpoint_rows(offs)
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
    torch.autograd.graph.increment_version(states)
    return out
