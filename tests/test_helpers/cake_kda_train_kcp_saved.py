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

The bytes of a KCP saved set (``cake_kda_train.cake_kcp.KcpSaved``) that its forward writes, for bitwise compares of
two saved sets.  ``S_in``, ``c1_recv`` and ``M_seg`` are written whole.  The carry plane (``P > 1``) is a fresh
uninitialized tensor of ``n_frag (P - 1)`` slots (``+ max(n_in, n_out)`` extra slots under the fused walks) of ``H``
[128, 128] tiles each, and the forward's carry merges write only

* the ``P - 1`` slots ``f.slot (P - 1) + i`` of every non-empty fragment ``f`` (``carry_start`` / ``carry_in``) and
* the extra slot ``n_frag (P - 1) + in_slot[f.slot]`` of every crossing-in fragment (the ``S_in`` copy of the fused
  walks).

The other slots keep the caching allocator's bytes, which depend on the allocation history: the extra slots
``n_in <= j < n_out`` (the backward publishes its ``dS_out`` copies there in the workspace pool, never in the saved
plane) and the slots of zero-length fragments.  The tape replay of the saved-mode backward reads exactly the written
slots; :func:`carry_slots` checks that against the plan's segment tables, and :func:`poison_unwritten_carry` lets a
test require bitwise gradients with NaN in the other slots.
"""

from __future__ import annotations

import torch

TILE = 128 * 128  # elements of one [128, 128] state tile of one head


def carry_slots(saved) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(written, unwritten)`` slot numbers of ``saved.carry`` (both empty without a carry plane)."""
    plane = saved.carry
    if plane is None:
        return (), ()
    ctx, H, P = saved.ctx, int(saved.H), int(saved.P)
    hp = ctx.heads(H)
    base = int(ctx.n_frag) * (P - 1)
    n_slot, rem = divmod(plane.numel(), H * TILE)
    assert int(hp.P) == P > 1 and rem == 0, (P, int(hp.P), plane.numel(), H)
    # the extra slots: max(n_in, n_out) under the fused walks, none under fused_walks=0
    extra = n_slot - base
    assert extra in (0, max(int(ctx.n_in), int(ctx.n_out))), (n_slot, base)
    live = [f for f in ctx.fragments if f.length > 0]
    written = {f.slot * (P - 1) + i for f in live for i in range(P - 1)}
    if extra:
        written |= {base + int(ctx.in_slot[f.slot]) for f in live if f.crossing_in}
    # the tape replay's initial states: the carry plane slot of every segment of a non-empty fragment (-1 = zero state)
    index = hp.h0_index_fused if extra else hp.carry_index
    read = {int(index[f.slot * P + p]) for f in live for p in range(P)} - {-1}
    assert written == read, (sorted(written), sorted(read))
    return tuple(sorted(written)), tuple(c for c in range(n_slot) if c not in written)


def _rows(saved, slots) -> tuple[torch.Tensor, torch.Tensor]:
    plane = saved.carry
    rows = plane.view(plane.numel() // (int(saved.H) * TILE), -1)
    return rows, torch.tensor(slots, dtype=torch.long, device=plane.device)


def written_carry(saved) -> torch.Tensor | None:
    """The written slots of ``saved.carry`` as ``[n_written, H * TILE]`` (``None`` without a carry plane)."""
    if saved.carry is None:
        return None
    rows, idx = _rows(saved, carry_slots(saved)[0])
    return rows.index_select(0, idx)


def written_tensors(saved) -> tuple[torch.Tensor, ...]:
    """``saved.tensors()`` with the carry plane cut to its written slots: what two saved sets of the same inputs
    hold bitwise equal."""
    return tuple(
        written_carry(saved) if t is saved.carry else t for t in saved.tensors()
    )


def poison_unwritten_carry(saved) -> int:
    """Fill the unwritten slots of ``saved.carry`` with NaN in place; returns their bytes."""
    if saved.carry is None:
        return 0
    unwritten = carry_slots(saved)[1]
    if unwritten:
        rows, idx = _rows(saved, unwritten)
        rows[idx] = float("nan")
    return len(unwritten) * int(saved.H) * TILE * saved.carry.element_size()
