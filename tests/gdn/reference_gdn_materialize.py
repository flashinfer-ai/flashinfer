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

---------------------------------------------------------------------------
fp32 reference for GDN prefix-cache state materialization.

Materialization answers: *what was this request's SSM state at some chosen
token in the recent past?*  Under the u-cache scheme the state is not written
every step; instead a checkpoint ``S0`` sits in the state pool and the
per-token update ingredients (L2-normalized k, delta-rule correction u,
cumulative log-decay g) accumulate in a small ring.  Replaying ``C`` ring
entries onto the checkpoint reconstructs the state after those C tokens:

    S_C = exp(G_C) * S0 + sum_{j<C} exp(G_C - g_j) * u_j (x) k_j,   G_C = g_{C-1}

which is exactly step 1 of ``tests/gdn/test_decode_ucache.py::_ref_fp32`` with
``P`` replaced by a caller-chosen ``C``.  The two differences from the decode
kernel's fold are that the replay is *partial* (C <= hist_len rather than the
whole window) and *non-destructive* (result goes to a different pool slot, so
the live request keeps decoding against its own checkpoint and ring).

Conventions, all inherited from the decode kernel and its oracle:

- ``g_cache`` holds cumulative log-decay measured from the window base, so the
  endpoint decay for a C-entry prefix is the stored g at logical row ``C-1``.
  No offset correction is needed as long as the replay starts at the window
  base, which materialization always does -- the source *is* the checkpoint.
- Logical row j lives at physical ring row ``(ring_start + j) % ring_slots``.
- ``k_cache`` carries H heads while ``u_cache``/``g_cache`` carry HV; value
  head ``hv`` reads k head ``hv // (HV // H)``.
- All math in fp32; a single rounding to the state dtype on store, matching
  the kernel's f32-accumulate / one-rounding fold epilogue.
"""

from typing import List, Optional, Sequence

import torch

__all__ = ["materialize_ref", "active_request_indices_from"]


def active_request_indices_from(
    replay_prefix_len: torch.Tensor,
) -> torch.Tensor:
    """Build the compacted active list: live request indices first, then -1.

    Mirrors ``_active_request_indices`` in flashinfer's mamba materialize
    tests.  A request is live when its replay prefix length is non-negative.
    Order among the live entries is arbitrary; only "live ones come first"
    matters.  NOTE the two consumers differ on malformed lists: THIS reference
    hard-stops at the first -1 (like the mamba kernel), while the GDN kernel
    SKIPS -1 entries and keeps walking -- identical results for any well-formed
    live-prefix-then-sentinels list, divergent if a live index hides after a
    -1.
    """
    indices = torch.full_like(replay_prefix_len, -1)
    active = torch.nonzero(replay_prefix_len >= 0, as_tuple=False).flatten()
    indices[: active.numel()] = active.to(indices.dtype)
    return indices


def materialize_ref(
    state: Sequence[torch.Tensor],
    k_cache: Sequence[torch.Tensor],
    u_cache: Sequence[torch.Tensor],
    g_cache: Sequence[torch.Tensor],
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    ring_start: torch.Tensor,
    replay_prefix_len: torch.Tensor,
    active_request_indices: Optional[torch.Tensor] = None,
    *,
    pad_slot_id: int = -1,
    max_window: Optional[int] = None,
) -> List[int]:
    """Replay a chosen ring prefix onto a checkpoint, into a separate slot.

    Every tensor argument is a per-layer sequence of length L; the slot tables
    are ``[L, B]`` so a request may occupy different pool slots in different
    layers.  ``state`` is mutated in place at ``dst_slots``; sources and rings
    are read-only.

    Parameters
    ----------
    state : sequence of L tensors, each ``[pool, HV, V, K]``
        Per-layer state pool.  Both the source checkpoint and the destination
        live here -- same pool, different slots.
    k_cache : sequence of L tensors, each ``[pool, H, ring_slots, K]``
        Ring of L2-normalized keys.
    u_cache : sequence of L tensors, each ``[pool, HV, ring_slots, V]``
        Ring of delta-rule corrections.  ``beta`` is already multiplied in at
        decode time, which is why materialization needs no beta argument.
    g_cache : sequence of L tensors, each ``[pool, HV, ring_slots]`` fp32
        Ring of cumulative log-decay, measured from the window base.
    src_slots, dst_slots : ``[L, B]`` int32
        Read-from and write-to slot per layer and request.  A (layer, request)
        pair is skipped when either equals ``pad_slot_id``.
    ring_start : ``[B]`` int32
        Physical ring row of logical row 0 -- the request's ``cache_base``.
    replay_prefix_len : ``[B]`` int32
        Number of ring entries to replay.  0 means an exact state copy.
        Negative means skip the request entirely.
    active_request_indices : ``[B]`` int32, optional
        Compacted live-request list terminated by -1.  Derived from
        ``replay_prefix_len`` when omitted.  Contents are not cross-checked
        against ``replay_prefix_len`` -- same contract as the mamba kernel.
    pad_slot_id : int
        Slot sentinel to skip.  Note 0 is a legal slot unless passed here.
    max_window : int, optional
        Replay lengths above this are skipped rather than raising, matching
        the kernel's ``count > W_RING`` guard (W_RING = 16, one MMA tile).
        Defaults to 16; pass the ring depth only to emulate the mamba kernel,
        whose MAX_WINDOW can reach its ring size.

    Returns
    -------
    list of int
        The physical request indices actually processed, in visit order.
        Useful for asserting that padding rows and post-sentinel entries were
        left alone.
    """
    layers = len(state)
    if not (len(k_cache) == len(u_cache) == len(g_cache) == layers):
        raise ValueError("per-layer sequences must all have length L")
    if src_slots.shape != dst_slots.shape:
        raise ValueError("src_slots and dst_slots must have the same shape")
    if src_slots.shape[0] != layers:
        raise ValueError("slot tables must be [L, B]")

    batch = src_slots.shape[1]
    for name, t in (
        ("ring_start", ring_start),
        ("replay_prefix_len", replay_prefix_len),
    ):
        if t.numel() != batch:
            raise ValueError(f"{name} must have {batch} entries")

    if active_request_indices is None:
        active_request_indices = active_request_indices_from(replay_prefix_len)
    elif active_request_indices.numel() != batch:
        raise ValueError("active_request_indices must have B entries")

    ring_slots = k_cache[0].shape[2]
    if max_window is None:
        max_window = 16  # the GDN kernel's W_RING replay-window guard

    # Pull the small metadata tensors to host once; this is a reference, and
    # per-element .item() inside the loops would dominate its runtime.
    active_host = active_request_indices.tolist()
    counts_host = replay_prefix_len.tolist()
    starts_host = ring_start.tolist()
    src_host = src_slots.tolist()
    dst_host = dst_slots.tolist()

    visited: List[int] = []
    for virtual_request in range(batch):
        physical_request = active_host[virtual_request]
        # Live entries are compacted to the front, so the first sentinel means
        # there is no work left anywhere -- stop, do not merely skip.
        if physical_request < 0:
            break
        count = counts_host[physical_request]
        if count < 0 or count > max_window:
            continue
        visited.append(physical_request)

        start = starts_host[physical_request]
        for layer in range(layers):
            src = src_host[layer][physical_request]
            dst = dst_host[layer][physical_request]
            if src == pad_slot_id or dst == pad_slot_id:
                continue
            _materialize_one(
                state[layer],
                k_cache[layer],
                u_cache[layer],
                g_cache[layer],
                src,
                dst,
                start,
                count,
                ring_slots,
            )
    return visited


def _materialize_one(
    state: torch.Tensor,
    k_cache: torch.Tensor,
    u_cache: torch.Tensor,
    g_cache: torch.Tensor,
    src: int,
    dst: int,
    start: int,
    count: int,
    ring_slots: int,
) -> None:
    """One (layer, request): replay `count` entries from `src` into `dst`."""
    if count == 0:
        # Exact copy. Cloned first so a src == dst call is still well defined.
        state[dst] = state[src].clone()
        return

    f = torch.float32
    hv = state.shape[1]
    h = k_cache.shape[1]
    grp = hv // h

    # Logical row j -> physical ring row (start + j) % ring_slots.
    rows = torch.tensor(
        [(start + j) % ring_slots for j in range(count)],
        dtype=torch.long,
        device=state.device,
    )
    kc = k_cache[src].index_select(1, rows).to(f)  # [H,  C, K]
    uc = u_cache[src].index_select(1, rows).to(f)  # [HV, C, V]
    gc = g_cache[src].index_select(1, rows).to(f)  # [HV, C]

    # Endpoint decay is the stored g at the LAST replayed entry.
    g_end = gc[:, count - 1]  # [HV]
    w = torch.exp(g_end[:, None] - gc)  # [HV, C]
    kc_hv = kc.repeat_interleave(grp, dim=0)  # [HV, C, K]

    s = torch.exp(g_end)[:, None, None] * state[src].to(f)
    s = s + torch.einsum("hcv,hck->hvk", w[:, :, None] * uc, kc_hv)
    state[dst] = s.to(state.dtype)
