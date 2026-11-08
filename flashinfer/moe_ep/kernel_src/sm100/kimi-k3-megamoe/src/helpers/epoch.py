# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v33-epoch: MEGAMOE_EPOCH_COUNTERS (prod1: default on; =0 disables) -- launch-parity double-buffered tail state, one tail NVLink barrier.

Stock tail (TokenCommNonDeterministic.reset_tail): grid sync + cross-rank barrier ("drain"), zero every
``tail_reset`` region of both workspaces, grid sync + cross-rank barrier ("publish"), exit.

With the flag:
- Every ``tail_reset`` region exists twice. ``DeviceWorkspace.finalize`` lays the tail block out once more right after
  itself (block rounded to 128 B), so copy 1 of a region sits at ``offset + block``. Both copies are inside the
  host-zeroed leading prefix (``require_zero_workspace_leading_bytes``), so allocation and the per-pairing barrier
  fence zero both.
- A per-rank launch counter ``EPOCH_REGION`` (local, zero_on_first_allocate, i.e. zeroed together with the NVLink
  barrier phase counter by the same host fence) selects the copy: launch e uses copy ``e & 1``. Every kernel that
  touches the workspace (router, MegaMoE) loads it once at entry (``DeviceWorkspace.load_epoch``); every
  ``ptr``/``tensor`` of a tail region adds ``(e & 1) * block``. Accessing a tail region before ``load_epoch``
  raises at trace time (catches host-side or missed sites).
- MegaMoE launch e zeroes copy ``(e + 1) & 1`` (launch e - 1's, dead) at kernel START, on the token-in transfer warps
  before they wait for the router (idle window), instead of zeroing copy ``e & 1`` in the tail.
- Tail: one grid sync + one cross-rank barrier (the old "drain"), then CTA 0 / transfer thread 0 advances the epoch.
  The "publish" barrier and the in-tail zero pass are gone.

Counter semantics, targets and comparisons are untouched (each launch still sees all-zero counters), so outputs are
bit-identical by construction. Int32 wrap of the epoch is harmless (only the parity is used; 2^32 is even).
The memory-ordering argument is in notes/fp8-critical-path.md (section "v33-epoch").
"""
from . import prod_config as _prod_config

REGION = "v33.epoch"


def enabled() -> bool:
    # default ON (every config); MEGAMOE_EPOCH_COUNTERS=0 restores the stock two-barrier tail.
    return _prod_config.epoch_enabled()


def tag() -> str:
    return "_epoch" if enabled() else ""
