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

Measured launch configurations for the PCIe IPC all-reduce.

Two tables, keyed on the interconnect rather than the GPU:

``rootcplx-noswitch``
    Every peer transfer crosses the CPU root complex. Two properties of that
    fabric drive the whole table: all-to-all peer writes collapse relative to
    neighbour writes, and the collapse worsens the more blocks write
    concurrently -- hence block counts of 1-2 where a switch-paired machine
    wants 8-96. Barriers are also far more expensive, which is why the
    barrier-free path keeps the small-batch TP8 range.

``pcieswitch-pairs``
    NUMA islands contain PCIe switch pairs, so some peers talk without
    reaching the host bridge. Inherited from the machine the kernels were
    originally tuned on.

Shapes outside the tables return ``None``: the caller must fall back rather
than run an untuned configuration, because the untuned default (max blocks,
no staging) is the worst direction on a switch-free fabric.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
from functools import lru_cache
from typing import Optional

from .pcie_ipc_topology import PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR

# Block counts above this are never useful on either fabric and the workspace
# is sized for it.
MAX_BLOCKS = 128


class IpcVariant(IntEnum):
    """Which kernel to launch; mirrors ``fi::Variant`` in the header.

    Values cross the FFI boundary as integers, so they are append-only.
    ``FLAT_STAGED`` is accepted at world size 8 only -- at 4 it would name the
    same kernel as ``STAGED``, and at 2 there is no staged-vs-flat distinction.
    """

    UNSTAGED = 0
    STAGED = 1
    STAGED_RING = 2
    FLAT_STAGED = 3


@dataclass(frozen=True)
class IpcLaunchConfig:
    blocks: int
    threads: int
    variant: IpcVariant


def _tp2(batch: int) -> Optional[IpcLaunchConfig]:
    # Shared by both profiles: two ranks push straight to their only peer, so
    # there is no all-to-all pattern to fix and the switch makes no difference.
    # At large batch this already sits at the link ceiling.
    if batch <= 1:
        return IpcLaunchConfig(32, 64, IpcVariant.UNSTAGED)
    if batch <= 2:
        return IpcLaunchConfig(128, 64, IpcVariant.UNSTAGED)
    if batch <= 4:
        return IpcLaunchConfig(64, 128, IpcVariant.STAGED)
    if batch <= 8:
        return IpcLaunchConfig(96, 64, IpcVariant.STAGED)
    if batch <= 12:
        return IpcLaunchConfig(16, 64, IpcVariant.UNSTAGED)
    if batch <= 16:
        return IpcLaunchConfig(64, 128, IpcVariant.UNSTAGED)
    if batch <= 28:
        return IpcLaunchConfig(16, 64, IpcVariant.UNSTAGED)
    if batch <= 32:
        return IpcLaunchConfig(64, 64, IpcVariant.STAGED)
    if batch <= 44:
        return IpcLaunchConfig(16, 128, IpcVariant.UNSTAGED)
    if batch <= 48:
        return IpcLaunchConfig(32, 128, IpcVariant.STAGED)
    return IpcLaunchConfig(16, 128, IpcVariant.UNSTAGED)


def _rootcplx(world_size: int, hidden: int, batch: int) -> Optional[IpcLaunchConfig]:
    if world_size == 4 and hidden == 4096:
        # Staged reduce-scatter/all-gather wins at every batch here, not only
        # above 20 as on the switch-paired machine.
        if batch <= 4:
            # Too little payload to amortise the staged kernel's 2*(N-1)
            # barriers, so keep the unstaged push.
            return IpcLaunchConfig(1, 128, IpcVariant.STAGED)
        # Neighbour-ordered staging trades all-to-all writes for neighbour
        # writes; with the contention gone, more blocks start to pay again.
        if batch <= 8:
            return IpcLaunchConfig(1, 256, IpcVariant.STAGED_RING)
        if batch <= 64:
            return IpcLaunchConfig(2, 256, IpcVariant.STAGED_RING)
        return IpcLaunchConfig(4, 256, IpcVariant.STAGED_RING)
    if world_size == 8 and hidden >= 6144:
        if batch <= 1:
            # Barrier-free pack kernel; below this the staged kernel's six
            # island barriers cost more than the traffic they save.
            return IpcLaunchConfig(32, 128, IpcVariant.UNSTAGED)
        if batch <= 3:
            # Ownership follows the data index, so a single CTA spans one or
            # two adjacent owners: the pushes stage without the six island
            # barriers the topology kernels pay for the same effect. Above this
            # a CTA spans enough owners that the staging stops happening and
            # the explicit barriers win instead.
            return IpcLaunchConfig(1, 128, IpcVariant.FLAT_STAGED)
        # Staged intra-island push, which beats the block kernel because that
        # one pushes to all three island peers at once instead of one at a
        # time. Staging also lifts the blocks % 4 constraint, and the low block
        # counts that frees up are most of the win.
        if batch <= 4:
            return IpcLaunchConfig(1, 128, IpcVariant.STAGED_RING)
        if batch <= 44:
            return IpcLaunchConfig(1, 256, IpcVariant.STAGED_RING)
        return IpcLaunchConfig(2, 256, IpcVariant.STAGED_RING)
    return None


def _switchpair(world_size: int, hidden: int, batch: int) -> Optional[IpcLaunchConfig]:
    if world_size == 4 and hidden == 4096:
        if batch <= 1:
            return IpcLaunchConfig(8, 128, IpcVariant.UNSTAGED)
        if batch <= 4:
            return IpcLaunchConfig(64, 128, IpcVariant.UNSTAGED)
        if batch <= 8:
            return IpcLaunchConfig(96, 128, IpcVariant.UNSTAGED)
        if batch <= 16:
            return IpcLaunchConfig(16, 128, IpcVariant.UNSTAGED)
        if batch <= 40:
            return IpcLaunchConfig(16, 128, IpcVariant.STAGED)
        return IpcLaunchConfig(32, 128, IpcVariant.STAGED)
    if world_size == 8 and hidden >= 6144:
        if batch <= 4:
            return IpcLaunchConfig(12, 256, IpcVariant.UNSTAGED)
        if batch <= 32:
            return IpcLaunchConfig(32, 128, IpcVariant.STAGED)
        return IpcLaunchConfig(8, 512, IpcVariant.STAGED)
    return None


def _is_launchable(world_size: int, config: IpcLaunchConfig, max_blocks: int) -> bool:
    """Reject configurations the kernels cannot accept.

    A violation here degrades to "unsupported shape" and a caller fallback,
    which is far better than reaching the kernel and failing a hard check.
    """
    if not 0 < config.blocks <= max_blocks:
        return False
    if not world_size <= config.threads <= 1024:
        return False
    # One configuration must name exactly one kernel, so the pairs the header
    # does not dispatch are rejected rather than aliased onto a neighbour.
    if world_size == 2 and config.variant not in (
        IpcVariant.UNSTAGED,
        IpcVariant.STAGED,
    ):
        return False
    if config.variant == IpcVariant.FLAT_STAGED and world_size != 8:
        return False
    # The block-partitioned TP8 kernel derives its chunk from blockIdx.x & 3.
    # Every other kernel uses flat grid-stride loops.
    if (
        world_size == 8
        and config.variant == IpcVariant.STAGED
        and config.blocks % 4 != 0
    ):
        return False
    return True


@lru_cache(maxsize=None)
def get_pcie_ipc_launch_config(
    profile: str,
    world_size: int,
    hidden: int,
    batch: int,
    max_blocks: int = MAX_BLOCKS,
) -> Optional[IpcLaunchConfig]:
    """Launch configuration for one shape, or ``None`` when untuned.

    Depends only on its arguments, so every rank in a group reaches the same
    answer -- a prerequisite, since a rank that opts out while its peers opt in
    deadlocks the collective.
    """
    if world_size == 2:
        config = _tp2(batch) if hidden <= 2048 else None
    elif profile == PROFILE_ROOTCPLX:
        config = _rootcplx(world_size, hidden, batch)
    elif profile == PROFILE_SWITCHPAIR:
        config = _switchpair(world_size, hidden, batch)
    else:
        raise ValueError(f"unknown profile {profile!r}")

    if config is None:
        return None
    config = replace(config, blocks=min(config.blocks, max_blocks))
    return config if _is_launchable(world_size, config, max_blocks) else None
