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

Launch configurations for the PCIe IPC all-reduce.

Two layers, with very different standing.

**Admission** (:func:`_admits`) is a capability question: which shapes the
kernels can run at all. It is not a performance judgement, and tuning cannot
change it.

**The seed** (:func:`_seed`) is a *default*, not a measurement. It picks the
one side of the one crossover that ports between machines -- push straight to
every peer while the payload is small, reduce-scatter/all-gather once it is
not -- and nothing finer.

Thresholds fitted per batch on one machine do not survive the trip to another,
so only the shape of the answer lives here; the numbers come from
:meth:`~flashinfer.comm.PcieIpcAllReduceWorkspace.tune`, which measures them
where they will run. Running untuned is warned about once per workspace.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
from functools import lru_cache
from typing import Optional


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


# Payload above which reduce-scatter/all-gather beats pushing to every peer.
# Keyed on bytes, not tokens: the crossover trades bytes moved against barrier
# latency, and only that ratio ports between fabrics.
_SEED_STAGE_BYTES = 32 * 1024

# The neighbour-ordered kernel has one outbound stream per rank whatever the
# grid, so extra blocks pay only once there are bytes enough to keep the link
# busy. Capped low: with no switch-local peer, concurrent transfers collapse
# rather than add.
_SEED_RING_BYTES_PER_BLOCK = 256 * 1024
_SEED_RING_MAX_BLOCKS = 4


def _admits(world_size: int, numel: int, elem_size: int) -> bool:
    """Whether the kernels can run this shape at all.

    Independent of which kernel is chosen: tuning launches every variant on
    whatever shape is admitted, so a precondition that held only for the
    variant the seed happens to pick would still deadlock the group under
    :meth:`~flashinfer.comm.PcieIpcAllReduceWorkspace.tune`.
    """
    if world_size not in (2, 4, 8):
        return False
    pack_elems = 16 // elem_size
    # Matches the launcher's own check; the kernels address whole 16-byte packs.
    if numel % pack_elems != 0:
        return False
    # Reduce-scatter gives each rank num_packs // world_size packs. Below one
    # pack per rank that split degenerates onto a single owner: correct, but it
    # leaves the other ranks idle, and a payload that small is better served by
    # another backend than by an IPC collective.
    return numel >= pack_elems * world_size


def _seed(
    world_size: int, numel: int, elem_size: int, max_blocks: int
) -> IpcLaunchConfig:
    """Default configuration for a shape nothing has measured yet."""
    payload = numel * elem_size

    if world_size == 2:
        # Staging moves the same bytes it would have pushed, so there is no
        # crossover here and no second branch to justify.
        return IpcLaunchConfig(min(16, max_blocks), 128, IpcVariant.UNSTAGED)

    if payload >= _SEED_STAGE_BYTES:
        # Neighbour-ordered rather than all-to-all: with no switch-local peer,
        # simultaneous writes to every peer collapse, and the penalty grows with
        # the payload. Picking wrong on this arm is unbounded rather than merely
        # slow, which is why the threshold sits low.
        blocks = max(
            1,
            min(
                _SEED_RING_MAX_BLOCKS,
                payload // _SEED_RING_BYTES_PER_BLOCK,
                max_blocks,
            ),
        )
        return IpcLaunchConfig(blocks, 256, IpcVariant.STAGED_RING)

    if world_size == 4:
        # Staging always cuts egress at four ranks and the all-to-all form pays
        # only two barriers, so the one-shot push is never the answer. One
        # block, because its grid multiplies the concurrency that collapses.
        return IpcLaunchConfig(1, 256, IpcVariant.STAGED)

    # Eight ranks below the crossover: the island-partitioned push has no
    # barriers, which is what the staged path's six island barriers must beat.
    return IpcLaunchConfig(min(16, max_blocks), 256, IpcVariant.UNSTAGED)


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
    world_size: int,
    numel: int,
    elem_size: int,
    max_blocks: int = MAX_BLOCKS,
) -> Optional[IpcLaunchConfig]:
    """Launch configuration for one shape, or ``None`` when unsupported.

    ``None`` means the kernels cannot run the shape, so the caller must use
    another backend. It never means "untuned": an untuned shape gets the seed.

    Depends only on its arguments, so every rank in a group reaches the same
    answer -- a prerequisite, since a rank that opts out while its peers opt in
    deadlocks the collective.
    """
    if not _admits(world_size, numel, elem_size):
        return None
    config = _seed(world_size, numel, elem_size, max_blocks)
    config = replace(config, blocks=min(config.blocks, max_blocks))
    return config if _is_launchable(world_size, config, max_blocks) else None
