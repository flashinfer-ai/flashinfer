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
    # Copy-engine ring. `blocks` on this variant is not a grid size: it carries
    # the ring's sub-chunk depth, which moves the collective by an order more
    # than the add kernel's geometry does -- the geometry is not free, it is
    # dominated -- so the one tunable integer goes to the knob that decides the
    # outcome. Reusing the field rather than widening IpcLaunchConfig keeps
    # the tactic a 3-tuple, which leaves the codec, pack_config and every arity
    # test untouched, and holds this variant to a handful of candidates.
    COPY_ENGINE_RING = 4
    # The 4+4 decomposition. Reachable only at world size 8 and only on a
    # fabric that grouping describes -- see _is_launchable, and the profile
    # filter in pcie_ipc_tuning.get_valid_tactics, which keeps it out of the
    # candidate list on fabrics it does not describe, where it lost to both
    # the flat ring and the SM path.
    COPY_ENGINE_ISLAND = 5


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


CE_MAX_PIECES = 4
# The copy-engine add kernel's block size. Fixed, not tuned -- see _is_launchable.
CE_THREADS = 256


def _is_launchable(
    world_size: int,
    config: IpcLaunchConfig,
    max_blocks: int,
    numel: Optional[int] = None,
    elem_size: int = 2,
) -> bool:
    """Reject configurations the kernels cannot accept.

    A violation here degrades to "unsupported shape" and a caller fallback,
    which is far better than reaching the kernel and failing a hard check.

    `numel` is optional because most callers ask whether a configuration can
    exist at all, not whether it fits one shape; passing it is what admits the
    shape-dependent rules below. The tuner must pass it: the copy-engine ring
    has a divisibility precondition that `_admits` does not imply, and a
    candidate that reaches the launcher and fails its hard check raises on one
    rank while its peers spin with no timeout.
    """
    if config.variant in (IpcVariant.COPY_ENGINE_RING, IpcVariant.COPY_ENGINE_ISLAND):
        if config.variant == IpcVariant.COPY_ENGINE_ISLAND and world_size != 8:
            return False
        # A copy-engine variant is not exempt from the world-size-2 rule just
        # because it names its own launcher: the launcher admits exactly two
        # variants there and rejects every other one by number, so a candidate
        # that gets past this point fails its hard check. The schedule is
        # correct at two ranks and costs nothing to leave out; the header of
        # pcie_ipc_ce_ring.cuh says why, and how far that has been established.
        if world_size == 2:
            return False
        if not 0 < config.blocks <= CE_MAX_PIECES:
            return False
        # The ring is bound by the fabric rather than by the SM, so the add
        # kernel's geometry is not worth a search dimension. The SM variants
        # are the opposite case, which is why their geometry stays tunable.
        if config.threads != CE_THREADS:
            return False
        if numel is not None:
            pack_elems = 16 // elem_size
            # The island schedule splits into four chunks whatever the world size.
            shards = (
                4 if config.variant == IpcVariant.COPY_ENGINE_ISLAND else world_size
            )
            if numel % (shards * pack_elems) != 0:
                return False
            if (numel // shards) % (config.blocks * pack_elems) != 0:
                return False
        return world_size <= config.threads <= 1024
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
