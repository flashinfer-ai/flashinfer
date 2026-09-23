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
    # Single-piece flat ring with CE stream publication, enabled jointly on SM120.
    COPY_ENGINE_RING_MEMOP = 6


@dataclass(frozen=True)
class IpcLaunchConfig:
    blocks: int
    threads: int
    variant: IpcVariant
    # Fused only: blocks that run the collective, the rest joining just for the
    # normalisation. 0 means "all of them", which is what the plain kernels do
    # and the only value they accept.
    transport_blocks: int = 0

    def effective_transport_blocks(self) -> int:
        return self.transport_blocks or self.blocks


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


# Packs one thread holds across a row's normalisation; mirrors
# kFusedMaxPacksPerThread in the header. Wider rows would have to be re-read from
# HBM, which is the traffic the fusion exists to remove.
FUSED_MAX_PACKS_PER_THREAD = 4
# Below this the block reduction is mostly idle warps, and the row is small
# enough that the thread count is not what limits it.
_FUSED_MIN_THREADS = 256


def fused_rms_norm_threads(hidden: int, elem_size: int) -> Optional[int]:
    """Threads per block for the fused kernel, or ``None`` if the row is too wide.

    A block covers a whole row so the denominator is a block reduction rather
    than a grid-wide one; that is what keeps the fused kernel free of the
    co-residency requirement a cross-block row reduction would impose.
    """
    pack_elems = 16 // elem_size
    if hidden <= 0 or hidden % pack_elems != 0:
        return None
    hidden_packs = hidden // pack_elems
    needed = -(-hidden_packs // FUSED_MAX_PACKS_PER_THREAD)
    threads = _FUSED_MIN_THREADS
    while threads < needed:
        threads *= 2
    return threads if threads <= 1024 else None


# Payload above which the fused path takes the neighbour-ordered transport.
# Keyed on world size, unlike the plain path's single threshold: the flat push
# has seven concurrent destinations at eight ranks against three at four, and
# it collapses in proportion, so the crossover moves. Measured at hidden 6144,
# where one row is 12 KiB -- TP4 crosses between two rows (27 us flat, 30 ring)
# and three (48 flat, 34 ring), TP8 between one (41 flat, 60 ring) and two (89
# flat, 67 ring).
_FUSED_SEED_RING_BYTES = {4: 32 * 1024, 8: 16 * 1024}

# Same cap as the plain ring, and measured rather than assumed: the row-affine
# normalisation has a claim on the grid that the plain kernel does not, so this
# was expected to want more blocks, and it does not. At TP4/hidden 6144 the
# neighbour-ordered transport still decides the total, and four blocks is the
# minimum of the curve at every batch from 16 up (batch 128: 231 us at four
# blocks against 264 at eight and 531 at one).
#
# Unlike the plain seed this does not scale the count with the payload. Below
# 256 KiB that formula yields a single block, which costs the normalisation
# more than the transport saves -- batch 16 is 85 us at one block against 56 at
# four.
_FUSED_SEED_RING_MAX_BLOCKS = 4


def _is_fused_launchable(
    world_size: int,
    config: IpcLaunchConfig,
    max_blocks: int,
    hidden: int,
    elem_size: int,
    sm_count: Optional[int] = None,
    numel: Optional[int] = None,
) -> bool:
    """Reject configurations the fused kernels cannot accept.

    Separate from :func:`_is_launchable` because the two dispatches accept
    different things: the fused entry point takes only the two transports, its
    thread count is bounded below by the row it has to hold in registers, and
    its grid is bounded above by what the device can hold.

    ``sm_count`` is that upper bound: every fused kernel ends in a rendezvous of
    all its blocks, which never completes if one of them was not scheduled, and
    ``__launch_bounds__(1024, 1)`` puts at most one block on an SM. The launcher
    refuses such a grid too; rejecting here keeps the tuner from ever proposing
    one. ``None`` skips the check, for callers that have no device in hand.
    """
    if not 0 < config.blocks <= max_blocks:
        return False
    if sm_count is not None and config.blocks > sm_count:
        return False
    if not 0 < config.effective_transport_blocks() <= config.blocks:
        return False
    if not world_size <= config.threads <= 1024:
        return False
    # Both data planes, on the same (world_size, variant) -> kernel mapping the
    # plain dispatch uses, so a tactic means the same thing on both paths. The
    # copy-engine variants reach the collective they already name followed by a
    # normalisation pass rather than a fused kernel -- the engine cannot
    # transform data, so there is nothing at the end of its all-gather to fold
    # into. Excluding them cost the fused path the fastest transport on any
    # fabric where the engine wins, which is a larger loss than the pass.
    copy_engine = config.variant in (
        IpcVariant.COPY_ENGINE_RING,
        IpcVariant.COPY_ENGINE_ISLAND,
        IpcVariant.COPY_ENGINE_RING_MEMOP,
    )
    if not copy_engine and config.variant not in (
        IpcVariant.UNSTAGED,
        IpcVariant.STAGED,
        IpcVariant.STAGED_RING,
        IpcVariant.FLAT_STAGED,
    ):
        return False
    if copy_engine:
        # `blocks` is the ring's sub-chunk depth here and `threads` belongs to
        # its add kernel, so the SM plane's bounds below do not apply. The
        # normalisation pass derives its own width, bounded by the widest block
        # rather than by this thread count.
        #
        # Refused without a payload rather than admitted on the checks that can
        # still be made: the ring's shard divisibility is a function of numel,
        # and a caller who cannot say what it is cannot be told this will run.
        if numel is None:
            return False
        if not _is_launchable(world_size, config, max_blocks, numel, elem_size):
            return False
        pack_elems = 16 // elem_size
        if hidden <= 0 or hidden % pack_elems != 0:
            return False
        return hidden // pack_elems <= FUSED_MAX_PACKS_PER_THREAD * 1024
    # Neighbour ordering needs peers to order; at two ranks it is one outbound
    # stream either way, and no ring kernel is dispatched there.
    if config.variant == IpcVariant.STAGED_RING and world_size == 2:
        return False
    # The topology-blind push is the world-8 fallback only, as on the plain path.
    if config.variant == IpcVariant.FLAT_STAGED and world_size != 8:
        return False
    # The world-8 island-block kernel takes its chunk from blockIdx.x & 3 and its
    # stride from transport_blocks >> 2: four blocks per chunk minimum, on both
    # counts, or the grid-stride loops never advance.
    if config.variant == IpcVariant.STAGED and world_size == 8:
        if config.blocks % 4 != 0 or config.effective_transport_blocks() % 4 != 0:
            return False
    pack_elems = 16 // elem_size
    if hidden <= 0 or hidden % pack_elems != 0:
        return False
    # The row is held in registers across the normalisation.
    return hidden // pack_elems <= FUSED_MAX_PACKS_PER_THREAD * config.threads


def get_pcie_ipc_fused_launch_config(
    world_size: int,
    numel: int,
    hidden: int,
    elem_size: int,
    max_blocks: int = MAX_BLOCKS,
    sm_count: Optional[int] = None,
) -> Optional[IpcLaunchConfig]:
    """Launch configuration for the fused all-reduce, or ``None`` if unsupported.

    The variant field names the *transport*, on the same crossover as
    :func:`_seed`: push to every peer at once while the payload is small,
    neighbour-ordered once it is not. The fused kernel is still reached through
    its own entry point rather than through the plain variant dispatch.
    """
    if not _admits(world_size, numel, elem_size):
        return None
    if numel % hidden != 0:
        return None
    threads = fused_rms_norm_threads(hidden, elem_size)
    if threads is None or threads < world_size:
        return None
    # Block count is capped by the row count under either transport: the
    # normalisation is row-affine, so blocks past the last row have nothing to
    # normalise. What the cap is below that differs, and only the ring's was
    # measured to be the same as the plain path's -- see the two constants.
    rows = numel // hidden
    if sm_count is not None:
        max_blocks = min(max_blocks, sm_count)
    # The grid is the normalisation's -- one block per row, capped by the device
    # -- and the collective runs on a slice of it. Sizing the launch to the
    # collective instead would starve the normalisation, which is HBM-bound.
    blocks = max(1, min(rows, max_blocks))
    ring_bytes = _FUSED_SEED_RING_BYTES.get(world_size)
    if ring_bytes is not None and numel * elem_size >= ring_bytes:
        config = IpcLaunchConfig(
            blocks,
            threads,
            IpcVariant.STAGED_RING,
            min(blocks, _FUSED_SEED_RING_MAX_BLOCKS),
        )
    elif world_size == 4:
        # Below the crossover at four ranks, staging still cuts egress and the
        # all-to-all form pays only two barriers, so the one-shot is never the
        # answer -- the same reasoning _seed() uses, and measured the same way
        # (batch 1, hidden 6144: 11.7 us staged against 32.8 one-shot).
        config = IpcLaunchConfig(blocks, threads, IpcVariant.STAGED, blocks)
    else:
        # Two ranks have nothing to stage, and eight below the crossover are
        # served by the island-decomposed one-shot, which has no barriers at all
        # (batch 1, hidden 6144: 17.0 us against 32.4 for the flat staged form).
        config = IpcLaunchConfig(blocks, threads, IpcVariant.UNSTAGED, blocks)
    if not _is_fused_launchable(
        world_size, config, max_blocks, hidden, elem_size, sm_count, numel
    ):
        return None
    return config


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
    if config.variant in (
        IpcVariant.COPY_ENGINE_RING,
        IpcVariant.COPY_ENGINE_ISLAND,
        IpcVariant.COPY_ENGINE_RING_MEMOP,
    ):
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
