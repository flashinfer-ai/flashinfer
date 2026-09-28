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

Autotuning for the PCIe IPC all-reduce.

The seed in :mod:`~flashinfer.comm.pcie_ipc_policy` is a default, not a
measurement: one crossover, and no constants fitted to any machine. This module
measures the same choice, over the launch configurations the dispatch can
actually reach.

Two properties of the surrounding code shape everything here:

**The autotuner never looks at a kernel's output**, and this kernel family's
characteristic failure is wrong *and* fast. So every candidate is verified
against a reference before it is timed, and the verdict is reduced across the
group -- see :meth:`PcieIpcAllReduceRunner.get_valid_tactics`.

**Every wait in the kernels is an unbounded spin.** Ranks that disagree on the
launch configuration, or that issue different numbers of calls, hang rather
than raise. So the candidate list is a pure function of group-identical
arguments, the verification verdict is reduced before it is used, and the
resolved configuration is checked for group agreement before it is cached.

The policy module keeps three jobs here: admission decides which shapes are
supported at all, and the seed is both tactic ``-1`` and the fallback whenever a
tuned answer cannot be used.
"""

import os
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from ..autotuner import (
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
    make_bucket_mapper,
)
from .pcie_ipc_topology import PROFILE_ROOTCPLX
from .pcie_ipc_policy import (
    MAX_BLOCKS,
    IpcLaunchConfig,
    IpcVariant,
    _is_fused_launchable,
    _is_launchable,
)

# Baked into every persisted cache key, so renaming it silently invalidates
# every cache file rather than mis-resolving one.
PCIE_IPC_CUSTOM_OP = "flashinfer::pcie_ipc_all_reduce"

# The fused op searches under its own name rather than as extra variants of the
# one above. Two reasons, and either alone would be enough: candidate_tactics()
# enumerates every IpcVariant, so a fused entry there would be swept by the
# plain search, which has no residual or weight to pass it; and the two ops
# rank different kernels over the same (blocks, threads) grid, so sharing a
# name would let one's winner be read back as the other's.
PCIE_IPC_FUSED_CUSTOM_OP = "flashinfer::pcie_ipc_all_reduce_fused_add_rmsnorm"

# Bump when a variant's meaning, the scratch-region assignment, or the
# candidate encoding changes. The autotuner's own metadata records library and
# driver versions but nothing about this op, and a dev checkout does not move
# the FlashInfer version.
#
# 2: the fused kernels moved from the block scratch region to the pack region,
# and gained a transport choice of their own.
# 3: fused tactics gained a fourth field, transport_blocks, and the fused
# kernels changed shape (collective verbatim, then a grid barrier, then the
# normalisation) -- a tactic measured against the old shape names a different
# kernel.
PCIE_IPC_TUNE_VERSION = 3
# Only workspaces that admit stream publication search the additional tactic.
# Preserve their existing namespace without invalidating original-path caches.
_PCIE_IPC_MEMOP_TUNE_VERSION = 4

# Versions the fused search alone, on top of the head above. Separate because
# what invalidates one search does not invalidate the other: both are keyed by
# the same workspace head and told apart only by op name, so folding this into
# PCIE_IPC_TUNE_VERSION would discard every tuned plain configuration for a
# change the plain dispatch never saw.
#
# 2: at world 8 kStaged moved from the flat push to the island-block kernel and
# the flat push moved to kFlatStaged, aligning the fused variant mapping with
# the plain one. The same persisted tactic now names a different kernel there.
PCIE_IPC_FUSED_TUNE_VERSION = 2

# Not all powers of two: the extra entries are block counts the search selected
# on real hardware, and it cannot converge on a configuration its own grid
# cannot name.
TUNE_BLOCKS: Tuple[int, ...] = (1, 2, 4, 8, 12, 16, 32, 64, 96, 128)
TUNE_THREADS: Tuple[int, ...] = (64, 128, 256, 512, 1024)

# Fused only: how many of the launched blocks run the collective. Short, and
# short on purpose -- the collective's optimum is a handful of blocks on this
# fabric, and 0 (meaning "all of them") covers the case where it is not.
TUNE_TRANSPORT_BLOCKS: Tuple[int, ...] = (1, 2, 4, 8, 0)

# Batch buckets. Floor semantics, so a bucket is always a batch the tuner
# actually measured. Matches the benchmark's default sweep.
TUNE_BATCHES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

# Higher than the library defaults, which time too short a span to resolve
# candidates for a collective of this scale.
TUNE_WARMUP = 10
TUNE_REPEAT = 50

# Reference tactic. The autotuner reserves -1 for "the fallback that implements
# any shape"; here that is the policy module's seed configuration.
TABLE_TACTIC = -1

# Inputs are drawn from [0, INIT_MAX_VALUE) so the group sum stays integral and
# exactly representable, which is what lets verification use a zero tolerance
# despite the kernels summing in a different order than NCCL.
INIT_MAX_VALUE = 16

# Epsilon used when screening fused candidates. Any value serves -- the
# reference is built with the same one -- but it is fixed here so the screening
# does not depend on what the caller happens to pass at serving time.
_TUNE_EPS = 1e-6


# Above this payload the grid is narrowed before profiling. The screen is a
# *policy* -- "not worth measuring here" -- not a capability claim, so it lives
# here and not in `_is_launchable`: an explicit `config=` naming a screened-out
# configuration must still run. The grid was sized for decode, where measuring
# every candidate is free; at prefill sizes the cost is dominated by candidates
# orders of magnitude off the winner, each still paying full warmup and repeat.
#
# Only the thread count is screened. A block-count bound was tried and removed:
# the block counts that win span the whole grid, so a bound fitted to one fabric
# silently drops another fabric's winner -- tuning faster and running slower,
# with no error. The thread floor stays because it sits below every winner a
# search has picked, with margin rather than fitted to them; guarded by
# `test_the_prefill_screen_keeps_every_measured_prefill_winner`.
PREFILL_SCREEN_BYTES = 4 * 1024 * 1024
PREFILL_MIN_THREADS = 256


def candidate_tactics(
    world_size: int,
    max_blocks: int = MAX_BLOCKS,
    blocks: Tuple[int, ...] = TUNE_BLOCKS,
    threads: Tuple[int, ...] = TUNE_THREADS,
    numel: Optional[int] = None,
    elem_size: int = 2,
    profile: Optional[str] = None,
    memop_supported: bool = False,
) -> Tuple[Tuple[int, int, int], ...]:
    """Every launch configuration worth profiling for this shape, as tactics.

    A pure function of group-identical arguments, so every rank derives the
    same list in the same order -- which the autotuner's collective profiling
    requires and cannot check. `numel` is group-identical too: every rank
    profiles the same shape.

    `numel=None` means "which configurations can exist at all", not "which are
    worth measuring here", and returns the unscreened grid.
    """
    return _candidate_tactics_cached(
        world_size,
        max_blocks,
        blocks,
        threads,
        numel,
        elem_size,
        profile,
        memop_supported,
    )


@lru_cache(maxsize=None)
def _candidate_tactics_cached(
    world_size, max_blocks, blocks, threads, numel, elem_size, profile, memop_supported
):
    screen = numel is not None and numel * elem_size >= PREFILL_SCREEN_BYTES
    out = []
    for variant in IpcVariant:
        if variant == IpcVariant.COPY_ENGINE_RING_MEMOP and not memop_supported:
            continue
        # The island schedule's 4+4 grouping describes one topology, and on
        # fabrics it does not describe it measured worse than the flat ring and
        # the SM path it competes with, so leaving it reachable is worse than
        # not having it. `profile=None` means the caller is asking which
        # configurations can exist rather than which to measure here, and does
        # not filter.
        if (
            variant == IpcVariant.COPY_ENGINE_ISLAND
            and profile is not None
            and profile != PROFILE_ROOTCPLX
        ):
            continue
        for b in blocks:
            if variant == IpcVariant.COPY_ENGINE_RING_MEMOP and b != 1:
                continue
            for t in threads:
                if screen and t < PREFILL_MIN_THREADS:
                    continue
                if _is_launchable(
                    world_size,
                    IpcLaunchConfig(b, t, variant),
                    max_blocks,
                    numel,
                    elem_size,
                ):
                    out.append((int(variant), b, t))
    return tuple(out)


def fused_candidate_tactics(
    world_size: int,
    hidden: int,
    elem_size: int,
    max_blocks: int = MAX_BLOCKS,
    sm_count: Optional[int] = None,
    blocks: Tuple[int, ...] = TUNE_BLOCKS,
    threads: Tuple[int, ...] = TUNE_THREADS,
    numel: Optional[int] = None,
) -> Tuple[Tuple[int, ...], ...]:
    """Every fused launch configuration the dispatch can reach, as tactics.

    Takes ``hidden`` where :func:`candidate_tactics` does not: the fused kernels
    hold a row in registers, so the row width sets a floor under the thread
    count and the candidate list genuinely differs between hidden sizes. It is
    a group-identical argument like the rest, so every rank still derives the
    same list in the same order.

    Every variant the plain op reaches, both data planes. ``numel`` is required
    for the copy-engine ones, whose shard divisibility is a function of the
    payload rather than of the shape alone; without it they are refused rather
    than guessed at.
    """
    return _fused_candidate_tactics_cached(
        world_size, hidden, elem_size, max_blocks, sm_count, blocks, threads, numel
    )


@lru_cache(maxsize=None)
def _fused_candidate_tactics_cached(
    world_size, hidden, elem_size, max_blocks, sm_count, blocks, threads, numel
):
    out = []
    for variant in IpcVariant:
        for b in blocks:
            for t in threads:
                for tb in TUNE_TRANSPORT_BLOCKS:
                    # 0 means every block; naming it explicitly as well would
                    # enumerate the same kernel twice.
                    if tb == b:
                        continue
                    config = IpcLaunchConfig(b, t, variant, tb)
                    if _is_fused_launchable(
                        world_size,
                        config,
                        max_blocks,
                        hidden,
                        elem_size,
                        sm_count,
                        numel,
                    ):
                        out.append(config_to_tactic(config))
    return tuple(out)


def config_to_tactic(config: IpcLaunchConfig) -> Tuple[int, ...]:
    """Encode a configuration as a tactic.

    Plain ints, because a tactic has to survive a JSON round-trip: the
    autotuner writes ``[0, 32, 128]`` and reads back ``(0, 32, 128)``.
    Self-describing rather than an index into :func:`candidate_tactics`, so
    editing the grid cannot repoint a persisted entry at a different kernel.
    """
    if config.transport_blocks:
        return (
            int(config.variant),
            int(config.blocks),
            int(config.threads),
            int(config.transport_blocks),
        )
    return (int(config.variant), int(config.blocks), int(config.threads))


def tactic_to_config(tactic: Sequence[int]) -> IpcLaunchConfig:
    """Decode a tactic. Raises ``ValueError`` on anything malformed.

    Three fields for the plain op, four for the fused one, whose extra field is
    ``transport_blocks``. Both lengths are accepted here because one persisted
    cache holds both, and a 3-element tactic means "every block transports",
    which is the only thing the plain kernels do.
    """
    if len(tactic) not in (3, 4):
        raise ValueError(f"expected a 3- or 4-element tactic, got {tactic!r}")
    values = [int(v) for v in tactic]
    variant, blocks, threads = values[0], values[1], values[2]
    transport_blocks = values[3] if len(values) == 4 else 0
    try:
        return IpcLaunchConfig(blocks, threads, IpcVariant(variant), transport_blocks)
    except ValueError as exc:
        raise ValueError(f"tactic {tactic!r} names no variant: {exc}") from exc


def _cache_key_head(
    world_size: int,
    profile: str,
    max_blocks: int,
    max_numel: int,
    memop_supported: bool,
) -> Tuple:
    head = (
        _PCIE_IPC_MEMOP_TUNE_VERSION if memop_supported else PCIE_IPC_TUNE_VERSION,
        int(world_size),
        str(profile),
        int(max_blocks),
        int(max_numel),
    )
    return head + (True,) if memop_supported else head


def cache_covers_workspace(
    world_size: int,
    profile: str,
    max_blocks: int,
    max_numel: int,
    memop_supported: bool = False,
) -> bool:
    """Whether the loaded cache holds any entry written for this workspace.

    ``max_numel`` is part of the key, so a workspace sized differently from the
    tuned one misses every entry at once rather than a few -- a configuration
    mistake rather than an untuned shape, and no single lookup can tell those
    apart, since the seed is a valid answer either way.

    dtype is not compared: one workspace serves both 2-byte dtypes and each gets
    its own entries, so a match would be required for a cache that covers the
    workspace perfectly well in the dtype the caller is not using.

    Scanned rather than parsed for the same reason
    :meth:`PcieIpcAllReduceWorkspace._cache_digest` scans -- the key format
    belongs to the autotuner.
    """
    from ..autotuner import AutoTuner

    # Both ops, because either one alone being tuned still means the workspace
    # has measurements. Matching on the plain name only would also match
    # nothing for the fused op -- its key is a longer string that does not start
    # with `('flashinfer::pcie_ipc_all_reduce'` once the closing quote is
    # included -- so the fused entries would be invisible here.
    prefixes = (f"('{PCIE_IPC_CUSTOM_OP}'", f"('{PCIE_IPC_FUSED_CUSTOM_OP}'")
    # cache_key_extras up to the dtype, with the closing paren traded for the
    # separator that must follow it.
    head = _cache_key_head(world_size, profile, max_blocks, max_numel, memop_supported)
    needle = repr(head)[:-1] + ", "
    return any(
        key.startswith(prefixes) and needle in key
        for key in AutoTuner.get()._file_configs
    )


def resolve_tuned_config(
    table_config: IpcLaunchConfig,
    tactic,
    world_size: int,
    max_blocks: int,
) -> IpcLaunchConfig:
    """Turn a tactic into a configuration, falling back to the seed.

    The autotuner does not check that a cached tactic can implement the shape
    it is being reused for, so a cache written against a larger ``max_blocks``
    would otherwise reach the launcher's hard checks and raise on every rank in
    the middle of a collective.
    """
    if tactic is None or tactic == TABLE_TACTIC:
        return table_config
    try:
        config = tactic_to_config(tactic)
    except (TypeError, ValueError):
        return table_config
    if not _is_launchable(world_size, config, max_blocks):
        return table_config
    return config


def resolve_tuned_fused_config(
    table_config: IpcLaunchConfig,
    tactic,
    world_size: int,
    max_blocks: int,
    hidden: int,
    elem_size: int,
    sm_count: Optional[int] = None,
    numel: Optional[int] = None,
) -> IpcLaunchConfig:
    """:func:`resolve_tuned_config` for the fused op, with its own admission.

    Same guard and the same reason: a cached tactic is not checked by the
    autotuner against the shape it is reused for, and here the row width bounds
    the thread count, so a cache written at one hidden size names configurations
    another cannot launch.
    """
    if tactic is None or tactic == TABLE_TACTIC:
        return table_config
    try:
        config = tactic_to_config(tactic)
    except (TypeError, ValueError):
        return table_config
    if not _is_fused_launchable(
        world_size, config, max_blocks, hidden, elem_size, sm_count, numel
    ):
        return table_config
    return config


def small_int_initializer(
    shapes: Tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Synthesize profiling inputs that can be compared at zero tolerance.

    The autotuner's default fills tensors with ``rand() * 10 - 5``, which no
    reference can be compared against exactly. Small integers keep the group
    sum exact in both supported dtypes, so verification uses ``torch.equal``
    and cannot mistake a reduction-order difference for a protocol bug. Zero is
    in the range on purpose: the sentinel kernels rewrite real zeros in the
    payload, and that path should be exercised.
    """
    return torch.randint(
        0, INIT_MAX_VALUE, shapes, device=device, dtype=torch.int32
    ).to(dtype)


@lru_cache(maxsize=None)
def pcie_ipc_tuning_config(batches: Tuple[int, ...] = TUNE_BATCHES) -> TuningConfig:
    """Tuning configuration for one bucket set.

    Cached so that the serving-side cache lookup and the tuning-side search
    share one object: the bucket mapper has to be identity-stable or the
    autotuner's profile lookup degenerates.

    Only the batch dimension is dynamic. Hidden stays static so it lands
    verbatim in the cache key -- the configuration follows the payload in bytes,
    and a bucketed hidden would silently reuse another payload's answer.
    """
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=batches,
                map_to_tuning_buckets=make_bucket_mapper(batches, round_map=False),
            ),
        ),
        tensor_initializers=((0, small_int_initializer),),
        # Capture is required, not preferred. Without it the profiler issues
        # each iteration separately, the host cannot keep up with a collective
        # this short, and the span it times is dominated by launch gaps -- it
        # would rank host overhead rather than kernels.
        use_cold_l2_cache=False,
        use_cuda_graph=True,
    )


def default_cache_path(world_size: int) -> str:
    """Where tuned configurations are persisted.

    World size is in the filename as well as the cache key so that a TP4 and a
    TP8 job on the same host never contend for one file.
    """
    import pathlib

    override = os.getenv("FLASHINFER_AUTOTUNE_DIR")
    if override:
        base = pathlib.Path(override)
    else:
        from ..jit.env import FLASHINFER_WORKSPACE_DIR

        base = FLASHINFER_WORKSPACE_DIR / "autotune"
    return str(base / f"pcie_ipc_all_reduce_ws{world_size}.json")


def cache_key_extras(
    world_size: int,
    profile: str,
    max_blocks: int,
    max_numel: int,
    dtype: torch.dtype,
    memop_supported: bool = False,
) -> Tuple:
    """Everything the autotuner's own cache key leaves out.

    That key is only the bucketed input shapes, so without these a TP4 and a
    TP8 entry at the same shape would collide, a configuration tuned on one
    fabric would be reused on the other, and a cache written for one workspace
    size would be applied to another. ``max_numel`` matters because the epoch
    double buffer places its halves ``world_size * max_numel`` apart, so the
    best block count genuinely depends on it.

    Every field is a workspace immutable or the input dtype, which is what the
    autotuner requires: the tuple must come out the same for the caller's real
    tensors and for the ones it synthesizes.
    """
    return _cache_key_head(
        world_size, profile, max_blocks, max_numel, memop_supported
    ) + (str(dtype),)


def pack_config(config: IpcLaunchConfig) -> int:
    """Pack a configuration into one integer for a cross-rank comparison.

    All four fields: ranks that agreed on (blocks, threads, variant) but not on
    transport_blocks would launch different numbers of transporting blocks, and
    the peer barriers pair block b with block b -- so the ones past the smaller
    count would wait for a partner that never arrives.
    """
    return (
        (int(config.variant) << 48)
        | (int(config.transport_blocks) << 32)
        | (int(config.blocks) << 16)
        | int(config.threads)
    )


# How many candidates the ranking pass keeps. A fixed count, not a "within Nx of
# the best" rule: a threshold makes the number of survivors a function of
# measured floats, and ranks whose lists come out different lengths do not pick
# different kernels, they deadlock in the autotuner's per-tactic timing
# reduction. The pass only has to cut the tail, and the count is generous
# because the cost of being wrong is asymmetric -- too many survivors costs
# tuning time, too few loses a configuration the search can then never choose.
TUNE_SURVIVORS = 48

# Rounds of the ranking pass. One round separates a hopeless candidate from a
# contender but not the contenders from each other, and that matters: with a
# single sample the *set* of survivors is itself a random variable, so the
# fine-ranking sees a different candidate set each run and the final choice
# moves with it. The extra rounds cost a small fraction of the launches the
# survivors then get.
#
# The score is the minimum over the rounds, then MAX-reduced across ranks.
# Minimum because what is estimated is how fast the configuration can go, and a
# sample is only ever inflated -- by a neighbour, a clock excursion, a queued
# launch -- never deflated. MAX because every rank has to truncate to the same
# list: ranks returning different numbers of tactics deadlock in a kernel that
# spins with no timeout. No number of rounds settles a choice between candidates
# that differ by less than the capture domain resolves; those remedies -- pinned
# clocks, a larger repeat, a longer capture window for small shapes -- live
# outside get_valid_tactics.
TUNE_RANK_ROUNDS = 3


def reduce_timings(times: "torch.Tensor", group) -> "torch.Tensor":
    """Make every rank rank the candidates identically.

    ``MAX`` because a collective costs what its slowest rank costs; a mean would
    also make the ranks agree, but it would agree on a number no rank observed.

    Agreement is the load-bearing part, not the statistic: the truncated list
    this feeds decides how many times each rank enters the profiler, and the
    autotuner's timing reduction has no check for that. TUNE_SURVIVORS says
    what a rank that truncates to a different length costs the group.
    """
    dist.all_reduce(times, op=dist.ReduceOp.MAX, group=group)
    return times


def reduce_verdict(wrong: "torch.Tensor", group) -> "torch.Tensor":
    """Make every rank agree on which candidates computed the wrong answer.

    ``MAX`` over a per-candidate "was wrong" flag, which is the same decision
    as ``MIN`` over "was right": one rank seeing a mismatch condemns the
    candidate everywhere. A rank-local verdict would let ranks profile
    different candidate sets, and the autotuner's timing reduction then
    deadlocks on the first divergence.

    Corruption is not necessarily uniform across ranks: the cross-island race
    this protocol can produce leaves some of them clean, so a rank-local verdict
    can miss it entirely.

    Factored out so a test can assert the operator without a GPU.
    """
    dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
    return wrong


MAX_GENERATED_BUCKETS = 14


def generate_tune_batches(
    hidden: int, max_numel: int, elem_size: int = 2
) -> Tuple[int, ...]:
    """A bucket ladder covering everything the workspace admits at this hidden.

    The default ``TUNE_BATCHES`` stops far below the payload a prefill-sized
    ``max_numel`` admits, so every large shape would be served by a measurement
    taken at a much smaller one -- and the configuration does change with the
    payload, the sub-chunk depth chosen at the bottom is not the one that wins
    at the top. Nothing warned, because ``tune_batches`` and ``max_numel`` are
    independent constructor arguments and only one of them is visible at the
    ``tune()`` call.

    So the ladder is derived from ``max_numel`` rather than fixed. Shape:

    * dense at the bottom, where the winning kernel genuinely moves from bucket
      to bucket;
    * dense again through the crossover, where the decision is which *data
      plane* to use and getting it wrong is the expensive mistake;
    * sparse above it -- geometric, so the count stays bounded -- where the
      bandwidth curve is flat, but not absent: the sub-chunk depth still moves
      across the largest payloads.

    ``MAX_GENERATED_BUCKETS`` caps the result, so an unexpectedly large
    ``max_numel`` cannot turn tuning into an all-night job. The bottom is
    thinned first: the decode end is the cheap end to measure, and the one with
    the most redundancy once the winner has settled.
    """
    # No rounding of `top` itself: the whole-pack constraint is on
    # ``batch * hidden``, not on the batch count, and it is already enforced by
    # _admits. Rounding here to a multiple of the pack size sent every
    # workspace whose largest batch is under eight to an empty ladder.
    top = int(max_numel) // int(hidden)
    if top < 1:
        return ()

    ladder = [1, 2, 4, 8, 16, 32, 64, 128]
    b = 256
    while b < top:
        ladder.append(b)
        b *= 4
    ladder.append(top)

    seen = sorted({b for b in ladder if 1 <= b <= top})
    if len(seen) > MAX_GENERATED_BUCKETS:
        head, tail = (
            seen[: len(seen) - MAX_GENERATED_BUCKETS],
            seen[-MAX_GENERATED_BUCKETS:],
        )
        seen = [head[0]] + tail if head else tail
    return tuple(seen)


def tuned_batches_for(
    hidden: int, batches: Tuple[int, ...], max_numel: int
) -> Tuple[int, ...]:
    """Drop buckets that would exceed the workspace at this hidden size."""
    return tuple(b for b in batches if b * hidden <= max_numel)


def warn_no_tune_group(what: str = "all-reduce", stacklevel: int = 2) -> None:
    """Say why a tuning session left this collective untuned.

    Raised from the places that reach the same dead end -- a runner, when the
    autotuner does ask it for candidates, and the workspace, when it declines
    to ask at all. The generic "nothing is tuned" advice does not fit here: the
    caller *is* tuning, so telling them to tune is a dead end. What they are
    missing is the reduction group, and that is what this names. ``what`` says
    which of the two searches declined, since they are tuned in one pass and
    only one of them may be affected.
    """
    warnings.warn(
        f"PCIe IPC {what} skipped autotuning: no matching "
        "autotune process group is installed on every rank. Call "
        "PcieIpcAllReduceWorkspace.tune(), or install one with "
        "set_autotune_process_group() before entering autotune().",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


class PcieIpcAllReduceRunner(TunableRunner):
    """Adapts the all-reduce to the autotuner, and screens candidates first.

    One instance per workspace, built once and kept: the autotuner puts
    ``hash(runner)`` in its in-memory cache key, so a fresh instance per call
    would miss every entry and re-tune.
    """

    def __init__(self, workspace) -> None:
        # A weak-ish coupling on purpose: the runner needs the raw launch and
        # the group, not the public API, whose admission checks would run once
        # per candidate and whose tracing decorator would recurse.
        self._ws = workspace
        # Named to end in _cache so the base __hash__ would skip it even if the
        # override below is ever removed.
        self._buf_cache: Dict[Tuple[Tuple[int, ...], torch.dtype], torch.Tensor] = {}

    def __hash__(self) -> int:
        # Everything that changes what this runner does, and nothing that
        # changes per call. The base implementation hashes __dict__ values and
        # would fold in the workspace object's identity, which differs between
        # processes and would defeat the persisted cache.
        ws = self._ws
        return hash(
            (
                type(self).__name__,
                *_cache_key_head(
                    ws.world_size,
                    ws.profile,
                    ws.max_blocks,
                    ws.max_numel,
                    ws.memop_supported,
                ),
            )
        )

    def get_cache_key_extras(self, inputs) -> Tuple:
        ws = self._ws
        return cache_key_extras(
            ws.world_size,
            ws.profile,
            ws.max_blocks,
            ws.max_numel,
            inputs[0].dtype,
            ws.memop_supported,
        )

    def _output_for(self, inp: torch.Tensor) -> torch.Tensor:
        key = (tuple(inp.shape), inp.dtype)
        out = self._buf_cache.get(key)
        if out is None:
            out = torch.empty_like(inp)
            self._buf_cache[key] = out
        return out

    def _table_config(self, inp: torch.Tensor) -> Optional[IpcLaunchConfig]:
        return self._ws.launch_config(inp)

    def can_profile(self, device) -> bool:
        """Whether a real search is safe, as a group decision.

        Reduced rather than read locally because the answer decides how many
        times each rank enters the profiler. Ranks that search different numbers
        of candidates do not disagree, they deadlock.
        """
        from ..autotuner import get_autotune_process_group

        group = get_autotune_process_group()
        ok = group is not None and dist.get_world_size(group) == self._ws.world_size
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._ws.group)
        return bool(flag.item())

    def get_valid_tactics(self, inputs, profile) -> List:
        """Candidates that computed the right answer, in a group-agreed order.

        This is the gate the autotuner does not have. It selects by ``argmin``
        on wall time and never inspects an output, while this kernel family's
        characteristic failure -- a sentinel poll returning stale data rather
        than waiting -- is wrong *and* fast. Screening here rather than during
        profiling keeps the verdict's collective out of the timed window, and
        costs one launch per candidate on a cache miss only.

        Cardinality is the hazard. Every rank must issue exactly these launches
        in exactly this order; an early return between the first launch and the
        verdict reduction leaves peers spinning inside a kernel this rank never
        issued, with no timeout. Hence: barrier first, every buffer allocated
        before the loop, and a loop body that does not allocate, synchronise
        with the host, or branch.
        """
        inp = inputs[0]
        ws = self._ws
        table_config = self._table_config(inp)
        if table_config is None:
            # The autotuner is being asked about a shape the kernels cannot run
            # at all. Nothing to choose between; the caller falls back.
            return [TABLE_TACTIC]

        if not self.can_profile(inp.device):
            # Tuning mode is process-global, so this op can be swept by a
            # caller that only meant to tune its GEMMs. Without a reduction
            # over the candidate timings the ranks would argmin independently
            # and pick different kernels, which this protocol does not survive.
            # Offering only the seed degrades that into a no-op.
            warn_no_tune_group(stacklevel=3)
            return [TABLE_TACTIC]

        tactics = candidate_tactics(
            ws.world_size,
            ws.max_blocks,
            numel=inp.numel(),
            elem_size=inp.element_size(),
            profile=ws.profile,
            memop_supported=ws.memop_supported,
        )
        configs = [table_config] + [tactic_to_config(t) for t in tactics]

        ref = inp.clone()
        dist.all_reduce(ref, group=ws.group)
        out = self._output_for(inp)
        wrong = torch.zeros(len(configs), dtype=torch.int32, device=inp.device)
        # Allocated before the loop, like every other buffer here: the loop body
        # must not allocate. Timing the launches costs nothing extra: this pass
        # already runs every candidate once.
        starts = [
            [torch.cuda.Event(enable_timing=True) for _ in configs]
            for _ in range(TUNE_RANK_ROUNDS)
        ]
        ends = [
            [torch.cuda.Event(enable_timing=True) for _ in configs]
            for _ in range(TUNE_RANK_ROUNDS)
        ]

        # One launch per distinct variant first. The first launch of a kernel
        # loads its module, which shows up as an order-of-magnitude outlier and
        # would demote whichever candidate happened to be that variant's first.
        # Per variant, not per candidate: block and thread counts do not pull in
        # a new module.
        seen_variants = set()
        for config in configs:
            if config.variant not in seen_variants:
                seen_variants.add(config.variant)
                ws._launch(inp, out, config)

        dist.barrier(group=ws.group)
        for r in range(TUNE_RANK_ROUNDS):
            for i, config in enumerate(configs):
                # A kernel that leaves part of the payload unwritten would
                # otherwise show the previous candidate's correct result.
                out.fill_(float("nan"))
                starts[r][i].record()
                ws._launch(inp, out, config)
                ends[r][i].record()
                if r == 0:
                    # Enqueued after the closing event: inside the span, the
                    # ranking would be timing a full-payload compare too.
                    wrong[i] = torch.ne(out, ref).any()
        reduce_verdict(wrong, ws.group)

        verdict = wrong.tolist()
        if verdict[0]:
            # The seed computing the wrong answer is not something to route
            # around: it is what every untuned shape and every cache miss falls
            # back to. The verdict is group-wide, so
            # every rank raises together and the group unwinds cleanly.
            raise RuntimeError(
                "the seed configuration for shape "
                f"{tuple(inp.shape)} ({table_config}) does not match a "
                "reference all-reduce; refusing to tune on top of it"
            )
        # Reading the events synchronises with the device, which is why it
        # happens here and not in the loop above.
        torch.cuda.synchronize(inp.device)
        times = torch.tensor(
            [
                min(
                    starts[r][i].elapsed_time(ends[r][i])
                    for r in range(TUNE_RANK_ROUNDS)
                )
                for i in range(len(configs))
            ],
            dtype=torch.float64,
            device=inp.device,
        )
        reduce_timings(times, ws.group)
        scores = times.tolist()

        survivors = [
            (t, scores[i + 1])
            for i, (t, bad) in enumerate(zip(tactics, verdict[1:], strict=True))
            if not bad
        ]
        # Sorted by measured time, tie-broken on the tactic itself so the order
        # is a function of group-identical inputs alone.
        survivors.sort(key=lambda ts: (ts[1], ts[0]))
        kept = [t for t, _ in survivors[:TUNE_SURVIVORS]]
        # The seed stays first: the autotuner breaks ties toward the earlier
        # element, so this is what makes an exact tie resolve to it.
        return [TABLE_TACTIC] + kept

    def forward(
        self, inputs, tactic=TABLE_TACTIC, do_preparation: bool = False, **kwargs
    ):
        inp = inputs[0]
        out = self._output_for(inp)
        if do_preparation:
            # Buffer now allocated; launching here would make the call counts
            # depend on whether the autotuner decided to prepare.
            return out
        table_config = self._table_config(inp)
        if table_config is None:
            raise RuntimeError(
                f"shape {tuple(inp.shape)} is not one the kernels support; "
                "the tuner must not have been asked about it"
            )
        config = resolve_tuned_config(
            table_config, tactic, self._ws.world_size, self._ws.max_blocks
        )
        self._ws._launch(inp, out, config)
        return out


class PcieIpcFusedRmsNormRunner(PcieIpcAllReduceRunner):
    """Adapts the fused all-reduce to the autotuner.

    Inherits the plain runner's bookkeeping -- the hash, the cache-key extras
    and the group-searchability check are the same question -- and replaces
    what a second output and two extra inputs change.

    ``residual_in`` and ``rms_gamma`` are built here rather than taken from
    ``inputs``. The fusion adds the residual once, on whichever rank owns the
    pack, so a group whose ranks disagree about it computes a result stitched
    together from several residuals; the autotuner's initializer draws from an
    unseeded generator, which is right for the contribution and wrong for these
    two. Seeded here, so every rank profiles the same problem.
    """

    # Same tolerance the correctness test uses. The residual is compared
    # exactly -- small-integer inputs make the group sum exact -- while the
    # normalisation goes through rsqrt and cannot be.
    _NORM_RTOL = 0.05
    _NORM_ATOL = 0.15
    _SHARED_SEED = 0

    def __init__(self, workspace) -> None:
        super().__init__(workspace)
        self._fused_cache: Dict[Tuple[Tuple[int, ...], torch.dtype], Tuple] = {}

    def get_cache_key_extras(self, inputs) -> Tuple:
        """The plain head, plus what versions this search and not the other.

        Both searches share a workspace head and are told apart by op name
        alone, so a fused-only invalidation has nowhere else to live. Appending
        rather than replacing keeps every field the plain key already carries --
        world size, profile, grid bound, capacity, dtype -- which all still
        decide what a fused tactic means.
        """
        return super().get_cache_key_extras(inputs) + (PCIE_IPC_FUSED_TUNE_VERSION,)

    def __hash__(self) -> int:
        return hash((super().__hash__(), PCIE_IPC_FUSED_TUNE_VERSION))

    def _buffers_for(self, inp: torch.Tensor) -> Tuple:
        """(residual_in, gamma, residual_out, norm_out) for a shape, cached.

        Allocated once per shape because get_valid_tactics() must not allocate
        inside its candidate loop: every rank has to issue exactly the same
        launches in the same order, and an allocation there could fail on one
        rank alone.
        """
        key = (tuple(inp.shape), inp.dtype)
        cached = self._fused_cache.get(key)
        if cached is not None:
            return cached
        hidden = inp.shape[-1]
        generator = torch.Generator(device=inp.device).manual_seed(self._SHARED_SEED)
        residual_in = torch.randint(
            0,
            INIT_MAX_VALUE,
            tuple(inp.shape),
            device=inp.device,
            dtype=torch.int32,
            generator=generator,
        ).to(inp.dtype)
        gamma = torch.randint(
            1,
            5,
            (hidden,),
            device=inp.device,
            dtype=torch.int32,
            generator=generator,
        ).to(inp.dtype)
        buffers = (
            residual_in,
            gamma,
            torch.empty_like(inp),
            torch.empty_like(inp),
        )
        self._fused_cache[key] = buffers
        return buffers

    def _table_config(self, inp: torch.Tensor) -> Optional[IpcLaunchConfig]:
        return self._ws.fused_launch_config(inp)

    def get_valid_tactics(self, inputs, profile) -> List:
        """Candidates that computed the right answer, in a group-agreed order.

        Two verdicts per candidate rather than one. The residual carries the
        collective and is exact, so it catches a transport that took stale data
        from a sentinel slot; the normalisation is what the row reduction
        produces, and a candidate can get the residual right while its
        denominator is wrong. Neither alone covers the kernel.

        The cardinality rule from the plain runner holds unchanged: barrier
        first, every buffer allocated before the loop, and a loop body that does
        not branch.
        """
        inp = inputs[0]
        ws = self._ws
        table_config = self._table_config(inp)
        if table_config is None:
            return [TABLE_TACTIC]
        if not self.can_profile(inp.device):
            warn_no_tune_group("fused rmsnorm", stacklevel=3)
            return [TABLE_TACTIC]

        hidden = int(inp.shape[-1])
        tactics = fused_candidate_tactics(
            ws.world_size,
            hidden,
            ws.elem_size,
            ws.max_blocks,
            ws.sm_count,
            numel=inp.numel(),
        )
        configs = [table_config] + [tactic_to_config(t) for t in tactics]

        residual_in, gamma, residual_out, norm_out = self._buffers_for(inp)
        reduced = inp.clone()
        dist.all_reduce(reduced, group=ws.group)
        ref_residual = (reduced + residual_in).to(inp.dtype)
        pre = ref_residual.to(torch.float64)
        ref_norm = (
            pre
            * torch.rsqrt(pre.pow(2).mean(dim=-1, keepdim=True) + _TUNE_EPS)
            * gamma.to(torch.float64)
        )
        wrong = torch.zeros(len(configs), dtype=torch.int32, device=inp.device)

        dist.barrier(group=ws.group)
        for i, config in enumerate(configs):
            # A kernel that leaves part of the payload unwritten would
            # otherwise show the previous candidate's correct result.
            residual_out.fill_(float("nan"))
            norm_out.fill_(float("nan"))
            ws._launch_fused(
                inp, residual_in, gamma, residual_out, norm_out, _TUNE_EPS, config
            )
            wrong[i] = (
                torch.ne(residual_out, ref_residual).any()
                | torch.isclose(
                    norm_out.to(torch.float64),
                    ref_norm,
                    rtol=self._NORM_RTOL,
                    atol=self._NORM_ATOL,
                )
                .logical_not()
                .any()
            )
        reduce_verdict(wrong, ws.group)

        verdict = wrong.tolist()
        if verdict[0]:
            raise RuntimeError(
                "the seed configuration for shape "
                f"{tuple(inp.shape)} ({table_config}) does not match a "
                "reference fused all-reduce; refusing to tune on top of it"
            )
        survivors = zip(tactics, verdict[1:], strict=True)
        return [TABLE_TACTIC] + [t for t, bad in survivors if not bad]

    def forward(
        self, inputs, tactic=TABLE_TACTIC, do_preparation: bool = False, **kwargs
    ):
        inp = inputs[0]
        residual_in, gamma, residual_out, norm_out = self._buffers_for(inp)
        if do_preparation:
            return norm_out
        table_config = self._table_config(inp)
        if table_config is None:
            raise RuntimeError(
                f"shape {tuple(inp.shape)} is not one the fused kernels "
                "support; the tuner must not have been asked about it"
            )
        config = resolve_tuned_fused_config(
            table_config,
            tactic,
            self._ws.world_size,
            self._ws.max_blocks,
            int(inp.shape[-1]),
            self._ws.elem_size,
            self._ws.sm_count,
            inp.numel(),
        )
        self._ws._launch_fused(
            inp, residual_in, gamma, residual_out, norm_out, _TUNE_EPS, config
        )
        return norm_out


__all__ = [
    "PCIE_IPC_CUSTOM_OP",
    "PCIE_IPC_FUSED_CUSTOM_OP",
    "PcieIpcAllReduceRunner",
    "PcieIpcFusedRmsNormRunner",
    "fused_candidate_tactics",
    "resolve_tuned_fused_config",
    "PCIE_IPC_TUNE_VERSION",
    "PCIE_IPC_FUSED_TUNE_VERSION",
    "TABLE_TACTIC",
    "TUNE_BATCHES",
    "TUNE_BLOCKS",
    "TUNE_REPEAT",
    "TUNE_THREADS",
    "TUNE_WARMUP",
    "cache_key_extras",
    "candidate_tactics",
    "config_to_tactic",
    "default_cache_path",
    "pack_config",
    "pcie_ipc_tuning_config",
    "reduce_verdict",
    "resolve_tuned_config",
    "small_int_initializer",
    "tactic_to_config",
    "tuned_batches_for",
]
