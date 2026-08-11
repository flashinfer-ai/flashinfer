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

The hand-written table in :mod:`~flashinfer.comm.pcie_ipc_policy` fits one
machine. This module measures the same choice instead, over the launch
configurations the dispatch can actually reach.

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

The table keeps three jobs: it decides which shapes are supported at all, it is
tactic ``-1``, and it is the fallback whenever a tuned answer cannot be used.
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
from .pcie_ipc_policy import (
    MAX_BLOCKS,
    IpcLaunchConfig,
    IpcVariant,
    _is_launchable,
)

# Baked into every persisted cache key, so renaming it silently invalidates
# every cache file rather than mis-resolving one.
PCIE_IPC_CUSTOM_OP = "flashinfer::pcie_ipc_all_reduce"

# Bump when a variant's meaning, the scratch-region assignment, or the
# candidate encoding changes. The autotuner's own metadata records library and
# driver versions but nothing about this op, and a dev checkout does not move
# the FlashInfer version.
PCIE_IPC_TUNE_VERSION = 1

# Must be able to name every configuration the shipping tables produce, or
# tuning could only reach the table through tactic -1 rather than explore around
# it -- hence the entries that are not powers of two.
TUNE_BLOCKS: Tuple[int, ...] = (1, 2, 4, 8, 12, 16, 32, 64, 96, 128)
TUNE_THREADS: Tuple[int, ...] = (64, 128, 256, 512, 1024)

# Batch buckets. Floor semantics, so a bucket is always a batch the tuner
# actually measured. Matches the benchmark's default sweep.
TUNE_BATCHES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

# Higher than the library defaults, which time too short a span to resolve
# candidates for a collective of this scale.
TUNE_WARMUP = 10
TUNE_REPEAT = 50

# Reference tactic. The autotuner reserves -1 for "the fallback that implements
# any shape"; here that is the hand-written table's own answer.
TABLE_TACTIC = -1

# Inputs are drawn from [0, INIT_MAX_VALUE) so the group sum stays integral and
# exactly representable, which is what lets verification use a zero tolerance
# despite the kernels summing in a different order than NCCL.
INIT_MAX_VALUE = 16


def candidate_tactics(
    world_size: int,
    max_blocks: int = MAX_BLOCKS,
    blocks: Tuple[int, ...] = TUNE_BLOCKS,
    threads: Tuple[int, ...] = TUNE_THREADS,
) -> Tuple[Tuple[int, int, int], ...]:
    """Every launch configuration the dispatch can reach, as tactics.

    A pure function of group-identical arguments, so every rank derives the
    same list in the same order -- which the autotuner's collective profiling
    requires and cannot check.
    """
    return _candidate_tactics_cached(world_size, max_blocks, blocks, threads)


@lru_cache(maxsize=None)
def _candidate_tactics_cached(world_size, max_blocks, blocks, threads):
    out = []
    for variant in IpcVariant:
        for b in blocks:
            for t in threads:
                if _is_launchable(
                    world_size, IpcLaunchConfig(b, t, variant), max_blocks
                ):
                    out.append((int(variant), b, t))
    return tuple(out)


def config_to_tactic(config: IpcLaunchConfig) -> Tuple[int, int, int]:
    """Encode a configuration as a tactic.

    Plain ints, because a tactic has to survive a JSON round-trip: the
    autotuner writes ``[0, 32, 128]`` and reads back ``(0, 32, 128)``.
    Self-describing rather than an index into :func:`candidate_tactics`, so
    editing the grid cannot repoint a persisted entry at a different kernel.
    """
    return (int(config.variant), int(config.blocks), int(config.threads))


def tactic_to_config(tactic: Sequence[int]) -> IpcLaunchConfig:
    """Decode a tactic. Raises ``ValueError`` on anything malformed."""
    if len(tactic) != 3:
        raise ValueError(f"expected a 3-element tactic, got {tactic!r}")
    variant, blocks, threads = (int(v) for v in tactic)
    try:
        return IpcLaunchConfig(blocks, threads, IpcVariant(variant))
    except ValueError as exc:
        raise ValueError(f"tactic {tactic!r} names no variant: {exc}") from exc


def resolve_tuned_config(
    table_config: IpcLaunchConfig,
    tactic,
    world_size: int,
    max_blocks: int,
) -> IpcLaunchConfig:
    """Turn a tactic into a configuration, falling back to the table.

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
    verbatim in the cache key -- the table is keyed on an exact hidden size and
    a bucketed one would silently reuse another shape's answer.
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
    return (
        PCIE_IPC_TUNE_VERSION,
        int(world_size),
        str(profile),
        int(max_blocks),
        int(max_numel),
        str(dtype),
    )


def pack_config(config: IpcLaunchConfig) -> int:
    """Pack a configuration into one integer for a cross-rank comparison."""
    return (
        (int(config.variant) << 32) | (int(config.blocks) << 16) | int(config.threads)
    )


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


def tuned_batches_for(
    hidden: int, batches: Tuple[int, ...], max_numel: int
) -> Tuple[int, ...]:
    """Drop buckets that would exceed the workspace at this hidden size."""
    return tuple(b for b in batches if b * hidden <= max_numel)


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
                PCIE_IPC_TUNE_VERSION,
                ws.world_size,
                ws.profile,
                ws.max_blocks,
                ws.max_numel,
            )
        )

    def get_cache_key_extras(self, inputs) -> Tuple:
        ws = self._ws
        return cache_key_extras(
            ws.world_size, ws.profile, ws.max_blocks, ws.max_numel, inputs[0].dtype
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

    def _searchable(self, device) -> bool:
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
            # The autotuner is being asked about a shape the table does not
            # claim. Nothing to choose between; the caller falls back.
            return [TABLE_TACTIC]

        if not self._searchable(inp.device):
            # Tuning mode is process-global, so this op can be swept by a
            # caller that only meant to tune its GEMMs. Without a reduction
            # over the candidate timings the ranks would argmin independently
            # and pick different kernels, which this protocol does not survive.
            # Offering only the table degrades that into a no-op.
            warnings.warn(
                "PCIe IPC all-reduce skipped autotuning: no matching "
                "autotune process group is installed on every rank. Call "
                "PcieIpcAllReduceWorkspace.tune(), or install one with "
                "set_autotune_process_group() before entering autotune().",
                RuntimeWarning,
                stacklevel=2,
            )
            return [TABLE_TACTIC]

        tactics = candidate_tactics(ws.world_size, ws.max_blocks)
        configs = [table_config] + [tactic_to_config(t) for t in tactics]

        ref = inp.clone()
        dist.all_reduce(ref, group=ws.group)
        out = self._output_for(inp)
        wrong = torch.zeros(len(configs), dtype=torch.int32, device=inp.device)

        dist.barrier(group=ws.group)
        for i, config in enumerate(configs):
            # A kernel that leaves part of the payload unwritten would
            # otherwise show the previous candidate's correct result.
            out.fill_(float("nan"))
            ws._launch(inp, out, config)
            wrong[i] = torch.ne(out, ref).any()
        reduce_verdict(wrong, ws.group)

        verdict = wrong.tolist()
        if verdict[0]:
            # The table's own configuration computing the wrong answer is not
            # something to route around: it is what every untuned shape and
            # every cache miss falls back to. The verdict is group-wide, so
            # every rank raises together and the group unwinds cleanly.
            raise RuntimeError(
                "the tuning table's own configuration for shape "
                f"{tuple(inp.shape)} ({table_config}) does not match a "
                "reference all-reduce; refusing to tune on top of it"
            )
        survivors = zip(tactics, verdict[1:], strict=True)
        return [TABLE_TACTIC] + [t for t, bad in survivors if not bad]

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
                f"shape {tuple(inp.shape)} is not in the tuning table; "
                "the tuner must not have been asked about it"
            )
        config = resolve_tuned_config(
            table_config, tactic, self._ws.world_size, self._ws.max_blocks
        )
        self._ws._launch(inp, out, config)
        return out


__all__ = [
    "PCIE_IPC_CUSTOM_OP",
    "PcieIpcAllReduceRunner",
    "PCIE_IPC_TUNE_VERSION",
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
