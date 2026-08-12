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
"""

import functools
import hashlib
import os
import warnings
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ..api_logging import flashinfer_api
from ..trace.templates.comm import pcie_ipc_all_reduce_trace
from ..jit.comm import gen_pcie_ipc_comm_module
from ..utils import register_custom_op
from .cuda_ipc import create_shared_buffer, free_shared_buffer
from .pcie_ipc_policy import IpcLaunchConfig, get_pcie_ipc_launch_config
from .pcie_ipc_topology import resolve_pcie_ipc_profile
from .pcie_ipc_tuning import (
    PCIE_IPC_CUSTOM_OP,
    TUNE_BATCHES,
    TUNE_REPEAT,
    TUNE_WARMUP,
    PcieIpcAllReduceRunner,
    cache_covers_workspace,
    default_cache_path,
    pack_config,
    pcie_ipc_tuning_config,
    resolve_tuned_config,
    tuned_batches_for,
)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
# Mirrors the launcher, which hard-checks a 2-byte element size: the kernels
# address whole 16-byte packs and are instantiated for half and nv_bfloat16
# only. Rejecting here turns that into an unsupported shape rather than an
# ICHECK partway through a collective.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@functools.cache
def get_pcie_ipc_comm_module():
    module = gen_pcie_ipc_comm_module().build_and_load()

    @register_custom_op("flashinfer::pcie_ipc_workspace_size", mutates_args=[])
    def workspace_size(
        world_size: int, max_numel: int, elem_size: int, max_blocks: int
    ) -> int:
        return module.pcie_ipc_workspace_size(
            world_size, max_numel, elem_size, max_blocks
        )

    @register_custom_op("flashinfer::pcie_ipc_init", mutates_args=["ipc_ptrs"])
    def init(
        ipc_ptrs: List[int],
        rank: int,
        max_numel: int,
        elem_size: int,
        max_blocks: int,
    ) -> int:
        return module.pcie_ipc_init(ipc_ptrs, rank, max_numel, elem_size, max_blocks)

    @register_custom_op("flashinfer::pcie_ipc_dispose", mutates_args=["handle"])
    def dispose(handle: int) -> None:
        module.pcie_ipc_dispose(handle)

    @register_custom_op("flashinfer::pcie_ipc_all_reduce", mutates_args=["out"])
    def all_reduce(
        handle: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        blocks: int,
        threads: int,
        variant: int,
        enable_pdl: bool,
    ) -> None:
        module.pcie_ipc_all_reduce(
            handle, inp, out, blocks, threads, variant, enable_pdl
        )

    return SimpleNamespace(
        workspace_size=workspace_size,
        init=init,
        dispose=dispose,
        all_reduce=all_reduce,
    )


class PcieIpcAllReduceWorkspace:
    """Shared workspace for the PCIe IPC all-reduce.

    Allocates one slab per rank, shares it over CUDA IPC, and binds it to the
    kernels. The workspace is sized once and cannot grow, so ``max_numel`` must
    cover the largest collective that will be issued; anything larger must fall
    back to another backend.

    This is a **collective**, and an unusually strict one. The kernels spin on
    peer flags with no timeout and no metadata exchange, so every rank must
    issue the same sequence of calls, with the same shape, dtype and launch
    configuration, in the same order. A rank that skips a call, reorders two,
    or passes a different explicit ``config`` does not get an error -- the
    group hangs, or worse, one rank reads a neighbour's partial sums as if they
    were finished. :meth:`launch_config` is a pure function of shape, dtype and
    the workspace's own immutable attributes precisely so that every rank
    derives the same answer without having to agree on one at runtime; passing
    ``config`` explicitly moves that obligation to the caller.

    One workspace serves **one CUDA stream**. Its epoch and arrival counters
    assume the calls sharing it are totally ordered, which stream order gives
    and concurrent streams do not; the second stream is rejected. Build a
    separate workspace per stream.

    Size ``max_numel`` to the real workload rather than to a round number. The
    epoch double buffer places its two halves ``world_size * max_numel``
    elements apart, so an oversized workspace spreads them further than the
    payload needs and costs measurable time at small batch. The multiplier is
    the world size, not 2 -- rounding ``max_numel`` up by 4x at 8 ranks moves
    the halves 32x the payload apart.

    Parameters
    ----------
    group : ProcessGroup
        Process group whose ranks share the workspace. Every rank must build
        the workspace with identical arguments.
    max_numel : int
        Largest element count that will be all-reduced.
    dtype : torch.dtype
        bfloat16 or float16. Only the element *size* is binding, so one
        workspace serves both.
    max_blocks : int
        Upper bound on the block count any launch may request. Sizes the
        barrier and epoch slots.
    profile : str, optional
        Force the interconnect label (``"rootcplx"`` or ``"pcieswitch"``)
        instead of probing for it. The label does not pick a kernel; it
        partitions the tune cache so two topologies do not read each other's
        measurements. Probing is collective and runs before any allocation.
    tune_cache : str, optional
        Where tuned configurations are read from at construction and written by
        :meth:`tune`. Defaults to ``FLASHINFER_AUTOTUNE_DIR`` (or the workspace
        directory). Give the same path to both, or a tuned result will not be
        found by the next process.

    Launch configurations start from a seed default that is workable rather
    than fast (see :mod:`~flashinfer.comm.pcie_ipc_policy`). Tune once to
    replace it with measurements from this machine; the result is persisted and
    later processes pick it up when the workspace is built. Tuning never changes
    which shapes are supported, only which kernel a supported shape runs.

    Examples
    --------
    >>> ws = PcieIpcAllReduceWorkspace(group=tp_group, max_numel=max_tokens * hidden)
    >>> if ws.supports(x):
    ...     out = ws.all_reduce(x)
    >>> ws.destroy()

    Tuning, once per machine:

    >>> ws.tune([hidden])  # collective; every rank calls it
    """

    def __init__(
        self,
        group: ProcessGroup,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        max_blocks: int = 128,
        profile: Optional[str] = None,
        tune_batches: Sequence[int] = TUNE_BATCHES,
        tune_cache: Optional[str] = None,
    ) -> None:
        # Construction is a staged transaction. Every rank must execute the same
        # sequence of collectives, so a rank that finds a problem does NOT raise
        # where it finds it -- it records an outcome and raises only at the next
        # gather, together with everyone else. Raising early would leave the
        # peers blocked in a collective that their partner has already left.
        self._ipc_ptrs: Optional[List[int]] = None
        self._handle: Optional[int] = None
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        # Bound on first executing use; see _check_stream.
        self._stream: Optional[torch.cuda.Stream] = None
        self.elem_size = 0
        self.max_numel = max_numel
        self.max_blocks = max_blocks
        self.profile = ""
        self.profile_reason = ""
        # Resolved launch configurations, keyed exactly. Consulted before any
        # AutoTuner call because even a pure cache lookup there takes a global
        # lock, which is real overhead at this operator's scale.
        self._tuned: Dict[Tuple[int, int, torch.dtype], IpcLaunchConfig] = {}
        self._runner: Optional[PcieIpcAllReduceRunner] = None
        self._tune_group: Optional[ProcessGroup] = None
        self._tune_batches = tuple(int(b) for b in tune_batches)
        self._tune_cache = tune_cache or default_cache_path(self.world_size)
        self._tune_cache_exists = False
        self._tuned_configs_loaded = False
        self._warned_untuned = False

        # --- stage 1: local validation, encoded rather than raised -----------
        error: Optional[str] = None
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            error = (
                f"world size {self.world_size} unsupported; "
                f"expected one of {_SUPPORTED_WORLD_SIZES}"
            )
        elif dtype not in _SUPPORTED_DTYPES:
            error = f"dtype {dtype} unsupported; expected one of {_SUPPORTED_DTYPES}"
        else:
            self.elem_size = torch.empty((), dtype=dtype).element_size()
            pack_elems = 16 // self.elem_size
            if max_numel <= 0 or max_numel % pack_elems != 0:
                # The kernels address the scratch in 16-byte packs, so a
                # capacity that is not a whole number of packs is rejected by
                # the launcher on every call. Catch it here instead.
                error = (
                    f"max_numel must be a positive multiple of {pack_elems} "
                    f"for {dtype}, got {max_numel}"
                )
            elif max_blocks <= 0:
                error = f"max_blocks must be positive, got {max_blocks}"

        # Layout must be identical on every rank, or one of them reads a peer
        # slab at the wrong offsets. Gather the config alongside the outcome so
        # a single collective settles both.
        local = {
            "error": error,
            "max_numel": max_numel,
            "elem_size": self.elem_size,
            "max_blocks": max_blocks,
            "profile": profile,
            # Buckets pick which shape a tuned entry is reused for, so ranks
            # that disagree would resolve different configurations.
            "tune_batches": self._tune_batches,
            "tune_cache": self._tune_cache,
        }
        self._joint_check(local, "validating arguments")

        # --- stage 2: topology, then module + workspace size -----------------
        # Both before any allocation, so an unsupported topology or a failed
        # JIT build costs nothing to unwind.
        try:
            decision = resolve_pcie_ipc_profile(group, requested=profile)
            self.profile = decision.profile
            self.profile_reason = decision.reason
            module = get_pcie_ipc_comm_module()
            nbytes = module.workspace_size(
                self.world_size, max_numel, self.elem_size, max_blocks
            )
        except Exception as e:  # noqa: BLE001 - re-raised jointly below
            nbytes = 0
            self._joint_check({"error": f"{type(e).__name__}: {e}"}, "preparing")
            raise  # unreachable: _joint_check raises on every rank
        self._joint_check({"error": None}, "preparing")

        # --- stage 3: allocate and share, then bind --------------------------
        # NOTE: create_shared_buffer() runs its own all_gather_object and
        # barrier internally. A failure *inside* it leaves the group in a state
        # this constructor cannot repair; that is a property of the shared
        # helper, not something worked around here.
        self._ipc_ptrs = create_shared_buffer(nbytes, group=group)
        bind_error: Optional[str] = None
        try:
            self._handle = module.init(
                self._ipc_ptrs, self.rank, max_numel, self.elem_size, max_blocks
            )
            # init() zeroes this rank's slab; no peer may push into it until
            # every rank has done so.
            torch.cuda.synchronize(self.device)
        except Exception as e:  # noqa: BLE001 - re-raised jointly below
            bind_error = f"{type(e).__name__}: {e}"

        # Whether to tear down is a group decision: the cleanup itself contains
        # barriers, so one rank must never enter it alone.
        try:
            self._joint_check({"error": bind_error}, "binding the workspace")
        except Exception:
            self.destroy()
            raise

        # --- stage 4: tuned configurations, if any have been persisted -------
        # Loaded once, here, and never reloaded: a rank that picks up a file
        # update its peers have not seen would choose a different kernel, and
        # the group hangs rather than erroring.
        try:
            self._init_tuning()
        except Exception:
            self.destroy()
            raise
        dist.barrier(group=group)

    def _joint_check(self, local: dict, what: str) -> None:
        """Gather per-rank outcomes and fail the whole group, or none of it.

        Raises the same error on every rank, so the caller can rely on all
        ranks taking the same branch afterwards.
        """
        gathered: List[Optional[dict]] = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.group)
        entries = [g for g in gathered if g is not None]

        failed = {i: g["error"] for i, g in enumerate(entries) if g.get("error")}
        if failed:
            raise ValueError(f"pcie ipc workspace failed while {what}: {failed}")

        mismatched = {
            key: [g[key] for g in entries]
            for key in local
            if key != "error" and len({repr(g[key]) for g in entries}) > 1
        }
        if mismatched:
            raise ValueError(
                "every rank must build the workspace with identical arguments, "
                f"but these differ across the group: {mismatched}"
            )

    @property
    def handle(self) -> int:
        if self._handle is None:
            raise RuntimeError("workspace has been destroyed")
        return self._handle

    def _check_stream(self) -> None:
        """Bind the workspace to one stream, and reject use from another.

        The workspace carries mutable protocol state -- the epoch that selects
        which half of the scratch a call stages through, and the arrival
        counter that commits it. Both are advanced by the kernels themselves
        and are only well defined if the calls that share this workspace are
        totally ordered. Stream order gives that; two streams do not, and
        concurrent calls would interleave their epoch reads and commits and
        silently corrupt each other.

        Capture is exempt: `torch.cuda.graph` records on a side stream but
        nothing executes, and the captured nodes form a linear chain that
        replays in order. Replaying such a graph concurrently with other calls
        on the same workspace is still unsupported and cannot be checked from
        here.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        current = torch.cuda.current_stream(self.device)
        if self._stream is None:
            self._stream = current
        elif current != self._stream:
            raise RuntimeError(
                "this workspace is already bound to "
                f"{self._stream}, but all_reduce was called on {current}. "
                "One workspace serves one stream: its epoch and arrival "
                "counters assume the calls sharing it are totally ordered. "
                "Build a second workspace for the second stream."
            )

    def rebind_stream(self) -> None:
        """Allow the next call to come from a different stream.

        The workspace rejects a second stream because it cannot tell "used
        sequentially from another stream" from "used concurrently", and only
        the latter is unsafe. A caller that knows the previous stream's work
        has completed -- because it synchronized, or recorded and waited on an
        event -- can say so here and move the binding.

        This is an assertion by the caller, not a check: calling it without
        actually ordering the two streams reintroduces the corruption it exists
        to prevent.
        """
        self._stream = None

    def launch_config(self, inp: torch.Tensor) -> Optional[IpcLaunchConfig]:
        """Seed launch configuration for ``inp``, or ``None`` if unsupported.

        The seed is a default, not a measurement -- see
        :mod:`~flashinfer.comm.pcie_ipc_policy`. :meth:`tuned_launch_config`
        is what returns a measured answer once :meth:`tune` has run.

        Depends only on shape, dtype and the workspace's own immutable
        attributes, never on rank-local state: every rank must reach the same
        answer or the collective deadlocks.

        Raises
        ------
        ValueError
            If ``inp`` is not on the workspace's device. This is deliberately
            not reported as "unsupported": a caller checking :meth:`supports`
            reads ``False`` as "use another backend", so answering ``False``
            here would turn a local bug into a silent fallback on one rank --
            and one rank taking a different branch hangs the rest.
        """
        # Checked before the workspace state so the diagnosis is the same
        # whether or not the workspace is still alive.
        if inp.device != self.device:
            raise ValueError(
                f"input is on {inp.device} but the workspace was built on {self.device}"
            )
        if self._handle is None:
            return None
        if inp.dtype not in _SUPPORTED_DTYPES:
            return None
        if inp.element_size() != self.elem_size:
            return None
        if not inp.is_contiguous() or inp.dim() == 0:
            return None
        numel = inp.numel()
        if numel > self.max_numel:
            return None
        return get_pcie_ipc_launch_config(
            self.world_size, numel, self.elem_size, self.max_blocks
        )

    def supports(self, inp: torch.Tensor) -> bool:
        """Whether the kernels can run ``inp`` at all.

        A capability question -- dtype, contiguity, workspace capacity, and
        enough payload for the reduce-scatter to give every rank a share. It
        does not mean the shape has been measured on this machine; call
        :meth:`tune` for that.

        Raises the same way :meth:`launch_config` does on a device mismatch --
        that is a caller bug, not an unsupported shape.

        Autotuning never changes this answer: it only picks a faster
        configuration for a shape that is already supported.
        """
        return self.launch_config(inp) is not None

    def _init_tuning(self) -> None:
        """Build the runner and load any persisted configurations. Collective."""
        self._runner = PcieIpcAllReduceRunner(self)
        path = self._tune_cache
        exists = os.path.isfile(path)
        # Whether the file is there has to be a group fact before anyone acts
        # on it: half a group running tuned configurations and half running the
        # seed is a hang, not a slowdown.
        self._joint_check({"error": None, "cache": exists}, "checking the tune cache")
        self._tune_cache_exists = exists
        if exists:
            from ..autotuner import AutoTuner

            AutoTuner.get().load_configs(path)
        # Settled against the loaded keys, where the answer is known, rather
        # than inferred from a miss later.
        self._tuned_configs_loaded = exists and cache_covers_workspace(
            self.world_size, self.profile, self.max_blocks, self.max_numel
        )
        self._joint_check(
            {
                "error": None,
                "digest": self._cache_digest(),
                "covers": self._tuned_configs_loaded,
            },
            "loading the tune cache",
        )

    def _warn_if_untuned(self) -> None:
        """Say once that this workspace resolved to seed configurations.

        Two causes with different fixes, so two messages: a machine nobody
        tuned, or a cache keyed for a different workspace (see
        :func:`~flashinfer.comm.pcie_ipc_tuning.cache_covers_workspace`).

        On the cold path only, so the steady state is untouched: a serving loop
        reaches this at most once per distinct shape, and the flag makes it once
        per workspace. Warning here rather than in ``__init__`` keeps it tied to
        actually using the kernels, not to building a workspace the caller may
        never route to.
        """
        if self._tuned_configs_loaded or self._warned_untuned:
            return
        self._warned_untuned = True
        if self._tune_cache_exists:
            warnings.warn(
                f"PCIe IPC all-reduce loaded {self._tune_cache} but it holds no "
                f"entry for this workspace ({self.world_size} ranks, "
                f"max_numel={self.max_numel}, max_blocks={self.max_blocks}, "
                f"profile={self.profile}); it was tuned for a different one, so "
                "every shape falls back to a seed configuration. Re-tune with "
                "this workspace's parameters, or build it with the ones the "
                "cache was written for.",
                UserWarning,
                stacklevel=4,
            )
        else:
            warnings.warn(
                "PCIe IPC all-reduce is running seed launch configurations: "
                f"nothing has been tuned for {self.world_size} ranks on this "
                f"machine ({self._tune_cache} does not exist). The seed picks a "
                "workable kernel, not a fast one. Call workspace.tune([hidden]) "
                "once per machine; the result is persisted and later processes "
                "pick it up.",
                UserWarning,
                stacklevel=4,
            )

    def _cache_digest(self) -> str:
        """Fingerprint of the tuned entries this rank will actually use.

        ``load_configs`` silently drops entries whose metadata does not match
        the machine, so "we all read the same file" is not the same as "we all
        hold the same table".
        """
        from ..autotuner import AutoTuner

        prefix = f"('{PCIE_IPC_CUSTOM_OP}'"
        tuner = AutoTuner.get()
        entries = sorted(
            (key, repr(value))
            for key, value in tuner._file_configs.items()
            if key.startswith(prefix)
        )
        return hashlib.sha256(repr(entries).encode()).hexdigest()[:16]

    def tuned_launch_config(self, inp: torch.Tensor) -> Optional[IpcLaunchConfig]:
        """Launch configuration for ``inp``, measured if one has been persisted.

        Admission is asked first and is final: a shape the kernels cannot run
        returns ``None`` here too, whatever the cache holds.

        Inside an ``autotune(True)`` context this runs the search; outside one
        it is a lookup. Same split as the other tunable ops in this library.
        """
        seed = self.launch_config(inp)
        if seed is None:
            return None
        from ..autotuner import AutoTuner

        tuner = AutoTuner.get()
        key = (inp.numel(), inp.shape[-1], inp.dtype)
        # The hot cache is skipped while tuning, so a search that has more
        # shapes to cover is not short-circuited by an earlier answer.
        if not tuner.is_tuning_mode:
            cached = self._tuned.get(key)
            if cached is not None:
                return cached
        config = self._resolve_tuned(inp, seed, tuner)
        self._tuned[key] = config
        return config

    def _resolve_tuned(
        self, inp: torch.Tensor, seed: IpcLaunchConfig, tuner
    ) -> IpcLaunchConfig:
        """Cold path: search or look up, then make the group agree."""
        hidden = inp.shape[-1]
        batch = inp.numel() // hidden
        tuning_config = pcie_ipc_tuning_config(self._tune_batches)
        if tuner.is_tuning_mode:
            _, tactic = tuner.choose_one(
                PCIE_IPC_CUSTOM_OP, [self._runner], tuning_config, [inp]
            )
        else:
            _, _, tactic, _ = tuner.search_cache(
                PCIE_IPC_CUSTOM_OP,
                [self._runner],
                ((batch, hidden),),
                tuning_config,
                inputs=[inp],
            )
        config = resolve_tuned_config(seed, tactic, self.world_size, self.max_blocks)

        # Unconditional, even when the cache missed and `config is seed`. The
        # ranks would otherwise have to agree on whether to run this collective
        # before running it, and disagreeing about that is the hang it exists
        # to prevent. It costs one small reduction per distinct shape.
        packed = pack_config(config)
        bounds = torch.tensor([packed, -packed], dtype=torch.int64, device=self.device)
        dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=self.group)
        if int(bounds[0]) != -int(bounds[1]):
            # Fall back rather than raise: the seed is a pure function, so it
            # is agreed by construction and the group stays alive.
            warnings.warn(
                "ranks resolved different tuned configurations for shape "
                f"{tuple(inp.shape)}; falling back to the seed configuration. "
                "The tune cache is inconsistent across ranks -- delete "
                f"{self._tune_cache} and re-tune.",
                RuntimeWarning,
                stacklevel=3,
            )
            return seed
        if not tuner.is_tuning_mode:
            self._warn_if_untuned()
        return config

    @flashinfer_api(trace=pcie_ipc_all_reduce_trace)
    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        config: Optional[IpcLaunchConfig] = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        """Out-of-place all-reduce.

        Parameters
        ----------
        inp : torch.Tensor
            Contiguous CUDA tensor whose byte size is a multiple of 16.
        out : torch.Tensor, optional
            Destination. Allocated when omitted.
        config : IpcLaunchConfig, optional
            Launch geometry and kernel selection. Resolved from the tune cache
            or the seed when omitted; pass one explicitly only to benchmark or
            to reach a kernel neither would choose. Ranks that disagree on it
            hang -- see the collective contract in the class docstring.
        enable_pdl : bool
            Programmatic dependent launch. **Currently rejected.** The TP8
            block kernel triggers launch completion before it writes its
            island ack and barrier flag, so a dependent kernel could start
            while this call's protocol state is still being written.

        Returns
        -------
        torch.Tensor
            The reduced tensor.

        Raises
        ------
        ValueError
            If the kernels cannot run this shape. Check :meth:`supports` first
            and fall back to another backend.
        """
        if config is None:
            config = self.tuned_launch_config(inp)
            if config is None:
                raise ValueError(
                    f"unsupported shape {tuple(inp.shape)} dtype {inp.dtype} "
                    f"at {self.world_size} ranks; check supports() first"
                )
        self._check_stream()
        # Raise rather than fall back: a device mismatch is a caller bug, and
        # silently opting this rank out would hang every other rank.
        if inp.device != self.device:
            raise ValueError(
                f"input is on {inp.device} but the workspace was built on {self.device}"
            )
        if out is None:
            out = torch.empty_like(inp)
        elif out.device != self.device:
            raise ValueError(
                f"output is on {out.device} but the workspace was built on "
                f"{self.device}"
            )
        self._launch(inp, out, config, enable_pdl)
        return out

    def _launch(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        config: IpcLaunchConfig,
        enable_pdl: bool = False,
    ) -> None:
        """Issue one collective with an explicit configuration.

        The launch without the admission, device and stream checks around it.
        Callers that have already done those -- the tuner, which sweeps many
        configurations over one validated pair of buffers -- use this so the
        checks do not run once per candidate.
        """
        get_pcie_ipc_comm_module().all_reduce(
            self.handle,
            inp,
            out,
            config.blocks,
            config.threads,
            int(config.variant),
            enable_pdl,
        )

    def tune(
        self,
        hiddens: Sequence[int],
        *,
        dtype: torch.dtype = torch.bfloat16,
        cache: Optional[str] = None,
        tune_group=None,
        warmup: int = TUNE_WARMUP,
        repeat: int = TUNE_REPEAT,
    ) -> Dict[Tuple[int, int], IpcLaunchConfig]:
        """Measure the launch configuration for every tuned shape. Collective.

        A convenience wrapper around the library's usual tuning idiom::

            with flashinfer.autotune(True, cache=path):
                for batch in batches:
                    ws.all_reduce(sample(batch))

        which also works, and does the same thing. This adds what a collective
        needs on top of it: a gloo subgroup for the timing reduction so every
        rank picks the same kernel, longer timing runs than the library default
        (the library defaults resolve too little at this scale), a check that
        every rank agrees on the arguments, and a single writer for the result
        file.

        Every rank must call this with identical arguments, and clocks should be
        pinned first (``nvidia-smi -lgc``): boost drift is larger than the
        differences being ranked.

        Parameters
        ----------
        hiddens : Sequence[int]
            Hidden sizes to tune -- the ones this job will actually run. There
            is no default: admission does not constrain the hidden size, so
            there is no finite set to enumerate, and guessing would quietly tune
            a shape nobody uses.

            The **batch** dimension is not here. It comes from ``tune_batches``
            on the constructor, because the buckets have to be the same on the
            tuning side and the lookup side, which makes them a property of the
            workspace rather than of one call.
        dtype : torch.dtype
            Which of the two supported dtypes to measure. Both are 2 bytes so
            the traffic is identical, but they take different conversion paths.
        cache : str, optional
            Where to persist results. Defaults to the workspace's
            ``tune_cache``, which is also where the next process reads them.
        tune_group : ProcessGroup, optional
            Group used to reduce per-candidate timings so every rank picks the
            same winner. Built here as a gloo subgroup when the workspace spans
            the default process group; must be supplied otherwise, because
            ``new_group`` is collective over the *default* group and building
            one here would hang a job whose workspace is a strict subgroup.
        warmup, repeat : int
            Untimed and timed iterations per candidate. The library defaults
            time too short a span to resolve candidates for a collective this
            fast, so these default higher.

        Returns
        -------
        dict
            ``{(hidden, batch): config}`` for every shape that was measured, so
            the caller can see what tuning actually covered and what it chose.

        Raises
        ------
        ValueError
            If none of ``hiddens`` yields a shape the kernels admit -- otherwise
            the call is a silent no-op.
        """
        from ..autotuner import (
            AutoTuner,
            autotune,
            get_autotune_process_group,
            set_autotune_process_group,
        )

        hiddens = tuple(int(h) for h in hiddens)
        path = cache or self._tune_cache
        # Everything the collective profiling contract requires to match, in
        # one gather. A blocklist set on one rank alone silently shortens that
        # rank's candidate list, and the timing reduction then deadlocks on the
        # first divergence.
        self._joint_check(
            {
                "error": None,
                "hiddens": hiddens,
                "dtype": str(dtype),
                "cache": path,
                "warmup": warmup,
                "repeat": repeat,
                "tune_batches": self._tune_batches,
                "blocklist": os.environ.get("FLASHINFER_TACTICS_BLOCKLIST", ""),
                "digest": self._cache_digest(),
            },
            "starting a tuning run",
        )

        if tune_group is None:
            tune_group = self._make_tune_group()
        elif dist.get_world_size(tune_group) != self.world_size:
            raise ValueError(
                f"tune_group spans {dist.get_world_size(tune_group)} ranks but "
                f"the workspace spans {self.world_size}"
            )

        tuner = AutoTuner.get()
        previous_group = get_autotune_process_group()
        previous_counts = (tuner.warmup, tuner.repeat)
        set_autotune_process_group(tune_group)
        # The library defaults time too short a span to resolve candidates at
        # this operator's scale.
        tuner.warmup, tuner.repeat = warmup, repeat
        covered: List[Tuple[int, int]] = []
        skipped: List[int] = []
        try:
            for hidden in hiddens:
                batches = [
                    b
                    for b in tuned_batches_for(
                        hidden, self._tune_batches, self.max_numel
                    )
                    if self.launch_config(
                        torch.empty((b, hidden), dtype=dtype, device=self.device)
                    )
                    is not None
                ]
                if not batches:
                    # Recorded rather than skipped silently: the call would
                    # otherwise return cleanly having measured nothing.
                    skipped.append(hidden)
                    continue
                torch.cuda.synchronize(self.device)
                self.rebind_stream()
                with autotune(True, tuning_buckets=tuple(batches), round_up=False):
                    for batch in batches:
                        inp = torch.randint(
                            0,
                            16,
                            (batch, hidden),
                            dtype=torch.int32,
                            device=self.device,
                        ).to(dtype)
                        tuner.choose_one(
                            PCIE_IPC_CUSTOM_OP,
                            [self._runner],
                            pcie_ipc_tuning_config(self._tune_batches),
                            [inp],
                        )
                        covered.append((hidden, batch))
        finally:
            tuner.warmup, tuner.repeat = previous_counts
            # Restore rather than clear: a caller may be tuning something else
            # around this.
            set_autotune_process_group(previous_group)

        if skipped:
            message = (
                f"tune() measured nothing for hidden {skipped} at "
                f"{self.world_size} ranks: the kernels do not support those "
                "shapes, and tuning does not widen what is supported."
            )
            if not covered:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        # Winners live in the in-memory cache now, so drop anything this
        # workspace resolved from the seed.
        self._tuned.clear()
        self._tuned_configs_loaded = True
        dist.barrier(group=self.group)
        if self.rank == 0:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tuner.save_configs(path)
        # Nobody leaves before the file is on disk: a peer that rebuilt its
        # workspace first would load a half-written table.
        dist.barrier(group=self.group)
        return {
            (hidden, batch): self.tuned_launch_config(
                torch.empty((batch, hidden), dtype=dtype, device=self.device)
            )
            for hidden, batch in covered
        }

    def _make_tune_group(self):
        """A gloo subgroup for reducing candidate timings.

        gloo because the reduction carries one float64 and an NCCL collective
        immediately after a spin-waiting IPC kernel is exactly the interference
        a timing loop does not want.
        """
        if self._tune_group is not None:
            return self._tune_group
        ranks = dist.get_process_group_ranks(self.group)
        if len(ranks) != dist.get_world_size():
            raise ValueError(
                "tune() cannot build its own reduction group for a workspace "
                "that spans a strict subgroup: new_group() is collective over "
                "the default process group, so every process would have to "
                "call it. Pass tune_group= built by all ranks instead."
            )
        self._tune_group = dist.new_group(ranks=ranks, backend="gloo")
        return self._tune_group

    def destroy(self) -> None:
        """Release the handle and the shared slab.

        Collective: every rank must call this, and the peer unmapping is
        separated from the free by a barrier inside ``free_shared_buffer``.
        """
        if self._handle is not None:
            # all_reduce() launches asynchronously, so a collective may still be
            # running or spinning on this slab. free_shared_buffer() unmaps the
            # peers, and unmapping memory a live kernel is still touching is a
            # use-after-free -- wait for the device before tearing anything
            # down. This is the conservative choice; a stream-scoped wait would
            # need the workspace to track every stream it has been used on.
            torch.cuda.synchronize(self.device)
            get_pcie_ipc_comm_module().dispose(self._handle)
            self._handle = None
        if self._ipc_ptrs is not None:
            free_shared_buffer(self._ipc_ptrs, group=self.group)
            self._ipc_ptrs = None
        if self._tune_group is not None:
            dist.destroy_process_group(self._tune_group)
            self._tune_group = None
        self._tuned.clear()

    def __enter__(self) -> "PcieIpcAllReduceWorkspace":
        return self

    def __exit__(self, *exc_info) -> None:
        self.destroy()
