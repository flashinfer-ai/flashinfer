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
from .pcie_ipc_policy import (
    IpcLaunchConfig,
    get_pcie_ipc_fused_launch_config,
    get_pcie_ipc_launch_config,
)
from .pcie_ipc_topology import resolve_pcie_ipc_profile
from .pcie_ipc_tuning import (
    PCIE_IPC_CUSTOM_OP,
    PCIE_IPC_FUSED_CUSTOM_OP,
    TUNE_REPEAT,
    TUNE_WARMUP,
    generate_tune_batches,
    PcieIpcAllReduceRunner,
    PcieIpcFusedRmsNormRunner,
    cache_covers_workspace,
    default_cache_path,
    pack_config,
    pcie_ipc_tuning_config,
    TABLE_TACTIC,
    resolve_tuned_config,
    resolve_tuned_fused_config,
    tuned_batches_for,
    warn_no_tune_group,
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

    @register_custom_op(
        "flashinfer::pcie_ipc_all_reduce_fused_add_rmsnorm",
        mutates_args=["residual_out", "norm_out"],
    )
    def all_reduce_fused_add_rmsnorm(
        handle: int,
        inp: torch.Tensor,
        residual_in: torch.Tensor,
        gamma: torch.Tensor,
        residual_out: torch.Tensor,
        norm_out: torch.Tensor,
        hidden: int,
        eps: float,
        blocks: int,
        threads: int,
        transport_blocks: int,
        variant: int,
        enable_pdl: bool,
    ) -> None:
        module.pcie_ipc_all_reduce_fused_add_rmsnorm(
            handle,
            inp,
            residual_in,
            gamma,
            residual_out,
            norm_out,
            hidden,
            eps,
            blocks,
            threads,
            transport_blocks,
            variant,
            enable_pdl,
        )

    return SimpleNamespace(
        workspace_size=workspace_size,
        init=init,
        dispose=dispose,
        all_reduce=all_reduce,
        all_reduce_fused_add_rmsnorm=all_reduce_fused_add_rmsnorm,
        memop_supported=module.pcie_ipc_memop_supported,
        set_memop_enabled=module.pcie_ipc_set_memop_enabled,
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
        tune_batches: Optional[Sequence[int]] = None,
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
        # Bounds the fused grid: every fused kernel ends in a rendezvous of all
        # its blocks, which only completes if all of them are resident, and
        # __launch_bounds__(1024, 1) puts at most one block on an SM.
        self.sm_count = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self.profile = ""
        self.memop_supported = False
        self.profile_reason = ""
        # Resolved launch configurations, keyed exactly. Consulted before any
        # AutoTuner call because even a pure cache lookup there takes a global
        # lock, which is real overhead at this operator's scale.
        self._tuned: Dict[Tuple[int, int, torch.dtype], IpcLaunchConfig] = {}
        # Kept apart from _tuned: the two ops rank different kernels over the
        # same (blocks, threads) grid, so one key space would let a plain
        # winner be read back as a fused one.
        self._fused_tuned: Dict[Tuple[int, int, torch.dtype], IpcLaunchConfig] = {}
        self._runner: Optional[PcieIpcAllReduceRunner] = None
        self._fused_runner: Optional[PcieIpcFusedRmsNormRunner] = None
        self._tune_group: Optional[ProcessGroup] = None
        # None means "derive a ladder from max_numel at tune() time". An
        # explicit list is honoured as given and only checked for coverage; see
        # _warn_if_coverage_falls_short for why that is a warning, not an error.
        self._tune_batches = (
            None if tune_batches is None else tuple(int(b) for b in tune_batches)
        )
        self._tune_cache = tune_cache or default_cache_path(self.world_size)
        self._tune_cache_exists = False
        self._tuned_configs_loaded = False
        self._warned_untuned = False
        self._warned_stale_entry = False

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
            # The fused kernels end in a device-local rendezvous, so their grid
            # may not exceed what the device can hold. Gathered rather than
            # read locally: ranks that disagree would admit different
            # configurations, and on a collective that is a hang.
            "sm_count": self.sm_count,
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
            local_memop_supported = bool(module.memop_supported())
            nbytes = module.workspace_size(
                self.world_size, max_numel, self.elem_size, max_blocks
            )
        except Exception as e:  # noqa: BLE001 - re-raised jointly below
            nbytes = 0
            self._joint_check({"error": f"{type(e).__name__}: {e}"}, "preparing")
            raise  # unreachable: _joint_check raises on every rank
        # Carry eligibility in the existing preparation exchange. Mixed groups
        # agree on the original protocol without adding a collective to it.
        prepared = self._joint_check(
            {"error": None, "memop_supported": local_memop_supported},
            "preparing",
            require_identical=False,
        )
        self.memop_supported = self.world_size in (4, 8) and all(
            entry["memop_supported"] for entry in prepared
        )

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
            if self.memop_supported:
                module.set_memop_enabled(self._handle, True)
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

    def _joint_check(
        self, local: dict, what: str, *, require_identical: bool = True
    ) -> List[dict]:
        """Gather per-rank outcomes and fail the whole group, or none of it.

        Raises the same error on every rank, so the caller can rely on all
        ranks taking the same branch afterwards. Capability outcomes may differ
        when require_identical is false; callers receive the successful entries
        and choose one protocol for the whole group.
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
            if require_identical
            and key != "error"
            and len({repr(g[key]) for g in entries}) > 1
        }
        if mismatched:
            raise ValueError(
                "every rank must build the workspace with identical arguments, "
                f"but these differ across the group: {mismatched}"
            )
        return entries

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
        self._fused_runner = PcieIpcFusedRmsNormRunner(self)
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
            self.world_size,
            self.profile,
            self.max_blocks,
            self.max_numel,
            self.memop_supported,
        )
        self._joint_check(
            {
                "error": None,
                "digest": self._cache_digest(),
                "covers": self._tuned_configs_loaded,
            },
            "loading the tune cache",
        )

    def _warn_stale_tuned_entry(self, tactic) -> None:
        """Say once that the cache holds entries this build no longer accepts."""
        if self._warned_stale_entry:
            return
        self._warned_stale_entry = True
        warnings.warn(
            f"PCIe IPC all-reduce ignored a tuned entry from {self._tune_cache}: "
            f"tactic {tactic!r} is not launchable in this build, so this shape "
            "falls back to a seed configuration. The cache key still matches, "
            "so this is not a stale workspace -- it is a cache written before "
            "the set of legal configurations narrowed. Re-tune to regain the "
            "measured configurations; other shapes in the same file may still "
            "be in use, so the loss is partial and otherwise unsignalled.",
            UserWarning,
            stacklevel=4,
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

        # Both ops: a rank whose *fused* table differs from its peers' resolves
        # different fused configurations, and that is a hang just the same.
        prefixes = (f"('{PCIE_IPC_CUSTOM_OP}'", f"('{PCIE_IPC_FUSED_CUSTOM_OP}'")
        tuner = AutoTuner.get()
        entries = sorted(
            (key, repr(value))
            for key, value in tuner._file_configs.items()
            if key.startswith(prefixes)
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
        # Resolving is collective and reads the verdict back to the host, so it
        # cannot happen inside a graph capture. Say that here: the CUDA-level
        # failure is "Cannot copy between CPU and CUDA tensors during CUDA
        # graph capture", which names neither this workspace nor the fix.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"the launch configuration for shape {tuple(inp.shape)} dtype "
                f"{inp.dtype} has not been resolved yet, and resolving it "
                "inside a CUDA graph capture is not possible: the ranks agree "
                "on it with a collective whose result is read back on the "
                "host. Call workspace.prepare() with every shape you intend to "
                "capture -- after tune(), which clears this cache -- or pass "
                "config= explicitly at the call site."
            )
        config = self._resolve_tuned(inp, seed, tuner)
        self._tuned[key] = config
        return config

    def _resolve_tuned(
        self, inp: torch.Tensor, seed: IpcLaunchConfig, tuner
    ) -> IpcLaunchConfig:
        """Cold path: search or look up, then make the group agree."""
        hidden = inp.shape[-1]
        batch = inp.numel() // hidden
        # The lookup has to use the same buckets the search used, or a
        # configuration profiled under one mapping is read back under another.
        # With a derived ladder that means regenerating it for this hidden --
        # pcie_ipc_tuning_config is lru_cached on the tuple, so an equal ladder
        # yields the identical mapper object, which _find_nearest_profile's own
        # cache requires.
        tuning_config = pcie_ipc_tuning_config(self._batches_for(hidden))
        can_profile = tuner.is_tuning_mode and self._runner.can_profile(inp.device)
        if can_profile:
            _, tactic = tuner.choose_one(
                PCIE_IPC_CUSTOM_OP, [self._runner], tuning_config, [inp]
            )
        else:
            # An enclosing autotune context may belong to another operator and
            # may have replaced the global file cache. Without a matching
            # distributed tune group, profiling this collective is unsafe.
            # Restore the workspace's explicit tune cache and perform a lookup
            # with its own bucket policy instead of retaining the seed tactic.
            if tuner.is_tuning_mode:
                # The runner would have said this had it been asked for
                # candidates; short-circuiting before ``choose_one`` is what
                # would otherwise swallow it. It is the actionable half of the
                # diagnosis -- the caller is tuning, so "go tune" is not.
                warn_no_tune_group(stacklevel=4)
                if self._tune_cache_exists:
                    tuner.load_configs(self._tune_cache)
            _, _, tactic, _ = tuner.search_cache(
                PCIE_IPC_CUSTOM_OP,
                [self._runner],
                ((batch, hidden),),
                tuning_config,
                inputs=[inp],
            )
        config = resolve_tuned_config(seed, tactic, self.world_size, self.max_blocks)
        if tactic != TABLE_TACTIC and config is seed:
            # A matching key with an unusable value: world size, profile,
            # max_blocks, max_numel, dtype and the tune version are all in the
            # key, so the legal set narrowed without the version moving with it.
            #
            # Worth its own message because the two existing ones cannot reach
            # it: both are gated on `_tuned_configs_loaded`, which is settled
            # from the loaded *keys*, so they ask "did anyone tune this?" and
            # not "is this value still legal?". The loss is otherwise silent
            # and partial -- some shapes keep their tuned configuration, others
            # quietly drop to the seed.
            self._warn_stale_tuned_entry(tactic)

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

    def fused_launch_config(self, inp: torch.Tensor) -> Optional[IpcLaunchConfig]:
        """Configuration for the fused all-reduce, or ``None`` if unsupported.

        A pure function of shape, dtype and the workspace's immutable attributes,
        for the same reason :meth:`launch_config` is: every rank must reach the
        same answer without negotiating one.
        """
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
        if not inp.is_contiguous() or inp.dim() < 2:
            return None
        numel = inp.numel()
        if numel > self.max_numel:
            return None
        return get_pcie_ipc_fused_launch_config(
            self.world_size,
            numel,
            inp.shape[-1],
            self.elem_size,
            self.max_blocks,
            self.sm_count,
        )

    def tuned_fused_launch_config(self, inp: torch.Tensor) -> Optional[IpcLaunchConfig]:
        """Fused configuration for ``inp``, measured if one has been persisted.

        The fused counterpart of :meth:`tuned_launch_config`, and split from it
        the same way: admission first and final, a hot cache consulted outside
        tuning mode, and the search or lookup only on a miss.
        """
        seed = self.fused_launch_config(inp)
        if seed is None:
            return None
        from ..autotuner import AutoTuner

        tuner = AutoTuner.get()
        key = (inp.numel(), inp.shape[-1], inp.dtype)
        if not tuner.is_tuning_mode:
            cached = self._fused_tuned.get(key)
            if cached is not None:
                return cached
        # As in tuned_launch_config: resolving is collective and its verdict is
        # read back on the host, so a shape first seen inside a capture cannot
        # be resolved there. The fused op has its own cache, so a shape prepared
        # for the plain call is still unresolved here.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"the fused launch configuration for shape {tuple(inp.shape)} "
                f"dtype {inp.dtype} has not been resolved yet, and resolving it "
                "inside a CUDA graph capture is not possible: the ranks agree "
                "on it with a collective whose result is read back on the "
                "host. Call workspace.prepare() with every shape you intend to "
                "capture -- after tune(), which clears this cache -- or pass "
                "config= explicitly at the call site."
            )
        config = self._resolve_tuned_fused(inp, seed, tuner)
        self._fused_tuned[key] = config
        return config

    def _fused_seed_for_bucket(self, inp: torch.Tensor) -> Optional[IpcLaunchConfig]:
        """The seed as the tuner measured it: on the bucket, not on the shape.

        Tactic ``-1`` means "the seed was fastest", and the autotuner timed it at
        the bucket shape. Re-deriving it from the real shape can name a
        *different* transport, because the seed's crossover is a byte threshold
        and the buckets are powers of two -- at four ranks the threshold falls
        between batch 2 and 3, which share bucket 2. Serving what was measured
        keeps a constant fitted on one machine from deciding a shape that was
        tuned on another.
        """
        hidden = int(inp.shape[-1])
        batch = inp.numel() // hidden
        # Through _batches_for, not the raw field: the ladder is derived from
        # the payload when the caller did not name one, and the serving lookup
        # has to land on the bucket the search actually measured.
        bucket = max(
            (b for b in self._batches_for(hidden) if b <= batch), default=batch
        )
        return get_pcie_ipc_fused_launch_config(
            self.world_size,
            bucket * hidden,
            hidden,
            self.elem_size,
            self.max_blocks,
            self.sm_count,
        )

    def _resolve_tuned_fused(
        self, inp: torch.Tensor, seed: IpcLaunchConfig, tuner
    ) -> IpcLaunchConfig:
        """Cold path: search or look up, then make the group agree."""
        hidden = int(inp.shape[-1])
        batch = inp.numel() // hidden
        tuning_config = pcie_ipc_tuning_config(self._batches_for(hidden))
        can_profile = tuner.is_tuning_mode and self._fused_runner.can_profile(
            inp.device
        )
        if can_profile:
            _, tactic = tuner.choose_one(
                PCIE_IPC_FUSED_CUSTOM_OP, [self._fused_runner], tuning_config, [inp]
            )
        else:
            # Same reasoning as _resolve_tuned, and the same two consequences of
            # getting it wrong: an enclosing autotune context that belongs to
            # another operator has replaced the global file cache, and profiling
            # a collective without a matching tune group is unsafe. Profiling
            # anyway persists the seed tactic for the life of the process.
            if tuner.is_tuning_mode:
                warn_no_tune_group("fused rmsnorm", stacklevel=4)
                if self._tune_cache_exists:
                    tuner.load_configs(self._tune_cache)
            _, _, tactic, _ = tuner.search_cache(
                PCIE_IPC_FUSED_CUSTOM_OP,
                [self._fused_runner],
                ((batch, hidden),),
                tuning_config,
                inputs=[inp],
            )
        # `-1` resolves to the configuration the tuner actually timed; anything
        # unusable still falls back to the seed for the real shape.
        table_config = self._fused_seed_for_bucket(inp) or seed
        config = resolve_tuned_fused_config(
            table_config,
            tactic,
            self.world_size,
            self.max_blocks,
            hidden,
            self.elem_size,
            self.sm_count,
            inp.numel(),
        )

        # Unconditional, for the reason given in _resolve_tuned: the ranks
        # would otherwise have to agree on whether to run this collective
        # before running it.
        packed = pack_config(config)
        bounds = torch.tensor([packed, -packed], dtype=torch.int64, device=self.device)
        dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=self.group)
        if int(bounds[0]) != -int(bounds[1]):
            warnings.warn(
                "ranks resolved different tuned fused configurations for shape "
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

    def supports_fused_add_rms_norm(self, inp: torch.Tensor) -> bool:
        """Whether the fused kernel can run ``inp``.

        Stricter than :meth:`supports`: the fused kernel holds a whole row in
        registers, so a row wider than the grid mapping allows is refused here
        while the plain all-reduce still takes it.
        """
        return self.fused_launch_config(inp) is not None

    @flashinfer_api
    def all_reduce_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        *,
        residual_in: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float = 1e-6,
        residual_out: Optional[torch.Tensor] = None,
        norm_out: Optional[torch.Tensor] = None,
        config: Optional[IpcLaunchConfig] = None,
        enable_pdl: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-reduce ``inp``, add ``residual_in``, and RMSNorm the result.

        Equivalent to ``all_reduce`` followed by
        :func:`~flashinfer.norm.fused_add_rmsnorm`, in one launch. Both outputs
        are full shape on every rank.

        ``norm_out`` is exactly ``rmsnorm(residual_out)``: the sum of squares is
        taken over the stored, rounded residual rather than an unrounded
        accumulator, because a rank that does not own a pack only ever sees the
        rounded value. That matches the other fusion backends, so the three are
        interchangeable.

        A **collective**, with the same contract as :meth:`all_reduce`: every
        rank issues it with the same shape, dtype and configuration, in the same
        order.

        Parameters
        ----------
        inp : torch.Tensor
            ``[rows, hidden]``, this rank's contribution. Not modified.
        residual_in : torch.Tensor
            Added once, by whichever rank owns the pack, after the reduction.
            **Must be identical on every rank** -- it is replicated in tensor
            parallelism, and a group that disagrees about it gets a result
            stitched together from several residuals rather than an error.
        rms_gamma : torch.Tensor
            ``[hidden]`` RMSNorm weight. Identical on every rank, for the same
            reason as ``residual_in``.
        rms_eps : float
            Added to the mean square before the reciprocal square root.
        residual_out, norm_out : torch.Tensor, optional
            Destinations. Allocated when omitted. Must not alias each other.
        config : IpcLaunchConfig, optional
            Launch geometry and transport. Resolved from the tune cache or the
            seed when omitted; pass one explicitly only to benchmark or to reach
            a configuration neither would choose. ``variant`` selects the
            transport, and names the same kernel it names on :meth:`all_reduce`:
            ``UNSTAGED`` the one-shot, ``STAGED`` the staged push (island-block
            at eight ranks), ``STAGED_RING`` neighbour order, ``FLAT_STAGED``
            the topology-blind push at eight ranks only. Ranks that disagree on
            it hang.

        Returns
        -------
        tuple
            ``(norm_out, residual_out)``.

        Raises
        ------
        ValueError
            If the fused kernel cannot run this shape. Check
            :meth:`supports_fused_add_rms_norm` first and fall back.
        """
        if config is None:
            config = self.tuned_fused_launch_config(inp)
            if config is None:
                raise ValueError(
                    f"fused rmsnorm does not support shape {tuple(inp.shape)} "
                    f"dtype {inp.dtype} at {self.world_size} ranks; check "
                    "supports_fused_add_rms_norm() first"
                )
        self._check_stream()
        for name, tensor in (("residual_in", residual_in), ("rms_gamma", rms_gamma)):
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device} but the workspace was built "
                    f"on {self.device}"
                )
        if residual_out is None:
            residual_out = torch.empty_like(inp)
        if norm_out is None:
            norm_out = torch.empty_like(inp)
        if residual_out.data_ptr() == norm_out.data_ptr():
            raise ValueError("residual_out and norm_out must not alias")

        get_pcie_ipc_comm_module().all_reduce_fused_add_rmsnorm(
            self.handle,
            inp,
            residual_in,
            rms_gamma,
            residual_out,
            norm_out,
            int(inp.shape[-1]),
            float(rms_eps),
            config.blocks,
            config.threads,
            config.effective_transport_blocks(),
            int(config.variant),
            enable_pdl,
        )
        return norm_out, residual_out

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

    def _launch_fused(
        self,
        inp: torch.Tensor,
        residual_in: torch.Tensor,
        gamma: torch.Tensor,
        residual_out: torch.Tensor,
        norm_out: torch.Tensor,
        eps: float,
        config: IpcLaunchConfig,
        enable_pdl: bool = False,
    ) -> None:
        """Issue one fused collective with an explicit configuration.

        The launch without the admission, device and stream checks around it,
        for the same caller :meth:`_launch` has: the tuner, sweeping many
        configurations over one validated set of buffers.
        """
        get_pcie_ipc_comm_module().all_reduce_fused_add_rmsnorm(
            self.handle,
            inp,
            residual_in,
            gamma,
            residual_out,
            norm_out,
            int(inp.shape[-1]),
            float(eps),
            config.blocks,
            config.threads,
            config.effective_transport_blocks(),
            int(config.variant),
            enable_pdl,
        )

    def prepare(
        self,
        shapes: Sequence[Tuple[int, int]],
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[Tuple[int, int], Optional[IpcLaunchConfig]]:
        """Resolve the launch configuration for each shape now. **Collective.**

        Resolution is lazy by default: the first call at a given shape looks the
        configuration up and then makes the group agree on it, which costs one
        small reduction whose verdict is read back on the host. That is fine in
        eager mode and impossible inside a CUDA graph capture, so a shape first
        used inside a capture fails to capture.

        This moves that work to a point the caller chooses. Nothing else
        changes -- the same lookup, the same agreement, the same number of
        collectives -- and afterwards every listed shape is served from the
        in-process cache, so a capture of it touches no collective at all.

        Call it **after** :meth:`tune`, which clears that cache, and list every
        shape that will be captured: shapes left out are still resolved lazily
        and still cannot be captured. Serving frameworks pad the batch to the
        bucket they capture, so list the padded sizes, not the real ones.

        Parameters
        ----------
        shapes : sequence of (batch, hidden)
            Shapes to resolve. Every rank must pass the same list in the same
            order -- resolution is collective, so a rank with a different list
            deadlocks the group rather than disagreeing.
        dtype : torch.dtype
            Which of the two supported dtypes to resolve for. The cache is
            keyed by dtype, so resolve each one that will be used.

        Both entry points are prepared: :meth:`all_reduce` and
        :meth:`all_reduce_fused_add_rms_norm` resolve against separate caches,
        so preparing one would leave the other uncapturable.

        Returns
        -------
        dict
            ``{(batch, hidden): config}`` for the plain all-reduce, with ``None``
            for shapes the kernels do not support -- those fall back to another
            backend at call time and never reach a capture. The fused
            configuration is resolved too but not returned; read it with
            :meth:`fused_launch_config` if it is wanted.
        """
        shapes = [(int(batch), int(hidden)) for batch, hidden in shapes]
        # Same reasoning as tune(): the loop below issues one collective per
        # shape, so a rank with a different list hangs rather than disagrees.
        self._joint_check(
            {"error": None, "shapes": shapes, "dtype": str(dtype)},
            "preparing launch configurations",
        )
        resolved: Dict[Tuple[int, int], Optional[IpcLaunchConfig]] = {}
        for batch, hidden in shapes:
            probe = torch.empty((batch, hidden), dtype=dtype, device=self.device)
            resolved[(batch, hidden)] = self.tuned_launch_config(probe)
            # The fused op keeps its own cache, so preparing the plain call
            # leaves it unresolved and uncapturable. Which of the two the caller
            # will use is not knowable here, and the extra cost is one small
            # reduction per shape on a call that is already collective and
            # made once. Shapes the fused kernels cannot run resolve to None
            # without issuing anything.
            self.tuned_fused_launch_config(probe)
        return resolved

    def tune(
        self,
        hiddens: Sequence[int],
        *,
        dtype: torch.dtype = torch.bfloat16,
        cache: Optional[str] = None,
        tune_group=None,
        warmup: int = TUNE_WARMUP,
        repeat: int = TUNE_REPEAT,
        fused: bool = True,
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
        fused : bool
            Also measure :meth:`all_reduce_fused_add_rms_norm`, which searches
            its own kernels under its own name and roughly doubles the run.
            Shapes whose row is too wide for the fused kernels are skipped
            without affecting the plain search. Its winners are not in the
            return value; read them back with
            :meth:`tuned_fused_launch_config`.

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
                # A rank that searched the fused op while its peers did not
                # would issue a whole extra sequence of collectives into a
                # group that is not expecting them.
                "fused": bool(fused),
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
        fused_covered: List[Tuple[int, int]] = []
        skipped: List[int] = []
        try:
            for hidden in hiddens:
                requested = self._batches_for(hidden)
                if self._tune_batches is not None:
                    self._warn_if_coverage_falls_short(requested, hidden, dtype)
                batches = [
                    b
                    for b in tuned_batches_for(hidden, requested, self.max_numel)
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
                            pcie_ipc_tuning_config(tuple(requested)),
                            [inp],
                        )
                        covered.append((hidden, batch))
                        # The fused op is a separate search under its own name,
                        # over its own candidate set. Skipped rather than
                        # refused where the row is too wide for it: the plain
                        # op still has an answer for that shape.
                        if fused and self.supports_fused_add_rms_norm(inp):
                            tuner.choose_one(
                                PCIE_IPC_FUSED_CUSTOM_OP,
                                [self._fused_runner],
                                # The same ladder the plain search just used at
                                # this hidden: two searches bucketing one shape
                                # differently would resolve against each other.
                                pcie_ipc_tuning_config(tuple(requested)),
                                [inp],
                            )
                            fused_covered.append((hidden, batch))
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
        self._fused_tuned.clear()
        self._tuned_configs_loaded = True
        dist.barrier(group=self.group)
        # What makes this worth refusing rather than warning is stated to the
        # caller below. The mechanism is not: the copy-engine path has several
        # internal streams and kernels to load, which a shortened warmup does
        # not absorb, so too few samples can pick the wrong data plane outright
        # rather than merely a worse block count within the right one.
        if warmup < TUNE_WARMUP or repeat < TUNE_REPEAT:
            if self.rank == 0:
                warnings.warn(
                    f"not persisting the tuned configurations: they were "
                    f"measured with warmup={warmup} repeat={repeat}, below the "
                    f"defaults ({TUNE_WARMUP}/{TUNE_REPEAT}). Too few samples "
                    f"can pick the wrong variant outright, and a file written "
                    f"here is indistinguishable from a good one to every "
                    f"process that later loads it. The results are live in this "
                    f"process; re-run at the default counts to persist them.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif self.rank == 0:
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

    def _batches_for(self, hidden: int) -> Tuple[int, ...]:
        """Single source for the ladder: the tuning search and the serving
        lookup must agree on it exactly, so both read it from here rather than
        each deriving it.
        """
        if self._tune_batches is not None:
            return self._tune_batches
        ladder = generate_tune_batches(hidden, self.max_numel, self.elem_size)
        # A workspace too small to hold even one row at this hidden yields an
        # empty ladder, and an empty bucket set is rejected outright by
        # autotune(). Such a shape is not admissible anyway -- the caller is
        # about to be told so -- but the lookup path reaches here first, so give
        # it a one-bucket ladder rather than an exception from three frames down.
        return ladder or (1,)

    def _warn_if_coverage_falls_short(
        self, batches: Sequence[int], hidden: int, dtype: torch.dtype
    ) -> None:
        """Say when an explicit bucket list leaves the top of the range untuned.

        Buckets map with floor semantics, so a shape above the largest one is
        served by whatever was measured there. That is fine when the gap is
        small and wrong when it is two orders of magnitude: at hidden 6144 the
        default list stops at 1.5 MiB, and the configuration it picks there runs
        a 96 MiB collective 30% slower than the one measured at 96 MiB.

        A warning rather than an error, because tuning only the decode range is
        a legitimate choice -- a deployment that never issues a prefill
        collective has no reason to pay for measuring one.
        """
        if not batches:
            return
        elem = torch.empty((), dtype=dtype).element_size()
        tuned_bytes = max(batches) * hidden * elem
        admitted_bytes = self.max_numel * elem
        if tuned_bytes * 8 >= admitted_bytes:
            return
        warnings.warn(
            f"tuning stops at {max(batches)} rows ({tuned_bytes >> 20} MiB at "
            f"hidden {hidden}) but this workspace admits up to "
            f"{admitted_bytes >> 20} MiB. Buckets map downwards, so every "
            f"larger shape will run the configuration measured at the top of "
            f"this list. Pass tune_batches covering the shapes you serve, or "
            f"leave it unset to have the ladder derived from max_numel.",
            RuntimeWarning,
            stacklevel=3,
        )

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
