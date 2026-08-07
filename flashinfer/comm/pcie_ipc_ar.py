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
from types import SimpleNamespace
from typing import List, Optional

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

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
# float32 is deliberately absent. The kernels handle it, but the tuning tables
# carry no dtype dimension while the crossovers depend on payload *bytes*, so a
# 4-byte dtype would run with a configuration chosen for half the traffic.
# Re-tune before re-enabling.
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
        stream_mode: bool,
        ring_push: bool,
        enable_pdl: bool,
    ) -> None:
        module.pcie_ipc_all_reduce(
            handle, inp, out, blocks, threads, stream_mode, ring_push, enable_pdl
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
        Force a tuning table (``"rootcplx"`` or ``"pcieswitch"``) instead of
        probing the interconnect. Probing is collective and runs before any
        allocation.

    Examples
    --------
    >>> ws = PcieIpcAllReduceWorkspace(group=group, max_numel=128 * 6144)
    >>> if ws.supports(x):
    ...     out = ws.all_reduce(x)
    >>> ws.destroy()
    """

    def __init__(
        self,
        group: ProcessGroup,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        max_blocks: int = 128,
        profile: Optional[str] = None,
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
        """Tuned launch configuration for ``inp``, or ``None`` when untuned.

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
        if numel > self.max_numel or (numel * self.elem_size) % 16 != 0:
            return None
        hidden = inp.shape[-1]
        if hidden <= 0 or numel % hidden != 0:
            return None
        return get_pcie_ipc_launch_config(
            self.profile, self.world_size, hidden, numel // hidden, self.max_blocks
        )

    def supports(self, inp: torch.Tensor) -> bool:
        """Whether this workspace has a tuned configuration for ``inp``.

        An untuned shape is reported unsupported rather than run with default
        launch parameters: on a switch-free fabric the untuned default (max
        blocks, no staging) is the worst direction, so falling back to another
        backend is strictly better.

        Raises the same way :meth:`launch_config` does on a device mismatch --
        that is a caller bug, not an untuned shape.
        """
        return self.launch_config(inp) is not None

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
            Launch geometry and kernel selection. Taken from the tuning table
            when omitted; pass one explicitly only to benchmark or to test a
            configuration the table does not choose. Ranks that disagree on it
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
            If no tuned configuration exists for this shape. Check
            :meth:`supports` first and fall back to another backend.
        """
        if config is None:
            config = self.launch_config(inp)
            if config is None:
                raise ValueError(
                    f"no tuned configuration for shape {tuple(inp.shape)} "
                    f"dtype {inp.dtype} at {self.world_size} ranks "
                    f"(profile {self.profile}); check supports() first"
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
        get_pcie_ipc_comm_module().all_reduce(
            self.handle,
            inp,
            out,
            config.blocks,
            config.threads,
            config.stream_mode,
            config.ring_push,
            enable_pdl,
        )
        return out

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

    def __enter__(self) -> "PcieIpcAllReduceWorkspace":
        return self

    def __exit__(self, *exc_info) -> None:
        self.destroy()
