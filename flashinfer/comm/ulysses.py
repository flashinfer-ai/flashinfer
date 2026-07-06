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

import ctypes
import re
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .ulysses_topology import UlyssesBackendDecision, resolve_ulysses_backend

_INT32_MAX = 2**31 - 1
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# communicator lifecycle states; CLOSED is only reached after a fully
# successful teardown so a failed close() can be retried
_OPEN, _CLOSING, _CLOSED = "open", "closing", "closed"


class UlyssesCommunicator:
    r"""Ulysses sequence-parallelism all-to-all communicator.

    Provides the two layout transforms of Ulysses attention over the 4-D
    layout ``[B, S, H, D]`` (a typical attention layer makes four collective
    calls: q/k/v through :meth:`scatter_heads`, the output through
    :meth:`gather_heads`):

    - :meth:`scatter_heads`: ``[B, S_local, H, D] -> [B, S_global, H_local, D]``
      (each rank keeps a head slice of the *full* sequence)
    - :meth:`gather_heads`:  ``[B, S_global, H_local, D] -> [B, S_local, H, D]``
      (each rank gets all heads of its *local* sequence shard back)

    where ``H`` is the global head count, ``H_local = H // world_size`` and
    ``S_global = S_local * world_size``. Both backends produce bit-identical
    results.

    Backend selection happens in the constructor, strictly before any IPC
    allocation or JIT compilation (see
    :func:`~flashinfer.comm.resolve_ulysses_backend`):

    - ``backend="auto"``: the fused-transpose NVLink-P2P kernel when the group
      is a verified single-node all-pairs NVLink mesh with a supported world
      size (2/4/6/8); NCCL otherwise — including when NVLink runtime
      initialization fails after a positive topology decision. Inspect
      :attr:`backend` and :attr:`fallback_reason` for the outcome.
    - ``backend="nvlink"``: force the fused kernel; raises on every rank
      (before any IPC/JIT for topology failures) when it cannot be used.
    - ``backend="nccl"``: force the ``dist.all_to_all_single`` path; skips
      the topology/NVML probe and all IPC/JIT entirely (the constructor
      still resolves and guards the CUDA device and performs CUDA-backed
      metadata collectives over ``group``). Supports any world size.

    All ranks must request the same ``backend``. The NCCL path with
    ``world_size > 1`` requires ``group`` to support CUDA all-to-all (an
    NCCL process group); this is checked at construction.
    ``world_size == 1`` is a passthrough: both collectives return the input
    tensor unchanged (no copy).

    Constraints
    -----------
    - The constructor is always collective: every rank of ``group`` must
      call it together. :meth:`close` is collective only when the NVLink
      backend was armed (its resources are IPC-shared); for the pure NCCL
      backend, ``world_size == 1``, or an auto fallback whose NVLink cleanup
      already completed, ``close`` is local and idempotent. Rank-local
      failures inside the constructor's NVLink initialization or inside a
      collective ``close`` are exchanged as group outcomes, so all ranks
      jointly clean up and raise (or fall back) instead of deadlocking; a
      failed ``close`` may be retried by all ranks.
    - Collectives run on the *current* CUDA stream of this rank; every rank
      must issue the same sequence of calls with consistently-shaped operands
      (a shape or call-order mismatch across ranks is a collective failure:
      expect hangs or garbage, exactly as with any collective library). At
      most one collective may be in flight per communicator at a time (the
      NVLink signal buffers assume serialized calls); do not call one
      communicator concurrently from multiple streams or threads.
    - Operand tensors must be contiguous 4-D CUDA tensors of the construction
      ``dtype`` (float16 / bfloat16 / float32) on the construction device,
      with every dim positive and total elements at most ``max_elems``;
      :meth:`scatter_heads` additionally requires ``H % world_size == 0`` and
      :meth:`gather_heads` requires ``S_global % world_size == 0``.
    - Each rank may use a different CUDA device (e.g. ``cuda:rank``); ranks
      must agree on ``max_elems``, ``dtype`` and ``backend``.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup, optional
        Process group of the Ulysses ranks. Defaults to ``dist.group.WORLD``.
    max_elems : int
        Capacity: the largest element count of any single all-to-all operand
        (input and output have equal ``numel``, so this is ``B*S_local*H*D``
        for the largest call). Must be at most ``2**31 - 1`` (the kernel's
        int32 index range). Sizes the NVLink staging buffer once at
        construction.
    dtype : torch.dtype
        Element type of all operands (float16 / bfloat16 / float32); enforced
        on every call.
    backend : str
        ``"auto"`` | ``"nvlink"`` | ``"nccl"`` (see above).
    device : torch.device or str or int, optional
        CUDA device of this rank; normalized to an explicit index (bare
        ``"cuda"`` means the current device, an int is a CUDA ordinal).
        Defaults to the current CUDA device.

    Examples
    --------
    >>> with UlyssesCommunicator(group, max_elems=B*S*H*D, dtype=torch.bfloat16) as comm:
    ...     q_ = comm.scatter_heads(q)   # [B,S_local,H,D] -> [B,S_global,H_local,D]
    ...     ...
    ...     o = comm.gather_heads(o_)    # [B,S_global,H_local,D] -> [B,S_local,H,D]
    """

    def __init__(
        self,
        group: Optional[ProcessGroup] = None,
        *,
        max_elems: int,
        dtype: torch.dtype,
        backend: str = "auto",
        device: Optional[torch.device] = None,
    ):
        self._state = _CLOSED  # flipped to OPEN only when construction succeeds
        self._nvlink_armed = False  # joint property: set on all ranks or none
        self._fa: Optional[int] = None
        self._out_ptrs: Optional[List[int]] = None
        self._sig_ptrs: Optional[List[int]] = None
        # rank-local resource tracking for staged init/teardown
        self._exports: List[int] = []  # device ptrs this rank cudaMalloc'ed
        self._imports: List[int] = []  # peer ptrs this rank IpcOpen'ed

        if group is None:
            group = dist.group.WORLD
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)

        # ---- collective-safe config validation ------------------------------
        # The bound device must be resolved BEFORE the first collective: NCCL
        # object collectives stage through a tensor on the *current* device,
        # so an explicit device="cuda:rank" without a prior set_device(rank)
        # would otherwise land every rank's metadata collective on GPU 0.
        # _resolve_device never raises; an unparsable input yields the current
        # device as a safe gather guard and the joint config validation right
        # after rejects it on every rank together.
        self.device = self._resolve_device(device)
        # Encode the local config with zero user code (exact type checks and
        # interpreter/torch-provided names only), gather, then validate the
        # identical list jointly so an invalid single-rank config raises the
        # same error on every rank instead of hanging peers in a later gather.
        # Devices are validated per rank but may legitimately differ across
        # ranks (cuda:rank); only max_elems and dtype must match.
        config = self._encode_config(max_elems, dtype, device)
        configs = self._gather(config)
        self._validate_configs_jointly(configs)

        self.max_elems = max_elems
        self.dtype = dtype

        # ---- backend selection: strictly before any IPC/JIT -----------------
        # topology_decision is what the probe concluded; decision is the
        # *effective* backend after runtime initialization (they differ only
        # when NVLink init failed at runtime and auto fell back to NCCL).
        self.topology_decision: UlyssesBackendDecision = resolve_ulysses_backend(
            backend, group=group, device=self.device
        )
        self.decision: UlyssesBackendDecision = self.topology_decision
        self.backend = self.decision.backend
        self.fallback_reason = (
            self.decision.reason
            if self.backend == "nccl" and backend != "nccl"
            else None
        )

        if self.backend == "nvlink":
            err = self._nvlink_init_transaction()
            if err is not None:
                # all ranks cleaned up (verified group-wide by the staged
                # cleanup) and hold the same joint error
                if backend == "nvlink":
                    raise RuntimeError(f"NVLink backend initialization failed: {err}")
                self.backend = "nccl"
                self.fallback_reason = f"nvlink init failed: {err}"
                self.decision = UlyssesBackendDecision("nccl", self.fallback_reason)

        # NCCL fallback needs a group that can move CUDA tensors; deterministic
        # in the (identical) group object, so a plain raise is group-uniform.
        if self.backend == "nccl" and self.world_size > 1:
            supported, observed = self._group_supports_cuda_alltoall()
            if not supported:
                raise ValueError(
                    "the Ulysses NCCL backend requires a process group "
                    f"supporting CUDA all-to-all (nccl), got '{observed}'"
                )

        self._state = _OPEN

    # ---- collective helpers ---------------------------------------------------

    def _group_supports_cuda_alltoall(self) -> Tuple[bool, str]:
        """A plain get_backend substring check would reject legitimate
        multi-backend groups (init_process_group(backend=None) reports
        "undefined" while its CUDA backend is ProcessGroupNCCL); check the
        backend actually bound to the CUDA device in that case. Deterministic
        in the group, so the resulting raise is group-uniform."""
        try:
            observed = str(dist.get_backend(self.group))
        except Exception as e:  # noqa: BLE001
            observed = f"<error: {type(e).__name__}: {e}>"
        if "nccl" in observed.lower():
            return True, observed
        try:
            cuda_backend = self.group._get_backend(torch.device("cuda"))
            if (
                cuda_backend is not None
                and "nccl" in type(cuda_backend).__name__.lower()
            ):
                return True, f"{observed} (cuda: {type(cuda_backend).__name__})"
        except Exception:  # noqa: BLE001 — no CUDA backend bound
            pass
        return False, observed

    def _gather(self, payload: Any) -> List[Any]:
        out: List[Any] = [None] * self.world_size
        # once the communicator device is resolved, metadata collectives must
        # not run on whatever the caller's current device happens to be
        device = getattr(self, "device", None)
        if device is not None:
            with torch.cuda.device(device):
                dist.all_gather_object(out, payload, group=self.group)
        else:
            dist.all_gather_object(out, payload, group=self.group)
        return out

    @staticmethod
    def _parse_cuda_ordinal(device) -> Tuple[Optional[int], Optional[str]]:
        """Strictly parse a device spec into a CUDA ordinal.

        Returns ``(index_or_None_for_current, error_or_None)``. torch.device
        wraps ordinals into a signed byte (``cuda:256`` silently becomes
        ``cuda:0``), so raw str/int ordinals are validated BEFORE any torch
        normalization; pre-built torch.device objects can only be checked for
        the surviving (possibly wrapped) index range.
        """
        count = torch.cuda.device_count()
        if device is None:
            return None, None
        if isinstance(device, bool):
            return None, f"invalid type: {type(device).__name__}"
        if isinstance(device, int):
            if 0 <= device < count:
                return device, None
            return None, f"ordinal {device} outside visible device count {count}"
        if isinstance(device, str):
            m = re.fullmatch(r"\s*cuda(?::(\d+))?\s*", device)
            if m is None:
                try:
                    parsed = torch.device(device)
                except (RuntimeError, ValueError, TypeError) as e:
                    return None, f"unparsable device: {e}"
                if parsed.type != "cuda":
                    return None, f"device must be a CUDA device, got {parsed}"
                return parsed.index, None
            if m.group(1) is None:
                return None, None  # bare "cuda" == current device
            idx = int(m.group(1))
            if 0 <= idx < count:
                return idx, None
            return None, f"ordinal {idx} outside visible device count {count}"
        if isinstance(device, torch.device):
            if device.type != "cuda":
                return None, f"device must be a CUDA device, got {device}"
            if device.index is None:
                return None, None
            if 0 <= device.index < count:
                return device.index, None
            return None, f"index {device.index} outside visible device count {count}"
        return None, f"invalid type: {type(device).__name__}"

    @classmethod
    def _resolve_device(cls, device) -> torch.device:
        """Never raises: yields the bound device for valid input and a safe
        gather-guard device (the current one) otherwise — the joint config
        validation rejects the invalid input right after."""
        try:
            index, err = cls._parse_cuda_ordinal(device)
            if err is not None:
                index = None
            if index is None:
                index = torch.cuda.current_device()
            return torch.device("cuda", index)
        except Exception:  # noqa: BLE001
            return torch.device("cuda", 0)

    @classmethod
    def _encode_config(cls, max_elems, dtype, device) -> Tuple[str, str, str]:
        if type(max_elems) is not int:  # bool is an int subclass: reject it too
            elems = f"<invalid type: {type(max_elems).__name__}>"
        else:
            elems = str(max_elems)
        if isinstance(dtype, torch.dtype):
            dt = str(dtype)
        else:
            dt = f"<invalid type: {type(dtype).__name__}>"
        try:
            index, err = cls._parse_cuda_ordinal(device)
        except Exception as e:  # noqa: BLE001
            index, err = None, f"{type(e).__name__}: {e}"
        if err is not None:
            dev = f"<invalid device: {err}>"
        elif index is None:
            dev = "cuda"
        else:
            dev = f"cuda:{index}"
        return (elems, dt, dev)

    def _validate_configs_jointly(self, configs) -> None:
        supported = tuple(str(d) for d in _SUPPORTED_DTYPES)
        problems = {}
        for r, (elems, dt, dev) in enumerate(configs):
            errs = []
            if not elems.isdigit() or int(elems) <= 0:
                errs.append(f"max_elems must be a positive int, got {elems}")
            elif int(elems) > _INT32_MAX:
                errs.append(
                    f"max_elems must be at most {_INT32_MAX} (int32 kernel "
                    f"index range), got {elems}"
                )
            if dt not in supported:
                errs.append(f"dtype must be one of {supported}, got {dt}")
            if not dev.startswith("cuda"):
                errs.append(f"device must be a CUDA device, got {dev}")
            if errs:
                problems[r] = "; ".join(errs)
        if problems:
            raise ValueError(f"invalid UlyssesCommunicator config by rank: {problems}")
        shared = {(elems, dt) for (elems, dt, _dev) in configs}
        if len(shared) > 1:
            raise ValueError(
                f"inconsistent UlyssesCommunicator configs across ranks: "
                f"(max_elems, dtype) = {sorted(shared)}; all ranks must agree"
            )

    # ---- staged NVLink initialization (collective-safe transaction) -----------
    #
    # Every stage ends with an outcome all-gather, so a rank-local failure at
    # any point (JIT compile, cudaMalloc, IPC get-handle, IPC open, kernel
    # init) is seen by all ranks together; they then run the same staged
    # cleanup (close imports -> gather -> free exports -> gather) and return
    # the same joint error. No bare barrier is ever reached by only a subset
    # of ranks.

    def _nvlink_init_transaction(self) -> Optional[str]:
        # stage J: JIT compile / load both modules and read the signal size.
        # Every import (including cudart below) lives inside a stage envelope:
        # an import failing on one rank must become a gathered outcome, not an
        # exception escaping before a gather.
        try:
            from .ulysses_a2a import get_ulysses_a2a_module
            from .vllm_ar import meta_size

            with torch.cuda.device(self.device):
                get_ulysses_a2a_module()
                sig_bytes = int(meta_size())
            outcome: Tuple[str, ...] = ("ok", str(sig_bytes))
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} JIT/meta: {type(e).__name__}: {e}")
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return err  # nothing allocated anywhere yet

        # stage A: allocate this rank's export buffers and IPC handles
        out_bytes = self.max_elems * self.dtype.itemsize
        handles: Optional[Tuple[Any, Any]] = None
        try:
            from .cuda_ipc import cudart

            with torch.cuda.device(self.device):
                out_ptr = cudart.cudaMalloc(out_bytes)
                self._exports.append(out_ptr.value)
                out_handle = cudart.cudaIpcGetMemHandle(out_ptr)
                sig_ptr = cudart.cudaMalloc(sig_bytes)
                self._exports.append(sig_ptr.value)
                sig_handle = cudart.cudaIpcGetMemHandle(sig_ptr)
            handles = (out_handle, sig_handle)
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} alloc: {type(e).__name__}: {e}")
        gathered = self._gather((outcome, handles))
        err = self._first_error([o for (o, _h) in gathered])
        if err is not None:
            return self._staged_cleanup(err)

        # stage B: open every peer's handles
        all_handles = [h for (_o, h) in gathered]
        out_ptrs: List[int] = [0] * self.world_size
        sig_ptrs: List[int] = [0] * self.world_size
        try:
            from .cuda_ipc import cudart

            with torch.cuda.device(self.device):
                for i, pair in enumerate(all_handles):
                    if i == self.rank:
                        out_ptrs[i] = self._exports[0]
                        sig_ptrs[i] = self._exports[1]
                        continue
                    p = cudart.cudaIpcOpenMemHandle(pair[0])
                    self._imports.append(p.value)
                    out_ptrs[i] = p.value
                    p = cudart.cudaIpcOpenMemHandle(pair[1])
                    self._imports.append(p.value)
                    sig_ptrs[i] = p.value
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} IPC open: {type(e).__name__}: {e}")
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return self._staged_cleanup(err)

        # stage C: create the kernel handle (zeroes this rank's signal buffer)
        # and synchronize the bound device before reporting success — the
        # zeroing uses cudaMemset, which is asynchronous with respect to the
        # host, so neither the API returning nor the following host-side
        # gather is a CUDA completion fence on its own.
        try:
            from .ulysses_a2a import init_ulysses_a2a

            with torch.cuda.device(self.device):
                self._fa = init_ulysses_a2a(
                    out_ptrs, sig_ptrs, self.rank, self.world_size, True
                )
                torch.cuda.synchronize()
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} init: {type(e).__name__}: {e}")
        # once every rank passes this gather, every rank's signal buffer is
        # both zeroed on-device (explicit synchronize above) and visible.
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return self._staged_cleanup(err)

        self._out_ptrs = out_ptrs
        self._sig_ptrs = sig_ptrs
        self._nvlink_armed = True
        return None

    @staticmethod
    def _first_error(outcomes: List[Tuple[str, ...]]) -> Optional[str]:
        errs = [o[1] for o in outcomes if o and o[0] == "err"]
        return "; ".join(errs) if errs else None

    def _staged_cleanup(self, err: str) -> str:
        """Joint init-failure cleanup: all ranks arrive here together (they
        all saw the same failed outcome gather) and run the full teardown
        protocol. Cleanup completion is *verified* group-wide; if it cannot
        be completed the constructor fails jointly on every rank (auto is not
        allowed to fall back to NCCL while NVLink resources may linger)."""
        cleanup_err = self._teardown_protocol(sync_first=True)
        if cleanup_err is not None:
            raise RuntimeError(
                f"NVLink backend initialization failed ({err}) and cleanup "
                f"could not be completed: {cleanup_err}"
            )
        return err

    # ---- staged teardown protocol ---------------------------------------------
    #
    # Fixed stage sequence executed by EVERY rank whenever it runs, regardless
    # of how many resources the rank still holds locally (a rank with nothing
    # left still participates in every gather — otherwise a retry after a
    # partial failure deadlocks the ranks that do have work left). Each stage
    # drains with bounded retries; the retry/stop decision is taken from the
    # gathered remaining-counts, so every rank takes the same branch.

    _TEARDOWN_ATTEMPTS = 3

    def _teardown_protocol(self, *, sync_first: bool) -> Optional[str]:
        stages = []
        if sync_first:
            # collectives/memsets are async enqueues: never unmap while the
            # bound device may still be executing one
            stages.append(("synchronize device", self._try_sync))
        stages.append(("dispose kernel handle", self._try_dispose))
        stages.append(("close peer mappings", self._try_close_imports))
        # exports are freed only after the gathered remaining-import count is
        # zero on EVERY rank: freeing a buffer a peer still has mapped is
        # undefined behavior
        stages.append(("free exports", self._try_free_exports))

        for name, step in stages:
            for attempt in range(1, self._TEARDOWN_ATTEMPTS + 1):
                # broad envelope around the WHOLE step: a helper that raises
                # (module import, device-guard enter/exit, anything) must
                # become a nonzero remaining-count, never skip the gather and
                # strand the peers
                try:
                    remaining, detail = step()
                except Exception as e:  # noqa: BLE001
                    remaining = 1
                    detail = (
                        f"rank {self.rank} stage '{name}' raised: "
                        f"{type(e).__name__}: {e}"
                    )
                outcomes = self._gather((remaining, detail))
                if all(r == 0 for (r, _d) in outcomes):
                    break  # stage complete on every rank
                if attempt == self._TEARDOWN_ATTEMPTS:
                    per_rank = {r: d for r, (n, d) in enumerate(outcomes) if n > 0}
                    return f"stage '{name}' incomplete after {attempt} attempts: {per_rank}"
        return None

    def _try_sync(self) -> Tuple[int, Optional[str]]:
        try:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
            return (0, None)
        except Exception as e:  # noqa: BLE001
            return (1, f"rank {self.rank} synchronize: {type(e).__name__}: {e}")

    def _try_dispose(self) -> Tuple[int, Optional[str]]:
        if self._fa is None:
            return (0, None)
        try:
            from .ulysses_a2a import dispose_ulysses_a2a

            with torch.cuda.device(self.device):
                dispose_ulysses_a2a(self._fa)
                # ledger update inside the guard: a __exit__ raise after a
                # successful dispose must not lead to a double-delete on retry
                self._fa = None
            return (0, None)
        except Exception as e:  # noqa: BLE001
            return (
                0 if self._fa is None else 1,
                f"rank {self.rank} dispose: {type(e).__name__}: {e}",
            )

    # The release helpers update the resource ledger immediately after each
    # successful release (inside the per-pointer try, device guard included),
    # so a later failure — even a device-guard __exit__ raising — can never
    # lead to a double-close/double-free on the next attempt.

    def _try_close_imports(self) -> Tuple[int, Optional[str]]:
        from .cuda_ipc import cudart

        last = None
        for ptr in list(self._imports):
            try:
                with torch.cuda.device(self.device):
                    cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
                    # ledger update inside the guard: even a __exit__ raise
                    # after a successful close cannot cause a double-close
                    self._imports.remove(ptr)
            except Exception as e:  # noqa: BLE001 — keep for retry
                last = f"rank {self.rank} close import: {type(e).__name__}: {e}"
        return (len(self._imports), last)

    def _try_free_exports(self) -> Tuple[int, Optional[str]]:
        from .cuda_ipc import cudart

        last = None
        for ptr in list(self._exports):
            try:
                with torch.cuda.device(self.device):
                    cudart.cudaFree(ctypes.c_void_p(ptr))
                    self._exports.remove(ptr)
            except Exception as e:  # noqa: BLE001 — keep for retry
                last = f"rank {self.rank} free export: {type(e).__name__}: {e}"
        return (len(self._exports), last)

    # ---- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        r"""Release the communicator. Idempotent once fully closed.

        Collective when the NVLink backend was armed: every rank must call
        ``close`` together, and every rank runs the same fixed teardown stage
        sequence even if it holds no resources locally — synchronize the
        bound device (collectives are asynchronous kernel launches; unmapping
        a peer buffer still in use would be undefined behavior), dispose the
        kernel handle, close peer mappings, and only after the group confirms
        all mappings are closed, free the exports. Each stage drains with
        bounded group-coordinated retries. If teardown still cannot complete,
        the call raises the same error on **all** ranks and the state stays
        CLOSING; every rank may retry ``close()``. The state becomes CLOSED
        only after a fully successful group-wide teardown. The pure-NCCL
        backend holds no resources and closes locally.
        """
        if self._state == _CLOSED:
            return
        self._state = _CLOSING

        if not getattr(self, "_nvlink_armed", False):
            # never held NVLink resources on ANY rank (armed is a joint
            # property: the init transaction either succeeds or cleans up on
            # every rank together), so closing locally cannot desync peers
            self._state = _CLOSED
            return

        err = self._teardown_protocol(sync_first=True)
        if err is not None:
            raise RuntimeError(
                f"UlyssesCommunicator.close failed (retry close() on all ranks): {err}"
            )
        self._out_ptrs = None
        self._sig_ptrs = None
        self._nvlink_armed = False
        self._state = _CLOSED

    def __enter__(self) -> "UlyssesCommunicator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ---- collectives -----------------------------------------------------------

    def scatter_heads(self, x: torch.Tensor) -> torch.Tensor:
        r"""``[B, S_local, H, D] -> [B, S_global, H_local, D]``.

        Scatter the global heads across ranks and gather the full sequence:
        afterwards this rank holds head slice
        ``[rank * H_local, (rank+1) * H_local)`` of every token. Runs on the
        current CUDA stream. Returns the input unchanged when
        ``world_size == 1``.
        """
        self._validate(x, "scatter_heads")
        B, S_local, H, D = x.shape
        if H % self.world_size != 0:
            raise ValueError(
                f"scatter_heads requires the global head count (dim 2) to be "
                f"divisible by world size {self.world_size}, got shape "
                f"{tuple(x.shape)}"
            )
        if self.world_size == 1:
            return x
        if self.backend == "nvlink":
            from .ulysses_a2a import ulysses_a2a

            out = torch.empty(
                B,
                S_local * self.world_size,
                H // self.world_size,
                D,
                dtype=x.dtype,
                device=x.device,
            )
            ulysses_a2a(self._fa, x, out, B, S_local, H, D, 0)
            return out
        return self._nccl_scatter_heads(x)

    def gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        r"""``[B, S_global, H_local, D] -> [B, S_local, H, D]``.

        Inverse of :meth:`scatter_heads`: gather all head slices for this
        rank's local sequence shard. Runs on the current CUDA stream. Returns
        the input unchanged when ``world_size == 1``.
        """
        self._validate(x, "gather_heads")
        B, S_global, H_local, D = x.shape
        if S_global % self.world_size != 0:
            raise ValueError(
                f"gather_heads requires the global sequence length (dim 1) to "
                f"be divisible by world size {self.world_size}, got shape "
                f"{tuple(x.shape)}"
            )
        if self.world_size == 1:
            return x
        S_local = S_global // self.world_size
        H = H_local * self.world_size
        if self.backend == "nvlink":
            from .ulysses_a2a import ulysses_a2a

            out = torch.empty(B, S_local, H, D, dtype=x.dtype, device=x.device)
            ulysses_a2a(self._fa, x, out, B, S_local, H, D, 1)
            return out
        return self._nccl_gather_heads(x)

    # ---- NCCL fallback ---------------------------------------------------------
    # The conventional all_to_all_single path with explicit permute/contiguous
    # glue before and after (exactly the data movement the fused NVLink kernel
    # folds into its cross-GPU writes). Bit-identical to the NVLink backend.

    def _nccl_scatter_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S_local, H, D = x.shape
        W = self.world_size
        H_local = H // W
        xt = x.reshape(B, S_local, W, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        # chunk j == rank j's contribution to sequence block j
        return recv.permute(1, 0, 2, 3, 4).reshape(B, W * S_local, H_local, D)

    def _nccl_gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S_global, H_local, D = x.shape
        W = self.world_size
        S_local = S_global // W
        xt = x.reshape(B, W, S_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        # chunk p == this rank's sequence block, head slice p
        return (
            recv.permute(1, 2, 0, 3, 4).reshape(B, S_local, W * H_local, D).contiguous()
        )

    # ---- validation ------------------------------------------------------------

    def _validate(self, x, op: str) -> None:
        if self._state != _OPEN:
            raise RuntimeError(
                f"{op} called on a {self._state} UlyssesCommunicator (use-after-close)"
            )
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{op} expects a torch.Tensor, got {type(x).__name__}")
        if x.dim() != 4:
            raise ValueError(
                f"{op} expects a 4-D [B, S, H, D] tensor, got {x.dim()}-D shape "
                f"{tuple(x.shape)}"
            )
        if x.device != self.device:
            raise ValueError(
                f"{op} tensor is on {x.device}, but this communicator is bound "
                f"to {self.device}"
            )
        if x.dtype != self.dtype:
            raise ValueError(
                f"{op} tensor dtype {x.dtype} does not match the communicator "
                f"dtype {self.dtype}"
            )
        if not x.is_contiguous():
            raise ValueError(f"{op} tensor must be contiguous")
        if any(s <= 0 for s in x.shape):
            raise ValueError(
                f"{op} tensor dims must all be positive, got shape {tuple(x.shape)}"
            )
        if x.numel() > self.max_elems:
            raise ValueError(
                f"{op} tensor has {x.numel()} elements, exceeding the "
                f"communicator capacity max_elems={self.max_elems} "
                f"(which is capped at the int32 index range {_INT32_MAX})"
            )
