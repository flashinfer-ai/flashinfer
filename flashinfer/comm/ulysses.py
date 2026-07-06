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

    Provides the two collectives of Ulysses attention over the 4-D layout
    ``[B, S, H, D]``:

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
    - ``backend="nccl"``: force the ``dist.all_to_all_single`` path; never
      probes CUDA/NVML. Supports any world size.

    The NCCL path with ``world_size > 1`` requires ``group`` to support CUDA
    all-to-all (an NCCL process group); this is checked at construction.
    ``world_size == 1`` is a passthrough: both collectives return the input
    tensor unchanged (no copy).

    Constraints
    -----------
    - The constructor and :meth:`close` are collective: every rank of
      ``group`` must call them together. Rank-local failures inside the
      constructor's NVLink initialization or inside ``close`` are exchanged as
      group outcomes, so all ranks jointly clean up and raise (or fall back)
      instead of deadlocking; a failed ``close`` may be retried by all ranks.
    - Collectives run on the *current* CUDA stream of this rank; every rank
      must issue the same sequence of calls with consistently-shaped operands
      (a shape or call-order mismatch across ranks is a collective failure:
      expect hangs or garbage, exactly as with any collective library). At
      most one collective may be in flight per communicator at a time (the
      NVLink signal buffers assume serialized calls); do not call one
      communicator concurrently from multiple streams or threads.
    - Operand tensors must be contiguous 4-D CUDA tensors of the construction
      ``dtype`` on the construction device, with every dim positive and total
      elements at most ``max_elems``.
    - Each rank may use a different CUDA device (e.g. ``cuda:rank``); ranks
      must agree on ``max_elems`` and ``dtype``.

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
    device : torch.device, optional
        CUDA device of this rank; normalized to an explicit index (bare
        ``"cuda"`` means the current device). Defaults to the current CUDA
        device.

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
        if device is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device(device)
            # bare "cuda" means the *current* device, not GPU 0
            index = (
                device.index
                if device.index is not None
                else (torch.cuda.current_device())
            )
            self.device = torch.device("cuda", index)

        # ---- backend selection: strictly before any IPC/JIT -----------------
        self.decision: UlyssesBackendDecision = resolve_ulysses_backend(
            backend, group=group, device=self.device
        )
        self.backend = self.decision.backend
        self.fallback_reason = (
            self.decision.reason
            if self.backend == "nccl" and backend != "nccl"
            else None
        )

        if self.backend == "nvlink":
            err = self._nvlink_init_transaction()
            if err is not None:
                # all ranks cleaned up and hold the same joint error
                if backend == "nvlink":
                    raise RuntimeError(f"NVLink backend initialization failed: {err}")
                self.backend = "nccl"
                self.fallback_reason = f"nvlink init failed: {err}"

        # NCCL fallback needs a group that can move CUDA tensors; deterministic
        # in the (identical) group object, so a plain raise is group-uniform.
        if self.backend == "nccl" and self.world_size > 1:
            pg_backend = str(dist.get_backend(self.group))
            if "nccl" not in pg_backend.lower():
                raise ValueError(
                    "the Ulysses NCCL backend requires a process group "
                    f"supporting CUDA all-to-all (nccl), got '{pg_backend}'"
                )

        self._state = _OPEN

    # ---- collective helpers ---------------------------------------------------

    def _gather(self, payload: Any) -> List[Any]:
        out: List[Any] = [None] * self.world_size
        dist.all_gather_object(out, payload, group=self.group)
        return out

    @staticmethod
    def _encode_config(max_elems, dtype, device) -> Tuple[str, str, str]:
        if type(max_elems) is not int:  # bool is an int subclass: reject it too
            elems = f"<invalid type: {type(max_elems).__name__}>"
        else:
            elems = str(max_elems)
        if isinstance(dtype, torch.dtype):
            dt = str(dtype)
        else:
            dt = f"<invalid type: {type(dtype).__name__}>"
        if device is None:
            dev = "cuda"
        elif isinstance(device, (torch.device, str, int)):
            try:
                dev = str(torch.device(device))
            except (RuntimeError, ValueError, TypeError) as e:
                dev = f"<invalid device: {e}>"
        else:
            dev = f"<invalid type: {type(device).__name__}>"
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
        from .cuda_ipc import cudart

        # stage J: JIT compile / load both modules and read the signal size
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
        try:
            from .ulysses_a2a import init_ulysses_a2a

            with torch.cuda.device(self.device):
                self._fa = init_ulysses_a2a(
                    out_ptrs, sig_ptrs, self.rank, self.world_size, True
                )
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} init: {type(e).__name__}: {e}")
        # this gather is also the required post-init synchronization point: the
        # signal zeroing is a synchronous memset, and no rank can pass this
        # gather before every rank finished stage C.
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return self._staged_cleanup(err)

        self._out_ptrs = out_ptrs
        self._sig_ptrs = sig_ptrs
        return None

    @staticmethod
    def _first_error(outcomes: List[Tuple[str, ...]]) -> Optional[str]:
        errs = [o[1] for o in outcomes if o and o[0] == "err"]
        return "; ".join(errs) if errs else None

    def _staged_cleanup(self, err: str) -> str:
        """Joint failure cleanup: all ranks arrive here together (they all saw
        the same failed outcome gather). Close imports first, confirm via
        gather, then free exports — never free an export while a peer may
        still have it mapped."""
        self._release_local(imports=True, exports=False)
        self._gather(("cleanup-imports-done",))
        self._release_local(imports=False, exports=True)
        self._gather(("cleanup-exports-done",))
        return err

    def _release_local(self, *, imports: bool, exports: bool) -> None:
        from .cuda_ipc import cudart

        if self._fa is not None:
            from .ulysses_a2a import dispose_ulysses_a2a

            dispose_ulysses_a2a(self._fa)
            self._fa = None
        with torch.cuda.device(self.device):
            if imports:
                remaining = []
                for ptr in self._imports:
                    try:
                        cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
                    except Exception:  # noqa: BLE001 — keep for retry
                        remaining.append(ptr)
                self._imports = remaining
            if exports:
                remaining = []
                for ptr in self._exports:
                    try:
                        cudart.cudaFree(ctypes.c_void_p(ptr))
                    except Exception:  # noqa: BLE001 — keep for retry
                        remaining.append(ptr)
                self._exports = remaining

    # ---- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        r"""Release the communicator. Idempotent once fully closed.

        Collective for the NVLink backend: every rank must call ``close``
        together. The bound device is synchronized first (collectives are
        asynchronous kernel launches; unmapping a peer buffer still in use
        would be undefined behavior), then teardown proceeds in stages with
        group outcome exchanges: close peer mappings -> confirm -> free own
        allocations -> confirm. If any rank reports a failure the call raises
        on **all** ranks, already-released resources stay released, and every
        rank may retry ``close()``; the state becomes CLOSED only after a
        fully successful teardown. The NCCL backend holds no resources and
        closes locally.
        """
        if self._state == _CLOSED:
            return
        self._state = _CLOSING

        has_resources = bool(self._imports or self._exports or self._fa)
        if self.backend != "nvlink" or not has_resources:
            self._state = _CLOSED
            return

        # collectives are async enqueues: never unmap while the device may
        # still be executing one
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()

        # stage 1: close peer imports everywhere
        self._release_local(imports=True, exports=False)
        o1 = (
            ("ok",)
            if not self._imports
            else (
                "err",
                f"rank {self.rank} failed to close {len(self._imports)} peer mapping(s)",
            )
        )
        outcomes1 = self._gather(o1)

        # stage 2: free own exports only after every rank confirmed stage 1
        # (freeing a buffer a peer still has mapped is undefined behavior; if
        # any rank failed stage 1 we still must not free, so skip and report)
        err1 = self._first_error(outcomes1)
        if err1 is None:
            self._release_local(imports=False, exports=True)
            o2 = (
                ("ok",)
                if not self._exports
                else (
                    "err",
                    f"rank {self.rank} failed to free {len(self._exports)} export(s)",
                )
            )
        else:
            o2 = ("err", f"rank {self.rank} skipped export free: peers failed stage 1")
        outcomes2 = self._gather(o2)

        err = self._first_error(outcomes1) or self._first_error(outcomes2)
        if err is not None:
            raise RuntimeError(
                f"UlyssesCommunicator.close failed (retry close() on all ranks): {err}"
            )
        self._out_ptrs = None
        self._sig_ptrs = None
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
