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
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .ulysses_topology import UlyssesBackendDecision, resolve_ulysses_backend

_INT32_MAX = 2**31 - 1
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
      size (2/4/6/8); NCCL otherwise. Inspect :attr:`backend` and
      :attr:`fallback_reason` for the outcome.
    - ``backend="nvlink"``: force the fused kernel; raises
      :class:`~flashinfer.comm.UlyssesBackendError` on every rank (before any
      IPC/JIT) when the topology cannot support it.
    - ``backend="nccl"``: force the ``dist.all_to_all_single`` path; never
      probes CUDA/NVML. Supports any world size.

    ``world_size == 1`` is a passthrough: both collectives return the input
    tensor unchanged (no copy).

    Constraints
    -----------
    - The constructor and :meth:`close` are collective: every rank of
      ``group`` must call them together (constructor performs all-gathers and
      barriers; close unmaps/frees IPC buffers for the NVLink backend).
    - Collectives run on the *current* CUDA stream of this rank; all ranks
      must issue the same sequence of calls. At most one collective may be in
      flight per communicator at a time (the NVLink signal buffers assume
      serialized calls); do not call one communicator concurrently from
      multiple streams or threads.
    - Operand tensors must be contiguous 4-D CUDA tensors of the construction
      ``dtype`` on the construction device, with every dim positive, total
      elements within int32 range and at most ``max_elems``.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup, optional
        Process group of the Ulysses ranks. Defaults to ``dist.group.WORLD``.
    max_elems : int
        Capacity: the largest element count of any single all-to-all operand
        (input and output have equal ``numel``; for Wan-style usage this is
        ``B * S_global * H * D // world_size``... i.e. ``B*S_local*H*D``).
        Sizes the NVLink staging buffer once at construction.
    dtype : torch.dtype
        Element type of all operands (float16 / bfloat16 / float32); enforced
        on every call.
    backend : str
        ``"auto"`` | ``"nvlink"`` | ``"nccl"`` (see above).
    device : torch.device, optional
        CUDA device of this rank. Defaults to the current CUDA device.

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
        self._closed = False
        self._fa: Optional[int] = None
        self._out_ptrs: Optional[List[int]] = None
        self._sig_ptrs: Optional[List[int]] = None

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
        config = self._encode_config(max_elems, dtype, device)
        configs: List[Optional[Tuple[str, ...]]] = [None] * self.world_size
        dist.all_gather_object(configs, config, group=group)
        self._validate_configs_jointly(configs)

        self.max_elems = max_elems
        self.dtype = dtype
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )

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
            self._init_nvlink()

    @staticmethod
    def _encode_config(max_elems, dtype, device) -> Tuple[str, ...]:
        if type(max_elems) is not int:  # bool is an int subclass: reject it too
            elems = f"<invalid type: {type(max_elems).__name__}>"
        else:
            elems = str(max_elems)
        if isinstance(dtype, torch.dtype):
            dt = str(dtype)
        else:
            dt = f"<invalid type: {type(dtype).__name__}>"
        if device is None:
            dev = "cuda:<current>"
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
            if dt not in supported:
                errs.append(f"dtype must be one of {supported}, got {dt}")
            if not dev.startswith("cuda"):
                errs.append(f"device must be a CUDA device, got {dev}")
            if errs:
                problems[r] = "; ".join(errs)
        if problems:
            raise ValueError(f"invalid UlyssesCommunicator config by rank: {problems}")
        if len(set(configs)) > 1:
            raise ValueError(
                f"inconsistent UlyssesCommunicator configs across ranks "
                f"(max_elems, dtype, device): {configs}; all ranks must agree "
                "on max_elems and dtype"
            )

    def _init_nvlink(self) -> None:
        # Imported here so the NCCL fallback path never touches the IPC helper
        # or the JIT module machinery.
        from .cuda_ipc import create_shared_buffer
        from .ulysses_a2a import init_ulysses_a2a
        from .vllm_ar import meta_size

        try:
            self._out_ptrs = create_shared_buffer(
                self.max_elems * self.dtype.itemsize, group=self.group
            )
            self._sig_ptrs = create_shared_buffer(meta_size(), group=self.group)
            self._fa = init_ulysses_a2a(
                self._out_ptrs, self._sig_ptrs, self.rank, self.world_size, True
            )
            # init zeroed this rank's signal buffer; make the zeroing globally
            # visible before the first collective.
            dist.barrier(group=self.group)
        except Exception:
            # Roll back rank-local resources only: peers may be failing too,
            # so no collectives here (free_shared_buffer would barrier).
            self._rollback_local()
            raise

    def _rollback_local(self) -> None:
        from .cuda_ipc import cudart

        if self._fa is not None:
            from .ulysses_a2a import dispose_ulysses_a2a

            dispose_ulysses_a2a(self._fa)
            self._fa = None
        for ptrs in (self._out_ptrs, self._sig_ptrs):
            if ptrs is None:
                continue
            for i, ptr in enumerate(ptrs):
                if ptr is None:
                    continue
                try:
                    if i == self.rank:
                        cudart.cudaFree(ctypes.c_void_p(ptr))
                    else:
                        cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass
        self._out_ptrs = None
        self._sig_ptrs = None

    # ---- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        r"""Release the communicator. Idempotent.

        Collective for the NVLink backend: every rank must call ``close``
        together, with no collective still executing (synchronize the streams
        that issued :meth:`scatter_heads` / :meth:`gather_heads` first). The
        NCCL backend holds no resources and closes locally.
        """
        if self._closed:
            return
        self._closed = True
        if self._fa is not None:
            from .ulysses_a2a import dispose_ulysses_a2a

            dispose_ulysses_a2a(self._fa)
            self._fa = None
        from .cuda_ipc import free_shared_buffer

        if self._out_ptrs is not None:
            free_shared_buffer(self._out_ptrs, group=self.group)
            self._out_ptrs = None
        if self._sig_ptrs is not None:
            free_shared_buffer(self._sig_ptrs, group=self.group)
            self._sig_ptrs = None

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
        if self._closed:
            raise RuntimeError(
                f"{op} called on a closed UlyssesCommunicator (use-after-close)"
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
        if x.numel() > _INT32_MAX:
            raise ValueError(
                f"{op} tensor has {x.numel()} elements, exceeding the int32 "
                f"kernel index range ({_INT32_MAX})"
            )
        if x.numel() > self.max_elems:
            raise ValueError(
                f"{op} tensor has {x.numel()} elements, exceeding the "
                f"communicator capacity max_elems={self.max_elems}"
            )
