# flashinfer/ep/group.py
#
# EpGroup class and top-level group lifecycle functions.
#
# Python is a THIN PASSTHROUGH — all heavy work (routing computation,
# NCCL alltoall, weighted reduction) happens in the JIT-compiled C++ module
# (csrc/ep/bindings.cu). Python only:
#   - Converts torch types ↔ TVM FFI
#   - Extracts ncclComm_t from the PyTorch process group
#   - Wraps C++ return values in Python dataclasses

import functools
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.distributed as dist

from flashinfer.ep.types import Backend, OutputLayout, EpStatus, EpError


# ─── JIT module loading ─────────────────────────────────────────────

@functools.cache
def _get_ep_module():
    """Load the core EP JIT module.

    Triggers JIT compilation (ninja + nvcc) on first call.
    Subsequent calls return the cached module.
    """
    from flashinfer.jit.ep import gen_ep_module
    return gen_ep_module().build_and_load()


@functools.cache
def _get_ep_deepep_module():
    """Load the DeepEP backend module. Returns None if not installed."""
    try:
        from flashinfer.jit.ep import gen_ep_deepep_module
        return gen_ep_deepep_module().build_and_load()
    except (ImportError, RuntimeError):
        return None


@functools.cache
def _get_ep_nccl_module():
    """Load the NCCL-EP backend module. Returns None if headers not available."""
    try:
        from flashinfer.jit.ep import gen_ep_nccl_module
        return gen_ep_nccl_module().build_and_load()
    except (ImportError, RuntimeError):
        return None


# ─── ncclComm_t extraction ──────────────────────────────────────────

def _extract_nccl_comm_ptr(process_group: dist.ProcessGroup) -> int:
    """Extract raw ncclComm_t pointer from a PyTorch process group.

    Uses the _get_backend()._get_nccl_comm() pattern established by
    vLLM and SGLang (Issue #14, #35).
    Returns the pointer as an int64 for passing through TVM FFI.

    Raises RuntimeError with diagnostic info if extraction fails.
    """
    import ctypes
    device = torch.device("cuda", torch.cuda.current_device())

    # Step 1: Get the NCCL backend from the process group
    try:
        pg_backend = process_group._get_backend(device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to get NCCL backend from process group: {e}\n"
            f"  process_group type: {type(process_group).__name__}\n"
            f"  device: {device}\n"
            f"  Available methods: {[m for m in dir(process_group) if 'backend' in m.lower()]}"
        ) from e

    # Step 2: Extract ncclComm_t — try multiple known APIs
    nccl_comm = None
    errors = []

    # Method 1: _get_nccl_comm() — vLLM/SGLang pattern (PyTorch >= 2.1)
    if hasattr(pg_backend, '_get_nccl_comm'):
        try:
            nccl_comm = pg_backend._get_nccl_comm()
        except Exception as e:
            errors.append(f"_get_nccl_comm(): {e}")

    # Method 2: _get_backend_name() + direct attribute (some PT builds)
    if nccl_comm is None and hasattr(pg_backend, 'nccl_comm'):
        try:
            nccl_comm = pg_backend.nccl_comm
        except Exception as e:
            errors.append(f"pg_backend.nccl_comm: {e}")

    # Method 3: bound_device_id variant (PyTorch nightly)
    if nccl_comm is None and hasattr(pg_backend, 'get_nccl_comm'):
        try:
            nccl_comm = pg_backend.get_nccl_comm()
        except Exception as e:
            errors.append(f"get_nccl_comm(): {e}")

    if nccl_comm is None:
        backend_methods = [m for m in dir(pg_backend) if 'nccl' in m.lower() or 'comm' in m.lower()]
        raise RuntimeError(
            f"Could not extract ncclComm_t from process group backend.\n"
            f"  Backend type: {type(pg_backend).__name__}\n"
            f"  Backend name: {getattr(pg_backend, 'name', 'unknown')}\n"
            f"  NCCL/comm-related methods: {backend_methods}\n"
            f"  All methods tried:\n" +
            "\n".join(f"    - {err}" for err in errors)
        )

    # Step 3: Convert to int64 pointer value
    ptr = _nccl_comm_to_int(nccl_comm)
    if ptr == 0:
        raise RuntimeError(
            f"ncclComm_t extraction returned a NULL pointer.\n"
            f"  Raw value: {nccl_comm!r} (type: {type(nccl_comm).__name__})\n"
            f"  This usually means the NCCL communicator has not been "
            f"initialized yet. Try performing a dummy allreduce on the "
            f"process group before creating the EP group."
        )
    return ptr


def _nccl_comm_to_int(nccl_comm) -> int:
    """Convert various ncclComm_t representations to an int64 pointer value."""
    import ctypes

    # Direct int (already a pointer value)
    if isinstance(nccl_comm, int):
        return nccl_comm

    # Has __int__ (PyCapsule on some PyTorch builds)
    if hasattr(nccl_comm, '__int__'):
        return int(nccl_comm)

    # ctypes pointer (e.g. ctypes.c_void_p)
    if hasattr(nccl_comm, 'value'):
        return nccl_comm.value or 0

    # PyCapsule — extract via ctypes
    if type(nccl_comm).__name__ == 'PyCapsule':
        try:
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            return ctypes.pythonapi.PyCapsule_GetPointer(nccl_comm, None) or 0
        except Exception:
            pass

    # Last resort
    try:
        return int(nccl_comm)
    except (TypeError, ValueError):
        return 0


# ─── Buffer sizing ───────────────────────────────────────────────────

@dataclass
class KernelConfig:
    """Kernel configuration hints."""
    recommended_num_sms: int
    nvl_buffer_hint: int
    rdma_buffer_hint: int


def get_buffer_size_hint(
    backend: Backend,
    hidden_dim: int,
    world_size: int,
    num_experts: int,
    top_k: int,
    max_tokens: int,
) -> Tuple[int, int]:
    """Return (nvl_buffer_bytes, rdma_buffer_bytes)."""
    elem_size = 2  # BF16
    if backend == Backend.DEEP_EP:
        nvl = num_experts * max_tokens * hidden_dim * elem_size
        rdma = nvl
    else:
        nvl = world_size * max_tokens * hidden_dim * elem_size
        rdma = 0
    return (nvl, rdma)


def get_low_latency_rdma_size_hint(
    backend: Backend,
    max_tokens: int,
    hidden_dim: int,
    world_size: int,
    num_experts: int,
) -> int:
    """DeepEP-specific RDMA buffer hint. Returns 0 for NCCL-EP."""
    if backend == Backend.NCCL_EP:
        return 0
    return num_experts * max_tokens * hidden_dim * 2


def get_dispatch_config(
    backend: Backend,
    world_size: int,
    hidden_dim: int,
    num_experts: int,
) -> KernelConfig:
    """Get recommended kernel configuration for dispatch."""
    num_sms = 24 if backend == Backend.DEEP_EP else 16
    nvl = get_buffer_size_hint(backend, hidden_dim, world_size, num_experts, 8, 4096)[0]
    return KernelConfig(recommended_num_sms=num_sms, nvl_buffer_hint=nvl, rdma_buffer_hint=0)


def get_combine_config(
    backend: Backend,
    world_size: int,
    hidden_dim: int,
    num_experts: int,
) -> KernelConfig:
    """Get recommended kernel configuration for combine."""
    return get_dispatch_config(backend, world_size, hidden_dim, num_experts)


# ─── Result types ────────────────────────────────────────────────────

@dataclass
class DispatchResult:
    """Result of a dispatch operation."""
    recv_hidden: torch.Tensor
    recv_topk_idx: torch.Tensor
    recv_topk_weights: torch.Tensor
    recv_expert_counts: torch.Tensor       # [num_local_experts], int32
    recv_scales: Optional[torch.Tensor]
    handle: "EpHandle"
    dep: Optional["StreamDep"]
    status: EpStatus


@dataclass
class CombineResult:
    """Result of a combine operation."""
    combined_hidden: torch.Tensor          # [num_tokens, hidden_dim]
    dep: Optional["StreamDep"]
    status: EpStatus


@dataclass
class LayoutInfo:
    """Pre-computed routing pattern from get_dispatch_layout()."""
    num_tokens_per_rank: torch.Tensor      # [world_size]
    num_tokens_per_expert: torch.Tensor    # [num_experts]
    is_token_in_rank: Optional[torch.Tensor]
    dep: Optional["StreamDep"]


# ─── Handle types ────────────────────────────────────────────────────

class EpHandle:
    """Opaque per-forward handle. Wraps C++ handle_id."""

    def __init__(self, _handle_id: int = -1, _group_id: int = -1):
        self._handle_id = _handle_id
        self._group_id = _group_id
        self._destroyed = False

    def __enter__(self) -> "EpHandle":
        return self

    def __exit__(self, *exc):
        self.destroy()

    def invoke_deferred(self) -> EpStatus:
        """Invoke deferred recv callback."""
        if self._handle_id < 0:
            return EpStatus(error_code=0)
        try:
            mod = _get_ep_module()
            mod.ep_handle_invoke_deferred(self._handle_id)
            return EpStatus(error_code=0)
        except Exception as e:
            return EpStatus(error_code=1, error_msg=str(e))

    def destroy(self):
        """Release handle resources. Safe to call multiple times."""
        if not self._destroyed and self._handle_id >= 0:
            try:
                mod = _get_ep_module()
                mod.ep_destroy_handle(self._handle_id)
            except Exception:
                pass
            self._destroyed = True
            self._handle_id = -1

    def get_num_recv_tokens(self) -> int:
        """Return number of tokens received by this handle."""
        if self._destroyed or self._handle_id < 0:
            raise RuntimeError("Handle has been destroyed")
        mod = _get_ep_module()
        n = mod.ep_handle_get_num_recv(self._handle_id)
        if n < 0:
            raise RuntimeError("Handle is invalid or destroyed")
        return n

    def complete(self, stream: Optional[torch.cuda.Stream] = None,
                 timeout_ms: int = 0):
        """Complete async operation on the given stream."""
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.synchronize()


class RoutingCache:
    """Pre-built routing table for low_latency_dispatch()."""

    def __init__(self, _cache=None, _shape=None):
        self._cache = _cache
        self._shape = _shape
        self._valid = True

    def __enter__(self) -> "RoutingCache":
        return self

    def __exit__(self, *exc):
        self.destroy()

    @property
    def valid(self) -> bool:
        return self._valid

    def is_valid_for(self, topk_idx: torch.Tensor) -> bool:
        """Hot-path check: shape-only comparison (~0 us, no D2H)."""
        if not self._valid or self._shape is None:
            return False
        return tuple(topk_idx.shape) == self._shape

    def destroy(self):
        self._valid = False
        self._cache = None


class StreamDep:
    """CUDA stream dependency."""

    def __init__(self, _event=None):
        self._event = _event

    def wait_on(self, target_stream: torch.cuda.Stream):
        if self._event is not None:
            target_stream.wait_event(self._event)

    def is_complete(self) -> bool:
        if self._event is not None:
            return self._event.query()
        return True

    def synchronize(self):
        if self._event is not None:
            self._event.synchronize()

    @staticmethod
    def create(stream: Optional[torch.cuda.Stream] = None) -> "StreamDep":
        event = torch.cuda.Event()
        if stream is not None:
            event.record(stream)
        else:
            event.record()
        return StreamDep(_event=event)


# ─── EpGroup class ───────────────────────────────────────────────────

class EpGroup:
    """Long-lived communication group.

    Thin Python wrapper — all dispatch/combine work is in C++.
    """

    def __init__(self, _group_id: int = -1, _backend=None, _config=None, _pg=None):
        self._group_id = _group_id
        self._backend = _backend
        self._config = _config or {}
        self._pg = _pg
        self._destroyed = False
        self._comm_stream = torch.cuda.Stream()

    def __enter__(self) -> "EpGroup":
        return self

    def __exit__(self, *exc):
        self.destroy()

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def world_size(self) -> int:
        return self._config.get("world_size", 0)

    @property
    def rank(self) -> int:
        return self._config.get("rank", 0)

    @property
    def num_experts(self) -> int:
        return self._config.get("num_experts", 0)

    @property
    def num_local_experts(self) -> int:
        return self._config.get("num_local_experts", 0)

    @property
    def top_k(self) -> int:
        return self._config.get("top_k", 0)

    def get_comm_stream(self) -> torch.cuda.Stream:
        return self._comm_stream

    def get_config(self) -> dict:
        return dict(self._config)

    # ─── Layout ──────────────────────────────────────────────────────

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        previous_dep: Optional["StreamDep"] = None,
        async_finish: bool = True,
    ) -> "LayoutInfo":
        """Pre-compute routing pattern. D2H memcpy — NOT graph-capturable."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "get_dispatch_layout() performs D2H copies and cannot be "
                "called inside CUDA graph capture."
            )

        mod = _get_ep_module()
        expert_counts = mod.ep_get_dispatch_layout(topk_idx, self._group_id)
        expert_counts_tensor = torch.from_dlpack(expert_counts)

        dep = StreamDep.create() if async_finish else None
        return LayoutInfo(
            num_tokens_per_rank=torch.zeros(self.world_size, dtype=torch.int32),
            num_tokens_per_expert=expert_counts_tensor,
            is_token_in_rank=None,
            dep=dep,
        )

    # ─── High-Throughput Dispatch ────────────────────────────────────

    def dispatch(
        self,
        hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        layout: "LayoutInfo",
        output_layout: OutputLayout = OutputLayout.FLAT_2D,
        previous_dep: Optional["StreamDep"] = None,
        previous_handle: Optional["EpHandle"] = None,
        allocate_on_comm_stream: bool = True,
        async_finish: bool = True,
    ) -> "DispatchResult":
        """Route tokens to experts — delegates entirely to C++."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "HT dispatch uses dynamic sizes and cannot be captured "
                "in a CUDA graph. Use low_latency_dispatch() instead."
            )

        scales = None
        if isinstance(hidden, tuple):
            hidden, scales = hidden

        try:
            mod = _get_ep_module()
            layout_int = 0 if output_layout == OutputLayout.FLAT_2D else 1

            # C++ ep_dispatch returns Tuple(recv_hidden, expert_counts, handle_id)
            result_tuple = mod.ep_dispatch(
                hidden, topk_idx, self._group_id, layout_int
            )

            recv_hidden = torch.from_dlpack(result_tuple[0])
            expert_counts = torch.from_dlpack(result_tuple[1])
            handle_id = int(result_tuple[2])

            handle = EpHandle(_handle_id=handle_id, _group_id=self._group_id)
            dep = StreamDep.create() if async_finish else None

            return DispatchResult(
                recv_hidden=recv_hidden,
                recv_topk_idx=topk_idx,
                recv_topk_weights=topk_weights,
                recv_expert_counts=expert_counts,
                recv_scales=scales,
                handle=handle,
                dep=dep,
                status=EpStatus(error_code=0),
            )
        except Exception as e:
            return DispatchResult(
                recv_hidden=torch.empty(0, device=hidden.device),
                recv_topk_idx=topk_idx,
                recv_topk_weights=topk_weights,
                recv_expert_counts=torch.zeros(
                    self.num_local_experts, device=hidden.device, dtype=torch.int32
                ),
                recv_scales=None,
                handle=EpHandle(),
                dep=None,
                status=EpStatus(error_code=1, error_msg=str(e)),
            )

    # ─── High-Throughput Combine ─────────────────────────────────────

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: "EpHandle",
        topk_weights: Optional[torch.Tensor] = None,
        previous_dep: Optional["StreamDep"] = None,
        allocate_on_comm_stream: bool = True,
        async_finish: bool = True,
    ) -> "CombineResult":
        """Gather expert outputs — delegates entirely to C++."""
        if handle._handle_id < 0:
            return CombineResult(
                combined_hidden=torch.empty(0, device=expert_output.device),
                dep=None,
                status=EpStatus(error_code=1, error_msg="Invalid handle"),
            )

        try:
            mod = _get_ep_module()
            # C++ ep_combine returns Tensor [num_tokens, hidden_dim]
            combined_tensor = mod.ep_combine(
                expert_output, topk_weights, handle._handle_id
            )
            combined = torch.from_dlpack(combined_tensor)

            dep = StreamDep.create() if async_finish else None
            return CombineResult(
                combined_hidden=combined,
                dep=dep,
                status=EpStatus(error_code=0),
            )
        except Exception as e:
            return CombineResult(
                combined_hidden=torch.empty(0, device=expert_output.device),
                dep=None,
                status=EpStatus(error_code=1, error_msg=str(e)),
            )

    # ─── Low-Latency Dispatch ────────────────────────────────────────

    def low_latency_dispatch(
        self,
        hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        max_tokens_per_rank: int,
        output_layout: OutputLayout = OutputLayout.FLAT_2D,
        previous_dep: Optional["StreamDep"] = None,
        previous_handle: Optional["EpHandle"] = None,
        allocate_on_comm_stream: bool = True,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        cached_routing: Optional["RoutingCache"] = None,
    ) -> "DispatchResult":
        """Route tokens to experts (low-latency mode).

        Uses the same C++ dispatch path. LL vs HT distinction matters
        when the full DeepEP/NCCL-EP backends are compiled.
        """
        scales = None
        if isinstance(hidden, tuple):
            hidden, scales = hidden

        num_tokens = hidden.shape[0] if hidden.ndim >= 1 else 0

        # Overflow check
        if num_tokens > max_tokens_per_rank * self.world_size:
            return DispatchResult(
                recv_hidden=torch.empty(0, device=hidden.device),
                recv_topk_idx=topk_idx,
                recv_topk_weights=torch.empty(0, device=hidden.device),
                recv_expert_counts=torch.zeros(
                    self.num_local_experts, device=hidden.device, dtype=torch.int32
                ),
                recv_scales=None,
                handle=EpHandle(),
                dep=None,
                status=EpStatus(
                    error_code=2,
                    error_msg=f"Buffer overflow: {num_tokens} tokens > "
                              f"max_tokens_per_rank * world_size = "
                              f"{max_tokens_per_rank * self.world_size}",
                ),
            )

        # Uniform weights for LL mode
        top_k = topk_idx.shape[1] if topk_idx.ndim > 1 else 1
        topk_weights = torch.ones(
            num_tokens, top_k,
            device=hidden.device, dtype=torch.float32,
        ) / top_k

        # Delegate to C++ dispatch (same path as HT)
        try:
            mod = _get_ep_module()
            layout_int = 0 if output_layout == OutputLayout.FLAT_2D else 1
            result_tuple = mod.ep_dispatch(
                hidden, topk_idx, self._group_id, layout_int
            )

            recv_hidden = torch.from_dlpack(result_tuple[0])
            expert_counts = torch.from_dlpack(result_tuple[1])
            handle_id = int(result_tuple[2])

            handle = EpHandle(_handle_id=handle_id, _group_id=self._group_id)

            return DispatchResult(
                recv_hidden=recv_hidden,
                recv_topk_idx=topk_idx,
                recv_topk_weights=topk_weights,
                recv_expert_counts=expert_counts,
                recv_scales=scales,
                handle=handle,
                dep=None,
                status=EpStatus(error_code=0),
            )
        except Exception as e:
            return DispatchResult(
                recv_hidden=torch.empty(0, device=hidden.device),
                recv_topk_idx=topk_idx,
                recv_topk_weights=topk_weights,
                recv_expert_counts=torch.zeros(
                    self.num_local_experts, device=hidden.device, dtype=torch.int32
                ),
                recv_scales=None,
                handle=EpHandle(),
                dep=None,
                status=EpStatus(error_code=1, error_msg=str(e)),
            )

    # ─── Low-Latency Combine ────────────────────────────────────────

    def low_latency_combine(
        self,
        expert_output: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: "EpHandle",
        previous_dep: Optional["StreamDep"] = None,
        allocate_on_comm_stream: bool = True,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> "CombineResult":
        """Gather expert outputs (low-latency mode) — delegates to C++."""
        return self.combine(
            expert_output=expert_output,
            handle=handle,
            topk_weights=topk_weights,
            previous_dep=previous_dep,
            allocate_on_comm_stream=allocate_on_comm_stream,
            async_finish=async_finish,
        )

    # ─── Routing Cache ───────────────────────────────────────────────

    def create_routing_cache(
        self,
        topk_idx: torch.Tensor,
    ) -> "RoutingCache":
        """Pre-build a routing cache from a representative topk_idx."""
        return RoutingCache(_cache=None, _shape=tuple(topk_idx.shape))

    # ─── Cleanup ─────────────────────────────────────────────────────

    def destroy(self):
        """Release all resources. Safe to call multiple times."""
        if not self._destroyed and self._group_id >= 0:
            try:
                mod = _get_ep_module()
                mod.ep_destroy_group(self._group_id)
            except Exception:
                pass
            self._destroyed = True


# ─── Group creation ──────────────────────────────────────────────────

def create_group(
    backend: Backend,
    process_group: dist.ProcessGroup,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    hidden_dim: int,
    max_dispatch_elem_size: int = 2,
    nvl_buffer_bytes: Optional[int] = None,
    rdma_buffer_bytes: Optional[int] = None,
    num_sms: int = 0,
    num_qps_per_rank: int = 0,
    enable_pcie_fallback: bool = False,
    cuda_graph_max_tokens: int = 0,
    timeout_ms: int = 30000,
    stream: Optional[torch.cuda.Stream] = None,
) -> "EpGroup":
    """Create a communication group for expert parallelism.

    Collective operation — all ranks must call.
    Extracts ncclComm_t and passes it to C++ for direct NCCL P2P calls.
    """
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    config = {
        "backend": backend.value,
        "rank": rank,
        "world_size": world_size,
        "num_experts": num_experts,
        "num_local_experts": num_local_experts,
        "top_k": top_k,
        "hidden_dim": hidden_dim,
        "max_dispatch_elem_size": max_dispatch_elem_size,
        "cuda_graph_max_tokens": cuda_graph_max_tokens,
        "timeout_ms": timeout_ms,
    }

    # Extract ncclComm_t from the PyTorch process group.
    # NOTE: PyTorch lazily creates ncclComm_t on the first collective.
    # If _get_nccl_comm() returns NULL, the caller must run at least one
    # collective (e.g. dist.all_reduce) on the process group before
    # calling create_group(). The error message from _extract_nccl_comm_ptr
    # will explain this if it happens.
    nccl_comm_ptr = _extract_nccl_comm_ptr(process_group)

    # Pass everything to C++
    mod = _get_ep_module()
    backend_int = 0 if backend == Backend.DEEP_EP else 1
    group_id = mod.ep_create_group(
        rank, world_size, num_experts, num_local_experts,
        top_k, hidden_dim, backend_int, cuda_graph_max_tokens,
        nccl_comm_ptr,
    )

    return EpGroup(
        _group_id=group_id,
        _backend=backend,
        _config=config,
        _pg=process_group,
    )
