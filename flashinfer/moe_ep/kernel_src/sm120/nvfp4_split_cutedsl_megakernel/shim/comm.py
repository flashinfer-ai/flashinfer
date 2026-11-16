"""Workspace-owned NCCL memory and peer mappings for SM120 NVFP4."""

from __future__ import annotations

import ctypes as C
import math
import os
from functools import lru_cache

import torch


def _no_dist() -> bool:
    return bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def ensure_not_capturing(operation: str) -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{operation} cannot run during CUDA Graph capture; call warmup() "
            "collectively on every EP rank before capture."
        )


@lru_cache(maxsize=1)
def _host_runtime():
    from moe_sm120_nvfp4_split.transport.gin_runtime import ROOT, validate_artifacts

    library = validate_artifacts()
    host = C.CDLL(str(ROOT / "gin_host.so"))
    host.eq_error.restype = C.c_char_p
    host.eq_set_library.argtypes = [C.c_char_p]
    host.eq_unique_id.argtypes = [C.c_void_p]
    host.eq_create.argtypes = [
        C.c_void_p,
        C.c_int,
        C.c_int,
        C.c_int,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_ulonglong),
        C.POINTER(C.c_int),
    ]
    host.eq_register.argtypes = [C.c_void_p, C.c_ulonglong, C.c_ulonglong]
    host.eq_peer.argtypes = [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_ulonglong)]
    host.eq_destroy.argtypes = [C.c_void_p]
    if host.eq_set_library(str(library).encode()):
        raise RuntimeError("Cannot configure NCCL memory runtime")
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(ROOT / "gin_host.so"), "eq_pool_allocate", "eq_pool_free"
    )
    # Keep the allocator alive: MemPool does not own its Python wrapper.
    return host, allocator


class PeerMemorySession:
    """One symmetric arena per workspace, shared by all layers and buckets.

    The kernel uses one peer-offset table for every symmetric tensor. Therefore
    all these tensors must be slices of the SAME registered allocation. Multiple
    independent allocations cannot share that table, even if their sizes match.
    """

    def __init__(
        self,
        rank: int,
        world: int,
        capacity: int,
        control_group=None,
        *,
        network_peer_mask: int = 0,
    ):
        ensure_not_capturing("NCCL workspace initialization")
        import torch.distributed as dist

        if network_peer_mask < 0 or network_peer_mask >> world:
            raise ValueError("Network peer mask is outside the EP world")
        if network_peer_mask & (1 << rank):
            raise ValueError("The local EP rank cannot be a network peer")
        if network_peer_mask:
            # Optional deployment mapping is indexed by visible CUDA device,
            # never by EP rank (an EP subgroup need not start at device zero).
            hca_map = os.environ.get("MEGAMOE_GIN_HCA_MAP")
            if hca_map:
                hcas = hca_map.split(",")
                device = torch.cuda.current_device()
                if device >= len(hcas) or not hcas[device].strip():
                    raise ValueError("GIN HCA map does not cover the CUDA device")
                os.environ["NCCL_IB_HCA"] = "=" + hcas[device].strip() + ":1"
            # Match the qualified GDAKI runtime; explicit deployment overrides
            # remain authoritative for queue depth and NIC handling.
            for key, value in {
                "NCCL_GIN_ENABLE": "1",
                "NCCL_GIN_TYPE": "3",
                "NCCL_IB_MERGE_NICS": "0",
                "NCCL_GIN_GDAKI_NIC_HANDLER": "2",
                "NCCL_GIN_GDAKI_QP_DEPTH": "1024",
                "NCCL_NVLS_ENABLE": "0",
            }.items():
                os.environ.setdefault(key, value)
        self.host, self.allocator = _host_runtime()
        self.world = world
        self.network_peer_mask = network_peer_mask
        self.network = bool(network_peer_mask)
        self.control_group = control_group
        self.cursor = 0
        self.handle = C.c_void_p()
        self.closed = False
        self.registered_arenas = []
        self.capacity = (capacity + (2 << 20) - 1) // (2 << 20) * (2 << 20)
        signatures = [None] * world
        signature = (self.capacity, self.network)
        dist.all_gather_object(signatures, signature, group=control_group)
        if any(value != signature for value in signatures):
            raise ValueError(
                "NCCL workspace capacity and network mode must match across EP ranks"
            )
        uid = C.create_string_buffer(128)
        if rank == 0:
            self._check(self.host.eq_unique_id(uid))
        source = (
            dist.get_global_rank(control_group, 0) if control_group is not None else 0
        )
        ids = [bytes(uid) if rank == 0 else None]
        dist.broadcast_object_list(ids, src=source, group=control_group)
        C.memmove(uid, ids[0], 128)
        state = C.c_ulonglong()
        info = (C.c_int * 6)()
        self._check(
            self.host.eq_create(
                uid,
                rank,
                world,
                int(self.network),
                C.byref(self.handle),
                C.byref(state),
                info,
            )
        )
        self.state_ptr = state.value
        self.runtime_info = tuple(info)
        try:
            self.pool = torch.cuda.MemPool(self.allocator.allocator())
            with torch.cuda.use_mem_pool(self.pool):
                self.arena = torch.empty(
                    self.capacity, dtype=torch.uint8, device="cuda"
                )
            self._check(
                self.host.eq_register(self.handle, self.arena.data_ptr(), self.capacity)
            )
            offsets = []
            for peer in range(world):
                address = C.c_ulonglong()
                self._check(self.host.eq_peer(self.handle, 0, peer, C.byref(address)))
                if not address.value and not (network_peer_mask & (1 << peer)):
                    raise RuntimeError(f"EP peer {peer} is not P2P reachable")
                # Network peers must be excluded by the compute kernel's
                # transport mask. Zero is a placeholder, never a peer mapping.
                offsets.append(
                    address.value - self.arena.data_ptr() if address.value else 0
                )
            self.offsets = tuple(offsets)
        except BaseException:
            self.close()
            raise

    def _check(self, rc):
        if rc:
            raise RuntimeError(self.host.eq_error().decode())

    def register_network_arena(self, capacity: int) -> torch.Tensor:
        """Register an additional arena for one execution bucket's GIN buffers.

        Its suballocations are network windows, not members of the P2P arena;
        they must never be passed to ``compute_peer_offsets``. All EP ranks
        register execution buckets in the same collective order before capture.
        """
        ensure_not_capturing("GIN network arena registration")
        if self.closed or not self.network:
            raise RuntimeError("GIN registration requires an active network session")
        if capacity <= 0:
            raise ValueError("Network arena capacity must be positive")
        import torch.distributed as dist

        capacity = (capacity + (2 << 20) - 1) // (2 << 20) * (2 << 20)
        signatures = [None] * self.world
        signature = (len(self.registered_arenas), capacity)
        dist.all_gather_object(signatures, signature, group=self.control_group)
        if any(value != signature for value in signatures):
            raise ValueError("GIN window order and capacity must match across EP ranks")
        with torch.cuda.use_mem_pool(self.pool):
            arena = torch.empty(capacity, dtype=torch.uint8, device="cuda")
        self._check(self.host.eq_register(self.handle, arena.data_ptr(), capacity))
        self.registered_arenas.append(arena)
        return arena

    def bind_network_kernel(self, kernel, role: str) -> None:
        """Bind a compiled specialization to this workspace's GIN session."""
        if self.closed or not self.network:
            raise RuntimeError("GIN kernel binding requires an active network session")
        context_bases = {
            "dispatch_plan": 0,
            "dispatch_tx": 8,
            "dispatch_rx": 16,
            "return_tx": 24,
            "return_rx": 32,
        }
        kernel.gin_context_base = context_bases[role]
        kernel.gin_enabled = True
        kernel.gin_state_ptr = self.state_ptr

    def zeros(self, shape, dtype):
        ensure_not_capturing("NCCL workspace suballocation")
        if self.closed:
            raise RuntimeError("NCCL workspace is closed")
        n = math.prod(shape) * dtype.itemsize
        aligned = (n + 4095) // 4096 * 4096
        start = self.cursor
        if start + aligned > self.capacity:
            raise MemoryError("NCCL workspace allocation exceeds its planned capacity")
        self.cursor += aligned
        tensor = self.arena[start : start + n].view(dtype).reshape(shape)
        tensor.zero_()
        tensor._mega_peer_session = self
        return tensor

    def close(self):
        if not self.closed:
            self._check(self.host.eq_destroy(self.handle))
            self.closed = True


def sym_zeros(shape, dtype, *, session=None):
    if _no_dist():
        tensor = torch.zeros(shape, dtype=dtype, device="cuda")
        tensor._mega_plain_alloc = True
        return tensor
    if session is None:
        raise RuntimeError("An explicit NCCL workspace session is required")
    return session.zeros(shape, dtype)


def sym_byte_view(shape, dtype, *, session=None):
    root = sym_zeros((math.prod(shape) * dtype.itemsize,), torch.uint8, session=session)
    return root.view(dtype).reshape(shape), root


def free_sym_tensor(tensor):
    # Suballocations share one arena. Workspace destruction deregisters the
    # arena once; PyTorch storage ownership controls its final deallocation.
    return None


def compute_peer_offsets(symmetric_tensor, world_size):
    local_base = int(symmetric_tensor.data_ptr())
    if _no_dist():
        return local_base, tuple(0 for _ in range(world_size))
    session = getattr(symmetric_tensor, "_mega_peer_session", None)
    if session is None or session.closed or session.world != world_size:
        raise ValueError("Tensor does not belong to the active EP workspace")
    return local_base, session.offsets
