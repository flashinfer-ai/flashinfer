"""Single-allocation symmetric workspace for the dispatch kernel.

Mirrors mega_moe's pattern (``DeepGEMM/csrc/apis/mega.hpp`` chained
``Buffer + slice_input_buffers``): one ``nvshmem.core.tensor((N,), uint8)``
per rank, every named tensor is a torch view into that byte buffer. Because
all ranks perform the same single collective allocation in the same order,
the peer-pointer delta is one constant per peer that works for every
sub-region (mega_moe ``SymBuffer.offsets`` / ``sym_buffer.cuh:34-37``).

The kernel side uses ``sym_buffer.SymBuffer.map`` to translate any local
pointer to its peer-rank counterpart -- no per-tensor peer-view lists
needed.  ``Workspace.symmetric_base`` + ``peer_offsets_list`` carry the
host-side payload used to build a ``SymBufferHost`` runtime argument.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist

import nvshmem.core

try:
    # cuda-core < 1.0 (APIs were still under the experimental namespace).
    from cuda.core.experimental import Device
except ImportError:
    # cuda-core >= 1.0 promoted everything to the top-level namespace.
    from cuda.core import Device

from .config import DSV4, DSV4Config, TOKEN_METADATA_BYTES


# ----------------------------------------------------------------------
# Region spec
# ----------------------------------------------------------------------

# nvshmem4py 0.3.0 doesn't support uint32/uint64 dtypes; we substitute signed
# int32/int64 storage. Kernel atomics use raw PTX so bit-pattern semantics
# are preserved end-to-end; host-side inspection wraps with ``.view(np.uint*)``.
_DTYPE_BYTES = {
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.float32: 4,
    torch.float64: 8,
    torch.bfloat16: 2,
    torch.float16: 2,
}


@dataclass(frozen=True)
class _Region:
    """One sub-region of the symmetric byte buffer.

    ``align`` is in bytes; the region's offset is rounded up to ``align``
    before placement. Default 16B matches the TMA descriptor minimum;
    large FP8 token rows use 128B for ``cp.async.bulk`` efficiency.
    """

    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    align: int = 16


@dataclass(frozen=True)
class _Shapes:
    """Pool shapes derived from (world_size, num_tokens_per_rank).

    Mirrors ``layout/mega_moe.cuh:get_num_max_pool_tokens`` /
    ``get_num_padded_sf_pool_tokens``. Stored on ``Workspace`` so callers
    don't recompute when ``num_tokens_per_rank`` is overridden per-launch.
    """

    num_tokens_per_rank: int
    num_max_pool_tokens: int
    num_max_pool_blocks: int
    num_max_task_tiles: int
    num_padded_sf_pool_tokens: int


def _derive_shapes(
    world_size: int,
    num_tokens_per_rank: int,
    config: DSV4Config = DSV4,
) -> _Shapes:
    K = config.num_topk
    BM = config.block_m
    SFBM = config.sf_block_m
    CTM = config.effective_cluster_tile_m
    E_local = config.num_experts_per_rank

    num_max_recv = world_size * num_tokens_per_rank
    num_max_per_tok = min(K, E_local)
    raw = num_max_recv * num_max_per_tok + E_local * (BM - 1)
    P_TOK = ((raw + BM - 1) // BM) * BM
    P_BLK = P_TOK // BM
    # Release-counter slot count: sized at ``cluster_tile_m`` granularity
    # rather than ``block_m`` (see ``fc12_integrate_comm.md`` §4 C3).
    P_TT = (P_TOK + CTM - 1) // CTM
    P_SF = P_BLK * SFBM
    return _Shapes(
        num_tokens_per_rank=num_tokens_per_rank,
        num_max_pool_tokens=P_TOK,
        num_max_pool_blocks=P_BLK,
        num_max_task_tiles=P_TT,
        num_padded_sf_pool_tokens=P_SF,
    )


def _build_spec(
    world_size: int,
    shapes: _Shapes,
    config: DSV4Config = DSV4,
) -> Tuple[_Region, ...]:
    """Region table for one DSV4 dispatch workspace at the given shapes.

    ``config`` selects the wire format: DSV4 (FP8, hidden_bytes=7168) or
    DSV4_nvfp4 (NVFP4, hidden_bytes=3584). Token buffers (uint8) and the L1
    pool are sized by ``config.hidden_bytes``; SF buffers use
    ``config.sf_uint32_per_token``.
    """
    R = world_size
    K = config.num_topk
    H_BYTES = config.hidden_bytes
    SFU = config.sf_uint32_per_token
    E_local = config.num_experts_per_rank
    E_total = config.num_total_experts
    T = shapes.num_tokens_per_rank
    P_TOK = shapes.num_max_pool_tokens
    P_TT = shapes.num_max_task_tiles
    P_SF = shapes.num_padded_sf_pool_tokens
    MAX_SLOT = T * K

    return (
        # Input region (per-rank source side).
        _Region("input_token_buffer", torch.uint8, (T, H_BYTES), align=128),
        _Region("input_sf_buffer", torch.int32, (T, SFU), align=16),
        _Region("input_topk_idx_buffer", torch.int64, (T, K), align=16),
        _Region("input_topk_weights_buffer", torch.float32, (T, K), align=16),
        # Counters (atomicAdd targets).
        _Region("expert_send_count", torch.int64, (E_total,), align=16),
        _Region("expert_recv_count", torch.int64, (R, E_local), align=16),
        _Region("expert_recv_count_sum", torch.int64, (E_local,), align=16),
        # Routing workspace.
        _Region("src_token_topk_idx", torch.int32, (E_local, R, MAX_SLOT), align=16),
        _Region(
            "token_src_metadata", torch.uint8, (P_TOK, TOKEN_METADATA_BYTES), align=16
        ),
        # L1 receive-side pool.  Note: ``l1_arrival_count`` is sized at
        # ``cluster_tile_m`` granularity (num_max_task_tiles), not at
        # ``block_m`` granularity -- see fc12_integrate_comm.md §4 C3.
        _Region("l1_arrival_count", torch.int32, (P_TT,), align=16),
        _Region("l1_token_buffer", torch.uint8, (P_TOK, H_BYTES), align=128),
        _Region("l1_sf_buffer", torch.int32, (SFU, P_SF), align=16),
        _Region("l1_topk_weights_buffer", torch.float32, (P_TOK,), align=16),
        # Cross-rank barrier signal (slot 0 = pre-pull, slot 1 = kernel-tail).
        _Region("nvlink_barrier_signal", torch.int32, (R, 2), align=16),
        # Local-only counter for software_grid_sync (carried in symmetric heap
        # so layout is uniform; intra-rank, no peer reads).
        _Region("grid_sync_counter", torch.int32, (2,), align=16),
    )


def _layout(spec: Tuple[_Region, ...]) -> Tuple[Dict[str, int], int]:
    """Place regions sequentially with per-region alignment; return (offsets, total).

    Final total is rounded up to 16B so the next allocation in a future
    fused workspace stays TMA-friendly.
    """
    offsets: Dict[str, int] = {}
    cursor = 0
    for region in spec:
        cursor = ((cursor + region.align - 1) // region.align) * region.align
        # Sub-region offset must satisfy dtype alignment for ``.view(dtype)``.
        assert cursor % _DTYPE_BYTES[region.dtype] == 0, (
            f"Region {region.name}: offset {cursor} not aligned for dtype {region.dtype}"
        )
        offsets[region.name] = cursor
        cursor += math.prod(region.shape) * _DTYPE_BYTES[region.dtype]
    total = ((cursor + 15) // 16) * 16
    return offsets, total


# ----------------------------------------------------------------------
# Workspace
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Workspace:
    """Symmetric-memory workspace for one rank's dispatch kernel run.

    Every named tensor is a typed torch view into ``_byte_buf``, the single
    symmetric byte allocation. ``peer_offsets_list[r] = peer_r_base -
    local_base`` (bytes) and is the only cross-rank pointer table the
    kernel needs --
    the same delta works for every sub-region (mega_moe ``SymBuffer.map``
    semantics, ``sym_buffer.cuh:34-37``).
    """

    # Input region.
    input_token_buffer: torch.Tensor
    input_sf_buffer: torch.Tensor
    input_topk_idx_buffer: torch.Tensor
    input_topk_weights_buffer: torch.Tensor
    # Counters.
    expert_send_count: torch.Tensor
    expert_recv_count: torch.Tensor
    expert_recv_count_sum: torch.Tensor
    # Routing workspace.
    src_token_topk_idx: torch.Tensor
    token_src_metadata: torch.Tensor
    # L1 receive pool.
    l1_arrival_count: torch.Tensor
    l1_token_buffer: torch.Tensor
    l1_sf_buffer: torch.Tensor
    l1_topk_weights_buffer: torch.Tensor
    # Cross-rank barrier.
    nvlink_barrier_signal: torch.Tensor
    grid_sync_counter: torch.Tensor
    # Peer-pointer delta data (HOST-side Python ints; the generated host
    # wrapper packs them into a SymBuffer{world_size} kernel param. No
    # GMEM allocation -- offsets live in the kernel param bank via CUDA
    # byval ABI). ``peer_offsets_list[r] = peer_r_base - local_base`` in
    # bytes; ``peer_offsets_list[local_rank] == 0``.
    symmetric_base: int
    peer_offsets_list: tuple  # tuple[int, ...] length = world_size
    local_rank: int
    # Derived pool shapes (so callers don't recompute when num_tokens_per_rank
    # is overridden per-launch).
    num_tokens_per_rank: int
    num_max_pool_tokens: int
    num_max_pool_blocks: int
    num_max_task_tiles: int
    num_padded_sf_pool_tokens: int
    # Keep-alive root buffer (do NOT drop -- views are non-owning).
    _byte_buf: torch.Tensor

    def release(self) -> None:
        """Free the symmetric byte buffer back to the NVSHMEM heap.

        Must be called collectively on every rank BEFORE
        ``nvshmem.core.finalize()`` (otherwise NVSHMEM logs a
        "Symmetric memory was not freed explicitly" error and the leak
        can surface as a hang in the next allocation cycle).
        """
        # NVSHMEM tracks the allocation by `data_ptr()` of the root byte
        # buffer. Drop our reference first so torch's caching allocator
        # doesn't hold a phantom view while we call free_tensor.
        try:  # noqa: SIM105
            nvshmem.core.free_tensor(self._byte_buf)
        except Exception:  # noqa: BLE001
            # Best-effort: if NVSHMEM has already torn down (e.g. the
            # process is exiting mid-fault), don't shadow the real error.
            pass


def alloc_workspace(
    rank: int,
    world_size: int = DSV4.num_ranks,
    num_tokens_per_rank: int = DSV4.num_tokens_per_rank,
    *,
    config: DSV4Config = DSV4,
    verbose: bool = False,
) -> Workspace:
    """Allocate ONE symmetric byte buffer and slice into typed views.

    All ranks call this collectively with identical arguments; nvshmem4py
    keeps the heap layout symmetric across ranks, so peer-pointer deltas
    are a single constant per peer.

    ``config`` selects the wire format (DSV4=FP8 default, DSV4_nvfp4=NVFP4).
    The byte buffer is zeroed (NVSHMEM allocation is not guaranteed
    initialised), then ``token_src_metadata`` is filled with the 0xFF
    pool-slot-empty sentinel that both kernel and host-check rely on.
    """
    assert world_size > 0, f"world_size must be positive, got {world_size}"

    shapes = _derive_shapes(world_size, num_tokens_per_rank, config)
    spec = _build_spec(world_size, shapes, config)
    offsets, total_bytes = _layout(spec)

    if verbose and rank == 0:
        print(
            f"[bootstrap] alloc_workspace: world_size={world_size} "
            f"num_tokens_per_rank={num_tokens_per_rank} "
            f"total_bytes={total_bytes:,} ({total_bytes / (1 << 20):.1f} MiB)"
        )
        for region in spec:
            nbytes = math.prod(region.shape) * _DTYPE_BYTES[region.dtype]
            print(
                f"[bootstrap]   {region.name:30s} off={offsets[region.name]:>11,}  "
                f"bytes={nbytes:>12,}  shape={tuple(region.shape)}  dtype={region.dtype}"
            )

    byte_buf = nvshmem.core.tensor((total_bytes,), dtype=torch.uint8)
    byte_buf.zero_()

    views: Dict[str, torch.Tensor] = {}
    for region in spec:
        off = offsets[region.name]
        nbytes = math.prod(region.shape) * _DTYPE_BYTES[region.dtype]
        sub = byte_buf.narrow(0, off, nbytes)
        v = sub.view(region.dtype).reshape(region.shape)
        # Layout invariant: every typed view must point exactly at its
        # offset within byte_buf. If this fails, peer_offsets won't work
        # for that region and cross-rank ops will land at wrong addresses.
        assert v.data_ptr() == byte_buf.data_ptr() + off, (
            f"Region {region.name}: data_ptr {v.data_ptr():#x} != "
            f"byte_buf {byte_buf.data_ptr():#x} + off {off}"
        )
        views[region.name] = v

    # Pool-slot-empty sentinel: kernel writes 0xFF only on padding slots,
    # host-check uses (token_src_metadata != 0xFF).any() to find populated
    # entries. mega_moe init_padding_kernel does the same.
    views["token_src_metadata"].fill_(0xFF)

    # Build peer offsets table as plain Python ints.
    # ``nvshmem.core.get_peer_tensor()`` tracks allocations by exact
    # ``data_ptr()``; it must be called on the root byte buffer, since
    # subview lookups would not match any tracked allocation.
    # No GMEM tensor: callers pass these ints via SymBufferHost, and the
    # generated host wrapper packs them into SymBuffer{world_size} before
    # launching the device kernel (CUDA byval ABI -> param bank).
    local_addr = int(byte_buf.data_ptr())
    peer_offsets_list = tuple(
        int(nvshmem.core.get_peer_tensor(byte_buf, r).data_ptr()) - local_addr
        for r in range(world_size)
    )
    assert peer_offsets_list[rank] == 0, (
        f"peer_offsets_list[rank={rank}] expected 0, got {peer_offsets_list[rank]}"
    )

    return Workspace(
        **views,
        symmetric_base=local_addr,
        peer_offsets_list=peer_offsets_list,
        local_rank=rank,
        num_tokens_per_rank=shapes.num_tokens_per_rank,
        num_max_pool_tokens=shapes.num_max_pool_tokens,
        num_max_pool_blocks=shapes.num_max_pool_blocks,
        num_max_task_tiles=shapes.num_max_task_tiles,
        num_padded_sf_pool_tokens=shapes.num_padded_sf_pool_tokens,
        _byte_buf=byte_buf,
    )


# ----------------------------------------------------------------------
# Reusable bootstrap: torch.distributed + NVSHMEM init/finalize.
#
# Pattern follows dlarch-fastkernels CuTeDSL distributed/dispatch_and_combine
# (`torchrun_uid_init_bcast`). One init function for every launcher
# (torchrun / srun / mp.spawn) so test scripts don't each re-roll their own.
# ----------------------------------------------------------------------


def _discover_ranks() -> tuple[int, int, int, str | None]:
    """Detect launcher type and return (local_rank, global_rank, world_size,
    init_method). ``init_method`` is None for torchrun (env:// default) or
    a ``tcp://master:port`` URL for srun.

    Recognises:
      - torchrun / torch.distributed.run / pytest+torchrun:
            RANK + LOCAL_RANK + WORLD_SIZE
      - srun / sbatch:
            SLURM_PROCID + SLURM_LOCALID + SLURM_NTASKS + MASTER_ADDR
      - mp.spawn (single-node) with manually-set env:
            RANK + LOCAL_RANK + WORLD_SIZE (same as torchrun; caller is
            responsible for setting MASTER_ADDR + MASTER_PORT)
    """
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        return (
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["RANK"]),
            int(os.environ.get("WORLD_SIZE", "1")),
            None,
        )
    if "SLURM_PROCID" in os.environ:
        global_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        # Mirror SLURM_NTASKS_PER_NODE -> LOCAL_WORLD_SIZE so downstream
        # code that wants a "node_id" computation has it without caring
        # which launcher it ran under.
        os.environ.setdefault(
            "LOCAL_WORLD_SIZE",
            os.environ.get("SLURM_NTASKS_PER_NODE", str(world_size)),
        )
        return local_rank, global_rank, world_size, f"tcp://{master_addr}:{master_port}"
    raise RuntimeError(
        "Cannot bootstrap: neither RANK/LOCAL_RANK (torchrun) nor "
        "SLURM_PROCID (srun) are set in the environment."
    )


def init_dist_and_nvshmem() -> tuple[int, int, int, Device]:
    """Bootstrap torch.distributed (gloo+nccl) and NVSHMEM via UID broadcast.

    Returns ``(local_rank, rank, world_size, dev)``. The
    ``cuda.core.experimental.Device`` handle is the same one NVSHMEM
    was initialised with, so callers can reuse it for ``cute.compile``.

    Re-entrancy: if torch.distributed is already initialised (e.g. the
    caller staged it themselves before calling here) the existing group
    is reused; NVSHMEM is initialised exactly once.
    """
    local_rank, rank, world_size, init_method = _discover_ranks()

    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()

    if not dist.is_initialized():
        if init_method is None:
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        else:
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl",
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )

    # Only global rank 0 generates the real UID; everyone else allocates
    # an empty placeholder that the broadcast overwrites. Using local_rank
    # here would have one rank-0 per node, all colliding in NVSHMEM init.
    uid = nvshmem.core.get_unique_id(empty=(rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

    nvshmem.core.init(
        device=dev,
        uid=uid,
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )
    return local_rank, rank, world_size, dev


def finalize_dist_and_nvshmem(workspace: Workspace | None = None) -> None:
    """Tear down NVSHMEM and torch.distributed.

    Pass ``workspace`` (if any) so its symmetric byte buffer is freed
    before ``nvshmem.core.finalize()``. Without this, NVSHMEM logs
    "Symmetric memory was not freed explicitly" on every rank.
    """
    if workspace is not None:
        workspace.release()
    try:  # noqa: SIM105
        nvshmem.core.finalize()
    except Exception:  # noqa: BLE001
        pass
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
