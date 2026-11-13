# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton implementation of push-wait all-gather matmul. For details of the algorithm, see all_gather_matmul.py."""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor
from .broadcast_input import broadcast_input
from .configs import Configs


@triton.jit
def barrier_triton_kernel(
    signal_pads,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Kernel-level barrier: deposit into peer's pad[rank], then withdraw from my pad[peer]."""
    # Deposit: signal_pads[peer][rank] 0 -> 1
    for peer in tl.static_range(world_size):
        deposit_pad = signal_pads[peer]
        while tl.atomic_cas(deposit_pad + rank, 0, 1) != 0:
            pass

    # Withdraw: signal_pads[rank][peer] 1 -> 0
    withdraw_pad = signal_pads[rank]
    for peer in tl.static_range(world_size):
        while tl.atomic_cas(withdraw_pad + peer, 1, 0) != 1:
            pass


def barrier_triton(
    signal_pads: tuple[torch.Tensor],
    rank: int,
) -> None:
    """Triton barrier: same semantics as AllGatherMatmul.barrier (cuTile). Launches one program per peer."""
    world_size = len(signal_pads)
    grid = (1,)
    barrier_triton_kernel[grid](
        signal_pads,
        rank=rank,
        world_size=world_size,
    )


@triton.jit
def swizzle_2d_from_bid(num_bid_m, num_bid_n, GROUP_SIZE_M: tl.constexpr, bid):
    """Map a 1D block ID to 2D tile coordinates with grouped column-first swizzle for L2 cache locality."""
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@triton.jit
def wait_signal_matmul_triton_kernel(
    inp_desc,
    inp_scratch_desc,
    w_desc,
    out_desc,
    signal_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CHUNK_SIZE_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """
    Persistent Triton kernel for matrix multiplication with progress waiting.

    Uses a 1D persistent grid (capped to NUM_SMS) with swizzled 2D tile mapping
    for L2 cache locality. Each program iterates over all chunks and all peers,
    work-stealing across tiles within each chunk.
    """
    num_tiles_m = tl.cdiv(CHUNK_SIZE_M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_chunks = tl.cdiv(M, CHUNK_SIZE_M)
    total_tiles = num_tiles_m * num_tiles_n

    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for shift in tl.static_range(WORLD_SIZE):
        peer = (RANK - shift + WORLD_SIZE) % WORLD_SIZE
        if shift == 0:
            cur_inp_desc = inp_desc
            offset_m_peer_base = 0
        else:
            cur_inp_desc = inp_scratch_desc
            offset_m_peer_base = M * peer

        for chunk_idx in range(num_chunks):
            if shift > 0:
                signal_off = peer * num_chunks + chunk_idx
                while tl.load(signal_ptr + signal_off, volatile=True) == 0:
                    pass

            tile_id = pid
            while tile_id < total_tiles:
                m_tile_idx, n_tile_idx = swizzle_2d_from_bid(
                    num_tiles_m, num_tiles_n, GROUP_SIZE_M, tile_id
                )
                offset_m = (m_tile_idx + num_tiles_m * chunk_idx) * BLOCK_SIZE_M
                offset_n = n_tile_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for k in range(0, num_tiles_k):
                    offset_k = k * BLOCK_SIZE_K
                    a = tl.load_tensor_descriptor(
                        cur_inp_desc, (offset_m_peer_base + offset_m, offset_k)
                    )
                    b = tl.load_tensor_descriptor(w_desc, (offset_k, offset_n))
                    accumulator = tl.dot(a, b, accumulator)

                accumulator = accumulator.to(out_desc.dtype)
                offset_m_out = M * peer + offset_m
                tl.store_tensor_descriptor(
                    out_desc, (offset_m_out, offset_n), accumulator
                )

                tile_id += num_programs


def triton_wait_signal_matmul(
    inp: torch.Tensor,
    inp_scratch: torch.Tensor,
    w: torch.Tensor,
    signal_pad: torch.Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    chunk_size_m: int,
    group_size_m: int,
    rank: int,
    world_size: int,
    *,
    verbose: bool = False,
):
    """Triton implementation of push-wait all-gather matmul. Caller must run
    barrier and broadcast_input (same as cutile_wait_signal_matmul) and provide
    inp_list and signal_pad. Uses same tile/chunk/group layout as cutile path.
    """
    device = inp.device
    M, K = inp.shape[0], inp.shape[1]
    N = w.shape[1]
    assert w.shape[0] == K, "reduction dimension mismatch"

    assert M % tile_m == 0 and N % tile_n == 0 and K % tile_k == 0

    M_out = M * world_size
    out = torch.empty(M_out, N, device=device, dtype=inp.dtype)

    num_tiles_m = (chunk_size_m + tile_m - 1) // tile_m
    num_tiles_n = (N + tile_n - 1) // tile_n
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid_size = min(NUM_SMS, num_tiles_m * num_tiles_n)
    grid = (grid_size,)

    if verbose and rank == 0:
        print(
            f"triton_wait_signal_matmul: tile_m={tile_m}, tile_n={tile_n}, tile_k={tile_k}, "
            f"chunk_size_m={chunk_size_m}, group_size_m={group_size_m}, grid={grid}, NUM_SMS={NUM_SMS}"
        )

    inp_desc = TensorDescriptor.from_tensor(inp, (tile_m, tile_k))
    inp_scratch_desc = TensorDescriptor.from_tensor(inp_scratch, (tile_m, tile_k))
    w_desc = TensorDescriptor.from_tensor(w, (tile_k, tile_n))
    out_desc = TensorDescriptor.from_tensor(out, (tile_m, tile_n))

    wait_signal_matmul_triton_kernel[grid](
        inp_desc=inp_desc,
        inp_scratch_desc=inp_scratch_desc,
        w_desc=w_desc,
        out_desc=out_desc,
        signal_ptr=signal_pad,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=tile_m,
        BLOCK_SIZE_N=tile_n,
        BLOCK_SIZE_K=tile_k,
        CHUNK_SIZE_M=chunk_size_m,
        GROUP_SIZE_M=group_size_m,
        RANK=rank,
        WORLD_SIZE=world_size,
    )
    return out


def all_gather_matmul_triton(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
):
    """Drop-in Triton equivalent of cutile_wait_signal_matmul: barrier, broadcast,
    wait_signal_matmul kernel, then cleanup. Requires NVSHMEM backend.
    """
    Configs.initialize()
    device = inp.device
    if symm_mem.get_backend(device) != "NVSHMEM":
        raise ValueError(
            "NVSHMEM backend is required for push-wait based all-gather matmul"
        )

    M, K = inp.shape[0], inp.shape[1]
    N = w.shape[1]
    assert w.shape[0] == K, "reduction dimension mismatch"

    handle = symm_mem.rendezvous(inp, group.group_name)
    world_size = handle.world_size
    rank = handle.rank

    GROUP_SIZE_M = 8
    tile_m, tile_n, tile_k = 128, 128, 128

    assert M % tile_m == 0 and N % tile_n == 0 and K % tile_k == 0

    # Larger chunks seems to be better for performance
    chunk_size_m = M
    num_chunks = (M + chunk_size_m - 1) // chunk_size_m

    inp_scratch = symm_mem.empty(world_size, M, K, device=device, dtype=inp.dtype)
    scratch_handle = symm_mem.rendezvous(inp_scratch, group.group_name)
    signal_pad = scratch_handle.get_signal_pad(
        rank, (world_size, num_chunks), Configs.SIGNAL_DTYPE, 0
    )

    main_stream = torch.cuda.current_stream()
    barrier_sigpads = tuple(
        handle.get_signal_pad(r, (world_size,), Configs.SIGNAL_DTYPE, 0)
        for r in range(world_size)
    )
    barrier_triton(barrier_sigpads, rank)

    comm_stream = torch.cuda.Stream()
    comm_stream.wait_stream(main_stream)
    with torch.cuda.stream(comm_stream):
        broadcast_input(inp, inp_scratch, scratch_handle, chunk_size_m)

    inp_scratch_2d = inp_scratch.reshape(world_size * M, K)
    out = triton_wait_signal_matmul(
        inp,
        inp_scratch_2d,
        w,
        signal_pad,
        tile_m,
        tile_n,
        tile_k,
        chunk_size_m,
        GROUP_SIZE_M,
        rank,
        world_size,
        verbose=verbose,
    )

    signal_pad.zero_()
    main_stream.wait_stream(comm_stream)
    return out
