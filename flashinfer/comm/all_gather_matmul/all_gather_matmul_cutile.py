# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
cuTile implementation of push-wait all-gather matmul. Requires SM >= 100 (Blackwell+).

See all_gather_matmul.py for the public routing entry point and full algorithm description.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cuda.tile as ct

from .broadcast_input import broadcast_input
from .configs import Configs


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Map a 1D block ID to 2D tile coordinates with grouped column-first swizzle
    # for better L2 cache locality.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


# Each bid deposits to a peer; then waits for a peer's deposit.
# This is a kernel-level, not a bid-level barrier.
@ct.kernel
def barrier(
    signal_pad_list,
    rank: ct.Constant[int],
):
    peer = ct.bid(0)
    # Deposit
    peer_signal_pad = signal_pad_list[peer]
    while ct.atomic_cas(peer_signal_pad, (rank,), 0, 1) == 1:
        pass
    # Withdraw
    my_signal_pad = signal_pad_list[rank]
    while ct.atomic_cas(my_signal_pad, (peer,), 1, 0) == 0:
        pass


# cuTile kernel for waiting for per-chunk ready signal then matmul
@ct.kernel
def wait_signal_matmul_kernel(
    inp_list,
    w,
    out,
    signal_pad,
    rank: ct.Constant[int],
    world_size: ct.Constant[int],
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    tile_k: ct.Constant[int],
    chunk_size_m: ct.Constant[int],
    num_chunks: ct.Constant[int],
    group_size_m: ct.Constant[int],
):
    # Number of k tiles along reduction dimension
    num_tiles_k = ct.num_tiles(w, axis=0, shape=(tile_k, tile_n))
    num_tiles_m = ct.cdiv(chunk_size_m, tile_m)

    bid = ct.bid(0)
    M = inp_list[0].shape[0]
    N = w.shape[1]
    num_tiles_m_per_peer = ct.cdiv(M, tile_m)

    # Loop over input from all peers, starting from self because it is ready
    for shift in range(0, world_size):
        peer = (rank - shift + world_size) % world_size
        # Select ct.Array from inp_list
        peer_inp = inp_list[peer]
        peer_offset_out = num_tiles_m_per_peer * peer

        zero_pad = ct.PaddingMode.ZERO
        # Convert fp32 to tf32 to use tensorcore
        dtype = ct.tfloat32 if peer_inp.dtype == ct.float32 else peer_inp.dtype

        for chunk_idx in range(num_chunks):
            # Initialize accumulator
            accumulator = ct.full((tile_m, tile_n), 0, dtype=ct.float32)

            # Swizzle 1D block ID to 2D tile coordinates for better L2 cache locality
            real_chunk_size_m = min(M - chunk_idx * chunk_size_m, chunk_size_m)
            m_tile_idx, n_tile_idx = swizzle_2d_from_bid(
                real_chunk_size_m, N, tile_m, tile_n, group_size_m, bid
            )
            if m_tile_idx * tile_m >= real_chunk_size_m:
                break

            # Wait for input ready signal
            if shift > 0:
                signal_index = (peer, chunk_idx)
                signal = ct.load(
                    signal_pad, index=signal_index, shape=(), padding_mode=zero_pad
                )
                while signal == 0:
                    signal = ct.load(
                        signal_pad, index=signal_index, shape=(), padding_mode=zero_pad
                    )

            # Index of this tile in the peer's input
            m_tile_idx_in_peer = m_tile_idx + num_tiles_m * chunk_idx

            for k in range(num_tiles_k):
                # Load remote input tile
                a = ct.load(
                    peer_inp,
                    index=(m_tile_idx_in_peer, k),
                    shape=(tile_m, tile_k),
                    padding_mode=zero_pad,
                ).astype(dtype)
                # Load weight tile
                b = ct.load(
                    w,
                    index=(k, n_tile_idx),
                    shape=(tile_k, tile_n),
                    padding_mode=zero_pad,
                ).astype(dtype)
                # Perform matrix multiplication
                accumulator = ct.mma(a, b, accumulator)

            # Cast result back to output dtype
            accumulator = ct.astype(accumulator, out.dtype)

            # Store result tile
            global_m_tile_idx = m_tile_idx_in_peer + peer_offset_out
            ct.store(out, index=(global_m_tile_idx, n_tile_idx), tile=accumulator)


# Launcher for push-wait based all-gather matmul (cuTile / SM >= 100)
def all_gather_matmul_cutile(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
):
    Configs.initialize()
    device = inp.device
    if symm_mem.get_backend(device) != "NVSHMEM":
        raise ValueError(
            "NVSHMEM backend is required for push-wait based all-gather matmul"
        )

    M = inp.shape[0]
    K = inp.shape[1]
    N = w.shape[1]
    assert w.shape[0] == K, "reduction dimension mismatch"

    # Tune tile sizes and group size
    if inp.dtype.itemsize == 2:  # Likely torch.float16 or torch.bfloat16
        tile_m, tile_n, tile_k = (
            128,
            256,
            64,
        )  # Larger tiles for Tensor Core friendly types
        GROUP_SIZE_M = 19  # GB200 number of SMs = 152
    else:  # Likely torch.float32 or other
        tile_m, tile_n, tile_k = 128, 128, 128  # Smaller, more general tiles
        GROUP_SIZE_M = 8

    assert M % tile_m == 0
    assert N % tile_n == 0
    assert K % tile_k == 0

    handle = symm_mem.rendezvous(inp, group.group_name)
    world_size = handle.world_size
    rank = handle.rank

    # Create output tensor
    M_out = M * world_size
    out = torch.empty(M_out, N, device=device, dtype=inp.dtype)

    # Create input scratch pad to receive input shards from peers
    # Starting from torch 2.11, an implicit memory pool is used for ``symm_mem.empty``
    inp_scratch = symm_mem.empty(world_size, M, K, device=device, dtype=inp.dtype)
    scratch_handle = symm_mem.rendezvous(inp_scratch, group.group_name)
    # Feeding input shards to the kernel, self input is in inp, other inputs are in inp_scratch
    inp_list = [
        inp if peer == rank else inp_scratch[peer] for peer in range(world_size)
    ]

    # Signal pad arrangement is related to the number of chunks
    # chunk_size_m must cover at least GROUP_SIZE_M tiles for the swizzle to be effective
    chunk_size_m = min(M, GROUP_SIZE_M * tile_m)
    num_chunks = ct.cdiv(M, chunk_size_m)
    signal_pad = scratch_handle.get_signal_pad(
        rank, (world_size, num_chunks), Configs.SIGNAL_DTYPE, 0
    )

    # Map each output tile to a block (1D grid with swizzled mapping)
    num_tiles_m = ct.cdiv(chunk_size_m, tile_m)
    num_tiles_n = ct.cdiv(N, tile_n)
    grid_size = num_tiles_m * num_tiles_n
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # grid_size = min(NUM_SMS, grid_size)
    grid = (grid_size,)
    if verbose and rank == 0:
        print(
            f"tile_m: {tile_m}, tile_n: {tile_n}, tile_k: {tile_k}, GROUP_SIZE_M: {GROUP_SIZE_M}"
        )
        print(
            f"num_tiles_m: {num_tiles_m}, num_tiles_n: {num_tiles_n}, launching kernel with grid: {grid}, NUM_SMS: {NUM_SMS}"
        )

    # Barrier to ensure all ranks have initialized their inputs, cleared signal pads, etc.
    main_stream = torch.cuda.current_stream()
    barrier_grid = (world_size,)
    barrier_sigpads = [
        handle.get_signal_pad(r, (world_size,), Configs.SIGNAL_DTYPE, 0)
        for r in range(world_size)
    ]
    ct.launch(
        main_stream,
        barrier_grid,
        barrier,
        (barrier_sigpads, rank),
    )

    comm_stream = torch.cuda.Stream()
    comm_stream.wait_stream(main_stream)

    # Broadcast input to peers
    with torch.cuda.stream(comm_stream):
        broadcast_input(inp, inp_scratch, scratch_handle, chunk_size_m)

    # Launch GEMM kernel
    ct.launch(
        main_stream,
        grid,
        wait_signal_matmul_kernel,
        (
            inp_list,
            w,
            out,
            signal_pad,
            rank,
            world_size,
            tile_m,
            tile_n,
            tile_k,
            chunk_size_m,
            num_chunks,
            GROUP_SIZE_M,
        ),
    )

    # Clean signal pad
    signal_pad.zero_()
    main_stream.wait_stream(comm_stream)

    return out
