# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Host-side broadcast of local input shard to peer scratch pads, with chunk-level signals."""

import torch

from .configs import Configs


def broadcast_input(
    inp: torch.Tensor,
    out: torch.Tensor,
    out_hdl,
    chunk_size_m: int,
):
    M = inp.shape[0]
    num_chunks = (M + chunk_size_m - 1) // chunk_size_m
    rank = out_hdl.rank
    world_size = out_hdl.world_size
    signal_pad = out_hdl.get_signal_pad(
        rank, (world_size, num_chunks), Configs.SIGNAL_DTYPE, 0
    )
    signal_val = 1

    for shift in range(1, world_size):
        peer = (rank + shift) % world_size
        if Configs.TRANSFER == "CE":
            peer_out = out_hdl.get_remote_tensor(peer, out.shape, out.dtype)[rank]
            peer_signal_pad = out_hdl.get_signal_pad(
                peer, (world_size, num_chunks), Configs.SIGNAL_DTYPE, 0
            )
        m_start = 0
        while m_start < M:
            m_end = min(m_start + chunk_size_m, M)
            if Configs.TRANSFER == "NVSHMEM":
                torch.ops.symm_mem.nvshmem_put_with_signal(
                    inp[m_start:m_end],
                    out[rank][m_start:m_end],
                    signal_pad[rank, m_start // chunk_size_m],
                    signal_val,
                    peer,
                )
            elif Configs.TRANSFER == "CE":
                peer_out[m_start:m_end].copy_(inp[m_start:m_end])
                torch.ops.symm_mem.stream_write_value32_(
                    peer_signal_pad[rank], m_start // chunk_size_m, signal_val
                )
            m_start = m_end
