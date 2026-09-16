"""MUSA logical-chunk boundaries for the SSD variable-length scheduler.

Like the CUDA implementation, the input logical chunks must be ordered by
sequence id. Each output is a lower bound in that ordered list. Independent
searches avoid a global histogram/scan workspace and work during graph capture.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _seq_chunk_lower_bounds(
    seq_idx, chunk_indices, chunk_offsets, output,
    seq_stride: tl.constexpr, indices_stride: tl.constexpr,
    offsets_stride: tl.constexpr, chunk_size: tl.constexpr,
    num_chunks: tl.constexpr, num_seqs: tl.constexpr,
    search_steps: tl.constexpr, BLOCK: tl.constexpr,
):
    sequence = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = sequence < num_seqs
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), num_chunks, tl.int32)
    for _ in range(search_steps):
        mid = (lo + hi) // 2
        searching = valid & (lo < hi)
        physical = tl.load(chunk_indices + mid * indices_stride, searching, 0).to(tl.int64)
        offset = tl.load(chunk_offsets + mid * offsets_stride, searching, 0).to(tl.int64)
        seq = tl.load(seq_idx + (physical * chunk_size + offset) * seq_stride, searching, 0)
        move_right = searching & (seq < sequence)
        hi = tl.where(searching & ~move_right, mid, hi)
        lo = tl.where(move_right, mid + 1, lo)
    tl.store(output + sequence, lo, valid)
    tl.store(output + sequence, num_chunks, sequence == num_seqs)


def seq_chunk_cumsum(
    seq_idx: torch.Tensor, chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor, chunk_size: int, num_seqs: int,
    out: torch.Tensor | None = None, tile_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exclusive chunk counts with the CUDA metadata/stride contract.

    ``tile_state`` is accepted for call-site compatibility but unused: this
    implementation has no inter-block scan or auxiliary workspace.
    """
    if seq_idx.device.type != "musa":
        raise ValueError("MUSA sequence chunk kernel requires MUSA tensors")
    if seq_idx.ndim != 2 or seq_idx.shape[0] != 1:
        raise ValueError("seq_idx must have shape [1, total_seqlen]")
    if seq_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_idx must be int32 or int64")
    if chunk_size <= 0 or num_seqs < 0:
        raise ValueError("chunk_size must be positive and num_seqs nonnegative")
    for tensor in (chunk_indices, chunk_offsets):
        if tensor.ndim != 1 or tensor.dtype != torch.int32 or tensor.device != seq_idx.device:
            raise ValueError("chunk metadata must be 1D int32 on seq_idx.device")
    if chunk_indices.shape != chunk_offsets.shape:
        raise ValueError("chunk_indices and chunk_offsets must have equal length")
    if tile_state is not None and (tile_state.device != seq_idx.device or tile_state.dtype != torch.uint8):
        raise ValueError("tile_state must be uint8 on seq_idx.device")
    if out is None:
        out = torch.empty(num_seqs + 1, dtype=torch.int32, device=seq_idx.device)
    if out.shape != (num_seqs + 1,) or out.dtype != torch.int32 or out.device != seq_idx.device or not out.is_contiguous():
        raise ValueError("out must be contiguous int32 [num_seqs + 1] on seq_idx.device")
    chunks = chunk_indices.numel()
    with torch.musa.device(seq_idx.device):
        _seq_chunk_lower_bounds[(triton.cdiv(num_seqs + 1, 128),)](
            seq_idx, chunk_indices, chunk_offsets, out,
            seq_idx.stride(1), chunk_indices.stride(0), chunk_offsets.stride(0),
            chunk_size, chunks, num_seqs, chunks.bit_length(), 128,
            num_warps=4,
        )
    return out
