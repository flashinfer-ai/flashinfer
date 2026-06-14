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

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..utils import is_sm12x_supported
from .jit import gen_build_k2q_csr_module


@functools.cache
def _get_build_k2q_csr_module():
    return gen_build_k2q_csr_module().build_and_load()


# Compiled CuTe-DSL CSR builders, keyed by (topk, has_schedule).
_csr_compile_cache: dict = {}

# Target queries handled sequentially by one warp (one (head, chunk) unit).
_CSR_Q_PER_CHUNK = 64


def _fake_i32(ndim):
    import cutlass
    import cutlass.cute as cute

    return cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        tuple(cute.sym_int() for _ in range(ndim)),
        stride_order=tuple(reversed(range(ndim))),
        assumed_align=4,
    )


def _get_compiled_csr(topk: int, has_schedule: bool):
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.build_k2q_csr_sm12x import BuildK2qCsrSm12x

    key = (topk, has_schedule)
    compiled = _csr_compile_cache.get(key)
    if compiled is not None:
        return compiled

    kernel_obj = BuildK2qCsrSm12x(topk=topk, has_schedule=has_schedule)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel_obj,
        _fake_i32(3),  # q2k
        _fake_i32(2),  # row_map
        _fake_i32(1),  # batch_of_q
        _fake_i32(1),  # qloc_of_q
        _fake_i32(2),  # row_coords
        _fake_i32(2),  # row_ptr
        _fake_i32(2),  # q_idx
        _fake_i32(2),  # qsplit (or dummy)
        _fake_i32(2),  # split_counts (or dummy)
        _fake_i32(2),  # scheduler_metadata (or dummy)
        _fake_i32(1),  # work_count (or dummy)
        _fake_i32(3),  # tile_counts scratch
        _fake_i32(2),  # row_counts scratch
        *(cutlass.Int32(1) for _ in range(8)),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    _csr_compile_cache[key] = compiled
    return compiled


def _csr_geometry(cu_q_dev, cu_k_cpu, total_rows, max_kv_blocks, total_q, device):
    """Host/device precompute of the static row geometry (replaces the CUDA
    ``build_row_map`` kernel): round-robin row map, per-row (batch, kv-block),
    and per-query (batch, batch-local index)."""
    seqlens_k = (cu_k_cpu[1:] - cu_k_cpu[:-1]).to(torch.int64)
    B = seqlens_k.numel()
    rows_per_batch = (seqlens_k + 127) // 128  # blk_kv == 128

    # round-robin order = flatten of active[level, batch] (level outer)
    levels = torch.arange(max_kv_blocks).view(-1, 1)
    active = levels < rows_per_batch.view(1, -1)  # [max_kv, B], order (level, batch)
    order = active.reshape(-1)
    ranks = torch.cumsum(order.to(torch.int32), 0) - 1
    row_map_flat = torch.full((max_kv_blocks * B,), -1, dtype=torch.int32)
    row_map_flat[order] = ranks[order].to(torch.int32)
    # [max_kv, B] -> [B, max_kv]
    row_map = row_map_flat.view(max_kv_blocks, B).t().contiguous().to(device)

    flat_idx = order.nonzero(as_tuple=False).squeeze(1)
    row_coords = (
        torch.stack(
            [(flat_idx % B).to(torch.int32), (flat_idx // B).to(torch.int32)], dim=1
        )
        .contiguous()
        .to(device)
    )

    qi = torch.arange(total_q, device=device, dtype=torch.int32)
    batch_of_q = (torch.searchsorted(cu_q_dev, qi, right=True) - 1).to(torch.int32)
    qloc_of_q = (qi - cu_q_dev[batch_of_q.to(torch.long)]).to(torch.int32)
    return row_map, batch_of_q, qloc_of_q, row_coords


def _run_csr_cudsl(
    q2k_indices,
    cu_q_dev,
    cu_k_cpu,
    total_rows,
    max_kv_blocks,
    topk,
    row_ptr,
    q_indices,
    target_q_per_cta=1,
    qsplit_indices=None,
    split_counts=None,
    scheduler_metadata=None,
    work_count=None,
    work_capacity=0,
):
    """Run the CuTe-DSL CSR builder. ``has_schedule`` is inferred from the
    presence of the schedule outputs."""
    has_schedule = scheduler_metadata is not None
    H, total_q, _ = q2k_indices.shape
    device = q2k_indices.device

    # outputs initialized to the CUDA kernel's defaults. Only the buffers whose
    # *unwritten* entries are actually read need clearing:
    #   - row_ptr: needed for the degenerate early-return below (total_rows==0);
    #     in the normal path the kernel writes every entry.
    #   - q_indices: unused tail slots must read back as -1.
    #   - split_counts: queries with no valid blocks are never written -> 0.
    #   - work_count: the parallel scheduler atomic-accumulates onto it.
    # qsplit_indices and scheduler_metadata tails are never read (the forward
    # consumes only [row_ptr[r], row_ptr[r+1]) / [0, work_count)), matching the
    # CUDA kernel which leaves them uninitialized -> skip those memsets.
    row_ptr.zero_()
    q_indices.fill_(-1)
    if has_schedule:
        split_counts.zero_()
        work_count.zero_()

    if total_q == 0 or total_rows == 0 or H == 0 or max_kv_blocks == 0:
        return

    nchunks = (total_q + _CSR_Q_PER_CHUNK - 1) // _CSR_Q_PER_CHUNK
    q_per_chunk = (total_q + nchunks - 1) // nchunks
    nchunks = (total_q + q_per_chunk - 1) // q_per_chunk

    row_map, batch_of_q, qloc_of_q, row_coords = _csr_geometry(
        cu_q_dev, cu_k_cpu, total_rows, max_kv_blocks, total_q, device
    )
    tile_counts = torch.zeros(
        (H, nchunks, total_rows), dtype=torch.int32, device=device
    )
    row_counts = torch.empty((H, total_rows), dtype=torch.int32, device=device)

    d2 = torch.empty((1, 1), dtype=torch.int32, device=device)
    d1 = torch.empty((1,), dtype=torch.int32, device=device)
    compiled = _get_compiled_csr(topk, has_schedule)
    compiled(
        q2k_indices,
        row_map,
        batch_of_q,
        qloc_of_q,
        row_coords,
        row_ptr,
        q_indices,
        qsplit_indices if has_schedule else d2,
        split_counts if has_schedule else d2,
        scheduler_metadata if has_schedule else d2,
        work_count if has_schedule else d1,
        tile_counts,
        row_counts,
        int(H),
        int(total_q),
        int(total_rows),
        int(max_kv_blocks),
        int(nchunks),
        int(q_per_chunk),
        int(target_q_per_cta),
        int(work_capacity),
    )


@dataclass
class MsaAttentionSchedule:
    """CSR + flat work schedule for the KV-major sparse forward kernel."""

    row_ptr: torch.Tensor  # [Hkv, total_rows + 1] int32
    q_indices: torch.Tensor  # [Hkv, total_q * topk] int32
    qsplit_indices: torch.Tensor  # [Hkv, total_q * topk] int32 (q | slot << 24)
    split_counts: torch.Tensor  # [total_q, Hkv] int32
    scheduler_metadata: torch.Tensor  # [capacity, 6] int32
    work_count: torch.Tensor  # [1] int32 (device)
    work_capacity: int
    total_rows: int
    max_kv_blocks: int
    topk: int


@flashinfer_api
def msa_build_k2q_csr(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    blk_kv: int = 128,
    row_ptr: Optional[torch.Tensor] = None,
    q_indices: Optional[torch.Tensor] = None,
    _backend: str = "cudsl",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert per-query top-K KV block indices into a KV-major CSR structure.

    Given ``q2k_indices`` mapping each (head, query token) to its selected KV
    blocks (the output of :func:`msa_topk_select`, transposed to head-major),
    builds the inverse mapping: for each (head, KV block row), the sorted list
    of query tokens that selected it. This KV-major layout is what the sparse
    attention kernel consumes.

    Rows are packed round-robin across batches: row order is
    (level 0 of every batch that has it, level 1 of every batch, ...), where a
    "level" is the i-th KV block of a sequence. Query indices within each row
    are in ascending (batch-local) order.

    Parameters
    ----------
    q2k_indices : torch.Tensor
        Shape ``(num_qo_heads, total_qo_len, topk)``, dtype int32, contiguous.
        KV-block indices per (head, query token); ``-1`` marks invalid slots.
        ``topk`` must be 4, 8, 16, or 32.
    cu_seqlens_q : torch.Tensor
        Shape ``(batch_size + 1,)``, dtype int32. Cumulative query lengths.
    cu_seqlens_k : torch.Tensor
        Shape ``(batch_size + 1,)``, dtype int32. Cumulative KV lengths.
        Must be on CPU or will be copied to CPU to compute row geometry.
    blk_kv : int
        KV block size. Must be 128.
    row_ptr : torch.Tensor, optional
        Pre-allocated output, shape ``(num_qo_heads, total_rows + 1)``, int32.
    q_indices : torch.Tensor, optional
        Pre-allocated output, shape ``(num_qo_heads, total_qo_len * topk)``,
        int32.

    Returns
    -------
    row_ptr : torch.Tensor
        Shape ``(num_qo_heads, total_rows + 1)``, dtype int32. CSR row
        pointers; row r of head h covers
        ``q_indices[h, row_ptr[h, r]:row_ptr[h, r + 1]]``.
    q_indices : torch.Tensor
        Shape ``(num_qo_heads, total_qo_len * topk)``, dtype int32.
        Batch-local query indices, ascending within each row; unused tail
        slots are ``-1``.
    """
    if not is_sm12x_supported(q2k_indices.device):
        raise RuntimeError(
            "msa_build_k2q_csr requires SM120 or SM121 (Blackwell) and CUDA >= 12.8"
        )

    if q2k_indices.dtype != torch.int32:
        raise ValueError(f"q2k_indices must be int32, got {q2k_indices.dtype}")
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous")
    if q2k_indices.ndim != 3:
        raise ValueError(
            f"q2k_indices must be 3D (num_qo_heads, total_qo_len, topk), "
            f"got {q2k_indices.ndim}D"
        )
    if blk_kv != 128:
        raise ValueError(f"blk_kv must be 128, got {blk_kv}")

    num_qo_heads, total_qo_len, topk = q2k_indices.shape
    if topk not in (4, 8, 16, 32):
        raise ValueError(f"topk must be 4, 8, 16, or 32, got {topk}")

    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be int32")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1D")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")

    # Row geometry (computed on CPU): rows per batch = ceil(seqlen_k / blk_kv)
    cu_k_cpu = cu_seqlens_k.cpu()
    seqlens_k = cu_k_cpu[1:] - cu_k_cpu[:-1]
    rows_per_batch = (seqlens_k + blk_kv - 1) // blk_kv
    total_rows = int(rows_per_batch.sum().item())
    max_kv_blocks = int(rows_per_batch.max().item()) if rows_per_batch.numel() else 0

    device = q2k_indices.device
    if row_ptr is None:
        row_ptr = torch.empty(
            (num_qo_heads, total_rows + 1), dtype=torch.int32, device=device
        )
    else:
        if row_ptr.shape != (num_qo_heads, total_rows + 1):
            raise ValueError(
                f"row_ptr shape must be ({num_qo_heads}, {total_rows + 1}), "
                f"got {tuple(row_ptr.shape)}"
            )
        if row_ptr.dtype != torch.int32:
            raise ValueError(f"row_ptr must be int32, got {row_ptr.dtype}")

    if q_indices is None:
        q_indices = torch.empty(
            (num_qo_heads, total_qo_len * topk), dtype=torch.int32, device=device
        )
    else:
        if q_indices.shape != (num_qo_heads, total_qo_len * topk):
            raise ValueError(
                f"q_indices shape must be ({num_qo_heads}, {total_qo_len * topk}), "
                f"got {tuple(q_indices.shape)}"
            )
        if q_indices.dtype != torch.int32:
            raise ValueError(f"q_indices must be int32, got {q_indices.dtype}")

    cu_q_dev = cu_seqlens_q.to(device, non_blocking=True)
    cu_k_dev = cu_seqlens_k.to(device, non_blocking=True)

    if _backend == "cuda":
        _get_build_k2q_csr_module().build_k2q_csr(
            q2k_indices,
            cu_q_dev,
            cu_k_dev,
            row_ptr,
            q_indices,
            topk,
            blk_kv,
            total_rows,
            max_kv_blocks,
        )
    else:
        _run_csr_cudsl(
            q2k_indices,
            cu_q_dev,
            cu_k_cpu,
            total_rows,
            max_kv_blocks,
            topk,
            row_ptr,
            q_indices,
        )

    return row_ptr, q_indices


@flashinfer_api
def msa_build_k2q_csr_schedule(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    blk_kv: int = 128,
    target_q_per_cta: int = 128,
    max_seqlen_q: int = 0,
    _backend: str = "cudsl",
) -> MsaAttentionSchedule:
    """Build the KV-major CSR plus the flat work schedule for the sparse
    forward kernel.

    In addition to :func:`msa_build_k2q_csr`'s outputs, this produces:

    - ``qsplit_indices``: like ``q_indices`` but each entry packs the
      batch-local query index (low 24 bits) with the query's *split slot*
      (high 8 bits) — the rank of this KV block among the query's valid
      selected blocks. The forward kernel writes its partial output for the
      query at that slot.
    - ``split_counts [total_q, Hkv]``: number of valid selected blocks per
      (query token, kv head); the combine step reduces over slots
      ``[0, count)``.
    - ``scheduler_metadata [capacity, 6]`` / ``work_count [1]``: flat work
      items ``{kv_head, row, q_begin, q_count, batch_idx, kv_block_idx}``;
      rows are chunked into items of at most ``target_q_per_cta`` queries.
    """
    if not is_sm12x_supported(q2k_indices.device):
        raise RuntimeError(
            "msa_build_k2q_csr_schedule requires SM120 or SM121 (Blackwell) and CUDA >= 12.8"
        )
    if q2k_indices.dtype != torch.int32 or q2k_indices.ndim != 3:
        raise ValueError("q2k_indices must be int32 of shape (Hkv, total_q, topk)")
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous")
    if blk_kv != 128:
        raise ValueError(f"blk_kv must be 128, got {blk_kv}")

    num_kv_heads, total_q, topk = q2k_indices.shape
    device = q2k_indices.device

    cu_k_cpu = cu_seqlens_k.cpu()
    seqlens_k = cu_k_cpu[1:] - cu_k_cpu[:-1]
    rows_per_batch = (seqlens_k + blk_kv - 1) // blk_kv
    total_rows = int(rows_per_batch.sum().item())
    max_kv_blocks = int(rows_per_batch.max().item()) if rows_per_batch.numel() else 0

    # Strict upper bound on emitted work items:
    # sum_rows ceil(count / target) <= total_entries / target + nonzero_rows
    work_capacity = num_kv_heads * (
        (total_q * topk + target_q_per_cta - 1) // target_q_per_cta + total_rows
    )

    row_ptr = torch.empty(
        (num_kv_heads, total_rows + 1), dtype=torch.int32, device=device
    )
    q_indices = torch.empty(
        (num_kv_heads, total_q * topk), dtype=torch.int32, device=device
    )
    qsplit_indices = torch.empty_like(q_indices)
    split_counts = torch.empty(
        (total_q, num_kv_heads), dtype=torch.int32, device=device
    )
    scheduler_metadata = torch.empty(
        (work_capacity, 6), dtype=torch.int32, device=device
    )
    work_count = torch.empty((1,), dtype=torch.int32, device=device)

    cu_q_dev = cu_seqlens_q.to(device, non_blocking=True)
    cu_k_dev = cu_seqlens_k.to(device, non_blocking=True)

    if _backend == "cuda":
        _get_build_k2q_csr_module().build_k2q_csr_schedule(
            q2k_indices,
            cu_q_dev,
            cu_k_dev,
            row_ptr,
            q_indices,
            qsplit_indices,
            split_counts,
            scheduler_metadata,
            work_count,
            topk,
            blk_kv,
            total_rows,
            max_kv_blocks,
            target_q_per_cta,
            max_seqlen_q,
        )
    else:
        _run_csr_cudsl(
            q2k_indices,
            cu_q_dev,
            cu_k_cpu,
            total_rows,
            max_kv_blocks,
            topk,
            row_ptr,
            q_indices,
            target_q_per_cta=target_q_per_cta,
            qsplit_indices=qsplit_indices,
            split_counts=split_counts,
            scheduler_metadata=scheduler_metadata,
            work_count=work_count,
            work_capacity=work_capacity,
        )

    return MsaAttentionSchedule(
        row_ptr=row_ptr,
        q_indices=q_indices,
        qsplit_indices=qsplit_indices,
        split_counts=split_counts,
        scheduler_metadata=scheduler_metadata,
        work_count=work_count,
        work_capacity=work_capacity,
        total_rows=total_rows,
        max_kv_blocks=max_kv_blocks,
        topk=topk,
    )
