"""Validation and canonical-to-derived metadata (proposal §3.2).

Mirrors ``flashinfer/mla/_batch_mla/_planning.py``: everything here is
backend-neutral. Structural checks are host-only; value checks read the host
mirrors the contract guarantees; derivation is pure device ops with no sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ._contracts import _expect, _expect_page_size


@dataclass
class Derived:
    q_seq_lens: torch.Tensor  # (b,)  device — diff(qo_indptr)
    cum_kv_seq_lens: torch.Tensor  # (b+1,) device — for trtllm / cuDNN cu_seq_len_kv
    kv_page_indptr: torch.Tensor  # (b+1,) device CSR page-unit indptr
    kv_page_indices: torch.Tensor  # flat page ids (CSR-compacted prefix up to
    # kv_page_indptr[-1]; any tail is untouched scratch never read by kernels,
    # which bound reads by the indptr)
    block_tables: Optional[torch.Tensor]  # (b, width) dense — given or derived;
    # None only when no candidate backend needs the dense form


def validate_structure(
    device: torch.device,
    qo_indptr,
    kv_seq_lens,
    block_tables,
    kv_page_indices,
    page_size,
    max_q_len,
    max_kv_len,
    kv_input_form,
) -> None:
    """Closed-set structural validation. Cheap (host-only, shape-derived)."""
    checks = [
        ("qo_indptr", qo_indptr, 1),
        ("kv_seq_lens", kv_seq_lens, 1),
    ]
    if block_tables is not None:
        checks.append(("block_tables", block_tables, 2))
    if kv_page_indices is not None:
        checks.append(("kv_page_indices", kv_page_indices, 1))
    for name, t, dim in checks:
        _expect(isinstance(t, torch.Tensor), f"{name} must be a torch.Tensor")
        _expect(
            t.is_cuda and t.device == device,
            f"{name} must be on CUDA device {device}, got {t.device}",
        )
        _expect(
            t.dtype == torch.int32,
            f"{name} must be int32, got {t.dtype} — torch cumsum/arange "
            "default to int64; build with dtype=torch.int32 or .int()",
        )
        _expect(t.dim() == dim, f"{name} must be {dim}-D, got shape {tuple(t.shape)}")

    b = kv_seq_lens.shape[0]
    _expect(b >= 1, "batch size must be >= 1")
    _expect(
        qo_indptr.shape[0] == b + 1,
        f"qo_indptr must have shape (batch_size+1,) = ({b + 1},), got "
        f"{tuple(qo_indptr.shape)} — it is a token-unit prefix sum "
        "(qo_indptr[0] = 0)",
    )
    if block_tables is not None:
        _expect(
            block_tables.shape[0] == b,
            f"block_tables must have shape (batch_size, max_pages) with "
            f"batch_size={b}, got {tuple(block_tables.shape)}",
        )
    _expect_page_size(page_size, kv_input_form)
    for nm, v in (("max_q_len", max_q_len), ("max_kv_len", max_kv_len)):
        _expect(
            isinstance(v, int) and v >= 1,
            f"{nm} must be a positive host int (required; it kills the "
            f"hidden device sync), got {v!r}",
        )
    if block_tables is not None:
        capacity = block_tables.shape[1] * page_size
        _expect(
            max_kv_len <= capacity,
            f"max_kv_len ({max_kv_len}) exceeds block_tables capacity "
            f"({block_tables.shape[1]} pages x page_size {page_size} = "
            f"{capacity})",
        )


def validate_values(
    qo_indptr,
    kv_seq_lens,
    block_tables,
    kv_page_indices,
    page_size,
    max_q_len,
    max_kv_len,
    qo_indptr_cpu,
    kv_seq_lens_cpu,
    causal,
) -> None:
    """Value-level validation against host mirrors. Always runs.

    These checks are what turns "silently wrong" into "loud error" for
    value corruption: an under-claimed max, an indptr that does not sum
    to the token count, or KV lens exceeding the table capacity would
    otherwise reach a kernel that trusts them as layout/scheduling truth.
    """
    _expect(
        qo_indptr_cpu.device.type == "cpu"
        and tuple(qo_indptr_cpu.shape) == tuple(qo_indptr.shape),
        "qo_indptr_cpu must be a CPU mirror with the same shape as qo_indptr",
    )
    _expect(
        kv_seq_lens_cpu.device.type == "cpu"
        and tuple(kv_seq_lens_cpu.shape) == tuple(kv_seq_lens.shape),
        "kv_seq_lens_cpu must be a CPU mirror with the same shape as kv_seq_lens",
    )
    d = qo_indptr_cpu.diff()
    if not bool((d > 0).all()):
        bad = int((d <= 0).nonzero()[0])
        raise ValueError(
            f"qo_indptr must be strictly increasing (q_len >= 1); entry "
            f"{bad}->{bad + 1} is {int(qo_indptr_cpu[bad])}->"
            f"{int(qo_indptr_cpu[bad + 1])} — zero-length requests are "
            "outside the v1 envelope; filter them before plan()"
        )
    _expect(int(qo_indptr_cpu[0]) == 0, "qo_indptr[0] must be 0")
    _expect(
        int(d.max()) <= max_q_len,
        f"max_q_len ({max_q_len}) is smaller than the actual longest "
        f"query ({int(d.max())}) — this would silently corrupt scheduling "
        "or graph shapes downstream",
    )
    if not bool((kv_seq_lens_cpu >= 1).all()):
        bad = int((kv_seq_lens_cpu < 1).nonzero()[0])
        raise ValueError(
            f"kv_seq_lens must be >= 1 (request {bad} has "
            f"{int(kv_seq_lens_cpu[bad])}) — zero-length KV rows are "
            "outside the v1 envelope; filter empty requests before plan()"
        )
    if causal and not bool((d <= kv_seq_lens_cpu).all()):
        bad = int((d > kv_seq_lens_cpu).nonzero()[0])
        raise ValueError(
            f"causal masking requires q_len_i <= kv_len_i for every "
            f"request; request {bad} has q_len {int(d[bad])} > kv_len "
            f"{int(kv_seq_lens_cpu[bad])} (fully-masked rows have "
            "backend-divergent LSE semantics and are outside the v1 "
            "envelope)"
        )
    _expect(
        int(kv_seq_lens_cpu.max()) <= max_kv_len,
        f"max_kv_len ({max_kv_len}) is smaller than the actual longest "
        f"KV ({int(kv_seq_lens_cpu.max())})",
    )
    if block_tables is not None:
        capacity = block_tables.shape[1] * page_size
        if not bool((kv_seq_lens_cpu <= capacity).all()):
            bad = int((kv_seq_lens_cpu > capacity).nonzero()[0])
            raise ValueError(
                f"kv_seq_lens[{bad}] = {int(kv_seq_lens_cpu[bad])} exceeds "
                f"block_tables capacity ({block_tables.shape[1]} pages x "
                f"page_size {page_size} = {capacity}) — widen block_tables "
                "or fix the length"
            )
    else:
        total_pages = int(torch.sum((kv_seq_lens_cpu + page_size - 1) // page_size))
        _expect(
            kv_page_indices.shape[0] >= total_pages,
            f"kv_page_indices has {kv_page_indices.shape[0]} entries but "
            f"kv_seq_lens require {total_pages} pages at page_size "
            f"{page_size} — the flat page-id list must cover "
            "sum(ceil(kv_len/page_size)) entries in request order",
        )


def derive(
    qo_indptr,
    kv_seq_lens,
    block_tables,
    kv_page_indices,
    page_size,
    max_kv_len,
    *,
    needs_dense: bool,
) -> Derived:
    """Canonical → derived forms.  Pure device ops, zero sync.

    In production this is one fused kernel (proposal §3.2); torch ops keep the
    prototype readable.  All output shapes are static functions of the input
    shapes and host ints:

    - dense given → flat indices by capacity scatter (a boolean masked-select
      would sync to size its result; scatter does not);
    - flat indices given → dense (when a candidate needs it) by a gather of
      width ceil(max_kv_len / page_size), with each row's tail CLAMPED TO THE
      REQUEST'S OWN LAST PAGE.  The per-row clamp is load-bearing: cuDNN
      gathers K/V pages by table width before masking, so a tail that
      pointed into the over-allocated (possibly uninitialized) region of
      kv_page_indices produced NaN outputs / out-of-pool reads (found
      empirically by the NaN-page probe; see the fuzzer's
      csr_overallocated_nan_tail mutation).
    """
    dev = kv_seq_lens.device
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    q_seq_lens = qo_indptr.diff()
    cum_kv = torch.cat([zero, torch.cumsum(kv_seq_lens, 0, dtype=torch.int32)])
    pages = (kv_seq_lens + page_size - 1) // page_size  # (b,)
    kv_page_indptr = torch.cat([zero, torch.cumsum(pages, 0, dtype=torch.int32)])
    b = kv_seq_lens.shape[0]

    if block_tables is not None:
        width = block_tables.shape[1]
        capacity = b * width
        col = torch.arange(width, device=dev, dtype=torch.int32)
        valid = col.unsqueeze(0) < pages.unsqueeze(1)  # (b, width)
        # compact destination of each (row, col) lane; invalid lanes all
        # target a dummy tail slot (duplicate writes there are benign)
        dst = kv_page_indptr[:-1].unsqueeze(1).to(torch.int64) + col.unsqueeze(0).to(
            torch.int64
        )
        dst = torch.where(valid, dst, torch.full_like(dst, capacity))
        buf = torch.empty(capacity + 1, dtype=torch.int32, device=dev)
        buf.scatter_(0, dst.reshape(-1), block_tables.reshape(-1))
        return Derived(q_seq_lens, cum_kv, kv_page_indptr, buf[:capacity], block_tables)

    dense = None
    if needs_dense:
        width = (max_kv_len + page_size - 1) // page_size  # host int, no sync
        col = torch.arange(width, device=dev, dtype=torch.int64)
        src = kv_page_indptr[:-1].to(torch.int64).unsqueeze(1) + col.unsqueeze(0)
        # clamp each row's tail to the request's OWN last live page (kv_len
        # >= 1 is validated, so every row owns at least one).  Load-bearing:
        # cuDNN gathers K/V pages by table width before masking, so a tail
        # pointing into a neighbouring request or the over-allocated
        # (possibly uninitialized) region of kv_page_indices produced NaN
        # outputs / out-of-pool reads (fuzzer: csr_overallocated_nan_tail).
        row_last = kv_page_indptr[1:].to(torch.int64).unsqueeze(1) - 1
        src = torch.minimum(src, row_last)
        dense = (
            kv_page_indices.to(torch.int64)
            .gather(0, src.reshape(-1))
            .reshape(b, width)
            .to(torch.int32)
        )
    return Derived(q_seq_lens, cum_kv, kv_page_indptr, kv_page_indices, dense)


__all__ = ["Derived", "derive", "validate_structure", "validate_values"]
