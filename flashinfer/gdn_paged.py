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

Gated Delta Rule (GDN) on a paged recurrent-state cache.

The caller keeps the recurrent states in one pool ``state_pool[num_pages, H,
V, K]`` and gives every sequence a list of page ids in CSR form
(``page_indices`` / ``page_indptr``), as for a paged KV cache. Page ``j`` of a
sequence holds the state after the last token of that page, so a sequence that
shares a full-page prefix with another one can start from the last shared
page. Both entry points derive every pool row on the device from the page
table (one small launch, or none when the caller passes rows precomputed with
:func:`gdn_paged_prefill_slots` / :func:`gdn_paged_decode_slots`, which every
layer of a step can share) and run the existing kernels with separate read
and write rows; nothing synchronizes with the host.
"""

import math
from typing import Literal, NamedTuple, Optional

import torch

from .api_logging import flashinfer_api
from .gdn_decode import gated_delta_rule_decode_pretranspose
from .gdn_prefill import chunk_gated_delta_rule
from .trace.templates.gdn import gdn_decode_paged_trace, gdn_prefill_paged_trace


def _check_page_table(
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    last_page_len: torch.Tensor,
    num_seqs: int,
    device: torch.device,
) -> None:
    for name, tensor, shape in (
        ("page_indices", page_indices, None),
        ("page_indptr", page_indptr, (num_seqs + 1,)),
        ("last_page_len", last_page_len, (num_seqs,)),
    ):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be int32 or int64, got {tensor.dtype}")
        if tensor.ndim != 1 or (shape is not None and tuple(tensor.shape) != shape):
            expected = "1D" if shape is None else f"of shape {shape}"
            raise ValueError(f"{name} must be {expected}, got {tuple(tensor.shape)}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        # The slot kernels and the GDN kernels index these with unit stride.
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _check_cu_seqlens(cu_seqlens: torch.Tensor, device: torch.device) -> None:
    if (
        cu_seqlens.dtype not in (torch.int32, torch.int64)
        or cu_seqlens.ndim != 1
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be a contiguous 1D int32 or int64 tensor")
    if cu_seqlens.device != device:
        raise ValueError(f"cu_seqlens must be on {device}, got {cu_seqlens.device}")


def _check_page_size(page_size: int) -> None:
    if type(page_size) is not int:
        raise ValueError(
            f"page_size must be a Python int, got {type(page_size).__name__}"
        )
    if page_size <= 0 or page_size % 64 != 0:
        raise ValueError(
            f"page_size must be a positive multiple of 64, got {page_size}"
        )


def _check_slots(
    slots: "GDNPagedPrefillSlots | GDNPagedDecodeSlots",
    num_seqs: int,
    device: torch.device,
) -> None:
    for name, tensor in zip(slots._fields, slots, strict=True):
        if (
            tensor.dtype != torch.int32
            or tuple(tensor.shape) != (num_seqs,)
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"slots.{name} must be a contiguous int32 tensor of shape "
                f"({num_seqs},) on {device}"
            )


# Sequences per program of the slot kernels.
_SLOTS_BLOCK = 128


class GDNPagedPrefillSlots(NamedTuple):
    """Per-sequence pool rows of a paged prefill batch, all ``[num_seqs]``
    int32. See :func:`gdn_paged_prefill_slots`."""

    state_indices: torch.Tensor
    """Page holding the last prefix token, read as the initial state (``-1``:
    empty prefix, zero state)."""
    output_state_indices: torch.Tensor
    """Last page, written with the final state. ``-1`` means there is no new
    token or the final state is the checkpoint of a page that ends there."""
    checkpoint_cu_starts: torch.Tensor
    """``page_indices`` entry of the first page boundary the new tokens cross."""
    prefix_lens: torch.Tensor
    """Tokens before the new ones."""


class GDNPagedDecodeSlots(NamedTuple):
    """Per-sequence pool rows of a paged decode batch, both ``[B]`` int32. See
    :func:`gdn_paged_decode_slots`."""

    initial_state_indices: torch.Tensor
    """Page read as the state before the decoded token."""
    output_state_indices: torch.Tensor
    """Last page, written with the state after the decoded token."""


@flashinfer_api
def gdn_paged_prefill_slots(
    cu_seqlens: torch.Tensor,
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    page_size: int,
    last_page_len: torch.Tensor,
) -> GDNPagedPrefillSlots:
    r"""Pool rows that :func:`chunk_gated_delta_rule_paged` derives from a page
    table, computed in one launch without a host synchronization.

    The rows depend only on the batch (``cu_seqlens`` and the page table), not
    on the layer, so a model can compute them once per step and pass them to
    every layer's call as ``slots``.

    Parameters
    ----------
    cu_seqlens, page_indices, page_indptr, page_size, last_page_len
        As in :func:`chunk_gated_delta_rule_paged`.

    Returns
    -------
    GDNPagedPrefillSlots
        ``[num_seqs]`` int32 read rows, final-state rows, first checkpoint
        entries and prefix lengths.
    """
    _check_page_size(page_size)
    device = cu_seqlens.device
    if device.type != "cuda":
        raise ValueError(f"cu_seqlens must be a CUDA tensor, got {device}")
    _check_cu_seqlens(cu_seqlens, device)
    num_seqs = cu_seqlens.size(0) - 1
    _check_page_table(page_indices, page_indptr, last_page_len, num_seqs, device)
    slots = GDNPagedPrefillSlots(
        *(torch.empty(num_seqs, dtype=torch.int32, device=device) for _ in range(4))
    )
    if num_seqs > 0:
        from .triton.gdn_paged import gdn_paged_prefill_slots_kernel

        with torch.cuda.device(device):
            gdn_paged_prefill_slots_kernel[
                ((num_seqs + _SLOTS_BLOCK - 1) // _SLOTS_BLOCK,)
            ](
                cu_seqlens,
                page_indices,
                page_indptr,
                last_page_len,
                *slots,
                num_seqs,
                page_size,
                BLOCK=_SLOTS_BLOCK,
            )
    return slots


@flashinfer_api
def gdn_paged_decode_slots(
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    last_page_len: torch.Tensor,
) -> GDNPagedDecodeSlots:
    r"""Pool rows that :func:`gated_delta_rule_decode_paged` derives from a page
    table, computed in one launch without a host synchronization. Compute them
    once per decode step and pass them to every layer's call as ``slots``.

    Parameters
    ----------
    page_indices, page_indptr, last_page_len
        As in :func:`gated_delta_rule_decode_paged`.

    Returns
    -------
    GDNPagedDecodeSlots
        ``[B]`` int32 read rows (the previous page when the decoded token
        opens a new page, else the last page) and write rows (the last page).
    """
    num_seqs = page_indptr.size(0) - 1
    device = page_indptr.device
    _check_page_table(page_indices, page_indptr, last_page_len, num_seqs, device)
    slots = GDNPagedDecodeSlots(
        *(torch.empty(num_seqs, dtype=torch.int32, device=device) for _ in range(2))
    )
    if num_seqs > 0:
        from .triton.gdn_paged import gdn_paged_decode_slots_kernel

        with torch.cuda.device(device):
            gdn_paged_decode_slots_kernel[
                ((num_seqs + _SLOTS_BLOCK - 1) // _SLOTS_BLOCK,)
            ](
                page_indices,
                page_indptr,
                last_page_len,
                *slots,
                num_seqs,
                BLOCK=_SLOTS_BLOCK,
            )
    return slots


def _chunk_gated_delta_rule_paged_cake_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    scale: Optional[float],
    *,
    cu_seqlens: torch.Tensor,
    state_pool: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    slots: GDNPagedPrefillSlots,
    use_qk_l2norm_in_kernel: bool,
    output: Optional[torch.Tensor],
    gate_domain: str,
    use_gate_in_kernel: bool,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    beta_is_logit: bool,
    max_seqlen: Optional[int],
) -> torch.Tensor:
    from .gdn_kernels.blackwell import (
        _chunk_gated_delta_rule_gdn_cp_paged_sm100,
        _gdn_cp_paged_rejection_reason,
    )
    from .gdn_kernels.gates import materialize_gates, validate_gate_inputs

    if _chunk_gated_delta_rule_gdn_cp_paged_sm100 is None:
        raise ValueError('backend="cake_gdn" needs the Cake GDN CP programs')
    reason = _gdn_cp_paged_rejection_reason(q, k, v, state_pool, page_size=page_size)
    if reason is not None:
        raise ValueError(f'backend="cake_gdn": {reason}')
    if (
        gate_domain != "linear"
        or use_gate_in_kernel
        or beta_is_logit
        or A_log is not None
        or dt_bias is not None
    ):
        validate_gate_inputs(
            q,
            v,
            g,
            beta,
            gate_domain,
            use_gate_in_kernel,
            A_log,
            dt_bias,
            beta_is_logit,
        )
    if g is not None and (gate_domain != "linear" or use_gate_in_kernel):
        g, _ = materialize_gates(
            g, None, gate_domain, use_gate_in_kernel, A_log, dt_bias, False
        )
    if beta is not None and beta_is_logit:
        _, beta = materialize_gates(None, beta, "linear", False, None, None, True)
    total = q.size(0)
    num_sab_heads = max(q.size(1), v.size(1))
    gate_shape = (total, num_sab_heads)
    alpha = (
        torch.ones(gate_shape, dtype=torch.float32, device=q.device)
        if g is None
        else g.float().contiguous()
    )
    beta = (
        torch.ones(gate_shape, dtype=torch.float32, device=q.device)
        if beta is None
        else beta.float().contiguous()
    )
    if output is None:
        output = torch.empty(
            (total, num_sab_heads, q.size(2)), dtype=q.dtype, device=q.device
        )
    _chunk_gated_delta_rule_gdn_cp_paged_sm100(
        output,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        float(scale) if scale else 1.0 / math.sqrt(q.size(2)),
        state_pool=state_pool,
        state_indices=slots.state_indices,
        output_state_indices=slots.output_state_indices,
        checkpoint_state_indices=page_indices,
        checkpoint_cu_starts=slots.checkpoint_cu_starts,
        prefix_lens=slots.prefix_lens,
        page_size=page_size,
        max_seqlen=max_seqlen,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return output


@flashinfer_api(trace=gdn_prefill_paged_trace)
def chunk_gated_delta_rule_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    *,
    cu_seqlens: torch.Tensor,
    state_pool: torch.Tensor,
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    page_size: int,
    last_page_len: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    gate_domain: Literal["linear", "log"] = "linear",
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
    slots: Optional[GDNPagedPrefillSlots] = None,
    backend: Literal["flashinfer", "cake_gdn"] = "flashinfer",
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    r"""Chunked GDN prefill that reads and writes a paged recurrent-state pool.

    For every sequence ``b`` the call processes its new tokens
    ``q[cu_seqlens[b]:cu_seqlens[b + 1]]``, which are the last tokens of the
    sequence's ``(num_pages_b - 1) * page_size + last_page_len[b]`` tokens
    (``num_pages_b = page_indptr[b + 1] - page_indptr[b]``). The tokens before
    them form its prefix. The call

    - reads the initial state from the page that holds the last prefix token
      (the zero state when the prefix is empty),
    - writes the state at every page boundary it crosses to that page, and
    - writes the final state to the last page.

    Pages that end at or before the prefix are only read, so a full-page
    prefix shared with other sequences is never written. A prefix that ends
    inside a page continues that (private) page in place. Prefix lengths need
    not be multiples of the 64-token chunk.

    Parameters
    ----------
    q, k, v, g, beta, scale, use_qk_l2norm_in_kernel, output, gate_domain, use_gate_in_kernel, A_log, dt_bias, beta_is_logit
        As in :func:`flashinfer.gdn_prefill.chunk_gated_delta_rule`.
    cu_seqlens : torch.Tensor
        ``[num_seqs + 1]`` contiguous int32/int64 offsets of the new tokens of
        each sequence in ``q``/``k``/``v``. Zero-length sequences are allowed
        and leave the pool unchanged.
    state_pool : torch.Tensor
        ``[num_pages, num_sab_heads, head_size, head_size]`` state pool,
        updated in place. Each row must be one dense ``[H, V, K]`` block; the
        row stride may be padded. Its dtype is the state dtype.
    page_indices : torch.Tensor
        ``[num_indices]`` contiguous int32/int64 page ids. The ids may be in
        any order and need not be consecutive.
    page_indptr : torch.Tensor
        ``[num_seqs + 1]`` contiguous int32/int64 CSR offsets into
        ``page_indices``.
        Sequence ``b`` owns ``page_indices[page_indptr[b]:page_indptr[b + 1]]``
        and must own at least one page.
    page_size : int
        Tokens per page, a positive multiple of 64.
    last_page_len : torch.Tensor
        ``[num_seqs]`` contiguous int32/int64 number of tokens in the last page
        of each sequence after this call, in ``[1, page_size]``.
    slots : GDNPagedPrefillSlots, optional
        Rows from :func:`gdn_paged_prefill_slots` for the same ``cu_seqlens``
        and page table, typically shared by every layer of a step. Without
        them the call computes them (one extra launch). They are not checked
        against the page table.
    backend : str
        ``"flashinfer"`` (default) runs the serial chunked kernel.
        ``"cake_gdn"`` runs the Cake four-stage context-parallel prefill,
        which splits each sequence across SMs and writes the page states from
        inside the split; it needs SM100, SM103 or SM107, a float32
        ``state_pool`` whose row stride is a multiple of 4 elements, and every
        prefix length (the tokens before ``cu_seqlens[b]``) a multiple of 64.
        The caller must keep the prefixes 64-aligned; they are not checked.
        For a sequence whose prefix is not, every page row it would write (the
        page of each page boundary it crosses and its last page) is filled
        with NaN instead of a state; the other sequences are unaffected. Its
        results are not bitwise those of
        ``"flashinfer"`` (about 4e-3 relative on the states), and its
        checkpoint bits depend on the batch shape. It helps when few states
        run in parallel: on GB300 it is up to 4.3x faster than
        ``"flashinfer"`` at small ``num_seqs * num_sab_heads`` with long
        sequences, about even at 32 with 2048 new tokens per sequence, and
        slower (0.4-0.8x) at 64 or more.
    max_seqlen : int, optional
        With ``backend="cake_gdn"``, an upper bound of the new tokens of any
        sequence; it sizes the split. Without it the bound is the total.

    Returns
    -------
    torch.Tensor
        The output ``[total_seq_len, num_o_heads, head_size]``.

    Notes
    -----
    A page written by one sequence must not be read or written by another
    sequence of the same call. The page table and lengths are caller
    preconditions; they are not validated against each other on the device.
    """
    _check_page_size(page_size)
    _check_cu_seqlens(cu_seqlens, q.device)
    num_seqs = cu_seqlens.size(0) - 1
    _check_page_table(page_indices, page_indptr, last_page_len, num_seqs, q.device)
    if state_pool.ndim != 4:
        raise ValueError(
            "state_pool must be [num_pages, num_sab_heads, head_size, head_size], "
            f"got shape {tuple(state_pool.shape)}"
        )
    if backend not in ("flashinfer", "cake_gdn"):
        raise ValueError(f'backend must be "flashinfer" or "cake_gdn", got {backend!r}')
    if slots is None:
        slots = gdn_paged_prefill_slots(
            cu_seqlens, page_indices, page_indptr, page_size, last_page_len
        )
    else:
        _check_slots(slots, num_seqs, q.device)

    if backend == "cake_gdn":
        return _chunk_gated_delta_rule_paged_cake_cp(
            q,
            k,
            v,
            g,
            beta,
            scale,
            cu_seqlens=cu_seqlens,
            state_pool=state_pool,
            page_indices=page_indices,
            page_size=page_size,
            slots=slots,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output=output,
            gate_domain=gate_domain,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            beta_is_logit=beta_is_logit,
            max_seqlen=max_seqlen,
        )

    output, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=state_pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=output,
        output_state=state_pool,
        state_checkpoints=state_pool,
        checkpoint_cu_starts=slots.checkpoint_cu_starts,
        checkpoint_every_n_tokens=page_size,
        use_cp=False,
        state_indices=slots.state_indices,
        gate_domain=gate_domain,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        beta_is_logit=beta_is_logit,
        output_state_indices=slots.output_state_indices,
        checkpoint_state_indices=page_indices,
        prefix_lens=slots.prefix_lens,
    )
    return output


@flashinfer_api(trace=gdn_decode_paged_trace)
def gated_delta_rule_decode_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    *,
    state_pool: torch.Tensor,
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    last_page_len: torch.Tensor,
    backend: Literal["auto", "flashinfer", "cake_gdn", "rubin"] = "auto",
    slots: Optional[GDNPagedDecodeSlots] = None,
) -> torch.Tensor:
    r"""Single-token GDN decode on a paged recurrent-state pool.

    The decoded token is the last token of its sequence: it sits at position
    ``last_page_len[b]`` of the sequence's last page. When the token opens a
    new page (``last_page_len[b] == 1``) the state is read from the previous
    page and written to the new one, so the previous page, which may be a
    shared full page, is not modified. Otherwise the last page is read and
    updated in place.

    Parameters
    ----------
    q, k, v, A_log, a, dt_bias, b, scale, output, use_qk_l2norm, backend
        As in :func:`flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose`
        with ``T = 1``.
    state_pool : torch.Tensor
        ``[num_pages, HV, V, K]`` state pool, updated in place. Supported
        dtypes and layouts are those of the ``initial_state`` pool of
        :func:`~flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose`.
    page_indices, page_indptr : torch.Tensor
        Contiguous int32/int64 CSR page table, as in
        :func:`chunk_gated_delta_rule_paged`.
    last_page_len : torch.Tensor
        ``[B]`` contiguous int32/int64 number of tokens in the last page
        including the decoded token, in ``[1, page_size]``.
    slots : GDNPagedDecodeSlots, optional
        Rows from :func:`gdn_paged_decode_slots` for the same page table,
        typically shared by every layer of a step. Without them the call
        computes them (one extra launch). They are not checked against the
        page table.

    Returns
    -------
    torch.Tensor
        The output ``[B, 1, HV, V]``.

    Notes
    -----
    Every sequence must hold at least one token before the decoded one, so a
    sequence whose decoded token opens a new page owns at least two pages.
    """
    B, T = q.shape[:2]
    if T != 1:
        raise ValueError(f"gated_delta_rule_decode_paged decodes one token, got T={T}")
    _check_page_table(page_indices, page_indptr, last_page_len, B, q.device)
    if slots is None:
        slots = gdn_paged_decode_slots(page_indices, page_indptr, last_page_len)
    else:
        _check_slots(slots, B, q.device)
    output, _ = gated_delta_rule_decode_pretranspose(
        q,
        k,
        v,
        None,
        A_log,
        a,
        dt_bias,
        b,
        scale=scale,
        output=output,
        use_qk_l2norm=use_qk_l2norm,
        initial_state=state_pool,
        initial_state_indices=slots.initial_state_indices,
        output_state_indices=slots.output_state_indices,
        backend=backend,
    )
    return output
