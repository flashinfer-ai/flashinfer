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
table and run the existing kernels with separate read and write rows; neither
synchronizes with the host.
"""

from typing import Literal, Optional

import torch

from .api_logging import flashinfer_api
from .gdn_decode import gated_delta_rule_decode_pretranspose
from .gdn_prefill import chunk_gated_delta_rule


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


def _gather_rows(
    page_indices: torch.Tensor, entries: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """``page_indices[entries]`` where ``valid``, else ``-1``."""
    if page_indices.numel() == 0:
        return torch.full_like(entries, -1, dtype=page_indices.dtype)
    rows = page_indices[entries.clamp(0, page_indices.numel() - 1)]
    return torch.where(valid, rows, torch.full_like(rows, -1))


@flashinfer_api
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
    page_len: int,
    last_page_len: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    cu_seqlens_cpu=None,
    gate_domain: Literal["linear", "log"] = "linear",
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
) -> torch.Tensor:
    r"""Chunked GDN prefill that reads and writes a paged recurrent-state pool.

    For every sequence ``b`` the call processes its new tokens
    ``q[cu_seqlens[b]:cu_seqlens[b + 1]]``, which are the last tokens of the
    sequence's ``(num_pages_b - 1) * page_len + last_page_len[b]`` tokens
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
    q, k, v, g, beta, scale, use_qk_l2norm_in_kernel, output, max_seqlen, cu_seqlens_cpu, gate_domain, use_gate_in_kernel, A_log, dt_bias, beta_is_logit
        As in :func:`flashinfer.gdn_prefill.chunk_gated_delta_rule`.
    cu_seqlens : torch.Tensor
        ``[num_seqs + 1]`` int32/int64 offsets of the new tokens of each
        sequence in ``q``/``k``/``v``. Zero-length sequences are allowed and
        leave the pool unchanged.
    state_pool : torch.Tensor
        ``[num_pages, num_sab_heads, head_size, head_size]`` state pool,
        updated in place. Each row must be one dense ``[H, V, K]`` block; the
        row stride may be padded. Its dtype is the state dtype.
    page_indices : torch.Tensor
        ``[num_indices]`` contiguous int32/int64 page ids. The ids may be in
        any order and need not be consecutive.
    page_indptr : torch.Tensor
        ``[num_seqs + 1]`` int32/int64 CSR offsets into ``page_indices``.
        Sequence ``b`` owns ``page_indices[page_indptr[b]:page_indptr[b + 1]]``
        and must own at least one page.
    page_len : int
        Tokens per page, a positive multiple of 64.
    last_page_len : torch.Tensor
        ``[num_seqs]`` int32/int64 number of tokens in the last page of each
        sequence after this call, in ``[1, page_len]``.

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
    if type(page_len) is not int:
        raise ValueError(
            f"page_len must be a Python int, got {type(page_len).__name__}"
        )
    if page_len <= 0 or page_len % 64 != 0:
        raise ValueError(f"page_len must be a positive multiple of 64, got {page_len}")
    num_seqs = cu_seqlens.size(0) - 1
    _check_page_table(page_indices, page_indptr, last_page_len, num_seqs, q.device)
    # The kernels read page_indices directly as checkpoint rows.
    if not page_indices.is_contiguous():
        raise ValueError("page_indices must be contiguous")
    if state_pool.ndim != 4:
        raise ValueError(
            "state_pool must be [num_pages, num_sab_heads, head_size, head_size], "
            f"got shape {tuple(state_pool.shape)}"
        )

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    page_starts = page_indptr[:-1]
    total_lens = (page_indptr[1:] - page_starts - 1) * page_len + last_page_len
    prefix_lens = total_lens - seq_lens
    # The prefix's last token lives on entry (prefix - 1) // page_len; the
    # first boundary crossed closes entry prefix // page_len.
    read_entries = page_starts + torch.div(
        prefix_lens - 1, page_len, rounding_mode="floor"
    )
    state_indices = _gather_rows(page_indices, read_entries, prefix_lens > 0)
    # A final state that ends on a page boundary is that page's checkpoint.
    output_state_indices = _gather_rows(
        page_indices,
        page_indptr[1:] - 1,
        (seq_lens > 0) & (total_lens % page_len != 0),
    )
    checkpoint_starts = page_starts + torch.div(
        prefix_lens, page_len, rounding_mode="floor"
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
        checkpoint_cu_starts=checkpoint_starts,
        checkpoint_every_n_tokens=page_len,
        use_cp=False,
        state_indices=state_indices,
        max_seqlen=max_seqlen,
        gate_domain=gate_domain,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        beta_is_logit=beta_is_logit,
        cu_seqlens_cpu=cu_seqlens_cpu,
        output_state_indices=output_state_indices,
        checkpoint_state_indices=page_indices,
        prefix_lens=prefix_lens,
    )
    return output


@flashinfer_api
def gated_delta_rule_decode_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    output: Optional[torch.Tensor] = None,
    *,
    state_pool: torch.Tensor,
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    last_page_len: torch.Tensor,
    backend: Literal["auto", "flashinfer", "cake_gdn", "rubin"] = "auto",
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
    q, k, v, A_log, a, dt_bias, b, scale, use_qk_l2norm, output, backend
        As in :func:`flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose`
        with ``T = 1``.
    state_pool : torch.Tensor
        ``[num_pages, HV, V, K]`` state pool, updated in place. Supported
        dtypes and layouts are those of the ``initial_state`` pool of
        :func:`~flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose`.
    page_indices, page_indptr : torch.Tensor
        int32/int64 CSR page table, as in :func:`chunk_gated_delta_rule_paged`.
    last_page_len : torch.Tensor
        ``[B]`` int32/int64 number of tokens in the last page including the
        decoded token, in ``[1, page_len]``.

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

    last_entries = page_indptr[1:] - 1
    output_state_indices = page_indices[last_entries]
    previous = page_indices[(last_entries - 1).clamp(min=0)]
    initial_state_indices = torch.where(
        last_page_len > 1, output_state_indices, previous
    )
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
        initial_state_indices=initial_state_indices,
        output_state_indices=output_state_indices,
        backend=backend,
    )
    return output
