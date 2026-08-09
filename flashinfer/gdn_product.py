"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Gated DeltaProduct (arXiv:2502.10297) -- API layer.

DeltaProduct takes ``num_householder`` delta-rule steps per token instead of one:

    A(x_i) = alpha_i * prod_{j=1..n_h} (I - beta_{i,j} k_{i,j} k_{i,j}^T)

Phase 1 implements this as a host-side expansion over the existing GDN kernels:
GDP is GDN run on a sequence ``n_h`` times longer, with the decay neutralised on
all but the first micro-step of each token and the query read only from the last.
No new kernel is required for correctness.

The recurrent state is UNCHANGED in size -- ``n_h`` costs FLOPs, not bytes.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .gdn_prefill import chunk_gated_delta_rule


def chunk_gated_delta_product(
    q: torch.Tensor,  # [total_seq_len,      num_q_heads, head_size]
    k: torch.Tensor,  # [total_seq_len, n_h, num_k_heads, head_size]
    v: torch.Tensor,  # [total_seq_len, n_h, num_v_heads, head_size]
    g: Optional[torch.Tensor] = None,  # [total_seq_len,      num_sab_heads]
    beta: Optional[torch.Tensor] = None,  # [total_seq_len, n_h, num_sab_heads]
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated DeltaProduct attention for prefill.

    Mirrors :func:`flashinfer.gdn_prefill.chunk_gated_delta_rule`, with a
    householder axis added to ``k``, ``v`` and ``beta``. At ``n_h == 1`` this
    delegates straight through and is bit-identical to the GDN entry point.

    Parameters
    ----------
    q : torch.Tensor
        Queries, ``[total_seq_len, num_q_heads, head_size]``. **One per real
        token** -- the query axis is not expanded.
    k, v : torch.Tensor
        Keys/values, ``[total_seq_len, num_householder, num_*_heads, head_size]``.
        The householder axis sits immediately after the token axis so that
        ``reshape(total_seq_len * n_h, ...)`` yields the ``(t n) h d`` ordering
        the expansion needs.
    g : torch.Tensor, optional
        Forget gate (alpha), ``[total_seq_len, num_sab_heads]``. **One per real
        token** -- the gate models time passing, while the ``n_h`` householders
        are all the same time step. MULTIPLICATIVE, neutral value ``1.0``
        (not the log-space decay FLA uses, whose neutral value is ``0.0``).
    beta : torch.Tensor, optional
        Update gate, ``[total_seq_len, num_householder, num_sab_heads]`` -- one
        per (token, householder).
    initial_state, output_state, state_indices, cu_seqlens, ...
        As in ``chunk_gated_delta_rule``. Note the state shape does NOT depend
        on ``num_householder``.

    Returns
    -------
    Same contract as ``chunk_gated_delta_rule``: ``output`` when
    ``output_final_state`` is False, else ``(output, final_state)``. ``output``
    has one row per REAL token, ``[total_seq_len, num_o_heads, head_size]``.
    """
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"k/v must carry a householder axis [T, n_h, H, D]; "
            f"got k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}"
        )
    num_householder = k.size(1)
    if v.size(1) != num_householder:
        raise ValueError(
            f"k/v householder counts differ: {num_householder} vs {v.size(1)}"
        )
    if beta is not None and beta.dim() == 3 and beta.size(1) != num_householder:
        raise ValueError(f"beta householder count {beta.size(1)} != {num_householder}")
    if g is not None and g.dim() != 2:
        raise ValueError(
            f"g is one gate per REAL token, expected [T, H]; got {tuple(g.shape)}"
        )
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required (varlen mode), as for GDN")

    # n_h == 1 is plain GDN. Delegate so this path stays bit-identical.
    if num_householder == 1:
        return chunk_gated_delta_rule(
            q,
            k.squeeze(1),
            v.squeeze(1),
            g,
            beta.squeeze(1) if beta is not None else None,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            use_qk_l2norm_in_kernel,
            output=output,
            output_state=output_state,
            state_indices=state_indices,
        )

    # -----------------------------------------------------------------------
    # TODO(you): the expansion.
    #
    # Build the n_h-times-longer sequence, call chunk_gated_delta_rule on it,
    # then slice back to one row per real token.
    #
    #   k, v      reshape [T, n_h, H, D] -> [T * n_h, H, D]
    #   beta      reshape [T, n_h, H]    -> [T * n_h, H]
    #   g         scatter: g_flat[0::n_h] = g, ELSE 1.0   (multiplicative!)
    #   q         scatter: q_flat[n_h-1::n_h] = q, else anything (rows discarded)
    #   cu_seqlens                       -> cu_seqlens * n_h
    #
    #   out_flat = chunk_gated_delta_rule(...)   # [T * n_h, num_o_heads, D]
    #   out      = out_flat[n_h - 1 :: n_h]      # one row per real token
    #
    # Three things to get right, in rough order of how easy they are to miss:
    #
    #  * `output=` / `output_state=`. The caller's `output` buffer is sized for
    #    T rows, but the kernel writes T * n_h. Allocate the expanded buffer
    #    here and copy the strided slice into the caller's, or pass None and
    #    copy afterwards. `output_state` is NOT expanded -- state shape is
    #    independent of n_h -- so it can be forwarded unchanged.
    #
    #  * The strided slice `[n_h - 1 :: n_h]` is only valid because every
    #    sequence's expanded length is a multiple of n_h, so every sequence
    #    still starts at a multiple of n_h. That is what makes
    #    `cu_seqlens * n_h` sufficient rather than needing per-sequence fixups.
    #
    #  * Zeroed q rows are safe (the in-kernel l2 norm has an eps), but the
    #    rows are discarded anyway -- so replicating q is equally fine and
    #    avoids depending on that eps.
    #
    # `tests/gdn/test_prefill_delta_product.py::test_reference_equals_expanded_
    # delta_rule` already validates this exact expansion in pure torch. If the
    # kernel disagrees, the fault is in the marshalling here, not the algebra.
    # -----------------------------------------------------------------------
    raise NotImplementedError("chunk_gated_delta_product: the n_h expansion")
