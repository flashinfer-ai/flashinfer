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
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .gdp_prefill import chunk_gated_delta_product as _gdp_prefill


def chunk_gated_delta_product(
    q: torch.Tensor,  # [total_seq_len,      num_q_heads, head_size]
    k: torch.Tensor,  # [total_seq_len, n_h, num_k_heads, head_size]
    v: torch.Tensor,  # [total_seq_len, n_h, num_v_heads, head_size_v]
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

    The expanded-axis view of :func:`flashinfer.chunk_gated_delta_product`:
    ``k``, ``v`` and ``beta`` carry the householder axis next to the token axis
    instead of folded into it, and ``n_h`` comes from the shape rather than an
    argument. Dispatches to ``backend="flashinfer"``. At ``n_h == 1`` this is
    bit-identical to :func:`flashinfer.chunk_gated_delta_rule`.

    Parameters
    ----------
    q : torch.Tensor
        Queries, ``[total_seq_len, num_q_heads, head_size]``. **One per real
        token** -- the query axis is not expanded.
    k : torch.Tensor
        Keys, ``[total_seq_len, num_householder, num_k_heads, head_size]``.
        The householder axis sits immediately after the token axis so that
        ``reshape(total_seq_len * n_h, ...)`` yields the ``(t n) h d`` ordering
        the expansion needs.
    v : torch.Tensor
        Values, ``[total_seq_len, num_householder, num_v_heads, head_size_v]``,
        same householder placement as ``k``.  ``head_size_v`` may be narrower
        than ``head_size`` (rectangular state); see
        :func:`~flashinfer.gdn_prefill.chunk_gated_delta_rule` for which
        architectures implement that.
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
    has one row per REAL token, ``[total_seq_len, num_o_heads, head_size_v]``.
    When ``output`` is supplied it is written in place and returned; otherwise
    a freshly allocated tensor is returned.
    """
    if q.dim() != 3 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "expected q=[T, H, D] and k/v=[T, n_h, H, D]; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    num_householder = k.size(1)
    if num_householder < 1:
        raise ValueError("num_householder must be at least 1")
    total_tokens, num_q_heads, head_size = q.shape
    num_v_heads = v.size(2)
    head_size_v = v.size(3)
    num_sab_heads = max(num_q_heads, num_v_heads)
    if k.size(0) != total_tokens or v.size(0) != total_tokens:
        raise ValueError("q, k, and v must have the same token dimension")
    if v.size(1) != num_householder:
        raise ValueError(
            f"k/v householder counts differ: {num_householder} vs {v.size(1)}"
        )
    if k.size(3) != head_size:
        raise ValueError("q and k must have the same head size")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if beta is not None:
        expected_beta = (total_tokens, num_householder, num_sab_heads)
        if beta.shape != expected_beta:
            raise ValueError(
                f"expected beta shape {expected_beta}, got {tuple(beta.shape)}"
            )
        if beta.dtype != torch.float32 or beta.device != q.device:
            raise ValueError("beta must be float32 on the same device as q")
    if g is not None:
        expected_g = (total_tokens, num_sab_heads)
        if g.shape != expected_g:
            raise ValueError(f"expected g shape {expected_g}, got {tuple(g.shape)}")
        if g.dtype != torch.float32 or g.device != q.device:
            raise ValueError("g must be float32 on the same device as q")
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required (varlen mode), as for GDN")

    # GDP = GDN with a sequence n_h times longer
    k = torch.flatten(k, start_dim=0, end_dim=1)
    v = torch.flatten(v, start_dim=0, end_dim=1)
    if beta is not None:
        beta = torch.flatten(beta, start_dim=0, end_dim=1)

    if output is not None:
        expected_output_shape = (total_tokens, num_sab_heads, head_size_v)
        if output.shape != expected_output_shape:
            raise ValueError(
                f"expected output shape {expected_output_shape}, got {tuple(output.shape)}"
            )
        if output.device != q.device:
            raise ValueError("output must be on the same device as q")

    # k / v / beta already carry the householder axis next to the token axis, so
    # the micro-step view above is a free reshape.  q / g / output stay per REAL
    # token, so GDP needs no expansion scratch at all.
    if output is None:
        output = torch.empty(
            total_tokens,
            num_sab_heads,
            head_size_v,
            dtype=q.dtype,
            device=q.device,
        )

    out = _gdp_prefill(
        q,
        k,
        v,
        g,
        beta,
        num_householder,
        scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=output,
        output_state=output_state,
        backend="flashinfer",
        state_indices=state_indices,
    )

    if output_final_state:
        return output, out[-1]
    else:
        return output


# Sentinel written into `a` on the non-first micro-steps of each token, to make
# the FUSED gate evaluate to alpha == 1.0 (no decay).
#
# Unlike prefill -- where `g` is a plain multiplicative tensor and the neutral
# value is simply 1.0 -- the decode kernel computes the gate itself:
#
#     alpha = exp(-exp(A_log) * softplus(a + dt_bias))
#
# so neutralising it means driving softplus to zero through `a`. At -1e4 the
# inner exp underflows to exactly 0, hence log1p(0) == 0 and alpha == 1.0
# bit-exactly, for ANY A_log and dt_bias. Do NOT tighten this to -30: that is
# one ULP short of 1.0 once exp(A_log)*softplus() exceeds 2^-24, and do NOT use
# -inf, which becomes NaN in the kernel's `(1-use_softplus) * x` blend.
GATE_NEUTRAL_A_SENTINEL = -1.0e4


def gated_delta_product_mtp(
    q: torch.Tensor,  # [B, T,      num_q_heads, K]
    k: torch.Tensor,  # [B, T, n_h, num_k_heads, K]
    v: torch.Tensor,  # [B, T, n_h, num_v_heads, V]
    initial_state: torch.Tensor,  # [pool_size, HV, V, K] fp32 -- the state POOL
    initial_state_indices: torch.Tensor,  # [B] read slot per batch row
    A_log: torch.Tensor,  # [HV]
    a: torch.Tensor,  # [B, T,      HV]  decay logits, ONE per real token
    dt_bias: torch.Tensor,  # [HV]
    b: torch.Tensor,  # [B, T, n_h, HV]  update-gate logits, per householder
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,  # [B, T, HV, V]
    ssm_state_indices: Optional[torch.Tensor] = None,  # [B, T] per-token scatter
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,  # [B]
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated DeltaProduct decode / MTP.

    GDP decode is :func:`flashinfer.gdn_decode.gated_delta_rule_mtp` with
    ``T -> T * num_householder``: one real token becomes ``n_h`` micro-steps.
    With speculative decoding on top, ``T`` is already ``num_spec + 1``, so the
    expanded axis is ``n_h * (num_spec + 1)``.

    Parameters
    ----------
    k, v, b : torch.Tensor
        Carry a householder axis at dim 2. ``q`` and ``a`` do not: one query and
        one gate per REAL token.
    ssm_state_indices : torch.Tensor, optional
        ``[B, T]`` int32, one pool slot per REAL token, as for GDN MTP.

    Returns
    -------
    ``(output, initial_state)``, matching ``gated_delta_rule_mtp``. ``output``
    is ``[B, T, HV, V]`` -- one row per REAL token.
    """
    if k.dim() != 5 or v.dim() != 5:
        raise ValueError(
            f"k/v must carry a householder axis [B, T, n_h, H, D]; "
            f"got k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}"
        )
    num_householder = k.size(2)
    if v.size(2) != num_householder:
        raise ValueError(
            f"k/v householder counts differ: {num_householder} vs {v.size(2)}"
        )
    if b.dim() != 4 or b.size(2) != num_householder:
        raise ValueError(
            f"b must be [B, T, n_h, HV] with n_h={num_householder}; "
            f"got {tuple(b.shape)}"
        )
    if a.dim() != 3:
        raise ValueError(
            f"a is one decay logit per REAL token, expected [B, T, HV]; "
            f"got {tuple(a.shape)}"
        )

    from .gdn_decode import gated_delta_rule_mtp

    # n_h == 1 is plain GDN MTP. Delegate so this path stays bit-identical.
    if num_householder == 1:
        return gated_delta_rule_mtp(
            q,
            k.squeeze(2),
            v.squeeze(2),
            initial_state,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b.squeeze(2),
            scale=scale,
            output=output,
            ssm_state_indices=ssm_state_indices,
            disable_state_update=disable_state_update,
            use_qk_l2norm=use_qk_l2norm,
            output_state_indices=output_state_indices,
        )

    # GDP = GDN with a sequence n_h times longer
    k = torch.flatten(k, start_dim=1, end_dim=2)
    v = torch.flatten(v, start_dim=1, end_dim=2)
    b = torch.flatten(b, start_dim=1, end_dim=2)

    if initial_state_indices.data_ptr() % 16 != 0:
        # GDN kernel requires 16-byte alignment on the index tensor
        initial_state_indices = initial_state_indices.clone()

    return gated_delta_rule_mtp(
        q,
        k,
        v,
        initial_state,
        initial_state_indices,
        A_log,
        a,
        dt_bias,
        b,
        scale=scale,
        output=output,
        ssm_state_indices=ssm_state_indices,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
        output_state_indices=output_state_indices,
        num_householder=num_householder,
    )
