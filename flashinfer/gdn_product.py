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
    # expansion scratch -- required for cudagraph capture, else allocated here
    expanded_q: Optional[torch.Tensor] = None,  # [T*n_h, num_q_heads,  D]
    expanded_g: Optional[torch.Tensor] = None,  # [T*n_h, num_sab_heads]
    expanded_output: Optional[torch.Tensor] = None,  # [T*n_h, num_o_heads,  D]
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
    expanded_q : torch.Tensor, optional
        Scratch for the expanded query,
        ``[total_seq_len * num_householder, num_q_heads, head_size]``, dtype of
        ``q``.
    expanded_g : torch.Tensor, optional
        Scratch for the expanded gate,
        ``[total_seq_len * num_householder, num_sab_heads]``. **Must be
        float32** -- ``g`` and ``beta`` are always fp32 in this API even when
        ``q``/``k``/``v`` are fp16/bf16, and a strided assignment into a
        half-precision buffer downcasts silently.
        Must be pre-filled with ``1.0``: only rows ``[0 :: n_h]`` are written
        (the per-token gate), and the remaining micro-steps rely on the
        multiplicative neutral value already being in place.
    expanded_output : torch.Tensor, optional
        Scratch for the kernel's output,
        ``[total_seq_len * num_householder, num_o_heads, head_size]`` where
        ``num_o_heads = max(num_q_heads, num_v_heads)``. Note this is **not**
        ``num_q_heads`` -- under GVA the output is wider than the query.
        Takes ``output``'s dtype when ``output`` is supplied, else ``q``'s, so
        the final strided copy never silently casts.

    Returns
    -------
    Same contract as ``chunk_gated_delta_rule``: ``output`` when
    ``output_final_state`` is False, else ``(output, final_state)``. ``output``
    has one row per REAL token, ``[total_seq_len, num_o_heads, head_size]``.
    When ``output`` is supplied it is written in place and returned; otherwise
    a freshly allocated tensor is returned.
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

    # GDP = GDN with a sequence n_h times longer
    k = torch.flatten(k, start_dim=0, end_dim=1)
    v = torch.flatten(v, start_dim=0, end_dim=1)
    if beta is not None:
        beta = torch.flatten(beta, start_dim=0, end_dim=1)

    if expanded_q is None:
        expanded_q = torch.empty(
            q.size(0) * num_householder, *q.shape[1:], dtype=q.dtype, device=q.device
        )
    elif expanded_q.shape != (q.size(0) * num_householder, *q.shape[1:]):
        raise ValueError("expanded_q shape must be [T*n_h, num_q_heads,  D]")
    elif expanded_q.dtype != q.dtype:
        raise ValueError(
            f"expanded_q.dtype and q.dtype must match, got {expanded_q.dtype} != {q.dtype}"
        )

    expanded_q[num_householder - 1 :: num_householder] = q

    if g is not None:
        if g.dtype != torch.float32:
            raise ValueError(
                f"g must be float32 (g/beta are always fp32 in this API, "
                f"unlike q/k/v); got {g.dtype}"
            )

        if expanded_g is None:
            expanded_g = torch.ones(
                g.size(0) * num_householder,
                *g.shape[1:],
                dtype=g.dtype,
                device=g.device,
            )
        elif expanded_g.shape != (g.size(0) * num_householder, *g.shape[1:]):
            raise ValueError("expanded_g shape must be [T*n_h, num_sab_heads]")
        elif expanded_g.dtype != torch.float32:
            raise ValueError(
                f"expanded_g must be float32 (g/beta are always fp32 in this API, "
                f"unlike q/k/v); got {expanded_g.dtype}"
            )
        expanded_g[::num_householder] = g
    else:
        expanded_g = None

    if expanded_output is None:
        expanded_output = torch.empty(
            expanded_q.size(0),
            max(q.size(1), v.size(1)),
            q.size(2),
            dtype=output.dtype if output is not None else q.dtype,
            device=output.device if output is not None else q.device,
        )
    elif expanded_output.shape != (
        expanded_q.size(0),
        max(q.size(1), v.size(1)),
        q.size(2),
    ):
        raise ValueError("expanded_output shape must be [T*n_h, num_o_heads,  D]")

    out = chunk_gated_delta_rule(
        expanded_q,
        k,
        v,
        expanded_g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens * num_householder,
        use_qk_l2norm_in_kernel,
        output=expanded_output,
        output_state=output_state,
        state_indices=state_indices,
    )

    if output is not None:
        output[:] = expanded_output[num_householder - 1 :: num_householder]
    else:
        output = expanded_output[num_householder - 1 :: num_householder].clone()

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
    scratch_state_indices: Optional[torch.Tensor] = None,  # [B] throwaway slots
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,  # [B]
    # expansion scratch -- see chunk_gated_delta_product
    expanded_q: Optional[torch.Tensor] = None,  # [B, T*n_h, num_q_heads, K]
    expanded_a: Optional[torch.Tensor] = None,  # [B, T*n_h, HV]
    expanded_output: Optional[torch.Tensor] = None,  # [B, T*n_h, HV, V]
    expanded_ssm_state_indices: Optional[torch.Tensor] = None,  # [B, T*n_h] int32
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated DeltaProduct decode / MTP.

    GDP decode is :func:`flashinfer.gdn_decode.gated_delta_rule_mtp` with
    ``T -> T * num_householder``: one real token becomes ``n_h`` micro-steps.
    With speculative decoding on top, ``T`` is already ``num_spec + 1``, so the
    expanded axis is ``n_h * (num_spec + 1)``.

    Two things differ from the prefill wrapper, both because the decode kernel
    is less of a blank slate:

    1. **The gate is fused.** Prefill takes ``g`` directly; here the kernel
       derives alpha from ``A_log``/``a``/``dt_bias``. Neutralising the gate on
       micro-steps ``1..n_h-1`` therefore happens through ``a``, using
       :data:`GATE_NEUTRAL_A_SENTINEL` -- not by writing 1.0 anywhere.
    2. **The per-token state scatter is unguarded.** ``gated_delta_rule_mtp``
       writes ``h_{t+1}`` to ``initial_state[ssm_state_indices[i, t]]`` for
       every ``t``, with no "skip this one" sentinel (unlike FLA, whose kernel
       guards on a non-positive slot id). Intermediate micro-steps must
       therefore be pointed at a **throwaway pool row** -- one shared row for
       the whole batch is enough; see ``scratch_state_indices``.

    Parameters
    ----------
    k, v, b : torch.Tensor
        Carry a householder axis at dim 2. ``q`` and ``a`` do not: one query and
        one gate per REAL token.
    ssm_state_indices : torch.Tensor, optional
        ``[B, T]`` int32, one pool slot per REAL token, as for GDN MTP. The
        wrapper expands this to ``[B, T*n_h]``, routing micro-steps
        ``1..n_h-1`` to the scratch rows and only the last micro-step of each
        token to the caller's slot.
    scratch_state_indices : torch.Tensor, optional
        ``[B]`` or ``[1]`` int32. Pool rows that absorb the intermediate
        micro-steps' state writes and are never read. Required when
        ``ssm_state_indices`` is given and ``n_h > 1``.

        The only constraint is that these rows are **disjoint from every live
        slot** (anything in ``initial_state_indices`` / ``ssm_state_indices`` /
        ``output_state_indices``). They need *not* be distinct from one another:
        a single shared row for the whole batch is correct, because the kernel
        reads the pool exactly once -- before the recurrence, through
        ``initial_state_indices`` -- and the per-token scatter is write-only
        thereafter. Concurrent stores to an address nobody reads are benign.

        Prefer one shared row. A per-row scratch costs ``B`` state slots
        (``HV * V * K * 4`` bytes each -- ~2 MiB at HV=32, K=V=128, so ~512 MiB
        at B=256) to buy only better write locality, for traffic that is pure
        waste either way. Phase 2's scatter guard removes these writes entirely.
    expanded_* : torch.Tensor, optional
        Scratch for the expansion; required for CUDA graph capture. See
        :func:`chunk_gated_delta_product` for why.

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

    if expanded_q is None:
        expanded_q = torch.empty(
            q.size(0),
            q.size(1) * num_householder,
            *q.shape[2:],
            dtype=q.dtype,
            device=q.device,
        )
    elif expanded_q.shape != (q.size(0), q.size(1) * num_householder, *q.shape[2:]):
        raise ValueError("expanded_q shape must be [B, T*n_h, num_q_heads,  D]")
    elif expanded_q.dtype != q.dtype:
        raise ValueError(
            f"expanded_q.dtype and q.dtype must match, got {expanded_q.dtype} != {q.dtype}"
        )

    expanded_q[:, num_householder - 1 :: num_householder] = q

    if expanded_a is None:
        expanded_a = torch.full(
            (a.size(0), a.size(1) * num_householder, *a.shape[2:]),
            GATE_NEUTRAL_A_SENTINEL,
            dtype=a.dtype,
            device=a.device,
        )
    elif expanded_a.shape != (a.size(0), a.size(1) * num_householder, *a.shape[2:]):
        raise ValueError("expanded_a shape must be [B, T*n_h, num_sab_heads]")
    elif expanded_a.dtype != a.dtype:
        raise ValueError(f"expanded_a dtype must match a dtype ({a.dtype})")
    expanded_a[:, ::num_householder] = a

    if expanded_output is None:
        expanded_output = torch.empty(
            expanded_q.size(0),
            expanded_q.size(1),
            max(q.size(2), v.size(2)),
            q.size(3),
            # kernel hardcodes bf16...
            dtype=torch.bfloat16,
            device=output.device if output is not None else q.device,
        )
    elif expanded_output.shape != (
        expanded_q.size(0),
        expanded_q.size(1),
        max(q.size(2), v.size(2)),
        q.size(3),
    ):
        raise ValueError("expanded_output shape must be [B, T*n_h, num_o_heads,  D]")

    if ssm_state_indices is not None:
        if scratch_state_indices is None:
            raise ValueError(
                "scratch_state_indices are required if ssm_state_indices is given"
            )
        if expanded_ssm_state_indices is None:
            expanded_ssm_state_indices = torch.empty(
                ssm_state_indices.size(0),
                ssm_state_indices.size(1) * num_householder,
                dtype=ssm_state_indices.dtype,
                device=ssm_state_indices.device,
            )
        elif expanded_ssm_state_indices.shape != (
            ssm_state_indices.size(0),
            ssm_state_indices.size(1) * num_householder,
        ):
            raise ValueError("expanded_ssm_state_indices shape must be [B, T*n_h]")
        elif expanded_ssm_state_indices.dtype != ssm_state_indices.dtype:
            raise ValueError(
                f"expanded_ssm_state_indices dtype must be {ssm_state_indices.dtype}"
            )

        expanded_ssm_state_indices[:] = scratch_state_indices[:, None]
        expanded_ssm_state_indices[:, num_householder - 1 :: num_householder] = (
            ssm_state_indices
        )
    else:
        expanded_ssm_state_indices = None

    _, state = gated_delta_rule_mtp(
        expanded_q,
        k,
        v,
        initial_state,
        initial_state_indices,
        A_log,
        expanded_a,
        dt_bias,
        b,
        scale=scale,
        output=expanded_output,
        ssm_state_indices=expanded_ssm_state_indices,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
        output_state_indices=output_state_indices,
    )

    if output is not None:
        output[:] = expanded_output[:, num_householder - 1 :: num_householder]
    else:
        output = expanded_output[:, num_householder - 1 :: num_householder].clone()

    return output.to(q.dtype), state
