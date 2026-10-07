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

Gated DeltaProduct (arXiv:2502.10297) -- decode API layer.
"""

from typing import Optional, Tuple

import torch

from .gdn_decode import gated_delta_rule_mtp


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
    # expansion scratch -- see chunk_gated_delta_product
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated DeltaProduct decode / MTP.

    GDP decode is :func:`flashinfer.gdn_decode.gated_delta_rule_mtp` with
    ``T -> T * num_householder``: one real token becomes ``n_h`` micro-steps.
    With speculative decoding on top, ``T`` is already ``num_spec + 1``, so the
    expanded axis is ``n_h * (num_spec + 1)``.

    **The gate is fused**, unlike the prefill kernel. Prefill takes ``g``
    directly; here the kernel derives alpha from ``A_log``/``a``/``dt_bias``.
    Neutralising the gate on micro-steps ``1..n_h-1`` therefore happens through
    ``a``, using :data:`GATE_NEUTRAL_A_SENTINEL` -- not by writing 1.0 anywhere.

    Parameters
    ----------
    k, v, b : torch.Tensor
        Carry a householder axis at dim 2. ``q`` and ``a`` do not: one query and
        one gate per REAL token.
    ssm_state_indices : torch.Tensor, optional
        ``[B, T]`` int32, one pool slot per REAL token, as for GDN MTP. The
        wrapper expands this to ``[B, T*n_h]``, giving micro-steps ``1..n_h-1``
        a negative slot -- which the kernel's scatter skips -- and routing only
        the last micro-step of each token to the caller's slot.
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
        # the kernel requires 16-byte alignment on the index tensor
        initial_state_indices = initial_state_indices.clone()

    # k/v/b already carry the householder axis next to the token axis, so the
    # micro-step view is a free reshape.  q, a and the output stay one row per
    # REAL token: the kernel indexes them directly, so GDP allocates no
    # expansion scratch at all.
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
