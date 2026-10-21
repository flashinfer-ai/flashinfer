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

Cake Gated DeltaProduct (GDP) decode and MTP verification for Nemotron 4
(SM100 / SM103).

GDP (arXiv:2502.10297) generalizes the gated delta rule from one Householder
update per token to ``n = num_householder`` of them, with one per-head scalar
decay per token: the decay acts before the token's updates and the readout
follows the last one.  Per batch row and head, over the FP32 state
``S [V = 64, K = 128]`` of the row's pool slot, for each real token ``t`` and
its sub-steps ``j = 0 .. n - 1``::

    alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias))      # softplus threshold 20
    beta_tj = sigmoid(b_tj)
    S = alpha_t * S
    for j:  kq = l2norm(k_tj);  u = S kq;  S += (beta_tj * (v_tj - u)) (x) kq
    out_t = scale * S l2norm(q_t)

``num_householder == 1`` is the gated delta rule of
:func:`flashinfer.gated_delta_rule_mtp` at ``K = 128``, ``V = 64``.  The
state pool uses FlashInfer's K-last layout ``[pool_size, HV, V, K]``; batch
rows read the slot ``initial_state_indices[b]`` and write the slot
``output_state_indices[b]`` (the read slot by default) after their last
token, or scatter the state after every token to ``ssm_state_indices[b, t]``.

Usage::

    out, pool = cake_gated_delta_product_mtp(
        q, k, v, pool, slots, A_log, a, dt_bias, b, disable_state_update=False
    )
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_gdp_backend as _backend
from .api_logging import flashinfer_api
from .cake_gdp_backend import (  # noqa: F401
    ARCHES,
    HEAD_SIZE_K,
    HEAD_SIZE_V,
    NUM_HOUSEHOLDER,
    SUPPORTED_COMPUTE_CAPABILITIES,
    is_cake_gdp_supported,
    route_key,
)

# The stable entry point is a real ``def`` here (public-API tooling reads the
# decorated signatures of this module); its body lives in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
def cake_gated_delta_product_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Gated DeltaProduct decode / MTP verification over an FP32 state pool.

    One fused kernel per call: for every batch row and head the state is read
    from its pool slot, decayed once per token, updated by the token's
    ``num_householder`` Householder sub-steps, read out at the real token and,
    after the last token, written back to the row's write slot.  The gate is
    fused: ``alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias))`` per token
    (softplus threshold 20) and ``beta_tj = sigmoid(b_tj)`` per sub-step.  The
    parameters are those of :func:`flashinfer.gated_delta_rule_mtp`, with
    ``k``, ``v`` and ``b`` carrying the Householder axis.

    Parameters
    ----------
    q : torch.Tensor
        Queries ``[B, T, H, 128]``, bfloat16 or float16, one per real token.
        ``q`` and ``k`` views are read in place when their last dimension is
        dense and the view is 8-byte aligned (base address and outer strides
        multiples of four elements); other layouts are refused.  float16
        inputs are converted to bfloat16 for the kernel.
    k : torch.Tensor
        Keys ``[B, T, num_householder, H, 128]`` in ``q.dtype``: the
        ``num_householder`` (``1`` to ``4``) sub-steps of token ``t`` are
        ``k[:, t, 0 .. n - 1]``.
    v : torch.Tensor
        Values ``[B, T, num_householder, HV, 64]`` in ``q.dtype``.  ``HV``
        must equal ``H`` (Nemotron 4 has no GQA / GVA).
    initial_state : torch.Tensor
        FP32 state pool ``[pool_size, HV, 64, 128]`` (K-last), read and
        written in place.  Its rows must be dense, its strides multiples of
        four elements and its base address 16-byte aligned; a non-contiguous
        pool with these properties (padding between heads or slots) is used
        as is.
    initial_state_indices : torch.Tensor
        Read slots ``[B]`` (int32 or int64).  A negative entry marks a padding
        row: the kernel skips it entirely -- no state read, no recurrence, no
        output store, no writeback and no scatter -- so its output row keeps
        the value of the caller's ``output`` (zero when ``output`` is
        ``None``).
    A_log : torch.Tensor
        Log decay parameter ``[HV]``, float32 (read in place).
    a : torch.Tensor
        Input-dependent decay ``[B, T, HV]`` in ``q.dtype``, one per real
        token.
    dt_bias : torch.Tensor
        Decay bias ``[HV]``.  A float32 vector is read in place; a bfloat16
        or float16 vector is converted once per call, which adds one small
        launch -- keep the parameter in float32 for the single-launch path.
    b : torch.Tensor
        Update-gate inputs ``[B, T, num_householder, HV]`` in ``q.dtype``, one
        per sub-step.
    scale : float, optional
        Query scale; ``1 / sqrt(128)`` when ``None``.  Any other value,
        ``0.0`` included, is applied as given.
    output : torch.Tensor, optional
        Pre-allocated output ``[B, T, HV, 64]``, bfloat16 or float16.  A
        bfloat16 buffer whose last dimension is dense and whose base address
        and strides are 8-byte aligned is written directly; any other buffer
        receives a copy after the launch (its values in skipped rows are
        preserved through a bfloat16 round trip).  Allocated as zeros in
        ``q.dtype`` when ``None``.
    ssm_state_indices : torch.Tensor, optional
        Per-token scatter slots ``[B, T]``, int32.  When given, the state
        after every real token (its ``num_householder`` sub-steps completed)
        is written to ``initial_state[ssm_state_indices[b, t]]`` (a negative
        entry skips that write) and the end-of-sequence writeback to the
        write slot is skipped, as in :func:`flashinfer.gated_delta_rule_mtp`.
        Requires ``disable_state_update=False``; ``T == 1`` is accepted.
    disable_state_update : bool, optional
        ``True`` leaves the pool unchanged (verification without commit).
        ``None`` (the default) means ``False``: the state is written back.
    use_qk_l2norm : bool
        Apply the L2 normalization (``x * rsqrt(sum(x^2) + 1e-6)``) to ``q``
        and ``k`` inside the kernel.  Default ``True``.
    output_state_indices : torch.Tensor, optional
        Write slots ``[B]`` (int32 or int64); ``initial_state_indices`` when
        ``None``.  A negative entry skips the writeback of that row only (its
        output is still computed and stored).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(output, initial_state)``: the output ``[B, T, HV, 64]`` in the
        requested dtype and the pool with the writebacks / scatters applied
        (unchanged when ``disable_state_update=True``).

    Notes
    -----
    - Requires an SM100 or SM103 device (``sm_100a`` / ``sm_103a`` programs).
    - ``T`` is a runtime value up to the exported maximum (16); ``T`` of 1, 4
      and 8 with ``num_householder == 3`` run programs compiled for that token
      count, selected by ``B x HV``.
    - The call reads no tensor contents on the host and synchronizes nothing.
      With bfloat16 activations, float32 ``A_log`` / ``dt_bias``, int32
      indices and a bfloat16 ``output`` it launches the kernel alone; it
      allocates only the output (when ``output`` is ``None``) and, when
      ``ssm_state_indices`` is ``None``, an uninitialized ``[B, T]`` int32
      scatter table the kernel never reads.  It is CUDA-graph safe.
    - Every batch row is independent of the others and three launches on the
      same inputs are bitwise identical.
    """

    return _backend.cake_gated_delta_product_mtp(
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
    )


__all__ = [
    "ARCHES",
    "HEAD_SIZE_K",
    "HEAD_SIZE_V",
    "NUM_HOUSEHOLDER",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "cake_gated_delta_product_mtp",
    "is_cake_gdp_supported",
    "route_key",
]
