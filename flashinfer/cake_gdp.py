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

Cake Gated DeltaProduct (GDP) decode and MTP verification
(SM90 / SM100 / SM103 / SM107).

GDP (arXiv:2502.10297) generalizes the gated delta rule from one Householder
update per token to ``n = num_householder`` of them, with one per-head scalar
decay per token: the decay acts before the token's updates and the readout
follows the last one.  Per batch row and head, over the state
``S [V = 64, K = 128]`` of the row's pool slot (kept in FP32 within a call),
for each real token ``t`` and its sub-steps ``j = 0 .. n - 1``::

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
The pool is FP32, or FP16 on SM100 / SM103 / SM107: an FP16 pool is widened to
FP32 on load and each store rounds the FP32 state to FP16, to nearest even or,
with a ``rand_seed``, stochastically (Philox random bits and the hardware
``cvt.rs`` conversion).

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
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Gated DeltaProduct decode / MTP verification over an FP32 or FP16 state pool.

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
        must equal ``H`` (no GQA / GVA).
    initial_state : torch.Tensor
        State pool ``[pool_size, HV, 64, 128]`` (K-last), float32 or (SM100 /
        SM103 / SM107 only) float16, read and written in place.  Its rows must be
        dense, its strides multiples of four elements and its base address
        16-byte (float32) or 8-byte (float16) aligned; a non-contiguous pool
        with these properties (padding between heads or slots) is used as
        is.  The recurrence runs in FP32 either way: a float16 pool is
        widened exactly on load, and every store (the writeback and each
        ``ssm_state_indices`` snapshot) rounds the FP32 state to float16 --
        to nearest even without ``rand_seed``, stochastically with it.  A
        value that rounds past the largest float16 (65504) is stored as an
        infinity.
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
        output is still computed and stored).  The write slot of a row that
        writes back must not be another row's read or write slot (see Notes).
    rand_seed : torch.Tensor, optional
        Stochastic rounding of a float16 pool: a one-element int64 tensor on
        the pool's device, read by the kernel on the device (a captured CUDA
        graph sees in-place updates).  It must not overlap ``initial_state``.
        Each stored element adds 13 random
        bits below its float16 precision and truncates.  The bits come from
        Philox-4x32 of ``philox_rounds`` rounds keyed by the seed's 64 bits.
        One draw serves the columns ``k .. k + 3`` and ``k + 64 .. k + 67``
        (``k`` a multiple of 4 below 64) of one state row; its counter is the
        element offset, from the base address of the ``initial_state`` view,
        of column ``k`` of that state row in the batch row's read slot,
        whichever slot the store writes.  So a fixed seed gives bitwise
        reproducible stores.  The counter holds no step or layer index:
        advance the seed every decode step and use a different one for every
        layer.  It holds no token or batch-row index either: the snapshots of
        one call, and the rows that read the same slot, share their random
        bits.  ``None`` (the default) rounds to nearest even.  A float32 pool
        takes no seed (``ValueError``).
    philox_rounds : int
        Philox rounds of the stochastic rounding, an ``int`` from 5 to 16
        (default 10); read only with ``rand_seed``.  Fewer than 5 rounds
        bias the low bits of neighbouring counters and are refused.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(output, initial_state)``: the output ``[B, T, HV, 64]`` in the
        requested dtype and the pool with the writebacks / scatters applied
        (unchanged when ``disable_state_update=True``).

    Notes
    -----
    - Requires an SM90, SM100, SM103 or SM107 device (``sm_90a`` /
      ``sm_100a`` / ``sm_103a`` / ``sm_107a`` programs); a float16 pool
      requires SM100, SM103 or SM107 (the hardware ``cvt.rs`` conversion) and
      is refused with ``ValueError`` on SM90.
    - ``T`` is a runtime value up to the exported maximum (16); ``T`` of 1 to 8
      and 16 with ``num_householder == 3`` run programs compiled for that token
      count, selected by ``B x HV``.
    - The call reads no tensor contents on the host and synchronizes nothing.
      With bfloat16 activations, float32 ``A_log`` / ``dt_bias``, int32
      indices and a bfloat16 ``output`` it launches the kernel alone; it
      allocates only the output (when ``output`` is ``None``), when
      ``ssm_state_indices`` is ``None`` an uninitialized ``[B, T]`` int32
      scatter table the kernel never reads, and, for a float16 pool without
      ``rand_seed`` whose storage offset is not a multiple of four elements,
      a one-element int64 seed placeholder the kernel never reads.  It is
      CUDA-graph safe.
    - Batch rows run concurrently in no fixed order.  A row writes its write
      slot (``output_state_indices[b]``, which defaults to
      ``initial_state_indices[b]``) or, with ``ssm_state_indices``, its
      scatter slots; no other row of the same call may read or write those
      slots, otherwise the stored states and outputs of the rows involved are
      undefined.  So when rows write back without ``output_state_indices``,
      the live read slots must be unique; with distinct write slots, with
      ``ssm_state_indices`` or with ``disable_state_update=True``, read slots
      may repeat (those rows start from the same state and share their random
      bits).  This is a caller precondition, not checked on the host (no
      synchronization).  Given it, a row's output and stores do not depend on
      the other rows, and repeated launches on the same inputs (the same seed
      included) are bitwise identical.  A float16 pool runs the twin of the
      program a float32 pool would run, with the same arithmetic: the output
      equals the float32 pool's bit for bit when both pools hold the same
      values.
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
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
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
