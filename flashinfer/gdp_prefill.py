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

from typing import Literal, Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .trace.templates.gdp import gdp_prefill_trace

_CU_SEQLENS_DTYPES = (torch.int32, torch.int64)


@flashinfer_api(trace=gdp_prefill_trace)
def chunk_gated_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    num_householder: int = 1,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    *,
    backend: Literal["auto", "cudnn", "cake"] = "auto",
    state_indices: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated DeltaProduct (GDP) attention for prefill.

    GDP generalizes :func:`flashinfer.chunk_gated_delta_rule` from one
    beta-gated Householder update per token to ``n = num_householder`` of
    them, with one per-head scalar decay per token: the decay acts before the
    token's updates and the readout follows the last one. Per real token, with
    sub-token rows ``t*n .. t*n + n - 1`` of ``k`` / ``v`` / ``beta``,

    .. math::

        S &\mathrel{*}= \alpha_t \\
        v^{new}_{t,j} &= \beta_{t,j} (v_{t,j} - S k_{t,j}),
            \quad S \mathrel{+}= v^{new}_{t,j} \otimes k_{t,j},
            \quad j = 0 \ldots n - 1 \\
        o_t &= \mathrm{scale} \cdot S q_t

    ``num_householder == 1`` is exactly
    :func:`flashinfer.chunk_gated_delta_rule`.

    Only varlen (packed / THD) input is accepted, matching
    :func:`flashinfer.chunk_gated_delta_rule`: pass ``cu_seqlens`` over the
    real tokens and lay the tokens of all sequences end to end.

    Parameters
    ----------
    q : torch.Tensor
        ``[total_seq_len, num_q_heads, head_size]``, float16 or bfloat16, at
        real-token rows.  Strides are honored, so only the innermost dimension
        has to be contiguous.
    k, v : torch.Tensor
        ``[total_seq_len * num_householder, num_k_heads / num_v_heads,
        head_size]``, float16 or bfloat16, on the expanded sub-token timeline:
        the ``n`` Householder updates of token ``t`` occupy rows
        ``t*n .. t*n + n - 1``.  ``num_k_heads`` must equal ``num_q_heads`` or
        ``num_v_heads``.
    g : torch.Tensor, optional
        Per-head forget gate in linear space (``alpha``, elementwise in
        ``(0, 1]``), shape ``[total_seq_len, num_sab_heads]`` at real-token
        rows, at float32, bfloat16 or float16.  All-ones (no decay) when
        ``None``.
    beta : torch.Tensor, optional
        Per-head, per-Householder update gate ``[total_seq_len *
        num_householder, num_sab_heads]``, post-sigmoid, in float32 or
        ``q.dtype``.  All-ones when ``None``.
    num_householder : int
        Householder updates per token (``n >= 1``).
    scale : float, optional
        Query scale; ``1 / sqrt(head_size)`` when ``None`` or ``0.0``, matching
        :func:`flashinfer.chunk_gated_delta_rule`.
    initial_state : torch.Tensor, optional
        Incoming recurrent state ``[num_seqs, num_sab_heads, head_size,
        head_size]``, V-major (``[..., V, K]``), float32 or bfloat16 (also
        float16 on the Cake backend on SM100 / SM103).  A zero state when
        ``None``.
    output_final_state : bool
        Return the outgoing recurrent state alongside the output.
    cu_seqlens : torch.Tensor
        Cumulative real-token sequence lengths ``[num_seqs + 1]``, int32 or
        int64.  Required.
    use_qk_l2norm_in_kernel : bool
        Fuse the q/k L2 normalization into the kernel instead of expecting
        pre-normalized inputs.
    output : torch.Tensor, optional
        Pre-allocated output ``[total_seq_len, num_o_heads, head_size]`` at
        real-token rows, written in place.  Allocated internally when ``None``.
    output_state : torch.Tensor, optional
        Pre-allocated final-state buffer, written in place.  Allocated
        internally when ``None`` and ``output_final_state`` is set.  It must
        not alias ``initial_state`` without ``state_indices``; the kernel splits one sequence across
        CTAs, so the CTA reading the incoming state would race the one writing
        the outgoing state.  Its dtype is the state dtype of the call: when
        ``initial_state`` is given too, both share it, except that a bfloat16
        ``initial_state`` with a float32 ``output_state`` (or one allocated
        internally, always float32) is accepted and converted.
    backend : Literal["auto", "cudnn", "cake"], optional
        ``"auto"`` (default) and ``"cudnn"`` run cuDNN's fused SM100
        linear-attention engine through
        :func:`flashinfer.cudnn.cudnn_chunk_gated_delta_product`.  ``"cake"``
        runs the Cake GDP prefill kernel
        (:func:`flashinfer.cake_gdp_backend.cake_chunk_gated_delta_product`):
        ``head_size == 128`` with a 64-wide value / state row, one head count
        for q, k and v, ``num_householder`` in 1..4 on an SM90 (H100 / H200)
        device and 3 on an SM100 (B200 / GB200) or SM103 (B300 / GB300) device; the argument
        contract is this function's.  On SM90 ``initial_state`` /
        ``output_state`` are float32 (a bfloat16 per-sequence
        ``initial_state`` is converted) and pools contiguous.  On SM100 /
        SM103 the state is float32, bfloat16 or float16, one dtype for the
        initial and the output state (the bfloat16-to-float32 conversion
        above aside): a bfloat16 / float16 state is read widened exactly and
        its final value rounded once when it is stored, to nearest even, or
        stochastically with ``rand_seed``; pools and dense bfloat16 / float16
        state tensors may pad the slot stride (see ``state_indices``), a dense
        float32 ``output_state`` must be contiguous (a padded dense float32
        ``initial_state`` is copied once).  One Cake kernel launch per call;
        the split programs add a workspace clear and the item-table upload
        (from pinned host memory into a cached device buffer, so the call
        captures into a CUDA graph when ``cu_seqlens_cpu`` is given).  On
        SM100 / SM103 one launch serves a mixed batch: the long sequences are
        planned as if alone and the short ones run on the least-loaded CTAs
        after them.  On SM90 one launch serves a mixed batch too: a long
        sequence that fills the SMs (when that is the lighter launch) is cut
        into one slot per SM, each piece after a cut starting from a zero
        state a warm-up before it (a cut whose gates do not decay by
        2^-14.43 within the warm-up is run through), and the short sequences
        run in the same launch; any other batch runs one kernel body chosen
        by a modeled kernel time.  The results agree with any other batch
        composition within rounding and the dropped warm-up decay.  With
        ``cu_seqlens_cpu`` given, no device-to-host synchronization is issued
        (the split rule reads the host copy); without it the sequence offsets
        are copied to the host once.  On SM90 ``g=None`` runs the kernel's
        unit-gate program and ``beta=None`` the unit-beta program, from a zero
        state (on SM100 / SM103 unit tensors are filled in); ``g=None``
        together with ``beta=None``, on SM90 a unit-gate / unit-beta call with
        an ``initial_state`` or a ``state_indices`` pool, on SM90 a bfloat16 /
        float16 state, ``output_state_indices`` or ``rand_seed``, and a device
        without an exported route raise ``ValueError`` naming what is
        exported.  Only ``backend="cake"`` takes ``output_state_indices`` and
        ``rand_seed``; the other backends raise ``ValueError`` when they are
        given.
    cu_seqlens_cpu : torch.Tensor, optional
        The host copy of ``cu_seqlens`` (a CPU int32 / int64 1-D tensor of ``N +
        1`` entries, as vLLM keeps ``query_start_loc_cpu``).  The Cake backend
        reads the sequence lengths on the host for the kernel's split rule of
        record (a long sequence is cut into items for the gate-saturation split
        program; a batch the rule leaves whole runs the exact program; the two
        agree bitwise).  When it is None the Cake backend reads
        ``cu_seqlens.cpu()``: one device-to-host synchronization, which CUDA
        Graph capture does not allow, so a graph-captured caller passes it.  It
        is checked for device, rank, length and dtype, never compared with
        ``cu_seqlens`` value by value.  The cuDNN backends accept and ignore it.
        The Cake backend plans the launch from these host lengths, so a CUDA
        graph it captures is valid only for the batch it captured (the
        sequence offsets, ``N`` and ``T``): a batch with other lengths needs a
        new capture or an eager call.  On SM100 / SM103 the kernel compares the
        planned offsets with ``cu_seqlens`` on the device; a replay over changed
        offsets computes nothing and writes NaN to ``output[0, 0, :2]`` (and to
        ``output_state[0, 0, 0, 0]`` when a dense float32 final state is
        stored; a pool or a bfloat16 / float16 state gets no marker, so the
        state is left as it was), so the failure is visible.  On SM90 the
        split program has the same check; the other SM90 programs have none,
        and the result of their replay over changed offsets is unspecified.
        The cuDNN backends read the lengths on the device and accept such a
        replay.

    state_indices : torch.Tensor, optional
        Contiguous int32 ``[num_seqs]`` slots in ``initial_state`` and
        ``output_state`` pools, requiring cuDNN frontend 1.31+. Pools have
        shape ``[N_pool, H, V, K]``, dense inner rows, and a 16-byte aligned
        slot stride. Slot values must be unique and in range.
        When ``output_final_state=True``, an explicit output pool is required;
        it may be the identical input view or disjoint memory. The returned
        state is the whole output pool, with unselected slots unchanged.
        When false, neither pool is modified. Values are not read on the CPU.
        The Cake backend also takes int64 slots (converted once; an entry
        outside the int32 range counts as a slot outside the pool) and slots
        outside the pool.  On SM100 / SM103 its pools are float32, bfloat16 or
        float16 and may pad the slot stride (a multiple of 4 elements, rows
        dense); the read slot is checked against ``initial_state`` and the
        write slot against ``output_state``: a read slot outside
        ``initial_state`` (such as -1) starts that sequence from a zero state,
        and a write slot outside ``output_state`` stores nothing.  On SM90 its
        pools are float32 and contiguous, and a slot outside either pool
        starts that sequence from a zero state and stores nothing.
    output_state_indices : torch.Tensor, optional
        ``backend="cake"`` on SM100 / SM103 only: int32 / int64
        ``[num_seqs]`` slots of ``output_state`` the final states are written
        to (a slot outside the pool, such as -1, stores nothing).
        ``state_indices`` stays the read slots.  Defaults to
        ``state_indices``; requires it.  A write slot must not be another
        sequence's read or write slot (not checked).
    rand_seed : torch.Tensor, optional
        ``backend="cake"`` with a bfloat16 / float16 state only: a
        one-element int64 tensor on the state's device, read by the kernel at
        each launch (so a captured graph rounds with the seed it holds at
        replay).  With it, the final state is rounded stochastically: each
        element takes 16 (bfloat16) or 13 (float16) random bits of a
        Philox-4x32 draw keyed by the seed, through the hardware ``cvt.rs``
        instruction.  One draw covers columns ``k .. k+3`` and
        ``k+64 .. k+67`` (``k`` a multiple of 4 below 64) of one state row,
        and its counter is the element offset of column ``k`` of that row
        from the base of ``initial_state`` at the sequence's read slot when
        the call reads an initial state and the slot is inside the pool,
        else from the base of ``output_state`` at its write slot.  The
        counter carries no step, layer or token term: advance the seed per
        step and use one per layer.  Sequences that read one slot get the
        same random bits.  Without a seed the final state is rounded to
        nearest even.  A seed with a float32 state raises ``ValueError``.
    philox_rounds : int
        Philox rounds of a stochastic rounding, 5 to 16 (default 10); read
        only with ``rand_seed``.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``output``, or ``(output, final_state)`` when ``output_final_state``.

    Note
    ----
    Requires an SM100-family (Blackwell) device and cudnn-frontend 1.29+ with
    the ``cutedsl`` extra (``pip install 'nvidia-cudnn-frontend[cutedsl]'``).
    Everything finer -- head dims, input dtypes, head-count relations -- is the
    engine's call: a graph it cannot serve is declined by cuDNN (the per-engine
    reason lands in the frontend's log).
    """
    if backend not in ("auto", "cudnn", "cake"):
        raise ValueError(f'backend must be "auto", "cudnn" or "cake", got {backend!r}')
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for varlen mode")
    if cu_seqlens.dtype not in _CU_SEQLENS_DTYPES:
        raise ValueError(
            f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
        )

    if backend != "cake" and (
        output_state_indices is not None or rand_seed is not None
    ):
        raise ValueError(
            f"{'output_state_indices' if output_state_indices is not None else 'rand_seed'} "
            f'needs backend="cake"; the {backend!r} backend does not take it'
        )

    if backend == "cake":
        from .cake_gdp_backend import cake_chunk_gated_delta_product

        return cake_chunk_gated_delta_product(
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
            state_indices=state_indices,
            cu_seqlens_cpu=cu_seqlens_cpu,
            output_state_indices=output_state_indices,
            rand_seed=rand_seed,
            philox_rounds=philox_rounds,
        )

    from .cudnn import cudnn_chunk_gated_delta_product

    return cudnn_chunk_gated_delta_product(
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
        state_indices=state_indices,
    )
