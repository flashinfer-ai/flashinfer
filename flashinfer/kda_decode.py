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
"""

"""
Kimi Delta Attention Decode - API Layer
=======================================

This file provides the public API for recurrent KDA decode operations.
Kernel implementations are in ``flashinfer.kda_kernels``; callers may
explicitly select Cake or use its narrow native auto-dispatch contract.
"""

from typing import Literal, Optional

import torch

from .api_logging import flashinfer_api
from .trace.templates.kda import (
    fused_kda_decode_trace,
    packed_kda_decode_trace,
    recurrent_kda_trace,
)

try:
    from .kda_kernels.fused_kda_decode import (
        run_fused_kda_decode as _run_fused_kda_decode,
    )

    _FUSED_KDA_DECODE_AVAILABLE = True
except (ImportError, RuntimeError):
    _run_fused_kda_decode = None
    _FUSED_KDA_DECODE_AVAILABLE = False

from .kda_kernels import run_packed_kda_decode as _run_packed_kda_decode
from .kda_kernels import run_recurrent_kda as _run_recurrent_kda

try:
    from .kda_kernels.kda_decode_wy_output_only import (
        kda_wy_output_only as _run_kda_output_only,
    )

    _KDA_OUTPUT_ONLY_AVAILABLE = True
except (ImportError, RuntimeError):
    _run_kda_output_only = None
    _KDA_OUTPUT_ONLY_AVAILABLE = False


# None when the CuTe DSL is missing or cannot target this device
# (see flashinfer/kda_kernels/__init__.py).
_RECURRENT_KDA_AVAILABLE = _run_recurrent_kda is not None


@flashinfer_api(trace=recurrent_kda_trace)
def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_spec_tokens: Optional[int] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
    *,
    disable_state_update: bool = False,
    correction_cache: Optional[torch.Tensor] = None,
    kg_cache: Optional[torch.Tensor] = None,
    backend: Literal["cute-dsl", "cake", "auto"] = "cute-dsl",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Recurrent KDA (Kimi Delta Attention) decode kernel.

    This public API supports the existing CuTe DSL implementation and an
    explicit exported Cake backend in
    ``flashinfer.kda_kernels.recurrent_kda``. It supports single-token decode,
    fused speculative decode, GQA, optional cu_seqlens packing, and the same
    gate modes as the selected backend implementation.

    Args:
        q (torch.Tensor):
            Current query of shape ``[B, 1, H, K]``, or ``[1, total_tokens, H, K]``
            when using ``cu_seqlens``. Must be bfloat16.
        k (torch.Tensor):
            Current key of shape ``[B, 1, H, K]``. Must be bfloat16.
        v (torch.Tensor):
            Current value of shape ``[B, 1, HV, V]``. Must be bfloat16.
            GQA is applied when ``HV != H``.
        g (torch.Tensor):
            Per-K-dimension gate of shape ``[B, 1, HV, K]``. Must be bfloat16.
            Log-space if pre-computed, raw input if ``use_gate_in_kernel=True``.
        beta (torch.Tensor):
            Delta-rule learning rate of shape ``[B, 1, HV]``. Must be bfloat16.
            Pre-sigmoided unless ``beta_is_logit=True``.
        A_log (Optional[torch.Tensor]):
            Log decay parameter of shape ``[H]``. Must be float32.
            Required when ``use_gate_in_kernel=True``.
        dt_bias (Optional[torch.Tensor]):
            Per-head-K decay bias of shape ``[H*K]``. Must be float32.
        scale (Optional[float]):
            Scale factor for queries. If ``None``, defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[N, HV, V, K]``. Must be bfloat16.
            If ``None``, zero-initialized. Updated in-place. For batched spec
            decode without ``cu_seqlens``, ``N`` is the packed checkpoint-slot
            count ``B * (1 + num_spec_tokens)`` when ``ssm_state_indices`` is
            omitted.
        output_final_state (bool):
            Whether to return the final state. Default: ``False``.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2 normalization to Q and K. Default: ``True``.
        use_gate_in_kernel (bool):
            Whether to compute the gate inside the kernel from ``A_log`` and
            ``g``. Default: ``False``.
        lower_bound (Optional[float]):
            If set, uses ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
            gate formula. If ``None``, uses
            ``-exp(A_log) * softplus(g + dt_bias)``. A supplied bound must be
            negative.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape ``[N+1]``. Must be int32.
        ssm_state_indices (Optional[torch.Tensor]):
            State cache indices. Shape ``[N]`` int32 for standard decode, or
            ``[N, 1+S]`` int32 for spec decode (``num_spec_tokens`` must also
            be set).
        num_spec_tokens (Optional[int]):
            Number of speculative tokens (S). When set, processes 1+S tokens in
            a single fused kernel launch. Must be >= 1.
        num_accepted_tokens (Optional[torch.Tensor]):
            Per-sequence accepted token count from the previous spec decode
            round. Shape ``[N]`` int32. If ``None``, initial state is loaded
            from ``ssm_state_indices[n, 0]``. Values above ``1+S`` are clamped
            to the final checkpoint slot.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor. Shape ``[B, 1, HV, V]`` for standard
            decode, ``[1, N*(1+S), HV, V]`` for spec decode with
            ``cu_seqlens``. If ``None``, a new tensor is allocated.
        initial_state_source (Optional[torch.Tensor]):
            Optional read-only committed state pool ``[N0, HV, V, K]``. When
            provided, token 0 is loaded from this pool instead of
            ``initial_state``.
        initial_state_indices (Optional[torch.Tensor]):
            Source slot per sequence, shape ``[N]`` int32. Required together
            with ``initial_state_source``.
        beta_is_logit (bool):
            If ``True``, apply sigmoid to ``beta`` inside the recurrent kernel.
        disable_state_update (bool):
            Frozen / speculative-verify mode (mirrors GDN's
            ``gated_delta_rule_mtp`` flag): compute the outputs for up to 16
            tokens per sequence from the committed state and never write any
            state back; ``final_state`` is always ``None``. Dispatches
            internally to the WY-parallel tensor-core kernel or a grouped
            register recurrence by problem size. Supports the batched
            ``[B, T, ...]`` form directly and the packed ``cu_seqlens`` form
            (ragged per-sequence lengths). ``backend="cake"`` raises in this
            mode (no frozen-state Cake kernels; no silent fallback), and
            ``output_final_state=True`` is rejected. Requires ``K == V ==
            128`` and a bf16 state pool.
        correction_cache (Optional[torch.Tensor]):
            Only with ``disable_state_update=True``. Slot-indexed float32
            buffer ``[num_slots, HV, T_max, V]`` receiving the per-token
            delta-rule corrections ``sigmoid-or-raw(beta) * (v - u)`` for a
            downstream commit/recovery kernel (the analog of GDN's
            slot-indexed ``intermediate_states_buffer``). Rows past each
            sequence's length and null slots are left untouched.
        kg_cache (Optional[torch.Tensor]):
            Only with ``disable_state_update=True``. Slot-indexed bf16 buffer
            ``[num_slots, HV, T_max, 2*K]`` receiving the raw (unnormalized) key
            in ``[..., :K]`` and the raw gate in ``[..., K:]`` per token,
            matching the vLLM RecoverSSM cache convention.
        backend (Literal["cute-dsl", "cake", "auto"]):
            Implementation backend. ``"cute-dsl"`` preserves the existing
            FlashInfer implementation. ``"cake"`` strictly selects an
            exported Cake kernel and raises when the call does not match one
            of its supported contracts. ``"auto"`` selects Cake only for its
            equal-head/D128/T1 unbounded-softplus contract, preserving CuTe
            DSL for every other decode surface. Default: ``"cute-dsl"``.

    Returns:
        Tuple of ``(output, final_state)`` where ``final_state`` is ``None``
        when ``output_final_state=False``. See
        :func:`flashinfer.kda_kernels.recurrent_kda.run_recurrent_kda` for the
        backend implementation.
    """
    if backend not in ("cute-dsl", "cake", "auto"):
        raise ValueError(
            f"backend must be 'cute-dsl', 'cake', or 'auto', got {backend!r}"
        )
    if (correction_cache is not None or kg_cache is not None) and (
        not disable_state_update
    ):
        raise ValueError(
            "correction_cache/kg_cache are speculative-verify caches and "
            "require disable_state_update=True"
        )
    if disable_state_update:
        if backend == "cake":
            raise ValueError(
                "backend='cake' has no frozen-state kernels; "
                "disable_state_update=True requires the CuTe-DSL backends"
            )
        if output_final_state:
            raise ValueError(
                "output_final_state=True is incompatible with "
                "disable_state_update=True (no state is produced)"
            )
        if num_accepted_tokens is not None:
            raise ValueError(
                "num_accepted_tokens applies to the state-updating fused "
                "spec path, not the frozen-verify mode"
            )
        return _run_frozen_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_spec_tokens=num_spec_tokens,
            output=output,
            initial_state=initial_state,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            beta_is_logit=beta_is_logit,
            correction_cache=correction_cache,
            kg_cache=kg_cache,
        )
    if _run_recurrent_kda is None:
        raise NotImplementedError("recurrent KDA backend is unavailable")

    run_kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=num_accepted_tokens,
        output=output,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        beta_is_logit=beta_is_logit,
    )
    return _run_recurrent_kda(**run_kwargs, backend=backend)


@flashinfer_api(trace=packed_kda_decode_trace)
def packed_kda_decode(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Run serving-native packed Kimi K3 recurrent decode.

    This operator consumes the post-convolution packed QKV row and raw gate
    and beta logits directly. It fuses Q/K extraction and L2 normalization,
    the Kimi K3 lower-bound gate transform, beta sigmoid, and one recurrent
    state update into a single exported Cake kernel. It is specialized for
    ``T=1``, ``H=12``, and ``K=V=128`` on exact SM100a and SM103a devices.

    The fixed numerical contract uses ``scale=1/sqrt(128)``, L2 epsilon
    ``1e-6``, and ``lower_bound=-5``. ``state`` is updated in place on the
    caller's current PyTorch CUDA stream. Batches below 32 use the eight-row
    value tile; batches of 32 or more use the sixteen-row value tile.

    Args:
        mixed_qkv:
            Post-convolution packed QKV with shape ``[B, 3 * 12 * 128]`` and
            dtype bfloat16. The last dimension must be contiguous; positive
            padding between batch rows is allowed.
        raw_gate:
            Raw per-channel recurrence gate with shape ``[B, 12 * 128]`` and
            dtype bfloat16. The last dimension must be contiguous.
        raw_beta:
            Raw delta-rule learning-rate logits with shape ``[B, 12]`` and
            dtype bfloat16. The last dimension must be contiguous.
        A_log:
            Contiguous float32 log-decay parameter with shape ``[12]``.
        dt_bias:
            Contiguous float32 per-channel decay bias with shape ``[12 * 128]``.
        state:
            Caller-owned bfloat16 recurrent-state pool with shape
            ``[N, 12, 128, 128]``. Its inner three dimensions must be compact;
            the outer slot stride may contain arbitrary positive padding.
        state_indices:
            Contiguous CUDA int32 cache slot for each row, with shape ``[B]``.
            Active indices must be unique and in bounds. ``-1`` marks an
            inactive CUDA-graph padding row, which produces zero output and
            does not access or update ``state``. These value constraints are
            not host-validated, avoiding a device synchronization.
        output:
            Optional caller-owned contiguous bfloat16 output with shape
            ``[B, 1, 12, 128]``. Supplying it avoids allocation and is required
            for an allocation-free CUDA-graph replay path.

    Returns:
        The bfloat16 output with shape ``[B, 1, 12, 128]`` by default. When
        ``output`` is supplied, the returned tensor is that exact allocation.
    """
    return _run_packed_kda_decode(
        mixed_qkv=mixed_qkv,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state=state,
        state_indices=state_indices,
        output=output,
        backend="cake",
    )


@flashinfer_api(trace=fused_kda_decode_trace)
def fused_kda_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    lower_bound: Optional[float] = -5.0,
    norm_eps: float = 1e-5,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Run the fused Kimi KDA decode pipeline.

    This operator fuses a width-four depthwise causal convolution with SiLU,
    one recurrent KDA update, and gated RMSNorm. It is specialized for
    head dimension 128 and 12, 24, 32, 48, or 96 heads. ``conv_state`` and
    ``state`` are updated in-place.

    Slot zero is reserved as a null slot. Rows whose ``state_indices`` value
    is non-positive produce zeros and do not update either cache.

    Args:
        x:
            Packed QKV projection with shape ``[num_rows, 3 * H * 128]`` and
            dtype bfloat16. The channel dimension must be contiguous.
        weight:
            Depthwise convolution weights with shape ``[3, 4, H * 128]`` and
            dtype float32.
        conv_state:
            Paged convolution cache with shape
            ``[num_slots, 3 * H * 128, 3]`` and dtype bfloat16. Each slot must
            use the sequence-dimension cache layout with strides
            ``[slot_stride, 1, 3 * H * 128]``.
        raw_gate:
            Raw per-channel recurrence gate with shape
            ``[1, num_rows, H, 128]`` and dtype bfloat16.
        raw_beta:
            Raw delta-rule learning-rate logits with shape
            ``[1, num_rows, H]`` and dtype bfloat16.
        A_log:
            Log decay parameter with ``H`` elements and dtype float32.
        dt_bias:
            Per-channel decay bias with ``H * 128`` elements and dtype float32.
        state_indices:
            Cache slot selected by each decode row. Must be a contiguous int32
            tensor with ``num_rows`` elements. Live indices must be in
            ``[1, num_slots)``; non-positive indices select the null path.
        state:
            Paged recurrent state with shape
            ``[num_slots, H, 128, 128]`` and dtype float32 or bfloat16. The
            recurrence is evaluated in float32; a bfloat16 state is rounded
            when written back. Each slot's ``[H, 128, 128]`` contents must be
            contiguous. ``state`` and ``conv_state`` must have the same
            ``num_slots``.
        output_gate:
            Gated RMSNorm logits with shape ``[num_rows, H, 128]`` or
            ``[1, num_rows, H, 128]`` and dtype bfloat16.
        norm_weight:
            RMSNorm weight with 128 elements and dtype float32.
        lower_bound:
            Negative lower bound used by the recurrence gate. Defaults to
            ``-5.0`` for Kimi K3. Pass ``None`` to use the original
            Kimi-Linear softplus gate.
        norm_eps:
            Non-negative RMSNorm epsilon. Defaults to ``1e-5``.
        output:
            Optional preallocated contiguous bfloat16 output with shape
            ``[1, num_rows, H, 128]``.

    Returns:
        The bfloat16 output tensor with shape ``[1, num_rows, H, 128]``.
    """
    if _run_fused_kda_decode is None:
        raise NotImplementedError("fused KDA decode backend is unavailable")
    return _run_fused_kda_decode(
        x=x,
        weight=weight,
        conv_state=conv_state,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_indices=state_indices,
        state=state,
        output_gate=output_gate,
        norm_weight=norm_weight,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        output=output,
    )


# Cached trivial metadata for the frozen mode's uniform batched form: the
# arange cu_seqlens / slot indices are content-stable per (device, B, T), so
# building them once keeps the hot path allocation- and launch-free (required
# for CUDA-graph capture and honest kernel-time benchmarking).
_FROZEN_META: dict = {}


def _frozen_arange(device, n, step=1):
    key = (str(device), int(n), int(step))
    t = _FROZEN_META.get(key)
    if t is None:
        with torch.inference_mode(False):
            t = torch.arange(0, n * step, step, dtype=torch.int32, device=device)
        _FROZEN_META[key] = t
    return t


def _run_frozen_recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    lower_bound,
    cu_seqlens,
    ssm_state_indices,
    num_spec_tokens,
    output,
    initial_state,
    initial_state_source,
    initial_state_indices,
    beta_is_logit,
    correction_cache,
    kg_cache,
):
    """recurrent_kda's frozen / speculative-verify mode.

    Dispatch, mirroring the GDN precedent (mode on the op, kernels internal):

    - batched ``[B, T, ...]`` without caches -> the output-only dispatcher
      (WY-parallel tensor-core kernel or grouped register recurrence by
      problem size);
    - any call requesting ``correction_cache``/``kg_cache``, and packed
      ``cu_seqlens`` calls, -> the packed frozen-verify kernel (WY, ragged
      lengths and null slots supported), which writes the slot-indexed fp32
      correction and bf16 kg caches.

    Never writes any state pool; returns ``(output, None)``.
    """
    if _run_kda_output_only is None:
        raise NotImplementedError(
            "disable_state_update=True requires the CuTe-DSL frozen-state "
            "kernels (missing cutlass DSL deps)"
        )
    if not use_qk_l2norm_in_kernel:
        raise ValueError(
            "disable_state_update=True always applies Q/K L2 normalization "
            "(use_qk_l2norm_in_kernel=False is not supported)"
        )
    pool = initial_state_source if initial_state_source is not None else initial_state
    if pool is None:
        raise ValueError(
            "disable_state_update=True requires a committed state pool via "
            "initial_state_source (preferred) or initial_state (read-only)"
        )
    slots = (
        initial_state_indices
        if initial_state_indices is not None
        else ssm_state_indices
    )
    want_caches = correction_cache is not None or kg_cache is not None
    if want_caches and (correction_cache is None or kg_cache is None):
        raise ValueError("correction_cache and kg_cache must be provided together")

    from .kda_kernels.kda_decode_wy_output_only import (
        _dummy_f32,
        kda_recoverssm_verify as _packed_frozen_verify,
    )

    if cu_seqlens is None and not want_caches:
        # Batched [B, T, ...] frozen decode without verify caches.
        out = _run_kda_output_only(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state_source=pool,
            initial_state_indices=slots,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound if use_gate_in_kernel else None,
            beta_is_logit=beta_is_logit,
            output=output,
            backend="auto",
        )
        return out, None

    # Packed frozen verify (and every cache-requesting call): slot-indexed
    # caches, ragged lengths, null slots.
    if q.shape[-2] != v.shape[-2]:
        raise NotImplementedError(
            "GQA (HV != H) is not supported by the packed frozen-verify path "
            "(correction/kg caches or cu_seqlens inputs) yet — the underlying "
            "vLLM RecoverSSM contract is equal-head (Kimi K3). Use the "
            "batched [B, T, ...] form without caches for GQA."
        )
    device = q.device
    if cu_seqlens is None:
        # Uniform batched form: build the trivial cu_seqlens and flatten to
        # the packed [1, total, ...] views (free reshapes of contiguous
        # tensors; .reshape falls back to a copy for non-contiguous inputs).
        B, T_in = q.shape[0], q.shape[1]
        H, HV = q.shape[2], v.shape[2]
        qsl = _frozen_arange(device, B + 1, T_in)
        qp = q.reshape(1, B * T_in, H, q.shape[3])
        kp = k.reshape(1, B * T_in, H, k.shape[3])
        vp = v.reshape(1, B * T_in, HV, v.shape[3])
        gp = g.reshape(1, B * T_in, HV, g.shape[3])
        bp = beta.reshape(1, B * T_in, HV)
        outp = (
            output.reshape(1, B * T_in, HV, v.shape[3]) if output is not None else None
        )
        spec_len = T_in
        reshape_out = (B, T_in)
    else:
        if not want_caches:
            raise ValueError(
                "packed (cu_seqlens) frozen-verify calls currently require "
                "correction_cache/kg_cache; for a plain frozen decode pass "
                "batched [B, T, ...] tensors instead"
            )
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError(
                f"cu_seqlens must be 1-D with at least 2 entries; "
                f"got {tuple(cu_seqlens.shape)}"
            )
        B = cu_seqlens.shape[0] - 1
        qp, kp, vp, gp, bp, outp, qsl = q, k, v, g, beta, output, cu_seqlens
        if num_spec_tokens is not None:
            spec_len = num_spec_tokens + 1
        elif correction_cache is not None:
            spec_len = correction_cache.shape[2]
        else:
            raise ValueError(
                "packed frozen-verify calls need the window size via "
                "num_spec_tokens or the caches' token dimension"
            )
        reshape_out = None
    if slots is None:
        slots = _frozen_arange(device, B)
    elif slots.ndim != 1 or slots.shape[0] != B:
        raise ValueError(
            f"state slot indices must be [B={B}]; got {tuple(slots.shape)}"
        )
    if want_caches:
        corr_t, kg_t = correction_cache, kg_cache
    else:
        # Batched call without caches but through the packed kernel is not
        # reachable (handled above); guard for completeness.
        raise AssertionError("unreachable: packed path without caches")
    H = qp.shape[2]
    K_dim = qp.shape[3]
    A_log_eff = A_log if A_log is not None else _dummy_f32(device, H)
    dt_bias_eff = dt_bias if dt_bias is not None else _dummy_f32(device, H * K_dim)
    out = _packed_frozen_verify(
        qp,
        kp,
        vp,
        gp,
        bp,
        A_log_eff,
        dt_bias_eff,
        lower_bound if use_gate_in_kernel else None,
        pool,
        corr_t,
        kg_t,
        qsl,
        slots,
        spec_len,
        outp,
        use_gate_in_kernel=use_gate_in_kernel,
        beta_is_logit=beta_is_logit,
        scale=scale,
        # recurrent_kda slot convention: 0 is a valid slot; only negative
        # slots are padding (the vLLM drop-in reserves slot 0 as null).
        null_min=0,
    )
    if reshape_out is not None:
        out = out.reshape(reshape_out[0], reshape_out[1], out.shape[2], out.shape[3])
    return out, None
