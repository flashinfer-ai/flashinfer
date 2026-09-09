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
Kimi Delta Attention - Public Facade
====================================

This phase-neutral facade preserves the top-level recurrent KDA entry point.
Eligible ordinary multi-token prefill calls use ``flashinfer.kda_prefill``;
decode and speculative decode retain the backend exposed by
``flashinfer.kda_decode``.
"""

import threading
from typing import Literal, Optional

import torch

from . import kda_decode as _kda_decode
from . import kda_prefill as _kda_prefill
from . import kda_prefill_cute as _kda_prefill_cute
from .jit import flash_kda_indexed as _flash_kda_indexed
from .api_logging import flashinfer_api, flashinfer_experimental_api
from .kda_backward import (
    RecurrentKDABackwardWorkspace as RecurrentKDABackwardWorkspace,
)
from .kda_backward import recurrent_kda_backward as recurrent_kda_backward
from .kda_training import (
    RecurrentKDATrainingContext as RecurrentKDATrainingContext,
)
from .kda_training import (
    recurrent_kda_training_backward as recurrent_kda_training_backward,
)
from .kda_training import (
    recurrent_kda_training_forward as recurrent_kda_training_forward,
)
from .trace.templates.kda import recurrent_kda_trace
from .utils import get_compute_capability


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
    seq_order: Optional[torch.Tensor] = None,
    prefill_workspace: Optional[_kda_prefill.RecurrentKDAPrefillWorkspace] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_state_indices: Optional[torch.Tensor] = None,
    *,
    disable_state_update: bool = False,
    correction_cache: Optional[torch.Tensor] = None,
    kg_cache: Optional[torch.Tensor] = None,
    backend: Literal["auto", "cute-dsl", "cake"] = "auto",
) -> (
    tuple[torch.Tensor, Optional[torch.Tensor]]
    | tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
):
    r"""Recurrent KDA (Kimi Delta Attention) decode and prefill kernel.

    This is the public API layer for the CuTe DSL implementation in
    ``flashinfer.kda_kernels.recurrent_kda``. It supports single-token decode,
    fused speculative decode, GQA, optional cu_seqlens packing, and the same
    gate modes as the backend implementation. On SM120a, eligible ordinary
    multi-token prefill uses the architecture-specific CuTe DSL backend. On
    SM100a (B200/GB200) and SM103a (B300/GB300), the FlashKDA-compatible subset
    can use either the frozen Cake schedules or the source-level CuTe DSL BT=16
    kernel. The Cake backend includes a generated two-stage BT=16
    prepare/chain portfolio with device- and shape-specific S7/S8/S9 pipeline
    selection. ``backend="auto"`` prefers CuTe DSL for supported plain prefill
    contracts and keeps Cake as the feature-complete fallback; use
    ``backend="cake"`` to select and benchmark the generated portfolio
    explicitly.
    Compatible equal-head D128 unbounded-softplus T=1 decode calls use their
    frozen Cake specialization automatically. The Cake path accepts any
    positive runtime head count, so Kimi-Linear tensor parallelism maps global
    H32 to per-rank H32/H16/H8/H4 without an adapter. Other decode and
    speculative-decode calls retain the CuTe DSL backend.

    Args:
        q (torch.Tensor):
            Query of shape ``[B, T, H, K]``, or
            ``[1, total_tokens, H, K]`` when using ``cu_seqlens``. Must be
            bfloat16. ``T=1`` selects decode; eligible ``T>1`` calls may select
            the frozen prefill backend.
        k (torch.Tensor):
            Key with the same shape as ``q``. Must be bfloat16.
        v (torch.Tensor):
            Value of shape ``[B, T, HV, V]``, or
            ``[1, total_tokens, HV, V]`` when packed. Must be bfloat16. GQA is
            applied when ``HV != H``.
        g (torch.Tensor):
            Per-K-dimension gate of shape ``[B, T, HV, K]``, or
            ``[1, total_tokens, HV, K]`` when packed. Must be bfloat16.
            Log-space if pre-computed, raw input if
            ``use_gate_in_kernel=True``.
        beta (torch.Tensor):
            Delta-rule learning rate of shape ``[B, T, HV]``, or
            ``[1, total_tokens, HV]`` when packed. Must be bfloat16.
            Pre-sigmoided unless ``beta_is_logit=True``. Eligible frozen
            prefill accepts non-overlapping token-row-strided storage with a
            unit head stride, including a view into a fused projection.
        A_log (Optional[torch.Tensor]):
            Log decay parameter of shape ``[H]``. Must be float32.
            Required when ``use_gate_in_kernel=True``.
        dt_bias (Optional[torch.Tensor]):
            Per-head-K decay bias of shape ``[H*K]`` or ``[H, K]``. Must be
            float32.
        scale (Optional[float]):
            Scale factor for queries. If ``None``, defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[N, HV, V, K]``. Eligible prefill
            backends accept bfloat16 or float32; other modes may be stricter.
            If ``None``, zero-initialized. Updated in-place. For
            batched spec decode without ``cu_seqlens``, ``N`` is the packed
            checkpoint-slot count ``B * (1 + num_spec_tokens)`` when
            ``ssm_state_indices`` is omitted. For eligible frozen prefill with
            ``ssm_state_indices``, this is a state pool
            ``[N_pool, H, 128, 128]`` whose inner slots are contiguous;
            padding between pool slots is allowed.
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
            Contiguous CUDA cumulative sequence lengths of shape ``[N+1]``.
            May be int32 or int64. Frozen prefill converts int32 offsets to
            int64 outside graph capture; graph capture requires caller-provided
            int64 offsets. For frozen prefill, values must start at zero, be
            non-decreasing, and end at the total token count. This value
            contract is not normally host-validated. Cake may read these values
            to prepare and cache host scheduling metadata. CuTe DSL generates
            packed sequence ordering on the device and can generate decomposed
            chunk metadata there when using
            :class:`RecurrentKDAPrefillWrapper`. Eligible 148-SM B200 and
            152-SM GB200 Cake calls additionally cache persistent worker task
            bins.
        ssm_state_indices (Optional[torch.Tensor]):
            State cache indices. Shape ``[N]`` int32 for standard decode, or
            ``[N, 1+S]`` int32 for spec decode (``num_spec_tokens`` must also
            be set). Eligible frozen packed prefill accepts contiguous CUDA
            int32 ``[N_seq]`` indices and updates the selected
            ``initial_state`` pool slots directly.
        num_spec_tokens (Optional[int]):
            Number of speculative tokens (S). When set, processes 1+S tokens in
            a single fused kernel launch. Must be >= 1.
        num_accepted_tokens (Optional[torch.Tensor]):
            Per-sequence accepted token count from the previous spec decode
            round. Shape ``[N]`` int32. If ``None``, initial state is loaded
            from ``ssm_state_indices[n, 0]``. Values above ``1+S`` are clamped
            to the final checkpoint slot.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor. Shape ``[B, T, HV, V]`` for fixed
            layout, or the corresponding packed/speculative shape when using
            ``cu_seqlens``. If ``None``, a new tensor is allocated. Frozen
            prefill requires storage disjoint from Q, K, V, G, beta, and
            ``initial_state``.
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
            Frozen / speculative-verify mode: compute outputs for up to 16
            tokens per sequence from the committed state and never write any
            state back (``final_state`` is ``None``). See
            :func:`flashinfer.kda_decode.recurrent_kda` for the full mode
            contract, including the optional slot-indexed
            ``correction_cache`` / ``kg_cache`` verify outputs.
        correction_cache (Optional[torch.Tensor]):
            Frozen-verify only: slot-indexed float32 per-token delta-rule
            corrections ``[num_slots, HV, T_max, V]``.
        kg_cache (Optional[torch.Tensor]):
            Frozen-verify only: slot-indexed bf16 (raw key | raw gate)
            cache ``[num_slots, HV, T_max, 2*K]``.
        seq_order (Optional[torch.Tensor]):
            Optional packed-prefill sequence order, as a contiguous CUDA int32
            permutation of shape ``[N]``. CuTe DSL packed engine calls generate
            a stable longest-sequence-first order on the device when this is
            omitted; CuTe DSL decomp keeps the original order unless the graph
            wrapper requests device-generated metadata. Cake constructs and
            caches its own eager host metadata. On Cake, supplying an order
            disables persistent host task-bin planning but does not force direct
            M128; the selected non-persistent route may still be BT16
            prepare/chain, M64, small-BH, or direct according to the input shape.
            Fixed-layout prefill and decode calls must leave it as ``None``.
        prefill_workspace (Optional[RecurrentKDAPrefillWorkspace]):
            Caller-owned workspace for SM100-family and SM120 prefill backends.
            It is optional for eager execution and required for CUDA graph
            capture. Warm it eagerly with the exact tensors on the capture
            stream before capture. Use one workspace per captured
            ``recurrent_kda`` invocation. Explicit workspaces and CUDA Graph
            capture use non-persistent schedules, including eligible BT16,
            M64, small-BH, and direct routes. Persistent task planning is an
            eager-only B200/GB200 route because its bins depend on host-visible
            sequence lengths.
        state_checkpoints (Optional[torch.Tensor]):
            Checkpoint output or pool ``[C, H, 128, 128]`` for prefill. CuTe DSL
            accepts BF16 or FP32 and requires it to match ``initial_state`` when
            present; Cake accepts BF16. Without ``checkpoint_state_indices``,
            row zero for each sequence is its initial state and later rows are
            states before token blocks beginning at ``N, 2N, ...``. CuTe DSL
            allocates this packed output during eager execution when omitted;
            CUDA graph capture requires a caller-owned tensor.
        checkpoint_cu_starts (Optional[torch.Tensor]):
            Contiguous CUDA int64 cumulative checkpoint counts ``[N_seq+1]``.
            The first value must be zero. Without ``checkpoint_state_indices``,
            each difference is ``ceil(seq_len / N)``. With indices, each
            difference is ``floor(seq_len / N)`` and counts completed periodic
            boundaries, including an aligned final boundary and excluding the
            initial state.
        checkpoint_state_indices (Optional[torch.Tensor]):
            CuTe DSL-only contiguous CUDA int32 destination rows. Entry ``i``
            selects the row of the ``state_checkpoints`` pool written for packed
            completed-boundary entry ``i``. The kernel writes the pool directly;
            no temporary checkpoint tensor or scatter is used.
        checkpoint_every_n_tokens (int):
            Checkpoint interval. Zero disables checkpoints; a positive value
            must be divisible by 32, except that the SM100-family exact-N16
            frozen route also accepts multiples of 16. SGLang normally uses
            64 or a larger cache-page-aligned multiple.
        backend (Literal["auto", "cute-dsl", "cake"]):
            Implementation backend. ``"auto"`` selects the architecture-
            appropriate CuTe DSL kernel for supported ordinary multi-token
            prefill, including the SM120 backend and SM100-family state
            checkpoints, and otherwise falls back to an exported frozen Cake
            specialization.
            ``"cake"`` and ``"cute-dsl"`` select those backends strictly. The
            Cake prefill path chooses among direct, persistent, small-BH, and
            two-stage BT16 schedules from the input shape and physical device.
            The SM100-family kernel additionally needs
            ``nvidia-cutlass-dsl>=4.7``; below that ``"auto"`` uses Cake there
            and ``"cute-dsl"`` raises :class:`ImportError`.

    Returns:
        Tuple of ``(output, final_state)`` where ``final_state`` is ``None``
        when ``output_final_state=False``. When checkpointing is enabled, a
        triple ``(output, final_state, state_checkpoints)`` is returned. See
        :func:`flashinfer.kda_kernels.recurrent_kda.run_recurrent_kda` for the
        backend implementation.
    """
    if prefill_workspace is not None and not isinstance(
        prefill_workspace, _kda_prefill.RecurrentKDAPrefillWorkspace
    ):
        raise TypeError("prefill_workspace must be a RecurrentKDAPrefillWorkspace")
    if backend not in ("auto", "cute-dsl", "cake"):
        raise ValueError(
            f"backend must be 'auto', 'cute-dsl', or 'cake', got {backend!r}"
        )
    if checkpoint_state_indices is not None and backend == "cake":
        raise ValueError("checkpoint_state_indices is supported only by CuTe DSL")

    # SM120 is an architecture-specific CuTe DSL implementation. Try it before
    # the SM100-family CuTe DSL path, whose eligibility check rejects SM120.
    sm120_rejection: Optional[str] = None
    if backend in ("auto", "cute-dsl") and checkpoint_state_indices is None:
        sm120_prefill_kwargs = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
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
            seq_order=seq_order,
            prefill_workspace=prefill_workspace,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )
        if _kda_prefill._sm120_kda_prefill_is_eligible(**sm120_prefill_kwargs):
            assert A_log is not None
            assert dt_bias is not None
            assert lower_bound is not None
            return _kda_prefill._run_sm120_kda_prefill(
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
                lower_bound=lower_bound,
                cu_seqlens=cu_seqlens,
                output=output,
                prefill_workspace=prefill_workspace,
            )
        if (
            backend == "cute-dsl"
            and q.is_cuda
            and get_compute_capability(q.device) == (12, 0)
        ):
            # Recorded, not raised: a decode or any other call this backend does
            # not take must keep falling through exactly as before.  It is used
            # only where the CC 10.0/10.3 block already refuses an explicit
            # request, which on this architecture can only answer with the
            # contract when the reason is known right here.
            sm120_rejection = _kda_prefill._sm120_kda_prefill_rejection_reason(
                **sm120_prefill_kwargs
            )

    if (correction_cache is not None or kg_cache is not None) and (
        not disable_state_update
    ):
        raise ValueError(
            "correction_cache/kg_cache are speculative-verify caches and "
            "require disable_state_update=True"
        )
    if disable_state_update:
        # Frozen / speculative-verify mode (mirrors GDN's
        # gated_delta_rule_mtp flag): outputs only, no state writes,
        # optional slot-indexed correction/kg caches. Handled ahead of the
        # prefill routing — none of the prefill-only features apply.
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
        if (
            seq_order is not None
            or prefill_workspace is not None
            or state_checkpoints is not None
            or checkpoint_cu_starts is not None
            or checkpoint_state_indices is not None
            or checkpoint_every_n_tokens != 0
        ):
            raise ValueError(
                "prefill-only arguments are incompatible with disable_state_update=True"
            )
        return _kda_decode._run_frozen_recurrent_kda(
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

    is_plain_prefill = _kda_prefill._is_plain_multi_token_prefill(
        q, cu_seqlens, num_spec_tokens
    )
    use_generated_indexed_prefill = (
        backend == "cake"
        and is_plain_prefill
        and _flash_kda_indexed.flash_kda_indexed_prefill_is_eligible(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
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
            seq_order=seq_order,
            prefill_workspace=prefill_workspace,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )
    )
    if use_generated_indexed_prefill:
        assert A_log is not None
        assert dt_bias is not None
        assert initial_state is not None
        assert ssm_state_indices is not None
        assert lower_bound is not None
        return _flash_kda_indexed._run_flash_kda_indexed_prefill(
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
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            output=output,
            state_indices=ssm_state_indices,
        )
    try_cute_dsl_prefill = backend in ("auto", "cute-dsl")
    if try_cute_dsl_prefill and is_plain_prefill:
        cute_dsl_eligible = _kda_prefill_cute._is_cute_dsl_kda_prefill_eligible(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            seq_order=seq_order,
            ssm_state_indices=ssm_state_indices,
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens=num_accepted_tokens,
            output=output,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            beta_is_logit=beta_is_logit,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_state_indices=checkpoint_state_indices,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )
        if backend == "cute-dsl" and not cute_dsl_eligible:
            if _kda_prefill_cute._is_cute_dsl_kda_prefill_dsl_too_old(q):
                raise ImportError(
                    "backend='cute-dsl' requires nvidia-cutlass-dsl>=4.7.0 "
                    "(cutlass.experimental); backend='auto' falls back to Cake"
                )
            raise ValueError(
                "backend='cute-dsl' does not support this recurrent_kda "
                "prefill contract"
                if sm120_rejection is None
                else "backend='cute-dsl' selects the SM120 prefill backend on "
                "a CC 12.0 device, and it does not support this call: "
                + sm120_rejection
            )
        if cute_dsl_eligible:
            assert A_log is not None
            assert dt_bias is not None
            assert lower_bound is not None
            return _kda_prefill_cute._run_cute_dsl_kda_prefill(
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
                lower_bound=lower_bound,
                cu_seqlens=cu_seqlens,
                seq_order=seq_order,
                output=output,
                prefill_workspace=prefill_workspace,
                state_indices=ssm_state_indices,
                state_checkpoints=state_checkpoints,
                checkpoint_cu_starts=checkpoint_cu_starts,
                checkpoint_state_indices=checkpoint_state_indices,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            )

    use_flash_kda_prefill = (
        not (
            isinstance(initial_state, torch.Tensor)
            and initial_state.dtype == torch.float32
            and (
                checkpoint_every_n_tokens != 0
                or state_checkpoints is not None
                or checkpoint_cu_starts is not None
            )
        )
        and backend != "cute-dsl"
        and checkpoint_state_indices is None
        and _kda_prefill._flash_kda_prefill_is_eligible(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
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
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )
    )
    if (
        backend in ("auto", "cake")
        and is_plain_prefill
        and isinstance(initial_state, torch.Tensor)
        and initial_state.dtype == torch.float32
        and (
            checkpoint_every_n_tokens != 0
            or state_checkpoints is not None
            or checkpoint_cu_starts is not None
        )
    ):
        raise ValueError("FP32 state checkpoints are not supported by Cake prefill")
    if use_flash_kda_prefill:
        assert A_log is not None
        assert dt_bias is not None
        return _kda_prefill._run_flash_kda_prefill(
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
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            output=output,
            seq_order=seq_order,
            prefill_workspace=prefill_workspace,
            state_indices=ssm_state_indices,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )

    if backend == "cake" and is_plain_prefill:
        raise ValueError(
            "backend='cake' does not support this recurrent_kda prefill contract"
        )

    if (
        checkpoint_every_n_tokens != 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
        or checkpoint_state_indices is not None
    ):
        raise ValueError(
            "state checkpoints are supported only by eligible frozen "
            "SM100/SM103 recurrent_kda prefill"
        )

    if prefill_workspace is not None:
        raise ValueError(
            "prefill_workspace is only supported by eligible ordinary "
            "multi-token prefill backends"
        )
    if seq_order is not None:
        raise ValueError(
            "seq_order is only supported by eligible packed ordinary "
            "SM100-family prefill"
        )
    if _kda_decode._run_recurrent_kda is None:
        raise NotImplementedError("recurrent KDA backend is unavailable")

    return _kda_decode._run_recurrent_kda(
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
        backend=backend,
    )


class RecurrentKDAPrefillWrapper:
    """Plan-and-run wrapper for packed recurrent-KDA prefill.

    .. warning::
        ``RecurrentKDAPrefillWrapper`` is experimental: it provides no
        compatibility guarantees and may change or be removed without
        deprecation.  It has not appeared in a release; the plan-and-run shape
        is expected to change as the ``recurrent_kda`` surface is unified
        (see `#4936 <https://github.com/flashinfer-ai/flashinfer/issues/4936>`_).

    Compute capability 10.0 and 10.3 only.  ``run`` forces ``backend="cute-dsl"``
    and always passes the ``seq_order`` it planned, and the CC 12.0 backend
    supports neither, so a CC 12.0 caller should use
    :func:`flashinfer.kda.recurrent_kda` directly.

    ``plan`` runs outside CUDA Graph capture and copies ``cu_seqlens`` into a
    fixed-address device buffer without reading its values on the host. ``run``
    generates the stable descending-length sequence order and cumulative chunk
    prefix on the GPU before the recurrent kernels consume them.

    The number of sequences and packed token tensor extent are fixed after the
    first warmup run so device buffer addresses, workspace capacity, and launch
    geometry remain valid across CUDA Graph replays. Call ``plan`` again before
    replay to update individual lengths in place.

    This wrapper is specific to the CuTe DSL backend and intentionally uses
    its non-persistent schedule. One wrapper instance is a single-writer
    resource: do not call ``plan`` concurrently with ``run`` or while a kernel
    launched by ``run`` may still be reading the wrapper's planned buffers.
    """

    def __init__(
        self,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("RecurrentKDAPrefillWrapper requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._workspace = _kda_prefill.RecurrentKDAPrefillWorkspace(self.device)
        self._cu_seqlens_buf: Optional[torch.Tensor] = None
        self._seq_order_buf: Optional[torch.Tensor] = None
        self._cu_chunks_buf: Optional[torch.Tensor] = None
        self._num_sequences: Optional[int] = None
        self._total_tokens: Optional[int] = None
        self._planned = False
        self._lock = threading.Lock()

    @flashinfer_experimental_api
    def plan(
        self,
        cu_seqlens: torch.Tensor,
        *,
        non_blocking: bool = True,
    ) -> None:
        """Plan a packed prefill sequence order outside CUDA Graph capture.

        ``cu_seqlens`` may reside on CPU or on this wrapper's CUDA device and
        may use int32 or int64 storage. Its values are copied into a stable
        int64 CUDA buffer; GPU metadata generation during ``run`` handles
        repeated offsets for zero-length sequences.
        """

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "RecurrentKDAPrefillWrapper.plan must run outside CUDA graph capture"
            )
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if (
            cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.numel() < 2
        ):
            raise ValueError(
                "cu_seqlens must be a contiguous int32 or int64 tensor with "
                "at least two entries"
            )
        if cu_seqlens.is_cuda and cu_seqlens.device != self.device:
            raise ValueError(
                f"cu_seqlens must be on {self.device} or CPU, got {cu_seqlens.device}"
            )

        num_sequences = cu_seqlens.numel() - 1

        with self._lock:
            if self._num_sequences is None:
                self._num_sequences = num_sequences
                self._cu_seqlens_buf = torch.empty(
                    num_sequences + 1, dtype=torch.int64, device=self.device
                )
                self._seq_order_buf = torch.empty(
                    num_sequences, dtype=torch.int32, device=self.device
                )
                self._cu_chunks_buf = torch.empty(
                    num_sequences + 1, dtype=torch.int32, device=self.device
                )
            elif num_sequences != self._num_sequences:
                raise ValueError(
                    "the number of sequences is fixed after the first plan call: "
                    f"expected {self._num_sequences}, got {num_sequences}"
                )
            assert self._cu_seqlens_buf is not None
            assert self._seq_order_buf is not None
            assert self._cu_chunks_buf is not None
            self._cu_seqlens_buf.copy_(cu_seqlens, non_blocking=non_blocking)
            self._workspace.__dict__["_cute_dsl_cu_chunks"] = self._cu_chunks_buf
            self._workspace.__dict__["_cute_dsl_generate_planned_metadata"] = True
            self._planned = True

    @flashinfer_experimental_api
    def run(
        self,
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
        output: Optional[torch.Tensor] = None,
        beta_is_logit: bool = False,
        state_checkpoints: Optional[torch.Tensor] = None,
        checkpoint_cu_starts: Optional[torch.Tensor] = None,
        checkpoint_every_n_tokens: int = 0,
        checkpoint_state_indices: Optional[torch.Tensor] = None,
        ssm_state_indices: Optional[torch.Tensor] = None,
    ) -> (
        tuple[torch.Tensor, Optional[torch.Tensor]]
        | tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ):
        """Run packed recurrent-KDA prefill using the most recent plan."""

        with self._lock:
            if not self._planned:
                raise RuntimeError("call plan before run")
            token_count = q.shape[0] * q.shape[1] if q.ndim == 4 else None
            if self._total_tokens is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm RecurrentKDAPrefillWrapper.run once before CUDA "
                        "graph capture"
                    )
                if token_count is None:
                    raise ValueError("q must be a rank-4 tensor")
                self._total_tokens = token_count
            elif token_count != self._total_tokens:
                raise ValueError(
                    "q token count is fixed after the first run: "
                    f"expected {self._total_tokens}, got "
                    f"{token_count if token_count is not None else 'invalid rank'}"
                )
            cu_seqlens = self._cu_seqlens_buf
            seq_order = self._seq_order_buf
        assert cu_seqlens is not None
        assert seq_order is not None
        return recurrent_kda(
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
            output=output,
            beta_is_logit=beta_is_logit,
            ssm_state_indices=ssm_state_indices,
            seq_order=seq_order,
            prefill_workspace=self._workspace,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_state_indices=checkpoint_state_indices,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            backend="cute-dsl",
        )
