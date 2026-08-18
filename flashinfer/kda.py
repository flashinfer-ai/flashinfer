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

from typing import Literal, Optional

import torch

from . import kda_decode as _kda_decode
from . import kda_prefill as _kda_prefill
from . import kda_prefill_cute as _kda_prefill_cute
from .api_logging import flashinfer_api
from .trace.templates.kda import recurrent_kda_trace


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
    *,
    backend: Literal["auto", "cute-dsl", "cake"] = "auto",
) -> (
    tuple[torch.Tensor, Optional[torch.Tensor]]
    | tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
):
    r"""Recurrent KDA (Kimi Delta Attention) decode and prefill kernel.

    This is the public API layer for the CuTe DSL implementation in
    ``flashinfer.kda_kernels.recurrent_kda``. It supports single-token decode,
    fused speculative decode, GQA, optional cu_seqlens packing, and the same
    gate modes as the backend implementation. On SM100a (B200/GB200) and
    SM103a (B300/GB300), the FlashKDA-compatible subset of ordinary multi-token
    prefill can use either the frozen Cake schedules or the source-level CuTe
    DSL BT=16 kernel. ``backend="auto"`` prefers CuTe DSL for supported plain
    prefill contracts and keeps Cake as the feature-complete fallback.

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
            Initial state of shape ``[N, HV, V, K]``. Must be bfloat16.
            If ``None``, zero-initialized. Updated in-place. For batched spec
            decode without ``cu_seqlens``, ``N`` is the packed checkpoint-slot
            count ``B * (1 + num_spec_tokens)`` when ``ssm_state_indices`` is
            omitted. For eligible frozen prefill with ``ssm_state_indices``,
            this is a state pool ``[N_pool, H, 128, 128]`` whose inner slots
            are contiguous; padding between pool slots is allowed.
        output_final_state (bool):
            Whether to return the final state. Default: ``False``.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2 normalization to Q and K. Default: ``True``.
        use_gate_in_kernel (bool):
            Whether to compute the gate inside the kernel from ``A_log`` and
            ``g``. Default: ``False``.
        lower_bound (Optional[float]):
            If set, uses ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
            gate formula instead of softplus. Must be negative.
        cu_seqlens (Optional[torch.Tensor]):
            Contiguous CUDA cumulative sequence lengths of shape ``[N+1]``.
            May be int32 or int64. Frozen prefill converts int32 offsets to
            int64 outside graph capture; graph capture requires caller-provided
            int64 offsets. For frozen prefill, values must start at zero, be
            strictly increasing, and end at the total token count. This value
            contract is not normally host-validated. Eager calls without an
            explicit workspace or ``seq_order`` read these values once per
            unchanged offsets tensor to schedule longer sequences first on
            Cake. CuTe DSL instead generates the order on device every launch;
            eligible 148-SM B200 and 152-SM GB200 Cake calls also cache
            persistent worker task bins.
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
        seq_order (Optional[torch.Tensor]):
            Optional packed-prefill sequence order, as a contiguous CUDA int32
            permutation of shape ``[N]``. It overrides automatic ordering in
            both Cake and CuTe DSL. Without it, CuTe DSL runs a device-side
            stable descending-length sort before each non-persistent launch;
            Cake constructs and caches eager host metadata. On Cake, supplying
            it keeps the direct schedule so caller-owned ordering is not
            replaced by persistent task bins.
            Fixed-layout prefill and decode calls must leave it as ``None``.
        prefill_workspace (Optional[RecurrentKDAPrefillWorkspace]):
            Caller-owned workspace for SM100-family prefill backends.
            It is optional for eager execution and required for CUDA graph
            capture. Warm it eagerly with the exact tensors on the capture
            stream before capture. Use one workspace per captured
            ``recurrent_kda`` invocation. Explicit workspaces and CUDA Graph
            capture use direct/M64 schedules; persistent task planning is an
            eager-only B200/GB200 route because its bins depend on host-visible
            sequence lengths.
        state_checkpoints (Optional[torch.Tensor]):
            Caller-owned BF16 checkpoint output ``[C, H, 128, 128]`` for
            frozen prefill. Row zero for each sequence is its initial state;
            later rows are the states before token blocks beginning at
            ``N, 2N, ...``. Required when ``checkpoint_every_n_tokens > 0``.
        checkpoint_cu_starts (Optional[torch.Tensor]):
            Contiguous CUDA int64 cumulative checkpoint counts ``[N_seq+1]``.
            Each count must equal ``ceil(seq_len / checkpoint_every_n_tokens)``.
        checkpoint_every_n_tokens (int):
            Checkpoint interval. Zero disables checkpoints; a positive value
            must be divisible by 32. SGLang normally uses 64 or a larger
            cache-page-aligned multiple.
        backend (Literal["auto", "cute-dsl", "cake"]):
            Implementation backend. ``"auto"`` selects the ported BT=16
            CuTe DSL kernel for supported ordinary multi-token prefill and
            falls back to an exported frozen Cake specialization for contracts
            such as checkpointing.
            ``"cake"`` and ``"cute-dsl"`` select those backends strictly.

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

    is_plain_prefill = _kda_prefill._is_plain_multi_token_prefill(
        q, cu_seqlens, num_spec_tokens
    )
    cute_dsl_feature_contract = (
        checkpoint_every_n_tokens == 0
        and state_checkpoints is None
        and checkpoint_cu_starts is None
    )
    try_cute_dsl_prefill = backend == "cute-dsl" or (
        backend == "auto" and cute_dsl_feature_contract
    )
    if try_cute_dsl_prefill and is_plain_prefill:
        if (
            checkpoint_every_n_tokens != 0
            or state_checkpoints is not None
            or checkpoint_cu_starts is not None
        ):
            raise ValueError(
                "state checkpoints are not yet supported by backend='cute-dsl'"
            )
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
        )
        if backend == "cute-dsl" and not cute_dsl_eligible:
            raise ValueError(
                "backend='cute-dsl' does not support this recurrent_kda "
                "prefill contract"
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
            )

    use_flash_kda_prefill = (
        backend != "cute-dsl"
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
    if use_flash_kda_prefill:
        assert A_log is not None
        assert dt_bias is not None
        assert lower_bound is not None
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
            backend="cake",
        )

    if backend == "cake" and is_plain_prefill:
        raise ValueError(
            "backend='cake' does not support this recurrent_kda prefill contract"
        )

    if (
        checkpoint_every_n_tokens != 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
    ):
        raise ValueError(
            "state checkpoints are supported only by eligible frozen "
            "SM100/SM103 recurrent_kda prefill"
        )

    if prefill_workspace is not None:
        raise ValueError(
            "prefill_workspace is only supported by eligible ordinary "
            "SM100-family prefill backends"
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
        backend="cake" if backend == "cake" else "cute-dsl",
    )
