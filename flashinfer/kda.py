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

from typing import Optional

import torch

from . import kda_decode as _kda_decode
from . import kda_prefill as _kda_prefill
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
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Recurrent KDA (Kimi Delta Attention) decode and prefill kernel.

    This is the public API layer for the CuTe DSL implementation in
    ``flashinfer.kda_kernels.recurrent_kda``. It supports single-token decode,
    fused speculative decode, GQA, optional cu_seqlens packing, and the same
    gate modes as the backend implementation. On NVIDIA B200, the exact
    FlashKDA-compatible subset of ordinary multi-token prefill is dispatched to
    frozen SM100a kernels. All existing decode and speculative-decode calls
    retain the CuTe DSL backend.

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
            Pre-sigmoided unless ``beta_is_logit=True``.
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
            gate formula instead of softplus. Must be negative.
        cu_seqlens (Optional[torch.Tensor]):
            Contiguous CUDA cumulative sequence lengths of shape ``[N+1]``.
            May be int32 or int64. Frozen prefill converts int32 offsets to
            int64 outside graph capture; graph capture requires caller-provided
            int64 offsets. For frozen prefill, values must start at zero, be
            strictly increasing, and end at the total token count. This value
            contract is not host-validated to avoid a device synchronization.
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
            permutation of shape ``[N]``. Sorting by descending sequence length
            improves tail utilization. It is only consumed by the frozen
            FlashKDA prefill backend; prepare it before CUDA graph capture or
            timed launches. Fixed-layout prefill and decode calls must leave it
            as ``None``.
        prefill_workspace (Optional[RecurrentKDAPrefillWorkspace]):
            Caller-owned workspace for the frozen B200 prefill backend. It is
            optional for eager execution and required for CUDA graph capture.
            Warm it eagerly with the exact tensors on the capture stream before
            capture. Use one workspace per captured ``recurrent_kda``
            invocation.

    Returns:
        Tuple of ``(output, final_state)`` where ``final_state`` is ``None``
        when ``output_final_state=False``. See
        :func:`flashinfer.kda_kernels.recurrent_kda.run_recurrent_kda` for the
        backend implementation.
    """
    if prefill_workspace is not None and not isinstance(
        prefill_workspace, _kda_prefill.RecurrentKDAPrefillWorkspace
    ):
        raise TypeError("prefill_workspace must be a RecurrentKDAPrefillWorkspace")

    use_flash_kda_prefill = _kda_prefill._flash_kda_prefill_is_eligible(
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
        )

    if prefill_workspace is not None:
        raise ValueError(
            "prefill_workspace is only supported by eligible ordinary "
            "prefill on the frozen B200 FlashKDA backend"
        )
    if seq_order is not None:
        raise ValueError(
            "seq_order is only supported by eligible packed ordinary prefill "
            "on the frozen B200 FlashKDA backend"
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
    )
