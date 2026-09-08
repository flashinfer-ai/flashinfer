# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host integration for the small-BH CuTe DSL KDA prefill kernel."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .kda_prefill import (
    RecurrentKDAPrefillWorkspace,
    _bind_workspace,
    _cached_packed_task_metadata,
    _check_output_does_not_overlap_inputs,
    _get_stream_workspace,
)
from .utils import get_compute_capability

_HEAD_DIM = 128
_TILE_TOKENS = 64
_CHUNK_TOKENS = 16
_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}


def _is_valid_lower_bound(lower_bound: Optional[float]) -> bool:
    if lower_bound is None:
        return True
    try:
        value = float(lower_bound)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value < 0.0


def _is_kda_prefill_cute_small_bh_eligible(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    seq_order: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    beta_is_logit: bool,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> bool:
    """Whether the small-BH CuTe DSL KDA prefill kernel can serve a call."""

    if not isinstance(q, torch.Tensor) or q.ndim != 4 or q.shape[1] <= 1:
        return False
    if (
        any(
            value is not None
            for value in (
                ssm_state_indices,
                num_spec_tokens,
                num_accepted_tokens,
                initial_state_source,
                initial_state_indices,
                state_checkpoints,
                checkpoint_cu_starts,
            )
        )
        or checkpoint_every_n_tokens != 0
    ):
        return False
    if not (
        use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and beta_is_logit
        and _is_valid_lower_bound(lower_bound)
    ):
        return False
    if (
        not q.is_cuda
        or get_compute_capability(q.device) not in _SUPPORTED_COMPUTE_CAPABILITIES
        or q.dtype != torch.bfloat16
        or not q.is_contiguous()
    ):
        return False

    batch_size, token_count, num_heads, head_dim = q.shape
    if batch_size <= 0 or token_count <= 1 or num_heads <= 0 or head_dim != _HEAD_DIM:
        return False
    for tensor in (k, v, g):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != q.device
            or tensor.dtype != q.dtype
            or tensor.shape != q.shape
            or not tensor.is_contiguous()
        ):
            return False
    if (
        not isinstance(beta, torch.Tensor)
        or beta.device != q.device
        or beta.dtype != q.dtype
        or beta.shape != (batch_size, token_count, num_heads)
        or not beta.is_contiguous()
        or beta.data_ptr() % 16 != 0
    ):
        return False
    if (
        not isinstance(A_log, torch.Tensor)
        or A_log.device != q.device
        or A_log.dtype != torch.float32
        or A_log.shape != (num_heads,)
        or not A_log.is_contiguous()
    ):
        return False
    if (
        not isinstance(dt_bias, torch.Tensor)
        or dt_bias.device != q.device
        or dt_bias.dtype != torch.float32
        or dt_bias.numel() != num_heads * _HEAD_DIM
        or dt_bias.ndim not in (1, 2)
        or not dt_bias.is_contiguous()
    ):
        return False
    if dt_bias.ndim == 2 and dt_bias.shape != (num_heads, _HEAD_DIM):
        return False

    if cu_seqlens is None:
        num_sequences = batch_size
        if seq_order is not None:
            return False
    else:
        if (
            batch_size != 1
            or not isinstance(cu_seqlens, torch.Tensor)
            or cu_seqlens.device != q.device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.numel() <= 1
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1
        if seq_order is not None and (
            not isinstance(seq_order, torch.Tensor)
            or seq_order.device != q.device
            or seq_order.dtype != torch.int32
            or seq_order.shape != (num_sequences,)
            or not seq_order.is_contiguous()
        ):
            return False

    if initial_state is not None and (
        not isinstance(initial_state, torch.Tensor)
        or initial_state.device != q.device
        or initial_state.dtype != q.dtype
        or initial_state.shape != (num_sequences, num_heads, _HEAD_DIM, _HEAD_DIM)
        or not initial_state.is_contiguous()
        or initial_state.data_ptr() % 16 != 0
    ):
        return False
    if output is not None and (
        not isinstance(output, torch.Tensor)
        or output.device != q.device
        or output.dtype != q.dtype
        or output.shape != q.shape
        or not output.is_contiguous()
    ):
        return False
    return True


def _workspace_signature(
    q: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> tuple:
    return (
        q.device.type,
        q.device.index,
        q.dtype,
        tuple(q.shape),
        None if cu_seqlens is None else cu_seqlens.numel(),
    )


def _packed_max_sequence_length(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
) -> int:
    """Resolve the exact packed maximum once, then reuse the workspace cache."""

    metadata_workspace = (
        _get_stream_workspace(q.device)
        if prefill_workspace is None
        else prefill_workspace
    )
    metadata = _cached_packed_task_metadata(
        metadata_workspace,
        cu_seqlens,
        total_tokens=q.shape[0] * q.shape[1],
        num_heads=q.shape[2],
        sm_count=0,
        build_persistent_plan=False,
    )
    return max(metadata[4])


def _allocate_workspace_tensors(
    q: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    matrix_shape: tuple[int, ...]
    gate_shape: tuple[int, ...]
    if cu_seqlens is None:
        batch_size, token_count, num_heads, head_dim = q.shape
        chunk_count = ((token_count + _TILE_TOKENS - 1) // _TILE_TOKENS) * (
            _TILE_TOKENS // _CHUNK_TOKENS
        )
        sequence_shape = q.shape
        matrix_shape = (
            batch_size,
            chunk_count,
            num_heads,
            _CHUNK_TOKENS,
            _CHUNK_TOKENS,
        )
        gate_shape = (batch_size, chunk_count, num_heads, head_dim)
    else:
        _, token_count, num_heads, head_dim = q.shape
        num_sequences = cu_seqlens.numel() - 1
        padded_token_count = token_count + num_sequences * _TILE_TOKENS
        chunk_count = (padded_token_count + _CHUNK_TOKENS - 1) // _CHUNK_TOKENS
        sequence_shape = (padded_token_count, num_heads, head_dim)
        matrix_shape = (
            chunk_count,
            num_heads,
            _CHUNK_TOKENS,
            _CHUNK_TOKENS,
        )
        gate_shape = (chunk_count, num_heads, head_dim)
    return (
        *(
            torch.empty(sequence_shape, dtype=q.dtype, device=q.device)
            for _ in range(3)
        ),
        torch.empty(matrix_shape, dtype=q.dtype, device=q.device),
        torch.empty(matrix_shape, dtype=q.dtype, device=q.device),
        torch.empty(gate_shape, dtype=torch.float32, device=q.device),
    )


def _run_kda_prefill_cute_small_bh(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the small-BH CuTe DSL KDA prefill kernel."""

    from .kda_kernels.kda_chunked_bt16_small_bh import chunk_kda_fwd

    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and prefill_workspace is None:
        raise RuntimeError(
            "CUDA graph capture of backend='small-bh' recurrent_kda prefill "
            "requires an explicit RecurrentKDAPrefillWorkspace warmed with "
            "the exact shape on the capture stream"
        )
    if output is None:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires a preallocated output tensor for "
                "backend='small-bh' recurrent_kda prefill"
            )
        out = torch.empty_like(q)
    else:
        out = output
    _check_output_does_not_overlap_inputs(
        out,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
    )

    scale_value = _HEAD_DIM**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")

    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
    if initial_state is not None:
        state_buffer = initial_state
    elif output_final_state:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires caller-owned initial_state when "
                "output_final_state=True for backend='small-bh'"
            )
        state_buffer = torch.empty(
            num_sequences,
            q.shape[2],
            _HEAD_DIM,
            _HEAD_DIM,
            dtype=q.dtype,
            device=q.device,
        )
    else:
        state_buffer = None

    gate_lower_bound = lower_bound is not None
    gate_scale = float(lower_bound) if gate_lower_bound else 1.0
    signature = _workspace_signature(q, cu_seqlens)
    max_sequence_length = (
        None
        if cu_seqlens is None
        else _packed_max_sequence_length(q, cu_seqlens, prefill_workspace)
    )

    def launch(workspace_tensors: tuple[torch.Tensor, ...]) -> None:
        if cu_seqlens is None:
            q_arg, k_arg, v_arg, g_arg, beta_arg, out_arg = q, k, v, g, beta, out
        else:
            q_arg, k_arg, v_arg, g_arg = q[0], k[0], v[0], g[0]
            beta_arg, out_arg = beta[0], out[0]
        chunk_kda_fwd(
            q_arg,
            k_arg,
            v_arg,
            g_arg,
            beta_arg,
            dt_bias.reshape(q.shape[2], _HEAD_DIM),
            A_log,
            scale=scale_value,
            gate_scale=gate_scale,
            gate_lower_bound=gate_lower_bound,
            return_state=state_buffer is not None,
            transpose_state=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_sequence_length,
            output=out_arg,
            state_input=initial_state,
            state_output=state_buffer,
            workspace_tensors=workspace_tensors,
        )

    if prefill_workspace is None:
        launch(_allocate_workspace_tensors(q, cu_seqlens))
    else:
        with prefill_workspace._lock:
            stream_ptr = int(torch.cuda.current_stream(q.device).cuda_stream)
            _bind_workspace(
                prefill_workspace,
                device=q.device,
                stream_ptr=stream_ptr,
                capturing=capturing,
                explicit=True,
            )
            workspace_tensors = getattr(
                prefill_workspace,
                "_kda_prefill_cute_small_bh_workspace_tensors",
                None,
            )
            warmed_signature = getattr(
                prefill_workspace,
                "_kda_prefill_cute_small_bh_workspace_signature",
                None,
            )
            if workspace_tensors is None or warmed_signature != signature:
                if capturing:
                    raise RuntimeError(
                        "backend='small-bh' prefill workspace is not warmed for "
                        "this CUDA graph shape"
                    )
                workspace_tensors = _allocate_workspace_tensors(q, cu_seqlens)
                prefill_workspace.__dict__[
                    "_kda_prefill_cute_small_bh_workspace_tensors"
                ] = workspace_tensors
                prefill_workspace.__dict__[
                    "_kda_prefill_cute_small_bh_workspace_signature"
                ] = signature
            launch(workspace_tensors)
            if capturing:
                prefill_workspace._captured = True

    return out, state_buffer if output_final_state else None
