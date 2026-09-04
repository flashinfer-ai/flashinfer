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

"""CuTe DSL BT=16 recurrent-KDA prefill adapter.

This module keeps its FlashInfer-facing validation, allocation, stream, and
state semantics separate from the kernel source.
"""

import math
from typing import Optional

import torch

from .kda_prefill import (
    RecurrentKDAPrefillWorkspace,
    _bind_workspace,
    _check_output_does_not_overlap_inputs,
    _identity_seq_order,
)
from .utils import get_compute_capability

_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
_HEAD_DIM = 128


def _is_cute_dsl_kda_runtime_available() -> bool:
    """Whether the BT=16 kernel can be imported at all.

    It is built on ``cutlass.experimental``, which needs CuTe DSL >= 4.7. The import
    is guarded because a broken DSL install can fail on load.
    """
    try:
        from .cute_dsl.availability import is_cute_dsl_experimental_available
    except (ImportError, RuntimeError):
        return False
    return is_cute_dsl_experimental_available()


def _is_cute_dsl_kda_prefill_dsl_too_old(q: torch.Tensor) -> bool:
    """Whether this device wants the BT=16 kernel but the installed DSL lacks it.

    Architecture-scoped so that devices served by another CuTe DSL prefill backend,
    such as SM120, are not told to upgrade a DSL they do not need.
    """
    return (
        isinstance(q, torch.Tensor)
        and q.is_cuda
        and get_compute_capability(q.device) in _SUPPORTED_COMPUTE_CAPABILITIES
        and not _is_cute_dsl_kda_runtime_available()
    )


def _is_cute_dsl_kda_prefill_eligible(
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
    """Return whether the ported BT=16 kernel can serve this call.

    Covers both the kernel contract and whether the installed CuTe DSL provides it.
    """

    if not isinstance(q, torch.Tensor) or q.ndim != 4 or q.shape[1] <= 1:
        return False
    if num_spec_tokens is not None:
        return False
    if any(
        value is not None
        for value in (
            num_accepted_tokens,
            initial_state_source,
            initial_state_indices,
        )
    ):
        return False
    if not (
        use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and beta_is_logit
        and lower_bound is not None
        and math.isfinite(float(lower_bound))
        and float(lower_bound) < 0.0
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
            or tensor.dtype != torch.bfloat16
            or tensor.shape != q.shape
            or not tensor.is_contiguous()
        ):
            return False
    if (
        not isinstance(beta, torch.Tensor)
        or beta.device != q.device
        or beta.dtype != torch.bfloat16
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
        or cu_seqlens is None
        or seq_order.device != q.device
        or seq_order.dtype != torch.int32
        or seq_order.ndim != 1
        or not seq_order.is_contiguous()
        or seq_order.numel() != num_sequences
    ):
        return False

    if ssm_state_indices is not None:
        if (
            initial_state is None
            or not isinstance(ssm_state_indices, torch.Tensor)
            or ssm_state_indices.device != q.device
            or ssm_state_indices.dtype != torch.int32
            or ssm_state_indices.ndim != 1
            or ssm_state_indices.numel() != num_sequences
            or not ssm_state_indices.is_contiguous()
        ):
            return False
    if initial_state is not None:
        if (
            not isinstance(initial_state, torch.Tensor)
            or initial_state.device != q.device
            or initial_state.dtype != torch.bfloat16
            or initial_state.ndim != 4
            or initial_state.shape[0] <= 0
            or tuple(initial_state.shape[1:]) != (num_heads, _HEAD_DIM, _HEAD_DIM)
            or initial_state.data_ptr() % 16 != 0
            or initial_state.stride(-1) != 1
            or initial_state.stride(-2) != _HEAD_DIM
            or initial_state.stride(-3) != _HEAD_DIM * _HEAD_DIM
            or initial_state.stride(0) < num_heads * _HEAD_DIM * _HEAD_DIM
            or initial_state.stride(0) * initial_state.element_size() % 16 != 0
        ):
            return False
        if ssm_state_indices is None and initial_state.shape[0] != num_sequences:
            return False
    if output is not None and (
        not isinstance(output, torch.Tensor)
        or output.device != q.device
        or output.dtype != torch.bfloat16
        or output.shape != q.shape
        or not output.is_contiguous()
    ):
        return False
    if (
        checkpoint_every_n_tokens < 0
        or checkpoint_every_n_tokens > torch.iinfo(torch.int32).max
        or checkpoint_every_n_tokens % 32 != 0
    ):
        return False
    if checkpoint_every_n_tokens:
        if (
            not isinstance(state_checkpoints, torch.Tensor)
            or state_checkpoints.device != q.device
            or state_checkpoints.dtype != torch.bfloat16
            or state_checkpoints.ndim != 4
            or tuple(state_checkpoints.shape[1:]) != (num_heads, _HEAD_DIM, _HEAD_DIM)
            or state_checkpoints.shape[0] > torch.iinfo(torch.int32).max
            or not state_checkpoints.is_contiguous()
            or not isinstance(checkpoint_cu_starts, torch.Tensor)
            or checkpoint_cu_starts.device != q.device
            or checkpoint_cu_starts.dtype != torch.int64
            or checkpoint_cu_starts.ndim != 1
            or checkpoint_cu_starts.numel() != num_sequences + 1
            or not checkpoint_cu_starts.is_contiguous()
        ):
            return False
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        return False
    # Probed last so that calls rejected above reach Cake exactly as before.
    return _is_cute_dsl_kda_runtime_available()


def _get_compiled_cute_dsl_kda(
    *,
    lower_bound: float,
    has_state_in: bool,
    has_state_out: bool,
    has_state_ckpt: bool,
    has_state_indices: bool,
):
    # Keep the large CuTe DSL module lazy so normal Cake and decode imports do
    # not initialize its compilation stack.
    import cutlass

    from .kda_kernels.kda_chunked_bt16 import compile

    return compile(
        dtype=cutlass.BFloat16,
        state_dtype=cutlass.BFloat16,
        gate_dtype=cutlass.BFloat16,
        safe_gate=True,
        gate_lower_bound=lower_bound,
        has_state_in=has_state_in,
        has_state_out=has_state_out,
        has_state_ckpt=has_state_ckpt,
        has_state_indices=has_state_indices,
        mode=None,
    )


def _run_cute_dsl_kda_prefill(
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
    lower_bound: float,
    cu_seqlens: Optional[torch.Tensor],
    seq_order: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    state_indices: Optional[torch.Tensor] = None,
) -> (
    tuple[torch.Tensor, Optional[torch.Tensor]]
    | tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
):
    """Launch the CuTe DSL BT=16 prefill kernel on the current stream."""

    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and prefill_workspace is None:
        raise RuntimeError(
            "CUDA graph capture of backend='cute-dsl' recurrent_kda prefill "
            "requires an explicit RecurrentKDAPrefillWorkspace warmed with "
            "the exact tensors on the capture stream"
        )
    if output is None:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires a preallocated output tensor for "
                "backend='cute-dsl' recurrent_kda prefill"
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
    if seq_order is None and cu_seqlens is None:
        seq_order = _identity_seq_order(
            device=q.device,
            num_sequences=num_sequences,
        )
    if initial_state is not None:
        final_state = initial_state
    elif output_final_state:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires caller-owned initial_state when "
                "output_final_state=True for backend='cute-dsl'"
            )
        final_state = torch.empty(
            num_sequences,
            q.shape[2],
            _HEAD_DIM,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        final_state = None

    if cu_seqlens is not None and cu_seqlens.dtype != torch.int64:
        if capturing:
            raise RuntimeError(
                "packed backend='cute-dsl' prefill requires int64 cu_seqlens "
                "during CUDA graph capture"
            )
        cu_seqlens = cu_seqlens.to(torch.int64)

    compiled = _get_compiled_cute_dsl_kda(
        lower_bound=float(lower_bound),
        has_state_in=initial_state is not None,
        has_state_out=final_state is not None,
        has_state_ckpt=state_checkpoints is not None,
        has_state_indices=state_indices is not None,
    )
    planned_cu_chunks = (
        getattr(prefill_workspace, "_cute_dsl_cu_chunks", None)
        if prefill_workspace is not None
        else None
    )
    planned_total_chunks = (
        getattr(prefill_workspace, "_cute_dsl_total_chunks", None)
        if prefill_workspace is not None
        else None
    )
    if (planned_cu_chunks is None) != (planned_total_chunks is None):
        raise RuntimeError("incomplete CuTe DSL chunk plan on prefill workspace")
    if planned_cu_chunks is not None:
        workspace_bytes = compiled.workspace_size_from_total_chunks(
            num_sequences,
            q.shape[2],
            planned_total_chunks,
            q.device,
        )
    elif cu_seqlens is None:
        workspace_bytes = compiled.workspace_size(
            None,
            q.shape[2],
            batch=q.shape[0],
            seqlen=q.shape[1],
        )
    else:
        workspace_bytes = compiled.workspace_size(cu_seqlens, q.shape[2])

    workspace_owner = prefill_workspace
    if workspace_owner is None:
        workspace = (
            torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
            if workspace_bytes
            else None
        )
        lock = None
    else:
        lock = workspace_owner._lock
        workspace = None

    def launch(workspace_arg: Optional[torch.Tensor]) -> None:
        checkpoint_kwargs = {}
        if checkpoint_every_n_tokens:
            checkpoint_kwargs = {
                "state_ckpt": state_checkpoints,
                "checkpoint_cu_starts": checkpoint_cu_starts,
                "ckpt_interval": checkpoint_every_n_tokens,
            }
        compiled(
            q,
            k,
            v,
            g,
            A_log,
            dt_bias.reshape(q.shape[2], _HEAD_DIM),
            beta,
            cu_seqlens,
            initial_state,
            out,
            final_state,
            workspace_arg,
            torch.cuda.current_stream(q.device).cuda_stream,
            scale_value,
            state_indices=state_indices,
            seq_order=seq_order,
            planned_cu_chunks=planned_cu_chunks,
            planned_total_chunks=planned_total_chunks,
            **checkpoint_kwargs,
        )

    if lock is None:
        launch(workspace)
    else:
        with lock:
            stream_ptr = int(torch.cuda.current_stream(q.device).cuda_stream)
            _bind_workspace(
                workspace_owner,
                device=q.device,
                stream_ptr=stream_ptr,
                capturing=capturing,
                explicit=True,
            )
            workspace = getattr(workspace_owner, "_cute_dsl_workspace", None)
            if workspace_bytes and (
                workspace is None or workspace.numel() < workspace_bytes
            ):
                if capturing:
                    raise RuntimeError(
                        "backend='cute-dsl' prefill workspace is not large enough "
                        "for CUDA graph capture; warm the largest shape first"
                    )
                workspace = torch.empty(
                    workspace_bytes, dtype=torch.uint8, device=q.device
                )
                workspace_owner.__dict__["_cute_dsl_workspace"] = workspace
            if workspace is not None:
                workspace = workspace[:workspace_bytes]
            launch(workspace)
            if capturing:
                workspace_owner._captured = True
    result = (out, final_state if output_final_state else None)
    if checkpoint_every_n_tokens:
        assert state_checkpoints is not None
        return (*result, state_checkpoints)
    return result


__all__ = [
    "_is_cute_dsl_kda_prefill_dsl_too_old",
    "_is_cute_dsl_kda_prefill_eligible",
    "_is_cute_dsl_kda_runtime_available",
    "_run_cute_dsl_kda_prefill",
]
