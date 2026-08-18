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

The kernel implementation is ported from DKG MR 26001.  This module keeps its
FlashInfer-facing validation, allocation, stream, and state semantics separate
from the generated-style kernel source.
"""

import math
from typing import Optional

import torch

from .kda_prefill import (
    RecurrentKDAPrefillWorkspace,
    _bind_workspace,
    _check_output_does_not_overlap_inputs,
)
from .utils import get_compute_capability

_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
_HEAD_DIM = 128


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
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    beta_is_logit: bool,
) -> bool:
    """Return whether the call matches the ported BT=16 kernel contract."""

    if not isinstance(q, torch.Tensor) or q.ndim != 4 or q.shape[1] <= 1:
        return False
    if num_spec_tokens is not None:
        return False
    if any(
        value is not None
        for value in (
            ssm_state_indices,
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
        beta.device != q.device
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
            or cu_seqlens.device != q.device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.numel() <= 1
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1

    if initial_state is not None and (
        initial_state.device != q.device
        or initial_state.dtype != torch.bfloat16
        or initial_state.shape != (num_sequences, num_heads, _HEAD_DIM, _HEAD_DIM)
        or not initial_state.is_contiguous()
    ):
        return False
    if output is not None and (
        output.device != q.device
        or output.dtype != torch.bfloat16
        or output.shape != q.shape
        or not output.is_contiguous()
    ):
        return False
    return True


def _get_compiled_cute_dsl_kda(
    *, lower_bound: float, has_state_in: bool, has_state_out: bool
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
        has_state_ckpt=False,
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
    output: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the ported DKG BT=16 prefill kernel on the current stream."""

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
    )
    if cu_seqlens is not None:
        from .kda_kernels.kda_chunked_bt16 import _cu_seqlens_contents

        offsets = _cu_seqlens_contents(cu_seqlens)
        if (
            not offsets
            or offsets[0] != 0
            or offsets[-1] != q.shape[1]
            or any(
                right <= left
                for left, right in zip(offsets[:-1], offsets[1:], strict=True)
            )
        ):
            raise ValueError(
                "cu_seqlens must start at zero, be strictly increasing, and "
                "end at the packed token count"
            )
    if cu_seqlens is None:
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
            workspace = workspace_owner._cute_dsl_workspace
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
                workspace_owner._cute_dsl_workspace = workspace
            if workspace is not None:
                workspace = workspace[:workspace_bytes]
            launch(workspace)
            if capturing:
                workspace_owner._captured = True
    return out, final_state if output_final_state else None


__all__ = [
    "_is_cute_dsl_kda_prefill_eligible",
    "_run_cute_dsl_kda_prefill",
]
