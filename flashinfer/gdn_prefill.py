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

import functools
import math
from types import SimpleNamespace
from typing import Optional, Union, Tuple
import torch

from .api_logging import flashinfer_api
from .jit.gdn import gen_gdn_prefill_sm90_module
from .utils import (
    register_custom_op,
    register_fake_op,
    get_device_sm_count,
    is_sm100a_supported,
    _get_cache_buf,
)
from .gdn_kernels import chunk_gated_delta_rule_sm100, _has_blackwell_prefill


@functools.cache
def get_gdn_prefill_module():
    module = gen_gdn_prefill_sm90_module().build_and_load()

    @register_custom_op(
        "flashinfer::gdn_prefill",
        mutates_args=("output", "output_state", "state_checkpoints"),
    )
    def gdn_prefill(
        output: torch.Tensor,
        output_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        scale: float,
        workspace_buffer: torch.Tensor,
        state_checkpoints: Optional[torch.Tensor],
        checkpoint_cu_starts: Optional[torch.Tensor],
        checkpoint_every_n_tokens: int,
    ) -> None:
        module.gdn_prefill(
            output,
            output_state,
            q,
            k,
            v,
            cu_seqlens,
            initial_state,
            g,
            beta,
            scale,
            workspace_buffer,
            state_checkpoints,
            checkpoint_cu_starts,
            checkpoint_every_n_tokens,
        )

    @register_fake_op("flashinfer::gdn_prefill")
    def _fake_gdn_prefill(
        output: torch.Tensor,
        output_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        scale: float,
        workspace_buffer: torch.Tensor,
        state_checkpoints: Optional[torch.Tensor],
        checkpoint_cu_starts: Optional[torch.Tensor],
        checkpoint_every_n_tokens: int,
    ) -> None:
        pass

    return SimpleNamespace(gdn_prefill=gdn_prefill)


@flashinfer_api
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule (GDN) attention for prefill.

    This implements the gated delta rule linear attention mechanism for efficient
    training and inference. Supports both GQA (grouped query attention) and GVA
    (grouped value attention) configurations.

    Args:
        q (torch.Tensor):
            Queries of shape ``[total_seq_len, num_q_heads, head_size]``.
            Must be contiguous and on CUDA.
        k (torch.Tensor):
            Keys of shape ``[total_seq_len, num_k_heads, head_size]``.
            Must be contiguous and on CUDA.
        v (torch.Tensor):
            Values of shape ``[total_seq_len, num_v_heads, head_size]``.
            Must be contiguous and on CUDA.
        g (Optional[torch.Tensor]):
            Forget gate (alpha) of shape ``[total_seq_len, num_sab_heads]`` where
            ``num_sab_heads = max(num_q_heads, num_v_heads)``. Must be float32.
            If None, defaults to all ones. Default: ``None``.
        beta (Optional[torch.Tensor]):
            Update gate (beta) of shape ``[total_seq_len, num_sab_heads]``.
            Must be float32. If None, defaults to all ones. Default: ``None``.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, defaults to ``1 / sqrt(head_size)``. Default: ``None``.
        initial_state (Optional[torch.Tensor]):
            Initial KV state of shape ``[num_seqs, num_sab_heads, head_size, head_size]``.
            Must be float32. If None, starts from zero state. Default: ``None``.
        output_final_state (bool):
            Whether to output the final state. Default: ``False``.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape ``[num_seqs + 1]``, int64.
            Required for variable-length sequences (varlen mode).
        use_qk_l2norm_in_kernel (bool):
            Whether to use QK L2 normalization in kernel. Default: ``False``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[total_seq_len, num_o_heads, head_size]``
            where ``num_o_heads = max(num_q_heads, num_v_heads)``.
            If None, will be allocated automatically. Default: ``None``.
        output_state (Optional[torch.Tensor]):
            Pre-allocated output state tensor of shape
            ``[num_seqs, num_sab_heads, head_size, head_size]``, float32.
            Required if ``output_final_state=True``. Default: ``None``.
        state_checkpoints (Optional[torch.Tensor]):
            Pre-allocated checkpoint tensor of shape
            ``[total_checkpoints, num_sab_heads, head_size, head_size]``, float32.
            Must be provided when ``checkpoint_every_n_tokens > 0``.
            Default: ``None``.
        checkpoint_cu_starts (Optional[torch.Tensor]):
            Cumulative checkpoint counts of shape ``[num_seqs + 1]``, int64.
            ``checkpoint_cu_starts[i+1] - checkpoint_cu_starts[i]`` is the number
            of checkpoints for sequence *i* (= ``seq_len_i // checkpoint_every_n_tokens``).
            Must be provided when ``checkpoint_every_n_tokens > 0``.
            Default: ``None``.
        checkpoint_every_n_tokens (int):
            Store intermediate state every N tokens. Must be a multiple of
            the chunk size (64). 0 means disabled (default).

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If ``output_final_state=False``: Returns output tensor of shape
              ``[total_seq_len, num_o_heads, head_size]``.
            - If ``output_final_state=True``: Returns tuple of (output, final_state) where
              final_state has shape ``[num_seqs, num_sab_heads, head_size, head_size]``.

    Note:
        - Supports GQA: ``num_q_heads > num_k_heads = num_v_heads``
        - Supports GVA: ``num_v_heads > num_q_heads = num_k_heads``
        - The final state layout is ``[N, H, V, K]``.
        - Requires SM90 (Hopper) or SM100 (Blackwell) architecture.
        - SM100 path requires head_size == 128.
        - SM100 path requires ``nvidia-cutlass-dsl[cu13]>=4.4.2``
          (install via ``pip install flashinfer-python[cu13]``).
    """
    if checkpoint_every_n_tokens < 0:
        raise ValueError(
            f"checkpoint_every_n_tokens must be non-negative, "
            f"got {checkpoint_every_n_tokens}"
        )
    if checkpoint_every_n_tokens > 0:
        if checkpoint_every_n_tokens % 64 != 0:
            raise ValueError(
                f"checkpoint_every_n_tokens must be a multiple of the chunk size (64), "
                f"got {checkpoint_every_n_tokens}"
            )
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError(
                "state_checkpoints and checkpoint_cu_starts must both be provided "
                "when checkpoint_every_n_tokens > 0"
            )
    if checkpoint_every_n_tokens == 0 and (
        state_checkpoints is not None or checkpoint_cu_starts is not None
    ):
        raise ValueError(
            "state_checkpoints and checkpoint_cu_starts must be None "
            "when checkpoint_every_n_tokens == 0"
        )

    assert cu_seqlens is not None, "cu_seqlens is required for varlen mode"

    num_seqs = cu_seqlens.size(0) - 1
    total_seq_len = q.size(0)
    num_q_heads = q.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    if checkpoint_every_n_tokens > 0:
        assert state_checkpoints is not None and checkpoint_cu_starts is not None
        if state_checkpoints.dtype != torch.float32:
            raise ValueError(
                f"state_checkpoints must be float32, got {state_checkpoints.dtype}"
            )
        if state_checkpoints.ndim != 4:
            raise ValueError(
                f"state_checkpoints must be 4D "
                f"[total_checkpoints, num_sab_heads, head_size, head_size], "
                f"got {state_checkpoints.ndim}D"
            )
        if checkpoint_cu_starts.dtype != torch.int64:
            raise ValueError(
                f"checkpoint_cu_starts must be int64, got {checkpoint_cu_starts.dtype}"
            )
        if checkpoint_cu_starts.ndim != 1:
            raise ValueError(
                f"checkpoint_cu_starts must be 1D [num_seqs + 1], "
                f"got {checkpoint_cu_starts.ndim}D"
            )
        if checkpoint_cu_starts.size(0) != num_seqs + 1:
            raise ValueError(
                f"checkpoint_cu_starts must have {num_seqs + 1} elements, "
                f"got {checkpoint_cu_starts.size(0)}"
            )
        expected_shape = (
            state_checkpoints.size(0),
            num_sab_heads,
            head_size,
            head_size,
        )
        if tuple(state_checkpoints.shape[1:]) != expected_shape[1:]:
            raise ValueError(
                f"state_checkpoints shape mismatch: expected "
                f"[*, {num_sab_heads}, {head_size}, {head_size}], "
                f"got {list(state_checkpoints.shape)}"
            )

    # Allocate output if not provided
    if output is None:
        output = torch.empty(
            (total_seq_len, num_o_heads, head_size),
            dtype=q.dtype,
            device=q.device,
        )

    # Allocate output_state if needed
    if output_final_state and output_state is None:
        output_state = torch.empty(
            (num_seqs, num_sab_heads, head_size, head_size),
            dtype=torch.float32,
            device=q.device,
        )
    elif not output_final_state and output_state is None:
        # Still need to allocate since kernel always writes state
        output_state = torch.empty(
            (num_seqs, num_sab_heads, head_size, head_size),
            dtype=torch.float32,
            device=q.device,
        )

    device = q.device
    _scale = scale if scale is not None else 1.0 / math.sqrt(head_size)

    _cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    if _has_blackwell_prefill and is_sm100a_supported(device) and _cuda_major >= 13:
        # Blackwell SM100 and SM103 path (CuTe DSL kernel)
        assert head_size == 128, (
            f"Blackwell GDN prefill requires head_size=128, got {head_size}"
        )

        _g = (
            g
            if g is not None
            else torch.ones(
                total_seq_len, num_sab_heads, dtype=torch.float32, device=device
            )
        )
        _beta = (
            beta
            if beta is not None
            else torch.ones(
                total_seq_len, num_sab_heads, dtype=torch.float32, device=device
            )
        )

        # Convert checkpoint_cu_starts from int64 cu_starts to int32 cu_checkpoints
        _cu_checkpoints = None
        if checkpoint_every_n_tokens > 0 and checkpoint_cu_starts is not None:
            _cu_checkpoints = checkpoint_cu_starts.to(torch.int32)

        chunk_gated_delta_rule_sm100(
            q,
            k,
            v,
            _g,
            _beta,
            output,
            cu_seqlens.to(torch.int32),
            initial_state,
            output_state if output_final_state else None,
            _scale,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            cu_checkpoints=_cu_checkpoints,
            output_checkpoints=state_checkpoints,
        )
    else:
        # SM90 Hopper path (C++ JIT kernel)
        workspace_size = get_device_sm_count(device) * 128
        workspace_buffer = _get_cache_buf(
            "gdn_prefill_workspace", workspace_size, device
        )

        get_gdn_prefill_module().gdn_prefill(
            output,
            output_state,
            q,
            k,
            v,
            cu_seqlens.to(torch.int64),
            initial_state,
            g,
            beta,
            scale if scale is not None else 0.0,
            workspace_buffer,
            state_checkpoints,
            checkpoint_cu_starts.to(torch.int64)
            if checkpoint_cu_starts is not None
            else None,
            checkpoint_every_n_tokens,
        )

    if output_final_state:
        return output, output_state
    else:
        return output
