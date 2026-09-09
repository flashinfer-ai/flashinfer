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

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit.fused_moe import gen_trtllm_gen_moe_gather_activation_module
from flashinfer.trace.templates.moe import trtllm_gen_moe_gather_activation_trace
from flashinfer.utils import (
    backend_requirement,
    device_support_pdl,
    register_custom_op,
    supported_compute_capability,
)

# The trtllm_gen_moe_gather_activation module is compiled for SM 10.x and 12.x
# only (same gate as the fused_moe_trtllm_sm100 module the dev kernels ship in).
_TRTLLM_GEN_MOE_GATHER_ACTIVATION_SUPPORTED_CC = [100, 103, 120, 121]


@supported_compute_capability(_TRTLLM_GEN_MOE_GATHER_ACTIVATION_SUPPORTED_CC)
def _check_trtllm_gen_moe_gather_activation_supported(
    activation_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    top_k: int,
    *,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> bool:
    if activation_output.dim() != 2:
        raise ValueError(
            f"activation_output must be 2D [num_padded_rows, intermediate_size], "
            f"got {tuple(activation_output.shape)}"
        )
    if activation_output.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"activation_output must be bfloat16 or float16, "
            f"got {activation_output.dtype}"
        )
    if expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise ValueError(
            f"expanded_idx_to_permuted_idx must be int32, "
            f"got {expanded_idx_to_permuted_idx.dtype}"
        )
    if expanded_idx_to_permuted_idx.device != activation_output.device:
        raise ValueError(
            f"expanded_idx_to_permuted_idx must be on the same device as "
            f"activation_output ({expanded_idx_to_permuted_idx.device} vs "
            f"{activation_output.device})"
        )
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    num_slots = expanded_idx_to_permuted_idx.numel()
    if num_slots % top_k:
        raise ValueError(
            f"expanded_idx_to_permuted_idx must hold a whole number of tokens: "
            f"{num_slots} elements is not a multiple of top_k ({top_k})"
        )
    if out is not None:
        expected = (num_slots // top_k, top_k, activation_output.shape[1])
        if tuple(out.shape) != expected:
            raise ValueError(f"out must have shape {expected}, got {tuple(out.shape)}")
        if out.dtype != activation_output.dtype:
            raise ValueError(
                f"out dtype ({out.dtype}) must match activation_output dtype "
                f"({activation_output.dtype})"
            )
        if out.device != activation_output.device:
            raise ValueError(
                f"out must be on the same device as activation_output "
                f"({out.device} vs {activation_output.device})"
            )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    return True


@functools.cache
def get_trtllm_gen_moe_gather_activation_module():
    """Build, load, and cache the trtllm_gen_moe_gather_activation JIT module."""
    module = gen_trtllm_gen_moe_gather_activation_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_gen_moe_gather_activation",
        mutates_args=["out"],
    )
    def trtllm_gen_moe_gather_activation(
        activation_output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        out: torch.Tensor,
        enable_pdl: bool,
    ) -> None:
        module.trtllm_gen_moe_gather_activation(
            activation_output,
            expanded_idx_to_permuted_idx,
            out,
            enable_pdl,
        )

    return SimpleNamespace(
        trtllm_gen_moe_gather_activation=trtllm_gen_moe_gather_activation
    )


@backend_requirement({}, common_check=_check_trtllm_gen_moe_gather_activation_supported)
@flashinfer_api(trace=trtllm_gen_moe_gather_activation_trace)
def trtllm_gen_moe_gather_activation(
    activation_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    top_k: int,
    *,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Gather the trtllm-gen MoE post-activation FC1 output back into expanded order.

    The routed MoE ops hand back the post-SwiGLU FC1 output
    (``gemm1_activation_output``) in the kernels' permuted layout, plus the
    ``expanded_idx_to_permuted_idx`` map. Any consumer of that tensor — a LoRA
    down-projection delta such as
    :func:`flashinfer.fused_moe.bgmv_moe_gemm2_lora_delta`, an expert-level
    probe — first has to undo the permutation. This op does that in a single
    copy kernel:

    .. math::
        \text{out}[t, k] = \begin{cases}
            \text{activation\_output}[\pi(t, k)] & \pi(t, k) \ge 0 \\
            0 & \text{otherwise}
        \end{cases}

    where :math:`\pi` is ``expanded_idx_to_permuted_idx``. Negative entries mark
    slots routed to an expert outside the local expert-parallel shard; their
    output rows are exactly zero.

    Parameters
    ----------
    activation_output : torch.Tensor
        Permuted post-activation FC1 output of shape
        ``(num_padded_rows, intermediate_size)``, ``bfloat16`` or ``float16``.
        The row count is the routing-padded count, which is larger than
        ``num_tokens * top_k``; padding rows are never read.
    expanded_idx_to_permuted_idx : torch.Tensor
        ``int32`` tensor with ``num_tokens * top_k`` elements (any shape)
        mapping each expanded ``(token, slot)`` index to its permuted row;
        negative entries mark inactive slots.
    top_k : int
        Number of routed slots per token, i.e. the width of the map — with
        fused shared experts that is ``top_k + num_fused_shared_experts``.
        ``num_tokens`` is derived as
        ``expanded_idx_to_permuted_idx.numel() // top_k``.
    out : Optional[torch.Tensor]
        Preallocated output of shape ``(num_tokens, top_k, intermediate_size)``,
        same dtype as ``activation_output``. Allocated if not given.
    enable_pdl : Optional[bool]
        Whether to launch with programmatic dependent launch. Defaults to
        auto-detection. The kernel reads every input after its grid-dependency
        sync, so enabling it is safe for a producer on the same stream.

    Returns
    -------
    torch.Tensor
        The gathered activation of shape
        ``(num_tokens, top_k, intermediate_size)``.

    Notes
    -----
    The permuted indices are only bounds-checked by a device-side assert, so a
    release build with an index at or past ``num_padded_rows`` reads out of
    bounds; the equivalent torch gather raises instead.
    ``num_tokens == 0`` is a no-op.
    """
    num_tokens = expanded_idx_to_permuted_idx.numel() // top_k
    intermediate_size = activation_output.shape[1]
    if out is None:
        out = torch.empty(
            (num_tokens, top_k, intermediate_size),
            dtype=activation_output.dtype,
            device=activation_output.device,
        )
    if enable_pdl is None:
        enable_pdl = device_support_pdl(activation_output.device)

    get_trtllm_gen_moe_gather_activation_module().trtllm_gen_moe_gather_activation(
        activation_output.contiguous(),
        expanded_idx_to_permuted_idx.contiguous(),
        out,
        enable_pdl,
    )
    return out
