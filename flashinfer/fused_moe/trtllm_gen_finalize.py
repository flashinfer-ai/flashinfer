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
from typing import Optional, Tuple

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit.fused_moe import gen_trtllm_gen_moe_finalize_module
from flashinfer.trace.templates.moe import trtllm_gen_moe_finalize_trace
from flashinfer.utils import (
    backend_requirement,
    device_support_pdl,
    register_custom_op,
    supported_compute_capability,
)

# The trtllm_gen_moe_finalize module is compiled for SM 10.x and 12.x only
# (same gate as the fused_moe_trtllm_sm100 module the finalize kernels ship in).
_TRTLLM_GEN_MOE_FINALIZE_SUPPORTED_CC = [100, 103, 120, 121]

# The vectorized finalize kernel loads 128 bits per thread; both supported
# element dtypes (bfloat16 / float16) are 16-bit.
_ELEMS_PER_128B = 8


@supported_compute_capability(_TRTLLM_GEN_MOE_FINALIZE_SUPPORTED_CC)
def _check_trtllm_gen_moe_finalize_supported(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    *,
    lora_delta: Optional[torch.Tensor] = None,
    lora_delta_scale: float = 1.0,
    lora_apply_expert_weights: bool = False,
    hidden_size: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> bool:
    if gemm2_output.dim() != 2:
        raise ValueError(
            f"gemm2_output must be 2D [num_padded, hidden_padded], "
            f"got {tuple(gemm2_output.shape)}"
        )
    if gemm2_output.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"gemm2_output must be bfloat16 or float16, got {gemm2_output.dtype}"
        )
    if expert_weights.dim() != 2:
        raise ValueError(
            f"expert_weights must be 2D [num_tokens, top_k], "
            f"got {tuple(expert_weights.shape)}"
        )
    if expert_weights.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"expert_weights must be bfloat16 or float32, got {expert_weights.dtype}"
        )
    num_tokens, top_k = expert_weights.shape
    if not 1 <= top_k <= 64:
        raise ValueError(
            f"top_k must be in [1, 64] (the vectorized finalize kernel's "
            f"shared-memory bound), got {top_k}"
        )
    if expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise ValueError(
            f"expanded_idx_to_permuted_idx must be int32, "
            f"got {expanded_idx_to_permuted_idx.dtype}"
        )
    if expanded_idx_to_permuted_idx.numel() != num_tokens * top_k:
        raise ValueError(
            f"expanded_idx_to_permuted_idx must have num_tokens * top_k "
            f"({num_tokens * top_k}) elements, "
            f"got {expanded_idx_to_permuted_idx.numel()}"
        )
    hidden = hidden_size if hidden_size is not None else gemm2_output.shape[1]
    if not 0 < hidden <= gemm2_output.shape[1]:
        raise ValueError(
            f"hidden_size must be in (0, gemm2_output.shape[1]], got {hidden}"
        )
    if hidden % _ELEMS_PER_128B or gemm2_output.shape[1] % _ELEMS_PER_128B:
        raise ValueError(
            f"hidden dimensions must be multiples of {_ELEMS_PER_128B}, got "
            f"{hidden} and {gemm2_output.shape[1]}"
        )
    if lora_delta is not None:
        if lora_delta.dtype != gemm2_output.dtype:
            raise ValueError(
                f"lora_delta dtype ({lora_delta.dtype}) must match "
                f"gemm2_output dtype ({gemm2_output.dtype})"
            )
        expected: Tuple[int, ...]
        if lora_delta.dim() == 2:
            expected = (num_tokens, hidden)
        elif lora_delta.dim() == 3:
            expected = (num_tokens, top_k, hidden)
        else:
            raise ValueError(
                f"lora_delta must be 2D [num_tokens, hidden] or 3D "
                f"[num_tokens, top_k, hidden], got {tuple(lora_delta.shape)}"
            )
        if tuple(lora_delta.shape) != expected:
            raise ValueError(
                f"lora_delta shape {tuple(lora_delta.shape)} does not match "
                f"expected {expected}"
            )
        if lora_apply_expert_weights and lora_delta.dim() != 3:
            raise ValueError(
                "lora_apply_expert_weights requires the 3D "
                "[num_tokens, top_k, hidden] lora_delta layout"
            )
    elif lora_apply_expert_weights:
        raise ValueError("lora_apply_expert_weights requires lora_delta")
    if out is not None:
        if out.shape != (num_tokens, hidden):
            raise ValueError(
                f"out must have shape ({num_tokens}, {hidden}), got {tuple(out.shape)}"
            )
        if out.dtype != gemm2_output.dtype:
            raise ValueError(
                f"out dtype ({out.dtype}) must match gemm2_output dtype "
                f"({gemm2_output.dtype})"
            )
    return True


@functools.cache
def get_trtllm_gen_moe_finalize_module():
    """Build, load, and cache the trtllm_gen_moe_finalize JIT module."""
    module = gen_trtllm_gen_moe_finalize_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_gen_moe_finalize",
        mutates_args=["out"],
    )
    def trtllm_gen_moe_finalize(
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        out: torch.Tensor,
        lora_delta: Optional[torch.Tensor],
        lora_delta_scale: float,
        lora_apply_expert_weights: bool,
        enable_pdl: bool,
    ) -> None:
        module.trtllm_gen_moe_finalize(
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            out,
            lora_delta,
            lora_delta_scale,
            lora_apply_expert_weights,
            enable_pdl,
        )

    return SimpleNamespace(trtllm_gen_moe_finalize=trtllm_gen_moe_finalize)


@backend_requirement({}, common_check=_check_trtllm_gen_moe_finalize_supported)
@flashinfer_api(trace=trtllm_gen_moe_finalize_trace)
def trtllm_gen_moe_finalize(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    *,
    lora_delta: Optional[torch.Tensor] = None,
    lora_delta_scale: float = 1.0,
    lora_apply_expert_weights: bool = False,
    hidden_size: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Standalone trtllm-gen MoE finalize (token combine), with optional fused delta.

    Runs the same finalize kernels the trtllm-gen fused MoE launchers execute
    when ``do_finalize=True`` (``moe::dev::finalize::run``): gathers the
    permuted FC2 output rows through ``expanded_idx_to_permuted_idx``, applies
    the expert weights, and reduces over the ``top_k`` slots. This completes
    the documented ``do_finalize=False`` contract of the routed MoE ops as a
    separate op, and can fuse a per-token delta — e.g. a LoRA down-projection
    delta produced by :func:`bgmv_moe_gemm2_lora_delta` machinery — into the
    combine, removing the extra full-output read-modify-write pass a separate
    addition would cost.

    .. math::
        \text{out}[t] = \sum_k w[t, k] \cdot \text{gemm2\_output}[\pi(t, k)]
        + s \cdot \sum_k \sigma[t, k] \cdot \Delta[t, k]

    where :math:`\pi` is ``expanded_idx_to_permuted_idx`` (slots with
    :math:`\pi = -1` are skipped in the first sum only), :math:`s` is
    ``lora_delta_scale`` and :math:`\sigma[t, k]` is ``expert_weights`` when
    ``lora_apply_expert_weights`` else 1.

    Parameters
    ----------
    gemm2_output : torch.Tensor
        Permuted, unfinalized FC2 output of shape
        ``(num_padded, hidden_padded)``, ``bfloat16`` or ``float16`` — the
        first tensor returned by the routed MoE ops with
        ``do_finalize=False``.
    expert_weights : torch.Tensor
        Per-slot routing weights of shape ``(num_tokens, top_k)``,
        ``bfloat16`` or ``float32``.
    expanded_idx_to_permuted_idx : torch.Tensor
        ``int32`` tensor with ``num_tokens * top_k`` elements mapping each
        expanded (token, slot) index to its permuted row; ``-1`` marks
        inactive slots (e.g. experts outside the local expert-parallel
        shard).
    lora_delta : Optional[torch.Tensor]
        Optional delta fused into the combine, same dtype as
        ``gemm2_output``. Either ``(num_tokens, top_k, hidden)`` with one row
        per slot, or ``(num_tokens, hidden)`` pre-combined. Rows are
        addressed by the expanded index, NOT the permutation, and are
        accumulated unconditionally — rows of inactive slots must be
        zero-filled by the producer.
    lora_delta_scale : float
        Scalar factor applied to the delta sum (e.g. a routed scaling
        factor).
    lora_apply_expert_weights : bool
        Multiply each per-slot delta row by its expert weight. Requires the
        3D ``lora_delta`` layout.
    hidden_size : Optional[int]
        Unpadded output hidden size. Defaults to ``gemm2_output.shape[1]``
        (i.e. no padding). Must be a multiple of 8.
    out : Optional[torch.Tensor]
        Preallocated output of shape ``(num_tokens, hidden_size)``, same
        dtype as ``gemm2_output``. Allocated if not given.
    enable_pdl : Optional[bool]
        Whether to launch with programmatic dependent launch. Defaults to
        auto-detection. With PDL enabled the kernel prefetches
        ``expert_weights`` and ``expanded_idx_to_permuted_idx`` before its
        grid-dependency sync (matching the fused pipeline, where routing
        completed several kernels earlier) — pass ``False`` if either tensor
        is produced by the immediately preceding kernel on the stream.

    Returns
    -------
    torch.Tensor
        The combined MoE output of shape ``(num_tokens, hidden_size)``.

    Notes
    -----
    ``top_k`` must be at most 64 (the vectorized kernel's shared-memory
    bound). All tensors must be 16-byte aligned (fresh torch allocations
    are; contiguous-but-offset views may not be). ``num_tokens == 0`` is a
    no-op.
    """
    num_tokens = expert_weights.shape[0]
    hidden = hidden_size if hidden_size is not None else gemm2_output.shape[1]
    if out is None:
        out = torch.empty(
            (num_tokens, hidden), dtype=gemm2_output.dtype, device=gemm2_output.device
        )
    if enable_pdl is None:
        enable_pdl = device_support_pdl(gemm2_output.device)

    get_trtllm_gen_moe_finalize_module().trtllm_gen_moe_finalize(
        gemm2_output.contiguous(),
        expert_weights.contiguous(),
        expanded_idx_to_permuted_idx.contiguous(),
        out,
        lora_delta.contiguous() if lora_delta is not None else None,
        float(lora_delta_scale),
        lora_apply_expert_weights,
        enable_pdl,
    )
    return out
