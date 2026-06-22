"""Layout bridge between EP dispatch output and the unified MoE compute API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ......fused_moe.api import MoEActivationPack


def build_activation_pack(
    expert_tensors: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    is_nvfp4: bool,
    global_scale: Optional[torch.Tensor] = None,
) -> "MoEActivationPack":
    """Translate the 3D expert-major dispatch output into a token-major pack."""
    if expert_tensors.dim() != 3:
        raise ValueError(
            "build_activation_pack expects a 3D [num_local_experts, cap, hidden] "
            f"dispatch tensor, got shape {tuple(expert_tensors.shape)}"
        )
    num_local_experts, cap, hidden = expert_tensors.shape
    flat = expert_tensors.reshape(num_local_experts * cap, hidden)
    m = flat.shape[0]
    device = flat.device

    row_expert = torch.arange(num_local_experts, device=device, dtype=torch.int32)
    selected_experts = (
        row_expert.repeat_interleave(cap).reshape(m, 1) + local_expert_offset
    )
    final_scales = torch.ones(m, 1, dtype=torch.float32, device=device)

    return _quantize_and_pack(
        flat,
        selected_experts,
        final_scales,
        is_nvfp4=is_nvfp4,
        global_scale=global_scale,
    )


def build_activation_pack_rank_major(
    recv_tensors: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
    local_expert_offset: int = 0,
    is_nvfp4: bool,
    global_scale: Optional[torch.Tensor] = None,
) -> "MoEActivationPack":
    """Translate the 3D RANK_MAJOR dispatch output into a token-major pack."""
    if recv_tensors.dim() != 3:
        raise ValueError(
            "build_activation_pack_rank_major expects a 3D [world, "
            "max_tokens_per_rank, hidden] dispatch tensor, got shape "
            f"{tuple(recv_tensors.shape)}"
        )
    d0, d1, hidden = recv_tensors.shape
    flat = recv_tensors.reshape(d0 * d1, hidden)
    m = flat.shape[0]

    idx = recv_topk_idx
    if idx.dtype != torch.int64:
        idx = idx.to(torch.int64)
    weights = recv_topk_weights
    if weights.dtype != torch.float32:
        weights = weights.to(torch.float32)
    if idx.shape[0] != m or weights.shape[0] != m:
        raise ValueError(
            f"recv_topk_idx/weights row count ({idx.shape[0]}/{weights.shape[0]}) "
            f"must match the flattened recv token count ({m})."
        )

    is_local = idx >= 0
    selected_experts = torch.where(
        is_local,
        idx + local_expert_offset,
        torch.full_like(idx, local_expert_offset),
    ).to(torch.int32)
    final_scales = torch.where(is_local, weights, torch.zeros_like(weights))

    return _quantize_and_pack(
        flat,
        selected_experts,
        final_scales,
        is_nvfp4=is_nvfp4,
        global_scale=global_scale,
    )


def _quantize_and_pack(
    flat: torch.Tensor,
    selected_experts: torch.Tensor,
    final_scales: torch.Tensor,
    *,
    is_nvfp4: bool,
    global_scale: Optional[torch.Tensor],
) -> "MoEActivationPack":
    from ......fused_moe.api import MoEActivationPack

    device = flat.device
    if is_nvfp4:
        from ......quantization.fp4_quantization import fp4_quantize

        if global_scale is None:
            global_scale = torch.ones(1, dtype=torch.float32, device=device)
        hidden_states_q, hidden_states_scale = fp4_quantize(
            flat,
            global_scale=global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=False,
        )
        if hidden_states_scale.dim() > 2:
            hidden_states_scale = hidden_states_scale.squeeze(-1)
    else:
        hidden_states_q = flat
        hidden_states_scale = torch.empty(0, device=device)

    return MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        selected_experts=selected_experts,
        final_scales=final_scales,
    )


def reshape_for_combine(out_2d: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Reshape compute output back to the 3D layout EP combine consumes."""
    hidden = out_2d.shape[-1]
    return out_2d.reshape(dim0, dim1, hidden)
