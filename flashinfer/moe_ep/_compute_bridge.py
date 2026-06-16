"""Layout bridge between EP dispatch output and the unified MoE compute API.

The NCCL/NIXL EP ``dispatch`` returns an **expert-major, padded** tensor
``[num_local_experts, cap, hidden]`` (``cap = max_tokens_per_rank * world_size``),
whereas the unified compute runners (``flashinfer.fused_moe``) consume a
**token-major** :class:`MoEActivationPack` (``[M, hidden]`` + ``selected_experts`` /
``final_scales``).

Every dispatched row is already assigned to exactly one local expert — the expert is
the first-dim index, i.e. ``expert = row // cap``.  So we flatten the buffer to
``[num_local_experts*cap, hidden]`` and synthesize a **top_k=1, pre-routed** batch with
``final_scales = 1``.  The real top-k reweight + reduction is owned by EP ``combine``;
the local compute must NOT apply routing weights (that would double-weight).  Running
the runner with ``do_finalize=True`` at ``top_k=1`` is an identity-order scatter, so the
output comes back in the same row order and reshapes straight back to the 3D combine
layout via :func:`reshape_for_combine`.

Padded rows beyond the per-expert ``recv_count`` compute garbage that ``combine`` never
gathers — correct, at a perf cost that a future recv_count-aware grouping can remove.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ..fused_moe.api import MoEActivationPack


def build_activation_pack(
    expert_tensors: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    is_nvfp4: bool,
    global_scale: Optional[torch.Tensor] = None,
) -> "MoEActivationPack":
    """Translate the 3D expert-major dispatch output into a token-major pack.

    Parameters
    ----------
    expert_tensors : torch.Tensor
        Dispatch output, shape ``[num_local_experts, cap, hidden]`` (bf16).
    local_expert_offset : int
        Global id of this rank's first local expert.  Synthesized expert ids are
        ``(row // cap) + local_expert_offset``; the compute runner subtracts the same
        offset, so the two must agree (they share one :class:`MoEConfig`).
    is_nvfp4 : bool
        When True, quantize the flattened activations to NVFP4 (``hidden_states_q`` +
        ``hidden_states_scale``).  When False (bf16 path), the raw bf16 activations are
        carried in ``hidden_states_q`` and ``hidden_states_scale`` is a 0-d placeholder
        (the bf16 runner ignores it).
    global_scale : torch.Tensor, optional
        Per-tensor global scale for NVFP4 quantization (shape ``[1]``, float32).
    """
    from ..fused_moe.api import MoEActivationPack

    if expert_tensors.dim() != 3:
        raise ValueError(
            "build_activation_pack expects a 3D [num_local_experts, cap, hidden] "
            f"dispatch tensor, got shape {tuple(expert_tensors.shape)}"
        )
    num_local_experts, cap, hidden = expert_tensors.shape
    flat = expert_tensors.reshape(num_local_experts * cap, hidden)
    m = flat.shape[0]
    device = flat.device

    # Synthesized routing: each row -> its own (single) expert, weight 1.0.
    # combine() applies the real topk_weights, so final_scales must stay 1 here.
    row_expert = torch.arange(num_local_experts, device=device, dtype=torch.int32)
    selected_experts = (
        row_expert.repeat_interleave(cap).reshape(m, 1) + local_expert_offset
    )
    final_scales = torch.ones(m, 1, dtype=torch.float32, device=device)

    if is_nvfp4:
        from ..quantization.fp4_quantization import fp4_quantize

        if global_scale is None:
            # NVFP4 requires a global scale (fp4Quantize asserts it is set when
            # sfUseUE8M0 is False).  Use a unit global scale, matching the weight
            # preparation convention in flashinfer.fused_moe.prepare (which passes
            # global_scale=1.0 and folds dequant into per-block scale factors +
            # the runner's alpha tensors).
            global_scale = torch.ones(1, dtype=torch.float32, device=device)
        # Activations use the LINEAR scale-factor layout (is_sf_swizzled_layout=
        # False), matching the runners' expectation (see create_moe_tensors in
        # tests/moe/test_b12x_fused_moe.py).  The default swizzled layout makes
        # the kernel index the scale tensor out of bounds → illegal memory access.
        hidden_states_q, hidden_states_scale = fp4_quantize(
            flat,
            global_scale=global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=False,
        )
        # Runners expect a 2D [M, H//16] scale; fp4_quantize may return a trailing dim.
        if hidden_states_scale.dim() > 2:
            hidden_states_scale = hidden_states_scale.squeeze(-1)
    else:
        # bf16 path: carry the raw activations; the bf16 runner reads them directly.
        hidden_states_q = flat
        hidden_states_scale = torch.empty(0, device=device)

    return MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        selected_experts=selected_experts,
        final_scales=final_scales,
    )


def reshape_for_combine(
    out_2d: torch.Tensor, num_local_experts: int, cap: int
) -> torch.Tensor:
    """Reshape compute output ``[num_local_experts*cap, hidden]`` back to the 3D
    ``[num_local_experts, cap, hidden]`` layout that EP ``combine`` consumes."""
    hidden = out_2d.shape[-1]
    return out_2d.reshape(num_local_experts, cap, hidden)
