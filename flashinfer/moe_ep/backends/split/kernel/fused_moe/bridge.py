"""Layout bridge between EP dispatch output and the unified MoE compute API.

The NCCL/NIXL EP ``dispatch`` returns an **expert-major, padded** tensor
``[num_local_experts, cap, hidden]`` (``cap = max_tokens_per_rank * world_size``),
whereas the unified compute runners (``flashinfer.fused_moe``) consume a
**token-major** :class:`MoEActivationPack` (``[M, hidden]`` + ``topk_ids`` /
``topk_weights``).

Every dispatched row is already assigned to exactly one local expert — the expert is
the first-dim index, i.e. ``expert = row // cap``.  So we flatten the buffer to
``[num_local_experts*cap, hidden]`` and synthesize a **top_k=1, pre-routed** batch with
``topk_weights = 1``.  The real top-k reweight + reduction is owned by EP ``combine``;
the local compute must NOT apply routing weights (that would double-weight).  Running
the runner with ``do_finalize=True`` at ``top_k=1`` is an identity-order scatter, so the
output comes back in the same row order and reshapes straight back to the 3D combine
layout via :func:`reshape_for_combine`.

Padded rows beyond the per-expert ``recv_count`` compute garbage that ``combine`` never
gathers — correct, at a perf cost that a future recv_count-aware grouping can remove.

For the LL **RANK_MAJOR** layout the recv buffer is instead
``[world, max_tokens_per_rank, hidden]`` (tokens grouped by source rank, each
carrying its received ``topk_idx`` / ``topk_weights``), and
:func:`build_activation_pack_rank_major` drives the runner at the real ``top_k``
with the received routing masked to this rank's local experts — see its docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ......fused_moe.api import MoEActivationPack, QuantVariant


def build_activation_pack(
    expert_tensors: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    quant_variant: "QuantVariant",
    per_token_activation: bool = False,
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
    quant_variant : QuantVariant
        Activation and weight quantization contract. NVFP4 quantizes the
        dispatched BF16 activations; BF16 and W4A16 carry them unchanged.
    per_token_activation : bool
        Use per-token NVFP4 activation scaling when quantizing the input.
    global_scale : torch.Tensor, optional
        Per-tensor global scale for NVFP4 quantization (shape ``[1]``, float32).
    """
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
    # combine() applies the real topk_weights, so topk_weights must stay 1 here.
    row_expert = torch.arange(num_local_experts, device=device, dtype=torch.int32)
    selected_experts = (
        row_expert.repeat_interleave(cap).reshape(m, 1) + local_expert_offset
    )
    final_scales = torch.ones(m, 1, dtype=torch.float32, device=device)

    return _quantize_and_pack(
        flat,
        selected_experts,
        final_scales,
        quant_variant=quant_variant,
        per_token_activation=per_token_activation,
        global_scale=global_scale,
    )


def build_activation_pack_rank_major(
    recv_tensors: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
    local_expert_offset: int = 0,
    quant_variant: "QuantVariant",
    per_token_activation: bool = False,
    global_scale: Optional[torch.Tensor] = None,
) -> "MoEActivationPack":
    """Translate the 3D RANK_MAJOR dispatch output into a token-major pack.

    The RANK_MAJOR recv buffer is ``[world, max_tokens_per_rank, hidden]`` — tokens
    grouped by *source rank*, received once each, **carrying their received
    ``topk_idx`` / ``topk_weights``**.  The library returns ``topk_idx`` as
    **LOCAL expert indices** (0-based within this rank's experts), with ``-1``
    marking a pick routed to a NON-local expert (handled by another rank).  Each
    received token must run through whichever of *this rank's* local experts appear
    in its top-k, weighted and summed (the "pre-reduce across local experts" the
    layout contract calls for).

    This is the **faithful** path: we drive the unified runner at the model's real
    ``top_k`` with ``do_finalize=True`` (which does the per-token weighted sum), and
    mask the non-local picks to weight 0 (pointing them at a valid local expert so
    indexing stays in range).  ``combine`` then just sums across ranks — the real
    weights are applied here, not on receive.  (The runner still evaluates all
    ``top_k`` slots per token, so the masked picks cost compute they don't
    contribute; an exact local-only gather is a future optimization.)

    Parameters
    ----------
    recv_tensors : torch.Tensor
        Dispatch output, shape ``[world, max_tokens_per_rank, hidden]`` (bf16).
    recv_topk_idx : torch.Tensor
        Received per-token LOCAL expert indices, ``[M, top_k]`` (``M = world *
        max_tokens_per_rank``); ``-1`` marks a non-local pick.
    recv_topk_weights : torch.Tensor
        Received per-token routing weights, ``[M, top_k]`` fp32 (same per-pick order
        as ``recv_topk_idx``).
    num_local_experts : int
        Number of experts this rank owns (carried for API symmetry / validation).
    local_expert_offset : int
        Global id of this rank's first local expert (see EXPERT_MAJOR builder).
    quant_variant, per_token_activation, global_scale :
        Same semantics as :func:`build_activation_pack`.
    """
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
    if idx.shape != weights.shape or idx.shape[0] != m:
        raise ValueError(
            f"recv_topk_idx/weights must share shape [M={m}, top_k]; got "
            f"{tuple(idx.shape)} / {tuple(weights.shape)}."
        )

    # The RANK_MAJOR dispatch returns LOCAL expert indices (0-based within this
    # rank's experts) for each token's picks, with -1 marking a pick routed to a
    # NON-local expert (handled by another rank). Convert local->global
    # (idx + local_expert_offset) for the runner, which expects global ids and
    # filters by local_expert_offset. Only [0, num_local_experts) is local; -1 and
    # any out-of-range id are masked to a valid local id with weight 0, so the
    # weighted finalize sum drops them; combine then sums each token's per-rank
    # partial reductions across ranks.
    is_local = (idx >= 0) & (idx < num_local_experts)
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
        quant_variant=quant_variant,
        per_token_activation=per_token_activation,
        global_scale=global_scale,
    )


def _quantize_and_pack(
    flat: torch.Tensor,
    selected_experts: torch.Tensor,
    final_scales: torch.Tensor,
    *,
    quant_variant: "QuantVariant",
    per_token_activation: bool,
    global_scale: Optional[torch.Tensor],
) -> "MoEActivationPack":
    """Prepare ``flat`` for the configured activation path and assemble the pack.

    Shared by the EXPERT_MAJOR and RANK_MAJOR builders, which differ only in how
    ``selected_experts`` / ``final_scales`` are synthesized.
    """
    from ......fused_moe.api import MoEActivationPack, QuantVariant

    device = flat.device
    per_token_scale = None
    if per_token_activation and quant_variant is not QuantVariant.NVFP4:
        raise ValueError("per-token activation scaling requires QuantVariant.NVFP4")
    if quant_variant is QuantVariant.NVFP4:
        from ......quantization.fp4_quantization import (
            fp4_quantize,
            nvfp4_quantize,
        )

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
        if per_token_activation:
            from ......quantization.nvfp4_quantization_utils import (
                current_nvfp4_4over6_config,
                make_nvfp4_global_scale,
            )
            from ......tllm_enums import SfLayout

            global_scale = make_nvfp4_global_scale(
                flat,
                per_token_activation=True,
                nvfp4_4over6_config=current_nvfp4_4over6_config(),
            )
            hidden_states_q, hidden_states_scale, per_token_scale = nvfp4_quantize(
                flat,
                global_scale,
                sfLayout=SfLayout.layout_linear,
                per_token_activation=True,
                backend="cute-dsl",
            )
        else:
            hidden_states_q, hidden_states_scale = fp4_quantize(
                flat,
                global_scale=global_scale,
                sf_vec_size=16,
                is_sf_swizzled_layout=False,
                backend="cute-dsl",
            )
        # Runners expect a 2D [M, H//16] scale; fp4_quantize may return a trailing dim.
        if hidden_states_scale.dim() > 2:
            hidden_states_scale = hidden_states_scale.squeeze(-1)
    elif quant_variant in (QuantVariant.BF16, QuantVariant.W4A16):
        # BF16 and W4A16 runners consume the dispatched activations directly.
        hidden_states_q = flat
        hidden_states_scale = None
    else:
        raise ValueError(f"fused_moe split bridge does not support {quant_variant!r}")

    return MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        topk_ids=selected_experts,
        topk_weights=final_scales,
        per_token_scale=per_token_scale,
    )


def reshape_for_combine(out_2d: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Reshape compute output ``[dim0*dim1, hidden]`` back to the 3D
    ``[dim0, dim1, hidden]`` layout that EP ``combine`` consumes.

    EXPERT_MAJOR passes ``(num_local_experts, cap)``; RANK_MAJOR passes
    ``(world, max_tokens_per_rank)`` — the reshape is identical either way."""
    hidden = out_2d.shape[-1]
    return out_2d.reshape(dim0, dim1, hidden)
