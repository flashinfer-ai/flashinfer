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
    from flashinfer.fused_moe.api import MoEActivationPack, QuantConfig


_NCCL_EP_LL_BF16_WIDTHS = (2048, 2560, 4096, 5120, 6144, 7168, 8192)


def packed_mxfp8_dispatch_width(hidden: int) -> int:
    """Return the smallest supported BF16 row that holds MXFP8 data + scales."""
    if hidden % 64:
        raise ValueError(f"mxfp8_dispatch requires hidden % 64 == 0, got {hidden}")
    need = (hidden + hidden // 32) // 2
    for width in _NCCL_EP_LL_BF16_WIDTHS:
        if width >= need:
            return width
    raise ValueError(f"packed MXFP8 row for hidden={hidden} is too large")


def pack_mxfp8_dispatch_payload(x: torch.Tensor) -> torch.Tensor:
    """Pack per-token MXFP8 values and linear scale bytes into BF16 rows."""
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise ValueError(
            "mxfp8_dispatch expects 2D BF16 [num_tokens, hidden] tokens, "
            f"got {x.dtype} shape {tuple(x.shape)}"
        )
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    m, hidden = x.shape
    send_width = packed_mxfp8_dispatch_width(hidden)
    q, sf = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
    packed = torch.zeros(m, 2 * send_width, dtype=torch.uint8, device=x.device)
    packed[:, :hidden] = q.view(torch.uint8)
    packed[:, hidden : hidden + hidden // 32] = sf.view(torch.uint8).reshape(
        m, hidden // 32
    )
    return packed.view(torch.bfloat16)


def _mask_padding_rows_nonlocal(
    selected_experts: torch.Tensor,
    final_scales: torch.Tensor,
    recv_count: torch.Tensor,
    *,
    num_local_experts: int,
    cap: int,
    local_expert_offset: int,
    num_experts: Optional[int],
):
    """Mask padded rows by marking them as belonging to an expert this rank does not own.

    Only the first recv_count[e] rows of each expert's block hold real tokens, the rest is padding. The
    routing kernel behind moe_sort only counts an expert into its histogram when that expert is local, so pointing a padding row at
    a remote expert drops it from the tile count, from ``permutedIdxSize``, and from
    every stage bounded by them - the permute/gather, both GEMMs, the activation and
    the finalize.

    Returns the inputs unchanged when there is no remote expert to point at.
    """
    if num_experts is None or num_local_experts >= num_experts:
        return selected_experts, final_scales

    expected = (num_local_experts,)
    if tuple(recv_count.shape) != expected:
        raise ValueError(
            f"recv_count must have shape {list(expected)}, got {list(recv_count.shape)}"
        )

    device = selected_experts.device
    m = selected_experts.shape[0]
    remote_id = (
        local_expert_offset + num_local_experts
    ) % num_experts  # First expert past this ranks block

    if m != num_local_experts * cap:
        raise ValueError(
            f"EXPERT_MAJOR pack must be num_local_experts * cap rows "
            f"({num_local_experts} * {cap} = {num_local_experts * cap}), got {m}"
        )

    # Mask padding rows by marking them as remote; efficient as broadcast compare.
    recv_count = recv_count.to(device)
    col = torch.arange(cap, device=device, dtype=recv_count.dtype)
    is_real = (col.unsqueeze(0) < recv_count.unsqueeze(1)).reshape(m, 1)

    selected_experts = torch.where(is_real, selected_experts, remote_id)
    final_scales = torch.where(is_real, final_scales, 0.0)
    return selected_experts, final_scales


def build_activation_pack(
    expert_tensors: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    quant: "QuantConfig",
    per_token_activation: bool = False,
    global_scale: Optional[torch.Tensor] = None,
    mxfp8_dispatch: bool = False,
    hidden_size: Optional[int] = None,
    recv_count: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
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
    quant : QuantConfig
        Weight / activation / output formats. NVFP4 activations quantize the
        dispatched BF16 rows; BF16 activations (including both W4A16 encodings)
        pass them through.
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

    if recv_count is not None:
        selected_experts, final_scales = _mask_padding_rows_nonlocal(
            selected_experts,
            final_scales,
            recv_count,
            num_local_experts=num_local_experts,
            cap=cap,
            local_expert_offset=local_expert_offset,
            num_experts=num_experts,
        )

    return _quantize_and_pack(
        flat,
        selected_experts,
        final_scales,
        quant=quant,
        per_token_activation=per_token_activation,
        global_scale=global_scale,
        mxfp8_dispatch=mxfp8_dispatch,
        hidden_size=hidden_size,
    )


def build_activation_pack_rank_major(
    recv_tensors: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
    local_expert_offset: int = 0,
    num_experts: Optional[int] = None,
    quant: "QuantConfig",
    per_token_activation: bool = False,
    global_scale: Optional[torch.Tensor] = None,
    mxfp8_dispatch: bool = False,
    hidden_size: Optional[int] = None,
    recv_scale: Optional[torch.Tensor] = None,
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

    The runner applies the real ``top_k`` weights and sums local contributions;
    ``combine`` sums across ranks. Masked picks use a valid remote expert ID
    when ``num_experts`` provides a remote shard, so the runner excludes them
    from expert compute. Otherwise they use a local ID with zero weight.

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
    num_experts : int, optional
        Global expert count. Enables exclusion of masked picks from local compute
        when this rank owns fewer than all experts.
    quant, per_token_activation, global_scale :
        Same semantics as :func:`build_activation_pack`.
    recv_scale : torch.Tensor, optional
        ``[M, *]`` received scale factors when the tokens were quantized by
        :func:`quantize_for_dispatch` before dispatch; ``recv_tensors`` then
        holds the quantized values and is not quantized again.
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

    # Zero weight alone does not skip GEMMs; a remote ID excludes the route.
    masked_expert = local_expert_offset
    if num_experts is not None and num_local_experts < num_experts:
        masked_expert = (local_expert_offset + num_local_experts) % num_experts
    is_local = (idx >= 0) & (idx < num_local_experts)
    selected_experts = torch.where(
        is_local,
        idx + local_expert_offset,
        torch.full_like(idx, masked_expert),
    ).to(torch.int32)
    final_scales = torch.where(is_local, weights, torch.zeros_like(weights))

    return _quantize_and_pack(
        flat,
        selected_experts,
        final_scales,
        quant=quant,
        per_token_activation=per_token_activation,
        global_scale=global_scale,
        mxfp8_dispatch=mxfp8_dispatch,
        hidden_size=hidden_size,
        flat_scale=recv_scale,
    )


def dispatch_quant_format(
    quant: "QuantConfig", *, per_token_activation: bool, mxfp8_dispatch: bool
):
    """Activation format tokens can be quantized to before a rank-major dispatch.

    NVFP4 with a unit global scale, MXFP8 and DeepSeek FP8 (1x128 blocks)
    quantize each token row on its own, so quantizing before dispatch yields
    the same values the receiver would compute. Per-token NVFP4 scaling and
    the packed ``mxfp8_dispatch`` row are excluded, as is any unquantized
    activation. Returns ``None`` when the tokens must be dispatched
    unquantized.
    """
    from flashinfer.fused_moe.api import QuantFormat

    activation = quant.pair[1]
    if activation is QuantFormat.NVFP4 and not per_token_activation:
        return QuantFormat.NVFP4
    if activation is QuantFormat.MXFP8 and not mxfp8_dispatch:
        return QuantFormat.MXFP8
    if activation is QuantFormat.DeepSeekFp8:
        return QuantFormat.DeepSeekFp8
    return None


def quantize_for_dispatch(
    x: torch.Tensor, quant: "QuantConfig"
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Quantize ``[num_tokens, hidden]`` tokens for a rank-major dispatch.

    Returns ``(values, scales)`` as uint8 rows, which every rank-major
    transport carries as raw bytes; :func:`build_activation_pack_rank_major`
    restores the dtypes and scale layout the runners expect.
    """
    from flashinfer.fused_moe.api import QuantFormat

    if quant.pair[1] is QuantFormat.DeepSeekFp8:
        q, scale = _deepseek_fp8_rows(x)
        return q.view(torch.uint8), scale.view(torch.uint8)
    q, scale, _ = _quantize_activations(
        x,
        quant=quant,
        per_token_activation=False,
        global_scale=None,
        mxfp8_dispatch=False,
        hidden_size=None,
    )
    return q.view(torch.uint8), scale.view(torch.uint8)


def _quantize_and_pack(
    flat: torch.Tensor,
    selected_experts: torch.Tensor,
    final_scales: torch.Tensor,
    *,
    quant: "QuantConfig",
    per_token_activation: bool,
    global_scale: Optional[torch.Tensor],
    mxfp8_dispatch: bool,
    hidden_size: Optional[int],
    flat_scale: Optional[torch.Tensor] = None,
) -> "MoEActivationPack":
    """Prepare ``flat`` for the configured activation path and assemble the pack.

    Shared by the EXPERT_MAJOR and RANK_MAJOR builders, which differ only in how
    ``selected_experts`` / ``final_scales`` are synthesized. With
    ``flat_scale``, ``flat`` already holds the values
    :func:`quantize_for_dispatch` produced and is only reinterpreted.
    """
    from flashinfer.fused_moe.api import MoEActivationPack

    if flat_scale is None:
        hidden_states_q, hidden_states_scale, per_token_scale = _quantize_activations(
            flat,
            quant=quant,
            per_token_activation=per_token_activation,
            global_scale=global_scale,
            mxfp8_dispatch=mxfp8_dispatch,
            hidden_size=hidden_size,
        )
    else:
        hidden_states_q, hidden_states_scale = _view_dispatched_activations(
            flat, flat_scale, quant
        )
        per_token_scale = None

    return MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        topk_ids=selected_experts,
        topk_weights=final_scales,
        per_token_scale=per_token_scale,
    )


def _view_dispatched_activations(
    values: torch.Tensor, scales: torch.Tensor, quant: "QuantConfig"
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Give received :func:`quantize_for_dispatch` rows the runners' dtypes."""
    from flashinfer.fused_moe.api import QuantFormat

    weight, activation = quant.pair
    values = values.view(torch.uint8)
    scales = scales.view(torch.uint8)
    if activation is QuantFormat.DeepSeekFp8:
        # Runners take the 1x128 block scales transposed, as [hidden // 128, M].
        values = values.reshape(-1, values.shape[-1])
        scales = scales.reshape(values.shape[0], -1).view(torch.float32)
        return values.view(torch.float8_e4m3fn), scales.t().contiguous()
    if activation is QuantFormat.MXFP8:
        values = values.view(torch.float8_e4m3fn)
        if weight is QuantFormat.MXFP4:
            scales = scales.view(torch.float8_e4m3fn)
    return values, scales


def _check_deepseek_fp8_hidden(hidden: int) -> None:
    if hidden % 128:
        raise ValueError(
            f"DeepSeek FP8 activations need hidden % 128 == 0, got {hidden}"
        )


def _deepseek_fp8_rows(flat: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """DeepSeek FP8 (1x128 block) quantization with row-major scales.

    Returns float8_e4m3fn values ``[M, hidden]`` and float32 scales
    ``[M, hidden // 128]``, so each token row carries its own scales.
    """
    from flashinfer.fused_moe.prepare import _deepseek_fp8_quantize_activations

    _check_deepseek_fp8_hidden(flat.shape[1])
    q, scale_t = _deepseek_fp8_quantize_activations(flat.contiguous())
    return q, scale_t.t().contiguous()


def _quantize_activations(
    flat: torch.Tensor,
    *,
    quant: "QuantConfig",
    per_token_activation: bool,
    global_scale: Optional[torch.Tensor],
    mxfp8_dispatch: bool,
    hidden_size: Optional[int],
) -> "tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]":
    """Quantize ``flat`` to the configured activation format.

    Returns ``(values, scales, per_token_scale)`` in the dtypes the runners
    consume; ``scales`` is None for BF16 activations.
    """
    from flashinfer.fused_moe.api import QuantFormat

    pair = quant.pair
    activation = pair[1]
    device = flat.device
    per_token_scale = None
    if per_token_activation and activation is not QuantFormat.NVFP4:
        raise ValueError("per-token activation scaling requires NVFP4 activations")
    if activation is QuantFormat.NVFP4:
        from flashinfer.quantization.fp4_quantization import (
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
            from flashinfer.quantization.nvfp4_quantization_utils import (
                make_nvfp4_global_scale,
                resolve_nvfp4_4over6,
            )
            from flashinfer.tllm_enums import SfLayout

            global_scale = make_nvfp4_global_scale(
                flat,
                per_token_activation=True,
                nvfp4_4over6_config=resolve_nvfp4_4over6(),
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
    elif activation is QuantFormat.MXFP8:
        from flashinfer.quantization.fp8_quantization import mxfp8_quantize

        if mxfp8_dispatch:
            if hidden_size is None:
                raise ValueError("mxfp8_dispatch requires the logical hidden size")
            send_width = packed_mxfp8_dispatch_width(hidden_size)
            if flat.dtype != torch.bfloat16 or flat.shape[1] != send_width:
                raise ValueError(
                    f"packed MXFP8 input must be BF16 width {send_width}, got "
                    f"{flat.dtype} width {flat.shape[1]}"
                )
            packed = flat.contiguous().view(torch.uint8)
            hidden_states_q = (
                packed[:, :hidden_size].contiguous().view(torch.float8_e4m3fn)
            )
            hidden_states_scale = packed[
                :, hidden_size : hidden_size + hidden_size // 32
            ].contiguous()
        else:
            hidden_states_q, hidden_states_scale = mxfp8_quantize(
                flat.contiguous(), is_sf_swizzled_layout=False
            )
            hidden_states_scale = hidden_states_scale.view(torch.uint8).reshape(
                flat.shape[0], flat.shape[1] // 32
            )
        if pair[0] is QuantFormat.MXFP4:
            # W4A8 packs carry UE8M0 bytes in float8 storage. CuTe reinterprets
            # them as uint8; TRT-LLM requires the float8 view.
            hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn)
    elif activation is QuantFormat.DeepSeekFp8:
        from flashinfer.fused_moe.prepare import _deepseek_fp8_quantize_activations

        _check_deepseek_fp8_hidden(flat.shape[1])
        hidden_states_q, hidden_states_scale = _deepseek_fp8_quantize_activations(
            flat.contiguous()
        )
    elif activation is QuantFormat.BF16:
        # BF16 and both W4A16 encodings consume the dispatched activations.
        hidden_states_q = flat
        hidden_states_scale = None
    else:
        raise ValueError(
            f"fused_moe split bridge does not support weight={pair[0].name}, "
            f"activation={pair[1].name}"
        )

    return hidden_states_q, hidden_states_scale, per_token_scale


def reshape_for_combine(out_2d: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Reshape compute output ``[dim0*dim1, hidden]`` back to the 3D
    ``[dim0, dim1, hidden]`` layout that EP ``combine`` consumes.

    EXPERT_MAJOR passes ``(num_local_experts, cap)``; RANK_MAJOR passes
    ``(world, max_tokens_per_rank)`` — the reshape is identical either way."""
    hidden = out_2d.shape[-1]
    return out_2d.reshape(dim0, dim1, hidden)
