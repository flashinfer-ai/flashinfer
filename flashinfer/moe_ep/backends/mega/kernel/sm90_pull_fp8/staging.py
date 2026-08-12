"""Stage bf16 activations + routing into SM90 FP8 mega-MoE symmetric buffers."""

from __future__ import annotations

import torch

from .....core.validation.common import MoEEpConfigError

# E8M0 encodes 2^(byte - 127): byte 127 == 1.0.  The per-tensor SF wire plane
# is dispatched but unused by GEMM dequantization; unit values match the drop
# driver's staging.
_E8M0_ONE_BYTE = 127


def _fp8_data_dtype(kind: str) -> torch.dtype:
    # Backend talks only to the pull_style_cutedsl_megakernel shim (never src/
    # directly); the package import also bootstraps sys.path for the kernel
    # packages.
    from .....kernel_src.sm90.pull_style_cutedsl_megakernel import kind_data_dtype

    return kind_data_dtype(kind)


def _note_staged_tokens(workspace_topk_idx: torch.Tensor, num_tokens: int) -> None:
    """Remember the live token count for ``compute(output=None)``.

    The sm100 trees memoize this in their fused-stage module; the sm90 tree
    has no staging kernel yet, so the count rides on the staging tensor object
    itself (same lifetime as the workspace, overwritten every stage).
    """
    workspace_topk_idx._sm90_staged_tokens = num_tokens  # type: ignore[attr-defined]


def staged_tokens(workspace_topk_idx: torch.Tensor) -> int | None:
    """Token count from the last stage into this workspace, or None."""
    return getattr(workspace_topk_idx, "_sm90_staged_tokens", None)


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    kind: str = "fp8_e4m3",
    fp8_scale_mode: str = "per_tensor",
    fc1_activation_dequant_scale: float = 1.0,
) -> None:
    """bf16 ``hidden_states`` → FP8 activation + per-mode scale staging.

    Quantization matches the kernel's dequant convention (``fp32 ~=
    payload * scale``; see ``weights.py``):

    * ``per_tensor``: ``payload = x / fc1_activation_dequant_scale`` (the
      static calibration scalar every EP rank shares); the E8M0 wire plane is
      filled with unit scales (dispatched, unused by dequantization).
    * ``blockwise``: DeepGEMM-style per-token/128-block fp32 scales
      (``quantize_fp8_per_token_block``: ``scale = absmax / fp8_max``), staged
      into the fp32 ``x_sf`` columns.

    PORT NOTE: torch-composed staging only — the sm100 fused single-launch
    ``DataPreprocess`` staging kernel has no sm90 counterpart yet.
    """
    num_tokens, hidden = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden % 64 != 0:
        raise ValueError("hidden_size must be a multiple of 64.")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")

    data_dtype = _fp8_data_dtype(kind)
    activation_fp32 = hidden_states.to(torch.float32)

    if fp8_scale_mode == "blockwise":
        # Backend talks only to the pull_style_cutedsl_megakernel shim boundary.
        from .....kernel_src.sm90.pull_style_cutedsl_megakernel import (
            Fp8BlockScaleK,
            quantize_fp8_per_token_block,
        )

        if hidden % Fp8BlockScaleK != 0:
            raise ValueError(
                f"blockwise FP8 requires hidden ({hidden}) divisible by "
                f"{Fp8BlockScaleK}."
            )
        q, sf = quantize_fp8_per_token_block(
            activation_fp32, data_dtype, block_k=Fp8BlockScaleK
        )
        sf_cols = hidden // Fp8BlockScaleK
        if x_sf.shape[1] < sf_cols:
            raise ValueError(
                f"x_sf trailing dim ({x_sf.shape[1]}) is smaller than required "
                f"{sf_cols}."
            )
        x_fp8[:num_tokens].view(torch.uint8).copy_(q.view(torch.uint8))
        x_sf[:num_tokens].zero_()
        x_sf[:num_tokens, :sf_cols].copy_(sf)
    else:
        scale = float(fc1_activation_dequant_scale)
        if scale <= 0.0:
            raise ValueError(
                f"fc1_activation_dequant_scale must be positive, got {scale}."
            )
        # torch fp8 casts saturate, so out-of-range values clip to fp8 max —
        # same behaviour the reference's requant chain relies on.
        q = (activation_fp32 / scale).to(data_dtype)
        x_fp8[:num_tokens].view(torch.uint8).copy_(q.view(torch.uint8))
        # Unit E8M0 wire plane (byte 127 == 2^0); full padded row so the
        # dispatch LDG.32 words carry defined bytes.
        x_sf[:num_tokens].view(torch.uint8).fill_(_E8M0_ONE_BYTE)

    topk_idx_out[:num_tokens].copy_(topk_ids)
    topk_weights_out[:num_tokens].copy_(topk_weights)

    capacity = x_fp8.shape[0]
    if num_tokens < capacity:
        topk_idx_out[num_tokens:capacity].fill_(-1)
    _note_staged_tokens(topk_idx_out, num_tokens)


def validate_sm90_fp8_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    kind: str = "fp8_e4m3",
    fp8_scale_mode: str = "per_tensor",
    scales: torch.Tensor | None = None,
) -> None:
    """SM90 FP8 mega-path validation (bf16 staging or pre-staged fp8)."""
    from .....core.validation.common import validate_mega_forward_inputs

    if quantize_input:
        validate_mega_forward_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            fleet_params,
            top_k=top_k,
            quantize_input=True,
        )
        return

    num_tokens = hidden_states.shape[0]
    hidden = fleet_params.token_hidden_size
    data_dtype = _fp8_data_dtype(kind)
    blockwise = fp8_scale_mode == "blockwise"
    if scales is None:
        raise MoEEpConfigError(
            "MoEEpTensors.scales is required when MegaConfig.quantize_input=False"
        )
    if num_tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError(
            f"token count {num_tokens} exceeds "
            f"max_tokens_per_rank={fleet_params.max_tokens_per_rank}"
        )
    if hidden_states.ndim != 2 or hidden_states.shape[1] != hidden:
        raise MoEEpConfigError(
            f"pre-staged FP8 hidden_states must be 2D with shape "
            f"[num_tokens, {hidden}], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != data_dtype:
        raise MoEEpConfigError(
            f"pre-staged FP8 hidden_states must have dtype {data_dtype}, "
            f"got {hidden_states.dtype}"
        )
    if topk_ids.shape != (num_tokens, top_k):
        raise MoEEpConfigError(
            f"topk_ids must have shape ({num_tokens}, {top_k}), "
            f"got {tuple(topk_ids.shape)}"
        )
    if topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("topk_weights and topk_ids must have the same shape")

    if scales.ndim != 2 or scales.shape[0] != num_tokens:
        raise MoEEpConfigError(
            f"scales must be 2D with leading dim {num_tokens}, got {tuple(scales.shape)}"
        )
    if blockwise:
        # Backend talks only to the pull_style_cutedsl_megakernel shim boundary.
        from .....kernel_src.sm90.pull_style_cutedsl_megakernel import Fp8BlockScaleK

        sf_cols = hidden // Fp8BlockScaleK
        if scales.dtype != torch.float32:
            raise MoEEpConfigError(
                f"blockwise scales must have dtype torch.float32, got {scales.dtype}"
            )
    else:
        from .....kernel_src.sm90.pull_style_cutedsl_megakernel import (
            Fp8E8M0SfVecSize,
            ceil_div,
        )

        sf_cols = ceil_div(hidden, Fp8E8M0SfVecSize)
        if scales.dtype != torch.float8_e8m0fnu:
            raise MoEEpConfigError(
                "per-tensor scales must have dtype torch.float8_e8m0fnu, "
                f"got {scales.dtype}"
            )
    if scales.shape[1] < sf_cols:
        raise MoEEpConfigError(
            f"scales.shape[1] ({scales.shape[1]}) must be >= {sf_cols} "
            f"for hidden={hidden} (fp8_scale_mode={fp8_scale_mode!r})"
        )
