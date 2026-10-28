"""Stage bf16 activations + routing into MXFP8 mega-MoE symmetric buffers."""

from __future__ import annotations

import os

import torch

from ......core.validation.common import MoEEpConfigError


def _use_fused_stage() -> bool:
    # Bisection escape hatch back to the multi-kernel torch staging path.
    return os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") != "0"


def _mxfp8_data_dtype(kind: str) -> torch.dtype:
    """Resolve activation dtype, using E4M3 for mixed MXFP4/MXFP8 weights."""
    # Backend talks only to the cutedsl_megamoe shim (never src/ directly); the
    # package import also bootstraps sys.path for the kernel packages.
    from ......kernel_src.sm100.cutedsl_megamoe import kind_data_dtype

    return kind_data_dtype("mxfp8_e4m3" if kind == "mxfp4_mxfp8" else kind)


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    kind: str = "mxfp8_e4m3",
) -> None:
    """bf16 ``hidden_states`` → MXFP8 activation + E8M0 block scales.

    Default path is the fused single-launch ``DataPreprocess`` staging kernel
    (quant + routing repack in one launch); ``FLASHINFER_MEGA_FUSED_STAGE=0``
    falls back to the original torch-composed staging below.
    """
    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from ......kernel_src.sm100.cutedsl_megamoe import (
        Mxfp8BlockSize,
        ceil_div,
        fused_quant_stage,
        fused_quant_stage_supported,
        note_staged_tokens,
        mxfp8_quantize_per_block_32,
        round_up,
    )

    num_tokens, hidden = hidden_states.shape
    if num_tokens == 0:
        # A zero-token step still owns the buffer: rows a previous batch left
        # routed must be re-masked (and the live-count memo set to 0) or they
        # would dispatch as stale live tokens.
        topk_idx_out.fill_(-1)
        note_staged_tokens(topk_idx_out, 0)
        return
    if hidden % 64 != 0:
        raise ValueError("hidden_size must be a multiple of 64.")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")

    if _use_fused_stage() and fused_quant_stage_supported(
        hidden_states, quant_type="mxfp8"
    ):
        fused_quant_stage(
            hidden_states,
            topk_ids,
            topk_weights,
            x_fp8,
            x_sf,
            topk_idx_out,
            topk_weights_out,
            quant_type="mxfp8_e4m3" if kind == "mxfp4_mxfp8" else kind,
        )
        return

    data_dtype = _mxfp8_data_dtype(kind)
    activation_fp32 = hidden_states.to(torch.float32)
    q, sf = mxfp8_quantize_per_block_32(activation_fp32, data_dtype)

    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)
    if x_sf.shape[1] < hidden_sf_cols_padded:
        raise ValueError(
            f"x_sf trailing dim ({x_sf.shape[1]}) is smaller than required "
            f"{hidden_sf_cols_padded}."
        )

    x_fp8[:num_tokens].view(torch.uint8).copy_(q.view(torch.uint8))
    x_sf[:num_tokens].zero_()
    x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(sf.view(torch.uint8))
    topk_idx_out[:num_tokens].copy_(topk_ids)
    topk_weights_out[:num_tokens].copy_(topk_weights)

    capacity = x_fp8.shape[0]
    if num_tokens < capacity:
        topk_idx_out[num_tokens:capacity].fill_(-1)
    note_staged_tokens(topk_idx_out, num_tokens)


def validate_mxfp8_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    kind: str = "mxfp8_e4m3",
    scales: torch.Tensor | None = None,
) -> None:
    """MXFP8 mega-path validation (bf16 staging or pre-staged fp8)."""
    from ......core.validation.common import validate_mega_forward_inputs

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

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from ......kernel_src.sm100.cutedsl_megamoe import (
        Mxfp8BlockSize,
        Mxfp8ScaleDtype,
        ceil_div,
    )

    num_tokens = hidden_states.shape[0]
    hidden = fleet_params.token_hidden_size
    data_dtype = _mxfp8_data_dtype(kind)
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
            f"pre-staged MXFP8 hidden_states must be 2D with shape "
            f"[num_tokens, {hidden}], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != data_dtype:
        raise MoEEpConfigError(
            f"pre-staged MXFP8 hidden_states must have dtype {data_dtype}, "
            f"got {hidden_states.dtype}"
        )
    if topk_ids.shape != (num_tokens, top_k):
        raise MoEEpConfigError(
            f"topk_ids must have shape ({num_tokens}, {top_k}), "
            f"got {tuple(topk_ids.shape)}"
        )
    if topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("topk_weights and topk_ids must have the same shape")

    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    if scales.ndim != 2 or scales.shape[0] != num_tokens:
        raise MoEEpConfigError(
            f"scales must be 2D with leading dim {num_tokens}, got {tuple(scales.shape)}"
        )
    if scales.dtype != Mxfp8ScaleDtype:
        raise MoEEpConfigError(
            f"scales must have dtype {Mxfp8ScaleDtype}, got {scales.dtype}"
        )
    if scales.shape[1] < hidden_sf_cols:
        raise MoEEpConfigError(
            f"scales.shape[1] ({scales.shape[1]}) must be >= {hidden_sf_cols} "
            f"for hidden={hidden}"
        )


def validate_genphase_forward_inputs(
    t, fleet_params, *, top_k, quantize_input, kind="mxfp8_e4m3"
):
    """GenPhase metadata checks; callers provide valid, unique expert routing."""
    for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
        if getattr(t, name) is not None:
            raise MoEEpConfigError(f"Blackwell GenPhase does not accept {name}.")
    if t.topk_ids.ndim != 2 or t.topk_weights.ndim != 2:
        raise MoEEpConfigError("GenPhase routing tensors must be 2D.")
    validate_mxfp8_forward_inputs(
        t.hidden_states,
        t.topk_ids,
        t.topk_weights,
        fleet_params,
        top_k=top_k,
        quantize_input=quantize_input,
        scales=t.scales,
        kind=kind,
    )
    if quantize_input and t.hidden_states.dtype != torch.bfloat16:
        raise MoEEpConfigError("GenPhase quantize_input=True requires BF16 inputs.")
    if t.topk_ids.dtype not in (torch.int32, torch.int64):
        raise MoEEpConfigError("GenPhase expert IDs must be int32 or int64.")
    if t.topk_weights.dtype != torch.float32:
        raise MoEEpConfigError("GenPhase top-k weights must be float32.")
    device = torch.device("cuda", torch.cuda.current_device())
    tensors = (t.hidden_states, t.topk_ids, t.topk_weights)
    if not quantize_input:
        tensors += (t.scales,)
    if any(tensor.device != device for tensor in tensors):
        raise MoEEpConfigError("GenPhase inputs must be on the current CUDA device.")


def stage_genphase_inputs(t, workspace, *, quantize_input: bool) -> None:
    """Stage the GenPhase int32 router and its padded linear SF rows."""
    # Match the legacy SM100 input contract without per-step device scans:
    # valid routes are unique IDs in [0, E) with finite scores; -1 is masked.
    num_tokens = t.hidden_states.shape[0]
    sf_cols = workspace.config.hidden // workspace.config.sf_vec_size
    if quantize_input:
        # Only the logical SF columns are staged. The padded columns were
        # zeroed at allocation and are never written by either input path.
        stage_mega_moe_inputs(
            t.hidden_states,
            t.topk_weights,
            t.topk_ids,
            workspace.x,
            workspace.x_sf[:, :sf_cols],
            workspace.topk_idx,
            workspace.topk_weights,
            kind=workspace.config.quant_kind,
        )
    else:
        from ......kernel_src.sm100.cutedsl_megamoe import note_staged_tokens

        workspace.x[:num_tokens].view(torch.uint8).copy_(
            t.hidden_states.view(torch.uint8)
        )
        workspace.x_sf[:num_tokens, :sf_cols].view(torch.uint8).copy_(
            t.scales[:, :sf_cols].view(torch.uint8)
        )
        workspace.topk_idx[:num_tokens].copy_(t.topk_ids)
        workspace.topk_weights[:num_tokens].copy_(t.topk_weights)
        workspace.topk_idx[num_tokens:].fill_(-1)
        # A later fused stage may share this workspace with a different layer.
        note_staged_tokens(workspace.topk_idx, num_tokens)
    workspace.note_staged_tokens(num_tokens)
