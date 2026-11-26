"""Stage MXFP8 activations and routing into the NVLinkOneSided MegaMoE workspace.

The Sentinel dispatch arms every landing byte with ``0xFF`` and treats a row
as arrived once none of its bytes is ``0xFF``, so staging rewrites the only
inputs that could collide with the sentinel: an E4M3 ``0xFF`` (a negative NaN)
becomes ``0x7F`` (NaN) and a UE8M0 ``0xFF`` (a NaN scale) becomes ``0xFE``.
"""

from __future__ import annotations

import torch

from ......core.validation.common import (
    MoEEpConfigError,
    validate_mega_forward_inputs,
)

_SF_VEC = 32


def validate_forward_inputs(t, fleet_params, *, top_k: int, quantize_input: bool):
    """Metadata checks; callers provide unique expert IDs per row (-1 masks a route)."""
    for name in (
        "fc1_alpha",
        "fc2_alpha",
        "fc1_norm_const",
        "fc1_activation_per_token_scale",
    ):
        if getattr(t, name) is not None:
            raise MoEEpConfigError(
                f"the MXFP4-weight NVLinkOneSided MegaMoE does not accept {name}"
            )
    if t.hidden_states.ndim != 2 or t.topk_ids.ndim != 2 or t.topk_weights.ndim != 2:
        raise MoEEpConfigError("hidden_states and routing tensors must be 2D")
    validate_mega_forward_inputs(
        t.hidden_states,
        t.topk_ids,
        t.topk_weights,
        fleet_params,
        top_k=top_k,
        quantize_input=quantize_input,
        scales=t.scales,
    )
    if t.topk_ids.dtype not in (torch.int32, torch.int64):
        raise MoEEpConfigError(
            f"topk_ids must be int32 or int64, got {t.topk_ids.dtype}"
        )
    if t.topk_weights.dtype != torch.float32:
        raise MoEEpConfigError(
            f"topk_weights must be float32, got {t.topk_weights.dtype}"
        )
    tensors = [t.hidden_states, t.topk_ids, t.topk_weights]
    if quantize_input:
        if t.hidden_states.dtype != torch.bfloat16:
            raise MoEEpConfigError("quantize_input=True requires BF16 hidden_states")
    else:
        if t.hidden_states.dtype != torch.float8_e4m3fn:
            raise MoEEpConfigError(
                "pre-quantized hidden_states must be float8_e4m3fn, got "
                f"{t.hidden_states.dtype}"
            )
        sf_cols = fleet_params.token_hidden_size // _SF_VEC
        scales = t.scales
        if (
            scales.ndim != 2
            or scales.shape[0] != t.hidden_states.shape[0]
            or scales.shape[1] < sf_cols
        ):
            raise MoEEpConfigError(
                f"scales must have shape [num_tokens, >= {sf_cols}], got "
                f"{tuple(scales.shape)}"
            )
        if scales.dtype not in (torch.float8_e8m0fnu, torch.uint8):
            raise MoEEpConfigError(
                f"scales must be E8M0 (float8_e8m0fnu or uint8), got {scales.dtype}"
            )
        tensors.append(scales)
    device = torch.device("cuda", torch.cuda.current_device())
    if any(tensor.device != device for tensor in tensors):
        raise MoEEpConfigError("MegaMoE inputs must be on the current CUDA device")


def _fused_prequantized_layouts(t, workspace) -> bool:
    """Whether the one-launch stager takes these pre-quantized inputs.

    It reads and writes every row at once, so inputs that overlap the
    workspace's staging planes keep the Torch path, which runs in sequence.
    """
    inputs = (t.hidden_states, t.scales, t.topk_ids, t.topk_weights)
    planes = (workspace.x, workspace.x_sf, workspace.topk_idx, workspace.topk_weights)
    contiguous = (
        t.hidden_states,
        t.topk_ids,
        t.topk_weights,
        workspace.x,
        workspace.topk_idx,
        workspace.topk_weights,
    )
    if not (
        all(tensor.is_contiguous() for tensor in contiguous)
        and t.scales.stride(1) == 1
        and workspace.x_sf.stride(1) == 1
        and workspace.topk_idx.dtype == torch.int32
    ):
        return False

    def extent(tensor):
        """Byte range of a 2D tensor whose rows have unit column stride."""
        rows, cols = tensor.shape
        size = (rows - 1) * tensor.stride(0) + cols if rows else 0
        return tensor.data_ptr(), tensor.data_ptr() + size * tensor.element_size()

    return not any(
        max(a[0], b[0]) < min(a[1], b[1])
        for a in map(extent, inputs)
        for b in map(extent, planes)
    )


def stage_inputs(t, workspace, *, quantize_input: bool) -> None:
    """Write activations, scales and routing for ``t.num_tokens`` rows.

    Rows past the token count are masked (``topk_idx = -1``) so a smaller
    batch never dispatches rows a previous batch left behind. With a shared
    expert the scales also go to its scale plane. Pre-quantized inputs stage
    in one launch (``staging_kernel``) where Triton is available and the
    layouts allow it, which writes the scale plane for the live rows only:
    the shared expert's outputs past the token count are never read.
    """
    from ......cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.kernel_helpers import (
        quantize_mxfp8_block32,
    )

    num_tokens = t.hidden_states.shape[0]
    sf_cols = workspace.config.hidden // _SF_VEC
    if not quantize_input and _fused_prequantized_layouts(t, workspace):
        try:
            from .staging_kernel import stage_prequantized
        except ImportError:
            stage_prequantized = None
        if stage_prequantized is not None:
            stage_prequantized(
                t.hidden_states,
                t.scales,
                t.topk_ids,
                t.topk_weights,
                workspace,
                sf_cols,
            )
            workspace.note_staged_tokens(num_tokens)
            return
    if num_tokens:
        data = workspace.x[:num_tokens].view(torch.uint8)
        scales = workspace.x_sf[:num_tokens, :sf_cols].view(torch.uint8)
        if quantize_input:
            q, sf = quantize_mxfp8_block32(t.hidden_states, torch.float8_e4m3fn)
            data.copy_(q.view(torch.uint8))
            # The quantizer clamps scale codes to [1, 254]: none is 0xFF.
            scales.copy_(sf.view(torch.uint8))
        else:
            data.copy_(t.hidden_states.view(torch.uint8))
            scales.copy_(t.scales[:, :sf_cols].view(torch.uint8))
            scales.masked_fill_(scales == 0xFF, 0xFE)
        data.masked_fill_(data == 0xFF, 0x7F)
        workspace.topk_idx[:num_tokens].copy_(t.topk_ids)
        workspace.topk_weights[:num_tokens].copy_(t.topk_weights)
    workspace.topk_idx[num_tokens:].fill_(-1)
    if workspace.config.shared_expert is not None:
        workspace.stage_shared_expert_scales()
    workspace.note_staged_tokens(num_tokens)
