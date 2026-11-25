"""Shared-expert configuration, weight ownership and input staging for Rubin."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from .....core.validation.common import MoEEpConfigError
from .....config import Sm107SharedExpertConfig as Sm107SharedExpertConfig


def validate_feature_config(config):
    from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import validate_feature_config

    validate_feature_config(config)


@dataclass(frozen=True)
class Sm107SharedExpertWeights(Sequence):
    """Prepared routed FC1/FC2 pairs and one replicated shared FC1/FC2 pair.

    Indexing and iteration retain the existing routed-pair protocol. The shared
    pair owns its storage separately and is bound along with the routed weights.
    """

    routed: tuple
    shared: tuple

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.routed[index]


def prepare_shared_weights(config, source, routed, fleet):
    shared = config.shared_expert
    pack = getattr(source, "shared_expert", None)
    if shared is None:
        if pack is not None:
            raise MoEEpConfigError("shared weights require shared_expert configuration")
        return routed
    from .....weights import MoEWeightPack
    from .mxfp8_mxfp8_bf16_cutedsl.weights import preprocess_mega_weights

    if not isinstance(pack, MoEWeightPack) or pack.shared_expert is not None:
        raise MoEEpConfigError("shared_expert requires one nonnested MoEWeightPack")
    if pack.w13.shape[0] != 1 or pack.w2.shape[0] != 1:
        raise MoEEpConfigError(
            "shared weights must contain one replicated FFN per rank"
        )
    transformed = preprocess_mega_weights(
        pack,
        intermediate_size=shared.intermediate_size,
        hidden_size=fleet.token_hidden_size,
        kind=shared.kind,
    )
    return Sm107SharedExpertWeights(routed, transformed)


def validate_shared_weights(config, weights, fleet):
    if config.shared_expert is None:
        if isinstance(weights, Sm107SharedExpertWeights):
            raise MoEEpConfigError(
                "prepared shared weights require shared_expert configuration"
            )
        return weights
    if not isinstance(weights, Sm107SharedExpertWeights):
        raise MoEEpConfigError(
            "shared_expert requires Sm107SharedExpertWeights when supplying prepared weights"
        )
    from .mxfp8_mxfp8_bf16_cutedsl.weights import validate_transformed_mega_weights

    shared = config.shared_expert
    validate_transformed_mega_weights(
        weights.shared,
        intermediate_size=shared.intermediate_size,
        hidden_size=fleet.token_hidden_size,
        kind=shared.kind,
        world_size=1,
        num_experts=1,
    )
    if weights.shared[0][0].device != weights.routed[0][0].device:
        raise MoEEpConfigError("routed and shared weights must use the same device")
    return weights.routed


def bind_shared_weights(workspace, weights):
    pair = weights.shared if isinstance(weights, Sm107SharedExpertWeights) else None
    if (
        pair is None
        and getattr(getattr(workspace, "config", None), "shared_expert", None) is None
    ):
        return
    workspace.bind_shared_weights(pair)


def validate_shared_inputs(config, tensors, fleet, *, quantize_input):
    shared = config.shared_expert
    x, sf = (
        getattr(tensors, "shared_hidden_states", None),
        getattr(tensors, "shared_scales", None),
    )
    if shared is None:
        if x is not None or sf is not None:
            raise MoEEpConfigError("shared inputs require shared_expert configuration")
        return
    import torch

    if x is None and sf is not None:
        raise MoEEpConfigError("shared_scales requires shared_hidden_states")
    if x is None:
        routed_kind = getattr(config, "kind", "nvfp4")
        compatible = routed_kind == shared.kind or (
            routed_kind == "mxfp4_mxfp8" and shared.kind == "mxfp8_e4m3"
        )
        if not quantize_input and not compatible:
            raise MoEEpConfigError(
                "provide shared_hidden_states when routed inputs use a different quantized format"
            )
        x, sf = tensors.hidden_states, None if quantize_input else tensors.scales
    dtype = torch.float8_e4m3fn if shared.kind == "mxfp8_e4m3" else torch.float8_e5m2
    if (
        x.shape != (tensors.num_tokens, fleet.token_hidden_size)
        or x.device != tensors.hidden_states.device
    ):
        raise MoEEpConfigError(
            "shared input must match live rows, hidden size and routed device"
        )
    if x.dtype == torch.bfloat16:
        if sf is not None:
            raise MoEEpConfigError("BF16 shared input does not take scales")
    elif (
        x.dtype != dtype
        or sf is None
        or sf.dtype not in (torch.float8_e8m0fnu, torch.uint8)
        or sf.shape != (tensors.num_tokens, fleet.token_hidden_size // 32)
        or sf.device != x.device
    ):
        raise MoEEpConfigError(
            "prequantized shared input requires matching MXFP8 data and row-major E8M0 scales"
        )


@cache
def _shared_bf16_stager():
    import torch

    from flashinfer.utils import version_at_least

    # Torch added the saturating CUDA E4M3 cast in 2.12 for CUDA 13 builds.
    if not (
        version_at_least(torch.__version__, "2.12")
        and version_at_least(torch.version.cuda or "0", "13.0")
    ):
        return None
    # Triton remains optional for the existing shared-expert path.
    try:
        from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.shared_staging import (
            stage_shared_bf16,
        )
    except ModuleNotFoundError as error:
        if error.name == "triton":
            return None
        raise
    return stage_shared_bf16


def stage_shared_inputs(tensors, workspace, *, quantize_input):
    shared = workspace.config.shared_expert
    if shared is None:
        return
    import torch

    x, sf = (
        getattr(tensors, "shared_hidden_states", None),
        getattr(tensors, "shared_scales", None),
    )
    if x is None:
        x, sf = tensors.hidden_states, None if quantize_input else tensors.scales
    # The fused launch pads grid.y to 128 rows and uses int32 offsets.
    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[-1] % 128 == 0
        and shared.kind == "mxfp8_e4m3"
        and os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") != "0"
        and workspace.shared_x.shape[0] <= 65_535 // 128 * 128
        and workspace.shared_x.numel() <= 2**31
        and workspace.shared_x_sf.numel() <= 2**31
    ):
        stage = _shared_bf16_stager()
        if stage is not None:
            stage(x, workspace)
            return

    from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import (
        quantize_mxfp8_block32,
        to_blocked,
    )

    if tensors.num_tokens and x.dtype == torch.bfloat16:
        dtype = (
            torch.float8_e4m3fn if shared.kind == "mxfp8_e4m3" else torch.float8_e5m2
        )
        x, sf = quantize_mxfp8_block32(x, dtype)
    count = tensors.num_tokens
    # The dense shared GEMMs cover capacity, including rows outside the routed
    # live extent. Clear those rows and every scale-padding byte on each replay.
    workspace.shared_x.view(torch.uint8).zero_()
    workspace.shared_x_sf_linear.view(torch.uint8).zero_()
    if count:
        workspace.shared_x[:count].view(torch.uint8).copy_(x.view(torch.uint8))
        workspace.shared_x_sf_linear[:count].view(torch.uint8).copy_(
            sf.view(torch.uint8)
        )
    workspace.shared_x_sf.view(torch.uint8).copy_(
        to_blocked(workspace.shared_x_sf_linear.view(torch.uint8))
    )
