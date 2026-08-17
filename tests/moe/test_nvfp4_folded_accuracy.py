"""Layer-level folded-FP8 accuracy gate for the bundled ModelOpt golden.

This gate covers deterministic weight conversion error. Full-model logits and
perplexity remain external validation requirements before changing the default
weight policy.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from flashinfer.fused_moe.nvfp4_checkpoint import (
    NVFP4Checkpoint,
    load_modelopt_nvfp4_state_dict,
    reference_dequantize_nvfp4,
)
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
    fold_nvfp4_checkpoint_to_fp8_blockscale,
)
from tests.moe.test_nvfp4_checkpoint import _load_safetensors_subset


_GOLDEN = Path(__file__).with_name("data") / "modelopt_w4a16_nvfp4_v1.safetensors"
_PREFIX = "model.layers.0.self_attn.q_proj"
_MIN_COSINE = 0.999
_MAX_NORMALIZED_L2 = 0.03
_MAX_RELATIVE_ERROR = 0.25


def _expanded_golden_checkpoint() -> NVFP4Checkpoint:
    names = tuple(
        f"{_PREFIX}.{suffix}" for suffix in ("weight", "weight_scale", "weight_scale_2")
    )
    tiny = load_modelopt_nvfp4_state_dict(
        _load_safetensors_subset(_GOLDEN, names),
        prefix=_PREFIX,
        logical_shape=(1, 16, 16),
        expert_mapping=(0,),
        source_format_version="nvidia-modelopt.0.45.0.w4a16-nvfp4",
    )
    return NVFP4Checkpoint(
        packed_e2m1=tiny.packed_e2m1.repeat(1, 8, 8).contiguous(),
        scale_e4m3_per16=tiny.scale_e4m3_per16.repeat(1, 8, 8).contiguous(),
        global_alpha=tiny.global_alpha.clone(),
        logical_shape=(1, 128, 128),
        expert_mapping=(0,),
        source_format_version=tiny.source_format_version,
    )


def _decode_folded(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    experts, rows, columns = weight.shape
    return (
        weight.float().reshape(experts, rows // 128, 128, columns // 128, 128)
        * scales[:, :, None, :, None]
    ).reshape(experts, rows, columns)


def test_modelopt_golden_folded_fp8_layer_accuracy_gate() -> None:
    checkpoint = _expanded_golden_checkpoint()
    packed_weight = reference_dequantize_nvfp4(checkpoint).float()
    folded_weight, folded_scales = fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)
    folded_weight = _decode_folded(folded_weight, folded_scales)

    activation = torch.linspace(-1.0, 1.0, 3 * 128, dtype=torch.float32).reshape(3, 128)
    packed_reference = activation @ packed_weight[0].T
    folded = activation @ folded_weight[0].T

    difference = folded - packed_reference
    normalized_l2 = float(
        difference.square().sum().sqrt()
        / packed_reference.square().sum().sqrt().clamp_min(1e-12)
    )
    cosine = float(
        F.cosine_similarity(folded.flatten(), packed_reference.flatten(), dim=0)
    )
    significance_floor = packed_reference.abs().amax().clamp_min(1e-12) * 1e-3
    significant = packed_reference.abs() >= significance_floor
    assert significant.any()
    max_relative_error = float(
        (difference[significant].abs() / packed_reference[significant].abs()).amax()
    )

    assert cosine >= _MIN_COSINE
    assert normalized_l2 <= _MAX_NORMALIZED_L2
    assert max_relative_error <= _MAX_RELATIVE_ERROR
