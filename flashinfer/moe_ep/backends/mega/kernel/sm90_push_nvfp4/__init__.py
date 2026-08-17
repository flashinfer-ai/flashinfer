"""SM90 push NVFP4 mega-MoE backend."""

from .backend import Sm90PushNvFp4MegaKernelBackend
from .config import Sm90PushNvFp4MegaMoeConfig
from .weights import (
    estimate_residency,
    load_modelopt_dual_weights,
    load_modelopt_folded_fp8_weights,
    load_modelopt_hot_folded_weights,
    load_modelopt_transformed_weights,
    make_folded_fp8_weights_from_checkpoints,
    make_dual_weights_from_checkpoints,
    make_hot_folded_weights_from_checkpoints,
    make_transformed_weights_from_checkpoints,
    preprocess_mega_weights,
    quantize_bf16_to_nvfp4_checkpoint,
    validate_transformed_mega_weights,
)


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from .weights import TransformedMegaWeights

        return TransformedMegaWeights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Sm90PushNvFp4MegaKernelBackend",
    "Sm90PushNvFp4MegaMoeConfig",
    "TransformedMegaWeights",
    "estimate_residency",
    "load_modelopt_dual_weights",
    "load_modelopt_folded_fp8_weights",
    "load_modelopt_hot_folded_weights",
    "load_modelopt_transformed_weights",
    "make_folded_fp8_weights_from_checkpoints",
    "make_dual_weights_from_checkpoints",
    "make_hot_folded_weights_from_checkpoints",
    "make_transformed_weights_from_checkpoints",
    "preprocess_mega_weights",
    "quantize_bf16_to_nvfp4_checkpoint",
    "validate_transformed_mega_weights",
]
