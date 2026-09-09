"""SM90 push FP8 mega-MoE backend."""

from .backend import Sm90PushFp8MegaKernelBackend
from .config import Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig
from .weights import (
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from .weights import TransformedMegaWeights

        return TransformedMegaWeights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Sm90PushFp8MegaKernelBackend",
    "Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
