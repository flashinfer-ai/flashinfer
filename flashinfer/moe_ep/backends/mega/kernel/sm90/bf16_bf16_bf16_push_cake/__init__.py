"""SM90 native BF16 (Cake-generated GEMMs) push mega-MoE backend."""

from .cake_backend import Sm90CakeBf16MegaKernelBackend
from .cake_config import Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig
from .cake_weights import (
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from .cake_weights import TransformedMegaWeights

        return TransformedMegaWeights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Sm90CakeBf16MegaKernelBackend",
    "Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
