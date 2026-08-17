"""SM90 push BF16 mega-MoE backend."""

from .backend import Sm90PushBf16MegaKernelBackend
from .config import Sm90PushBf16MegaMoeConfig
from .weights import preprocess_mega_weights, validate_transformed_mega_weights


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from .weights import TransformedMegaWeights

        return TransformedMegaWeights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Sm90PushBf16MegaKernelBackend",
    "Sm90PushBf16MegaMoeConfig",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
