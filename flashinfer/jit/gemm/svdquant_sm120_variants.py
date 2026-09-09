"""Names shared by SM120 SVDQuant LoRA diagnostic variants."""

from typing import Final


LORA_LADDER_RUNGS: Final[tuple[str, ...]] = (
    "residual",
    "transfer",
    "production_residual",
    "production_transfer",
)
PRODUCTION_LADDER_RUNGS: Final[tuple[str, ...]] = (
    "production_residual",
    "production_transfer",
)
