"""Public shim exports for the vendored SM120 W4A8 split kernel."""

from ._paths import bootstrap_paths

bootstrap_paths()

from .runtime import (  # noqa: E402
    MegaMoESm120W4A8Config,
    MegaMoESm120W4A8Inputs,
    MegaMoESm120W4A8Workspace,
    allocate_workspace,
    run_split_mega_moe,
)
from .staging import ACTIVATION_DTYPE, stage_inputs  # noqa: E402
from .weights import (  # noqa: E402
    FP4_DTYPE,
    SCALE_DTYPE,
    TransformedWeights,
    transform_prequantized_weights,
)

__all__ = [
    "ACTIVATION_DTYPE",
    "FP4_DTYPE",
    "MegaMoESm120W4A8Config",
    "MegaMoESm120W4A8Inputs",
    "MegaMoESm120W4A8Workspace",
    "SCALE_DTYPE",
    "TransformedWeights",
    "allocate_workspace",
    "bootstrap_paths",
    "run_split_mega_moe",
    "stage_inputs",
    "transform_prequantized_weights",
]
