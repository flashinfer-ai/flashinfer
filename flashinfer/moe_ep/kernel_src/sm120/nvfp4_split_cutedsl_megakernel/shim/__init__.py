"""Public shim exports for the vendored SM120 NVFP4 split kernel."""

from ._paths import bootstrap_paths

bootstrap_paths()

from .runtime import (  # noqa: E402
    DECODE_GRAPH_COMPILE_BUCKETS,
    MegaMoESm120Nvfp4Config,
    MegaMoESm120Nvfp4Inputs,
    MegaMoESm120Nvfp4Workspace,
    allocate_workspace,
    run_split_mega_moe,
    select_graph_compile_bucket,
    set_compile_tokens_per_rank,
)
from .staging import ACTIVATION_DTYPE, stage_inputs  # noqa: E402
from .weights import (  # noqa: E402
    FP4_DTYPE,
    SCALE_DTYPE,
    TransformedWeights,
    ceil_div,
    round_up,
    scale_storage_size,
    transform_weights,
)

__all__ = [
    "DECODE_GRAPH_COMPILE_BUCKETS",
    "ACTIVATION_DTYPE",
    "FP4_DTYPE",
    "MegaMoESm120Nvfp4Config",
    "MegaMoESm120Nvfp4Inputs",
    "MegaMoESm120Nvfp4Workspace",
    "SCALE_DTYPE",
    "TransformedWeights",
    "allocate_workspace",
    "bootstrap_paths",
    "run_split_mega_moe",
    "select_graph_compile_bucket",
    "set_compile_tokens_per_rank",
    "stage_inputs",
    "ceil_div",
    "round_up",
    "scale_storage_size",
    "transform_weights",
]
