# SPDX-License-Identifier: Apache-2.0
"""SM107 NVFP4 MegaMoE for callers supplying kernel-layout weights."""

from .kernel_src.sm107.next_cutedsl_megamoe.shim.submission_config import (
    bucket_for,
    decode_kernel,
    geometry_unsupported_reason,
    l2_margin,
    layer_buckets,
)
from .kernel_src.sm107.next_cutedsl_megamoe.shim.submission_genphase import (
    GenphaseRuntime,
)
from .kernel_src.sm107.next_cutedsl_megamoe.shim.submission_runtime import (
    Nvfp4MegaMoERuntime,
)
from .kernel_src.sm107.next_cutedsl_megamoe.shim.submission_staging import (
    scale_multiplier,
)

__all__ = [
    "GenphaseRuntime",
    "Nvfp4MegaMoERuntime",
    "bucket_for",
    "decode_kernel",
    "geometry_unsupported_reason",
    "l2_margin",
    "layer_buckets",
    "scale_multiplier",
]
