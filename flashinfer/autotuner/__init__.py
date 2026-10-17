"""FlashInfer autotuner.

Environment variables:
    FLASHINFER_AUTOTUNE_INDEPENDENT (default "0")
        Set to "1" to enable independent per-rank profiling: each rank
        profiles with its own timings and handles its own OOM detection,
        skipping all collectives during the tuning loop. Use this on
        homogeneous TP deployments where cold-start autotuning exceeds
        the distributed timeout (fixes #5898). Leave at "0" (default)
        when rank agreement on tactic choice is required.
"""

from flashinfer.autotuner.initializers import (
    autotuner_initializer_empty,
    autotuner_initializer_zeros,
    autotuner_initializer_ones,
    autotuner_initializer_randn,
    autotuner_initializer_rand,
    autotuner_initializer_rand_scaled,
)
from flashinfer.autotuner.autotuner import (
    AutoTuner,
    AutoTunerStatistics,
    ConstraintSpec,
    Dim,
    DynamicDim,
    DynamicTensorSpec,
    OptimizationProfile,
    ProfilingCacheKey,
    StaticDim,
    TunableRunner,
    TuningConfig,
    ValueProfileArena,
    _METADATA_KEY,
    _collect_metadata,
    _json_to_tactic,
    _tactic_to_json,
    _tactic_to_json_hashable,
    autotune,
    get_autotune_process_group,
    is_in_profile_measurement,
    make_bucket_mapper,
    round_to_nearest_bucket,
    set_autotune_process_group,
)

__all__ = [
    "AutoTuner",
    "AutoTunerStatistics",
    "ConstraintSpec",
    "Dim",
    "DynamicDim",
    "DynamicTensorSpec",
    "OptimizationProfile",
    "ProfilingCacheKey",
    "StaticDim",
    "TunableRunner",
    "TuningConfig",
    "ValueProfileArena",
    "_METADATA_KEY",
    "_collect_metadata",
    "_json_to_tactic",
    "_tactic_to_json",
    "_tactic_to_json_hashable",
    "autotune",
    "autotuner_initializer_empty",
    "autotuner_initializer_ones",
    "autotuner_initializer_rand",
    "autotuner_initializer_rand_scaled",
    "autotuner_initializer_randn",
    "autotuner_initializer_zeros",
    "get_autotune_process_group",
    "is_in_profile_measurement",
    "make_bucket_mapper",
    "round_to_nearest_bucket",
    "set_autotune_process_group",
]
