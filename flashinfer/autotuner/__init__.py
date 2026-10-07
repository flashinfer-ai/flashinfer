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
