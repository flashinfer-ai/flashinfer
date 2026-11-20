"""DeepSeek V4 Pro EP4 parameters, indexed by tokens per rank.

These are the original-pull configurations used for the calibrated CI baseline.
They retain supported settings from the tuning study; this file does not claim
an exhaustive retuning of the current main branch. Use get_params() to see the
complete arguments for one token size. Importing this module needs no CUDA.
"""

# Shared workload and default kernel arguments. None disables an optional flag.
BASE_PARAMS = dict(
    topk=6,
    experts=384,
    hidden=7168,
    intermediate=3072,
    quant_kind="nvfp4",
    gateup_clamp="10",
    topk_apply_position="pre_fc1_quant",
    route_distribution="balanced",
    combine_format="bf16",
    token_back_mode="epi_warps",
    work_id_mode="atomic_counter",
    cluster_shape_mnk="4,1,1",
    fallback_cluster_shape_mnk="2,1,1",
    fc2_use_bulk=True,
    seed=45494,
)

# Per-token kernel settings override the shared arguments above.
COMMON = dict(mma_tiler_mnk="256,256,256", schedule_policy="phase_interleave,3",
              token_in_flag_batch=4, fc2_tma_stages=2, epi_flag_batches="1,4")
# Only arguments supported by the unmodified main runner. Dispatch uses its
# default fused pull path; no push, packing, tail-TMA or custom dispatch knobs.
PULL = dict(COMMON, fc2_tma_stages=1)
BEST_PARAMS = {
    128: dict(COMMON),
    256: dict(COMMON, cluster_shape_mnk="2,1,1", fallback_cluster_shape_mnk=None),
    512: dict(PULL),
    1024: dict(PULL),
    2048: dict(PULL),
    4096: dict(PULL, schedule_policy="phase_interleave"),
    8192: dict(PULL, epi_flag_batches="1,1"),
    16384: dict(PULL),
    32768: dict(PULL),
}


def get_params(tokens_per_rank):
    """Return a fresh dictionary with the complete config for one token size."""
    return dict(max_tokens_per_rank=tokens_per_rank, **BASE_PARAMS) | BEST_PARAMS[tokens_per_rank]
