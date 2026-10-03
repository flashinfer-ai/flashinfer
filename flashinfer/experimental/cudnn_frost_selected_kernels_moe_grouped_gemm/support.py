# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Shared geometry policy for dtype-specific Frost automatic candidates."""

_SHORTLIST_TOP_K = {
    (12, 7168, 3072): (1, 2, 4),
    (8, 4096, 14336): (2,),
    (64, 2048, 1408): (6,),
}


def shortlisted_moe_geometry(config, act, *, hidden_size=None):
    """Check the model/top-k and token range covered by offline stage selection."""
    geometry = (
        config.routing.num_experts,
        act.hidden_states_q.shape[1] if hidden_size is None else hidden_size,
        config.experts.intermediate_size,
    )
    return (
        config.routing.top_k in _SHORTLIST_TOP_K.get(geometry, ())
        and 0 < act.num_tokens <= 12288
    )


def quantized_moe_geometry(config, act, *, hidden_size=None):
    """Cheap native-plan bounds; artifact contracts are checked by the runner."""
    t = act.num_tokens
    h = act.hidden_states_q.shape[1] if hidden_size is None else hidden_size
    i = config.experts.intermediate_size
    e, k = config.routing.num_experts, config.routing.top_k
    return (
        0 < t <= min(1 << 20, config.execution.tune_max_num_tokens)
        and 0 < k <= e <= 1024
        and t * k < 2**31 - 128 * e
        and 0 < h <= 1 << 20
        and 0 < i <= 1 << 20
        and h % 128 == 0
        and i % 128 == 0
    )
