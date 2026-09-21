# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Shared geometry policy for dtype-specific Frost automatic candidates."""

_SHORTLIST_TOP_K = {
    (12, 7168, 3072): (1, 2, 4),
    (8, 4096, 14336): (2,),
    (64, 2048, 1408): (6,),
}


def shortlisted_moe_geometry(config, act):
    """Check the model/top-k and token range covered by offline stage selection."""
    geometry = (
        config.routing.num_experts,
        act.hidden_states_q.shape[1],
        config.experts.intermediate_size,
    )
    return (
        config.routing.top_k in _SHORTLIST_TOP_K.get(geometry, ())
        and 0 < act.num_tokens <= 12288
    )
