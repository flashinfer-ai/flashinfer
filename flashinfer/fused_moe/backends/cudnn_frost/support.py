# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Weight-view aliases and input geometry policy for Frost MoELayer candidates."""

# Every dtype packages sources for these exact targets; no other architecture
# runs them.
SUPPORTED_ARCHS = ("sm_107a", "sm_120a")
SUPPORTED_ARCH_NUMBERS = (107, 120)


def weight_view(weights, backend_key, canonical_key):
    """Use the backend's named view, or share an existing canonical view."""
    key = backend_key if backend_key in weights.native_views else canonical_key
    return weights.get_view(key)


def moe_geometry(config, act, *, hidden_size=None):
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
