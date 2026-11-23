"""Host adaptation for the pinned Kimi kernel export."""

from flashinfer.moe_ep.shims.sm100.kimi_k3.weights import (
    KimiWeights,
    interleave_gate_up,
    preprocess_weights,
    swizzle_scales,
    validate_transformed_weights,
)


def create_workspace(config, fleet_params, *, rank, world_size, process_group):
    from flashinfer.moe_ep.shims.sm100.kimi_k3.runtime import KimiWorkspace

    return KimiWorkspace(
        config,
        fleet_params,
        rank=rank,
        world_size=world_size,
        process_group=process_group,
    )


def make_reducer(*args, **kwargs):
    from flashinfer.moe_ep.shims.sm100.kimi_k3.reduce import KimiTopkReducer

    return KimiTopkReducer(*args, **kwargs)


def get_combine_format(name):
    from flashinfer.moe_ep.shims.sm100.kimi_k3.formats import (
        get_combine_format as resolve,
    )

    return resolve(name)


__all__ = [
    "KimiWeights",
    "create_workspace",
    "preprocess_weights",
    "validate_transformed_weights",
    "interleave_gate_up",
    "swizzle_scales",
    "make_reducer",
    "get_combine_format",
]
