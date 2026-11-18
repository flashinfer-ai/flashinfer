# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Shape contracts for automatic selective policy selection."""

MODEL_CONFIGS = {
    "DS-V3.2": (64, 128, 2048, 1),
    "DS-V4-Flash": (64, 128, 512, 4),
    "DS-V4-Pro": (64, 128, 1024, 4),
    "GLM-5.3-Flash": (32, 128, 512, 4),
    "GLM-5.3": (32, 128, 2048, 1),
    "Qwen3.8-Flash-Next": (4, 128, 512, 4),
    "Kimi-K3.1-Pro": (32, 128, 2048, 1),
    "DS-V4.1-Flash": (32, 128, 512, 2),
}


def resolve_model_config(heads, head_dim, top_k, compression_ratio, model_config=None):
    """Infer a policy; an optional name asserts the same configuration.

    Unrecognized configurations retain the general numerical contract. In
    particular, H4 inputs do not imply Qwen's constant weights.
    Qwen's numerical constraints apply only to an explicitly named model.
    Kimi shares GLM-5.3's shape and must be named to select its policy.
    """
    config = (heads, head_dim, top_k, compression_ratio)
    if model_config == "DS-V4.1-Flash" and config in (
        (32, 128, 512, 1),
        (32, 128, 512, 2),
    ):
        return model_config
    inferred = next(
        (name for name, value in MODEL_CONFIGS.items() if value == config), None
    )
    if model_config is not None and (
        not isinstance(model_config, str)
        or model_config not in MODEL_CONFIGS
        or MODEL_CONFIGS[model_config] != config
    ):
        raise ValueError(
            "cutedsl model_config must match num_heads, head_dim, top_k and compression_ratio"
        )
    if inferred == "Qwen3.8-Flash-Next" and model_config is None:
        return None
    return model_config if model_config is not None else inferred
