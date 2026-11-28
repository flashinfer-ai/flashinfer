# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""K-stage alternatives that preserve the canonical native MMA fold."""


def quantized_batch_invariant_tiles(kind, *, arch, implementation="standard"):
    if implementation != "standard":
        raise ValueError("K-stage alternatives require standard inference")
    if arch not in ("sm_100", "sm_103", "sm_107"):
        raise ValueError("unsupported quantized MegaMoE architecture")
    if kind not in ("nvfp4", "mxfp4", "mxfp4_mxfp8", "mxfp8_e4m3", "mxfp8_e5m2"):
        raise ValueError("unsupported quantized MegaMoE kind")
    canonical = 256 if arch == "sm_107" and kind in ("nvfp4", "mxfp4") else 128
    return ((256, 128, canonical), (256, 128, 2 * canonical))


def validate_quantized_batch_invariant_tile(
    config, *, kind, arch, knobs=None, hidden=None, intermediate=None
):
    """Reject changes to zero padding, native instruction, M/N or CTA geometry."""
    values = knobs or {}
    tiles = quantized_batch_invariant_tiles(kind, arch=arch)
    tile = (
        values.get("mma_tiler_mnk", getattr(config, "mma_tiler_mnk", None)) or tiles[0]
    )
    if (
        type(tile) is not tuple
        or any(type(x) is not int for x in tile)
        or tile not in tiles
    ):
        raise ValueError(
            f"batch_invariant unvalidated mma_tiler_mnk K-stage geometry {tile!r}"
        )
    if tile == tiles[0]:
        return
    if getattr(config, "combine_dtype", "bf16") != "bf16" or getattr(
        config, "enable_fc1_activation_per_token_scale", False
    ):
        raise ValueError(
            "batch_invariant alternative K stages require BF16 combine without per-token FC1 scaling"
        )
    h = hidden if hidden is not None else getattr(config, "hidden", None)
    i = (
        intermediate
        if intermediate is not None
        else getattr(config, "intermediate", getattr(config, "intermediate_size", None))
    )
    for name, value in (("hidden", h), ("intermediate", i)):
        if value is None:
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"batch_invariant {name} must be a positive integer")
        canonical = tiles[0][2]
        if (value + tile[2] - 1) // tile[2] * tile[2] != (
            value + canonical - 1
        ) // canonical * canonical:
            raise ValueError(
                f"batch_invariant alternative tile K changes canonical {name} padding"
            )
