# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Dispatch the independently versioned block-scaled numerical policies."""


def _kind(config):
    kind = getattr(config, "quant_kind", getattr(config, "kind", None))
    if kind is None and type(config).__name__ in (
        "Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
        "Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
        "MegaMoENvfp4Config",
    ):
        kind = "nvfp4"
    if kind not in ("nvfp4", "mxfp4", "mxfp4_mxfp8", "mxfp8_e4m3", "mxfp8_e5m2"):
        raise ValueError(f"unsupported batch-invariant block-scaled kind: {kind!r}")
    return kind


def validate_block_scaled_batch_invariant(
    config, knobs=None, *, implementation=None, arch=None
):
    if type(getattr(config, "batch_invariant", False)) is not bool:
        raise ValueError("batch_invariant must be a bool")
    if not getattr(config, "batch_invariant", False):
        return
    kind = _kind(config)
    if kind == "nvfp4":
        from .nvfp4_batch_invariant import validate_nvfp4_batch_invariant as validate
    elif kind in ("mxfp8_e4m3", "mxfp8_e5m2"):
        from .mxfp8_batch_invariant import validate_mxfp8_batch_invariant as validate
    else:
        from .mxfp4_batch_invariant import validate_mxfp4_batch_invariant as validate
    validate(config, knobs, implementation=implementation, arch=arch)


def block_scaled_batch_invariant_fingerprint(config, **kwargs):
    kind = _kind(config)
    if kind == "nvfp4":
        from .nvfp4_batch_invariant import (
            nvfp4_batch_invariant_fingerprint as fingerprint,
        )
    elif kind in ("mxfp8_e4m3", "mxfp8_e5m2"):
        from .mxfp8_batch_invariant import (
            mxfp8_batch_invariant_fingerprint as fingerprint,
        )
    else:
        from .mxfp4_batch_invariant import (
            mxfp4_batch_invariant_fingerprint as fingerprint,
        )
    return fingerprint(config, **kwargs)


def block_scaled_batch_invariant_version(config):
    kind = _kind(config)
    if kind == "nvfp4":
        from .nvfp4_batch_invariant import NVFP4_BATCH_INVARIANT_VERSION

        return NVFP4_BATCH_INVARIANT_VERSION
    if kind in ("mxfp8_e4m3", "mxfp8_e5m2"):
        from .mxfp8_batch_invariant import MXFP8_BATCH_INVARIANT_VERSION

        return MXFP8_BATCH_INVARIANT_VERSION
    from .mxfp4_batch_invariant import MXFP4_BATCH_INVARIANT_VERSION

    return MXFP4_BATCH_INVARIANT_VERSION


def block_scaled_batch_invariant_metadata(config):
    if not getattr(config, "batch_invariant", False):
        return {}
    kind = _kind(config)
    if kind == "nvfp4":
        from .nvfp4_batch_invariant import nvfp4_batch_invariant_metadata as metadata
    elif kind in ("mxfp8_e4m3", "mxfp8_e5m2"):
        from .mxfp8_batch_invariant import mxfp8_batch_invariant_metadata as metadata
    else:
        from .mxfp4_batch_invariant import mxfp4_batch_invariant_metadata as metadata
    return metadata(config)


def block_scaled_batch_invariant_default_knobs(config, implementation, *, arch=None):
    kind = _kind(config)
    if kind == "nvfp4":
        from .nvfp4_batch_invariant import nvfp4_batch_invariant_default_knobs

        return nvfp4_batch_invariant_default_knobs(implementation, arch=arch)
    if kind in ("mxfp8_e4m3", "mxfp8_e5m2"):
        from .mxfp8_batch_invariant import mxfp8_batch_invariant_default_knobs

        return mxfp8_batch_invariant_default_knobs(
            implementation, quant_kind=kind, arch=arch
        )
    from .mxfp4_batch_invariant import (
        mxfp4_batch_invariant_default_knobs,
        mxfp4_quant_kind,
    )

    return mxfp4_batch_invariant_default_knobs(
        implementation, quant_kind=mxfp4_quant_kind(config), arch=arch
    )
