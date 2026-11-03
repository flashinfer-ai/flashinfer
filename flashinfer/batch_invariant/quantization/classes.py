# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The quantization classes' entries of the mode's registry (:data:`flashinfer.batch_invariant.CLASSES`): the class names
and the fingerprints behind ``fingerprint("mxfp8_quantize")``, ``fingerprint("mxfp4_quantize")``,
``fingerprint("nvfp4_quantize")`` and ``fingerprint("nvfp4_quantize_per_token", K=)``. A quantizer's class has no shape dimension: one specification per format
serves every row count and row length (the scale-factor layouts are placements of the store), so the fingerprints take
no dimensions; the NVFP4 per-token variant is a different class (its global scale is derived from the row)."""

from __future__ import annotations

from typing import Any, Dict


def _mxfp8_quant_fingerprint() -> str:
    from .mx_spec import MXFP8, spec_fingerprint

    return spec_fingerprint(MXFP8)


def _mxfp4_quant_fingerprint() -> str:
    from .mx_spec import MXFP4, spec_fingerprint

    return spec_fingerprint(MXFP4)


def _nvfp4_quant_fingerprint() -> str:
    from .nvfp4_spec import PER_TENSOR, class_fingerprint

    return class_fingerprint(PER_TENSOR)


def _nvfp4_quant_per_token_fingerprint(K: int) -> str:
    from .nvfp4_spec import PER_TOKEN, class_fingerprint

    return class_fingerprint(PER_TOKEN, int(K))


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the quantization classes to the mode's registry (called by ``flashinfer.batch_invariant`` at import):
    ``mxfp8_quantize``, ``mxfp4_quantize`` and ``nvfp4_quantize`` (the per-tensor class) take no dimension;
    ``nvfp4_quantize_per_token`` (what ``nvfp4_quantize(..., per_token_activation=True)`` runs) takes ``K``."""
    classes["mxfp8_quantize"] = class_info("mxfp8.quant", (), _mxfp8_quant_fingerprint)
    classes["mxfp4_quantize"] = class_info("mxfp4.quant", (), _mxfp4_quant_fingerprint)
    classes["nvfp4_quantize"] = class_info("nvfp4.quant", (), _nvfp4_quant_fingerprint)
    classes["nvfp4_quantize_per_token"] = class_info(
        "nvfp4.quant.pertoken", ("K",), _nvfp4_quant_per_token_fingerprint
    )
