# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The sampling class's entries of the mode's registry (:data:`flashinfer.batch_invariant.CLASSES`): the class name and
the fingerprint behind ``fingerprint("softmax", N=, dtype=, temperature=)`` and ``fingerprint("log_probs", ...)``.

The two APIs run one class, ``lse.row``, whose specification is a function of the vocabulary ``N`` (the two row trees),
the input dtype (the output conversion of the probabilities; the leaves of a 16-bit input are its exact widenings) and
whether a temperature enters the scale (``c = LOG2E`` or ``div(LOG2E, T)``)."""

from __future__ import annotations

from typing import Any, Dict

_DTYPE_NAMES = {
    "bf16": "bf16",
    "f16": "f16",
    "fp16": "f16",
    "f32": "f32",
    "fp32": "f32",
}


def _in_fmt(dtype) -> str:
    """The class's name of an input dtype: ``"bf16"`` / ``"f16"`` / ``"f32"`` (a string or a ``torch.dtype``)."""
    if isinstance(dtype, str):
        if dtype in _DTYPE_NAMES:
            return _DTYPE_NAMES[dtype]
        raise ValueError(f"dtype {dtype!r}: one of bf16 / f16 / f32")
    import torch

    names = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
    if dtype in names:
        return names[dtype]
    raise ValueError(f"dtype {dtype!r}: one of torch.bfloat16 / float16 / float32")


def _lse_row_fingerprint(N: int, dtype, temperature) -> str:
    from .spec import spec_fingerprint

    return spec_fingerprint(int(N), _in_fmt(dtype), bool(temperature))


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the sampling class to the mode's registry (called by ``flashinfer.batch_invariant`` at import): ``softmax``
    and ``log_probs`` both take the vocabulary ``N``, the input ``dtype`` and whether a ``temperature`` is given."""
    dims = ("N", "dtype", "temperature")
    classes["softmax"] = class_info("lse.row", dims, _lse_row_fingerprint)
    classes["log_probs"] = class_info("lse.row", dims, _lse_row_fingerprint)
