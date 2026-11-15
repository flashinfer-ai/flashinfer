# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The sampling classes' entries of the mode's registry (:data:`flashinfer.batch_invariant.CLASSES`): the class name and
the fingerprint behind ``fingerprint("softmax", N=, dtype=, temperature=[, out_dtype=])``, ``fingerprint("log_softmax",
...)``, ``fingerprint("log_probs", ...)``, ``fingerprint("sampling_from_probs", N=, dtype=)`` and
``fingerprint("sampling_from_logits", N=, dtype=)``.

The first three APIs run one class, ``lse.row``, whose specification is a function of the vocabulary ``N`` (the two row
trees), the input dtype (the output conversion of the probabilities; the leaves of a 16-bit input are its exact
widenings, so ``softmax`` with ``out_dtype=torch.float32`` is the fp32 input's specification) and whether a temperature
enters the scale (``c = LOG2E`` or ``div(LOG2E, T)``). The two samplers run the draw class ``sample.icdf.v1``
(``draw_spec.py``): its specification is a function of the vocabulary (the sum tree and the prefix nodes read from it)
and of whether the weights are the given probabilities or the ``lse.row`` weights of the logits; the input dtype is a
validated dimension that enters nothing (the leaves of a 16-bit input are its exact widenings and the draw converts no
output)."""

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


def _lse_row_fingerprint(N: int, dtype, temperature, out_dtype=None) -> str:
    from .spec import spec_fingerprint

    in_fmt = _in_fmt(dtype)
    if out_dtype is not None:
        out_fmt = _in_fmt(out_dtype)
        if out_fmt == "f32":
            in_fmt = "f32"  # the probabilities before the output conversion: the fp32 input's specification
        elif out_fmt != in_fmt:
            raise ValueError(
                f"out_dtype {out_dtype!r}: the input dtype or f32 (the class converts its probabilities to the input dtype or leaves them in fp32)"
            )
    return spec_fingerprint(int(N), in_fmt, bool(temperature))


def _draw_fingerprint(N: int, dtype, from_logits: bool = False) -> str:
    from .draw_spec import spec_fingerprint

    in_fmt = _in_fmt(dtype)
    return spec_fingerprint(int(N), bool(from_logits), in_fmt)


def _draw_probs_fingerprint(N: int, dtype) -> str:
    return _draw_fingerprint(N, dtype, False)


def _draw_logits_fingerprint(N: int, dtype) -> str:
    return _draw_fingerprint(N, dtype, True)


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the sampling classes to the mode's registry (called by ``flashinfer.batch_invariant`` at import): ``softmax``,
    ``log_softmax`` and ``log_probs`` take the vocabulary ``N``, the input ``dtype`` and whether a ``temperature`` is
    given; ``softmax`` optionally ``out_dtype`` (f32: the probabilities before the output conversion);
    ``sampling_from_probs`` and ``sampling_from_logits`` (the draw class) take ``N`` and the input ``dtype``."""
    dims = ("N", "dtype", "temperature")
    classes["softmax"] = class_info(
        "lse.row", dims, _lse_row_fingerprint, ("out_dtype",)
    )
    classes["log_softmax"] = class_info("lse.row", dims, _lse_row_fingerprint)
    classes["log_probs"] = class_info("lse.row", dims, _lse_row_fingerprint)
    # the inverse-CDF draw (draw_api.py): one class with two weight forms (the given probabilities / the lse.row weights
    # of the logits), hence two fingerprints per vocabulary; the input dtype enters nothing
    classes["sampling_from_probs"] = class_info(
        "sample.icdf.v1", ("N", "dtype"), _draw_probs_fingerprint
    )
    classes["sampling_from_logits"] = class_info(
        "sample.icdf.v1", ("N", "dtype"), _draw_logits_fingerprint
    )
