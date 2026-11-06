# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The sampling class's entries of the mode's registry (:data:`flashinfer.batch_invariant.CLASSES`): the class name and
the fingerprint behind ``fingerprint("softmax", N=, dtype=, temperature=[, out_dtype=])``, ``fingerprint("log_softmax",
...)``, ``fingerprint("log_probs", ...)`` and ``fingerprint("entropy", ...)``.

The four APIs run one class, ``lse.row``, whose specification is a function of the vocabulary ``N`` (the row trees),
the input dtype (the output conversion of the probabilities; the leaves of a 16-bit input are its exact widenings, so
``softmax`` with ``out_dtype=torch.float32`` is the fp32 input's specification) and whether a temperature enters the
scale (``c = LOG2E`` or ``div(LOG2E, T)``). The entropy member (``entropy``, and ``log_probs(return_entropy=True)``)
carries its own fingerprint: its third tree and chain; the input dtype does not enter it. The tensor-parallel entries
(``log_probs_tp_partials``, ``log_probs_tp_merge``, ``log_probs_tp``, ``log_softmax_tp``, ``softmax_tp``; ``tp_api.py``)
are the class itself on a vocabulary shard: a rank's partials are the class's own sub-trees and the merge rebuilds its
trees, so their outputs carry the class's fingerprint -- the wire format (``shards.ShardLayout``) is a layout, not a
class."""

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


def _entropy_fingerprint(N: int, dtype, temperature) -> str:
    """The entropy member's fingerprint: the same dimensions as the class's (the dtype is validated and does not enter:
    the leaves of every input dtype are exact widenings)."""
    from .spec import entropy_fingerprint

    _in_fmt(dtype)
    return entropy_fingerprint(int(N), bool(temperature))


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the sampling class to the mode's registry (called by ``flashinfer.batch_invariant`` at import): ``softmax``,
    ``log_softmax``, ``log_probs`` and ``entropy`` take the vocabulary ``N``, the input ``dtype`` and whether a
    ``temperature`` is given; ``softmax`` optionally ``out_dtype`` (f32: the probabilities before the output
    conversion); ``entropy`` names the member's own fingerprint."""
    dims = ("N", "dtype", "temperature")
    classes["softmax"] = class_info(
        "lse.row", dims, _lse_row_fingerprint, ("out_dtype",)
    )
    classes["log_softmax"] = class_info("lse.row", dims, _lse_row_fingerprint)
    classes["log_probs"] = class_info("lse.row", dims, _lse_row_fingerprint)
    classes["entropy"] = class_info("lse.row", dims, _entropy_fingerprint)
    # the tensor-parallel protocol (tp_api.py): the same class and fingerprint -- a rank's partials are the class's own
    # sub-trees over its shard and the merge rebuilds the class's trees, so the merged statistics and every sharded output
    # are the full-row bits; the wire format (shards.ShardLayout) is a layout of those sub-trees, not a class
    for tp_op in (
        "log_probs_tp_partials",
        "log_probs_tp_merge",
        "log_probs_tp",
        "log_softmax_tp",
        "softmax_tp",
    ):
        classes[tp_op] = class_info("lse.row", dims, _lse_row_fingerprint)
