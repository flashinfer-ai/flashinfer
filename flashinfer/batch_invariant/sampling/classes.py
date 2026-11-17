# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The sampling classes' entries of the mode's registry (:data:`flashinfer.batch_invariant.CLASSES`): the class name and
the fingerprint behind ``fingerprint("softmax", N=, dtype=, temperature=[, out_dtype=])``, ``fingerprint("log_softmax",
N=, dtype=, temperature=[, out_dtype=])``, ``fingerprint("log_probs", N=, dtype=, temperature=[, top_k=, top_p=, min_p=,
filter_apply_order=])`` (with a filter the kept-set log-probability: the row class's receipt joined with the support
class's),
``fingerprint("sampling_from_probs", N=, dtype=)``, ``fingerprint("sampling_from_logits", N=, dtype=[, temperature=])``,
and the truncated samplers' ``fingerprint("top_k_sampling_from_probs", N=, dtype=)``,
``fingerprint("top_p_sampling_from_probs", ...)``, ``fingerprint("min_p_sampling_from_probs", ...)``,
``fingerprint("top_k_top_p_sampling_from_probs", N=, dtype=[, filter_apply_order=])`` and
``fingerprint("top_k_top_p_sampling_from_logits", N=, dtype=[, filter_apply_order=, temperature=])``.

The first three APIs run one class, ``lse.row``, whose specification is a function of the vocabulary ``N`` (the two row
trees), the input dtype (the output conversion of the probabilities; the leaves of a 16-bit input are its exact
widenings, so ``softmax`` with ``out_dtype=torch.float32`` is the fp32 input's specification) and whether a temperature
enters the scale (``c = LOG2E`` or ``div(LOG2E, T)``). The two samplers run the draw class ``sample.icdf.v1``
(``draw_spec.py``): its specification is a function of the vocabulary (the sum tree and the prefix nodes read from it)
and of whether the weights are the given probabilities or the ``lse.row`` weights of the logits; the input dtype is a
validated dimension that enters nothing (the leaves of a 16-bit input are its exact widenings and the draw converts no
output). The five truncated samplers run no class of their own: the kept set is ``support.row.v1`` (``support_spec.py``:
the filter set and order, the selection rules, the vocabulary's sum tree and the weight form) and the draw over it is
``sample.icdf.v1``'s reading of the masked tree, so their receipt joins the two fingerprints."""

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


def _log_softmax_fingerprint(N: int, dtype, temperature, out_dtype=None) -> str:
    """``log_softmax``'s receipt: the class fingerprint, or with ``out_dtype`` the input's 16-bit dtype the 16-bit output
    member's (the log-softmax element converted once to the input format: a different output, a different receipt)."""
    from .spec import spec_fingerprint

    in_fmt = _in_fmt(dtype)
    cvt = False
    if out_dtype is not None:
        out_fmt = _in_fmt(out_dtype)
        if out_fmt == in_fmt and in_fmt != "f32":
            cvt = True
        elif out_fmt != "f32":
            raise ValueError(
                f"out_dtype {out_dtype!r}: the input dtype (the 16-bit output member) or f32 (the fp32 row)"
            )
    return spec_fingerprint(int(N), in_fmt, bool(temperature), logprobs_cvt=cvt)


def _draw_fingerprint(
    N: int, dtype, from_logits: bool = False, temperature: bool = False
) -> str:
    from .draw_spec import spec_fingerprint

    in_fmt = _in_fmt(dtype)
    return spec_fingerprint(int(N), bool(from_logits), in_fmt, bool(temperature))


def _support_fingerprint(
    N: int, dtype, filters, from_logits: bool, temperature: bool = False
) -> str:
    """The support class's fingerprint (``support_spec.py``): the filter set, the selection rules, the vocabulary
    (the sum tree the kept mass is read from), the weight form (with or without a temperature in the scale) and the
    output format. The class has no registry entry of its own: the truncated samplers' and ``log_probs``' receipts join
    it (the operators' renormalisation entries keep their own kernels; a flag there is deferred)."""
    from .support_spec import spec_fingerprint

    return spec_fingerprint(
        filters, int(N), bool(from_logits), _in_fmt(dtype), bool(temperature)
    )


def _log_probs_fingerprint(
    N: int,
    dtype,
    temperature,
    top_k=None,
    top_p=None,
    min_p=None,
    filter_apply_order: str = "top_k_first",
) -> str:
    """``log_probs``'s receipt: the row class's, or with a filter the kept-set log-probability's -- the row class's (the
    token's chain, the temperature form) joined with the support class's (the filter set and order, the vocabulary's
    tree); the parameter values move nothing, their presence does."""
    from .support_spec import Filters

    base = _lse_row_fingerprint(N, dtype, temperature)
    if top_k is None and top_p is None and min_p is None:
        return base
    filters = Filters(
        top_k=1 if top_k is not None else None,
        top_p=1.0 if top_p is not None else None,
        min_p=0.0 if min_p is not None else None,
        order=filter_apply_order,
    )
    return f"{base}+{_support_fingerprint(N, dtype, filters, True, bool(temperature))}"


def _draw_support_fingerprint(
    N: int, dtype, filters, from_logits: bool, temperature: bool = False
) -> str:
    """The truncated samplers' receipt: the draw class's fingerprint (the tree the kept mass and the prefixes are read
    from) joined with the support class's (the kept set); no class of its own. For logits both halves embed the scale
    ``c``, so a temperature in the scale is another receipt."""
    return (
        f"{_draw_fingerprint(N, dtype, from_logits, temperature)}+"
        f"{_support_fingerprint(N, dtype, filters, from_logits, temperature)}"
    )


def _top_k_sampling_fingerprint(N: int, dtype) -> str:
    from .support_spec import Filters

    return _draw_support_fingerprint(N, dtype, Filters(top_k=1), False)


def _top_p_sampling_fingerprint(N: int, dtype) -> str:
    from .support_spec import Filters

    return _draw_support_fingerprint(N, dtype, Filters(top_p=1.0), False)


def _min_p_sampling_fingerprint(N: int, dtype) -> str:
    from .support_spec import Filters

    return _draw_support_fingerprint(N, dtype, Filters(min_p=0.0), False)


def _top_k_top_p_sampling_probs_fingerprint(
    N: int, dtype, filter_apply_order: str = "top_k_first"
) -> str:
    from .support_spec import Filters

    return _draw_support_fingerprint(
        N, dtype, Filters(top_k=1, top_p=1.0, order=filter_apply_order), False
    )


def _top_k_top_p_sampling_logits_fingerprint(
    N: int, dtype, filter_apply_order: str = "top_k_first", temperature: bool = False
) -> str:
    from .support_spec import Filters

    return _draw_support_fingerprint(
        N,
        dtype,
        Filters(top_k=1, top_p=1.0, order=filter_apply_order),
        True,
        bool(temperature),
    )


def _draw_probs_fingerprint(N: int, dtype) -> str:
    return _draw_fingerprint(N, dtype, False)


def _draw_logits_fingerprint(N: int, dtype, temperature=False) -> str:
    """The logits form's receipt: the draw over ``lse.row``'s weights, with or without a temperature in the scale."""
    return _draw_fingerprint(N, dtype, True, bool(temperature))


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the sampling classes to the mode's registry (called by ``flashinfer.batch_invariant`` at import): ``softmax``,
    ``log_softmax`` and ``log_probs`` take the vocabulary ``N``, the input ``dtype`` and whether a ``temperature`` is
    given; ``softmax`` optionally ``out_dtype`` (f32: the probabilities before the output conversion), ``log_softmax``
    optionally ``out_dtype`` (the input's 16-bit dtype: the converted row, the 16-bit output member);
    ``sampling_from_probs`` and ``sampling_from_logits`` (the draw class) take ``N`` and the input ``dtype``,
    ``sampling_from_logits`` optionally whether a ``temperature`` enters the scale; the five truncated samplers take ``N``
    and ``dtype``, the two top-k + top-p ones optionally ``filter_apply_order``, the logits one optionally whether a
    ``temperature`` enters the scale; ``log_probs`` optionally the filters (``top_k`` / ``top_p`` / ``min_p`` /
    ``filter_apply_order``: the kept-set log-probability's joined receipt)."""
    dims = ("N", "dtype", "temperature")
    classes["softmax"] = class_info(
        "lse.row", dims, _lse_row_fingerprint, ("out_dtype",)
    )
    classes["log_softmax"] = class_info(
        "lse.row", dims, _log_softmax_fingerprint, ("out_dtype",)
    )
    classes["log_probs"] = class_info(
        "lse.row",
        dims,
        _log_probs_fingerprint,
        ("top_k", "top_p", "min_p", "filter_apply_order"),
    )
    # the inverse-CDF draw (draw_api.py): one class with two weight forms (the given probabilities / the lse.row weights
    # of the logits, with or without a temperature in the scale), hence the fingerprints per vocabulary; the input dtype
    # enters nothing
    classes["sampling_from_probs"] = class_info(
        "sample.icdf.v1", ("N", "dtype"), _draw_probs_fingerprint
    )
    classes["sampling_from_logits"] = class_info(
        "sample.icdf.v1", ("N", "dtype"), _draw_logits_fingerprint, ("temperature",)
    )
    # the truncated samplers (draw_support_api.py): the support class's kept set, then the draw class over it -- the
    # fingerprint joins the two (the filter set per entry, the order a dimension of the top-k + top-p entries); the dtype
    # is validated and enters nothing (both constituents widen a 16-bit input exactly and convert no output)
    joined = "sample.icdf.v1+support.row.v1"
    classes["top_k_sampling_from_probs"] = class_info(
        joined, ("N", "dtype"), _top_k_sampling_fingerprint
    )
    classes["top_p_sampling_from_probs"] = class_info(
        joined, ("N", "dtype"), _top_p_sampling_fingerprint
    )
    classes["min_p_sampling_from_probs"] = class_info(
        joined, ("N", "dtype"), _min_p_sampling_fingerprint
    )
    classes["top_k_top_p_sampling_from_probs"] = class_info(
        joined,
        ("N", "dtype"),
        _top_k_top_p_sampling_probs_fingerprint,
        ("filter_apply_order",),
    )
    classes["top_k_top_p_sampling_from_logits"] = class_info(
        joined,
        ("N", "dtype"),
        _top_k_top_p_sampling_logits_fingerprint,
        ("filter_apply_order", "temperature"),
    )
