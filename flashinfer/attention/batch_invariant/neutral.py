# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The wrapper arguments the batch-invariant attention classes do not consume, and the values of each that ask for nothing.

Serving engines pass a fixed argument list on every ``run()`` call, with neutral values where a feature is off (unit scales
for an operand that is not quantized). A class takes none of these as an operand, so each has one rule here: a value that
cannot change what the caller gets is accepted and has no effect on the output bits; any other value raises the class's
typed error.

==========================================  =====================================================================================
argument                                    accepted (no effect on the bits); everything else raises
==========================================  =====================================================================================
``q_scale`` / ``o_scale``                   ``None`` or exactly ``1.0`` for every class (the query and the output are bf16 and
                                            unscaled), as a number or a one-element floating-point tensor; a device tensor is
                                            read back once per tensor version (``resolve_scale``), and before any CUDA-graph
                                            capture
``k_scale`` / ``v_scale``                   ``None`` or exactly ``1.0`` (in the same forms) for a bf16 KV cache and for the
                                            token-head-scale classes (their scales are the per-key tensors of
                                            ``kv_cache_sf``); the per-tensor low-bit KV classes take them as operands (any
                                            value their chain admits, ``engine.check_sm_scale`` / ``engine.check_v_scale``),
                                            read through the same path (``scale_value``)
==========================================  =====================================================================================

Every scale, unit-only or an operand of the class, is a Python number or a one-element floating-point tensor and reaches the
host through ``resolve_scale``: a device tensor costs one readback per tensor version, never one per ``run()``.
"""

from __future__ import annotations

import numbers
import weakref
from typing import Dict, Sequence, Tuple

import torch

from ...batch_invariant.support import unsupported

SCALE_NAMES = ("q_scale", "k_scale", "v_scale", "o_scale")
_UNIT_SCALES = "unscaled bf16 operands (q_scale / k_scale / v_scale / o_scale of None or exactly 1.0, as a number or a one-element floating-point tensor)"


_SCALE_FORMS = (
    "a Python number or a one-element floating-point tensor (a device tensor is read once per tensor version; inside a "
    "CUDA-graph capture its value must already be known: pass a Python number or a CPU tensor, or call once before the capture)"
)
# device scale tensors already read: id(tensor) -> (weak reference to the tensor, its version when read, the value)
_DEVICE_SCALES: Dict[int, Tuple[weakref.ref, int, float]] = {}


def _read_device_scale(value: torch.Tensor) -> float:
    """The one device readback of a scale tensor (a synchronisation point)."""
    return float(value.item())


def resolve_scale(class_name: str, name: str, value, limit: str = _UNIT_SCALES):
    """The host number of one scale argument, for every wrapper and for eager and CUDA graph mode alike (``None``, no
    scale, stays ``None``):

    * a real number (``numbers.Real``: a Python int or float, a numpy integer or floating scalar; not a bool) is itself;
    * a one-element floating-point CPU tensor is read on the host at every call;
    * a one-element floating-point device tensor is read back at most once per tensor version: the value is remembered
      for the tensor object (held through a weak reference, so a recycled ``id`` cannot alias it) and its ``_version``,
      and later calls with the same unmodified tensor issue no readback and no synchronisation. An in-place write bumps
      the version and the next call reads again. Storage changed without a version bump (``set_``, writes through a raw
      pointer or another framework's view) is the caller's responsibility. A CUDA-graph capture cannot read the device:
      a tensor whose value is not known when ``run()`` executes inside a capture raises, so it is resolved by a call
      before the capture (the warm-up ``run()``).

    Any other form (several elements, an integer or bool tensor, a bool, a string or another non-number) raises the
    typed error. ``limit``: the calling class's statement of the scales it takes, quoted in those refusals (the unit-scale
    text by default; a class that owns the operand passes its own)."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise unsupported(
                class_name,
                f"{name} as a tensor of {value.numel()} elements",
                limit,
            )
        if not value.dtype.is_floating_point:
            raise unsupported(
                class_name, f"{name} as a tensor of dtype {value.dtype}", limit
            )
        if value.device.type == "cpu":
            return value.item()
        key = id(value)
        known = _DEVICE_SCALES.get(key)
        if known is not None and known[0]() is value and known[1] == value._version:
            return known[2]
        if value.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise unsupported(
                class_name,
                f"{name} as a tensor on {value.device} first seen inside a CUDA-graph capture (its value cannot be read "
                "there)",
                _SCALE_FORMS,
            )
        number = _read_device_scale(value)

        def forget(ref, key=key):
            entry = _DEVICE_SCALES.get(key)
            if entry is not None and entry[0] is ref:
                del _DEVICE_SCALES[key]

        _DEVICE_SCALES[key] = (weakref.ref(value, forget), value._version, number)
        return number
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise unsupported(
            class_name, f"{name}={value!r} ({type(value).__name__})", limit
        )
    return value


def check_unit_scales(
    class_name: str, scales: Sequence, free: Sequence[str] = ()
) -> None:
    """``scales`` in the order of :data:`SCALE_NAMES`: each must be ``None`` or exactly one (compared with ``1.0`` exactly,
    no tolerance: engines pass the exact value; ``resolve_scale`` says which forms of the argument are read). A unit scale
    multiplies nothing, so it is the unscaled class; any other value is a term the class does not have. ``free`` names
    the scales a class takes as operands (the per-tensor ``k_scale`` / ``v_scale`` of a low-bit KV class): those are not
    checked here. The query and the output scale are unit-only for every class."""
    for name, value in zip(SCALE_NAMES, scales, strict=False):
        if value is None or name in free:
            continue
        shown = resolve_scale(class_name, name, value, _UNIT_SCALES)
        if float(shown) != 1.0:
            raise unsupported(class_name, f"{name}={shown!r}", _UNIT_SCALES)


def scale_value(x, name: str, class_name: str) -> float:
    """A per-tensor scale a class takes as an operand (the ``k_scale`` / ``v_scale`` of a per-tensor low-bit KV class): ``None``
    (1.0), or the host number of ``resolve_scale`` -- so a one-element device tensor is read back once per tensor version, not
    at every ``run()``, and its value is on the host when the class folds it into its constants. A form that is not a scale
    raises the class's typed error, quoting the forms an operand takes (not the unit-scale rule)."""
    if x is None:
        return 1.0
    return float(
        resolve_scale(
            class_name,
            name,
            x,
            f"{name} as the class's per-tensor scale: {_SCALE_FORMS}",
        )
    )
