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
                                            unscaled), as a host number or a one-element CPU tensor (a device tensor would
                                            cost a readback per ``run()``: refused)
``k_scale`` / ``v_scale``                   ``None`` or exactly ``1.0`` (host values, as above) for a bf16 KV cache and for the
                                            token-head-scale classes (their scales are the per-key tensors of
                                            ``kv_cache_sf``); the per-tensor low-bit KV
                                            classes take them as operands (any value their chain admits,
                                            ``engine.check_sm_scale`` / ``engine.check_v_scale``)
==========================================  =====================================================================================

A scale a class takes as an operand is a Python number or a one-element floating-point tensor (read on the host: a
synchronisation for a device tensor, so a captured CUDA graph should pass numbers); a unit-only scale is a host value.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ...batch_invariant.support import unsupported

SCALE_NAMES = ("q_scale", "k_scale", "v_scale", "o_scale")
_UNIT_SCALES = "unscaled bf16 operands (q_scale / k_scale / v_scale / o_scale of None or exactly 1.0, as host numbers)"


def scale_value(x, name: str, class_name: str) -> float:
    """A per-tensor scale as FlashInfer's wrappers take it: ``None`` (1.0), a Python number, or a one-element floating-point
    tensor (read on the host); anything else raises the class's typed error."""
    if x is None:
        return 1.0
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() != 1 or not x.dtype.is_floating_point:
            raise unsupported(
                class_name,
                f"a {name} tensor of shape {tuple(x.shape)} and dtype {x.dtype}",
                f"{name} as a Python number or a 1-element floating-point tensor",
            )
        return float(x.item())
    raise unsupported(
        class_name,
        f"{name} of type {type(x).__name__}",
        f"{name} as a Python number or a 1-element tensor",
    )


def check_unit_scales(
    class_name: str, scales: Sequence, free: Sequence[str] = ()
) -> None:
    """``scales`` in the order of :data:`SCALE_NAMES`: each must be ``None`` or exactly one (compared with ``1.0`` exactly,
    no tolerance: engines pass the exact value), as a host number or a one-element CPU tensor -- a device tensor is refused
    whatever its value, since reading it would synchronise every ``run()``. A unit scale multiplies nothing, so it is the
    unscaled class; any other value is a term the class does not have. ``free`` names the scales a class takes as operands
    (the per-tensor ``k_scale`` / ``v_scale`` of a low-bit KV class): those are not checked here. The query and the output
    scale are unit-only for every class."""
    limit = _UNIT_SCALES
    if free:
        fixed = [n for n in SCALE_NAMES if n not in free]
        limit = (
            f"an unscaled bf16 query and output ({' / '.join(fixed)} of None or exactly 1.0, as host numbers; "
            f"{' / '.join(free)} are the class's per-tensor scales)"
        )
    for name, value in zip(SCALE_NAMES, scales, strict=False):
        if value is None or name in free:
            continue
        shown = value
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise unsupported(
                    class_name,
                    f"{name} as a tensor on {value.device} (reading it would stall every run())",
                    limit,
                )
            if value.numel() != 1:
                raise unsupported(
                    class_name,
                    f"{name} as a tensor of {value.numel()} elements",
                    limit,
                )
            shown = value.item()
        try:
            unit = float(shown) == 1.0
        except (TypeError, ValueError):
            unit = False
        if not unit:
            raise unsupported(class_name, f"{name}={shown!r}", limit)
