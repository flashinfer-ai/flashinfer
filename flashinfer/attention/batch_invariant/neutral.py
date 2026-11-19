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
                                            unscaled)
``k_scale`` / ``v_scale``                   ``None`` or exactly ``1.0`` for a bf16 KV cache; the low-bit KV classes take them as
                                            operands (any value their chain admits, ``engine.check_sm_scale`` /
                                            ``engine.check_v_scale``)
==========================================  =====================================================================================

A scale is a Python number or a one-element floating-point tensor (read on the host: a synchronisation for a device tensor,
so a captured CUDA graph should pass numbers).
"""

from __future__ import annotations

from typing import Sequence

import torch

from ...batch_invariant.support import unsupported

SCALE_NAMES = ("q_scale", "k_scale", "v_scale", "o_scale")


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
    no tolerance: engines pass the exact value). A unit scale multiplies nothing, so it is the unscaled operand; any other
    value is a term the class does not have. ``free`` names the scales a class takes as operands (the per-tensor
    ``k_scale`` / ``v_scale`` of a low-bit KV class): those are not checked here. The query and the output scale are
    unit-only for every class."""
    fixed = [n for n in SCALE_NAMES if n not in free]
    limit = (
        f"unscaled bf16 operands ({' / '.join(fixed)} of None or exactly 1.0)"
        if not free
        else f"an unscaled bf16 query and output ({' / '.join(fixed)} of None or exactly 1.0; "
        f"{' / '.join(free)} are the class's per-tensor scales)"
    )
    for name, value in zip(SCALE_NAMES, scales, strict=False):
        if value is None or name in free:
            continue
        shown = scale_value(value, name, class_name)
        if shown != 1.0:
            raise unsupported(class_name, f"{name}={shown!r}", limit)
