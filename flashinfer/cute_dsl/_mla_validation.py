# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host-side checks for scalar values consumed by the Rubin MTP kernel.

Keep these outside ``attention`` so adapters can import them without the DSL.
"""

import functools
import math
import struct


def _float32(value: float) -> float:
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


_LOG2_E_FP32 = _float32(math.log2(math.e))


@functools.lru_cache(maxsize=128)
def _validate_mtp_scales(softmax_scale=None, output_scale=None) -> None:
    """Validate the effective FP32 values, including the kernel's log2 scale.

    A zero or infinite softmax scale can turn masked logits into NaNs. Checking
    Python finiteness alone misses FP32 narrowing and the subsequent multiply.
    Cache the bounded scalar configurations used by repeated prepared launches.
    """
    if softmax_scale is not None:
        scale = _float32(softmax_scale)
        log2_scale = _float32(scale * _LOG2_E_FP32)
        if not math.isfinite(log2_scale) or log2_scale <= 0:
            raise ValueError(
                "cute-dsl-rubin-mtp requires a positive softmax scale with a "
                "finite, nonzero FP32 log2 value."
            )
    if output_scale is not None and not math.isfinite(_float32(output_scale)):
        raise ValueError("cute-dsl-rubin-mtp requires a finite FP32 output scale.")
