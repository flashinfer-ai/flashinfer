# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 device leaves; existing MXFP4/NVFP4 interpretations stay unchanged."""

import cutlass

from .mxfp4_device import PackedF32 as _PackedF32
from .mxfp4_device import ScalarF32 as _ScalarF32


class ScalarF32(_ScalarF32):
    @staticmethod
    def cvt_out(value, fmt):
        if fmt == "e5m2":
            return value.to(cutlass.Float8E5M2)
        return _ScalarF32.cvt_out(value, fmt)


class PackedF32(_PackedF32):
    @staticmethod
    def cvt_out(value, fmt):
        return (ScalarF32.cvt_out(value[0], fmt), ScalarF32.cvt_out(value[1], fmt))
