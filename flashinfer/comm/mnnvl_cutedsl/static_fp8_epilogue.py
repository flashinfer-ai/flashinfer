# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference static E4M3 epilogue shared by the LL, BT, and HT protocols.

Source-only prototype: not GPU-compiled or numerically validated. The packed
BF16 argument preserves the existing norm rounding point before quantization.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float8E4M3FN, Float32, Int64, Uint32

from ...cute_dsl.fp4_common import cvt_f32_to_e4m3
from .cute_dsl_primitives import (
    packed_u32x4_to_bf16x8,
    store_global_u32x2,
    store_global_u32x4,
)


@cute.jit
def store_rmsnorm_output(
    packed_bf16,
    output: cute.Tensor,
    element_offset: Int64,
    output_scale: cute.Tensor | None,
    norm_output_bf16: cute.Tensor | None,
) -> None:
    """Store eight normalized values, optionally quantizing with q = norm / s.

    ``s`` is a positive finite device FP32 scalar shared by all tokens/ranks.
    It is an input dequantization scale, not a scale estimated by this kernel.
    The output element type and optional companion are compile-time choices.
    """
    if cutlass.const_expr(output.element_type == Float8E4M3FN):
        # Read on the device, after the caller's existing dependency waits.
        # Keeping the scalar as a tensor permits in-place updates during replay.
        inverse_scale = Float32(1.0) / output_scale[0]
        values = packed_u32x4_to_bf16x8(packed_bf16).to(Float32)
        packed_fp8 = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
        for word in cutlass.range_constexpr(2):
            bits = Uint32(0)
            for byte in cutlass.range_constexpr(4):
                # Native round-to-nearest, finite-saturating E4M3 conversion.
                # A follow-up can pair conversions; this version favors clarity.
                encoded = cvt_f32_to_e4m3(values[word * 4 + byte] * inverse_scale)
                bits = bits | ((encoded & Uint32(0xFF)) << (byte * 8))
            packed_fp8[word] = bits
        store_global_u32x2(
            Int64((output.iterator + element_offset).toint()), packed_fp8.load()
        )
        if cutlass.const_expr(norm_output_bf16 is not None):
            store_global_u32x4(
                Int64((norm_output_bf16.iterator + element_offset).toint()), packed_bf16
            )
    else:
        # Existing BF16 path: no scale read, conversion, or extra output store.
        store_global_u32x4(
            Int64((output.iterator + element_offset).toint()), packed_bf16
        )
