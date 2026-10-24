# Copyright (c) 2025 by the b12x authors.
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

"""Numeric format limits."""

from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32


# =============================================================================
# Constants
# =============================================================================

FLOAT4_E2M1_MAX = 6.0  # Maximum value representable in FP4 E2M1

FLOAT8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3

SF_VEC_SIZE = 16  # Elements per scale factor block

COPY_BITS = 128  # 128-bit vectorized loads

_FP4_MAG_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# UE8M0 0xFF is the encoding's NaN.  No MX block scale may be NaN, and a
# clamp would silently change every dequantized weight in the block, so the
# preparation path rejects it instead of carrying it into the kernel.
E8M0_NAN_BYTE = 0xFF

MX_SF_VEC_SIZE = 32  # Elements per UE8M0 scale block (MXFP8 w4a8 activations)

_INV_FLOAT8_E4M3_MAX = 1.0 / FLOAT8_E4M3_MAX

# ``mma_m16n8k16_f32_bf16`` was proven during the SM120 sparse-MLA port to be
# bit-identical (same signature, same emitted
# ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``, same constraint
# string and operand grouping) to the pre-existing ``bf16_mma_m16n8k16_f32``
# above. Expose it under the port's name as a thin alias so the sparse-MLA
# kernels can import it without duplicating the inline asm.
mma_m16n8k16_f32_bf16 = bf16_mma_m16n8k16_f32

# ===== TRELLIS-3.0 (QTIP/EXL3 3INST cb=1) dequant =====
_TRELLIS_MCG = 0xCBAC1FED  # codebook.cuh:39  MCG multiplier (cb==1)

_TRELLIS_MASK = 0x8FFF8FFF  # codebook.cuh:41  lop3 operand b

_TRELLIS_OR = 0x3B603B60  # codebook.cuh:41  lop3 operand c; immLut 0x6a == (a & b) ^ c

_TRELLIS_MUL1 = 0x83DCD12D  # codebook.cuh:107  MUL1 multiplier (cb==2/3)
