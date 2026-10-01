"""Element type: how a 32-bit word of the input decomposes into elements, keys and values.

Everything dtype-specific in the top-k phases goes through one ``Elements`` object chosen at
Python level before tracing.  fp32 rows have one element per word; fp16 and bf16 rows have
two (low half first), and their ordered keys live in 16 bits so the radix rounds are two
instead of four.  Values are always handled as fp32 in the phases: thresholds, bin scales and
comparisons are fp32 arithmetic, exact for every 16-bit grid value.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from ...device.keys import (
    f16_bits_of_key,
    f32_bits_of_key,
    key_of_f16_bits,
    key_of_f32_bits,
)

__all__ = ["Elements"]


@dataclass(frozen=True)
class Elements:
    """Compile-time description of the input element type plus its jit accessors.

    Fields: ``dtype`` (cutlass numeric type), ``is_f32``, ``is_bf16``, ``bytes``,
    ``per_word`` (1 or 2), ``per_vector`` (elements in 16 bytes: 4 or 8),
    ``log2_per_vector``, ``key_shifts`` (radix digit positions, most significant first).
    A frozen dataclass on purpose: the DSL drops read-only frozen dataclasses from the values
    it carries through runtime loops, so phases may use ``elems`` inside any loop.
    """

    dtype: type
    is_f32: bool
    is_bf16: bool
    bytes: int
    per_word: int
    per_vector: int
    log2_per_vector: int
    key_shifts: tuple

    @classmethod
    def of(cls, dtype) -> "Elements":
        assert dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16), dtype
        is_f32 = dtype is cutlass.Float32
        per_word = 1 if is_f32 else 2
        return cls(
            dtype=dtype,
            is_f32=is_f32,
            is_bf16=dtype is cutlass.BFloat16,
            bytes=4 if is_f32 else 2,
            per_word=per_word,
            per_vector=4 * per_word,
            log2_per_vector=2 if is_f32 else 3,
            key_shifts=(24, 16, 8, 0) if is_f32 else (8, 0),
        )

    @cute.jit
    def bits(self, word, h: cutlass.Constexpr):
        """Element ``h`` of a loaded Uint32 word as a bit pattern in the low bits."""
        if cutlass.const_expr(self.is_f32):
            return word
        else:
            return (word >> cutlass.Uint32(16 * h)) & cutlass.Uint32(0xFFFF)

    @cute.jit
    def key(self, bits):
        """Ordered key of an element bit pattern (see ``device.keys``)."""
        if cutlass.const_expr(self.is_f32):
            return key_of_f32_bits(bits)
        else:
            return key_of_f16_bits(bits)

    @cute.jit
    def value(self, bits):
        """The element's value as Float32 (exact for every 16-bit grid value)."""
        if cutlass.const_expr(self.is_f32):
            return bits.bitcast(cutlass.Float32)
        else:
            half = bits.to(cutlass.Uint16)
            if cutlass.const_expr(self.is_bf16):
                return half.bitcast(cutlass.BFloat16).to(cutlass.Float32)
            else:
                return half.bitcast(cutlass.Float16).to(cutlass.Float32)

    @cute.jit
    def value_of_key(self, key):
        """Float32 value of an ordered key (the inverse map, then widen)."""
        if cutlass.const_expr(self.is_f32):
            return f32_bits_of_key(key).bitcast(cutlass.Float32)
        else:
            return self.value(f16_bits_of_key(key))

    @cute.jit
    def load_bits(self, row_ptr, idx):
        """One element from a typed row pointer as a Uint32 bit pattern (scalar load)."""
        v = row_ptr[idx]
        if cutlass.const_expr(self.is_f32):
            return v.bitcast(cutlass.Uint32)
        else:
            return v.bitcast(cutlass.Uint16).to(cutlass.Uint32)
