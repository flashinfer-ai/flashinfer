"""Streaming a row from memory four vectors at a time.

A loop that loads one 16-byte vector per iteration and consumes it waits out the load's
latency every iteration: a pass over 64K fp32 elements with 1024 threads measured 7 us that
way (16 dependent loads per thread).  Issuing four loads before consuming any overlaps them
(the filter pass has always done this).  ``row_quads`` yields the loads; the caller consumes
them with ``quad_elements``, which enumerates (bits, index, valid) for the 16 words.

Vectors past the row's last one are clamped to it, so every load is in bounds; their elements
carry an index at or beyond ``length`` and ``valid`` is 0.
"""

import cutlass
import cutlass.cute as cute

from ...device.memory import load_global_readonly_16

__all__ = ["quad_stride", "load_quad", "element_of", "index_of_element"]


def quad_stride(threads: int) -> int:
    """Vectors covered by one iteration of a four-load loop."""
    return 4 * threads


@cute.jit
def _clamped(base, v, last_vector):
    vc = v
    if vc > last_vector:
        vc = last_vector
    return load_global_readonly_16(base + cutlass.Int64(vc) * 16)


@cute.jit
def load_quad(base, v, threads: cutlass.Constexpr, last_vector):
    """Sixteen Uint32 words: vectors ``v``, ``v + threads``, ``v + 2 threads``, ``v + 3 threads``
    (clamped to ``last_vector``), issued together."""
    a0, a1, a2, a3 = _clamped(base, v, last_vector)
    b0, b1, b2, b3 = _clamped(base, v + threads, last_vector)
    c0, c1, c2, c3 = _clamped(base, v + 2 * threads, last_vector)
    d0, d1, d2, d3 = _clamped(base, v + 3 * threads, last_vector)
    return a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3


@cute.jit
def index_of_element(elems, v, threads: cutlass.Constexpr, e: cutlass.Constexpr):
    """Row index of element ``e`` (0 .. 16 * per_word - 1) of the quad loaded at ``v``: vector
    ``e // per_vector`` of the quad, position ``e % per_vector`` in it."""
    return (
        (v + cutlass.Int32((e // elems.per_vector) * threads))
        << cutlass.Int32(elems.log2_per_vector)
    ) + cutlass.Int32(e % elems.per_vector)


@cute.jit
def element_of(
    elems,
    words,
    v,
    threads: cutlass.Constexpr,
    i: cutlass.Constexpr,
    h: cutlass.Constexpr,
):
    """(bits, index) of half ``h`` of word ``i`` (0..15) of the quad loaded at ``v``."""
    bits = elems.bits(words[i], h)
    idx = (
        (v + cutlass.Int32((i // 4) * threads)) << cutlass.Int32(elems.log2_per_vector)
    ) + cutlass.Int32((i % 4) * elems.per_word + h)
    return bits, idx
