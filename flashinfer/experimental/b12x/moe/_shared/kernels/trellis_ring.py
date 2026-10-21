"""t256 tail-biting ring window geometry shared by the W4A16 and W4A8 kernels.

One ``trellis_t256`` tile packs 256 codes into a ring of ``8*bits`` uint32
words (``256*bits`` bits total). Weight ``j`` of lane ``l`` occupies the
16-bit window ending at bit ``(8*l + j + 257) * bits``, taken modulo the
ring; consecutive weights overlap by ``16 - bits`` bits. The geometry
depends only on the lane id, the weight span, and the bitrate, so callers
hoist it out of their decode loops and feed the resulting word indices to a
64-bit funnel shift.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32


@cute.jit
def trellis256_lane_geom_bits(
    lane: Int32,
    weight_offset: cutlass.Constexpr[int],
    weight_count: cutlass.Constexpr[int],
    bits: cutlass.Constexpr[int],
):
    """Ring geometry for ``weight_count`` weights starting at ``weight_offset``.

    Returns ``(ia, ib, s2, span)``: the ring word indices of the first and
    last 32-bit words covering the lane's windows, the funnel shift that
    aligns the merged 64-bit read on the final window, and the word span
    ``i2 - i0`` before ring wrap-around.
    """

    bits_i32 = Int32(int(bits))
    ring_u32 = Int32(8 * int(bits))
    t_offset = Int32(8) * lane + Int32(weight_offset)
    b1 = (t_offset + Int32(257)) * bits_i32
    b0 = b1 - Int32(16)
    b2 = b1 + Int32((int(weight_count) - 1) * int(bits))
    i0 = b0 >> Int32(5)
    i2 = (b2 - Int32(1)) >> Int32(5)
    ia = i0 - ring_u32 * (i0 >= ring_u32).to(Int32)
    ib = i2 - ring_u32 * (i2 >= ring_u32).to(Int32)
    s2 = (i2 + Int32(1)) * Int32(32) - b2
    return ia, ib, s2, i2 - i0
