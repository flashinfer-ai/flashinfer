"""Bit-equality of the direct-table decode against the value-table decode.

The rate-indexed direct table precomposes the lut_e4m3 permutation with the
value table (``direct[window] = values[rank(window) >> 4]``), so the
two decode intrinsics must produce identical bytes for every window. The
probe drives both intrinsics on the same random ring windows and compares
the packed results.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    packed_decode_lut_e4m3_to_e4m3x8,
    packed_decode_lut_e4m3_direct_to_e4m3x8,
)
from b12x._lib.quant.lut_e4m3 import (
    lut_e4m3_direct_table_cpu,
    lut_e4m3_value_table_cpu,
)
from b12x._lib.utils import current_cuda_stream
from b12x.testing.reference.helpers import require_b12x

require_b12x(allow_module_level=True)


class _DecodeProbe:
    def __init__(self, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(
        self,
        wins: cute.Tensor,
        value_table: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(wins, value_table, direct, out).launch(
            grid=(1, 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        wins: cute.Tensor,
        value_table: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = Int32(tidx)
        value_table_addr = value_table.iterator.toint()
        dir_addr = direct.iterator.toint()
        wa = Uint32(wins[2 * lane])
        wb = Uint32(wins[2 * lane + 1])
        lo_t, hi_t = packed_decode_lut_e4m3_to_e4m3x8(
            wa, wb, value_table_addr, self.bits, value_table_in_shared=False
        )
        lo_d, hi_d = packed_decode_lut_e4m3_direct_to_e4m3x8(
            wa, wb, dir_addr, self.bits, rate_indexed=True
        )
        out[4 * lane] = Int32(lo_t)
        out[4 * lane + 1] = Int32(hi_t)
        out[4 * lane + 2] = Int32(lo_d)
        out[4 * lane + 3] = Int32(hi_d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_direct_table_decode_bit_equals_value_table(bits: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260812 + bits)
    wins = torch.zeros(64, dtype=torch.int32, device=device)
    value_table = lut_e4m3_value_table_cpu().to(device)
    direct = lut_e4m3_direct_table_cpu().to(device)
    out = torch.zeros(128, dtype=torch.int32, device=device)

    def args():
        return (
            from_dlpack(wins, assumed_align=16),
            from_dlpack(value_table, assumed_align=16),
            from_dlpack(direct, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_DecodeProbe(bits), *args())
    mismatched = 0
    for _ in range(512):
        wins.copy_(
            torch.randint(
                -(2**31), 2**31 - 1, (64,), dtype=torch.int32, device=device
            )
        )
        compiled(*args())
        torch.cuda.synchronize()
        o = out.view(32, 4)
        mismatched += int((o[:, 0] != o[:, 2]).sum())
        mismatched += int((o[:, 1] != o[:, 3]).sum())
    assert mismatched == 0, mismatched
