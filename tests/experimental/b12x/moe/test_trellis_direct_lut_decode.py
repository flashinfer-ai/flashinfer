"""Bit-equality of the direct-table decode against the value-table decode.

The rate-indexed direct table precomposes the lut_e4m3 permutation with the
value table (``direct[window] = values[rank(window) >> 4]``), so the
two decode intrinsics must produce identical bytes for every window. The
probe drives both intrinsics on the same random ring windows and compares
the packed results.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    packed_decode_lut_e4m3_to_e4m3x8,
    packed_decode_lut_e4m3_direct_to_e4m3x8,
    shared_ptr_to_u32,
)
from b12x._lib.quant.lut_e4m3 import (
    lut_e4m3_direct_table_cpu,
    lut_e4m3_value_table_cpu,
)
from b12x._lib.utils import current_cuda_stream
from b12x.testing.reference.helpers import require_b12x

require_b12x(allow_module_level=True)


class _DecodeProbe:
    def __init__(self, bits: int, shared: bool = False):
        self.bits = int(bits)
        self.shared = shared

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
            grid=(cute.size(wins) // 64, 1, 1), block=[32, 1, 1], stream=stream
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
        bidx, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) + Int32(bidx) * 32
        value_table_addr = value_table.iterator.toint()
        dir_addr = direct.iterator.toint()
        if cutlass.const_expr(self.shared):
            table = cutlass.utils.SmemAllocator().allocate_tensor(
                cutlass.Uint8, cute.make_layout(65536), byte_alignment=16
            )
            for offset in range(Int32(tidx), 65536, 32):
                table[offset] = direct[(self.bits - 2) * 65536 + offset]
            cute.arch.sync_threads()
            dir_addr = shared_ptr_to_u32(table.iterator)
        wa = Uint32(wins[2 * lane])
        wb = Uint32(wins[2 * lane + 1])
        lo_t, hi_t = packed_decode_lut_e4m3_to_e4m3x8(
            wa, wb, value_table_addr, self.bits, value_table_in_shared=False
        )
        lo_d, hi_d = packed_decode_lut_e4m3_direct_to_e4m3x8(
            wa,
            wb,
            dir_addr,
            self.bits,
            rate_indexed=not self.shared,
            in_shared=self.shared,
        )
        out[4 * lane] = Int32(lo_t)
        out[4 * lane + 1] = Int32(hi_t)
        out[4 * lane + 2] = Int32(lo_d)
        out[4 * lane + 3] = Int32(hi_d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("shared", [False, True])
def test_direct_table_decode_bit_equals_value_table(bits: int, shared: bool) -> None:
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

    compiled = b12x_compile(_DecodeProbe(bits, shared), *args())
    mismatched = 0
    for _ in range(512):
        wins.copy_(
            torch.randint(-(2**31), 2**31 - 1, (64,), dtype=torch.int32, device=device)
        )
        compiled(*args())
        torch.cuda.synchronize()
        o = out.view(32, 4)
        mismatched += int((o[:, 0] != o[:, 2]).sum())
        mismatched += int((o[:, 1] != o[:, 3]).sum())
    assert mismatched == 0, mismatched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_shared_direct_lut_covers_all_codewords(bits: int) -> None:
    # Both 16-bit windows visit every codeword; random high bits also exercise
    # the shifted windows that cross the low-16-bit boundary.
    device = torch.device("cuda")
    torch.manual_seed(731 + bits)
    states = torch.arange(65536, device=device, dtype=torch.int32)
    upper = torch.randint(0, 65536, (65536, 2), device=device, dtype=torch.int32)
    wins = ((upper << 16) | states[:, None]).flatten()
    value_table = lut_e4m3_value_table_cpu().to(device)
    direct = lut_e4m3_direct_table_cpu().to(device)
    out = torch.empty(65536 * 4, device=device, dtype=torch.int32)
    args = tuple(
        from_dlpack(tensor, assumed_align=16)
        for tensor in (wins, value_table, direct, out)
    ) + (current_cuda_stream(),)
    compiled = b12x_compile(_DecodeProbe(bits, shared=True), *args)
    compiled(*args)
    torch.cuda.synchronize()
    result = out.view(-1, 4)
    torch.testing.assert_close(result[:, :2], result[:, 2:], rtol=0, atol=0)
