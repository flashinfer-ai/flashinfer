"""Fragment-mapping probe for the w4a8 (FP4 weights x E4M3 activations) MMA path.

Pins, once and bit-tight, the hardest w4a8 sub-problem before touching
dynamic.py: how packed-FP4 B words expand into the two m16n8k32 mxf8f6f4
MMAs that replace one m16n8k64 mxf4nvf4 MMA, which k-permutation the A-side
quantizer must write, and the empirical (lane, byte) -> (row/col, k-block)
mapping of the scale_vec::1X SFA/SFB operands.

K-permutation claim under test: lane t (c = t%4, g = t/4) holding the packed
B word for original k in [K0+8c, K0+8c+8) expands via e2m1x8_to_e4m3x8 into
exactly the (b0, b1) registers of the k32 atom for block K0, provided the
A operand for atom register r is the plain contiguous 4-byte load
A[g (+8 for odd r), K0 + 8c + 4*(r//2)].  Dot products commute, so this
lane-local relabeling is exact as long as A and B agree and original k32
blocks stay within one atom (they do), keeping SFA/SFB block-aligned.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import (
    _fp4_encode_nibbles,
    broadcast_f32_to_half2,
    e2m1x8_mul_residual_to_e4m3x8,
    e2m1x8_to_e4m3x8,
    fp8_e4m3_to_f32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    packed_decode_trellis_mul1_e4m3_to_e4m3x8,
)

from b12x.testing.reference.helpers import require_b12x

_M, _N, _K = 16, 8, 64  # one m16n8 tile, two k32 blocks


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


class _W4A8ProbeKernel:
    num_threads = 32

    def __init__(self, *, use_residual: bool):
        self.use_residual = bool(use_residual)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,        # [16, 16] u32 = 16x64 e4m3 bytes, row-major
        mB: cute.Tensor,        # [8, 8] u32 = 8x32 packed fp4 bytes (k-major)
        mResidual: cute.Tensor, # [8] u32 = 8x4 e4m3 residual bytes (n, k16)
        mSfaTable: cute.Tensor, # [32] u32 per-lane SFA register
        mSfbTable: cute.Tensor, # [32] u32 per-lane SFB register
        mOut: cute.Tensor,      # [16, 8] f32
        stream: cuda.CUstream,
    ):
        self.kernel(mA, mB, mResidual, mSfaTable, mSfbTable, mOut).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mResidual: cute.Tensor,
        mSfaTable: cute.Tensor,
        mSfbTable: cute.Tensor,
        mOut: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        c = lane % Int32(4)
        g = lane // Int32(4)
        sfa = Uint32(mSfaTable[lane])
        sfb = Uint32(mSfbTable[lane])

        d0 = Float32(0.0)
        d1 = Float32(0.0)
        d2 = Float32(0.0)
        d3 = Float32(0.0)
        for kb in cutlass.range_constexpr(2):
            # B: the packed word a k64-style fragment hands this lane for
            # this k32 block (original k in [32*kb + 8c, +8)).
            w = Uint32(mB[g, Int32(4 * kb) + c])
            b0 = Uint32(0)
            b1 = Uint32(0)
            if const_expr(self.use_residual):
                res_word = Uint32(mResidual[g])
                res_byte = (res_word >> ((Uint32(2 * kb) + (Uint32(c) >> 1)) * Uint32(8))) & Uint32(0xFF)
                res_h2 = broadcast_f32_to_half2(fp8_e4m3_to_f32(res_byte))
                b0, b1 = e2m1x8_mul_residual_to_e4m3x8(w, res_h2)
            else:
                b0, b1 = e2m1x8_to_e4m3x8(w)

            # A: plain contiguous 4-byte loads under the k-permutation.
            a0 = Uint32(mA[g, Int32(8 * kb) + Int32(2) * c])
            a1 = Uint32(mA[g + Int32(8), Int32(8 * kb) + Int32(2) * c])
            a2 = Uint32(mA[g, Int32(8 * kb) + Int32(2) * c + Int32(1)])
            a3 = Uint32(mA[g + Int32(8), Int32(8 * kb) + Int32(2) * c + Int32(1)])

            sfa_kb = (sfa >> (Uint32(kb) * Uint32(16))) & Uint32(0xFFFF)
            sfb_kb = (sfb >> (Uint32(kb) * Uint32(16))) & Uint32(0xFFFF)
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                d0, d1, d2, d3,
                a0, a1, a2, a3,
                b0, b1,
                sfa_kb,
                sfb_kb,
            )

        col = Int32(2) * c
        mOut[g, col] = d0
        mOut[g, col + Int32(1)] = d1
        mOut[g + Int32(8), col] = d2
        mOut[g + Int32(8), col + Int32(1)] = d3


class _TrellisW4A8ProbeKernel:
    """Direct MUL1-E4M3 decode into the production E4M3 MMA fragment."""

    num_threads = 32

    def __init__(self, *, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mWinA: cute.Tensor,
        mWinB: cute.Tensor,
        mSfaTable: cute.Tensor,
        mSfbTable: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mA, mWinA, mWinB, mSfaTable, mSfbTable, mOut).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mWinA: cute.Tensor,
        mWinB: cute.Tensor,
        mSfaTable: cute.Tensor,
        mSfbTable: cute.Tensor,
        mOut: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        c = lane % Int32(4)
        g = lane // Int32(4)
        sfa = Uint32(mSfaTable[lane])
        sfb = Uint32(mSfbTable[lane])

        d0 = Float32(0.0)
        d1 = Float32(0.0)
        d2 = Float32(0.0)
        d3 = Float32(0.0)
        for kb in cutlass.range_constexpr(2):
            group = Int32(4 * kb) + c
            b0, b1 = packed_decode_trellis_mul1_e4m3_to_e4m3x8(
                Uint32(mWinA[g, group]),
                Uint32(mWinB[g, group]),
                self.bits,
            )

            a0 = Uint32(mA[g, Int32(8 * kb) + Int32(2) * c])
            a1 = Uint32(mA[g + Int32(8), Int32(8 * kb) + Int32(2) * c])
            a2 = Uint32(mA[g, Int32(8 * kb) + Int32(2) * c + Int32(1)])
            a3 = Uint32(
                mA[g + Int32(8), Int32(8 * kb) + Int32(2) * c + Int32(1)]
            )

            sfa_kb = (sfa >> (Uint32(kb) * Uint32(16))) & Uint32(0xFFFF)
            sfb_kb = (sfb >> (Uint32(kb) * Uint32(16))) & Uint32(0xFFFF)
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                d0,
                d1,
                d2,
                d3,
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
                sfa_kb,
                sfb_kb,
            )

        col = Int32(2) * c
        mOut[g, col] = d0
        mOut[g, col + Int32(1)] = d1
        mOut[g + Int32(8), col] = d2
        mOut[g + Int32(8), col + Int32(1)] = d3


class _NativeTrellisW4A8ProbeKernel:
    """Decode two native EXL 16x16 tiles straight into one k32 MMA.

    Native EXL lanes each reconstruct four K values for one output in each
    N8 half.  A fixed within-K32 A permutation lets an MMA lane retain its
    own four-byte word and fetch the other word with one xor-2 warp shuffle.
    No reconstructed weight is widened or written to memory.
    """

    num_threads = 32

    def __init__(self, *, bits: int, n_high: bool):
        self.bits = int(bits)
        self.n_high = bool(n_high)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mPacked: cute.Tensor,
        mSfaTable: cute.Tensor,
        mSfbTable: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mA, mPacked, mSfaTable, mSfbTable, mOut).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mPacked: cute.Tensor,
        mSfaTable: cute.Tensor,
        mSfbTable: cute.Tensor,
        mOut: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        c = lane % Int32(4)
        g = lane // Int32(4)
        bits = Int32(self.bits)
        ring_u32 = Int32(8 * self.bits)
        t_offset = Int32(8) * lane
        b1 = (t_offset + Int32(257)) * bits
        b0 = b1 - Int32(16)
        b2 = b1 + Int32(7 * self.bits)
        i0 = b0 >> Int32(5)
        i2 = (b2 - Int32(1)) >> Int32(5)
        ia = i0 - ring_u32 * (i0 >= ring_u32).to(Int32)
        ib = i2 - ring_u32 * (i2 >= ring_u32).to(Int32)
        s2 = (i2 + Int32(1)) * Int32(32) - b2

        e0 = Uint32(0)
        e1 = Uint32(0)
        for kt in cutlass.range_constexpr(2):
            a = Uint32(mPacked[kt, ia])
            b = Uint32(mPacked[kt, ib])
            merged = (Int64(a) << Int64(32)) | Int64(b)
            win_a = Uint32(merged >> Int64(s2))
            win_b = Uint32(
                merged >> Int64(s2 + Int32(4 * self.bits))
            )
            lo, hi = packed_decode_trellis_mul1_e4m3_to_e4m3x8(
                win_a, win_b, self.bits
            )
            chosen = hi if const_expr(self.n_high) else lo
            if const_expr(kt == 0):
                e0 = chosen
            else:
                e1 = chosen

        # c=0/1 consumes K16 tile 0; c=2/3 consumes tile 1.  The peer
        # register deliberately contains the opposite selection so xor-2
        # sends tile-0 words 2/3 to lanes 0/1 and tile-1 words 0/1 to 2/3.
        own = e0
        send = e1
        if c >= Int32(2):
            own = e1
            send = e0
        peer = Uint32(cute.arch.shuffle_sync_bfly(send, offset=2))

        a0 = Uint32(mA[g, Int32(2) * c])
        a1 = Uint32(mA[g + Int32(8), Int32(2) * c])
        a2 = Uint32(mA[g, Int32(2) * c + Int32(1)])
        a3 = Uint32(mA[g + Int32(8), Int32(2) * c + Int32(1)])
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            a0,
            a1,
            a2,
            a3,
            own,
            peer,
            Uint32(mSfaTable[lane]) & Uint32(0xFFFF),
            Uint32(mSfbTable[lane]) & Uint32(0xFFFF),
        )

        col = Int32(2) * c
        mOut[g, col] = d0
        mOut[g, col + Int32(1)] = d1
        mOut[g + Int32(8), col] = d2
        mOut[g + Int32(8), col + Int32(1)] = d3


def _run_probe(
    a_bytes: torch.Tensor,
    b_packed: torch.Tensor,
    residual_bytes: torch.Tensor | None,
    sfa_table: torch.Tensor,
    sfb_table: torch.Tensor,
) -> torch.Tensor:
    device = a_bytes.device
    out = torch.zeros(_M, _N, device=device, dtype=torch.float32)
    res = (
        residual_bytes
        if residual_bytes is not None
        else torch.zeros(_N, 4, dtype=torch.uint8, device=device)
    )
    args = (
        _to_cute_tensor(a_bytes.view(torch.int32), cutlass.Uint32),
        _to_cute_tensor(b_packed.view(torch.int32), cutlass.Uint32),
        _to_cute_tensor(res.view(torch.int32).view(-1), cutlass.Uint32),
        _to_cute_tensor(sfa_table, cutlass.Uint32),
        _to_cute_tensor(sfb_table, cutlass.Uint32),
        _to_cute_tensor(out, cutlass.Float32),
        stream := cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    kernel = _W4A8ProbeKernel(use_residual=residual_bytes is not None)
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return out


def _run_trellis_probe(
    a_bytes: torch.Tensor,
    win_a: torch.Tensor,
    win_b: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    device = a_bytes.device
    out = torch.zeros(_M, _N, device=device, dtype=torch.float32)
    args = (
        _to_cute_tensor(a_bytes.view(torch.int32), cutlass.Uint32),
        _to_cute_tensor(win_a, cutlass.Uint32),
        _to_cute_tensor(win_b, cutlass.Uint32),
        _to_cute_tensor(_unit_tables(device), cutlass.Uint32),
        _to_cute_tensor(_unit_tables(device), cutlass.Uint32),
        _to_cute_tensor(out, cutlass.Float32),
        stream := cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    kernel = _TrellisW4A8ProbeKernel(bits=bits)
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return out


def _run_native_trellis_probe(
    a_bytes_permuted: torch.Tensor,
    packed_tiles: torch.Tensor,
    bits: int,
    *,
    n_high: bool,
) -> torch.Tensor:
    device = a_bytes_permuted.device
    out = torch.zeros(16, 8, device=device, dtype=torch.float32)
    args = (
        _to_cute_tensor(a_bytes_permuted.view(torch.int32), cutlass.Uint32),
        _to_cute_tensor(packed_tiles.view(torch.int32), cutlass.Uint32),
        _to_cute_tensor(_unit_tables(device), cutlass.Uint32),
        _to_cute_tensor(_unit_tables(device), cutlass.Uint32),
        _to_cute_tensor(out, cutlass.Float32),
        stream := cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    kernel = _NativeTrellisW4A8ProbeKernel(bits=bits, n_high=n_high)
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return out


def _decode_mul1_e4m3_reference(
    win_a: torch.Tensor, win_b: torch.Tensor, bits: int
) -> torch.Tensor:
    """Reference eight E4M3 values for each pair of trellis funnel words."""
    a = win_a.to(torch.int64) & 0xFFFFFFFF
    b = win_b.to(torch.int64) & 0xFFFFFFFF
    shifts = (3 * bits, 2 * bits, bits, 0)
    states = torch.stack(
        [
            *((b >> shift) & 0xFFFF for shift in shifts),
            *((a >> shift) & 0xFFFF for shift in shifts),
        ],
        dim=-1,
    )
    product = (states * 0x83DCD12D) & 0xFFFFFFFF
    byte_sum = (
        (product & 0xFF)
        + ((product >> 8) & 0xFF)
        + ((product >> 16) & 0xFF)
        + ((product >> 24) & 0xFF)
    )
    accumulator = (byte_sum + 0x6400).to(torch.int16).view(torch.float16)
    inv = torch.tensor(0x1EEE, dtype=torch.int16, device=win_a.device).view(
        torch.float16
    )
    bias = torch.tensor(
        0xC931 - 0x10000, dtype=torch.int16, device=win_a.device
    ).view(torch.float16)
    reconstructed = (accumulator.double() * inv.double() + bias.double()).to(
        torch.float16
    )
    return reconstructed.to(torch.float8_e4m3fn).to(torch.float32)


def _decode_mul1_e4m3_states_reference(states: torch.Tensor) -> torch.Tensor:
    product = (states.to(torch.int64) * 0x83DCD12D) & 0xFFFFFFFF
    byte_sum = (
        (product & 0xFF)
        + ((product >> 8) & 0xFF)
        + ((product >> 16) & 0xFF)
        + ((product >> 24) & 0xFF)
    )
    accumulator = (byte_sum + 0x6400).to(torch.int16).view(torch.float16)
    inv = torch.tensor(0x1EEE, dtype=torch.int16, device=states.device).view(
        torch.float16
    )
    bias = torch.tensor(
        0xC931 - 0x10000, dtype=torch.int16, device=states.device
    ).view(torch.float16)
    reconstructed = (accumulator.double() * inv.double() + bias.double()).to(
        torch.float16
    )
    return reconstructed.to(torch.float8_e4m3fn).to(torch.float32)


def _native_trellis_tile(edges: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack 256 edge symbols in EXL's native SWAP16 tile order."""
    values = edges.to(torch.int64) & ((1 << bits) - 1)
    spans = values.reshape(16, 16)
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=edges.device)
    bitstream = ((spans[..., None] >> symbol_shifts) & 1).reshape(16, bits * 16)
    word_shifts = torch.arange(15, -1, -1, device=edges.device)
    words = (bitstream.reshape(16, bits, 16) << word_shifts).sum(dim=-1)
    flat = words.reshape(16 * bits)
    return flat.reshape(-1, 2).flip(-1).reshape(-1).to(torch.int16).contiguous()


def _cyclic_trellis_states(edges: torch.Tensor, bits: int) -> torch.Tensor:
    states = torch.zeros_like(edges, dtype=torch.int64)
    for lag in range((16 + bits - 1) // bits):
        states |= torch.roll(edges.to(torch.int64), shifts=lag) << (lag * bits)
    return (states & 0xFFFF).to(torch.int32)


# MMA A bytes matching the one-shuffle native-B assignment in
# _NativeTrellisW4A8ProbeKernel.  This is purely a permutation within K32, so
# the activation's UE8M0 scale group remains unchanged.
_NATIVE_MMA_K32_PERM = (
    0, 1, 8, 9, 4, 5, 12, 13,
    2, 3, 10, 11, 6, 7, 14, 15,
    20, 21, 28, 29, 16, 17, 24, 25,
    22, 23, 30, 31, 18, 19, 26, 27,
)


def _native_tiles_reference(
    edge_tiles: torch.Tensor, bits: int
) -> torch.Tensor:
    """Reconstruct two native K16xN16 tiles as logical B[N=16,K=32]."""
    b = torch.empty(16, 32, dtype=torch.float32, device=edge_tiles.device)
    for kt in range(2):
        states = _cyclic_trellis_states(edge_tiles[kt], bits).reshape(32, 8)
        values = _decode_mul1_e4m3_states_reference(states)
        for lane in range(32):
            r = lane % 4
            n0 = 2 * (lane // 8) + ((lane >> 2) & 1)
            ks = (2 * r, 2 * r + 1, 2 * r + 8, 2 * r + 9)
            for j, k in enumerate(ks):
                b[n0, 16 * kt + k] = values[lane, j]
                b[n0 + 8, 16 * kt + k] = values[lane, j + 4]
    return b


def _unit_tables(device: torch.device) -> torch.Tensor:
    return torch.full((32,), 0x7F7F7F7F, dtype=torch.int64, device=device).to(torch.int32)


def _pack_b(values: torch.Tensor) -> torch.Tensor:
    """Pack FP4-grid values [N, K] into k-major nibbles [N, K/2] uint8."""
    nib = _fp4_encode_nibbles(values)
    pair = nib.view(values.shape[0], values.shape[1] // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).contiguous()


def _e4m3_bytes(values: torch.Tensor) -> torch.Tensor:
    return values.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_probe_data_path_exact_unit_scales() -> None:
    """Expansion + k-permutation + double-k32 MMA, bit-exact on dyadic data."""
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(0)
    # Dyadic values: every product and partial sum is exact in f32, so the
    # accumulation order cannot matter and equality is exact.
    a_vals = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0, -0.5, 4.0, -2.0], device=device)
    a = a_vals[torch.randint(0, 8, (_M, _K), device=device)]
    fp4_grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
    )
    b = fp4_grid[torch.randint(0, 15, (_N, _K), device=device)]

    out = _run_probe(
        _e4m3_bytes(a), _pack_b(b), None, _unit_tables(device), _unit_tables(device)
    )
    ref = a @ b.T
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_mul1_e4m3_trellis_decode_feeds_w4a8_mma_exactly(bits: int) -> None:
    """Trellis windows decode directly into the expected E4M3 B fragment."""
    require_b12x()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0xE4A8 + bits)
    a_grid = torch.tensor(
        [0.0, 0.25, 0.5, 1.0, 2.0, -0.25, -0.5, -1.0, -2.0],
        device=device,
    )
    a = a_grid[
        torch.randint(0, a_grid.numel(), (_M, _K), device=device, generator=generator)
    ]
    win_a = torch.randint(
        -(1 << 31),
        1 << 31,
        (_N, _K // 8),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    win_b = torch.randint(
        -(1 << 31),
        1 << 31,
        (_N, _K // 8),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )

    out = _run_trellis_probe(_e4m3_bytes(a), win_a, win_b, bits)
    b = _decode_mul1_e4m3_reference(win_a, win_b, bits).reshape(_N, _K)
    ref = a @ b.T
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("n_high", [False, True])
def test_native_mul1_e4m3_tiles_feed_w4a8_mma_exactly(
    bits: int, n_high: bool
) -> None:
    """Native tile decode + one-shuffle B mapping is an exact E4M3 MMA."""
    require_b12x()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(
        0x16A832 + 17 * bits + int(n_high)
    )
    edge_tiles = torch.randint(
        0,
        1 << bits,
        (2, 256),
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    packed_tiles = torch.stack(
        [_native_trellis_tile(edge_tiles[kt], bits) for kt in range(2)]
    )
    b16 = _native_tiles_reference(edge_tiles, bits)
    b = b16[8:] if n_high else b16[:8]

    # Dyadic inputs keep the FP32 MMA and Torch references bit-exact while
    # still making a wrong K permutation immediately visible.
    a_grid = torch.tensor(
        [0.0, 0.25, 0.5, 1.0, 2.0, -0.25, -0.5, -1.0, -2.0],
        device=device,
    )
    a = a_grid[
        torch.randint(
            0,
            a_grid.numel(),
            (16, 32),
            device=device,
            generator=generator,
        )
    ]
    perm = torch.tensor(_NATIVE_MMA_K32_PERM, device=device)
    out = _run_native_trellis_probe(
        _e4m3_bytes(a[:, perm]), packed_tiles, bits, n_high=n_high
    )
    ref = a @ b.T
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_probe_residual_path_exact() -> None:
    """The NVFP4 residual multiply inside the expansion, bit-exact."""
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(1)
    a_vals = torch.tensor([0.5, 1.0, 2.0, -1.0], device=device)
    a = a_vals[torch.randint(0, 4, (_M, _K), device=device)]
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -6.0], device=device)
    b = fp4_grid[torch.randint(0, 9, (_N, _K), device=device)]
    # Dyadic residuals are exact under f16 multiply + e4m3 round.
    res_vals = torch.tensor([1.0, 0.5, 2.0, 0.25], device=device)
    residual = res_vals[torch.randint(0, 4, (_N, 4), device=device)]

    out = _run_probe(
        _e4m3_bytes(a),
        _pack_b(b),
        _e4m3_bytes(residual).view(_N, 4),
        _unit_tables(device),
        _unit_tables(device),
    )
    b_eff = b.view(_N, 4, 16) * residual.unsqueeze(-1)
    ref = a @ b_eff.view(_N, _K).T
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


def _discover_mapping(role: str) -> dict[tuple[int, int], tuple[int, int]]:
    """Empirically map (lane, byte) of SFA/SFB to (row-or-col, k-block).

    All payloads are 1.0 and the other operand's scales are unit.  The kernel
    routes the lane register's low 16 bits to the k-block-0 atom and the high
    16 bits to k-block 1, so each role takes two runs: populate only one
    16-bit half with unique powers of two (the other half's ue8m0 byte 0
    decodes to 2^-127, invisible next to the live term), then read the single
    consumed exponent per output row/col:  out = 32 * 2^e + dust.
    """
    import math

    device = torch.device("cuda")
    a = torch.ones(_M, _K, device=device)
    b = torch.ones(_N, _K, device=device)
    unit = _unit_tables(device)

    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for kb, byte_lo in ((0, 0), (1, 2)):
        # Unique exponent per (lane, half-byte): e = lane*2 + b - 32 in [-32, 31].
        table = torch.zeros(32, dtype=torch.int64, device=device)
        for lane in range(32):
            word = 0
            for b_off in range(2):
                exp = lane * 2 + b_off - 32
                word |= (127 + exp) << (8 * (byte_lo + b_off))
            table[lane] = word
        table = table.to(torch.int32)

        if role == "sfa":
            out = _run_probe(_e4m3_bytes(a), _pack_b(b), None, table, unit)
            idx_count = _M
            series = out[:, 0]
            assert torch.allclose(out, series.unsqueeze(1).expand(_M, _N)), out
        else:
            out = _run_probe(_e4m3_bytes(a), _pack_b(b), None, unit, table)
            idx_count = _N
            series = out[0, :]
            assert torch.allclose(out, series.unsqueeze(0).expand(_M, _N)), out

        for idx in range(idx_count):
            # series = 32 * (2^e + 2^-127-dust): recover e, then (lane, byte).
            val = series[idx].item() / 32.0
            assert val > 0, (role, kb, idx, val)
            e = int(round(math.log2(val)))
            assert abs(val - 2.0**e) <= 2.0**e * 1e-6, (role, kb, idx, val, e)
            slot = e + 32
            lane_id, b_off = slot // 2, slot % 2
            mapping[(lane_id, byte_lo + b_off)] = (idx, kb)
    return mapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_probe_sf_selector_mapping() -> None:
    """Discover and pin the scale_vec::1X SFA/SFB (lane, byte) mapping."""
    require_b12x()
    sfa_map = _discover_mapping("sfa")
    sfb_map = _discover_mapping("sfb")
    print("SFA mapping (lane, byte) -> (row, kblock):", sorted(sfa_map.items()))
    print("SFB mapping (lane, byte) -> (col, kblock):", sorted(sfb_map.items()))

    # Candidate convention (to be asserted once discovered): with the kernel
    # splitting the lane register into per-k-block 16-bit halves and the
    # wrapper defaults bid=0/tid=0, the hardware is expected to read the SF
    # for output row/col `i` of k-block kb from lane (i % 8) * 4 + ... ; the
    # assertions below lock whatever the hardware actually does so Phase 4
    # can build registers against it.
    assert len(sfa_map) == 2 * _M
    assert len(sfb_map) == 2 * _N
    # Self-consistency: each (row/col, kblock) appears exactly once.
    assert len(set(sfa_map.values())) == 2 * _M
    assert len(set(sfb_map.values())) == 2 * _N


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_probe_full_scaled_vs_oracle() -> None:
    """Random data + real block scales routed per the discovered mapping."""
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(2)

    sfa_map = _discover_mapping("sfa")
    sfb_map = _discover_mapping("sfb")

    a_vals = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0, -0.5, 4.0, -2.0], device=device)
    a = a_vals[torch.randint(0, 8, (_M, _K), device=device)]
    fp4_grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
    )
    b = fp4_grid[torch.randint(0, 15, (_N, _K), device=device)]
    sfa_exp = torch.randint(-3, 4, (_M, 2), device=device)
    sfb_exp = torch.randint(-3, 4, (_N, 2), device=device)

    sfa_table = torch.zeros(32, dtype=torch.int64, device=device)
    for (lane, byte), (row, kb) in sfa_map.items():
        sfa_table[lane] |= int(127 + sfa_exp[row, kb].item()) << (8 * byte)
    sfb_table = torch.zeros(32, dtype=torch.int64, device=device)
    for (lane, byte), (col, kb) in sfb_map.items():
        sfb_table[lane] |= int(127 + sfb_exp[col, kb].item()) << (8 * byte)

    out = _run_probe(
        _e4m3_bytes(a),
        _pack_b(b),
        None,
        sfa_table.to(torch.int32),
        sfb_table.to(torch.int32),
    )
    a_eff = a.view(_M, 2, 32) * torch.exp2(sfa_exp.float()).unsqueeze(-1)
    b_eff = b.view(_N, 2, 32) * torch.exp2(sfb_exp.float()).unsqueeze(-1)
    ref = a_eff.view(_M, _K) @ b_eff.view(_N, _K).T
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
