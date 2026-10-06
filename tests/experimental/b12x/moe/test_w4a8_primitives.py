"""Unit tests for the w4a8 (MXFP8) quantization/expansion primitives.

Gates the Phase-1 building blocks bit-tight against pure-Torch references
before any kernel integration:
  * e2m1x8_to_e4m3x8 prmt-LUT nibble expansion (lossless)
  * e2m1x8_mul_residual_to_e4m3x8 (NVFP4 residual decomposition path)
  * quantize_block_fp8_mx / silu_mul_quantize_block_fp8_mx (UE8M0 + E4M3)
  * the pure-Torch grouped quantizer helpers
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import (
    MX_SF_VEC_SIZE,
    broadcast_f32_to_half2,
    e2m1x8_mul_residual_to_e4m3x8,
    e2m1x8_to_e4m3x8,
    max_abs_32,
    pow2_ceil_ue8m0_torch,
    quant_dequant_mxfp8_torch,
    quantize_block_fp8_mx,
    quantize_grouped_mxfp8_torch,
    silu_mul_quantize_block_fp8_mx,
)

from ..conftest import require_b12x

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


class _ExpandKernel:
    num_threads = 128

    @cute.jit
    def __call__(
        self,
        mIn: cute.Tensor,
        mLo: cute.Tensor,
        mHi: cute.Tensor,
        stream: cuda.CUstream,
    ):
        n = cute.size(mIn.shape)
        self.kernel(mIn, mLo, mHi).launch(
            grid=(cute.ceil_div(n, self.num_threads), 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mLo: cute.Tensor, mHi: cute.Tensor):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        idx = bidx * self.num_threads + tidx
        if idx < cute.size(mIn.shape):
            lo, hi = e2m1x8_to_e4m3x8(Uint32(mIn[idx]))
            mLo[idx] = lo
            mHi[idx] = hi


class _ExpandResidualKernel:
    num_threads = 128

    @cute.jit
    def __call__(
        self,
        mIn: cute.Tensor,
        mResidual: cute.Tensor,
        mLo: cute.Tensor,
        mHi: cute.Tensor,
        stream: cuda.CUstream,
    ):
        n = cute.size(mIn.shape)
        self.kernel(mIn, mResidual, mLo, mHi).launch(
            grid=(cute.ceil_div(n, self.num_threads), 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIn: cute.Tensor,
        mResidual: cute.Tensor,
        mLo: cute.Tensor,
        mHi: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        idx = bidx * self.num_threads + tidx
        if idx < cute.size(mIn.shape):
            r_h2 = broadcast_f32_to_half2(Float32(mResidual[idx]))
            lo, hi = e2m1x8_mul_residual_to_e4m3x8(Uint32(mIn[idx]), r_h2)
            mLo[idx] = lo
            mHi[idx] = hi


class _QuantBlockKernel:
    num_threads = 128
    fuse_silu = False

    @cute.jit
    def __call__(
        self,
        mIn: cute.Tensor,
        mUp: cute.Tensor,
        mPayload: cute.Tensor,
        mScale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = cute.size(mIn.shape[0])
        self.kernel(mIn, mUp, mPayload, mScale).launch(
            grid=(cute.ceil_div(rows, self.num_threads), 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIn: cute.Tensor,
        mUp: cute.Tensor,
        mPayload: cute.Tensor,
        mScale: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        row = bidx * self.num_threads + tidx
        if row < cute.size(mIn.shape[0]):
            vals = cute.make_rmem_tensor((32,), Float32)
            for j in cutlass.range_constexpr(32):
                vals[j] = Float32(mIn[row, j])
            if cutlass.const_expr(self.fuse_silu):
                ups = cute.make_rmem_tensor((32,), Float32)
                for j in cutlass.range_constexpr(32):
                    ups[j] = Float32(mUp[row, j])
                payload, scale_byte = silu_mul_quantize_block_fp8_mx(vals, ups)
            else:
                payload, scale_byte = quantize_block_fp8_mx(vals, max_abs_32(vals))
            for j in cutlass.range_constexpr(8):
                mPayload[row, j] = payload[j]
            mScale[row] = Int32(scale_byte)


class _SiluQuantBlockKernel(_QuantBlockKernel):
    fuse_silu = True


def _run_expand(values_u32: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    device = values_u32.device
    lo = torch.zeros_like(values_u32)
    hi = torch.zeros_like(values_u32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if residual is None:
        kernel = _ExpandKernel()
        args = (
            _to_cute_tensor(values_u32, cutlass.Uint32),
            _to_cute_tensor(lo, cutlass.Uint32),
            _to_cute_tensor(hi, cutlass.Uint32),
            stream,
        )
    else:
        kernel = _ExpandResidualKernel()
        args = (
            _to_cute_tensor(values_u32, cutlass.Uint32),
            _to_cute_tensor(residual, cutlass.Float32),
            _to_cute_tensor(lo, cutlass.Uint32),
            _to_cute_tensor(hi, cutlass.Uint32),
            stream,
        )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return lo, hi


def _nibbles(x: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, device=x.device, dtype=torch.int32) * 4
    return (x.unsqueeze(-1) >> shifts) & 0xF


def _bytes_of_pair(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    both = torch.stack([lo, hi], dim=-1)
    shifts = torch.arange(4, device=lo.device, dtype=torch.int32) * 8
    return ((both.unsqueeze(-1) >> shifts) & 0xFF).reshape(*lo.shape, 8)


def _e2m1_lut_f32(device: torch.device) -> torch.Tensor:
    mags = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
    return torch.cat([mags, -mags])


def _native_trellis_tile(edges: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack 256 cyclic edge symbols with EXL's native SWAP16 ordering."""
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_e2m1x8_to_e4m3x8_exact() -> None:
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(0)
    # All 16 codes in every nibble position, plus random patterns.
    sweep = torch.arange(16, device=device, dtype=torch.int64)
    sweep = (sweep.unsqueeze(-1) * (2 ** (4 * torch.arange(8, device=device, dtype=torch.int64)))).sum(
        dim=0, keepdim=False
    )
    patterns = torch.cat(
        [
            torch.tensor([0x76543210, 0xFEDCBA98, 0x0, 0xFFFFFFFF], device=device, dtype=torch.int64),
            torch.randint(0, 2**32, (4096,), device=device, dtype=torch.int64),
        ]
    ).to(torch.int32)

    lo, hi = _run_expand(patterns, None)
    got = _bytes_of_pair(lo, hi)

    lut_f32 = _e2m1_lut_f32(device)
    expected_bytes = lut_f32.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    nib = _nibbles(patterns).to(torch.int64)
    expected = expected_bytes[nib]
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_e2m1x8_mul_residual_to_e4m3x8_matches_f16_reference() -> None:
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(1)
    n = 4096
    patterns = torch.randint(0, 2**32, (n,), device=device, dtype=torch.int64).to(torch.int32)
    # Residuals in (2^-9, 2): the NVFP4 decomposition range, incl. subnormal-ish tails.
    exponents = torch.randint(-9, 1, (n,), device=device, dtype=torch.float32)
    mantissa = 1.0 + torch.rand(n, device=device) * 0.999
    residual = (mantissa * torch.exp2(exponents)).to(torch.float32)

    lo, hi = _run_expand(patterns, residual)
    got = _bytes_of_pair(lo, hi)

    lut_f32 = _e2m1_lut_f32(device)
    nib = _nibbles(patterns).to(torch.int64)
    vals_f16 = lut_f32[nib].to(torch.float16)
    r_f16 = residual.to(torch.float16).unsqueeze(-1)
    prod_f16 = vals_f16 * r_f16
    expected = (
        prod_f16.to(torch.float32)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(torch.int32)
    )
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def _quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocked = x.to(torch.float32)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    rounded, byte = pow2_ceil_ue8m0_torch(block_max * (1.0 / 448.0))
    del rounded
    inv_bits = (254 - byte.to(torch.int32)).clamp(min=0) << 23
    inv = torch.where(byte == 0, torch.zeros_like(byte, dtype=torch.float32), inv_bits.view(torch.float32))
    payload = (
        (blocked * inv)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return payload, byte.squeeze(-1)


def _run_quant(kernel_cls, x: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    rows = x.shape[0]
    payload = torch.zeros(rows, 8, device=device, dtype=torch.int32)
    scale = torch.zeros(rows, device=device, dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = kernel_cls()
    args = (
        _to_cute_tensor(x, cutlass.Float32),
        _to_cute_tensor(up, cutlass.Float32),
        _to_cute_tensor(payload, cutlass.Uint32),
        _to_cute_tensor(scale, cutlass.Int32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()
    payload_bytes = (
        (payload.unsqueeze(-1) >> (torch.arange(4, device=device, dtype=torch.int32) * 8)) & 0xFF
    ).reshape(rows, 32)
    return payload_bytes, scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_block_fp8_mx_bit_exact() -> None:
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(2)
    blocks = [
        torch.randn(512, 32, device=device) * 4.0,
        torch.randn(64, 32, device=device) * 1e-4,
        torch.randn(64, 32, device=device) * 300.0,
        torch.zeros(4, 32, device=device),
    ]
    # A spiky block: one huge outlier to stress the shared block scale.
    spiky = torch.randn(64, 32, device=device)
    spiky[:, 7] = 250.0
    blocks.append(spiky)
    x = torch.cat(blocks).float().contiguous()

    got_payload, got_scale = _run_quant(_QuantBlockKernel, x, torch.zeros_like(x))
    ref_payload, ref_scale = _quant_reference(x)

    torch.testing.assert_close(got_scale, ref_scale.to(torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(got_payload, ref_payload.to(torch.int32), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_mul_quantize_block_fp8_mx_close_to_torch() -> None:
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(3)
    rows = 512
    gate = (torch.randn(rows, 32, device=device) * 3.0).float().contiguous()
    up = (torch.randn(rows, 32, device=device) * 3.0).float().contiguous()

    got_payload, got_scale = _run_quant(_SiluQuantBlockKernel, gate, up)

    activated = torch.nn.functional.silu(gate) * up
    scale_byte = got_scale.float().unsqueeze(-1)
    scale = torch.where(
        scale_byte == 0,
        torch.zeros(rows, 1, device=device),
        torch.exp2(scale_byte - 127.0),
    )
    dequant = (
        got_payload.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32) * scale
    )
    # One E4M3 RN step is up to 2^-4 relative at the bottom of a binade, plus
    # a subnormal floor of the block scale; the kernel's rcp.approx/ex2.approx
    # sigmoid adds only ~1ulp f32 on top.
    block_max = activated.abs().amax(dim=-1, keepdim=True)
    tol = torch.clamp(block_max / 448.0, min=1e-6) * 0.75 + activated.abs() * 0.0725 + 1e-3
    assert torch.all((dequant - activated).abs() <= tol), (
        (dequant - activated).abs().max().item(),
        tol.min().item(),
    )


def test_pow2_ceil_ue8m0_torch_semantics() -> None:
    scale = torch.tensor([0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 2.0**-130, 448.0])
    rounded, byte = pow2_ceil_ue8m0_torch(scale)
    expected_rounded = torch.tensor(
        [0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 2.0**-126, 512.0]
    )
    torch.testing.assert_close(rounded, expected_rounded, atol=0, rtol=0)
    expected_byte = torch.tensor([0, 126, 127, 127, 128, 128, 129, 1, 136], dtype=torch.uint8)
    torch.testing.assert_close(byte, expected_byte, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_grouped_mxfp8_torch_roundtrip() -> None:
    device = torch.device("cuda")
    torch.manual_seed(4)
    groups, rows, cols = 3, 24, 128
    x = torch.randn(groups, rows, cols, device=device) * 2.0
    row_counts = torch.tensor([24, 7, 0], device=device)

    payload, scale_view = quantize_grouped_mxfp8_torch(x, row_counts)
    assert payload.shape == (rows, cols, groups)
    assert payload.dtype == torch.uint8

    # Round-trip the valid rows through the oracle helper and compare against
    # an independent dequant of payload * 2^(scale-127).  The swizzled scale
    # view indexing is exercised end-to-end by the kernel integration tests;
    # here the scales are re-derived from the inputs.
    qd = quant_dequant_mxfp8_torch(x[0, :, :])
    payload_g0 = payload[:, :, 0].view(torch.float8_e4m3fn).to(torch.float32)
    sf_blocks = cols // MX_SF_VEC_SIZE
    blocked = x[0].float().view(rows, sf_blocks, MX_SF_VEC_SIZE)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    rounded, _ = pow2_ceil_ue8m0_torch(block_max * (1.0 / 448.0))
    dequant = (payload_g0.view(rows, sf_blocks, MX_SF_VEC_SIZE) * rounded).view(rows, cols)
    torch.testing.assert_close(dequant, qd, atol=0, rtol=0)

    # Group with zero valid rows must be all-zero payload.
    assert int(payload[:, :, 2].abs().sum().item()) == 0


class _QdE4M3Kernel:
    num_threads = 128

    @cute.jit
    def __call__(self, mIn: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        rows = cute.size(mIn.shape[0])
        self.kernel(mIn, mOut).launch(
            grid=(cute.ceil_div(rows, self.num_threads), 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOut: cute.Tensor):
        from b12x._lib.intrinsics import mx_scale_from_amax32, quant_dequant_e4m3_2

        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        row = bidx * self.num_threads + tidx
        if row < cute.size(mIn.shape[0]):
            peak = Float32(0.0)
            for j in cutlass.range_constexpr(32):
                v = Float32(mIn[row, j])
                m = v
                if v < Float32(0.0):
                    m = -v
                if m > peak:
                    peak = m
            scale32, inv32 = mx_scale_from_amax32(peak)
            for j in cutlass.range_constexpr(16):
                v0 = Float32(mIn[row, j * 2])
                v1 = Float32(mIn[row, j * 2 + 1])
                f0, f1 = quant_dequant_e4m3_2(v0, v1, inv32, scale32)
                mOut[row, j * 2] = f0
                mOut[row, j * 2 + 1] = f1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quant_dequant_e4m3_2_matches_prefill_quantizer() -> None:
    """The micro a8_mx round-trip must bit-match quant_dequant_mxfp8_torch."""
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(5)
    x = torch.cat(
        [
            torch.randn(512, 32, device=device) * 4.0,
            torch.zeros(4, 32, device=device),
            torch.randn(64, 32, device=device) * 300.0,
        ]
    ).float().contiguous()
    out = torch.zeros_like(x)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = _QdE4M3Kernel()
    args = (
        _to_cute_tensor(x, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()
    ref = quant_dequant_mxfp8_torch(x)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
