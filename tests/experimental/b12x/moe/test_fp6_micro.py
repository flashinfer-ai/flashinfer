"""Tests for the MX-FP6 (W6A6) BS1 decode micro-kernel building blocks.

This suite grows alongside the FP6 micro-kernel build. Step 1 covers the
device-side FP6 decode LUT primitive (the software-GEMV compute core), validated
against the host OCP-minifloat reference decode.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import pytest
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import (
    cvt_e8m0_to_f32,
    fabs_f32,
    fmax_f32,
    swizzle_block_scale,
)
from b12x._lib.fp6 import (
    _decode_fp6_e2m3,
    _decode_fp6_e3m2,
    build_fp6_decode_lut,
    dequant_mxfp6_torch,
    fp6_decode_code,
    fp6_unpack4_codes,
    mxfp6_swizzled_scale_offset,
    pack_fp6_codes_tensor,
    quantize_grouped_mxfp6_torch,
)
from b12x.moe._shared.kernels.mxfp6_moe import moe_mxfp6_quantize_input_block_containers

# TODO(port): the FP6 BS1 decode micro kernel's geometry/dispatch surface
# (MXFP6_BLOCK_SIZE, _MAX_DIRECT_K_SEGMENTS, _mxfp6_micro_k_segments_for_k,
# _mxfp6_micro_shapes_ok, is_supported_mxfp6) was not ported upstream, and the
# retired pre-facade b12x.integration.tp_moe host API (allocate_tp_moe_workspace /
# b12x_moe_fp6 / clear_tp_moe_caches) is not part of the current API. Tests that
# depend on either are kept as reference behind pytest.skip gates; rewrite them
# against b12x.moe.fused_moe (quant_mode="w6a8_mx") once the run path
# lands. The decode/scale/quant primitive tests below remain live.
try:
    from b12x.moe._shared.kernels.micro import (
        MXFP6_BLOCK_SIZE,
        _MAX_DIRECT_K_SEGMENTS,
        _mxfp6_micro_k_segments_for_k,
        _mxfp6_micro_shapes_ok,
        MoEMicroKernelBackend,
    )
    from b12x.moe._shared.kernels.silu import MoEMicroKernelSilu

    _HAS_FP6_MICRO = True
except ImportError:  # pragma: no cover - upstream has no FP6 micro kernel yet
    _HAS_FP6_MICRO = False

from b12x.testing.reference.helpers import require_b12x

_FP6_MICRO_SKIP = "pending: FP6 micro kernel port (symbols not upstream)"


class _Fp6DecodeLutKernel:
    """Decode every input FP6 code through the shared-memory LUT primitive."""

    n = 64

    @cute.jit
    def __call__(
        self,
        mLut: cute.Tensor,
        mCodes: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mLut, mCodes, mOut).launch(
            grid=(1, 1, 1),
            block=[self.n, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mLut: cute.Tensor, mCodes: cute.Tensor, mOut: cute.Tensor):
        tidx = cute.arch.thread_idx()[0]
        if tidx < self.n:
            code = Uint32(mCodes[tidx])
            mOut[tidx] = fp6_decode_code(mLut, code)


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_fp6_decode_lut_matches_host(fmt: str) -> None:
    require_b12x()
    device = torch.device("cuda")

    lut = build_fp6_decode_lut(fmt, device=device)
    codes = torch.arange(64, dtype=torch.uint8, device=device)
    out = torch.empty(64, dtype=torch.float32, device=device)

    kernel = _Fp6DecodeLutKernel()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute_tensor(lut, cutlass.Float32),
        _to_cute_tensor(codes, cutlass.Uint8),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    decode = _decode_fp6_e3m2 if fmt == "e3m2" else _decode_fp6_e2m3
    ref = torch.tensor(
        [decode(i) for i in range(64)], dtype=torch.float32, device=device
    )
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


# --- Step 2: FP6 micro geometry + is_supported_mxfp6 (CPU, no GPU) ---------


def test_mxfp6_micro_k_segments_math() -> None:
    if not _HAS_FP6_MICRO:
        pytest.skip(_FP6_MICRO_SKIP)
    # 32 scale-blocks of 32 elements => 1024 contraction elements per segment.
    assert _mxfp6_micro_k_segments_for_k(1024) == 1
    assert _mxfp6_micro_k_segments_for_k(2048) == 2
    assert _mxfp6_micro_k_segments_for_k(1) == 1
    assert _mxfp6_micro_k_segments_for_k(1025) == 2


def test_mxfp6_micro_shapes_predicate_accepts_real_shapes() -> None:
    if not _HAS_FP6_MICRO:
        pytest.skip(_FP6_MICRO_SKIP)
    # Qwen3.6-35B-A3B: K(hidden)=2048, N(inter)=512, topk up to 8, E=256.
    assert _mxfp6_micro_shapes_ok(1, 2048, 512, 8, 256) is True
    assert _mxfp6_micro_shapes_ok(2, 2048, 512, 8, 256) is True
    assert _mxfp6_micro_shapes_ok(8, 128, 128, 8, 4) is True


def test_mxfp6_micro_shapes_predicate_rejects_invalid() -> None:
    if not _HAS_FP6_MICRO:
        pytest.skip(_FP6_MICRO_SKIP)
    assert _mxfp6_micro_shapes_ok(3, 2048, 512, 8, 256) is False  # m not in {1,2,4,8}
    assert _mxfp6_micro_shapes_ok(1, 96, 512, 8, 256) is False  # k % 128 != 0
    assert _mxfp6_micro_shapes_ok(1, 160, 512, 8, 256) is False  # k % 128 != 0
    assert _mxfp6_micro_shapes_ok(1, 2048, 48, 8, 256) is False  # n % 32 != 0
    assert _mxfp6_micro_shapes_ok(1, 2048, 512, 0, 256) is False  # topk == 0
    assert _mxfp6_micro_shapes_ok(1, 2048, 512, 33, 256) is False  # topk > 32
    assert _mxfp6_micro_shapes_ok(1, 2048, 512, 8, 0) is False  # weight_E == 0
    too_large_k = (_MAX_DIRECT_K_SEGMENTS + 1) * 32 * MXFP6_BLOCK_SIZE
    assert (
        _mxfp6_micro_shapes_ok(1, too_large_k, 512, 8, 256) is False
    )  # too many segments


def test_is_supported_mxfp6_default_off(monkeypatch) -> None:
    if not _HAS_FP6_MICRO:
        pytest.skip(_FP6_MICRO_SKIP)
    monkeypatch.delenv("B12X_ENABLE_FP6_MICRO", raising=False)
    assert MoEMicroKernelBackend.is_supported_mxfp6(1, 2048, 512, 8, 256) is False
    assert MoEMicroKernelSilu.is_supported_mxfp6(1, 128, 128, 8, 4) is False


def test_is_supported_mxfp6_opt_in(monkeypatch) -> None:
    if not _HAS_FP6_MICRO:
        pytest.skip(_FP6_MICRO_SKIP)
    monkeypatch.setenv("B12X_ENABLE_FP6_MICRO", "1")
    assert MoEMicroKernelBackend.is_supported_mxfp6(1, 2048, 512, 8, 256) is True
    assert MoEMicroKernelSilu.is_supported_mxfp6(2, 2048, 512, 8, 256) is True
    # Shape predicate still applies when enabled.
    assert MoEMicroKernelBackend.is_supported_mxfp6(3, 2048, 512, 8, 256) is False


def test_is_supported_mxfp6_env_falsey_values(monkeypatch) -> None:
    if not _HAS_FP6_MICRO:
        pytest.skip(_FP6_MICRO_SKIP)
    for val in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("B12X_ENABLE_FP6_MICRO", val)
        assert MoEMicroKernelBackend.is_supported_mxfp6(1, 2048, 512, 8, 256) is False


# --- Step 3a: FP6 GEMV inner loop (LUT decode + UE8M0 scale + f32 accum) ---
#
# Validates the micro kernel's core compute in isolation against an exact torch
# oracle: y[r] = sum_k decode(code[r,k]) * blockscale[r, k//32] * x[k]. Uses random
# codes / scales (no quantizer dependency) so the reference is bit-faithful to the
# kernel's arithmetic; only f32 accumulation-order noise differs.


class _Fp6GemvKernel:
    """Per-row FP6 GEMV: one thread per output row, full-K f32 accumulation."""

    def __init__(self, rows: int, k: int):
        self.rows = int(rows)
        self.k = int(k)
        self.k_blocks = int(k) // 32

    @cute.jit
    def __call__(
        self,
        mLut: cute.Tensor,
        mW: cute.Tensor,
        mScale: cute.Tensor,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mLut, mW, mScale, mX, mOut).launch(
            grid=(1, 1, 1),
            block=[self.rows, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mLut: cute.Tensor,
        mW: cute.Tensor,
        mScale: cute.Tensor,
        mX: cute.Tensor,
        mOut: cute.Tensor,
    ):
        r = cute.arch.thread_idx()[0]
        if r < self.rows:
            acc = Float32(0.0)
            for blk in cutlass.range_constexpr(self.k_blocks):
                scale = cvt_e8m0_to_f32(Uint32(mScale[r, blk]))
                for j in cutlass.range_constexpr(32):
                    kk = blk * 32 + j
                    val = fp6_decode_code(mLut, Uint32(mW[r, kk]))
                    acc = acc + val * scale * Float32(mX[kk])
            mOut[r] = acc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_fp6_gemv_inner_loop_matches_reference(fmt: str) -> None:
    require_b12x()
    torch.manual_seed(0)
    device = torch.device("cuda")
    rows, k = 64, 256
    k_blocks = k // 32

    lut = build_fp6_decode_lut(fmt, device=device)
    codes = torch.randint(0, 64, (rows, k), dtype=torch.uint8, device=device)
    # UE8M0 exponent bytes around 2^0 (127) so block scales are O(1) and non-zero.
    ue = torch.randint(120, 135, (rows, k_blocks), dtype=torch.uint8, device=device)
    x = torch.randn(k, dtype=torch.float32, device=device)
    out = torch.empty(rows, dtype=torch.float32, device=device)

    kernel = _Fp6GemvKernel(rows, k)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute_tensor(lut, cutlass.Float32),
        _to_cute_tensor(codes, cutlass.Uint8),
        _to_cute_tensor(ue, cutlass.Uint8),
        _to_cute_tensor(x, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    block_scale = torch.pow(
        torch.tensor(2.0, device=device), ue.float() - 127.0
    )  # ue in [120,135] => never 0
    w_dq = lut[codes.long()] * block_scale.repeat_interleave(32, dim=1)
    ref = (w_dq.float() @ x).float()
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


# --- Step 3b: production-storage addressing (swizzled scales + packed 3:4) -


class _Fp6SwizzledScaleKernel:
    """Read swizzled UE8M0 scales at logical (row, block) via the offset primitive."""

    def __init__(self, rows: int, num_blocks: int, cols_padded_div4: int):
        self.rows = int(rows)
        self.num_blocks = int(num_blocks)
        self.cpd4 = int(cols_padded_div4)

    @cute.jit
    def __call__(self, mSwz: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mSwz, mOut).launch(
            grid=(1, 1, 1), block=[self.rows, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mSwz: cute.Tensor, mOut: cute.Tensor):
        r = cute.arch.thread_idx()[0]
        if r < self.rows:
            for blk in cutlass.range_constexpr(self.num_blocks):
                off = mxfp6_swizzled_scale_offset(
                    Int32(r), Int32(blk), Int32(self.cpd4)
                )
                mOut[r, blk] = cvt_e8m0_to_f32(Uint32(mSwz[off]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp6_swizzled_scale_addressing_matches_unswizzled() -> None:
    require_b12x()
    torch.manual_seed(1)
    device = torch.device("cuda")
    rows, k = 128, 256  # rows multiple of 128 so the swizzle has no row padding
    num_blocks = k // 32
    cols_padded = ((num_blocks + 3) // 4) * 4

    ue = torch.randint(120, 135, (rows, num_blocks), dtype=torch.uint8, device=device)
    swz = swizzle_block_scale(ue.view(torch.float8_e8m0fnu)).view(torch.uint8)
    assert swz.shape == (rows, cols_padded)
    swz_flat = swz.reshape(-1).contiguous()
    out = torch.empty(rows, num_blocks, dtype=torch.float32, device=device)

    kernel = _Fp6SwizzledScaleKernel(rows, num_blocks, cols_padded // 4)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute_tensor(swz_flat, cutlass.Uint8),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    ref = torch.pow(torch.tensor(2.0, device=device), ue.float() - 127.0)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


class _Fp6PackedDecodeKernel:
    """Decode packed 3:4 FP6 weights to values via the unpack + LUT primitives."""

    def __init__(self, rows: int, k: int):
        self.rows = int(rows)
        self.k = int(k)
        self.groups = int(k) // 4
        self.packed_k = (int(k) * 3) // 4

    @cute.jit
    def __call__(
        self,
        mLut: cute.Tensor,
        mPacked: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mLut, mPacked, mOut).launch(
            grid=(1, 1, 1), block=[self.rows, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mLut: cute.Tensor, mPacked: cute.Tensor, mOut: cute.Tensor):
        r = cute.arch.thread_idx()[0]
        if r < self.rows:
            for g in cutlass.range_constexpr(self.groups):
                b0 = Uint32(mPacked[r, g * 3 + 0])
                b1 = Uint32(mPacked[r, g * 3 + 1])
                b2 = Uint32(mPacked[r, g * 3 + 2])
                c0, c1, c2, c3 = fp6_unpack4_codes(b0, b1, b2)
                mOut[r, g * 4 + 0] = fp6_decode_code(mLut, c0)
                mOut[r, g * 4 + 1] = fp6_decode_code(mLut, c1)
                mOut[r, g * 4 + 2] = fp6_decode_code(mLut, c2)
                mOut[r, g * 4 + 3] = fp6_decode_code(mLut, c3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_fp6_packed_weight_decode_matches_reference(fmt: str) -> None:
    require_b12x()
    torch.manual_seed(2)
    device = torch.device("cuda")
    rows, k = 64, 256

    lut = build_fp6_decode_lut(fmt, device=device)
    codes = torch.randint(0, 64, (rows, k), dtype=torch.uint8, device=device)
    packed = pack_fp6_codes_tensor(codes).contiguous()  # (rows, 3K/4)
    out = torch.empty(rows, k, dtype=torch.float32, device=device)

    kernel = _Fp6PackedDecodeKernel(rows, k)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute_tensor(lut, cutlass.Float32),
        _to_cute_tensor(packed, cutlass.Uint8),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    ref = lut[codes.long()]
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


# --- Step 3c: in-kernel BF16->FP6 activation quant (32-elem UE8M0) ---------


class _Fp6ActQuantKernel:
    """Quantize bf16 activations to FP6 per 32-elem block, then dequant in place.

    Mirrors the static path's activation handling: ``block_max`` over each 32-elem
    block, the shared ``moe_mxfp6_quantize_input_block_containers`` primitive, then
    dequant ``decode(code) * 2^(scale-127) / gs`` for the GEMV input.
    """

    def __init__(self, rows: int, k: int, fmt_a: str):
        self.rows = int(rows)
        self.k = int(k)
        self.k_blocks = int(k) // 32
        self.fmt_a = str(fmt_a)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mLut: cute.Tensor,
        mGs: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mLut, mGs, mOut).launch(
            grid=(1, 1, 1), block=[self.rows, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mLut: cute.Tensor,
        mGs: cute.Tensor,
        mOut: cute.Tensor,
    ):
        r = cute.arch.thread_idx()[0]
        if r < self.rows:
            gs = Float32(mGs[0])
            for blk in cutlass.range_constexpr(self.k_blocks):
                vals = cute.make_rmem_tensor((32,), Float32)
                bmax = Float32(0.0)
                for j in cutlass.range_constexpr(32):
                    v = Float32(mX[r, blk * 32 + j])
                    vals[j] = v
                    bmax = fmax_f32(bmax, fabs_f32(v))
                containers, scale_byte = moe_mxfp6_quantize_input_block_containers(
                    vals, bmax, gs, self.fmt_a
                )
                bscale = cvt_e8m0_to_f32(Uint32(scale_byte))
                for j in cutlass.range_constexpr(32):
                    val = fp6_decode_code(mLut, Uint32(containers[j]))
                    mOut[r, blk * 32 + j] = val * bscale / gs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fmt_a", ["e3m2", "e2m3"])
def test_fp6_activation_quant_matches_torch_reference(fmt_a: str) -> None:
    require_b12x()
    torch.manual_seed(3)
    device = torch.device("cuda")
    rows, k = 8, 128

    x = (torch.randn(rows, k, device=device) / 2).to(torch.bfloat16)
    lut = build_fp6_decode_lut(fmt_a, device=device)
    gs = torch.ones(1, dtype=torch.float32, device=device)
    out = torch.empty(rows, k, dtype=torch.float32, device=device)

    kernel = _Fp6ActQuantKernel(rows, k, fmt_a)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute_tensor(x, cutlass.BFloat16),
        _to_cute_tensor(lut, cutlass.Float32),
        _to_cute_tensor(gs, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    # Torch reference: same per-32-block UE8M0 quant (bf16-rounded, matching the GPU
    # cvt path), then dequant back to float.
    row_counts = torch.tensor([rows], dtype=torch.int32, device=device)
    packed, scales = quantize_grouped_mxfp6_torch(
        x.float().unsqueeze(0), row_counts, gs, fmt=fmt_a, bf16_round=True
    )
    ref = dequant_mxfp6_torch(
        packed[..., 0], scales, num_fp6=k, fmt=fmt_a, scales_swizzled=True
    ).reshape(rows, k)

    cos = torch.nn.functional.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    )
    assert cos.item() > 0.9999, f"activation quant cosine too low: {cos.item()}"
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


# --- Step 3d-1: FC1 GEMV on production FP6 storage (packed W + swizzled SF) -


class _Fp6Fc1Kernel:
    """Single-expert FC1: h[row] = sum_k decode(W1[row,k]) * wscale[row,k//32] * a_dq[k].

    Reads the production packed weight ``(2N, 3K/4)`` and swizzled UE8M0 scale
    storage (one expert) exactly as ``FP6MoEWeights`` lays them out, with a
    pre-quantized FP6 activation input. One thread per FC1 output row.
    """

    def __init__(self, rows: int, k: int, cols_padded_div4: int):
        self.rows = int(rows)
        self.k = int(k)
        self.k_blocks = int(k) // 32
        self.cpd4 = int(cols_padded_div4)

    @cute.jit
    def __call__(
        self,
        mLutW: cute.Tensor,
        mPackedW: cute.Tensor,
        mScaleW: cute.Tensor,
        mAdq: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mLutW, mPackedW, mScaleW, mAdq, mOut).launch(
            grid=(1, 1, 1), block=[self.rows, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mLutW: cute.Tensor,
        mPackedW: cute.Tensor,
        mScaleW: cute.Tensor,
        mAdq: cute.Tensor,
        mOut: cute.Tensor,
    ):
        r = cute.arch.thread_idx()[0]
        if r < self.rows:
            acc = Float32(0.0)
            for blk in cutlass.range_constexpr(self.k_blocks):
                off = mxfp6_swizzled_scale_offset(
                    Int32(r), Int32(blk), Int32(self.cpd4)
                )
                wscale = cvt_e8m0_to_f32(Uint32(mScaleW[off]))
                for gg in cutlass.range_constexpr(8):
                    g = blk * 8 + gg
                    b0 = Uint32(mPackedW[r, g * 3 + 0])
                    b1 = Uint32(mPackedW[r, g * 3 + 1])
                    b2 = Uint32(mPackedW[r, g * 3 + 2])
                    c0, c1, c2, c3 = fp6_unpack4_codes(b0, b1, b2)
                    base_k = blk * 32 + gg * 4
                    acc = acc + fp6_decode_code(mLutW, c0) * wscale * Float32(
                        mAdq[base_k + 0]
                    )
                    acc = acc + fp6_decode_code(mLutW, c1) * wscale * Float32(
                        mAdq[base_k + 1]
                    )
                    acc = acc + fp6_decode_code(mLutW, c2) * wscale * Float32(
                        mAdq[base_k + 2]
                    )
                    acc = acc + fp6_decode_code(mLutW, c3) * wscale * Float32(
                        mAdq[base_k + 3]
                    )
            mOut[r] = acc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp6_fc1_gemv_on_production_storage() -> None:
    require_b12x()
    from b12x.quantization.mxfp6 import quantize_moe_weights_to_fp6

    torch.manual_seed(4)
    device = torch.device("cuda")
    e, k, n = 2, 128, 128  # GPU quantizer needs M%128==0 and K%128==0
    two_n = 2 * n
    act_fmt = "e3m2"  # mxfp6_default activations

    w1 = (torch.randn(e, two_n, k, device=device) / 8).to(torch.bfloat16)
    w2 = (torch.randn(e, k, n, device=device) / 8).to(torch.bfloat16)
    weights = quantize_moe_weights_to_fp6(
        w1, w2, source_format="mxfp6_default", activation="silu", use_gpu=True
    )
    weight_fmt = weights.weight_fmt  # "e2m3"
    lut_w = build_fp6_decode_lut(weight_fmt, device=device)

    # Build a real FP6-quantized activation (validated in 3c) as kernel input.
    x = (torch.randn(k, device=device) / 2).to(torch.bfloat16)
    gs = torch.ones(1, dtype=torch.float32, device=device)
    packed_a, scales_a = quantize_grouped_mxfp6_torch(
        x.float().reshape(1, 1, k),
        torch.tensor([1], dtype=torch.int32, device=device),
        gs,
        fmt=act_fmt,
        bf16_round=True,
    )
    a_dq = (
        dequant_mxfp6_torch(
            packed_a[..., 0], scales_a, num_fp6=k, fmt=act_fmt, scales_swizzled=True
        )
        .reshape(k)
        .contiguous()
    )

    expert = 0
    w1_packed = weights.w1_fp6[expert].contiguous()  # (2N, 3K/4)
    w1_scale = weights.w1_blockscale[expert].reshape(-1).contiguous()  # swizzled flat
    cols_padded = ((k // 32 + 3) // 4) * 4
    out = torch.empty(two_n, dtype=torch.float32, device=device)

    kernel = _Fp6Fc1Kernel(two_n, k, cols_padded // 4)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute_tensor(lut_w, cutlass.Float32),
        _to_cute_tensor(w1_packed, cutlass.Uint8),
        _to_cute_tensor(w1_scale, cutlass.Uint8),
        _to_cute_tensor(a_dq, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    # Reference: dequant production weights and matmul with the same FP6 activation.
    w1_dq = dequant_mxfp6_torch(
        w1_packed,
        weights.w1_blockscale[expert],
        num_fp6=k,
        fmt=weight_fmt,
        scales_swizzled=True,
    ).reshape(two_n, k)
    ref = (w1_dq.double() @ a_dq.double()).float()  # f64 to avoid TF32 noise

    cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=0)
    assert cos.item() > 0.999, f"FC1 GEMV cosine too low: {cos.item()}"
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# --- Step 3d-2: FC1 + SiLU(gate)*up gating -> intermediate ------------------


class _Fp6Fc1GateKernel:
    """FC1 into smem, then gated activation: inter[n] = silu(gate[n]) * up[n].

    FC1 rows are ``[up; gate]`` (up = rows [0:N], gate = rows [N:2N]) matching the
    validated numeric reference; SiLU is applied to the gate (second) half.
    """

    def __init__(self, n: int, k: int, cols_padded_div4: int):
        self.n = int(n)
        self.two_n = 2 * int(n)
        self.k = int(k)
        self.k_blocks = int(k) // 32
        self.cpd4 = int(cols_padded_div4)

    @cute.jit
    def __call__(
        self,
        mLutW: cute.Tensor,
        mPackedW: cute.Tensor,
        mScaleW: cute.Tensor,
        mAdq: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mLutW, mPackedW, mScaleW, mAdq, mOut).launch(
            grid=(1, 1, 1), block=[self.two_n, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mLutW: cute.Tensor,
        mPackedW: cute.Tensor,
        mScaleW: cute.Tensor,
        mAdq: cute.Tensor,
        mOut: cute.Tensor,
    ):
        r = cute.arch.thread_idx()[0]
        smem = cutlass_utils.SmemAllocator()
        sh = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(self.two_n),
            byte_alignment=128,
        )
        if r < self.two_n:
            acc = Float32(0.0)
            for blk in cutlass.range_constexpr(self.k_blocks):
                off = mxfp6_swizzled_scale_offset(
                    Int32(r), Int32(blk), Int32(self.cpd4)
                )
                wscale = cvt_e8m0_to_f32(Uint32(mScaleW[off]))
                for gg in cutlass.range_constexpr(8):
                    g = blk * 8 + gg
                    b0 = Uint32(mPackedW[r, g * 3 + 0])
                    b1 = Uint32(mPackedW[r, g * 3 + 1])
                    b2 = Uint32(mPackedW[r, g * 3 + 2])
                    c0, c1, c2, c3 = fp6_unpack4_codes(b0, b1, b2)
                    base_k = blk * 32 + gg * 4
                    acc = acc + fp6_decode_code(mLutW, c0) * wscale * Float32(
                        mAdq[base_k + 0]
                    )
                    acc = acc + fp6_decode_code(mLutW, c1) * wscale * Float32(
                        mAdq[base_k + 1]
                    )
                    acc = acc + fp6_decode_code(mLutW, c2) * wscale * Float32(
                        mAdq[base_k + 2]
                    )
                    acc = acc + fp6_decode_code(mLutW, c3) * wscale * Float32(
                        mAdq[base_k + 3]
                    )
            sh[r] = acc
        cute.arch.sync_threads()
        if r < self.n:
            up = sh[r]
            gate = sh[self.n + r]
            sigmoid_g = cute.arch.rcp_approx(
                Float32(1.0) + cute.math.exp(-gate, fastmath=False)
            )
            mOut[r] = sigmoid_g * gate * up


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp6_fc1_gate_intermediate() -> None:
    require_b12x()
    from b12x.quantization.mxfp6 import quantize_moe_weights_to_fp6

    torch.manual_seed(5)
    device = torch.device("cuda")
    e, k, n = 2, 128, 128
    two_n = 2 * n
    act_fmt = "e3m2"

    w1 = (torch.randn(e, two_n, k, device=device) / 8).to(torch.bfloat16)
    w2 = (torch.randn(e, k, n, device=device) / 8).to(torch.bfloat16)
    weights = quantize_moe_weights_to_fp6(
        w1, w2, source_format="mxfp6_default", activation="silu", use_gpu=True
    )
    weight_fmt = weights.weight_fmt
    lut_w = build_fp6_decode_lut(weight_fmt, device=device)

    x = (torch.randn(k, device=device) / 2).to(torch.bfloat16)
    gs = torch.ones(1, dtype=torch.float32, device=device)
    packed_a, scales_a = quantize_grouped_mxfp6_torch(
        x.float().reshape(1, 1, k),
        torch.tensor([1], dtype=torch.int32, device=device),
        gs,
        fmt=act_fmt,
        bf16_round=True,
    )
    a_dq = (
        dequant_mxfp6_torch(
            packed_a[..., 0], scales_a, num_fp6=k, fmt=act_fmt, scales_swizzled=True
        )
        .reshape(k)
        .contiguous()
    )

    expert = 0
    w1_packed = weights.w1_fp6[expert].contiguous()
    w1_scale = weights.w1_blockscale[expert].reshape(-1).contiguous()
    cols_padded = ((k // 32 + 3) // 4) * 4
    out = torch.empty(n, dtype=torch.float32, device=device)

    kernel = _Fp6Fc1GateKernel(n, k, cols_padded // 4)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute_tensor(lut_w, cutlass.Float32),
        _to_cute_tensor(w1_packed, cutlass.Uint8),
        _to_cute_tensor(w1_scale, cutlass.Uint8),
        _to_cute_tensor(a_dq, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    w1_dq = dequant_mxfp6_torch(
        w1_packed,
        weights.w1_blockscale[expert],
        num_fp6=k,
        fmt=weight_fmt,
        scales_swizzled=True,
    ).reshape(two_n, k)
    h = w1_dq.double() @ a_dq.double()  # [2N]
    up_ref, gate_ref = h[:n], h[n:]
    inter_ref = (torch.sigmoid(gate_ref) * gate_ref * up_ref).float()

    cos = torch.nn.functional.cosine_similarity(out.float(), inter_ref.float(), dim=0)
    assert cos.item() > 0.999, f"FC1+gate cosine too low: {cos.item()}"
    torch.testing.assert_close(out, inter_ref, atol=2e-2, rtol=2e-2)


# --- Step 3d-3: full fused BS1 routed MoE micro vs static b12x_moe_fp6 ------


class _Fp6MoeBs1Kernel:
    """Full fused routed MX-FP6 MoE for small-batch decode (one CTA per token).

    Per token, for each of ``topk`` routed experts: FC1 (W1 @ a_dq) into smem,
    SiLU(gate)*up gating, in-kernel FP6 re-quant of the intermediate, FC2
    (W2 @ inter_dq), and router-weighted accumulation into the output. Mirrors the
    static path's math with the proven decode/scale/quant primitives.
    """

    def __init__(
        self, n: int, k: int, topk: int, act_fmt: str, cpd4_w1: int, cpd4_w2: int
    ):
        self.n = int(n)
        self.two_n = 2 * int(n)
        self.k = int(k)
        self.topk = int(topk)
        self.act_fmt = str(act_fmt)
        self.k_blocks = int(k) // 32
        self.n_blocks = int(n) // 32
        self.cpd4_w1 = int(cpd4_w1)
        self.cpd4_w2 = int(cpd4_w2)
        self.blk_dim = max(self.two_n, self.k)

    @cute.jit
    def __call__(
        self,
        mLutW: cute.Tensor,
        mLutA: cute.Tensor,
        mAdq: cute.Tensor,
        mW1p: cute.Tensor,
        mW1s: cute.Tensor,
        mW2p: cute.Tensor,
        mW2s: cute.Tensor,
        mIds: cute.Tensor,
        mTw: cute.Tensor,
        mGs: cute.Tensor,
        mOut: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mLutW, mLutA, mAdq, mW1p, mW1s, mW2p, mW2s, mIds, mTw, mGs, mOut
        ).launch(grid=(m, 1, 1), block=[self.blk_dim, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mLutW: cute.Tensor,
        mLutA: cute.Tensor,
        mAdq: cute.Tensor,
        mW1p: cute.Tensor,
        mW1s: cute.Tensor,
        mW2p: cute.Tensor,
        mW2s: cute.Tensor,
        mIds: cute.Tensor,
        mTw: cute.Tensor,
        mGs: cute.Tensor,
        mOut: cute.Tensor,
    ):
        t = cute.arch.block_idx()[0]
        r = cute.arch.thread_idx()[0]
        smem = cutlass_utils.SmemAllocator()
        sh = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(self.two_n),
            byte_alignment=128,
        )
        sinter = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout(self.n), byte_alignment=128
        )
        sinter_dq = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout(self.n), byte_alignment=128
        )
        sout = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout(self.k), byte_alignment=128
        )
        gs = Float32(mGs[0])

        if r < self.k:
            sout[r] = Float32(0.0)
        cute.arch.sync_threads()

        for slot in cutlass.range_constexpr(self.topk):
            e = Int32(mIds[t, slot])
            # ---- FC1: h[r] = sum_k decode(W1[e,r,k]) * wscale * a_dq[t,k] ----
            if r < self.two_n:
                acc = Float32(0.0)
                for blk in cutlass.range_constexpr(self.k_blocks):
                    off = mxfp6_swizzled_scale_offset(
                        Int32(r), Int32(blk), Int32(self.cpd4_w1)
                    )
                    wsc = cvt_e8m0_to_f32(Uint32(mW1s[e, off]))
                    for gg in cutlass.range_constexpr(8):
                        g = blk * 8 + gg
                        c0, c1, c2, c3 = fp6_unpack4_codes(
                            Uint32(mW1p[e, r, g * 3 + 0]),
                            Uint32(mW1p[e, r, g * 3 + 1]),
                            Uint32(mW1p[e, r, g * 3 + 2]),
                        )
                        bk = blk * 32 + gg * 4
                        acc = acc + fp6_decode_code(mLutW, c0) * wsc * Float32(
                            mAdq[t, bk + 0]
                        )
                        acc = acc + fp6_decode_code(mLutW, c1) * wsc * Float32(
                            mAdq[t, bk + 1]
                        )
                        acc = acc + fp6_decode_code(mLutW, c2) * wsc * Float32(
                            mAdq[t, bk + 2]
                        )
                        acc = acc + fp6_decode_code(mLutW, c3) * wsc * Float32(
                            mAdq[t, bk + 3]
                        )
                sh[r] = acc
            cute.arch.sync_threads()
            # ---- gate: inter[n] = silu(gate)*up (up=rows[:N], gate=rows[N:]) ----
            if r < self.n:
                up = sh[r]
                gate = sh[self.n + r]
                sig = cute.arch.rcp_approx(
                    Float32(1.0) + cute.math.exp(-gate, fastmath=False)
                )
                sinter[r] = sig * gate * up
            cute.arch.sync_threads()
            # ---- re-quant intermediate to FP6 (one thread per 32-elem block) ----
            if r < self.n_blocks:
                vals = cute.make_rmem_tensor((32,), Float32)
                bmax = Float32(0.0)
                for j in cutlass.range_constexpr(32):
                    v = sinter[r * 32 + j]
                    vals[j] = v
                    bmax = fmax_f32(bmax, fabs_f32(v))
                containers, sb = moe_mxfp6_quantize_input_block_containers(
                    vals, bmax, gs, self.act_fmt
                )
                bsc = cvt_e8m0_to_f32(Uint32(sb))
                for j in cutlass.range_constexpr(32):
                    sinter_dq[r * 32 + j] = (
                        fp6_decode_code(mLutA, Uint32(containers[j])) * bsc / gs
                    )
            cute.arch.sync_threads()
            # ---- FC2: y[r] = sum_n decode(W2[e,r,n]) * wscale * inter_dq[n] ----
            if r < self.k:
                acc2 = Float32(0.0)
                for blk in cutlass.range_constexpr(self.n_blocks):
                    off = mxfp6_swizzled_scale_offset(
                        Int32(r), Int32(blk), Int32(self.cpd4_w2)
                    )
                    wsc = cvt_e8m0_to_f32(Uint32(mW2s[e, off]))
                    for gg in cutlass.range_constexpr(8):
                        g = blk * 8 + gg
                        c0, c1, c2, c3 = fp6_unpack4_codes(
                            Uint32(mW2p[e, r, g * 3 + 0]),
                            Uint32(mW2p[e, r, g * 3 + 1]),
                            Uint32(mW2p[e, r, g * 3 + 2]),
                        )
                        bn = blk * 32 + gg * 4
                        acc2 = (
                            acc2 + fp6_decode_code(mLutW, c0) * wsc * sinter_dq[bn + 0]
                        )
                        acc2 = (
                            acc2 + fp6_decode_code(mLutW, c1) * wsc * sinter_dq[bn + 1]
                        )
                        acc2 = (
                            acc2 + fp6_decode_code(mLutW, c2) * wsc * sinter_dq[bn + 2]
                        )
                        acc2 = (
                            acc2 + fp6_decode_code(mLutW, c3) * wsc * sinter_dq[bn + 3]
                        )
                sout[r] = sout[r] + Float32(mTw[t, slot]) * acc2
            cute.arch.sync_threads()

        if r < self.k:
            mOut[t, r] = sout[r]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("m", [1, 4])
def test_fp6_moe_bs1_micro_matches_static(m: int) -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    require_b12x()
    from b12x.integration.tp_moe import (  # retired pre-facade API (module TODO)
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )
    from b12x.quantization.mxfp6 import quantize_moe_weights_to_fp6

    clear_tp_moe_caches()
    torch.manual_seed(7)
    device = torch.device("cuda")
    k, n, experts, topk = 128, 128, 2, 2
    two_n = 2 * n
    act_fmt = "e3m2"

    x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).float()

    w1 = (torch.randn(experts, two_n, k, device=device) * 0.15).to(torch.bfloat16)
    w2 = (torch.randn(experts, k, n, device=device) * 0.15).to(torch.bfloat16)
    weights = quantize_moe_weights_to_fp6(
        w1, w2, source_format="mxfp6_default", activation="silu", use_gpu=True
    )

    # Ground truth: the validated static fused FP6 path.
    workspace = allocate_tp_moe_workspace(
        x,
        weights.a1_gscale,
        weights.w1_fp6,
        weights.a2_gscale,
        weights.w2_fp6,
        topk_ids,
        quant_mode="w6a6",
        input_scales_static=True,
    )
    static_out = b12x_moe_fp6(
        x,
        weights.a1_gscale,
        weights.w1_fp6,
        weights.w1_blockscale,
        weights.w1_alphas,
        weights.a2_gscale,
        weights.w2_fp6,
        weights.w2_blockscale,
        weights.w2_alphas,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        source_format="mxfp6_default",
    )
    torch.cuda.synchronize()

    # Pre-quantize activations to FP6 (validated in 3c) for the micro kernel input.
    a_dq = torch.empty(m, k, dtype=torch.float32, device=device)
    gs1 = torch.ones(1, dtype=torch.float32, device=device)
    for t in range(m):
        packed_a, scales_a = quantize_grouped_mxfp6_torch(
            x[t].float().reshape(1, 1, k),
            torch.tensor([1], dtype=torch.int32, device=device),
            gs1,
            fmt=act_fmt,
            bf16_round=True,
        )
        a_dq[t] = dequant_mxfp6_torch(
            packed_a[..., 0], scales_a, num_fp6=k, fmt=act_fmt, scales_swizzled=True
        ).reshape(k)

    lut_w = build_fp6_decode_lut(weights.weight_fmt, device=device)
    lut_a = build_fp6_decode_lut(act_fmt, device=device)
    w1s = weights.w1_blockscale.reshape(experts, -1).contiguous()
    w2s = weights.w2_blockscale.reshape(experts, -1).contiguous()
    cpd4_w1 = ((k // 32) + 3) // 4
    cpd4_w2 = ((n // 32) + 3) // 4
    micro_out = torch.empty(m, k, dtype=torch.float32, device=device)

    kernel = _Fp6MoeBs1Kernel(n, k, topk, act_fmt, cpd4_w1, cpd4_w2)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute_tensor(lut_w, cutlass.Float32),
        _to_cute_tensor(lut_a, cutlass.Float32),
        _to_cute_tensor(a_dq, cutlass.Float32),
        _to_cute_tensor(weights.w1_fp6.contiguous(), cutlass.Uint8),
        _to_cute_tensor(w1s, cutlass.Uint8),
        _to_cute_tensor(weights.w2_fp6.contiguous(), cutlass.Uint8),
        _to_cute_tensor(w2s, cutlass.Uint8),
        _to_cute_tensor(topk_ids.contiguous(), cutlass.Int32),
        _to_cute_tensor(topk_weights.contiguous(), cutlass.Float32),
        _to_cute_tensor(weights.a2_gscale, cutlass.Float32),
        _to_cute_tensor(micro_out, cutlass.Float32),
        Int32(m),
        stream,
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    cos = torch.nn.functional.cosine_similarity(
        micro_out.reshape(-1).float(), static_out.reshape(-1).float(), dim=0
    )
    assert cos.item() > 0.99, f"micro vs static cosine too low: {cos.item()}"


# --- Step 4: dispatch wiring -- b12x_moe_fp6 routes to micro via env gate ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("m", [1, 4])
def test_fp6_micro_dispatch_matches_static_via_b12x_moe_fp6(
    monkeypatch, m: int
) -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    require_b12x()
    from b12x.integration.tp_moe import (  # retired pre-facade API (module TODO)
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )
    from b12x.quantization.mxfp6 import quantize_moe_weights_to_fp6

    torch.manual_seed(11)
    device = torch.device("cuda")
    k, n, experts, topk = 128, 128, 2, 2
    two_n = 2 * n

    x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).float()
    w1 = (torch.randn(experts, two_n, k, device=device) * 0.15).to(torch.bfloat16)
    w2 = (torch.randn(experts, k, n, device=device) * 0.15).to(torch.bfloat16)
    weights = quantize_moe_weights_to_fp6(
        w1, w2, source_format="mxfp6_default", activation="silu", use_gpu=True
    )

    def _run() -> torch.Tensor:
        clear_tp_moe_caches()
        ws = allocate_tp_moe_workspace(
            x,
            weights.a1_gscale,
            weights.w1_fp6,
            weights.a2_gscale,
            weights.w2_fp6,
            topk_ids,
            quant_mode="w6a6",
            input_scales_static=True,
        )
        out = b12x_moe_fp6(
            x,
            weights.a1_gscale,
            weights.w1_fp6,
            weights.w1_blockscale,
            weights.w1_alphas,
            weights.a2_gscale,
            weights.w2_fp6,
            weights.w2_blockscale,
            weights.w2_alphas,
            topk_weights,
            topk_ids,
            workspace=ws,
            input_scales_static=True,
            source_format="mxfp6_default",
        )
        torch.cuda.synchronize()
        return out

    monkeypatch.setenv("B12X_ENABLE_FP6_MICRO", "0")
    static_out = _run()
    monkeypatch.setenv("B12X_ENABLE_FP6_MICRO", "1")
    micro_out = _run()

    cos = torch.nn.functional.cosine_similarity(
        micro_out.reshape(-1).float(), static_out.reshape(-1).float(), dim=0
    )
    assert cos.item() > 0.99, f"dispatch micro vs static cosine too low: {cos.item()}"


# --- Step 5: larger-shape numeric (row-stride loops + runtime contraction) ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("k,n", [(512, 512)])
def test_fp6_micro_dispatch_larger_shapes(monkeypatch, m: int, k: int, n: int) -> None:
    """K/N > one CTA exercise the row-stride loops; K,N=512 the runtime blk loop."""
    pytest.skip("pending: fused_moe-based rewrite")
    require_b12x()
    from b12x.integration.tp_moe import (  # retired pre-facade API (module TODO)
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )
    from b12x.quantization.mxfp6 import quantize_moe_weights_to_fp6

    torch.manual_seed(23)
    device = torch.device("cuda")
    experts, topk = 4, 2
    two_n = 2 * n

    x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).float()
    w1 = (torch.randn(experts, two_n, k, device=device) * 0.1).to(torch.bfloat16)
    w2 = (torch.randn(experts, k, n, device=device) * 0.1).to(torch.bfloat16)
    weights = quantize_moe_weights_to_fp6(
        w1, w2, source_format="mxfp6_default", activation="silu", use_gpu=True
    )

    def _run() -> torch.Tensor:
        clear_tp_moe_caches()
        ws = allocate_tp_moe_workspace(
            x,
            weights.a1_gscale,
            weights.w1_fp6,
            weights.a2_gscale,
            weights.w2_fp6,
            topk_ids,
            quant_mode="w6a6",
            input_scales_static=True,
        )
        out = b12x_moe_fp6(
            x,
            weights.a1_gscale,
            weights.w1_fp6,
            weights.w1_blockscale,
            weights.w1_alphas,
            weights.a2_gscale,
            weights.w2_fp6,
            weights.w2_blockscale,
            weights.w2_alphas,
            topk_weights,
            topk_ids,
            workspace=ws,
            input_scales_static=True,
            source_format="mxfp6_default",
        )
        torch.cuda.synchronize()
        return out

    monkeypatch.setenv("B12X_ENABLE_FP6_MICRO", "0")
    static_out = _run()
    monkeypatch.setenv("B12X_ENABLE_FP6_MICRO", "1")
    micro_out = _run()

    cos = torch.nn.functional.cosine_similarity(
        micro_out.reshape(-1).float(), static_out.reshape(-1).float(), dim=0
    )
    assert cos.item() > 0.99, (
        f"larger-shape micro vs static cosine too low: {cos.item()}"
    )
