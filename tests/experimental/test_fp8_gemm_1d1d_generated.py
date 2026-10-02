"""Prepared FP8 1D1D GEMM: random-input references at several shapes, both routes,
in-place accumulation, non-default-stream submission and CUDA-graph replay."""

import pytest
import torch

from flashinfer.experimental.deepgemm_fp8_gemm import (
    pack_ue8m0_words,
    prepare_fp8_gemm_1d1d,
)
from flashinfer.experimental.deepgemm_fp8_gemm import runtime as _runtime


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        return _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))


def _ue8m0_exponent(amax):
    """UE8M0 exponent byte of the power-of-two scale covering ``amax`` at E4M3 range."""
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp(min=1e-4) / 448.0)))
    return (scale.view(torch.int32) >> 23).to(torch.uint8), scale


def _quantize_rows(x, block_rows):
    """Cast BF16 ``x`` ``[R, K]`` to E4M3 with one UE8M0 scale per ``block_rows`` x 128 block."""
    rows, k = x.shape
    blocks = x.float().reshape(rows // block_rows, block_rows, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    exponent, scale = _ue8m0_exponent(amax)
    fp8 = (blocks / scale).to(torch.float8_e4m3fn)
    dequant = fp8.float() * scale
    return (
        fp8.reshape(rows, k).view(torch.uint8),
        exponent.reshape(rows // block_rows, k // 128),
        dequant.reshape(rows, k),
    )


def _case(M, N, K, accumulate, seed):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    pad_m = (-M) % 128
    pad_n = (-N) % 128
    a_u8, sfa_u8, a_deq = _quantize_rows(
        torch.nn.functional.pad(a, (0, 0, 0, pad_m)), 1
    )
    # Forward B carries one scale per 128 x 128 block; wgrad B is per-token cast.
    b_u8, sfb_u8, b_deq = _quantize_rows(
        torch.nn.functional.pad(b, (0, 0, 0, pad_n)), 1 if accumulate else 128
    )
    a_u8, sfa_u8, a_deq = a_u8[:M], sfa_u8[:M], a_deq[:M]
    b_u8, b_deq = b_u8[:N], b_deq[:N]
    if accumulate:
        sfb_u8 = sfb_u8[:N]
    sfa = pack_ue8m0_words(sfa_u8, M)
    sfb = pack_ue8m0_words(sfb_u8, N)
    reference = a_deq @ b_deq.T
    return a_u8.contiguous(), b_u8.contiguous(), sfa, sfb, reference


def _output(M, N, dtype):
    """``[M, N]`` output with a 16-byte row pitch: contiguous when ``N`` allows it,
    otherwise the leading columns of a wider buffer (the documented route for
    ``N`` values whose dense pitch TMA cannot encode)."""
    per_pitch = 16 // torch.empty((), dtype=dtype).element_size()
    pitch = (N + per_pitch - 1) // per_pitch * per_pitch
    return torch.empty(M, pitch, device="cuda", dtype=dtype)[:, :N]


SHAPES = [
    (4096, 7168, 4096),  # deployment row
    (1024, 2048, 1024),
    (3000, 4000, 1152),  # ragged M, N not a multiple of 224, 9 K tiles
    (128, 1024, 512),  # single M tile pair
    (300, 7168, 2048),
    (1, 7168, 512),  # decode row: M % 4 != 0 packs a padded scale pitch
    (3, 130, 256),  # M and N both off the 4-row pitch; out needs a padded pitch
]


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_fp8_gemm_1d1d_matches_dequantized_reference(M, N, K, accumulate):
    _skip_unless_supported()
    a, b, sfa, sfb, reference = _case(M, N, K, accumulate, seed=M * 31 + N * 7 + K)
    if accumulate:
        initial = torch.randn(M, N, device="cuda", dtype=torch.float32) * 32.0
        out = _output(M, N, torch.float32)
        out.copy_(initial)
        expected = reference + initial
    else:
        out = _output(M, N, torch.bfloat16)
        expected = reference
    address = out.data_ptr()
    plan = prepare_fp8_gemm_1d1d(a, b, sfa, sfb, out, accumulate=accumulate)
    plan.run()
    torch.cuda.synchronize()
    assert out.data_ptr() == address
    if accumulate:
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-3)
        plan.run()
        torch.testing.assert_close(out, expected + reference, atol=2e-2, rtol=1e-3)
    else:
        torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("accumulate", [False, True])
def test_fp8_gemm_1d1d_replays_on_a_stream_and_in_a_graph(accumulate):
    _skip_unless_supported()
    M, N, K = 1024, 2048, 1024
    a, b, sfa, sfb, reference = _case(M, N, K, accumulate, seed=11)
    initial = (
        torch.randn(M, N, device="cuda", dtype=torch.float32) if accumulate else None
    )
    out = torch.empty(
        M, N, device="cuda", dtype=torch.float32 if accumulate else torch.bfloat16
    )
    plan = prepare_fp8_gemm_1d1d(a, b, sfa, sfb, out, accumulate=accumulate)

    def reset():
        if accumulate:
            out.copy_(initial)
        else:
            out.zero_()

    expected = reference + initial if accumulate else reference
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reset()
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
        reset()
        graph.replay()
    stream.synchronize()
    tolerance = dict(atol=1e-2, rtol=1e-3) if accumulate else dict(atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out.float(), expected, **tolerance)


def test_fp8_gemm_1d1d_rejects_invalid_operands():
    _skip_unless_supported()
    a, b, sfa, sfb, _ = _case(256, 448, 512, False, seed=3)
    out = torch.empty(256, 448, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        prepare_fp8_gemm_1d1d(a, b, sfa, sfb, out.float())
    with pytest.raises(TypeError):
        prepare_fp8_gemm_1d1d(a, b, sfa[:, :128], sfb, out)
    with pytest.raises(ValueError):
        prepare_fp8_gemm_1d1d(a[:, :500], b[:, :500], sfa, sfb, out)
    # K=512 packs into one word per row; row counts that are multiples of four
    # keep a dense pitch, and a 4-byte pitch is rejected.
    assert tuple(sfa.stride()) == (256, 1) and tuple(sfb.stride()) == (448, 1)
    with pytest.raises(ValueError):
        prepare_fp8_gemm_1d1d(
            a, b, sfa.expand(1, 256).as_strided((1, 256), (1, 1)), sfb, out
        )


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 130, 256])
def test_pack_ue8m0_words_pads_the_row_pitch(rows):
    _skip_unless_supported()
    scales = torch.randint(100, 140, (rows, 9), device="cuda", dtype=torch.uint8)
    packed = pack_ue8m0_words(scales, rows)
    assert tuple(packed.shape) == (3, rows)
    assert packed.stride(1) == 1 and packed.stride(0) % 4 == 0
    assert packed.stride(0) == (rows + 3) // 4 * 4 and packed.data_ptr() % 16 == 0
    dense = torch.nn.functional.pad(scales, (0, 3)).contiguous().view(torch.uint32)
    assert torch.equal(
        packed.contiguous().view(torch.int32),
        dense.transpose(0, 1).contiguous().view(torch.int32),
    )
