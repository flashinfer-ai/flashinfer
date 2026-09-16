# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Single-GPU regression tests for fused SM90 MXFP4 safe quantization.

Run in an isolated Hopper pytest process: the SM90 and SM100 vendor trees use
colliding module names. References below use only test-owned Torch arithmetic;
the raw epilogue import is exclusively the device function under test.
"""

import pytest
import torch


pytestmark = pytest.mark.arch_hopper


@pytest.fixture(scope="module")
def quantizers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from flashinfer.utils import get_compute_capability, is_sm90a_supported

    device = torch.device("cuda", torch.cuda.current_device())
    if get_compute_capability(device) != (9, 0) or not is_sm90a_supported(device):
        pytest.skip("an SM90a Hopper GPU is required")
    pytest.importorskip("cutlass.cute")

    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
        safe_quantize,
        staging,
    )

    return safe_quantize, staging, device


def _legacy_quantize(x):
    fp32 = x.float()
    d = (fp32.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1.0e-30)
    return (fp32 / d).to(torch.float8_e4m3fn), d


def _assert_fp32_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(
        actual.contiguous().view(torch.int32),
        expected.contiguous().view(torch.int32),
        rtol=0,
        atol=0,
    )


def _assert_quantized_bits(actual, expected):
    assert actual[0].dtype == expected[0].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        actual[0].contiguous().view(torch.uint8),
        expected[0].contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )
    _assert_fp32_bits(actual[1], expected[1])


def _boundary_amaxes():
    """CPU FP32 values on both sides of each arithmetic-path boundary."""
    finfo = torch.finfo(torch.float32)
    thresholds = torch.tensor(
        [448.0e-30, 448.0 / finfo.max, finfo.tiny], dtype=torch.float32
    )
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.float32),
            torch.nextafter(torch.zeros(1), torch.ones(1)),
            torch.nextafter(thresholds, torch.zeros_like(thresholds)),
            thresholds,
            torch.nextafter(thresholds, torch.full_like(thresholds, float("inf"))),
            torch.tensor([1.0e-39, 1.0e-35, 1.0e-30, 1.0e-28, 1.0e-20, 1.0, 448.0]),
        )
    )


def _tiny_scale_reference(amax):
    """Independently round both scale boundaries using CPU FP64 arithmetic.

    Round the denominator bound to FP32 *before* dividing, then round q before
    computing d. In particular, computing d as amax/448 is not this contract.
    """
    assert amax.device.type == "cpu" and amax.dtype == torch.float32
    finfo = torch.finfo(torch.float32)
    bound = torch.tensor(448.0 / finfo.max, dtype=torch.float32).double()
    q = (448.0 / amax.double().clamp_min(bound)).float().clamp_max(finfo.max)
    d = (1.0 / q.double()).float()
    tiny = (amax > 0) & (amax < torch.tensor(448.0e-30, dtype=torch.float32))
    return tiny, q, d


@pytest.mark.parametrize("hidden", [256, 7168])
def test_normal_rows_are_byte_exact_and_legacy_is_default(quantizers, hidden):
    safe_quantize, staging, device = quantizers
    generator = torch.Generator(device=device).manual_seed(501)
    x = torch.randn((65, hidden), device=device, generator=generator).bfloat16()
    x[0].zero_()
    expected = _legacy_quantize(x)
    _assert_quantized_bits(
        staging._quantize_e4m3_per_token_full_hidden(x, safe_quantization=True),
        expected,
    )
    fp32 = x.float()
    _assert_quantized_bits(
        safe_quantize.quantize_from_amax(fp32, fp32.abs().amax(1, keepdim=True)),
        expected,
    )

    tiny = (fp32 * 1.0e-35).bfloat16()
    _assert_quantized_bits(
        staging._quantize_e4m3_per_token_full_hidden(tiny), _legacy_quantize(tiny)
    )
    _assert_quantized_bits(
        staging._quantize_e4m3_per_token_full_hidden(tiny, safe_quantization=False),
        _legacy_quantize(tiny),
    )


def test_input_quantizer_fp32_boundaries(quantizers):
    safe_quantize, _, device = quantizers
    x_cpu = _boundary_amaxes()[:, None] * torch.linspace(-1.0, 1.0, 256)
    x = x_cpu.to(device)
    amax = x.abs().amax(1, keepdim=True)
    actual_payload, actual_d = safe_quantize.quantize_from_amax(x, amax)
    old_payload, old_d = _legacy_quantize(x)
    tiny, q_ref, d_ref = _tiny_scale_reference(x_cpu.abs().amax(1, keepdim=True))
    scaled_ref = (x_cpu.double() * q_ref.double()).float()
    safe_payload = scaled_ref.to(torch.float8_e4m3fn)
    expected_bytes = torch.where(
        tiny, safe_payload.view(torch.uint8), old_payload.cpu().view(torch.uint8)
    )
    expected_d = torch.where(tiny, d_ref, old_d.cpu())
    torch.testing.assert_close(
        actual_payload.cpu().view(torch.uint8), expected_bytes, rtol=0, atol=0
    )
    _assert_fp32_bits(actual_d.cpu(), expected_d)
    assert torch.isfinite(actual_payload.float()).all()
    assert torch.isfinite(actual_d).all() and (actual_d > 0).all()
    # The smallest FP32 inputs can legitimately become FP8 zero. This larger
    # tiny row must distinguish the fix from the legacy all-zero result.
    witness = torch.full((1, 256), 1.0e-35, dtype=torch.float32, device=device)
    new_payload, _ = safe_quantize.quantize_from_amax(
        witness, witness.abs().amax(1, keepdim=True)
    )
    assert torch.count_nonzero(new_payload.float()) == witness.numel()
    assert torch.count_nonzero(_legacy_quantize(witness)[0].float()) == 0


@pytest.mark.parametrize("use_default_stream", [True, False])
def test_current_stream_and_graph_replay(quantizers, use_default_stream):
    safe_quantize, _, device = quantizers
    caller_stream = torch.cuda.current_stream(device)
    stream = (
        torch.cuda.default_stream(device)
        if use_default_stream
        else torch.cuda.Stream(device=device)
    )
    generator = torch.Generator(device=device).manual_seed(502)
    x = torch.randn((17, 256), device=device, generator=generator)
    stream.wait_stream(caller_stream)
    with torch.cuda.stream(stream):
        amax = x.abs().amax(1, keepdim=True)
        _assert_quantized_bits(
            safe_quantize.quantize_from_amax(x, amax), _legacy_quantize(x)
        )
    # CUDA Graph capture itself requires a non-default stream. Test eager
    # current-stream inheritance above for both streams, then replay from it.
    capture_stream = (
        stream if not use_default_stream else torch.cuda.Stream(device=device)
    )
    capture_stream.wait_stream(stream)
    with torch.cuda.stream(capture_stream):
        # Compile/cache population must happen outside capture.
        safe_quantize.quantize_from_amax(x, x.abs().amax(1, keepdim=True))
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        amax = x.abs().amax(1, keepdim=True)
        payload, d = safe_quantize.quantize_from_amax(x, amax)
    stream.wait_stream(capture_stream)
    with torch.cuda.stream(stream):
        for factor in (0.5, 1.25):
            x.mul_(factor)
            graph.replay()
            _assert_quantized_bits((payload, d), _legacy_quantize(x))
    caller_stream.wait_stream(stream)
    stream.synchronize()


def test_empty_batch_does_not_compile_or_launch(quantizers, monkeypatch):
    safe_quantize, staging, device = quantizers

    def unexpected_compile(*args, **kwargs):
        pytest.fail("empty quantization must not enter the compiled launch path")

    monkeypatch.setattr(safe_quantize, "_compiled_quantizer", unexpected_compile)
    x = torch.empty((0, 256), device=device, dtype=torch.float32)
    amax = torch.empty((0, 1), device=device, dtype=torch.float32)
    pairs = (
        safe_quantize.quantize_from_amax(x, amax),
        staging._quantize_e4m3_per_token_full_hidden(x, safe_quantization=True),
    )
    for payload, d in pairs:
        assert payload.shape == x.shape and payload.dtype == torch.float8_e4m3fn
        assert d.shape == (0, 1) and d.dtype == torch.float32
        assert payload.device == d.device == device


def test_epilogue_q_and_d_are_independently_correct(quantizers):
    _, _, device = quantizers
    import cutlass
    import cutlass.cute as cute

    try:
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
            bootstrap_paths,
        )

        bootstrap_paths()
    except RuntimeError as error:
        pytest.skip(f"SM90 kernel tree unavailable in this process: {error}")
    from moe_hopper_fp8.epilogue_fp8_swapab import _mxfp4_safe_scale_pair

    @cute.kernel
    def scale_pair_probe(inputs: cute.Tensor, output: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        if tid < cute.size(inputs):
            a = inputs[tid]
            old_d = a * cutlass.Float32(1.0 / 448.0)
            if old_d < cutlass.Float32(1.0e-30):
                old_d = cutlass.Float32(1.0e-30)
            old_q = cutlass.Float32(1.0) / old_d
            d, q = _mxfp4_safe_scale_pair(a, old_d)
            output[tid, 0] = d
            output[tid, 1] = q
            output[tid, 2] = old_d
            output[tid, 3] = old_q

    @cute.jit
    def launch(inputs: cute.Tensor, output: cute.Tensor, stream):
        scale_pair_probe(inputs, output).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream
        )

    a_cpu = _boundary_amaxes()
    assert a_cpu.numel() <= 128
    inputs = a_cpu.to(device)
    output = torch.empty((a_cpu.numel(), 4), device=device, dtype=torch.float32)
    fake_inputs = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, inputs.shape, assumed_align=4
    )
    fake_output = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, output.shape, stride_order=(1, 0), assumed_align=4
    )
    compiled = cute.compile(
        launch,
        fake_inputs,
        fake_output,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    compiled(inputs, output)
    result = output.cpu()
    tiny, q_ref, d_ref = _tiny_scale_reference(a_cpu)
    _assert_fp32_bits(result[tiny, 1], q_ref[tiny])
    _assert_fp32_bits(result[tiny, 0], d_ref[tiny])
    _assert_fp32_bits(result[~tiny, 0], result[~tiny, 2])
    _assert_fp32_bits(result[~tiny, 1], result[~tiny, 3])
    assert torch.isfinite(result).all() and (result > 0).all()
    # An explicit known pair makes an incorrect reciprocal recomputation or
    # flush-to-zero visible without relying only on the reference formula.
    min_subnormal_row = 1
    assert result[min_subnormal_row, 1].item() == torch.finfo(torch.float32).max
    assert result[min_subnormal_row, 0].item() == 2.0**-128
