"""Projection routes: exact scale packing, changed inputs, stream replay, token coverage."""

import pytest
import torch
from flashinfer.experimental.deepgemm_batched_gemm import batched_gemm as _runtime
from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm

# The ten originally exported routes.
_CASES = [(t, True, None) for t in (1, 4, 16, 128, 512, 4096)]
_CASES += [(t, False, alpha) for t in (4, 128) for alpha in (None, 0.5)]
# Token counts outside the original route table that select an exported
# schedule (swap_ab BM16 with one and two M tiles, swap_ab BM64, n256 with
# four and eight M tiles, the general alpha schedule).
_CASES += [(t, True, None) for t in (2, 8, 17, 32, 100, 127, 500, 1000)]
_CASES += [(t, False, 0.5) for t in (1, 17, 1000)]
# Token counts whose selected schedule is not exported.
_UNEXPORTED = [(64, True, None), (8, False, None), (1000, False, None)]


def _skip_unless_exported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        arch = _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms not in _runtime.supported_num_sms(arch):
        pytest.skip(
            f"The exported {arch} schedules were selected on {_runtime.supported_num_sms(arch)} SMs, "
            f"this device has {sms}"
        )


def _operands(tokens, heads=8, inner=4096, width=1024):
    aq = torch.ones((tokens, heads, inner), dtype=torch.float8_e4m3fn, device="cuda")
    bq = torch.full(
        (heads, width, inner), 1 / 64, dtype=torch.float8_e4m3fn, device="cuda"
    )
    asf = torch.ones((tokens, heads, inner // 128), dtype=torch.float32, device="cuda")
    bsf = torch.ones(
        (heads, width // 128, inner // 128), dtype=torch.float32, device="cuda"
    )
    return aq, bq, asf, bsf


@pytest.mark.parametrize("tokens,fp8,alpha", _CASES)
def test_projection_values_scales_current_stream_graph(tokens, fp8, alpha):
    _skip_unless_exported()
    inner = 4096
    aq, bq, asf, bsf = _operands(tokens)
    plan = prepare_fp8_batched_gemm((aq, asf), (bq, bsf), output_fp8=fp8, alpha=alpha)
    expected_route = _runtime.route_config(
        tokens,
        8,
        inner,
        1024,
        torch.cuda.get_device_properties(0).multi_processor_count,
        "fp8" if fp8 else "alpha" if alpha is not None else "bf16",
    )
    assert plan.config == expected_route
    assert plan.program == _runtime.ROUTES[_runtime.schedule_key(expected_route)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for sign, bvalue, exponent in (
        (1, 1 / 64, 125),
        (-1, 1 / 32, 126),
        (1, 1 / 64, 125),
    ):
        with torch.cuda.stream(stream):
            aq.fill_(sign)
            bq.fill_(bvalue)
            plan.values.view(torch.uint8).fill_(255)
            if fp8:
                plan.scales.fill_(-1234567)
            graph.replay()
        stream.synchronize()
        if fp8:
            expected = torch.full_like(plan.values, sign * 256)
            torch.testing.assert_close(
                plan.values.view(torch.uint8),
                expected.view(torch.uint8),
                atol=0,
                rtol=0,
            )
            word = exponent * 0x01010101
            torch.testing.assert_close(
                plan.scales, torch.full_like(plan.scales, word), atol=0, rtol=0
            )
        else:
            expected = sign * inner * bvalue * (1 if alpha is None else alpha)
            torch.testing.assert_close(
                plan.values,
                torch.full_like(plan.values, expected),
                atol=0.01,
                rtol=0.01,
            )
    # The alpha value is a launch argument, independent of its compiled route.
    if alpha is not None:
        other = prepare_fp8_batched_gemm(
            (aq, asf), (bq, bsf), output_fp8=False, alpha=-0.25
        )
        other.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            other.values, torch.full_like(other.values, -16), atol=0.01, rtol=0.01
        )


@pytest.mark.parametrize("tokens,fp8,alpha", _UNEXPORTED)
def test_unexported_schedule_raises_at_preparation(tokens, fp8, alpha):
    _skip_unless_exported()
    aq, bq, asf, bsf = _operands(tokens)
    with pytest.raises(
        NotImplementedError, match="general schedule .* is not exported"
    ):
        prepare_fp8_batched_gemm((aq, asf), (bq, bsf), output_fp8=fp8, alpha=alpha)


def test_route_table_is_a_function_of_tokens_sms_and_epilogue():
    """Every original route key maps onto the program its token count selects."""
    routes = {
        (1, "fp8"): "swap_ab_bm16_bn128_s12_fp8",
        (4, "fp8"): "swap_ab_bm16_bn128_s12_fp8",
        (16, "fp8"): "swap_ab_bm16_bn128_s12_fp8",
        (128, "fp8"): "swap_ab_bm64_bn128_s10_fp8",
        (512, "fp8"): "n256_bm128_bn256_s5_fp8",
        (4096, "fp8"): "n256_bm128_bn256_s5_fp8",
        (4, "bf16"): "bf16_t4_bm16_bn128_s12_bf16",
        (128, "bf16"): "bf16_t128_bm64_bn128_s10_bf16",
        (4, "alpha"): "general_bm128_bn128_s5_alpha",
        (128, "alpha"): "general_bm128_bn128_s5_alpha",
    }
    for sms in _runtime.PINNED_NUM_SMS:
        for (tokens, epilogue), key in routes.items():
            config = _runtime.route_config(tokens, 8, 4096, 1024, sms, epilogue)
            assert _runtime.schedule_key(config) == key
            assert key in _runtime.ROUTES
    assert set(_runtime.ROUTES) == set(routes.values())
    assert set(_runtime.ROUTES.values()) <= set(_runtime.PROGRAMS)
