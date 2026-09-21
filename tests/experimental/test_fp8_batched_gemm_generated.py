"""All projection routes: exact scale packing, changed inputs and stream replay."""
import pytest
import torch
from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm

_CASES = [(t, True, None) for t in (1, 4, 16, 128, 512, 4096)]
_CASES += [(t, False, alpha) for t in (4, 128) for alpha in (None, .5)]


@pytest.mark.parametrize('tokens,fp8,alpha', _CASES)
def test_projection_values_scales_current_stream_graph(tokens, fp8, alpha):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip('SM103a required')
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip('The exported schedules require 152 SMs')
    heads, inner, width = 8, 4096, 1024
    aq = torch.ones((tokens, heads, inner), dtype=torch.float8_e4m3fn, device='cuda')
    bq = torch.full((heads, width, inner), 1/64, dtype=torch.float8_e4m3fn, device='cuda')
    asf = torch.ones((tokens, heads, inner//128), dtype=torch.float32, device='cuda')
    bsf = torch.ones((heads, width//128, inner//128), dtype=torch.float32, device='cuda')
    plan = prepare_fp8_batched_gemm((aq, asf), (bq, bsf), output_fp8=fp8, alpha=alpha)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for sign, bvalue, exponent in ((1, 1/64, 125), (-1, 1/32, 126), (1, 1/64, 125)):
        with torch.cuda.stream(stream):
            aq.fill_(sign)
            bq.fill_(bvalue)
            plan.values.view(torch.uint8).fill_(255)
            if fp8:
                plan.scales.fill_(-1234567)
            graph.replay()
        stream.synchronize()
        if fp8:
            expected = torch.full_like(plan.values, sign*256)
            torch.testing.assert_close(plan.values.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)
            word = exponent * 0x01010101
            torch.testing.assert_close(plan.scales, torch.full_like(plan.scales, word), atol=0, rtol=0)
        else:
            expected = sign * inner * bvalue * (1 if alpha is None else alpha)
            torch.testing.assert_close(plan.values, torch.full_like(plan.values, expected), atol=.01, rtol=.01)
    # The alpha value is a launch argument, independent of its compiled route.
    if alpha is not None:
        other = prepare_fp8_batched_gemm((aq, asf), (bq, bsf), output_fp8=False, alpha=-.25)
        other.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(other.values, torch.full_like(other.values, -16), atol=.01, rtol=.01)
