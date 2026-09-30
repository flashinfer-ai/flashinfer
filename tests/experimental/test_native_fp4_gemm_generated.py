"""Analytical E2M1/UE8M0 inputs, runtime alpha, current-stream graph replay."""

import pytest
import torch
from flashinfer.experimental.deepgemm_fp4_gemm import fp4_gemm as _runtime
from flashinfer.fp4_gemm import prepare_fp4_gemm

_MODEL = [
    (m, n, k, None)
    for m in (16, 128, 512, 4096)
    for n, k in ((4608, 5120), (5120, 2304))
]
_CASES = [
    (m, n, k, stages, alpha) for m, n, k, stages in _MODEL for alpha in (1.0, -0.75)
]
_CASES += [
    (256, 128, 2048, 7, 0.5),
    (256, 128, 256, None, 1.0),
    (256, 128, 256, None, -0.75),
]


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
            f"The exported {arch} schedules cover {_runtime.supported_num_sms(arch)} SMs, "
            f"this device has {sms}"
        )


@pytest.mark.parametrize("m,n,k,num_stages,alpha", _CASES)
def test_fp4_values_alpha_stream_replay(m, n, k, num_stages, alpha):
    _skip_unless_exported()
    a = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    sfa = torch.full((k // 128, m), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
    sfb = torch.full((k // 128, n), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
    plan = prepare_fp4_gemm(a, b, sfa, sfb, m=m, alpha=alpha, num_stages=num_stages)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for negative in (False, True, False):
        with torch.cuda.stream(stream):
            a.fill_(0xAA if negative else 0x22)
            plan.storage.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        expected = torch.full_like(plan.output, (-1 if negative else 1) * k * alpha)
        torch.testing.assert_close(plan.output, expected, atol=0, rtol=0)
