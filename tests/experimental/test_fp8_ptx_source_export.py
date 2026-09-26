"""Exercise the prepared FP8 1D1D PTX source export without the source compiler installed."""

import pytest
import torch
from flashinfer.experimental.deepgemm_fp8_gemm import prepare_fp8_gemm_1d1d
from flashinfer.experimental.deepgemm_fp8_gemm import runtime as _runtime


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
            f"The exported {arch} programs cover {_runtime.supported_num_sms(arch)} SMs, "
            f"this device has {sms}"
        )
    return arch


@pytest.mark.parametrize("accumulate", [False, True])
def test_fp8_prepared_output_and_accumulator(tmp_path, accumulate):
    _skip_unless_exported()
    a = (
        torch.ones((4096, 4096), device="cuda", dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    b = (
        torch.ones((7168, 4096), device="cuda", dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    sfa = torch.full((8, 4096), 0x7F7F7F7F, device="cuda", dtype=torch.uint32)
    sfb = torch.full((8, 7168), 0x7F7F7F7F, device="cuda", dtype=torch.uint32)
    out = torch.full(
        (4096, 7168),
        0.25 if accumulate else 0,
        device="cuda",
        dtype=torch.float32 if accumulate else torch.bfloat16,
    )
    address = out.data_ptr()
    plan = prepare_fp8_gemm_1d1d(
        a, b, sfa, sfb, out, accumulate=accumulate, cache_dir=tmp_path
    )
    plan.run()
    torch.testing.assert_close(
        out, torch.full_like(out, 4096.25 if accumulate else 4096), atol=0, rtol=0
    )
    assert out.data_ptr() == address
    if accumulate:
        plan.run()
        torch.testing.assert_close(out, torch.full_like(out, 8192.25), atol=0, rtol=0)
    # The prepared submission replays on a non-default stream and inside a CUDA graph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out.fill_(0.25 if accumulate else 0)
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
        out.fill_(0.25 if accumulate else 0)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(
        out, torch.full_like(out, 4096.25 if accumulate else 4096), atol=0, rtol=0
    )
