import pytest
import torch
import tvm_ffi

from flashinfer.tllm_utils import delay_kernel, get_trtllm_utils_module


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.gpu_exclusive,
]


def _elapsed_on_stream(stream, call):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        call()
        end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _reference_delay_ms(stream):
    module = get_trtllm_utils_module()
    with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
        module.delay_kernel(1)
        stream.synchronize()
        return _elapsed_on_stream(stream, lambda: module.delay_kernel(5000))


@pytest.mark.parametrize("default_stream", [False, True])
def test_delay_kernel_uses_torch_stream_and_restores_ffi_context(default_stream):
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    other = torch.cuda.Stream()
    device = tvm_ffi.device(str(stream.device))
    reference_ms = _reference_delay_ms(stream)
    try:
        with tvm_ffi.use_raw_stream(device, other.cuda_stream):
            elapsed_ms = _elapsed_on_stream(stream, lambda: delay_kernel(5000))
            assert tvm_ffi.get_raw_stream(device) == other.cuda_stream
        # Compare against the same native delay, not its approximate microsecond
        # argument. A launch on the other stream leaves this event window empty.
        assert elapsed_ms >= reference_ms / 2
    finally:
        torch.cuda.synchronize()


def test_delay_kernel_is_captured_on_torch_stream():
    stream = torch.cuda.Stream()
    other = torch.cuda.Stream()
    device = tvm_ffi.device(str(stream.device))
    reference_ms = _reference_delay_ms(stream)
    output = torch.zeros(1, device=stream.device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with tvm_ffi.use_raw_stream(device, other.cuda_stream):
        with torch.cuda.graph(graph, stream=stream):
            delay_kernel(5000)
            output.add_(1)
        assert tvm_ffi.get_raw_stream(device) == other.cuda_stream
    for value in (3.0, 9.0):
        with torch.cuda.stream(stream):
            output.fill_(value)
        elapsed_ms = _elapsed_on_stream(stream, graph.replay)
        assert elapsed_ms >= reference_ms / 2
        torch.testing.assert_close(output, torch.full_like(output, value + 1))
