import functools

import tvm_ffi

from .jit.tllm_utils import gen_trtllm_utils_module


@functools.cache
def get_trtllm_utils_module():
    return gen_trtllm_utils_module().build_and_load()


def delay_kernel(stream_delay_micro_secs: int):
    """Enqueue a delay on the current PyTorch CUDA stream."""
    # This scalar-only FFI call cannot infer Torch's stream from tensor arguments.
    # Bind it explicitly so the delay precedes the autotuner's timing events.
    with tvm_ffi.use_torch_stream():
        get_trtllm_utils_module().delay_kernel(stream_delay_micro_secs)
