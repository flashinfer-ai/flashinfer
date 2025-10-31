import functools

from .jit.tllm_utils import gen_trtllm_utils_module


@functools.cache
def get_trtllm_utils_module():
    return gen_trtllm_utils_module().build_and_load()


def delay_kernel(stream_delay_micro_secs):
    get_trtllm_utils_module().delay_kernel(stream_delay_micro_secs)
