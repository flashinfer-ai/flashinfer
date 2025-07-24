import functools

from .jit import env as jit_env
from .jit import gen_jit_spec


def get_trtllm_utils_spec():
    return gen_jit_spec(
        "trtllm_utils",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/delayStream.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        ],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "cutlass_extensions"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels",
        ],
    )


@functools.cache
def get_trtllm_utils_module():
    return get_trtllm_utils_spec().build_and_load()


def delay_kernel(stream_delay_micro_secs):
    get_trtllm_utils_module().delay_kernel(stream_delay_micro_secs)
