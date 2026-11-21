"""JIT spec for the SM103 fp8 paged-MQA top-k sparse-prefill indexer."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags


# Thin TVM-FFI binding/launch TU. The framework-independent device code it
# wraps lives under include/flashinfer/attention/fp8_paged_mqa_topk_indexer_sm103/
# (device.cuh, device_legacy.cuh), which is on the default JIT include path.
def gen_fp8_paged_mqa_topk_indexer_module() -> JitSpec:
    return gen_jit_spec(
        "fp8_paged_mqa_topk_indexer_sm103",
        [jit_env.FLASHINFER_CSRC_DIR / "fp8_paged_mqa_topk_indexer_sm103.cu"],
        extra_cuda_cflags=sm103a_nvcc_flags + ["--use_fast_math", "-lineinfo"],
    )
