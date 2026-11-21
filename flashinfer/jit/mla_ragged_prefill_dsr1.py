"""JIT spec for the SM103 DeepSeek-R1-geometry ragged MLA prefill fast
path (fp8 e4m3 q/k/v, bf16 out)."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags


def gen_mla_ragged_prefill_dsr1_module() -> JitSpec:
    # The framework-independent device code lives under
    # include/flashinfer/attention/mla_ragged_prefill_dsr1_sm103/ (on the
    # default JIT include path); the csrc TU below is the binding/launch TU.
    return gen_jit_spec(
        "mla_ragged_prefill_dsr1_sm103",
        [jit_env.FLASHINFER_CSRC_DIR / "mla_ragged_prefill_dsr1_sm103.cu"],
        extra_cuda_cflags=sm103a_nvcc_flags + ["--use_fast_math", "-lineinfo"],
        # cuTensorMapEncodeTiled (TMA descriptor encoding) lives in the
        # driver library.
        extra_ldflags=["-lcuda"],
    )
