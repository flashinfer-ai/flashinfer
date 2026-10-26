from .core import JitSpec, gen_jit_spec, current_compilation_context, sm100a_nvcc_flags
from .cpp_ext import is_cuda_version_at_least
from . import env as jit_env


def gen_tinygemm2_module() -> JitSpec:
    """Generate JIT spec for tinygemm2 kernel (SM90+ BF16 small GEMM with bias)."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12],
        map_sm107_to_100f=True,
    )
    return gen_jit_spec(
        "tinygemm2",
        [jit_env.FLASHINFER_CSRC_DIR / "tinygemm2.cu"],
        extra_cuda_cflags=nvcc_flags,
    )


def gen_tinygemm2_sm100_module() -> JitSpec:
    """Generate the JIT spec for the SM100/SM103 generated tinygemm2 kernel.

    ``csrc/tinygemm2_sm100.cu`` is a single translation unit holding one
    kernel template instantiated six ways (pipeline ring depth 4/8/16 x PDL
    off/on) plus its TVM-FFI binding, mirroring the incumbent
    ``csrc/tinygemm2.cu`` layout. The kernel is a generated port of the
    TensorRT-LLM tinygemm2 kernel with bit-identical outputs; the ring depth
    is selected inside the binding from the problem shape.
    """
    return gen_jit_spec(
        "tinygemm2_sm100",
        [jit_env.FLASHINFER_CSRC_DIR / "tinygemm2_sm100.cu"],
        extra_cuda_cflags=sm100a_nvcc_flags
        + ["-gencode=arch=compute_103a,code=sm_103a"]
        + (
            ["-gencode=arch=compute_107a,code=sm_107a"]
            if is_cuda_version_at_least("13.4")
            else []
        ),
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    )
