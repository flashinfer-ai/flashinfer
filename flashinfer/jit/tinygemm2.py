from .core import (
    JitSpec,
    gen_jit_spec,
    current_compilation_context,
    common_nvcc_flags,
)
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


_SM10X_MIN_CUDA = {"3a": "12.9", "7a": "13.4"}


def _sm100_family_nvcc_flags() -> list[str]:
    flags = []
    for major, minor in sorted(current_compilation_context.TARGET_CUDA_ARCHS):
        if major != 10:
            continue
        min_cuda = _SM10X_MIN_CUDA.get(minor)
        if min_cuda is not None and not is_cuda_version_at_least(min_cuda):
            continue
        flags.append(f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}")
    if not flags:
        raise RuntimeError("No supported SM10x CUDA architecture for tinygemm2_sm100.")
    return flags + common_nvcc_flags


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
        extra_cuda_cflags=_sm100_family_nvcc_flags(),
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    )
