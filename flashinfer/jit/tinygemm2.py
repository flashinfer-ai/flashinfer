import dataclasses
from typing import Optional

from .core import (
    JitSpec,
    JitSpecNvcc,
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


def _sm100_family_archs() -> list[tuple[int, str]]:
    archs = []
    for major, minor in sorted(current_compilation_context.TARGET_CUDA_ARCHS):
        if major != 10:
            continue
        min_cuda = _SM10X_MIN_CUDA.get(minor)
        if min_cuda is not None and not is_cuda_version_at_least(min_cuda):
            continue
        archs.append((major, minor))
    return archs


def tinygemm2_sm100_has_targets() -> bool:
    return bool(_sm100_family_archs())


@dataclasses.dataclass
class _PrebuiltOnlyJitSpec(JitSpecNvcc):
    """A spec that loads its prebuilt (AOT) module but never compiles.

    Used when the local nvcc can build none of the module's targets, for
    example an SM103 GPU with CUDA 12.8 (sm_103a needs 12.9): an installed
    jit-cache module still loads, while a miss raises instead of compiling
    with the global ``-gencode`` flags of the visible GPUs.
    """

    reason: str = ""

    def build(self, verbose: Optional[bool] = None, need_lock: bool = False) -> None:
        raise RuntimeError(self.reason)


def gen_tinygemm2_sm100_module() -> JitSpec:
    """Generate the JIT spec for the SM100/SM103 generated tinygemm2 kernel.

    ``csrc/tinygemm2_sm100.cu`` is a single translation unit holding one
    kernel template instantiated six ways (pipeline ring depth 4/8/16 x PDL
    off/on) plus its TVM-FFI binding, mirroring the incumbent
    ``csrc/tinygemm2.cu`` layout. The kernel is a generated port of the
    TensorRT-LLM tinygemm2 kernel with bit-identical outputs; the ring depth
    is selected inside the binding from the problem shape.

    The SM10x targets are the build targets the local nvcc supports. Without
    any, the spec only loads a prebuilt module (see ``_PrebuiltOnlyJitSpec``).
    """
    archs = _sm100_family_archs()
    spec = gen_jit_spec(
        "tinygemm2_sm100",
        [jit_env.FLASHINFER_CSRC_DIR / "tinygemm2_sm100.cu"],
        extra_cuda_cflags=[
            f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
            for major, minor in archs
        ]
        + common_nvcc_flags,
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    )
    if archs:
        return spec
    targets = sorted(current_compilation_context.TARGET_CUDA_ARCHS)
    return _PrebuiltOnlyJitSpec(
        **{field.name: getattr(spec, field.name) for field in dataclasses.fields(spec)},
        reason=(
            "tinygemm2_sm100 is not in the installed jit-cache and the local "
            f"CUDA toolkit cannot build it for the target architectures {targets} "
            "(sm_103a needs CUDA 12.9, sm_107a CUDA 13.4). Install a "
            "flashinfer-jit-cache that ships it, use a newer CUDA toolkit, or set "
            "FLASHINFER_DISABLE_TINYGEMM2_SM100=1."
        ),
    )
