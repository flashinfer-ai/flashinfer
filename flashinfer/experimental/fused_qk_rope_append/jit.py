"""JIT loader for fused QK RMSNorm/RoPE/paged-append kernels."""

from __future__ import annotations

import functools
from pathlib import Path

from ...jit import current_compilation_context
from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec


def _flashinfer_csrc_path() -> Path:
    if (jit_env.FLASHINFER_CSRC_DIR / "tvm_ffi_utils.h").is_file():
        return jit_env.FLASHINFER_CSRC_DIR
    checkout = Path(__file__).resolve().parents[3] / "csrc"
    if (checkout / "tvm_ffi_utils.h").is_file():
        return checkout
    raise FileNotFoundError("FlashInfer TVM-FFI JIT headers were not found")


@functools.cache
def gen_fused_qk_rope_append_module() -> JitSpec:
    source_dir = Path(__file__).resolve().parent / "csrc"
    sources = [source_dir / "hpc_rope.cu", source_dir / "hpc_rope_jit_binding.cu"]
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError("fused QK/RoPE CUDA sources were not found")
    flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10]
    )
    return gen_jit_spec(
        "fused_qk_rope_append",
        sources,
        extra_cuda_cflags=["-lineinfo", *flags],
        extra_include_paths=[source_dir / "include", _flashinfer_csrc_path()],
    )


@functools.cache
def load_fused_qk_rope_append_module():
    return gen_fused_qk_rope_append_module().build_and_load()


__all__ = ["gen_fused_qk_rope_append_module", "load_fused_qk_rope_append_module"]
