"""JIT loader for the adaptive sparse block-mask CUDA kernel."""

from __future__ import annotations

import functools
from pathlib import Path

from ...jit import current_compilation_context
from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec


def _flashinfer_include_paths() -> list[str | Path]:
    installed_csrc = jit_env.FLASHINFER_CSRC_DIR
    if (installed_csrc / "tvm_ffi_utils.h").is_file():
        return [installed_csrc]

    checkout = Path(__file__).resolve().parents[3]
    checkout_csrc = checkout / "csrc"
    if (checkout_csrc / "tvm_ffi_utils.h").is_file():
        return [checkout_csrc]
    raise FileNotFoundError("FlashInfer TVM-FFI JIT headers were not found")


@functools.cache
def gen_adaptive_sparse_block_mask_module() -> JitSpec:
    source = Path(__file__).resolve().parent / "csrc" / "adaptive_sparse_block_mask.cu"
    if not source.is_file():
        raise FileNotFoundError(
            "adaptive sparse block-mask CUDA source was not found in the package"
        )
    flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]
    )
    return gen_jit_spec(
        "adaptive_sparse_block_mask",
        [source],
        extra_cuda_cflags=["-lineinfo", *flags],
        extra_include_paths=_flashinfer_include_paths(),
    )


@functools.cache
def load_adaptive_sparse_block_mask_module():
    return gen_adaptive_sparse_block_mask_module().build_and_load()


__all__ = [
    "gen_adaptive_sparse_block_mask_module",
    "load_adaptive_sparse_block_mask_module",
]
