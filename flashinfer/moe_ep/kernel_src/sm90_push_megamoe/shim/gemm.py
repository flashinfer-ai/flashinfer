"""Private grouped FP8 GEMM module for the SM90 push MegaMoE backend."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .....jit import env as jit_env
from .....jit.core import JitSpec, gen_jit_spec, sm90a_nvcc_flags
from .....jit.cpp_ext import is_cuda_version_at_least

__all__ = [
    "create_sm90_push_fp8_moe_gemm_runner",
    "gen_sm90_push_fp8_moe_gemm_module",
    "sm90_push_fp8_moe_gemm_uri",
]

_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "fp8_gemm"
_SOURCE_TREE_CSRC_DIR = Path(__file__).resolve().parents[5] / "csrc"
_SOURCE_NAMES = (
    "fp8_moe_binding.cu",
    "fp8_moe_fc1_fused.cuh",
    "fp8_moe_jit.cuh",
    "fp8_moe_launcher.cuh",
    "fp8_moe_scheduler.cuh",
)
_DEPENDENCY_NAMES = (
    "tvm_ffi_utils.h",
    "nv_internal/cpp/common/envUtils.cpp",
    "nv_internal/cpp/common/logger.cpp",
    "nv_internal/cpp/common/memoryUtils.cu",
    "nv_internal/cpp/common/stringUtils.cpp",
    "nv_internal/cpp/common/tllmException.cpp",
    "nv_internal/include/tensorrt_llm/common/assert.h",
    "nv_internal/include/tensorrt_llm/common/cudaBf16Wrapper.h",
    "nv_internal/include/tensorrt_llm/common/cudaFp8Utils.h",
    "nv_internal/include/tensorrt_llm/common/cudaUtils.h",
    "nv_internal/include/tensorrt_llm/common/logger.h",
    "nv_internal/include/tensorrt_llm/common/stringUtils.h",
    "nv_internal/include/tensorrt_llm/common/tllmException.h",
    "nv_internal/tensorrt_llm/common/cudaDriverWrapper.h",
    "nv_internal/tensorrt_llm/common/cudaTypeUtils.cuh",
    "nv_internal/tensorrt_llm/common/envUtils.h",
    "nv_internal/tensorrt_llm/common/memoryUtils.h",
    "nv_internal/tensorrt_llm/deep_gemm/compiler.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/fp8_gemm.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/fp8_gemm_impl.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/jit_utils.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/nvrtc_cutlass.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/nvrtc_std.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/runtime.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/scheduler.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/tma_utils.cuh",
    "nv_internal/tensorrt_llm/deep_gemm/utils.cuh",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/"
    "fp8_blockscale_gemm_kernel.cuh",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/"
    "fp8_blockscale_mma_utils.cuh",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/"
    "fp8_blockscale_tma_utils.cuh",
)


def _canonical_source(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _source_blobs() -> dict[str, bytes]:
    return {name: _canonical_source(_SOURCE_DIR / name) for name in _SOURCE_NAMES}


def _csrc_dir() -> Path:
    if jit_env.FLASHINFER_CSRC_DIR.is_dir():
        return jit_env.FLASHINFER_CSRC_DIR
    return _SOURCE_TREE_CSRC_DIR


def _dependency_blobs() -> dict[str, bytes]:
    csrc = _csrc_dir()
    return {name: _canonical_source(csrc / name) for name in _DEPENDENCY_NAMES}


def _module_flags() -> tuple[str, ...]:
    return tuple(sm90a_nvcc_flags) + (
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        *(("-DENABLE_FP8_BLOCK_SCALE",) if is_cuda_version_at_least("12.8") else ()),
        "-DCUTLASS_ENABLE_GDC_FOR_SM90=1",
    )


def _digest(
    sources: dict[str, bytes],
    dependencies: dict[str, bytes],
    flags: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    for name in _SOURCE_NAMES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(sources[name])
        digest.update(b"\0")
    for name in _DEPENDENCY_NAMES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(dependencies[name])
        digest.update(b"\0")
    digest.update(json.dumps(flags, separators=(",", ":")).encode())
    return digest.hexdigest()


def _snapshot_matches(path: Path, content: bytes) -> bool:
    try:
        return path.read_bytes() == content
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return False


def _write_snapshot_atomic(path: Path, content: bytes) -> None:
    if _snapshot_matches(path, content):
        return
    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "wb") as snapshot:
            snapshot.write(content)
        os.replace(temp_path, path)
        temp_path = None
    except PermissionError:
        if not _snapshot_matches(path, content):
            raise
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()


def sm90_push_fp8_moe_gemm_uri(
    cuda_flags: tuple[str, ...] | None = None,
) -> str:
    """Return the content-addressed module name for the private GEMM sources."""
    flags = _module_flags() if cuda_flags is None else cuda_flags
    source_digest = _digest(_source_blobs(), _dependency_blobs(), flags)
    return f"sm90_push_fp8_moe_gemm_{source_digest[:20]}"


def gen_sm90_push_fp8_moe_gemm_module() -> JitSpec:
    """Snapshot the private GEMM sources and return their SM90 JIT spec."""
    if not is_cuda_version_at_least("12.8"):
        raise RuntimeError(
            "SM90 push FP8 MoE GEMM requires CUDA Toolkit 12.8 or newer."
        )
    flags = _module_flags()
    sources = _source_blobs()
    source_digest = _digest(sources, _dependency_blobs(), flags)
    uri = f"sm90_push_fp8_moe_gemm_{source_digest[:20]}"
    output_dir = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    for name, content in sources.items():
        _write_snapshot_atomic(output_dir / name, content)
    build_config = (
        "#pragma once\n\n"
        f'#define FLASHINFER_SM90_PUSH_FP8_MOE_SOURCE_DIGEST "{source_digest[:20]}"\n'
    ).encode()
    _write_snapshot_atomic(output_dir / "fp8_moe_build_config.h", build_config)

    csrc = _csrc_dir()
    return gen_jit_spec(
        uri,
        [
            output_dir / "fp8_moe_binding.cu",
            csrc / "nv_internal/cpp/common/envUtils.cpp",
            csrc / "nv_internal/cpp/common/logger.cpp",
            csrc / "nv_internal/cpp/common/stringUtils.cpp",
            csrc / "nv_internal/cpp/common/tllmException.cpp",
            csrc / "nv_internal/cpp/common/memoryUtils.cu",
        ],
        extra_cuda_cflags=list(flags),
        extra_ldflags=["-lnvrtc", "-lcuda"],
        extra_include_paths=[
            output_dir,
            csrc,
            csrc / "nv_internal",
            csrc / "nv_internal" / "include",
            csrc / "nv_internal" / "tensorrt_llm",
            csrc / "nv_internal" / "tensorrt_llm" / "cutlass_extensions" / "include",
            csrc
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels"
            / "include",
            csrc / "nv_internal" / "tensorrt_llm" / "kernels" / "cutlass_kernels",
        ],
    )


@functools.cache
def _load_sm90_push_fp8_moe_gemm_module_cached(uri: str):
    spec = gen_sm90_push_fp8_moe_gemm_module()
    if spec.name != uri:
        raise RuntimeError(
            "SM90 push FP8 MoE sources changed while constructing the JIT module"
        )
    module = spec.build_and_load()
    generated_source_dir = jit_env.FLASHINFER_GEN_SRC_DIR / spec.name
    deepgemm_source_dir = _csrc_dir() / "nv_internal" / "tensorrt_llm"
    module.set_deepgemm_jit_include_dirs(
        [str(generated_source_dir), str(deepgemm_source_dir)]
    )
    return module


def _load_sm90_push_fp8_moe_gemm_module():
    return _load_sm90_push_fp8_moe_gemm_module_cached(sm90_push_fp8_moe_gemm_uri())


def create_sm90_push_fp8_moe_gemm_runner():
    """Return a new stateful private FP8 MoE GEMM runner."""
    return _load_sm90_push_fp8_moe_gemm_module().init_moe()
