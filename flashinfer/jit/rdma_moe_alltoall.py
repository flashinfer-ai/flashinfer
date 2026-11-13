# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Independent JIT modules for quantization and NCCL GIN RDMA.

Importing this module does not discover NCCL or compile CUDA code.
Standalone quantization needs neither NCCL headers nor an RDMA provider.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
import re

# This is a validated-version policy, not the first NCCL release with GIN.
# Broader admission requires checking source-completion semantics and running
# the transport tests against matching headers and runtime.
_VERIFIED_NCCL_VERSION = (2, 30, 7)


def _nccl_identity(include: Path, library: Path) -> str:
    """Hash external dependencies that FlashInfer's source hash cannot see."""
    digest = hashlib.sha256(f"{include}\n{library}\n".encode())
    headers: list[Path] = []
    visited: set[Path] = set()

    def visit(directory: Path) -> None:
        # SDKs may symlink nccl_device/ outside include/. Follow those paths,
        # while preventing aliases or parent links from forming a walk cycle.
        resolved = directory.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                visit(path)
            elif path.is_file() and path.suffix in (".h", ".hpp", ".cuh"):
                headers.append(path)

    visit(include)
    dependencies = [(path, path.relative_to(include)) for path in headers]
    dependencies.append(
        (library / "libnccl.so.2", (library / "libnccl.so.2").resolve())
    )
    for path, identity in dependencies:
        digest.update(str(identity).encode())
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def resolve_nccl_paths() -> tuple[Path, Path]:
    """Find host/device headers and a runtime from the same NCCL prefix.

    NCCL_HOME takes precedence over the nvidia.nccl wheel. The constructor
    also verifies the actually loaded runtime, which may have been loaded
    earlier by the host framework.
    """
    home = os.environ.get("NCCL_HOME")
    if home:
        prefix = Path(home).expanduser().resolve()
    else:
        try:
            import nvidia.nccl

            prefix = Path(nvidia.nccl.__path__[0]).resolve()
        except (ImportError, AttributeError, IndexError) as exc:
            raise RuntimeError(
                "RDMA MoE AlltoAll currently supports matching NCCL 2.30.7 headers and runtime. "
                "Set NCCL_HOME to that installation."
            ) from exc
    include = Path(os.environ.get("NCCL_INCLUDE_DIR", str(prefix / "include")))
    include = include.expanduser().resolve()
    if "NCCL_LIB_DIR" in os.environ:
        library = Path(os.environ["NCCL_LIB_DIR"]).expanduser().resolve()
    else:
        library = next(
            (
                candidate
                for candidate in (prefix / "lib", prefix / "lib64", prefix)
                if (candidate / "libnccl.so.2").is_file()
            ),
            prefix / "lib",
        )
    if not (include / "nccl.h").is_file() or not (include / "nccl_device.h").is_file():
        raise RuntimeError(f"NCCL host/device headers are missing from {include}")
    if not (library / "libnccl.so.2").is_file():
        raise RuntimeError(f"NCCL runtime libnccl.so.2 is missing from {library}")
    source = (include / "nccl.h").read_text()
    values = [
        re.search(rf"^\s*#define\s+NCCL_{part}\s+(\d+)", source, re.MULTILINE)
        for part in ("MAJOR", "MINOR", "PATCH")
    ]
    if not all(values):
        raise RuntimeError(f"Cannot determine the NCCL header version in {include}")
    version = tuple(int(match.group(1)) for match in values if match is not None)
    if version != _VERIFIED_NCCL_VERSION:
        raise RuntimeError(
            "RDMA MoE AlltoAll currently verifies NCCL 2.30.7 GDAKI "
            f"completion semantics; found headers {'.'.join(map(str, version))}."
        )
    return include, library


def _spec(name: str, sources: list[str], *, nccl: bool = False):
    from . import env as jit_env
    from .core import (
        current_compilation_context,
        gen_jit_spec,
        sm120a_nvcc_flags,
    )

    # The inherited numerical kernels use architecture-specific FP4 PTX and
    # were validated as sm_120a. FlashInfer's CUDA >=12.9 normalization picks
    # 120f by default; admit that requested device, but keep this module's
    # original exact-architecture image instead of claiming SM121 support.
    if not any(
        major == 12 and minor in ("0", "0a", "0f")
        for major, minor in current_compilation_context.TARGET_CUDA_ARCHS
    ):
        raise RuntimeError("RDMA MoE AlltoAll JIT requires an SM120 compilation target")
    flags = sm120a_nvcc_flags
    includes = [
        jit_env.FLASHINFER_CSRC_DIR,
        jit_env.FLASHINFER_INCLUDE_DIR,
    ]
    defines = ["-DDISABLE_NVSHMEM", "-DDISABLE_AGGRESSIVE_PTX_INSTRS"]
    ldflags = []
    if nccl:
        include, library = resolve_nccl_paths()
        includes.extend([include, include / "nccl_device"])
        defines.append("-DDEEP_EP_USE_GIN")
        ldflags.extend([str(library / "libnccl.so.2"), f"-Wl,-rpath,{library}"])
        # External device headers and the linked binary are outside the normal
        # source closure; changing either must invalidate this cached module.
        name += "_nccl23007_" + _nccl_identity(include, library)
    return gen_jit_spec(
        name,
        [jit_env.FLASHINFER_CSRC_DIR / source for source in sources],
        extra_include_paths=includes,
        extra_cuda_cflags=["--expt-relaxed-constexpr", *defines, *flags],
        extra_cflags=defines,
        extra_ldflags=ldflags,
        # Keep the original protocol's explicit arithmetic/rounding behavior.
        use_fast_math=False,
    )


@functools.cache
def get_quantization_module():
    return _spec(
        "rdma_moe_alltoall_quantization",
        ["rdma_moe_alltoall_quantization.cu", "rdma_moe_alltoall_quant_binding.cu"],
    ).build_and_load()


@functools.cache
def get_gin_module():
    return _spec(
        "rdma_moe_alltoall_gin",
        ["rdma_moe_alltoall_gin_kernels.cu", "rdma_moe_alltoall_gin_runtime.cu"],
        nccl=True,
    ).build_and_load()
