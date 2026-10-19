"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# JIT loader for the Cake W8A8 MXFP8 MegaMoE EP16 route (exact sm_103a).
#
# The device sources are one translation unit per physical program
# (``cake_w8a8_megamoe_ep16_<stage>_kernel.cu``) plus one hand-written
# tvm-ffi binding per program (``cake_w8a8_megamoe_ep16_<stage>_binding.cu``)
# under ``csrc/``. The module name carries the one exact target and a digest
# of the listed sources, so a changed drop never reuses a stale library and
# two targets never share one cached library. Targets follow FlashInfer's
# compilation context (``FLASHINFER_CUDA_ARCH_LIST`` or the visible devices),
# restricted to the single capability the programs are generated for.

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

from ...compilation_context import CompilationContext
from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags

MODULE_PREFIX = "cake_w8a8_megamoe_ep16"
TARGET_CAPABILITY = (10, 3)
TARGET_ARCH = "sm_103a"
SUPPORTED_CAPABILITIES = (TARGET_CAPABILITY,)
DIGEST_HEX_DIGITS = 20
KERNEL_SUFFIX = "_kernel.cu"
BINDING_SUFFIX = "_binding.cu"
MANIFEST_NAME = "cake_w8a8_megamoe_ep16_manifest.json"

# Numerics follow the manifest's ``compile.receipt_options``, which list
# ``--use_fast_math`` and nothing else: the programs were generated and
# measured with fast math and no further numerics flags. ``NUMERICS_FLAGS``
# stays empty unless a regenerated manifest lists more; such flags go after
# the target flags (the later flag wins).
USE_FAST_MATH = True
NUMERICS_FLAGS: tuple[str, ...] = ()


def default_csrc_dir() -> Path:
    """Directory holding the generated kernels, bindings and manifest.

    ``csrc/cake_w8a8_megamoe_ep16/`` beside this module when present, else
    ``csrc/`` itself (both layouts are shipped as package data).
    """

    package_sources = Path(__file__).resolve().parent / "csrc"
    nested = package_sources / MODULE_PREFIX
    if nested.is_dir():
        return nested
    return package_sources


def _get_flashinfer_header_dirs() -> list[Path]:
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer" / "layout.cuh"
    ).is_file():
        return installed

    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer" / "layout.cuh"
    ).is_file():
        return source

    raise FileNotFoundError("FlashInfer JIT headers were not found")


def discover_sources(csrc_dir: Path | str | None = None) -> list[Path]:
    """Kernel translation units then binding translation units, each sorted.

    Every ``*_kernel.cu`` must have a ``*_binding.cu`` partner and vice versa.
    """

    root = Path(default_csrc_dir() if csrc_dir is None else csrc_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Cake W8A8 MegaMoE EP16 sources were not found under {root}"
        )
    kernels = sorted(root.glob(f"*{KERNEL_SUFFIX}"))
    bindings = sorted(root.glob(f"*{BINDING_SUFFIX}"))
    if not kernels:
        raise FileNotFoundError(
            f"no *{KERNEL_SUFFIX} device sources under {root}; the generated "
            "programs are not installed"
        )
    kernel_stages = [p.name[: -len(KERNEL_SUFFIX)] for p in kernels]
    binding_stages = [p.name[: -len(BINDING_SUFFIX)] for p in bindings]
    if kernel_stages != binding_stages:
        raise FileNotFoundError(
            "kernel/binding translation units do not pair up under "
            f"{root}: kernels={kernel_stages}, bindings={binding_stages}"
        )
    for path in (*kernels, *bindings):
        if not path.name.startswith(MODULE_PREFIX + "_"):
            raise FileNotFoundError(
                f"unexpected source name {path.name}; every file must start with {MODULE_PREFIX}_"
            )
    return [*kernels, *bindings]


def source_digest(sources: list[Path]) -> str:
    """SHA-256 over the listed sources (name and bytes), 20 hex digits."""

    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:DIGEST_HEX_DIGITS]


def module_name(sources: list[Path]) -> str:
    return f"{MODULE_PREFIX}_{TARGET_ARCH}_{source_digest(sources)}"


def supported_capabilities() -> tuple[tuple[int, int], ...]:
    """Admitted capabilities that are also FlashInfer build targets."""

    targets = CompilationContext().TARGET_CUDA_ARCHS
    return tuple(
        capability
        for capability in SUPPORTED_CAPABILITIES
        if (capability[0], f"{capability[1]}a") in targets
    )


def require_supported_capability(capability: tuple[int, int]) -> tuple[int, int]:
    """Return ``capability`` when the route is built for it, else raise naming the fix."""

    capability = (int(capability[0]), int(capability[1]))
    if capability not in SUPPORTED_CAPABILITIES:
        raise RuntimeError(
            f"Cake W8A8 MXFP8 MegaMoE EP16 requires compute capability "
            f"{TARGET_CAPABILITY[0]}.{TARGET_CAPABILITY[1]} ({TARGET_ARCH}), got "
            f"{capability[0]}.{capability[1]}"
        )
    if capability not in supported_capabilities():
        raise RuntimeError(
            "Cake W8A8 MXFP8 MegaMoE EP16 is not a build target for compute capability "
            f"{capability[0]}.{capability[1]}; include {capability[0]}.{capability[1]}a "
            "in FLASHINFER_CUDA_ARCH_LIST"
        )
    return capability


def gen_cake_w8a8_megamoe_ep16_spec(csrc_dir: Path | str | None = None) -> JitSpec:
    """Exact-architecture JIT specification of the route's programs.

    ``csrc_dir`` overrides source discovery (tests); the default is the
    installed drop beside this module.
    """

    require_supported_capability(TARGET_CAPABILITY)
    sources = discover_sources(csrc_dir)
    root = sources[0].parent
    spec = gen_jit_spec(
        name=module_name(sources),
        sources=sources,
        extra_cuda_cflags=[*sm103a_nvcc_flags, *NUMERICS_FLAGS],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *_get_flashinfer_header_dirs()],
        use_fast_math=USE_FAST_MATH,
    )
    logger.info(f"Generated Cake W8A8 MegaMoE EP16 JIT spec: {spec.name}")
    return spec


@functools.cache
def _build_and_load() -> Any:
    module = gen_cake_w8a8_megamoe_ep16_spec().build_and_load()
    logger.info("Loaded Cake W8A8 MegaMoE EP16 module for sm_103a")
    return module


def load_module(device: Any = None) -> Any:
    """Build or load the module for ``device`` (which must be an admitted sm_103 device)."""

    import torch

    from ...utils import get_compute_capability

    resolved = torch.device("cuda") if device is None else torch.device(device)
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    require_supported_capability(get_compute_capability(resolved))
    return _build_and_load()


def available(csrc_dir: Path | str | None = None) -> bool:
    """True when the generated sources are installed and sm_103a is a build target."""

    try:
        discover_sources(csrc_dir)
    except FileNotFoundError:
        return False
    return TARGET_CAPABILITY in supported_capabilities()


__all__ = [
    "BINDING_SUFFIX",
    "KERNEL_SUFFIX",
    "MANIFEST_NAME",
    "MODULE_PREFIX",
    "NUMERICS_FLAGS",
    "SUPPORTED_CAPABILITIES",
    "TARGET_ARCH",
    "TARGET_CAPABILITY",
    "USE_FAST_MATH",
    "available",
    "default_csrc_dir",
    "discover_sources",
    "gen_cake_w8a8_megamoe_ep16_spec",
    "load_module",
    "module_name",
    "require_supported_capability",
    "source_digest",
    "supported_capabilities",
]
