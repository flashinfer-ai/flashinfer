"""JIT wiring for the generated Blackwell (SM100/SM103) Ulysses all-to-all kernels.

The generated kernels are one source per program for every Blackwell target.
Build targets follow ``FLASHINFER_CUDA_ARCH_LIST`` (AOT hosts) or the visible
devices through the process compilation context
(``flashinfer.jit.core.current_compilation_context``); the generated module is built
whenever a major-10 target is present and compiled for each such target. A
device whose capability is not among the built targets keeps the portable
NVLink kernel of ``ulysses_all_to_all.cu``.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Optional

import torch

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec

SUPPORTED_MAJOR_VERSIONS = (10,)
GENERATED_DEFINE = "-DFLASHINFER_ULYSSES_GENERATED=1"
# csrc-relative device translation units of the generated programs (one file
# per program for every supported target).
GENERATED_KERNEL_SOURCES: list[str] = [
    "generated/ulysses/cake_ulysses_a2a_45915ee873e8d754fc8a_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_b3e242482bd4ee11df46_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_2862f9aa8f469dc4a895_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_b3fa2ce10ba84e438405_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_a43ae84038aed9942618_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_d2e1d5cc0834259b4b8a_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_f01765220b0ea3e0a458_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_8f16ac9ca3b30bbdb8ad_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_646d36c61be114175d4f_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_67f08c8843354b114fa0_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_329637d2bca31c68941f_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_1b864007ccae50b2c64b_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_7f52c5383930c70fcdfc_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_13d5bcbfe2305fa3b33f_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_37f3eb80fcfbdc94ec58_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_6c0f4daca24cb49e50b3_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_aa7f336404576b7995f9_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_7d3317288d67a76a904a_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_5e2b99e811e0f2a39e8c_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_0bfaf1875f1eae295499_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_9f53afbc054ade876d63_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_aa3de711d5012d241018_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_3b65cc535a8bb3d2f951_kernel.cu",
    "generated/ulysses/cake_ulysses_a2a_80ac1f368b5e2e162a4f_kernel.cu",
]


def _capability_of(major: Any, minor: Any) -> tuple[int, int]:
    """``(major, minor)`` of a ``TARGET_CUDA_ARCHS`` entry (``(10, "0a")`` -> ``(10, 0)``)."""
    return int(major), int(re.sub(r"[a-z]+$", "", str(minor)))


def target_capabilities() -> tuple[tuple[int, int], ...]:
    """Supported compute capabilities among the build targets, ascending.

    Read from the live process context on every call so a pinned target list
    (tests, AOT hosts) is honoured without cache invalidation.
    """
    context = current_compilation_context
    return tuple(
        sorted(
            {
                _capability_of(major, minor)
                for major, minor in context.TARGET_CUDA_ARCHS
                if int(major) in SUPPORTED_MAJOR_VERSIONS
            }
        )
    )


def supported_capabilities() -> tuple[tuple[int, int], ...]:
    """Capabilities the generated kernels are built for in this process."""
    return target_capabilities()


def supported_capability(capability: tuple[int, int]) -> Optional[tuple[int, int]]:
    """``(major, minor)`` when the generated kernels are built for it, else ``None``."""
    key = (int(capability[0]), int(capability[1]))
    return key if key in target_capabilities() else None


@functools.cache
def _device_capability(device_index: int) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(device_index))


def generated_module_name() -> Optional[str]:
    """Module name derived from the supported build targets; ``None`` without one."""
    targets = sorted(
        (int(major), str(minor))
        for major, minor in current_compilation_context.TARGET_CUDA_ARCHS
        if int(major) in SUPPORTED_MAJOR_VERSIONS
    )
    if not targets:
        return None
    return "ulysses_a2a_" + "_".join(f"sm{major}{minor}" for major, minor in targets)


def nvcc_flags() -> list[str]:
    """``-gencode`` flags for every supported build target plus FlashInfer's common flags."""
    return current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=list(SUPPORTED_MAJOR_VERSIONS)
    )


def extra_cuda_cflags() -> list[str]:
    return nvcc_flags() + [GENERATED_DEFINE]


def generated_ulysses_spec() -> Optional[JitSpec]:
    """The generated module when it is built for this process and its current device.

    Returns ``None`` when no build target has a supported major version, or when
    the current CUDA device's capability is not among the built targets; the
    caller then keeps the portable NVLink kernel.
    """
    name = generated_module_name()
    if name is None:
        return None
    if torch.cuda.is_available():
        capability = _device_capability(torch.cuda.current_device())
        if supported_capability(capability) is None:
            return None
    sources = [
        "ulysses_all_to_all.cu",
        "cake_ulysses_dispatch.cu",
        *GENERATED_KERNEL_SOURCES,
    ]
    return gen_jit_spec(
        name,
        [jit_env.FLASHINFER_CSRC_DIR / path for path in sources],
        extra_cuda_cflags=extra_cuda_cflags(),
    )
