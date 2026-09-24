"""JIT wiring for the generated SM100/SM103 DCP all-to-all kernels."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from . import env as jit_env
from .core import (
    JitSpec,
    current_compilation_context,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

# Generated source inventory per exact target, relative to the csrc root.
# Filled by the Cake source exporter; an empty entry keeps the portable helix
# kernel for that target.
GENERATED_SOURCES: Dict[str, List[str]] = {
    "sm_100a": [
        "generated/dcp_alltoall/sm_100a/cake_dcp_alltoall_4e05bb34258ed0b17208_kernel.cu",
        "generated/dcp_alltoall/sm_100a/cake_dcp_alltoall_4e05bb34258ed0b17208_binding.cu",
        "generated/dcp_alltoall/sm_100a/cake_dcp_alltoall_093259c7e2860c83269f_kernel.cu",
        "generated/dcp_alltoall/sm_100a/cake_dcp_alltoall_093259c7e2860c83269f_binding.cu",
    ],
    "sm_103a": [
        "generated/dcp_alltoall/sm_103a/cake_dcp_alltoall_3038adb817c49d1907b4_kernel.cu",
        "generated/dcp_alltoall/sm_103a/cake_dcp_alltoall_3038adb817c49d1907b4_binding.cu",
        "generated/dcp_alltoall/sm_103a/cake_dcp_alltoall_187a3dc44ec9fa42b278_kernel.cu",
        "generated/dcp_alltoall/sm_103a/cake_dcp_alltoall_187a3dc44ec9fa42b278_binding.cu",
    ],
}

_TARGETS = {
    frozenset({(10, "0a")}): ("sm100a", "sm_100a", sm100a_nvcc_flags),
    frozenset({(10, "3a")}): ("sm103a", "sm_103a", sm103a_nvcc_flags),
}


def generated_module_name(arch: str) -> str:
    """JIT module name of the generated route for ``arch`` (``sm_100a``/``sm_103a``)."""
    for name, candidate, _flags in _TARGETS.values():
        if candidate == arch:
            return "dcp_alltoall_" + name
    raise ValueError(f"no generated DCP all-to-all target for {arch!r}")


def generated_dcp_alltoall_spec(
    helix_sources: Sequence[Path], extra_include_paths: Sequence[Union[str, Path]]
) -> Optional[JitSpec]:
    """Return the generated-kernel module spec for an exact Blackwell target.

    Target selection follows ``CompilationContext`` (``FLASHINFER_CUDA_ARCH_LIST``
    or the visible GPUs). Any other target set, including multi-target AOT
    builds, keeps the portable helix module.
    """
    selected = _TARGETS.get(frozenset(current_compilation_context.TARGET_CUDA_ARCHS))
    if selected is None:
        return None
    name, arch, arch_flags = selected
    generated = GENERATED_SOURCES.get(arch)
    if not generated:
        return None
    sources = [
        *helix_sources,
        jit_env.FLASHINFER_CSRC_DIR / "cake_dcp_alltoall_dispatch.cu",
        *(jit_env.FLASHINFER_CSRC_DIR / path for path in generated),
    ]
    return gen_jit_spec(
        "dcp_alltoall_" + name,
        sources,
        extra_include_paths=list(extra_include_paths),
        extra_cuda_cflags=arch_flags
        + [
            "-DFLASHINFER_DCP_GENERATED=1",
            "-DCAKE_DCP_GENERATED_TARGET_" + name.upper() + "=1",
        ],
    )
