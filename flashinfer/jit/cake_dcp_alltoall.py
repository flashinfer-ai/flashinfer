"""JIT wiring for the generated SM100/SM103 DCP all-to-all kernels."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec

# Generated kernel sources, one per CP size, relative to the csrc root. Both
# Blackwell targets compile the same text. Filled by the Cake source exporter;
# an empty list keeps the portable helix kernel everywhere.
GENERATED_SOURCES: List[str] = [
    "generated/dcp_alltoall/cake_dcp_alltoall_335c14a7e34177967411_kernel.cu",
    "generated/dcp_alltoall/cake_dcp_alltoall_baedc53532c2be692cbd_kernel.cu",
]

# (major, minor, define) of every target the generated kernels are built for.
# The define tells the dispatcher which compute capabilities have a generated
# kernel image; every other device runs the portable helix kernel.
GENERATED_TARGETS: List[Tuple[int, str, str]] = [
    (10, "0a", "CAKE_DCP_GENERATED_SM100A"),
    (10, "3a", "CAKE_DCP_GENERATED_SM103A"),
]

MODULE_NAME = "dcp_alltoall_generated"


def generated_targets() -> List[Tuple[int, str, str]]:
    """Blackwell targets of the current build that have generated kernels."""
    archs = current_compilation_context.TARGET_CUDA_ARCHS
    return [t for t in GENERATED_TARGETS if (t[0], t[1]) in archs]


def generated_dcp_alltoall_spec(
    helix_sources: Sequence[Path], extra_include_paths: Sequence[Union[str, Path]]
) -> Optional[JitSpec]:
    """Return the module spec with the generated kernels, or ``None``.

    The generated module is built whenever the compilation context
    (``FLASHINFER_CUDA_ARCH_LIST`` or the visible GPUs) contains at least one
    Blackwell target, including multi-target AOT builds; the dispatcher
    selects the generated kernel per device capability at launch time. Builds
    without a Blackwell target keep the portable helix module.
    """
    targets = generated_targets()
    if not targets or not GENERATED_SOURCES:
        return None
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]
    )
    sources = [
        *helix_sources,
        jit_env.FLASHINFER_CSRC_DIR / "cake_dcp_alltoall_dispatch.cu",
        *(jit_env.FLASHINFER_CSRC_DIR / path for path in GENERATED_SOURCES),
    ]
    return gen_jit_spec(
        MODULE_NAME,
        sources,
        extra_include_paths=list(extra_include_paths),
        extra_cuda_cflags=nvcc_flags
        + ["-DFLASHINFER_DCP_GENERATED=1"]
        + [f"-D{define}=1" for _major, _minor, define in targets],
    )
