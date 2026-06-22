"""
Copyright (c) 2025 by FlashInfer team.

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

import functools
import os
import shutil
from pathlib import Path

from . import env as jit_env
from .core import gen_jit_spec, logger, sm90a_nvcc_flags


def _get_megamoe_csrc_dir() -> Path:
    """Get the path to the MoE monokernel (megamoe) CUDA source directory.

    Handles both the installed package (data/csrc/fused_moe/megamoe) and a
    development checkout (../../csrc/fused_moe/megamoe relative to this file).
    """
    standard_path = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "megamoe"
    if standard_path.exists():
        return standard_path

    dev_path = (
        Path(__file__).parent.parent.parent
        / "csrc"
        / "fused_moe"
        / "megamoe"
    )
    if dev_path.exists():
        return dev_path

    raise FileNotFoundError(
        f"MoE monokernel (megamoe) CUDA sources not found. Checked:\n"
        f"  - {standard_path}\n"
        f"  - {dev_path}\n"
        f"Please ensure the csrc/fused_moe/megamoe/ directory exists."
    )


def get_megamoe_uri() -> str:
    """Generate the unique identifier for the megamoe module."""
    return "megamoe"


# Files actually fed to nvcc as compilation units.  Everything else in the
# megamoe tree is `#include`d into one of these two TUs (whole-program unity
# build), so listing it here would cause duplicate-symbol link errors.
#
#   - megamoe_binding.cu   #includes moe_wrapper.cu -> src/moe.cu (which in turn
#                          #includes the up/down/routing/prepare/scale .cu files)
#   - src/moe_tma.cu       standalone host TU: definitions of the create_*_tma_desc
#                          factories (declared in src/moe_tma.h, never #included)
_SOURCE_FILES = [
    "megamoe_binding.cu",
    "src/moe_tma.cu",
]

# Header / include-only sources copied alongside the compiled TUs so the
# `#include` graph resolves inside the gen directory.  These are NOT compiled.
_INCLUDE_FILES = [
    "moe_wrapper.cu",
    "src/moe.cu",
    "src/moe_up_projection.cu",
    "src/moe_down_projection.cu",
    "src/moe_routing.cu",
    "src/moe_prepare.cu",
    "src/moe_scale_inputs.cu",
    "src/moe_interface.h",
    "src/moe_internal.h",
    "src/moe_grid_barrier.h",
    "src/moe_tma.h",
    "src/ptx_utils.h",
]


@functools.cache
def gen_megamoe_module():
    """
    Generate the JIT compilation spec for the MoE monokernel (megamoe).

    Compiles the single-kernel top-K MoE pipeline (routing, up-projection,
    SiLU, down-projection, reduction) for the Qwen3.5-35B block-FP8
    WGMMA+TMA path.  Hopper (SM90a) only — the kernel uses
    `wgmma.mma_async` and TMA, which require SM90.

    Returns:
        JitSpec that can be built and loaded.
    """
    csrc_dir = _get_megamoe_csrc_dir()
    uri = get_megamoe_uri()

    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    # `src/` must exist so the `#include "src/..."` paths resolve and so the
    # `src/*.cu` include-only files land where moe.cu expects them.
    os.makedirs(gen_directory / "src", exist_ok=True)

    def _copy(rel_name: str) -> Path:
        src_path = csrc_dir / rel_name
        if not src_path.exists():
            raise FileNotFoundError(f"megamoe source file not found: {src_path}")
        dest_path = gen_directory / rel_name
        os.makedirs(dest_path.parent, exist_ok=True)
        shutil.copy(src_path, dest_path)
        return dest_path

    sources = [_copy(fname) for fname in _SOURCE_FILES]
    for fname in _INCLUDE_FILES:
        _copy(fname)

    # Hopper (SM90a) ONLY: the kernel uses wgmma.mma_async + TMA and is
    # hard-specialized to a single SM90a shape — it cannot target any
    # other architecture.  Use the `sm90a_nvcc_flags` constant directly
    # (matching gen_cutlass_fused_moe_sm90_module / gen_gemm_sm90_module)
    # rather than the multi-arch `get_nvcc_flags_list` path: both resolve
    # to `compute_90a,sm_90a` today, but the constant is the honest,
    # robust choice for a kernel that is SM90a-only by construction (it
    # always emits the `a` suffix and never depends on the detected /
    # FLASHINFER_CUDA_ARCH_LIST arch set).
    spec = gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=sm90a_nvcc_flags
        + [
            "-DFLASHINFER_ENABLE_BF16",
            "-DFLASHINFER_ENABLE_FP8_E4M3",
        ],
        extra_include_paths=[
            str(gen_directory),
            str(gen_directory / "src"),
            str(jit_env.FLASHINFER_INCLUDE_DIR),
            str(jit_env.FLASHINFER_CSRC_DIR),
        ],
    )

    logger.info(f"Generated megamoe JIT spec: {spec.name}")
    return spec


@functools.cache
def load_megamoe_module():
    """
    Build and load the MoE monokernel (megamoe) CUDA extension via
    FlashInfer's JIT system.

    Returns the loaded module exposing `moe_monokernel_topk`.
    """
    spec = gen_megamoe_module()
    module = spec.build_and_load()
    logger.info("megamoe module loaded successfully")
    return module
