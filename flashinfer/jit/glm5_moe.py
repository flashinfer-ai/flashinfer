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

JIT loader for the low-token GLM5 block-FP8 fused MoE kernels.
"""

import functools
import os
import shutil
from pathlib import Path

from . import env as jit_env
from .core import current_compilation_context, gen_jit_spec, logger


_SOURCE_FILES = [
    "glm5_fused_expert_up.cu",
    "glm5_fused_expert_down.cu",
]
_INCLUDE_FILES = ["topk_reduce.cuh"]
_FLASHINFER_SUPPORT_HEADERS = ["tvm_ffi_utils.h"]


def _get_glm5_moe_csrc_dir() -> Path:
    standard_path = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "glm5"
    if standard_path.exists():
        return standard_path

    dev_path = Path(__file__).parent.parent.parent / "csrc" / "fused_moe" / "glm5"
    if dev_path.exists():
        return dev_path

    raise FileNotFoundError(
        f"GLM5 fused MoE sources were not found under {standard_path} or {dev_path}."
    )


@functools.cache
def gen_glm5_moe_module():
    """Create the SM100-family JIT specification for GLM5 fused MoE."""
    csrc_dir = _get_glm5_moe_csrc_dir()
    dev_root = Path(__file__).parent.parent.parent
    uri = "glm5_fused_moe_sm100"
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    def _copy(name: str) -> Path:
        source = csrc_dir / name
        if not source.exists():
            raise FileNotFoundError(f"GLM5 fused MoE source not found: {source}")
        destination = gen_directory / name
        shutil.copy(source, destination)
        return destination

    sources = [_copy(name) for name in _SOURCE_FILES]
    for name in _INCLUDE_FILES:
        _copy(name)
    for name in _FLASHINFER_SUPPORT_HEADERS:
        candidates = (
            jit_env.FLASHINFER_CSRC_DIR / name,
            dev_root / "csrc" / name,
        )
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(
                f"FlashInfer support header {name!r} was not found in {candidates}."
            )
        shutil.copy(source, gen_directory / name)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )
    spec = gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DFLASHINFER_ENABLE_BF16",
            "-DFLASHINFER_ENABLE_FP8_E4M3",
        ],
        extra_include_paths=[
            gen_directory,
            dev_root / "include",
            dev_root / "csrc",
            jit_env.FLASHINFER_INCLUDE_DIR,
            jit_env.FLASHINFER_CSRC_DIR,
            *jit_env.CUTLASS_INCLUDE_DIRS,
        ],
    )
    logger.info("Generated GLM5 fused MoE JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_glm5_moe_module():
    """Build and load the GLM5 fused MoE module."""
    return gen_glm5_moe_module().build_and_load()
