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
from .core import gen_jit_spec, logger, current_compilation_context


def _get_bgmv_moe_csrc_dir() -> Path:
    """Get the path to the BGMV MoE CUDA source directory.

    Handles both installed package (data/csrc/bgmv_moe) and
    development checkout (../../csrc/bgmv_moe relative to this file).
    """
    # Standard path via FlashInfer's data directory
    standard_path = jit_env.FLASHINFER_CSRC_DIR / "bgmv_moe"
    if standard_path.exists():
        return standard_path

    # Development fallback: relative to this file
    dev_path = Path(__file__).parent.parent.parent / "csrc" / "bgmv_moe"
    if dev_path.exists():
        return dev_path

    raise FileNotFoundError(
        f"BGMV MoE CUDA sources not found. Checked:\n"
        f"  - {standard_path}\n"
        f"  - {dev_path}\n"
        f"Please ensure the csrc/bgmv_moe/ directory exists."
    )


def get_bgmv_moe_uri() -> str:
    """Generate unique identifier for the BGMV MoE module."""
    return "bgmv_moe"


@functools.cache
def gen_bgmv_moe_module():
    """
    Generate the JIT compilation spec for the BGMV MoE CUDA kernels.

    This compiles the multi-LoRA MoE BGMV shrink/expand kernel pair.
    Supports SM70+ (V100, A100, H100, B200).

    Returns:
        JitSpec that can be built and loaded.
    """
    csrc_dir = _get_bgmv_moe_csrc_dir()
    uri = get_bgmv_moe_uri()

    # Create generation directory
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    # Source files to copy
    source_files = [
        "moe_bgmv_binding.cu",
        "moe_bgmv_bf16_bf16_bf16.cu",
        "moe_bgmv_bf16_fp32_bf16.cu",
        "moe_bgmv_fp16_fp16_fp16.cu",
        "moe_bgmv_fp16_fp32_fp16.cu",
        "moe_bgmv_fp32_bf16_bf16.cu",
        "moe_bgmv_fp32_fp16_fp16.cu",
    ]

    # Header files to copy (includes moe_bgmv_ops.cu which is #included by binding)
    header_files = [
        "moe_bgmv_impl.cuh",
        "moe_bgmv_config.h",
        "moe_bgmv_ops.h",
        "moe_bgmv_ops.cu",
        "kernel_config.h",
    ]

    # Copy sources to gen directory
    sources = []
    for fname in source_files:
        src_path = csrc_dir / fname
        if not src_path.exists():
            raise FileNotFoundError(f"BGMV MoE source file not found: {src_path}")
        dest_path = gen_directory / fname
        shutil.copy(src_path, dest_path)
        sources.append(dest_path)

    # Copy headers to gen directory
    for fname in header_files:
        src_path = csrc_dir / fname
        if not src_path.exists():
            raise FileNotFoundError(f"BGMV MoE header file not found: {src_path}")
        shutil.copy(src_path, gen_directory / fname)

    # Get nvcc flags for supported architectures (SM70+)
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]  # SM90+ (H100, B200)
    )

    spec = gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DFLASHINFER_ENABLE_BF16",
            "-DFLASHINFER_ENABLE_F16",
        ],
        extra_include_paths=[
            str(gen_directory),
            str(jit_env.FLASHINFER_INCLUDE_DIR),
            str(jit_env.FLASHINFER_CSRC_DIR),
        ],
    )

    logger.info(f"Generated BGMV MoE JIT spec: {spec.name}")
    return spec


@functools.cache
def load_bgmv_moe_module():
    """
    Build and load the BGMV MoE CUDA extension via FlashInfer's JIT system.

    Returns the loaded module with `bgmv_moe_shrink` and `bgmv_moe_expand` functions.
    """
    spec = gen_bgmv_moe_module()
    module = spec.build_and_load()
    logger.info("BGMV MoE module loaded successfully")
    return module
