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

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

CakeMoeFinalizeArch = Literal["sm_100a", "sm_103a"]

SOURCE_PACKAGE = "cake_moe_finalize_allreduce_fusion"
HIDDEN_DIM = 7168
WORLD_SIZES = (2, 4, 8)
DTYPES = ("float16", "bfloat16")
OUTPUT_PROFILES = ("110", "111")  # residual + norm, optionally + NVFP4 quant

_ARCH_BY_CAPABILITY: dict[tuple[int, int], CakeMoeFinalizeArch] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake MoE finalize CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def kernel_names() -> tuple[str, ...]:
    """The twelve generated kernels: dtype x world size x output profile."""
    return tuple(
        f"cake_trtllm_moe_finalize_{dtype}_ws{world_size}_o{profile}"
        for dtype in DTYPES
        for world_size in WORLD_SIZES
        for profile in OUTPUT_PROFILES
    )


@functools.cache
def target_arch(device_index: int) -> CakeMoeFinalizeArch:
    """Return the generated architecture for one CUDA device (cached per index)."""
    capability = torch.cuda.get_device_capability(device_index)
    arch = _ARCH_BY_CAPABILITY.get(capability)
    if arch is None:
        raise ValueError(
            "Cake MoE finalize requires SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return arch


@functools.cache
def gen_cake_moe_finalize_module(arch: CakeMoeFinalizeArch) -> JitSpec:
    """One JIT module per architecture: the device sources plus the dispatch binding.

    The device sources are architecture independent; SM103 builds them with the
    sm_103a flags.
    """
    source_dir = _source_dir()
    sources = [source_dir / "sm_100a" / f"{name}_device.cu" for name in kernel_names()]
    sources.append(source_dir / f"{SOURCE_PACKAGE}_binding.cu")
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}_{arch}",
        sources=sources,
        extra_cuda_cflags=_NVCC_FLAGS[arch],
    )


@functools.cache
def get_cake_moe_finalize_module(arch: CakeMoeFinalizeArch):
    return gen_cake_moe_finalize_module(arch).build_and_load()


def run_cake_moe_finalize(
    *,
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_out: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    quant_out: torch.Tensor | None,
    scale_out: torch.Tensor | None,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    eps: float,
    shared_expert_output: torch.Tensor | None,
    expert_scale_factor: torch.Tensor | None,
    routed_scaling_factor: float | None,
    weight_bias: float | None,
) -> None:
    """Launch the Cake finalize kernel for ``allreduce_in``'s device.

    Tensor shapes, dtypes and the workspace table are validated once, in the
    binding. ``quant_out`` / ``scale_out`` are byte buffers (packed FP4 and E4M3
    scales in SWIZZLED_128x4 layout) and may use any element type.
    """
    if norm_out is None or residual_out is None:
        raise ValueError('norm_out and residual_out are required when backend="cake"')
    if expert_scale_factor is None:
        raise ValueError('expert_scale_factor is required when backend="cake"')
    device_index = allreduce_in.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    module = get_cake_moe_finalize_module(target_arch(device_index))
    module.cake_moe_finalize_allreduce_fusion(
        allreduce_in,
        residual_in,
        norm_weight,
        expanded_idx_to_permuted_idx,
        expert_scale_factor,
        shared_expert_output,
        residual_out,
        norm_out,
        quant_out,
        scale_out,
        workspace_ptrs,
        world_rank,
        world_size,
        float(eps),
        1.0 if routed_scaling_factor is None else float(routed_scaling_factor),
        0.0 if weight_bias is None else float(weight_bias),
        bool(launch_with_pdl),
    )


__all__ = [
    "DTYPES",
    "HIDDEN_DIM",
    "OUTPUT_PROFILES",
    "SOURCE_PACKAGE",
    "WORLD_SIZES",
    "gen_cake_moe_finalize_module",
    "get_cake_moe_finalize_module",
    "kernel_names",
    "run_cake_moe_finalize",
    "target_arch",
]
