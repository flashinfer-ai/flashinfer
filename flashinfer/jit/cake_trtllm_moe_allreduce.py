"""Source-built TRT-LLM MoE all-reduce fusion kernels for SM100 and SM103.

This bundle serves ``trtllm_moe_allreduce_fusion(backend="cake")`` when no
``moe_allreduce_out`` is requested; calls with the all-reduce output are routed
to :mod:`flashinfer.jit.cake_trtllm_moe_allreduce_union`.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

SOURCE_PACKAGE = "cake_trtllm_moe_allreduce_fusion"

_ARCH_BY_CAPABILITY = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# The kernel source holds the generic kernels plus the SM103 single-token
# group (``_sm103_t1``); only the SM103 module compiles that group.
_ARCH_CUDA_CFLAGS = {
    "sm_100a": [*sm100a_nvcc_flags, "-DCAKE_MOE_AR_SM103_T1=0"],
    "sm_103a": [*sm103a_nvcc_flags, "-DCAKE_MOE_AR_SM103_T1=1"],
}


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake TRT-LLM MoE all-reduce fusion sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def target_arch(device_index: int) -> str:
    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = _ARCH_BY_CAPABILITY.get(capability)
    if arch is None:
        raise ValueError(
            "Cake TRT-LLM MoE all-reduce requires SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return arch


@functools.cache
def spec(arch: str) -> JitSpec:
    if arch not in _ARCH_CUDA_CFLAGS:
        raise ValueError(f"unsupported architecture {arch!r}")
    source_dir = _source_dir()
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}_{arch}",
        sources=[
            source_dir / f"{SOURCE_PACKAGE}_kernels.cu",
            source_dir / f"{SOURCE_PACKAGE}_launcher.cu",
        ],
        extra_cuda_cflags=_ARCH_CUDA_CFLAGS[arch],
        extra_include_paths=[source_dir.parent],
    )


@functools.cache
def load(device_index: int) -> Any:
    return spec(target_arch(device_index)).build_and_load()


__all__ = ["load", "spec", "target_arch"]
