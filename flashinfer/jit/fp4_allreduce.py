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

# JIT loader of the NVFP4-transport tensor-parallel all-reduce programs
# (``csrc/fp4_allreduce``).  The programs are generated: one kernel per
# (family, world size, input format[, output format], variant, rank) at a
# fixed row capacity, listed in ``PROGRAMS`` next to the hand-written binding
# of each (family, world size).  Two families: ``allreduce`` (the bfloat16
# all-reduce, world sizes 2 and 4; TP4 in three phase-B variants) and
# ``rmsnorm`` (world size 2, the all-reduce fused with the residual add and the
# Gemma RMSNorm; bfloat16, NVFP4 or MXFP8 output; the quantised outputs in two
# receive-poll variants; eps and the weight bias are compiled in).  Every
# program is compiled for the device it runs on (sm_100a / sm_103a / sm_107a)
# with the exact flag set below and without fast math; a program whose device
# code differs between those architectures keeps the difference under
# ``__CUDA_ARCH__`` guards inside its one source file.

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import torch

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

Fp4AllReduceArch = Literal["sm_100a", "sm_103a", "sm_107a"]

SOURCE_PACKAGE = "fp4_allreduce"
HIDDEN = 6144
RECORD_BYTES = 3456  # NVFP4 input record: 3072 E2M1 value bytes + 384 E4M3 scale bytes
CAPACITY = 16384  # rows the delivered programs are specialised for
FAMILIES = ("allreduce", "rmsnorm")
WORLD_SIZES = (2, 4)  # allreduce family
RMSNORM_WORLD_SIZES = (2,)  # rmsnorm family
INPUT_FORMATS = ("bf16", "nvfp4")
OUTPUT_FORMATS = ("bf16", "nvfp4", "mxfp8")  # rmsnorm family (allreduce: bf16 only)
RMSNORM_EPS = 1e-06  # compiled into the fused-norm programs
RMSNORM_WEIGHT_BIAS = (
    1.0  # Gemma: y = z * rsqrt(mean(z^2) + eps) * (weight_bias + weight)
)

PROGRAMS: dict = {
    "tp2": {
        "family": "allreduce",
        "algo": "oneshot",
        "binding": "fp4_allreduce_tp2_binding.cu",
        "block": 352,
        "dynamic_smem_bytes": 226176,
        "variants": 1,
        "variant_rule": None,
        "ctrl_words": 16384,
        "table_entries": 4,
        "row_multiple": 1,
        "max_grid": 4096,
        "capacity": 16384,
        "kernels": [
            {
                "symbol": "kernel_fp4_allreduce_tp2_oneshot_bf16_fused_r0_cap16384",
                "source": "fp4_allreduce_tp2_oneshot_bf16_fused_r0_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 0,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_oneshot_bf16_fused_r1_cap16384",
                "source": "fp4_allreduce_tp2_oneshot_bf16_fused_r1_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 1,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_oneshot_nvfp4_fused_r0_cap16384",
                "source": "fp4_allreduce_tp2_oneshot_nvfp4_fused_r0_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 0,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_oneshot_nvfp4_fused_r1_cap16384",
                "source": "fp4_allreduce_tp2_oneshot_nvfp4_fused_r1_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 1,
                "variant": 0,
            },
        ],
    },
    "tp4": {
        "family": "allreduce",
        "algo": "twoshot",
        "binding": "fp4_allreduce_tp4_binding.cu",
        "block": 352,
        "dynamic_smem_bytes": 226304,
        "variants": 3,
        "variant_rule": {"team_max": 40, "by_quad_min": 64},
        "ctrl_words": 4,
        "table_entries": 4,
        "row_multiple": 4,
        "max_grid": 65536,
        "capacity": 16384,
        "kernels": [
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r0_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r0_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 0,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r1_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r1_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 1,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r2_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r2_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 2,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r3_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r3_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 3,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv1_r0_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv1_r0_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 0,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv1_r1_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv1_r1_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 1,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv1_r2_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv1_r2_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 2,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv1_r3_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv1_r3_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 3,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv2_r0_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv2_r0_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 0,
                "variant": 2,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv2_r1_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv2_r1_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 1,
                "variant": 2,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv2_r2_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv2_r2_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 2,
                "variant": 2,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv2_r3_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_bf16_fused_bv2_r3_cap16384_device.cu",
                "input_format": "bf16",
                "rank": 3,
                "variant": 2,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv0_r0_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv0_r0_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 0,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv0_r1_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv0_r1_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 1,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv0_r2_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv0_r2_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 2,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv0_r3_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv0_r3_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 3,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r0_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r0_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 0,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r1_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r1_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 1,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r2_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r2_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 2,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r3_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r3_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 3,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv2_r0_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv2_r0_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 0,
                "variant": 2,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv2_r1_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv2_r1_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 1,
                "variant": 2,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv2_r2_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv2_r2_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 2,
                "variant": 2,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv2_r3_cap16384",
                "source": "fp4_allreduce_tp4_twoshot_nvfp4_fused_bv2_r3_cap16384_device.cu",
                "input_format": "nvfp4",
                "rank": 3,
                "variant": 2,
            },
        ],
    },
    "tp2_rmsnorm": {
        "family": "rmsnorm",
        "algo": "oneshot",
        "binding": "fp4_allreduce_tp2_rmsnorm_binding.cu",
        "block": {"bf16": 256, "nvfp4": 256},
        "dynamic_smem_bytes": {"bf16": 219776, "nvfp4": 215424},
        "formats": [
            ["bf16", "bf16"],
            ["nvfp4", "bf16"],
            ["bf16", "nvfp4"],
            ["bf16", "mxfp8"],
        ],
        "eps": 1e-06,
        "weight_bias": 1.0,
        "variants": 2,
        "variant_rule": {"probe_min_rows_per_cta": 28, "outputs": ["nvfp4", "mxfp8"]},
        "ctrl_words": 4,
        "table_entries": 2,
        "row_multiple": 4,
        "max_grid": 65536,
        "capacity": 16384,
        "nvfp4_value_bytes": 3072,
        "nvfp4_scales_per_row": 384,
        "mxfp8_value_bytes": 6144,
        "mxfp8_scales_per_row": 192,
        "kernels": [
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_bf16_fused_r0_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_bf16_fused_r0_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "bf16",
                "rank": 0,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_bf16_fused_r1_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_bf16_fused_r1_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "bf16",
                "rank": 1,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_r0_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_r0_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "mxfp8",
                "rank": 0,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_r1_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_r1_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "mxfp8",
                "rank": 1,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_pa_r0_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_pa_r0_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "mxfp8",
                "rank": 0,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_pa_r1_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_pa_r1_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "mxfp8",
                "rank": 1,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_r0_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_r0_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "nvfp4",
                "rank": 0,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_r1_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_r1_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "nvfp4",
                "rank": 1,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_pa_r0_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_pa_r0_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "nvfp4",
                "rank": 0,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_pa_r1_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_pa_r1_cap16384_device.cu",
                "input_format": "bf16",
                "output_format": "nvfp4",
                "rank": 1,
                "variant": 1,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r0_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r0_cap16384_device.cu",
                "input_format": "nvfp4",
                "output_format": "bf16",
                "rank": 0,
                "variant": 0,
            },
            {
                "symbol": "kernel_fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r1_cap16384",
                "source": "fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r1_cap16384_device.cu",
                "input_format": "nvfp4",
                "output_format": "bf16",
                "rank": 1,
                "variant": 0,
            },
        ],
    },
}

COMPILE_FLAGS: list[str] = []

_ARCH_BY_CAPABILITY: dict[tuple[int, int], Fp4AllReduceArch] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "fp4_allreduce CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _table_key(world_size: int, family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"fp4_allreduce families are {FAMILIES}, got {family!r}")
    base = f"tp{int(world_size)}"
    return base if family == "allreduce" else f"{base}_{family}"


def world_sizes(family: str = "allreduce") -> tuple[int, ...]:
    """World sizes delivered for ``family``."""
    if family == "allreduce":
        return WORLD_SIZES
    if family == "rmsnorm":
        return RMSNORM_WORLD_SIZES
    raise ValueError(f"fp4_allreduce families are {FAMILIES}, got {family!r}")


def program_table(world_size: int, family: str = "allreduce") -> dict:
    """The delivered program set of one (family, world size) (``PROGRAMS["tp<world_size>[_<family>]"]``)."""
    key = _table_key(world_size, family)
    if key not in PROGRAMS:
        raise ValueError(
            f"fp4_allreduce family {family!r} supports world sizes {world_sizes(family)}, got {world_size}"
        )
    return PROGRAMS[key]


def kernel_names(
    world_size: int | None = None, family: str = "allreduce"
) -> tuple[str, ...]:
    """Kernel symbols of one (family, world size) (or of every world size of the family)."""
    sizes = world_sizes(family) if world_size is None else (int(world_size),)
    return tuple(
        kernel["symbol"]
        for size in sizes
        for kernel in program_table(size, family)["kernels"]
    )


def kernel_name(
    world_size: int,
    input_format: str,
    rank: int,
    variant: int,
    family: str = "allreduce",
    output_format: str = "bf16",
) -> str:
    """Kernel symbol of one (family, world size, input format, output format, rank, phase-B variant)."""
    for kernel in program_table(world_size, family)["kernels"]:
        if (
            kernel["input_format"] == input_format
            and kernel.get("output_format", "bf16") == output_format
            and int(kernel["rank"]) == int(rank)
            and int(kernel["variant"]) == int(variant)
        ):
            return kernel["symbol"]
    raise KeyError(
        f"no fp4_allreduce program for family={family!r} world_size={world_size} input_format={input_format!r} "
        f"output_format={output_format!r} rank={rank} variant={variant}"
    )


@functools.cache
def target_arch(device_index: int) -> Fp4AllReduceArch:
    """The generated architecture of one CUDA device (cached per index)."""
    capability = torch.cuda.get_device_capability(device_index)
    arch = _ARCH_BY_CAPABILITY.get(capability)
    if arch is None:
        raise ValueError(
            "fp4_allreduce requires SM100, SM103 or SM107, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return arch


def phase_b_variant(
    rows: int,
    grid: int,
    world_size: int,
    family: str = "allreduce",
    output_format: str = "bf16",
) -> int:
    """The program variant the binding selects for ``rows`` on ``grid`` CTAs.

    The two-shot (world size 4) all-reduce kernels come in three phase-B
    variants chosen from the rows each CTA owns; the one-shot all-reduce
    (world size 2) has a single program.  The fused-norm family runs its
    probe-all receive-poll program (variant 1) for the NVFP4 / MXFP8 outputs
    when a CTA owns more than ``probe_min_rows_per_cta`` rows and the grouped
    program (variant 0) otherwise; its bfloat16 output has one program.  Every
    rank evaluates this rule on the same grid, so every rank runs the same
    variant.
    """
    rule = program_table(world_size, family)["variant_rule"]
    if rule is None:
        return 0
    per_cta = int(rows) // int(grid)
    if family == "rmsnorm":
        if output_format not in rule["outputs"]:
            return 0
        return 1 if per_cta > int(rule["probe_min_rows_per_cta"]) else 0
    if per_cta > int(rule["by_quad_min"]):
        return 2
    if per_cta > int(rule["team_max"]):
        return 0
    return 1


def _jit_spec(world_size: int, arch: Fp4AllReduceArch, family: str) -> JitSpec:
    table = program_table(world_size, family)
    source_dir = _source_dir()
    sources = [source_dir / kernel["source"] for kernel in table["kernels"]]
    sources.append(source_dir / table["binding"])
    suffix = "" if family == "allreduce" else f"_{family}"
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}{suffix}_tp{int(world_size)}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*_NVCC_FLAGS[arch], *COMPILE_FLAGS],
        use_fast_math=False,
    )


@functools.cache
def gen_fp4_allreduce_module(world_size: int, arch: Fp4AllReduceArch) -> JitSpec:
    """One JIT module per (world size, architecture) of the all-reduce family: the generated device units plus the binding."""
    return _jit_spec(int(world_size), arch, "allreduce")


@functools.cache
def get_fp4_allreduce_module(world_size: int, arch: Fp4AllReduceArch):
    return gen_fp4_allreduce_module(int(world_size), arch).build_and_load()


@functools.cache
def gen_fp4_allreduce_rmsnorm_module(
    world_size: int, arch: Fp4AllReduceArch
) -> JitSpec:
    """One JIT module per (world size, architecture) of the fused-norm family: the generated device units plus the binding."""
    return _jit_spec(int(world_size), arch, "rmsnorm")


@functools.cache
def get_fp4_allreduce_rmsnorm_module(world_size: int, arch: Fp4AllReduceArch):
    return gen_fp4_allreduce_rmsnorm_module(int(world_size), arch).build_and_load()


def get_module(world_size: int, arch: Fp4AllReduceArch, family: str = "allreduce"):
    """The built module of one (family, world size, architecture)."""
    if family == "allreduce":
        return get_fp4_allreduce_module(int(world_size), arch)
    if family == "rmsnorm":
        return get_fp4_allreduce_rmsnorm_module(int(world_size), arch)
    raise ValueError(f"fp4_allreduce families are {FAMILIES}, got {family!r}")


__all__ = [
    "CAPACITY",
    "COMPILE_FLAGS",
    "FAMILIES",
    "HIDDEN",
    "INPUT_FORMATS",
    "OUTPUT_FORMATS",
    "PROGRAMS",
    "RECORD_BYTES",
    "RMSNORM_EPS",
    "RMSNORM_WEIGHT_BIAS",
    "RMSNORM_WORLD_SIZES",
    "SOURCE_PACKAGE",
    "WORLD_SIZES",
    "gen_fp4_allreduce_module",
    "gen_fp4_allreduce_rmsnorm_module",
    "get_fp4_allreduce_module",
    "get_fp4_allreduce_rmsnorm_module",
    "get_module",
    "kernel_name",
    "kernel_names",
    "phase_b_variant",
    "program_table",
    "target_arch",
    "world_sizes",
]
