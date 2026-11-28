# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""JIT specs for the fused TP MoE kernels (see flashinfer/fused_moe/tp_moe.py)."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_tp_moe_sm90_module(variant) -> JitSpec:
    """One SM90a module per (model, TP size, M-range variant); ``variant`` is a
    ``flashinfer.fused_moe._tp_moe_sm90.Variant``.

    The numerics are pinned by explicit flags rather than inherited defaults:
    no ``-use_fast_math`` (the kernels write their fast-math operations as
    explicit PTX where the reference numerics need them, everything else is
    IEEE: FTZ / approximate division must not leak in), ``-O3``, sm_90a only.
    """
    sources = {
        "dsv4_flash": "tp_moe_sm90_flash.cu",
        "dsv41_flash": "tp_moe_sm90_dsv41.cu",
    }
    src_dir = jit_env.FLASHINFER_CSRC_DIR / "tp_moe" / "sm90"
    flags = [f"-DTP_MOE_NDEV={variant.tp_size}"]
    flags += [f"-D{name}={value}" for name, value in variant.flags]
    return gen_jit_spec(
        variant.name,
        [src_dir / sources[variant.model]],
        extra_cuda_cflags=[
            "-gencode=arch=compute_90a,code=sm_90a",
            "-O3",
            "--expt-relaxed-constexpr",
            *flags,
        ],
        extra_include_paths=[src_dir],
        use_fast_math=False,
    )


def gen_tp_moe_sm100_module(build) -> JitSpec:
    """One SM100a kernel build (``flashinfer.fused_moe._tp_moe_sm100.Build``): the family's host entry compiled with the
    build's ``-DTPMOE_*`` flags (model shape, TP size, token range, entry, per-M tuning)."""
    src = jit_env.FLASHINFER_CSRC_DIR / "tp_moe" / "sm100" / f"{build.family}.cu"
    return gen_jit_spec(
        build.uri,
        [src],
        extra_cuda_cflags=[
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            # the validated builds' device-code environment (torch extension defaults)
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            *(f"-DTPMOE_{name}={value}" for name, value in build.flags),
        ],
        extra_ldflags=["-lcuda"],
        # IEEE fp32 (no flush-to-zero, exact division / sqrt): the MXFP8 / UE8M0 scale arithmetic depends on it
        use_fast_math=False,
    )
