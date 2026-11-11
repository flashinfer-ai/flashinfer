# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

BlackwellBf16BmmTarget = Literal["sm100a", "sm103a"]
_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}

# One source set serves both exact targets: the generated kernels carry no
# architecture-specific text, so each target compiles the same translation
# units with its own exact nvcc flag set and gets its own cached library.
_SOURCE_DIR = "cake_bf16_bmm"
_BINDING = "cake_bf16_bmm_binding.cu"
_KERNELS = (
    "cake_bf16_bmm_007557bb28b86b0ad447_kernel.cu",
    "cake_bf16_bmm_1428cc8db4c88a1a4faf_kernel.cu",
    "cake_bf16_bmm_148ff052ab7f16459f74_kernel.cu",
    "cake_bf16_bmm_215b1124d8a15af5cdd3_kernel.cu",
    "cake_bf16_bmm_49bbceba1010338dc287_kernel.cu",
    "cake_bf16_bmm_5143d7dde41d5c40f467_kernel.cu",
    "cake_bf16_bmm_5ca33069bbbad5c7016d_kernel.cu",
    "cake_bf16_bmm_5fb134eefb0c3d4d3572_kernel.cu",
    "cake_bf16_bmm_60c16df3a3f9b440c477_kernel.cu",
    "cake_bf16_bmm_6bb5fd7ca479b4132d28_kernel.cu",
    "cake_bf16_bmm_6d8fa1a07f5f07cab697_kernel.cu",
    "cake_bf16_bmm_73c9472b3d1b2ce56afe_kernel.cu",
    "cake_bf16_bmm_74b735e0c2a4a0ab11b5_kernel.cu",
    "cake_bf16_bmm_75d6b3b338da1b83fdd2_kernel.cu",
    "cake_bf16_bmm_82a7d836ee60f3f8e54d_kernel.cu",
    "cake_bf16_bmm_86f88c11bbab28a112a4_kernel.cu",
    "cake_bf16_bmm_8bbbf16d966b721a5216_kernel.cu",
    "cake_bf16_bmm_971934c57c7ec8e277e8_kernel.cu",
    "cake_bf16_bmm_9fc7d7b7d693e3601c8e_kernel.cu",
    "cake_bf16_bmm_a620001ffb4be97be504_kernel.cu",
    "cake_bf16_bmm_aae1fbf4100e7044666d_kernel.cu",
    "cake_bf16_bmm_b3614c1b24fcc0d9db41_kernel.cu",
    "cake_bf16_bmm_b45cfbc87416c25a8d38_kernel.cu",
    "cake_bf16_bmm_bc0d392cafbe0e91f5aa_kernel.cu",
    "cake_bf16_bmm_bda02d92fb3e9e813ac2_kernel.cu",
    "cake_bf16_bmm_beecb1dd16619150e5dc_kernel.cu",
    "cake_bf16_bmm_cb886960b55646251e65_kernel.cu",
    "cake_bf16_bmm_cc44de244d2ddb40e3e0_kernel.cu",
    "cake_bf16_bmm_cd337271d0ea1ad303f9_kernel.cu",
    "cake_bf16_bmm_d1f4a5edaf1b6f5d5e54_kernel.cu",
    "cake_bf16_bmm_d53a138927da127b43f3_kernel.cu",
    "cake_bf16_bmm_daddfc793176eeaf96ba_kernel.cu",
    "cake_bf16_bmm_dd1aa52c1efb755d163e_kernel.cu",
    "cake_bf16_bmm_ed865140f48d71197566_kernel.cu",
)


def gen_blackwell_bf16_bmm_module(target: BlackwellBf16BmmTarget) -> JitSpec:
    if target not in _FLAGS:
        raise ValueError(f"unsupported CAKE BF16 BMM target: {target}")
    checkout = Path(__file__).resolve().parents[3]
    source_root = checkout / "csrc"
    include_root = checkout / "include"
    if not source_root.is_dir():
        source_root = jit_env.FLASHINFER_CSRC_DIR
        include_root = jit_env.FLASHINFER_INCLUDE_DIR
    source_dir = source_root / _SOURCE_DIR
    # Preserve the public AOT registration; native Ninja owns JIT freshness.
    name = f"blackwell_bf16_bmm_cake_{target}"
    return gen_jit_spec(
        name,
        [source_dir / _BINDING] + [source_dir / p for p in _KERNELS],
        extra_cuda_cflags=_FLAGS[target] + ["--use_fast_math"],
        extra_include_paths=[source_root, source_dir, include_root],
    )


__all__ = ["gen_blackwell_bf16_bmm_module"]
