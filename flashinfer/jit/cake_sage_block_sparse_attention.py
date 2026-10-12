# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""JIT loader for generated Cake Sage block-sparse attention kernels.

Sources live under ``csrc/cake_sage_block_sparse_attention``:

* ``sm_120a/`` -- the SM120 (RTX PRO 6000 / RTX 5090) ``mma.sync`` kernels, one
  kernel source per specialization of the five metadata flags and one launcher
  source shared by all of them (the kernel symbol is passed as a compile
  definition).
* ``sm_100a/`` -- the SM100-family (B200 / B300) variable-shape ``tcgen05``
  attention kernel plus the two fused Sage-FP8 quantization kernels, one
  kernel/launcher pair per stage, compiled for ``sm_100a`` and ``sm_103a``
  (the few lowering differences sit under ``__CUDA_ARCH__`` guards).
"""

from __future__ import annotations

import functools
from pathlib import Path

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm120a_nvcc_flags,
)

_GENERATED_ROOT = "csrc/cake_sage_block_sparse_attention"
_PREFIX = "cake_sage_block_sparse_attention"
_COMPILE_FLAGS = ("--use_fast_math",)

# (HAS_BLOCK_NUMS, BLOCK_SIZES_MODE, FULL_K64_TILES, UNIFORM_NONEMPTY,
#  CONTIGUOUS_BLOCK_INDICES) -> SM120 kernel source under sm_120a/. The five flags
# are folded into the generated kernel body; every reachable combination has one
# entry, and the kernel symbol is ``kernel_<source stem without _kernel>``.
SM120_MODULES: dict[tuple[int, int, int, int, int], str] = {
    (0, 0, 0, 0, 0): "cake_sage_block_sparse_attention_c214204e0e4b13466e20_kernel.cu",
    (0, 0, 0, 0, 1): "cake_sage_block_sparse_attention_0f9c7da2c88932b636d1_kernel.cu",
    (0, 0, 0, 1, 0): "cake_sage_block_sparse_attention_5e1d9bd7d208ae70cfc3_kernel.cu",
    (0, 0, 0, 1, 1): "cake_sage_block_sparse_attention_883c6c3450cc823b2d9b_kernel.cu",
    (0, 0, 1, 0, 0): "cake_sage_block_sparse_attention_ac8c0628f84e77fb565d_kernel.cu",
    (0, 0, 1, 0, 1): "cake_sage_block_sparse_attention_33204ab76367b6d9b25f_kernel.cu",
    (0, 0, 1, 1, 0): "cake_sage_block_sparse_attention_0a00c8bb89c685b0c222_kernel.cu",
    (0, 0, 1, 1, 1): "cake_sage_block_sparse_attention_2adb38e9fb64cab29bf4_kernel.cu",
    (0, 1, 0, 0, 0): "cake_sage_block_sparse_attention_70d9ca316721813db07f_kernel.cu",
    (0, 1, 0, 0, 1): "cake_sage_block_sparse_attention_4379d118d88e49a36ec3_kernel.cu",
    (0, 1, 0, 1, 0): "cake_sage_block_sparse_attention_08683144a7b6c64433ad_kernel.cu",
    (0, 1, 0, 1, 1): "cake_sage_block_sparse_attention_aa9589fa9d54f340e6b1_kernel.cu",
    (0, 2, 0, 0, 0): "cake_sage_block_sparse_attention_6a1fcc34a68d31545b23_kernel.cu",
    (0, 2, 0, 0, 1): "cake_sage_block_sparse_attention_59d2437f3af320cc32c8_kernel.cu",
    (0, 2, 0, 1, 0): "cake_sage_block_sparse_attention_3e61674cc1a4b6793646_kernel.cu",
    (0, 2, 0, 1, 1): "cake_sage_block_sparse_attention_8b24fd2b689f1ead23c9_kernel.cu",
    (0, 3, 0, 0, 0): "cake_sage_block_sparse_attention_82751a008faeeeb37778_kernel.cu",
    (0, 3, 0, 0, 1): "cake_sage_block_sparse_attention_184ad19669dcb84b24ff_kernel.cu",
    (0, 3, 0, 1, 0): "cake_sage_block_sparse_attention_a17d0e6f30423c654a19_kernel.cu",
    (0, 3, 0, 1, 1): "cake_sage_block_sparse_attention_813b7e41eaabcd64b34d_kernel.cu",
    (1, 0, 0, 0, 0): "cake_sage_block_sparse_attention_386f636f65fcfae07807_kernel.cu",
    (1, 0, 1, 0, 0): "cake_sage_block_sparse_attention_2069e62b1e24fb1be284_kernel.cu",
    (1, 1, 0, 0, 0): "cake_sage_block_sparse_attention_bd48d91a04a229511902_kernel.cu",
    (1, 2, 0, 0, 0): "cake_sage_block_sparse_attention_073e7b711576560b267c_kernel.cu",
    (1, 3, 0, 0, 0): "cake_sage_block_sparse_attention_f9e50348d20ed28e5654_kernel.cu",
}
_SM120_BINDING = "cake_sage_block_sparse_attention_binding.cu"

SM100_ARCHES = ("sm_100a", "sm_103a")
SM100_STAGES = ("attention", "quantize_qk_vamax", "quantize_v")
# stage -> (kernel source, launcher source) under sm_100a/, shared by both arches.
_SM100_SOURCES = {
    "attention": (
        "cake_sage_block_sparse_attention_attention_kernel.cu",
        "cake_sage_block_sparse_attention_attention_binding.cu",
    ),
    "quantize_qk_vamax": (
        "cake_sage_block_sparse_attention_quantize_qk_vamax_kernel.cu",
        "cake_sage_block_sparse_attention_quantize_qk_vamax_binding.cu",
    ),
    "quantize_v": (
        "cake_sage_block_sparse_attention_quantize_v_kernel.cu",
        "cake_sage_block_sparse_attention_quantize_v_binding.cu",
    ),
}
_SM100_ARCH_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
# Module ids per (arch, stage): the JIT spec names (and hence the build cache
# directories) are unchanged from the per-architecture delivery they replace.
_SM100_MODULE_IDS = {
    ("sm_100a", "attention"): "f63b7fc8674de9925f7e",
    ("sm_100a", "quantize_qk_vamax"): "e46b194b07d454419fba",
    ("sm_100a", "quantize_v"): "43e2bcadf25e569b50fb",
    ("sm_103a", "attention"): "62c3142a797348e374c0",
    ("sm_103a", "quantize_qk_vamax"): "7374900a242cff867ec4",
    ("sm_103a", "quantize_v"): "d961cdf9b05b70c09c59",
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_sage_block_sparse_attention"
    if (installed / "sm_120a" / _SM120_BINDING).is_file():
        return installed
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_sage_block_sparse_attention"
    )
    if (checkout / "sm_120a" / _SM120_BINDING).is_file():
        return checkout
    raise FileNotFoundError("Cake Sage block-sparse attention sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _spec(name: str, sources: list[Path], arch_flags, extra: list[str]) -> JitSpec:
    spec = gen_jit_spec(
        name=name,
        sources=sources,
        extra_cuda_cflags=[*arch_flags, *_COMPILE_FLAGS, *extra],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[_get_csrc_dir().parent, _get_include_dir()],
    )
    logger.info("Generated Cake Sage block-sparse attention JIT spec: %s", spec.name)
    return spec


@functools.cache
def gen_cake_sage_block_sparse_attention_module(
    has_block_nums: int,
    block_sizes_mode: int,
    full_k64_tiles: int,
    uniform_nonempty: int,
    contiguous_block_indices: int,
) -> JitSpec:
    key = (
        int(has_block_nums),
        int(block_sizes_mode),
        int(full_k64_tiles),
        int(uniform_nonempty),
        int(contiguous_block_indices),
    )
    source = SM120_MODULES.get(key)
    if source is None:
        raise RuntimeError(
            f"no generated SM120 Sage block-sparse attention module for specialization {key}"
        )
    name = source.removesuffix("_kernel.cu")
    root = _get_csrc_dir() / "sm_120a"
    return _spec(
        name,
        [root / source, root / _SM120_BINDING],
        sm120a_nvcc_flags,
        [f"-DCAKE_SAGE_SM120_KERNEL=kernel_{name}"],
    )


@functools.cache
def load_cake_sage_block_sparse_attention_module(
    has_block_nums: int,
    block_sizes_mode: int,
    full_k64_tiles: int,
    uniform_nonempty: int,
    contiguous_block_indices: int,
):
    """Build and load the SM120 module of one specialization."""
    return gen_cake_sage_block_sparse_attention_module(
        has_block_nums,
        block_sizes_mode,
        full_k64_tiles,
        uniform_nonempty,
        contiguous_block_indices,
    ).build_and_load()


@functools.cache
def gen_cake_sage_sm100_module(arch: str, stage: str) -> JitSpec:
    if arch not in _SM100_ARCH_FLAGS:
        raise RuntimeError(
            f"Cake SM100 Sage block-sparse attention supports {list(SM100_ARCHES)}, got {arch!r}"
        )
    if stage not in SM100_STAGES:
        raise RuntimeError(
            f"unknown Cake SM100 Sage stage {stage!r}; expected one of {SM100_STAGES}"
        )
    root = _get_csrc_dir() / "sm_100a"
    return _spec(
        f"{_PREFIX}_{_SM100_MODULE_IDS[(arch, stage)]}_{arch}",
        [root / name for name in _SM100_SOURCES[stage]],
        _SM100_ARCH_FLAGS[arch],
        [],
    )


@functools.cache
def load_cake_sage_sm100_module(arch: str, stage: str):
    """Build and load the (``arch``, ``stage``) SM100-family module."""
    return gen_cake_sage_sm100_module(arch, stage).build_and_load()


__all__ = [
    "SM100_ARCHES",
    "SM100_STAGES",
    "SM120_MODULES",
    "gen_cake_sage_block_sparse_attention_module",
    "gen_cake_sage_sm100_module",
    "load_cake_sage_block_sparse_attention_module",
    "load_cake_sage_sm100_module",
]
