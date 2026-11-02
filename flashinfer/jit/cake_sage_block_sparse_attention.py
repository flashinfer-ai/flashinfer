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


"""JIT loader facade of the Cake Sage block-sparse attention package.

The kernels ship in :mod:`flashinfer.cake_sage_block_sparse_attention_backend`:
``build.json`` names every library (sources, flags, digests), ``cake_build.py``
builds or loads it and the ``cake_jit`` registry records what each module
launches.  This module keeps the names the rest of FlashInfer and its tests
import, each a thin call into that package:

* ``SM120_MODULES`` -- specialization key ``(HAS_BLOCK_NUMS, BLOCK_SIZES_MODE,
  FULL_K64_TILES, UNIFORM_NONEMPTY, CONTIGUOUS_BLOCK_INDICES)`` -> the library
  (JIT module) name of the SM120 (RTX PRO 6000 / RTX 5090) ``mma.sync`` kernel,
  read off the registry's ``select_rule`` records.  Every specialization is one
  shared kernel source and one shared launcher compiled under its own compile
  definitions (``sm120_compile_definitions``, the ``-D`` flags of its manifest
  entry): ``CAKE_SAGE_SM120_KERNEL`` names its kernel symbol and the five
  ``CAKE_SAGE_*`` flag macros select its text.
* ``SM100_ARCHES`` / ``SM100_STAGES`` and ``gen_cake_sage_sm100_module`` /
  ``load_cake_sage_sm100_module`` -- the SM100-family (B200 / B300) ``tcgen05``
  attention and the two fused Sage-FP8 quantizers, compiled for ``sm_100a`` and
  ``sm_103a``.
"""

from __future__ import annotations

import functools
from typing import Any

from flashinfer.cake_sage_block_sparse_attention_backend import cake_build, cake_jit

SM100_ARCHES = tuple(cake_jit.SM100_ARCHES)
# stage -> the registry record (and build.json library stem) of the SM100 family.
SM100_STAGES = ("attention", "quantize_qk_vamax", "quantize_v")
SM120_KERNEL_MACRO = "CAKE_SAGE_SM120_KERNEL"

Sm120Key = tuple[int, int, int, int, int]


def _sm120_key(*flags: Any) -> Sm120Key:
    values = tuple(int(flag) for flag in flags)
    if len(values) != 5:
        raise RuntimeError(f"an SM120 specialization key has five flags, got {values}")
    return values  # type: ignore[return-value]


# (HAS_BLOCK_NUMS, BLOCK_SIZES_MODE, FULL_K64_TILES, UNIFORM_NONEMPTY,
#  CONTIGUOUS_BLOCK_INDICES) -> SM120 library name, one row per registry record
# of role ``attention_sm120``.  CONTIGUOUS_BLOCK_INDICES is set only together
# with UNIFORM_NONEMPTY: with no block to load, the contiguous specialization
# writes the same zeros as its (..., 0, 0) sibling, so the dispatch rule
# (``cake_jit.sm120_module_key``) clears the flag instead.
SM120_MODULES: dict[Sm120Key, str] = {
    key: cake_jit.MODULES[record]["library"]
    for key, record in sorted(cake_jit.SM120_RECORDS.items())
}


def sm120_compile_definitions(key: Sm120Key) -> list[str]:
    """Per-module compile definitions of one SM120 specialization.

    The ``-D`` flags of the specialization's ``build.json`` entry: the first
    names the module's kernel symbol (``-DCAKE_SAGE_SM120_KERNEL=kernel_<name>``),
    the next five set the flag macros of the shared kernel source to the key's
    values, in key order.
    """
    library = f"{SM120_MODULES[_sm120_key(*key)]}_{cake_jit.SM120_ARCH}"
    entry = cake_build.libraries_by_name()[library]
    return [flag for flag in entry["nvcc_flags"] if flag.startswith("-D")]


@functools.cache
def gen_cake_sage_block_sparse_attention_module(
    has_block_nums: int,
    block_sizes_mode: int,
    full_k64_tiles: int,
    uniform_nonempty: int,
    contiguous_block_indices: int,
) -> Any:
    """The JIT spec of one SM120 specialization (``cake_build.jit_spec``)."""
    key = _sm120_key(
        has_block_nums,
        block_sizes_mode,
        full_k64_tiles,
        uniform_nonempty,
        contiguous_block_indices,
    )
    name = SM120_MODULES.get(key)
    if name is None:
        raise RuntimeError(
            f"no generated SM120 Sage block-sparse attention module for specialization {key}"
        )
    return cake_build.jit_spec(f"{name}_{cake_jit.SM120_ARCH}")


@functools.cache
def load_cake_sage_block_sparse_attention_module(
    has_block_nums: int,
    block_sizes_mode: int,
    full_k64_tiles: int,
    uniform_nonempty: int,
    contiguous_block_indices: int,
):
    """Build (or load the prebuilt) and return the SM120 module of one specialization."""
    record = cake_jit.select_sm120(
        has_block_nums=has_block_nums,
        block_sizes_mode=block_sizes_mode,
        full_k64_tiles=full_k64_tiles,
        uniform_nonempty=uniform_nonempty,
        contiguous_block_indices=contiguous_block_indices,
    )
    return cake_jit.load(record, cake_jit.SM120_ARCH)


def _sm100_record(arch: str, stage: str) -> str:
    if arch not in SM100_ARCHES:
        raise RuntimeError(
            f"Cake SM100 Sage block-sparse attention supports {list(SM100_ARCHES)}, got {arch!r}"
        )
    if stage not in SM100_STAGES:
        raise RuntimeError(
            f"unknown Cake SM100 Sage stage {stage!r}; expected one of {SM100_STAGES}"
        )
    return stage


@functools.cache
def gen_cake_sage_sm100_module(arch: str, stage: str) -> Any:
    """The JIT spec of the (``arch``, ``stage``) SM100-family module."""
    record = _sm100_record(arch, stage)
    return cake_build.jit_spec(cake_jit.library_name(record, arch))


@functools.cache
def load_cake_sage_sm100_module(arch: str, stage: str):
    """Build (or load the prebuilt) and return the (``arch``, ``stage``) SM100-family module."""
    return cake_jit.load(_sm100_record(arch, stage), arch)


__all__ = [
    "SM100_ARCHES",
    "SM100_STAGES",
    "SM120_KERNEL_MACRO",
    "SM120_MODULES",
    "gen_cake_sage_block_sparse_attention_module",
    "gen_cake_sage_sm100_module",
    "load_cake_sage_block_sparse_attention_module",
    "load_cake_sage_sm100_module",
    "sm120_compile_definitions",
]
