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

"""JIT compilation specification for the experimental VibeCUDA KDA prefill backend."""

import functools
import hashlib
from pathlib import Path
from typing import Literal

from ...jit._kda_jit_common import (
    get_flashinfer_include_dir,
    get_kda_csrc_dir,
)
from ...jit.core import (
    JitSpec,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

VibeCUDAPrefillTarget = Literal["sm100a", "sm103a"]

_CSRC = Path(__file__).resolve().parent / "csrc"

_VIBECUDA_FLASH_KDA_SOURCES: tuple[str, ...] = (
    "vibecuda_flashkda_binding.cu",
    "vibecuda_flashkda_bf16_fused_m128_binding.cu",
    "vibecuda_flashkda_bf16_fused_m128_slab_binding.cu",
    "vibecuda_flashkda_bf16_fused_m64_binding.cu",
    "vibecuda_flashkda_bf16_persistent_m128_binding.cu",
    "vibecuda_flashkda_fam2_binding_h12.cu",
    "vibecuda_flashkda_fam2_binding_h64.cu",
    "vibecuda_flashkda_fam2_binding_h96.cu",
    "vibecuda_bt16_binding_fused_prepare.cu",
    "vibecuda_bt16_binding_fused_chain_s7.cu",
    "vibecuda_bt16_binding_fused_chain_s8.cu",
    "vibecuda_bt16_binding_fused_chain_s9.cu",
)

_VIBECUDA_FLASH_KDA_EXTRA_HASH_FILES: tuple[str, ...] = (
    "vibecuda_flashkda_tma.cuh",
    "vibecuda_flashkda_bf16_fused_m128.cu",
    "vibecuda_flashkda_bf16_fused_m128_slab.cu",
    "vibecuda_flashkda_bf16_fused_m64.cu",
    "vibecuda_flashkda_bf16_persistent_m128.cu",
    "vibecuda_flashkda_fam2_gen.cu",
    "vibecuda_flashkda_fam2h12_gen.cu",
    "vibecuda_bt16_tma.cuh",
    "vibecuda_cakebt16_prepare_gen.cu",
    "vibecuda_cakebt16_chain_s7_gen.cu",
    "vibecuda_cakebt16_chain_s8_gen.cu",
    "vibecuda_cakebt16_chain_s9_gen.cu",
)

_VIBECUDA_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

_VIBECUDA_TARGET_DEFINES = {
    "sm100a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=3",
}


def get_vibecuda_csrc_dir() -> Path:
    """Return the experimental VibeCUDA KDA CUDA source directory."""

    if _CSRC.exists():
        return _CSRC
    raise FileNotFoundError(
        f"Experimental VibeCUDA KDA CUDA sources were not found at {_CSRC}"
    )


@functools.cache
def _get_vibecuda_flash_kda_digest() -> str:
    csrc_dir = get_vibecuda_csrc_dir()
    kda_csrc_dir = get_kda_csrc_dir()
    hasher = hashlib.sha256()
    for name in _VIBECUDA_FLASH_KDA_SOURCES + _VIBECUDA_FLASH_KDA_EXTRA_HASH_FILES:
        hasher.update((csrc_dir / name).read_bytes())
        hasher.update(b"\0")
    hasher.update((kda_csrc_dir / "flashkda_binding_common.cuh").read_bytes())
    hasher.update(b"\0")
    return hasher.hexdigest()[:10]


def get_vibecuda_flash_kda_uri(target: VibeCUDAPrefillTarget = "sm100a") -> str:
    if target not in _VIBECUDA_TARGET_DEFINES:
        raise ValueError(
            "VibeCUDA FlashKDA prefill supports target in {'sm100a', 'sm103a'}, "
            f"got {target!r}"
        )
    digest = _get_vibecuda_flash_kda_digest()
    return f"flash_kda_bf16_vibecuda_{digest}_{target}"


@functools.cache
def gen_vibecuda_flash_kda_module(
    target: VibeCUDAPrefillTarget = "sm100a",
) -> JitSpec:
    if target not in _VIBECUDA_TARGET_DEFINES:
        raise ValueError(
            "VibeCUDA FlashKDA prefill supports target in {'sm100a', 'sm103a'}, "
            f"got {target!r}"
        )
    csrc_dir = get_vibecuda_csrc_dir()
    kda_csrc_dir = get_kda_csrc_dir()
    include_dir = get_flashinfer_include_dir()
    sources = [csrc_dir / name for name in _VIBECUDA_FLASH_KDA_SOURCES]
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        raise FileNotFoundError(
            "VibeCUDA FlashKDA prefill sources were not found:\n  - "
            + "\n  - ".join(missing)
        )
    # Round-230+: KDA_FUSE_FIXUP is compiled unconditionally on both sm100a and
    # sm103a so split launches fold the carry contribution in the correction
    # epilogue without allocating an out-sized split_out scratch buffer.
    extra_cflags = ["-O3", "-Xptxas=-v", "-DKDA_FUSE_FIXUP"]
    if target == "sm103a":
        extra_cflags.extend(["-DKDA_SM103", "-DKDA_W8PF", "-DKDA_FIN_SPLIT"])
    return gen_jit_spec(
        name=get_vibecuda_flash_kda_uri(target),
        sources=sources,
        extra_cuda_cflags=[
            *_VIBECUDA_NVCC_FLAGS[target],
            _VIBECUDA_TARGET_DEFINES[target],
            *extra_cflags,
        ],
        extra_include_paths=[
            csrc_dir,
            kda_csrc_dir,
            kda_csrc_dir.parent,
            include_dir,
        ],
    )


@functools.cache
def load_vibecuda_flash_kda_module(target: VibeCUDAPrefillTarget = "sm100a"):
    return gen_vibecuda_flash_kda_module(target).build_and_load()
