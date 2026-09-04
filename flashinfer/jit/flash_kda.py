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

import functools
from typing import Literal

from ._kda_jit_common import (
    gen_kda_jit_spec,
    get_flashinfer_include_dir as _get_flash_kda_include_dir,
    get_kda_csrc_dir as _get_flash_kda_csrc_dir,
)
from .core import JitSpec, logger

FlashKDAVariant = Literal[
    "m64",
    "m128",
    "m128_tensor_state_decay",
    "m128_h12_short",
    "m128_h12_long",
    "m128_n16",
    "m128_n16_checkpoint",
    "m128_n16_short",
    "persistent_m128",
    "piece_persistent_m128",
    "small_bh_m128",
    "bt16_prepare",
    "bt16_prepare_beta_tma",
    "bt16_chain_m64_s7",
    "bt16_chain_m64_s8",
    "bt16_chain_m64_s9",
    "bt16_prepare_chain_m64_s8",
]
FlashKDATarget = Literal["sm100a", "sm100f"]

FLASH_KDA_VARIANTS: tuple[FlashKDAVariant, ...] = (
    "m64",
    "m128",
    "m128_tensor_state_decay",
    "m128_h12_short",
    "m128_h12_long",
    "m128_n16",
    "m128_n16_checkpoint",
    "m128_n16_short",
    "persistent_m128",
    "piece_persistent_m128",
    "small_bh_m128",
    "bt16_prepare",
    "bt16_prepare_beta_tma",
    "bt16_chain_m64_s7",
    "bt16_chain_m64_s8",
    "bt16_chain_m64_s9",
    "bt16_prepare_chain_m64_s8",
)

_FLASH_KDA_TARGETS: tuple[FlashKDATarget, ...] = ("sm100a", "sm100f")
_FLASH_KDA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
    "sm100f": "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100",
}

# Keep every frozen cache key tied to its complete generated-plus-integration
# implementation. This prevents an installed JIT/AOT cache from satisfying a
# refreshed export or binding specialization after an in-place package upgrade.
_FLASH_KDA_MODULE_IDENTS = {
    "m64": "41828f5029",
    "m128": "8ab0c31bf7",
    "m128_tensor_state_decay": "95731d19b1",
    "m128_h12_short": "93f0f54206",
    "m128_h12_long": "7d4006b429",
    "m128_n16": "749b6dff22",
    # Generated body, binding, and shared binding header, separated by NUL
    # bytes without a trailing separator. Keep this route's cache key tied to
    # all compiled content.
    "m128_n16_checkpoint": "8ee53de13d",
    "m128_n16_short": "d8a8cc97b2",
    "persistent_m128": "cba62a2b86",
    "piece_persistent_m128": "e2b3ec690d",
    "small_bh_m128": "6192b9d793",
    "bt16_prepare": "2c6cc4c1f6",
    "bt16_prepare_beta_tma": "d9394ce430",
    "bt16_chain_m64_s7": "350dbb8897",
    "bt16_chain_m64_s8": "9e1ea1ef2d",
    "bt16_chain_m64_s9": "e83ce16115",
    "bt16_prepare_chain_m64_s8": "6c392ef667",
}

_FLASH_KDA_BINDING_STEMS = {
    "m64": "flashkda_bf16_fused_m64",
    "m128": "flashkda_bf16_fused_m128",
    "m128_tensor_state_decay": "flashkda_bf16_fused_m128",
    "m128_h12_short": "cake_flashkda_bf16_fused_m128_h12",
    "m128_h12_long": "cake_flashkda_bf16_fused_m128_h12",
    "m128_n16": "cake_flashkda_bf16_fused_m128_n16",
    "m128_n16_checkpoint": "flashkda_bf16_fused_m128_n16_checkpoint",
    "m128_n16_short": "cake_flashkda_bf16_fused_m128_n16",
    "persistent_m128": "cake_flashkda_bf16_persistent_m128",
    "piece_persistent_m128": "cake_flashkda_bf16_piece_persistent_m128",
    "small_bh_m128": "cake_flashkda_bf16_small_bh_m128",
    "bt16_prepare": "cake_flashkda_bf16_bt16_prepare",
    "bt16_prepare_beta_tma": "cake_flashkda_bf16_bt16_prepare_beta_tma",
    "bt16_chain_m64_s7": "cake_flashkda_bf16_bt16_chain_m64_s7",
    "bt16_chain_m64_s8": "cake_flashkda_bf16_bt16_chain_m64",
    "bt16_chain_m64_s9": "cake_flashkda_bf16_bt16_chain_m64_s9",
}

_FLASH_KDA_VARIANT_DEFINES = {
    "m128_n16_short": "-DFLASHINFER_FLASH_KDA_N16_SHORT=1",
    "m128_tensor_state_decay": "-DFLASHINFER_FLASH_KDA_TENSOR_STATE_DECAY=1",
    "m128_h12_short": "-DFLASHINFER_FLASH_KDA_H12_SHORT=1",
    "m128_h12_long": "-DFLASHINFER_FLASH_KDA_H12_LONG=1",
}


def get_flash_kda_uri(variant: FlashKDAVariant, target: FlashKDATarget) -> str:
    """Return the target-specific JIT/AOT key for one schedule."""

    if variant not in FLASH_KDA_VARIANTS:
        raise ValueError(f"unsupported FlashKDA variant: {variant}")
    if target not in _FLASH_KDA_TARGETS:
        raise ValueError(f"unsupported FlashKDA target: {target}")
    module_ident = _FLASH_KDA_MODULE_IDENTS[variant]
    return f"flash_kda_bf16_{variant}_{module_ident}_{target}"


@functools.cache
def gen_flash_kda_module(variant: FlashKDAVariant, target: FlashKDATarget) -> JitSpec:
    """Generate one legacy exact-SM100a or SM100-family JIT module.

    Each physical schedule is compiled in its own translation unit because the
    checked-in frozen sources intentionally retain generated helper names and
    macros. ``gen_jit_spec`` supplies FlashInfer's standard ``-use_fast_math``
    flag. CUDA 12.8 uses the exact ``sm_100a`` target on B200. CUDA 12.9 and
    newer use one ``sm_100f`` target validated on CC 10.0 and CC 10.3.
    """

    csrc_dir = _get_flash_kda_csrc_dir()
    include_dir = _get_flash_kda_include_dir()
    uri = get_flash_kda_uri(variant, target)
    if variant == "bt16_prepare_chain_m64_s8":
        sources = [
            csrc_dir / "cake_flashkda_bf16_bt16_prepare_binding.cu",
            csrc_dir / "cake_flashkda_bf16_bt16_chain_m64_binding.cu",
            csrc_dir / "cake_flashkda_bf16_bt16_prepare_chain_m64_binding.cu",
        ]
    else:
        sources = [csrc_dir / f"{_FLASH_KDA_BINDING_STEMS[variant]}_binding.cu"]
    missing_sources = [source for source in sources if not source.exists()]
    if missing_sources:
        raise FileNotFoundError(
            f"FlashKDA binding source not found: {missing_sources[0]}"
        )

    extra_cuda_cflags = [
        *(
            [_FLASH_KDA_VARIANT_DEFINES[variant]]
            if variant in _FLASH_KDA_VARIANT_DEFINES
            else []
        ),
        *(
            ["-DFLASHINFER_FLASH_KDA_COMBINED_BT16=1"]
            if variant == "bt16_prepare_chain_m64_s8"
            else []
        ),
    ]
    spec = gen_kda_jit_spec(
        name=uri,
        sources=sources,
        target=target,
        target_define=_FLASH_KDA_TARGET_DEFINE[target],
        csrc_dir=csrc_dir,
        include_dir=include_dir,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    logger.info(f"Generated FlashKDA {variant} {target} JIT spec: {spec.name}")
    return spec


def gen_flash_kda_m64_module(target: FlashKDATarget) -> JitSpec:
    """Generate the fixed N=1, H=64 two-CTA M64 module."""

    return gen_flash_kda_module("m64", target)


def gen_flash_kda_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the general packed/fixed M128 module."""

    return gen_flash_kda_module("m128", target)


def gen_flash_kda_m128_tensor_state_decay_module(
    target: FlashKDATarget,
) -> JitSpec:
    """Generate the full-tile SM103 tensor state-decay M128 module."""

    return gen_flash_kda_module("m128_tensor_state_decay", target)


def gen_flash_kda_m128_h12_short_module(target: FlashKDATarget) -> JitSpec:
    """Generate the short-sequence H12 N32 M128 module."""

    return gen_flash_kda_module("m128_h12_short", target)


def gen_flash_kda_m128_h12_long_module(target: FlashKDATarget) -> JitSpec:
    """Generate the pair-packed-beta H12 N32 M128 module."""

    return gen_flash_kda_module("m128_h12_long", target)


def gen_flash_kda_m128_n16_module(target: FlashKDATarget) -> JitSpec:
    """Generate the H12 packed/fixed M128 module with a 16-token chunk."""

    return gen_flash_kda_module("m128_n16", target)


def gen_flash_kda_m128_n16_checkpoint_module(target: FlashKDATarget) -> JitSpec:
    """Generate the N16 M128 module with checkpoint TMA stores."""

    return gen_flash_kda_module("m128_n16_checkpoint", target)


def gen_flash_kda_m128_n16_short_module(target: FlashKDATarget) -> JitSpec:
    """Generate the generic one-tile M128 module with one N16 stage."""

    return gen_flash_kda_module("m128_n16_short", target)


def gen_flash_kda_persistent_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the SM100-only static-binned persistent M128 module."""

    return gen_flash_kda_module("persistent_m128", target)


def gen_flash_kda_piece_persistent_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the recurrence-piece persistent M128 module."""

    return gen_flash_kda_module("piece_persistent_m128", target)


def gen_flash_kda_small_bh_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the fixed-layout small-BH owner/helper M128 module."""

    return gen_flash_kda_module("small_bh_m128", target)


def gen_flash_kda_bt16_prepare_module(target: FlashKDATarget) -> JitSpec:
    """Generate the scalar-beta BT16 factor-preparation module."""

    return gen_flash_kda_module("bt16_prepare", target)


def gen_flash_kda_bt16_prepare_beta_tma_module(target: FlashKDATarget) -> JitSpec:
    """Generate the beta-TMA BT16 factor-preparation module."""

    return gen_flash_kda_module("bt16_prepare_beta_tma", target)


def gen_flash_kda_bt16_chain_m64_s7_module(target: FlashKDATarget) -> JitSpec:
    """Generate the two-resident S7 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s7", target)


def gen_flash_kda_bt16_chain_m64_s8_module(target: FlashKDATarget) -> JitSpec:
    """Generate the canonical S8 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s8", target)


def gen_flash_kda_bt16_chain_m64_s9_module(target: FlashKDATarget) -> JitSpec:
    """Generate the underfilled-grid S9 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s9", target)


def gen_flash_kda_bt16_prepare_chain_m64_s8_module(
    target: FlashKDATarget,
) -> JitSpec:
    """Generate the combined scalar-prepare plus S8 chain launcher."""

    return gen_flash_kda_module("bt16_prepare_chain_m64_s8", target)


@functools.cache
def load_flash_kda_module(variant: FlashKDAVariant, target: FlashKDATarget):
    """Build or load one physical, target-specific FlashKDA module."""

    module = gen_flash_kda_module(variant, target).build_and_load()
    logger.info(f"Loaded FlashKDA {variant} {target} module")
    return module


def load_flash_kda_m64_module(target: FlashKDATarget):
    """Load the fixed N=1, H=64 two-CTA M64 module."""

    return load_flash_kda_module("m64", target)


def load_flash_kda_m128_module(target: FlashKDATarget):
    """Load the general packed/fixed M128 module."""

    return load_flash_kda_module("m128", target)


def load_flash_kda_m128_tensor_state_decay_module(target: FlashKDATarget):
    """Load the full-tile SM103 tensor state-decay M128 module."""

    return load_flash_kda_module("m128_tensor_state_decay", target)


def load_flash_kda_m128_h12_short_module(target: FlashKDATarget):
    """Load the short-sequence H12 N32 M128 module."""

    return load_flash_kda_module("m128_h12_short", target)


def load_flash_kda_m128_h12_long_module(target: FlashKDATarget):
    """Load the pair-packed-beta H12 N32 M128 module."""

    return load_flash_kda_module("m128_h12_long", target)


def load_flash_kda_m128_n16_module(target: FlashKDATarget):
    """Load the H12 packed/fixed M128 module with a 16-token chunk."""

    return load_flash_kda_module("m128_n16", target)


def load_flash_kda_m128_n16_short_module(target: FlashKDATarget):
    """Load the generic one-tile M128 module with one N16 stage."""

    return load_flash_kda_module("m128_n16_short", target)


def load_flash_kda_persistent_m128_module(target: FlashKDATarget):
    """Load the SM100-only static-binned persistent M128 module."""

    return load_flash_kda_module("persistent_m128", target)


def load_flash_kda_piece_persistent_m128_module(target: FlashKDATarget):
    """Load the recurrence-piece persistent M128 module."""

    return load_flash_kda_module("piece_persistent_m128", target)


def load_flash_kda_small_bh_m128_module(target: FlashKDATarget):
    """Load the fixed-layout small-BH owner/helper M128 module."""

    return load_flash_kda_module("small_bh_m128", target)


def load_flash_kda_bt16_prepare_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_prepare", target)


def load_flash_kda_bt16_prepare_beta_tma_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_prepare_beta_tma", target)


def load_flash_kda_bt16_chain_m64_s7_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s7", target)


def load_flash_kda_bt16_chain_m64_s8_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s8", target)


def load_flash_kda_bt16_chain_m64_s9_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s9", target)


def load_flash_kda_bt16_prepare_chain_m64_s8_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_prepare_chain_m64_s8", target)


def get_flash_kda_prefill_module(variant: FlashKDAVariant, target: FlashKDATarget):
    """Return the loaded module used by the recurrent-KDA prefill dispatcher."""

    return load_flash_kda_module(variant, target)


__all__ = [
    "FLASH_KDA_VARIANTS",
    "FlashKDATarget",
    "FlashKDAVariant",
    "gen_flash_kda_bt16_chain_m64_s7_module",
    "gen_flash_kda_bt16_chain_m64_s8_module",
    "gen_flash_kda_bt16_chain_m64_s9_module",
    "gen_flash_kda_bt16_prepare_chain_m64_s8_module",
    "gen_flash_kda_bt16_prepare_beta_tma_module",
    "gen_flash_kda_bt16_prepare_module",
    "gen_flash_kda_m64_module",
    "gen_flash_kda_m128_module",
    "gen_flash_kda_m128_tensor_state_decay_module",
    "gen_flash_kda_m128_h12_short_module",
    "gen_flash_kda_m128_h12_long_module",
    "gen_flash_kda_m128_n16_module",
    "gen_flash_kda_m128_n16_checkpoint_module",
    "gen_flash_kda_m128_n16_short_module",
    "gen_flash_kda_piece_persistent_m128_module",
    "gen_flash_kda_persistent_m128_module",
    "gen_flash_kda_small_bh_m128_module",
    "gen_flash_kda_module",
    "get_flash_kda_prefill_module",
    "get_flash_kda_uri",
    "load_flash_kda_m64_module",
    "load_flash_kda_m128_module",
    "load_flash_kda_m128_tensor_state_decay_module",
    "load_flash_kda_m128_h12_short_module",
    "load_flash_kda_m128_h12_long_module",
    "load_flash_kda_m128_n16_module",
    "load_flash_kda_m128_n16_short_module",
    "load_flash_kda_piece_persistent_m128_module",
    "load_flash_kda_persistent_m128_module",
    "load_flash_kda_small_bh_m128_module",
    "load_flash_kda_bt16_chain_m64_s7_module",
    "load_flash_kda_bt16_chain_m64_s8_module",
    "load_flash_kda_bt16_chain_m64_s9_module",
    "load_flash_kda_bt16_prepare_chain_m64_s8_module",
    "load_flash_kda_bt16_prepare_beta_tma_module",
    "load_flash_kda_bt16_prepare_module",
    "load_flash_kda_module",
]
