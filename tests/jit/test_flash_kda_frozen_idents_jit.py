# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#

import hashlib

import pytest

from flashinfer.jit import flash_kda

# Body compiled into each common-header variant. Bindings come from
# _FLASH_KDA_BINDING_STEMS; the ones shared by several variants
# (m128_n16 / m128_n16_short) pick their body with a -D flag.
_COMMON_HEADER_VARIANT_BODIES = {
    "m64": "flashkda_bf16_fused_m64.cu",
    "m128": "flashkda_bf16_fused_m128.cu",
    "m128_tensor_state_decay": "cake_flashkda_bf16_fused_m128_tensor_state_decay.cu",
    "m128_h12_short": "cake_flashkda_bf16_fused_m128_h12_short.cu",
    "m128_h12_long": "cake_flashkda_bf16_fused_m128_h12_long.cu",
    "m128_n16": "cake_flashkda_bf16_fused_m128_n16.cu",
    "m128_n16_checkpoint": "flashkda_bf16_fused_m128_n16_checkpoint.cu",
    "m128_n16_short": "cake_flashkda_bf16_fused_m128_n16_short.cu",
    "persistent_m128": "cake_flashkda_bf16_persistent_m128.cu",
    "piece_persistent_m128": "cake_flashkda_bf16_piece_persistent_m128.cu",
    "small_bh_m128": "cake_flashkda_bf16_small_bh_m128.cu",
}

# The bt16 routes seal a different source closure, so their idents are not
# reproducible from the common-header formula below.
_VARIANTS_WITHOUT_COMMON_HEADER_IDENTS = frozenset(
    {
        "bt16_prepare",
        "bt16_prepare_beta_tma",
        "bt16_chain_m64_s7",
        "bt16_chain_m64_s8",
        "bt16_chain_m64_s9",
        "bt16_prepare_chain_m64_s8",
    }
)


def _ident_from_sources(variant: str) -> str:
    csrc_dir = flash_kda._get_flash_kda_csrc_dir()
    stem = flash_kda._FLASH_KDA_BINDING_STEMS[variant]
    sources = (
        csrc_dir / _COMMON_HEADER_VARIANT_BODIES[variant],
        csrc_dir / f"{stem}_binding.cu",
        csrc_dir / "flashkda_binding_common.cuh",
    )
    return hashlib.sha256(
        b"\0".join(source.read_bytes() for source in sources)
    ).hexdigest()[:10]


@pytest.mark.parametrize("variant", sorted(_COMMON_HEADER_VARIANT_BODIES))
def test_frozen_module_ident_describes_its_sources(variant):
    """The frozen ident is the JIT/AOT cache key, so it must track the sources.

    A prebuilt module is loaded by name alone, and the ident is the only part of
    that name carrying source identity.
    """

    assert flash_kda._FLASH_KDA_MODULE_IDENTS[variant] == _ident_from_sources(
        variant
    ), (
        f"frozen ident for {variant!r} does not describe the sources on disk.\n"
        f"  ident table read from: {flash_kda.__file__}\n"
        f"  sources read from:     {flash_kda._get_flash_kda_csrc_dir()}\n"
        "When those two paths are in different trees, the imported flashinfer "
        "and its data/csrc symlink disagree and the package needs reinstalling. "
        "When they are the same tree, regenerate the ident for this variant."
    )


def test_every_frozen_ident_is_accounted_for():
    assert set(flash_kda._FLASH_KDA_MODULE_IDENTS) == (
        set(_COMMON_HEADER_VARIANT_BODIES) | _VARIANTS_WITHOUT_COMMON_HEADER_IDENTS
    )
