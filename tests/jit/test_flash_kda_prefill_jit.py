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

import hashlib

import pytest

from flashinfer.jit import flash_kda


_H12_CASES = (
    (
        "m128_h12_short",
        "2e3dd633b9",
        "d25044154d",
        "-DFLASHINFER_FLASH_KDA_H12_SHORT=1",
        "cake_flashkda_bf16_fused_m128_h12_short.cu",
        "3472d562b61a2eb865f4a075cbae14bf199a357abc2d9476127350106be40b27",
    ),
    (
        "m128_h12_long",
        "ebe95af50a",
        "88cedfb168",
        "-DFLASHINFER_FLASH_KDA_H12_LONG=1",
        "cake_flashkda_bf16_fused_m128_h12_long.cu",
        "edc4085329fa659498b0a790407579afc0aeab48bac08b6b57e5de462e7754f7",
    ),
)

_COMMON_HEADER_VARIANT_BODIES = (
    ("m64", "flashkda_bf16_fused_m64.cu"),
    ("m128", "flashkda_bf16_fused_m128.cu"),
    (
        "m128_tensor_state_decay",
        "cake_flashkda_bf16_fused_m128_tensor_state_decay.cu",
    ),
    ("m128_h12_short", "cake_flashkda_bf16_fused_m128_h12_short.cu"),
    ("m128_h12_long", "cake_flashkda_bf16_fused_m128_h12_long.cu"),
    ("m128_n16", "cake_flashkda_bf16_fused_m128_n16.cu"),
    (
        "m128_n16_checkpoint",
        "flashkda_bf16_fused_m128_n16_checkpoint.cu",
    ),
    ("m128_n16_short", "cake_flashkda_bf16_fused_m128_n16_short.cu"),
    ("persistent_m128", "cake_flashkda_bf16_persistent_m128.cu"),
    ("small_bh_m128", "cake_flashkda_bf16_small_bh_m128.cu"),
)


@pytest.mark.parametrize(("variant", "body_name"), _COMMON_HEADER_VARIANT_BODIES)
def test_common_header_prefill_variant_cache_key(variant, body_name):
    csrc_dir = flash_kda._get_flash_kda_csrc_dir()
    spec = flash_kda.gen_flash_kda_module(variant, "sm100f")
    assert len(spec.sources) == 1
    digest = hashlib.sha256(
        b"\0".join(
            source.read_bytes()
            for source in (
                csrc_dir / body_name,
                spec.sources[0],
                csrc_dir / "flashkda_binding_common.cuh",
            )
        )
    ).hexdigest()[:10]
    assert flash_kda._FLASH_KDA_MODULE_IDENTS[variant] == digest


@pytest.mark.parametrize(
    (
        "variant",
        "module_ident",
        "frozen_module_ident",
        "variant_define",
        "source_name",
        "source_sha256",
    ),
    _H12_CASES,
)
@pytest.mark.parametrize(
    ("target", "target_define"),
    (
        ("sm100a", "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0"),
        ("sm100f", "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100"),
    ),
)
def test_h12_prefill_jit_spec_and_frozen_source(
    variant,
    module_ident,
    frozen_module_ident,
    variant_define,
    source_name,
    source_sha256,
    target,
    target_define,
):
    flash_kda.gen_flash_kda_module.cache_clear()
    spec = flash_kda.gen_flash_kda_module(variant, target)

    assert spec.name == f"flash_kda_bf16_{variant}_{module_ident}_{target}"
    assert spec.sources == [
        flash_kda._get_flash_kda_csrc_dir()
        / "cake_flashkda_bf16_fused_m128_h12_binding.cu"
    ]
    assert variant_define in spec.extra_cuda_cflags
    assert target_define in spec.extra_cuda_cflags
    assert (
        sum(
            flag.startswith("-DFLASHINFER_FLASH_KDA_H12_")
            for flag in spec.extra_cuda_cflags
        )
        == 1
    )
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1

    frozen_source = flash_kda._get_flash_kda_csrc_dir() / source_name
    payload = frozen_source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == source_sha256
    text = payload.decode()
    assert f"flashkda_bf16_fused_m128_{frozen_module_ident}." in text

    flash_kda.gen_flash_kda_module.cache_clear()


def test_h12_prefill_variants_are_in_the_aot_inventory():
    assert "m128_h12_short" in flash_kda.FLASH_KDA_VARIANTS
    assert "m128_h12_long" in flash_kda.FLASH_KDA_VARIANTS


@pytest.mark.parametrize(
    ("target", "target_define"),
    (
        ("sm100a", "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0"),
        ("sm100f", "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100"),
    ),
)
def test_combined_bt16_jit_spec(target, target_define):
    flash_kda.gen_flash_kda_module.cache_clear()
    spec = flash_kda.gen_flash_kda_module("bt16_prepare_chain_m64_s8", target)

    csrc_dir = flash_kda._get_flash_kda_csrc_dir()
    assert spec.name == (
        f"flash_kda_bf16_bt16_prepare_chain_m64_s8_6c392ef667_{target}"
    )
    assert spec.sources == [
        csrc_dir / "cake_flashkda_bf16_bt16_prepare_binding.cu",
        csrc_dir / "cake_flashkda_bf16_bt16_chain_m64_binding.cu",
        csrc_dir / "cake_flashkda_bf16_bt16_prepare_chain_m64_binding.cu",
    ]
    assert "-DFLASHINFER_FLASH_KDA_COMBINED_BT16=1" in spec.extra_cuda_cflags
    assert target_define in spec.extra_cuda_cflags
    assert "bt16_prepare_chain_m64_s8" in flash_kda.FLASH_KDA_VARIANTS

    flash_kda.gen_flash_kda_module.cache_clear()


@pytest.mark.parametrize(
    ("target", "target_define"),
    (
        ("sm100a", "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0"),
        ("sm100f", "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100"),
    ),
)
def test_short_n16_jit_spec_and_frozen_source(target, target_define):
    flash_kda.gen_flash_kda_module.cache_clear()
    spec = flash_kda.gen_flash_kda_module("m128_n16_short", target)

    assert spec.name == f"flash_kda_bf16_m128_n16_short_71bc4450bf_{target}"
    assert spec.sources == [
        flash_kda._get_flash_kda_csrc_dir()
        / "cake_flashkda_bf16_fused_m128_n16_binding.cu"
    ]
    assert "-DFLASHINFER_FLASH_KDA_N16_SHORT=1" in spec.extra_cuda_cflags
    assert target_define in spec.extra_cuda_cflags
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1

    frozen_source = (
        flash_kda._get_flash_kda_csrc_dir()
        / "cake_flashkda_bf16_fused_m128_n16_short.cu"
    )
    payload = frozen_source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "7a9b1c01a0abd04c0d2baddbcfc6043b693c80be728691f1fc6d2325db6a238e"
    )
    text = payload.decode()
    assert "#define SMEM_TOTAL 112256" in text
    assert "__launch_bounds__(512)" in text
    assert "m128_n16_short" in flash_kda.FLASH_KDA_VARIANTS

    flash_kda.gen_flash_kda_module.cache_clear()


@pytest.mark.parametrize(
    ("target", "target_define"),
    (
        ("sm100a", "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0"),
        ("sm100f", "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100"),
    ),
)
def test_tensor_state_decay_jit_spec_and_frozen_source(target, target_define):
    flash_kda.gen_flash_kda_module.cache_clear()
    spec = flash_kda.gen_flash_kda_module("m128_tensor_state_decay", target)

    assert spec.name == (f"flash_kda_bf16_m128_tensor_state_decay_b3a1e8779c_{target}")
    assert spec.sources == [
        flash_kda._get_flash_kda_csrc_dir() / "flashkda_bf16_fused_m128_binding.cu"
    ]
    assert "-DFLASHINFER_FLASH_KDA_TENSOR_STATE_DECAY=1" in spec.extra_cuda_cflags
    assert target_define in spec.extra_cuda_cflags
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1

    frozen_source = (
        flash_kda._get_flash_kda_csrc_dir()
        / "cake_flashkda_bf16_fused_m128_tensor_state_decay.cu"
    )
    payload = frozen_source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "c84b37d139728dba0f96d021825695922dc2ed080d99d3530734b9ef7bfaea50"
    )
    assert "flashkda_bf16_fused_m128_0d8d9e6964." in payload.decode()
    assert "m128_tensor_state_decay" in flash_kda.FLASH_KDA_VARIANTS

    flash_kda.gen_flash_kda_module.cache_clear()
