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
        "fe0a070282",
        "-DFLASHINFER_FLASH_KDA_H12_SHORT=1",
        "flashkda_bf16_fused_m128_h12_short.cu",
        "d53eb6c03047625fce5c365c0e217bbd511fe64abf20112a1cfff496b27c39ad",
    ),
    (
        "m128_h12_long",
        "9e4219f788",
        "-DFLASHINFER_FLASH_KDA_H12_LONG=1",
        "flashkda_bf16_fused_m128_h12_long.cu",
        "abbebbc936a6af5a35c7cb5497b1c40136079896904edda6d10ef986556584c3",
    ),
)


@pytest.mark.parametrize(
    ("variant", "module_ident", "variant_define", "source_name", "source_sha256"),
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
        / "flashkda_bf16_fused_m128_h12_binding.cu"
    ]
    assert variant_define in spec.extra_cuda_cflags
    assert target_define in spec.extra_cuda_cflags
    assert sum(
        flag.startswith("-DFLASHINFER_FLASH_KDA_H12_")
        for flag in spec.extra_cuda_cflags
    ) == 1
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1

    frozen_source = flash_kda._get_flash_kda_csrc_dir() / source_name
    payload = frozen_source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == source_sha256
    text = payload.decode()
    assert f"flashkda_bf16_fused_m128_{module_ident}." in text

    flash_kda.gen_flash_kda_module.cache_clear()


def test_h12_prefill_variants_are_in_the_aot_inventory():
    assert "m128_h12_short" in flash_kda.FLASH_KDA_VARIANTS
    assert "m128_h12_long" in flash_kda.FLASH_KDA_VARIANTS
