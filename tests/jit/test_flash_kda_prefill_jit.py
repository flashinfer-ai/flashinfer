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
import inspect
import json
from dataclasses import replace

import pytest

import flashinfer.aot as aot_api
import flashinfer.kda_prefill as kda_prefill_api
from flashinfer.jit import flash_kda


_H12_CASES = (
    (
        "m128_h12_short",
        "47c46019cc",
        "d25044154d",
        "-DFLASHINFER_FLASH_KDA_H12_SHORT=1",
        "cake_flashkda_bf16_fused_m128_h12_short.cu",
        "3472d562b61a2eb865f4a075cbae14bf199a357abc2d9476127350106be40b27",
    ),
    (
        "m128_h12_long",
        "b813a7edd3",
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
    (
        "piece_persistent_m128",
        "cake_flashkda_bf16_piece_persistent_m128.cu",
    ),
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


def test_piece_persistent_prefill_variant_is_in_the_aot_inventory():
    assert "piece_persistent_m128" in flash_kda.FLASH_KDA_VARIANTS


def test_prefill_jit_route_inventory_is_source_complete_and_aot_registered(
    monkeypatch,
):
    """Keep dispatcher metadata, compiled sources, and AOT registration closed."""

    assert all(
        not variant.lower().endswith("_o1") for variant in flash_kda.FLASH_KDA_VARIANTS
    )
    for variant in flash_kda.FLASH_KDA_VARIANTS:
        spec = flash_kda.gen_flash_kda_module(variant, "sm100f")
        assert spec.name == flash_kda.get_flash_kda_uri(variant, "sm100f")
        assert spec.sources
        assert all(source.is_file() for source in spec.sources)
        assert all("o1" not in flag.lower() for flag in spec.extra_cuda_cflags)

    wrapper_by_variant = {}

    def record_variant(variant, target):
        assert target == "sm100f"
        wrapper_by_variant[variant] = active_wrapper[0]
        return object()

    monkeypatch.setattr(flash_kda, "gen_flash_kda_module", record_variant)
    active_wrapper = [""]
    for name, value in vars(flash_kda).items():
        if (
            name == "gen_flash_kda_module"
            or not name.startswith("gen_flash_kda_")
            or not name.endswith("_module")
            or not callable(value)
        ):
            continue
        if tuple(inspect.signature(value).parameters) != ("target",):
            continue
        active_wrapper[0] = name
        value("sm100f")

    assert set(wrapper_by_variant) == set(flash_kda.FLASH_KDA_VARIANTS)
    aot_source = inspect.getsource(aot_api)
    assert "get_flash_kda_generated_variant_ids" in aot_source
    assert "gen_flash_kda_generated_module" in aot_source
    assert "gen_flash_kda_m128_n16_checkpoint_module" in aot_source
    assert "gen_flash_kda_m128_module" not in aot_source


def test_generated_prefill_registry_is_receipt_closed_and_exact_targeted():
    registry = flash_kda.get_flash_kda_generated_registry()
    csrc_dir = flash_kda._get_flash_kda_csrc_dir()
    receipt = json.loads(
        (csrc_dir / flash_kda._FLASH_KDA_GENERATED_RECEIPT_NAME).read_text()
    )

    assert receipt["status"] == "passed"
    assert receipt["source_closure_status"] == "passed"
    assert receipt["physical_selector_schema_version"] == 1
    assert receipt["physical_selector_collision_count"] == 0
    assert receipt["variant_count"] == len(registry)
    assert receipt["binding_tu_count"] == len(registry)
    assert receipt["abi_wrapper_count"] == len(
        {module.abi_wrapper_relpath for module in registry.values()}
    )
    assert set(module.target for module in registry.values()) == {
        "sm100a",
        "sm103a",
    }
    assert "bf16_f32_dependency" in {module.state_mode for module in registry.values()}
    selector_registry = flash_kda.get_flash_kda_generated_selector_registry()
    assert receipt["physical_selector_count"] == len(selector_registry)

    for module in registry.values():
        assert module.physical_selectors
        assert [source.role for source in module.source_closure] == [
            "selector_binding",
            "sanitized_body",
            "abi_wrapper",
            "generated_common_wrapper",
            "bt16_descriptor_common",
            "public_common_include",
        ]
        assert module.source_closure[4].path == (
            "csrc/kda/flashkda_generated_bt16_descriptor_common.cuh"
        )
        for selector in module.physical_selectors:
            selector_key = {
                "arch": selector.arch,
                "route": selector.route,
                "route_role": selector.route_role,
                "abi_family": selector.abi_family,
                "state_mode": selector.state_mode,
                "family_specialization_vector": [
                    list(item) for item in selector.family_specialization
                ],
            }
            assert (
                kda_prefill_api._make_flash_kda_generated_selector_key(
                    target=module.target,
                    route=selector.route,
                    route_role=selector.route_role,
                    state_mode=selector.state_mode,
                    family_specialization=dict(selector.family_specialization),
                )
                == selector_key
            )
            assert (
                flash_kda.get_flash_kda_generated_module_for_selector(selector_key)
                is module
            )

    for target, gencode, target_define in (
        (
            "sm100a",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
        ),
        (
            "sm103a",
            "-gencode=arch=compute_103a,code=sm_103a",
            "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=3",
        ),
    ):
        variant_ids = flash_kda.get_flash_kda_generated_variant_ids(target)
        assert variant_ids
        assert variant_ids == tuple(
            variant_id
            for variant_id, module in registry.items()
            if module.target == target
        )
        for variant_id in variant_ids:
            module = registry[variant_id]
            spec = flash_kda.gen_flash_kda_generated_module(variant_id)
            assert spec.sources == [
                flash_kda._resolve_generated_source(csrc_dir, module.binding_relpath)
            ]
            assert spec.name == flash_kda.get_flash_kda_generated_uri(variant_id)
            assert spec.name.endswith(module.cache_ident)
            assert gencode in spec.extra_cuda_cflags
            assert target_define in spec.extra_cuda_cflags
            assert "-DFLASHKDA_GENERATED_EMBEDDED_CUBIN=1" in spec.extra_cuda_cflags
            assert "-DTVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API=1" in spec.extra_cuda_cflags
            assert (
                f"-DFLASHKDA_GENERATED_CUBIN_IDENT={module.module_ident}"
                in spec.extra_cuda_cflags
            )
            assert spec.embedded_cubin_factory is not None
            assert all("sm100f" not in flag for flag in spec.extra_cuda_cflags)
            assert all("_o1" not in flag.lower() for flag in spec.extra_cuda_cflags)


def test_generated_embedded_kernel_cache_is_translation_unit_local():
    common_header = (
        flash_kda._get_flash_kda_csrc_dir() / "flashkda_generated_binding_common.cuh"
    ).read_text()
    assert "static inline void ConfigureAndLaunch(" in common_header


def test_generated_prefill_selector_parser_rejects_unknown_and_incomplete_keys():
    valid = {
        "arch": "sm_100a",
        "route": "direct_m128",
        "route_role": "main",
        "abi_family": "direct_m128",
        "state_mode": "bf16",
        "family_specialization_vector": [["chunk", 32]],
    }
    parsed = flash_kda._parse_generated_selector_key(valid, label="test selector")
    assert parsed.family_specialization == (("chunk", 32),)

    for invalid in (
        {key: value for key, value in valid.items() if key != "route_role"},
        {**valid, "unknown": False},
        {
            **valid,
            "family_specialization_vector": [["chunk", 32], ["chunk", 16]],
        },
    ):
        with pytest.raises(ValueError):
            flash_kda._parse_generated_selector_key(invalid, label="test selector")

    unknown = {
        **valid,
        "family_specialization_vector": [["chunk", 17]],
    }
    with pytest.raises(ValueError, match="unsupported generated FlashKDA"):
        flash_kda.get_flash_kda_generated_module_for_selector(unknown)


def test_generated_prefill_selector_registry_rejects_duplicate_keys(monkeypatch):
    registry = flash_kda.get_flash_kda_generated_registry()
    modules = list(registry.values())
    assert len(modules) >= 2
    first, second = modules[:2]
    colliding_second = replace(
        second, physical_selectors=(first.physical_selectors[0],)
    )
    monkeypatch.setattr(
        flash_kda,
        "get_flash_kda_generated_registry",
        lambda: {
            first.variant_id: first,
            colliding_second.variant_id: colliding_second,
        },
    )
    flash_kda.get_flash_kda_generated_selector_registry.cache_clear()
    try:
        with pytest.raises(ValueError, match="not collision-free"):
            flash_kda.get_flash_kda_generated_selector_registry()
    finally:
        flash_kda.get_flash_kda_generated_selector_registry.cache_clear()


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

    assert spec.name == f"flash_kda_bf16_m128_n16_short_3f90fe2347_{target}"
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

    assert spec.name == (f"flash_kda_bf16_m128_tensor_state_decay_9614ba2d29_{target}")
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
