# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest

from flashinfer.jit import cake_kda_prefill_shared as shared


def test_shared_prefill_manifest_is_complete_and_hash_verified():
    shared.get_cake_kda_prefill_shared_module_specs.cache_clear()
    specs = shared.get_cake_kda_prefill_shared_module_specs()

    assert len(specs) == 16
    assert {(spec.target, spec.policy) for spec in specs} == {
        (target, policy)
        for target in ("sm100a", "sm103a")
        for policy in shared._EXPECTED_POLICIES
    }
    assert len({spec.device_path for spec in specs}) == 8
    assert len({spec.binding_path for spec in specs}) == 8
    assert all(spec.device_path.name.startswith("cake_kda_prefill_") for spec in specs)
    assert all(spec.binding_path.name.endswith("_binding.cu") for spec in specs)


@pytest.mark.parametrize(
    ("target", "target_define"),
    (
        ("sm100a", "-DFLASHINFER_CAKE_KDA_PREFILL_TARGET_MINOR=0"),
        ("sm103a", "-DFLASHINFER_CAKE_KDA_PREFILL_TARGET_MINOR=3"),
    ),
)
def test_shared_prefill_jit_compiles_device_and_binding_separately(
    target, target_define
):
    shared.gen_cake_kda_prefill_shared_module.cache_clear()
    module_spec = shared.get_cake_kda_prefill_shared_module_spec(
        target, "direct_m128_generic"
    )
    jit_spec = shared.gen_cake_kda_prefill_shared_module(target, "direct_m128_generic")

    assert jit_spec.name == (
        f"{module_spec.module_ident}_{target}_{module_spec.closure_sha256}"
    )
    assert jit_spec.sources == [module_spec.device_path, module_spec.binding_path]
    assert target_define in jit_spec.extra_cuda_cflags
    assert "--use_fast_math" in jit_spec.extra_cuda_cflags
    assert "--ptxas-options=-O1" in jit_spec.extra_cuda_cflags
    assert (
        sum("-gencode=arch=compute_" in flag for flag in jit_spec.extra_cuda_cflags)
        == 1
    )


def test_shared_prefill_binding_uses_flashinfer_tvm_ffi_runtime():
    specs = shared.get_cake_kda_prefill_shared_module_specs()
    for binding_path in {spec.binding_path for spec in specs}:
        source = binding_path.read_text()
        assert source.count('#include "tvm_ffi_utils.h"') == 1
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run" in source
        assert "TVM_FFI_EMBED_CUBIN" not in source
