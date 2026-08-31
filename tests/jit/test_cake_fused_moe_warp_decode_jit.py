# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import pytest

from flashinfer.jit import cake_fused_moe_warp_decode


def test_cake_warp_decode_jit_spec_is_exact_sm103a() -> None:
    cake_fused_moe_warp_decode.gen_cake_fused_moe_warp_decode_module.cache_clear()
    spec = cake_fused_moe_warp_decode.gen_cake_fused_moe_warp_decode_module()

    assert spec.name == "cake_fused_moe_warp_decode_sm103a"
    assert [source.name for source in spec.sources] == [
        "cake_adaptive_warp_decode_kernels.cu",
        "cake_warp_decode_binding.cu",
    ]
    assert all(source.is_file() for source in spec.sources)
    assert (
        spec.sources[0].parent / "cake_warp_decode_generated_manifest.cuh"
    ).is_file()
    assert (
        spec.sources[1].parent / "cake_warp_decode_contract.cuh"
    ).is_file()
    assert spec.sources[0].parent.name == "generated"
    assert spec.sources[1].parent == spec.sources[0].parent.parent
    assert spec.sources[1].parent in spec.extra_include_dirs
    assert spec.sources[0].parent in spec.extra_include_dirs
    assert spec.sources[1].parents[2] in spec.extra_include_dirs
    assert "-gencode=arch=compute_103a,code=sm_103a" in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert spec.extra_ldflags == ["-lcuda"]
    assert spec.needs_device_linking is False
    assert (
        sum(
            flag.startswith("-gencode=arch=compute_")
            for flag in spec.extra_cuda_cflags
        )
        == 1
    )

    binding = spec.sources[1].read_text()
    assert "CheckSm103a" in binding
    assert (
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_moe_warp_decode_workspace_size"
        in binding
    )
    assert (
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_moe_warp_decode_prepare_workspace"
        in binding
    )
    assert (
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_moe_warp_decode_release_workspace"
        in binding
    )
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_moe_warp_decode," in binding


def test_cake_warp_decode_rejects_other_targets() -> None:
    with pytest.raises(ValueError, match="unsupported Cake warp-decode target"):
        cake_fused_moe_warp_decode.get_cake_fused_moe_warp_decode_uri("sm100a")
    with pytest.raises(ValueError, match="unsupported Cake warp-decode target"):
        cake_fused_moe_warp_decode.gen_cake_fused_moe_warp_decode_module("sm120a")


def test_cake_warp_decode_getter_uses_exact_target(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        cake_fused_moe_warp_decode,
        "load_cake_fused_moe_warp_decode_module",
        lambda target="sm103a", device=None: (target, device, sentinel),
    )

    assert cake_fused_moe_warp_decode.get_cake_fused_moe_warp_decode_module(
        device="cuda:1"
    ) == (
        "sm103a",
        "cuda:1",
        sentinel,
    )


def test_cake_warp_decode_is_registered_for_sm103_aot(monkeypatch) -> None:
    from flashinfer import aot

    sentinel = SimpleNamespace(name="cake_fused_moe_warp_decode_sm103a")
    monkeypatch.setattr(
        aot,
        "gen_cake_fused_moe_warp_decode_module",
        lambda: sentinel,
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"sm103": True},
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )

    assert sentinel in specs


def test_cake_warp_decode_loader_rejects_non_sm103_before_build(monkeypatch) -> None:
    build_called = False
    checked_devices = []

    def fail_if_built(target):
        nonlocal build_called
        build_called = True
        return target

    def get_compute_capability(device=None):
        checked_devices.append(device)
        return 10, 0

    monkeypatch.setattr(
        cake_fused_moe_warp_decode,
        "_get_compute_capability",
        get_compute_capability,
    )
    monkeypatch.setattr(
        cake_fused_moe_warp_decode,
        "_build_and_load_cake_fused_moe_warp_decode_module",
        fail_if_built,
    )

    with pytest.raises(RuntimeError, match="exact compute capability 10.3"):
        cake_fused_moe_warp_decode.load_cake_fused_moe_warp_decode_module(
            device="cuda:1"
        )
    assert build_called is False
    assert checked_devices == ["cuda:1"]
