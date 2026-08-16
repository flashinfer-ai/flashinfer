# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import json
from types import SimpleNamespace

import pytest

from flashinfer.jit import cake_kda_decode
from flashinfer.jit import core as jit_core


VARIANTS = (
    "d128_t1_unbounded_softplus_direct_split16",
    "d128_t1_unbounded_softplus_direct_split8",
)


def test_cake_kda_decode_manifest_matches_frozen_sources():
    csrc_dir = cake_kda_decode._get_csrc_dir()
    manifest = json.loads(
        (
            csrc_dir / "cake_kda_decode_unbounded_softplus_import_manifest.json"
        ).read_text()
    )
    assert manifest["schema_version"] == 1
    assert set(manifest["variants"]) == {"split8", "split16"}
    for split, record in manifest["variants"].items():
        frozen = csrc_dir.parent.parent / record["frozen_path"]
        assert (
            hashlib.sha256(frozen.read_bytes()).hexdigest() == record["frozen_sha256"]
        )
        assert record["value_split"] == int(split.removeprefix("split"))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    ("target", "target_arch", "expected_flag", "target_kind"),
    [
        ("sm100a", (10, "0a"), "-gencode=arch=compute_100a,code=sm_100a", 1000),
        ("sm100f", (10, "0f"), "-gencode=arch=compute_100f,code=sm_100f", 100),
        ("sm103a", (10, "3a"), "-gencode=arch=compute_103a,code=sm_103a", 1003),
    ],
)
def test_cake_kda_decode_jit_spec(
    monkeypatch, tmp_path, variant, target, target_arch, expected_flag, target_kind
):
    monkeypatch.setattr(
        jit_core.current_compilation_context, "TARGET_CUDA_ARCHS", {target_arch}
    )
    monkeypatch.setattr(cake_kda_decode.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    cake_kda_decode.gen_cake_kda_decode_module.cache_clear()

    uri = cake_kda_decode.get_cake_kda_decode_uri(variant, target)
    spec = cake_kda_decode.gen_cake_kda_decode_module(variant, target)
    binding = spec.sources[0]
    binding_text = binding.read_text()

    assert uri == f"cake_kda_decode_{variant}_{target}"
    assert spec.name == uri
    assert binding == tmp_path / uri / "cake_kda_decode_binding.cu"
    assert expected_flag in spec.extra_cuda_cflags
    assert (
        f"-DFLASHINFER_CAKE_KDA_DECODE_TARGET_KIND={target_kind}"
        in spec.extra_cuda_cflags
    )
    assert (
        f'#define CAKE_KDA_DECODE_BODY_FILE "cake_kda_decode_{variant}.cu"'
        in binding_text
    )
    assert "#define CAKE_KDA_DECODE_DIRECT_IMPL 1" in binding_text


@pytest.mark.parametrize(
    ("capabilities", "expected_target"),
    [
        ({"cake_kda_decode_sm100a_legacy": True}, "sm100a"),
        ({"cake_kda_decode_sm100f": True}, "sm100f"),
        ({"cake_kda_decode_sm103a_direct": True}, "sm103a"),
    ],
)
def test_aot_registers_cake_kda_decode_portfolio(
    monkeypatch, capabilities, expected_target
):
    from flashinfer import aot

    calls = []

    def fake_cake_kda_decode(variant, target):
        calls.append((variant, target))
        return SimpleNamespace(name=f"cake_kda_decode_{variant}_{target}")

    monkeypatch.setattr(aot, "gen_cake_kda_decode_module", fake_cake_kda_decode)
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        capabilities,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    expected_calls = [(variant, expected_target) for variant in VARIANTS]
    assert calls == expected_calls
    assert [spec.name for spec in specs] == [
        "spdlog",
        *(f"cake_kda_decode_{variant}_{expected_target}" for variant in VARIANTS),
        "cudnn",
    ]
