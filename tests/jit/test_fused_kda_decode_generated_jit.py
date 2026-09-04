# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


generated = importlib.import_module("flashinfer.jit.fused_kda_decode_generated")


def _source(
    kernel_symbol="kernel_fused_kda_decode_test",
    *,
    abi_kind="standard",
    state_dtype="float32",
):
    parameters = []
    for index, (kind, name, dtype) in enumerate(
        generated.FUSED_KDA_DECODE_GENERATED_ABIS[abi_kind]
    ):
        if kind == "buffer":
            if dtype == "state_dtype":
                dtype = state_dtype
            c_type = {
                "bfloat16": "__nv_bfloat16*",
                "float32": "float*",
                "int32": "int*",
            }[dtype]
        else:
            assert index >= 12
            c_type = "float" if dtype == "float32_scalar" else "int"
        parameters.append(f"{c_type} {name}")
    return (
        'extern "C" {\n'
        "#define SMEM_TOTAL 2320\n"
        f"__global__ __launch_bounds__(256) void {kernel_symbol}(\n"
        + ",\n".join(parameters)
        + ") { }\n}\n"
    )


def _manifest(
    body: Path,
    *,
    kernel_symbol="kernel_fused_kda_decode_test",
    abi_kind="standard",
    state_dtype="float32",
):
    return {
        "schema_version": 1,
        "contract": json.loads(json.dumps(generated._CONTRACT)),
        "status": "complete",
        "variants": [
            {
                "name": "single_cta",
                "target": "sm100a",
                "body": body.name,
                "source_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
                "kernel_symbol": kernel_symbol,
                "abi_kind": abi_kind,
                "state_dtype": state_dtype,
                "extra_cuda_cflags": ["--use_fast_math", "--maxrregcount=128"],
                "launch": {
                    "threads": 256,
                    "dynamic_smem_bytes": 2320,
                },
                "eligibility": [
                    {
                        "heads": [12, 24, 32, 48, 96],
                        "minimum_rows": 1,
                        "maximum_rows": None,
                        "slot_classes": [
                            "positive_unique",
                            "unique_or_null",
                        ],
                        "lower_bound_values": "any",
                        "norm_eps_values": "any",
                        "strides": {
                            "x_row_stride": None,
                            "conv_slot_stride": None,
                            "beta_row_stride": None,
                            "state_slot_stride": None,
                            "output_gate_row_stride": None,
                        },
                    }
                ],
            }
        ],
        "remaining_generated_inputs": [],
    }


def _write_complete_manifest(
    tmp_path: Path, *, abi_kind="standard", state_dtype="float32"
):
    body = tmp_path / "fused_kda_decode_generated_single_cta.cu"
    body.write_text(
        _source(abi_kind=abi_kind, state_dtype=state_dtype), encoding="utf-8"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(body, abi_kind=abi_kind, state_dtype=state_dtype)),
        encoding="utf-8",
    )
    return body, manifest_path


def test_checked_in_manifest_enables_all_generated_routes():
    csrc_dir = generated.get_kda_csrc_dir()
    variants = generated.load_fused_kda_decode_generated_variants(
        manifest_path=csrc_dir / generated._MANIFEST_FILENAME,
        csrc_dir=csrc_dir,
    )
    assert len(variants) == 16
    assert len({variant.name for variant in variants}) == 16
    assert {variant.target for variant in variants} == {"sm100a"}
    assert generated.fused_kda_decode_generated_is_available()


def test_complete_manifest_verifies_source_identity_abi_and_launch(tmp_path):
    body, manifest_path = _write_complete_manifest(tmp_path)

    (variant,) = generated.load_fused_kda_decode_generated_variants(
        manifest_path=manifest_path,
        csrc_dir=tmp_path,
    )

    assert variant.name == "single_cta"
    assert variant.target == "sm100a"
    assert variant.body_path == body
    assert variant.kernel_symbol == "kernel_fused_kda_decode_test"
    assert variant.abi_kind == "standard"
    assert variant.state_dtype == "float32"
    assert variant.extra_cuda_cflags == ("--use_fast_math", "--maxrregcount=128")
    assert variant.eligibility[0].heads == (12, 24, 32, 48, 96)
    assert variant.eligibility[0].slot_classes == (
        "positive_unique",
        "unique_or_null",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["contract"].update(head_dim=64), "contract mismatch"),
        (
            lambda payload: payload["variants"][0].update(source_sha256="0" * 64),
            "source_sha256 mismatch",
        ),
        (
            lambda payload: payload["variants"][0]["launch"].update(
                dynamic_smem_bytes=4096
            ),
            "SMEM_TOTAL matching",
        ),
        (
            lambda payload: payload["variants"][0].update(
                extra_cuda_cflags=["-DUNVERIFIED=1"]
            ),
            "not an allowed generated-kernel flag",
        ),
    ],
)
def test_manifest_rejects_contract_source_and_launch_drift(tmp_path, mutation, message):
    body, manifest_path = _write_complete_manifest(tmp_path)
    payload = _manifest(body)
    mutation(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(generated.FusedKDADecodeGeneratedManifestError, match=message):
        generated.load_fused_kda_decode_generated_variants(
            manifest_path=manifest_path,
            csrc_dir=tmp_path,
        )


def test_manifest_rejects_kernel_parameter_order_drift(tmp_path):
    body, manifest_path = _write_complete_manifest(tmp_path)
    body.write_text(
        _source().replace("__nv_bfloat16* x", "__nv_bfloat16* x_renamed"),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(_manifest(body)), encoding="utf-8")

    with pytest.raises(
        generated.FusedKDADecodeGeneratedManifestError,
        match="parameter 0 must be named 'x'",
    ):
        generated.load_fused_kda_decode_generated_variants(
            manifest_path=manifest_path,
            csrc_dir=tmp_path,
        )


def test_repeated_safe_bfloat16_abi_includes_rows(tmp_path):
    body, manifest_path = _write_complete_manifest(
        tmp_path, abi_kind="repeated_safe", state_dtype="bfloat16"
    )
    payload = _manifest(body, abi_kind="repeated_safe", state_dtype="bfloat16")
    payload["variants"][0]["eligibility"][0]["slot_classes"].append("repeated_positive")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    (variant,) = generated.load_fused_kda_decode_generated_variants(
        manifest_path=manifest_path,
        csrc_dir=tmp_path,
    )

    assert variant.abi_kind == "repeated_safe"
    assert variant.state_dtype == "bfloat16"
    binding = generated._render_binding(variant)
    assert "#define FLASHINFER_FUSED_KDA_DECODE_HAS_ROWS 1" in binding
    assert "#define FLASHINFER_FUSED_KDA_DECODE_STATE_IS_BFLOAT16 1" in binding
    assert generated._ARG_PLAN_SHA256["repeated_safe"] in binding


def test_manifest_rejects_repeated_slots_for_standard_abi(tmp_path):
    body, manifest_path = _write_complete_manifest(tmp_path)
    payload = _manifest(body)
    payload["variants"][0]["eligibility"][0]["slot_classes"].append("repeated_positive")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        generated.FusedKDADecodeGeneratedManifestError,
        match="cannot send repeated positive slots to the standard ABI",
    ):
        generated.load_fused_kda_decode_generated_variants(
            manifest_path=manifest_path,
            csrc_dir=tmp_path,
        )


def test_manifest_selector_matches_exact_rules_and_falls_back_for_gaps(tmp_path):
    body, manifest_path = _write_complete_manifest(tmp_path)
    payload = _manifest(body)
    payload["variants"][0]["eligibility"][0].update(
        heads=[32],
        minimum_rows=4,
        maximum_rows=64,
        slot_classes=["positive_unique"],
        lower_bound_values=[-5.0],
        norm_eps_values=[1e-5],
        strides={
            "x_row_stride": 12305,
            "conv_slot_stride": None,
            "beta_row_stride": None,
            "state_slot_stride": None,
            "output_gate_row_stride": None,
        },
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    variants = generated.load_fused_kda_decode_generated_variants(
        manifest_path=manifest_path,
        csrc_dir=tmp_path,
    )
    facts = {
        "target": "sm100a",
        "num_heads": 32,
        "num_rows": 32,
        "state_dtype": "float32",
        "slot_class": "positive_unique",
        "lower_bound": -5.0,
        "norm_eps": 1e-5,
        "x_row_stride": 12305,
        "conv_slot_stride": 589824,
        "beta_row_stride": 33,
        "state_slot_stride": 552960,
        "output_gate_row_stride": 4103,
        "variants": variants,
    }

    assert generated.select_fused_kda_decode_generated_variant(**facts) is variants[0]
    assert (
        generated.select_fused_kda_decode_generated_variant(**{**facts, "num_rows": 65})
        is None
    )
    assert (
        generated.select_fused_kda_decode_generated_variant(
            **{**facts, "lower_bound": None}
        )
        is None
    )
    assert (
        generated.select_fused_kda_decode_generated_variant(
            **{**facts, "x_row_stride": 12304}
        )
        is None
    )


def test_manifest_rejects_body_path_traversal(tmp_path):
    body, manifest_path = _write_complete_manifest(tmp_path)
    payload = _manifest(body)
    payload["variants"][0]["body"] = "../fused_kda_decode_generated_single_cta.cu"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        generated.FusedKDADecodeGeneratedManifestError,
        match=r"must name one fused_kda_decode_generated_\*\.cu file",
    ):
        generated.load_fused_kda_decode_generated_variants(
            manifest_path=manifest_path,
            csrc_dir=tmp_path,
        )


def test_binding_renderer_preserves_verified_launch_and_source_contract(tmp_path):
    _, manifest_path = _write_complete_manifest(tmp_path)
    (variant,) = generated.load_fused_kda_decode_generated_variants(
        manifest_path=manifest_path,
        csrc_dir=tmp_path,
    )

    binding = generated._render_binding(variant)

    assert (
        "#define FLASHINFER_FUSED_KDA_DECODE_BODY_FILE "
        '"fused_kda_decode_generated_single_cta.cu"'
    ) in binding
    assert (
        "#define FLASHINFER_FUSED_KDA_DECODE_KERNEL kernel_fused_kda_decode_test"
    ) in binding
    assert "#define FLASHINFER_FUSED_KDA_DECODE_THREADS 256" in binding
    assert generated._ARG_PLAN_SHA256["standard"] in binding


def test_jit_spec_uses_verified_source_identity_and_binding(tmp_path, monkeypatch):
    body, manifest_path = _write_complete_manifest(tmp_path)
    (variant,) = generated.load_fused_kda_decode_generated_variants(
        manifest_path=manifest_path,
        csrc_dir=tmp_path,
    )
    (tmp_path / generated._BINDING_HEADER).write_text("// binding\n", encoding="utf-8")
    generated_dir = tmp_path / "generated"
    calls = []

    monkeypatch.setattr(
        generated,
        "get_fused_kda_decode_generated_variant",
        lambda name, target: variant,
    )
    monkeypatch.setattr(generated, "get_kda_csrc_dir", lambda: tmp_path)
    monkeypatch.setattr(generated, "get_flashinfer_include_dir", lambda: tmp_path)
    monkeypatch.setattr(generated.jit_env, "FLASHINFER_GEN_SRC_DIR", generated_dir)
    monkeypatch.setattr(
        generated,
        "gen_kda_jit_spec",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(name=kwargs["name"]),
    )
    generated.gen_fused_kda_decode_generated_module.cache_clear()
    try:
        spec = generated.gen_fused_kda_decode_generated_module("single_cta", "sm100a")
    finally:
        generated.gen_fused_kda_decode_generated_module.cache_clear()

    expected_uri = (
        "fused_kda_decode_generated_single_cta_"
        f"{hashlib.sha256(body.read_bytes()).hexdigest()[:16]}_sm100a"
    )
    assert spec.name == expected_uri
    assert calls[0]["target"] == "sm100a"
    assert calls[0]["extra_cuda_cflags"] == (
        "--use_fast_math",
        "--maxrregcount=128",
    )
    (binding_path,) = calls[0]["sources"]
    assert binding_path.read_text(encoding="utf-8") == generated._render_binding(
        variant
    )


def test_binding_header_packs_the_exact_physical_argument_count():
    header = (generated.get_kda_csrc_dir() / generated._BINDING_HEADER).read_text(
        encoding="utf-8"
    )

    assert "CheckArgumentCount<21>(args);" in header
    assert "CheckArgumentCount<22>(args);" in header
    assert header.count('CheckAlignment(state, "state", 16);') == 1
    assert header.count('CheckAlignment(state, "state", 32);') == 1
    assert "x.stride(0) >= qkv_size" not in header
    assert "raw_beta.stride(1) >= num_heads" not in header
    assert "output_gate.stride(0) >= hidden" not in header
    assert "cudaLaunchKernel(kernel" in header
    assert "for (int64_t row = 0; row < rows; ++row)" in header
    assert "rows_i32 = 1;" in header
    assert "fused KDA decode repeated-row launch" in header
