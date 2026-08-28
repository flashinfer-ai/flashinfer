# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.gemm import gemm_bf16_fp4_blackwell as blackwell_backend
from flashinfer.jit import blackwell_bf16_fp4 as blackwell_jit


def _manifest(target: str) -> dict:
    arch = {"sm100": "sm_100a", "sm103": "sm_103a"}[target]
    return {
        "schema_version": 3,
        "bundle": "flashinfer_blackwell_bf16_fp4_gemm",
        "arch": arch,
        "tma_abi": "pointer",
        "tensor_map_abi": {
            "public_type": "FlashInferTensorMap",
            "cuda_type": "CUtensorMap",
            "size_bytes": 128,
            "alignment_bytes": 128,
        },
        "adapter_boundary": "separate_translation_unit",
        "prepared_abis": {
            "cudnn": {
                "B": {"dtype": "uint8", "shape": ["N", "K/2"]},
                "B_descale": {
                    "dtype": "float8_e4m3fn",
                    "shape": ["N", "K/16"],
                },
            },
            "cute_dsl": {
                "B": {"dtype": "int32", "shape": ["K/16", "N*2"]},
                "B_descale": {"dtype": "uint8", "shape": ["K/16", "N"]},
            },
        },
        "ir_symbols": [f"synthetic_ir_{index}" for index in range(14)],
        "kernels": [
            {
                "kernel_symbol": f"synthetic_kernel_{index}",
                "arg_plan": [["tensor", "A"]],
                "tma_descriptors": [],
            }
            for index in range(74)
        ],
        "dispatch": {
            "selection": "ordered_first_match_after_input_validation",
            "inputs": [
                "backend",
                "out_dtype",
                "M",
                "N",
                "K",
                "has_alpha",
                "enable_pdl",
            ],
            "routes": [{} for _ in range(11)],
        },
    }


def _write_manifest(path: Path, manifest: dict) -> bytes:
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _source_header(target: str, manifest_raw: bytes) -> bytes:
    target_sm = {"sm100": 100, "sm103": 103}[target]
    raw_source_hash = hashlib.sha256(b"synthetic generated source").hexdigest()
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    return (
        "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1\n"
        "#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_VERSION 2\n"
        f"#define FLASHINFER_BLACKWELL_BF16_FP4_TARGET_SM {target_sm}\n"
        "#define FLASHINFER_BLACKWELL_BF16_FP4_RAW_SOURCE_SHA256 "
        f'"{raw_source_hash}"\n'
        "#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256 "
        f'"{manifest_hash}"\n'
    ).encode()


@pytest.mark.parametrize("target", ["sm100", "sm103"])
def test_schema_3_manifest_accepts_both_generated_abi_families(
    tmp_path: Path, target: str
) -> None:
    path = tmp_path / blackwell_jit._MANIFEST_NAMES[target]
    raw = _write_manifest(path, _manifest(target))

    parsed, parsed_raw = blackwell_jit._load_abi_manifest(path, target)

    assert parsed_raw == raw
    assert parsed["arch"] == blackwell_jit._NVCC_ARCH[target]
    assert set(parsed["prepared_abis"]) == {"cudnn", "cute_dsl"}
    blackwell_jit._validate_source_header(_source_header(target, raw), raw, target)


@pytest.mark.parametrize(
    ("capability", "target"), [((10, 0), "sm100"), ((10, 3), "sm103")]
)
def test_compute_capability_routing_is_exact(
    capability: tuple[int, int], target: str
) -> None:
    assert blackwell_jit._target_for_capability(capability) == target


@pytest.mark.parametrize("capability", [(9, 0), (10, 1), (12, 0)])
def test_compute_capability_routing_rejects_other_targets(
    capability: tuple[int, int]
) -> None:
    with pytest.raises(ValueError, match="requires compute capability 10.0 or 10.3"):
        blackwell_jit._target_for_capability(capability)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version=3"),
        ("arch", "sm_103a", "architecture does not match"),
        ("tma_abi", "value", "requires pointer TMA ABI"),
        ("tensor_map_abi", {}, "incompatible TensorMap ABI"),
        ("prepared_abis", {}, "incompatible prepared layouts"),
    ],
)
def test_manifest_rejects_incompatible_abi_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest = copy.deepcopy(_manifest("sm100"))
    manifest[field] = value
    path = tmp_path / "generated.abi.json"
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match=message):
        blackwell_jit._load_abi_manifest(path, "sm100")


def test_manifest_rejects_duplicate_keys(tmp_path: Path) -> None:
    raw = json.dumps(_manifest("sm100")).replace(
        '"schema_version": 3', '"schema_version": 3, "schema_version": 3', 1
    )
    path = tmp_path / "generated.abi.json"
    path.write_text(raw)

    with pytest.raises(ValueError, match="duplicate ABI manifest key"):
        blackwell_jit._load_abi_manifest(path, "sm100")


def test_source_and_manifest_both_participate_in_cache_identity() -> None:
    nvcc = Path("/opt/cuda/bin/nvcc")
    base = blackwell_jit._source_package_key(
        "sm100", b"source-a", b"manifest-a", b"binding", nvcc
    )

    assert base != blackwell_jit._source_package_key(
        "sm100", b"source-b", b"manifest-a", b"binding", nvcc
    )
    assert base != blackwell_jit._source_package_key(
        "sm100", b"source-a", b"manifest-b", b"binding", nvcc
    )


@pytest.mark.parametrize(
    ("backend", "out_dtype"),
    [
        ("blackwell-native", torch.bfloat16),
        ("blackwell-native", torch.float16),
        ("blackwell-tiled", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("explicit_alpha", [False, True])
def test_synthetic_launch_preserves_alpha_out_and_pdl(
    monkeypatch: pytest.MonkeyPatch,
    backend: blackwell_backend.BlackwellBf16Fp4Backend,
    out_dtype: torch.dtype,
    enable_pdl: bool,
    explicit_alpha: bool,
) -> None:
    a = torch.zeros((2, 16), dtype=torch.bfloat16)
    if backend == "blackwell-native":
        b = torch.zeros((64, 8), dtype=torch.uint8)
        b_descale = torch.zeros((64, 1), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        layout_code = 0
    else:
        b = torch.zeros((1, 128), dtype=torch.int32)
        b_descale = torch.zeros((1, 64), dtype=torch.uint8)
        layout_code = 1
    alpha = torch.ones((1,), dtype=torch.float32) if explicit_alpha else None
    out = torch.empty((2, 64), dtype=out_dtype)
    captured: list[tuple] = []

    monkeypatch.setattr(
        blackwell_backend, "_require_blackwell_source_arch", lambda _device: None
    )
    monkeypatch.setattr(
        blackwell_backend,
        "_get_blackwell_bf16_fp4_module",
        lambda: SimpleNamespace(run=lambda *args: captured.append(args)),
    )

    result = blackwell_backend._compute_blackwell_bf16_fp4(
        a,
        b,
        b_descale,
        alpha,
        out_dtype,
        out,
        16,
        enable_pdl,
        backend,
    )

    assert result is out
    assert len(captured) == 1
    args = captured[0]
    assert len(args) == 7
    assert args[0] is a
    assert args[1] is b
    assert args[2].data_ptr() == b_descale.data_ptr()
    assert args[2].dtype == torch.uint8
    assert args[4] is out
    assert args[5:] == (layout_code, enable_pdl)
    if explicit_alpha:
        assert args[3] is alpha
    else:
        assert args[3].data_ptr() == a.data_ptr()
