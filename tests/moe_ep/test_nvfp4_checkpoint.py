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

import ast
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_checkpoint import (
    NVFP4Checkpoint,
    load_modelopt_nvfp4_state_dict,
    reference_dequantize_nvfp4 as canonical_dequantize_nvfp4,
)

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _independent_dequantize_nvfp4(
    payload: torch.Tensor, scales: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=payload.device)
    low = table[(payload & 0x0F).long()]
    high = table[(payload >> 4).long()]
    decoded = torch.stack((low, high), dim=-1).flatten(-2)
    decoded = decoded.view(*decoded.shape[:-1], -1, 16)
    decoded = decoded * scales.float().unsqueeze(-1)
    if alpha.numel() == 1:
        multiplier = alpha.reshape(1, 1, 1, 1)
    else:
        multiplier = alpha.reshape(-1, 1, 1, 1)
    return (decoded * multiplier).flatten(-2)


_BUNDLED_MODELOPT_GOLDEN = (
    Path(__file__).with_name("data") / "modelopt_w4a16_nvfp4_v1.safetensors"
)
_MODELOPT_GOLDEN_VERSION = "0.45.0"
_MODELOPT_GOLDEN_COMMIT = "ec87a82927d003986d44fb7f4fa8b3d10c31b095"
_MODELOPT_GOLDEN_SHA256 = (
    "532857e12aa4d70279dcd1bdd2219d184d549844849d45c2222fd7b2ed05f513"
)
_MODELOPT_GOLDEN_DEQUANT_SHA256 = (
    "255c4393f1ff9a228bef639a018353e2d531459bc1a37694922f0c26318d39d5"
)
_MODELOPT_GOLDEN_PREFIX = "model.layers.0.self_attn.q_proj"


def _checkpoint(
    payload: torch.Tensor,
    scales: torch.Tensor,
    alpha: torch.Tensor,
    logical_shape: tuple[int, int, int] | None = None,
    expert_mapping: tuple[int, ...] | None = None,
) -> NVFP4Checkpoint:
    physical_shape = (payload.shape[0], payload.shape[1], payload.shape[2] * 2)
    return NVFP4Checkpoint(
        payload,
        scales,
        alpha,
        physical_shape if logical_shape is None else logical_shape,
        tuple(range(payload.shape[0])) if expert_mapping is None else expert_mapping,
        "test.nvfp4.v1",
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def _load_safetensors_subset(
    path: Path, tensor_names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    if sys.byteorder != "little":
        raise RuntimeError("the bundled safetensors fixture requires little-endian")
    blob = path.read_bytes()
    if len(blob) < 8:
        raise ValueError("safetensors fixture is truncated before its header")
    header_size = int.from_bytes(blob[:8], "little")
    data_start = 8 + header_size
    if data_start > len(blob):
        raise ValueError("safetensors fixture has an invalid header size")
    header = json.loads(blob[8:data_start].decode("utf-8"))
    dtype_map = {
        "U8": (torch.uint8, 1),
        "F8_E4M3": (torch.float8_e4m3fn, 1),
        "F32": (torch.float32, 4),
    }
    result = {}
    for name in tensor_names:
        entry = header[name]
        dtype_name = entry["dtype"]
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported safetensors dtype {dtype_name!r}")
        dtype, width = dtype_map[dtype_name]
        begin, end = entry["data_offsets"]
        if not (0 <= begin <= end <= len(blob) - data_start):
            raise ValueError(f"invalid safetensors offsets for {name!r}")
        raw = torch.frombuffer(
            bytearray(blob[data_start + begin : data_start + end]), dtype=torch.uint8
        )
        if raw.numel() % width:
            raise ValueError(f"misaligned safetensors storage for {name!r}")
        tensor = raw if dtype == torch.uint8 else raw.view(dtype)
        shape = tuple(entry["shape"])
        expected_elements = 1
        for dimension in shape:
            expected_elements *= dimension
        if tensor.numel() != expected_elements:
            raise ValueError(f"safetensors shape disagrees with storage for {name!r}")
        result[name] = tensor.reshape(shape).clone().contiguous()
    return result


def test_canonical_checkpoint_decoder_imports_only_stdlib_and_torch():
    spec = find_spec(
        "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_checkpoint"
    )
    assert spec is not None and spec.origin is not None
    source_path = Path(spec.origin)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "canonical decoder must not use relative imports"
            assert node.module is not None
            imports.add(node.module)

    assert imports == {"__future__", "collections.abc", "dataclasses", "torch"}
    dynamic_import_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"__import__", "eval", "exec"}
    }
    assert not dynamic_import_calls
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == "ops"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "torch"
        and any(alias.name == "ops" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_nvfp4_checkpoint_decodes_all_codes_low_nibble_first():
    payload = torch.tensor(
        [
            [
                [
                    (high << 4) | low
                    for low, high in zip(range(0, 16, 2), range(1, 16, 2), strict=True)
                ]
            ]
        ],
        dtype=torch.uint8,
    )
    scales = torch.ones((1, 1, 1), dtype=torch.float32).to(torch.float8_e4m3fn)
    decoded = canonical_dequantize_nvfp4(
        _checkpoint(payload, scales, torch.ones(1, dtype=torch.float32))
    )
    expected = torch.tensor(_E2M1_VALUES, dtype=torch.float32).reshape(1, 1, 16)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    assert not torch.signbit(decoded[0, 0, 0])
    assert torch.signbit(decoded[0, 0, 8])


def test_nvfp4_checkpoint_matches_existing_independent_reference():
    payload = torch.tensor(
        [
            [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]],
            [[0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]],
        ],
        dtype=torch.uint8,
    )
    scales = torch.tensor([0.5, 448.0], dtype=torch.float32).to(torch.float8_e4m3fn)[
        :, None, None
    ]
    alpha = torch.tensor([0.25, 2.0], dtype=torch.float32)
    actual = canonical_dequantize_nvfp4(
        _checkpoint(payload, scales, alpha, expert_mapping=(4, 9))
    )
    expected = _independent_dequantize_nvfp4(payload, scales, alpha)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvfp4_checkpoint_scale_extremes():
    raw_scales = torch.tensor([0x01, 0x7E], dtype=torch.uint8).view(torch.float8_e4m3fn)
    payload = torch.full((2, 1, 8), 0x77, dtype=torch.uint8)
    checkpoint = _checkpoint(
        payload,
        raw_scales[:, None, None],
        torch.ones(2, dtype=torch.float32),
    )
    decoded = canonical_dequantize_nvfp4(checkpoint)
    assert decoded[0, 0, 0].item() == 6.0 * (2.0**-9)
    assert decoded[1, 0, 0].item() == 6.0 * 448.0


@pytest.mark.parametrize(
    ("raw_scale", "match"),
    [
        (0x7F, "finite values"),
        (0xFF, "finite values"),
        (0x80, "negative zero"),
        (0x81, "non-negative values"),
    ],
)
def test_nvfp4_checkpoint_rejects_noncanonical_scale_bits(raw_scale, match):
    payload = torch.zeros((1, 1, 8), dtype=torch.uint8)
    scales = torch.tensor([raw_scale], dtype=torch.uint8).view(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=match):
        _checkpoint(payload, scales.reshape(1, 1, 1), torch.ones(1))


def test_nvfp4_checkpoint_crops_k_tail_and_padding():
    payload = torch.full((1, 4, 16), 0x77, dtype=torch.uint8)
    payload[0, 0, 8] = 0x21
    scales = torch.ones((1, 4, 2), dtype=torch.float32).to(torch.float8_e4m3fn)
    checkpoint = _checkpoint(
        payload,
        scales,
        torch.ones(1, dtype=torch.float32),
        logical_shape=(1, 2, 17),
    )
    decoded = canonical_dequantize_nvfp4(checkpoint)
    assert decoded.shape == (1, 2, 17)
    assert decoded[0, 0, 16].item() == 0.5
    assert decoded[0, 0, 15].item() == 6.0
    assert decoded[0, 1, 16].item() == 6.0


def test_nvfp4_checkpoint_supports_empty_expert_axis():
    payload = torch.empty((0, 2, 8), dtype=torch.uint8)
    scales = torch.empty((0, 2, 1), dtype=torch.float8_e4m3fn)
    checkpoint = _checkpoint(
        payload,
        scales,
        torch.empty((0,), dtype=torch.float32),
        logical_shape=(0, 2, 16),
        expert_mapping=(),
    )
    decoded = canonical_dequantize_nvfp4(checkpoint)
    assert decoded.shape == (0, 2, 16)


def test_modelopt_state_dict_loader_uses_global_decode_scale_directly():
    state_dict = {
        "experts.weight": torch.tensor([[0x21] * 8], dtype=torch.uint8),
        "experts.weight_scale": torch.ones((1, 1), dtype=torch.float32).to(
            torch.float8_e4m3fn
        ),
        "experts.weight_scale_2": torch.tensor([0.25], dtype=torch.float32),
    }
    checkpoint = load_modelopt_nvfp4_state_dict(
        state_dict,
        prefix="experts",
        expert_mapping=(7,),
        source_format_version="modelopt.test",
    )
    assert checkpoint.logical_shape == (1, 1, 16)
    assert checkpoint.expert_mapping == (7,)
    assert checkpoint.alpha_scope == "per_tensor"
    assert checkpoint.global_alpha.item() == 0.25
    assert canonical_dequantize_nvfp4(checkpoint)[0, 0, 0].item() == 0.125


def test_modelopt_nvfp4_bundled_golden():
    assert _BUNDLED_MODELOPT_GOLDEN.stat().st_size <= 16 * 1024
    assert (
        hashlib.sha256(_BUNDLED_MODELOPT_GOLDEN.read_bytes()).hexdigest()
        == _MODELOPT_GOLDEN_SHA256
    )
    names = tuple(
        f"{_MODELOPT_GOLDEN_PREFIX}.{suffix}"
        for suffix in ("weight", "weight_scale", "weight_scale_2")
    )
    state_dict = _load_safetensors_subset(_BUNDLED_MODELOPT_GOLDEN, names)
    assert state_dict[names[0]].dtype == torch.uint8
    assert state_dict[names[0]].shape == (16, 8)
    assert state_dict[names[1]].dtype == torch.float8_e4m3fn
    assert state_dict[names[1]].shape == (16, 1)
    assert state_dict[names[2]].dtype == torch.float32
    assert state_dict[names[2]].shape == ()

    checkpoint = load_modelopt_nvfp4_state_dict(
        state_dict,
        prefix=_MODELOPT_GOLDEN_PREFIX,
        logical_shape=(1, 16, 16),
        expert_mapping=(17,),
        source_format_version=(
            f"nvidia-modelopt.{_MODELOPT_GOLDEN_VERSION}.w4a16-nvfp4"
        ),
    )
    generator_source = (
        Path(__file__)
        .with_name("generate_modelopt_nvfp4_golden.py")
        .read_text(encoding="utf-8")
    )
    assert _MODELOPT_GOLDEN_COMMIT in generator_source
    assert "mtq.W4A16_NVFP4_CFG" in generator_source
    assert "export_hf_checkpoint(" in generator_source
    assert (
        _tensor_sha256(canonical_dequantize_nvfp4(checkpoint))
        == _MODELOPT_GOLDEN_DEQUANT_SHA256
    )


def test_single_expert_vector_alpha_keeps_per_expert_scope():
    payload = torch.zeros((1, 1, 8), dtype=torch.uint8)
    scales = torch.ones((1, 1, 1), dtype=torch.float32).to(torch.float8_e4m3fn)
    checkpoint = _checkpoint(payload, scales, torch.ones(1, dtype=torch.float32))
    assert checkpoint.alpha_scope == "per_expert"
    assert checkpoint.global_alpha.shape == (1,)


@pytest.mark.parametrize(
    "state_dict,match",
    [
        ({}, "missing ModelOpt"),
        (
            {
                "weight": torch.zeros((1, 8), dtype=torch.int8),
                "weight_scale": torch.ones((1, 1)).to(torch.float8_e4m3fn),
                "weight_scale_2": torch.ones(1, dtype=torch.float32),
            },
            "weight must have dtype",
        ),
        (
            {
                "weight": torch.zeros((1, 8), dtype=torch.uint8),
                "weight_scale": torch.ones((1, 1)).to(torch.float8_e4m3fn),
                "weight_scale_2": torch.zeros(1, dtype=torch.float32),
            },
            "finite and positive",
        ),
        (
            {
                "weight": torch.zeros(8, dtype=torch.uint8),
                "weight_scale": torch.ones((1, 1)).to(torch.float8_e4m3fn),
                "weight_scale_2": torch.ones(1, dtype=torch.float32),
            },
            r"weight must be \[N,K/2\]",
        ),
    ],
)
def test_modelopt_state_dict_loader_rejects_invalid_contract(state_dict, match):
    with pytest.raises((KeyError, ValueError), match=match):
        load_modelopt_nvfp4_state_dict(state_dict)


@pytest.mark.skipif(
    not os.environ.get("FI_NVFP4_GOLDEN_DIR"),
    reason="FI_NVFP4_GOLDEN_DIR is not configured",
)
def test_modelopt_nvfp4_real_golden():
    golden_dir = Path(os.environ["FI_NVFP4_GOLDEN_DIR"])
    state_dict = torch.load(
        golden_dir / "state_dict.pt", map_location="cpu", weights_only=True
    )
    metadata = torch.load(
        golden_dir / "metadata.pt", map_location="cpu", weights_only=True
    )
    expected = torch.load(
        golden_dir / "dequantized.pt", map_location="cpu", weights_only=True
    )
    checkpoint = load_modelopt_nvfp4_state_dict(
        state_dict,
        prefix=metadata["prefix"],
        logical_shape=tuple(metadata["logical_shape"]),
        expert_mapping=tuple(metadata["expert_mapping"]),
        source_format_version=metadata["source_format_version"],
    )
    torch.testing.assert_close(
        canonical_dequantize_nvfp4(checkpoint),
        expected.to(torch.float32),
        rtol=0,
        atol=0,
    )
