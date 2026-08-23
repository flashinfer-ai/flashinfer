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

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flashinfer.diffusion_ops import minimax_h3_bf16_pre_attention
from flashinfer.diffusion_ops.minimax_h3 import _validate_input_contract
from flashinfer.jit import env as jit_env
from flashinfer.jit.minimax_h3 import (
    _minimax_h3_cuda_source,
    _minimax_h3_include_dir,
)
from flashinfer.utils import get_compute_capability, is_sm100f_supported


HIDDEN = 5376
NUM_HEADS = 56
HEAD_DIM = 128
QKV_KINDS = 3
QKV_WIDTH = NUM_HEADS * QKV_KINDS * HEAD_DIM
ROPE_DIM = 96
ADALN_ROWS = 9
EPS = 1.0e-5

CENTER_SHAPES = [
    (33472, 1, "production_segments"),
    (16736, 2, "production_segments"),
    (8368, 4, "production_segments"),
    (4184, 8, "production_segments"),
    (38592, 1, "production_segments"),
    (19296, 2, "production_segments"),
    (9648, 4, "production_segments"),
    (4824, 8, "production_segments"),
    (48768, 1, "production_segments"),
    (24384, 2, "production_segments"),
    (12192, 4, "production_segments"),
    (6096, 8, "production_segments"),
    (58944, 1, "production_segments"),
    (29472, 2, "production_segments"),
    (14736, 4, "production_segments"),
    (7368, 8, "production_segments"),
    (74240, 1, "production_segments"),
    (37120, 2, "production_segments"),
    (18560, 4, "production_segments"),
    (9280, 8, "production_segments"),
    (109952, 1, "production_segments"),
    (54976, 2, "production_segments"),
    (27488, 4, "production_segments"),
    (13744, 8, "production_segments"),
]
ALIGNED_SHAPES = [
    (38528, 1, "production_segments"),
    (38656, 1, "production_segments"),
    (19264, 2, "production_segments"),
    (19328, 2, "production_segments"),
    (9632, 4, "production_segments"),
    (9664, 4, "production_segments"),
    (4816, 8, "production_segments"),
    (4832, 8, "production_segments"),
]
TAIL_SHAPES = [
    (38591, 1, "boundary_segments"),
    (38593, 1, "boundary_segments"),
    (19295, 2, "all_same"),
    (19297, 2, "all_same"),
    (9647, 4, "random"),
    (9649, 4, "random"),
    (4823, 8, "boundary_segments"),
    (4825, 8, "boundary_segments"),
]
SMOKE_SHAPES = [
    (1, 8, "all_same"),
    (127, 8, "boundary_segments"),
    (128, 8, "production_segments"),
    (129, 8, "random"),
]
FULL_CORRECTNESS_SHAPES = CENTER_SHAPES + ALIGNED_SHAPES + TAIL_SHAPES + SMOKE_SHAPES

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "csrc"
    / "minimax_h3_bf16_pre_attention_sm103a.cu"
)
_CUDA_DEVICE = torch.device("cuda")
_HAS_SM103A_RUNTIME = (
    _SOURCE.is_file()
    and torch.cuda.is_available()
    and get_compute_capability(_CUDA_DEVICE) == (10, 3)
    and is_sm100f_supported(_CUDA_DEVICE)
)
_RUN_FULL = os.environ.get("FLASHINFER_RUN_FULL_MINIMAX_H3_TESTS", "0") == "1"


def _make_meta_case(m: int = 129, p: int = 8):
    def tensor(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    return {
        "x": tensor((m, HIDDEN)),
        "x_norm_weight": tensor((HIDDEN,)),
        "adaln_scale": tensor((ADALN_ROWS, HIDDEN)),
        "adaln_shift": tensor((ADALN_ROWS, HIDDEN)),
        "adaln_index": tensor((m,), torch.int32),
        "qkv_weight": tensor((QKV_WIDTH, HIDDEN)),
        "q_norm_weight": tensor((HEAD_DIM,)),
        "k_norm_weight": tensor((HEAD_DIM,)),
        "rope_cos_sin": tensor((m, ROPE_DIM)),
        "out": tensor((p, m, NUM_HEADS // p, QKV_KINDS, HEAD_DIM)),
        "ulysses_degree": p,
        "eps": EPS,
    }


def _validate(case):
    _validate_input_contract(
        case["x"],
        case["x_norm_weight"],
        case["adaln_scale"],
        case["adaln_shift"],
        case["adaln_index"],
        case["qkv_weight"],
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["rope_cos_sin"],
        case["out"],
        ulysses_degree=case["ulysses_degree"],
        eps=case["eps"],
    )


def test_public_signature_requires_destination():
    signature = inspect.signature(minimax_h3_bf16_pre_attention)
    assert signature.parameters["out"].default is inspect.Parameter.empty
    assert signature.parameters["eps"].default == EPS


def test_fi_trace_contract():
    case = _make_meta_case(m=129, p=8)
    definition = minimax_h3_bf16_pre_attention.fi_trace(**case)
    assert definition["op_type"] == "minimax_h3_bf16_pre_attention"
    assert definition["axes"]["num_tokens"]["type"] == "var"
    assert definition["axes"]["hidden_size"]["value"] == HIDDEN
    assert definition["axes"]["ulysses_degree"]["type"] == "var"
    assert definition["axes"]["heads_per_destination"]["type"] == "var"
    assert "num_heads" not in definition["axes"]
    assert (
        "qkv_width == ulysses_degree * heads_per_destination * qkv_kinds * head_dim"
        in definition["constraints"]
    )
    expected_shape = [
        "ulysses_degree",
        "num_tokens",
        "heads_per_destination",
        "qkv_kinds",
        "head_dim",
    ]
    assert definition["outputs"]["out"]["shape"] == expected_shape
    assert definition["outputs"]["out"]["dtype"] == "bfloat16"
    assert case["ulysses_degree"] == case["out"].shape[0] == 8
    assert NUM_HEADS // case["ulysses_degree"] == case["out"].shape[2] == 7


def test_jit_source_resolution_supports_package_and_source_tree(monkeypatch, tmp_path):
    packaged_csrc = tmp_path / "data" / "csrc"
    packaged_csrc.mkdir(parents=True)
    packaged_source = packaged_csrc / _SOURCE.name
    packaged_source.write_text("packaged source", encoding="utf-8")
    packaged_include = tmp_path / "data" / "include"
    packaged_include.mkdir(parents=True)
    monkeypatch.setattr(jit_env, "FLASHINFER_CSRC_DIR", packaged_csrc)
    monkeypatch.setattr(jit_env, "FLASHINFER_INCLUDE_DIR", packaged_include)

    assert _minimax_h3_cuda_source() == packaged_source
    assert _minimax_h3_include_dir() == packaged_include
    packaged_source.unlink()
    packaged_csrc.rmdir()
    packaged_include.rmdir()
    assert _minimax_h3_cuda_source() == _SOURCE
    assert _minimax_h3_include_dir() == _SOURCE.parents[1] / "include"


def test_frozen_source_contains_index_and_tmem_safety_guards():
    source = _SOURCE.read_text(encoding="utf-8")
    assert "table_row >= 0 && table_row < 9" in source
    assert "tcgen05.fence::after_thread_sync;" in source
    assert "tcgen05.wait::ld.sync.aligned;" in source
    assert "tcgen05.fence::before_thread_sync;" in source


def test_frozen_source_uses_launch_parameter_tensor_map():
    source = _SOURCE.read_text(encoding="utf-8")
    assert "__grid_constant__ CUtensorMap qkv_weight" in source
    assert "cuMemAlloc" not in source
    assert "qkv_weight tensor-map cache" not in source


@pytest.mark.parametrize("p", [1, 2, 4, 8])
def test_valid_meta_contract(p):
    _validate(_make_meta_case(p=p))


@pytest.mark.parametrize(
    "field,replacement,match",
    [
        (
            "x",
            torch.empty((129, HIDDEN - 1), dtype=torch.bfloat16, device="meta"),
            "x must have shape",
        ),
        (
            "adaln_scale",
            torch.empty((ADALN_ROWS - 1, HIDDEN), dtype=torch.bfloat16, device="meta"),
            "adaln_scale shape",
        ),
        (
            "adaln_index",
            torch.empty((129,), dtype=torch.int64, device="meta"),
            "adaln_index dtype",
        ),
        (
            "qkv_weight",
            torch.empty((HIDDEN, QKV_WIDTH), dtype=torch.bfloat16, device="meta").t(),
            "qkv_weight must be contiguous",
        ),
        (
            "rope_cos_sin",
            torch.empty((129, HEAD_DIM), dtype=torch.bfloat16, device="meta"),
            "rope_cos_sin shape",
        ),
        (
            "out",
            torch.empty(
                (8, 129, 7, QKV_KINDS, HEAD_DIM - 1),
                dtype=torch.bfloat16,
                device="meta",
            ),
            "out shape",
        ),
        ("ulysses_degree", 3, "ulysses_degree"),
        ("eps", 1.0e-6, "eps must be"),
    ],
)
def test_invalid_meta_contract(field, replacement, match):
    case = _make_meta_case()
    case[field] = replacement
    with pytest.raises(ValueError, match=match):
        _validate(case)


def _make_adaln_index(m: int, profile: str, *, device, generator):
    rows = torch.arange(m, dtype=torch.int64, device=device)
    if profile == "production_segments":
        return (
            torch.div(rows * ADALN_ROWS, m, rounding_mode="floor")
            .clamp_max(8)
            .to(torch.int32)
        )
    if profile == "boundary_segments":
        return (
            torch.div(rows, 127, rounding_mode="floor")
            .remainder(ADALN_ROWS)
            .to(torch.int32)
        )
    if profile == "all_same":
        return torch.full((m,), 8, dtype=torch.int32, device=device)
    return torch.randint(
        0, ADALN_ROWS, (m,), dtype=torch.int32, device=device, generator=generator
    )


def _make_rope_cache(m: int, *, device):
    rows = torch.arange(m, dtype=torch.float32, device=device)
    axes = (
        torch.div(rows, 4096, rounding_mode="floor"),
        torch.div(rows, 64, rounding_mode="floor").remainder(64),
        rows.remainder(64),
    )
    inv_freq = torch.pow(
        torch.tensor(10000.0, dtype=torch.float32, device=device),
        -torch.arange(16, dtype=torch.float32, device=device) / 16.0,
    )
    phase = torch.cat([axis[:, None] * inv_freq[None, :] for axis in axes], dim=-1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(torch.bfloat16).contiguous()


def _apply_rope(x, rope_cos_sin):
    rotary = x[..., :ROPE_DIM].float()
    tail = x[..., ROPE_DIM:]
    cos_half = rope_cos_sin[:, :48].float()
    sin_half = rope_cos_sin[:, 48:].float()
    cos = torch.cat((cos_half, cos_half), dim=-1)[:, None, :]
    sin = torch.cat((sin_half, sin_half), dim=-1)[:, None, :]
    rotated_half = torch.cat((-rotary[..., 48:], rotary[..., :48]), dim=-1)
    rotated = (rotary * cos + rotated_half * sin).to(torch.bfloat16)
    return torch.cat((rotated, tail), dim=-1)


def _reference(case):
    norm = F.rms_norm(case["x"], (HIDDEN,), case["x_norm_weight"], eps=EPS).to(
        torch.bfloat16
    )
    index = case["adaln_index"].long()
    valid_index = (index >= 0) & (index < ADALN_ROWS)
    safe_index = index.clamp(0, ADALN_ROWS - 1)
    scale = case["adaln_scale"].index_select(0, safe_index)
    shift = case["adaln_shift"].index_select(0, safe_index)
    adaln = torch.addcmul(shift, norm, (scale + 1.0).to(torch.bfloat16)).to(
        torch.bfloat16
    )
    adaln = torch.where(valid_index[:, None], adaln, torch.zeros_like(adaln))
    qkv = F.linear(adaln, case["qkv_weight"]).to(torch.bfloat16)
    grouped = qkv.view(case["x"].shape[0], NUM_HEADS, QKV_KINDS, HEAD_DIM)
    q = F.rms_norm(grouped[:, :, 0, :], (HEAD_DIM,), case["q_norm_weight"], eps=EPS).to(
        torch.bfloat16
    )
    k = F.rms_norm(grouped[:, :, 1, :], (HEAD_DIM,), case["k_norm_weight"], eps=EPS).to(
        torch.bfloat16
    )
    q = _apply_rope(q, case["rope_cos_sin"])
    k = _apply_rope(k, case["rope_cos_sin"])
    fused = torch.stack((q, k, grouped[:, :, 2, :]), dim=2)
    p = case["ulysses_degree"]
    return (
        fused.view(case["x"].shape[0], p, NUM_HEADS // p, QKV_KINDS, HEAD_DIM)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def _make_cuda_case(m: int, p: int, profile: str):
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(4532 + m + p)

    def normal(shape, std):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return out.normal_(0.0, std, generator=generator)

    def uniform(shape, low, high):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return out.uniform_(low, high, generator=generator)

    return {
        "x": normal((m, HIDDEN), 0.5),
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "adaln_index": _make_adaln_index(
            m, profile, device=device, generator=generator
        ),
        "qkv_weight": normal((QKV_WIDTH, HIDDEN), 0.01),
        "q_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "rope_cos_sin": _make_rope_cache(m, device=device),
        "out": torch.empty(
            (p, m, NUM_HEADS // p, QKV_KINDS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        ),
        "ulysses_degree": p,
        "eps": EPS,
    }


def _run_correctness_shape(m: int, p: int, profile: str):
    case = _make_cuda_case(m, p, profile)
    expected = _reference(case)
    actual = minimax_h3_bf16_pre_attention(**case)
    assert actual.data_ptr() == case["out"].data_ptr()
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(
        actual[:, :, :, 0, ROPE_DIM:],
        expected[:, :, :, 0, ROPE_DIM:],
        atol=0.01,
        rtol=0.01,
    )


@pytest.mark.skipif(
    not _HAS_SM103A_RUNTIME,
    reason="requires the frozen SM103a CUDA source and an SM103a GPU",
)
@pytest.mark.parametrize("m,p,profile", SMOKE_SHAPES)
def test_sm103a_smoke_correctness(m, p, profile):
    _run_correctness_shape(m, p, profile)


@pytest.mark.skipif(
    not _HAS_SM103A_RUNTIME,
    reason="requires the frozen SM103a CUDA source and an SM103a GPU",
)
def test_sm103a_invalid_adaln_indices_produce_zero_rows():
    case = _make_cuda_case(6, 8, "all_same")
    case["adaln_index"] = torch.tensor(
        [0, -1, 8, 9, -(2**31), 2**31 - 1], dtype=torch.int32, device="cuda"
    )
    expected = _reference(case)
    actual = minimax_h3_bf16_pre_attention(**case)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    assert not torch.count_nonzero(actual[:, [1, 3, 4, 5]])


@pytest.mark.skipif(
    not _HAS_SM103A_RUNTIME,
    reason="requires the frozen SM103a CUDA source and an SM103a GPU",
)
def test_sm103a_cuda_graph_capture():
    case = _make_cuda_case(128, 8, "production_segments")
    expected = _reference(case)
    minimax_h3_bf16_pre_attention(**case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = minimax_h3_bf16_pre_attention(**case)
    case["out"].zero_()
    assert not torch.count_nonzero(case["out"])
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)


@pytest.mark.skipif(
    not (_HAS_SM103A_RUNTIME and _RUN_FULL),
    reason="set FLASHINFER_RUN_FULL_MINIMAX_H3_TESTS=1 to run the 44-shape suite",
)
@pytest.mark.parametrize("m,p,profile", FULL_CORRECTNESS_SHAPES)
def test_sm103a_full_correctness(m, p, profile):
    _run_correctness_shape(m, p, profile)
