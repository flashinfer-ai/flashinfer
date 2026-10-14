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
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flashinfer.diffusion_ops import minimax_h3_bf16_pre_attention
from flashinfer.diffusion_ops import minimax_h3 as minimax_h3_module
from flashinfer.diffusion_ops.minimax_h3 import _validate_input_contract
from flashinfer.jit import env as jit_env
from flashinfer.jit.cake_minimax_h3_bf16_pre_attention import (
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
# Production default table row count; the operator accepts any rows >= 1.
ADALN_ROWS = 9
# Engine modulation projection: tables are column chunks of a [rows, 6 * 5376] buffer.
ENGINE_TABLE_CHUNKS = 6
ENGINE_TABLE_ROWS = [3, 6, 12]
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
    / "cake_minimax_h3_bf16_pre_attention_sm103a.cu"
)
_CUDA_DEVICE = torch.device("cuda")
_HAS_BLACKWELL_RUNTIME = (
    _SOURCE.is_file()
    and torch.cuda.is_available()
    and get_compute_capability(_CUDA_DEVICE) in {(10, 0), (10, 3)}
    and is_sm100f_supported(_CUDA_DEVICE)
)
_RUN_FULL = os.environ.get("FLASHINFER_RUN_FULL_MINIMAX_H3_TESTS", "0") == "1"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)
requires_blackwell = pytest.mark.skipif(
    not _HAS_BLACKWELL_RUNTIME,
    reason="requires the frozen CUDA source and an SM100a or SM103a GPU",
)


def _meta(shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _make_empty_case(m: int = 129, p: int = 8, device: str = "meta"):
    """Uninitialised operands of the contract shapes (validation-only cases)."""

    def tensor(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)

    return {
        "x": tensor((m, HIDDEN)),
        "x_norm_weight": tensor((HIDDEN,)),
        "adaln_scale": tensor((ADALN_ROWS, HIDDEN)),
        "adaln_shift": tensor((ADALN_ROWS, HIDDEN)),
        "adaln_index": tensor((m,), torch.int64),
        "qkv_weight": tensor((QKV_WIDTH, HIDDEN)),
        "q_norm_weight": tensor((HEAD_DIM,)),
        "k_norm_weight": tensor((HEAD_DIM,)),
        "rope_cos_sin": tensor((m, ROPE_DIM)),
        "rope_positions": None,
        "out": tensor((p, m, NUM_HEADS // p, QKV_KINDS, HEAD_DIM)),
        "ulysses_degree": p,
        "eps": EPS,
        "qk_eps": None,
    }


def _make_meta_case(m: int = 129, p: int = 8):
    return _make_empty_case(m, p, device="meta")


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
        case["rope_positions"],
        case["out"],
        ulysses_degree=case["ulysses_degree"],
        eps=case["eps"],
        qk_eps=case["qk_eps"],
    )


def _engine_tables(rows: int, *, device, fill=None):
    """``(adaln_shift, adaln_scale)`` as column chunks 0 and 1 of a ``[rows, 6 * 5376]``
    modulation projection (row stride ``6 * 5376`` elements), never copied."""
    proj = torch.empty(
        (rows, ENGINE_TABLE_CHUNKS * HIDDEN), dtype=torch.bfloat16, device=device
    )
    if fill is not None:
        fill(proj)
    shift = proj[:, 0:HIDDEN]
    scale = proj[:, HIDDEN : 2 * HIDDEN]
    assert scale.stride() == (ENGINE_TABLE_CHUNKS * HIDDEN, 1)
    assert not scale.is_contiguous()
    return shift, scale


def test_public_signature_requires_destination():
    signature = inspect.signature(minimax_h3_bf16_pre_attention)
    assert signature.parameters["out"].default is inspect.Parameter.empty
    assert signature.parameters["eps"].default == EPS
    assert signature.parameters["qk_eps"].default is None
    assert signature.parameters["rope_positions"].default is None
    assert signature.parameters["rope_positions"].kind is inspect.Parameter.KEYWORD_ONLY


def test_fi_trace_contract():
    case = _make_meta_case(m=129, p=8)
    definition = minimax_h3_bf16_pre_attention.fi_trace(**case)
    assert definition["op_type"] == "minimax_h3_bf16_pre_attention"
    assert definition["axes"]["num_tokens"]["type"] == "var"
    assert definition["axes"]["hidden_size"]["value"] == HIDDEN
    assert definition["axes"]["adaln_rows"]["type"] == "var"
    assert definition["axes"]["rope_cache_rows"]["type"] == "var"
    assert definition["axes"]["ulysses_degree"]["type"] == "var"
    assert definition["axes"]["heads_per_destination"]["type"] == "var"
    assert "num_heads" not in definition["axes"]
    assert (
        "qkv_width == ulysses_degree * heads_per_destination * qkv_kinds * head_dim"
        in definition["constraints"]
    )
    assert definition["inputs"]["rope_positions"]["optional"] is True
    assert definition["inputs"]["qk_eps"]["optional"] is True
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
    # The int64 AdaLN index is range-checked against the runtime row count
    # before any table address is formed (out-of-range -> zero activation
    # row); neither bound is a literal.
    assert re.search(
        r"table_index >= 0(LL)? && table_index < \(?\s*\(?long long\)?\s*\)?\(?adaln_rows\)?",
        source,
    )
    # TMEM ownership protocol of the GEMM (same surface as the fc1 / out_proj stages).
    assert "tcgen05.fence::after_thread_sync;" in source
    assert "tcgen05.wait::ld.sync.aligned;" in source


def test_frozen_source_uses_launch_parameter_tensor_maps():
    source = _SOURCE.read_text(encoding="utf-8")
    assert "__grid_constant__ CUtensorMap activation" in source
    assert "__grid_constant__ CUtensorMap qkv_weight" in source
    assert "cuMemAlloc" not in source
    assert "qkv_weight tensor-map cache" not in source
    # Two launches: the plain norm/AdaLN kernel and the cluster-launched GEMM.
    assert "cudaLaunchKernelEx" in source
    assert "TensorView workspace" in source


@pytest.mark.parametrize("p", [1, 2, 4, 8])
def test_valid_meta_contract(p):
    _validate(_make_meta_case(p=p))


@pytest.mark.parametrize("rows", ENGINE_TABLE_ROWS)
def test_valid_meta_contract_engine_tables(rows):
    case = _make_meta_case()
    case["adaln_shift"], case["adaln_scale"] = _engine_tables(rows, device="meta")
    _validate(case)


def test_valid_meta_contract_rope_cache_and_positions():
    m = 129
    case = _make_meta_case(m=m)
    # Identity positions: the cache may be longer than M.
    case["rope_cos_sin"] = _meta((4 * m + 3, ROPE_DIM))
    _validate(case)
    # Explicit positions: the cache may be shorter than M.
    case["rope_cos_sin"] = _meta((1, ROPE_DIM))
    case["rope_positions"] = _meta((m,), torch.int64)
    _validate(case)
    # Distinct Q/K epsilon and an arbitrary input epsilon are runtime values.
    case["eps"] = 1.0e-6
    case["qk_eps"] = 1.0e-3
    _validate(case)


@pytest.mark.parametrize(
    "field,replacement,match",
    [
        ("x", _meta((129, HIDDEN - 1)), "x must have shape"),
        ("adaln_scale", _meta((HIDDEN,)), "adaln_scale shape"),
        ("adaln_scale", _meta((0, HIDDEN)), "adaln_scale shape"),
        (
            "adaln_scale",
            _meta((ADALN_ROWS, HIDDEN), torch.float16),
            "adaln_scale dtype",
        ),
        (
            "adaln_scale",
            _meta((HIDDEN, 3)).t(),
            "adaln_scale must have a unit last stride",
        ),
        ("adaln_shift", _meta((3, HIDDEN + 4))[:, :HIDDEN], "adaln_shift row pitch"),
        ("adaln_index", _meta((129,), torch.int32), "adaln_index dtype"),
        (
            "adaln_index",
            _meta((129, 2), torch.int64)[:, 0],
            "adaln_index must be contiguous",
        ),
        ("qkv_weight", _meta((HIDDEN, QKV_WIDTH)).t(), "qkv_weight must be contiguous"),
        ("rope_cos_sin", _meta((129, HEAD_DIM)), "rope_cos_sin shape"),
        ("rope_cos_sin", _meta((0, ROPE_DIM)), "rope_cos_sin shape"),
        ("rope_cos_sin", _meta((128, ROPE_DIM)), "rope_positions=None"),
        ("rope_positions", _meta((129,), torch.int32), "rope_positions dtype"),
        ("rope_positions", _meta((128,), torch.int64), "rope_positions shape"),
        (
            "out",
            _meta((8, 129, 7, QKV_KINDS, HEAD_DIM - 1)),
            "out shape",
        ),
        ("ulysses_degree", 3, "ulysses_degree"),
        ("eps", "not-a-number", "could not convert"),
    ],
)
def test_invalid_meta_contract(field, replacement, match):
    case = _make_meta_case()
    case[field] = replacement
    with pytest.raises(ValueError, match=match):
        _validate(case)


@requires_cuda
def test_rejects_misaligned_table_base():
    case = _make_empty_case(m=129, device="cuda")
    storage = torch.empty(ADALN_ROWS * HIDDEN + 8, dtype=torch.bfloat16, device="cuda")
    # An 8-byte offset into a 16-byte aligned allocation.
    case["adaln_scale"] = storage[4 : 4 + ADALN_ROWS * HIDDEN].view(ADALN_ROWS, HIDDEN)
    assert case["adaln_scale"].data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="16-byte aligned"):
        _validate(case)


def _make_adaln_index(m: int, profile: str, *, device, generator, rows=ADALN_ROWS):
    positions = torch.arange(m, dtype=torch.int64, device=device)
    if profile == "production_segments":
        return torch.div(positions * rows, m, rounding_mode="floor").clamp_max(rows - 1)
    if profile == "boundary_segments":
        return torch.div(positions, 127, rounding_mode="floor").remainder(rows)
    if profile == "all_same":
        return torch.full((m,), rows - 1, dtype=torch.int64, device=device)
    return torch.randint(
        0, rows, (m,), dtype=torch.int64, device=device, generator=generator
    )


def _make_rope_cache(rows: int, *, device):
    positions = torch.arange(rows, dtype=torch.float32, device=device)
    axes = (
        torch.div(positions, 4096, rounding_mode="floor"),
        torch.div(positions, 64, rounding_mode="floor").remainder(64),
        positions.remainder(64),
    )
    inv_freq = torch.pow(
        torch.tensor(10000.0, dtype=torch.float32, device=device),
        -torch.arange(16, dtype=torch.float32, device=device) / 16.0,
    )
    phase = torch.cat([axis[:, None] * inv_freq[None, :] for axis in axes], dim=-1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(torch.bfloat16).contiguous()


def _apply_rope(x, rope_rows):
    rotary = x[..., :ROPE_DIM].float()
    tail = x[..., ROPE_DIM:]
    cos_half = rope_rows[:, :48].float()
    sin_half = rope_rows[:, 48:].float()
    cos = torch.cat((cos_half, cos_half), dim=-1)[:, None, :]
    sin = torch.cat((sin_half, sin_half), dim=-1)[:, None, :]
    rotated_half = torch.cat((-rotary[..., 48:], rotary[..., :48]), dim=-1)
    rotated = (rotary * cos + rotated_half * sin).to(torch.bfloat16)
    return torch.cat((rotated, tail), dim=-1)


def _rope_rows(case):
    """Per-token RoPE rows: ``cache[positions]`` with positions clamped to the cache, or the
    first ``M`` cache rows for the identity."""
    cache = case["rope_cos_sin"]
    positions = case.get("rope_positions")
    m = case["x"].shape[0]
    if positions is None:
        return cache[:m]
    return cache.index_select(0, positions.clamp(0, cache.shape[0] - 1))


def _reference(case):
    eps = float(case["eps"])
    qk_eps = eps if case.get("qk_eps") is None else float(case["qk_eps"])
    rows = case["adaln_scale"].shape[0]
    norm = F.rms_norm(case["x"], (HIDDEN,), case["x_norm_weight"], eps=eps).to(
        torch.bfloat16
    )
    index = case["adaln_index"].long()
    valid_index = (index >= 0) & (index < rows)
    safe_index = index.clamp(0, rows - 1)
    scale = case["adaln_scale"].index_select(0, safe_index)
    shift = case["adaln_shift"].index_select(0, safe_index)
    adaln = torch.addcmul(shift, norm, (scale + 1.0).to(torch.bfloat16)).to(
        torch.bfloat16
    )
    adaln = torch.where(valid_index[:, None], adaln, torch.zeros_like(adaln))
    qkv = F.linear(adaln, case["qkv_weight"]).to(torch.bfloat16)
    # Engine-resident weight rows are [qkv_kind, head, head_dim]: the projection
    # columns are [q_all | k_all | v_all].
    grouped = qkv.view(case["x"].shape[0], QKV_KINDS, NUM_HEADS, HEAD_DIM).transpose(
        1, 2
    )
    q = F.rms_norm(
        grouped[:, :, 0, :], (HEAD_DIM,), case["q_norm_weight"], eps=qk_eps
    ).to(torch.bfloat16)
    k = F.rms_norm(
        grouped[:, :, 1, :], (HEAD_DIM,), case["k_norm_weight"], eps=qk_eps
    ).to(torch.bfloat16)
    rope_rows = _rope_rows(case)
    q = _apply_rope(q, rope_rows)
    k = _apply_rope(k, rope_rows)
    fused = torch.stack((q, k, grouped[:, :, 2, :]), dim=2)
    p = case["ulysses_degree"]
    return (
        fused.view(case["x"].shape[0], p, NUM_HEADS // p, QKV_KINDS, HEAD_DIM)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def _make_cuda_case(
    m: int,
    p: int,
    profile: str,
    *,
    adaln_rows: int = ADALN_ROWS,
    table_layout: str = "contract",
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(4532 + m + p)

    def normal(shape, std):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return out.normal_(0.0, std, generator=generator)

    def uniform(shape, low, high):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return out.uniform_(low, high, generator=generator)

    if table_layout == "contract":
        adaln_scale = uniform((adaln_rows, HIDDEN), -0.05, 0.05)
        adaln_shift = uniform((adaln_rows, HIDDEN), -0.05, 0.05)
    else:
        adaln_shift, adaln_scale = _engine_tables(
            adaln_rows,
            device=device,
            fill=lambda proj: proj.uniform_(-0.05, 0.05, generator=generator),
        )

    return {
        "x": normal((m, HIDDEN), 0.5),
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": adaln_scale,
        "adaln_shift": adaln_shift,
        "adaln_index": _make_adaln_index(
            m, profile, device=device, generator=generator, rows=adaln_rows
        ),
        "qkv_weight": normal((QKV_WIDTH, HIDDEN), 0.01),
        "q_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "rope_cos_sin": _make_rope_cache(m, device=device),
        "rope_positions": None,
        "out": torch.empty(
            (p, m, NUM_HEADS // p, QKV_KINDS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        ),
        "ulysses_degree": p,
        "eps": EPS,
        "qk_eps": None,
    }


def _run_case(case):
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
    return actual, expected


def _run_correctness_shape(m: int, p: int, profile: str):
    _run_case(_make_cuda_case(m, p, profile))


@requires_blackwell
@pytest.mark.parametrize("m,p,profile", SMOKE_SHAPES)
def test_blackwell_smoke_correctness(m, p, profile):
    _run_correctness_shape(m, p, profile)


@requires_blackwell
@pytest.mark.parametrize("rows", ENGINE_TABLE_ROWS)
@pytest.mark.parametrize("m,p", [(129, 8), (4824, 8)])
def test_blackwell_engine_layout_tables(rows, m, p):
    case = _make_cuda_case(
        m, p, "production_segments", adaln_rows=rows, table_layout="engine"
    )
    assert case["adaln_scale"].stride(0) == ENGINE_TABLE_CHUNKS * HIDDEN
    _run_case(case)
    # The strided views were passed through, not copied.
    assert case["adaln_scale"].stride(0) == ENGINE_TABLE_CHUNKS * HIDDEN
    assert not case["adaln_scale"].is_contiguous()


@requires_blackwell
@pytest.mark.parametrize("rows", [3, ADALN_ROWS])
def test_blackwell_invalid_adaln_indices_produce_zero_rows(rows):
    int64 = torch.iinfo(torch.int64)
    values = [0, -1, rows, rows + 1, -(2**40), 2**40, int64.min, int64.max]
    case = _make_cuda_case(len(values), 8, "all_same", adaln_rows=rows)
    case["adaln_index"] = torch.tensor(values, dtype=torch.int64, device="cuda")
    actual, _ = _run_case(case)
    assert not torch.count_nonzero(actual[:, 1:])
    assert torch.count_nonzero(actual[:, 0])


@requires_blackwell
def test_blackwell_rope_cache_with_positions():
    m, p = 129, 8
    case = _make_cuda_case(m, p, "production_segments")
    cache_rows = 2 * m + 7
    case["rope_cos_sin"] = _make_rope_cache(cache_rows, device="cuda")
    case["rope_positions"] = (
        torch.arange(m, dtype=torch.int64, device="cuda") * 7 + 3
    ).remainder(cache_rows)
    _run_case(case)
    # A different gather must change the rotated columns (the test is sensitive).
    other = dict(case)
    other["rope_positions"] = case["rope_positions"].flip(0)
    assert not torch.allclose(
        _reference(other)[:, :, :, 0, :ROPE_DIM],
        _reference(case)[:, :, :, 0, :ROPE_DIM],
        atol=0.01,
        rtol=0.01,
    )


@requires_blackwell
def test_blackwell_rope_positions_out_of_range_are_clamped():
    m, p = 64, 8
    case = _make_cuda_case(m, p, "production_segments")
    cache_rows = 40
    case["rope_cos_sin"] = _make_rope_cache(cache_rows, device="cuda")
    positions = torch.arange(m, dtype=torch.int64, device="cuda")
    positions[0] = -5
    positions[1] = cache_rows + 9
    positions[2] = -(2**40)
    positions[3] = 2**40
    case["rope_positions"] = positions
    _run_case(case)  # the reference clamps to [0, S)


@requires_blackwell
def test_blackwell_identity_positions_use_longer_cache_and_are_cached():
    m, p = 129, 8
    case = _make_cuda_case(m, p, "production_segments")
    case["rope_cos_sin"] = _make_rope_cache(3 * m, device="cuda")
    case["rope_positions"] = None
    _run_case(case)
    key = (m, torch.device("cuda").index or torch.cuda.current_device())
    cached = minimax_h3_module._IDENTITY_ROPE_POSITIONS[key]
    assert torch.equal(cached, torch.arange(m, dtype=torch.int64, device="cuda"))
    _run_case(case)
    assert minimax_h3_module._IDENTITY_ROPE_POSITIONS[key] is cached


@requires_blackwell
def test_blackwell_distinct_input_and_qk_eps():
    case = _make_cuda_case(129, 8, "production_segments")
    case["eps"] = 1.0e-2
    case["qk_eps"] = 5.0e-2
    _run_case(case)
    # Swapping the two epsilons must be visible within the tolerance.
    swapped = dict(case)
    swapped["eps"], swapped["qk_eps"] = case["qk_eps"], case["eps"]
    assert not torch.allclose(
        _reference(swapped), _reference(case), atol=0.01, rtol=0.01
    )


@requires_blackwell
def test_blackwell_cuda_graph_capture():
    case = _make_cuda_case(128, 8, "production_segments")
    expected = _reference(case)
    # Warm-up populates the identity-positions cache before capture.
    minimax_h3_bf16_pre_attention(**case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = minimax_h3_bf16_pre_attention(**case)
    case["out"].zero_()
    assert not torch.count_nonzero(case["out"])
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)


@pytest.mark.skipif(
    not (_HAS_BLACKWELL_RUNTIME and _RUN_FULL),
    reason="set FLASHINFER_RUN_FULL_MINIMAX_H3_TESTS=1 to run the 44-shape suite",
)
@pytest.mark.parametrize("m,p,profile", FULL_CORRECTNESS_SHAPES)
def test_blackwell_full_correctness(m, p, profile):
    _run_correctness_shape(m, p, profile)


@requires_blackwell
def test_blackwell_explicit_workspace_matches_and_is_overwritten():
    case = _make_cuda_case(129, 8, "production_segments")
    expected = _reference(case)
    workspace = torch.full(
        (129, HIDDEN), float("nan"), dtype=torch.bfloat16, device=_CUDA_DEVICE
    )
    actual = minimax_h3_bf16_pre_attention(**case, workspace=workspace)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    assert torch.isfinite(workspace.float()).all()


def test_workspace_contract_rejects_wrong_shape_and_dtype():
    case = _make_meta_case(m=129, p=8)
    with pytest.raises(ValueError, match="workspace shape"):
        minimax_h3_bf16_pre_attention(
            **case,
            workspace=torch.empty((128, HIDDEN), dtype=torch.bfloat16, device="meta"),
        )
    with pytest.raises(ValueError, match="workspace dtype"):
        minimax_h3_bf16_pre_attention(
            **case,
            workspace=torch.empty((129, HIDDEN), dtype=torch.float16, device="meta"),
        )
