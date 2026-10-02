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

from __future__ import annotations

import sys
from typing import Callable, Optional
from types import ModuleType

import pytest
import torch

from flashinfer.cake_minimax_h3 import MiniMaxH3Nvfp4PreAttention


_RUN_TENSOR_NAMES = (
    "x",
    "x_norm_weight",
    "adaln_scale",
    "adaln_shift",
    "adaln_index",
    "x_global_scale",
    "qkv_weight_q",
    "qkv_weight_sf",
    "w_global_scale",
    "q_norm_weight",
    "k_norm_weight",
    "rope_cos_sin",
    "out_global_scale",
    "out_q",
    "out_sf",
)


def test_prepared_api_preserves_caller_owned_outputs(monkeypatch) -> None:
    values = {name: object() for name in _RUN_TENSOR_NAMES}
    output = (values["out_q"], values["out_sf"])
    calls = []

    class _Prepared:
        def __call__(self):
            calls.append("run")
            return output

    generated = ModuleType("flashinfer.diffusion_ops.cake_minimax_h3_nvfp4")

    def _prepare(**kwargs):
        calls.append(kwargs)
        return _Prepared()

    generated.prepare_minimax_h3_nvfp4_pre_attention = _prepare
    monkeypatch.setitem(sys.modules, generated.__name__, generated)

    operation = MiniMaxH3Nvfp4PreAttention(
        **values,
        activation_q=object(),
        activation_sf=object(),
        P=8,
    )
    actual = operation.run(**values)

    assert calls[-1] == "run"
    assert actual[0] is values["out_q"]
    assert actual[1] is values["out_sf"]

    with pytest.raises(ValueError, match="x"):
        operation.run(**{**values, "x": object()})


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((10, 0), "sm100a"), ((10, 3), "sm103a")],
)
def test_exact_architecture_router(monkeypatch, capability, expected) -> None:
    router = pytest.importorskip("flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    assert router.minimax_h3_nvfp4_target(torch.device("cuda")) == expected


def test_architecture_router_rejects_cross_routing(monkeypatch) -> None:
    router = pytest.importorskip("flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    with pytest.raises(RuntimeError, match="exact compute capability 10.0 or 10.3"):
        router.minimax_h3_nvfp4_target(torch.device("cuda"))


def _prepare_zero_smoke(
    adaln_index: int,
    M: int = 1,
    P: int = 8,
    configure: Optional[Callable[[dict], None]] = None,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100a or SM103a")

    router = pytest.importorskip("flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention")

    hidden, qkv_width, head_dim, fp4_block = 5376, 21504, 128, 16
    device = torch.device("cuda")
    rows_per_destination = M * (56 // P) * 3
    out_sf_stride = ((rows_per_destination + 127) // 128 * 128) * (
        head_dim // fp4_block
    )
    activation_sf_len = ((M + 127) // 128 * 128) * (hidden // fp4_block)
    values = {
        "x": torch.zeros((M, hidden), dtype=torch.bfloat16, device=device),
        "x_norm_weight": torch.ones((hidden,), dtype=torch.bfloat16, device=device),
        "adaln_scale": torch.zeros((9, hidden), dtype=torch.bfloat16, device=device),
        "adaln_shift": torch.zeros((9, hidden), dtype=torch.bfloat16, device=device),
        "adaln_index": torch.full((M,), adaln_index, dtype=torch.int32, device=device),
        "x_global_scale": torch.ones((1,), dtype=torch.float32, device=device),
        "qkv_weight_q": torch.zeros(
            (qkv_width, hidden // 2), dtype=torch.uint8, device=device
        ),
        "qkv_weight_sf": torch.zeros(
            (qkv_width * (hidden // fp4_block),), dtype=torch.uint8, device=device
        ),
        "w_global_scale": torch.ones((1,), dtype=torch.float32, device=device),
        "q_norm_weight": torch.ones((head_dim,), dtype=torch.bfloat16, device=device),
        "k_norm_weight": torch.ones((head_dim,), dtype=torch.bfloat16, device=device),
        "rope_cos_sin": torch.zeros((M, 96), dtype=torch.bfloat16, device=device),
        "out_global_scale": torch.ones((1,), dtype=torch.float32, device=device),
        "out_q": torch.ones(
            (P, M, 56 // P, 3, head_dim // 2),
            dtype=torch.uint8,
            device=device,
        ),
        "out_sf": torch.full((P, out_sf_stride), 255, dtype=torch.uint8, device=device),
    }
    if configure is not None:
        # The fused route derives alpha and the CTA-pair-ordered weight scales at
        # preparation, so operand values a test depends on are set before the
        # operation is prepared (the E2M1 weight bytes are still read in place).
        configure(values)
    route = router.minimax_h3_nvfp4_route_record(device, P)
    assert set(route["stages"]) == {
        "norm_adaln_nvfp4_quantize",
        "qkv_nvfp4_gemm_fused_pack",
    }
    operation = MiniMaxH3Nvfp4PreAttention(
        **values,
        activation_q=torch.empty((M, hidden // 2), dtype=torch.uint8, device=device),
        activation_sf=torch.empty(
            (activation_sf_len,), dtype=torch.uint8, device=device
        ),
        P=P,
    )
    return operation, values


@pytest.mark.parametrize("invalid_index", [-1, 9, -(2**31), 2**31 - 1])
def test_invalid_adaln_row_writes_zero_to_caller_outputs(invalid_index) -> None:
    operation, values = _prepare_zero_smoke(invalid_index)
    actual_q, actual_sf = operation.run(**values)
    torch.cuda.synchronize()

    assert actual_q is values["out_q"]
    assert actual_sf is values["out_sf"]
    assert torch.count_nonzero(actual_q).item() == 0
    assert torch.count_nonzero(actual_sf).item() == 0


def test_prepared_api_cuda_graph_replay() -> None:
    operation, values = _prepare_zero_smoke(-1)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        operation.run(**values)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_q, captured_sf = operation.run(**values)
    values["out_q"].fill_(1)
    values["out_sf"].fill_(255)
    graph.replay()
    torch.cuda.synchronize()

    assert captured_q is values["out_q"]
    assert captured_sf is values["out_sf"]
    assert torch.count_nonzero(captured_q).item() == 0
    assert torch.count_nonzero(captured_sf).item() == 0


def test_prepared_instances_run_on_independent_streams() -> None:
    first, first_values = _prepare_zero_smoke(-1)

    def configure(values: dict) -> None:
        values["adaln_shift"][0].fill_(1)
        values["qkv_weight_q"].fill_(0x22)
        values["qkv_weight_sf"].fill_(0x38)

    second, second_values = _prepare_zero_smoke(0, configure=configure)
    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    first_stream.wait_stream(current_stream)
    second_stream.wait_stream(current_stream)

    with torch.cuda.stream(first_stream):
        first_q, first_sf = first.run(**first_values)
    with torch.cuda.stream(second_stream):
        second_q, second_sf = second.run(**second_values)
    first_stream.synchronize()
    second_stream.synchronize()

    assert first_q is first_values["out_q"]
    assert first_sf is first_values["out_sf"]
    assert second_q is second_values["out_q"]
    assert second_sf is second_values["out_sf"]
    assert torch.count_nonzero(first_q).item() == 0
    assert torch.count_nonzero(first_sf).item() == 0
    assert torch.count_nonzero(second_q).item() > 0
    assert torch.count_nonzero(second_sf).item() > 0


@pytest.mark.parametrize(
    ("M", "P", "expected_norm", "expected_gemm", "expected_row_tiles"),
    [
        (1, 8, (64, 1, 1), (168, 1, 1), 2),
        (129, 8, (128, 1, 1), (168, 1, 1), 2),
        (4824, 8, (2432, 1, 1), (3192, 1, 1), 38),
        (9648, 4, (4864, 1, 1), (6384, 1, 1), 76),
        (19296, 2, (9664, 1, 1), (12768, 1, 1), 152),
        (38592, 1, (19328, 1, 1), (25368, 1, 1), 302),
        (38591, 1, (19328, 1, 1), (25368, 1, 1), 302),
    ],
)
def test_stage_launch_grids_follow_the_runtime_token_count(
    M, P, expected_norm, expected_gemm, expected_row_tiles
) -> None:
    from flashinfer.diffusion_ops import cake_minimax_h3_nvfp4 as ops

    norm_record = {
        "launch_grid_rule": {
            "kind": "norm_rows",
            "rows_per_cta": 2,
            "row_alignment": 128,
        }
    }
    gemm_record = {
        "launch_grid_rule": {
            "kind": "gemm_cluster_tiles",
            "block_m": 128,
            "cta_group": 2,
            "n_tiles": 84,
        }
    }
    assert ops._stage_launch_grid(norm_record, M=M, P=P) == expected_norm
    assert ops._stage_launch_grid(gemm_record, M=M, P=P) == expected_gemm
    assert (
        ops._gemm_row_tiles(gemm_record["launch_grid_rule"], M=M) == expected_row_tiles
    )
    with pytest.raises(RuntimeError, match="launch grid rule"):
        ops._stage_launch_grid({"launch_grid_rule": {"kind": "other"}}, M=M, P=P)


def test_fused_gemm_weight_scale_repack_orders_cta_pairs() -> None:
    from flashinfer.diffusion_ops import cake_minimax_h3_nvfp4 as ops

    n_blocks, k_sets, tile = 168, 84, 512
    # Tile (row block n, K-set k) tagged by its position; the repack must map
    # it to (pair n // 2, K-set k, half n % 2).
    tags = torch.arange(n_blocks * k_sets, dtype=torch.int32).reshape(n_blocks, k_sets)
    source = (
        tags.unsqueeze(-1).expand(n_blocks, k_sets, tile).to(torch.int32).reshape(-1)
    )
    packed = ops.repack_minimax_h3_qkv_weight_scales_for_fused_gemm(
        (source % 251).to(torch.uint8)
    )
    assert packed.shape == (n_blocks * k_sets * tile,)
    view = packed.reshape(n_blocks // 2, k_sets, 2, tile)
    expected = (tags % 251).to(torch.uint8).reshape(n_blocks // 2, 2, k_sets)
    assert torch.equal(view[..., 0], expected.permute(0, 2, 1))
    assert torch.equal(view[..., tile - 1], expected.permute(0, 2, 1))
    with pytest.raises(ValueError, match="swizzled scale bytes"):
        ops.repack_minimax_h3_qkv_weight_scales_for_fused_gemm(
            torch.zeros((16,), dtype=torch.uint8)
        )


# ---------------------------------------------------------------------------
# Numerical reference
# ---------------------------------------------------------------------------

_HIDDEN, _QKV_WIDTH, _NUM_HEADS, _HEAD_DIM, _FP4_BLOCK, _ROPE_DIM, _ADALN_ROWS = (
    5376,
    21504,
    56,
    128,
    16,
    96,
    9,
)
_E2M1_MAX, _E4M3_MAX, _EPS = 6.0, 448.0, 1.0e-5
_E2M1_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even E2M1 codes (``cvt.rn.satfinite.e2m1x2`` semantics)."""
    magnitude = values.abs()
    codes = torch.zeros_like(magnitude, dtype=torch.uint8)
    codes[(magnitude > 0.25) & (magnitude < 0.75)] = 1
    codes[(magnitude >= 0.75) & (magnitude <= 1.25)] = 2
    codes[(magnitude > 1.25) & (magnitude < 1.75)] = 3
    codes[(magnitude >= 1.75) & (magnitude <= 2.5)] = 4
    codes[(magnitude > 2.5) & (magnitude < 3.5)] = 5
    codes[(magnitude >= 3.5) & (magnitude <= 5.0)] = 6
    codes[magnitude > 5.0] = 7
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def _decode_e2m1(packed: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1_TABLE, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    return table[(codes & 0x7).long()] * torch.where((codes & 0x8) != 0, -1.0, 1.0)


def _quantize_nvfp4(source: torch.Tensor, global_scale: torch.Tensor):
    """NVFP4 block-16 quantization with a static global encode scale.

    Returns packed E2M1 nibble pairs ``uint8 [rows, K // 2]`` and logical E4M3
    scale bytes ``uint8 [rows, K // 16]``: ``sf = e4m3(gs * amax / 6)``, encode
    factor ``1 / (float(sf) / gs)``, zero blocks give zero bytes.
    """
    rows, width = source.shape
    gs = global_scale.float().reshape(())
    blocks = source.float().reshape(rows, width // _FP4_BLOCK, _FP4_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    sf_fp8 = (gs * (amax * (1.0 / _E2M1_MAX))).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
    sf_value = sf_fp8.float()
    encode = torch.where(amax != 0.0, 1.0 / (sf_value * (1.0 / gs)), torch.zeros_like(sf_value))
    codes = _e2m1_codes(blocks * encode)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed.reshape(rows, width // 2), sf_fp8.reshape(rows, width // _FP4_BLOCK).view(torch.uint8)


def _dequantize_nvfp4(packed, logical_scales, global_scale) -> torch.Tensor:
    rows = packed.shape[0]
    scale = logical_scales.contiguous().view(torch.float8_e4m3fn).float()
    values = _decode_e2m1(packed).reshape(rows, -1, _FP4_BLOCK) * scale.reshape(rows, -1, 1)
    return values.reshape(rows, -1) / global_scale.float().reshape(())


def _scale_indices(rows, cols, padded_cols: int):
    """Swizzled 128x4 scale layout used by the NVFP4 scale tensors."""
    return (
        cols % 4
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (rows // 128) * (128 * padded_cols)
    )


def _swizzle_scales(logical: torch.Tensor) -> torch.Tensor:
    rows_count, cols_count = logical.shape
    padded_cols = _round_up(cols_count, 4)
    out = torch.zeros(_round_up(rows_count, 128) * padded_cols, dtype=torch.uint8, device=logical.device)
    rows = torch.arange(rows_count, device=logical.device).unsqueeze(1)
    cols = torch.arange(cols_count, device=logical.device).unsqueeze(0)
    out[_scale_indices(rows, cols, padded_cols).reshape(-1)] = logical.reshape(-1)
    return out


def _unswizzle_scales(physical: torch.Tensor, rows_count: int, cols_count: int) -> torch.Tensor:
    flat = physical.contiguous().reshape(-1)
    rows = torch.arange(rows_count, device=flat.device).unsqueeze(1)
    cols = torch.arange(cols_count, device=flat.device).unsqueeze(0)
    return flat[_scale_indices(rows, cols, _round_up(cols_count, 4))].reshape(rows_count, cols_count)


def _partial_neox_rope(x: torch.Tensor, rope_cos_sin: torch.Tensor) -> torch.Tensor:
    half = _ROPE_DIM // 2
    rotary, tail = x[..., :_ROPE_DIM].float(), x[..., _ROPE_DIM:]
    cos = torch.cat((rope_cos_sin[:, :half],) * 2, dim=-1).float()[:, None, :]
    sin = torch.cat((rope_cos_sin[:, half:],) * 2, dim=-1).float()[:, None, :]
    rotate_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    return torch.cat(((rotary * cos + rotate_half * sin).to(torch.bfloat16), tail), dim=-1)


def _reference(values: dict) -> dict:
    """Independent torch oracle of the two-stage route (BF16 rounding at every contract boundary)."""
    M = values["x"].shape[0]
    P = values["out_q"].shape[0]
    norm = torch.nn.functional.rms_norm(values["x"], (_HIDDEN,), values["x_norm_weight"], eps=_EPS).to(torch.bfloat16)
    index = values["adaln_index"].to(torch.int64)
    valid = (index >= 0) & (index < _ADALN_ROWS)
    safe = index.clamp(0, _ADALN_ROWS - 1)
    scale_plus_one = (values["adaln_scale"].index_select(0, safe) + 1.0).to(torch.bfloat16)
    adaln = torch.addcmul(values["adaln_shift"].index_select(0, safe), norm, scale_plus_one).to(torch.bfloat16)
    adaln = torch.where(valid[:, None], adaln, torch.zeros_like(adaln))

    activation_q, activation_sf = _quantize_nvfp4(adaln, values["x_global_scale"])
    activation = _dequantize_nvfp4(activation_q, activation_sf, values["x_global_scale"]).to(torch.bfloat16)
    weight_sf = _unswizzle_scales(values["qkv_weight_sf"], _QKV_WIDTH, _HIDDEN // _FP4_BLOCK)
    weight = _dequantize_nvfp4(values["qkv_weight_q"], weight_sf, values["w_global_scale"]).to(torch.bfloat16)
    qkv = torch.nn.functional.linear(activation, weight).to(torch.bfloat16)

    grouped = qkv.view(M, _NUM_HEADS, 3, _HEAD_DIM)
    q = torch.nn.functional.rms_norm(grouped[:, :, 0], (_HEAD_DIM,), values["q_norm_weight"], eps=_EPS).to(torch.bfloat16)
    k = torch.nn.functional.rms_norm(grouped[:, :, 1], (_HEAD_DIM,), values["k_norm_weight"], eps=_EPS).to(torch.bfloat16)
    q = _partial_neox_rope(q, values["rope_cos_sin"])
    k = _partial_neox_rope(k, values["rope_cos_sin"])
    fused = torch.stack((q, k, grouped[:, :, 2]), dim=2)
    destination = fused.view(M, P, _NUM_HEADS // P, 3, _HEAD_DIM).permute(1, 0, 2, 3, 4).contiguous()
    out_q, out_sf = [], []
    for shard in destination:
        packed, logical = _quantize_nvfp4(shard.reshape(-1, _HEAD_DIM), values["out_global_scale"])
        out_q.append(packed.reshape(*shard.shape[:-1], _HEAD_DIM // 2))
        out_sf.append(_swizzle_scales(logical))
    return {
        "adaln": adaln,
        "q": q,
        "k": k,
        "v": grouped[:, :, 2],
        "out_q": torch.stack(out_q),
        "out_sf": torch.stack(out_sf),
    }


def _dequantize_destination_major(out_q, out_sf, global_scale) -> torch.Tensor:
    restored = []
    for destination in range(out_q.shape[0]):
        packed = out_q[destination].reshape(-1, _HEAD_DIM // 2)
        logical = _unswizzle_scales(out_sf[destination], packed.shape[0], _HEAD_DIM // _FP4_BLOCK)
        restored.append(_dequantize_nvfp4(packed, logical, global_scale).reshape(*out_q.shape[1:-1], _HEAD_DIM))
    return torch.stack(restored)


def _global_encode_scale(amax: torch.Tensor) -> torch.Tensor:
    scale = torch.div(_E4M3_MAX * _E2M1_MAX, amax.float())
    return torch.where(scale == 0.0, torch.ones_like(scale), scale).reshape(1)


def _rope_cache(M: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(M, dtype=torch.float32, device=device)
    axes = (rows // 4096, (rows // 64).remainder(64), rows.remainder(64))
    width = _ROPE_DIM // 6
    inv_freq = torch.pow(10000.0, -torch.arange(width, dtype=torch.float32, device=device) / width)
    phase = torch.cat([axis[:, None] * inv_freq[None, :] for axis in axes], dim=-1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(torch.bfloat16).contiguous()


def _make_numerical_values(M: int, P: int, device: torch.device) -> dict:
    generator = torch.Generator(device=device).manual_seed(0x5791 + M * 16 + P)

    def uniform(shape, low, high, dtype=torch.bfloat16):
        return (torch.rand(shape, generator=generator, device=device, dtype=torch.float32) * (high - low) + low).to(dtype)

    x = uniform((M, _HIDDEN), -2.0, 2.0)
    adaln_index = torch.div(torch.arange(M, device=device) * _ADALN_ROWS, M, rounding_mode="floor").to(torch.int32)
    if M >= 3:
        adaln_index[1] = -1
        adaln_index[2] = _ADALN_ROWS
    x_norm_weight = uniform((_HIDDEN,), 0.8, 1.2)
    adaln_scale = uniform((_ADALN_ROWS, _HIDDEN), -0.5, 0.5)
    adaln_shift = uniform((_ADALN_ROWS, _HIDDEN), -0.5, 0.5)
    norm = torch.nn.functional.rms_norm(x, (_HIDDEN,), x_norm_weight, eps=_EPS)
    adaln_amax = (norm * (adaln_scale + 1.0).abs().amax(dim=0) + adaln_shift.abs().amax(dim=0)).abs().amax()
    weight = uniform((_QKV_WIDTH, _HIDDEN), -0.05, 0.05)
    w_global_scale = _global_encode_scale(weight.float().abs().amax())
    qkv_weight_q, weight_logical_sf = _quantize_nvfp4(weight, w_global_scale)
    rows_per_destination = M * (_NUM_HEADS // P) * 3
    return {
        "x": x,
        "x_norm_weight": x_norm_weight,
        "adaln_scale": adaln_scale,
        "adaln_shift": adaln_shift,
        "adaln_index": adaln_index,
        "x_global_scale": _global_encode_scale(adaln_amax),
        "qkv_weight_q": qkv_weight_q.contiguous(),
        "qkv_weight_sf": _swizzle_scales(weight_logical_sf).contiguous(),
        "w_global_scale": w_global_scale,
        "q_norm_weight": uniform((_HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((_HEAD_DIM,), 0.9, 1.1),
        "rope_cos_sin": _rope_cache(M, device),
        # Static output scale sized for post-norm Q/K (|v| <= ~4 after RoPE) and
        # the V projection magnitude of this fixture.
        "out_global_scale": _global_encode_scale(torch.tensor(8.0, device=device)),
        "out_q": torch.full((P, M, _NUM_HEADS // P, 3, _HEAD_DIM // 2), 0x7F, dtype=torch.uint8, device=device),
        "out_sf": torch.full(
            (P, _round_up(rows_per_destination, 128) * (_HEAD_DIM // _FP4_BLOCK)), 0xA5, dtype=torch.uint8, device=device
        ),
    }


@pytest.mark.parametrize(("M", "P"), [(1, 8), (129, 8), (2048, 1), (4097, 2)])
def test_numerical_reference(M: int, P: int) -> None:
    """The two stages match an independent torch oracle of the NVFP4 route.

    Intermediates (AdaLN output, post-norm/RoPE Q and K) are checked at BF16
    tolerance; the packed destination is checked after dequantization against
    the oracle's quantization of the kernel's own BF16 intermediates (one E2M1
    step at the block scale), and the Q/K scale blocks must be bit-identical.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100a or SM103a")
    pytest.importorskip("flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention")

    device = torch.device("cuda")
    values = _make_numerical_values(M, P, device)
    debug = {
        "debug_adaln_bf16": torch.full((M, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device),
        "debug_q_bf16": torch.full((M, _NUM_HEADS, _HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=device),
        "debug_k_bf16": torch.full((M, _NUM_HEADS, _HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=device),
    }
    operation = MiniMaxH3Nvfp4PreAttention(
        **values,
        activation_q=torch.empty((M, _HIDDEN // 2), dtype=torch.uint8, device=device),
        activation_sf=torch.empty((_round_up(M, 128) * (_HIDDEN // _FP4_BLOCK),), dtype=torch.uint8, device=device),
        P=P,
        **debug,
    )
    out_q, out_sf = operation.run(**values, **debug)
    torch.cuda.synchronize()
    expected = _reference(values)

    assert out_q is values["out_q"] and out_sf is values["out_sf"]
    # Stage 1: RMSNorm + AdaLN with invalid rows zeroed (BF16 tolerance).
    torch.testing.assert_close(debug["debug_adaln_bf16"], expected["adaln"], atol=1e-2, rtol=1e-2)
    # Fused epilogue intermediates: per-head RMSNorm + partial NeoX RoPE on the
    # NVFP4 GEMM result (BF16 tolerance; the GEMM accumulates in FP32).
    torch.testing.assert_close(debug["debug_q_bf16"], expected["q"], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(debug["debug_k_bf16"], expected["k"], atol=1e-2, rtol=1e-2)

    # Destination pack: the kernel quantizes its own fused BF16 values, so the
    # oracle packs the kernel's debug Q/K plus the oracle V to isolate the pack,
    # and both are compared after dequantization at one E2M1 step of the
    # block's scale (the largest E2M1 spacing is two units of the scale).
    kernel_fused = torch.stack((debug["debug_q_bf16"], debug["debug_k_bf16"], expected["v"]), dim=2)
    kernel_destination = kernel_fused.view(M, P, _NUM_HEADS // P, 3, _HEAD_DIM).permute(1, 0, 2, 3, 4).contiguous()
    gs = values["out_global_scale"]
    expected_q, expected_sf = [], []
    for shard in kernel_destination:
        packed, logical = _quantize_nvfp4(shard.reshape(-1, _HEAD_DIM), gs)
        expected_q.append(packed.reshape(*shard.shape[:-1], _HEAD_DIM // 2))
        expected_sf.append(_swizzle_scales(logical))
    expected_q, expected_sf = torch.stack(expected_q), torch.stack(expected_sf)

    # Scale padding rows (beyond the destination's real rows) are zero.
    rows_per_destination = M * (_NUM_HEADS // P) * 3
    padding = torch.ones_like(out_sf, dtype=torch.bool)
    rows = torch.arange(rows_per_destination, device=device).unsqueeze(1)
    cols = torch.arange(_HEAD_DIM // _FP4_BLOCK, device=device).unsqueeze(0)
    padding[:, _scale_indices(rows, cols, _HEAD_DIM // _FP4_BLOCK).reshape(-1)] = False
    assert torch.count_nonzero(out_sf[padding]).item() == 0

    actual = _dequantize_destination_major(out_q, out_sf, gs)
    oracle = _dequantize_destination_major(expected_q, expected_sf, gs)
    assert torch.isfinite(actual).all()
    scale_cols = _HEAD_DIM // _FP4_BLOCK
    logical_actual = torch.stack([_unswizzle_scales(out_sf[d], rows_per_destination, scale_cols) for d in range(P)])
    logical_expected = torch.stack([_unswizzle_scales(expected_sf[d], rows_per_destination, scale_cols) for d in range(P)])
    block_scale = torch.maximum(
        logical_actual.view(torch.float8_e4m3fn).float(), logical_expected.view(torch.float8_e4m3fn).float()
    ) / gs.float()
    step = (2.0 * block_scale).reshape(P, M, _NUM_HEADS // P, 3, scale_cols, 1).expand(-1, -1, -1, -1, -1, _FP4_BLOCK)
    error = (actual - oracle).abs()
    assert bool((error <= step.reshape(actual.shape) + 1e-6).all()), float(error.max())
    # Most elements agree exactly; the Q/K rows are quantized from identical
    # BF16 inputs and only accumulation-order effects on V may differ.
    assert float((out_q != expected_q).float().mean()) <= 0.01
    assert float((logical_actual != logical_expected).float().mean()) <= 0.01
