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

"""MiniMax-H3 MXFP8 pre-attention: prepared API, zero-input smoke and numerical reference.

The numerical tests emulate the norm stage in PyTorch at its own rounding points
(BF16 RMSNorm and AdaLN, UE8M0 block-32 scales rounded up to the next power of
two, E4M3 round-to-nearest) over its caller-owned intermediates, and the fused
GEMM's Q/K RMSNorm, RoPE and MXFP8 pack over the BF16 projection FlashInfer's
CUTLASS MXFP8 GEMM computes from those intermediates: the fused GEMM accumulates
in that GEMM's K order, so the projection its epilogue reads is that GEMM's
output.  V passes the epilogue unchanged, so its packed bytes and scale codes
are compared byte for byte with the MXFP8 quantization of that projection: a
change of the fused GEMM's K order fails the test.  The token counts are
representative, tail (one row around a production center) and smoke counts.
"""

from __future__ import annotations

import math
import sys
from types import ModuleType

import pytest
import torch

from flashinfer.cake_minimax_h3 import MiniMaxH3Mxfp8PreAttention

_HIDDEN = 5376
_HEADS = 56
_KINDS = 3
_HEAD_DIM = 128
_QKV_WIDTH = _HEADS * _KINDS * _HEAD_DIM
_ROPE_WIDTH = 96
_ADALN_ROWS = 9
_EPS = 1.0e-5
_SCALE_BLOCK = 32
_E4M3_MAX = 448.0

_RUN_TENSOR_NAMES = (
    "x",
    "x_norm_weight",
    "adaln_scale",
    "adaln_shift",
    "adaln_index",
    "qkv_weight_q",
    "qkv_weight_sf",
    "q_norm_weight",
    "k_norm_weight",
    "rope_cos_sin",
    "out_q",
    "out_sf",
)


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def test_prepared_api_preserves_caller_owned_outputs(monkeypatch) -> None:
    values = {name: object() for name in _RUN_TENSOR_NAMES}
    output = (values["out_q"], values["out_sf"])
    calls = []

    class _Prepared:
        def __call__(self):
            calls.append("run")
            return output

    generated = ModuleType("flashinfer.diffusion_ops.cake_minimax_h3_mxfp8")

    def _prepare(**kwargs):
        calls.append(kwargs)
        return _Prepared()

    generated.prepare_minimax_h3_mxfp8_pre_attention = _prepare
    monkeypatch.setitem(sys.modules, generated.__name__, generated)
    operation = MiniMaxH3Mxfp8PreAttention(
        **values,
        activation_q=object(),
        activation_sf=object(),
        qkv_bf16=object(),
        gemm_workspace=object(),
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
    [((10, 0), "sm_100a"), ((10, 3), "sm_103a")],
)
def test_exact_architecture_router(monkeypatch, capability, expected) -> None:
    host = pytest.importorskip("flashinfer.diffusion_ops.cake_minimax_h3_mxfp8")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    assert host.minimax_h3_mxfp8_target(torch.device("cuda")) == expected


def test_architecture_router_rejects_cross_routing(monkeypatch) -> None:
    host = pytest.importorskip("flashinfer.diffusion_ops.cake_minimax_h3_mxfp8")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    with pytest.raises(RuntimeError, match="exact compute capability 10.0 or 10.3"):
        host.minimax_h3_mxfp8_target(torch.device("cuda"))


@pytest.mark.parametrize(
    ("M", "norm_ctas", "gemm_ctas"),
    [
        (1, 1, 2 * 84),  # one row tile, padded to a CTA pair
        (128, 128, 2 * 84),
        (129, 129, 2 * 84),  # two row tiles: one CTA pair
        (257, 257, 4 * 84),  # three row tiles, padded to two CTA pairs
        (32640, 32640, 256 * 84),
    ],
)
def test_launch_grid_follows_the_token_count(M, norm_ctas, gemm_ctas) -> None:
    host = pytest.importorskip("flashinfer.diffusion_ops.cake_minimax_h3_mxfp8")
    assert host.minimax_h3_mxfp8_launch_grid("norm", M) == (norm_ctas, 1, 1)
    assert host.minimax_h3_mxfp8_launch_grid("gemm", M) == (gemm_ctas, 1, 1)
    with pytest.raises(ValueError, match="unknown stage"):
        host.minimax_h3_mxfp8_launch_grid("post", M)


def _zero_smoke_tensors(adaln_index: int, M: int = 1, P: int = 8):
    """The run tensors and the intermediates of a smoke preparation."""
    if not _supported():
        pytest.skip("requires SM100a or SM103a")
    from flashinfer.gemm import gemm_base

    device = torch.device("cuda")
    rows_per_destination = M * (_HEADS // P) * _KINDS
    out_sf_stride = _round_up(rows_per_destination, 128) * (_HEAD_DIM // _SCALE_BLOCK)
    activation_sf_len = _round_up(M, 128) * (_HIDDEN // _SCALE_BLOCK)
    values = {
        "x": torch.zeros((M, _HIDDEN), dtype=torch.bfloat16, device=device),
        "x_norm_weight": torch.ones((_HIDDEN,), dtype=torch.bfloat16, device=device),
        "adaln_scale": torch.zeros(
            (_ADALN_ROWS, _HIDDEN), dtype=torch.bfloat16, device=device
        ),
        "adaln_shift": torch.zeros(
            (_ADALN_ROWS, _HIDDEN), dtype=torch.bfloat16, device=device
        ),
        "adaln_index": torch.full((M,), adaln_index, dtype=torch.int32, device=device),
        "qkv_weight_q": torch.zeros(
            (_QKV_WIDTH, _HIDDEN), dtype=torch.float8_e4m3fn, device=device
        ),
        "qkv_weight_sf": torch.zeros(
            (_QKV_WIDTH * (_HIDDEN // _SCALE_BLOCK),), dtype=torch.uint8, device=device
        ),
        "q_norm_weight": torch.ones((_HEAD_DIM,), dtype=torch.bfloat16, device=device),
        "k_norm_weight": torch.ones((_HEAD_DIM,), dtype=torch.bfloat16, device=device),
        "rope_cos_sin": torch.zeros(
            (M, _ROPE_WIDTH), dtype=torch.bfloat16, device=device
        ),
        "out_q": torch.ones(
            (P, M, _HEADS // P, _KINDS, _HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        "out_sf": torch.full((P, out_sf_stride), 255, dtype=torch.uint8, device=device),
    }
    intermediates = {
        "activation_q": torch.empty(
            (M, _HIDDEN), dtype=torch.float8_e4m3fn, device=device
        ),
        "activation_sf": torch.empty(
            (activation_sf_len,), dtype=torch.uint8, device=device
        ),
        "qkv_bf16": torch.empty((M, _QKV_WIDTH), dtype=torch.bfloat16, device=device),
        "gemm_workspace": torch.empty(
            (int(gemm_base.DEFAULT_WORKSPACE_SIZE),),
            dtype=torch.uint8,
            device=device,
        ),
    }
    return values, intermediates


def _prepare_zero_smoke(adaln_index: int, M: int = 1, P: int = 8):
    values, intermediates = _zero_smoke_tensors(adaln_index, M, P)
    operation = MiniMaxH3Mxfp8PreAttention(**values, **intermediates, P=P)
    return operation, values


def _offset_copy(tensor: torch.Tensor, offset: int) -> torch.Tensor:
    """A contiguous copy of ``tensor`` at ``offset`` bytes past a 256-byte boundary."""
    size = tensor.numel() * tensor.element_size()
    storage = torch.zeros((size + 256,), dtype=torch.uint8, device=tensor.device)
    view = storage[offset : offset + size].view(tensor.dtype).view(tensor.shape)
    view.copy_(tensor)
    assert view.is_contiguous() and view.data_ptr() % 256 == offset
    return view


@pytest.mark.parametrize(
    ("name", "offset"),
    [
        ("x", 8),
        ("q_norm_weight", 8),
        ("k_norm_weight", 8),
        ("rope_cos_sin", 8),
        ("qkv_weight_sf", 8),
        ("activation_q", 8),
        ("activation_sf", 4),
        ("out_q", 4),
        ("out_sf", 1),
        ("out_sf", 2),
        ("debug_q_bf16", 8),
    ],
)
def test_misaligned_tensors_are_rejected_at_preparation(name, offset) -> None:
    values, intermediates = _zero_smoke_tensors(-1)
    tensors = {**values, **intermediates}
    if name.startswith("debug_"):
        tensors["debug_q_bf16"] = torch.zeros(
            (1, _HEADS, _HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )
        tensors["debug_k_bf16"] = torch.zeros_like(tensors["debug_q_bf16"])
    tensors[name] = _offset_copy(tensors[name], offset)
    with pytest.raises(ValueError, match=f"^{name} must have a"):
        MiniMaxH3Mxfp8PreAttention(**tensors, P=8)


def test_word_aligned_out_sf_matches_an_aligned_one() -> None:
    """``out_sf`` needs 4-byte alignment only: one 32-bit store per output row."""
    results = []
    for offset in (0, 4):
        values, intermediates = _zero_smoke_tensors(0)
        values["adaln_shift"][0].fill_(1)
        values["qkv_weight_q"].fill_(1)
        values["qkv_weight_sf"].fill_(127)
        values["out_sf"] = _offset_copy(values["out_sf"], offset)
        operation = MiniMaxH3Mxfp8PreAttention(**values, **intermediates, P=8)
        actual_q, actual_sf = operation.run(**values)
        torch.cuda.synchronize()
        assert actual_sf is values["out_sf"]
        results.append((actual_q.view(torch.uint8).clone(), actual_sf.clone()))
    assert torch.count_nonzero(results[0][1]).item() > 0
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("invalid_index", [-1, 9, -(2**31), 2**31 - 1])
def test_invalid_adaln_row_writes_zero_to_caller_outputs(invalid_index) -> None:
    operation, values = _prepare_zero_smoke(invalid_index)
    actual_q, actual_sf = operation.run(**values)
    torch.cuda.synchronize()
    assert actual_q is values["out_q"]
    assert actual_sf is values["out_sf"]
    assert torch.count_nonzero(actual_q.view(torch.uint8)).item() == 0
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
    assert torch.count_nonzero(captured_q.view(torch.uint8)).item() == 0
    assert torch.count_nonzero(captured_sf).item() == 0


def test_prepared_instances_run_on_independent_streams() -> None:
    first, first_values = _prepare_zero_smoke(-1)
    second, second_values = _prepare_zero_smoke(0)
    second_values["adaln_shift"][0].fill_(1)
    second_values["qkv_weight_q"].fill_(1)
    second_values["qkv_weight_sf"].fill_(127)
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
    assert torch.count_nonzero(first_q.view(torch.uint8)).item() == 0
    assert torch.count_nonzero(first_sf).item() == 0
    assert torch.count_nonzero(second_q.view(torch.uint8)).item() > 0
    assert torch.count_nonzero(second_sf).item() > 0


# --------------------------------------------------------------------------
# Numerical reference for both generated stages
# --------------------------------------------------------------------------


def _sf_128x4_offsets(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """Byte offsets of logical ``[rows, cols]`` block scales in the swizzled 128x4 layout."""
    assert rows % 128 == 0 and cols % 4 == 0
    r = torch.arange(rows, device=device)[:, None]
    c = torch.arange(cols, device=device)[None, :]
    return (
        (r // 128) * (128 * cols)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (c % 4)
    )


def _unswizzle_sf(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Logical ``[round_up(rows, 128), cols]`` UE8M0 codes of one swizzled scale tile."""
    padded_rows = _round_up(rows, 128)
    flat = sf.reshape(-1)
    assert flat.numel() == padded_rows * cols
    return flat[_sf_128x4_offsets(padded_rows, cols, sf.device)]


def _ue8m0_scale(codes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes.float() - 127.0)


def _dequantize_blocks(q: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """``q`` E4M3 ``[..., K]`` with UE8M0 ``codes`` ``[..., K / 32]`` to float32."""
    scale = _ue8m0_scale(codes).repeat_interleave(_SCALE_BLOCK, dim=-1)
    return q.float() * scale


def _assert_tight_block_scales(q: torch.Tensor, codes: torch.Tensor) -> None:
    """Every non-zero block's scale is the smallest power of two keeping |q| <= 448.

    This pins the UE8M0 rounding direction (round up) using the kernel's own
    bytes: the largest E4M3 magnitude of a block scaled this way lies in
    [224, 448]; a block whose code is zero must be all zeros.
    """
    block_max = q.float().abs().reshape(*codes.shape, _SCALE_BLOCK).amax(dim=-1)
    nonzero = codes > 0
    assert torch.all(block_max[nonzero] >= _E4M3_MAX / 2)
    assert torch.all(block_max[nonzero] <= _E4M3_MAX)
    assert torch.all(block_max[~nonzero] == 0)


def _mxfp8_quantize_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 bytes and UE8M0 codes of FP32 rows ``[..., K]``, as the epilogue packs.

    The block scale is ``amax * RN(1 / 448)`` (the device's ``rcp.approx`` of
    448), its UE8M0 code the scale's biased exponent rounded up to a power of
    two (0 for an all-zero block) and each element code the E4M3
    round-to-nearest of ``value * 2^(127 - code)``.
    """
    blocks = values.reshape(*values.shape[:-1], -1, _SCALE_BLOCK)
    rcp = torch.tensor(1.0 / _E4M3_MAX, dtype=torch.float32, device=values.device)
    scale = blocks.abs().amax(dim=-1) * rcp
    bits = scale.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    round_up = (mantissa != 0) & ((exponent != 0) | (mantissa > 0x400000))
    codes = torch.where(
        scale > 0, (exponent + round_up.int()).clamp(max=254), torch.zeros_like(bits)
    )
    inverse = torch.where(
        (codes > 0) & (codes < 254),
        torch.exp2(127.0 - codes.float()),
        torch.zeros_like(scale),
    )
    quantized = (blocks * inverse[..., None]).clamp(-_E4M3_MAX, _E4M3_MAX)
    quantized = quantized.reshape(values.shape).to(torch.float8_e4m3fn)
    return quantized.view(torch.uint8), codes.to(torch.uint8)


def _make_inputs(M: int, P: int, seed: int, device: torch.device) -> dict:
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (
            torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            * scale
        ).to(torch.bfloat16)

    adaln_index = torch.randint(
        0, _ADALN_ROWS, (M,), generator=generator, device=device, dtype=torch.int32
    )
    if M >= 3:
        # Malformed AdaLN rows must quantize to all-zero bytes.
        adaln_index[1] = -1
        adaln_index[2] = _ADALN_ROWS
    angles = (
        torch.rand((M, _ROPE_WIDTH // 2), generator=generator, device=device)
        * 2.0
        * math.pi
    )
    weight = randn(_QKV_WIDTH, _HIDDEN, scale=0.02)
    qkv_weight_q, qkv_weight_sf = mxfp8_quantize(weight, is_sf_swizzled_layout=True)
    qkv_weight_sf = qkv_weight_sf.reshape(-1).contiguous()
    assert qkv_weight_sf.numel() == _QKV_WIDTH * (_HIDDEN // _SCALE_BLOCK)
    return {
        "x": randn(M, _HIDDEN),
        "x_norm_weight": (1.0 + randn(_HIDDEN, scale=0.1).float()).to(torch.bfloat16),
        "adaln_scale": randn(_ADALN_ROWS, _HIDDEN, scale=0.1),
        "adaln_shift": randn(_ADALN_ROWS, _HIDDEN, scale=0.1),
        "adaln_index": adaln_index,
        "qkv_weight_q": qkv_weight_q.contiguous(),
        "qkv_weight_sf": qkv_weight_sf,
        "q_norm_weight": (1.0 + randn(_HEAD_DIM, scale=0.1).float()).to(torch.bfloat16),
        "k_norm_weight": (1.0 + randn(_HEAD_DIM, scale=0.1).float()).to(torch.bfloat16),
        "rope_cos_sin": torch.cat((angles.cos(), angles.sin()), dim=-1).to(
            torch.bfloat16
        ),
    }


def _reference_activation(inputs: dict) -> torch.Tensor:
    """Stage-1 BF16 AdaLN output ``bf16(bf16(x * rstd * w) * bf16(1 + scale) + shift)``."""
    x = inputs["x"].float()
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + _EPS)
    normalized = (x * rstd * inputs["x_norm_weight"].float()).to(torch.bfloat16).float()
    index = inputs["adaln_index"].long()
    valid = (index >= 0) & (index < _ADALN_ROWS)
    safe = index.clamp(0, _ADALN_ROWS - 1)
    scale = (inputs["adaln_scale"].float()[safe] + 1.0).to(torch.bfloat16).float()
    shift = inputs["adaln_shift"].float()[safe]
    activation = (normalized * scale + shift).to(torch.bfloat16).float()
    activation[~valid] = 0.0
    return activation


def _bf16_ulp(magnitude: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(magnitude.clamp_min(2.0**-126))) - 7.0)


def _assert_within_round_points(
    actual: torch.Tensor,
    expected: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    what: str,
) -> None:
    """``|actual - expected| <= atol + rtol * |expected| + 2 * bf16_ulp(magnitude)``.

    The kernel rounds the normalized Q/K values to BF16 before the rotation
    (with an approximate ``rsqrt``), so each rotated output may differ from the
    reference by one BF16 ulp of each of its two inputs; ``magnitude`` carries
    ``|n * cos| + |partner * sin|`` for rotated elements and ``|expected|``
    elsewhere.
    """
    difference = (actual.float() - expected.float()).abs()
    # Derivation of the extra term: a rotated output is n * cos - partner * sin
    # where the kernel has already rounded n and partner to BF16.  Each rounding
    # moves its operand by at most half a BF16 ulp, so |n_k * cos - n_r * cos| +
    # |p_k * sin - p_r * sin| <= ulp(|n cos|) / 2 + ulp(|partner sin|) / 2
    # <= ulp(|n cos| + |partner sin|); the output BF16 rounding adds at most one
    # more ulp of the same magnitude.  Hence 2 * bf16_ulp(magnitude), with atol
    # and rtol covering the approximate rsqrt and the dequantized FP8 error.
    bound = atol + rtol * expected.float().abs() + 2.0 * _bf16_ulp(magnitude)
    excess = difference - bound
    violations = int((excess > 0).sum().item())
    worst = float(excess.max().item())
    assert violations == 0, (
        f"{what}: {violations} of {difference.numel()} elements exceed the round-point "
        f"bound (worst excess {worst:.6g})"
    )


def _reference_qkv(qkv_bf16: torch.Tensor, inputs: dict) -> tuple[torch.Tensor, ...]:
    """Stage-2 BF16 Q (normed + RoPE), K (normed + RoPE), V and the per-element
    input magnitudes that bound their BF16 rounding (see ``_assert_within_round_points``)."""
    M = qkv_bf16.shape[0]
    grouped = qkv_bf16.view(M, _HEADS, _KINDS, _HEAD_DIM).float()

    def normed(kind: int, weight: torch.Tensor) -> torch.Tensor:
        head = grouped[:, :, kind]
        rstd = torch.rsqrt(head.square().mean(dim=-1, keepdim=True) + _EPS)
        return (head * rstd * weight.float()).to(torch.bfloat16).float()

    half = _ROPE_WIDTH // 2
    cos = inputs["rope_cos_sin"][:, :half].float()[:, None, :]
    sin = inputs["rope_cos_sin"][:, half:].float()[:, None, :]

    def rope(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = values[..., :half]
        second = values[..., half:_ROPE_WIDTH]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).to(torch.bfloat16)
        tail = values[..., _ROPE_WIDTH:].to(torch.bfloat16)
        pair = (first * cos).abs() + (second * sin).abs()
        magnitude = torch.cat(
            (pair, (second * cos).abs() + (first * sin).abs(), tail.float().abs()),
            dim=-1,
        )
        return torch.cat((rotated, tail), dim=-1), magnitude

    q, q_magnitude = rope(normed(0, inputs["q_norm_weight"]))
    k, k_magnitude = rope(normed(1, inputs["k_norm_weight"]))
    v = grouped[:, :, 2].to(torch.bfloat16)
    return q, k, v, q_magnitude, k_magnitude


def _reference_projection(
    activation_q: torch.Tensor, activation_sf: torch.Tensor, inputs: dict
) -> torch.Tensor:
    """The BF16 QKV projection FlashInfer's CUTLASS MXFP8 GEMM computes from the stage-1 bytes."""
    from flashinfer.gemm import mm_mxfp8

    return mm_mxfp8(
        activation_q,
        inputs["qkv_weight_q"].t(),
        activation_sf,
        inputs["qkv_weight_sf"],
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )


def _assert_gemm_matches_dequantized_operands(
    qkv_bf16: torch.Tensor,
    activation_q: torch.Tensor,
    activation_codes: torch.Tensor,
    inputs: dict,
) -> None:
    """The reference GEMM consumed the stage-1 bytes in the layout stage 1 wrote them."""
    weight_codes = _unswizzle_sf(
        inputs["qkv_weight_sf"], _QKV_WIDTH, _HIDDEN // _SCALE_BLOCK
    )
    weight = _dequantize_blocks(inputs["qkv_weight_q"], weight_codes)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, qkv_bf16.shape[0], 2048):
            stop = min(start + 2048, qkv_bf16.shape[0])
            activation = _dequantize_blocks(
                activation_q[start:stop], activation_codes[start:stop]
            )
            expected = activation @ weight.t()
            torch.testing.assert_close(
                qkv_bf16[start:stop].float(), expected, atol=1e-2, rtol=1e-2
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize(
    ("M", "P"),
    [
        (1, 8),  # smoke
        (127, 8),  # smoke, one row short of a scale tile
        (129, 8),  # smoke, one row past a scale tile
        (4823, 8),  # tail, one row below the P=8 production center
        (4825, 8),  # tail, one row above the P=8 production center
        (9648, 4),  # P=4 production center
        (16736, 2),  # P=2 production center
        (33472, 1),  # P=1 production center
    ],
)
def test_numerical_reference_both_stages(M: int, P: int) -> None:
    if not _supported():
        pytest.skip("requires SM100a or SM103a")
    from flashinfer.gemm import gemm_base

    device = torch.device("cuda")
    inputs = _make_inputs(M, P, seed=M * 16 + P, device=device)
    heads_per_destination = _HEADS // P
    rows_per_destination = M * heads_per_destination * _KINDS
    out_sf_stride = _round_up(rows_per_destination, 128) * (_HEAD_DIM // _SCALE_BLOCK)
    activation_sf_len = _round_up(M, 128) * (_HIDDEN // _SCALE_BLOCK)
    out_q = torch.empty(
        (P, M, heads_per_destination, _KINDS, _HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    out_sf = torch.empty((P, out_sf_stride), dtype=torch.uint8, device=device)
    activation_q = torch.empty((M, _HIDDEN), dtype=torch.float8_e4m3fn, device=device)
    activation_sf = torch.empty((activation_sf_len,), dtype=torch.uint8, device=device)
    qkv_bf16 = torch.empty((M, _QKV_WIDTH), dtype=torch.bfloat16, device=device)
    debug_q = torch.empty((M, _HEADS, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    debug_k = torch.empty((M, _HEADS, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    operation = MiniMaxH3Mxfp8PreAttention(
        **inputs,
        out_q=out_q,
        out_sf=out_sf,
        activation_q=activation_q,
        activation_sf=activation_sf,
        qkv_bf16=qkv_bf16,
        gemm_workspace=torch.empty(
            (int(gemm_base.DEFAULT_WORKSPACE_SIZE),), dtype=torch.uint8, device=device
        ),
        P=P,
        debug_q_bf16=debug_q,
        debug_k_bf16=debug_k,
    )
    actual_q, actual_sf = operation.run(
        **inputs,
        out_q=out_q,
        out_sf=out_sf,
        debug_q_bf16=debug_q,
        debug_k_bf16=debug_k,
    )
    torch.cuda.synchronize()
    assert actual_q is out_q and actual_sf is out_sf

    # Stage 1: RMSNorm + indexed AdaLN + MXFP8 quantize (caller-owned intermediates).
    activation_codes = _unswizzle_sf(activation_sf, M, _HIDDEN // _SCALE_BLOCK)
    assert torch.all(activation_codes[M:] == 0), "scale padding rows must stay zero"
    activation_codes = activation_codes[:M]
    _assert_tight_block_scales(activation_q, activation_codes)
    expected_activation = _reference_activation(inputs)
    torch.testing.assert_close(
        _dequantize_blocks(activation_q, activation_codes),
        expected_activation,
        atol=1e-3,
        rtol=2.0**-4 + 2.0**-7,  # E4M3 round-to-nearest plus one BF16 ulp
    )
    invalid = (inputs["adaln_index"] < 0) | (inputs["adaln_index"] >= _ADALN_ROWS)
    assert torch.all(activation_q[invalid].view(torch.uint8) == 0)
    assert torch.all(activation_codes[invalid] == 0)

    # GEMM: the reference BF16 projection equals the dequantized stage-1 operand
    # times the dequantized prepacked weight (proves the swizzled scale layout
    # end to end).  The fused GEMM keeps its projection on chip; the epilogue
    # outputs below are checked against this projection.
    projection = _reference_projection(activation_q, activation_sf, inputs)
    _assert_gemm_matches_dequantized_operands(
        projection, activation_q, activation_codes, inputs
    )

    # Stage 2: per-head Q/K RMSNorm + split-half NeoX RoPE, checked on the BF16
    # debug copies, then destination-major MXFP8 packing of Q, K and V.
    expected_q, expected_k, expected_v, q_magnitude, k_magnitude = _reference_qkv(
        projection, inputs
    )
    _assert_within_round_points(
        debug_q, expected_q, q_magnitude, atol=1e-2, rtol=1e-2, what="debug_q_bf16"
    )
    _assert_within_round_points(
        debug_k, expected_k, k_magnitude, atol=1e-2, rtol=1e-2, what="debug_k_bf16"
    )
    expected = torch.stack((expected_q, expected_k, expected_v), dim=2).float()
    expected = expected.view(M, P, heads_per_destination, _KINDS, _HEAD_DIM)
    expected = expected.permute(1, 0, 2, 3, 4)
    magnitude = torch.stack((q_magnitude, k_magnitude, expected_v.float().abs()), dim=2)
    magnitude = magnitude.view(M, P, heads_per_destination, _KINDS, _HEAD_DIM)
    magnitude = magnitude.permute(1, 0, 2, 3, 4)
    # V passes the epilogue unchanged: its bytes and codes are exact.
    expected_v_q, expected_v_codes = _mxfp8_quantize_rows(expected_v.float())
    expected_v_q = expected_v_q.view(M, P, heads_per_destination, _HEAD_DIM)
    expected_v_q = expected_v_q.permute(1, 0, 2, 3)
    expected_v_codes = expected_v_codes.view(
        M, P, heads_per_destination, _HEAD_DIM // _SCALE_BLOCK
    ).permute(1, 0, 2, 3)
    for destination in range(P):
        codes = _unswizzle_sf(
            out_sf[destination], rows_per_destination, _HEAD_DIM // _SCALE_BLOCK
        )
        assert torch.all(codes[rows_per_destination:] == 0)
        codes = codes[:rows_per_destination].view(
            M, heads_per_destination, _KINDS, _HEAD_DIM // _SCALE_BLOCK
        )
        packed = out_q[destination]
        _assert_tight_block_scales(packed, codes)
        differing = int(
            (packed[:, :, 2].view(torch.uint8) != expected_v_q[destination])
            .sum()
            .item()
        )
        assert differing == 0, (
            f"destination {destination}: {differing} V bytes differ from the MXFP8 "
            "quantization of the CUTLASS projection"
        )
        assert torch.equal(codes[:, :, 2], expected_v_codes[destination])
        _assert_within_round_points(
            _dequantize_blocks(packed, codes),
            expected[destination],
            magnitude[destination],
            atol=1e-2,
            rtol=2.0**-4 + 2.0**-7,
            what=f"out_q/out_sf destination {destination}",
        )
