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


def test_prepared_api_e4m3_returns_no_scale_tile(monkeypatch) -> None:
    names = [n for n in _RUN_TENSOR_NAMES if n not in ("out_global_scale", "out_sf")]
    values = {name: object() for name in names}
    values["qkv_scales"] = object()
    calls = []

    class _Prepared:
        def __call__(self):
            calls.append("run")
            return values["out_q"], None

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
        out_format="e4m3",
    )
    actual = operation.run(**values)

    assert calls[0]["out_format"] == "e4m3"
    assert calls[0]["qkv_scales"] is values["qkv_scales"]
    assert calls[0]["out_global_scale"] is None and calls[0]["out_sf"] is None
    assert operation.out_format == "e4m3"
    assert actual[0] is values["out_q"] and actual[1] is None

    with pytest.raises(ValueError, match="qkv_scales"):
        operation.run(**{**values, "qkv_scales": object()})


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((10, 0), "sm_100a"), ((10, 3), "sm_103a")],
)
def test_exact_architecture_router(monkeypatch, capability, expected) -> None:
    from flashinfer.diffusion_ops import cake_minimax_h3_nvfp4 as ops

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    assert ops.minimax_h3_nvfp4_target(torch.device("cuda")) == expected
    norm, gemm = ops.minimax_h3_nvfp4_programs(torch.device("cuda"), "e4m3")
    assert expected in norm.targets and gemm.targets == (expected,)
    assert gemm is not ops.minimax_h3_nvfp4_programs(torch.device("cuda"), "nvfp4")[1]


@pytest.mark.parametrize(
    ("capability", "target"),
    [((10, 0), "sm_100a"), ((10, 3), "sm_103a")],
)
def test_norm_program_follows_target_and_destinations(
    monkeypatch, capability, target
) -> None:
    from flashinfer.diffusion_ops import cake_minimax_h3_nvfp4 as ops
    from flashinfer.jit import cake_minimax_h3_nvfp4_pre_attention as programs

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    for P in (None, 1, 2, 4, 8):
        for out_format in ("nvfp4", "e4m3"):
            for M in (None, 1, 4824, 6095, 6096, 13744):
                norm, _ = ops.minimax_h3_nvfp4_programs(
                    torch.device("cuda"), out_format, P, M
                )
                staged = (target, P) == ("sm_103a", 4) or (
                    (out_format, target, P) == ("nvfp4", "sm_103a", 8)
                    and M is not None
                    and M >= 6096
                )
                assert norm is (programs.NORM_V5 if staged else programs.NORM)
                assert target in norm.targets


@pytest.mark.parametrize("capability", [(9, 0), (10, 7), (12, 0)])
def test_architecture_router_rejects_cross_routing(monkeypatch, capability) -> None:
    from flashinfer.diffusion_ops import cake_minimax_h3_nvfp4 as ops

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    with pytest.raises(RuntimeError, match="exact compute capability 10.0 or 10.3"):
        ops.minimax_h3_nvfp4_target(torch.device("cuda"))


@pytest.mark.parametrize(
    ("out_format", "missing", "extra", "message"),
    [
        ("nvfp4", ("out_sf",), {}, "out_global_scale and out_sf are required"),
        (
            "nvfp4",
            ("out_global_scale",),
            {},
            "out_global_scale and out_sf are required",
        ),
        ("nvfp4", (), {"qkv_scales": "qkv_scales"}, "qkv_scales must be None"),
        ("e4m3", ("out_global_scale", "out_sf"), {}, "qkv_scales is required"),
        (
            "e4m3",
            ("out_global_scale",),
            {"qkv_scales": "qkv_scales"},
            "out_global_scale and out_sf must be None",
        ),
        ("mxfp8", (), {}, "out_format must be one of"),
    ],
)
def test_output_format_contract_is_checked_before_the_device(
    out_format, missing, extra, message
) -> None:
    """The format's operand set is validated before any CUDA tensor is required (a CPU-only check)."""
    from flashinfer.diffusion_ops import cake_minimax_h3_nvfp4 as ops

    x = torch.zeros((4, 5376), dtype=torch.bfloat16)
    placeholder = torch.zeros((1,), dtype=torch.float32)
    kwargs = {
        name: placeholder
        for name in _RUN_TENSOR_NAMES
        if name != "x" and name not in missing
    }
    kwargs.update({name: placeholder for name in extra})
    with pytest.raises(ValueError, match=message):
        ops.prepare_minimax_h3_nvfp4_pre_attention(
            x=x,
            activation_q=placeholder,
            activation_sf=placeholder,
            P=8,
            out_format=out_format,
            **kwargs,
        )


def _prepare_zero_smoke(
    adaln_index: int,
    M: int = 1,
    P: int = 8,
    configure: Optional[Callable[[dict], None]] = None,
    out_format: str = "nvfp4",
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100a or SM103a")

    from flashinfer.diffusion_ops import cake_minimax_h3_nvfp4 as ops

    hidden, qkv_width, head_dim, fp4_block = 5376, 21504, 128, 16
    device = torch.device("cuda")
    activation_sf_len = ((M + 127) // 128 * 128) * (hidden // fp4_block)
    shapes = ops.minimax_h3_nvfp4_output_shapes(M, P, out_format)
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
        # Every output byte starts non-zero so a zero result is the kernel's.
        "out_q": torch.full(
            shapes["out_q"][0], 0x7F, dtype=torch.uint8, device=device
        ).view(shapes["out_q"][1]),
    }
    if out_format == "nvfp4":
        assert shapes["out_sf"] is not None
        values["out_global_scale"] = torch.ones(
            (1,), dtype=torch.float32, device=device
        )
        values["out_sf"] = torch.full(
            shapes["out_sf"][0], 255, dtype=torch.uint8, device=device
        )
    else:
        values["qkv_scales"] = torch.ones((3,), dtype=torch.float32, device=device)
    if configure is not None:
        # The fused route derives alpha and the CTA-pair-ordered weight scales at
        # preparation, so operand values a test depends on are set before the
        # operation is prepared (the E2M1 weight bytes are still read in place).
        configure(values)
    norm, gemm = ops.minimax_h3_nvfp4_programs(device, out_format, P)
    target = ops.minimax_h3_nvfp4_target(device)
    assert target in norm.targets and gemm.targets == (target,)
    operation = MiniMaxH3Nvfp4PreAttention(
        **values,
        activation_q=torch.empty((M, hidden // 2), dtype=torch.uint8, device=device),
        activation_sf=torch.empty(
            (activation_sf_len,), dtype=torch.uint8, device=device
        ),
        P=P,
        out_format=out_format,
    )
    return operation, values


def _assert_zero_outputs(actual_q, actual_sf, values: dict) -> None:
    assert actual_q is values["out_q"]
    assert torch.count_nonzero(actual_q.view(torch.uint8)).item() == 0
    if "out_sf" in values:
        assert actual_sf is values["out_sf"]
        assert torch.count_nonzero(actual_sf).item() == 0
    else:
        assert actual_sf is None


@pytest.mark.parametrize("out_format", ["nvfp4", "e4m3"])
@pytest.mark.parametrize("invalid_index", [-1, 9, -(2**31), 2**31 - 1])
def test_invalid_adaln_row_writes_zero_to_caller_outputs(
    invalid_index, out_format
) -> None:
    operation, values = _prepare_zero_smoke(invalid_index, out_format=out_format)
    actual_q, actual_sf = operation.run(**values)
    torch.cuda.synchronize()
    _assert_zero_outputs(actual_q, actual_sf, values)


@pytest.mark.parametrize("out_format", ["nvfp4", "e4m3"])
def test_prepared_api_cuda_graph_replay(out_format) -> None:
    operation, values = _prepare_zero_smoke(-1, out_format=out_format)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        operation.run(**values)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_q, captured_sf = operation.run(**values)
    values["out_q"].view(torch.uint8).fill_(0x7F)
    if "out_sf" in values:
        values["out_sf"].fill_(255)
    graph.replay()
    torch.cuda.synchronize()
    _assert_zero_outputs(captured_q, captured_sf, values)


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


def test_prepared_api_runs_on_the_tensors_device_stream() -> None:
    """Tensors on a device that is not the current one launch on that device's
    current torch stream: the stages wait for the input written there and a
    consumer on that stream sees the finished outputs, byte-identical to a run
    with the tensors' device current."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    capability = torch.cuda.get_device_capability(1)
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100a or SM103a")
    pytest.importorskip("flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention")

    M, P = 129, 8
    device = torch.device("cuda", 1)
    values = _make_numerical_values(M, P, device)
    activation_q = torch.empty((M, _HIDDEN // 2), dtype=torch.uint8, device=device)
    activation_sf = torch.empty(
        (_round_up(M, 128) * (_HIDDEN // _FP4_BLOCK),), dtype=torch.uint8, device=device
    )
    with torch.cuda.device(device):
        operation = MiniMaxH3Nvfp4PreAttention(
            **values, activation_q=activation_q, activation_sf=activation_sf, P=P
        )
        operation.run(**values)
        torch.cuda.synchronize()
    expected_q = values["out_q"].clone()
    expected_sf = values["out_sf"].clone()

    x = values["x"]
    x_input = x.clone()
    x.zero_()
    values["out_q"].fill_(0x7F)
    values["out_sf"].fill_(0xA5)
    torch.cuda.synchronize(device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.device(device):
        previous = torch.cuda.current_stream()
        torch.cuda.set_stream(stream)
    try:
        with torch.cuda.device(0):
            assert torch.cuda.current_stream(device) == stream
            # The input arrives late on the stream; a launch on any other
            # stream of the device would read the zeroed input.
            with torch.cuda.stream(stream):
                torch.cuda._sleep(200_000_000)
                x.copy_(x_input)
            out_q, out_sf = operation.run(**values)
            with torch.cuda.stream(stream):
                actual_q = out_q.clone()
                actual_sf = out_sf.clone()
        stream.synchronize()
    finally:
        with torch.cuda.device(device):
            torch.cuda.set_stream(previous)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_sf, expected_sf)


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

    assert ops.minimax_h3_nvfp4_launch_grid("norm", M) == expected_norm
    assert ops.minimax_h3_nvfp4_launch_grid("gemm", M) == expected_gemm
    assert ops._gemm_row_tiles(M) == expected_row_tiles
    with pytest.raises(ValueError, match="unknown stage"):
        ops.minimax_h3_nvfp4_launch_grid("other", M)
    rows = M * (56 // P) * 3
    assert ops.minimax_h3_nvfp4_rows_per_destination(M, P) == rows
    assert ops.minimax_h3_nvfp4_output_shapes(M, P, "nvfp4") == {
        "out_q": ((P, M, 56 // P, 3, 64), torch.uint8),
        "out_sf": ((P, (rows + 127) // 128 * 128 * 8), torch.uint8),
    }
    assert ops.minimax_h3_nvfp4_output_shapes(M, P, "e4m3") == {
        "out_q": ((P, M, 56 // P, 3, 128), torch.float8_e4m3fn),
        "out_sf": None,
    }


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
    sf_fp8 = (
        (gs * (amax * (1.0 / _E2M1_MAX))).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
    )
    sf_value = sf_fp8.float()
    encode = torch.where(
        amax != 0.0, 1.0 / (sf_value * (1.0 / gs)), torch.zeros_like(sf_value)
    )
    codes = _e2m1_codes(blocks * encode)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed.reshape(rows, width // 2), sf_fp8.reshape(
        rows, width // _FP4_BLOCK
    ).view(torch.uint8)


def _dequantize_nvfp4(packed, logical_scales, global_scale) -> torch.Tensor:
    rows = packed.shape[0]
    scale = logical_scales.contiguous().view(torch.float8_e4m3fn).float()
    values = _decode_e2m1(packed).reshape(rows, -1, _FP4_BLOCK) * scale.reshape(
        rows, -1, 1
    )
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
    out = torch.zeros(
        _round_up(rows_count, 128) * padded_cols,
        dtype=torch.uint8,
        device=logical.device,
    )
    rows = torch.arange(rows_count, device=logical.device).unsqueeze(1)
    cols = torch.arange(cols_count, device=logical.device).unsqueeze(0)
    out[_scale_indices(rows, cols, padded_cols).reshape(-1)] = logical.reshape(-1)
    return out


def _unswizzle_scales(
    physical: torch.Tensor, rows_count: int, cols_count: int
) -> torch.Tensor:
    flat = physical.contiguous().reshape(-1)
    rows = torch.arange(rows_count, device=flat.device).unsqueeze(1)
    cols = torch.arange(cols_count, device=flat.device).unsqueeze(0)
    return flat[_scale_indices(rows, cols, _round_up(cols_count, 4))].reshape(
        rows_count, cols_count
    )


def _partial_neox_rope(x: torch.Tensor, rope_cos_sin: torch.Tensor) -> torch.Tensor:
    half = _ROPE_DIM // 2
    rotary, tail = x[..., :_ROPE_DIM].float(), x[..., _ROPE_DIM:]
    cos = torch.cat((rope_cos_sin[:, :half],) * 2, dim=-1).float()[:, None, :]
    sin = torch.cat((rope_cos_sin[:, half:],) * 2, dim=-1).float()[:, None, :]
    rotate_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    return torch.cat(
        ((rotary * cos + rotate_half * sin).to(torch.bfloat16), tail), dim=-1
    )


def _reference(
    values: dict, activation_q: torch.Tensor, activation_sf: torch.Tensor
) -> dict:
    """Torch oracle of the two-stage route, rounded where the kernels round.

    Stage 1 (RMSNorm + AdaLN -> BF16) is recomputed independently.  Stage 2
    starts from the kernel's own NVFP4 activation bytes (checked separately by
    ``_assert_activation_quantization``) so that the fused GEMM and its
    epilogue are isolated: the tensor core multiplies the exact FP4 x E4M3
    operands and applies the two global scales once to the fp32 accumulators,
    so the oracle dequantizes exactly and accumulates in fp64 before the BF16
    rounding of the GEMM output (rounding the dequantized operands to BF16
    first would add a ~2^-9 relative error per operand that the kernel does
    not have).  Per-head RMSNorm, BF16 rounding, partial NeoX RoPE and a
    second BF16 rounding follow the kernel's epilogue.
    """
    M = values["x"].shape[0]
    norm = torch.nn.functional.rms_norm(
        values["x"], (_HIDDEN,), values["x_norm_weight"], eps=_EPS
    ).to(torch.bfloat16)
    index = values["adaln_index"].to(torch.int64)
    valid = (index >= 0) & (index < _ADALN_ROWS)
    safe = index.clamp(0, _ADALN_ROWS - 1)
    scale_plus_one = (values["adaln_scale"].index_select(0, safe) + 1.0).to(
        torch.bfloat16
    )
    adaln = torch.addcmul(
        values["adaln_shift"].index_select(0, safe), norm, scale_plus_one
    ).to(torch.bfloat16)
    adaln = torch.where(valid[:, None], adaln, torch.zeros_like(adaln))

    activation_logical_sf = _unswizzle_scales(activation_sf, M, _HIDDEN // _FP4_BLOCK)
    activation = _dequantize_nvfp4(
        activation_q, activation_logical_sf, values["x_global_scale"]
    ).double()
    weight_sf = _unswizzle_scales(
        values["qkv_weight_sf"], _QKV_WIDTH, _HIDDEN // _FP4_BLOCK
    )
    weight = _dequantize_nvfp4(
        values["qkv_weight_q"], weight_sf, values["w_global_scale"]
    ).double()
    qkv = (activation @ weight.T).to(torch.bfloat16)
    # Magnitude of the fp32 accumulation per output: the kernel sums the
    # 64-wide MMA K-slabs in fp32, HIDDEN / 64 additions of partial sums
    # bounded by sum |a_k b_k|.
    accumulation = (_HIDDEN // 64) * 2.0**-24 * (activation.abs() @ weight.abs().T)

    grouped = qkv.view(M, _NUM_HEADS, 3, _HEAD_DIM)
    accumulation = accumulation.view(M, _NUM_HEADS, 3, _HEAD_DIM).float()
    out = {"adaln": adaln, "v": grouped[:, :, 2]}
    for name, slot, weight_name in (
        ("q", 0, "q_norm_weight"),
        ("k", 1, "k_norm_weight"),
    ):
        x = grouped[:, :, slot]
        head_weight = values[weight_name].float().abs()
        rstd = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + _EPS)
        pre = torch.nn.functional.rms_norm(
            x, (_HEAD_DIM,), values[weight_name], eps=_EPS
        ).to(torch.bfloat16)
        out[name] = _partial_neox_rope(pre, values["rope_cos_sin"])
        out[f"{name}_bound"] = _fused_epilogue_bound(
            x,
            pre,
            out[name],
            rstd * head_weight,
            accumulation[:, :, slot] * rstd * head_weight,
            values["rope_cos_sin"],
        )
    return out


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    magnitude = x.float().abs()
    exponent = torch.floor(
        torch.log2(magnitude.clamp_min(torch.finfo(torch.float32).tiny))
    )
    return torch.where(
        magnitude > 0,
        torch.finfo(torch.bfloat16).eps * torch.exp2(exponent),
        torch.zeros_like(magnitude),
    )


def _fused_epilogue_bound(
    x, pre, post, gain, accumulation, rope_cos_sin
) -> torch.Tensor:
    """Per-element bound for the kernel's post-norm/RoPE Q or K versus the oracle.

    Both round at the same three points (GEMM output -> BF16 ``x``, post-norm
    product -> BF16 ``pre``, rotated value -> BF16 ``post``); only fp32
    accumulation order and the fp32 RMSNorm/RoPE arithmetic differ, so each
    rounding may flip by one BF16 ulp.  Summing the flips through the data
    flow (``gain`` = rstd * |head weight|; rope dims mix element ``i`` with its
    partner ``j`` through cos/sin; tail dims are copied once, so their
    post-norm rounding is the output rounding):

    * ``ulp(x_i) * gain_i``: one flip of the GEMM output, scaled by the norm;
    * ``ulp(pre_i)``: one flip of the post-norm product;
    * ``rel_rstd * |post_i|``: the rstd change from flips of the GEMM outputs
      of the head, first order in sum ulp(x_k)|x_k| / sum x_k^2;
    * ``accumulation``: the fp32 accumulation magnitude from ``_reference``;
    * ``ulp(post_i)``: one flip of the output rounding (rope dims only).
    """
    half = _ROPE_DIM // 2
    x_f, pre_f, post_f = x.float(), pre.float(), post.float()
    rel_rstd = (_bf16_ulp(x_f) * x_f.abs()).sum(dim=-1, keepdim=True) / x_f.pow(2).sum(
        dim=-1, keepdim=True
    ).clamp_min(_EPS)
    per_element = _bf16_ulp(x_f) * gain + _bf16_ulp(pre_f) + accumulation
    bound = per_element + rel_rstd * post_f.abs()
    cos = rope_cos_sin[:, :half].float().abs()[:, None, :]
    sin = rope_cos_sin[:, half:].float().abs()[:, None, :]
    lower, upper = per_element[..., :half], per_element[..., half:_ROPE_DIM]
    rotated = torch.cat((lower * cos + upper * sin, upper * cos + lower * sin), dim=-1)
    bound[..., :_ROPE_DIM] = (
        rotated
        + _bf16_ulp(post_f[..., :_ROPE_DIM])
        + rel_rstd * post_f[..., :_ROPE_DIM].abs()
    )
    return bound


def _assert_within(
    actual: torch.Tensor, expected: torch.Tensor, bound: torch.Tensor, name: str
) -> None:
    error = (actual.float() - expected.float()).abs()
    outside = error > bound
    if bool(outside.any()):
        worst = int(
            torch.argmax(error / bound.clamp_min(torch.finfo(torch.float32).tiny))
        )
        index = tuple(
            int(i) for i in torch.unravel_index(torch.tensor(worst), error.shape)
        )
        raise AssertionError(
            f"{name}: {int(outside.sum())} / {error.numel()} elements outside the derived bound; worst at {index}: "
            f"actual {float(actual.float().flatten()[worst])!r} expected {float(expected.float().flatten()[worst])!r} "
            f"bound {float(bound.flatten()[worst])!r}"
        )


def _assert_activation_quantization(
    debug_adaln, activation_q, activation_sf, global_scale, M: int
) -> None:
    """The stage-1 quantizer reproduces the oracle's round-to-nearest NVFP4 encoding of its own BF16 output bit for bit."""
    expected_q, expected_sf = _quantize_nvfp4(debug_adaln, global_scale)
    logical_sf = _unswizzle_scales(activation_sf, M, _HIDDEN // _FP4_BLOCK)
    assert torch.equal(logical_sf, expected_sf), int((logical_sf != expected_sf).sum())
    assert torch.equal(activation_q, expected_q), int(
        (activation_q != expected_q).sum()
    )


def _dequantize_destination_major(out_q, out_sf, global_scale) -> torch.Tensor:
    restored = []
    for destination in range(out_q.shape[0]):
        packed = out_q[destination].reshape(-1, _HEAD_DIM // 2)
        logical = _unswizzle_scales(
            out_sf[destination], packed.shape[0], _HEAD_DIM // _FP4_BLOCK
        )
        restored.append(
            _dequantize_nvfp4(packed, logical, global_scale).reshape(
                *out_q.shape[1:-1], _HEAD_DIM
            )
        )
    return torch.stack(restored)


def _global_encode_scale(amax: torch.Tensor) -> torch.Tensor:
    scale = torch.div(_E4M3_MAX * _E2M1_MAX, amax.float())
    return torch.where(scale == 0.0, torch.ones_like(scale), scale).reshape(1)


def _rope_cache(M: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(M, dtype=torch.float32, device=device)
    axes = (rows // 4096, (rows // 64).remainder(64), rows.remainder(64))
    width = _ROPE_DIM // 6
    inv_freq = torch.pow(
        10000.0, -torch.arange(width, dtype=torch.float32, device=device) / width
    )
    phase = torch.cat([axis[:, None] * inv_freq[None, :] for axis in axes], dim=-1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(torch.bfloat16).contiguous()


def _make_numerical_values(M: int, P: int, device: torch.device) -> dict:
    generator = torch.Generator(device=device).manual_seed(0x5791 + M * 16 + P)

    def uniform(shape, low, high, dtype=torch.bfloat16):
        return (
            torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
            * (high - low)
            + low
        ).to(dtype)

    x = uniform((M, _HIDDEN), -2.0, 2.0)
    adaln_index = torch.div(
        torch.arange(M, device=device) * _ADALN_ROWS, M, rounding_mode="floor"
    ).to(torch.int32)
    if M >= 3:
        adaln_index[1] = -1
        adaln_index[2] = _ADALN_ROWS
    x_norm_weight = uniform((_HIDDEN,), 0.8, 1.2)
    adaln_scale = uniform((_ADALN_ROWS, _HIDDEN), -0.5, 0.5)
    adaln_shift = uniform((_ADALN_ROWS, _HIDDEN), -0.5, 0.5)
    norm = torch.nn.functional.rms_norm(x, (_HIDDEN,), x_norm_weight, eps=_EPS)
    adaln_amax = (
        (norm * (adaln_scale + 1.0).abs().amax(dim=0) + adaln_shift.abs().amax(dim=0))
        .abs()
        .amax()
    )
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
        "out_q": torch.full(
            (P, M, _NUM_HEADS // P, 3, _HEAD_DIM // 2),
            0x7F,
            dtype=torch.uint8,
            device=device,
        ),
        "out_sf": torch.full(
            (P, _round_up(rows_per_destination, 128) * (_HEAD_DIM // _FP4_BLOCK)),
            0xA5,
            dtype=torch.uint8,
            device=device,
        ),
    }


@pytest.mark.parametrize(
    ("M", "P"), [(1, 8), (129, 8), (2048, 1), (4097, 2), (2048, 4)]
)
def test_numerical_reference(M: int, P: int) -> None:
    """The two stages match an independent torch oracle of the NVFP4 route.

    The AdaLN output is checked at BF16 tolerance and its NVFP4 quantization
    bit for bit; post-norm/RoPE Q and K are checked against an oracle that
    starts from the kernel's activation bytes, within a per-element bound
    derived from the kernel's rounding points; the packed destination is
    checked after dequantization against
    the oracle's quantization of the kernel's own BF16 intermediates (one E2M1
    step at the block scale), and the Q/K scale blocks must be bit-identical.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100a or SM103a")
    pytest.importorskip("flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention")

    device = torch.device("cuda")
    values = _make_numerical_values(M, P, device)
    debug = {
        "debug_adaln_bf16": torch.full(
            (M, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
        ),
        "debug_q_bf16": torch.full(
            (M, _NUM_HEADS, _HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
        "debug_k_bf16": torch.full(
            (M, _NUM_HEADS, _HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    activation_q = torch.empty((M, _HIDDEN // 2), dtype=torch.uint8, device=device)
    activation_sf = torch.empty(
        (_round_up(M, 128) * (_HIDDEN // _FP4_BLOCK),), dtype=torch.uint8, device=device
    )
    operation = MiniMaxH3Nvfp4PreAttention(
        **values, activation_q=activation_q, activation_sf=activation_sf, P=P, **debug
    )
    out_q, out_sf = operation.run(**values, **debug)
    torch.cuda.synchronize()
    expected = _reference(values, activation_q, activation_sf)

    assert out_q is values["out_q"] and out_sf is values["out_sf"]
    # Stage 1: RMSNorm + AdaLN with invalid rows zeroed (BF16 tolerance), and
    # its NVFP4 quantization reproduced bit for bit from the kernel's output.
    torch.testing.assert_close(
        debug["debug_adaln_bf16"], expected["adaln"], atol=1e-2, rtol=1e-2
    )
    _assert_activation_quantization(
        debug["debug_adaln_bf16"],
        activation_q,
        activation_sf,
        values["x_global_scale"],
        M,
    )
    # Fused epilogue intermediates: per-head RMSNorm + partial NeoX RoPE on the
    # NVFP4 GEMM result, within the per-element bound derived from the
    # kernel's rounding points (see _fused_epilogue_bound).
    _assert_within(
        debug["debug_q_bf16"], expected["q"], expected["q_bound"], "debug_q_bf16"
    )
    _assert_within(
        debug["debug_k_bf16"], expected["k"], expected["k_bound"], "debug_k_bf16"
    )

    # Destination pack: the kernel quantizes its own fused BF16 values, so the
    # oracle packs the kernel's debug Q/K plus the oracle V to isolate the pack,
    # and both are compared after dequantization at one E2M1 step of the
    # block's scale (the largest E2M1 spacing is two units of the scale).
    kernel_fused = torch.stack(
        (debug["debug_q_bf16"], debug["debug_k_bf16"], expected["v"]), dim=2
    )
    kernel_destination = (
        kernel_fused.view(M, P, _NUM_HEADS // P, 3, _HEAD_DIM)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
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
    logical_actual = torch.stack(
        [
            _unswizzle_scales(out_sf[d], rows_per_destination, scale_cols)
            for d in range(P)
        ]
    )
    logical_expected = torch.stack(
        [
            _unswizzle_scales(expected_sf[d], rows_per_destination, scale_cols)
            for d in range(P)
        ]
    )
    block_scale = (
        torch.maximum(
            logical_actual.view(torch.float8_e4m3fn).float(),
            logical_expected.view(torch.float8_e4m3fn).float(),
        )
        / gs.float()
    )
    step = (
        (2.0 * block_scale)
        .reshape(P, M, _NUM_HEADS // P, 3, scale_cols, 1)
        .expand(-1, -1, -1, -1, -1, _FP4_BLOCK)
    )
    error = (actual - oracle).abs()
    assert bool((error <= step.reshape(actual.shape) + 1e-6).all()), float(error.max())
    # Most elements agree exactly; the Q/K rows are quantized from identical
    # BF16 inputs and only accumulation-order effects on V may differ.
    assert float((out_q != expected_q).float().mean()) <= 0.01
    assert float((logical_actual != logical_expected).float().mean()) <= 0.01


# ---------------------------------------------------------------------------
# E4M3 output mode
# ---------------------------------------------------------------------------


def _e4m3_cast(values: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
    """``e4m3(value * out_scale)``: float32 multiply, round to nearest even, saturating at +-448."""
    scaled = (values.float() * out_scale).clamp(-_E4M3_MAX, _E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn)


@pytest.mark.parametrize(
    ("M", "P"), [(129, 4), (300, 4), (2048, 4), (6096, 8), (7368, 8)]
)
def test_staged_norm_program_writes_the_norm_bytes(monkeypatch, M: int, P: int) -> None:
    """On ``sm_103a`` the staged norm program (``NORM_V5``) writes the bytes of ``NORM``.

    It serves four destinations and, for ``out_format="nvfp4"``, eight
    destinations from ``M = 6096`` tokens on.  Reference: the ``NORM`` program
    on the same inputs, every output and debug buffer compared byte for byte.
    The (129, 4) and (300, 4) inputs put E2M1 rounding ties in the activation:
    the kernels form the encode factor with approximate reciprocals and the
    torch oracle of ``test_numerical_reference`` with IEEE division, so at a tie
    the two can pick neighbouring codes; the program-to-program comparison is
    exact at every element.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103a")
    pytest.importorskip("flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention")
    from flashinfer.diffusion_ops import cake_minimax_h3_nvfp4 as ops
    from flashinfer.jit import cake_minimax_h3_nvfp4_pre_attention as programs

    device = torch.device("cuda")

    def run() -> dict:
        values = _make_numerical_values(M, P, device)
        debug = {
            "debug_adaln_bf16": torch.full(
                (M, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
            ),
            "debug_q_bf16": torch.full(
                (M, _NUM_HEADS, _HEAD_DIM),
                float("nan"),
                dtype=torch.bfloat16,
                device=device,
            ),
            "debug_k_bf16": torch.full(
                (M, _NUM_HEADS, _HEAD_DIM),
                float("nan"),
                dtype=torch.bfloat16,
                device=device,
            ),
        }
        activation_q = torch.full(
            (M, _HIDDEN // 2), 0xA5, dtype=torch.uint8, device=device
        )
        activation_sf = torch.full(
            (_round_up(M, 128) * (_HIDDEN // _FP4_BLOCK),),
            0xA5,
            dtype=torch.uint8,
            device=device,
        )
        operation = MiniMaxH3Nvfp4PreAttention(
            **values,
            activation_q=activation_q,
            activation_sf=activation_sf,
            P=P,
            **debug,
        )
        operation.run(**values, **debug)
        torch.cuda.synchronize()
        return {
            "activation_q": activation_q,
            "activation_sf": activation_sf,
            "out_q": values["out_q"],
            "out_sf": values["out_sf"],
            **debug,
        }

    assert ops.minimax_h3_nvfp4_programs(device, "nvfp4", P, M)[0] is programs.NORM_V5
    staged = run()
    with monkeypatch.context() as patch:
        patch.setattr(ops, "_NORM_PROGRAMS", {})
        patch.setattr(ops, "_NORM_PROGRAMS_FROM_M", {})
        assert ops.minimax_h3_nvfp4_programs(device, "nvfp4", P, M)[0] is programs.NORM
        shared = run()
    for name, tensor in staged.items():
        assert torch.equal(tensor.view(torch.uint8), shared[name].view(torch.uint8)), (
            name
        )


@pytest.mark.parametrize(
    ("M", "out_format", "staged"),
    [(6095, "nvfp4", False), (6096, "nvfp4", True), (6096, "e4m3", False)],
)
def test_prepared_operation_launches_the_routed_norm_program(
    M: int, out_format: str, staged: bool
) -> None:
    """A prepared eight-destination operation on ``sm_103a`` launches the norm
    program of its route: ``NORM_V5`` for ``out_format="nvfp4"`` from
    ``M = 6096`` tokens on, ``NORM`` below that and for ``e4m3``.

    The kernels of one run are read from the CUDA profiler trace, so the check
    covers the programs the operation prepared, not only the routing table.  A
    profiler session that loses its device records (``CUPTI-LOST-RECORDS``) is
    retried, and the test skips rather than reporting a launch it cannot see.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103a")
    pytest.importorskip("flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention")
    from flashinfer.jit import cake_minimax_h3_nvfp4_pre_attention as programs
    from tests.test_helpers.cake_profiler import cuda_kernel_names

    operation, values = _prepare_zero_smoke(-1, M=M, P=8, out_format=out_format)
    operation.run(**values)
    torch.cuda.synchronize()
    launched = sorted(
        name
        for name in cuda_kernel_names(lambda: operation.run(**values))
        if name.startswith("kernel_cake_minimax_h3_nvfp4_pre_attention_")
    )
    norm = programs.NORM_V5 if staged else programs.NORM
    gemm = programs.FUSED_PACK_K96 if out_format == "nvfp4" else programs.FUSED_E4M3_K96
    assert launched == sorted([norm.kernel, gemm.kernel])


def _e4m3_step(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One E4M3 code step at the larger magnitude: 2^(e - 3) for normals, 2^-9 below 2^-6."""
    magnitude = torch.maximum(a.abs(), b.abs()).clamp_min(2.0**-6)
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 3.0)


def _make_e4m3_values(M: int, P: int, qkv_scales, device: torch.device) -> dict:
    values = _make_numerical_values(M, P, device)
    del values["out_global_scale"], values["out_sf"]
    values["out_q"] = torch.full(
        (P, M, _NUM_HEADS // P, 3, _HEAD_DIM), 0x7F, dtype=torch.uint8, device=device
    ).view(torch.float8_e4m3fn)
    values["qkv_scales"] = torch.tensor(qkv_scales, dtype=torch.float32, device=device)
    return values


def test_e4m3_rejects_wrong_operands() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100a or SM103a")
    device = torch.device("cuda")
    values = _make_e4m3_values(1, 8, (0.5, 2.0, 0.25), device)
    common = {
        "activation_q": torch.empty(
            (1, _HIDDEN // 2), dtype=torch.uint8, device=device
        ),
        "activation_sf": torch.empty(
            (128 * (_HIDDEN // _FP4_BLOCK),), dtype=torch.uint8, device=device
        ),
        "P": 8,
        "out_format": "e4m3",
    }
    with pytest.raises(ValueError, match=r"qkv_scales must have shape \(3,\)"):
        MiniMaxH3Nvfp4PreAttention(
            **{**values, "qkv_scales": torch.ones((2,), device=device)}, **common
        )
    with pytest.raises(TypeError, match="out_q must have dtype torch.float8_e4m3fn"):
        MiniMaxH3Nvfp4PreAttention(
            **{**values, "out_q": values["out_q"].view(torch.uint8)}, **common
        )
    with pytest.raises(ValueError, match="out_q must have shape"):
        MiniMaxH3Nvfp4PreAttention(
            **{**values, "out_q": values["out_q"][..., : _HEAD_DIM // 2].contiguous()},
            **common,
        )


def test_nvfp4_rejects_a_misaligned_scale_tile() -> None:
    """The norm stage zeroes the padding rows of ``out_sf`` in 8-byte words."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100a or SM103a")
    device = torch.device("cuda")
    M, P = 1, 8
    values = _make_numerical_values(M, P, device)
    size = values["out_sf"].numel()
    storage = torch.zeros(size + 8, dtype=torch.uint8, device=device)
    misaligned = storage[4 : 4 + size].view(values["out_sf"].shape)
    common = {
        "activation_q": torch.empty(
            (M, _HIDDEN // 2), dtype=torch.uint8, device=device
        ),
        "activation_sf": torch.empty(
            (_round_up(M, 128) * (_HIDDEN // _FP4_BLOCK),),
            dtype=torch.uint8,
            device=device,
        ),
        "P": P,
    }
    with pytest.raises(ValueError, match="out_sf must be 8-byte aligned"):
        MiniMaxH3Nvfp4PreAttention(**{**values, "out_sf": misaligned}, **common)


@pytest.mark.parametrize(
    ("M", "P", "qkv_scales"),
    [
        (1, 8, (0.5, 2.0, 2.0**-8)),
        (129, 8, (4.0 / 448, 4.0 / 448, 8.0 / 448)),
        (2048, 1, (0.3, 1.7, 0.9)),
        (4097, 2, (4.0 / 448, 4.0 / 448, 8.0 / 448)),
        (2048, 4, (0.3, 1.7, 0.9)),
    ],
)
def test_e4m3_numerical_reference(M: int, P: int, qkv_scales) -> None:
    """``out_format="e4m3"`` writes the same fused values, cast per kind.

    The stage-1 and fused-epilogue checks are the NVFP4 route's (the e4m3
    program shares the stage-1 program and the GEMM + Q/K RMSNorm + RoPE of
    its NVFP4 sibling).  The Q and K bytes must equal the cast of the kernel's
    own BF16 debug values exactly; the V bytes must equal the cast of the
    oracle V within one E4M3 code step, with at most 1 % of the elements
    differing (fp32 accumulation order); ``out_sf`` is ``None``; a second
    launch reproduces the bytes.  The scales cover a saturating V (M = 1),
    the fixture's magnitudes at ~448 full scale, and non-power-of-two scales
    (M = 2048), which pin the multiply spelling and its ties.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100a or SM103a")

    device = torch.device("cuda")
    values = _make_e4m3_values(M, P, qkv_scales, device)
    debug = {
        "debug_adaln_bf16": torch.full(
            (M, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
        ),
        "debug_q_bf16": torch.full(
            (M, _NUM_HEADS, _HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
        "debug_k_bf16": torch.full(
            (M, _NUM_HEADS, _HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    activation_q = torch.empty((M, _HIDDEN // 2), dtype=torch.uint8, device=device)
    activation_sf = torch.empty(
        (_round_up(M, 128) * (_HIDDEN // _FP4_BLOCK),), dtype=torch.uint8, device=device
    )
    operation = MiniMaxH3Nvfp4PreAttention(
        **values,
        activation_q=activation_q,
        activation_sf=activation_sf,
        P=P,
        out_format="e4m3",
        **debug,
    )
    out_q, out_sf = operation.run(**values, **debug)
    torch.cuda.synchronize()
    assert out_q is values["out_q"] and out_sf is None
    expected = _reference(values, activation_q, activation_sf)

    torch.testing.assert_close(
        debug["debug_adaln_bf16"], expected["adaln"], atol=1e-2, rtol=1e-2
    )
    _assert_activation_quantization(
        debug["debug_adaln_bf16"],
        activation_q,
        activation_sf,
        values["x_global_scale"],
        M,
    )
    _assert_within(
        debug["debug_q_bf16"], expected["q"], expected["q_bound"], "debug_q_bf16"
    )
    _assert_within(
        debug["debug_k_bf16"], expected["k"], expected["k_bound"], "debug_k_bf16"
    )

    out_scale = (1.0 / values["qkv_scales"].double()).float()
    fused = torch.stack(
        (debug["debug_q_bf16"], debug["debug_k_bf16"], expected["v"]), dim=2
    )
    destination = fused.view(M, P, _NUM_HEADS // P, 3, _HEAD_DIM).permute(1, 0, 2, 3, 4)
    oracle = _e4m3_cast(destination, out_scale.view(1, 1, 1, 3, 1)).contiguous()
    actual_bytes = out_q.view(torch.uint8)
    oracle_bytes = oracle.view(torch.uint8)
    # Q and K: the kernel casts the BF16 values it also reports as debug outputs.
    assert torch.equal(actual_bytes[:, :, :, :2], oracle_bytes[:, :, :, :2])
    # V: the oracle's BF16 GEMM output may differ from the kernel's by a BF16
    # rounding flip (accumulation order), which moves the cast by one code at most.
    actual_v, oracle_v = out_q[:, :, :, 2].float(), oracle[:, :, :, 2].float()
    assert torch.isfinite(actual_v).all()
    error = (actual_v - oracle_v).abs()
    assert bool((error <= _e4m3_step(actual_v, oracle_v) + 1e-7).all()), float(
        error.max()
    )
    differing = actual_bytes[:, :, :, 2] != oracle_bytes[:, :, :, 2]
    assert float(differing.float().mean()) <= 0.01

    first = actual_bytes.clone()
    out_q.view(torch.uint8).fill_(0x55)
    operation.run(**values, **debug)
    torch.cuda.synchronize()
    assert torch.equal(out_q.view(torch.uint8), first)
