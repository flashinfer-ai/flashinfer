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

import re
from pathlib import Path

import pytest
import torch

import flashinfer
from flashinfer.jit.gated_act_mxfp8 import gen_gated_act_mxfp8_module
from flashinfer.quantization import SfLayout, mxfp8_quantize
from flashinfer.utils import get_compute_capability, has_flashinfer_jit_cache


MODES = (
    ("forward", True, False),
    ("forward", False, True),
    ("forward", True, True),
    ("backward", True, False),
    ("backward", False, True),
    ("backward", True, True),
)

FROZEN_LAUNCHER_SOURCES = (
    "gated_act_mxfp8_fwd_row_noalloc.cu",
    "gated_act_mxfp8_fwd_both_noalloc.cu",
    "gated_act_mxfp8_bwd_row.cu",
    "gated_act_mxfp8_bwd_row_sm103.cu",
)


def test_frozen_launcher_sources_use_cuda_tensor_map_definition() -> None:
    source_dir = Path(__file__).resolve().parents[2] / "csrc" / "gated_act_mxfp8"
    tensor_map_typedef = re.compile(r"\btypedef\b[^;]*\bCUtensorMap\s*;", re.DOTALL)

    for source_name in FROZEN_LAUNCHER_SOURCES:
        source = (source_dir / source_name).read_text()
        assert tensor_map_typedef.search(source) is None, (
            f"{source_name} must use CUtensorMap from <cuda.h>"
        )


def _supported() -> bool:
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda:0")
    ) in (
        (10, 0),
        (10, 3),
    )


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    if _supported():
        flashinfer.jit.build_jit_specs([gen_gated_act_mxfp8_module()], verbose=False)
    yield


def _make_inputs(m: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(20260820)
    gated_input = torch.randn(
        (m, 2 * k), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    grad_output = torch.randn(
        (m, k), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    return gated_input, grad_output


def _logical(
    direction: str, gated_input: torch.Tensor, grad_output: torch.Tensor
) -> torch.Tensor:
    k = gated_input.shape[1] // 2
    gate = gated_input[:, :k].float()
    up = gated_input[:, k:].float()
    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    if direction == "forward":
        return (silu_gate * up).bfloat16()
    dact = silu_gate * (1.0 - sigmoid_gate) + sigmoid_gate
    dgate = ((dact * grad_output.float()) * up).bfloat16()
    dup = (silu_gate * grad_output.float()).bfloat16()
    return torch.cat((dgate, dup), dim=1)


def _quantize_reference(
    logical: torch.Tensor, rowwise: bool, colwise: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    empty_q = logical.new_empty(0, dtype=torch.float8_e4m3fn)
    empty_s = logical.new_empty(0, dtype=torch.uint8)
    if rowwise:
        row_q, row_s = mxfp8_quantize(logical, sf_swizzle_layout=SfLayout.layout_128x4)
        row_s = row_s.reshape(logical.shape[0], logical.shape[1] // 32)
    else:
        row_q, row_s = empty_q, empty_s
    if colwise:
        col_q_t, col_s = mxfp8_quantize(
            logical.T.contiguous(), sf_swizzle_layout=SfLayout.layout_128x4
        )
        col_q = col_q_t.T
    else:
        col_q, col_s = empty_q, empty_s
    return row_q, col_q, row_s, col_s


def _unswizzle(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    values = sf.reshape(padded_rows // 128, padded_cols // 4, 32, 4, 4)
    return values.permute(0, 3, 2, 1, 4).reshape(padded_rows, padded_cols)[:rows, :cols]


def _e4m3_ordinal(bits: torch.Tensor) -> torch.Tensor:
    values = bits.to(torch.int16)
    return torch.where((values & 0x80) != 0, 0x80 - (values & 0x7F), 0x80 + values)


def _assert_backward_orientation(
    actual_q: torch.Tensor,
    actual_s: torch.Tensor,
    expected_q: torch.Tensor,
    expected_s: torch.Tensor,
    rowwise: bool,
) -> None:
    m, n = expected_q.shape
    if rowwise:
        actual_scale = _unswizzle(actual_s.view(torch.uint8), m, n // 32)
        expected_scale = _unswizzle(expected_s.view(torch.uint8), m, n // 32)
        actual_element_scale = actual_scale.repeat_interleave(32, dim=1)
        expected_element_scale = expected_scale.repeat_interleave(32, dim=1)
    else:
        actual_scale = _unswizzle(actual_s.view(torch.uint8), n, m // 32)
        expected_scale = _unswizzle(expected_s.view(torch.uint8), n, m // 32)
        actual_element_scale = actual_scale.repeat_interleave(32, dim=1).T
        expected_element_scale = expected_scale.repeat_interleave(32, dim=1).T

    scale_gap = (actual_scale.to(torch.int16) - expected_scale.to(torch.int16)).abs()
    invalid_scale = (actual_scale == 0xFF) | (expected_scale == 0xFF)
    assert not (((scale_gap > 1) & ~invalid_scale).any())
    assert not (invalid_scale & (actual_scale != expected_scale)).any(), (
        "invalid scale payloads must match"
    )
    scale_differences = int((actual_scale != expected_scale).sum())
    assert scale_differences <= max(8, int(1.0e-5 * actual_scale.numel()))

    invalid_elements = (actual_element_scale == 0xFF) | (expected_element_scale == 0xFF)
    exponent_delta = actual_element_scale.to(torch.int32) - expected_element_scale.to(
        torch.int32
    )
    ratio = torch.ldexp(
        torch.ones_like(actual_element_scale, dtype=torch.float32), exponent_delta
    )
    ratio = torch.where(invalid_elements, torch.ones_like(ratio), ratio)
    actual_bits = actual_q.contiguous().view(torch.uint8)
    expected_bits = expected_q.contiguous().view(torch.uint8)
    reencoded = (actual_q.float() * ratio).to(torch.float8_e4m3fn)
    common_bits = torch.where(
        invalid_elements, actual_bits, reencoded.contiguous().view(torch.uint8)
    )
    invalid_payload = invalid_elements & (
        (actual_element_scale != expected_element_scale)
        | (actual_bits != expected_bits)
    )
    code_gap = (_e4m3_ordinal(common_bits) - _e4m3_ordinal(expected_bits)).abs()
    assert not ((code_gap > 1) | invalid_payload).any()
    code_differences = int((code_gap != 0).sum())
    assert code_differences <= max(8, int(1.0e-5 * code_gap.numel()))


def _run(
    direction: str,
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    rowwise: bool,
    colwise: bool,
):
    if direction == "forward":
        return flashinfer.silu_and_mul_mxfp8_quantize(
            gated_input, rowwise=rowwise, colwise=colwise
        )
    return flashinfer.silu_and_mul_mxfp8_quantize_backward(
        gated_input, grad_output, rowwise=rowwise, colwise=colwise
    )


@pytest.mark.parametrize("m,k", [(128, 128), (256, 512)])
@pytest.mark.parametrize("direction,rowwise,colwise", MODES)
@torch.inference_mode()
def test_gated_act_mxfp8_correctness(m, k, direction, rowwise, colwise):
    if not _supported():
        pytest.skip("fused gated MXFP8 quantization requires SM100 or SM103")
    gated_input, grad_output = _make_inputs(m, k)
    actual = _run(direction, gated_input, grad_output, rowwise, colwise)
    logical = _logical(direction, gated_input, grad_output)
    expected = _quantize_reference(logical, rowwise, colwise)

    assert tuple(t.shape for t in actual) == tuple(t.shape for t in expected)
    assert tuple(t.stride() for t in actual) == tuple(t.stride() for t in expected)
    assert actual[0].dtype == actual[1].dtype == torch.float8_e4m3fn
    assert actual[2].dtype == actual[3].dtype == torch.float8_e8m0fnu

    if direction == "forward":
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_tensor.contiguous().view(torch.uint8),
                expected_tensor.contiguous().view(torch.uint8),
                rtol=0,
                atol=0,
            )
    else:
        if rowwise:
            _assert_backward_orientation(
                actual[0], actual[2], expected[0], expected[2], True
            )
        if colwise:
            _assert_backward_orientation(
                actual[1], actual[3], expected[1], expected[3], False
            )


@pytest.mark.parametrize("direction", ["forward", "backward"])
@torch.inference_mode()
def test_gated_act_mxfp8_mode_consistency(direction):
    if not _supported():
        pytest.skip("fused gated MXFP8 quantization requires SM100 or SM103")
    gated_input, grad_output = _make_inputs(256, 512)
    row = _run(direction, gated_input, grad_output, True, False)
    col = _run(direction, gated_input, grad_output, False, True)
    both = _run(direction, gated_input, grad_output, True, True)
    for actual, expected in (
        (both[0], row[0]),
        (both[2], row[2]),
        (both[1], col[1]),
        (both[3], col[3]),
    ):
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
