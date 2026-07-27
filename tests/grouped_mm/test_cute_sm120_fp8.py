"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for the cute SM120 FP8 float-scale zero-padding MoE GEMM entry
(:func:`flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise`).
"""

import math
from typing import Tuple

import pytest
import torch

from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise
from flashinfer.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
from flashinfer.utils import is_sm120a_supported

CALC_DIFF_THRESHOLD = 1e-3


def skip_if_not_sm120():
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("cute SM120 FP8 groupwise GEMM requires SM120a")


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum().item()
    if denom == 0:
        return 0.0
    return 1.0 - 2.0 * (x * y).sum().item() / denom


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * 3) // 4 * 4


def per_token_cast_to_fp8_for_moe_gemm(
    x: torch.Tensor,
    token_offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Python-eager reference for the cute SM120 FP8 moe_gemm zero-padding scale layout.

    Quantizes ``x`` with per-token ``(1, 128)`` float32 scales (DG helper
    per_token_cast_to_fp8; row-independent, so whole-tensor quantization equals
    per-expert quantization) and re-packs the scales into the contiguous MN-major
    ``(k_blocks, m_padded)`` layout with per-expert 4-row-aligned start columns;
    padding columns are zero-filled and ignored by the kernel.
    """
    assert x.dim() == 2
    assert token_offset.dtype == torch.int32
    assert token_offset[0].item() == 0

    token_num = x.shape[0]
    num_experts = token_offset.numel() - 1
    x_fp8, sf = per_token_cast_to_fp8(x)
    scale_k = sf.size(1)
    m_padded = compute_padded_offset(token_num, num_experts)
    padded = torch.zeros((scale_k, m_padded), dtype=torch.float32, device=x.device)
    for i in range(num_experts):
        start = int(token_offset[i].item())
        end = int(token_offset[i + 1].item())
        if start == end:
            continue
        padded_start = compute_padded_offset(start, i)
        padded[:, padded_start : padded_start + end - start] = sf[start:end].t()
    return x_fp8, padded


def make_inputs(m_per_expert_list, n, k):
    torch.random.manual_seed(0)
    num_experts = len(m_per_expert_list)
    offsets = [0]
    for m_pe in m_per_expert_list:
        offsets.append(offsets[-1] + m_pe)
    total_rows = offsets[-1]

    a = torch.randn((total_rows, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_experts, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    ref = torch.zeros((total_rows, n), dtype=torch.bfloat16, device="cuda")
    for i in range(num_experts):
        start, end = offsets[i], offsets[i + 1]
        if start < end:
            ref[start:end] = a[start:end] @ b[i].t()

    a_fp8, a_scale = per_token_cast_to_fp8_for_moe_gemm(a, m_indptr)
    b_fp8_list, b_sf_list = [], []
    for i in range(num_experts):
        b_i_fp8, b_i_sf = per_block_cast_to_fp8(b[i])
        b_fp8_list.append(b_i_fp8)
        b_sf_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_sf_list, dim=0).transpose(-1, -2).contiguous()
    return a_fp8, b_fp8, a_scale, b_scale, m_indptr, ref


@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("rows_per_expert", [1, 8, 192, 1024])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
def test_moe_gemm_fp8_nt_groupwise(num_experts, rows_per_expert, n, k):
    skip_if_not_sm120()
    a, b, a_scale, b_scale, m_indptr, ref = make_inputs(
        [rows_per_expert] * num_experts, n, k
    )
    out = moe_gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, m_indptr)
    diff = calc_diff(out.float(), ref.float())
    assert diff < CALC_DIFF_THRESHOLD, f"calc_diff={diff:.6e}"


@pytest.mark.parametrize(
    "m_per_expert_list",
    [
        [1, 1, 8, 16, 64, 128, 192, 256],
        [0, 8, 0, 256, 16, 0, 1, 64],
    ],
    ids=["uneven", "empty_expert"],
)
def test_moe_gemm_fp8_nt_groupwise_irregular(m_per_expert_list):
    skip_if_not_sm120()
    a, b, a_scale, b_scale, m_indptr, ref = make_inputs(m_per_expert_list, 4096, 7168)
    out = moe_gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, m_indptr)
    diff = calc_diff(out.float(), ref.float())
    assert diff < CALC_DIFF_THRESHOLD, f"calc_diff={diff:.6e}"


@pytest.mark.parametrize(
    "bad_input",
    [
        "granularity",
        "scale_major_mode",
        "backend",
        "m_indptr",
    ],
)
def test_moe_gemm_fp8_nt_groupwise_rejects_bad_input(bad_input):
    skip_if_not_sm120()
    a, b, a_scale, b_scale, m_indptr, _ = make_inputs([8] * 4, 256, 512)
    kwargs = {}
    if bad_input == "granularity":
        kwargs["scale_granularity_mnk"] = (1, 1, 128)
        exc = ValueError
    elif bad_input == "scale_major_mode":
        kwargs["scale_major_mode"] = "K"
        exc = ValueError
    elif bad_input == "backend":
        kwargs["backend"] = "cutlass"
        exc = NotImplementedError
    else:
        m_indptr = m_indptr[:-1]
        exc = ValueError
    with pytest.raises(exc):
        moe_gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, m_indptr, **kwargs)


@pytest.mark.parametrize("bad_scale", ["a_kb", "a_m", "b_expert", "b_kb", "b_n"])
def test_moe_gemm_fp8_nt_groupwise_rejects_bad_scale_shape(bad_scale):
    skip_if_not_sm120()
    # Non-square Kb != Nb so per-dimension shape damage is detectable.
    a, b, a_scale, b_scale, m_indptr, _ = make_inputs([8] * 4, 256, 512)

    expected_message = "a_scale must have zero-padding shape"
    if bad_scale == "a_kb":
        a_scale = a_scale[:-1, :].contiguous()
    elif bad_scale == "a_m":
        a_scale = a_scale[:, :-1].contiguous()
    elif bad_scale == "b_expert":
        b_scale = b_scale[:-1, :, :].contiguous()
        expected_message = "b_scale must have shape"
    elif bad_scale == "b_kb":
        b_scale = b_scale[:, :-1, :].contiguous()
        expected_message = "b_scale must have shape"
    else:
        b_scale = b_scale[:, :, :-1].contiguous()
        expected_message = "b_scale must have shape"

    with pytest.raises(Exception, match=expected_message):
        moe_gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, m_indptr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
