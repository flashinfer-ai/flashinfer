"""cute_dsl_sm12x_moe_gemm_fp8: correctness against a pure-torch reference."""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
from flashinfer.utils import is_sm120a_supported

pytestmark = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="cute_dsl not available"
)


def skip_if_not_sm120():
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum().item()
    return 0.0 if denom == 0 else 1.0 - 2.0 * (x * y).sum().item() / denom


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * 3) // 4 * 4


def per_token_cast_to_fp8_for_moe_gemm(x, token_offset):
    """Same MN-major padded-scale repacking as tests/grouped_mm/test_cute_sm120_fp8.py."""
    token_num = x.shape[0]
    num_experts = token_offset.numel() - 1
    x_fp8, sf = per_token_cast_to_fp8(x)
    scale_k = sf.size(1)
    m_padded = compute_padded_offset(token_num, num_experts)
    padded = torch.zeros((scale_k, m_padded), dtype=torch.float32, device=x.device)
    for i in range(num_experts):
        start, end = int(token_offset[i].item()), int(token_offset[i + 1].item())
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


def test_cute_dsl_sm12x_moe_gemm_fp8_matches_reference():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_moe_gemm_fp8

    a, b, a_scale, b_scale, m_indptr, ref = make_inputs([64, 64, 64, 64], 512, 512)
    out = cute_dsl_sm12x_moe_gemm_fp8(a, a_scale, b, b_scale, m_indptr)
    diff = calc_diff(out.float(), ref.float())
    assert diff < 1e-3, f"calc_diff={diff:.6e}"
