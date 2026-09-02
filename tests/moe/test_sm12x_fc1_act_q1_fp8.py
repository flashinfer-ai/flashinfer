"""cute_dsl_sm12x_fc1_act_q1_fp8: wiring smoke test (shape/dtype/no-NaN)."""

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


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * 3) // 4 * 4


def per_token_cast_to_fp8_for_moe_gemm(x, token_offset):
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


def make_gated_inputs(m_per_expert_list, n, k):
    torch.random.manual_seed(0)
    num_experts = len(m_per_expert_list)
    offsets = [0]
    for m_pe in m_per_expert_list:
        offsets.append(offsets[-1] + m_pe)
    total_rows = offsets[-1]
    a = torch.randn((total_rows, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_experts, 2 * n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    a_fp8, a_scale = per_token_cast_to_fp8_for_moe_gemm(a, m_indptr)
    b_fp8_list, b_sf_list = [], []
    for i in range(num_experts):
        b_i_fp8, b_i_sf = per_block_cast_to_fp8(b[i])
        b_fp8_list.append(b_i_fp8)
        b_sf_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_sf_list, dim=0).transpose(-1, -2).contiguous()
    return a_fp8, b_fp8, a_scale, b_scale, m_indptr


def test_cute_dsl_sm12x_fc1_act_q1_fp8_smoke():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_q1_fp8

    a, b, a_scale, b_scale, m_indptr = make_gated_inputs([64, 64, 64, 64], 512, 512)
    q, sf = cute_dsl_sm12x_fc1_act_q1_fp8(a, a_scale, b, b_scale, m_indptr)
    assert q.dtype == torch.float8_e4m3fn
    assert not torch.isnan(q.float()).any()
    assert sf.numel() > 0
