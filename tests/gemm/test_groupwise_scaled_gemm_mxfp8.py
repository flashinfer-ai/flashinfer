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

Tests for MXFP8 groupwise-scaled GEMM family (cute backend, SM120):

- `gemm_mxfp8_nt_groupwise`                 — dense Normal
- `batch_gemm_mxfp8_nt_groupwise`           — dense Batched
- `group_gemm_mxfp8_nt_groupwise`           — MoE contiguous (psum_layout True/False)
- `group_gemm_mxfp8_nt_groupwise_masked`    — MoE masked
- `group_gemm_mxfp8_nt_groupwise_zero_padding` — MoE zero-padded (uses dedicated quant helper)

Reference: dequant-self matmul (a_dequant @ b_dequant.T), accuracy threshold
`cosine_similarity > 0.99` (mirrors `test_mm_mxfp8_sm120.py` / `test_bmm_mxfp8.py`
convention for MXFP8 family — UE8M0 block quantization is more aggressive than
DG fp8 so element-wise tolerance is too tight; cosine similarity matches the
overall accuracy signal).
"""

import math

import pytest
import torch
import torch.nn.functional as F
from einops import einsum

from flashinfer.gemm import (
    batch_gemm_mxfp8_nt_groupwise,
    gemm_mxfp8_nt_groupwise,
    group_gemm_mxfp8_nt_groupwise,
    group_gemm_mxfp8_nt_groupwise_masked,
    group_gemm_mxfp8_nt_groupwise_zero_padding,
    quantize_mxfp8_for_zero_padding,
)
from flashinfer.quantization import (
    mxfp8_dequantize_per_token,
    mxfp8_quantize_per_block,
    mxfp8_quantize_per_token,
    mxfp8_transform_sf_layout,
)
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm120():
    cc = get_compute_capability(torch.device(device="cuda"))
    if cc[0] != 12:
        pytest.skip(
            "MXFP8 cute GEMM family is only supported on SM120 (RTX PRO 6000 Blackwell)."
        )


def _quant_a_per_token(a, k_gran, masked_m=None):
    """Return (a_fp8, a_sf_packed, a_dequant) where a_dequant reverses quantization."""
    a_fp8, a_sf = mxfp8_quantize_per_token(a, masked_m=masked_m, k_gran=k_gran)
    a_dequant = mxfp8_dequantize_per_token(a_fp8, a_sf, k_gran=k_gran, dtype=a.dtype)
    return a_fp8, a_sf, a_dequant


def _quant_b_per_block(b, k_gran, num_groups=None):
    """Return (b_fp8, b_sf_packed, b_dequant)."""
    b_fp8, b_sf_fp32 = mxfp8_quantize_per_block(b, k_gran=k_gran)
    n = b.size(-2)
    k = b.size(-1)
    b_sf = mxfp8_transform_sf_layout(
        b_sf_fp32,
        mn=n,
        k=k,
        recipe=(k_gran, k_gran),
        num_groups=num_groups,
        is_sfa=False,
    )
    b_dequant = mxfp8_dequantize_per_token(b_fp8, b_sf, k_gran=k_gran, dtype=b.dtype)
    return b_fp8, b_sf, b_dequant


# ───── Dense Normal ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("m", [128, 512, 4096])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
@pytest.mark.parametrize("k_gran", [32, 128])
def test_mxfp8_gemm_groupwise(m, n, k, k_gran):
    _skip_if_not_sm120()
    torch.random.manual_seed(0)
    out_dtype = torch.bfloat16

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") / math.sqrt(k)

    a_fp8, a_sf, a_deq = _quant_a_per_token(a, k_gran=k_gran)
    b_fp8, b_sf, b_deq = _quant_b_per_block(b, k_gran=k_gran)
    ref = einsum(a_deq, b_deq, "m k, n k -> m n").to(out_dtype)

    out = gemm_mxfp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_sf,
        b_sf,
        scale_granularity_mnk=(1, 1, k_gran),
        out_dtype=out_dtype,
    )
    cos_sim = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99"


# ───── Dense Batched ────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("m", [128, 512])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
@pytest.mark.parametrize("k_gran", [32, 128])
def test_mxfp8_batch_gemm_groupwise(num_groups, m, n, k, k_gran):
    _skip_if_not_sm120()
    torch.random.manual_seed(0)
    out_dtype = torch.bfloat16

    a = torch.randn((num_groups, m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_groups, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)

    a_fp8, a_sf, a_deq = _quant_a_per_token(a, k_gran=k_gran)
    b_fp8, b_sf, b_deq = _quant_b_per_block(b, k_gran=k_gran, num_groups=num_groups)
    ref = einsum(a_deq, b_deq, "g m k, g n k -> g m n").to(out_dtype)

    out = batch_gemm_mxfp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_sf,
        b_sf,
        scale_granularity_mnk=(1, 1, k_gran),
        out_dtype=out_dtype,
    )
    cos_sim = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99"


# ───── MoE contiguous (psum_layout=True) ────────────────────────────────────


@pytest.mark.parametrize("num_groups", [2, 4, 8])
@pytest.mark.parametrize("m_per_group", [128, 256])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
@pytest.mark.parametrize("k_gran", [32, 128])
def test_mxfp8_group_gemm_groupwise_psum_layout(num_groups, m_per_group, n, k, k_gran):
    _skip_if_not_sm120()
    torch.random.manual_seed(0)
    out_dtype = torch.bfloat16
    m = m_per_group * num_groups

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_groups, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indices = torch.tensor(
        [(i + 1) * m_per_group for i in range(num_groups)],
        dtype=torch.int32,
        device="cuda",
    )

    a_fp8, a_sf, a_deq = _quant_a_per_token(a, k_gran=k_gran)
    b_fp8, b_sf, b_deq = _quant_b_per_block(b, k_gran=k_gran, num_groups=num_groups)

    ref = torch.zeros(m, n, dtype=out_dtype, device="cuda")
    for j in range(num_groups):
        ref[j * m_per_group : (j + 1) * m_per_group] = (
            a_deq[j * m_per_group : (j + 1) * m_per_group] @ b_deq[j].t()
        ).to(out_dtype)

    out = group_gemm_mxfp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_sf,
        b_sf,
        m_indices,
        scale_granularity_mnk=(1, 1, k_gran),
        use_psum_layout=True,
        out_dtype=out_dtype,
    )
    cos_sim = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99"


# ───── MoE contiguous (psum_layout=False, DG canonical, BLOCK_M=128) ────────


@pytest.mark.parametrize("num_groups", [2, 4, 8])
@pytest.mark.parametrize("m_per_group", [128, 256])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
@pytest.mark.parametrize("k_gran", [32, 128])
def test_mxfp8_group_gemm_groupwise_mgroup_contiguous(
    num_groups, m_per_group, n, k, k_gran
):
    _skip_if_not_sm120()
    torch.random.manual_seed(0)
    out_dtype = torch.bfloat16
    m = m_per_group * num_groups  # Each group already BLOCK_M=128 aligned

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_groups, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    # m_indices: per-row group assignment, DG canonical layout
    m_indices = torch.empty(m, dtype=torch.int32, device="cuda")
    for j in range(num_groups):
        m_indices[j * m_per_group : (j + 1) * m_per_group] = j

    a_fp8, a_sf, a_deq = _quant_a_per_token(a, k_gran=k_gran)
    b_fp8, b_sf, b_deq = _quant_b_per_block(b, k_gran=k_gran, num_groups=num_groups)

    ref = torch.zeros(m, n, dtype=out_dtype, device="cuda")
    for j in range(num_groups):
        ref[j * m_per_group : (j + 1) * m_per_group] = (
            a_deq[j * m_per_group : (j + 1) * m_per_group] @ b_deq[j].t()
        ).to(out_dtype)

    out = group_gemm_mxfp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_sf,
        b_sf,
        m_indices,
        scale_granularity_mnk=(1, 1, k_gran),
        use_psum_layout=False,
        out_dtype=out_dtype,
    )
    cos_sim = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99"


# ───── MoE masked ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_groups", [2, 4, 8])
@pytest.mark.parametrize("max_m", [128, 4096])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
@pytest.mark.parametrize("k_gran", [32, 128])
def test_mxfp8_group_gemm_groupwise_masked(num_groups, max_m, n, k, k_gran):
    _skip_if_not_sm120()
    torch.random.manual_seed(0)
    out_dtype = torch.bfloat16

    a = torch.randn((num_groups, max_m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_groups, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    masked_m = torch.full((num_groups,), max_m // 2, dtype=torch.int32, device="cuda")

    a_fp8, a_sf, a_deq = _quant_a_per_token(a, masked_m=masked_m, k_gran=k_gran)
    b_fp8, b_sf, b_deq = _quant_b_per_block(b, k_gran=k_gran, num_groups=num_groups)

    ref = einsum(a_deq, b_deq, "g m k, g n k -> g m n").to(out_dtype)

    out = group_gemm_mxfp8_nt_groupwise_masked(
        a_fp8,
        b_fp8,
        a_sf,
        b_sf,
        masked_m,
        scale_granularity_mnk=(1, 1, k_gran),
        out_dtype=out_dtype,
    )
    for j in range(num_groups):
        m_valid = int(masked_m[j].item())
        cos_sim = F.cosine_similarity(
            out[j, :m_valid].reshape(-1).float(),
            ref[j, :m_valid].reshape(-1).float(),
            dim=0,
        ).item()
        assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99 at group {j}"


# ───── MoE zero_padding (uses dedicated CUDA quant helper for A) ────────────


@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("rows_per_group", [64, 128])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
@pytest.mark.parametrize("k_gran", [32, 128])
def test_mxfp8_group_gemm_groupwise_zero_padding(
    num_groups, rows_per_group, n, k, k_gran
):
    _skip_if_not_sm120()
    torch.random.manual_seed(0)
    out_dtype = torch.bfloat16
    token_num = rows_per_group * num_groups

    a = torch.randn((token_num, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_groups, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(
        [i * rows_per_group for i in range(num_groups + 1)],
        dtype=torch.int32,
        device="cuda",
    )

    # A path uses the dedicated CUDA helper for kernel call (zero_padding-specific
    # padded sf layout with per-expert 3-row gaps for TMA alignment).
    a_fp8, a_sf = quantize_mxfp8_for_zero_padding(a, m_indptr, gran_k=k_gran)

    # For the dequant reference: independently compute per-expert per-token quant
    # via mxfp8_quantize_per_token (no padded layout, just per-row sf). UE8M0 ceil
    # is deterministic so per-row fp8/sf values match the kernel's input; only sf
    # storage layout differs. Cos-sim is robust to this layout-only difference.
    a_deq = torch.zeros_like(a)
    for j in range(num_groups):
        start = int(m_indptr[j].item())
        end = int(m_indptr[j + 1].item())
        a_j_fp8, a_j_sf = mxfp8_quantize_per_token(a[start:end], k_gran=k_gran)
        a_deq[start:end] = mxfp8_dequantize_per_token(
            a_j_fp8,
            a_j_sf,
            k_gran=k_gran,
            dtype=a.dtype,
        )

    b_fp8, b_sf, b_deq = _quant_b_per_block(b, k_gran=k_gran, num_groups=num_groups)

    ref = torch.zeros(token_num, n, dtype=out_dtype, device="cuda")
    for j in range(num_groups):
        start = int(m_indptr[j].item())
        end = int(m_indptr[j + 1].item())
        ref[start:end] = (a_deq[start:end] @ b_deq[j].t()).to(out_dtype)

    out = group_gemm_mxfp8_nt_groupwise_zero_padding(
        a_fp8,
        b_fp8,
        a_sf,
        b_sf,
        m_indptr,
        scale_granularity_mnk=(1, 1, k_gran),
        out_dtype=out_dtype,
    )
    cos_sim = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99"
