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
"""

import math

import pytest
import torch
import torch.nn.functional as F

from flashinfer.grouped_mm import (
    grouped_mm_mxfp8_nt_groupwise_zero_padding,
    quantize_mxfp8_for_zero_padding,
)
from flashinfer.quantization import (
    mxfp8_dequantize_per_token,
    mxfp8_quantize_per_block,
    mxfp8_quantize_per_token,
    mxfp8_transform_sf_layout,
)
from flashinfer.utils import is_sm120a_supported

COS_SIM_THRESHOLD = 0.99


def _skip_if_not_sm120():
    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip(
            "MXFP8 cute grouped GEMM zero_padding requires SM120 (RTX PRO 6000 Blackwell)."
        )


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


@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("rows_per_group", [64, 128])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
@pytest.mark.parametrize("k_gran", [32, 128])
def test_grouped_mm_mxfp8_nt_groupwise_zero_padding(
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

    a_fp8, a_sf = quantize_mxfp8_for_zero_padding(a, m_indptr, gran_k=k_gran)

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

    out = grouped_mm_mxfp8_nt_groupwise_zero_padding(
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
    assert cos_sim > COS_SIM_THRESHOLD, f"cos_sim={cos_sim:.4f} < {COS_SIM_THRESHOLD}"


if __name__ == "__main__":
    pytest.main([__file__])
