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

import math

import pytest
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.gemm import (
    group_gemm_nvfp4_nt_groupwise,
)
from flashinfer.utils import get_compute_capability


def gemm_nvfp4_nt_groupwise_ref(
    a_float: torch.Tensor,
    b_float: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out_ref = torch.empty(
        (a_float.shape[0] * a_float.shape[1], b_float.shape[1]),
        dtype=out_dtype,
        device=a_float.device,
    )
    for i in range(a_float.shape[0]):
        out_ref[i * a_float.shape[1] : (i + 1) * a_float.shape[1]] = torch.mm(
            a_float[i].float(), b_float[i].float().T
        ).to(out_dtype)
    return out_ref


def _quantize_nvfp4_group_inputs(
    a_float: torch.Tensor,
    b_float: torch.Tensor,
    m_indptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a_fp4_chunks = []
    a_scale_chunks = []
    b_fp4_chunks = []
    b_scale_chunks = []
    alpha_chunks = []
    for a_group, b_group in zip(a_float, b_float, strict=True):
        a_global_sf = (448 * 6) / a_group.float().abs().nan_to_num().max()
        b_global_sf = (448 * 6) / b_group.float().abs().nan_to_num().max()
        a_fp4_group, a_scale_group = nvfp4_quantize(
            a_group, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
        b_fp4_group, b_scale_group = nvfp4_quantize(
            b_group, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
        a_fp4_chunks.append(a_fp4_group)
        a_scale_chunks.append(a_scale_group)
        b_fp4_chunks.append(b_fp4_group)
        b_scale_chunks.append(b_scale_group)
        alpha_chunks.append(1.0 / (a_global_sf * b_global_sf))

    return (
        torch.cat(a_fp4_chunks, dim=0),
        torch.stack(b_fp4_chunks, dim=0),
        torch.cat(a_scale_chunks, dim=0),
        torch.stack(b_scale_chunks, dim=0),
        torch.tensor(alpha_chunks, dtype=torch.float32, device=a_float.device),
    )


@pytest.mark.parametrize("m", [4, 128, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("group_size", [1, 2, 4])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_group_gemm_nvfp4(
    m: int,
    n: int,
    k: int,
    group_size: int,
    out_dtype: torch.dtype,
):
    device = torch.device("cuda")
    compute_capability = get_compute_capability(device)
    if compute_capability[0] not in [12]:
        pytest.skip(
            "group_gemm_nvfp4_nt_groupwise is only supported on SM120/SM121 GPUs."
        )

    torch.random.manual_seed(0)
    a_float = torch.randn((group_size, m, k), dtype=torch.bfloat16, device=device)
    b_float = torch.randn((group_size, n, k), dtype=torch.bfloat16, device=device)
    m_indptr = torch.arange(
        0, (group_size + 1) * m, m, dtype=torch.int32, device=device
    )

    a_fp4, b_fp4, a_scale, b_scale, alpha = _quantize_nvfp4_group_inputs(a_float, b_float)
    out_ref = gemm_nvfp4_nt_groupwise_ref(a_float, b_float, out_dtype)

    for tile_k in [128, 256]:
        out = group_gemm_nvfp4_nt_groupwise(
            a_fp4,
            b_fp4,
            a_scale,
            b_scale,
            m_indptr,
            alpha,
            tile_m=128,
            tile_n=128,
            tile_k=tile_k,
            out_dtype=out_dtype,
        )
        cos_sim = F.cosine_similarity(
            out_ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
        )
        assert cos_sim > 0.97


if __name__ == "__main__":
    pytest.main([__file__])
