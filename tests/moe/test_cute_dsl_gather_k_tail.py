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

"""Regression coverage for partial K tiles in Blackwell gather GEMM1."""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import get_compute_capability


@pytest.mark.skipif(not is_cute_dsl_available(), reason="CuTe DSL is not available")
@pytest.mark.parametrize("k", [128, 256, 384, 512])
@pytest.mark.parametrize("num_tokens", [1, 17])
@pytest.mark.parametrize("tile_m", [128, 256])
def test_gather_gemm_k_tail(k, num_tokens, tile_m):
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("Requires Blackwell SM100 or SM103")

    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion,
    )

    device = "cuda"
    n = 256
    # Two FP4 ones per byte. An extra row keeps the old kernel's tail reads
    # inside allocated storage, so a failure cannot poison the CUDA context.
    a_storage = torch.full(
        (num_tokens + 1, k // 2), 0x22, dtype=torch.uint8, device=device
    )
    a = a_storage[:num_tokens]
    # E4M3 1.0 = 0x38; 0x7f is NaN. Reading the extra scale row must not
    # contaminate the last real row, even though the weight K tail is zero.
    scale_storage = torch.full(
        (num_tokens + 1, k // 16), 0x7F, dtype=torch.uint8, device=device
    )
    a_scale = scale_storage[:num_tokens]
    a_scale.fill_(0x38)
    b = torch.full((1, n, k // 2), 0x22, dtype=torch.uint8, device=device)
    b_scale = torch.full(
        (32, 4, n // 128, 4, k // 64, 1), 0x38, dtype=torch.uint8, device=device
    )
    mapping = torch.full((tile_m,), -1, dtype=torch.int32, device=device)
    mapping[:num_tokens] = torch.arange(num_tokens - 1, -1, -1, device=device)

    output, _ = blockscaled_contiguous_gather_grouped_gemm_act_fusion(
        a=a,
        b=b,
        a_scale=a_scale,
        b_scale=b_scale,
        alpha=torch.full((1,), 1.0 / k, device=device),
        tile_idx_to_expert_idx=torch.zeros(1, dtype=torch.int32, device=device),
        tile_idx_to_mn_limit=torch.tensor(
            [num_tokens], dtype=torch.int32, device=device
        ),
        token_id_mapping=mapping,
        num_non_exiting_tiles=torch.ones(1, dtype=torch.int32, device=device),
        topk=1,
        c_dtype="bfloat16",
        mma_tiler_mn=(tile_m, 128),
        cluster_shape_mn=(tile_m // 128, 1),
        enable_pdl=False,
    )
    # Both projections are dot(ones, ones) / K = 1, hence SwiGLU = sigmoid(1).
    expected = torch.full_like(
        output[:num_tokens], torch.sigmoid(torch.tensor(1.0)).item()
    )
    torch.testing.assert_close(output[:num_tokens], expected, rtol=0, atol=0)
