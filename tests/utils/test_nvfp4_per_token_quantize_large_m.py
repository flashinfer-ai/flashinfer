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

Regression test for row addressing in the per-token NVFP4 CuTe-DSL
quantizer at M large enough that a row's element offset (row * K)
exceeds 2**31 - 1.

The fused-MoE per-token path quantizes the GEMM1 intermediate of shape
(max_num_permuted_tokens, intermediate_size). For
NVIDIA-Nemotron-3-Super-120B (intermediate 2688, 32768 tokens, top_k 22,
512 experts) the 256-row tactics produce M=851456, whose upper rows have
element offsets past 2**31. Row slicing that folds row * K in Int32
wraps negative there and faults with cudaErrorIllegalAddress.
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available

K = 2688
# First row whose element offset row * K exceeds 2**31 - 1 is
# ceil(2**31 / 2688) = 798916. The two sizes bracket it: every row of the
# first fits in Int32 even including the in-row offset; the second has 124
# rows past the boundary.
M_BELOW_INT32_BOUNDARY = 798848
M_ABOVE_INT32_BOUNDARY = 799040

REQUIRED_FREE_BYTES = 8 * 1024**3


def _enough_gpu_memory() -> bool:
    if not torch.cuda.is_available():
        return False
    free, _total = torch.cuda.mem_get_info()
    return free >= REQUIRED_FREE_BYTES


@pytest.mark.skipif(not is_cute_dsl_available(), reason="CuteDSL not available")
@pytest.mark.skipif(
    not _enough_gpu_memory(),
    reason=f"requires >= {REQUIRED_FREE_BYTES / 2**30:.0f} GiB free GPU memory",
)
@pytest.mark.parametrize("rows", [M_BELOW_INT32_BOUNDARY, M_ABOVE_INT32_BOUNDARY])
@pytest.mark.parametrize("enable_pdl", [False, None])
def test_nvfp4_per_token_quantize_row_offset_past_int32(rows, enable_pdl):
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        SF_LAYOUT_128x4,
        nvfp4_quantize_per_token_cute_dsl,
    )

    device = torch.device("cuda")
    # Fill each row with a per-row constant so the returned per-token scale
    # (row amax with a global scale of 1.0) identifies the row that was
    # actually read. A wrapped row address either faults or fails the
    # exact-scale comparison below.
    row_values = (1.0 + (torch.arange(rows, device=device) % 7)).float()
    x = row_values.to(torch.bfloat16).unsqueeze(1).expand(rows, K).contiguous()
    global_scale_inv = torch.ones(1, dtype=torch.float32, device=device)

    fp4, sf, per_token_scale = nvfp4_quantize_per_token_cute_dsl(
        x, global_scale_inv, sf_layout=SF_LAYOUT_128x4, enable_pdl=enable_pdl
    )
    torch.cuda.synchronize()

    assert fp4.shape == (rows, K // 2)
    torch.testing.assert_close(per_token_scale, row_values, rtol=0.0, atol=0.0)
    # Every element of a constant row equals the row amax, so each packed
    # byte must decode to a pair of E2M1 6.0 values (0x77), including in the
    # rows whose element offsets exceed 2**31 - 1.
    tail = fp4[-64:]
    assert bool((tail == 0x77).all().item())
