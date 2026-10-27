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

import gc

import pytest
import torch

from flashinfer import (
    nvfp4_attention_sm120_fwd,
    nvfp4_attention_sm120_quantize_qkv,
)
from flashinfer.utils import is_sm120a_supported, is_sm121a_supported

# The compact correction alone takes 8-21 GiB in these cases.
pytestmark = pytest.mark.gpu_exclusive


@pytest.fixture(autouse=True)
def release_test_memory():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "seq_len,batch,causal,dtype,return_lse",
    [
        pytest.param(
            262016, 1, False, torch.bfloat16, False, id="below-int32-boundary"
        ),
        pytest.param(
            262144, 2, False, torch.bfloat16, True, id="at-int32-boundary-batch2-lse"
        ),
        pytest.param(
            297618, 1, False, torch.bfloat16, False, id="unpadded-bf16-noncausal"
        ),
        pytest.param(
            297618, 2, True, torch.float16, True, id="unpadded-fp16-batch2-causal-lse"
        ),
    ],
)
@torch.inference_mode()
def test_large_correction_matches_independent_heads(
    seq_len, batch, causal, dtype, return_lse
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if not (is_sm120a_supported(device) or is_sm121a_supported(device)):
        pytest.skip("SM120 or SM121 GPU is required")

    num_heads, head_dim = 4, 128
    padded_len = (seq_len + 127) // 128 * 128
    correction_bytes = batch * num_heads * (padded_len // 128) * padded_len * 4
    # Include quantization temporaries and the output comparisons in the budget.
    if torch.cuda.mem_get_info()[0] < 2 * correction_bytes + 6 * 2**30:
        pytest.skip("Insufficient free memory for the large correction regression")

    torch.manual_seed(26)
    q, k, v = [
        torch.randn((batch, num_heads, seq_len, head_dim), device=device, dtype=dtype)
        for _ in range(3)
    ]
    packed = nvfp4_attention_sm120_quantize_qkv(q, k, v, per_block_mean=True)
    del q, k, v
    batch_stride = num_heads * (padded_len // 128) * padded_len
    assert packed[-1].stride(0) == batch_stride
    assert (batch_stride >= 2**31) == (padded_len >= 262144)

    kwargs = dict(
        sm_scale=head_dim**-0.5,
        causal=causal,
        per_block_mean=True,
        unpadded_k_len=seq_len,
        return_lse=return_lse,
        out_dtype=dtype,
    )
    result = nvfp4_attention_sm120_fwd(*packed, **kwargs)
    out, lse = result if return_lse else (result, None)
    torch.cuda.synchronize()

    for bi in range(batch):
        for hi in range(num_heads):
            # Reuse the same quantized buffers, so the oracle changes only the
            # layout: each single-head launch has correction strides below 2^31.
            independent = tuple(
                x[bi : bi + 1, hi : hi + 1].contiguous() for x in packed
            )
            ref = nvfp4_attention_sm120_fwd(*independent, **kwargs)
            ref_out, ref_lse = ref if return_lse else (ref, None)
            torch.testing.assert_close(
                out[bi : bi + 1, hi : hi + 1], ref_out, rtol=0, atol=0
            )
            if return_lse:
                torch.testing.assert_close(
                    lse[bi : bi + 1, hi : hi + 1], ref_lse, rtol=0, atol=0
                )
            for start in range(0, seq_len, 4096):
                end = min(start + 4096, seq_len)
                assert torch.isfinite(ref_out[:, :, start:end]).all()
                if return_lse:
                    assert torch.isfinite(ref_lse[:, :, start:end]).all()
