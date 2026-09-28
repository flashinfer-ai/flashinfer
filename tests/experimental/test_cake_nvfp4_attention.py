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

import pytest
import torch

from flashinfer.prefill import prepare_nvfp4_attention


@pytest.mark.parametrize(
    "batch,heads,seqlen", [(4, 8, 4096), (1, 8, 32768), (8, 32, 8192)]
)
def test_nvfp4_attention(batch, heads, seqlen):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 required")
    torch.manual_seed(42)
    q = torch.randn((batch, heads, seqlen, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    attention = prepare_nvfp4_attention(q, k, v, out, backend="cake")
    assert attention() is out
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=1.0, rtol=0.1)
    snapshot = out.clone()
    out.zero_()
    assert attention() is out
    torch.testing.assert_close(out, snapshot, atol=0, rtol=0)


def test_nvfp4_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        prepare_nvfp4_attention(None, None, None, None, backend="unknown")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nvfp4_quantization_saturates_finite_scales(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flashinfer.experimental.nvfp4_attention.cake_backend import _quantize_nvfp4

    largest = torch.finfo(torch.bfloat16).max
    values = torch.tensor(
        [0.0, 1.0, 2688.0, 4096.0, -4096.0, largest, -largest],
        dtype=torch.bfloat16,
        device=device,
    )
    x = values[:, None].expand(-1, 16).contiguous()
    packed, scale_bytes = _quantize_nvfp4(x)
    scales = scale_bytes.view(torch.float8_e4m3fn).float()
    expected_scales = torch.tensor(
        [2.0**-9, 0.171875, 448.0, 448.0, 448.0, 448.0, 448.0], device=device
    )[:, None]
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    expected_packed = torch.tensor(
        [0x00, 0x77, 0x77, 0x77, 0xFF, 0x77, 0xFF],
        dtype=torch.uint8,
        device=device,
    )[:, None].expand(-1, 8)
    torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)
