"""
Copyright (c) 2023 by FlashInfer team.

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

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("kv_len", [7, 19, 39, 1170, 39275])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])#[64, 128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])#["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])#, "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
def test_single_decode_fp8_calibration_scale(
    kv_len, num_kv_heads, num_qo_heads, head_dim, kv_layout, pos_encoding_mode, fp8_dtype,
):
    torch.manual_seed(42)
    q = torch.randn(num_qo_heads, head_dim, dtype=torch.float16).to(0)
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16).to(0) if kv_layout == "NHD" else torch.randn(num_kv_heads, kv_len, head_dim).to(0)
    v = 0.1 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16).to(0) if kv_layout == "NHD" else 0.1 * torch.randn(num_kv_heads, kv_len, head_dim).to(0)

    o_fp16 = flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout=kv_layout, pos_encoding_mode=pos_encoding_mode
    )

    q_scale = q.amax().item() / 256
    k_scale = k.amax().item() / 256
    v_scale = v.amax().item() / 256
    q_fp8 = (q / q_scale).to(fp8_dtype)
    k_fp8 = (k / k_scale).to(fp8_dtype)
    v_fp8 = (v / v_scale).to(fp8_dtype)

    o_fp8 = flashinfer.single_decode_with_kv_cache(
        q_fp8, k_fp8, v_fp8, kv_layout=kv_layout, pos_encoding_mode=pos_encoding_mode,
        q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
    )

    np.testing.assert_allclose(o_fp16.cpu().numpy(), o_fp8.cpu().numpy(), atol=1e-2, rtol=2e-2)


if __name__ == "__main__":
    test_single_decode_fp8_calibration_scale(1170, 4, 32, 128, "NHD", "NONE", torch.float8_e4m3fn)
