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

from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import (
    is_sm100a_supported,
    is_sm110a_supported,
    is_sm12x_supported,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sf_layout", list(SfLayout))
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
@pytest.mark.parametrize("k", [32, 69632])
@pytest.mark.parametrize(
    "value,block_max,decode_scale,expected_nibble",
    [
        (-1.1796875, 2.0, 0.0035109748132526875, 0xD),
        (-0.27734375, 0.9375, 0.0033017112873494625, 0xC),
    ],
)
def test_nvfp4_per_token_rounding(
    dtype,
    sf_layout,
    backend,
    k,
    value,
    block_max,
    decode_scale,
    expected_nibble,
    monkeypatch,
):
    device = torch.device("cuda")
    if not (
        is_sm100a_supported(device)
        or is_sm110a_supported(device)
        or is_sm12x_supported(device)
    ):
        pytest.skip("NVFP4 requires Blackwell or newer")
    if backend == "cute-dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")
    monkeypatch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "0")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "0")
    x = torch.zeros((2, k), dtype=dtype, device="cuda")
    x[0, 0] = value
    x[0, 1] = block_max
    x[0, 16] = 4.0
    # A reciprocal round trip shifts these values across FP4 rounding boundaries.
    q, sf, pts = nvfp4_quantize(
        x,
        decode_scale / 4.0,
        sfLayout=sf_layout,
        per_token_activation=True,
        backend=backend,
    )
    assert (q[0, 0].item() & 0xF) == expected_nibble
    torch.testing.assert_close(
        pts, torch.tensor([decode_scale, 0.0], device="cuda"), rtol=0, atol=0
    )
    assert torch.count_nonzero(q[1]).item() == 0
    assert torch.isfinite(sf.view(torch.float8_e4m3fn).float()).all()
