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

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import (
    is_sm100a_supported,
    is_sm110a_supported,
    is_sm12x_supported,
)


def _is_nvfp4_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    return (
        is_sm100a_supported(device)
        or is_sm110a_supported(device)
        or is_sm12x_supported(device)
    )


pytestmark = [
    pytest.mark.skipif(
        not _is_nvfp4_supported(),
        reason="per-token NVFP4 CuTe-DSL quantizer requires Blackwell",
    ),
    pytest.mark.skipif(not is_cute_dsl_available(), reason="CuteDSL not available"),
]


def _exact_tile_amax(input: torch.Tensor, num_tiles: int) -> torch.Tensor:
    """Build exact per-tile maxima from values already rounded to input.dtype."""
    return torch.stack(
        [chunk.abs().float().amax(dim=1) for chunk in input.tensor_split(num_tiles, 1)],
        dim=1,
    ).contiguous()


@pytest.mark.parametrize("deterministic_quant", [False, True])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    "sf_layout",
    [
        pytest.param(0, id="128x4"),
        pytest.param(1, id="8x4"),
        pytest.param(2, id="linear"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_per_token_input_amax_is_bitwise_equal(
    monkeypatch, dtype, sf_layout, enable_pdl, deterministic_quant
):
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "1" if deterministic_quant else "0")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", "1")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_ERR_MODE", "MSE")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1")
    monkeypatch.setenv(
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
        "1" if deterministic_quant else "0",
    )

    torch.manual_seed(0)
    input = torch.randn(17, 256, device="cuda", dtype=torch.float32).to(dtype)
    input[0].zero_()
    input[1, 0] = 31.0
    input[2, -1] = -29.0
    global_scale_inv = torch.tensor([1.0 / 6.0], device="cuda", dtype=torch.float32)

    expected = nvfp4_quantize_per_token_cute_dsl(
        input,
        global_scale_inv,
        sf_layout=sf_layout,
        enable_pdl=enable_pdl,
    )

    # Both widths use the same presence-specialized compiled kernel. The second
    # width also exercises the grid-stride aux reduction past 128 CTA threads.
    for num_tiles in (7, 129):
        input_amax = _exact_tile_amax(input, num_tiles)
        actual = nvfp4_quantize_per_token_cute_dsl(
            input,
            global_scale_inv,
            sf_layout=sf_layout,
            enable_pdl=enable_pdl,
            input_amax=input_amax,
        )
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize(
    ("make_input_amax", "match"),
    [
        (lambda: [1.0], "torch.Tensor"),
        (
            lambda: torch.ones((4, 2), device="cuda", dtype=torch.float16),
            "torch.float32",
        ),
        (lambda: torch.ones((4, 2), dtype=torch.float32), "CUDA device"),
        (
            lambda: torch.ones((8,), device="cuda", dtype=torch.float32),
            "must be 2-D",
        ),
        (
            lambda: torch.ones((3, 2), device="cuda", dtype=torch.float32),
            "one row per input row",
        ),
        (
            lambda: torch.empty((4, 0), device="cuda", dtype=torch.float32),
            "must be nonempty",
        ),
        (
            lambda: torch.ones((2, 4), device="cuda", dtype=torch.float32).T,
            "must be contiguous",
        ),
    ],
)
def test_nvfp4_per_token_input_amax_validation(make_input_amax, match):
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    input = torch.ones((4, 32), device="cuda", dtype=torch.bfloat16)
    global_scale_inv = torch.ones((1,), device="cuda", dtype=torch.float32)

    with pytest.raises(AssertionError, match=match):
        nvfp4_quantize_per_token_cute_dsl(
            input,
            global_scale_inv,
            enable_pdl=False,
            input_amax=make_input_amax(),
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_nvfp4_per_token_input_amax_requires_same_device():
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    input = torch.ones((4, 32), device="cuda:0", dtype=torch.bfloat16)
    global_scale_inv = torch.ones((1,), device="cuda:0", dtype=torch.float32)
    input_amax = torch.ones((4, 2), device="cuda:1", dtype=torch.float32)

    with pytest.raises(AssertionError, match="same device"):
        nvfp4_quantize_per_token_cute_dsl(
            input,
            global_scale_inv,
            enable_pdl=False,
            input_amax=input_amax,
        )
