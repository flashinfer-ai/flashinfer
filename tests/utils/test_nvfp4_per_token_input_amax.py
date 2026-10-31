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
from flashinfer.quantization import NVFP44Over6Config
from flashinfer.quantization.nvfp4_quantization_utils import nvfp4_per_token_scale_inv
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


def _quant_config(monkeypatch, deterministic: bool) -> NVFP44Over6Config | None:
    monkeypatch.setenv(
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1" if deterministic else "0"
    )
    return (
        NVFP44Over6Config(e4m3_max=256, err_mode="MSE", err_use_fast_math=True)
        if deterministic
        else None
    )


def _exact_tile_amax(input: torch.Tensor, num_tiles: int) -> torch.Tensor:
    """Build native-width PTX maxNum tile maxima in blocked-8 layout."""
    assert num_tiles > 0
    row_major = torch.stack(
        [
            torch.where(torch.isnan(chunk), torch.zeros_like(chunk), chunk.abs())
            .float()
            .amax(dim=1)
            for chunk in input.tensor_split(num_tiles, 1)
        ],
        dim=1,
    ).to(input.dtype)
    padded = torch.zeros(
        ((input.shape[0] + 7) // 8 * 8, num_tiles),
        dtype=input.dtype,
        device=input.device,
    )
    padded[: input.shape[0]].copy_(row_major)
    return padded.reshape(-1, 8, num_tiles).permute(0, 2, 1).contiguous()


def _unpack_blocked8(input_amax: torch.Tensor, rows: int) -> torch.Tensor:
    """Return the logical [row, tile] view of a blocked-8 aux tensor."""
    return (
        input_amax.permute(0, 2, 1)
        .reshape(input_amax.shape[0] * 8, input_amax.shape[1])[:rows]
        .contiguous()
    )


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

    config = _quant_config(monkeypatch, deterministic_quant)

    torch.manual_seed(0)
    input = torch.randn(17, 256, device="cuda", dtype=torch.float32).to(dtype)
    input[0].zero_()
    input[1, 0] = 31.0
    input[2, -1] = -29.0
    global_scale_inv = torch.tensor(
        [nvfp4_per_token_scale_inv(config)], device="cuda", dtype=torch.float32
    )

    expected = nvfp4_quantize_per_token_cute_dsl(
        input,
        global_scale_inv,
        sf_layout=sf_layout,
        enable_pdl=enable_pdl,
        nvfp4_4over6=config,
    )

    # Both widths use the same presence-specialized compiled kernel. The second
    # width also exercises the half-warp's grid-stride aux reduction.
    for num_tiles in (7, 33):
        input_amax = _exact_tile_amax(input, num_tiles)
        actual = nvfp4_quantize_per_token_cute_dsl(
            input,
            global_scale_inv,
            sf_layout=sf_layout,
            enable_pdl=enable_pdl,
            nvfp4_4over6=config,
            input_amax=input_amax,
        )
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize("deterministic_quant", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_per_token_input_amax_preserves_maxnum_edge_cases(
    monkeypatch, dtype, deterministic_quant
):
    """The aux reduction must retain the legacy scan's maximumNumber contract."""
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    config = _quant_config(monkeypatch, deterministic_quant)

    input = torch.zeros((6, 256), device="cuda", dtype=dtype)
    nan = torch.tensor(float("nan"), device="cuda", dtype=dtype)
    inf = torch.tensor(float("inf"), device="cuda", dtype=dtype)
    tiny = torch.tensor(
        torch.finfo(dtype).smallest_normal / 2,
        device="cuda",
        dtype=dtype,
    )
    input[0, :2] = torch.stack((nan, inf))
    input[1, :2] = torch.stack((nan, -inf))
    input[2].fill_(nan)
    input[3, :2] = torch.tensor([-0.0, 0.0], device="cuda", dtype=dtype)
    input[4, :3] = torch.stack((nan, tiny, -tiny))
    input[5, :3] = torch.tensor([float("nan"), 3.0, -4.0], device="cuda", dtype=dtype)

    input_amax = _exact_tile_amax(input, num_tiles=4)
    row_amax = _unpack_blocked8(input_amax, input.shape[0]).float().amax(dim=1)
    assert torch.equal(
        row_amax,
        torch.stack(
            (
                inf,
                inf,
                input.new_zeros(()),
                input.new_zeros(()),
                tiny,
                input.new_tensor(4.0),
            )
        ).float(),
    )
    assert row_amax[2].view(torch.int32).item() == 0
    assert row_amax[3].view(torch.int32).item() == 0

    global_scale_inv = torch.tensor(
        [nvfp4_per_token_scale_inv(config)], device="cuda", dtype=torch.float32
    )
    legacy = nvfp4_quantize_per_token_cute_dsl(
        input,
        global_scale_inv,
        sf_layout=2,
        enable_pdl=False,
        nvfp4_4over6=config,
    )
    accelerated = nvfp4_quantize_per_token_cute_dsl(
        input,
        global_scale_inv,
        sf_layout=2,
        enable_pdl=False,
        nvfp4_4over6=config,
        input_amax=input_amax,
    )
    for accelerated_tensor, legacy_tensor in zip(accelerated, legacy, strict=True):
        assert torch.equal(
            accelerated_tensor.view(torch.uint8), legacy_tensor.view(torch.uint8)
        )


@pytest.mark.parametrize("deterministic_quant", [False, True])
@pytest.mark.parametrize("k", [256, 8192, 65552])
def test_nvfp4_per_token_input_amax_with_out_scale(monkeypatch, deterministic_quant, k):
    """Match the register-resident and streaming standalone paths, including folding."""
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    torch.manual_seed(1)
    config = _quant_config(monkeypatch, deterministic_quant)
    monkeypatch.delenv("FLASHINFER_NVFP4_PER_TOKEN_THREADS", raising=False)
    input = torch.randn(9, k, device="cuda", dtype=torch.bfloat16)
    input_amax = _exact_tile_amax(input, num_tiles=7)
    global_scale_inv = torch.tensor(
        [nvfp4_per_token_scale_inv(config)], device="cuda", dtype=torch.float32
    )
    out_scale = torch.tensor([0.3125], device="cuda", dtype=torch.float32)
    kwargs = {"sf_layout": 2, "enable_pdl": False, "nvfp4_4over6": config}
    unfolded = nvfp4_quantize_per_token_cute_dsl(input, global_scale_inv, **kwargs)
    legacy = nvfp4_quantize_per_token_cute_dsl(
        input, global_scale_inv, out_scale=out_scale, **kwargs
    )
    accelerated = nvfp4_quantize_per_token_cute_dsl(
        input,
        global_scale_inv,
        out_scale=out_scale,
        input_amax=input_amax,
        **kwargs,
    )
    for actual, expected in zip(accelerated, legacy, strict=True):
        assert torch.equal(actual, expected)
    assert torch.equal(accelerated[0], unfolded[0])
    assert torch.equal(accelerated[1], unfolded[1])
    assert torch.equal(accelerated[2], unfolded[2] * out_scale)


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_nvfp4_per_token_input_amax_respects_graph_replayed_valid_rows(enable_pdl):
    """A device bound changes under replay; inactive output rows stay untouched."""
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    torch.manual_seed(2)
    input = torch.randn(24, 256, device="cuda", dtype=torch.bfloat16)
    complete_amax = _exact_tile_amax(input, num_tiles=5)
    input_amax = complete_amax.clone()
    valid_rows = torch.tensor([8], device="cuda", dtype=torch.int32)
    global_scale_inv = torch.tensor(
        [nvfp4_per_token_scale_inv(None)], device="cuda", dtype=torch.float32
    )
    out_scale = torch.tensor([0.3125], device="cuda", dtype=torch.float32)
    kwargs = {
        "sf_layout": 2,
        "enable_pdl": enable_pdl,
        "out_scale": out_scale,
        "nvfp4_4over6": None,
    }

    legacy = nvfp4_quantize_per_token_cute_dsl(input, global_scale_inv, **kwargs)
    # Compile before capture. The captured allocations are reused on every
    # replay, allowing sentinel checks even when there are zero valid rows.
    nvfp4_quantize_per_token_cute_dsl(
        input,
        global_scale_inv,
        input_amax=input_amax,
        input_amax_valid_rows=valid_rows,
        **kwargs,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        accelerated = nvfp4_quantize_per_token_cute_dsl(
            input,
            global_scale_inv,
            input_amax=input_amax,
            input_amax_valid_rows=valid_rows,
            **kwargs,
        )

    for rows in (0, 8, 24, 16, 0):
        valid_rows.fill_(rows)
        input_amax.fill_(float("nan"))
        input_amax[: rows // 8].copy_(complete_amax[: rows // 8])
        for tensor in accelerated:
            tensor.fill_(17)
        graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(accelerated, legacy, strict=True):
            assert torch.equal(actual[:rows], expected[:rows])
            assert torch.equal(actual[rows:], torch.full_like(actual[rows:], 17))


@pytest.mark.parametrize(
    ("make_input_amax", "match"),
    [
        (lambda: [1.0], "torch.Tensor"),
        (
            lambda: torch.ones((1, 1, 8), device="cuda", dtype=torch.float16),
            "same dtype as input",
        ),
        (
            lambda: torch.ones((1, 1, 8), dtype=torch.bfloat16),
            "CUDA device",
        ),
        (
            lambda: torch.ones((4, 2), device="cuda", dtype=torch.bfloat16),
            "must be 3-D blocked-8",
        ),
        (
            lambda: torch.ones((2, 1, 8), device="cuda", dtype=torch.bfloat16),
            "row-block dimension",
        ),
        (
            lambda: torch.empty((1, 0, 8), device="cuda", dtype=torch.bfloat16),
            "at least one tile",
        ),
        (
            lambda: torch.ones((1, 1, 4), device="cuda", dtype=torch.bfloat16),
            "trailing dimension",
        ),
        (
            lambda: torch.ones(
                (1, 8, 4), device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2),
            "must be contiguous",
        ),
        (
            lambda: torch.ones((1, 1, 9), device="cuda", dtype=torch.bfloat16)[..., 1:],
            "4-byte aligned",
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


@pytest.mark.parametrize(
    ("make_valid_rows", "with_input_amax", "match"),
    [
        (lambda: torch.tensor([4], device="cuda", dtype=torch.int32), False, "only"),
        (lambda: [4], True, "torch.Tensor"),
        (
            lambda: torch.tensor([4], device="cuda", dtype=torch.int64),
            True,
            "torch.int32",
        ),
        (lambda: torch.tensor([4], dtype=torch.int32), True, "CUDA device"),
        (
            lambda: torch.tensor([[4]], device="cuda", dtype=torch.int32),
            True,
            r"shape \(1,\)",
        ),
    ],
)
def test_nvfp4_per_token_input_amax_valid_rows_validation(
    make_valid_rows, with_input_amax, match
):
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    input = torch.ones((4, 32), device="cuda", dtype=torch.bfloat16)
    global_scale_inv = torch.ones((1,), device="cuda", dtype=torch.float32)
    input_amax = (
        torch.ones((1, 1, 8), device="cuda", dtype=torch.bfloat16)
        if with_input_amax
        else None
    )
    with pytest.raises(AssertionError, match=match):
        nvfp4_quantize_per_token_cute_dsl(
            input,
            global_scale_inv,
            enable_pdl=False,
            input_amax=input_amax,
            input_amax_valid_rows=make_valid_rows(),
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_nvfp4_per_token_input_amax_requires_same_device():
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    input = torch.ones((4, 32), device="cuda:0", dtype=torch.bfloat16)
    global_scale_inv = torch.ones((1,), device="cuda:0", dtype=torch.float32)
    input_amax = torch.ones((1, 1, 8), device="cuda:1", dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="same device"):
        nvfp4_quantize_per_token_cute_dsl(
            input,
            global_scale_inv,
            enable_pdl=False,
            input_amax=input_amax,
        )
