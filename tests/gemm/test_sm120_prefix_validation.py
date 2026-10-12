"""Input validation for the SM120 smooth-quantize/LoRA-down prefix."""

from typing import Literal

import pytest
import torch
from typing_extensions import assert_never

from flashinfer.gemm.svdquant_sm120_cutlass import get_nvfp4_svdquant_sm120_module
from flashinfer.gemm.svdquant_sm120_routes import sm120_producer_variants


@pytest.fixture
def prefix_inputs() -> list[torch.Tensor]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    m, k = 64, 3072
    return [
        torch.ones((m, k), dtype=torch.bfloat16, device="cuda"),
        torch.ones(k, dtype=torch.bfloat16, device="cuda"),
        torch.ones(1, dtype=torch.float32, device="cuda"),
        torch.ones((k, 32), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
        torch.empty(128 * (k // 16), dtype=torch.uint8, device="cuda"),
        torch.empty((m, 32), dtype=torch.bfloat16, device="cuda"),
    ]


def _run_prefix(inputs: list[torch.Tensor], *, dynamic: bool) -> None:
    module = get_nvfp4_svdquant_sm120_module()
    if dynamic:
        family, tiling, policy = sm120_producer_variants(64, 3072)[0]
        module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
            *inputs, family, *tiling, policy
        )
    else:
        module.nvfp4_quantize_smooth_lora_down_sm120(*inputs)


@pytest.mark.parametrize("dynamic", (False, True))
@pytest.mark.parametrize(
    "index,shape,message",
    (
        (0, (64 * 3072,), "x must be 2-D"),
        (1, (1,), "pqs must have k elements"),
        (2, (0,), "global_scale must contain at least one element"),
        (3, (1, 32), r"l2t_smoothed must be \[k, 32\]"),
        (3, (3072, 16), r"l2t_smoothed must be \[k, 32\]"),
        (4, (64, 1), r"xq must be \[m, k/2\]"),
        (5, (1,), "sf is smaller than"),
        (6, (1, 32), r"down must be \[m, 32\]"),
    ),
)
def test_prefix_rejects_invalid_shapes_before_launch(
    prefix_inputs: list[torch.Tensor],
    dynamic: bool,
    index: int,
    shape: tuple[int, ...],
    message: str,
) -> None:
    original = prefix_inputs[index]
    prefix_inputs[index] = torch.empty(
        shape, dtype=original.dtype, device=original.device
    )
    with pytest.raises(RuntimeError, match=message):
        _run_prefix(prefix_inputs, dynamic=dynamic)


@pytest.mark.parametrize("dynamic", (False, True))
@pytest.mark.parametrize("index", range(1, 7))
def test_prefix_rejects_mixed_devices(
    prefix_inputs: list[torch.Tensor], dynamic: bool, index: int
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    other_device = (torch.cuda.current_device() + 1) % torch.cuda.device_count()
    prefix_inputs[index] = prefix_inputs[index].to(other_device)
    with pytest.raises(RuntimeError, match="device_id"):
        _run_prefix(prefix_inputs, dynamic=dynamic)


def test_dynamic_prefix_matches_default_prefix(
    prefix_inputs: list[torch.Tensor],
) -> None:
    _run_prefix(prefix_inputs, dynamic=False)
    expected = [tensor.clone() for tensor in prefix_inputs[4:]]
    for tensor in prefix_inputs[4:]:
        tensor.zero_()
    _run_prefix(prefix_inputs, dynamic=True)
    for actual, reference in zip(prefix_inputs[4:], expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("rank", (32, 64))
@pytest.mark.parametrize("entry", ("quantize", "fused", "dynamic", "cublaslt"))
@pytest.mark.parametrize("index,name", ((0, "x"), (1, "pqs")))
@pytest.mark.parametrize("offset", (1, 2, 4))
def test_native_prefix_rejects_misaligned_inputs_before_launch(
    prefix_inputs: list[torch.Tensor],
    rank: int,
    entry: Literal["quantize", "fused", "dynamic", "cublaslt"],
    index: int,
    name: str,
    offset: int,
) -> None:
    inputs = prefix_inputs
    inputs[3] = torch.ones((3072, rank), dtype=torch.bfloat16, device="cuda")
    inputs[6] = torch.empty((64, rank), dtype=torch.bfloat16, device="cuda")
    original = inputs[index]
    inputs[index] = torch.ones(
        original.numel() + offset, dtype=original.dtype, device=original.device
    )[offset:].view_as(original)
    assert inputs[index].is_contiguous()
    assert inputs[index].data_ptr() % 16 == offset * 2
    module = get_nvfp4_svdquant_sm120_module(rank)
    with pytest.raises(RuntimeError, match=f"{name} must be 16-byte aligned"):
        match entry:
            case "quantize":
                module.nvfp4_quantize_smooth(*inputs[:3], *inputs[4:6], False)
            case "fused":
                module.nvfp4_quantize_smooth_lora_down_sm120(*inputs)
            case "dynamic":
                family, tiling, policy = sm120_producer_variants(64, 3072, rank)[0]
                module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
                    *inputs, family, *tiling, policy
                )
            case "cublaslt":
                workspace = torch.empty(16, dtype=torch.uint8, device="cuda")
                module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
                    *inputs, workspace
                )
            case unreachable:
                assert_never(unreachable)
    torch.cuda.synchronize()
