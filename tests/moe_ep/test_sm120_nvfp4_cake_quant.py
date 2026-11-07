"""Single-GPU test: the fused Triton NVFP4 input quantizer is bit-identical to the torch reference."""

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake.cake_nvfp4_quant import (
    nvfp4_quantize_fused,
)
from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake.cake_nvfp4_weights import (
    nvfp4_quantize_torch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)

TIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -0.75, -5.0, 0.0)


def _make(kind: str, rows: int, hidden: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(rows * 7 + hidden)
    if kind == "randn":
        return torch.randn(rows, hidden, device="cuda", generator=g)
    if kind == "wide":
        e = torch.randint(-12, 10, (rows, hidden // 16, 1), device="cuda", generator=g)
        return (
            torch.randn(rows, hidden // 16, 16, device="cuda", generator=g)
            * torch.exp2(e.float())
        ).reshape(rows, hidden)
    if kind == "tiny":
        return torch.randn(rows, hidden, device="cuda", generator=g) * 1e-3
    if kind == "ties":
        t = torch.tensor(TIES, device="cuda")
        return t[
            torch.randint(0, len(TIES), (rows, hidden), device="cuda", generator=g)
        ]
    return torch.zeros(rows, hidden, device="cuda")


@pytest.mark.parametrize("rows,hidden", [(1, 1024), (7, 4096), (257, 5120), (64, 7168)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("norm", [1.0, 0.37, 3.0, 128.0])
@pytest.mark.parametrize("kind", ["randn", "wide", "tiny", "ties", "zeros"])
def test_fused_quantizer_matches_torch(rows, hidden, dtype, norm, kind) -> None:
    x = _make(kind, rows, hidden).to(dtype)
    c0, s0 = nvfp4_quantize_torch(x, norm)
    c1, s1 = nvfp4_quantize_fused(x, norm)
    assert torch.equal(c0, c1) and torch.equal(s0, s1)


def test_fused_quantizer_writes_into_larger_buffers() -> None:
    x = _make("randn", 33, 4096).to(torch.bfloat16)
    codes = torch.full((64, 2048), 0xAB, dtype=torch.uint8, device="cuda")
    sf = torch.full((64, 256), 0xCD, dtype=torch.uint8, device="cuda")
    c, s = nvfp4_quantize_fused(x, 1.0, codes=codes, sf=sf)
    c0, s0 = nvfp4_quantize_torch(x, 1.0)
    assert torch.equal(c, c0) and torch.equal(s, s0)
    assert bool((codes[33:] == 0xAB).all()) and bool(
        (sf[33:] == 0xCD).all()
    )  # rows beyond are untouched
