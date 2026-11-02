from __future__ import annotations

import pytest
import torch

from b12x.quantization.mxfp6 import (
    FP6DenseWeight,
    dense_fp6_linear,
    load_fp6_dense_weight,
    quantize_dense_weight_to_fp6,
    save_fp6_dense_weight,
)
from b12x.testing.reference.helpers import require_b12x

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FP6 GPU tests"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


@pytest.mark.parametrize("source_format", ["mxfp6_e3m2", "mxfp6_e2m3"])
def test_dense_fp6_weight_pipeline_roundtrip_and_equivalence(
    source_format: str, tmp_path
) -> None:
    require_b12x()
    torch.manual_seed(3)
    m, n, k = 128, 256, 128
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.2

    qw = quantize_dense_weight_to_fp6(weight, source_format=source_format)
    assert qw.packed.shape == (n, 3 * k // 4)
    assert qw.out_features == n and qw.in_features == k

    y = dense_fp6_linear(x, qw)
    assert y.shape == (m, n)

    ref = x.float() @ weight.float().T
    assert _cos(y, ref) > 0.95

    # save/load round-trip: identical tensors and identical kernel output.
    path = str(tmp_path / "dense_fp6.safetensors")
    save_fp6_dense_weight(qw, path)
    loaded = load_fp6_dense_weight(path, device="cuda")
    assert isinstance(loaded, FP6DenseWeight)
    torch.testing.assert_close(loaded.packed, qw.packed, rtol=0, atol=0)
    torch.testing.assert_close(loaded.scale_storage, qw.scale_storage, rtol=0, atol=0)

    y_loaded = dense_fp6_linear(x, loaded)
    torch.testing.assert_close(y_loaded, y, rtol=0, atol=0)


def test_dense_fp6_linear_pads_non_tile_token_count() -> None:
    require_b12x()
    torch.manual_seed(5)
    m, n, k = 70, 256, 128  # m not a multiple of 128
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.2

    qw = quantize_dense_weight_to_fp6(weight, source_format="mxfp6_e3m2")
    y = dense_fp6_linear(x, qw)
    assert y.shape == (m, n)

    ref = x.float() @ weight.float().T
    assert _cos(y, ref) > 0.95
