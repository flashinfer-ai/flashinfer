"""Unit tests for the cuTile backend of flashinfer.mm_bf16.

The cuTile path lives in `flashinfer.cutile.gemm.mm_bf16_cutile` and is wired
into the upstream `flashinfer.mm_bf16` dispatcher with `backend="cutile"`.
Companion to `test_mm_bf16.py`, scoped to the cuTile-only quirks:

* `out_dtype == torch.bfloat16` only (raises ValueError otherwise)
* ignores `bias` / `pdl` (the cuTile kernel is alpha=1, beta=0)
* requires SM >= 90 and cuda-tile python package
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import mm_bf16
from flashinfer.utils import (
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm12x_supported,
)


def _cutile_available() -> bool:
    """The cuTile path is optional — gracefully skip when the runtime is missing."""
    try:
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    return True


def _supports_cutile_mm_bf16(device: torch.device) -> bool:
    """cuTile mm_bf16's autotune config space targets Hopper (SM90) through Blackwell (SM12x).

    Composed predicate because ``is_sm90a_supported`` matches only exact SM90; we accept
    any of {SM90a, SM100a, SM12x}.
    """
    return (
        is_sm90a_supported(device)
        or is_sm100a_supported(device)
        or is_sm12x_supported(device)
    )



@pytest.mark.parametrize("m", [16, 64, 256, 1024])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("k", [1536, 7168])
def test_mm_bf16_cutile(m: int, n: int, k: int):
    """cuTile mm_bf16 output must match the cuBLAS torch.mm reference within cos_sim > 0.99."""
    if not _cutile_available():
        pytest.skip("cuda-tile not installed in this environment.")
    if not _supports_cutile_mm_bf16(torch.device("cuda")):
        pytest.skip("cuTile mm_bf16 requires SM >= 90")

    torch.manual_seed(42)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    # The upstream mm_bf16 contract takes `b` shape (k, n) in column-major
    # (a transposed view of an (n, k) row-major tensor). torch.mm gives the
    # same arithmetic.
    reference = torch.mm(a, b.T)

    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    mm_bf16(a, b.T, None, False, out, torch.bfloat16, backend="cutile")

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99, f"cuTile mm_bf16 output mismatch: cos_sim={cos_sim:.6f}"


def test_mm_bf16_cutile_rejects_non_bf16_out():
    """The v1 cuTile path only emits bf16; fp16/fp32 out_dtype must raise."""
    if not _cutile_available():
        pytest.skip("cuda-tile not installed in this environment.")
    if not _supports_cutile_mm_bf16(torch.device("cuda")):
        pytest.skip("cuTile mm_bf16 requires SM >= 90")

    # a is (m, k) = (64, 1024); b is (n, k) = (2048, 1024); b.T is (k, n) = (1024, 2048)
    a = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(64, 2048, device="cuda", dtype=torch.float16)
    # The @backend_requirement decorator catches this in `_cutile_mm_bf16_requirement`
    # with the message "only supports bfloat16 output".
    with pytest.raises(ValueError, match="only supports bfloat16 output"):
        mm_bf16(a, b.T, None, False, out, torch.float16, backend="cutile")


def test_mm_bf16_cutile_repeat_uses_tune_cache():
    """Second call on the same shape should reuse the cached autotune result (no exception)."""
    if not _cutile_available():
        pytest.skip("cuda-tile not installed in this environment.")
    if not _supports_cutile_mm_bf16(torch.device("cuda")):
        pytest.skip("cuTile mm_bf16 requires SM >= 90")

    # a is (m, k) = (64, 1024); b is (n, k) = (2048, 1024); b.T is (k, n) = (1024, 2048)
    # mm_bf16 computes a @ b.T → (m, n) = (64, 2048).
    a = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(64, 2048, device="cuda", dtype=torch.bfloat16)

    # First call: warms tune cache.
    mm_bf16(a, b.T, None, False, out, torch.bfloat16, backend="cutile")
    out_first = out.clone()

    # Second call: must produce the same result and not raise.
    mm_bf16(a, b.T, None, False, out, torch.bfloat16, backend="cutile")
    cos_sim = F.cosine_similarity(out_first.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.999, "tune-cache reuse produced divergent output"


if __name__ == "__main__":
    pytest.main([__file__])
