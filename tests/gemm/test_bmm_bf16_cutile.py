"""Unit tests for the cuTile backend of flashinfer.gemm.bmm_bf16.

The cuTile path lives in ``flashinfer.cutile.bmm.bmm_bf16_cutile`` and is
wired into ``flashinfer.gemm.bmm_bf16`` with ``backend="cutile"``.

Scoped to the cuTile-only quirks:

* Output dtype must be bfloat16 (the cudnn/cutlass backends accept fp16/fp32
  outputs; cuTile path is bf16-only in v1).
* Requires SM >= 100 (Blackwell) and the cuda-tile python package.

Reference is torch.bmm at fp32 precision; we compare via torch.testing.assert_close
with bf16-friendly atol/rtol (1e-2).
"""

import math

import pytest
import torch

from flashinfer.gemm import bmm_bf16
from flashinfer.utils import get_compute_capability


def _cutile_available() -> bool:
    try:
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    return True


def _skip_if_not_supported():
    if not _cutile_available():
        pytest.skip("cuda-tile not installed in this environment.")
    cc = get_compute_capability(torch.device("cuda"))
    cc_num = cc[0] * 10 + cc[1]
    if cc_num < 100:
        pytest.skip(f"cuTile bmm_bf16 targets SM >= 100; detected sm{cc_num}.")


@pytest.mark.parametrize("b", [1, 4, 16])
@pytest.mark.parametrize("m", [128, 256])
@pytest.mark.parametrize("n", [128, 512])
@pytest.mark.parametrize("k", [128, 1024])
def test_bmm_bf16_cutile(b, m, n, k):
    """cuTile BMM BF16 must agree with torch.bmm reference within atol/rtol = 1e-2."""
    _skip_if_not_supported()

    torch.random.manual_seed(0)
    A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16)
    # Match flashinfer's convention: B is (b, k, n) col-major — same memory as
    # (b, n, k) row-major.
    B_nk = torch.randn(b, n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    B = B_nk.transpose(-2, -1)

    ref = torch.bmm(A.float(), B.float()).to(torch.bfloat16)

    out = bmm_bf16(A, B, out_dtype=torch.bfloat16, backend="cutile")
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_bmm_bf16_cutile_rejects_non_bf16_out():
    """The cuTile path only supports bfloat16 output; non-bf16 must raise."""
    _skip_if_not_supported()

    A = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16).transpose(-2, -1)

    with pytest.raises(ValueError, match="bfloat16"):
        bmm_bf16(A, B, out_dtype=torch.float16, backend="cutile")


def test_bmm_bf16_cutile_repeat_uses_tune_cache():
    """Two back-to-back calls at the same shape must hit the cuTile tune cache."""
    _skip_if_not_supported()

    torch.random.manual_seed(0)
    A = torch.randn(4, 128, 256, device="cuda", dtype=torch.bfloat16)
    B = (torch.randn(4, 256, 256, device="cuda", dtype=torch.bfloat16)
            .transpose(-2, -1))

    from flashinfer.cutile.bmm import _BMM_BF16_TUNE_CACHE
    _BMM_BF16_TUNE_CACHE.clear()
    out1 = bmm_bf16(A, B, out_dtype=torch.bfloat16, backend="cutile")
    assert len(_BMM_BF16_TUNE_CACHE) == 1, (
        f"first call should populate cache; got {len(_BMM_BF16_TUNE_CACHE)} entries"
    )
    out2 = bmm_bf16(A, B, out_dtype=torch.bfloat16, backend="cutile")
    assert len(_BMM_BF16_TUNE_CACHE) == 1, (
        f"second call must hit cache; got {len(_BMM_BF16_TUNE_CACHE)} entries"
    )
    torch.testing.assert_close(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__])
