"""
Tests for concat_mla_k kernel — verifies correctness across BF16, FP16, and FP8 dtypes.

concat_mla_k is a pure memory movement operation (copy + broadcast), so the output
must be **bit-exact** compared to the PyTorch slice-assign reference.
"""

import pytest
import torch

from flashinfer.concat_ops import concat_mla_k
from flashinfer.utils import get_compute_capability

NUM_LOCAL_HEADS = 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM


def _reference_concat(k_nope: torch.Tensor, k_rope: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: slice-assign with broadcast."""
    k = torch.empty(
        (*k_nope.shape[:-1], K_HEAD_DIM),
        dtype=k_nope.dtype,
        device=k_nope.device,
    )
    k[..., :QK_NOPE_HEAD_DIM] = k_nope
    k[..., QK_NOPE_HEAD_DIM:] = k_rope
    return k


def _make_tensors(num_tokens: int, dtype: torch.dtype, device: str = "cuda"):
    """Create contiguous k_nope, k_rope, and pre-allocated output k."""
    # Generate in BF16 then cast — FP8 doesn't support randn directly
    k_nope = (
        torch.randn(
            num_tokens,
            NUM_LOCAL_HEADS,
            QK_NOPE_HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        .to(dtype)
        .contiguous()
    )
    k_rope = (
        torch.randn(
            num_tokens, 1, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        .to(dtype)
        .contiguous()
    )
    k = torch.empty(num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, dtype=dtype, device=device)
    return k, k_nope, k_rope


# ────────────────────────── Core correctness tests ──────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 32, 1024, 8192])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float16,
        pytest.param(torch.float8_e4m3fn, id="fp8_e4m3"),
        pytest.param(torch.float8_e5m2, id="fp8_e5m2"),
    ],
)
def test_concat_mla_k_correctness(num_tokens, dtype):
    """Bit-exact correctness: flashinfer output == PyTorch reference."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        major, minor = get_compute_capability(torch.device("cuda"))
        if (major, minor) < (8, 9):
            pytest.skip("FP8 requires SM >= 89 (Ada/Hopper)")

    k, k_nope, k_rope = _make_tensors(num_tokens, dtype)
    concat_mla_k(k, k_nope, k_rope)

    ref = _reference_concat(k_nope, k_rope)

    # Pure copy — must be bit-exact
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        assert torch.equal(k.view(torch.uint8), ref.view(torch.uint8)), (
            f"Mismatch for dtype={dtype}, num_tokens={num_tokens}."
        )
    else:
        assert torch.equal(k, ref), (
            f"Mismatch for dtype={dtype}, num_tokens={num_tokens}. "
            f"max abs diff = {(k.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
        )


# ────────────────────────── Zero-token edge case ──────────────────────────


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float8_e4m3fn],
)
def test_concat_mla_k_zero_tokens(dtype):
    """num_tokens=0 should return immediately without error."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        major, minor = get_compute_capability(torch.device("cuda"))
        if (major, minor) < (8, 9):
            pytest.skip("FP8 requires SM >= 89")

    k, k_nope, k_rope = _make_tensors(0, dtype)
    concat_mla_k(k, k_nope, k_rope)  # should not crash


# ────────────────────────── Strided (non-contiguous last dim) inputs ──────


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        pytest.param(torch.float8_e4m3fn, id="fp8_e4m3"),
    ],
)
def test_concat_mla_k_strided_inputs(dtype):
    """Verify correctness when k_nope is a slice of a larger contiguous tensor."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        major, minor = get_compute_capability(torch.device("cuda"))
        if (major, minor) < (8, 9):
            pytest.skip("FP8 requires SM >= 89")

    num_tokens = 2048

    # k_nope is a slice — last-dim contiguous but has a stride gap on dim-1
    nope_container = torch.randn(
        num_tokens,
        NUM_LOCAL_HEADS,
        QK_NOPE_HEAD_DIM + 128,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(dtype)
    k_nope = nope_container[:, :, :QK_NOPE_HEAD_DIM]

    k_rope = (
        torch.randn(
            num_tokens, 1, QK_ROPE_HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        .to(dtype)
        .contiguous()
    )

    k = torch.empty(num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, dtype=dtype, device="cuda")
    concat_mla_k(k, k_nope, k_rope)

    ref = _reference_concat(k_nope, k_rope)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        assert torch.equal(k.view(torch.uint8), ref.view(torch.uint8))
    else:
        assert torch.equal(k, ref)


# ────────────────────────── Cross-dtype guard ──────────────────────────


def test_concat_mla_k_dtype_mismatch_raises():
    """Passing mismatched dtypes should raise an error from the C++ side."""
    num_tokens = 64
    k_nope = torch.randn(
        num_tokens,
        NUM_LOCAL_HEADS,
        QK_NOPE_HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_rope = torch.randn(
        num_tokens,
        1,
        QK_ROPE_HEAD_DIM,
        device="cuda",
        dtype=torch.float16,  # intentional mismatch
    )
    k = torch.empty(
        num_tokens,
        NUM_LOCAL_HEADS,
        K_HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.raises(RuntimeError):
        concat_mla_k(k, k_nope, k_rope)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
