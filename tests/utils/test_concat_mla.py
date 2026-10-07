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


def _cake_case(
    num_tokens: int,
    dtype: torch.dtype,
    input_layout: str = "contiguous",
    padded_output: bool = False,
):
    dtype_name = str(dtype).removeprefix("torch.")
    output_name = "padded" if padded_output else "contiguous"
    return pytest.param(
        num_tokens,
        dtype,
        input_layout,
        padded_output,
        id=f"t{num_tokens}-{dtype_name}-{input_layout}-{output_name}",
    )


_CAKE_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)
_CAKE_CONTRACT_CASES = [
    *(
        _cake_case(tokens, torch.bfloat16, "both_strided")
        for tokens in (2048, 4096, 8192, 16384, 32768)
    ),
    *(
        _cake_case(tokens, dtype)
        for tokens in (1, 32, 1024, 8192)
        for dtype in _CAKE_DTYPES
    ),
    *(_cake_case(0, dtype) for dtype in _CAKE_DTYPES[:3]),
    _cake_case(2048, torch.bfloat16, "nope_strided"),
    _cake_case(2048, torch.float8_e4m3fn, "nope_strided"),
    _cake_case(2048, torch.bfloat16),
    *(_cake_case(1, dtype, padded_output=True) for dtype in _CAKE_DTYPES),
    _cake_case(2, torch.bfloat16, padded_output=True),
    _cake_case(3, torch.float16),
    _cake_case(4, torch.float8_e4m3fn),
    _cake_case(5, torch.float8_e5m2, "both_strided", padded_output=True),
    _cake_case(31, torch.bfloat16),
    _cake_case(33, torch.float16),
    _cake_case(1023, torch.float8_e4m3fn),
    _cake_case(1025, torch.float8_e5m2),
]
assert len(_CAKE_CONTRACT_CASES) == 39


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


def _make_cake_tensors(
    num_tokens: int,
    dtype: torch.dtype,
    input_layout: str,
    padded_output: bool,
):
    if input_layout in ("nope_strided", "both_strided"):
        k_nope_container = torch.randn(
            num_tokens,
            NUM_LOCAL_HEADS,
            QK_NOPE_HEAD_DIM + 128,
            device="cuda",
            dtype=torch.bfloat16,
        ).to(dtype)
        k_nope = k_nope_container[..., :QK_NOPE_HEAD_DIM]
    else:
        k_nope = (
            torch.randn(
                num_tokens,
                NUM_LOCAL_HEADS,
                QK_NOPE_HEAD_DIM,
                device="cuda",
                dtype=torch.bfloat16,
            )
            .to(dtype)
            .contiguous()
        )
    if input_layout == "both_strided":
        k_rope_container = torch.randn(
            num_tokens,
            1,
            128 + QK_ROPE_HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        ).to(dtype)
        k_rope = k_rope_container[..., -QK_ROPE_HEAD_DIM:]
    else:
        k_rope = (
            torch.randn(
                num_tokens,
                1,
                QK_ROPE_HEAD_DIM,
                device="cuda",
                dtype=torch.bfloat16,
            )
            .to(dtype)
            .contiguous()
        )
    if padded_output:
        k_container = torch.empty(
            num_tokens,
            NUM_LOCAL_HEADS,
            256,
            dtype=dtype,
            device="cuda",
        )
        k = k_container[..., :K_HEAD_DIM]
    else:
        k = torch.empty(
            num_tokens,
            NUM_LOCAL_HEADS,
            K_HEAD_DIM,
            dtype=dtype,
            device="cuda",
        )
    return k, k_nope, k_rope


def _require_cake_concat_mla_k() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Cake concat MLA K requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("Cake concat MLA K requires SM100 or SM103")


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


@pytest.mark.parametrize(
    "num_tokens,dtype,input_layout,padded_output", _CAKE_CONTRACT_CASES
)
def test_cake_concat_mla_k_full_contract(
    num_tokens: int,
    dtype: torch.dtype,
    input_layout: str,
    padded_output: bool,
):
    """Run all 39 byte-exact SM100-family source-backend contract rows."""

    _require_cake_concat_mla_k()
    torch.manual_seed(17)
    k, k_nope, k_rope = _make_cake_tensors(
        num_tokens,
        dtype,
        input_layout,
        padded_output,
    )
    k_object = k
    k_metadata = (tuple(k.shape), tuple(k.stride()), k.storage_offset())
    k_nope_snapshot = k_nope.clone()
    k_rope_snapshot = k_rope.clone()
    result = concat_mla_k(k, k_nope, k_rope, backend="cake")

    expected = torch.empty_like(k)
    expected[..., :QK_NOPE_HEAD_DIM] = k_nope_snapshot
    expected[..., QK_NOPE_HEAD_DIM:] = k_rope_snapshot.expand(
        num_tokens, NUM_LOCAL_HEADS, -1
    )
    assert result is None
    assert k is k_object
    assert (tuple(k.shape), tuple(k.stride()), k.storage_offset()) == k_metadata
    assert torch.equal(
        k.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )
    assert torch.equal(
        k_nope.contiguous().view(torch.uint8),
        k_nope_snapshot.contiguous().view(torch.uint8),
    )
    assert torch.equal(
        k_rope.contiguous().view(torch.uint8),
        k_rope_snapshot.contiguous().view(torch.uint8),
    )


def test_cake_concat_mla_k_selects_sm100f_on_cc100(monkeypatch):
    from flashinfer.jit import cake_concat_mla_k

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.is_cuda_version_at_least",
        lambda version: version == "12.9",
    )

    assert cake_concat_mla_k.cake_concat_mla_k_target(torch.device("cuda")) == (
        "sm100f"
    )


@pytest.mark.parametrize(
    ("target", "target_arch", "expected_compute", "forbidden_compute"),
    [
        ("sm100f", (10, "0f"), "compute_100f", "compute_103a"),
        ("sm103a", (10, "3a"), "compute_103a", "compute_100f"),
    ],
)
def test_cake_concat_mla_k_jit_target_isolated(
    monkeypatch,
    target,
    target_arch,
    expected_compute,
    forbidden_compute,
):
    from flashinfer.jit import cake_concat_mla_k
    from flashinfer.jit import core as jit_core

    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {target_arch},
    )
    cake_concat_mla_k.gen_cake_concat_mla_k_module.cache_clear()

    spec = cake_concat_mla_k.gen_cake_concat_mla_k_module(target)

    assert f"_{target}_" in spec.name
    assert any(expected_compute in flag for flag in spec.extra_cuda_cflags)
    assert not any(forbidden_compute in flag for flag in spec.extra_cuda_cflags)


def test_cake_concat_mla_k_rejects_unsupported_arch(monkeypatch):
    _require_cake_concat_mla_k()
    k, k_nope, k_rope = _make_cake_tensors(1, torch.bfloat16, "contiguous", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    with pytest.raises(RuntimeError, match="requires compute capability 10.0"):
        concat_mla_k(k, k_nope, k_rope, backend="cake")


def test_concat_mla_k_rejects_unknown_backend():
    k = torch.empty(0)
    with pytest.raises(ValueError, match="unsupported concat_mla_k backend"):
        concat_mla_k(k, k, k, backend="unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
