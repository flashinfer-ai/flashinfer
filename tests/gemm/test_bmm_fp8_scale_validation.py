"""Tests for bmm_fp8 scale tensor validation.

Issue: https://github.com/flashinfer-ai/flashinfer/issues/4052
"""

import pytest
import torch

from flashinfer import autotune
from flashinfer.gemm import bmm_fp8
from flashinfer.utils import get_compute_capability


def _skip_if_backend_unsupported(backend: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bmm_fp8 test")
    major, minor = get_compute_capability(torch.device("cuda"))
    cc = major * 10 + minor
    if not bmm_fp8.is_backend_supported(backend, cc):
        pytest.skip(f"bmm_fp8 backend '{backend}' is not supported on SM{cc}")


def _make_bmm_inputs(b: int = 2, m: int = 16, k: int = 16, n: int = 16):
    torch.manual_seed(0)
    A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    # B must be column-major: create (b, n, k) then transpose to (b, k, n)
    B = (
        torch.randn(b, n, k, device="cuda", dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn)
        .transpose(-2, -1)
    )
    return A, B


@pytest.mark.parametrize("scale_shape", [(), (1,), (1, 1, 1)])
def test_bmm_fp8_valid_scalar_scale(scale_shape):
    """Valid 0-D and 1-element float32 CUDA scales match the scaled reference."""
    _skip_if_backend_unsupported("cublas")
    b, m, k, n = 2, 16, 16, 16
    A, B = _make_bmm_inputs(b, m, k, n)
    A_scale = torch.full(scale_shape, 2.0, dtype=torch.float32, device="cuda")
    B_scale = torch.full(scale_shape, 0.75, dtype=torch.float32, device="cuda")

    out = bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")
    assert out.shape == (b, m, n)
    assert out.dtype == torch.bfloat16

    expected = (torch.bmm(A.float(), B.float()) * (2.0 * 0.75)).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("backend", ["auto", "cublas"])
@pytest.mark.parametrize("scale_name", ["A_scale", "B_scale"])
def test_bmm_fp8_non_scalar_scale_rejected(scale_name, backend):
    """Non-scalar scales (e.g. per-row/per-column) are rejected before dispatch."""
    _skip_if_backend_unsupported("cublas")
    b, m, k, n = 2, 16, 16, 16
    A, B = _make_bmm_inputs(b, m, k, n)
    scalar = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    nonscalar = torch.ones(b, m, 1, dtype=torch.float32, device="cuda")
    A_scale, B_scale = (
        (nonscalar, scalar) if scale_name == "A_scale" else (scalar, nonscalar)
    )
    with pytest.raises(ValueError, match=f"{scale_name} must be a scalar"):
        bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend=backend)


def test_bmm_fp8_batch_invariant_scale_rejected():
    """batch_invariant=True enforces the 1-element float32 scale contract too."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bmm_fp8 test")
    from flashinfer import batch_invariant as bi
    from flashinfer.gemm.batch_invariant.api import BMM_FP8_ARCHS

    if not bi.is_supported_device(torch.device("cuda"), BMM_FP8_ARCHS):
        pytest.skip("batch_invariant bmm_fp8 requires SM100/SM103/SM107 with CuTe DSL")

    A, B = _make_bmm_inputs()
    scalar = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    nonscalar = torch.ones(2, 16, 1, dtype=torch.float32, device="cuda")
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="1-element float32"):
        bmm_fp8(
            A,
            B,
            nonscalar,
            scalar,
            torch.bfloat16,
            backend="auto",
            batch_invariant=True,
        )


def test_bmm_fp8_rejects_bad_scale_before_backend_selection(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bmm_fp8 test")

    from flashinfer.gemm import gemm_base

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("auto backend heuristic ran before scale validation")

    monkeypatch.setattr(gemm_base, "_match_sm_version", unexpected_dispatch)
    A = torch.empty((2, 16, 16), device="cuda", dtype=torch.float8_e4m3fn)
    B = torch.empty_like(A).transpose(-2, -1)
    invalid = torch.ones(2, dtype=torch.float32, device="cuda")
    scalar = torch.ones((), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="A_scale must be a scalar"):
        bmm_fp8(A, B, invalid, scalar, torch.bfloat16, backend="auto")


@pytest.mark.parametrize("scale_name", ["A_scale", "B_scale"])
def test_bmm_fp8_wrong_dtype_rejected(scale_name):
    """Non-float32 scale tensors are rejected."""
    _skip_if_backend_unsupported("cublas")
    A, B = _make_bmm_inputs()
    valid = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    wrong = torch.tensor(1.0, dtype=torch.float16, device="cuda")
    A_scale, B_scale = (wrong, valid) if scale_name == "A_scale" else (valid, wrong)

    with pytest.raises(ValueError, match=f"{scale_name} must be torch.float32"):
        bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")


@pytest.mark.parametrize("scale_name", ["A_scale", "B_scale"])
def test_bmm_fp8_wrong_device_rejected(scale_name):
    """Scales on a different device than the operand (e.g. CPU) are rejected."""
    _skip_if_backend_unsupported("cublas")
    A, B = _make_bmm_inputs()
    cuda_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    cpu_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    A_scale, B_scale = (
        (cpu_scale, cuda_scale) if scale_name == "A_scale" else (cuda_scale, cpu_scale)
    )

    with pytest.raises(
        ValueError, match=f"{scale_name} must be on the same CUDA device"
    ):
        bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")


def test_bmm_fp8_cpu_scale_validator_rejects_cpu_tensors():
    """Direct scale helper rejects CPU scales even if operands are also on CPU."""
    from flashinfer.gemm.gemm_base import _validate_bmm_fp8_scales

    A = torch.empty((2, 16, 16), dtype=torch.float8_e4m3fn)
    B = torch.empty((2, 16, 16), dtype=torch.float8_e4m3fn)
    s = torch.ones((), dtype=torch.float32)
    with pytest.raises(ValueError, match="A_scale must be on the same CUDA device"):
        _validate_bmm_fp8_scales(A, B, s, s)


@pytest.mark.parametrize("backend", ["cublas", "cutlass"])
@pytest.mark.parametrize("bad_case", ["nonscalar", "dtype", "device"])
def test_bmm_fp8_ffi_boundary_rejects_under_skip_check(backend, bad_case):
    """Native FFI bindings also validate scales before float* casts when skip_check=True."""
    _skip_if_backend_unsupported(backend)
    A, B = _make_bmm_inputs()
    valid = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    if bad_case == "nonscalar":
        bad = torch.ones(2, dtype=torch.float32, device="cuda")
    elif bad_case == "dtype":
        bad = torch.tensor(1.0, dtype=torch.float16, device="cuda")
    else:
        bad = torch.tensor(1.0, dtype=torch.float32, device="cpu")

    for tune in (False, True) if backend == "cublas" else (False,):
        with autotune(tune), pytest.raises((RuntimeError, ValueError)):
            bmm_fp8(A, B, bad, valid, torch.bfloat16, backend=backend, skip_check=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
