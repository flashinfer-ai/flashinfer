import pytest
import torch

from flashinfer.triton.kernels import sm_constraint_gemm


@pytest.mark.parametrize("kernel_name", ["gemm_kernel", "gemm_kernel_persistent"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "value,alpha,beta,initial_c",
    [
        pytest.param(1.0, 1 / 128, 0.0, 0.0, id="finite-control"),
        pytest.param(32.0, 1 / 128, 0.0, 0.0, id="scaled-overflow"),
        pytest.param(32.0, 0.5, 1.0, -1024.0, id="beta-cancellation"),
        pytest.param(2**-18, 2**16, 0.0, 0.0, id="scaled-underflow"),
    ],
)
def test_sm_constraint_gemm_epilogue(kernel_name, dtype, value, alpha, beta, initial_c):
    """Check scaling against an FP32 reference without intermediate output casts."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Requires SM80 or newer")

    a = torch.full((128, 128), value, dtype=dtype, device="cuda")
    b = a.clone()
    # Initialize C even when beta is zero to isolate accumulator precision
    # from the separate uninitialized-output issue (#4724).
    c = torch.full_like(a, initial_c)
    expected = torch.addmm(c.float(), a.float(), b.float(), alpha=alpha, beta=beta)

    kwargs = {"alpha": alpha, "beta": beta}
    if kernel_name == "gemm_kernel_persistent":
        kwargs["NUM_SMS"] = 1
    getattr(sm_constraint_gemm, kernel_name)[(1,)](
        a,
        b,
        c,
        128,
        128,
        128,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        **kwargs,
    )

    # Powers of two make the expected values exactly representable.
    torch.testing.assert_close(c, expected.to(dtype), rtol=0, atol=0)
