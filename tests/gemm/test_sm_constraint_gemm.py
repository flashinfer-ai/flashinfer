import pytest
import torch

import flashinfer
import flashinfer.triton
from flashinfer.utils import get_compute_capability


def torch_gemm(a, b, c, alpha, beta):
    x = torch.matmul(a, b.T)
    c = alpha * x + beta * c
    return c


def torch_addmm(a, b, c, alpha=1.0, beta=0.0):
    # Transpose b to match torch_gemm's matmul(a, b.T)
    C = torch.addmm(c, a, b.T, beta=beta, alpha=alpha)
    return C


@pytest.mark.parametrize("M", [128, 512, 1024, 8192])
@pytest.mark.parametrize("N", [128, 512, 1024, 8192])
@pytest.mark.parametrize("K", [128, 512, 1024, 8192])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("num_sms", [1, 16, 64, 128, 132, 133])
@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize(
    "EPILOGUE_SUBTILE", [True, False]
)  # only for descriptor persistent
def test_sm_constraint_gemm(M, N, K, alpha, beta, num_sms, dtype, EPILOGUE_SUBTILE):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    # TODO(P1): Most of these tests pass on Blackwell. We need triage these at some point.
    if compute_capability[0] != 9:
        pytest.skip("These tests are only guaranteed to work on Hopper GPUs.")

    out_dtype = dtype if dtype != torch.float8_e4m3fn else torch.bfloat16
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()
    c = torch.randn((M, N), device="cuda", dtype=out_dtype)
    c_unmodified = c.clone()
    c0 = c.clone()
    c1 = c.clone()

    # torch gemm
    c_torch = torch_gemm(a.to(out_dtype), b.to(out_dtype), c.to(out_dtype), alpha, beta)

    # triton gemm: persistent
    c_persistent = flashinfer.triton.sm_constraint_gemm.gemm_persistent(
        a, b.T, c=c, alpha=alpha, beta=beta, num_sms=num_sms
    )

    # triton gemm: naive
    c_naive = flashinfer.triton.sm_constraint_gemm.gemm(
        a, b.T, c=c0, alpha=alpha, beta=beta
    )

    c_descriptor = None
    # triton gemm: descriptor persistent
    if dtype != torch.float32:
        c_descriptor = flashinfer.triton.sm_constraint_gemm.gemm_descriptor_persistent(
            a,
            b,
            c=c1,
            alpha=alpha,
            beta=beta,
            num_sms=num_sms,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        )

    torch_atol = 20.0 if out_dtype == torch.bfloat16 else 1.0

    in_place_persistent = c_persistent.data_ptr() == c.data_ptr() and torch.allclose(
        c_persistent.to(out_dtype), c.to(out_dtype)
    )
    assert in_place_persistent  # modified in place

    in_place_naive = c_naive.data_ptr() == c0.data_ptr() and torch.allclose(
        c_naive.to(out_dtype), c0.to(out_dtype)
    )
    assert in_place_naive  # modified in place

    if c_descriptor is not None:
        in_place_descriptor = (
            c_descriptor.data_ptr() == c1.data_ptr()
            and torch.allclose(c_descriptor.to(out_dtype), c1.to(out_dtype))
        )
        assert in_place_descriptor  # modified in place

    # torch results vs triton results
    torch_vs_triton_persistent = torch.allclose(
        c_torch.to(out_dtype), c_persistent.to(out_dtype), atol=torch_atol
    )
    if not torch_vs_triton_persistent:
        print_all_on_failure(
            a, b, c_unmodified, c_torch, c_naive, c_persistent, c_descriptor, out_dtype
        )
        print("compare c_torch and c_persistent")
        print_max_diff_on_failure(c_torch, c_persistent, out_dtype)
    assert torch_vs_triton_persistent  # value is correct

    if c_descriptor is not None:
        torch_vs_triton_descriptor = torch.allclose(
            c_torch.to(out_dtype), c_descriptor.to(out_dtype), atol=torch_atol
        )
        if not torch_vs_triton_descriptor:
            print_all_on_failure(
                a,
                b,
                c_unmodified,
                c_torch,
                c_naive,
                c_persistent,
                c_descriptor,
            )
            print("compare c_torch and c_descriptor")
            print_max_diff_on_failure(c_torch, c_descriptor, out_dtype)
        assert torch_vs_triton_descriptor  # value is correct

    # triton naive results vs each other
    triton_atol = 10.0 if out_dtype == torch.bfloat16 else 1.0
    naive_vs_persistent = torch.allclose(
        c_naive.to(out_dtype), c_persistent.to(out_dtype), atol=triton_atol
    )
    if not naive_vs_persistent:
        print_all_on_failure(
            a, b, c_unmodified, c_torch, c_naive, c_persistent, c_descriptor, out_dtype
        )
        print("compare c_naive and c_persistent")
        print_max_diff_on_failure(c_naive, c_persistent, out_dtype)

    assert naive_vs_persistent  # value is correct

    if c_descriptor is not None:
        descriptor_atol = 10.0 if out_dtype == torch.bfloat16 else 1.0
        naive_vs_descriptor = torch.allclose(
            c_naive.to(out_dtype), c_descriptor.to(out_dtype), atol=descriptor_atol
        )
        if not naive_vs_descriptor:
            print_all_on_failure(
                a,
                b,
                c_unmodified,
                c_torch,
                c_naive,
                c_persistent,
                c_descriptor,
            )
            print("compare c_naive and c_descriptor")
            print_max_diff_on_failure(c_naive, c_descriptor, out_dtype)

        assert naive_vs_descriptor  # value is correct


@pytest.mark.parametrize("kernel", ["naive", "persistent", "descriptor_persistent"])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sm_constraint_gemm_beta_zero_does_not_read_c(kernel, alpha, dtype):
    # When c is not passed, the wrapper allocates it with torch.empty. With
    # beta == 0 the epilogue must not read that buffer: 0.0 * NaN is NaN, so a
    # single read turns the whole result into NaN.
    if kernel == "descriptor_persistent" and dtype == torch.float32:
        pytest.skip("descriptor persistent does not support float32")

    M = N = K = 256
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    if kernel == "naive":
        run = lambda c: flashinfer.triton.sm_constraint_gemm.gemm(
            a, b, c=c, alpha=alpha
        )
    elif kernel == "persistent":
        run = lambda c: flashinfer.triton.sm_constraint_gemm.gemm_persistent(
            a, b, c=c, alpha=alpha
        )
    else:
        bT = b.T.contiguous()
        run = lambda c: flashinfer.triton.sm_constraint_gemm.gemm_descriptor_persistent(
            a, bT, c=c, alpha=alpha
        )

    # Leave NaN behind in a block of exactly the output's size and dtype, so the
    # caching allocator hands it to the torch.empty inside the wrapper.
    run(torch.zeros((M, N), device="cuda", dtype=dtype))  # warm up (compile)
    stale = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
    stale_ptr = stale.data_ptr()
    del stale

    out = run(None)  # c=None, beta=0.0
    assert out.data_ptr() == stale_ptr  # the poisoned block was reused
    assert torch.isfinite(out).all()
    # Nothing about the buffer may reach the result.
    assert torch.equal(out, run(torch.zeros((M, N), device="cuda", dtype=dtype)))


@pytest.mark.parametrize("kernel", ["naive", "persistent", "descriptor_persistent"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sm_constraint_gemm_beta_nonzero_reads_c(kernel, dtype):
    # The other half of the contract: with beta != 0 the epilogue still folds c
    # in. alpha = 0 zeroes the product, so the result must be exactly beta * c.
    if kernel == "descriptor_persistent" and dtype == torch.float32:
        pytest.skip("descriptor persistent does not support float32")

    M = N = K = 256
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    c = torch.randn((M, N), device="cuda", dtype=dtype)
    expected = (0.5 * c.float()).to(dtype)

    if kernel == "naive":
        out = flashinfer.triton.sm_constraint_gemm.gemm(a, b, c=c, alpha=0.0, beta=0.5)
    elif kernel == "persistent":
        out = flashinfer.triton.sm_constraint_gemm.gemm_persistent(
            a, b, c=c, alpha=0.0, beta=0.5
        )
    else:
        out = flashinfer.triton.sm_constraint_gemm.gemm_descriptor_persistent(
            a, b.T.contiguous(), c=c, alpha=0.0, beta=0.5
        )

    assert torch.equal(out, expected)


def print_all_on_failure(
    a, b, c_unmodified, c_torch, c_naive, c_persistent, c_descriptor
):
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c_unmodified: {c_unmodified}")
    if c_torch is not None:
        print(f"c_torch: {c_torch}")
    print(f"c_naive: {c_naive}")
    print(f"c_persistent: {c_persistent}")
    if c_descriptor is not None:
        print(f"c_descriptor: {c_descriptor}")


def print_max_diff_on_failure(target1, target2, out_dtype):
    max_diff = torch.max(torch.abs(target1.to(out_dtype) - target2.to(out_dtype)))
    print(f"max diff: {max_diff}")
    max_diff_index = torch.argmax(
        torch.abs(target1.to(out_dtype) - target2.to(out_dtype))
    )
    print(f"max diff index: {max_diff_index}")
    if target1.dim() > 1:
        max_diff_index = torch.unravel_index(max_diff_index, target1.shape)
    print(f"target1[max_diff_index]: {target1[max_diff_index]}")
    print(f"target2[max_diff_index]: {target2[max_diff_index]}")
