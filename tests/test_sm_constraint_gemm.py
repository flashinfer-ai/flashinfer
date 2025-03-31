import pytest
import torch

import flashinfer
import flashinfer.triton

def torch_gemm(a, b, c, alpha, beta):
    x = torch.matmul(a, b.T)
    c = alpha * x + beta * c
    return c

def torch_addmm(a, b, c, alpha=1.0, beta=0.0):
    # Transpose b to match torch_gemm's matmul(a, b.T)
    C = torch.addmm(c, a, b.T, beta=beta, alpha=alpha)
    return C

# @pytest.mark.parametrize("M", [128, 256, 512, 1024, 8192])
# @pytest.mark.parametrize("N", [128, 256, 512, 1024, 8192])
# @pytest.mark.parametrize("K", [128, 256, 512, 1024, 8192])
# @pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
# @pytest.mark.parametrize("beta", [0.0, 0.5, 2.0]) 
# @pytest.mark.parametrize("num_sms", [1, 16, 64, 128, 132, 133])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("M", [128, 256, 512, 1024, 8192])
@pytest.mark.parametrize("N", [128, 256, 512, 1024, 8192])
@pytest.mark.parametrize("K", [128, 256, 512, 1024, 8192])
@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 2.0]) 
@pytest.mark.parametrize("num_sms", [1, 16, 64, 128, 132, 133])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [True, False]) # only for descriptor persistent
def test_sm_constraint_gemm(M, N, K, alpha, beta, num_sms, dtype, EPILOGUE_SUBTILE):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()
    c = torch.randn((M, N), device="cuda", dtype=torch.float16).to(dtype)

    # a = torch.tensor([[-1.6250,  2.7500, -0.8750,  0.9375],
    #     [-2.2500, -0.2812, -0.9375,  1.1250],
    #     [ 1.6250, -0.0938, -1.5000,  0.5625],
    #     [ 0.6250,  0.8750, -1.0000,  0.0586]], device="cuda", dtype=dtype)
    # b = torch.tensor([[ 1.1250,  1.5000, -0.6875, -1.0000],
    #     [ 0.0625, -1.0000, -0.6250, -1.2500],
    #     [-0.8750,  1.3750,  2.5000,  1.8750],
    #     [ 0.6250, -1.3750,  0.2344, -0.3750]], device="cuda", dtype=dtype)
    # c = torch.tensor([[ 0.7500, -0.6250, -1.2500,  0.6875],
    #     [-0.1250,  0.4062,  0.7500, -0.6875],
    #     [ 2.5000,  1.3750, -0.1406,  0.4375],
    #     [ 0.5000, -0.5625,  0.8750,  0.7500]], device="cuda", dtype=dtype)
    
    c_unmodified = c.clone()
    c0 = c.clone()
    c1 = c.clone()
    assert torch.allclose(c.to(torch.float16), c0.to(torch.float16))

    c_torch = torch_gemm(a, b, c, alpha, beta) if dtype == torch.float16 or dtype == torch.float32 or dtype == torch.bfloat16 else None
    c_triton = flashinfer.triton.sm_constraint_gemm.gemm_persistent(a, b.T, c, alpha, beta, num_sms)
    c_naive = flashinfer.triton.sm_constraint_gemm.gemm(a, b.T, c0, alpha, beta)
    c_descriptor = flashinfer.triton.sm_constraint_gemm.gemm_descriptor_persistent(a, b, c1, alpha, beta, num_sms, EPILOGUE_SUBTILE)

    cmp_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
    torch_atol = 10.0 if dtype == torch.bfloat16 else 1.0

    in_place_triton_persistent = c_triton.data_ptr() == c.data_ptr() and torch.allclose(c_triton.to(cmp_dtype), c.to(cmp_dtype))
    assert in_place_triton_persistent # modified in place

    in_place_naive = c_naive.data_ptr() == c0.data_ptr() and torch.allclose(c_naive.to(cmp_dtype), c0.to(cmp_dtype))
    assert in_place_naive # modified in place

    in_place_descriptor = c_descriptor.data_ptr() == c1.data_ptr() and torch.allclose(c_descriptor.to(cmp_dtype), c1.to(cmp_dtype))
    assert in_place_descriptor # modified in place

    if c_torch is not None:
        torch_vs_triton_persistent = torch.allclose(c_torch.to(cmp_dtype), c_triton.to(cmp_dtype), atol=torch_atol)
        if torch_vs_triton_persistent == False:
            print(f"c_torch: {c_torch}")
            print(f"c_triton: {c_triton}")
            print(f"max diff: {torch.max(torch.abs(c_torch.to(cmp_dtype) - c_triton.to(cmp_dtype)))}")
        assert torch_vs_triton_persistent # value is correct

        torch_vs_triton_descriptor = torch.allclose(c_torch.to(cmp_dtype), c_descriptor.to(cmp_dtype), atol=torch_atol)
        if torch_vs_triton_descriptor == False:
            print(f"c_torch: {c_torch}")
            print(f"c_descriptor: {c_descriptor}")
            print(f"max diff: {torch.max(torch.abs(c_torch.to(cmp_dtype) - c_descriptor.to(cmp_dtype)))}")
        assert torch_vs_triton_descriptor # value is correct

    triton_atol = 1.0
    naive_vs_persistent = torch.allclose(c_naive.to(cmp_dtype), c_triton.to(cmp_dtype), atol=triton_atol)
    if naive_vs_persistent == False:
        if c_torch is not None:
            print(f"c_torch: {c_torch}")
        print(f"c_naive: {c_naive}")
        print(f"c_triton: {c_triton}")
        print(f"max diff: {torch.max(torch.abs(c_naive.to(cmp_dtype) - c_triton.to(cmp_dtype)))}")
    assert naive_vs_persistent # value is correct
    
    descriptor_atol = 1.0 if dtype == torch.float8_e4m3fn or dtype == torch.float16 or dtype == torch.float32 else 10.0
    naive_vs_descriptor = torch.allclose(c_naive.to(cmp_dtype), c_descriptor.to(cmp_dtype), atol=descriptor_atol)
    if naive_vs_descriptor == False:
        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c_unmodified: {c_unmodified}")
        print(f"c_naive: {c_naive}")
        print(f"c_descriptor: {c_descriptor}")
        print(f"max diff: {torch.max(torch.abs(c_naive.to(cmp_dtype) - c_descriptor.to(cmp_dtype)))}")
        # Calculate the index of the maximum difference
        max_diff_index = torch.argmax(torch.abs(c_naive.to(cmp_dtype) - c_descriptor.to(cmp_dtype)))

        # If c_descriptor is multi-dimensional, convert the flat index to a tuple of indices
        if c_descriptor.dim() > 1:
            max_diff_index = torch.unravel_index(max_diff_index, c_descriptor.shape)

        # Print the value at the index of maximum difference
        print(f"max_diff_index: {max_diff_index}")
        print(f"c_unmodified of max diff: {c_unmodified[max_diff_index]}")
        print(f"c_naive of max diff: {c_naive[max_diff_index]}")
        print(f"c_persistent of max diff: {c_triton[max_diff_index]}")
        print(f"c_descriptor of max diff: {c_descriptor[max_diff_index]}")

    assert naive_vs_descriptor # value is correct
    # print(f"a: {a}")
    # print(f"b: {b}")
    # print(f"c_unmodified: {c_unmodified}")
    # print(f"c_naive: {c_naive}")
    # print(f"c_descriptor: {c_descriptor}")
    # assert False
