"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("n", [128])
@pytest.mark.parametrize("k", [1024])
def test_fp8_gemm(
    b,
    m,
    n,
    k,
):
    torch.manual_seed(42)

    A_shape = (b, m, k)
    B_shape = (b, k, n)
    C_shape = (b, m, n)

    # input data types - using FP8 representation for inputs
    # Note: PyTorch uses uint8 to store FP8 values
    # Using E4M3 format (4-bit exponent, 3-bit mantissa) as FP8 format
    input_type_a = torch.uint8  # Will be interpreted as E4M3 by cuDNN
    input_type_b = torch.uint8  # Will be interpreted as E4M3 by cuDNN

    # input tensors - both are E4M3 (FP8) stored as uint8
    # Create actual floating point values first, then convert to FP8 format
    a_float = (
        torch.randn(A_shape, dtype=torch.float32, device="cuda") * 0.5
    )  # FP32 values
    b_float = (
        torch.randn(B_shape, dtype=torch.float32, device="cuda") * 0.5
    )  # FP32 values

    # Convert to FP8 E4M3 format using PyTorch's built-in conversion
    a = a_float.to(torch.float8_e4m3fn).view(
        torch.uint8
    )  # Convert to FP8 E4M3 and view as uint8
    b_row_major = b_float.to(torch.float8_e4m3fn).view(
        torch.uint8
    )  # Convert to FP8 E4M3 and view as uint8
    b = torch.as_strided(b_row_major, B_shape, (n * k, 1, n))

    # Create a 1x1x1 scaling tensor
    scale_tensor = (
        torch.ones((1, 1, 1), dtype=torch.float32, device="cuda") * 0.5
    )  # Example scaling factor

    # reference output - use the FP8 values (converted back to float) for accurate reference
    a_fp8_as_float = a.view(torch.float8_e4m3fn).to(torch.float32)
    b_fp8_as_float = b_row_major.view(torch.float8_e4m3fn).to(torch.float32)
    b_fp8_strided = torch.as_strided(b_fp8_as_float, B_shape, (n * k, 1, n))

    type_func_pairs = zip(
        [torch.bfloat16, torch.half],
        [flashinfer.gemm.gemm_f8f8_f32_bf16, flashinfer.gemm.gemm_f8f8_f32_fp16],
    )
    for output_type, gemm_func in type_func_pairs:
        mma_data_type = output_type
        c_matmul_ref = torch.matmul(
            a_fp8_as_float.to(mma_data_type), b_fp8_strided.to(mma_data_type)
        ).to(torch.float)
        c_ref = (c_matmul_ref * scale_tensor).to(output_type)

        # place holder for cudnn output
        c = torch.randn_like(c_matmul_ref, device="cuda")
        c_final = torch.randn_like(c_ref, device="cuda")

        # run the gemm
        c_final = gemm_func(a, b, scale_tensor, c_final)

        # check the result
        assert torch.allclose(c_final, c_ref, atol=1e-4)


if __name__ == "__main__":
    test_fp8_gemm(1, 32, 128, 512)
