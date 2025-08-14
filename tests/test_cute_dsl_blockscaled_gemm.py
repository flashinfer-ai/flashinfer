"""
This is the test file for MaskedBatchedMatmulCuteDSL kernel.
`test_blockscaled_gemm_python_interface` is the python interface test. For pytorch DLFW, refer to this.
"""

from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
from flashinfer.cute_dsl.blockscaled_gemm import (
    Sm100BlockScaledPersistentDenseGemmKernel,  # not used in python interface
)
from flashinfer.cute_dsl.blockscaled_gemm import create_scale_factor_tensor
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    is_cute_dsl_available,
)


# todo(Yingyi): complete this test for target python interface
@pytest.mark.skipif(
    not is_cute_dsl_available(), reason="Please `pip install nvidia-cutlass-dsl`"
)
@pytest.mark.parametrize("lm", [(1, 1024), (2, 512), (4, 256)])
@pytest.mark.parametrize("kn", [(7168, 4096), (2048, 7168)])
@pytest.mark.parametrize(
    "ab_dtype,sf_dtype,c_dtype,sf_vec_size",
    [
        ("float4_e2m1fn", "float8_e8m0fnu", "float16", 16),
        ("float4_e2m1fn", "float8_e8m0fnu", "bfloat16", 16),
        ("float4_e2m1fn", "float8_e8m0fnu", "float32", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "float16", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "bfloat16", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "float32", 16),
        ("float8_e4m3fn", "float8_e8m0fnu", "bfloat16", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float16", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float32", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float8_e4m3fn", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float8_e5m2", 32),
        ("float8_e5m2", "float8_e8m0fnu", "bfloat16", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float16", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float32", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float8_e4m3fn", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float8_e5m2", 32),
    ],
)
@pytest.mark.parametrize("a_major", ["k"])
@pytest.mark.parametrize("b_major", ["k"])
@pytest.mark.parametrize("c_major", ["n"])
@pytest.mark.parametrize("mma_tiler_mn", [(128, 128)])
@pytest.mark.parametrize("cluster_shape_mn", [(1, 1)])
@pytest.mark.parametrize("tolerance", [1e-01])
@pytest.mark.parametrize("iterations", [3])
def test_blockscaled_gemm_python_interface(
    lm: Tuple[int, int],
    kn: Tuple[int, int],
    ab_dtype: cutlass.dtype,
    sf_dtype: cutlass.dtype,
    sf_vec_size: int,
    c_dtype: cutlass.dtype,
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    iterations: int,
):
    torch.manual_seed(42)
    l, m = lm
    k, n = kn
    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        get_cutlass_dtype(ab_dtype),
        get_cutlass_dtype(sf_dtype),
        sf_vec_size,
        get_cutlass_dtype(c_dtype),
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        pytest.skip(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    # not used for now
    def create_torch_tensor(l, mode0, mode1, is_mode0_major, cutlass_dtype, device):
        """
        Create a torch tensor with specified shape and dtype for testing. Optionally permute it.
        todo(Yingyi): Initialize it with specified init type and config
        For dtype, you should pass:
        - Float32: torch.float32
        - Float16: torch.float16
        - BFloat16: torch.bfloat16
        - Float8E5M2: torch.uint8
        - Float8E4M3FN: torch.uint8
        - Float8E4M3B11FNUZ: torch.uint8
        - Float8E8M0FNU: torch.uint8
        - Float4E2M1FN: torch.int8

        - Return: torch tensor with cutlass dtype
        """
        torch_type_map = {
            # TFloat32 is just alias of float32
            cutlass.TFloat32: torch.float32,
            cutlass.Float32: torch.float32,
            cutlass.BFloat16: torch.bfloat16,
            cutlass.Float16: torch.float16,
            cutlass.Float8E5M2: torch.int8,  # todo(Yingyi): removed after 2.8?
            cutlass.Float8E4M3FN: torch.int8,
            cutlass.Float8E4M3B11FNUZ: torch.int8,
            cutlass.Float4E2M1FN: torch.int8,
        }
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)

        if cutlass_dtype == cutlass.Float4E2M1FN:
            mode0 = mode0 // 2 if is_mode0_major else mode0
            mode1 = mode1 if is_mode0_major else mode1 // 2

        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        fp32_torch_tensor = torch.randn(*shape, dtype=torch.float32, device=device)
        dtype_torch_tensor = fp32_torch_tensor.to(dtype=torch_type_map[cutlass_dtype])
        dtype_torch_tensor = dtype_torch_tensor.permute(permute_order)

        return dtype_torch_tensor

    a_ref = cutlass_torch.matrix(l, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, c_major == "m", cutlass.Float32)
    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref,
        get_cutlass_dtype(c_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, sf_vec_size, get_cutlass_dtype(sf_dtype)
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype)
    )
    masked_m_tensor = torch.randint(0, m, (l,), dtype=torch.int32, device="cuda")

    for _ in range(iterations):
        assert a_major == "k" and b_major == "k" and c_major == "n"
        grouped_gemm_nt_masked(
            (a_torch, sfa_torch),
            (b_torch, sfb_torch),
            c_torch,
            masked_m_tensor,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
        torch.cuda.synchronize()

    # compute ref output
    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)

    # Convert c back to f32 for comparison.
    c_ref_device = c_ref.cuda()
    cute.testing.convert(
        c_tensor,
        from_dlpack(c_ref_device, assumed_align=16).mark_layout_dynamic(
            leading_dim=(1 if c_major == "n" else 0)
        ),
    )
    c_ref = c_ref_device.cpu()

    if c_dtype in ("float32", "float16", "bfloat16"):
        for i in range(l):
            torch.testing.assert_close(
                c_ref[: masked_m_tensor[i].item(), :, i],
                ref[: masked_m_tensor[i].item(), :, i],
                atol=tolerance,
                rtol=1e-02,
            )
    elif c_dtype in ("float8_e5m2", "float8_e4m3fn"):
        # Convert ref : f32 -> f8 -> f32
        ref_f8_ = torch.empty(*(l, m, n), dtype=torch.uint8, device="cuda").permute(
            1, 2, 0
        )
        ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        ref_f8.element_type = get_cutlass_dtype(c_dtype)
        ref_device = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
        ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        cute.testing.convert(ref_tensor, ref_f8)
        cute.testing.convert(ref_f8, ref_tensor)
        ref = ref_device.cpu()
        for i in range(l):
            torch.testing.assert_close(
                c_ref[: masked_m_tensor[i].item(), :, i],
                ref[: masked_m_tensor[i].item(), :, i],
                atol=tolerance,
                rtol=1e-02,
            )
