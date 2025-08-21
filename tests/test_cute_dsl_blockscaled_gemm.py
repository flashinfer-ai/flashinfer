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

from flashinfer.cute_dsl.blockscaled_gemm import (
    Sm100BlockScaledPersistentDenseGemmKernel,  # not used in python interface
    grouped_gemm_nt_masked,  # deepgemm-like python interface for DLFW integration
    create_scale_factor_tensor,
)
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    is_cute_dsl_available,
)


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
@pytest.mark.parametrize("fuse_alpha", [False, True])
@pytest.mark.parametrize("alpha_dtype", ["float32"])
@pytest.mark.parametrize("mma_tiler_mn", [(128, 128)])
@pytest.mark.parametrize("cluster_shape_mn", [(1, 1)])
@pytest.mark.parametrize("sm_count", [132, None])
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
    fuse_alpha: bool,
    alpha_dtype: cutlass.dtype,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    tolerance: float,
    iterations: int,
):
    torch.manual_seed(42)
    device = torch.device("cuda:0")
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

    if not (a_major == "k" and b_major == "k" and c_major == "n"):
        # not supported since we try to align deepgemm for now
        pytest.skip(
            f"Skip non deepgemm-like cases {a_major}, {b_major}, {c_major}. Might be added later"
        )

    a_ref = cutlass_torch.matrix(
        l, m, k, a_major == "m", cutlass.Float32, device=device
    )
    b_ref = cutlass_torch.matrix(
        l, n, k, b_major == "n", cutlass.Float32, device=device
    )
    c_ref = cutlass_torch.matrix(
        l, m, n, c_major == "m", cutlass.Float32, device=device
    )

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
    alpha_tensor = (
        torch.randn(l, dtype=torch.float32, device=device) if fuse_alpha else None
    )

    # for deepgemm-like python interface
    if ab_dtype == "float4_e2m1fn":
        m, k, l = a_torch.shape
        n, k, l = b_torch.shape
        # slice into half after flatten
        half_len_a = a_torch.numel() // 2
        half_len_b = b_torch.numel() // 2
        a_torch = (
            a_torch.permute(2, 0, 1)
            .flatten()[:half_len_a]
            .reshape(l, m, k // 2)
            .permute(1, 2, 0)
        )
        b_torch = (
            b_torch.permute(2, 0, 1)
            .flatten()[:half_len_b]
            .reshape(l, n, k // 2)
            .permute(1, 2, 0)
        )

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    masked_m_tensor = torch.randint(0, m, (l,), dtype=torch.int32, device=device)

    for _ in range(iterations):
        # deepgemm-like python interface: fp4 packed, for DLFW integration
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
            alpha=alpha_tensor,
            alpha_dtype=alpha_dtype,
            sm_count=sm_count,
        )

    # compute ref output
    if not fuse_alpha:
        alpha_tensor = torch.ones(l, dtype=torch.float32, device=device)
    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
    ref = torch.einsum("mnl,l->mnl", ref, alpha_tensor)

    # Convert c back to f32 for comparison.
    cute.testing.convert(
        c_tensor,
        from_dlpack(c_ref, assumed_align=16).mark_layout_dynamic(
            leading_dim=(1 if c_major == "n" else 0)
        ),
    )

    if c_dtype in ("float32", "float16", "bfloat16"):
        for i in range(l):
            # skip testing c_ref & ref
            torch.testing.assert_close(
                c_ref[: masked_m_tensor[i].item(), :, i],
                ref[: masked_m_tensor[i].item(), :, i],
                atol=tolerance,
                rtol=1e-02,
            )
    elif c_dtype in ("float8_e5m2", "float8_e4m3fn"):
        # Convert ref : f32 -> f8 -> f32
        ref_f8_ = torch.empty(*(l, m, n), dtype=torch.uint8, device=device).permute(
            1, 2, 0
        )
        ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        ref_f8.element_type = get_cutlass_dtype(c_dtype)
        ref = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0)
        ref_tensor = from_dlpack(ref, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        cute.testing.convert(ref_tensor, ref_f8)
        cute.testing.convert(ref_f8, ref_tensor)
        for i in range(l):
            # skip testing c_ref & ref
            torch.testing.assert_close(
                c_ref[: masked_m_tensor[i].item(), :, i],
                ref[: masked_m_tensor[i].item(), :, i],
                atol=tolerance,
                rtol=1e-02,
            )


if __name__ == "__main__":
    test_blockscaled_gemm_python_interface(
        lm=(1, 1024),
        kn=(7168, 4096),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        sf_vec_size=16,
        c_dtype="float16",
        a_major="k",
        b_major="k",
        c_major="n",
        fuse_alpha=False,
        alpha_dtype="float32",
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(2, 1),
        tolerance=1e-01,
        iterations=3,
        sm_count=132,
    )
