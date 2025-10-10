import math
import pytest
from typing import Tuple

import cutlass
import cutlass.torch as cutlass_torch
import torch

from flashinfer.cute_dsl.blockwise_gemm import BlockwiseGemmKernel, blockwise_gemm
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    get_num_sm,
    is_cute_dsl_available,
)


def create_tensors(
    l, m, n, k, a_major, b_major, cd_major, ab_dtype, c_dtype, scale_dtype, device
):
    torch.manual_seed(42)

    a_torch_cpu = cutlass_torch.matrix(
        l, m, k, a_major == "m", get_cutlass_dtype(ab_dtype), device=device
        )
    b_torch_cpu = cutlass_torch.matrix(
        l, n, k, b_major == "n", get_cutlass_dtype(ab_dtype), device=device
        )
    c_torch_cpu = cutlass_torch.matrix(
        l, m, n, cd_major == "m", get_cutlass_dtype(c_dtype), device=device
        )
    sfa_torch_cpu = cutlass_torch.matrix(
        l, m, math.ceil(k / 128), True, get_cutlass_dtype(scale_dtype), device=device
        )
    sfb_torch_cpu = cutlass_torch.matrix(
        l, math.ceil(n / 128), math.ceil(k / 128), False,
        get_cutlass_dtype(scale_dtype), device=device,
    )

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_torch_cpu, get_cutlass_dtype(ab_dtype), is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_torch_cpu, get_cutlass_dtype(ab_dtype), is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_torch_cpu, get_cutlass_dtype(c_dtype), is_dynamic_layout=True, assumed_align=16
    )
    sfa_tensor, sfa_torch = cutlass_torch.cute_tensor_like(
        sfa_torch_cpu, get_cutlass_dtype(scale_dtype), is_dynamic_layout=True, assumed_align=16
    )
    sfb_tensor, sfb_torch = cutlass_torch.cute_tensor_like(
        sfb_torch_cpu, get_cutlass_dtype(scale_dtype), is_dynamic_layout=True, assumed_align=16
    )

    return (
        a_tensor,
        a_torch,
        b_tensor,
        b_torch,
        c_tensor,
        c_torch,
        sfa_tensor,
        sfa_torch,
        sfb_tensor,
        sfb_torch,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
    )


@pytest.mark.skipif(
    not is_cute_dsl_available(), reason="Please `pip install nvidia-cutlass-dsl`"
)
@pytest.mark.parametrize("lm", [(1, 256)])
@pytest.mark.parametrize("kn", [(512, 256)])
@pytest.mark.parametrize(
    "ab_dtype,sf_dtype,c_dtype,acc_dtype",
    [
        ("float8_e4m3fn", "float32", "bfloat16", "float32"),
    ],
)
@pytest.mark.parametrize("a_major", ["k"])
@pytest.mark.parametrize("b_major", ["k"])
@pytest.mark.parametrize("c_major", ["n"])
@pytest.mark.parametrize("use_2cta_instrs", [False])
@pytest.mark.parametrize("mma_tiler_mn", [(128, 128)])
@pytest.mark.parametrize("cluster_shape_mn", [(1, 1)])
@pytest.mark.parametrize("tolerance", [1e-01])
@pytest.mark.parametrize("iterations", [3])
def test_blockwise_gemm_python_interface(
    lm: Tuple[int, int],
    kn: Tuple[int, int],
    ab_dtype: cutlass.dtype,
    sf_dtype: cutlass.dtype,
    c_dtype: cutlass.dtype,
    acc_dtype: cutlass.dtype,
    a_major: str,
    b_major: str,
    c_major: str,
    use_2cta_instrs: bool,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    iterations: int,
):
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)

    if not (major == 10 and minor == 0):
        pytest.skip("Cute-dsl backend is only supported on SM100.")

    l, m = lm
    k, n = kn

    sm_count = get_num_sm(device)

    print(f"device: {device}")

    if not BlockwiseGemmKernel.can_implement(
        get_cutlass_dtype(ab_dtype),
        get_cutlass_dtype(acc_dtype),
        get_cutlass_dtype(c_dtype),
        use_2cta_instrs,
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
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {c_dtype}, {acc_dtype}, {use_2cta_instrs} ,{mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )
    
    (
        a_tensor,
        a_torch,
        b_tensor,
        b_torch,
        c_tensor,
        c_torch,
        sfa_tensor,
        sfa_torch,
        sfb_tensor,
        sfb_torch,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
    ) = create_tensors(
        l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype, sf_dtype, device
    )

    for _ in range(iterations):
        blockwise_gemm(
          a_torch,
          sfa_torch,
          b_torch,
          sfb_torch,
          c_torch,
          ab_dtype=ab_dtype,
          sf_dtype=sf_dtype,
          c_dtype=c_dtype,
          acc_dtype=acc_dtype,
          sm_count=sm_count,
          mma_tiler_mn=mma_tiler_mn,
          cluster_shape_mn=cluster_shape_mn,
          use_2cta_instrs=use_2cta_instrs,
        )

    torch.cuda.synchronize()
    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(sfa_torch_cpu, a_torch_cpu)
    updated_b = pad_and_multiply(sfb_torch_cpu, b_torch_cpu)

    ref = torch.einsum("mkl,nkl->mnl", updated_a, updated_b).to(
        cutlass_torch.dtype(get_cutlass_dtype(c_dtype))
    )
    res = c_torch.view(cutlass_torch.dtype(get_cutlass_dtype(c_dtype)))

    torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-03)


if __name__ == "__main__":
    test_blockwise_gemm_python_interface(
        lm=(1, 256),
        kn=(512, 256),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype="bfloat16",
        acc_dtype="float32",
        a_major="k",
        b_major="k",
        c_major="n",
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        tolerance=1e-01,
        iterations=3,
    )