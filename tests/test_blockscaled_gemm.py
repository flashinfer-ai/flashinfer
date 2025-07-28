import argparse
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from flashinfer.cute_dsl.blockscaled_gemm import (
    MaskedBatchedMatmulCuteDSL,  # python interface
)
from flashinfer.cute_dsl.blockscaled_gemm import (
    Sm100BlockScaledPersistentDenseGemmKernel,  # not used in python interface
)
from flashinfer.cute_dsl.blockscaled_gemm import (
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,  # not used in python interface
)
from flashinfer.cute_dsl.blockscaled_gemm import create_scale_factor_tensor


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    """Execute a persistent batched dense blockscaled GEMM operation on Blackwell architecture with performance benchmarking.

    This function prepares input tensors, configures and launches the persistent GEMM kernel,
    optionally performs reference validation, and benchmarks the execution performance.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: Data type for input tensors A and B
    :type ab_dtype: Type[cutlass.Numeric]
    :param sf_dtype: Data type for scale factor tensor
    :type sf_dtype: Type[cutlass.Numeric]
    :param sf_vec_size: Vector size for scale factor tensor
    :type sf_vec_size: int
    :param c_dtype: Data type for output tensor C
    :type c_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: Memory layout of tensor A/B/C
    :type a_major/b_major/c_major: str
    :param mma_tiler_mn: MMA tiling size.
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster shape.
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: Tolerance value for reference validation comparison, defaults to 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :raises RuntimeError: If CUDA GPU is not available
    :raises ValueError: If the configuration is invalid or unsupported by the kernel
    :return: Execution time of the GEMM kernel
    :rtype: float
    """
    print(f"Running Sm100 Persistent Dense BlockScaled GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # Unpack parameters
    m, n, k, l = mnkl

    # Skip unsupported testcase
    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
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
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create tensor A/B/C

    a_ref = cutlass_torch.matrix(l, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, c_major == "m", cutlass.Float32)

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    masked_m_tensor_torch = torch.full((l,), m, dtype=torch.int32, device="cuda")
    masked_m_tensor = from_dlpack(
        masked_m_tensor_torch, assumed_align=1
    ).mark_layout_dynamic(leading_dim=0)

    # Mark tensor to be byte aligned
    a_tensor.mark_compact_shape_dynamic(
        mode=1 if a_major == "k" else 0,
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1 if b_major == "k" else 0,
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
    )
    c_tensor.mark_compact_shape_dynamic(
        mode=1 if c_major == "n" else 0,
        stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
        divisibility=2 if c_dtype == cutlass.Float4E2M1FN else 1,
    )

    # Create scale factor tensor SFA/SFB
    def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        ref_shape = (l, mn, sf_k)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        ref_permute_order = (1, 2, 0)
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        # Create f32 ref torch tensor (cpu)
        ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            ref_shape,
            torch.float32,
            permute_order=ref_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=1,
                max_val=3,
            ),
        )

        # Create f32 cute torch tensor (cpu)
        cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.float32,
            permute_order=mma_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=0,
                max_val=1,
            ),
        )

        # convert ref f32 tensor to cute f32 tensor
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
        cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

        # reshape makes memory contiguous
        ref_f32_torch_tensor_cpu = (
            ref_f32_torch_tensor_cpu.permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(l, mn, sf_k, sf_vec_size)
            .reshape(l, mn, sf_k * sf_vec_size)
            .permute(*ref_permute_order)
        )
        # prune to mkl for reference check.
        ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

        # Create dtype cute torch tensor (cpu)
        cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
            cute_f32_torch_tensor_cpu,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        # Convert f32 cute tensor to dtype cute tensor
        cute_tensor = cutlass_torch.convert_cute_tensor(
            cute_f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=True,
        )
        return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, sf_vec_size, sf_dtype
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, sf_dtype
    )

    # Configure gemm kernel
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    print(hardware_info)
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    print("a_tensor: ", a_tensor)
    print("b_tensor: ", b_tensor)
    print("c_tensor: ", c_tensor)
    print("sfa_tensor: ", sfa_tensor)
    print("sfb_tensor: ", sfb_tensor)
    print("masked_m_tensor: ", masked_m_tensor)

    # Compile gemm kernel
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        c_tensor,
        masked_m_tensor,
        max_active_clusters,
        current_stream,
    )

    # Compute reference result
    if not skip_ref_check:
        # Execute kernel once for reference checking
        compiled_gemm(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            masked_m_tensor,
            current_stream,
        )
        print("Verifying results...")
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

        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # Convert ref : f32 -> f8 -> f32
            ref_f8_ = torch.empty(*(l, m, n), dtype=torch.uint8, device="cuda").permute(
                1, 2, 0
            )
            ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            ref_f8.element_type = c_dtype
            ref_device = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            ref = ref_device.cpu()
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + sfa_torch.numel() * sfa_torch.element_size()
            + sfb_torch.numel() * sfb_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )


# run tests provided by cutlass example
# SHOULD BE REMOVED LATER
@pytest.mark.parametrize("mnkl", [(1500, 2048, 2048, 100)])
@pytest.mark.parametrize("ab_dtype", [cutlass.Float4E2M1FN])
@pytest.mark.parametrize("sf_dtype", [cutlass.Float8E8M0FNU])
@pytest.mark.parametrize("sf_vec_size", [16])
@pytest.mark.parametrize("c_dtype", [cutlass.Float16])
@pytest.mark.parametrize("a_major", ["k"])
@pytest.mark.parametrize("b_major", ["k"])
@pytest.mark.parametrize("c_major", ["n"])
@pytest.mark.parametrize("mma_tiler_mn", [(128, 128)])
@pytest.mark.parametrize("cluster_shape_mn", [(1, 1)])
@pytest.mark.parametrize("tolerance", [1e-01])
@pytest.mark.parametrize("warmup_iterations", [0])
@pytest.mark.parametrize("iterations", [1])
@pytest.mark.parametrize("skip_ref_check", [False])
@pytest.mark.parametrize("use_cold_l2", [False])
def test_blockscaled_gemm(
    mnkl: Tuple[int, int, int, int],
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
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool,
):
    m, n, k, l = mnkl
    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
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

    run(
        mnkl,
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        a_major,
        b_major,
        c_major,
        mma_tiler_mn,
        cluster_shape_mn,
        tolerance,
        warmup_iterations,
        iterations,
        skip_ref_check,
        use_cold_l2,
    )


# todo(Yingyi): complete this test for target python interface
@pytest.mark.parametrize("mnkl", [(1500, 2048, 2048, 100)])
@pytest.mark.parametrize("ab_dtype", [cutlass.Float4E2M1FN])
@pytest.mark.parametrize("sf_dtype", [cutlass.Float8E8M0FNU])
@pytest.mark.parametrize("sf_vec_size", [16])
@pytest.mark.parametrize("c_dtype", [cutlass.Float16])
@pytest.mark.parametrize("a_major", ["k"])
@pytest.mark.parametrize("b_major", ["k"])
@pytest.mark.parametrize("c_major", ["n"])
@pytest.mark.parametrize("mma_tiler_mn", [(128, 128)])
@pytest.mark.parametrize("cluster_shape_mn", [(1, 1)])
@pytest.mark.parametrize("tolerance", [1e-01])
def test_blockscaled_gemm_python_interface(
    mnkl: Tuple[int, int, int, int],
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
):
    m, n, k, l = mnkl

    # Create tensors on GPU first to initialize CUDA context before plan
    # 1. Create torch tensors using size fp32 and cast to torch_dtype
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
            cutlass.Float16: torch.float16,
            cutlass.Float8E5M2: torch.float8_e5m2,
            cutlass.Float8E4M3FN: torch.float8_e4m3fn,
            cutlass.Float8E4M3B11FNUZ: torch.float8_e4m3fnuz,
            cutlass.Float4E2M1FN: torch.int8,
        }
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)

        if cutlass_dtype == cutlass.Float4E2M1FN:
            mode0 = mode0 // 2 if is_mode0_major else mode0
            mode1 = mode1 if is_mode0_major else mode1 // 2

        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        # fp32_torch_tensor = torch.empty(*shape, dtype=torch.float32, device=device)
        # fp32_torch_tensor = fp32_torch_tensor.permute(permute_order)
        # dtype_torch_tensor = fp32_torch_tensor.to(dtype=torch_type_map[cutlass_dtype])
        dtype_torch_tensor = torch.empty(
            *shape, dtype=torch_type_map[cutlass_dtype], device=device
        )
        # todo(Yingyi): add init value
        dtype_torch_tensor = dtype_torch_tensor.permute(permute_order)

        return dtype_torch_tensor

    # create helper tensors for testing
    # todo(Yingyi): use int8 and 1/2 shape for fp4？
    a_tensor_gpu = create_torch_tensor(
        l, m, k, a_major == "m", cutlass.Float4E2M1FN, "cuda"
    )
    b_tensor_gpu = create_torch_tensor(
        l, n, k, b_major == "n", cutlass.Float4E2M1FN, "cuda"
    )
    c_tensor_gpu = create_torch_tensor(l, m, n, c_major == "m", cutlass.Float16, "cuda")
    _, _, sfa_tensor_gpu = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
    _, _, sfb_tensor_gpu = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)
    masked_m_tensor_gpu = torch.full((l,), m, dtype=torch.int32, device="cuda")

    wrapper = MaskedBatchedMatmulCuteDSL(use_cuda_graph=False)
    wrapper.plan(
        m=m,
        n=n,
        k=k,
        l=l,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        c_dtype=c_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    c = wrapper.run(
        a_tensor_gpu,
        b_tensor_gpu,
        sfa_tensor_gpu,
        sfb_tensor_gpu,
        c_tensor_gpu,
        masked_m_tensor_gpu,
    )
    print("PASS")

    # todo(Yingyi): add reference check


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Example of Sm100 Dense Persistent BlockScaled GEMM."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(1500, 2048, 2048, 100),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E8M0FNU)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    run(
        args.mnkl,
        args.ab_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
