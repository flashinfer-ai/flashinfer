import logging
import multiprocessing as mp
import pytest
import socket
from typing import Any, Tuple, Type

try:
    # cuda-python >= 12.9 (has cuda.bindings.driver)
    from cuda.bindings import driver as cuda
except ImportError:
    try:
        # cuda-python <= 12.9 (no cuda.bindings.driver, use cuda as driver)
        # from cuda import cuda is not available in cuda-python >= 13.0
        from cuda import cuda
    except ImportError as e:
        raise ImportError(
            "Could not import the 'cuda' module. "
            "Please install cuda-python that matches your CUDA version."
        ) from e

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch
import cutlass.utils as utils

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from flashinfer.cute_dsl.gemm_allreduce_two_shot import PersistentDenseGemmKernel
from flashinfer.utils import get_compute_capability


logger = logging.getLogger(__name__)


def create_mc_tensor(torch_tensor_cpu, dtype, leading_dim, is_dynamic_layout=True):
    torch_symm_tensor = symm_mem.empty(
        torch_tensor_cpu.shape, device="cuda", dtype=torch_tensor_cpu.dtype
    )
    torch_symm_tensor.copy_(torch_tensor_cpu)
    symm = symm_mem.rendezvous(torch_symm_tensor, group=dist.group.WORLD.group_name)
    mc_ptr = symm.multicast_ptr
    # create MC tensor memref
    cute_tensor_mc = from_dlpack(
        cutlass_torch.as_tensor(mc_ptr, torch_tensor_cpu.shape, torch_tensor_cpu.dtype),
        assumed_align=16,
    )
    if is_dynamic_layout:
        cute_tensor_mc = cute_tensor_mc.mark_layout_dynamic(leading_dim=leading_dim)
    torch_tensor_gpu = torch_symm_tensor
    cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
    cute_tensor.element_type = dtype
    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    cute_tensor = cutlass_torch.convert_cute_tensor(
        torch_tensor_gpu,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )
    return cute_tensor, cute_tensor_mc, torch_tensor_gpu


def create_tensors(
    l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype, is_all_reduce=False
):
    torch.manual_seed(1111)

    a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major == "m", ab_dtype)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", ab_dtype)
    c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major == "m", c_dtype)

    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    c_tensor_mc = None
    if is_all_reduce:
        c_tensor, c_tensor_mc, c_torch_gpu = create_mc_tensor(
            c_torch_cpu, c_dtype, (1 if c_major == "n" else 0), is_dynamic_layout=True
        )

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        c_tensor_mc,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        c_torch_gpu,
    )


def compare(
    a_torch_cpu, b_torch_cpu, c_torch_gpu, c_dtype, tolerance, do_all_reduce=False
):
    # Copy gpu result back
    kernel_result = c_torch_gpu.cpu()

    # Compute reference result
    ref = torch.einsum(
        "mkl,nkl->mnl",
        a_torch_cpu.to(dtype=torch.float32),
        b_torch_cpu.to(dtype=torch.float32),
    )

    if c_dtype == cutlass.Float8E5M2 or c_dtype == cutlass.Float8E4M3FN:
        ref = ref.to(torch.float16).cuda()
    else:
        ref = ref.cuda()
    if do_all_reduce:
        torch.distributed.all_reduce(ref, op=torch.distributed.ReduceOp.SUM)
    if c_dtype == cutlass.Float8E5M2:
        ref = ref.to(torch.float8_e5m2fnuz)
    elif c_dtype == cutlass.Float8E4M3FN:
        ref = ref.to(torch.float8_e4m3fn)

    # Convert ref to c_dtype
    _, ref_torch_gpu = cutlass_torch.cute_tensor_like(
        ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    ref_result = ref_torch_gpu.cpu()

    # Assert close results
    torch.testing.assert_close(kernel_result, ref_result, atol=tolerance, rtol=1e-05)


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    use_tma_store: bool = True,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    all_reduce: str = "none",
    **kwargs,
):
    """Execute a persistent batched dense GEMM operation on Blackwell architecture with performance benchmarking.

    This function prepares input tensors, configures and launches the persistent GEMM kernel,
    optionally performs reference validation, and benchmarks the execution performance.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: Data type for input tensors A and B
    :type ab_dtype: Type[cutlass.Numeric]
    :param c_dtype: Data type for output tensor C
    :type c_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Data type for accumulation during matrix multiplication
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: Memory layout of tensor A/B/C
    :type a_major/b_major/c_major: str
    :param mma_tiler_mn: MMA tiling size. If not specified in the decorator parameters, the autotuner will use the
        default value of (256, 256). Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type mma_tiler_mn: Tuple[int, int], optional
    :param cluster_shape_mn: Cluster shape. If not specified in the decorator parameters, the autotuner will use the
        default value of (2, 1). Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type cluster_shape_mn: Tuple[int, int], optional
    :param use_2cta_instrs: Whether to use 2CTA instructions. If not specified in the decorator parameters, the autotuner
        will use the default value of True. Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type use_2cta_instrs: bool, optional
    :param use_tma_store: Whether to use TMA store. If not specified in the decorator parameters, the autotuner will use
        the default value of True. Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type use_tma_store: bool, optional
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
    :param all_reduce: All-reduce mode, can be "none", "two_shot"
    :type all_reduce: str, optional
    :raises RuntimeError: If CUDA GPU is not available
    :raises ValueError: If the configuration is invalid or unsupported by the kernel
    :return: Execution time of the GEMM kernel
    :rtype: float
    """
    print("Running Blackwell Persistent Dense GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Use TMA Store: {'True' if use_tma_store else 'False'}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")
    print(f"Fused AllReduce Op: {all_reduce}")

    # Unpack parameters
    m, n, k, l = mnkl
    can_implement = PersistentDenseGemmKernel.can_implement(
        ab_dtype,
        acc_dtype,
        c_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
        all_reduce,
    )
    if not can_implement:
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {acc_dtype}, {c_dtype}, {use_2cta_instrs}, {mma_tiler_mn}, {cluster_shape_mn}, {use_tma_store}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    (
        a_tensor,
        b_tensor,
        c_tensor,
        c_tensor_mc,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        c_torch_gpu,
    ) = create_tensors(
        l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype, all_reduce != "none"
    )

    # Build GEMM object
    gemm = PersistentDenseGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        all_reduce=all_reduce,
    )

    if not can_implement:
        raise ValueError(
            f"The current config which is invalid/unsupported: use_2cta_instrs = {use_2cta_instrs}, "
            f"mma_tiler_mn = {mma_tiler_mn}, cluster_shape_mn = {cluster_shape_mn}, "
            f"use_tma_store = {use_tma_store}"
        )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    def create_barrier_flags():
        cta_tile_shape_mn = (
            mma_tiler_mn[0] // (2 if use_2cta_instrs else 1),
            mma_tiler_mn[1],
        )
        problem_shape_ntile_mn = (m // cta_tile_shape_mn[0], n // cta_tile_shape_mn[1])
        num_tiles = problem_shape_ntile_mn[0] * problem_shape_ntile_mn[1]
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

        # +num_sms for final barrier
        barrier_flag = symm_mem.empty(
            (num_tiles + num_sms,), device="cuda", dtype=torch.int32
        )
        barrier_flag.fill_(0)
        symm = symm_mem.rendezvous(barrier_flag, group=dist.group.WORLD.group_name)
        barrier_flag_mc = symm.multicast_ptr

        barrier_flag_memref = from_dlpack(barrier_flag)
        barrier_flag_memref = barrier_flag_memref.mark_layout_dynamic()
        barrier_flag_mc_memref = from_dlpack(
            cutlass_torch.as_tensor(
                barrier_flag_mc, barrier_flag.shape, barrier_flag.dtype
            ),
        )
        barrier_flag_mc_memref = barrier_flag_mc_memref.mark_layout_dynamic()

        return barrier_flag_memref, barrier_flag_mc_memref

    compiled_gemm = None
    if all_reduce == "none":
        compiled_gemm = cute.compile(
            gemm, a_tensor, b_tensor, c_tensor, max_active_clusters, current_stream
        )
    else:
        barrier_flag_memref, barrier_flag_mc_memref = create_barrier_flags()
        compiled_gemm = cute.compile(
            gemm,
            a_tensor,
            b_tensor,
            c_tensor,
            max_active_clusters,
            current_stream,
            c_mc=c_tensor_mc,
            barrier_flag=barrier_flag_memref,
            barrier_flag_mc=barrier_flag_mc_memref,
        )

    if not skip_ref_check:
        if all_reduce == "none":
            compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
        else:
            compiled_gemm(
                a_tensor,
                b_tensor,
                c_tensor,
                current_stream,
                c_mc=c_tensor_mc,
                barrier_flag=barrier_flag_memref,
                barrier_flag_mc=barrier_flag_mc_memref,
            )
        compare(
            a_torch_cpu,
            b_torch_cpu,
            c_torch_gpu,
            c_dtype,
            tolerance,
            do_all_reduce=all_reduce != "none",
        )

    def generate_tensors():
        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        if all_reduce != "none":
            c_tensor, c_tensor_mc, _ = create_mc_tensor(
                c_torch_cpu,
                c_dtype,
                (1 if c_major == "n" else 0),
                is_dynamic_layout=True,
            )
            barrier_flag_memref, barrier_flag_mc_memref = create_barrier_flags()
            return testing.JitArguments(
                a_tensor,
                b_tensor,
                c_tensor,
                current_stream,
                c_mc=c_tensor_mc,
                barrier_flag=barrier_flag_memref,
                barrier_flag_mc=barrier_flag_mc_memref,
            )
        else:
            return testing.JitArguments(a_tensor, b_tensor, c_tensor, current_stream)

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch_cpu.numel() * a_torch_cpu.element_size()
            + b_torch_cpu.numel() * b_torch_cpu.element_size()
            + c_torch_cpu.numel() * c_torch_cpu.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    if dist.get_rank() == 0:
        print(f"exec_time: {exec_time}")

    return exec_time  # Return execution time in microseconds


def _run_correctness_worker(world_size, rank, distributed_init_port):
    assert rank >= 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        init_method=distributed_init_method,
    )
    group = dist.group.WORLD
    rank_id = torch.distributed.get_rank()

    try:
        run(
            mnkl=(2048, 2048, 4096, 1),
            ab_dtype=cutlass.TFloat32,
            c_dtype=cutlass.Float32,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
            use_tma_store=False,
            tolerance=1e-01,
            warmup_iterations=0,
            iterations=1,
            skip_ref_check=False,
            use_cold_l2=False,
            all_reduce="two_shot",
        )
    except Exception as e:
        print(f"Rank {rank_id}: Exception during test: {e}")
        raise
    finally:
        torch.distributed.barrier(group)
        torch.distributed.destroy_process_group(group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


@pytest.mark.parametrize("world_size", [8])
def test_cute_dsl_gemm_allreduce_two_shot(world_size):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )

    if get_compute_capability(torch.device("cuda")) != (10, 0):
        pytest.skip("cute_dsl_gemm_allreduce_two_shot requires SM100")

    print(f"Running test for world_size={world_size}")
    multi_process_parallel(
        world_size,
        _run_correctness_worker,
        target_args=(),
    )
    print(f"cute_dsl_gemm_allreduce_two_shot on {world_size} GPUs: OK")
