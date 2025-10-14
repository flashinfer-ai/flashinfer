import math
import multiprocessing as mp
import pytest
import socket
from typing import Any, Tuple

import cutlass
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from flashinfer.cute_dsl.blockwise_gemm import (
    BlockwiseGemmKernel,
    blockwise_gemm,
)
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    get_num_sm,
    is_cute_dsl_available,
)


def create_mc_tensor(torch_tensor_cpu, dtype, is_dynamic_layout=True):
    m, n, l = torch_tensor_cpu.shape

    # Create flat symm_mem buffer
    total_elements = m * n * l
    torch_symm_flat = symm_mem.empty(
        (total_elements,), device="cuda", dtype=torch_tensor_cpu.dtype
    )

    # Reshape to match input's stride pattern using as_strided
    torch_symm_tensor = torch_symm_flat.as_strided(
        size=torch_tensor_cpu.shape, stride=torch_tensor_cpu.stride()
    )
    torch_symm_tensor.copy_(torch_tensor_cpu)

    symm = symm_mem.rendezvous(torch_symm_flat, group=dist.group.WORLD.group_name)
    mc_ptr = symm.multicast_ptr

    # Create MC tensor with same stride
    torch_tensor_mc_flat = cutlass_torch.as_tensor(
        mc_ptr, (total_elements,), torch_tensor_cpu.dtype
    )
    torch_tensor_mc = torch_tensor_mc_flat.as_strided(
        size=torch_tensor_cpu.shape, stride=torch_tensor_cpu.stride()
    )

    cute_tensor_mc = from_dlpack(torch_tensor_mc, assumed_align=16)

    if is_dynamic_layout:
        for i, stride in enumerate(torch_tensor_mc.stride()):
            if stride == 1:
                leading_dim = i
                break
        cute_tensor_mc = cute_tensor_mc.mark_layout_dynamic(leading_dim=leading_dim)

    torch_tensor_gpu = torch_symm_tensor
    cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
    cute_tensor.element_type = dtype

    if is_dynamic_layout:
        for i, stride in enumerate(torch_tensor_gpu.stride()):
            if stride == 1:
                leading_dim = i
                break
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    cute_tensor = cutlass_torch.convert_cute_tensor(
        torch_tensor_gpu,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )
    return cute_tensor, cute_tensor_mc, torch_tensor_gpu, torch_tensor_mc


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
        l,
        math.ceil(n / 128),
        math.ceil(k / 128),
        False,
        get_cutlass_dtype(scale_dtype),
        device=device,
    )

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_torch_cpu,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_torch_cpu,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    # c_tensor, c_torch = cutlass_torch.cute_tensor_like(
    #     c_torch_cpu,
    #     get_cutlass_dtype(c_dtype),
    #     is_dynamic_layout=True,
    #     assumed_align=16,
    # )
    c_tensor, c_tensor_mc, c_torch, c_torch_mc = create_mc_tensor(
        c_torch_cpu,
        get_cutlass_dtype(c_dtype),
        # (1 if c_major == "n" else 0),
        is_dynamic_layout=True,
    )
    sfa_tensor, sfa_torch = cutlass_torch.cute_tensor_like(
        sfa_torch_cpu,
        get_cutlass_dtype(scale_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    sfb_tensor, sfb_torch = cutlass_torch.cute_tensor_like(
        sfb_torch_cpu,
        get_cutlass_dtype(scale_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )

    return (
        a_tensor,
        a_torch,
        b_tensor,
        b_torch,
        c_tensor,
        c_torch,
        c_tensor_mc,
        c_torch_mc,
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


def create_barrier_flags(
    m, n, l, mma_tiler_mn, cluster_shape_mn, use_2cta_instrs, sm_count
):
    barrier_size = BlockwiseGemmKernel.compute_barrier_flag_size(
        m, n, l, mma_tiler_mn, cluster_shape_mn, use_2cta_instrs, sm_count
    )
    barrier_flag = symm_mem.empty((barrier_size,), device="cuda", dtype=torch.int32)

    barrier_flag.fill_(0)
    symm = symm_mem.rendezvous(barrier_flag, group=dist.group.WORLD.group_name)
    barrier_flag_mc_ptr = symm.multicast_ptr

    barrier_flag_memref = from_dlpack(barrier_flag)
    barrier_flag_memref = barrier_flag_memref.mark_layout_dynamic()
    barrier_flag_mc_torch = cutlass_torch.as_tensor(
        barrier_flag_mc_ptr, barrier_flag.shape, barrier_flag.dtype
    )
    barrier_flag_mc_memref = from_dlpack(
        barrier_flag_mc_torch,
    )
    barrier_flag_mc_memref = barrier_flag_mc_memref.mark_layout_dynamic()
    barrier_flag_torch = barrier_flag
    return (
        barrier_flag_memref,
        barrier_flag_mc_memref,
        barrier_flag_torch,
        barrier_flag_mc_torch,
    )


def run_blockwise_gemm_all_reduce_python_interface(
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
    all_reduce: str,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
):
    torch.manual_seed(42)
    device = torch.device("cuda", rank)
    major, minor = torch.cuda.get_device_capability(device)
    if not (major == 10 and minor == 0):
        pytest.skip("Cute-dsl backend is only supported on SM100.")

    l, m = lm
    k, n = kn
    sm_count = get_num_sm(device)

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
        all_reduce,
        group,
    ):
        pytest.skip(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {c_dtype}, {acc_dtype}, {use_2cta_instrs} ,{mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}, {all_reduce}"
        )

    (
        a_tensor,
        a_torch,
        b_tensor,
        b_torch,
        c_tensor,
        c_torch,
        c_tensor_mc,
        c_torch_mc,
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

    (
        barrier_flag_memref,
        barrier_flag_mc_memref,
        barrier_flag_torch,
        barrier_flag_mc_torch,
    ) = create_barrier_flags(
        m,
        n,
        l,
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs,
        sm_count,
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
            all_reduce="two_shot",
            out_mc=c_tensor_mc,
            out_mc_torch=c_torch_mc,
            barrier_flag=barrier_flag_memref,
            barrier_flag_mc=barrier_flag_mc_memref,
            barrier_flag_torch=barrier_flag_torch,
            barrier_flag_mc_torch=barrier_flag_mc_torch,
            process_group=group,
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

    updated_a = pad_and_multiply(sfa_torch_cpu, a_torch_cpu).to(
        cutlass_torch.dtype(get_cutlass_dtype(acc_dtype))
    )
    updated_b = pad_and_multiply(sfb_torch_cpu, b_torch_cpu).to(
        cutlass_torch.dtype(get_cutlass_dtype(acc_dtype))
    )

    ref = torch.einsum("mkl,nkl->mnl", updated_a, updated_b)
    # .to(
    #     cutlass_torch.dtype(get_cutlass_dtype(c_dtype))
    # )
    # ref = ref.contiguous()
    torch.distributed.all_reduce(
        ref, op=torch.distributed.ReduceOp.SUM, group=dist.group.WORLD
    )
    ref = ref.to(cutlass_torch.dtype(get_cutlass_dtype(c_dtype)))
    res = c_torch.view(cutlass_torch.dtype(get_cutlass_dtype(c_dtype)))

    torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-03)


def _run_correctness_worker(
    world_size,
    rank,
    distributed_init_port,
    lm,
    kn,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    a_major,
    b_major,
    c_major,
    use_2cta_instrs,
    mma_tiler_mn,
    cluster_shape_mn,
    tolerance,
    iterations,
    all_reduce,
):
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
        run_blockwise_gemm_all_reduce_python_interface(
            lm=lm,
            kn=kn,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            acc_dtype=acc_dtype,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            tolerance=tolerance,
            iterations=iterations,
            all_reduce=all_reduce,
            rank=rank,
            world_size=world_size,
            group=group,
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


@pytest.mark.skipif(
    not is_cute_dsl_available(), reason="Please `pip install nvidia-cutlass-dsl`"
)
@pytest.mark.parametrize("world_size", [8])
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
@pytest.mark.parametrize("all_reduce", ["two_shot"])
def test_cute_dsl_blockscaled_gemm_allreduce_two_shot(
    world_size,
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
    all_reduce,
):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )

    print(f"Running test for world_size={world_size}")
    multi_process_parallel(
        world_size,
        _run_correctness_worker,
        target_args=(
            lm,
            kn,
            ab_dtype,
            sf_dtype,
            c_dtype,
            acc_dtype,
            a_major,
            b_major,
            c_major,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            tolerance,
            iterations,
            all_reduce,
        ),
    )
    print(f"cute_dsl_blockwise_gemm_allreduce_two_shot on {world_size} GPUs: OK")
