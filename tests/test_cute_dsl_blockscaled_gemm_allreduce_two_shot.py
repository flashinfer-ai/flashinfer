import multiprocessing as mp
import pytest
import socket
from typing import Any, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from flashinfer.cute_dsl.blockscaled_gemm import (
    Sm100BlockScaledPersistentDenseGemmKernel,  # not used in python interface
    grouped_gemm_nt_masked,  # deepgemm-like python interface for DLFW integration
    create_scale_factor_tensor,
)
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
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


def create_barrier_flags(m, n, l, mma_tiler_mn, cluster_shape_mn, sm_count):
    barrier_size = Sm100BlockScaledPersistentDenseGemmKernel.compute_barrier_flag_size(
        m, n, l, mma_tiler_mn, cluster_shape_mn, sm_count
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


def run_blockscaled_gemm_all_reduce_python_interface(
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
    enable_dst_signals: int,
    all_reduce: str,
    rank: int,
    world_size: int,
):
    torch.manual_seed(42)
    device = torch.device("cuda", rank)
    major, minor = torch.cuda.get_device_capability(device)

    if not (major == 10 and minor == 0):
        pytest.skip("Cute-dsl backend is only supported on SM100.")
    if enable_dst_signals and (sm_count is None):
        pytest.skip("dst_signals require sm_count")

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
        l,
        m,
        n,
        c_major == "m",
        cutlass.Float32,
        device=device,
        init_type=cutlass_torch.TensorInitType.SCALAR,
        init_config=cutlass_torch.ScalarInitConfig(value=0.0),
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
    c_tensor, c_tensor_mc, c_torch, c_torch_mc = create_mc_tensor(
        c_ref,
        get_cutlass_dtype(c_dtype),
        # (1 if c_major == "n" else 0),
        is_dynamic_layout=True,
    )
    alpha_tensor = (
        torch.randn(l, dtype=torch.float32, device=device) if fuse_alpha else None
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
        sm_count,
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
    if rank == 0:
        masked_m_tensor = torch.randint(0, m, (l,), dtype=torch.int32, device=device)
    else:
        masked_m_tensor = torch.empty((l,), dtype=torch.int32, device=device)
    torch.distributed.broadcast(masked_m_tensor, src=0)
    for _ in range(iterations):
        dst_signals = (
            torch.zeros((l,), dtype=torch.uint32, device="cuda")
            if enable_dst_signals
            else None
        )

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
            dst_signals=dst_signals,
            all_reduce=all_reduce,
            out_mc=c_tensor_mc,
            out_mc_torch=c_torch_mc,
            barrier_flag=barrier_flag_memref,
            barrier_flag_mc=barrier_flag_mc_memref,
            barrier_flag_torch=barrier_flag_torch,
            barrier_flag_mc_torch=barrier_flag_mc_torch,
        )

        if enable_dst_signals:
            assert torch.all(dst_signals == sm_count), f"{dst_signals}"

    # compute ref output
    if not fuse_alpha:
        alpha_tensor = torch.ones(l, dtype=torch.float32, device=device)
    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
    ref = torch.einsum("mnl,l->mnl", ref, alpha_tensor)
    ref = ref.contiguous()
    torch.distributed.all_reduce(
        ref, op=torch.distributed.ReduceOp.SUM, group=dist.group.WORLD
    )
    # Convert c back to f32 for comparison.
    ref = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0)
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


def _run_correctness_worker(
    world_size,
    rank,
    distributed_init_port,
    lm,
    kn,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    c_dtype,
    a_major,
    b_major,
    c_major,
    fuse_alpha,
    alpha_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sm_count,
    tolerance,
    iterations,
    enable_dst_signals,
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
        run_blockscaled_gemm_all_reduce_python_interface(
            lm=lm,
            kn=kn,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            sf_vec_size=sf_vec_size,
            c_dtype=c_dtype,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            fuse_alpha=fuse_alpha,
            alpha_dtype=alpha_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            tolerance=tolerance,
            iterations=iterations,
            sm_count=sm_count,
            enable_dst_signals=enable_dst_signals,
            all_reduce=all_reduce,
            rank=rank,
            world_size=world_size,
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
@pytest.mark.parametrize("lm", [(1, 1024), (2, 512), (4, 256)])
@pytest.mark.parametrize("kn", [(7168, 4096), (2048, 7168)])
@pytest.mark.parametrize(
    "ab_dtype,sf_dtype,c_dtype,sf_vec_size",
    [
        ("float8_e5m2", "float8_e8m0fnu", "bfloat16", 32),
        ("float4_e2m1fn", "float8_e8m0fnu", "float16", 16),
        ("float4_e2m1fn", "float8_e8m0fnu", "bfloat16", 16),
        ("float4_e2m1fn", "float8_e8m0fnu", "float32", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "float16", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "bfloat16", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "float32", 16),
        ("float8_e4m3fn", "float8_e8m0fnu", "bfloat16", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float16", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float32", 32),
        # ("float8_e4m3fn", "float8_e8m0fnu", "float8_e4m3fn", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float8_e5m2", 32),
        ("float8_e5m2", "float8_e8m0fnu", "bfloat16", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float16", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float32", 32),
        # ("float8_e5m2", "float8_e8m0fnu", "float8_e4m3fn", 32),
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
@pytest.mark.parametrize("sm_count", [148])
@pytest.mark.parametrize("tolerance", [1e-01])
@pytest.mark.parametrize("iterations", [1])
@pytest.mark.parametrize("enable_dst_signals", [False, True])
@pytest.mark.parametrize("all_reduce", ["two_shot"])
def test_cute_dsl_blockscaled_gemm_allreduce_two_shot(
    world_size,
    lm,
    kn,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    c_dtype,
    a_major,
    b_major,
    c_major,
    fuse_alpha,
    alpha_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sm_count,
    tolerance,
    iterations,
    enable_dst_signals,
    all_reduce,
):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    major, minor = torch.cuda.get_device_capability(torch.device("cuda:0"))
    if not (major == 10 and minor == 0):
        pytest.skip("Cute-dsl backend is only supported on SM100.")
    if enable_dst_signals and (sm_count is None):
        pytest.skip("dst_signals require sm_count")

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
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(
        world_size,
        _run_correctness_worker,
        target_args=(
            lm,
            kn,
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            a_major,
            b_major,
            c_major,
            fuse_alpha,
            alpha_dtype,
            mma_tiler_mn,
            cluster_shape_mn,
            sm_count,
            tolerance,
            iterations,
            enable_dst_signals,
            all_reduce,
        ),
    )
    print(f"cute_dsl_blockscaled_gemm_allreduce_two_shot on {world_size} GPUs: OK")
