import os
import socket

import pytest
import torch
import torch.distributed as dist
from mpi4py import MPI

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend
from tests.test_helpers.comm import init_torch_distributed_from_mpi


FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
# E4M3FN's smallest positive subnormal is 2^-9.
FP8_MIN_SUBNORMAL = 2.0**-9
MIN_FP8_SCALE = FP8_MIN_SUBNORMAL / FP8_MAX


def _prefer_mpi_launcher_env() -> None:
    """Prefer MPI rank variables over Slurm rank variables for this test."""
    mpi_comm = MPI.COMM_WORLD
    if mpi_comm.Get_size() <= 1:
        return

    local_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
    os.environ["RANK"] = str(mpi_comm.Get_rank())
    os.environ["WORLD_SIZE"] = str(mpi_comm.Get_size())
    os.environ["LOCAL_RANK"] = str(local_comm.Get_rank())
    if "MASTER_ADDR" not in os.environ:
        master_addr = mpi_comm.bcast(
            socket.gethostname() if mpi_comm.Get_rank() == 0 else None, root=0
        )
        os.environ["MASTER_ADDR"] = master_addr
    local_comm.Free()

    for key in (
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_LOCALID",
        "SLURM_NODEID",
    ):
        os.environ.pop(key, None)


def _reference(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute torch references for MNNVL RMSNorm and dynamic FP8 quant."""
    reduced = allreduce_in.clone()
    dist.all_reduce(reduced)
    residual_out = reduced + residual
    norm_out = torch.nn.functional.rms_norm(
        residual_out, (residual_out.shape[-1],), weight=weight, eps=eps
    )
    scale = torch.clamp(
        norm_out.float().abs().amax(dim=-1, keepdim=True) / FP8_MAX,
        min=MIN_FP8_SCALE,
    )
    quant = (
        (norm_out.float() / scale)
        .clamp(
            min=float(torch.finfo(torch.float8_e4m3fn).min),
            max=float(torch.finfo(torch.float8_e4m3fn).max),
        )
        .to(torch.float8_e4m3fn)
    )
    return residual_out, norm_out, quant, scale


def _assert_dynamic_fp8_dequant_close(
    quant_out: torch.Tensor,
    scale_out: torch.Tensor,
    ref_quant: torch.Tensor,
    ref_scale: torch.Tensor,
) -> None:
    """Compare dequantized dynamic FP8 outputs with scale-aware tolerances."""
    scale_max = float(ref_scale.max().item())
    torch.testing.assert_close(
        quant_out.float() * scale_out,
        ref_quant.float() * ref_scale,
        atol=scale_max * 0.05,
        rtol=0.05,
    )


@pytest.mark.parametrize("use_oneshot", [True, False])
@pytest.mark.parametrize("with_norm_out", [False, True])
def test_mnnvl_allreduce_dynamic_fp8_quant(
    use_oneshot: bool,
    with_norm_out: bool,
) -> None:
    """Validate MNNVL dynamic FP8 fusion correctness."""
    if torch.cuda.device_count() == 0:
        pytest.skip("requires CUDA")

    _prefer_mpi_launcher_env()
    init_torch_distributed_from_mpi()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip(f"requires at least 2 ranks, got {world_size}")

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    M, H = (16, 4096) if use_oneshot else (128, 4096)
    dtype = torch.bfloat16
    eps = 1e-6

    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=world_size,
            rank=rank,
            max_token_num=M,
            hidden_dim=H,
            dtype=dtype,
            gpus_per_node=torch.cuda.device_count(),
            comm_backend=TorchDistBackend(),
            force_oneshot_support=use_oneshot,
        )
        torch.manual_seed(1234 + rank)
        allreduce_in = torch.randn(M, H, device="cuda", dtype=dtype)
        residual = torch.empty(M, H, device="cuda", dtype=dtype)
        weight = torch.empty(H, device="cuda", dtype=dtype)
        if rank == 0:
            residual.copy_(torch.randn_like(residual))
            weight.copy_(torch.randn_like(weight))
        dist.broadcast(residual, src=0)
        dist.broadcast(weight, src=0)

        residual_out = torch.empty_like(allreduce_in)
        norm_out = torch.empty_like(allreduce_in) if with_norm_out else None
        quant_out = torch.empty_like(allreduce_in, dtype=torch.float8_e4m3fn)
        scale_out = torch.empty(M, 1, device="cuda", dtype=torch.float32)
        pattern = (
            comm.AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant
            if with_norm_out
            else comm.AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant
        )

        comm.allreduce_fusion(
            input=allreduce_in,
            workspace=workspace,
            pattern=pattern,
            launch_with_pdl=False,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=weight,
            rms_eps=eps,
            use_oneshot=use_oneshot,
        )
        torch.cuda.synchronize()

        ref_residual, ref_norm, ref_quant, ref_scale = _reference(
            allreduce_in, residual, weight, eps
        )
        torch.testing.assert_close(
            residual_out.float(), ref_residual.float(), atol=8e-1, rtol=1e-2
        )
        torch.testing.assert_close(scale_out, ref_scale, atol=1e-3, rtol=1e-3)
        _assert_dynamic_fp8_dequant_close(quant_out, scale_out, ref_quant, ref_scale)
        if norm_out is not None:
            torch.testing.assert_close(
                norm_out.float(),
                ref_norm.float(),
                atol=8e-1,
                rtol=1e-2,
            )
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.barrier()


@pytest.mark.parametrize("use_oneshot", [True, False])
def test_mnnvl_allreduce_dynamic_fp8_cuda_graph_replay(use_oneshot: bool) -> None:
    """Validate MNNVL dynamic FP8 CUDA Graph replay."""
    if torch.cuda.device_count() == 0:
        pytest.skip("requires CUDA")

    _prefer_mpi_launcher_env()
    init_torch_distributed_from_mpi()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip(f"requires at least 2 ranks, got {world_size}")

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    M, H = (16, 4096) if use_oneshot else (128, 4096)
    dtype = torch.bfloat16
    eps = 1e-6

    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=world_size,
            rank=rank,
            max_token_num=M,
            hidden_dim=H,
            dtype=dtype,
            gpus_per_node=torch.cuda.device_count(),
            comm_backend=TorchDistBackend(),
            force_oneshot_support=use_oneshot,
        )
        torch.manual_seed(4321 + rank)
        static_input = torch.empty(M, H, device="cuda", dtype=dtype)
        residual = torch.empty(M, H, device="cuda", dtype=dtype)
        weight = torch.empty(H, device="cuda", dtype=dtype)
        if rank == 0:
            residual.copy_(torch.randn_like(residual))
            weight.copy_(torch.randn_like(weight))
        dist.broadcast(residual, src=0)
        dist.broadcast(weight, src=0)

        residual_out = torch.empty_like(static_input)
        quant_out = torch.empty_like(static_input, dtype=torch.float8_e4m3fn)
        scale_out = torch.empty(M, 1, device="cuda", dtype=torch.float32)

        def run_op() -> None:
            comm.allreduce_fusion(
                input=static_input,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant,
                launch_with_pdl=False,
                residual_in=residual,
                residual_out=residual_out,
                quant_out=quant_out,
                scale_out=scale_out,
                rms_gamma=weight,
                rms_eps=eps,
                use_oneshot=use_oneshot,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                static_input.copy_(torch.randn_like(static_input))
                run_op()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        static_input.copy_(torch.randn_like(static_input))
        with torch.cuda.graph(graph):
            run_op()
        torch.cuda.synchronize()

        for step in range(2):
            torch.manual_seed(8000 + step * world_size + rank)
            replay_input = torch.randn(M, H, device="cuda", dtype=dtype)
            static_input.copy_(replay_input)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()

            ref_residual, _, ref_quant, ref_scale = _reference(
                replay_input, residual, weight, eps
            )
            torch.testing.assert_close(
                residual_out.float(), ref_residual.float(), atol=8e-1, rtol=1e-2
            )
            torch.testing.assert_close(scale_out, ref_scale, atol=1e-3, rtol=1e-3)
            _assert_dynamic_fp8_dequant_close(
                quant_out, scale_out, ref_quant, ref_scale
            )
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.barrier()
