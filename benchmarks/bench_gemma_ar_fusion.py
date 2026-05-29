"""Benchmark: AllReduce + Gemma RMSNorm — fused vs unfused.

Compares the perf of two equivalent paths for Qwen3.5 / Gemma tensor-parallel
RMSNorm:

  Fused:    flashinfer.comm.allreduce_fusion(pattern=kARResidualRMSNorm,
                                             weight_bias=1.0)
  Unfused:  torch.distributed.all_reduce(input) + residual add
            + flashinfer.norm.gemma_fused_add_rmsnorm

Launch with mpirun for the rank count you care about, e.g.:

  mpirun -np 2 python benchmarks/bench_gemma_ar_fusion.py \\
      --hidden-sizes 2048 4096 8192 --num-tokens 16 128 1024

(OpenMPI as root: add `--allow-run-as-root`. The flag is rejected by MPICH.)
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.norm import gemma_fused_add_rmsnorm
from flashinfer.testing.utils import bench_gpu_time


def _init_distributed() -> tuple[int, int, int]:
    """Stand up a TorchDist NCCL group via mpi4py (works for OpenMPI + MPICH)."""
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()
    local_rank = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    return rank, world_size, local_rank


def _bench_fused(workspace, x, residual, rms_gamma, rms_eps, num_iters, dry_run_iters):
    norm_out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    def _run(inp):
        comm.allreduce_fusion(
            input=inp,
            workspace=workspace,
            pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            weight_bias=1.0,
        )
        return norm_out

    return bench_gpu_time(
        fn=_run,
        input_args=(x,),
        dry_run_iters=dry_run_iters,
        repeat_iters=num_iters,
        sleep_after_run=False,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )


def _bench_unfused(x, residual, rms_gamma, rms_eps, group, num_iters, dry_run_iters):
    # Pre-allocate scratch so we don't time allocation. The standalone Gemma
    # kernel mutates input and residual in place; allocate fresh copies each
    # call (matches how an inference loop would feed it).
    scratch_x = torch.empty_like(x)
    scratch_r = torch.empty_like(residual)

    def _run(inp):
        scratch_x.copy_(inp)
        scratch_r.copy_(residual)
        dist.all_reduce(scratch_x, group=group)
        gemma_fused_add_rmsnorm(scratch_x, scratch_r, rms_gamma, eps=rms_eps)
        return scratch_x

    return bench_gpu_time(
        fn=_run,
        input_args=(x,),
        dry_run_iters=dry_run_iters,
        repeat_iters=num_iters,
        sleep_after_run=False,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-sizes", nargs="+", type=int, default=[2048, 4096, 8192]
    )
    parser.add_argument("--num-tokens", nargs="+", type=int, default=[16, 128, 1024])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--max-token-num", type=int, default=2048)
    args = parser.parse_args()

    rank, world_size, _ = _init_distributed()
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    rms_eps = 1e-6

    # Build one workspace sized for the largest config we'll exercise.
    max_hidden = max(args.hidden_sizes)
    workspace = comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=world_size,
        rank=rank,
        max_token_num=args.max_token_num,
        hidden_dim=max_hidden,
        dtype=dtype,
        comm_backend=TorchDistBackend(),
    )

    if rank == 0:
        print(
            f"\n=== Gemma AR-Fusion Benchmark (TP={world_size}, dtype={args.dtype}) ===\n"
        )
        print(
            f"{'tokens':>8} {'hidden':>8} {'fused_us':>12} {'unfused_us':>12} "
            f"{'speedup':>10}"
        )
        print("-" * 56)

    try:
        for tok in args.num_tokens:
            for hidden in args.hidden_sizes:
                if hidden > max_hidden:
                    continue

                torch.manual_seed(42)
                rms_gamma = torch.randn(hidden, dtype=dtype, device=device)
                torch.manual_seed(42 + rank)
                x = torch.randn(tok, hidden, dtype=dtype, device=device)
                residual = torch.randn(tok, hidden, dtype=dtype, device=device)

                # Synchronize across ranks for fair timing.
                dist.barrier()
                torch.cuda.synchronize()

                fused_times = _bench_fused(
                    workspace,
                    x,
                    residual,
                    rms_gamma,
                    rms_eps,
                    args.num_iters,
                    args.dry_run_iters,
                )
                dist.barrier()

                unfused_times = _bench_unfused(
                    x,
                    residual,
                    rms_gamma,
                    rms_eps,
                    dist.group.WORLD,
                    args.num_iters,
                    args.dry_run_iters,
                )
                dist.barrier()

                # Use max across ranks (sync collectives) → median across iters.
                # bench_gpu_time returns per-iter time in milliseconds.
                fused_local_ms = np.median(fused_times)
                unfused_local_ms = np.median(unfused_times)
                # Reduce-max across ranks
                fused_t = torch.tensor([fused_local_ms], device=device)
                unfused_t = torch.tensor([unfused_local_ms], device=device)
                dist.all_reduce(fused_t, op=dist.ReduceOp.MAX)
                dist.all_reduce(unfused_t, op=dist.ReduceOp.MAX)
                fused_us = fused_t.item() * 1e3  # ms -> us
                unfused_us = unfused_t.item() * 1e3

                if rank == 0:
                    speedup = unfused_us / fused_us if fused_us > 0 else float("nan")
                    print(
                        f"{tok:>8d} {hidden:>8d} "
                        f"{fused_us:>12.2f} {unfused_us:>12.2f} "
                        f"{speedup:>9.2f}x"
                    )
    finally:
        workspace.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
