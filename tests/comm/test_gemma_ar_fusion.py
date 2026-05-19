"""Focused multi-rank test for Gemma / Qwen3.5 RMSNorm in AllReduce fusion.

The big parametrized test in test_trtllm_allreduce_fusion.py uses a 1728-
config inner loop and a loose bf16 tolerance (8e-1). This test runs a single
focused config with two reference paths:

  1. Hand-rolled torch FP32 reference (`(1 + gamma) * x * rsqrt(...)`).
  2. flashinfer.norm.gemma_fused_add_rmsnorm applied to the post-AllReduce
     tensor — bit-equivalent to the fused kernel modulo the residual-add
     dtype (standalone is fp32 add; fusion is bf16 add).

Tolerance is tighter than the big test (atol=2e-2 bf16) so a Gemma-specific
regression that the loose 8e-1 check would miss still trips this one.
"""

import multiprocessing as mp
import socket

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.norm import gemma_fused_add_rmsnorm


def _run_gemma_worker(world_size, rank, distributed_init_port):
    """One rank of the Gemma-fusion correctness probe."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        torch.manual_seed(42 + rank)
        dtype = torch.bfloat16
        token_num = 8
        hidden_dim = 2048
        rms_eps = 1e-6
        weight_bias = 1.0

        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=128,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )

        # Same data across ranks for the rms_gamma; per-rank shards for input.
        torch.manual_seed(42)
        rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)
        torch.manual_seed(42 + rank)
        input_shard = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
        residual_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)

        # ---- Fused path: AR + residual add + Gemma RMSNorm ------------------
        fused_in = input_shard.clone()
        fused_resid_out = torch.empty_like(residual_in)
        fused_norm_out = torch.empty_like(residual_in)
        comm.allreduce_fusion(
            input=fused_in,
            workspace=workspace,
            pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=False,
            residual_in=residual_in,
            residual_out=fused_resid_out,
            norm_out=fused_norm_out,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            weight_bias=weight_bias,
            use_oneshot=True,
        )
        torch.cuda.synchronize()

        # ---- Reference A: hand-rolled FP32 RMSNorm --------------------------
        ar_clone = input_shard.clone()
        dist.all_reduce(ar_clone, group=group)
        pre_fp32 = ar_clone.to(torch.float32) + residual_in.to(torch.float32)
        var = pre_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed_fp32 = pre_fp32 * torch.rsqrt(var + rms_eps)
        normed_fp32 = normed_fp32 * (weight_bias + rms_gamma.to(torch.float32))
        ref_resid = pre_fp32.to(dtype)
        ref_norm = normed_fp32.to(dtype)

        # ---- Reference B: gemma_fused_add_rmsnorm on post-AR tensor ---------
        # AR done manually then handed to the standalone Gemma kernel:
        #   step 1: residual += input              (in fp32 inside the kernel)
        #   step 2: input = (residual/RMS) * (1+w)
        # `standalone_norm` ends up holding the norm result; `standalone_resid`
        # ends up holding the post-add residual.
        # Note: the standalone path uses fp32 residual-add (norm.cuh:418-425),
        # while the fused kernel does the add in bf16 — so residuals may differ
        # by ~1 bf16 ulp. Norm outputs should still agree within bf16 tol.
        standalone_norm = input_shard.clone()
        dist.all_reduce(standalone_norm, group=group)
        standalone_resid = residual_in.clone()
        gemma_fused_add_rmsnorm(
            standalone_norm, standalone_resid, rms_gamma, eps=rms_eps
        )

        # ---- Asserts --------------------------------------------------------
        atol = 2e-2
        rtol = 1e-2

        torch.testing.assert_close(
            fused_resid_out.to(torch.float32),
            ref_resid.to(torch.float32),
            atol=atol,
            rtol=rtol,
            msg="fused residual_out mismatches torch FP32 reference",
        )
        torch.testing.assert_close(
            fused_norm_out.to(torch.float32),
            ref_norm.to(torch.float32),
            atol=atol,
            rtol=rtol,
            msg="fused norm_out mismatches torch FP32 reference",
        )
        torch.testing.assert_close(
            fused_norm_out.to(torch.float32),
            standalone_norm.to(torch.float32),
            atol=atol,
            rtol=rtol,
            msg=(
                "fused norm_out diverges from standalone gemma_fused_add_rmsnorm; "
                "both kernels should compute the same Gemma RMSNorm output."
            ),
        )

        workspace.destroy()
        if rank == 0:
            print(
                f"[rank {rank}] Gemma AR fusion: torch ref + cross-kernel "
                f"agreement OK (tol={atol})"
            )
    finally:
        dist.barrier(group=group)
        dist.destroy_process_group(group=group)


def _get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


@pytest.mark.parametrize("world_size", [2])
def test_gemma_rmsnorm_ar_fusion(world_size):
    """End-to-end Gemma RMSNorm correctness for fused AllReduce path."""
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"world_size {world_size} > available GPUs {available_gpus}")

    mp.set_start_method("spawn", force=True)
    port = _get_open_port()
    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=_run_gemma_worker,
            args=(world_size, rank, port),
            name=f"gemma-ar-rank-{rank}",
        )
        p.start()
        procs.append(p)

    for i, p in enumerate(procs):
        p.join()
        assert p.exitcode == 0, (
            f"Gemma AR-fusion worker rank {i} failed (exitcode={p.exitcode})"
        )


if __name__ == "__main__":
    test_gemma_rmsnorm_ar_fusion(2)
