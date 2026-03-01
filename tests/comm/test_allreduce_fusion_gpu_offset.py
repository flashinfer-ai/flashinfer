"""Test allreduce fusion with GPU offset (base_gpu_id > 0).

When using frameworks like sglang in colocate mode, the inference engine
may run on GPUs 4-7 while TP ranks are 0-3. This means
torch.cuda.current_device() != tp_rank. This test validates that
allreduce fusion works correctly in this scenario.

See: https://github.com/flashinfer-ai/flashinfer/pull/2662
"""

import multiprocessing as mp
import socket
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend


TOKEN_NUMS: list[int] = [1, 128, 1024]
HIDDEN_DIM: int = 1024
MAX_TOKEN_NUM: int = 2048


def _worker_with_gpu_offset(
    world_size: int,
    rank: int,
    gpu_offset: int,
    dtype: torch.dtype,
    hidden_dim: int,
    distributed_init_port: int,
    legacy_api: bool,
) -> None:
    """Worker that sets CUDA device to rank + gpu_offset, simulating base_gpu_id > 0."""
    device_id: int = rank + gpu_offset
    device: torch.device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        assert torch.cuda.current_device() == device_id, (
            f"Expected current_device={device_id}, got {torch.cuda.current_device()}"
        )

        lamport_use_fp32: bool = dtype == torch.float32

        if legacy_api:
            ipc_handles, workspace_tensor, workspace_metadata = (
                comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                    rank,
                    world_size,
                    MAX_TOKEN_NUM,
                    hidden_dim,
                    group=group,
                    use_fp32_lamport=lamport_use_fp32,
                    create_metadata=True,
                )
            )
        else:
            workspace = comm.create_allreduce_fusion_workspace(
                backend="trtllm",
                world_size=world_size,
                rank=rank,
                max_token_num=MAX_TOKEN_NUM,
                hidden_dim=hidden_dim,
                dtype=dtype,
                comm_backend=TorchDistBackend(),
            )

        assert torch.cuda.current_device() == device_id, (
            f"Workspace creation changed device! Expected {device_id}, got {torch.cuda.current_device()}"
        )

        pattern_code = comm.AllReduceFusionPattern.kAllReduce

        for token_num in TOKEN_NUMS:
            dist.barrier(group=group)

            message_size: int = token_num * hidden_dim
            allreduce_in: torch.Tensor = torch.randn(message_size, dtype=dtype, device=device)
            allreduce_in_clone: torch.Tensor = allreduce_in.clone()
            all_reduce_out: torch.Tensor = torch.zeros(message_size, dtype=dtype, device=device)

            residual_in: torch.Tensor = torch.randn(message_size, dtype=dtype, device=device)
            residual_out: torch.Tensor = torch.empty_like(residual_in)
            norm_out: torch.Tensor = torch.empty_like(residual_in)
            quant_out: torch.Tensor = torch.empty(message_size, dtype=dtype, device=device)
            scale_out: torch.Tensor = torch.empty(message_size // 16, dtype=dtype, device=device)
            rms_gamma: torch.Tensor = torch.randn(hidden_dim, dtype=dtype, device=device)
            rms_eps: float = 1e-3

            # Warmup
            stream: torch.cuda.Stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    if legacy_api:
                        comm.trtllm_allreduce_fusion(
                            allreduce_in=allreduce_in,
                            world_size=world_size,
                            world_rank=rank,
                            token_num=token_num,
                            hidden_dim=hidden_dim,
                            workspace_ptrs=workspace_tensor,
                            launch_with_pdl=False,
                            use_oneshot=None,
                            trigger_completion_at_end=False,
                            fp32_acc=False,
                            pattern_code=pattern_code,
                            allreduce_out=all_reduce_out,
                            residual_in=residual_in,
                            residual_out=residual_out,
                            norm_out=norm_out,
                            quant_out=quant_out,
                            scale_out=scale_out,
                            rms_gamma=rms_gamma,
                            rms_eps=rms_eps,
                            metadata=workspace_metadata,
                        )
                    else:
                        comm.allreduce_fusion(
                            input=allreduce_in.view(token_num, hidden_dim),
                            workspace=workspace,
                            launch_with_pdl=False,
                            output=all_reduce_out.view(token_num, hidden_dim),
                            residual_in=residual_in.view(token_num, hidden_dim),
                            residual_out=residual_out.view(token_num, hidden_dim),
                            norm_out=norm_out.view(token_num, hidden_dim),
                            quant_out=quant_out.view(token_num, hidden_dim),
                            scale_out=scale_out,
                            rms_gamma=rms_gamma,
                            rms_eps=rms_eps,
                            pattern=pattern_code,
                            use_oneshot=None,
                            fp32_acc=False,
                        )

            # CUDA graph capture
            graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(3):
                    if legacy_api:
                        comm.trtllm_allreduce_fusion(
                            allreduce_in=allreduce_in,
                            world_size=world_size,
                            world_rank=rank,
                            token_num=token_num,
                            hidden_dim=hidden_dim,
                            workspace_ptrs=workspace_tensor,
                            launch_with_pdl=False,
                            use_oneshot=None,
                            trigger_completion_at_end=False,
                            fp32_acc=False,
                            pattern_code=pattern_code,
                            allreduce_out=all_reduce_out,
                            residual_in=residual_in,
                            residual_out=residual_out,
                            norm_out=norm_out,
                            quant_out=quant_out,
                            scale_out=scale_out,
                            rms_gamma=rms_gamma,
                            rms_eps=rms_eps,
                            metadata=workspace_metadata,
                        )
                    else:
                        comm.allreduce_fusion(
                            input=allreduce_in.view(token_num, hidden_dim),
                            workspace=workspace,
                            launch_with_pdl=False,
                            output=all_reduce_out.view(token_num, hidden_dim),
                            residual_in=residual_in.view(token_num, hidden_dim),
                            residual_out=residual_out.view(token_num, hidden_dim),
                            norm_out=norm_out.view(token_num, hidden_dim),
                            quant_out=quant_out.view(token_num, hidden_dim),
                            scale_out=scale_out,
                            rms_gamma=rms_gamma,
                            rms_eps=rms_eps,
                            pattern=pattern_code,
                            use_oneshot=None,
                            fp32_acc=False,
                        )

            graph.replay()
            torch.cuda.synchronize()

            all_reduce_out = all_reduce_out.view(token_num, hidden_dim)

            # Reference: nccl all_reduce
            dist.all_reduce(allreduce_in_clone, group=group)
            ref_out: torch.Tensor = allreduce_in_clone.view(token_num, hidden_dim).to(torch.float32)

            tolerance: float = 8e-2 if dtype == torch.float16 else 8e-1
            torch.testing.assert_close(
                all_reduce_out.to(torch.float32),
                ref_out,
                atol=tolerance,
                rtol=1e-2,
            )

            assert torch.cuda.current_device() == device_id, (
                f"Device changed after allreduce! Expected {device_id}, got {torch.cuda.current_device()}"
            )

            dist.barrier(group=group)
            print(
                f"RANK {rank} (gpu {device_id}): token_num={token_num} hidden_dim={hidden_dim} passed"
            )

    finally:
        dist.barrier(group=group)
        if legacy_api:
            comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                ipc_handles, group=group
            )
        elif workspace is not None:
            workspace.destroy()
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


def _run_multi_process(
    world_size: int,
    gpu_offset: int,
    dtype: torch.dtype,
    hidden_dim: int,
    test_target: Any,
    target_args: tuple[Any, ...] = (),
) -> None:
    mp.set_start_method("spawn", force=True)

    distributed_init_port: int = _get_open_port()
    procs: list[mp.Process] = []
    for i in range(world_size):
        proc_args: tuple[Any, ...] = (
            world_size,
            i,
            gpu_offset,
            dtype,
            hidden_dim,
            distributed_init_port,
        ) + target_args
        proc: mp.Process = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"Process {i} failed with exit code {proc.exitcode}"


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("legacy_api", [True, False])
def test_allreduce_fusion_gpu_offset(
    world_size: int,
    dtype: torch.dtype,
    legacy_api: bool,
) -> None:
    """Test allreduce fusion when CUDA device index != TP rank (base_gpu_id > 0)."""
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    available_gpus: int = torch.cuda.device_count()
    # Need world_size GPUs for the workers + at least 1 extra for offset
    gpu_offset: int = available_gpus - world_size
    if gpu_offset <= 0:
        pytest.skip(
            f"Need more than {world_size} GPUs to test gpu_offset > 0 "
            f"(have {available_gpus})"
        )

    api_str: str = "legacy" if legacy_api else "unified"
    print(
        f"Running gpu_offset test: world_size={world_size}, gpu_offset={gpu_offset}, "
        f"{api_str} API (GPUs {gpu_offset}..{gpu_offset + world_size - 1})"
    )

    _run_multi_process(
        world_size=world_size,
        gpu_offset=gpu_offset,
        dtype=dtype,
        hidden_dim=HIDDEN_DIM,
        test_target=_worker_with_gpu_offset,
        target_args=(legacy_api,),
    )
    print(f"gpu_offset allreduce fusion tp={world_size} ({api_str} API): OK")


if __name__ == "__main__":
    test_allreduce_fusion_gpu_offset(world_size=2, dtype=torch.bfloat16, legacy_api=True)
    print()
    test_allreduce_fusion_gpu_offset(world_size=2, dtype=torch.bfloat16, legacy_api=False)
