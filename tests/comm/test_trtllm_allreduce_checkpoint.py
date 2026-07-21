import os

import pytest
import torch
import torch.distributed as dist

from tests.test_helpers.comm import init_torch_distributed_from_mpi, setup_mpi_and_cuda


def _distributed_world_size() -> int:
    """Return world size from launcher env vars without importing mpi4py."""
    for key in ("SLURM_NTASKS", "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MPI_LOCALNRANKS"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


def test_protocol_restore_resets_inference_flags(monkeypatch) -> None:
    from flashinfer.comm.allreduce import MNNVLAllReduceFusionWorkspace
    from flashinfer.comm.workspace_base import AllReduceFusionWorkspace

    workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    AllReduceFusionWorkspace.__init__(workspace, world_size=1, rank=0)
    workspace.handle = type(
        "Handle",
        (),
        {"lamport_initialize": lambda self, rank, dtype: None},
    )()
    workspace.buffer_size_bytes = 1024
    with torch.inference_mode():
        workspace.buffer_flags = torch.ones(9, dtype=torch.uint32)
    workspace._destroyed = True
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    assert torch.is_inference(workspace.buffer_flags)
    assert not torch.is_inference_mode_enabled()
    workspace._initialize_protocol()
    assert not torch.is_inference_mode_enabled()
    assert torch.equal(
        workspace.buffer_flags,
        torch.tensor([0, 2, 1024, 0, 0, 0, 0, 0, 0], dtype=torch.uint32),
    )


def test_checkpoint_lifecycle_rejects_torch_symmetric_memory_backing() -> None:
    from flashinfer.comm.allreduce import (
        MNNVLAllReduceFusionWorkspace,
        TRTLLMAllReduceFusionWorkspace,
    )
    from flashinfer.comm.workspace_base import AllReduceFusionWorkspace

    trt_workspace = object.__new__(TRTLLMAllReduceFusionWorkspace)
    AllReduceFusionWorkspace.__init__(trt_workspace, world_size=1, rank=0)
    trt_workspace.mem_handles = []
    trt_workspace._destroyed = True

    mnnvl_workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    AllReduceFusionWorkspace.__init__(mnnvl_workspace, world_size=1, rank=0)
    mnnvl_workspace.handle = object()
    mnnvl_workspace._destroyed = True

    for workspace in (trt_workspace, mnnvl_workspace):
        with pytest.raises(NotImplementedError, match="torch symmetric memory"):
            workspace.checkpoint_prepare()
        with pytest.raises(NotImplementedError, match="torch symmetric memory"):
            workspace.checkpoint_restore(None)


@pytest.mark.parametrize("num_tokens,use_oneshot", [(1, True), (4, False)])
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or (torch.cuda.device_count() < 2 and _distributed_world_size() < 2),
    reason="checkpointable TRT-LLM all-reduce requires 2+ GPUs or 2+ distributed ranks",
)
def test_graph_replay_after_symmetric_memory_remap(
    num_tokens: int, use_oneshot: bool
) -> None:
    import flashinfer.comm as comm
    from flashinfer.comm.mnnvl import TorchDistBackend

    rank, world_size, _ = setup_mpi_and_cuda()
    init_torch_distributed_from_mpi()

    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # allreduce sum of (rank+1) across all ranks: 1+2+...+world_size
    expected_first = world_size * (world_size + 1) // 2
    # after input_.fill_(rank+2): sum of (rank+2) across all ranks
    expected_second = world_size * (world_size + 1) // 2 + world_size

    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=num_tokens,
            hidden_dim=4096,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(),
        )
        input_ = torch.full(
            (num_tokens, 4096), rank + 1, dtype=torch.bfloat16, device=device
        )
        output = torch.empty_like(input_)

        def all_reduce() -> None:
            comm.allreduce_fusion(
                input=input_,
                workspace=workspace,
                output=output,
                pattern=comm.AllReduceFusionPattern.kAllReduce,
                use_oneshot=use_oneshot,
            )

        all_reduce()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, expected_first))

        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            all_reduce()

        dist.barrier()
        workspace.checkpoint_prepare()
        workspace.checkpoint_prepare()
        with pytest.raises(RuntimeError, match="not attached"):
            all_reduce()
        dist.barrier()
        fresh_backend = TorchDistBackend()
        workspace.checkpoint_restore(fresh_backend)
        workspace.checkpoint_restore(fresh_backend)
        dist.barrier()

        input_.fill_(rank + 2)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, expected_second))
    finally:
        if workspace is not None:
            workspace.destroy()
