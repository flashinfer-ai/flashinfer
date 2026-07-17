import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_worker(
    rank: int, world_size: int, port: int, num_tokens: int, use_oneshot: bool
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    workspace = None
    try:
        import flashinfer.comm as comm
        from flashinfer.comm.mnnvl import TorchDistBackend

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
            (num_tokens, 4096), rank + 1, dtype=torch.bfloat16, device="cuda"
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
        torch.testing.assert_close(output, torch.full_like(output, 3))

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
        torch.testing.assert_close(output, torch.full_like(output, 5))
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("num_tokens,use_oneshot", [(1, True), (4, False)])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="checkpointable TRT-LLM all-reduce requires two CUDA devices",
)
def test_graph_replay_after_symmetric_memory_remap(
    num_tokens: int, use_oneshot: bool
) -> None:
    mp.spawn(
        _run_worker,
        args=(2, _free_port(), num_tokens, use_oneshot),
        nprocs=2,
        join=True,
    )
