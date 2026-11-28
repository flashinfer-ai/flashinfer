import socket
import sys
from pathlib import Path

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


def test_specialized_checkpoint_lifecycle_covers_kernel_state_and_fallback(
    monkeypatch,
) -> None:
    """GPU-free: prepare / restore / destroy of a specialized workspace act on both the
    fallback workspace and the specialized kernel's region, and repeated calls are
    no-ops."""
    from flashinfer.comm import _allreduce_fusion_ws2_specialized as ws2
    from flashinfer.comm.allreduce import SpecializedAllReduceFusionWorkspace
    from flashinfer.comm.workspace_base import AllReduceFusionWorkspace

    events = []

    class Fallback(AllReduceFusionWorkspace):
        @property
        def backend(self):
            return "mnnvl"

        def is_buffer_size_sufficient(self, *args, **kwargs):
            return True

        def checkpoint_prepare(self):
            events.append("fallback.prepare")

        def checkpoint_restore(self, comm_backend):
            events.append("fallback.restore")

        def destroy(self):
            events.append("fallback.destroy")
            self._destroyed = True

    class Comm:
        def barrier(self):
            events.append("barrier")

    class Handle:
        mapped = True
        comm_backend = Comm()

        def _unmap_and_release_handles(self):
            events.append("region.unmap")
            self.mapped = False

        def _create_and_map_handles(self, comm_backend):
            events.append("region.map")
            self.mapped = True

        def get_buffer_ptrs_host(self):
            return [1, 2]

    class Module:
        def allreduce_fusion_ws2_init(self, epoch, own_ptr, capacity):
            events.append("region.init")

    monkeypatch.setattr(ws2, "_module", Module())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    state = ws2._State(
        handle=Handle(),
        own_ptr=1,
        peer_ptr=2,
        epoch=torch.zeros(1, dtype=torch.int32),
        capacity=513,
        grid=1,
        device_idx=0,
        rank=0,
        region_bytes=0,
    )
    workspace = SpecializedAllReduceFusionWorkspace(Fallback(2, 0), state)
    assert workspace.backend == "specialized"
    assert workspace.fallback_backend == "mnnvl"

    workspace.checkpoint_prepare()
    workspace.checkpoint_prepare()
    assert events == [
        "fallback.prepare",
        "region.unmap",
        "barrier",
        "fallback.prepare",
    ]
    events.clear()
    restore_comm = Comm()
    workspace.checkpoint_restore(restore_comm)
    workspace.checkpoint_restore(restore_comm)
    assert events == [
        "fallback.restore",
        "region.map",
        "region.init",
        "barrier",
        "fallback.restore",
    ]
    events.clear()
    workspace.destroy()
    workspace.destroy()
    assert events == ["fallback.destroy"]
    assert workspace._ar_ws2 is None


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
    # mp.spawn starts fresh interpreters that need to re-import this module;
    # ensure the repo root is on sys.path so 'tests.comm' is findable.
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    mp.spawn(
        _run_worker,
        args=(2, _free_port(), num_tokens, use_oneshot),
        nprocs=2,
        join=True,
    )


def _run_ws2_worker(rank: int, world_size: int, port: int, backend: str) -> None:
    """A specialized workspace (backend="specialized", or "auto", which picks it here):
    the two-rank AR + residual + RMSNorm kernel (compute capability 10.3, hidden size 6144)
    and a call on the fallback backend, captured in one graph, survive prepare/restore of
    the kernel's own region and of the fallback workspace."""
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
        from flashinfer.comm._allreduce_fusion_ws2_specialized import (
            _allreduce_fusion_ws2_specialized_stats as stats,
        )
        from flashinfer.comm.mnnvl import TorchDistBackend

        # The vLLM workspace size and a token count in the first allowlisted range.
        max_token_num, hidden_dim = 5461, 6144
        lo, hi = stats()["allowlist_num_tokens"][0]
        num_tokens = min((lo + hi) // 2, max_token_num)
        workspace = comm.create_allreduce_fusion_workspace(
            backend=backend,
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(),
        )
        assert isinstance(workspace, comm.SpecializedAllReduceFusionWorkspace)
        assert workspace.has_specialized_kernels, stats()["last_state_rejection"]
        input_ = torch.full(
            (num_tokens, hidden_dim), rank + 1, dtype=torch.bfloat16, device="cuda"
        )
        residual = torch.ones_like(input_)
        gamma = torch.zeros(hidden_dim, dtype=torch.bfloat16, device="cuda")
        residual_out = torch.empty_like(input_)
        norm_out = torch.empty_like(input_)
        # A decode-sized all-reduce on the same workspace: served by the fallback backend.
        small_in = torch.full(
            (4, hidden_dim), rank + 1, dtype=torch.bfloat16, device="cuda"
        )
        small_out = torch.empty_like(small_in)

        def fused() -> None:
            comm.allreduce_fusion(
                input=input_,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_out=residual_out,
                norm_out=norm_out,
                residual_in=residual,
                rms_gamma=gamma,
                rms_eps=1e-6,
                fp32_acc=True,
                weight_bias=1.0,
            )

        def small() -> None:
            comm.allreduce_fusion(
                input=small_in,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kAllReduce,
                output=small_out,
            )

        before = stats()["dispatch_count"]
        fused()
        small()
        torch.cuda.synchronize()
        assert stats()["dispatch_count"] == before + 1
        # (1 + 2) + 1 = 4; 4 * rsqrt(16) * (1 + 0) = 1
        torch.testing.assert_close(residual_out, torch.full_like(residual_out, 4))
        torch.testing.assert_close(norm_out, torch.ones_like(norm_out))
        torch.testing.assert_close(small_out, torch.full_like(small_out, 3))

        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            fused()
            small()
        assert stats()["dispatch_count"] == before + 2

        dist.barrier()
        workspace.checkpoint_prepare()
        workspace.checkpoint_prepare()
        with pytest.raises(RuntimeError, match="not attached"):
            fused()
        with pytest.raises(RuntimeError, match="not attached"):
            small()
        dist.barrier()
        fresh_backend = TorchDistBackend()
        workspace.checkpoint_restore(fresh_backend)
        workspace.checkpoint_restore(fresh_backend)
        dist.barrier()

        # Two replays, each checked on its own (fresh inputs, NaN-filled outputs): the two
        # receive slots of the kernel's region after the restore, and the fallback's buffers.
        for replay in range(2):
            input_.fill_(rank + 2 + replay)
            small_in.fill_(rank + 2 + replay)
            residual_out.fill_(float("nan"))
            norm_out.fill_(float("nan"))
            small_out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            # (2 + 3 + 2 * replay) + 1 = 6 + 2 * replay; norm = 1 (gamma 0, weight_bias 1)
            what = f"rank {rank} replay {replay}"
            torch.testing.assert_close(
                residual_out,
                torch.full_like(residual_out, 6 + 2 * replay),
                msg=lambda m: f"{what} residual_out: {m}",
            )
            torch.testing.assert_close(
                norm_out,
                torch.ones_like(norm_out),
                msg=lambda m: f"{what} norm_out: {m}",
            )
            torch.testing.assert_close(
                small_out,
                torch.full_like(small_out, 5 + 2 * replay),
                msg=lambda m: f"{what} small_out: {m}",
            )
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group()


def _is_compute_capability_10_3() -> bool:
    """Both GPUs the two workers bind (cuda:0 and cuda:1) have compute capability 10.3."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    from flashinfer.utils import get_compute_capability

    return all(
        get_compute_capability(torch.device(f"cuda:{i}")) == (10, 3) for i in (0, 1)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not _is_compute_capability_10_3(),
    reason="the specialized two-rank kernel needs two compute capability 10.3 devices",
)
@pytest.mark.parametrize("backend", ["specialized", "auto"])
def test_graph_replay_after_symmetric_memory_remap_ws2_specialized(backend) -> None:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    mp.spawn(_run_ws2_worker, args=(2, _free_port(), backend), nprocs=2, join=True)
