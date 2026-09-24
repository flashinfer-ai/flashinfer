"""Generated Blackwell DCP all-to-all kernels behind the public DCP API.

The generated CP2/CP4 kernels share the helix MNNVL workspace layout, so the
public ``decode_cp_a2a_*`` functions are the only surface under test. Runs
with ``torch.distributed`` spawn workers (no MPI needed):

    pytest tests/comm/test_cake_dcp_alltoall.py -v
"""

import pytest
import torch
import torch.distributed as dist

from tests.comm.test_ulysses_a2a import multi_process_parallel

# (batch, head_dim, stats_dim, dtype). The first four rows are the generated
# routes; the last two exercise the portable helix fallback through the same
# module (fp16 D128 also matches the generated route's byte layout).
SHAPES = [
    (1, 128, 2, torch.bfloat16),
    (16, 128, 2, torch.bfloat16),
    (64, 128, 2, torch.bfloat16),
    (128, 128, 2, torch.bfloat16),
    (16, 128, 2, torch.float16),
    (16, 256, 4, torch.bfloat16),
]


def _supported_platform(world_size: int) -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.device_count() < world_size:
        return f"requires {world_size} GPUs"
    if torch.cuda.get_device_capability(0) not in {(10, 0), (10, 3)}:
        return "generated kernels target Blackwell SM100/SM103"
    import pynvml

    from flashinfer.comm.mnnvl import MnnvlMemory

    pynvml.nvmlInit()
    if not MnnvlMemory.supports_mnnvl():
        return "MNNVL fabric memory is not available on this platform"
    return None


def _allocate_workspace(rank: int, world_size: int) -> torch.Tensor:
    from flashinfer.comm import decode_cp_a2a_allocate_mnnvl_workspace
    from flashinfer.comm.comm_backend import TorchDistBackend
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig

    mapping = Mapping(
        world_size=world_size, rank=rank, cp_size=world_size, tp_size=1, pp_size=1
    )
    return decode_cp_a2a_allocate_mnnvl_workspace(
        mapping, mnnvl_config=MnnvlConfig(comm_backend=TorchDistBackend())
    )


def _to_torch(t):
    return t if isinstance(t, torch.Tensor) else torch.from_dlpack(t)


def _check_transpose(rank, world_size, partial_o, softmax_stats, recv_o, recv_s):
    all_o = [torch.empty_like(partial_o) for _ in range(world_size)]
    all_s = [torch.empty_like(softmax_stats) for _ in range(world_size)]
    dist.all_gather(all_o, partial_o)
    dist.all_gather(all_s, softmax_stats)
    for peer in range(world_size):
        torch.testing.assert_close(
            recv_o[..., peer, :], all_o[peer][..., rank, :], atol=0, rtol=0
        )
        torch.testing.assert_close(
            recv_s[..., peer, :], all_s[peer][..., rank, :], atol=0, rtol=0
        )


def _correctness_worker(world_size, rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from flashinfer.comm import decode_cp_a2a_alltoall, decode_cp_a2a_init_workspace

        workspace = _allocate_workspace(rank, world_size)
        decode_cp_a2a_init_workspace(workspace, rank, world_size)
        torch.cuda.synchronize()
        dist.barrier()

        for round_idx, (batch, head_dim, stats_dim, dtype) in enumerate(SHAPES):
            torch.manual_seed(0xA2A + rank * 100 + round_idx)
            partial_o = torch.randn(
                batch, world_size, head_dim, dtype=dtype, device="cuda"
            )
            softmax_stats = torch.randn(
                batch, world_size, stats_dim, dtype=torch.float32, device="cuda"
            )

            # Allocating form.
            recv_o, recv_s = decode_cp_a2a_alltoall(
                partial_o, softmax_stats, workspace, rank, world_size
            )
            recv_o, recv_s = _to_torch(recv_o), _to_torch(recv_s)
            torch.cuda.synchronize()
            dist.barrier()
            _check_transpose(rank, world_size, partial_o, softmax_stats, recv_o, recv_s)

            # Preallocated-output form, repeated so the FIFO wraps.
            out_o = torch.empty_like(partial_o)
            out_s = torch.empty_like(softmax_stats)
            for _ in range(5):
                out_o.zero_()
                out_s.zero_()
                ret_o, ret_s = decode_cp_a2a_alltoall(
                    partial_o,
                    softmax_stats,
                    workspace,
                    rank,
                    world_size,
                    out=(out_o, out_s),
                )
                assert ret_o is out_o and ret_s is out_s
                torch.cuda.synchronize()
                dist.barrier()
                _check_transpose(
                    rank, world_size, partial_o, softmax_stats, out_o, out_s
                )
    finally:
        dist.destroy_process_group()


def _graph_worker(world_size, rank, port):
    """The preallocated-output form must capture and replay in a CUDA graph."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from flashinfer.comm import decode_cp_a2a_alltoall, decode_cp_a2a_init_workspace

        workspace = _allocate_workspace(rank, world_size)
        decode_cp_a2a_init_workspace(workspace, rank, world_size)
        torch.cuda.synchronize()
        dist.barrier()

        torch.manual_seed(0xA2A + rank)
        partial_o = torch.randn(
            64, world_size, 128, dtype=torch.bfloat16, device="cuda"
        )
        softmax_stats = torch.randn(
            64, world_size, 2, dtype=torch.float32, device="cuda"
        )
        out_o = torch.empty_like(partial_o)
        out_s = torch.empty_like(softmax_stats)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            decode_cp_a2a_alltoall(
                partial_o,
                softmax_stats,
                workspace,
                rank,
                world_size,
                out=(out_o, out_s),
            )
        stream.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            decode_cp_a2a_alltoall(
                partial_o,
                softmax_stats,
                workspace,
                rank,
                world_size,
                out=(out_o, out_s),
            )
        for _ in range(3):
            partial_o.normal_()
            softmax_stats.normal_()
            out_o.zero_()
            out_s.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            _check_transpose(rank, world_size, partial_o, softmax_stats, out_o, out_s)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_generated_dcp_alltoall_correctness(world_size):
    reason = _supported_platform(world_size)
    if reason:
        pytest.skip(reason)
    multi_process_parallel(world_size, _correctness_worker)


@pytest.mark.parametrize("world_size", [2, 4])
def test_generated_dcp_alltoall_cuda_graph(world_size):
    reason = _supported_platform(world_size)
    if reason:
        pytest.skip(reason)
    multi_process_parallel(world_size, _graph_worker)


def test_generated_spec_selection(monkeypatch):
    """The generated module is selected only for one exact Blackwell target."""
    from flashinfer.jit import cake_dcp_alltoall
    from flashinfer.jit.comm import gen_dcp_alltoall_module
    from flashinfer.jit.core import current_compilation_context

    monkeypatch.setattr(
        cake_dcp_alltoall,
        "GENERATED_SOURCES",
        {"sm_100a": ["generated/dcp_alltoall/sm_100a/probe_kernel.cu"]},
    )
    monkeypatch.setattr(current_compilation_context, "TARGET_CUDA_ARCHS", {(10, "0a")})
    spec = gen_dcp_alltoall_module()
    assert spec.name == "dcp_alltoall_sm100a"
    assert any(str(s).endswith("cake_dcp_alltoall_dispatch.cu") for s in spec.sources)

    # sm_103a has no sources registered -> portable module.
    monkeypatch.setattr(current_compilation_context, "TARGET_CUDA_ARCHS", {(10, "3a")})
    assert gen_dcp_alltoall_module().name == "dcp_alltoall"

    # Multi-target builds keep the portable module.
    monkeypatch.setattr(
        current_compilation_context, "TARGET_CUDA_ARCHS", {(9, "0a"), (10, "0a")}
    )
    assert gen_dcp_alltoall_module().name == "dcp_alltoall"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
