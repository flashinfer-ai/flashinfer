"""Unit tests for split-path inner kernels (identity + registry).

4-GPU multirank test:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_split_kernels.py -v -m "nvep and gpu_4" --backend=nccl_ep      # or nixl_ep
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

# First-use JIT compile of reference kernels (e.g. fused_moe_trtllm_sm100)
# can exceed torch's 10-min default watchdog while other ranks wait in a
# collective; a cold cache is not a hang.
_PG_TIMEOUT = timedelta(minutes=60)


class TestKernelRequiresWeights:
    def test_identity_does_not_require_weights(self) -> None:
        from flashinfer.moe_ep import IdentityConfig, kernel_requires_weights

        assert kernel_requires_weights(IdentityConfig()) is False
        assert kernel_requires_weights(IdentityConfig(kernel_name="identity")) is False

    def test_fused_moe_requires_moe_config(self) -> None:
        from flashinfer.moe_ep import FusedMoeKernelConfig

        with pytest.raises(TypeError):
            FusedMoeKernelConfig()  # type: ignore[call-arg]


class TestIdentitySplitKernel:
    def test_passes_expert_tensors_through_unchanged(self) -> None:
        import torch

        from flashinfer.moe_ep import (
            FleetParams,
            IdentityConfig,
            SplitKernelContext,
            run_split_kernel,
        )

        expert = torch.randn(8, 128)
        ctx = SplitKernelContext(
            expert_tensors=expert,
            num_tokens=8,
            fleet_params=FleetParams(
                num_experts=1,
                max_tokens_per_rank=8,
                token_hidden_size=128,
            ),
        )
        out = run_split_kernel(IdentityConfig(), ctx)
        assert out is expert

    def test_create_returns_identity_kernel(self) -> None:
        from flashinfer.moe_ep import IdentityConfig
        from flashinfer.moe_ep.backends.split.kernel.identity.backend import (
            IdentitySplitKernelBackend,
        )
        from flashinfer.moe_ep.core.kernel.registry import create_split_kernel

        kernel = create_split_kernel(IdentityConfig())
        assert isinstance(kernel, IdentitySplitKernelBackend)


class TestSplitKernelRegistry:
    def test_unknown_kernel_raises(self) -> None:
        import torch

        from flashinfer.moe_ep import (
            FleetParams,
            SplitKernelContext,
            run_split_kernel,
        )

        class _UnknownKernel:
            kernel_name = "unknown"

        ctx = SplitKernelContext(
            expert_tensors=torch.zeros(2, 4),
            num_tokens=2,
            fleet_params=FleetParams(
                num_experts=2,
                max_tokens_per_rank=2,
                token_hidden_size=4,
            ),
        )
        with pytest.raises(KeyError, match="unknown"):
            run_split_kernel(_UnknownKernel(), ctx)


@pytest.fixture
def capturing_stub_fleet():
    """Stub fleet that records tensors at dispatch and combine boundaries."""
    from flashinfer.moe_ep.core.comm.fleet import _BACKEND_REGISTRY

    captured: dict[str, object] = {}

    class _StubHandle:
        def dispatch(self, params):
            from flashinfer.moe_ep import DispatchOutput

            captured["dispatch_in"] = params.x[0]
            return DispatchOutput(
                expert_tensors=params.x[0],
                num_tokens=params.x[0].size(0),
            )

        def combine(self, params):
            from flashinfer.moe_ep import CombineOutput

            captured["combine_in"] = params.x[0]
            return CombineOutput(x=params.out)

        def complete(self):
            pass

        def destroy(self):
            pass

    class _StubFleet:
        def __init__(self, bootstrap, params, algo_knobs):
            pass

        def create_handle(self, params, algo_knobs=()):
            return _StubHandle()

        def update_topology(self, bootstrap, algo_knobs=()):
            pass

        def destroy(self):
            pass

    from unittest import mock

    saved = _BACKEND_REGISTRY.get("nccl_ep")
    _BACKEND_REGISTRY["nccl_ep"] = _StubFleet
    with mock.patch("flashinfer.moe_ep.modes.split_layer.validate_arch_for_backend"):
        yield captured
    if saved is not None:
        _BACKEND_REGISTRY["nccl_ep"] = saved
    else:
        _BACKEND_REGISTRY.pop("nccl_ep", None)


def test_split_layer_identity_kernel_wires_dispatch_to_combine(capturing_stub_fleet):
    """MoEEpSplitLayer + IdentityConfig passes dispatch output into combine."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        dummy_moe_weights,
        IdentityConfig,
        MoEEpSplitLayer,
        MoEEpTensors,
        NCCLEPConfig,
        SplitConfig,
    )

    captured = capturing_stub_fleet
    x = torch.arange(24, dtype=torch.float32, device="cuda").view(4, 6)
    split = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=FleetParams(
            num_experts=2,
            max_tokens_per_rank=4,
            token_hidden_size=6,
        ),
        weights=dummy_moe_weights(num_local_experts=2, hidden=6),
        backend=SplitConfig(comm=NCCLEPConfig(), kernel=IdentityConfig()),
    )
    out = split(
        MoEEpTensors(
            hidden_states=x,
            topk_ids=torch.zeros(4, 2, dtype=torch.int64, device="cuda"),
            topk_weights=torch.ones(4, 2, device="cuda") * 0.5,
        )
    )

    torch.testing.assert_close(captured["dispatch_in"], x)
    torch.testing.assert_close(captured["combine_in"], x)
    assert out.shape == x.shape


def pytest_generate_tests(metafunc):
    if "comm_backend" not in metafunc.fixturenames:
        return
    cli = metafunc.config.getoption("--backend", default=None)
    if cli == "both" or cli is None:
        metafunc.parametrize("comm_backend", ["nccl_ep", "nixl_ep"])
    else:
        metafunc.parametrize("comm_backend", [cli])


@pytest.mark.nvep
@pytest.mark.gpu_4
def test_identity_split_kernel_multirank_roundtrip(comm_backend):
    """SplitConfig + IdentityConfig roundtrips hidden_states on 4+ GPUs."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetParams,
        dummy_moe_weights,
        IdentityConfig,
        MoEEpSplitLayer,
        MoEEpTensors,
        NCCLEPConfig,
        NvepConfig,
        SplitConfig,
    )

    backend_name = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend_name,
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=_PG_TIMEOUT,
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    num_tokens = 64
    num_experts = 8
    hidden = 4096
    topk = 4

    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        device="cuda",
        dtype=torch.int64,
        generator=g,
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda", generator=g), dim=-1
    )

    if comm_backend == "nixl_ep":
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        tcp_store = dist.TCPStore(
            host_name=master_addr,
            port=master_port + 1,
            world_size=world_size,
            is_master=(rank == 0),
        )
    else:
        tcp_store = None

    comm = NvepConfig() if comm_backend == "nixl_ep" else NCCLEPConfig()
    split = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
            nccl_comm=None,
            tcp_store=tcp_store,
        ),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=num_tokens,
            token_hidden_size=hidden,
            dtype_bytes=2,
            algorithm=EpAlgorithm.LOW_LATENCY,
        ),
        weights=dummy_moe_weights(
            num_local_experts=num_experts // world_size,
            hidden=hidden,
        ),
        backend=SplitConfig(comm=comm, kernel=IdentityConfig()),
    )

    y = split(
        MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    )
    torch.cuda.synchronize()
    dist.barrier()

    assert y.shape == x.shape
    torch.testing.assert_close(y, x, atol=5e-2, rtol=5e-2)
    split.destroy()
    print(f"rank {rank}: identity split kernel {comm_backend} roundtrip OK")
