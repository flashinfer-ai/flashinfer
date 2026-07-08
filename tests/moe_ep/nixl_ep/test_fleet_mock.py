"""Host-only unit tests for NixlEpFleet / NixlEpHandle (mocked Buffer).

These tests never touch a real GPU, RDMA fabric, or staged
``nixl_ep_cpp*.so``. The ``patched_loader`` fixture stubs ``_load_nixl_ep``
(so no ``libnixl.so`` is needed), patches ``_require_built`` (so a host
lacking a built backend doesn't raise ``MoEEpNotBuiltError``), and injects a
fake ``nixl_ep.Buffer`` whose methods record their call args.

What they verify is **call sequencing and arg marshaling**, not numerics:
that ``Buffer.update_memory_buffers`` + ``connect_ranks`` fire with the right
sizes at Fleet construction, that ``update_topology`` diffs the rank set, and
that combine rejects a missing topk-weights knob. Real end-to-end behavior is
covered by the on-cluster smoke + multirank tests (``tests/moe_ep/smoke_*.py``,
``tests/moe_ep/test_moe_ep_layer_multirank.py``).
"""

from __future__ import annotations

from unittest import mock

import pytest


def _skip_unless_ep_capable():
    """Skip on hosts that can't construct an EP Fleet even with mocks.

    ``create_fleet`` runs ``validate_arch_for_backend``, which requires a
    CUDA device and a CUDA-13 torch build (the EP runtime wheels ship
    CUDA-13 binaries only), so on older stacks these tests would fail in
    validation before reaching the mocked Buffer.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cuda_ver = torch.version.cuda
    try:
        cuda_major = int(cuda_ver.split(".")[0]) if cuda_ver else None
    except ValueError:
        cuda_major = None
    if cuda_major is not None and cuda_major < 13:
        pytest.skip(f"moe_ep requires a CUDA-13 torch build (got CUDA {cuda_ver})")


@pytest.fixture
def fake_buffer_cls():
    """Build a `Buffer` class that records ctor + method calls."""

    class _FakeBuffer:
        instances: list = []  # noqa: RUF012

        def __init__(
            self,
            rank=0,
            low_latency_mode=True,
            tcp_store_group=None,
            **kwargs,
        ):
            self.rank = rank
            self.low_latency_mode = low_latency_mode
            self.tcp_store_group = tcp_store_group
            self.kwargs = kwargs
            self.calls: list = []
            _FakeBuffer.instances.append(self)

        @staticmethod
        def get_rdma_size_hint(max_tokens, hidden, num_ranks, num_experts):
            # Toy formula: tokens * hidden * 2 * num_ranks bytes.
            return max_tokens * hidden * 2 * num_ranks

        def update_memory_buffers(
            self, num_ranks, num_experts_per_rank, num_rdma_bytes
        ):
            self.calls.append(
                (
                    "update_memory_buffers",
                    num_ranks,
                    num_experts_per_rank,
                    num_rdma_bytes,
                )
            )

        def connect_ranks(self, ranks, activate=True):
            self.calls.append(("connect_ranks", list(ranks)))

        def disconnect_ranks(self, ranks):
            self.calls.append(("disconnect_ranks", list(ranks)))

        def low_latency_dispatch(self, x, topk_idx, max_tokens, num_experts, **kw):
            import torch

            self.calls.append(("dispatch", max_tokens, num_experts, kw))
            # Return (recv_x, recv_count, handle, event, hook).
            recv_x = torch.empty(
                num_experts, max_tokens, x.size(1), dtype=x.dtype, device=x.device
            )
            recv_count = torch.tensor(
                [max_tokens // num_experts] * num_experts, device=x.device
            )
            handle = ("dummy_handle",)
            event = mock.Mock(current_stream_wait=mock.Mock())
            hook = mock.Mock()
            return recv_x, recv_count, handle, event, hook

        def low_latency_combine(self, x, topk_idx, topk_weights, handle, **kw):
            import torch

            self.calls.append(("combine", handle, kw))
            combined = torch.empty(
                topk_idx.size(0), x.size(-1), dtype=x.dtype, device=x.device
            )
            event = mock.Mock(current_stream_wait=mock.Mock())
            return combined, event, None

    return _FakeBuffer


@pytest.fixture
def fake_nixl_ep_module(fake_buffer_cls):
    """Inject a fake ``nixl_ep`` module + a fake ``_load_nixl_ep`` shim."""
    import sys

    fake_mod = mock.Mock()
    fake_mod.Buffer = fake_buffer_cls
    sys.modules["nixl_ep"] = fake_mod
    yield fake_mod
    del sys.modules["nixl_ep"]


@pytest.fixture
def patched_loader(fake_nixl_ep_module):
    """Bypass _load_nixl_ep so we don't need libnixl.so on the dev box."""
    from flashinfer.moe_ep.nixl_ep import fleet

    with (
        mock.patch.object(fleet, "_load_nixl_ep", return_value=fake_nixl_ep_module),
        mock.patch.object(fleet, "_require_built", return_value=None),
    ):
        yield fake_nixl_ep_module


def test_fleet_init_calls_update_memory_and_connect(patched_loader, fake_buffer_cls):
    _skip_unless_ep_capable()

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetParams,
        create_fleet,
    )

    # NIXL needs tcp_store; the mock doesn't actually use it.
    bootstrap = BootstrapConfig(world_size=4, rank=0, tcp_store=mock.Mock())
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=4096,
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
    )
    _ = create_fleet(bootstrap, params, [], backend="nixl_ep")

    assert len(fake_buffer_cls.instances) == 1
    buf = fake_buffer_cls.instances[-1]
    # update_memory_buffers + connect_ranks both got called.
    methods = [c[0] for c in buf.calls]
    assert "update_memory_buffers" in methods
    assert "connect_ranks" in methods
    # update_memory_buffers arg shape: (num_ranks=4, experts_per_rank=8/4=2, rdma_bytes>0).
    umb = next(c for c in buf.calls if c[0] == "update_memory_buffers")
    assert umb[1] == 4
    assert umb[2] == 2
    assert umb[3] > 0
    # connect_ranks targets [0, 1, 2, 3].
    cr = next(c for c in buf.calls if c[0] == "connect_ranks")
    assert cr[1] == [0, 1, 2, 3]
    # cleanup
    fake_buffer_cls.instances.clear()


def test_handle_combine_requires_topk_weights(patched_loader, fake_buffer_cls):
    import torch

    _skip_unless_ep_capable()

    from flashinfer.moe_ep import (
        BootstrapConfig,
        CombineInputParams,
        DispatchInputParams,
        EpAlgorithm,
        FleetParams,
        HandleParams,
        create_fleet,
    )

    bootstrap = BootstrapConfig(world_size=4, rank=0, tcp_store=mock.Mock())
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=4096,
        algorithm=EpAlgorithm.LOW_LATENCY,
    )
    fleet = create_fleet(bootstrap, params, [], backend="nixl_ep")

    topk = torch.zeros(64, 4, dtype=torch.int64, device="cuda")
    h = fleet.create_handle(HandleParams(topk_ids=topk))
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    _ = h.dispatch(DispatchInputParams(x=[x]))

    # combine without HandleAlgoKnobTopKWeights → ValueError.
    with pytest.raises(ValueError, match="HandleAlgoKnobTopKWeights"):
        h.combine(CombineInputParams(x=[x]))

    fake_buffer_cls.instances.clear()


def test_update_topology_diffs_ranks(patched_loader, fake_buffer_cls):
    _skip_unless_ep_capable()

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        create_fleet,
    )

    bootstrap = BootstrapConfig(world_size=4, rank=0, tcp_store=mock.Mock())
    params = FleetParams(num_experts=8, max_tokens_per_rank=64, token_hidden_size=4096)
    fleet = create_fleet(bootstrap, params, [], backend="nixl_ep")

    # Grow from 4 → 6 ranks: new ranks [4, 5] should appear in connect_ranks.
    fleet.update_topology(BootstrapConfig(world_size=6, rank=0, tcp_store=mock.Mock()))
    buf = fake_buffer_cls.instances[-1]
    added = next(
        (c for c in buf.calls if c[0] == "connect_ranks" and c[1] == [4, 5]), None
    )
    assert added is not None, f"expected connect_ranks([4, 5]) in {buf.calls}"

    fake_buffer_cls.instances.clear()
