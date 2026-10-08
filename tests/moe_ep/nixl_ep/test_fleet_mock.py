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
    """Skip on hosts that can't run the CUDA tensor parts of these tests.

    The backend arch check is mocked below because these tests validate Buffer
    call sequencing and argument marshaling, not sm_90+ runtime support.
    They still allocate CUDA tensors in the handle tests, and the EP runtime
    wheels require a CUDA-13 torch build.
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
            self.num_ranks = num_ranks
            self.num_experts_per_rank = num_experts_per_rank
            # 0xFF init: every entry starts masked, self unmasked.
            self.mask = [-1] * num_ranks
            self.mask[self.rank] = 0
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
            if activate:
                for r in ranks:
                    self.mask[r] = 0  # 0 == unmasked/active in NIXL

        def disconnect_ranks(self, ranks):
            self.calls.append(("disconnect_ranks", list(ranks)))

        # --- mask API, faithful to the transport's semantics -------------
        # The real buffer is 0xFF-memset at allocation, so an untouched entry
        # reads back as -1, and the kernels test `mask_buffer[r] != 0`.

        def update_mask_buffer(self, rank_to_mask, mask=False):
            self.mask[rank_to_mask] = 1 if mask else 0
            self.calls.append(("update_mask", rank_to_mask, mask))

        def query_mask_buffer(self, out):
            import torch

            assert out.numel() == self.num_ranks, (
                f"query_mask_buffer wants exactly max_num_ranks ({self.num_ranks}) "
                f"entries, got {out.numel()}"
            )
            out.copy_(torch.tensor(self.mask, dtype=torch.int32, device=out.device))
            self.calls.append(("query_mask",))

        def clean_mask_buffer(self):
            # Zeroes ALL capacity entries -- including the never-connected
            # tail, which it thereby marks ACTIVE.
            self.mask = [0] * self.num_ranks
            self.calls.append(("clean_mask",))

        def low_latency_dispatch(self, x, topk_idx, max_tokens, num_experts, **kw):
            import torch

            kw = dict(kw, stream=torch.cuda.current_stream().cuda_stream)
            self.calls.append(("dispatch", max_tokens, num_experts, kw))
            # Return (recv_x, recv_count, handle, event, hook) with the real
            # LL EXPERT_MAJOR recv shape [num_local, max_tokens * ranks, hidden].
            num_local = self.num_experts_per_rank
            recv_x = torch.empty(
                num_local,
                max_tokens * self.num_ranks,
                x.size(1),
                dtype=x.dtype,
                device=x.device,
            )
            if kw.get("use_fp8"):
                scales = torch.empty(
                    num_local,
                    max_tokens * self.num_ranks,
                    x.size(1) // 128,
                    dtype=torch.float32,
                    device=x.device,
                )
                recv_x = (recv_x.to(torch.float8_e4m3fn), scales)
            recv_count = torch.zeros(num_local, dtype=torch.int32, device=x.device)
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
    from flashinfer.moe_ep.backends.split.comm.nixl_ep import fleet

    with (
        mock.patch.object(fleet, "_load_nixl_ep", return_value=fake_nixl_ep_module),
        mock.patch.object(fleet, "_require_built", return_value=None),
        mock.patch.object(fleet, "validate_arch_for_backend", return_value=None),
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
        FleetAlgoKnobTopologyCapacity,
        FleetParams,
        create_fleet,
    )

    bootstrap = BootstrapConfig(world_size=4, rank=0, tcp_store=mock.Mock())
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=4096,
    )
    # Capacity must cover the grown world: the transport's per-rank buffers
    # are sized once, at construction.
    fleet = create_fleet(
        bootstrap, params, [FleetAlgoKnobTopologyCapacity(n=8)], backend="nixl_ep"
    )

    # Grow from 4 → 6 ranks: new ranks [4, 5] should appear in connect_ranks.
    fleet.update_topology(BootstrapConfig(world_size=6, rank=0, tcp_store=mock.Mock()))
    buf = fake_buffer_cls.instances[-1]
    added = next(
        (c for c in buf.calls if c[0] == "connect_ranks" and c[1] == [4, 5]), None
    )
    assert added is not None, f"expected connect_ranks([4, 5]) in {buf.calls}"

    fake_buffer_cls.instances.clear()


def _make_fleet(create_fleet, BootstrapConfig, FleetParams, knobs=(), **bs_kw):
    bs_kw.setdefault("tcp_store", mock.Mock())
    bootstrap = BootstrapConfig(world_size=4, rank=0, **bs_kw)
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=4096,
    )
    return create_fleet(bootstrap, params, knobs, backend="nixl_ep")


def test_dispatch_surfaces_counts_and_num_tokens(patched_loader, fake_buffer_cls):
    import torch

    _skip_unless_ep_capable()

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DispatchInputParams,
        FleetParams,
        HandleParams,
        create_fleet,
    )

    fleet = _make_fleet(create_fleet, BootstrapConfig, FleetParams)
    topk = torch.zeros(64, 4, dtype=torch.int64, device="cuda")
    h = fleet.create_handle(HandleParams(topk_ids=topk))
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    out = h.dispatch(DispatchInputParams(x=[x]))

    # The library's recv counts must be surfaced, not discarded.
    assert out.expert_counts is not None
    assert out.expert_counts.shape == (2,)  # 8 experts / 4 ranks
    assert out.expert_scales is None  # bf16 dispatch
    # num_tokens is the per-expert row count of the recv buffer
    # (max_tokens_per_rank * ranks — same semantics as nccl_ep LL).
    assert out.get_num_tokens() == 64 * 4
    assert out.get_num_tokens() == out.expert_tensors.size(1)

    fake_buffer_cls.instances.clear()


def test_dispatch_fp8_surfaces_scales(patched_loader, fake_buffer_cls):
    import torch

    _skip_unless_ep_capable()

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DispatchInputParams,
        FleetAlgoKnobQuantization,
        FleetParams,
        HandleParams,
        QuantType,
        create_fleet,
    )

    fleet = _make_fleet(
        create_fleet,
        BootstrapConfig,
        FleetParams,
        knobs=[FleetAlgoKnobQuantization(quants=frozenset({QuantType.FP8E4M3}))],
    )
    topk = torch.zeros(64, 4, dtype=torch.int64, device="cuda")
    h = fleet.create_handle(HandleParams(topk_ids=topk))
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    out = h.dispatch(DispatchInputParams(x=[x]))

    # use_fp8 reached the library, and the (data, scales) pair is surfaced.
    disp = next(c for c in fake_buffer_cls.instances[-1].calls if c[0] == "dispatch")
    assert disp[3]["use_fp8"] is True
    assert out.expert_tensors.dtype == torch.float8_e4m3fn
    assert out.expert_scales is not None
    assert out.expert_scales.dtype == torch.float32

    fake_buffer_cls.instances.clear()


def test_user_stream_knob_accepted_but_not_redirected(patched_loader, fake_buffer_cls):
    """The user-stream knob is accepted without error but must NOT wrap the
    Buffer calls in a foreign stream context: doing so breaks NIXL's async
    RDMA completion and deadlocks combine. NIXL runs on the current stream.
    """
    import torch

    _skip_unless_ep_capable()

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DispatchInputParams,
        FleetParams,
        HandleAlgoKnobUserStream,
        HandleParams,
        create_fleet,
    )

    fleet = _make_fleet(create_fleet, BootstrapConfig, FleetParams)
    topk = torch.zeros(64, 4, dtype=torch.int64, device="cuda")
    side_stream = torch.cuda.Stream()
    # Passing the knob must not raise and must not switch the active stream.
    h = fleet.create_handle(
        HandleParams(topk_ids=topk),
        algo_knobs=[HandleAlgoKnobUserStream(stream=side_stream.cuda_stream)],
    )
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    default_stream = torch.cuda.current_stream().cuda_stream
    _ = h.dispatch(DispatchInputParams(x=[x]))

    disp = next(c for c in fake_buffer_cls.instances[-1].calls if c[0] == "dispatch")
    # The Buffer saw the CURRENT stream, not the side stream from the knob.
    assert disp[3]["stream"] == default_stream
    assert disp[3]["stream"] != side_stream.cuda_stream

    fake_buffer_cls.instances.clear()


def test_store_derived_from_default_group(patched_loader, fake_buffer_cls, tmp_path):
    import torch.distributed as dist

    _skip_unless_ep_capable()

    from flashinfer.moe_ep import BootstrapConfig, FleetParams, create_fleet

    if dist.is_initialized():
        pytest.skip("needs an uninitialized torch.distributed")
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(str(tmp_path / "store"), 1),
        rank=0,
        world_size=1,
    )
    try:
        bootstrap = BootstrapConfig(world_size=1, rank=0)
        params = FleetParams(
            num_experts=8,
            max_tokens_per_rank=64,
            token_hidden_size=4096,
        )
        _ = create_fleet(bootstrap, params, [], backend="nixl_ep")
        buf = fake_buffer_cls.instances[-1]
        # No tcp_store passed: the fleet derives a namespaced PrefixStore
        # from the default store instead of erroring out.
        assert isinstance(buf.tcp_store_group, dist.PrefixStore)
    finally:
        dist.destroy_process_group()
        fake_buffer_cls.instances.clear()


def test_missing_store_raises_without_dist(patched_loader, fake_buffer_cls):
    import torch.distributed as dist

    _skip_unless_ep_capable()

    from flashinfer.moe_ep import BootstrapConfig, FleetParams, create_fleet

    if dist.is_initialized():
        pytest.skip("needs an uninitialized torch.distributed")
    bootstrap = BootstrapConfig(world_size=4, rank=0)
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=4096,
    )
    with pytest.raises(ValueError, match="rendezvous store"):
        create_fleet(bootstrap, params, [], backend="nixl_ep")

    fake_buffer_cls.instances.clear()


# ------------------------------------------------------------ fault tolerance


def _ft_fleet(world_size=4, capacity=None, **knob_kwargs):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetAlgoKnobFaultTolerance,
        FleetAlgoKnobTopologyCapacity,
        FleetParams,
        create_fleet,
    )

    knobs = [FleetAlgoKnobFaultTolerance(**knob_kwargs)]
    if capacity is not None:
        knobs.append(FleetAlgoKnobTopologyCapacity(n=capacity))
    return create_fleet(
        BootstrapConfig(world_size=world_size, rank=0, tcp_store=mock.Mock()),
        FleetParams(num_experts=8, max_tokens_per_rank=64, token_hidden_size=4096),
        knobs,
        backend="nixl_ep",
    )


def test_ft_timeout_reaches_buffer_ctor(patched_loader, fake_buffer_cls):
    _skip_unless_ep_capable()

    _ft_fleet(timeout_ms=2500)
    assert fake_buffer_cls.instances[-1].kwargs["timeout_ms"] == 2500


def test_ft_zero_timeout_leaves_transport_default(patched_loader, fake_buffer_cls):
    _skip_unless_ep_capable()

    _ft_fleet(timeout_ms=0)
    assert "timeout_ms" not in fake_buffer_cls.instances[-1].kwargs


def test_ctor_without_timeout_support_is_actionable(patched_loader, fake_buffer_cls):
    """An older vendored nixl_ep build has no timeout_ms parameter."""
    _skip_unless_ep_capable()

    from flashinfer.moe_ep.errors import MoEEpFaultToleranceUnsupportedError

    orig_init = fake_buffer_cls.__init__

    def _no_timeout_init(self, rank=0, low_latency_mode=True, tcp_store_group=None):
        orig_init(self, rank, low_latency_mode, tcp_store_group)

    with (
        mock.patch.object(fake_buffer_cls, "__init__", _no_timeout_init),
        pytest.raises(MoEEpFaultToleranceUnsupportedError, match="timeout_ms"),
    ):
        _ft_fleet(timeout_ms=2500)


def test_query_active_mask_polarity_and_trim(patched_loader, fake_buffer_cls):
    """The regression test for NIXL's inverted, capacity-length mask buffer.

    Raw is `nonzero == masked` with a 0xFF init, so an untouched tail entry
    reads back as -1. `(raw == 0)` is the only correct normalization: a
    `1 - raw` inversion would return 2 for those tail ranks.
    """
    _skip_unless_ep_capable()

    import torch

    fleet = _ft_fleet(world_size=4, capacity=8)
    buf = fake_buffer_cls.instances[-1]
    buf.mask = [-1, 0, 0, 0, -1, -1, -1, -1]  # rank 0 dead; tail never connected

    got = fleet.query_active_mask()
    assert got.tolist() == [0, 1, 1, 1]  # trimmed to world_size, 1 = active
    assert got.dtype == torch.int32
    assert got.numel() == 4


def test_connect_ranks_leaves_capacity_tail_masked(patched_loader, fake_buffer_cls):
    _skip_unless_ep_capable()

    _ft_fleet(world_size=4, capacity=8)
    buf = fake_buffer_cls.instances[-1]
    assert buf.mask[:4] == [0, 0, 0, 0]  # live world active
    assert buf.mask[4:] == [-1, -1, -1, -1]  # tail still masked from the 0xFF init


def test_set_active_mask_pushes_only_the_diff(patched_loader, fake_buffer_cls):
    """Each update_mask_buffer is a kernel launch, so a no-op set must be free."""
    _skip_unless_ep_capable()

    fleet = _ft_fleet(world_size=4)
    buf = fake_buffer_cls.instances[-1]

    fleet.set_active_mask([1, 1, 0, 1])
    updates = [c for c in buf.calls if c[0] == "update_mask"]
    assert updates == [("update_mask", 2, True)]

    # Applying the identical mask again must issue nothing.
    fleet.set_active_mask([1, 1, 0, 1])
    assert [c for c in buf.calls if c[0] == "update_mask"] == updates

    # Un-masking rank 2 pushes exactly one un-mask.
    fleet.set_active_mask([1, 1, 1, 1])
    assert [c for c in buf.calls if c[0] == "update_mask"][-1] == (
        "update_mask",
        2,
        False,
    )


def test_set_active_mask_rejects_masking_self(patched_loader, fake_buffer_cls):
    _skip_unless_ep_capable()

    fleet = _ft_fleet(world_size=4)
    with pytest.raises(ValueError, match="cannot mask itself"):
        fleet.set_active_mask([0, 1, 1, 1])


def test_query_fault_diffs_against_applied(patched_loader, fake_buffer_cls):
    _skip_unless_ep_capable()

    fleet = _ft_fleet(world_size=4)
    assert fleet.query_fault() is False
    # Transport self-masks rank 3 on timeout.
    fake_buffer_cls.instances[-1].mask[3] = 1
    assert fleet.query_fault() is True


def test_clear_faults_readmit_remasks_the_capacity_tail(
    patched_loader, fake_buffer_cls
):
    """clean_mask_buffer zeroes ALL capacity entries, marking the
    never-connected tail ACTIVE. Without the re-mask that is a live bug on any
    fleet sized above its world."""
    _skip_unless_ep_capable()

    fleet = _ft_fleet(world_size=4, capacity=8)
    buf = fake_buffer_cls.instances[-1]
    buf.mask[2] = 1  # a rank died

    fleet.clear_faults(readmit=True)

    assert buf.mask[:4] == [0, 0, 0, 0], "survivors + readmitted rank active"
    assert buf.mask[4:] == [1, 1, 1, 1], "capacity tail must be re-masked"
    kinds = [c[0] for c in buf.calls]
    assert "clean_mask" in kinds
    assert kinds.index("clean_mask") < len(kinds) - 1  # re-masks come after


def test_clear_faults_without_readmit_is_a_noop(patched_loader, fake_buffer_cls):
    """NIXL has no sticky host error flag to re-arm."""
    _skip_unless_ep_capable()

    fleet = _ft_fleet(world_size=4)
    buf = fake_buffer_cls.instances[-1]
    before = list(buf.calls)
    fleet.clear_faults()
    assert buf.calls == before


def test_ft_methods_raise_without_the_knob(patched_loader, fake_buffer_cls):
    _skip_unless_ep_capable()

    from flashinfer.moe_ep import BootstrapConfig, FleetParams, create_fleet
    from flashinfer.moe_ep.errors import MoEEpFaultToleranceUnsupportedError

    fleet = create_fleet(
        BootstrapConfig(world_size=4, rank=0, tcp_store=mock.Mock()),
        FleetParams(num_experts=8, max_tokens_per_rank=64, token_hidden_size=4096),
        [],
        backend="nixl_ep",
    )
    assert fleet.supports_fault_tolerance is False
    with pytest.raises(MoEEpFaultToleranceUnsupportedError):
        fleet.query_active_mask()


def test_query_fault_rejects_graph_capture(patched_loader, fake_buffer_cls):
    """Both backends must agree: query_fault is not capturable.

    On nixl it already raised via query_active_mask; on nccl it silently
    returned the capture-time answer until the guard was added. This pins the
    shared behaviour.
    """
    _skip_unless_ep_capable()

    import torch

    fleet = _ft_fleet(world_size=4)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), pytest.raises(RuntimeError, match="CUDA-graph capture"):
        fleet.query_fault()
