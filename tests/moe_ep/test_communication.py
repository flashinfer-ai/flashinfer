"""Host-only tests of the MoEEpCommunication layer and its backends.

The backends' kernels and transports are replaced by stubs, so these tests
check the contract plumbing (payload order, id translation, state handling,
registry and split-layer routing) without GPUs. Multi-GPU correctness lives in
test_moe_ep_communication_multirank.py.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.fused_moe import QuantFormat
from flashinfer.moe_ep import (
    BootstrapConfig,
    CakeAlltoAll,
    CakeAlltoAllConfig,
    EpAlgorithm,
    EpLayout,
    FleetParams,
    IdentityConfig,
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpConfigError,
    MoEEpDispatchResult,
    MoEEpSplitLayer,
    MoEEpTensors,
    NCCLEPConfig,
    NVLinkOneSidedAlltoAll,
    NVLinkOneSidedConfig,
    NVLinkTwoSidedAlltoAll,
    SplitConfig,
    create_communication,
    dummy_moe_weights,
    register_communication,
)
from flashinfer.moe_ep.core.comm.communication import (
    _COMMUNICATION_REGISTRY,
    is_communication_backend,
)


class _LoopbackConfig:
    backend_name = "test_loopback"


@pytest.fixture
def loopback_backend():
    """Single-rank backend that receives its own tokens, padded to capacity."""
    log: dict = {}

    @register_communication("test_loopback")
    class _Loopback(MoEEpCommunication):
        def __init__(self, bootstrap, params, config=None, tag=None):
            super().__init__(bootstrap, params)
            log["config"] = config
            log["tag"] = tag

        @classmethod
        def is_platform_supported(cls) -> bool:
            return True

        def dispatch(
            self,
            hidden_states,
            topk_ids,
            topk_weights=None,
            *,
            hidden_states_scale=None,
            max_tokens_per_rank=None,
            eplb_local_stats=None,
        ):
            rows = self.params.max_tokens_per_rank
            n = hidden_states.shape[0]
            recv = hidden_states.new_zeros(rows, hidden_states.shape[1])
            recv[:n] = hidden_states
            ids = torch.full(
                (rows, topk_ids.shape[1]),
                self.params.invalid_expert_id,
                dtype=torch.int32,
            )
            ids[:n] = topk_ids
            weights = torch.zeros(rows, topk_ids.shape[1])
            weights[:n] = topk_weights
            log["local_num_tokens"] = n
            return MoEEpDispatchResult(
                hidden_states=recv,
                topk_ids=ids,
                topk_weights=weights,
                tokens_per_rank=rows,
            )

        def combine(self, expert_output, *, output=None):
            result = expert_output[: log["local_num_tokens"]]
            if output is not None:
                output.copy_(result)
                return output
            return result

        def destroy(self):
            log["destroyed"] = True

    yield log
    _COMMUNICATION_REGISTRY.pop("test_loopback", None)


def _params(**overrides) -> MoEEpCommParams:
    fields = dict(num_experts=4, top_k=2, max_tokens_per_rank=3, hidden_size=8)
    fields.update(overrides)
    return MoEEpCommParams(**fields)


class TestCommParams:
    def test_rejects_non_positive_sizes(self) -> None:
        with pytest.raises(ValueError, match="max_tokens_per_rank"):
            _params(max_tokens_per_rank=0)

    def test_rejects_top_k_above_num_experts(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            _params(top_k=5)

    def test_rejects_invalid_id_inside_expert_range(self) -> None:
        with pytest.raises(ValueError, match="invalid_expert_id"):
            _params(invalid_expert_id=2)
        assert _params(invalid_expert_id=4).invalid_expert_id == 4

    def test_token_dtype_defaults_to_bf16(self) -> None:
        assert _params().token_dtype == torch.bfloat16
        assert _params(dtype=torch.float32).token_dtype == torch.float32

    def test_dispatch_bytes_follow_the_dispatch_format(self) -> None:
        def dispatch_bytes(**overrides):
            return _params(hidden_size=7168, **overrides).dispatch_bytes_per_token

        assert dispatch_bytes() == 2 * 7168
        assert dispatch_bytes(dtype=torch.float32) == 4 * 7168
        assert dispatch_bytes(dispatch_format=QuantFormat.MXFP8) == 7168 + 224
        assert dispatch_bytes(dispatch_format=QuantFormat.NVFP4) == 3584 + 448
        with pytest.raises(ValueError, match="dispatch_format"):
            _params(dispatch_format=QuantFormat.INT4)


class TestRegistry:
    def test_builtin_backends_are_registered(self) -> None:
        for name in ("nvlink_one_sided", "cake", "nvlink_two_sided"):
            assert is_communication_backend(name)
        assert is_communication_backend(NVLinkOneSidedConfig())
        assert is_communication_backend(CakeAlltoAllConfig())
        # NCCL-EP and NIXL-EP are Fleet/Handle transports.
        assert not is_communication_backend("nccl_ep")
        assert not is_communication_backend("nixl_ep")
        assert not is_communication_backend(object())

    def test_cuda_graph_capability(self) -> None:
        assert NVLinkOneSidedAlltoAll.supports_cuda_graph
        assert CakeAlltoAll.supports_cuda_graph
        assert NVLinkTwoSidedAlltoAll.supports_cuda_graph

    def test_create_by_name_and_by_config(self, loopback_backend) -> None:
        bootstrap = BootstrapConfig(world_size=1, rank=0)
        comm = create_communication(bootstrap, _params(), "test_loopback", tag=7)
        assert loopback_backend["config"] is None
        assert loopback_backend["tag"] == 7
        assert comm.backend_name == "test_loopback"
        assert comm.num_local_experts == 4

        config = _LoopbackConfig()
        create_communication(bootstrap, _params(), config)
        assert loopback_backend["config"] is config

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown communication backend"):
            create_communication(
                BootstrapConfig(world_size=1, rank=0), _params(), "no_such_backend"
            )

    def test_experts_must_divide_across_ranks(self, loopback_backend) -> None:
        with pytest.raises(ValueError, match="divisible"):
            create_communication(
                BootstrapConfig(world_size=3, rank=0), _params(), "test_loopback"
            )


@pytest.fixture
def isolated_single_rank_gloo(tmp_path):
    import torch.distributed as dist

    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            init_method=(tmp_path / "gloo-rendezvous").as_uri(),
        )
    try:
        yield
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _split_layer(comm, layout=EpLayout.RANK_MAJOR, num_experts=4, hidden=8):
    return MoEEpSplitLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=3,
            token_hidden_size=hidden,
            dtype_bytes=4,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=layout,
        ),
        weights=dummy_moe_weights(num_local_experts=num_experts, hidden=hidden),
        backend=SplitConfig(comm=comm, kernel=IdentityConfig()),
    )


class TestSplitLayerRouting:
    def test_fleet_backends_keep_the_fleet_path(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "flashinfer.moe_ep.modes.split_layer.validate_arch_for_backend",
            lambda name: None,
        )
        layer = _split_layer(NCCLEPConfig(), layout=EpLayout.EXPERT_MAJOR)
        assert not layer._uses_communication()

    def test_unregistered_backend_is_rejected_at_init(self) -> None:
        with pytest.raises(MoEEpConfigError, match="registered as neither"):
            _split_layer("no_such_backend")

    def test_communication_backend_requires_rank_major(self, loopback_backend) -> None:
        with pytest.raises(MoEEpConfigError, match="RANK_MAJOR"):
            _split_layer(_LoopbackConfig(), layout=EpLayout.EXPERT_MAJOR)

    def test_communication_path_translates_routing_to_local_ids(
        self, loopback_backend, isolated_single_rank_gloo
    ) -> None:
        layer = _split_layer(_LoopbackConfig())
        assert layer._uses_communication()

        seen = {}
        compute = layer._kernel.compute

        def capture(ctx):
            seen["ctx"] = ctx
            return compute(ctx)

        layer._kernel.compute = capture
        x = torch.arange(16, dtype=torch.float32).view(2, 8)
        t = MoEEpTensors(
            hidden_states=x,
            topk_ids=torch.tensor([[0, 3], [2, 1]]),
            topk_weights=torch.full((2, 2), 0.5),
        )
        out = layer(t)

        torch.testing.assert_close(out, x)
        ctx = seen["ctx"]
        assert ctx.expert_tensors.shape == (1, 3, 8)
        assert ctx.num_tokens == 3
        # Padding row carries the invalid id, which is never local.
        assert ctx.recv_topk_idx.tolist() == [[0, 3], [2, 1], [-1, -1]]

        state = layer.create_graph_state(t)
        out = layer(t, graph_state=state)
        assert out is state.out
        torch.testing.assert_close(out, x)
        layer.destroy()
        assert state.destroyed
        assert loopback_backend["destroyed"]

    def test_graph_state_requires_a_capturable_backend(
        self, loopback_backend, isolated_single_rank_gloo, monkeypatch
    ) -> None:
        layer = _split_layer(_LoopbackConfig())
        t = MoEEpTensors(
            hidden_states=torch.zeros(2, 8),
            topk_ids=torch.zeros(2, 2, dtype=torch.int64),
            topk_weights=torch.ones(2, 2),
        )
        monkeypatch.setattr(
            _COMMUNICATION_REGISTRY["test_loopback"], "supports_cuda_graph", False
        )
        with pytest.raises(MoEEpConfigError, match="CUDA graph"):
            layer.create_graph_state(t)
        layer.destroy()


class _FakeCakeOps:
    """Records the ``moe_a2a_*`` ops CakeAlltoAll calls; dispatch returns zeros."""

    def __init__(self) -> None:
        self.calls: dict = {}

    def workspace_size_per_rank(self, *args, **kwargs):
        self.calls["workspace_size"] = (args, kwargs)
        return 1 << 12

    def dispatch(self, topk_ids, payloads, workspace, metainfo, rt, *args, **kwargs):
        self.calls["dispatch"] = (payloads, rt, args, kwargs)
        ep_size = workspace.shape[0]
        recv = [torch.zeros(ep_size, rt, *p.shape[1:], dtype=p.dtype) for p in payloads]
        return recv, 0, None

    def sanitize(self, expert_ids, workspace, metainfo, ep_rank, invalid_id, **kwargs):
        self.calls["sanitize"] = (expert_ids, invalid_id, kwargs)

    def combine(self, payload, local_num_tokens, *args, **kwargs):
        self.calls["combine"] = (payload, local_num_tokens, args, kwargs)
        return torch.zeros(local_num_tokens, payload.shape[-1])


@pytest.fixture
def fake_cake(monkeypatch):
    import flashinfer.comm.mnnvl as mnnvl
    import flashinfer.comm.trtllm_moe_alltoall as a2a
    from flashinfer.comm.mapping import Mapping
    from flashinfer.moe_ep.backends.split.comm.cake import communication as cake

    ops = _FakeCakeOps()
    monkeypatch.setattr(
        a2a, "moe_a2a_get_workspace_size_per_rank", ops.workspace_size_per_rank
    )
    monkeypatch.setattr(a2a, "moe_a2a_dispatch", ops.dispatch)
    monkeypatch.setattr(a2a, "moe_a2a_sanitize_expert_ids", ops.sanitize)
    monkeypatch.setattr(a2a, "moe_a2a_combine", ops.combine)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "initialize", staticmethod(lambda: None))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    def acquire(self, mapping, workspace_size_per_rank):
        ops.calls["acquire"] = workspace_size_per_rank
        return {
            "workspace": torch.zeros(
                self.ep_size, workspace_size_per_rank, dtype=torch.uint8
            ),
            "metainfo": torch.zeros(1, dtype=torch.int64),
            "views": {},
            "refcount": 1,
            "key": None,
        }

    monkeypatch.setattr(CakeAlltoAll, "_acquire_workspace", acquire)
    monkeypatch.setattr(
        cake,
        "mnnvl_mapping_and_config",
        lambda bootstrap, comm_backend: (
            Mapping(
                world_size=bootstrap.world_size,
                rank=bootstrap.rank,
                tp_size=bootstrap.world_size,
                moe_ep_size=bootstrap.world_size,
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        CakeAlltoAll, "is_platform_supported", classmethod(lambda cls: True)
    )
    return ops


def test_cake_sizes_dispatch_by_format(fake_cake) -> None:
    create_communication(
        BootstrapConfig(world_size=2, rank=0),
        _params(hidden_size=64, dispatch_format=QuantFormat.NVFP4),
        "cake",
    )
    args, kwargs = fake_cake.calls["workspace_size"]
    assert kwargs["backend"] == "cake"
    _, _, dispatch_bytes, combine_bytes, _ = args
    # NVFP4 values and scales, then int32 ids and FP32 weights for top_k=2.
    assert dispatch_bytes == 32 + 4 + 2 * 4 * 2
    assert combine_bytes == 64 * 2


def test_cake_payload_plumbing(fake_cake) -> None:
    comm = create_communication(
        BootstrapConfig(world_size=2, rank=0),
        _params(),
        CakeAlltoAllConfig(use_low_precision_combine=True),
    )
    assert isinstance(comm, CakeAlltoAll)

    hidden = torch.ones(2, 8, dtype=torch.bfloat16)
    scale = torch.ones(2, 1, dtype=torch.uint8)
    ids = torch.tensor([[0, 3], [1, 2]])
    weights = torch.ones(2, 2)
    result = comm.dispatch(
        hidden, ids, weights, hidden_states_scale=scale, max_tokens_per_rank=2
    )

    payloads, rt, _, kwargs = fake_cake.calls["dispatch"]
    assert rt == 2
    assert [p.dtype for p in payloads] == [
        torch.bfloat16,
        torch.uint8,
        torch.int32,
        torch.float32,
    ]
    assert kwargs["backend"] == "cake"
    sanitized_ids, invalid_id, kwargs = fake_cake.calls["sanitize"]
    assert sanitized_ids.dtype == torch.int32 and sanitized_ids.shape == (2, 2, 2)
    assert invalid_id == -1 and kwargs["backend"] == "cake"
    assert result.tokens_per_rank == 2
    assert result.hidden_states.shape == (4, 8)
    assert result.hidden_states_scale.shape == (4, 1)
    assert result.topk_ids.shape == (4, 2)
    assert result.topk_weights.dtype == torch.float32

    buffer = comm.get_combine_input_buffer(torch.bfloat16)
    assert buffer.shape == (4, 8)
    comm.combine(buffer)
    payload, local_num_tokens, args, kwargs = fake_cake.calls["combine"]
    assert payload.shape == (2, 2, 8) and local_num_tokens == 2
    # Positional payload_in_workspace follows the combine payload offset.
    assert args[-1] is True
    assert kwargs["use_low_precision"] is True and kwargs["backend"] == "cake"

    comm.dispatch(hidden, ids, weights)
    comm.combine(torch.zeros(6, 8, dtype=torch.bfloat16))
    assert fake_cake.calls["combine"][2][-1] is False

    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        comm.dispatch(hidden, ids, weights, max_tokens_per_rank=4)
    with pytest.raises(ValueError, match="enable_rank_mask"):
        comm.dispatch(hidden, ids, weights, active_rank_mask=torch.ones(4))
    with pytest.raises(RuntimeError, match="before dispatch"):
        comm.combine(torch.zeros(6, 8, dtype=torch.bfloat16))

    comm.destroy()
    comm.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        comm.dispatch(hidden, ids, weights)


def test_cake_checkpoint_needs_an_idle_live_instance(fake_cake) -> None:
    comm = create_communication(
        BootstrapConfig(world_size=2, rank=0), _params(), "cake"
    )
    comm.dispatch(
        torch.ones(2, 8, dtype=torch.bfloat16), torch.tensor([[0, 3], [1, 2]])
    )
    with pytest.raises(RuntimeError, match="between dispatch and combine"):
        comm.checkpoint_prepare()
    comm.combine(torch.zeros(6, 8, dtype=torch.bfloat16))
    comm.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        comm.checkpoint_prepare()
    with pytest.raises(RuntimeError, match="destroyed"):
        comm.checkpoint_restore(None)


def test_cake_needs_compute_capability_10_0_or_10_3(monkeypatch) -> None:
    from flashinfer.moe_ep.backends.split.comm.cake import communication as cake

    monkeypatch.setattr(cake, "nvlink_platform_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    assert not CakeAlltoAll.is_platform_supported()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 3))
    assert CakeAlltoAll.is_platform_supported()


_ONE_SIDED_LAYOUT = {
    "NUM_METAINFO_FIELDS": 25,
    "COMBINE_INPUT_SIZE_INDEX": 16,
    "WORKSPACE_SIZE_INDEX": 24,
    "MAX_RANKS": 256,
    "MAX_PAYLOADS": 4,
    "WORKSPACE_ALIGNMENT": 256,
}


class _FakeOneSidedModule:
    """Records the NVLink one-sided ops; dispatch packs payloads from offset 0."""

    def __init__(self) -> None:
        self.calls: dict = {}

    def moe_a2a_get_workspace_layout(self, *args):
        self.calls["layout"] = args[:-1]
        metainfo = args[-1]
        metainfo.zero_()
        metainfo[_ONE_SIDED_LAYOUT["COMBINE_INPUT_SIZE_INDEX"]] = 1 << 16

    def moe_a2a_dispatch(self, ids, payloads, workspace, metainfo, rt, *args):
        self.calls["dispatch"] = (ids, payloads, rt, args)
        offsets, offset = [], 0
        for payload in payloads:
            offsets.append(offset)
            nbytes = workspace.shape[0] * rt * payload.shape[1] * payload.element_size()
            offset += -(-nbytes // 128) * 128
        return offsets, 1 << 15, -1

    def moe_a2a_sanitize_expert_ids(self, *args):
        self.calls["sanitize"] = args

    def moe_a2a_combine(self, *args):
        self.calls["combine"] = args


@pytest.fixture
def fake_one_sided(monkeypatch):
    """NVLinkOneSidedAlltoAll on a fake kernel module and a host workspace."""
    from types import SimpleNamespace

    import flashinfer.comm.mnnvl as mnnvl
    import flashinfer.utils
    from flashinfer.comm.mapping import Mapping
    from flashinfer.moe_ep.backends.split.comm.nvlink_one_sided import (
        communication as one_sided,
    )

    module = _FakeOneSidedModule()
    probe: dict = {"reason": None, "probed": 0}

    def acquire(self, mapping, comm, metainfo, cft_capable):
        return {
            "key": None,
            "workspace": torch.zeros(self.ep_size, 1 << 16, dtype=torch.uint8),
            "metainfo": metainfo,
            "views": {},
            "refcount": 1,
            "cft_ready": cft_capable,
        }

    def unsupported_reason(device_index):
        probe["probed"] += 1
        return probe["reason"]

    monkeypatch.setattr(one_sided, "get_nvlink_one_sided_module", lambda: module)
    monkeypatch.setattr(
        one_sided, "layout_constants", lambda: SimpleNamespace(**_ONE_SIDED_LAYOUT)
    )
    monkeypatch.setattr(one_sided, "_cft_unsupported_reason", unsupported_reason)
    monkeypatch.setattr(
        one_sided,
        "mnnvl_mapping_and_config",
        lambda bootstrap, comm_backend: (
            Mapping(
                world_size=bootstrap.world_size,
                rank=bootstrap.rank,
                tp_size=bootstrap.world_size,
                moe_ep_size=bootstrap.world_size,
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        NVLinkOneSidedAlltoAll, "is_platform_supported", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(NVLinkOneSidedAlltoAll, "_acquire_workspace", acquire)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "initialize", staticmethod(lambda: None))
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "set_comm_from_config",
        staticmethod(lambda mapping, config=None: None),
    )
    monkeypatch.setattr(
        mnnvl.MnnvlMemory, "get_comm", staticmethod(lambda mapping: None)
    )
    monkeypatch.setattr(mnnvl, "all_ranks_agree", lambda comm, local: local)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(flashinfer.utils, "device_support_pdl", lambda device: True)
    return SimpleNamespace(module=module, probe=probe)


def _one_sided(params=None, **config):
    return create_communication(
        BootstrapConfig(world_size=2, rank=0),
        _params() if params is None else params,
        NVLinkOneSidedConfig(**config),
    )


def test_nvlink_one_sided_sizes_workspace_by_format(fake_one_sided) -> None:
    _one_sided(
        _params(hidden_size=64, dispatch_format=QuantFormat.NVFP4),
        extra_payload_bytes_per_token=16,
        eplb_stats_num_experts=4,
    )
    ep, max_tokens, top_k, dispatch, combine_input, combine_recv, eplb, cft = (
        fake_one_sided.module.calls["layout"]
    )
    assert (ep, max_tokens, top_k, eplb, cft) == (2, 3, 2, 4, True)
    tokens = ep * max_tokens
    # NVFP4 values and scales, int32 ids, FP32 weights and the extra payload,
    # plus one alignment unit for each payload's own boundary.
    assert dispatch == tokens * (32 + 4 + 2 * 4 * 2 + 16) + 4 * 256
    assert combine_input == tokens * 64 * 2
    assert combine_recv == tokens * 64 * 2


def test_nvlink_one_sided_cft_selection(fake_one_sided) -> None:
    comm = _one_sided(cft=False)
    assert not comm.cft_enabled
    assert fake_one_sided.probe["probed"] == 0
    assert fake_one_sided.module.calls["layout"][5] == 0

    fake_one_sided.probe["reason"] = "the device does not support it"
    assert not _one_sided().cft_enabled

    fake_one_sided.probe["reason"] = None
    comm = _one_sided(use_low_precision_combine=True, cft_max_tokens_for_dispatch=2)
    assert comm.cft_enabled
    # FP8 wire format sizes the CFT receive inbox.
    assert fake_one_sided.module.calls["layout"][5] == 6 * 8
    assert comm._use_cft(2, 2) and not comm._use_cft(3, 2)
    assert _one_sided(cft=True)._use_cft(3, 2)


def test_nvlink_one_sided_checkpoint_needs_an_idle_live_instance(
    fake_one_sided, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    comm = _one_sided(cft=False)
    comm.dispatch(
        torch.ones(2, 8, dtype=torch.bfloat16), torch.tensor([[0, 3], [1, 2]])
    )
    with pytest.raises(RuntimeError, match="between dispatch and combine"):
        comm.checkpoint_prepare()
    comm._round = None
    comm.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        comm.checkpoint_prepare()
    with pytest.raises(RuntimeError, match="destroyed"):
        comm.checkpoint_restore(None)


def test_nvlink_one_sided_payload_plumbing(fake_one_sided) -> None:
    calls = fake_one_sided.module.calls
    comm = _one_sided(
        _params(num_experts=8, top_k=4), cft_max_tokens_for_dispatch=2, timeout_sec=60
    )
    hidden = torch.ones(2, 8, dtype=torch.bfloat16)
    scale = torch.ones(2, 16, dtype=torch.uint8)
    ids = torch.tensor([[0, 3, 5, 7], [1, 2, 4, 6]])
    weights = torch.ones(2, 4)
    result = comm.dispatch(
        hidden, ids, weights, hidden_states_scale=scale, max_tokens_per_rank=2
    )

    sent_ids, payloads, rt, args = calls["dispatch"]
    assert sent_ids.dtype == torch.int32 and rt == 2
    assert [p.dtype for p in payloads] == [
        torch.bfloat16,
        torch.uint8,
        torch.int32,
        torch.float32,
    ]
    rank, ep, top_k, num_experts, eplb, use_cft, id_index, invalid = args[:8]
    assert (rank, ep, top_k, num_experts, eplb) == (0, 2, 4, 8, None)
    # Every payload row is a multiple of 16 bytes, so this small step uses CFT,
    # which also fills the padding rows' expert ids.
    assert use_cft and id_index == 2 and invalid == -1
    assert args[-2] == 60
    assert "sanitize" not in calls
    assert result.tokens_per_rank == 2
    assert result.hidden_states.shape == (4, 8)
    assert result.hidden_states_scale.shape == (4, 16)
    assert result.topk_ids.shape == (4, 4)
    assert result.topk_weights.dtype == torch.float32

    buffer = comm.get_combine_input_buffer(torch.bfloat16)
    assert buffer.shape == (4, 8)
    out = comm.combine(buffer)
    assert out.shape == (2, 8)
    payload, local_num_tokens = calls["combine"][:2]
    assert payload.shape == (2, 2, 8) and local_num_tokens == 2
    # A 16-byte BF16 row allows CFT for the combine as well.
    assert calls["combine"][11] is True and calls["combine"][14] is out

    # The full-capacity step exceeds the CFT limit: the fence path runs and
    # the padding rows are sanitized separately.
    comm.dispatch(hidden, ids, weights)
    assert calls["dispatch"][3][5] is False and calls["dispatch"][3][6] == -1
    assert calls["sanitize"][0].shape == (2, 3, 4)
    output = torch.empty(2, 8, dtype=torch.bfloat16)
    assert (
        comm.combine(torch.zeros(6, 8, dtype=torch.bfloat16), output=output) is output
    )

    # Eight-byte expert-id rows cannot use CFT counted writes.
    narrow = _one_sided(_params(num_experts=8, top_k=2))
    narrow.dispatch(hidden, ids[:, :2], max_tokens_per_rank=2)
    assert calls["dispatch"][3][5] is False

    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        comm.dispatch(hidden, ids, weights, max_tokens_per_rank=4)
    with pytest.raises(RuntimeError, match="before dispatch"):
        comm.combine(buffer)


def test_nvlink_one_sided_rank_mask(fake_one_sided) -> None:
    comm = _one_sided(enable_rank_mask=True)
    mask = comm.active_rank_mask([0, 1, 255 % comm.ep_size])
    assert mask.dtype == torch.uint64 and mask.tolist() == [3, 0, 0, 0]
    with pytest.raises(ValueError, match="out of range"):
        comm.active_rank_mask([2])
    with pytest.raises(ValueError, match="enable_rank_mask"):
        _one_sided().dispatch(
            torch.ones(2, 8, dtype=torch.bfloat16),
            torch.tensor([[0, 3], [1, 2]]),
            active_rank_mask=mask,
        )


def test_nvlink_one_sided_validates_config(fake_one_sided) -> None:
    with pytest.raises(ValueError, match="timeout_sec"):
        _one_sided(timeout_sec=0)
    with pytest.raises(ValueError, match="top_k"):
        _one_sided(_params(num_experts=8, top_k=3))
    with pytest.raises(ValueError, match="eplb_stats_num_experts"):
        _one_sided(eplb_stats_num_experts=5)


def test_nvlink_two_sided_marks_padding_rows_invalid(monkeypatch) -> None:
    import flashinfer.comm.mnnvl as mnnvl
    import flashinfer.comm.trtllm_alltoall as two_sided_ops
    from flashinfer.comm.mapping import Mapping
    from flashinfer.moe_ep.backends.split.comm.nvlink_two_sided import (
        communication as two_sided,
    )

    calls: dict = {"comm": []}

    def moe_prepare(*args):
        calls["prepare"] = args
        recv_ids = torch.tensor([[1, 3], [4, 4], [4, 4]], dtype=torch.int32)
        indices = [torch.zeros(1, dtype=torch.int32) for _ in range(5)]
        return (recv_ids, torch.ones(3, 2), *indices, None)

    def moe_comm(x, send_cumsum, send_indices, output, *args):
        calls["comm"].append((x, output, args))
        output.fill_(1.0)

    monkeypatch.setattr(two_sided_ops, "moe_prepare", moe_prepare)
    monkeypatch.setattr(two_sided_ops, "moe_comm", moe_comm)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "initialize", staticmethod(lambda: None))
    monkeypatch.setattr(
        NVLinkTwoSidedAlltoAll,
        "_acquire_workspaces",
        lambda self, mapping: {
            "workspace": "workspace",
            "prepare_workspace": "prepare_workspace",
        },
    )
    monkeypatch.setattr(
        two_sided,
        "mnnvl_mapping_and_config",
        lambda bootstrap, comm_backend: (Mapping(), None),
    )
    monkeypatch.setattr(
        NVLinkTwoSidedAlltoAll,
        "is_platform_supported",
        classmethod(lambda cls: True),
    )

    comm = create_communication(
        BootstrapConfig(world_size=1, rank=0), _params(), "nvlink_two_sided"
    )
    result = comm.dispatch(
        torch.ones(1, 8), torch.tensor([[1, 3]]), torch.ones(1, 2, dtype=torch.bfloat16)
    )
    prepare_args = calls["prepare"]
    assert prepare_args[0].dtype == torch.int32
    assert prepare_args[1].dtype == torch.float32
    assert prepare_args[3] == "prepare_workspace"
    assert result.topk_ids.tolist() == [[1, 3], [-1, -1], [-1, -1]]
    # Hidden states arrive in ep_size * tokens_per_rank rows.
    _, recv_hidden, args = calls["comm"][-1]
    assert recv_hidden.shape == (3, 8) and args[2] == "workspace"

    out = torch.empty(1, 8)
    assert comm.combine(torch.ones(3, 8), output=out) is out
    # One row per top-k slot of the single local token, summed over the slots.
    _, per_slot, _ = calls["comm"][-1]
    assert per_slot.shape == (2, 8)
    torch.testing.assert_close(out, torch.full((1, 8), 2.0))

    comm.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        comm.dispatch(torch.ones(1, 8), torch.tensor([[1, 3]]))

    with pytest.raises(ValueError, match="divisible by 4"):
        create_communication(
            BootstrapConfig(world_size=1, rank=0),
            _params(num_experts=6),
            "nvlink_two_sided",
        )
