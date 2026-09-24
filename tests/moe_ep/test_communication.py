"""Host-only tests of the MoEEpCommunication layer and its backends.

The backends' kernels and transports are replaced by stubs, so these tests
check the contract plumbing (payload order, id translation, state handling,
registry and split-layer routing) without GPUs. Multi-GPU correctness lives in
test_moe_ep_communication_multirank.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    CombineOutput,
    DispatchOutput,
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
    NcclEpCommunication,
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


class TestRegistry:
    def test_builtin_backends_are_registered(self) -> None:
        for name in ("nccl_ep", "nvlink_one_sided", "nvlink_two_sided"):
            assert is_communication_backend(name)
        assert is_communication_backend(NVLinkOneSidedConfig())
        assert not is_communication_backend("nixl_ep")
        assert not is_communication_backend(object())

    def test_cuda_graph_capability(self) -> None:
        assert NVLinkOneSidedAlltoAll.supports_cuda_graph
        assert NVLinkTwoSidedAlltoAll.supports_cuda_graph
        # Each NCCL-EP dispatch creates a handle on the host.
        assert not NcclEpCommunication.supports_cuda_graph

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

    def test_communication_backend_requires_rank_major(self, loopback_backend) -> None:
        with pytest.raises(MoEEpConfigError, match="RANK_MAJOR"):
            _split_layer(_LoopbackConfig(), layout=EpLayout.EXPERT_MAJOR)

    def test_round_trip_translates_routing_to_local_ids(
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
        out = layer(
            MoEEpTensors(
                hidden_states=x,
                topk_ids=torch.tensor([[0, 3], [2, 1]]),
                topk_weights=torch.full((2, 2), 0.5),
            )
        )

        torch.testing.assert_close(out, x)
        ctx = seen["ctx"]
        assert ctx.expert_tensors.shape == (1, 3, 8)
        assert ctx.num_tokens == 3
        # Padding row carries the invalid id, which is never local.
        assert ctx.recv_topk_idx.tolist() == [[0, 3], [2, 1], [-1, -1]]

        with pytest.raises(MoEEpConfigError, match="graph states"):
            layer.create_graph_state(
                MoEEpTensors(
                    hidden_states=x,
                    topk_ids=torch.zeros(2, 2, dtype=torch.int64),
                    topk_weights=torch.ones(2, 2),
                )
            )
        layer.destroy()
        assert loopback_backend["destroyed"]


class _StubFleetHandle:
    def __init__(self, log, output):
        self.log = log
        self.output = output
        self.destroyed = False

    def dispatch(self, params):
        self.log["dispatch_x"] = params.x
        return self.output

    def combine(self, params):
        self.log["combine_x"] = params.x[0]
        return CombineOutput(x=params.x[0].sum(dim=0))

    def complete(self):
        self.log["complete"] = True

    def destroy(self):
        self.destroyed = True


@pytest.fixture
def stub_nccl_fleet(monkeypatch):
    from flashinfer.moe_ep.core.comm.fleet import _BACKEND_REGISTRY

    log: dict = {}

    class _StubFleet:
        def __init__(self, bootstrap, params, algo_knobs):
            log["fleet_params"] = params
            self.destroyed = False

        def create_handle(self, params, algo_knobs=()):
            log["topk_ids"] = params.topk_ids
            log["knobs"] = list(algo_knobs)
            handle = _StubFleetHandle(log, log["dispatch_output"])
            log["handle"] = handle
            return handle

        def destroy(self):
            log["fleet_destroyed"] = True

    monkeypatch.setitem(_BACKEND_REGISTRY, "nccl_ep", _StubFleet)
    monkeypatch.setattr(
        NcclEpCommunication, "is_platform_supported", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *a, **k: SimpleNamespace(cuda_stream=0),
    )
    return log


def test_nccl_ep_communication_reports_global_ids(stub_nccl_fleet) -> None:
    log = stub_nccl_fleet
    # Rank 1 of 2 owns experts {2, 3}; each source rank sends up to 3 tokens.
    comm = create_communication(
        BootstrapConfig(world_size=2, rank=1), _params(), NCCLEPConfig()
    )
    assert comm.fleet_params.layout is EpLayout.RANK_MAJOR
    assert comm.fleet_params.algorithm is EpAlgorithm.LOW_LATENCY

    recv_idx = torch.tensor(
        [[0, -1], [1, 0], [5, 5], [-1, 1], [7, 7], [7, 7]], dtype=torch.int32
    )
    log["dispatch_output"] = DispatchOutput(
        expert_tensors=torch.ones(2, 3, 8),
        num_tokens=6,
        recv_topk_idx=recv_idx,
        recv_topk_weights=torch.ones(6, 2),
        expert_counts=torch.tensor([2, 1], dtype=torch.int32),
    )
    weights = torch.full((2, 2), 0.5)
    result = comm.dispatch(torch.ones(2, 8), torch.tensor([[2, 0], [3, 1]]), weights)

    assert result.tokens_per_rank == 3
    assert result.hidden_states.shape == (6, 8)
    assert result.topk_ids.tolist() == [
        [2, -1],
        [3, 2],
        [-1, -1],  # past source rank 0's two tokens
        [-1, 3],
        [-1, -1],  # past source rank 1's single token
        [-1, -1],
    ]
    assert any(getattr(k, "weights", None) is weights for k in log["knobs"])

    with pytest.raises(RuntimeError, match="twice"):
        comm.dispatch(torch.ones(2, 8), torch.zeros(2, 2), weights)

    out = comm.combine(torch.ones(6, 8))
    assert log["combine_x"].shape == (2, 3, 8)
    assert out.shape == (3, 8)
    assert log["complete"] and log["handle"].destroyed
    with pytest.raises(RuntimeError, match="before dispatch"):
        comm.combine(torch.ones(6, 8))

    comm.destroy()
    assert log["fleet_destroyed"]


def test_nccl_ep_communication_rejects_unsupported_inputs(stub_nccl_fleet) -> None:
    comm = create_communication(
        BootstrapConfig(world_size=2, rank=0), _params(), NCCLEPConfig()
    )
    x, ids = torch.ones(2, 8), torch.zeros(2, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match="topk_weights"):
        comm.dispatch(x, ids)
    with pytest.raises(NotImplementedError, match="scale"):
        comm.dispatch(x, ids, torch.ones(2, 2), hidden_states_scale=torch.ones(2, 1))


class _FakeMoeAlltoAll:
    instances: list = []

    def __init__(self, mapping, **kwargs):
        self.kwargs = kwargs
        self.ep_size = mapping.moe_ep_size
        self.eplb_gathered_stats = None
        self.calls: dict = {}
        _FakeMoeAlltoAll.instances.append(self)

    def dispatch(self, token_selected_experts, input_payloads, rt, **kwargs):
        self.calls["dispatch"] = (input_payloads, rt, kwargs)
        return [
            torch.zeros(self.ep_size, rt, *p.shape[1:], dtype=p.dtype)
            for p in input_payloads
        ]

    def get_combine_payload_tensor_in_workspace(self, rt, hidden, dtype):
        buf = torch.zeros(self.ep_size, rt, hidden, dtype=dtype)
        self.calls["buffer"] = buf
        return buf

    def combine(self, payload, rt, **kwargs):
        self.calls["combine"] = (payload, rt, kwargs)
        return torch.zeros(2, payload.shape[-1])


@pytest.fixture
def fake_one_sided(monkeypatch):
    import flashinfer.comm.trtllm_moe_alltoall as a2a
    from flashinfer.comm.mapping import Mapping
    from flashinfer.moe_ep.backends.split.comm.nvlink_one_sided import (
        communication as one_sided,
    )

    _FakeMoeAlltoAll.instances = []
    monkeypatch.setattr(a2a, "MoeAlltoAll", _FakeMoeAlltoAll)
    monkeypatch.setattr(
        a2a, "moe_a2a_get_workspace_size_per_rank", lambda *a, **k: 1 << 20
    )
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
        NVLinkOneSidedAlltoAll,
        "is_platform_supported",
        classmethod(lambda cls: True),
    )


def test_nvlink_one_sided_payload_plumbing(fake_one_sided) -> None:
    comm = create_communication(
        BootstrapConfig(world_size=2, rank=0),
        _params(),
        NVLinkOneSidedConfig(kernel="cake", use_low_precision_combine=True),
    )
    a2a = _FakeMoeAlltoAll.instances[-1]
    assert a2a.kwargs["backend"] == "cake"
    assert a2a.kwargs["max_num_tokens"] == 3

    hidden = torch.ones(2, 8, dtype=torch.bfloat16)
    scale = torch.ones(2, 1, dtype=torch.uint8)
    ids = torch.tensor([[0, 3], [1, 2]])
    weights = torch.ones(2, 2)
    result = comm.dispatch(
        hidden, ids, weights, hidden_states_scale=scale, max_tokens_per_rank=2
    )

    payloads, rt, kwargs = a2a.calls["dispatch"]
    assert rt == 2
    assert [p.dtype for p in payloads] == [
        torch.bfloat16,
        torch.uint8,
        torch.int32,
        torch.float32,
    ]
    assert kwargs["expert_id_payload_index"] == 2
    assert kwargs["invalid_token_expert_id"] == -1
    assert result.tokens_per_rank == 2
    assert result.hidden_states.shape == (4, 8)
    assert result.hidden_states_scale.shape == (4, 1)
    assert result.topk_ids.shape == (4, 2)
    assert result.topk_weights.dtype == torch.float32

    buffer = comm.get_combine_input_buffer(torch.bfloat16)
    assert buffer.shape == (4, 8)
    comm.combine(buffer)
    payload, rt, kwargs = a2a.calls["combine"]
    assert payload.shape == (2, 2, 8)
    assert kwargs["payload_in_workspace"] is True
    assert kwargs["use_low_precision"] is True

    comm.dispatch(hidden, ids, weights)
    comm.combine(torch.zeros(6, 8, dtype=torch.bfloat16))
    assert a2a.calls["combine"][2]["payload_in_workspace"] is False

    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        comm.dispatch(hidden, ids, weights, max_tokens_per_rank=4)


def test_nvlink_two_sided_marks_padding_rows_invalid(monkeypatch) -> None:
    import flashinfer.comm.mnnvl as mnnvl
    import flashinfer.comm.trtllm_alltoall as two_sided_ops
    from flashinfer.comm.mapping import Mapping
    from flashinfer.moe_ep.backends.split.comm.nvlink_two_sided import (
        communication as two_sided,
    )

    calls: dict = {}

    class _FakeMnnvlMoe:
        @staticmethod
        def get_moe_workspaces(mapping, config):
            return "workspace"

        @staticmethod
        def get_moe_prepare_workspace(mapping, config):
            return "prepare_workspace"

        @staticmethod
        def mnnvl_moe_alltoallv_prepare_without_allgather(*args):
            calls["prepare"] = args
            recv_ids = torch.tensor([[1, 3], [4, 4], [4, 4]], dtype=torch.int32)
            return "info", recv_ids, torch.ones(3, 2), None

        @staticmethod
        def mnnvl_moe_alltoallv(x, info, workspace, rank, size):
            return torch.zeros(3, x.shape[1], dtype=x.dtype)

        @staticmethod
        def mnnvl_moe_alltoallv_combine(x, info, workspace, **kwargs):
            calls["combine"] = kwargs
            return torch.full((kwargs["token_count"], x.shape[1]), 2.0)

    monkeypatch.setattr(two_sided_ops, "MnnvlMoe", _FakeMnnvlMoe)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "initialize", staticmethod(lambda: None))
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
    assert result.topk_ids.tolist() == [[1, 3], [-1, -1], [-1, -1]]

    out = torch.empty(1, 8)
    assert comm.combine(torch.ones(3, 8), output=out) is out
    assert calls["combine"]["token_count"] == 1
    torch.testing.assert_close(out, torch.full((1, 8), 2.0))

    with pytest.raises(ValueError, match="divisible by 4"):
        create_communication(
            BootstrapConfig(world_size=1, rank=0),
            _params(num_experts=6),
            "nvlink_two_sided",
        )
