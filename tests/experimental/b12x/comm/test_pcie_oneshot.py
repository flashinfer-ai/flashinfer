from __future__ import annotations

import gc
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie.pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    PCIeOneshotAllReducePool,
    TP2_PLAIN_REMOTE_PUSH_MAX_BYTES,
    _abort_collective_ipc_setup,
    _broadcast_gather_object,
    _compute_crossover_size,
    _finish_collective_runtime_setup,
    _group_ranks,
    _object_broadcast_device,
    _OwnedSharedBuffer,
    _require_no_retained_ipc_setup,
    _require_full_grid_residency,
    _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    _RETAINED_FAILED_IPC_EXPORTS,
    _CuTeOneshotBackend,
    _CuTeOneshotState,
    _eager_storage_shards,
    _enable_device_slot_selection,
    _tp2_plain_remote_push_enabled,
    _transport_policy_contract,
    _uses_sharded_eager_storage,
    parse_pcie_oneshot_max_size,
)
from b12x.comm.pcie._oneshot_cute import _FusedOneshotLaunch, _OneshotLaunch


@pytest.fixture(autouse=True)
def _isolate_pcie_global_registries():
    _RETAINED_FAILED_IPC_EXPORTS.clear()
    _ABANDONED_PCIE_RUNTIME_QUARANTINE.clear()
    yield
    _RETAINED_FAILED_IPC_EXPORTS.clear()
    _ABANDONED_PCIE_RUNTIME_QUARANTINE.clear()


class _FakeExt:
    def __init__(self):
        self.init_calls = []
        self.register_pcie_buffers_calls = []
        self.register_buffer_calls = []
        self.all_reduce_calls = []
        self.all_reduce_fused_add_rms_norm_calls = []
        self.dispose_calls = []
        self.register_graph_buffers_calls = []
        self.handle_bytes = [1, 2, 3]
        self.offsets = [0, 64]

    def init_custom_ar(self, signal_ptrs, rank_data, rank):
        self.init_calls.append((tuple(signal_ptrs), rank_data.device.type, rank))
        return 12345

    def register_pcie_buffers(self, ptr, ptrs0, ptrs1):
        self.register_pcie_buffers_calls.append((ptr, tuple(ptrs0), tuple(ptrs1)))

    def register_buffer(self, ptr, peer_input_ptrs):
        self.register_buffer_calls.append((ptr, tuple(peer_input_ptrs)))

    def all_reduce(self, ptr, inp, out, reg_buffer, reg_buffer_bytes):
        self.all_reduce_calls.append(
            (
                ptr,
                int(inp.data_ptr()),
                int(out.data_ptr()),
                reg_buffer,
                reg_buffer_bytes,
            )
        )
        out.copy_(inp)

    def all_reduce_fused_add_rms_norm(
        self,
        ptr,
        inp,
        residual,
        weight,
        out,
        residual_out,
        epsilon,
        reg_buffer,
        reg_buffer_bytes,
    ):
        self.all_reduce_fused_add_rms_norm_calls.append(
            (
                ptr,
                int(inp.data_ptr()),
                int(residual.data_ptr()),
                int(weight.data_ptr()),
                int(out.data_ptr()),
                int(residual_out.data_ptr()),
                epsilon,
                reg_buffer,
                reg_buffer_bytes,
            )
        )
        residual_value = residual.clone()
        residual_out.copy_(inp)
        residual_out.add_(residual_value)
        variance = residual_out.float().square().mean(dim=-1, keepdim=True)
        normalized = residual_out.float() * torch.rsqrt(variance + epsilon)
        out.copy_((normalized * weight.float()).to(out.dtype))

    def dispose(self, ptr):
        self.dispose_calls.append(ptr)

    def meta_size(self):
        return 256

    def get_graph_buffer_ipc_meta(self, ptr):
        return list(self.handle_bytes), list(self.offsets)

    def register_graph_buffers(self, ptr, handles, offsets):
        self.register_graph_buffers_calls.append((ptr, handles, offsets))


def _make_runtime(
    *,
    rank=0,
    world_size=2,
    exchange_group=None,
    max_size=8 * 1024 * 1024,
    eager=False,
    ext=None,
    stream_affine=True,
):
    ext = ext or _FakeExt()
    kwargs = {}
    if eager:
        kwargs["eager_buffer_ptrs0"] = tuple(range(200, 200 + world_size))
        kwargs["eager_buffer_ptrs1"] = tuple(range(300, 300 + world_size))
        kwargs["eager_buffer_bytes"] = max_size
    return PCIeOneshotAllReduce(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        signal_ptrs=tuple(range(100, 100 + world_size)),
        exchange_group=exchange_group,
        max_size=max_size,
        ext_module=ext,
        stream_affine=stream_affine,
        **kwargs,
    )


def test_parse_pcie_oneshot_max_size_accepts_auto_and_suffixes():
    assert parse_pcie_oneshot_max_size(None) is None
    assert parse_pcie_oneshot_max_size("auto") is None
    assert parse_pcie_oneshot_max_size("64KB") == 64 * 1024
    assert parse_pcie_oneshot_max_size("2m") == 2 * 1024 * 1024
    assert parse_pcie_oneshot_max_size(4096) == 4096


def test_full_grid_residency_accepts_device_with_enough_visible_sms(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=84),
    )

    _require_full_grid_residency(
        owner="test collective",
        required_sms=64,
        device=torch.device("cuda:0"),
        exchange_group=None,
    )


def test_full_grid_residency_rejects_mig_like_capacity_collectively(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=84),
    )
    monkeypatch.setenv("B12X_PCIE_TEST_VISIBLE_SM_COUNT", "32")
    exchanged = []

    def exchange(
        local_error,
        *,
        exchange_group,
        phase,
        exports_retained_on_exchange_failure,
    ):
        assert not exports_retained_on_exchange_failure
        exchanged.append((local_error, exchange_group, phase))
        return ((f"RuntimeError: {local_error}",), ())

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._exchange_setup_failures", exchange
    )

    group = object()
    with pytest.raises(RuntimeError, match="requires at least 64 visible SMs") as exc:
        _require_full_grid_residency(
            owner="test collective",
            required_sms=64,
            device=torch.device("cuda:0"),
            exchange_group=group,
        )

    assert exchanged
    assert exchanged[0][1:] == (group, "test collective resident-grid capability")
    assert "exports were retained" not in str(exc.value)


def test_compute_crossover_size_runs_fine_sweep():
    seen_sizes = []

    def benchmark(size_bytes: int) -> tuple[float, float]:
        seen_sizes.append(size_bytes)
        if size_bytes <= 48 * 1024:
            return 1.0, 2.0
        return 3.0, 2.0

    crossover, results = _compute_crossover_size(
        benchmark,
        ceiling_bytes=64 * 1024,
        fine_step_bytes=8 * 1024,
    )

    assert crossover == 48 * 1024
    assert 40 * 1024 in seen_sizes
    assert 48 * 1024 in seen_sizes
    assert 56 * 1024 in seen_sizes
    assert results[-1].size_bytes == 64 * 1024


def test_graph_slot_bias_preserves_the_next_host_slot() -> None:
    state = _CuTeOneshotState(
        rank=0,
        world_size=2,
        signal_ptrs=(100, 200),
        rank_data=torch.empty(256, dtype=torch.uint8),
        signal_table_address=0,
        next_table_offset=128,
        registered_tables={},
        eager_tables=(300, 400),
        eager_slot=5,
    )

    _enable_device_slot_selection(state, capturing=True)

    assert state.device_slot_selection
    assert state.slot_bias == 1
    state.eager_slot = 8
    _enable_device_slot_selection(state, capturing=True)
    assert state.slot_bias == 1


def test_plain_tp2_remote_push_rejects_unstaged_input() -> None:
    with pytest.raises(ValueError, match="requires a graph-owned staged"):
        _OneshotLaunch(
            "bfloat16",
            world_size=2,
            rank=0,
            stage_input=False,
            device_slot_selection=False,
            slot_bias=0,
            transport="tp2_remote_push",
            threads=128,
        )


def test_plain_tp2_graph_remote_push_accepts_full_launch_geometry() -> None:
    _OneshotLaunch(
        "bfloat16",
        world_size=2,
        rank=0,
        stage_input=True,
        device_slot_selection=True,
        slot_bias=0,
        transport="tp2_remote_push",
        threads=512,
    )


def _make_cute_state(
    world_size: int,
    *,
    eager: bool = True,
    eager_buffer_bytes: int = TP2_PLAIN_REMOTE_PUSH_MAX_BYTES,
) -> _CuTeOneshotState:
    transport_policy = _transport_policy_contract()
    sharded_eager_storage = _uses_sharded_eager_storage(
        world_size,
        transport_policy,
    )
    eager_ptrs = (
        (
            tuple(range(1_000, 1_000 + world_size)),
            tuple(range(2_000, 2_000 + world_size)),
        )
        if eager
        else None
    )
    plain_remote_push_region_packs = (
        (world_size if sharded_eager_storage else 1) * eager_buffer_bytes // 16
        if eager and world_size == 2 and transport_policy[2]
        else 0
    )
    return _CuTeOneshotState(
        rank=0,
        world_size=world_size,
        signal_ptrs=tuple(range(100, 100 + world_size)),
        rank_data=torch.empty(256, dtype=torch.uint8),
        signal_table_address=0,
        next_table_offset=128,
        registered_tables={},
        eager_tables=(300, 400) if eager else None,
        eager_ptrs=eager_ptrs,
        eager_buffer_bytes=eager_buffer_bytes if eager else None,
        transport_policy=transport_policy,
        sharded_eager_storage=sharded_eager_storage,
        plain_remote_push_region_packs=plain_remote_push_region_packs,
    )


@pytest.mark.parametrize(
    ("rows", "expected"),
    (
        (1, ("tp2_remote_push", 64, 36)),
        (2, ("tp2_remote_push_stream", 128, 36)),
        (4, ("tp2_remote_push_stream", 256, 36)),
        (6, ("tp2_remote_push", 64, 16)),
        (8, ("tp2_remote_push", 256, 32)),
        (16, ("tp2_remote_push_stream", 64, 36)),
        (24, ("tp2_remote_push_stream", 128, 32)),
        (32, ("tp2_remote_push", 128, 16)),
        (128, ("tp2_remote_push", 128, 16)),
        (256, ("tp2_remote_push", 128, 16)),
        (512, ("tp2_remote_push", 128, 16)),
    ),
)
def test_plain_tp2_remote_push_uses_qualified_geometry(
    monkeypatch,
    rows: int,
    expected: tuple[str, int, int],
) -> None:
    monkeypatch.delenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "1")
    state = _make_cute_state(2)

    assert (
        _CuTeOneshotBackend._plain_launch_config(
            state,
            torch.empty((rows, 4096), dtype=torch.bfloat16),
            graph_channel=True,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("world_size", "shape", "dtype", "eager"),
    (
        (2, (513, 4096), torch.bfloat16, True),
        (2, (4, 6144), torch.bfloat16, True),
        (2, (4, 4096), torch.float32, True),
        (2, (4, 4096), torch.bfloat16, False),
        (4, (1, 4096), torch.bfloat16, True),
        (8, (1, 4096), torch.bfloat16, True),
        (16, (1, 4096), torch.bfloat16, True),
    ),
)
def test_plain_tp2_remote_push_preserves_generic_fallbacks(
    monkeypatch,
    world_size: int,
    shape: tuple[int, int],
    dtype: torch.dtype,
    eager: bool,
) -> None:
    monkeypatch.delenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "1")
    transport, _, _ = _CuTeOneshotBackend._plain_launch_config(
        _make_cute_state(world_size, eager=eager),
        torch.empty(shape, dtype=dtype),
        graph_channel=True,
    )

    assert transport == "pull"


def test_plain_tp2_transport_policy_is_frozen_at_channel_setup(monkeypatch) -> None:
    monkeypatch.delenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "1")
    state = _make_cute_state(2)
    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "0")

    transport, _, _ = _CuTeOneshotBackend._plain_launch_config(
        state,
        torch.empty((1, 4096), dtype=torch.bfloat16),
        graph_channel=True,
    )

    assert transport == "tp2_remote_push"


def test_plain_graph_plan_freezes_prepared_channel_contract(monkeypatch) -> None:
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "1")
    monkeypatch.setattr(
        _CuTeOneshotBackend,
        "_device_index",
        staticmethod(lambda _device: 0),
    )
    state = _make_cute_state(2)
    inp = torch.empty((6, 4096), dtype=torch.bfloat16)

    plan = _CuTeOneshotBackend._plain_graph_plan(state, inp)
    state.transport_policy = (False, False, False, False, False)
    state.eager_buffer_bytes = 0
    state.plain_remote_push_region_packs = 0

    assert plan.transport == "tp2_remote_push"
    assert (plan.threads, plan.blocks) == (64, 16)
    assert plan.eager_buffer_bytes == TP2_PLAIN_REMOTE_PUSH_MAX_BYTES
    assert plan.transport_policy[2] is True
    assert plan.remote_push_region_packs > 0
    assert plan.remote_push_slot_ptrs == (1_000, 1_001, 2_000, 2_001)


def test_prepare_plain_graph_all_reduce_retains_immutable_plan(monkeypatch) -> None:
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "1")
    monkeypatch.setattr(
        _CuTeOneshotBackend,
        "_device_index",
        staticmethod(lambda _device: 0),
    )
    launchers = []

    def fake_get_oneshot_launcher(*args):
        launcher = _OneshotLaunch(*args[:-1])
        launchers.append(launcher)
        return launcher

    monkeypatch.setattr(
        "b12x.comm.pcie._oneshot_cute.get_oneshot_launcher",
        fake_get_oneshot_launcher,
    )
    backend = _CuTeOneshotBackend()
    state = _make_cute_state(2)
    backend._states[7] = state
    inp = torch.empty((6, 4096), dtype=torch.bfloat16)

    backend._prepare_all_reduce(7, inp)

    key = backend._plain_graph_plan_key(inp)
    plan = state.plain_graph_plans[key]
    assert plan.transport == "tp2_remote_push"
    assert (plan.threads, plan.blocks) == (64, 16)
    assert launchers[0]._transport == "pull"
    assert not launchers[0]._device_slot_selection
    assert [
        (launcher._device_slot_selection, launcher._slot_bias, launcher._transport)
        for launcher in launchers[1:]
    ] == [(True, 0, "tp2_remote_push"), (True, 1, "tp2_remote_push")]


def test_capture_binds_unseen_shape_for_a_prepared_launcher(monkeypatch) -> None:
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "1")
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._is_current_stream_capturing",
        lambda _device: True,
    )
    monkeypatch.setattr(
        _CuTeOneshotBackend,
        "_device_index",
        staticmethod(lambda _device: 0),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        "b12x.comm.pcie._oneshot_cute.is_oneshot_launcher_prepared",
        lambda *args: True,
    )
    launches = []
    monkeypatch.setattr(
        "b12x.comm.pcie._oneshot_cute.get_oneshot_launcher",
        lambda *args: lambda *launch_args: launches.append((args, launch_args)),
    )
    backend = _CuTeOneshotBackend()
    state = _make_cute_state(2)
    state.device_slot_selection = True
    backend._states[7] = state
    inp = torch.empty((5, 4096), dtype=torch.bfloat16)

    backend.all_reduce(7, inp, torch.empty_like(inp), 0, 0)

    key = backend._plain_graph_plan_key(inp)
    assert state.plain_graph_plans[key].transport == "tp2_remote_push"
    assert (
        state.plain_graph_plans[key].threads,
        state.plain_graph_plans[key].blocks,
    ) == (
        64,
        16,
    )
    assert len(launches) == 1


def test_capture_rejects_unseen_shape_for_a_cold_launcher(monkeypatch) -> None:
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "1")
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._is_current_stream_capturing",
        lambda _device: True,
    )
    monkeypatch.setattr(
        _CuTeOneshotBackend,
        "_device_index",
        staticmethod(lambda _device: 0),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie._oneshot_cute.is_oneshot_launcher_prepared",
        lambda *args: False,
    )
    backend = _CuTeOneshotBackend()
    state = _make_cute_state(2)
    state.device_slot_selection = True
    backend._states[7] = state
    inp = torch.empty((5, 4096), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="cold PCIe oneshot CUDA graph capture"):
        backend.all_reduce(7, inp, torch.empty_like(inp), 0, 0)

    assert backend._plain_graph_plan_key(inp) not in state.plain_graph_plans


def test_plain_tp2_remote_push_is_not_selected_for_eager_execution(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "1")
    transport, _, _ = _CuTeOneshotBackend._plain_launch_config(
        _make_cute_state(2),
        torch.empty((6, 4096), dtype=torch.bfloat16),
    )

    assert transport == "pull"


def test_tp8_owner_launcher_rejects_float32() -> None:
    with pytest.raises(ValueError, match="requires float16 or bfloat16"):
        _FusedOneshotLaunch(
            "float32",
            world_size=8,
            rank=0,
            mode="stage_tp8_owner",
            single_cta=True,
            register_normalize=True,
            device_slot_selection=False,
            slot_bias=0,
            threads=256,
        )


@pytest.mark.parametrize(
    ("rows", "hidden", "dtype", "expected"),
    (
        (1, 6144, torch.bfloat16, "stage_pull"),
        (2, 4096, torch.float16, "stage_tp8_owner"),
        (4, 6144, torch.bfloat16, "stage_tp8_owner"),
        (8, 6144, torch.bfloat16, "stage_tp8_owner"),
        (9, 6144, torch.bfloat16, "stage_pull"),
        (4, 8192, torch.bfloat16, "stage_pull"),
        (4, 6144, torch.float32, "stage_pull"),
    ),
)
def test_tp8_owner_reduce_default_has_a_bounded_shape_contract(
    monkeypatch,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    expected: str,
) -> None:
    monkeypatch.delenv("B12X_PCIE_ONESHOT_PUSH", raising=False)
    monkeypatch.delenv("B12X_PCIE_TP8_OWNER_REDUCE", raising=False)
    mode, _, _, _ = _CuTeOneshotBackend._fused_launch_config(
        _make_cute_state(8), torch.empty((rows, hidden), dtype=dtype)
    )
    assert mode == expected


def test_tp8_owner_reduce_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("B12X_PCIE_ONESHOT_PUSH", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP8_OWNER_REDUCE", "0")
    mode, _, _, _ = _CuTeOneshotBackend._fused_launch_config(
        _make_cute_state(8), torch.empty((4, 6144), dtype=torch.bfloat16)
    )
    assert mode == "stage_pull"


@pytest.mark.parametrize(
    (
        "world_size",
        "env_name",
        "shape",
        "expected_by_default",
        "expected_when_enabled",
    ),
    (
        (
            2,
            "B12X_PCIE_TP2_REMOTE_PUSH",
            (4, 4096),
            "stage_pull",
            "stage_remote_push",
        ),
        (
            4,
            "B12X_PCIE_TP4_REMOTE_PUSH",
            (32, 6144),
            "stage_pull",
            "stage_remote_push",
        ),
    ),
)
def test_tp2_tp4_remote_push_defaults_and_overrides(
    monkeypatch,
    world_size: int,
    env_name: str,
    shape: tuple[int, int],
    expected_by_default: str,
    expected_when_enabled: str,
) -> None:
    monkeypatch.delenv("B12X_PCIE_ONESHOT_PUSH", raising=False)
    monkeypatch.delenv(env_name, raising=False)
    inp = torch.empty(shape, dtype=torch.bfloat16)
    mode, _, _, _ = _CuTeOneshotBackend._fused_launch_config(
        _make_cute_state(world_size), inp
    )
    assert mode == expected_by_default

    monkeypatch.setenv(env_name, "0")
    mode, _, _, _ = _CuTeOneshotBackend._fused_launch_config(
        _make_cute_state(world_size), inp
    )
    assert mode == "stage_pull"

    monkeypatch.setenv(env_name, "1")
    mode, _, _, _ = _CuTeOneshotBackend._fused_launch_config(
        _make_cute_state(world_size), inp
    )
    assert mode == expected_when_enabled


def test_topology_policy_is_immutable_after_channel_setup(monkeypatch) -> None:
    monkeypatch.delenv("B12X_PCIE_ONESHOT_PUSH", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP4_REMOTE_PUSH", "1")
    state = _make_cute_state(4)

    monkeypatch.setenv("B12X_PCIE_TP4_REMOTE_PUSH", "0")
    mode, _, _, _ = _CuTeOneshotBackend._fused_launch_config(
        state,
        torch.empty((4, 6144), dtype=torch.bfloat16),
    )

    assert mode == "stage_remote_push"


@pytest.mark.parametrize(
    ("world_size", "env_name", "shape"),
    (
        (2, "B12X_PCIE_TP2_REMOTE_PUSH", (4, 6144)),
        (2, "B12X_PCIE_TP2_REMOTE_PUSH", (33, 4096)),
        (4, "B12X_PCIE_TP4_REMOTE_PUSH", (33, 6144)),
        (4, "B12X_PCIE_TP4_REMOTE_PUSH", (4, 8192)),
    ),
)
def test_tp2_tp4_remote_push_falls_back_outside_qualified_shapes(
    monkeypatch,
    world_size: int,
    env_name: str,
    shape: tuple[int, int],
) -> None:
    monkeypatch.delenv("B12X_PCIE_ONESHOT_PUSH", raising=False)
    monkeypatch.setenv(env_name, "1")
    mode, _, _, _ = _CuTeOneshotBackend._fused_launch_config(
        _make_cute_state(world_size),
        torch.empty(shape, dtype=torch.bfloat16),
    )
    assert mode == "stage_pull"


def test_registered_fused_input_never_selects_topology_transport(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_TP4_REMOTE_PUSH", "1")
    mode, _, _, _ = _CuTeOneshotBackend._fused_launch_config(
        _make_cute_state(4, eager=False),
        torch.empty((4, 6144), dtype=torch.bfloat16),
    )
    assert mode == "registered"


def test_topology_transport_storage_policy_matches_opt_in_defaults(
    monkeypatch,
) -> None:
    for name in (
        "B12X_PCIE_ONESHOT_PUSH",
        "B12X_PCIE_TP2_REMOTE_PUSH",
        "B12X_PCIE_TP2_PLAIN_REMOTE_PUSH",
        "B12X_PCIE_TP4_REMOTE_PUSH",
        "B12X_PCIE_TP8_OWNER_REDUCE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert not _uses_sharded_eager_storage(2)
    assert not _uses_sharded_eager_storage(4)
    assert _uses_sharded_eager_storage(8)
    assert _eager_storage_shards(2) == 3
    assert _eager_storage_shards(4) == 1
    assert _eager_storage_shards(8) == 8
    assert _transport_policy_contract() == (False, False, True, False, True)

    monkeypatch.setenv("B12X_PCIE_TP8_OWNER_REDUCE", "0")
    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "0")
    monkeypatch.setenv("B12X_PCIE_TP4_REMOTE_PUSH", "1")
    assert _uses_sharded_eager_storage(4)
    assert not _uses_sharded_eager_storage(8)
    assert _eager_storage_shards(4) == 4
    assert _transport_policy_contract() == (False, False, False, True, False)

    monkeypatch.setenv("B12X_PCIE_TP4_REMOTE_PUSH", "0")
    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "1")
    assert _eager_storage_shards(2) == 4


def test_plain_tp2_remote_push_override_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "1")
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "0")
    assert not _tp2_plain_remote_push_enabled()

    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "0")
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "1")
    assert _tp2_plain_remote_push_enabled()


def test_register_buffer_is_idempotent_for_same_mapping():
    runtime = _make_runtime()
    ext = runtime._ext

    runtime.register_buffer((111, 222))
    runtime.register_buffer((111, 222))

    assert ext.register_buffer_calls == [(12345, (111, 222))]


def test_register_buffer_rejects_mismatched_mapping_for_same_local_ptr():
    runtime = _make_runtime()

    runtime.register_buffer((111, 222))

    with pytest.raises(ValueError, match="already registered"):
        runtime.register_buffer((111, 333))


def test_all_reduce_registers_explicit_peer_ptrs_once():
    runtime = _make_runtime()
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out0 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))
    out1 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))

    assert torch.equal(out0, inp)
    assert torch.equal(out1, inp)
    assert ext.register_buffer_calls == [(12345, (inp.data_ptr(), 222))]
    assert len(ext.all_reduce_calls) == 2


def test_all_reduce_requires_registration_without_eager_buffers():
    runtime = _make_runtime()
    inp = torch.arange(8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="peer_input_ptrs are required"):
        runtime.all_reduce(inp)


def test_eager_buffers_allow_all_reduce_without_peer_ptrs():
    runtime = _make_runtime(eager=True)
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out = runtime.all_reduce(inp)

    assert torch.equal(out, inp)
    assert ext.register_pcie_buffers_calls == [(12345, (200, 201), (300, 301))]
    assert ext.register_buffer_calls == []
    assert len(ext.all_reduce_calls) == 1


def test_fused_add_rms_norm_returns_norm_and_residual_outputs():
    runtime = _make_runtime(eager=True)
    ext = runtime._ext
    inp = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 8
    residual = torch.linspace(-0.5, 0.5, 16, dtype=torch.bfloat16).reshape(2, 8)
    weight = torch.linspace(0.75, 1.25, 8, dtype=torch.bfloat16)

    out, residual_out = runtime.all_reduce_fused_add_rms_norm(
        inp,
        residual,
        weight,
        1e-6,
    )

    expected_residual = inp + residual
    variance = expected_residual.float().square().mean(dim=-1, keepdim=True)
    expected_out = (
        expected_residual.float() * torch.rsqrt(variance + 1e-6) * weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(out, expected_out)
    assert len(ext.all_reduce_fused_add_rms_norm_calls) == 1


def test_fused_add_rms_norm_supports_inplace_residual_output():
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)
    residual = torch.ones_like(inp)
    original_residual_ptr = residual.data_ptr()

    _, residual_out = runtime.all_reduce_fused_add_rms_norm(
        inp,
        residual,
        torch.ones(8, dtype=torch.bfloat16),
        1e-6,
        residual_out=residual,
    )

    assert residual_out.data_ptr() == original_residual_ptr
    torch.testing.assert_close(residual_out, inp + 1)


def test_fused_add_rms_norm_requires_pack_aligned_rows():
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

    with pytest.raises(ValueError, match="last input dimension"):
        runtime.all_reduce_fused_add_rms_norm(
            inp,
            torch.zeros_like(inp),
            torch.ones(4, dtype=torch.bfloat16),
            1e-6,
        )


def test_fused_add_rms_norm_validates_weight():
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)

    with pytest.raises(ValueError, match="weight tensor"):
        runtime.all_reduce_fused_add_rms_norm(
            inp,
            torch.zeros_like(inp),
            torch.ones(8, dtype=torch.float32),
            1e-6,
        )


def test_world_size_10_is_supported_by_eager_pool():
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(rank=3, world_size=10, eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=3,
        world_size=10,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )

    runtime = pool.for_stream()

    assert runtime.world_size == 10
    assert runtime._ext.init_calls == [
        ((100, 101, 102, 103, 104, 105, 106, 107, 108, 109), "cpu", 3)
    ]
    assert runtime._ext.register_pcie_buffers_calls == [
        (
            12345,
            (200, 201, 202, 203, 204, 205, 206, 207, 208, 209),
            (300, 301, 302, 303, 304, 305, 306, 307, 308, 309),
        )
    ]
    assert created == [(None, runtime)]


def test_runtime_rejects_reuse_from_another_stream_key(monkeypatch):
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16)
    state = {"stream_key": 11}

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: state["stream_key"],
    )

    runtime.all_reduce(inp)
    state["stream_key"] = 22

    with pytest.raises(RuntimeError, match="stream-affine"):
        runtime.all_reduce(inp)


def test_runtime_can_disable_stream_affinity(monkeypatch):
    runtime = _make_runtime(eager=True, stream_affine=False)
    inp = torch.arange(8, dtype=torch.bfloat16)
    state = {"stream_key": 11}

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: state["stream_key"],
    )

    runtime.all_reduce(inp)
    state["stream_key"] = 22
    runtime.all_reduce(inp)


def test_should_allreduce_checks_device_dtype_size_alignment_and_contiguity():
    runtime = _make_runtime(max_size=16)

    good = torch.arange(8, dtype=torch.bfloat16)
    assert runtime.should_allreduce(good) is True
    assert runtime.should_allreduce(torch.arange(4, dtype=torch.int32)) is False
    assert runtime.should_allreduce(torch.arange(16, dtype=torch.bfloat16)) is False
    assert runtime.should_allreduce(torch.arange(7, dtype=torch.bfloat16)) is False
    assert (
        runtime.should_allreduce(torch.arange(16, dtype=torch.bfloat16)[::2]) is False
    )


def test_plain_allreduce_route_rejects_overlapping_storage_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", raising=False)
    monkeypatch.delenv("B12X_PCIE_TP2_REMOTE_PUSH", raising=False)
    runtime = _make_runtime(world_size=2, eager=True)
    transposed = torch.empty((4096, 4), dtype=torch.bfloat16).transpose(0, 1)
    overlapping = torch.empty((4, 4096), dtype=torch.bfloat16).as_strided(
        (4, 4096), (0, 1)
    )

    assert runtime.should_allreduce(transposed)
    assert runtime.should_route_plain_allreduce(
        transposed,
        generic_max_bytes=0,
        graph_capture=True,
    )
    assert not runtime.should_allreduce(overlapping)
    assert not runtime.should_route_plain_allreduce(
        overlapping,
        generic_max_bytes=0,
        graph_capture=True,
    )


@pytest.mark.parametrize(
    ("world_size", "shape", "generic_max_bytes", "graph_capture", "expected"),
    (
        (2, (64, 4096), 84 * 1024, True, True),
        (2, (65, 4096), 84 * 1024, True, False),
        (2, (512, 4096), 84 * 1024, True, False),
        (2, (1, 4096), 84 * 1024, False, True),
        (2, (16, 4096), 84 * 1024, False, False),
        (2, (1, 6144), 84 * 1024, True, True),
        (2, (8, 6144), 84 * 1024, True, False),
        (4, (8, 4096), 84 * 1024, True, True),
        (4, (32, 4096), 84 * 1024, True, False),
        (8, (40, 4096), 84 * 1024, True, False),
    ),
)
def test_plain_allreduce_route_uses_qualified_tp_bands(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    shape: tuple[int, int],
    generic_max_bytes: int,
    graph_capture: bool,
    expected: bool,
) -> None:
    monkeypatch.delenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", raising=False)
    monkeypatch.delenv("B12X_PCIE_TP2_REMOTE_PUSH", raising=False)
    runtime = _make_runtime(world_size=world_size, eager=True)

    assert (
        runtime.should_route_plain_allreduce(
            torch.empty(shape, dtype=torch.bfloat16),
            generic_max_bytes=generic_max_bytes,
            graph_capture=graph_capture,
        )
        is expected
    )


def test_plain_allreduce_route_honors_tp2_remote_push_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP2_REMOTE_PUSH", "0")
    runtime = _make_runtime(world_size=2, eager=True)
    inp = torch.empty((32, 4096), dtype=torch.bfloat16)

    assert not runtime.should_route_plain_allreduce(
        inp,
        generic_max_bytes=84 * 1024,
        graph_capture=True,
    )


def test_plain_allreduce_route_rejects_negative_generic_band() -> None:
    runtime = _make_runtime(world_size=2, eager=True)

    with pytest.raises(ValueError, match="must be non-negative"):
        runtime.should_route_plain_allreduce(
            torch.empty((1, 4096), dtype=torch.bfloat16),
            generic_max_bytes=-1,
        )


def test_eager_runtime_rejects_threshold_above_buffer_capacity():
    with pytest.raises(ValueError, match="exceeds eager buffer capacity"):
        PCIeOneshotAllReduce(
            rank=0,
            world_size=2,
            device=torch.device("cpu"),
            signal_ptrs=(100, 200),
            eager_buffer_ptrs0=(300, 400),
            eager_buffer_ptrs1=(500, 600),
            eager_buffer_bytes=16,
            max_size=32,
            ext_module=_FakeExt(),
        )


def test_cuda_direct_runtime_requires_collective_factory():
    with pytest.raises(ValueError, match="exchange_group is required"):
        PCIeOneshotAllReduce(
            rank=0,
            world_size=2,
            device=torch.device("cuda:0"),
            signal_ptrs=(100, 200),
            ext_module=_FakeExt(),
        )


@pytest.mark.parametrize(
    "runtime_class", (PCIeOneshotAllReduce, PCIeOneshotAllReducePool)
)
def test_process_group_max_input_sets_threshold_and_capacity(
    monkeypatch, runtime_class
):
    captured = {}

    def fake_from_exchange_group(cls, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        runtime_class, "from_exchange_group", classmethod(fake_from_exchange_group)
    )

    runtime_class.from_process_group(
        process_group=object(),
        device=torch.device("cuda:0"),
        max_input_bytes=1 << 20,
    )

    assert captured["eager_buffer_bytes"] == 1 << 20
    assert captured["max_size"] == 1 << 20


def test_autotune_never_benchmarks_past_eager_capacity(monkeypatch):
    runtime = _make_runtime(eager=True, max_size=4096)
    runtime.device = torch.device("cuda:0")
    observed = {}

    def fake_compute(benchmark, *, ceiling_bytes, fine_step_bytes):
        observed["ceiling_bytes"] = ceiling_bytes
        return ceiling_bytes, []

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._compute_crossover_size", fake_compute
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda *, device: object())

    assert runtime.find_crossover_size(object(), ceiling_bytes=1 << 20) == 4096
    assert observed["ceiling_bytes"] == 4096


def test_graph_buffer_api_exposes_explicit_registration_hooks():
    runtime = _make_runtime()
    ext = runtime._ext

    assert runtime.get_graph_buffer_ipc_meta() == ([1, 2, 3], [0, 64])

    runtime.register_graph_buffers_from_ranks(
        ([1, 2, 3], [4, 5, 6]),
        ([0, 64], [8, 72]),
    )

    assert ext.register_graph_buffers_calls == [
        (12345, [[1, 2, 3], [4, 5, 6]], [[0, 64], [8, 72]])
    ]


def test_allocate_shared_buffer_cleans_up_on_failed_peer_open(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.freed = []

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            pass

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            if handle == b"remote1":
                raise RuntimeError("open failed")
            return 2000

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 3)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_object, group: (
            [local_object, (), ()]
            if isinstance(local_object, tuple)
            else [local_object, b"remote0", b"remote1"]
        ),
    )

    with pytest.raises(RuntimeError, match="peer group rank 2"):
        PCIeOneshotAllReduce._allocate_shared_buffer(
            object(), 256, zero_fill=True, ipc=ipc
        )

    assert ipc.closed == [2000]
    assert ipc.freed == [1000]


def test_failed_setup_export_free_is_retained_and_retryable(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.free_attempts = 0

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            raise RuntimeError("injected setup failure")

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaFree(self, ptr):
            self.free_attempts += 1
            if self.free_attempts == 1:
                raise RuntimeError("transient free failure")

    ipc = FakeIPC()
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_object, group: [local_object, ()],
    )

    with pytest.raises(RuntimeError, match="exports were retained") as exc:
        PCIeOneshotAllReduce._allocate_shared_buffer(
            object(), 256, zero_fill=True, ipc=ipc
        )

    retained = exc.value.retryable_export
    assert _RETAINED_FAILED_IPC_EXPORTS[retained.key] is retained
    retained.retry()
    assert ipc.free_attempts == 2
    assert retained.local_ptr == 0
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_failed_peer_setup_export_free_is_retained_and_retryable(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.free_attempts = 0

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            pass

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            return 2000

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.free_attempts += 1
            if self.free_attempts == 1:
                raise RuntimeError("transient free failure")

    ipc = FakeIPC()
    status_round = 0

    def exchange(local_object, group):
        nonlocal status_round
        if not isinstance(local_object, tuple):
            return [local_object, b"remote"]
        status_round += 1
        if status_round == 1:  # retained-setup gate
            return [local_object, ()]
        if status_round == 2:  # export preparation
            return [local_object, ()]
        if status_round == 3:  # import-open verdict
            return [local_object, ("peer open failed",)]
        if status_round == 4:  # rollback-unmap verdict
            return [local_object, ()]
        if status_round == 5:  # rollback-export-free verdict
            return [local_object, ()]
        if status_round == 6:  # retry export-free verdict
            return [local_object, ()]
        raise AssertionError(f"unexpected status round {status_round}")

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )

    with pytest.raises(RuntimeError, match="exports were retained") as exc:
        PCIeOneshotAllReduce._allocate_shared_buffer(
            object(), 256, zero_fill=True, ipc=ipc
        )

    retained = exc.value.retryable_export
    assert ipc.closed == [2000]
    assert _RETAINED_FAILED_IPC_EXPORTS[retained.key] is retained
    retained.retry()
    assert ipc.free_attempts == 2
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_peer_setup_and_unmap_failure_retains_export_and_closes_each_import_once(
    monkeypatch,
):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.freed = []

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            pass

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            return {b"remote0": 2000, b"remote1": 3000}[handle]

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()
    status_round = 0

    def exchange(local_object, group):
        nonlocal status_round
        if not isinstance(local_object, tuple):
            return [local_object, b"remote0", b"remote1"]
        status_round += 1
        if status_round == 1:  # retained-setup gate
            return [local_object, (), ()]
        if status_round == 2:  # export preparation
            return [local_object, (), ()]
        if status_round == 3:  # import-open verdict
            return [local_object, ("peer open failed",), ()]
        if status_round == 4:  # rollback-unmap verdict
            return [local_object, ("peer unmap failed",), ()]
        if status_round in (5, 6):  # retry unmap / retry export free
            return [local_object, (), ()]
        raise AssertionError(f"unexpected status round {status_round}")

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 3)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )

    with pytest.raises(RuntimeError, match=r"retained.*peer unmap failed") as exc:
        PCIeOneshotAllReduce._allocate_shared_buffer(
            object(), 256, zero_fill=True, ipc=ipc
        )

    assert ipc.closed == [2000, 3000]
    assert ipc.freed == []
    exc.value.retryable_setup.retry()
    assert ipc.closed == [2000, 3000]
    assert ipc.freed == [1000]
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_handle_exchange_failure_retains_export_until_collective_retry(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.freed = []

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            pass

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()
    status_round = 0
    fail_handle_exchange = True

    def exchange(local_object, group):
        nonlocal status_round, fail_handle_exchange
        if not isinstance(local_object, tuple):
            if fail_handle_exchange:
                fail_handle_exchange = False
                raise RuntimeError("injected handle exchange failure")
            return [local_object, b"remote"]
        status_round += 1
        return [local_object, ()]

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )

    with pytest.raises(RuntimeError, match="handle exchange failed") as exc:
        PCIeOneshotAllReduce._allocate_shared_buffer(
            object(), 256, zero_fill=True, ipc=ipc
        )

    retained = exc.value.retryable_setup
    assert retained.local_ptr == 1000
    assert ipc.freed == []
    retained.retry()
    assert ipc.freed == [1000]
    assert status_round == 4  # gate, export verdict, retry unmap, retry free
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_import_status_exchange_failure_retains_imports_for_retry(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.freed = []

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            pass

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            return 2000

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()
    status_round = 0

    def exchange(local_object, group):
        nonlocal status_round
        if not isinstance(local_object, tuple):
            return [local_object, b"remote"]
        status_round += 1
        if status_round == 3:
            raise RuntimeError("injected import verdict exchange failure")
        return [local_object, ()]

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )

    with pytest.raises(RuntimeError, match="import-open status") as exc:
        PCIeOneshotAllReduce._allocate_shared_buffer(
            object(), 256, zero_fill=True, ipc=ipc
        )

    retained = exc.value.retryable_setup
    assert retained.remote_ptrs == [2000]
    assert ipc.closed == []
    retained.retry()
    assert ipc.closed == [2000]
    assert ipc.freed == [1000]
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_failed_native_cleanup_is_owned_and_retried_before_export_free(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.freed = []

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()
    cleanup_calls = 0

    def cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise RuntimeError("transient native dispose failure")

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    with pytest.raises(RuntimeError, match="rollback native cleanup") as exc:
        _abort_collective_ipc_setup(
            owner="test runtime",
            setup_phase="native initialization",
            setup_statuses=(("peer init failed",), ()),
            exchange_group=object(),
            ipc=ipc,
            local_ptr=1000,
            remote_ptrs=(2000,),
            local_error=RuntimeError("peer init failed"),
            local_cleanup=cleanup,
        )

    retained = exc.value.retryable_setup
    assert cleanup_calls == 1
    assert retained.state == "native"
    assert ipc.closed == []
    assert ipc.freed == []
    retained.retry()
    assert cleanup_calls == 2
    assert ipc.closed == [2000]
    assert ipc.freed == [1000]
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_peer_native_cleanup_failure_blocks_every_rank_from_unmapping(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.freed = []

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()
    cleanup_calls = 0
    status_round = 0

    def cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1

    def exchange(local_status, group):
        nonlocal status_round
        status_round += 1
        if status_round == 1:
            return [local_status, ("peer native dispose failed",)]
        return [local_status, ()]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )

    with pytest.raises(RuntimeError, match="rollback native cleanup") as exc:
        _abort_collective_ipc_setup(
            owner="test runtime",
            setup_phase="native initialization",
            setup_statuses=(("peer init failed",), ()),
            exchange_group=object(),
            ipc=ipc,
            local_ptr=1000,
            remote_ptrs=(2000,),
            local_error=RuntimeError("peer init failed"),
            local_cleanup=cleanup,
        )

    retained = exc.value.retryable_setup
    assert retained.state == "native"
    assert retained.local_cleanup is None
    assert cleanup_calls == 1
    assert ipc.closed == []
    assert ipc.freed == []

    retained.retry()
    assert cleanup_calls == 1
    assert ipc.closed == [2000]
    assert ipc.freed == [1000]
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_native_cleanup_verdict_exchange_failure_retains_imports(monkeypatch):
    events = []

    class FakeIPC:
        def cudaIpcCloseMemHandle(self, ptr):
            events.append(("close", ptr))

        def cudaFree(self, ptr):
            events.append(("free", ptr))

    ipc = FakeIPC()
    status_round = 0

    def exchange(local_status, group):
        nonlocal status_round
        status_round += 1
        if status_round == 1:
            raise RuntimeError("injected native cleanup verdict exchange failure")
        return [local_status, ()]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )

    with pytest.raises(RuntimeError, match="cleanup status exchange") as exc:
        _abort_collective_ipc_setup(
            owner="test runtime",
            setup_phase="native initialization",
            setup_statuses=(("peer init failed",), ()),
            exchange_group=object(),
            ipc=ipc,
            local_ptr=1000,
            remote_ptrs=(2000,),
            local_error=RuntimeError("peer init failed"),
            local_cleanup=lambda: events.append(("dispose", 3000)),
        )

    retained = exc.value.retryable_setup
    assert retained.state == "native"
    assert retained.local_cleanup is None
    assert events == [("dispose", 3000)]

    retained.retry()
    assert events == [("dispose", 3000), ("close", 2000), ("free", 1000)]
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_initial_native_verdict_exchange_retains_complete_setup_until_retry(
    monkeypatch,
):
    events = []

    class FakeIPC:
        def cudaIpcCloseMemHandle(self, ptr):
            events.append(("close", ptr))

        def cudaFree(self, ptr):
            events.append(("free", ptr))

    ipc = FakeIPC()
    group = object()
    status_round = 0

    def exchange(local_status, exchange_group):
        nonlocal status_round
        assert exchange_group is group
        status_round += 1
        if status_round == 1:
            raise RuntimeError("injected first native verdict exchange failure")
        return [local_status, ()]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )

    with pytest.raises(RuntimeError, match="must be coordinated by every rank") as exc:
        _finish_collective_runtime_setup(
            owner="PCIe twoshot proof",
            exchange_group=group,
            ipc=ipc,
            shared=_OwnedSharedBuffer(
                local_ptr=1000,
                peer_ptrs=(1000, 2000),
                remote_ptrs=(2000,),
            ),
            local_error=None,
            local_cleanup=lambda: events.append(("dispose", 3000)),
        )

    retained = exc.value.retryable_setup
    assert retained.local_ptr == 1000
    assert retained.remote_ptrs == [2000]
    assert retained.local_cleanup is not None
    assert events == []
    assert _RETAINED_FAILED_IPC_EXPORTS[retained.key] is retained

    # The retained generation gate rejects another setup on this process group
    # before CUDA allocation.  A caller may only retry after arranging the same
    # retry call on every rank that participated in the failed exchange.
    with pytest.raises(RuntimeError, match="retained CUDA IPC setup gate"):
        _require_no_retained_ipc_setup(group)
    assert events == []

    retained.retry()
    assert events == [("dispose", 3000), ("close", 2000), ("free", 1000)]
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_free_status_exchange_failure_keeps_coordination_ticket_without_double_free(
    monkeypatch,
):
    class FakeIPC:
        def __init__(self):
            self.freed = []

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()
    status_round = 0

    def exchange(local_status, group):
        nonlocal status_round
        status_round += 1
        if status_round == 2:
            raise RuntimeError("injected free verdict exchange failure")
        return [local_status, ()]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )

    with pytest.raises(RuntimeError, match="free status exchange") as exc:
        _abort_collective_ipc_setup(
            owner="test runtime",
            setup_phase="native initialization",
            setup_statuses=(("peer init failed",), ()),
            exchange_group=object(),
            ipc=ipc,
            local_ptr=1000,
            remote_ptrs=(),
            local_error=RuntimeError("peer init failed"),
        )

    retained = exc.value.retryable_setup
    assert retained.local_ptr == 0
    assert ipc.freed == [1000]
    retained.retry()
    assert ipc.freed == [1000]
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_retained_generation_blocks_second_setup_before_cuda_allocation(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.malloc_calls = 0

        def cudaMalloc(self, size):
            self.malloc_calls += 1
            raise RuntimeError("injected no-local-export allocation failure")

    ipc = FakeIPC()
    group = object()
    status_round = 0

    def exchange(local_status, exchange_group):
        nonlocal status_round
        status_round += 1
        if status_round == 2:
            raise RuntimeError("injected preparation verdict exchange failure")
        return [local_status, ()]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", exchange
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)

    with pytest.raises(RuntimeError, match="preparation status") as exc:
        PCIeOneshotAllReduce._allocate_shared_buffer(
            group, 256, zero_fill=True, ipc=ipc
        )
    retained = exc.value.retryable_setup
    first_key = retained.key
    assert retained.local_ptr == 0
    assert ipc.malloc_calls == 1

    with pytest.raises(RuntimeError, match="retained CUDA IPC setup gate"):
        PCIeOneshotAllReduce._allocate_shared_buffer(
            group, 256, zero_fill=True, ipc=ipc
        )
    assert ipc.malloc_calls == 1
    assert tuple(_RETAINED_FAILED_IPC_EXPORTS) == (first_key,)

    retained.retry()
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def test_eager_channel_buffers_use_single_ipc_slab(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.malloc_sizes = []
            self.memsets = []
            self.opened = []

        def cudaMalloc(self, size):
            self.malloc_sizes.append(size)
            return 1000

        def cudaMemset(self, ptr, value, size):
            self.memsets.append((ptr, value, size))

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            ptr = {b"remote0": 2000, b"remote1": 3000}[handle]
            self.opened.append(ptr)
            return ptr

        def cudaIpcCloseMemHandle(self, ptr):
            raise AssertionError("success path should not close remote ptrs")

        def cudaFree(self, ptr):
            raise AssertionError("success path should not free local ptr")

    ipc = FakeIPC()
    exchange_group = object()
    signal_bytes = 300
    eager_bytes = 128
    eager0_offset = IPC_SLAB_ALIGNMENT * 2
    eager1_offset = eager0_offset + IPC_SLAB_ALIGNMENT

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 3)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_object, group: (
            [local_object, (), ()]
            if isinstance(local_object, tuple)
            else [local_object, b"remote0", b"remote1"]
        ),
    )

    buffers = PCIeOneshotAllReduce._allocate_eager_channel_buffers(
        exchange_group,
        signal_bytes=signal_bytes,
        eager_buffer_bytes=eager_bytes,
        eager_storage_shards=1,
        ipc=ipc,
    )

    assert ipc.malloc_sizes == [eager1_offset + eager_bytes]
    assert ipc.memsets == [(1000, 0, eager1_offset + eager_bytes)]
    assert ipc.opened == [2000, 3000]
    assert buffers.owned_buffer.local_ptr == 1000
    assert buffers.owned_buffer.remote_ptrs == (2000, 3000)
    assert buffers.signal_ptrs == (1000, 2000, 3000)
    assert buffers.eager0_ptrs == (
        1000 + eager0_offset,
        2000 + eager0_offset,
        3000 + eager0_offset,
    )
    assert buffers.eager1_ptrs == (
        1000 + eager1_offset,
        2000 + eager1_offset,
        3000 + eager1_offset,
    )


def test_shared_buffer_maps_only_selected_peers(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.opened = []

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            pass

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            ptr = {
                b"remote-1": 2001,
                b"remote-2": 2002,
                b"remote-3": 2003,
            }[handle]
            self.opened.append(ptr)
            return ptr

    ipc = FakeIPC()
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 4)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_object, group: (
            [local_object, (), (), ()]
            if isinstance(local_object, tuple)
            else [local_object, b"remote-1", b"remote-2", b"remote-3"]
        ),
    )

    shared = PCIeOneshotAllReduce._allocate_shared_buffer(
        object(),
        256,
        zero_fill=True,
        ipc=ipc,
        peer_ranks=(1, 3),
    )

    assert ipc.opened == [2001, 2003]
    assert shared.local_ptr == 1000
    assert shared.peer_ptrs == (1000, 2001, 0, 2003)
    assert shared.remote_ptrs == (2001, 2003)


def test_eager_channel_buffers_cleanup_when_slab_zero_fails(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.freed = []

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            raise RuntimeError("memset failed")

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            return {b"remote0": 2000, b"remote1": 3000}[handle]

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 3)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_object, group: (
            [local_object, (), ()]
            if isinstance(local_object, tuple)
            else [local_object, b"remote0", b"remote1"]
        ),
    )

    with pytest.raises(RuntimeError, match="memset failed"):
        PCIeOneshotAllReduce._allocate_eager_channel_buffers(
            object(),
            signal_bytes=300,
            eager_buffer_bytes=128,
            eager_storage_shards=1,
            ipc=ipc,
        )

    assert ipc.closed == []
    assert ipc.freed == [1000]


def test_register_graph_buffers_uses_exchange_group_broadcast(monkeypatch):
    remote_meta = {
        0: ([1, 2, 3], [0, 64]),
        1: ([9, 8, 7], [16, 80]),
    }

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks", lambda group=None: [0, 1]
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._object_broadcast_device",
        lambda group: "cpu",
    )

    def fake_broadcast(object_list, src, group=None, device=None):
        object_list[0] = remote_meta[src]

    monkeypatch.setattr("torch.distributed.broadcast_object_list", fake_broadcast)

    runtime = _make_runtime()
    runtime.exchange_group = object()
    ext = runtime._ext
    runtime.register_graph_buffers()

    assert ext.register_graph_buffers_calls == [
        (12345, [[1, 2, 3], [9, 8, 7]], [[0, 64], [16, 80]])
    ]


def test_object_broadcast_uses_cpu_for_gloo(monkeypatch):
    group = object()
    monkeypatch.setattr("torch.distributed.get_backend", lambda group=None: "gloo")

    assert _object_broadcast_device(group) == torch.device("cpu")


def test_nonmonotonic_process_group_order_is_preserved_for_handle_slots(
    monkeypatch,
):
    group = object()
    sources: list[int] = []
    nonmonotonic_ranks = [7, 2, 9]
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 3)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 1)
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda group=None: list(nonmonotonic_ranks),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._object_broadcast_device",
        lambda group: "cpu",
    )

    def fake_broadcast(object_list, src, group=None, device=None):
        sources.append(src)
        object_list[0] = f"from-{src}"

    monkeypatch.setattr("torch.distributed.broadcast_object_list", fake_broadcast)

    assert _group_ranks(group) == nonmonotonic_ranks
    assert _broadcast_gather_object("local", group) == [
        "from-7",
        "from-2",
        "from-9",
    ]
    assert sources == nonmonotonic_ranks


def test_register_graph_buffers_noops_when_no_rank_registered_buffers(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks", lambda group=None: [0, 1]
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._object_broadcast_device",
        lambda group: "cpu",
    )
    monkeypatch.setattr(
        "torch.distributed.broadcast_object_list",
        lambda object_list, src, group=None, device=None: object_list.__setitem__(
            0, ([], [])
        ),
    )

    runtime = _make_runtime()
    runtime.exchange_group = object()
    ext = runtime._ext
    ext.handle_bytes = []
    ext.offsets = []

    runtime.register_graph_buffers()

    assert ext.register_graph_buffers_calls == []


def test_capture_registers_graph_buffers_after_context(monkeypatch):
    runtime = _make_runtime()
    runtime.exchange_group = object()
    calls = []

    monkeypatch.setattr(
        runtime, "register_graph_buffers", lambda: calls.append("registered")
    )

    with runtime.capture():
        pass

    assert calls == ["registered"]


def test_eager_capture_skips_graph_buffer_registration(monkeypatch):
    runtime = _make_runtime(eager=True)
    runtime.exchange_group = object()
    calls = []

    monkeypatch.setattr(
        runtime, "register_graph_buffers", lambda: calls.append("registered")
    )

    with runtime.capture():
        pass

    assert calls == []


def test_pool_creates_distinct_channels_per_stream_key(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 7 if stream is None else int(stream),
    )

    ch7 = pool.for_stream()
    ch8 = pool.for_stream(8)

    assert pool.for_stream() is ch7
    assert pool.for_stream(8) is ch8
    assert ch7 is not ch8
    assert [entry[0] for entry in created] == [7, 8]


def test_collective_logical_channel_preparation_is_canonical(monkeypatch):
    created = []
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: created.append(stream_key) or object(),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )

    pool.prepare_channels(("draft", "target"))
    pool.prepare_channels(("target", "draft"))

    assert list(pool._logical_channels) == ["draft", "target"]
    assert created == [None, None]


def test_collective_logical_channel_mismatch_rejects_before_allocation(monkeypatch):
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if not local_state or isinstance(local_state[0], str):
            return [local_state, ()]
        _requested, existing = local_state
        return [local_state, (("other",), existing)]

    monkeypatch.setattr("b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather)

    with pytest.raises(RuntimeError, match="differs across ranks"):
        pool.prepare_channels(("target",))


def test_invalid_logical_channel_id_is_rejected_collectively_before_allocation(
    monkeypatch,
):
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    with pytest.raises(RuntimeError, match="must not be empty"):
        pool.prepare_channels(("  ",))


def test_empty_logical_channel_set_still_enters_collective_contract(monkeypatch):
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()

    def gather(local_state, group):
        if not local_state or isinstance(local_state[0], str):
            return [local_state, ()]
        _, existing = local_state
        return [local_state, (("peer-channel",), existing)]

    monkeypatch.setattr("b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather)

    with pytest.raises(RuntimeError, match="differs across ranks"):
        pool.prepare_channels(())


def test_collective_capture_requires_stable_semantic_id(monkeypatch):
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()

    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    with (
        pytest.raises(RuntimeError, match="stable semantic channel_id"),
        pool.capture(7),
    ):
        pass


def test_collective_capture_allows_opposite_order_from_agreed_catalog(monkeypatch):
    target = _make_runtime(eager=True)
    draft = _make_runtime(eager=True)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels.update({"graph:draft": draft, "graph:target": target})
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: int(stream),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        requested, catalog = local_state
        assert requested == "graph:target"
        return [local_state, ("graph:draft", catalog)]

    monkeypatch.setattr("b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather)

    with pool.capture(7, channel_id="graph:target") as channel:
        assert channel is target

    assert pool._captured_channel_ids == {"graph:target"}


def test_collective_capture_rejects_divergent_catalog_before_allocation(monkeypatch):
    target = _make_runtime(eager=True)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["graph:target"] = target
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        return [local_state, ("graph:target", ())]

    monkeypatch.setattr("b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather)

    with (
        pytest.raises(RuntimeError, match="prepared channel catalog differs"),
        pool.capture(7, channel_id="graph:target"),
    ):
        pass


def test_collective_capture_rejects_differing_unprepared_ids(monkeypatch):
    target = _make_runtime(eager=True)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["graph:target"] = target
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        _, catalog = local_state
        return [local_state, ("graph:unknown", catalog)]

    monkeypatch.setattr("b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather)

    with (
        pytest.raises(RuntimeError, match="unprepared logical channels"),
        pool.capture(7, channel_id="graph:target"),
    ):
        pass


def test_collective_capture_preserves_same_id_convenience_allocation(monkeypatch):
    created = []
    channel = _make_runtime(eager=True)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: created.append(stream_key) or channel,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: int(stream),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )

    with pool.capture(7, channel_id="graph:target") as captured:
        assert captured is channel

    assert created == [None]
    assert pool._logical_channels == {"graph:target": channel}


def test_collective_capture_routes_eager_warmup_to_graph_channel(monkeypatch):
    eager = _make_runtime(eager=True)
    target = _make_runtime(eager=True)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels.update({"eager:model": eager, "graph:target": target})
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 7 if stream is None else int(stream),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )

    with pool.capture(7, channel_id="graph:target") as captured:
        assert captured is target
        assert pool.for_stream(7, channel_id="eager:model") is target
        with pytest.raises(RuntimeError, match="stream-affine"):
            pool.for_stream(8, channel_id="eager:model")

    assert pool.for_stream(7, channel_id="eager:model") is eager


def test_collective_eager_channel_requires_id_and_rejects_duplicate_stream_owner(
    monkeypatch,
):
    eager = _make_runtime(eager=True)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: eager,
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["eager:model"] = eager
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: int(stream),
    )

    with pytest.raises(RuntimeError, match="explicit semantic channel_id"):
        pool.for_stream(7)

    assert pool.for_stream(7, channel_id="eager:model") is eager
    with pytest.raises(RuntimeError, match="stream-affine"):
        pool.for_stream(8, channel_id="eager:model")


def test_pool_reuses_single_channel_across_stream_keys(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        single_channel=True,
        channel_factory=make_channel,
    )

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 7 if stream is None else int(stream),
    )

    ch7 = pool.for_stream()
    ch8 = pool.for_stream(8)

    assert ch7 is ch8
    assert [entry[0] for entry in created] == [None]


def test_pool_requires_precreated_channel_during_capture(monkeypatch):
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 7,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._is_current_stream_capturing",
        lambda device: True,
    )

    with pytest.raises(RuntimeError, match=r"pool\.capture\(\)"):
        pool.for_stream()

    pool._channels[7] = _make_runtime(eager=True)

    with pytest.raises(RuntimeError, match=r"pool\.capture\(\)"):
        pool.for_stream()


def test_nested_capture_reuses_its_outer_channel(monkeypatch):
    created = []
    current_stream = [7]
    capturing = [False]

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    with pool.capture(7) as target_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is target_channel
        capturing[0] = False

    with pool.capture(8) as draft_channel:
        capturing[0] = True
        current_stream[0] = 80
        assert pool.for_stream() is draft_channel
        capturing[0] = False

    assert target_channel is not draft_channel
    assert pool._channels == {}
    assert target_channel in pool._all_channels
    assert draft_channel in pool._all_channels
    assert [entry[0] for entry in created] == [7, 8]


def test_pool_restores_nested_capture_mappings_in_lifo_order(monkeypatch):
    current_stream = [7]
    capturing = [False]
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    eager_channel = pool.for_stream()
    with pool.capture(7) as outer_channel:
        assert pool._channels == {7: outer_channel}

        current_stream[0] = 8
        with pool.capture() as inner_channel:
            capturing[0] = True
            current_stream[0] = 80
            assert pool.for_stream() is inner_channel
            assert pool._channels == {
                7: outer_channel,
                8: inner_channel,
                80: inner_channel,
            }
            capturing[0] = False

        assert pool._channels == {7: outer_channel}
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is outer_channel
        capturing[0] = False

    assert eager_channel is not outer_channel
    assert outer_channel is not inner_channel
    assert pool._channels == {7: eager_channel}
    assert pool._capture_channel_stack == []


def test_reused_capture_stream_keys_get_distinct_channels(monkeypatch):
    created = []
    current_stream = [7]
    capturing = [False]

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    with pool.capture(7) as target_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is target_channel
        capturing[0] = False

    # CUDA may recycle both handles for the next graph manager. Neither stale
    # mapping may make the draft graph retain the target graph's IPC channel.
    with pool.capture(7) as draft_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is draft_channel
        capturing[0] = False

    assert target_channel is not draft_channel
    assert pool._channels == {}
    assert target_channel in pool._all_channels
    assert draft_channel in pool._all_channels
    assert [entry[0] for entry in created] == [7, 7]

    pool.close()
    assert target_channel._ext.dispose_calls == [12345]
    assert draft_channel._ext.dispose_calls == [12345]


def test_pool_rolls_back_throwaway_capture_channels(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 3 if stream is None else int(stream),
    )

    eager_channel = pool.for_stream()
    checkpoint = pool.checkpoint_channels()
    with pool.capture(7) as profile_channel:
        pass

    pool.rollback_channels(checkpoint)

    assert pool._all_channels == [eager_channel]
    assert pool._channels == {3: eager_channel}
    assert profile_channel._ext.dispose_calls == [12345]
    assert eager_channel._ext.dispose_calls == []


def test_pool_coordinates_ipc_teardown_across_ranks(monkeypatch):
    events = []

    class FakeChannel:
        def _close_ipc_imports(self):
            events.append("close-imports")

        def _free_ipc_exports(self):
            events.append("free-exports")

    group = object()
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        exchange_group=group,
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    retained = FakeChannel()
    transient = FakeChannel()
    pool._all_channels = [retained]
    pool._channels = {3: retained}
    checkpoint = pool.checkpoint_channels()
    pool._all_channels.append(transient)
    pool._channels[7] = transient
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group: events.append("barrier"),
    )

    pool.rollback_channels(checkpoint)

    assert events == [
        "barrier",
        "close-imports",
        "barrier",
        "free-exports",
        "barrier",
    ]
    assert pool._all_channels == [retained]
    assert pool._channels == {3: retained}


def test_pool_binds_cuda_device_for_teardown_barriers(monkeypatch):
    barriers = []
    synchronizations = []

    class FakeChannel:
        def _close_ipc_imports(self):
            pass

        def _free_ipc_exports(self):
            pass

    group = object()
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cuda:3"),
        exchange_group=group,
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    retained = FakeChannel()
    transient = FakeChannel()
    pool._all_channels = [retained]
    checkpoint = pool.checkpoint_channels()
    pool._all_channels.append(transient)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.torch.cuda.synchronize",
        lambda device: synchronizations.append(device),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group, device_ids=None: barriers.append((group, device_ids)),
    )

    pool.rollback_channels(checkpoint)

    assert synchronizations == [torch.device("cuda:3")]
    assert barriers == [(group, [3])] * 3


def test_channel_destructor_retains_all_cuda_ownership_without_side_effects():
    events = []

    class FailingExt:
        def dispose_best_effort(self, ptr):
            events.append(("dispose_best_effort", ptr))

    class FailingIPC:
        def cudaIpcCloseMemHandle(self, ptr):
            events.append(("close", ptr))
            raise RuntimeError("close failed")

        def cudaFree(self, ptr):
            raise AssertionError("GC must not free exported IPC allocations")

    class SharedBuffer:
        def __init__(self, local_ptr, remote_ptrs):
            self.local_ptr = local_ptr
            self.remote_ptrs = remote_ptrs

    channel = object.__new__(PCIeOneshotAllReduce)
    channel._closed = False
    channel._ipc_imports_closed = False
    channel._ipc_exports_freed = False
    channel.exchange_group = None
    channel.device = torch.device("cpu")
    channel._ptr = 123
    channel._ext = FailingExt()
    channel._ipc = FailingIPC()
    channel._owned_buffers = [
        SharedBuffer(1000, (2000, 3000)),
        SharedBuffer(4000, (5000,)),
    ]
    channel._registered_input_ptrs = {1: (2,)}
    channel._coordinated_close_complete = False

    channel.__del__()

    assert events == []
    assert channel._ptr == 123
    assert not channel._closed
    assert not channel._ipc_imports_closed
    assert not channel._ipc_exports_freed
    assert [shared.local_ptr for shared in channel._owned_buffers] == [1000, 4000]
    assert channel._registered_input_ptrs == {1: (2,)}
    assert _ABANDONED_PCIE_RUNTIME_QUARANTINE.pop(id(channel)) is channel
    channel._coordinated_close_complete = True


def test_channel_gc_quarantines_rank_data_tensor_and_native_pointer():
    events = []

    class RankData:
        def __del__(self):
            events.append("rank_data_freed")

    channel = object.__new__(PCIeOneshotAllReduce)
    channel._ptr = 123
    channel._owned_buffers = []
    channel._coordinated_close_complete = False
    channel.rank_data = RankData()
    channel_id = id(channel)

    del channel
    gc.collect()

    retained = _ABANDONED_PCIE_RUNTIME_QUARANTINE.pop(channel_id)
    assert retained._ptr == 123
    assert events == []

    # The production quarantine is process-lifetime. Tests may release this
    # fake only after proving the owned allocation survived object GC.
    retained._coordinated_close_complete = True
    del retained
    gc.collect()
    assert events == ["rank_data_freed"]


def test_pool_rejects_channel_rollback_during_capture():
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    checkpoint = pool.checkpoint_channels()

    with pool.capture(7), pytest.raises(RuntimeError, match="during capture"):
        pool.rollback_channels(checkpoint)
