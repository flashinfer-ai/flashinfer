# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-only lifecycle contracts for the SM90 MXFP4 Green split session."""

from __future__ import annotations

import dataclasses
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
    MegaMoEHopperMxfp4SplitConfig,
    MegaMoEHopperMxfp4SplitSession,
    Mxfp4SplitLifecycleError,
    Mxfp4SplitSessionPoisonedError,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4_split import (
    _make_shape_validation_config,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)


def _config(**changes) -> MegaMoEHopperMxfp4SplitConfig:
    fields = dict(
        rank=0,
        world_size=2,
        num_tokens_per_rank=32,
        num_topk=2,
        num_total_experts=8,
        hidden=256,
        intermediate=128,
        k1_mma_tiler_mnk=(128, 64, 128),
        k2_mma_tiler_mnk=(128, 64, 128),
        k1_cluster_shape_mnk=(1, 1, 1),
        k2_cluster_shape_mnk=(1, 1, 1),
        k1_sm_count=80,
        k2_sm_count=52,
        k1_group_hint=80,
        k2_group_hint=52,
        k1_num_sched_stages=1,
        k2_num_sched_stages=3,
        counter_epoch_banks=1,
        graph_variant="steady_k3_reset",
    )
    fields.update(changes)
    return MegaMoEHopperMxfp4SplitConfig(**fields)


def _module_for_session():
    return sys.modules[MegaMoEHopperMxfp4SplitSession.__module__]


def _install_module(monkeypatch, name: str, **attributes):
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        qualified = ".".join(parts[:index])
        module = sys.modules.get(qualified)
        if module is None:
            module = ModuleType(qualified)
            if index != len(parts):
                module.__path__ = []
            monkeypatch.setitem(sys.modules, qualified, module)
        if index > 1:
            parent = sys.modules[".".join(parts[: index - 1])]
            monkeypatch.setattr(parent, parts[index - 1], module, raising=False)
    module = sys.modules[name]
    for key, value in attributes.items():
        monkeypatch.setattr(module, key, value, raising=False)
    return module


def test_split_config_routing_profile_is_strict_kw_only_session_identity() -> None:
    block = _config()
    exact = _config(
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    routing_field = dataclasses.fields(MegaMoEHopperMxfp4SplitConfig)[-1]
    assert routing_field.name == "routing_profile"
    assert routing_field.kw_only
    assert block.routing_profile == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert block != exact
    assert (
        _make_shape_validation_config(exact).routing_profile
        == SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
    )
    with pytest.raises(ValueError, match="routing_profile"):
        _config(routing_profile="published_exact_balanced")


def test_role_k_divisibility_uses_hidden_for_k1_and_public_i_for_k2() -> None:
    k1_k256 = _config(k1_mma_tiler_mnk=(128, 64, 256))
    assert k1_k256.hidden == 256 and k1_k256.intermediate == 128

    k2_k256 = _config(
        hidden=128,
        intermediate=256,
        k2_mma_tiler_mnk=(128, 64, 256),
    )
    assert k2_k256.hidden == 128 and k2_k256.intermediate == 256

    with pytest.raises(ValueError, match="hidden.*K1|K1.*hidden"):
        _config(hidden=128, k1_mma_tiler_mnk=(128, 64, 256))
    with pytest.raises(ValueError, match="intermediate.*K2|K2.*intermediate"):
        _config(intermediate=128, k2_mma_tiler_mnk=(128, 64, 256))


def test_every_tactic_graph_partition_and_bank_axis_is_session_identity() -> None:
    baseline = _config()
    variants = (
        dataclasses.replace(baseline, k1_mma_tiler_mnk=(256, 64, 128)),
        dataclasses.replace(baseline, k2_mma_tiler_mnk=(256, 64, 128)),
        dataclasses.replace(baseline, k1_group_hint=72),
        dataclasses.replace(baseline, k2_group_hint=44),
        dataclasses.replace(baseline, k1_num_sched_stages=2),
        dataclasses.replace(baseline, k2_num_sched_stages=2),
        dataclasses.replace(baseline, k1_sm_count=72),
        dataclasses.replace(baseline, k2_sm_count=60),
        dataclasses.replace(baseline, counter_epoch_banks=2),
        dataclasses.replace(baseline, graph_variant="cold_k0"),
    )
    assert all(candidate != baseline for candidate in variants)
    assert len({repr(candidate) for candidate in variants}) == len(variants)

    sessions = [MegaMoEHopperMxfp4SplitSession(c) for c in (baseline, *variants)]
    assert len({session.generation for session in sessions}) == len(sessions)
    assert all(
        session.config is config
        for session, config in zip(sessions, (baseline, *variants), strict=False)
    )


def test_plan_preserves_independent_tactics_stages_hints_and_sm_counts(
    monkeypatch,
) -> None:
    class SplitMegaTactic:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SplitMegaPlan:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _install_module(
        monkeypatch,
        "moe_hopper_fp8.split_mega_runner",
        SplitMegaPlan=SplitMegaPlan,
        SplitMegaTactic=SplitMegaTactic,
    )
    config = _config(
        k1_mma_tiler_mnk=(256, 128, 128),
        k2_mma_tiler_mnk=(128, 64, 128),
        k1_group_hint=240,
        k2_group_hint=156,
    )
    plan = MegaMoEHopperMxfp4SplitSession(config)._make_plan()

    assert plan.k1_sm_count == 80 and plan.k2_sm_count == 52
    assert plan.fc1_impl.mma_tiler_mnk == (256, 128, 128)
    assert plan.fc2_impl.mma_tiler_mnk == (128, 64, 128)
    assert plan.fc1_impl.group_hint == 240
    assert plan.fc2_impl.group_hint == 156
    assert plan.fc1_impl.num_sched_stages == 1
    assert plan.fc2_impl.num_sched_stages == 3
    assert plan.fc1_impl.load_balance_mode == "static"
    assert plan.fc2_impl.load_balance_mode == "static"


def test_prepare_maps_public_post_swiglu_i_to_donor_gateup_2i(monkeypatch) -> None:
    captured = {}
    cleanup_events = []

    class _StopAtFactory(RuntimeError):
        pass

    green = SimpleNamespace(
        sm_counts=(80, 52), close=lambda: cleanup_events.append("green")
    )
    support = SimpleNamespace(supported=True, total_sms=132, reason="")

    class GreenContextSplit:
        @staticmethod
        def create(k1_sm_count, *, device_ordinal):
            assert (k1_sm_count, device_ordinal) == (80, 0)
            return green

    def factory(problem, plan, **kwargs):
        captured["problem"] = problem
        captured["plan"] = plan
        captured["kwargs"] = kwargs
        raise _StopAtFactory("factory seam")

    _install_module(monkeypatch, "cuda.bindings.driver")
    _install_module(monkeypatch, "cutlass.cute")
    _install_module(monkeypatch, "cutlass.torch")
    _install_module(
        monkeypatch,
        "moe_hopper_fp8.green_context",
        GreenContextSplit=GreenContextSplit,
        check_green_context_support=lambda device: support,
    )
    _install_module(monkeypatch, "moe_hopper_fp8.green_graph", GreenGraph=object)
    _install_module(
        monkeypatch,
        "moe_hopper_fp8.split_mega_runner",
        build_mxfp4_split_kernel_pair=factory,
    )
    _install_module(monkeypatch, "src.sym_buffer", SymBufferHost=object)

    module = _module_for_session()
    monkeypatch.setattr(module, "ensure_not_capturing", lambda operation: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    session = MegaMoEHopperMxfp4SplitSession(_config(intermediate=128))
    session._inputs = object()
    session._make_plan = lambda: "independent-plan"

    with pytest.raises(_StopAtFactory, match="factory seam"):
        session._prepare()
    problem = captured["problem"]
    assert problem.intermediate == 256
    assert problem.hidden == 256
    assert problem.num_experts_per_rank == 4
    assert captured["kwargs"]["fp8_scale_mode"] == "mxfp4_hybrid"
    assert session.poisoned
    assert session._green_contexts is None
    assert cleanup_events == ["green"]

    # A failed prepare is poisoned/non-reusable, but explicit teardown remains
    # idempotent and must not release the already-cleaned context twice.
    session.destroy()
    assert session.destroyed and cleanup_events == ["green"]


def test_fixed_pointer_identity_includes_every_tensor_shape_device_and_stream(
    monkeypatch,
) -> None:
    class Tensor:
        def __init__(self, pointer, shape):
            self._pointer = pointer
            self.shape = shape

        def data_ptr(self):
            return self._pointer

    names = (
        "activation",
        "activation_sf",
        "topk_idx",
        "topk_weights",
        "fc1_weight",
        "fc1_weight_sf",
        "fc1_activation_dequant_scale",
        "fc1_weight_dequant_scale",
        "fc2_weight",
        "fc2_weight_sf",
        "fc2_activation_dequant_scale",
        "fc2_weight_dequant_scale",
        "output_activation",
    )
    inputs = SimpleNamespace(
        **{name: Tensor(index + 1, (index + 2,)) for index, name in enumerate(names)}
    )
    stream = {"value": 17}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=stream["value"]),
    )
    first = MegaMoEHopperMxfp4SplitSession._pointer_key(inputs)
    stream["value"] = 19
    second = MegaMoEHopperMxfp4SplitSession._pointer_key(inputs)
    assert first != second
    stream["value"] = 17
    inputs.fc2_weight._pointer += 100
    assert first != MegaMoEHopperMxfp4SplitSession._pointer_key(inputs)


def test_bind_inputs_rejects_pointer_change_without_rebinding(monkeypatch) -> None:
    session = MegaMoEHopperMxfp4SplitSession(_config())
    validated = []
    session._input_validator = SimpleNamespace(
        _validate_inputs=lambda inputs, num_tokens: validated.append(num_tokens)
    )
    session._pointer_key = lambda inputs: inputs.key
    first = SimpleNamespace(key=(1,), activation=SimpleNamespace(shape=(8, 256)))
    changed = SimpleNamespace(key=(2,), activation=SimpleNamespace(shape=(8, 256)))

    session.bind_inputs(first)
    with pytest.raises(Mxfp4SplitLifecycleError, match="fixed pointer"):
        session.bind_inputs(changed)
    assert session.fixed_pointer_key == (1,)
    assert session._inputs is first
    assert validated == [8, 8]


def test_capture_and_replay_failures_poison_the_session(monkeypatch) -> None:
    capture_session = MegaMoEHopperMxfp4SplitSession(_config())
    capture_session.warmup = lambda inputs: None
    capture_session._dist_barrier = lambda: None
    capture_session._initialize_epoch_state = lambda: None
    capture_session._executor = SimpleNamespace(
        capture=lambda: (_ for _ in ()).throw(RuntimeError("capture boom"))
    )
    with pytest.raises(Mxfp4SplitSessionPoisonedError, match="capture failed"):
        capture_session.capture(object())
    assert capture_session.poisoned
    with pytest.raises(Mxfp4SplitSessionPoisonedError, match="poisoned"):
        capture_session.capture(object())

    replay_session = MegaMoEHopperMxfp4SplitSession(_config())
    replay_session._captured = True
    replay_session._inputs = SimpleNamespace(output_activation=object())
    replay_session._prepared = SimpleNamespace(parent_stream=17)
    replay_session._executor = SimpleNamespace(
        launch=lambda: (_ for _ in ()).throw(RuntimeError("replay boom"))
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=17)
    )
    with pytest.raises(Mxfp4SplitSessionPoisonedError, match="replay failed"):
        replay_session.replay()
    assert replay_session.poisoned


def test_synchronize_failure_poisons_and_rejects_later_use() -> None:
    calls = []

    def fail_synchronize():
        calls.append("synchronize")
        raise RuntimeError("synchronize boom")

    session = MegaMoEHopperMxfp4SplitSession(_config())
    session._captured = True
    session._inputs = SimpleNamespace(output_activation=object())
    session._executor = SimpleNamespace(
        launch=lambda: calls.append("launch"),
        synchronize=fail_synchronize,
    )

    with pytest.raises(Mxfp4SplitSessionPoisonedError, match="synchronization failed"):
        session.synchronize()
    assert session.poisoned
    assert calls == ["synchronize"]

    with pytest.raises(Mxfp4SplitSessionPoisonedError, match="poisoned"):
        session.synchronize()
    with pytest.raises(Mxfp4SplitSessionPoisonedError, match="poisoned"):
        session.replay()
    assert calls == ["synchronize"]


def test_replay_requires_capture_and_the_fixed_primary_stream(monkeypatch) -> None:
    session = MegaMoEHopperMxfp4SplitSession(_config())
    with pytest.raises(Mxfp4SplitLifecycleError, match="capture"):
        session.replay()

    launched = []
    session._captured = True
    session._inputs = SimpleNamespace(output_activation=object())
    session._prepared = SimpleNamespace(parent_stream=17)
    session._executor = SimpleNamespace(launch=lambda: launched.append(True))
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=19)
    )
    with pytest.raises(Mxfp4SplitLifecycleError, match="fixed primary"):
        session.replay()
    assert not launched
    assert not session.poisoned


def test_handoff_metadata_view_maps_the_exact_local_workspace_region() -> None:
    class Contract:
        local_total_bytes = 32

        @staticmethod
        def region(address_space, name):
            assert (address_space, name) == ("local", "token_src_metadata")
            return SimpleNamespace(
                byte_offset=5,
                byte_size=12,
                shape=(3, 4),
                stride=(4, 1),
                dtype="Uint8",
                alignment=1,
            )

    session = MegaMoEHopperMxfp4SplitSession(_config())
    storage = torch.arange(32, dtype=torch.uint8)
    session._workspace_contract = Contract()
    session._local_workspace = storage

    metadata = session.handoff_metadata_view()
    assert metadata.dtype == torch.uint8
    assert metadata.shape == (3, 4)
    assert metadata.data_ptr() == storage.data_ptr() + 5
    torch.testing.assert_close(
        metadata,
        torch.arange(5, 17, dtype=torch.uint8).reshape(3, 4),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("counter_banks", [0, 2])
def test_initialize_epoch_state_orders_host_reset_before_cold_or_steady_reset(
    monkeypatch, counter_banks
) -> None:
    events = []

    class Driver:
        @staticmethod
        def cuStreamSynchronize(stream):
            events.append(("stream_sync", stream))
            return (0,)

    banks = tuple(
        SimpleNamespace(reset_memsets=(f"bank{index}",))
        for index in range(counter_banks)
    )
    prepared = SimpleNamespace(
        counter_banks=banks,
        k3_stream=23,
        driver=Driver(),
        _launch_memsets=lambda memsets, stream: events.append(
            ("memsets", memsets, stream)
        ),
        launch_k0=lambda stream: events.append(("k0", stream)),
    )
    session = MegaMoEHopperMxfp4SplitSession(_config())
    session._prepared = prepared
    session._reset_host_state = lambda: events.append(("host_reset",))
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: events.append(("cuda_sync",))
    )

    session._initialize_epoch_state()

    expected = [("host_reset",), ("cuda_sync",)]
    if counter_banks:
        expected.extend(
            ("memsets", (f"bank{index}",), 23) for index in range(counter_banks)
        )
    else:
        expected.append(("k0", 23))
    expected.append(("stream_sync", 23))
    assert events == expected


def test_destroy_is_idempotent_and_releases_graph_stream_context_workspace_order(
    monkeypatch,
) -> None:
    events = []

    class Driver:
        @staticmethod
        def cuStreamDestroy(stream):
            events.append(("stream", stream))
            return (0,)

    session = MegaMoEHopperMxfp4SplitSession(_config())
    session._executor = SimpleNamespace(close=lambda: events.append(("graph", None)))
    session._driver = Driver()
    session._k3_stream = 23
    session._green_contexts = SimpleNamespace(
        close=lambda: events.append(("green", None))
    )
    shared = object()
    session._shared_workspace = shared
    monkeypatch.setattr(
        _module_for_session(),
        "free_sym_tensor",
        lambda tensor: events.append(("shared", tensor)),
    )

    session.destroy()
    session.destroy()
    assert events == [
        ("graph", None),
        ("stream", 23),
        ("green", None),
        ("shared", shared),
    ]
    assert session.destroyed and not session.captured
    with pytest.raises(Mxfp4SplitLifecycleError, match="destroyed"):
        session.replay()
