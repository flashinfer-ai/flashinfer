"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CPU-only tests for the SM100 world-size-4 Cake MoE all-reduce union export."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import trtllm_ar
from flashinfer.jit import cake_trtllm_moe_allreduce_union as union

_RAW_POINTERS = {
    "workspace_control",
    *(f"workspace_payload_{peer}" for peer in range(4)),
}
_EXPECTED_ROUTE_KEYS = {
    (dtype, pdl, union.SPECIALIZATION_GENERIC)
    for dtype in ("bfloat16", "float16")
    for pdl in (False, True)
} | {
    ("bfloat16", False, union.SPECIALIZATION_T1_E8_SERIAL_CLEAR),
    ("float16", False, union.SPECIALIZATION_T64_E12_RESIDENT),
    ("float16", True, union.SPECIALIZATION_T128_E16_OWNER_FORWARD),
}


def test_module_inventory_is_verified_source_only() -> None:
    assert union.MODULES
    for name, record in union.MODULES.items():
        assert name.startswith("cake_") and record["cache_name"].startswith("cake_")
        assert record["kernel_symbol"].startswith("kernel_cake_")
        assert record["ffi_entry"] == "run"
        assert record["compile_flags"] == ["--use_fast_math"]
        paths = union.verified_sources(name)
        assert len(paths) == 2 and all(path.suffix == ".cu" for path in paths)
        kinds = [kind for kind, _key in record["arg_plan"]]
        assert set(kinds) <= {"buffer", "raw_pointer", "parameter", "grid"}
        assert record["arg_plan"][-3:] == [
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ]
        assert {
            key for kind, key in record["arg_plan"] if kind == "raw_pointer"
        } == _RAW_POINTERS
        assert ("buffer", "workspace_tensor") in record["arg_plan"]
        launch = record["launch"]
        assert tuple(launch["block"]) == (224, 1, 1) and tuple(launch["cluster"]) == (
            4,
            1,
            1,
        )


def test_routes_cover_exactly_the_reviewed_specializations() -> None:
    assert set(union.ROUTES) == _EXPECTED_ROUTE_KEYS
    for (_dtype, pdl, specialization), names in union.ROUTES.items():
        assert len(names) == union.WORLD_SIZE
        for name in names:
            record = union.MODULES[name]
            assert record["launch"]["use_pdl"] is pdl
            expected_cooperative = specialization in (
                union.SPECIALIZATION_T64_E12_RESIDENT,
                union.SPECIALIZATION_T128_E16_OWNER_FORWARD,
            )
            assert record["launch"]["cooperative"] is expected_cooperative
        if specialization == union.SPECIALIZATION_T128_E16_OWNER_FORWARD:
            # Owner forwarding is rank-specialized: four distinct physical modules.
            assert len(set(names)) == union.WORLD_SIZE


@pytest.mark.parametrize(
    "dtype_name,pdl,tokens,experts,expected",
    [
        ("bfloat16", False, 1, 8, union.SPECIALIZATION_T1_E8_SERIAL_CLEAR),
        ("bfloat16", True, 1, 8, union.SPECIALIZATION_GENERIC),
        ("float16", False, 1, 8, union.SPECIALIZATION_GENERIC),
        ("float16", False, 64, 12, union.SPECIALIZATION_T64_E12_RESIDENT),
        ("float16", True, 64, 12, union.SPECIALIZATION_GENERIC),
        ("float16", True, 128, 16, union.SPECIALIZATION_T128_E16_OWNER_FORWARD),
        ("bfloat16", False, 128, 16, union.SPECIALIZATION_GENERIC),
        ("bfloat16", True, 2048, 12, union.SPECIALIZATION_GENERIC),
    ],
)
def test_select_specialization_rules(
    dtype_name, pdl, tokens, experts, expected
) -> None:
    assert union.select_specialization(dtype_name, pdl, tokens, experts) == expected
    names = union.route_module_names(
        dtype_name=dtype_name,
        launch_with_pdl=pdl,
        token_num=tokens,
        active_experts=experts,
    )
    assert names == union.ROUTES[(dtype_name, pdl, expected)]
    assert (
        union.route_module_name(
            dtype_name=dtype_name,
            launch_with_pdl=pdl,
            token_num=tokens,
            active_experts=experts,
            world_rank=3,
        )
        == names[3]
    )
    with pytest.raises(ValueError):
        union.route_module_name(
            dtype_name=dtype_name,
            launch_with_pdl=pdl,
            token_num=tokens,
            active_experts=experts,
            world_rank=4,
        )


def test_route_scope_is_world_size_4_sm100_with_allreduce_output() -> None:
    assert union.route_applies(
        world_size=4, device_capability=(10, 0), emit_moe_allreduce=True
    )
    assert not union.route_applies(
        world_size=2, device_capability=(10, 0), emit_moe_allreduce=True
    )
    assert not union.route_applies(
        world_size=8, device_capability=(10, 0), emit_moe_allreduce=True
    )
    assert not union.route_applies(
        world_size=4, device_capability=(10, 3), emit_moe_allreduce=True
    )
    assert not union.route_applies(
        world_size=4, device_capability=(10, 0), emit_moe_allreduce=False
    )


def test_launch_grid_rule() -> None:
    assert union.launch_grid_x(1, False, 148) == 4
    assert union.launch_grid_x(2048, False, 148) == 148
    assert union.launch_grid_x(64, True, 148) == 256
    assert union.launch_grid_x(128, True, 148) == 512
    with pytest.raises(ValueError):
        union.launch_grid_x(0, False, 148)


def test_workspace_pointer_registry_round_trip() -> None:
    table = torch.arange(1, 14, dtype=torch.int64)
    pointers = [int(value) * 4096 for value in range(1, 14)]
    union.register_workspace_pointers(table, pointers)
    assert union.workspace_pointers(table, 4) == tuple(pointers)
    unregistered = torch.arange(101, 114, dtype=torch.int64)
    assert union.workspace_pointers(unregistered, 4) == tuple(range(101, 114))
    with pytest.raises(ValueError):
        union.workspace_pointers(torch.zeros(12, dtype=torch.int64), 4)
    with pytest.raises(ValueError):
        union.register_workspace_pointers(table, pointers[:-1])


def _arguments(world_size: int, *, emit_allreduce: bool) -> dict:
    tokens, hidden, experts = 1, 8, 2
    dtype = torch.float16
    return {
        "world_size": world_size,
        "world_rank": 1,
        "token_num": tokens,
        "hidden_dim": hidden,
        "workspace_ptrs": torch.zeros(3 * world_size + 1, dtype=torch.int64),
        "launch_with_pdl": True,
        "residual_in": torch.zeros(tokens, hidden, dtype=dtype),
        "rms_gamma": torch.ones(hidden, dtype=dtype),
        "rms_eps": 1e-5,
        "scale_factor": 1.0,
        "moe_reduction_device_num_experts": experts,
        "moe_reduction_scale_input": torch.ones(experts, tokens, dtype=torch.float32),
        "moe_reduction_active_experts_token_input": torch.zeros(
            experts, tokens, hidden, dtype=dtype
        ),
        "moe_reduction_token_input": torch.zeros(tokens, hidden, dtype=dtype),
        "layout_code": None,
        "moe_allreduce_out": torch.empty(tokens, hidden, dtype=dtype)
        if emit_allreduce
        else None,
        "residual_out": torch.empty(tokens, hidden, dtype=dtype),
        "norm_out": torch.empty(tokens, hidden, dtype=dtype),
        "quant_out": None,
        "scale_out": None,
        "weight_bias": 1.0,
    }


def _isolate_backends(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> tuple[list, list]:
    union_calls: list[dict] = []
    legacy_calls: list[tuple] = []
    monkeypatch.setattr(trtllm_ar, "_validate_cake_moe_allreduce", lambda **kwargs: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        union,
        "run_cake_moe_allreduce_union",
        lambda **kwargs: union_calls.append(kwargs),
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_cake_moe_allreduce_module",
        lambda device_index: SimpleNamespace(
            run_reduction=lambda *args: legacy_calls.append(args)
        ),
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("TRT-LLM module must not load for backend='cake'"),
    )
    return union_calls, legacy_calls


def test_cake_backend_routes_world_size_4_sm100_to_the_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    union_calls, legacy_calls = _isolate_backends(monkeypatch, (10, 0))
    arguments = _arguments(4, emit_allreduce=True)

    trtllm_ar.trtllm_moe_allreduce_fusion(**arguments, backend="cake")

    assert legacy_calls == []
    assert len(union_calls) == 1
    call = union_calls[0]
    assert call["backend"] == "cake"
    assert call["world_size"] == 4 and call["world_rank"] == 1
    assert call["workspace_ptrs"] is arguments["workspace_ptrs"]
    assert call["moe_allreduce_out"] is arguments["moe_allreduce_out"]
    assert call["residual_out"] is arguments["residual_out"]
    assert call["norm_out"] is arguments["norm_out"]
    assert call["launch_with_pdl"] is True and call["weight_bias"] == 1.0


@pytest.mark.parametrize(
    "world_size,capability,emit_allreduce",
    [(2, (10, 0), True), (8, (10, 0), True), (4, (10, 3), True), (4, (10, 0), False)],
    ids=("tp2", "tp8", "sm103", "no-allreduce-output"),
)
def test_cake_backend_keeps_the_legacy_bundle_outside_the_union_scope(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    capability: tuple[int, int],
    emit_allreduce: bool,
) -> None:
    union_calls, legacy_calls = _isolate_backends(monkeypatch, capability)
    arguments = _arguments(world_size, emit_allreduce=emit_allreduce)

    trtllm_ar.trtllm_moe_allreduce_fusion(**arguments, backend="cake")

    assert union_calls == []
    assert len(legacy_calls) == 1 and len(legacy_calls[0]) == 18
    assert legacy_calls[0][0] == world_size


def test_workspace_creation_registers_the_pointer_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[tuple] = []
    monkeypatch.setattr(
        union,
        "register_workspace_pointers",
        lambda tensor, pointers: registered.append((tensor, tuple(pointers))),
    )
    table = torch.arange(13, dtype=torch.int64)
    trtllm_ar.register_cake_moe_allreduce_workspace_pointers(table, list(range(13)))
    assert registered == [(table, tuple(range(13)))]
