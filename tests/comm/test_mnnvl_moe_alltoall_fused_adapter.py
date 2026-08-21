# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI
import pytest
import torch

from flashinfer.comm import MoeAlltoAll, moe_a2a_active_rank_mask
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlMemory
from flashinfer.tllm_enums import SfLayout

from .conftest import mnnvl_available


def test_fused_module_keeps_the_public_python_contract():
    import flashinfer.comm.trtllm_moe_alltoall as api

    assert "moe_a2a_active_rank_mask" in api.__all__
    assert tuple(inspect.signature(api.moe_a2a_dispatch).parameters) == (
        "token_selected_experts",
        "input_payloads",
        "workspace",
        "metainfo",
        "runtime_max_tokens_per_rank",
        "ep_rank",
        "ep_size",
        "top_k",
        "num_experts",
        "enable_pdl",
        "eplb_local_stats",
        "enable_rank_mask",
        "active_rank_mask",
    )
    combine = inspect.signature(api.moe_a2a_combine)
    assert tuple(combine.parameters) == (
        "payload",
        "local_num_tokens",
        "workspace",
        "metainfo",
        "runtime_max_tokens_per_rank",
        "ep_rank",
        "ep_size",
        "top_k",
        "combine_payload_offset",
        "payload_in_workspace",
        "output_dtype",
        "output_scales",
        "output_scalar_scale",
        "sf_layout",
        "output",
        "use_low_precision",
        "enable_pdl",
        "enable_rank_mask",
        "active_rank_mask",
    )
    assert combine.parameters["sf_layout"].default is SfLayout.layout_linear
    assert combine.parameters["output"].default is None
    assert combine.parameters["use_low_precision"].kind is inspect.Parameter.KEYWORD_ONLY


def test_active_rank_mask_preserves_upper_u64_bits_in_dispatch():
    mask = moe_a2a_active_rank_mask((0, 31, 32, 63), 64)
    assert mask.dtype is torch.uint64
    assert mask.device.type == "cpu"
    assert mask.tolist() == [0x8000000180000001]

    dispatch_source = (
        Path(__file__).resolve().parents[2]
        / "csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllDispatch.cu"
    ).read_text()
    assert "1 << (unsigned long long)" not in dispatch_source
    assert dispatch_source.count("1ULL << (unsigned long long)") == 5


def test_fused_module_registers_the_existing_custom_ops(monkeypatch):
    import flashinfer.comm.trtllm_moe_alltoall as api

    registrations = []

    def register(name, *, mutates_args):
        def decorate(function):
            registrations.append((name, tuple(mutates_args), function.__name__))
            return function

        return decorate

    class FakeSpec:
        def build_and_load(self):
            return SimpleNamespace()

    api.get_moe_alltoall_module.cache_clear()
    monkeypatch.setattr(api, "register_custom_op", register)
    monkeypatch.setattr(api, "gen_moe_alltoall_module", lambda: FakeSpec())
    try:
        module = api.get_moe_alltoall_module()
        assert set(vars(module)) == {
            "moe_a2a_initialize",
            "moe_a2a_dispatch",
            "moe_a2a_combine",
            "moe_a2a_combine_into",
            "moe_a2a_sanitize_expert_ids",
            "moe_a2a_get_metainfo_index_pairs",
            "moe_a2a_get_aux_data_size",
        }
        assert {name for name, _, _ in registrations} == {
            "flashinfer::moe_a2a_initialize",
            "flashinfer::moe_a2a_dispatch",
            "flashinfer::moe_a2a_combine",
            "flashinfer::moe_a2a_combine_into",
            "flashinfer::moe_a2a_sanitize_expert_ids",
            "flashinfer::moe_a2a_get_metainfo_index_pairs",
            "flashinfer::moe_a2a_get_aux_data_size",
        }
        combine_into = next(row for row in registrations if row[0].endswith("combine_into"))
        assert combine_into[1] == ("workspace", "output")
    finally:
        api.get_moe_alltoall_module.cache_clear()


def test_fused_jit_inventory_is_self_contained(monkeypatch):
    import flashinfer.jit.comm as jit_comm

    captured = {}

    def capture(name, sources, **kwargs):
        captured.update(name=name, sources=tuple(sources), kwargs=kwargs)
        return object()

    monkeypatch.setattr(jit_comm, "gen_jit_spec", capture)
    jit_comm.gen_moe_alltoall_module()
    names = {path.name for path in captured["sources"]}
    assert captured["name"] == "mnnvl_moe_alltoall"
    assert "moeAlltoAllKernels.cu" not in names
    assert {
        "trtllm_moe_alltoall.cu",
        "moeAlltoAllFusedKernels.cu",
        "moeAlltoAllPrepareDispatch.cu",
        "moeAlltoAllDispatch.cu",
        "moeAlltoAllStageCombine.cu",
        "moeAlltoAllPublishCombine.cu",
        "moeAlltoAllCombine.cu",
        "moeAlltoAllQuantizeCombine.cu",
        "moeAlltoAllSanitize.cu",
    } <= names
    assert all(path.is_file() for path in captured["sources"])


def test_combine_inventory_is_specialized_and_fail_closed():
    source_root = (
        Path(__file__).resolve().parents[2]
        / "csrc/nv_internal/tensorrt_llm/kernels/communicationKernels"
    )
    combine_source = (source_root / "moeAlltoAllCombine.cu").read_text()
    launcher_source = (source_root / "moeAlltoAllFusedKernels.cu").read_text()
    supported_top_k = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 22)

    for top_k in supported_top_k:
        assert f"DEFINE_COMBINE_TOP_K({top_k})" in combine_source
        assert f"PRELOAD_COMBINE_TOP_K({top_k})" in launcher_source
        assert f"LAUNCH_COMBINE_TOP_K({top_k})" in launcher_source
    assert "unsupported top_k for moe_a2a_combine" in launcher_source
    assert "int ep_rank, int top_k" not in combine_source


def test_workspace_initialization_rendezvous_is_ordered_and_cached(monkeypatch):
    import flashinfer.comm.trtllm_moe_alltoall as api

    events = []
    workspace = object()
    metainfo = object()

    class FakeComm:
        def barrier(self):
            events.append("barrier")

    class FakeMnnvlMemory:
        allocated_map = {}

        def __init__(self, mapping, size):
            assert mapping is fake_mapping
            assert size == 4096
            events.append("allocate")
            self.ptr = 17
            self.allocated_map[self.ptr] = SimpleNamespace(comm=FakeComm())

        def as_torch_strided_tensor(self, dtype):
            assert dtype is torch.uint8
            events.append("view")
            return workspace

    def initialize(actual_workspace, ep_rank, ep_size, max_num_tokens, eplb_width):
        assert actual_workspace is workspace
        assert (ep_rank, ep_size, max_num_tokens, eplb_width) == (0, 2, 16, 5)
        events.append("initialize")
        return metainfo

    fake_mapping = object()
    monkeypatch.setattr(api, "MnnvlMemory", FakeMnnvlMemory)
    monkeypatch.setattr(api, "moe_a2a_initialize", initialize)
    monkeypatch.setattr(api.MoeAlltoAll, "_WORKSPACE_CACHE", {})

    first = api.MoeAlltoAll.get_workspace(4096, 0, 2, 16, fake_mapping, 5)
    second = api.MoeAlltoAll.get_workspace(4096, 0, 2, 16, fake_mapping, 5)

    assert first is second
    assert first["workspace"] is workspace
    assert first["metainfo"] is metainfo
    assert events == ["allocate", "view", "initialize", "barrier"]


def _payloads(rank, experts):
    tokens = experts.shape[0]
    columns = torch.arange(8, dtype=torch.float32, device="cuda")
    rows = torch.arange(tokens, dtype=torch.float32, device="cuda")[:, None]
    hidden = (rank * 100 + rows * 10 + columns).to(torch.bfloat16)
    weights = torch.arange(tokens * 2, dtype=torch.float32, device="cuda").reshape(tokens, 2)
    weights.add_(rank * 10)
    lora_ids = (rank * 10 + torch.arange(tokens, dtype=torch.int32, device="cuda"))[:, None]
    fp8 = (hidden.float() * 0.125).to(torch.float8_e4m3fn)
    packed = torch.arange(tokens * 4, dtype=torch.uint8, device="cuda").reshape(tokens, 4)
    packed.add_(rank * 20)
    return [hidden, experts, weights, lora_ids, fp8, packed]


def _owner(expert_id):
    return 0 if expert_id < 3 else 1


def _run_public_mpi2_cycle():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() != 2:
        pytest.skip("this independent source-adapter check requires two MPI ranks")

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    torch.cuda.set_device(node_comm.Get_rank())
    try:
        MnnvlMemory.initialize()
        if not mnnvl_available():
            pytest.skip("MNNVL not supported on this system")
    except Exception:
        pytest.skip("MNNVL not supported on this system")

    routes_by_rank = (
        ((0, 3), (4, 4), (2, 1)),
        ((3, 0), (1, 4), (3, 3)),
    )
    routes = torch.tensor(routes_by_rank[rank], dtype=torch.int32, device="cuda")
    payloads = _payloads(rank, routes)
    max_tokens = routes.shape[0]
    top_k = routes.shape[1]
    extra_payload_bytes = 4 + 8 + 4
    workspace_size = MoeAlltoAll.get_moe_workspace_size_per_rank(
        2,
        top_k,
        max_tokens,
        8,
        extra_payload_bytes_per_token=extra_payload_bytes,
        eplb_stats_num_experts=5,
    )
    mapping = Mapping(rank=rank, moe_ep_size=2, tp_size=2, world_size=2)
    collective = MoeAlltoAll(
        mapping,
        max_num_tokens=max_tokens,
        top_k=top_k,
        num_experts=5,
        workspace_size_per_rank=workspace_size,
        eplb_stats_num_experts=5,
        enable_rank_mask=True,
    )
    active_mask = moe_a2a_active_rank_mask((0, 1), 2)
    local_stats = rank * 100 + torch.arange(5, dtype=torch.int32, device="cuda")
    received = collective.dispatch(
        routes,
        payloads,
        max_tokens,
        invalid_token_expert_id=5,
        expert_id_payload_index=1,
        eplb_local_stats=local_stats,
        active_rank_mask=active_mask,
    )

    expected_stats = torch.stack(
        [source * 100 + torch.arange(5, dtype=torch.int32, device="cuda") for source in range(2)]
    )
    torch.testing.assert_close(collective.eplb_gathered_stats, expected_stats, atol=0, rtol=0)

    expected_payloads = [
        _payloads(
            source,
            torch.tensor(routes_by_rank[source], dtype=torch.int32, device="cuda"),
        )
        for source in range(2)
    ]
    valid_slots = {}
    for source in range(2):
        expected_tokens = [
            token
            for token, selected in enumerate(routes_by_rank[source])
            if rank in {_owner(expert) for expert in selected}
        ]
        slots = {
            int(received[3][source, slot, 0].item()): slot
            for slot in range(len(expected_tokens))
        }
        assert set(slots) == {source * 10 + token for token in expected_tokens}
        valid_slots[source] = slots
        for token in expected_tokens:
            slot = slots[source * 10 + token]
            for payload_index, source_payload in enumerate(expected_payloads[source]):
                torch.testing.assert_close(
                    received[payload_index][source, slot],
                    source_payload[token],
                    atol=0,
                    rtol=0,
                )
        if len(expected_tokens) < max_tokens:
            assert torch.all(received[1][source, len(expected_tokens) :] == 5)

    expert_output = torch.zeros_like(received[0])
    for source, slots in valid_slots.items():
        for slot in slots.values():
            expert_output[source, slot].copy_(received[0][source, slot] * (rank + 1))

    caller_output = torch.empty_like(payloads[0])
    result = collective.combine(
        expert_output,
        max_tokens,
        payload_in_workspace=False,
        output=caller_output,
        active_rank_mask=active_mask,
    )
    assert result is caller_output
    factors = torch.tensor(
        [
            sum({_owner(expert) + 1 for expert in selected})
            for selected in routes_by_rank[rank]
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    reference = payloads[0] * factors[:, None]
    torch.testing.assert_close(result, reference, atol=1e-2, rtol=1e-2)
    comm.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_public_mpi2_nondivisible_six_payload_eplb_and_external_output():
    _run_public_mpi2_cycle()
