# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI
import pytest
import torch

from flashinfer import mxfp4_quantize, nvfp4_quantize
from flashinfer.comm import MoeAlltoAll, moe_a2a_active_rank_mask
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlMemory
from flashinfer.tllm_enums import SfLayout
from tests.utils_fp8 import mxfp8_quantize_reference

from .conftest import mnnvl_available


def test_fused_module_keeps_the_public_python_contract():
    import flashinfer.comm as public_api
    import flashinfer.comm.trtllm_moe_alltoall as api

    expected_module_parameters = {
        "moe_a2a_active_rank_mask": ("active_ranks", "ep_size"),
        "moe_a2a_initialize": (
            "workspace",
            "ep_rank",
            "ep_size",
            "max_num_tokens",
            "eplb_stats_num_experts",
        ),
        "moe_a2a_wrap_payload_tensor_in_workspace": (
            "workspace",
            "leading_shape",
            "slice_start",
            "slice_end",
            "dtype",
        ),
        "moe_a2a_dispatch": (
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
        ),
        "moe_a2a_combine": (
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
        ),
        "moe_a2a_sanitize_expert_ids": (
            "expert_ids",
            "workspace",
            "metainfo",
            "ep_rank",
            "invalid_expert_id",
            "enable_pdl",
        ),
        "moe_a2a_get_workspace_size_per_rank": (
            "ep_size",
            "max_num_tokens",
            "total_dispatch_payload_size_per_token",
            "combine_payload_size_per_token",
            "eplb_stats_num_experts",
        ),
    }
    expected_module_defaults = {
        "moe_a2a_active_rank_mask": {},
        "moe_a2a_initialize": {"eplb_stats_num_experts": 0},
        "moe_a2a_wrap_payload_tensor_in_workspace": {},
        "moe_a2a_dispatch": {
            "enable_pdl": None,
            "eplb_local_stats": None,
            "enable_rank_mask": False,
            "active_rank_mask": None,
        },
        "moe_a2a_combine": {
            "payload_in_workspace": False,
            "output_dtype": None,
            "output_scales": None,
            "output_scalar_scale": 1.0,
            "sf_layout": SfLayout.layout_linear,
            "output": None,
            "use_low_precision": False,
            "enable_pdl": None,
            "enable_rank_mask": False,
            "active_rank_mask": None,
        },
        "moe_a2a_sanitize_expert_ids": {"enable_pdl": None},
        "moe_a2a_get_workspace_size_per_rank": {"eplb_stats_num_experts": 0},
    }
    assert set(api.__all__) == {"MoeAlltoAll", *expected_module_parameters}
    for name, expected_parameters in expected_module_parameters.items():
        assert getattr(public_api, name) is getattr(api, name)
        signature = inspect.signature(getattr(api, name))
        assert tuple(signature.parameters) == expected_parameters
        assert {
            parameter_name: parameter.default
            for parameter_name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        } == expected_module_defaults[name]

    assert public_api.MoeAlltoAll is api.MoeAlltoAll
    expected_class_parameters = {
        "__init__": (
            "self",
            "mapping",
            "max_num_tokens",
            "top_k",
            "num_experts",
            "workspace_size_per_rank",
            "hidden_size",
            "mnnvl_config",
            "eplb_stats_num_experts",
            "enable_rank_mask",
        ),
        "get_workspace": (
            "workspace_size_per_rank",
            "ep_rank",
            "ep_size",
            "max_num_tokens",
            "mapping",
            "eplb_stats_num_experts",
        ),
        "get_moe_workspace_size_per_rank": (
            "ep_size",
            "top_k",
            "max_num_tokens",
            "hidden_size",
            "extra_payload_bytes_per_token",
            "eplb_stats_num_experts",
        ),
        "checkpoint_prepare": ("self",),
        "checkpoint_restore": ("self", "comm_backend"),
        "dispatch": (
            "self",
            "token_selected_experts",
            "input_payloads",
            "runtime_max_tokens_per_rank",
            "invalid_token_expert_id",
            "expert_id_payload_index",
            "eplb_local_stats",
            "active_rank_mask",
        ),
        "combine": (
            "self",
            "payload",
            "runtime_max_tokens_per_rank",
            "payload_in_workspace",
            "output_dtype",
            "output_scales",
            "output_scalar_scale",
            "sf_layout",
            "output",
            "use_low_precision",
            "active_rank_mask",
        ),
        "get_combine_payload_tensor_in_workspace": (
            "self",
            "runtime_max_tokens_per_rank",
            "hidden_size",
            "dtype",
        ),
    }
    expected_class_defaults = {
        "__init__": {
            "workspace_size_per_rank": None,
            "hidden_size": None,
            "mnnvl_config": None,
            "eplb_stats_num_experts": 0,
            "enable_rank_mask": False,
        },
        "get_workspace": {"eplb_stats_num_experts": 0},
        "get_moe_workspace_size_per_rank": {
            "extra_payload_bytes_per_token": 0,
            "eplb_stats_num_experts": 0,
        },
        "checkpoint_prepare": {},
        "checkpoint_restore": {},
        "dispatch": {
            "invalid_token_expert_id": None,
            "expert_id_payload_index": None,
            "eplb_local_stats": None,
            "active_rank_mask": None,
        },
        "combine": {
            "payload_in_workspace": False,
            "output_dtype": None,
            "output_scales": None,
            "output_scalar_scale": 1.0,
            "sf_layout": SfLayout.layout_linear,
            "output": None,
            "use_low_precision": False,
            "active_rank_mask": None,
        },
        "get_combine_payload_tensor_in_workspace": {},
    }
    for name, expected_parameters in expected_class_parameters.items():
        signature = inspect.signature(getattr(api.MoeAlltoAll, name))
        assert tuple(signature.parameters) == expected_parameters
        assert {
            parameter_name: parameter.default
            for parameter_name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        } == expected_class_defaults[name]

    combine = inspect.signature(api.moe_a2a_combine)
    assert combine.parameters["use_low_precision"].kind is inspect.Parameter.KEYWORD_ONLY
    class_combine = inspect.signature(api.MoeAlltoAll.combine)
    assert (
        class_combine.parameters["use_low_precision"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


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
    monkeypatch.setattr(api, "gen_moe_alltoall_module", lambda target: FakeSpec())
    monkeypatch.setattr(api.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(api.torch.cuda, "get_device_capability", lambda device: (10, 0))
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


@pytest.mark.parametrize(
    ("target", "module_name", "generated_name", "arch_flag"),
    [
        (
            "sm100a",
            "mnnvl_moe_alltoall_sm100a",
            "mnnvl_moe_alltoall_sm100.cu",
            "compute_100a,code=sm_100a",
        ),
        (
            "sm103a",
            "mnnvl_moe_alltoall_sm103a",
            "mnnvl_moe_alltoall_sm103.cu",
            "compute_103a,code=sm_103a",
        ),
    ],
)
def test_fused_jit_inventory_is_exact_arch_and_self_contained(
    monkeypatch, target, module_name, generated_name, arch_flag
):
    import flashinfer.jit.comm as jit_comm

    captured = {}

    def capture(name, sources, **kwargs):
        captured.update(name=name, sources=tuple(sources), kwargs=kwargs)
        return object()

    monkeypatch.setattr(jit_comm, "gen_jit_spec", capture)
    jit_comm.gen_moe_alltoall_module(target)
    names = {path.name for path in captured["sources"]}
    assert captured["name"] == module_name
    assert generated_name in names
    assert {
        "trtllm_moe_alltoall.cu",
        "moeAlltoAllFusedKernels.cu",
    } <= names
    assert not {
        "moeAlltoAllKernels.cu",
        "moeAlltoAllPrepareDispatch.cu",
        "moeAlltoAllDispatch.cu",
        "moeAlltoAllStageCombine.cu",
        "moeAlltoAllPublishCombine.cu",
        "moeAlltoAllCombine.cu",
        "moeAlltoAllQuantizeCombine.cu",
        "moeAlltoAllSanitize.cu",
    } & names
    assert {
        path.name for path in captured["sources"] if path.parent.name == "generated"
    } == {generated_name}
    assert any(arch_flag in flag for flag in captured["kwargs"]["extra_cuda_cflags"])


def test_runtime_module_selection_uses_exact_current_device_capability(monkeypatch):
    import flashinfer.comm.trtllm_moe_alltoall as api

    monkeypatch.setattr(api.torch.cuda, "current_device", lambda: 7)
    selected = []
    monkeypatch.setattr(
        api,
        "_get_moe_alltoall_module_for_target",
        lambda target: selected.append(target) or target,
    )

    monkeypatch.setattr(
        api.torch.cuda, "get_device_capability", lambda device: (10, 0)
    )
    assert api.get_moe_alltoall_module() == "sm100a"
    monkeypatch.setattr(
        api.torch.cuda, "get_device_capability", lambda device: (10, 3)
    )
    assert api.get_moe_alltoall_module() == "sm103a"
    assert selected == ["sm100a", "sm103a"]

    monkeypatch.setattr(
        api.torch.cuda, "get_device_capability", lambda device: (12, 0)
    )
    with pytest.raises(RuntimeError, match="exact compute capability 10.0 or 10.3"):
        api.get_moe_alltoall_module()


def test_aot_registers_each_exact_mnnvl_moe_target(monkeypatch):
    from flashinfer import aot
    import flashinfer.jit.comm as jit_comm

    selected = []

    def spec(name):
        return SimpleNamespace(name=name)

    monkeypatch.setattr(aot, "gen_spdlog_module", lambda: spec("spdlog"))
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(aot, "gen_cudnn_fmha_module", lambda: spec("cudnn"))
    monkeypatch.setattr(jit_comm, "gen_comm_alltoall_module", lambda: spec("comm"))
    monkeypatch.setattr(jit_comm, "gen_vllm_comm_module", lambda: spec("vllm"))
    monkeypatch.setattr(jit_comm, "gen_pcie_ipc_comm_module", lambda: spec("pcie"))
    monkeypatch.setattr(
        jit_comm,
        "gen_moe_alltoall_module",
        lambda target: selected.append(target) or spec(f"mnnvl_{target}"),
    )

    aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"sm100a_exact": True, "sm103a_exact": True},
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    assert selected == ["sm100a", "sm103a"]


def test_quantized_combine_preserves_the_physical_accumulator_dtype():
    repo_root = Path(__file__).resolve().parents[2]
    adapter_source = (repo_root / "csrc/trtllm_moe_alltoall.cu").read_text()
    launcher_source = (
        repo_root
        / "csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllFusedKernels.cu"
    ).read_text()

    assert (
        "alloc_tensor({localNumTokens, elementsPerToken}, payload.dtype(), payload.device())"
        in adapter_source
    )
    assert "static_cast<uint8_t*>(params.accumulation_data)" in launcher_source
    assert "params.elements_per_token, dtypeBytes(params.dtype)," in launcher_source
    assert "quantized ? kDTypeFloat32" not in launcher_source


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


def test_bf16_topk8_route_and_stage_grid_keep_exact_boundaries():
    launcher_source = (
        Path(__file__).resolve().parents[2]
        / "csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllFusedKernels.cu"
    ).read_text()

    specialized_kernel = (
        "kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk8"
    )
    assert launcher_source.count(specialized_kernel) == 3
    assert (
        "params.top_k == 8 && params.dtype == nvinfer1::DataType::kBF16 &&"
        in launcher_source
    )
    assert "params.elements_per_token % 8 == 0" in launcher_source
    assert (
        'preloadKernel("mnnvl_moe_alltoall_combine_bf16_topk8",'
        in launcher_source
    )
    assert (
        '"mnnvl_moe_alltoall_combine_bf16_topk8", params.enable_pdl,'
        in launcher_source
    )
    assert "uint64_t{kCombineThreads * 16}" in launcher_source
    assert "std::min(128, ceilDiv(payload_bytes," in launcher_source
    assert (
        "unsigned long long, bool, int, bool);" in launcher_source
    )
    assert (
        "payload_bytes,\n      true, params.ep_rank, params.enable_pdl);"
        in launcher_source
    )
    assert (
        "unsigned long long, unsigned long long, unsigned long long, bool, int, int, bool,\n"
        "    bool, unsigned long long);"
        in launcher_source
    )
    assert (
        "completion_offset, false, params.ep_rank, params.ep_size, params.enable_pdl,"
        in launcher_source
    )


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


_HIDDEN_SIZE = 128
_ROUTES_BY_RANK = (
    ((0, 3), (4, 4), (2, 1)),
    ((3, 0), (1, 4), (3, 3)),
)
_TOPK8_ROUTES_BY_RANK = (
    (
        (0, 1, 2, 3, 8, 9, 10, 11),
        (4, 5, 6, 7, 12, 13, 14, 15),
        (0, 2, 4, 6, 8, 10, 12, 14),
    ),
    (
        (1, 3, 5, 7, 9, 11, 13, 15),
        (0, 1, 6, 7, 8, 9, 14, 15),
        (2, 3, 4, 5, 10, 11, 12, 13),
    ),
)
_QUANTIZATION_CELLS = (
    ("mxfp8", SfLayout.layout_linear),
    ("mxfp8", SfLayout.layout_128x4),
    ("mxfp8", SfLayout.layout_8x4),
    ("mxfp4", SfLayout.layout_128x4),
    ("nvfp4", SfLayout.layout_128x4),
)


def _payloads(rank, experts):
    tokens = experts.shape[0]
    columns = torch.arange(_HIDDEN_SIZE, dtype=torch.float32, device="cuda")
    rows = torch.arange(tokens, dtype=torch.float32, device="cuda")[:, None]
    hidden = (rank * 100 + rows * 10 + columns).to(torch.bfloat16)
    top_k = experts.shape[1]
    weights = torch.arange(
        tokens * top_k, dtype=torch.float32, device="cuda"
    ).reshape(tokens, top_k)
    weights.add_(rank * 10)
    lora_ids = (
        rank * 10 + torch.arange(tokens, dtype=torch.int32, device="cuda")
    )[:, None]
    fp8 = (hidden.float() * 0.125).to(torch.float8_e4m3fn)
    packed = torch.arange(
        tokens * (_HIDDEN_SIZE // 2), dtype=torch.uint8, device="cuda"
    ).reshape(tokens, _HIDDEN_SIZE // 2)
    packed.add_(rank * 20)
    return [hidden, experts, weights, lora_ids, fp8, packed]


def _owner(expert_id, num_experts):
    experts_per_rank = (num_experts + 1) // 2
    return min(expert_id // experts_per_rank, 1)


def _scale_extent(quantization, rows, columns, layout):
    vector_size = 16 if quantization == "nvfp4" else 32
    scale_columns = (columns + vector_size - 1) // vector_size
    if layout is SfLayout.layout_linear:
        return rows * scale_columns
    if layout is SfLayout.layout_128x4:
        padded_rows = (rows + 127) // 128 * 128
    elif layout is SfLayout.layout_8x4:
        padded_rows = (rows + 7) // 8 * 8
    else:
        raise ValueError(f"unsupported scale layout: {layout}")
    padded_columns = (scale_columns + 3) // 4 * 4
    return padded_rows * padded_columns


def _assert_exact_physical_bytes(actual, expected, label):
    actual_bytes = actual.contiguous().view(torch.uint8).reshape(-1)
    expected_bytes = expected.contiguous().view(torch.uint8).reshape(-1)
    assert actual_bytes.shape == expected_bytes.shape, (
        f"{label} byte extent mismatch: "
        f"{actual_bytes.shape} != {expected_bytes.shape}"
    )
    if not torch.equal(actual_bytes, expected_bytes):
        mismatches = torch.nonzero(actual_bytes != expected_bytes, as_tuple=False)
        first = int(mismatches[0, 0].item())
        raise AssertionError(
            f"{label} physical bytes differ at {len(mismatches)} positions; "
            f"first index {first}: actual={int(actual_bytes[first].item())}, "
            f"expected={int(expected_bytes[first].item())}"
        )


def _dispatch_public_round(
    collective,
    rank,
    routes,
    payloads,
    active_mask,
    active_sources,
    *,
    gather_eplb,
    routes_by_rank=_ROUTES_BY_RANK,
    num_experts=5,
):
    local_stats = None
    if gather_eplb:
        local_stats = rank * 100 + torch.arange(5, dtype=torch.int32, device="cuda")
    received = collective.dispatch(
        routes,
        payloads,
        routes.shape[0],
        invalid_token_expert_id=num_experts,
        expert_id_payload_index=1,
        eplb_local_stats=local_stats,
        active_rank_mask=active_mask,
    )

    if gather_eplb:
        expected_stats = torch.stack(
            [
                source * 100
                + torch.arange(5, dtype=torch.int32, device="cuda")
                for source in active_sources
            ]
        )
        torch.testing.assert_close(
            collective.eplb_gathered_stats, expected_stats, atol=0, rtol=0
        )
    else:
        assert collective.eplb_gathered_stats is None

    expected_payloads = [
        _payloads(
            source,
            torch.tensor(routes_by_rank[source], dtype=torch.int32, device="cuda"),
        )
        for source in active_sources
    ]
    valid_slots = {}
    for source, source_payloads in zip(
        active_sources, expected_payloads, strict=True
    ):
        expected_tokens = [
            token
            for token, selected in enumerate(routes_by_rank[source])
            if rank in {_owner(expert, num_experts) for expert in selected}
        ]
        slots = {
            int(received[3][source, slot, 0].item()): slot
            for slot in range(len(expected_tokens))
        }
        assert set(slots) == {source * 10 + token for token in expected_tokens}
        valid_slots[source] = slots
        for token in expected_tokens:
            slot = slots[source * 10 + token]
            for payload_index, source_payload in enumerate(source_payloads):
                _assert_exact_physical_bytes(
                    received[payload_index][source, slot],
                    source_payload[token],
                    f"dispatch payload {payload_index}",
                )
        if len(expected_tokens) < routes.shape[0]:
            assert torch.all(
                received[1][source, len(expected_tokens) :] == num_experts
            )
    return received, valid_slots


def _fill_expert_output_and_reference(
    expert_output,
    received,
    valid_slots,
    payloads,
    rank,
    active_owners,
    *,
    routes_by_rank=_ROUTES_BY_RANK,
    num_experts=5,
):
    expert_output.zero_()
    for source, slots in valid_slots.items():
        for slot in slots.values():
            lora_id = int(received[3][source, slot, 0].item())
            factor = rank + 1 + (lora_id + 1) / 16
            expert_output[source, slot].copy_(
                (received[0][source, slot].float() * factor).to(torch.bfloat16)
            )

    reference = torch.zeros_like(payloads[0])
    for token, selected in enumerate(routes_by_rank[rank]):
        lora_id = int(payloads[3][token, 0].item())
        contributions = [
            (payloads[0][token].float() * (owner + 1 + (lora_id + 1) / 16)).to(
                torch.bfloat16
            )
            for owner in {_owner(expert, num_experts) for expert in selected}
            if owner in active_owners
        ]
        if contributions:
            reference[token].copy_(
                torch.stack(contributions).float().sum(dim=0).to(torch.bfloat16)
            )
    return reference


def _quantized_reference(reference, quantization, layout, scalar_scale):
    if quantization == "mxfp8":
        return mxfp8_quantize_reference(reference, sf_swizzle_layout=layout)
    if quantization == "mxfp4":
        return mxfp4_quantize(reference, sfLayout=layout)
    if quantization == "nvfp4":
        global_scale = torch.tensor(
            [scalar_scale], dtype=torch.float32, device=reference.device
        )
        return nvfp4_quantize(
            reference,
            global_scale,
            sfLayout=layout,
            sf_vec_size=16,
            do_shuffle=False,
        )
    raise ValueError(f"unsupported quantization: {quantization}")


def _run_public_combine_round(
    collective,
    rank,
    routes,
    payloads,
    active_mask,
    active_sources,
    active_owners,
    *,
    payload_in_workspace,
    gather_eplb=False,
    quantization=None,
    layout=SfLayout.layout_linear,
    routes_by_rank=_ROUTES_BY_RANK,
    num_experts=5,
):
    received, valid_slots = _dispatch_public_round(
        collective,
        rank,
        routes,
        payloads,
        active_mask,
        active_sources,
        gather_eplb=gather_eplb,
        routes_by_rank=routes_by_rank,
        num_experts=num_experts,
    )
    if payload_in_workspace:
        expert_output = collective.get_combine_payload_tensor_in_workspace(
            routes.shape[0], _HIDDEN_SIZE, torch.bfloat16
        )
    else:
        expert_output = torch.empty_like(received[0])
    reference = _fill_expert_output_and_reference(
        expert_output,
        received,
        valid_slots,
        payloads,
        rank,
        active_owners,
        routes_by_rank=routes_by_rank,
        num_experts=num_experts,
    )

    if quantization is None:
        caller_output = torch.empty_like(reference)
        result = collective.combine(
            expert_output,
            routes.shape[0],
            payload_in_workspace=payload_in_workspace,
            output=caller_output,
            active_rank_mask=active_mask,
        )
        assert result is caller_output
        torch.testing.assert_close(result, reference, atol=1e-2, rtol=1e-2)
        return

    if quantization == "mxfp8":
        output_dtype = torch.float8_e4m3fn
        scale_dtype = torch.uint8
        output_columns = _HIDDEN_SIZE
    else:
        output_dtype = torch.uint8
        scale_dtype = (
            torch.float8_e4m3fn if quantization == "nvfp4" else torch.uint8
        )
        output_columns = _HIDDEN_SIZE // 2
    output_scales = torch.zeros(
        _scale_extent(quantization, routes.shape[0], _HIDDEN_SIZE, layout),
        dtype=scale_dtype,
        device="cuda",
    )
    caller_output = torch.empty(
        routes.shape[0], output_columns, dtype=output_dtype, device="cuda"
    )
    scalar_scale = 2.5
    result = collective.combine(
        expert_output,
        routes.shape[0],
        payload_in_workspace=payload_in_workspace,
        output_dtype=output_dtype,
        output_scales=output_scales,
        output_scalar_scale=scalar_scale,
        sf_layout=layout,
        output=caller_output,
        active_rank_mask=active_mask,
    )
    assert result is caller_output
    expected_output, expected_scales = _quantized_reference(
        reference, quantization, layout, scalar_scale
    )
    _assert_exact_physical_bytes(result, expected_output, f"{quantization} output")
    _assert_exact_physical_bytes(
        output_scales, expected_scales, f"{quantization} scales"
    )


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

    routes = torch.tensor(_ROUTES_BY_RANK[rank], dtype=torch.int32, device="cuda")
    payloads = _payloads(rank, routes)
    max_tokens = routes.shape[0]
    top_k = routes.shape[1]
    extra_payload_bytes = 4 + _HIDDEN_SIZE + _HIDDEN_SIZE // 2
    workspace_size = MoeAlltoAll.get_moe_workspace_size_per_rank(
        2,
        top_k,
        max_tokens,
        _HIDDEN_SIZE,
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
    _run_public_combine_round(
        collective,
        rank,
        routes,
        payloads,
        active_mask,
        (0, 1),
        (0, 1),
        payload_in_workspace=False,
        gather_eplb=True,
    )
    _run_public_combine_round(
        collective,
        rank,
        routes,
        payloads,
        active_mask,
        (0, 1),
        (0, 1),
        payload_in_workspace=True,
    )

    topk8_routes = torch.tensor(
        _TOPK8_ROUTES_BY_RANK[rank], dtype=torch.int32, device="cuda"
    )
    topk8_payloads = _payloads(rank, topk8_routes)
    topk8_workspace_size = MoeAlltoAll.get_moe_workspace_size_per_rank(
        2,
        8,
        max_tokens,
        _HIDDEN_SIZE,
        extra_payload_bytes_per_token=extra_payload_bytes,
    )
    topk8_collective = MoeAlltoAll(
        mapping,
        max_num_tokens=max_tokens,
        top_k=8,
        num_experts=16,
        workspace_size_per_rank=topk8_workspace_size,
        enable_rank_mask=True,
    )
    _run_public_combine_round(
        topk8_collective,
        rank,
        topk8_routes,
        topk8_payloads,
        active_mask,
        (0, 1),
        (0, 1),
        payload_in_workspace=False,
        routes_by_rank=_TOPK8_ROUTES_BY_RANK,
        num_experts=16,
    )
    _run_public_combine_round(
        topk8_collective,
        rank,
        topk8_routes,
        topk8_payloads,
        active_mask,
        (0, 1),
        (0, 1),
        payload_in_workspace=False,
        quantization="nvfp4",
        layout=SfLayout.layout_128x4,
        routes_by_rank=_TOPK8_ROUTES_BY_RANK,
        num_experts=16,
    )

    for quantization, layout in _QUANTIZATION_CELLS:
        _run_public_combine_round(
            collective,
            rank,
            routes,
            payloads,
            active_mask,
            (0, 1),
            (0, 1),
            payload_in_workspace=False,
            quantization=quantization,
            layout=layout,
        )

    comm.barrier()
    if rank == 0:
        rank_zero_mask = moe_a2a_active_rank_mask((0,), 2)
        _run_public_combine_round(
            collective,
            rank,
            routes,
            payloads,
            rank_zero_mask,
            (0,),
            (0,),
            payload_in_workspace=False,
        )
    comm.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_public_mpi2_nondivisible_six_payload_eplb_and_external_output():
    """Exercise the complete source-adapter boundary through public APIs only."""
    _run_public_mpi2_cycle()
