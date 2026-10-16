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
            "backend",
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
            "backend",
            "recv_view_cache",
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
            "backend",
        ),
        "moe_a2a_sanitize_expert_ids": (
            "expert_ids",
            "workspace",
            "metainfo",
            "ep_rank",
            "invalid_expert_id",
            "enable_pdl",
            "backend",
        ),
        "moe_a2a_get_workspace_size_per_rank": (
            "ep_size",
            "max_num_tokens",
            "total_dispatch_payload_size_per_token",
            "combine_payload_size_per_token",
            "eplb_stats_num_experts",
            "backend",
        ),
    }
    expected_module_defaults = {
        "moe_a2a_active_rank_mask": {},
        "moe_a2a_initialize": {"eplb_stats_num_experts": 0, "backend": "trtllm"},
        "moe_a2a_wrap_payload_tensor_in_workspace": {},
        "moe_a2a_dispatch": {
            "enable_pdl": None,
            "eplb_local_stats": None,
            "enable_rank_mask": False,
            "active_rank_mask": None,
            "backend": "trtllm",
            "recv_view_cache": None,
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
            "backend": "trtllm",
        },
        "moe_a2a_sanitize_expert_ids": {"enable_pdl": None, "backend": "trtllm"},
        "moe_a2a_get_workspace_size_per_rank": {
            "eplb_stats_num_experts": 0,
            "backend": "trtllm",
        },
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
        if "backend" in signature.parameters:
            assert (
                signature.parameters["backend"].kind is inspect.Parameter.KEYWORD_ONLY
            )

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
            "backend",
        ),
        "get_workspace": (
            "workspace_size_per_rank",
            "ep_rank",
            "ep_size",
            "max_num_tokens",
            "mapping",
            "eplb_stats_num_experts",
            "backend",
        ),
        "get_moe_workspace_size_per_rank": (
            "ep_size",
            "top_k",
            "max_num_tokens",
            "hidden_size",
            "extra_payload_bytes_per_token",
            "eplb_stats_num_experts",
            "backend",
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
            "backend": "trtllm",
        },
        "get_workspace": {"eplb_stats_num_experts": 0, "backend": "trtllm"},
        "get_moe_workspace_size_per_rank": {
            "extra_payload_bytes_per_token": 0,
            "eplb_stats_num_experts": 0,
            "backend": "trtllm",
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
        if "backend" in signature.parameters:
            assert (
                signature.parameters["backend"].kind is inspect.Parameter.KEYWORD_ONLY
            )

    combine = inspect.signature(api.moe_a2a_combine)
    assert (
        combine.parameters["use_low_precision"].kind is inspect.Parameter.KEYWORD_ONLY
    )
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


def test_backends_register_distinct_custom_ops(monkeypatch):
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
    operation_names = {
        "moe_a2a_initialize",
        "moe_a2a_dispatch",
        "moe_a2a_combine",
        "moe_a2a_combine_into",
        "moe_a2a_sanitize_expert_ids",
        "moe_a2a_get_metainfo_index_pairs",
        "moe_a2a_get_aux_data_size",
    }
    try:
        expected_registrations = set()
        for target in ("legacy", "sm100a", "sm103a"):
            module = api._get_moe_alltoall_module_for_target(target)
            assert api._get_moe_alltoall_module_for_target(target) is module
            assert set(vars(module)) == operation_names
            suffix = "" if target == "legacy" else f"_{target}"
            expected_registrations.update(
                f"flashinfer::{name}{suffix}" for name in operation_names
            )
            combine_into = next(
                row
                for row in registrations
                if row[0] == f"flashinfer::moe_a2a_combine_into{suffix}"
            )
            assert combine_into[1] == ("workspace", "output")
        assert {name for name, _, _ in registrations} == expected_registrations
        assert len(registrations) == len(expected_registrations)
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
    assert (
        not {
            "moeAlltoAllKernels.cu",
            "moeAlltoAllPrepareDispatch.cu",
            "moeAlltoAllDispatch.cu",
            "moeAlltoAllStageCombine.cu",
            "moeAlltoAllPublishCombine.cu",
            "moeAlltoAllCombine.cu",
            "moeAlltoAllQuantizeCombine.cu",
            "moeAlltoAllSanitize.cu",
        }
        & names
    )
    assert {
        path.name for path in captured["sources"] if path.parent.name == "generated"
    } == {generated_name}
    assert any(arch_flag in flag for flag in captured["kwargs"]["extra_cuda_cflags"])


def test_legacy_jit_inventory_remains_available(monkeypatch):
    import flashinfer.jit.comm as jit_comm

    captured = {}

    def capture(name, sources, **kwargs):
        captured.update(name=name, sources=tuple(sources), kwargs=kwargs)
        return object()

    monkeypatch.setattr(jit_comm, "gen_jit_spec", capture)
    jit_comm.gen_moe_alltoall_module()
    names = {path.name for path in captured["sources"]}
    assert captured["name"] == "mnnvl_moe_alltoall"
    assert "moeAlltoAllKernels.cu" in names
    assert "moeAlltoAllFusedKernels.cu" not in names
    assert not {name for name in names if name.startswith("mnnvl_moe_alltoall_sm")}
    assert captured["kwargs"]["extra_cuda_cflags"] == ["-DENABLE_BF16"]


@pytest.mark.parametrize(
    ("capability", "cake_target"),
    [
        ((8, 0), None),
        ((9, 0), None),
        ((10, 0), "sm100a"),
        ((10, 3), "sm103a"),
        ((11, 0), None),
        ((12, 0), None),
        ((12, 1), None),
    ],
)
def test_runtime_module_selection_requires_explicit_cake_opt_in(
    monkeypatch, capability, cake_target
):
    import flashinfer.comm.trtllm_moe_alltoall as api

    monkeypatch.setattr(api.torch.cuda, "current_device", lambda: 7)
    selected = []
    capability_queries = []

    def get_device_capability(device):
        capability_queries.append(device)
        return capability

    def select_target(target):
        selected.append(target)
        return target

    select_target.cache_clear = lambda: None
    monkeypatch.setattr(api, "_get_moe_alltoall_module_for_target", select_target)
    monkeypatch.setattr(api.torch.cuda, "get_device_capability", get_device_capability)

    try:
        api.get_moe_alltoall_module.cache_clear()
        assert api.get_moe_alltoall_module() == "legacy"
        assert api.get_moe_alltoall_module(backend="trtllm") == "legacy"
        assert capability_queries == []
        if cake_target is None:
            with pytest.raises(ValueError, match="cake"):
                api.get_moe_alltoall_module(backend="cake")
            assert selected == ["legacy", "legacy"]
        else:
            assert api.get_moe_alltoall_module(backend="cake") == cake_target
            assert selected == ["legacy", "legacy", cake_target]
        assert capability_queries == [7]
    finally:
        api.get_moe_alltoall_module.cache_clear()


@pytest.mark.parametrize("backend", ["auto", "legacy", "sm100a", "", None])
def test_runtime_module_selection_rejects_invalid_backends(backend):
    import flashinfer.comm.trtllm_moe_alltoall as api

    with pytest.raises(ValueError, match="backend"):
        api.get_moe_alltoall_module(backend=backend)


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
        jit_comm, "gen_pcie_ipc_ag_rs_module", lambda: spec("pcie_ag_rs")
    )
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


def test_bf16_vector_routes_and_stage_grid_keep_exact_boundaries():
    launcher_source = (
        Path(__file__).resolve().parents[2]
        / "csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllFusedKernels.cu"
    ).read_text()

    for top_k in (1, 2, 4):
        assert f"PRELOAD_COMBINE_BF16_VECTOR_TOP_K({top_k})" in launcher_source
        assert f"LAUNCH_COMBINE_BF16_VECTOR_TOP_K({top_k})" in launcher_source
    assert (
        "(params.top_k == 1 || params.top_k == 2 || params.top_k == 4) &&"
        in launcher_source
    )
    assert (
        "params.dtype == nvinfer1::DataType::kBF16 && params.elements_per_token % 8 == 0;"
        in launcher_source
    )
    assert "unsupported top_k for the BF16 vector moe_a2a_combine" in launcher_source
    # the exported dispatch kernel carries the in-kernel sanitize switch; this launcher
    # keeps the separate public sanitize launch and passes it disabled
    assert "int sanitize_payload_index, int invalid_expert_id);" in launcher_source
    assert "params.active_rank_mask[0], -1, 0);" in launcher_source
    for top_k in (6, 8):
        specialized_kernel = (
            f"kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk{top_k}"
        )
        assert launcher_source.count(specialized_kernel) == 3
        assert (
            f"params.top_k == {top_k} && params.dtype == nvinfer1::DataType::kBF16 &&"
            in launcher_source
        )
        assert (
            f"preloadKernel<PreloadKernelSlot::kCombineBf16TopK{top_k}>("
            in launcher_source
        )
        assert (
            f'"mnnvl_moe_alltoall_combine_bf16_topk{top_k}", params.enable_pdl,'
            in launcher_source
        )
    # TOP_K=6, TOP_K=8 and the TOP_K in {1,2,4} vector route share the exact 16-byte boundary
    assert launcher_source.count("params.elements_per_token % 8 == 0") == 3
    assert launcher_source.count("dim3 const topk6_grid(") == 1
    assert "constexpr int kBf16TopK6Threads = 32;" in launcher_source
    assert (
        "ceilDiv(params.elements_per_token, kBf16TopK6Threads * 8)" in launcher_source
    )
    assert (
        "kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk6, topk6_grid, "
        "kBf16TopK6Threads, 0," in launcher_source
    )
    assert launcher_source.count("bool const fuse_topk6_publication =") == 2
    assert (
        launcher_source.count(
            "useBf16TopK6Combine(params) && params.prepare_payload != nullptr"
        )
        == 2
    )
    assert launcher_source.count("if (!fuse_topk6_publication)") == 2
    assert (
        "params.workspace_stride_bytes, flag_offset, completion_offset,"
        in launcher_source
    )
    assert (
        "params.ep_rank, params.ep_size,\n"
        "        params.use_low_precision, params.enable_pdl, "
        "fuse_topk6_publication,\n"
        "        params.enable_rank_mask, params.active_rank_mask[0]" in launcher_source
    )

    generated_root = Path(__file__).resolve().parents[2] / "csrc/generated"
    for generated_name in (
        "mnnvl_moe_alltoall_sm100.cu",
        "mnnvl_moe_alltoall_sm103.cu",
    ):
        generated_source = (generated_root / generated_name).read_text()
        normalized_generated_source = " ".join(generated_source.split())
        assert (
            normalized_generated_source.count(
                "void kernel_flashinfer_mnnvl_moe_alltoall_"
            )
            == 33
        )
        for top_k in (6, 8):
            assert (
                normalized_generated_source.count(
                    f"void kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk{top_k}"
                )
                == 1
            )
        for top_k in (1, 2, 4):
            assert (
                normalized_generated_source.count(
                    "void kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_vector_top_k_"
                    f"{top_k}("
                )
                == 1
            )
        assert normalized_generated_source.count("if (fuse_publication) {") == 1
        assert (
            "unsigned long long workspace_stride_bytes, "
            "unsigned long long flag_val_offset," in normalized_generated_source
        )
        assert (
            "unsigned long long completion_flags_offset, "
            "unsigned long long topk_target_ranks_offset,"
            in normalized_generated_source
        )
        assert (
            "int ep_size, bool use_low_precision, bool enable_pdl, "
            "bool fuse_publication," in normalized_generated_source
        )
        assert (
            "bool enable_rank_mask, unsigned long long active_rank_mask)"
            in normalized_generated_source
        )
        topk6_source = normalized_generated_source.split(
            "void kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk6(",
            1,
        )[1].split(
            "void kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk8(",
            1,
        )[0]
        assert (
            'asm volatile("fence.acquire.sys;" ::: "memory"); } __syncthreads();'
            in topk6_source
        )
        assert (
            '__syncthreads(); asm volatile("fence.acquire.sys;" ::: "memory");'
            not in topk6_source
        )
        assert "if (bid == 0 && blockIdx.y == 0)" in topk6_source
        assert "int column_chunk = blockIdx.y * 256;" in topk6_source
        assert "int column = column_chunk + tid * 8;" in topk6_source
        dispatch_source = normalized_generated_source.split(
            "void kernel_flashinfer_mnnvl_moe_alltoall_dispatch(",
            1,
        )[1].split(
            "void kernel_flashinfer_mnnvl_moe_alltoall_stage_combine(",
            1,
        )[0]
        assert "if (top_k == 6) { if (warp == 0) {" in dispatch_source
        assert dispatch_source.count("__shfl_sync(0xFFFFFFFF, target_rank,") == 5
        assert "} else if (tid == 0) {" in dispatch_source
        assert "smem_target_ranks[route] = stored_rank;" in dispatch_source
        assert (
            "workspace_i32[topk_workspace_base + "
            "(unsigned long long)route] = stored_rank;" in dispatch_source
        )

    assert "uint64_t{kCombineThreads * 16}" in launcher_source
    assert "std::min(128, ceilDiv(payload_bytes," in launcher_source
    assert "unsigned long long, bool, int, bool);" in launcher_source
    assert (
        "payload_bytes, true,\n      params.ep_rank, params.enable_pdl);"
        in launcher_source
    )
    assert (
        "unsigned long long, unsigned long long, unsigned long long, bool, int, int, bool,\n"
        "    bool, unsigned long long);" in launcher_source
    )
    normalized_launcher_source = " ".join(launcher_source.split())
    assert (
        "completion_offset, false, params.ep_rank, params.ep_size, "
        "params.enable_pdl, params.enable_rank_mask, params.active_rank_mask[0]"
        in normalized_launcher_source
    )


_FUSED_QUANTIZED_COMBINE_TOP_K = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 22)
_FUSED_QUANTIZED_COMBINE_SYMBOL = (
    "kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_"
)


def _normalized_cxx(source: str) -> str:
    return " ".join(source.replace("\\\n", "\n").split())


def test_fused_quantized_combine_routes_mirror_the_cake_host_rule():
    repo_root = Path(__file__).resolve().parents[2]
    kernel_root = (
        repo_root / "csrc/nv_internal/tensorrt_llm/kernels/communicationKernels"
    )
    launcher_source = (kernel_root / "moeAlltoAllFusedKernels.cu").read_text()
    launcher = _normalized_cxx(launcher_source)
    header = _normalized_cxx((kernel_root / "moeAlltoAllKernels.h").read_text())
    legacy = _normalized_cxx((kernel_root / "moeAlltoAllKernels.cu").read_text())
    adapter = _normalized_cxx((repo_root / "csrc/trtllm_moe_alltoall.cu").read_text())

    # one extern declaration, preload slot and launch case per exported TOP_K
    for top_k in _FUSED_QUANTIZED_COMBINE_TOP_K:
        assert f"DECLARE_COMBINE_QUANTIZED_TOP_K({top_k})" in launcher_source
        assert f"kCombineQuantizedTopK{top_k}," in launcher_source
        assert f"PRELOAD_COMBINE_QUANTIZED_TOP_K({top_k});" in launcher_source
        assert f"LAUNCH_COMBINE_QUANTIZED_TOP_K({top_k});" in launcher_source
    assert launcher_source.count("DECLARE_COMBINE_QUANTIZED_TOP_K(") == 12
    assert launcher_source.count("PRELOAD_COMBINE_QUANTIZED_TOP_K(") == 12
    assert launcher_source.count("LAUNCH_COMBINE_QUANTIZED_TOP_K(") == 12
    # the extern declarations carry the generated argument list: five byte pointers, four
    # workspace offsets, seven ints, the NVFP4 scalar scale, three ints and two flags
    assert (
        "kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_##TOP_K( "
        "uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, unsigned long long, "
        "unsigned long long, unsigned long long, unsigned long long, int, int, int, "
        "int, int, int, int, float, int, int, int, bool, bool);" in launcher
    )
    # the selection rule is the Cake host rule `_select_fused_quantized_combine`
    assert "constexpr bool kFusedQuantizedCombineEnabled = true;" in launcher_source
    assert (
        "bool useFusedQuantizedCombine(MoeA2ACombineParams const& params) { "
        "return kFusedQuantizedCombineEnabled && "
        "params.quant_mode != MoeA2ACombineQuantMode::NONE && "
        "params.local_num_tokens > 0 && params.elements_per_token % 32 == 0 && "
        "!useBf16TopK6Combine(params) && !useBf16TopK8Combine(params) && "
        "!useBf16VectorCombine(params); }" in launcher
    )
    # the fused route replaces the combine preload, skips the quantize preload and
    # launch, and returns before the carrier path
    assert launcher_source.count("useFusedQuantizedCombine(params)") == 3
    assert (
        "bool const fused_quantized_combine = useFusedQuantizedCombine(params); "
        "if (fused_quantized_combine) { preloadFusedQuantizedCombineKernel(params.top_k); } "
        "else if (useBf16TopK6Combine(params)) {" in launcher
    )
    assert (
        "if (params.quant_mode != MoeA2ACombineQuantMode::NONE && !fused_quantized_combine) {"
        in launcher_source
    )
    assert (
        "#undef LAUNCH_COMBINE_QUANTIZED_TOP_K return; } void* accumulation ="
        in launcher
    )
    assert (
        "unsupported top_k for the fused quantized moe_a2a_combine" in launcher_source
    )
    # launch constants: 256 threads, no dynamic shared memory, one CTA per local token
    assert "constexpr int kCombineThreads = 256;" in launcher_source
    assert (
        "kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_##TOP_K, grid, "
        "kCombineThreads, 0, params.stream, params.workspace, output_bytes, output_bytes, "
        "scale_bytes, scale_bytes, params.workspace_stride_bytes, "
        "byteOffset(params.topk_target_ranks, rank_workspace), "
        "byteOffset(params.topk_send_indices, rank_workspace), "
        "byteOffset(params.recv_buffers[params.ep_rank], rank_workspace), "
        "params.max_tokens_per_rank, params.local_num_tokens, params.elements_per_token, "
        "dtypeBytes(params.dtype), dtypeCode(params.dtype), layout.quant_mode, "
        "static_cast<int>(params.swizzle_mode), params.output_scalar_scale, "
        "layout.blocks_per_row, layout.padded_scale_cols, params.ep_rank, "
        "params.use_low_precision, params.enable_pdl); break" in launcher
    )
    assert "int const grid = std::max(params.local_num_tokens, 1);" in launcher_source
    # both quantized routes take the scale geometry from one helper
    assert launcher_source.count("quantizedCombineLayout(params);") == 2
    # the host boundary skips the BF16/FP16 intermediate on the fused route; the trtllm
    # backend never fuses, so backend="trtllm" keeps its allocation
    assert (
        "bool moe_a2a_combine_fuses_quantization(MoeA2ACombineParams const& params);"
        in header
    )
    assert (
        "bool moe_a2a_combine_fuses_quantization(MoeA2ACombineParams const& params) { "
        "return useFusedQuantizedCombine(params); }" in launcher
    )
    assert (
        "bool moe_a2a_combine_fuses_quantization(MoeA2ACombineParams const&) "
        "{ return false; }" in legacy
    )
    assert (
        "if (!tl_throughput::moe_a2a_combine_fuses_quantization(params)) { "
        "accumulationStorage = alloc_tensor({localNumTokens, elementsPerToken}, "
        "payload.dtype(), payload.device()); "
        "params.accumulation_data = accumulationStorage.data_ptr(); }" in adapter
    )

    generated_root = repo_root / "csrc/generated"
    for generated_name in (
        "mnnvl_moe_alltoall_sm100.cu",
        "mnnvl_moe_alltoall_sm103.cu",
    ):
        generated = " ".join((generated_root / generated_name).read_text().split())
        for top_k in _FUSED_QUANTIZED_COMBINE_TOP_K:
            symbol = f"{_FUSED_QUANTIZED_COMBINE_SYMBOL}{top_k}("
            assert generated.count(f"void {symbol}") == 1
            # 256 threads per CTA (the launcher passes kCombineThreads) and one TOP_K
            # specialization per symbol
            assert (
                f"#define THREADS 256 #define TOP_K {top_k} "
                f'extern "C" {{ __global__ __launch_bounds__(THREADS) void {symbol}'
                in generated
            )
            signature, body = generated.split(f"void {symbol}", 1)[1].split(")", 1)
            assert signature == (
                "uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, "
                "uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, "
                "uint8_t* __restrict__ scales_fp8, "
                "unsigned long long workspace_stride_bytes, "
                "unsigned long long topk_target_ranks_offset, "
                "unsigned long long topk_send_indices_offset, "
                "unsigned long long combine_payload_offset, int max_tokens_per_rank, "
                "int local_num_tokens, int elements_per_token, int payload_element_bytes, "
                "int payload_dtype_code, int quant_mode, int scale_layout, "
                "float output_scalar_scale, int blocks_per_row, int padded_scale_cols, "
                "int ep_rank, bool use_low_precision, bool enable_pdl"
            )
            # no static or dynamic shared memory: the launcher passes 0 dynamic bytes
            body = body.split("__global__", 1)[0]
            assert "__shared__" not in body
            assert "SMEM_TOTAL" not in body
        # the dispatch kernel owns the only dynamic smem pool in the bundle; its generated
        # footprint must match the launcher constant
        assert generated.count("extern __shared__") == 1
        assert generated.count("#define SMEM_TOTAL ") == 1
        assert "#define SMEM_TOTAL 256" in generated
    assert "constexpr int kDispatchSharedBytes = 256;" in launcher_source


def test_kernel_preload_cache_is_per_kernel_and_device_without_global_tree():
    launcher_source = (
        Path(__file__).resolve().parents[2]
        / "csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllFusedKernels.cu"
    ).read_text()

    assert "template <PreloadKernelSlot Slot, typename KernelFn>" in launcher_source
    assert (
        "static std::array<std::once_flag, kMaxPreloadDevices> preloaded_devices;"
        in launcher_source
    )
    assert "std::call_once(preloaded_devices[device]" in launcher_source
    assert "preload_mutex" not in launcher_source
    assert "std::set<" not in launcher_source


def test_workspace_initialization_rendezvous_is_ordered_and_cached_per_backend(
    monkeypatch,
):
    import flashinfer.comm.trtllm_moe_alltoall as api

    events = []
    workspaces = []
    metainfo_by_backend = {"trtllm": object(), "cake": object()}

    class FakeComm:
        def barrier(self):
            events.append("barrier")

    class FakeMnnvlMemory:
        allocated_map = {}

        def __init__(self, mapping, size):
            assert mapping is fake_mapping
            assert size == 4096
            events.append("allocate")
            self.ptr = 17 + len(self.allocated_map)
            self.workspace = object()
            workspaces.append(self.workspace)
            self.allocated_map[self.ptr] = SimpleNamespace(comm=FakeComm())

        def as_torch_strided_tensor(self, dtype):
            assert dtype is torch.uint8
            events.append("view")
            return self.workspace

    def initialize(
        actual_workspace, ep_rank, ep_size, max_num_tokens, eplb_width, *, backend
    ):
        assert actual_workspace is workspaces[-1]
        assert (ep_rank, ep_size, max_num_tokens, eplb_width) == (0, 2, 16, 5)
        events.append(f"initialize:{backend}")
        return metainfo_by_backend[backend]

    fake_mapping = object()
    monkeypatch.setattr(api, "MnnvlMemory", FakeMnnvlMemory)
    monkeypatch.setattr(api, "moe_a2a_initialize", initialize)
    monkeypatch.setattr(api.MoeAlltoAll, "_WORKSPACE_CACHE", {})

    first = api.MoeAlltoAll.get_workspace(4096, 0, 2, 16, fake_mapping, 5)
    second = api.MoeAlltoAll.get_workspace(
        4096, 0, 2, 16, fake_mapping, 5, backend="trtllm"
    )
    cake = api.MoeAlltoAll.get_workspace(
        4096, 0, 2, 16, fake_mapping, 5, backend="cake"
    )

    assert first is second
    assert cake is api.MoeAlltoAll.get_workspace(
        4096, 0, 2, 16, fake_mapping, 5, backend="cake"
    )
    assert first["workspace"] is workspaces[0]
    assert cake["workspace"] is workspaces[1]
    assert first["workspace"] is not cake["workspace"]
    assert first["metainfo"] is metainfo_by_backend["trtllm"]
    assert cake["metainfo"] is metainfo_by_backend["cake"]
    assert events == [
        "allocate",
        "view",
        "initialize:trtllm",
        "barrier",
        "allocate",
        "view",
        "initialize:cake",
        "barrier",
    ]


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
    weights = torch.arange(tokens * top_k, dtype=torch.float32, device="cuda").reshape(
        tokens, top_k
    )
    weights.add_(rank * 10)
    lora_ids = (rank * 10 + torch.arange(tokens, dtype=torch.int32, device="cuda"))[
        :, None
    ]
    fp8 = (hidden.float() * 0.125).to(torch.float8_e4m3fn)
    packed = torch.arange(
        tokens * (_HIDDEN_SIZE // 2), dtype=torch.uint8, device="cuda"
    ).reshape(tokens, _HIDDEN_SIZE // 2)
    packed.add_(rank * 20)
    # With two ranks and three tokens, the 4-byte LoRA plane leaves the next
    # receive plane only 8-byte aligned. Keep it before the wider FP8/packed
    # payloads to exercise address-aware vector widths in the default backend.
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
        f"{label} byte extent mismatch: {actual_bytes.shape} != {expected_bytes.shape}"
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
                source * 100 + torch.arange(5, dtype=torch.int32, device="cuda")
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
    for source, source_payloads in zip(active_sources, expected_payloads, strict=True):
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
            assert torch.all(received[1][source, len(expected_tokens) :] == num_experts)
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
        scale_dtype = torch.float8_e4m3fn if quantization == "nvfp4" else torch.uint8
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


def _run_public_mpi2_cycle(backend):
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

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the Cake backend requires SM100 or SM103")

    from flashinfer.comm.trtllm_moe_alltoall import get_moe_alltoall_module

    # Keep both implementations loaded before exercising either collective.
    legacy_module = get_moe_alltoall_module()
    cake_module = get_moe_alltoall_module(backend="cake")
    assert get_moe_alltoall_module(backend="trtllm") is legacy_module
    assert cake_module is not legacy_module

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
        backend=backend,
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
        backend=backend,
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
        backend=backend,
    )
    topk8_collective = MoeAlltoAll(
        mapping,
        max_num_tokens=max_tokens,
        top_k=8,
        num_experts=16,
        workspace_size_per_rank=topk8_workspace_size,
        enable_rank_mask=True,
        backend=backend,
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
    return collective, topk8_collective


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_public_mpi2_nondivisible_six_payload_eplb_and_external_output():
    """Exercise both backends in one process through public APIs only."""
    legacy = _run_public_mpi2_cycle(backend="trtllm")
    cake = _run_public_mpi2_cycle(backend="cake")
    for legacy_collective, cake_collective in zip(legacy, cake, strict=True):
        assert (
            legacy_collective.workspace.data_ptr()
            != cake_collective.workspace.data_ptr()
        )
    _run_public_mpi2_cycle(backend="trtllm")

    # Check a dropped peer only after all full-rank reuse checks. A peer that
    # misses collective epochs cannot rejoin without coordinated reinitialization.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.barrier()
    if rank == 0:
        routes = torch.tensor(_ROUTES_BY_RANK[rank], dtype=torch.int32, device="cuda")
        payloads = _payloads(rank, routes)
        rank_zero_mask = moe_a2a_active_rank_mask((0,), 2)
        for collective in (legacy[0], cake[0]):
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
