"""CPU contract tests for the optional Cake MoE communication backend."""

import hashlib
import json
import re
from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import trtllm_ar
from flashinfer.jit import cake_moe_comm


def _reduction_args() -> dict:
    tokens = 1
    hidden = 4
    experts = 2
    dtype = torch.float16
    return {
        "world_size": 2,
        "world_rank": 0,
        "token_num": tokens,
        "hidden_dim": hidden,
        "workspace_ptrs": torch.zeros(7, dtype=torch.int64),
        "launch_with_pdl": False,
        "residual_in": torch.zeros(tokens * hidden, dtype=dtype),
        "rms_gamma": torch.ones(hidden, dtype=dtype),
        "rms_eps": 1e-6,
        "scale_factor": 1.0,
        "moe_reduction_device_num_experts": experts,
        "moe_reduction_scale_input": torch.ones(experts * tokens, dtype=torch.float32),
        "moe_reduction_active_experts_token_input": torch.zeros(
            experts * tokens * hidden, dtype=dtype
        ),
        "moe_reduction_token_input": torch.zeros(tokens * hidden, dtype=dtype),
        "layout_code": None,
        "moe_allreduce_out": None,
        "residual_out": torch.empty(tokens * hidden, dtype=dtype),
        "norm_out": torch.empty(tokens * hidden, dtype=dtype),
        "quant_out": None,
        "scale_out": None,
    }


def _finalize_args() -> dict:
    tokens = 1
    hidden = 4
    top_k = 8
    dtype = torch.bfloat16
    return {
        "allreduce_in": torch.zeros(3, hidden, dtype=dtype),
        "residual_in": torch.zeros(tokens, hidden, dtype=dtype),
        "norm_weight": torch.ones(hidden, dtype=dtype),
        "expanded_idx_to_permuted_idx": torch.zeros(tokens, top_k, dtype=torch.int32),
        "norm_out": torch.empty(tokens, hidden, dtype=dtype),
        "residual_out": torch.empty(tokens, hidden, dtype=dtype),
        "quant_out": None,
        "scale_out": None,
        "workspace_ptrs": torch.zeros(7, dtype=torch.int64),
        "launch_with_pdl": True,
        "world_rank": 0,
        "world_size": 2,
        "eps": 1e-6,
        "shared_expert_output": None,
        "expert_scale_factor": torch.ones(tokens, top_k, dtype=dtype),
        "routed_scaling_factor": None,
    }


def _cake_source_bundle_manifest(source: bytes, *, schema_version: int = 2) -> dict:
    if schema_version == 2:
        manifest = {"schema_version": 2, "arch": "sm_100a"}
    elif schema_version == 3:
        manifest = {
            "schema_version": 3,
            "architectures": ["sm_100a", "sm_103a"],
        }
    else:  # pragma: no cover - test helper callers use the schemas above.
        raise ValueError(f"unsupported test manifest schema {schema_version}")
    manifest.update(
        {
            "compile_flags": ["--use_fast_math"],
            "launch": {
                "block_threads": 224,
                "cluster_dim": [4, 1, 1],
                "dynamic_smem_bytes": 256,
            },
            "constraints": {
                "dtypes": ["float16", "bfloat16"],
                "hidden_dim": 7168,
                "max_tokens": 2048,
                "quantization": False,
                "world_sizes": [2, 4],
            },
            "kernel_symbols": list(cake_moe_comm._KERNEL_SYMBOLS),
            "workspace_protocol": {
                "flag_region_bytes_by_world_size": {"2": 2048, "4": 4288},
                "tp4_ack_tail": {
                    "offset_bytes": 4096,
                    "bytes": 192,
                    "generations": 3,
                    "consumers": 4,
                    "slot_bytes": 16,
                    "generation_stride_bytes": 64,
                    "empty_sentinel_u16": 32768,
                    "ready_u16": 0,
                },
            },
            "source_sha256": hashlib.sha256(source).hexdigest(),
        }
    )
    return manifest


def _write_cake_source_bundle(tmp_path, *, manifest_text: str | None = None):
    source = b'extern "C" __global__ void test_kernel() {}\n'
    source_path = tmp_path / "cake_moe_allreduce_fusion_kernels.cu"
    source_path.write_bytes(source)
    if manifest_text is None:
        manifest_text = json.dumps(_cake_source_bundle_manifest(source))
    (tmp_path / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return source_path, source


def _host_launch_argument_names(function_name: str) -> list[str]:
    function_start = cake_moe_comm._HOST_SOURCE.index(f"void {function_name}(")
    args_start = cake_moe_comm._HOST_SOURCE.index("void* args[] = {", function_start)
    args_end = cake_moe_comm._HOST_SOURCE.index("};", args_start)
    return re.findall(
        r"&([A-Za-z_]\w*)", cake_moe_comm._HOST_SOURCE[args_start:args_end]
    )


def test_cake_host_launch_abi_matches_pointer_table_bundle():
    assert _host_launch_argument_names("RunReduction") == [
        "p_active",
        "p_scales",
        "p_token",
        "p_residual",
        "p_gamma",
        "p_moe_out",
        "p_residual_out",
        "p_norm_out",
        "p_quant_out",
        "p_scale_out",
        "p_workspace",
        "rank32",
        "tokens32",
        "experts32",
        "eps32",
        "weight_bias32",
        "scale_factor32",
        "unused_layout",
    ]
    assert _host_launch_argument_names("RunFinalize") == [
        "p_allreduce",
        "p_indices",
        "p_scales",
        "p_shared",
        "p_residual",
        "p_gamma",
        "p_residual_out",
        "p_norm_out",
        "p_quant_out",
        "p_scale_out",
        "p_workspace",
        "rank32",
        "tokens32",
        "top_k32",
        "has_shared32",
        "routed32",
        "eps32",
        "weight_bias32",
        "unused_scale_factor",
    ]


def test_reduction_default_keeps_trtllm_dispatch(monkeypatch: pytest.MonkeyPatch):
    calls = []
    module = SimpleNamespace(
        trtllm_moe_allreduce_fusion=lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(trtllm_ar, "get_trtllm_comm_module", lambda: module)

    args = _reduction_args()
    trtllm_ar.trtllm_moe_reduction_allreduce_fusion(**args)

    assert len(calls) == 1
    assert calls[0]["world_size"] == args["world_size"]
    assert calls[0]["moe_reduction_token_input"] is args["moe_reduction_token_input"]
    assert (
        trtllm_ar.trtllm_moe_reduction_allreduce_fusion
        is trtllm_ar.trtllm_moe_allreduce_fusion
    )


def test_reduction_cake_dispatch_is_independent(monkeypatch: pytest.MonkeyPatch):
    calls = []
    module = SimpleNamespace(run_reduction=lambda *args: calls.append(args))
    monkeypatch.setattr(trtllm_ar, "_validate_cake_moe_reduction", lambda **kwargs: 3)
    monkeypatch.setattr(
        trtllm_ar, "get_cake_moe_comm_module", lambda device_index: module
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("TRT-LLM module must not load for backend='cake'"),
    )

    args = _reduction_args()
    trtllm_ar.trtllm_moe_reduction_allreduce_fusion(**args, backend="cake")

    assert len(calls) == 1
    assert calls[0][0:4] == (2, 0, 1, 4)


def test_finalize_cake_dispatch_defaults_routed_scale(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    module = SimpleNamespace(run_finalize=lambda *args: calls.append(args))
    monkeypatch.setattr(trtllm_ar, "_validate_cake_moe_finalize", lambda **kwargs: 1)
    monkeypatch.setattr(
        trtllm_ar, "get_cake_moe_comm_module", lambda device_index: module
    )

    trtllm_ar.trtllm_moe_finalize_allreduce_fusion(**_finalize_args(), backend="cake")

    assert len(calls) == 1
    assert calls[0][-2] == 1.0


def test_invalid_backend_fails_before_module_load(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("invalid backend must fail before module load"),
    )

    with pytest.raises(ValueError, match="unsupported MoE all-reduce backend"):
        trtllm_ar.trtllm_moe_reduction_allreduce_fusion(
            **_reduction_args(), backend="unknown"
        )


def test_cake_reduction_rejects_quantization(monkeypatch: pytest.MonkeyPatch):
    args = _reduction_args()
    args["hidden_dim"] = 7168
    args["quant_out"] = torch.empty(1, dtype=torch.float16)
    monkeypatch.setattr(
        trtllm_ar,
        "_check_cake_moe_common",
        lambda *args, **kwargs: (torch.device("cpu"), torch.float16, 0),
    )

    with pytest.raises(ValueError, match="does not support quantization"):
        trtllm_ar._validate_cake_moe_reduction(
            **{
                key: value
                for key, value in args.items()
                if key not in {"launch_with_pdl", "rms_eps", "scale_factor"}
            }
        )


def test_cake_finalize_rejects_wrong_shared_expert_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    tokens = 1
    hidden = 7168
    top_k = 8
    dtype = torch.bfloat16
    monkeypatch.setattr(
        trtllm_ar,
        "_check_cake_moe_common",
        lambda *args, **kwargs: (torch.device("cpu"), dtype, 0),
    )

    with pytest.raises(ValueError, match="shared_expert_output must contain"):
        trtllm_ar._validate_cake_moe_finalize(
            allreduce_in=torch.zeros(top_k, hidden, dtype=dtype),
            residual_in=torch.zeros(tokens, hidden, dtype=dtype),
            norm_weight=torch.ones(hidden, dtype=dtype),
            expanded_idx_to_permuted_idx=torch.zeros(tokens, top_k, dtype=torch.int32),
            norm_out=torch.empty(tokens, hidden, dtype=dtype),
            residual_out=torch.empty(tokens, hidden, dtype=dtype),
            quant_out=None,
            scale_out=None,
            workspace_ptrs=torch.zeros(7, dtype=torch.int64),
            world_rank=0,
            world_size=2,
            shared_expert_output=torch.zeros(tokens, hidden - 1, dtype=dtype),
            expert_scale_factor=torch.ones(tokens, top_k, dtype=dtype),
        )


def test_cake_finalize_rejects_shared_expert_on_another_device(
    monkeypatch: pytest.MonkeyPatch,
):
    tokens = 1
    hidden = 7168
    top_k = 8
    dtype = torch.float16
    monkeypatch.setattr(
        trtllm_ar,
        "_check_cake_moe_common",
        lambda *args, **kwargs: (torch.device("cpu"), dtype, 0),
    )

    with pytest.raises(ValueError, match="shared_expert_output must be on cpu"):
        trtllm_ar._validate_cake_moe_finalize(
            allreduce_in=torch.zeros(top_k, hidden, dtype=dtype),
            residual_in=torch.zeros(tokens, hidden, dtype=dtype),
            norm_weight=torch.ones(hidden, dtype=dtype),
            expanded_idx_to_permuted_idx=torch.zeros(tokens, top_k, dtype=torch.int32),
            norm_out=torch.empty(tokens, hidden, dtype=dtype),
            residual_out=torch.empty(tokens, hidden, dtype=dtype),
            quant_out=None,
            scale_out=None,
            workspace_ptrs=torch.zeros(7, dtype=torch.int64),
            world_rank=0,
            world_size=2,
            shared_expert_output=torch.empty(
                tokens, hidden, dtype=dtype, device="meta"
            ),
            expert_scale_factor=torch.ones(tokens, top_k, dtype=dtype),
        )


def test_cake_finalize_rejects_noncontiguous_allreduce_input(
    monkeypatch: pytest.MonkeyPatch,
):
    tokens = 1
    hidden = 7168
    top_k = 8
    dtype = torch.float16
    monkeypatch.setattr(
        trtllm_ar,
        "_check_cake_moe_common",
        lambda *args, **kwargs: (torch.device("cpu"), dtype, 0),
    )
    allreduce_in = torch.zeros(hidden, top_k, dtype=dtype).t()
    assert not allreduce_in.is_contiguous()

    with pytest.raises(ValueError, match="allreduce_in must be contiguous"):
        trtllm_ar._validate_cake_moe_finalize(
            allreduce_in=allreduce_in,
            residual_in=torch.zeros(tokens, hidden, dtype=dtype),
            norm_weight=torch.ones(hidden, dtype=dtype),
            expanded_idx_to_permuted_idx=torch.zeros(tokens, top_k, dtype=torch.int32),
            norm_out=torch.empty(tokens, hidden, dtype=dtype),
            residual_out=torch.empty(tokens, hidden, dtype=dtype),
            quant_out=None,
            scale_out=None,
            workspace_ptrs=torch.zeros(7, dtype=torch.int64),
            world_rank=0,
            world_size=2,
            shared_expert_output=None,
            expert_scale_factor=torch.ones(tokens, top_k, dtype=dtype),
        )


def test_cake_finalize_requires_one_allreduce_row_per_routed_token(
    monkeypatch: pytest.MonkeyPatch,
):
    tokens = 2
    hidden = 7168
    top_k = 8
    dtype = torch.bfloat16
    monkeypatch.setattr(
        trtllm_ar,
        "_check_cake_moe_common",
        lambda *args, **kwargs: (torch.device("cpu"), dtype, 0),
    )

    with pytest.raises(ValueError, match=r"token_num \* top_k rows"):
        trtllm_ar._validate_cake_moe_finalize(
            allreduce_in=torch.zeros(tokens * top_k - 1, hidden, dtype=dtype),
            residual_in=torch.zeros(tokens, hidden, dtype=dtype),
            norm_weight=torch.ones(hidden, dtype=dtype),
            expanded_idx_to_permuted_idx=torch.zeros(tokens, top_k, dtype=torch.int32),
            norm_out=torch.empty(tokens, hidden, dtype=dtype),
            residual_out=torch.empty(tokens, hidden, dtype=dtype),
            quant_out=None,
            scale_out=None,
            workspace_ptrs=torch.zeros(7, dtype=torch.int64),
            world_rank=0,
            world_size=2,
            shared_expert_output=None,
            expert_scale_factor=torch.ones(tokens, top_k, dtype=dtype),
        )


def test_missing_cake_source_bundle_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setattr(cake_moe_comm, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="source bundle is not installed"):
        cake_moe_comm._load_source_bundle()


def test_exact_cake_source_bundle_manifest_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    source_path, source = _write_cake_source_bundle(tmp_path)
    monkeypatch.setattr(cake_moe_comm, "_source_dir", lambda: tmp_path)

    loaded_path, loaded_source = cake_moe_comm._load_source_bundle()

    assert loaded_path == source_path
    assert loaded_source == source
    assert trtllm_ar._allreduce_fusion_flag_layout(2) == (2048, 0, 2048)
    assert trtllm_ar._allreduce_fusion_flag_layout(4) == (4096, 192, 4288)
    slots = {
        4096 + generation * 64 + consumer * 16
        for generation in range(3)
        for consumer in range(4)
    }
    assert len(slots) == 12
    assert min(slots) == 4096
    assert max(slots) + 16 == 4288
    assert all(offset % 16 == 0 for offset in slots)

    memsets = []
    initializers = []
    monkeypatch.setattr(
        trtllm_ar.cudart,
        "cudaMemset",
        lambda ptr, value, size: memsets.append((ptr.value, value, size)),
    )
    monkeypatch.setattr(
        trtllm_ar.cudart,
        "cudaMemcpy",
        lambda *args: None,
    )
    monkeypatch.setattr(
        trtllm_ar,
        "trtllm_lamport_initialize",
        lambda ptr, size, dtype: initializers.append((ptr, size, dtype)),
    )
    trtllm_ar._initialize_allreduce_fusion_protocol(
        ipc_handles=[[1000], [2000], [3000]],
        tp_rank=0,
        tp_size=4,
        flag_size=4288,
        lamport_buffer_size=6144,
        lamport_comm_size=2048,
        use_fp32_lamport=False,
        control_flag_ptr=4000,
    )
    assert memsets[0] == (2000, 0, 4096)
    assert initializers[0] == (2000 + 4096, 96, torch.float16)
    assert initializers[1] == (3000, 3072, torch.float16)
    with pytest.raises(ValueError, match="flag layout mismatch"):
        trtllm_ar._initialize_allreduce_fusion_protocol(
            ipc_handles=[[1000], [2000], [3000]],
            tp_rank=0,
            tp_size=4,
            flag_size=4096,
            lamport_buffer_size=6144,
            lamport_comm_size=2048,
            use_fp32_lamport=False,
            control_flag_ptr=4000,
        )
    with pytest.raises(ValueError, match=r"flag_size \(4096\).+4288"):
        trtllm_ar.check_trtllm_allreduce_fusion_workspace_metadata(
            token_num=1,
            hidden_dim=7168,
            world_size=4,
            dtype=torch.float16,
            metadata={
                "max_token_num": 2048,
                "tp_size": 4,
                "hidden_dim": 7168,
                "use_fp32_lamport": False,
                "flag_size": 4096,
            },
        )


def test_exact_dual_arch_cake_source_bundle_manifest_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    source_path, source = _write_cake_source_bundle(tmp_path)
    manifest = _cake_source_bundle_manifest(source, schema_version=3)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(cake_moe_comm, "_source_dir", lambda: tmp_path)

    assert cake_moe_comm._load_source_bundle() == (source_path, source)


@pytest.mark.parametrize(
    "capability,expected",
    (((10, 0), "sm_100a"), ((10, 3), "sm_103a")),
)
def test_cake_target_arch_maps_supported_blackwell_devices(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected: str,
):
    monkeypatch.setattr(
        cake_moe_comm.torch.cuda,
        "get_device_capability",
        lambda device_index: capability,
    )

    assert cake_moe_comm._target_arch(0) == expected


@pytest.mark.parametrize("capability", ((9, 0), (12, 0)))
def test_cake_target_arch_rejects_other_devices(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
):
    monkeypatch.setattr(
        cake_moe_comm.torch.cuda,
        "get_device_capability",
        lambda device_index: capability,
    )

    with pytest.raises(ValueError, match="requires SM100 or SM103"):
        cake_moe_comm._target_arch(0)


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
def test_public_cake_arch_guard_accepts_supported_blackwell_devices(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
):
    monkeypatch.setattr(
        trtllm_ar.torch.cuda,
        "get_device_capability",
        lambda device_index: capability,
    )

    trtllm_ar._check_cake_moe_arch(0)


@pytest.mark.parametrize("capability", ((9, 0), (12, 0)))
def test_public_cake_arch_guard_rejects_other_devices(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
):
    monkeypatch.setattr(
        trtllm_ar.torch.cuda,
        "get_device_capability",
        lambda device_index: capability,
    )

    with pytest.raises(ValueError, match="requires SM100 or SM103"):
        trtllm_ar._check_cake_moe_arch(0)


def test_dual_arch_cake_source_bundle_rejects_architecture_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    _, source = _write_cake_source_bundle(tmp_path)
    manifest = _cake_source_bundle_manifest(source, schema_version=3)
    manifest["architectures"] = ["sm_103a", "sm_100a"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(cake_moe_comm, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="mismatch for architectures"):
        cake_moe_comm._load_source_bundle()


def test_cake_source_bundle_rejects_unknown_manifest_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    source_path, source = _write_cake_source_bundle(tmp_path)
    manifest = _cake_source_bundle_manifest(source)
    manifest["unexpected"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(cake_moe_comm, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="top-level keys mismatch"):
        cake_moe_comm._load_source_bundle()

    assert source_path.is_file()


def test_cake_source_bundle_rejects_duplicate_manifest_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    _, source = _write_cake_source_bundle(tmp_path)
    manifest_text = json.dumps(_cake_source_bundle_manifest(source))
    manifest_text = manifest_text.replace(
        '{"schema_version": 2,',
        '{"schema_version": 2, "schema_version": 2,',
        1,
    )
    (tmp_path / "manifest.json").write_text(manifest_text, encoding="utf-8")
    monkeypatch.setattr(cake_moe_comm, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="contains duplicate key 'schema_version'"):
        cake_moe_comm._load_source_bundle()
