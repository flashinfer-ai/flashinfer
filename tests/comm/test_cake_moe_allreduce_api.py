"""CPU contract tests for the optional Cake MoE communication backend."""

import hashlib
import json
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
        "moe_reduction_scale_input": torch.ones(
            experts * tokens, dtype=torch.float32
        ),
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
        "expanded_idx_to_permuted_idx": torch.zeros(
            tokens, top_k, dtype=torch.int32
        ),
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


def _cake_source_bundle_manifest(source: bytes) -> dict:
    return {
        "schema_version": 1,
        "arch": "sm_100a",
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
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def _write_cake_source_bundle(tmp_path, *, manifest_text: str | None = None):
    source = b'extern "C" __global__ void test_kernel() {}\n'
    source_path = tmp_path / "cake_moe_allreduce_fusion_kernels.cu"
    source_path.write_bytes(source)
    if manifest_text is None:
        manifest_text = json.dumps(_cake_source_bundle_manifest(source))
    (tmp_path / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return source_path, source


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
    monkeypatch.setattr(
        trtllm_ar, "_validate_cake_moe_reduction", lambda **kwargs: 3
    )
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
    monkeypatch.setattr(
        trtllm_ar, "_validate_cake_moe_finalize", lambda **kwargs: 1
    )
    monkeypatch.setattr(
        trtllm_ar, "get_cake_moe_comm_module", lambda device_index: module
    )

    trtllm_ar.trtllm_moe_finalize_allreduce_fusion(
        **_finalize_args(), backend="cake"
    )

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
            expanded_idx_to_permuted_idx=torch.zeros(
                tokens, top_k, dtype=torch.int32
            ),
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
            expanded_idx_to_permuted_idx=torch.zeros(
                tokens, top_k, dtype=torch.int32
            ),
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
            expanded_idx_to_permuted_idx=torch.zeros(
                tokens, top_k, dtype=torch.int32
            ),
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
        '{"schema_version": 1,',
        '{"schema_version": 1, "schema_version": 1,',
        1,
    )
    (tmp_path / "manifest.json").write_text(manifest_text, encoding="utf-8")
    monkeypatch.setattr(cake_moe_comm, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="contains duplicate key 'schema_version'"):
        cake_moe_comm._load_source_bundle()
