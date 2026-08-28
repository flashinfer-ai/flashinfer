"""CPU-only contracts for the BF16 rank-major source session and pooling."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = (
    _PROJECT_ROOT
    / "flashinfer"
    / "moe_ep"
    / "kernel_src"
    / "blackwell_bf16_rank_major"
)
_SESSION_PATH = _PACKAGE_ROOT / "session.py"
_SOURCE_NAME = "flashinfer_blackwell_moe_ep_layer_sm100.cu"


@cache
def _session_module():
    """Load the host contract without importing the FlashInfer package tree."""
    module_name = "_flashinfer_bf16_rank_major_session_cpu_contract"
    spec = importlib.util.spec_from_file_location(module_name, _SESSION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(module, source: bytes) -> dict:
    stages = []
    for name in module._STAGE_NAMES:
        (
            grid,
            block,
            cluster,
            dynamic_smem_bytes,
            scalar_bindings,
            pdl_sync,
            pdl_launch,
            use_pdl,
        ) = module._STAGE_LAUNCH_CONTRACTS[name]
        stages.append(
            {
                "name": name,
                "symbol": f"test_{name}",
                "grid": list(grid),
                "block": list(block),
                "cluster": list(cluster),
                "dynamic_smem_bytes": dynamic_smem_bytes,
                "scalar_bindings": [
                    list(binding) for binding in scalar_bindings
                ],
                "pdl_sync": pdl_sync,
                "pdl_launch": pdl_launch,
                "use_pdl": use_pdl,
                "bindings": list(module._STAGE_BINDINGS[name]),
            }
        )
    return {
        "schema_version": 1,
        "arch": "sm_100a",
        "compile_flags": ["--use_fast_math"],
        "constraints": {
            "activation_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "world_size": 8,
            "tokens_per_rank": 128,
            "hidden_dim": 7168,
            "intermediate_dim": 2048,
            "num_experts": 256,
            "local_experts": 32,
            "top_k": 8,
            "layout": "rank_major",
        },
        "stages": stages,
        "kernel_symbols": [stage["symbol"] for stage in stages],
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def _write_bundle(tmp_path: Path, module, manifest: dict | None = None):
    source = b"// CPU manifest contract fixture\n"
    source_path = tmp_path / _SOURCE_NAME
    source_path.write_bytes(source)
    resolved_manifest = manifest or _manifest(module, source)
    (tmp_path / "manifest.json").write_text(json.dumps(resolved_manifest))
    return resolved_manifest, source_path


def test_session_import_defers_optional_cuda_driver_bindings():
    driver_before = sys.modules.get("cuda.bindings.driver")
    _session_module()
    assert sys.modules.get("cuda.bindings.driver") is driver_before


def test_packaged_source_manifest_is_complete_and_self_consistent():
    module = _session_module()
    manifest, source_path = module._load_manifest()

    assert source_path == _PACKAGE_ROOT / "src" / _SOURCE_NAME
    assert manifest["source_sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    assert tuple(stage["name"] for stage in manifest["stages"]) == (
        module._STAGE_NAMES
    )


def test_package_data_declares_the_immutable_source_pair():
    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    key = '"flashinfer.moe_ep.kernel_src.blackwell_bf16_rank_major" = ['
    assert key in pyproject
    package_block = pyproject.split(key, maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert '"*.md"' in package_block
    assert '"src/*.cu"' in package_block
    assert '"src/*.json"' in package_block


def test_manifest_accepts_only_the_exact_host_contract(tmp_path, monkeypatch):
    module = _session_module()
    expected, source_path = _write_bundle(tmp_path, module)
    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)

    actual, actual_source = module._load_manifest()

    assert actual == expected
    assert actual_source == source_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda manifest: manifest.update(schema_version=2), "schema must be 1"),
        (lambda manifest: manifest.update(arch="sm_103a"), "arch must be sm_100a"),
        (
            lambda manifest: manifest.update(compile_flags=["-lineinfo"]),
            "compile flags",
        ),
        (
            lambda manifest: manifest["constraints"].update(tokens_per_rank=127),
            "constraints drifted",
        ),
        (
            lambda manifest: manifest["stages"].reverse(),
            "stage order drifted",
        ),
        (
            lambda manifest: manifest["kernel_symbols"].reverse(),
            "kernel symbol order drifted",
        ),
        (
            lambda manifest: manifest.update(source_sha256="0" * 64),
            "source checksum differs",
        ),
        (
            lambda manifest: manifest["stages"][0].update(grid=[1, 0, 1]),
            "invalid grid",
        ),
        (
            lambda manifest: manifest["stages"][0].update(grid=[2, 1, 1]),
            "launch contract drifted",
        ),
        (
            lambda manifest: manifest["stages"][6].update(
                scalar_bindings=[["K", 2048]]
            ),
            "launch contract drifted",
        ),
        (
            lambda manifest: manifest["stages"][6].update(pdl_sync=False),
            "launch contract drifted",
        ),
        (
            lambda manifest: manifest["stages"][0].update(
                dynamic_smem_bytes=-1
            ),
            "invalid shared memory",
        ),
    ),
)
def test_manifest_drift_fails_closed(tmp_path, monkeypatch, mutation, message):
    module = _session_module()
    source = b"// CPU manifest contract fixture\n"
    manifest = _manifest(module, source)
    mutation(manifest)
    _write_bundle(tmp_path, module, manifest)
    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match=message):
        module._load_manifest()


@pytest.mark.parametrize("missing", (_SOURCE_NAME, "manifest.json"))
def test_manifest_requires_both_generated_files(tmp_path, monkeypatch, missing):
    module = _session_module()
    _write_bundle(tmp_path, module)
    (tmp_path / missing).unlink()
    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="package is incomplete"):
        module._load_manifest()


def test_source_tampering_is_rejected_by_manifest_checksum(tmp_path, monkeypatch):
    module = _session_module()
    _, source_path = _write_bundle(tmp_path, module)
    source_path.write_bytes(source_path.read_bytes() + b"// tampered\n")
    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="source checksum differs"):
        module._load_manifest()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        ("world_size", 4, 8),
        ("max_tokens_per_rank", 64, 128),
        ("hidden_size", 4096, 7168),
        ("intermediate_size", 1024, 2048),
        ("num_experts", 128, 256),
        ("top_k", 4, 8),
    ),
)
def test_session_rejects_coordinate_drift_before_cuda_use(field, value, expected):
    module = _session_module()
    kwargs = {
        "process_group": object(),
        "rank": 0,
        "world_size": 8,
        "max_tokens_per_rank": 128,
        "hidden_size": 7168,
        "intermediate_size": 2048,
        "num_experts": 256,
        "top_k": 8,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=rf"{field}={expected}"):
        module.BlackwellBf16RankMajorSession(**kwargs)


def test_exact_session_coordinate_fails_cleanly_without_cuda(monkeypatch):
    module = _session_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="session requires CUDA"):
        module.BlackwellBf16RankMajorSession(
            process_group=object(),
            rank=0,
            world_size=8,
            max_tokens_per_rank=128,
            hidden_size=7168,
            intermediate_size=2048,
            num_experts=256,
            top_k=8,
        )


def test_process_group_name_is_required():
    module = _session_module()
    assert module._group_name(SimpleNamespace(group_name="ep_group")) == "ep_group"
    with pytest.raises(RuntimeError, match="requires a named torch process group"):
        module._group_name(SimpleNamespace(group_name=""))


def _physical_row_order(rows: int) -> torch.Tensor:
    logical = torch.arange(rows)
    row_in_block = logical % 32
    return (logical // 32) * 32 + (row_in_block % 4) * 8 + row_in_block // 4


def _unpack_block_major(weight: torch.Tensor) -> torch.Tensor:
    experts, blocks, rows, width = weight.shape
    return weight.permute(0, 2, 1, 3).reshape(experts, rows, blocks * width)


def test_weight_transform_matches_full_fc1_and_fc2_logical_layout():
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_rank_major_cuda.weights import (
        preprocess_mega_weights,
    )

    experts = 2
    intermediate = hidden = 64
    elements_w13 = experts * 2 * intermediate * hidden
    elements_w2 = experts * hidden * intermediate
    w13 = (torch.arange(elements_w13) % 997).to(torch.bfloat16).reshape(
        experts, 2 * intermediate, hidden
    )
    w2 = (torch.arange(elements_w2) % 991).to(torch.bfloat16).reshape(
        experts, hidden, intermediate
    )

    transformed = preprocess_mega_weights(
        MoEWeightPack(w13, w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
        num_local_experts=experts,
    )
    physical_fc1 = _unpack_block_major(transformed.w13_block_major)
    physical_fc2 = _unpack_block_major(transformed.w2_block_major)
    fc1_rows = _physical_row_order(2 * intermediate)
    fc2_rows = _physical_row_order(hidden)
    logical_fc1 = physical_fc1.index_select(1, fc1_rows)
    logical_fc2 = physical_fc2.index_select(1, fc2_rows)
    gate, up = w13.split(intermediate, dim=1)
    expected_fc1 = torch.stack((up, gate), dim=2).reshape_as(logical_fc1)

    assert torch.equal(logical_fc1, expected_fc1)
    assert torch.equal(logical_fc2, w2)


def test_cpu_transformed_weights_cannot_cross_the_cuda_session_boundary():
    from flashinfer.moe_ep import MoEWeightPack, MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_rank_major_cuda.weights import (
        preprocess_mega_weights,
        validate_transformed_mega_weights,
    )

    hidden = intermediate = 64
    transformed = preprocess_mega_weights(
        MoEWeightPack(
            torch.zeros(1, 2 * intermediate, hidden, dtype=torch.bfloat16),
            torch.zeros(1, hidden, intermediate, dtype=torch.bfloat16),
        ),
        intermediate_size=intermediate,
        hidden_size=hidden,
        num_local_experts=1,
    )

    with pytest.raises(MoEEpConfigError, match="must be CUDA tensors"):
        validate_transformed_mega_weights(
            transformed,
            intermediate_size=intermediate,
            hidden_size=hidden,
            num_local_experts=1,
        )


def _exact_fleet():
    from flashinfer.moe_ep import EpAlgorithm, EpLayout, FleetParams

    return FleetParams(
        num_experts=256,
        max_tokens_per_rank=128,
        token_hidden_size=7168,
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
        layout=EpLayout.RANK_MAJOR,
    )


def _bound_backend(*, group, rank=1, world_size=8, config=None):
    from flashinfer.moe_ep import (
        Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_rank_major_cuda.backend import (
        Bf16RankMajorCudaMegaKernelBackend,
    )

    config = config or Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig()
    backend = Bf16RankMajorCudaMegaKernelBackend(config)
    bootstrap = object()
    backend._ep_bootstrap = bootstrap
    backend._ep_comm_group = group
    backend._ep_rank = rank
    backend._ep_world_size = world_size
    return backend, bootstrap


def test_workspace_pool_key_covers_every_allocation_coordinate():
    from flashinfer.moe_ep import (
        Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig,
    )

    fleet = _exact_fleet()
    group = object()
    backend, _ = _bound_backend(group=group)
    with mock.patch("torch.cuda.current_device", return_value=3):
        baseline = backend._workspace_pool_key(fleet)
        assert _bound_backend(group=group)[0]._workspace_pool_key(fleet) == baseline
        variants = (
            _bound_backend(group=object())[0]._workspace_pool_key(fleet),
            _bound_backend(group=group, rank=2)[0]._workspace_pool_key(fleet),
            _bound_backend(group=group, world_size=4)[0]._workspace_pool_key(fleet),
            _bound_backend(
                group=group,
                config=replace(
                    Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig(),
                    intermediate_size=1024,
                ),
            )[0]._workspace_pool_key(fleet),
            _bound_backend(
                group=group,
                config=replace(
                    Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig(),
                    top_k=4,
                ),
            )[0]._workspace_pool_key(fleet),
            backend._workspace_pool_key(
                SimpleNamespace(
                    num_experts=128,
                    max_tokens_per_rank=128,
                    token_hidden_size=7168,
                )
            ),
            backend._workspace_pool_key(
                SimpleNamespace(
                    num_experts=256,
                    max_tokens_per_rank=64,
                    token_hidden_size=7168,
                )
            ),
            backend._workspace_pool_key(
                SimpleNamespace(
                    num_experts=256,
                    max_tokens_per_rank=128,
                    token_hidden_size=4096,
                )
            ),
        )
        assert all(candidate != baseline for candidate in variants)

    with mock.patch("torch.cuda.current_device", return_value=4):
        assert backend._workspace_pool_key(fleet) != baseline


def test_workspace_pool_shares_session_and_refcounts_destroy(monkeypatch):
    from flashinfer.moe_ep.core.kernel import workspace_pool

    monkeypatch.setattr(workspace_pool, "_POOL", {})
    monkeypatch.setattr(workspace_pool, "_KEY_BY_ID", {})
    group = object()
    first, first_bootstrap = _bound_backend(group=group)
    second, second_bootstrap = _bound_backend(group=group)
    workspace = mock.MagicMock()
    first._allocate_workspace = mock.Mock(return_value=workspace)
    second._allocate_workspace = mock.Mock()

    with mock.patch("torch.cuda.current_device", return_value=3):
        first_workspace = first.prepare_workspace(first_bootstrap, _exact_fleet())
        second_workspace = second.prepare_workspace(second_bootstrap, _exact_fleet())

    assert first_workspace is second_workspace is workspace
    first._allocate_workspace.assert_called_once()
    second._allocate_workspace.assert_not_called()

    first.destroy(first_workspace)
    workspace.destroy.assert_not_called()
    second.destroy(second_workspace)
    workspace.destroy.assert_called_once_with()
