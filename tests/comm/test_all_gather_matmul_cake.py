import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _backend():
    return importlib.import_module(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul_cake"
    )


def _manifest(backend, source: bytes, arch: str):
    return {
        "schema_version": 1,
        "arch": arch,
        "compile_flags": ["--use_fast_math"],
        "tma_abi": "pointer",
        "kernel_count": 8,
        "launch": copy.deepcopy(backend._LAUNCH_CONTRACT),
        "constraints": copy.deepcopy(backend._CONSTRAINTS),
        "kernel_symbols": list(backend._KERNEL_SYMBOLS),
        "route_coverage": copy.deepcopy(backend._ROUTE_COVERAGE),
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def _write_bundle(tmp_path: Path, backend, arch: str = "sm_100a"):
    source = b'extern "C" __global__ void generated() {}\n'
    directory = tmp_path / arch.replace("sm_", "sm")
    directory.mkdir(parents=True)
    source_path = directory / "cake_all_gather_matmul_kernels.cu"
    source_path.write_bytes(source)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(backend, source, arch)), encoding="utf-8"
    )
    return source_path, manifest_path


def test_program_source_accepts_exact_ordered_manifest(tmp_path, monkeypatch):
    backend = _backend()
    source_path, _ = _write_bundle(tmp_path, backend)
    monkeypatch.setattr(backend, "_source_dir", lambda: tmp_path)

    actual_source, manifest = backend._program_source("sm_100a")

    assert actual_source == source_path
    assert manifest["kernel_symbols"] == list(backend._KERNEL_SYMBOLS)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.update(schema_version=2),
        lambda manifest: manifest.update(compile_flags=[]),
        lambda manifest: manifest.update(tma_abi="grid_constant"),
        lambda manifest: manifest["kernel_symbols"].reverse(),
        lambda manifest: manifest["constraints"].update(world_sizes=[2]),
        lambda manifest: manifest["launch"]["main"].update(block_threads=128),
        lambda manifest: manifest["route_coverage"]["ws4"]["main"].update(
            bfloat16=manifest["kernel_symbols"][5]
        ),
        lambda manifest: manifest.update(source_sha256="0" * 64),
        lambda manifest: manifest.update(unexpected=True),
    ],
)
def test_program_source_rejects_manifest_identity_drift(
    tmp_path, monkeypatch, mutation
):
    backend = _backend()
    _, manifest_path = _write_bundle(tmp_path, backend)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(document)
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(backend, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="manifest identity is invalid"):
        backend._program_source("sm_100a")


def test_descriptor_cache_keeps_each_layer_mapping_alive(monkeypatch):
    backend = _backend()
    inp = object()
    scratch = object()
    output = object()
    weights = [object() for _ in range(80)]
    allocations = []
    preparations = []
    workspace = SimpleNamespace(
        scratch=scratch,
        output=output,
        descriptor_cache={},
    )

    monkeypatch.setattr(backend, "_tensor_fingerprint", lambda tensor: tensor)

    def allocate(*args, **kwargs):
        storage = object()
        allocations.append((storage, args, kwargs))
        return storage

    def prepare(*args):
        preparations.append(args)

    monkeypatch.setattr(backend.torch, "empty", allocate)
    module = SimpleNamespace(prepare_descriptors=prepare)
    descriptors = [
        backend._descriptor_storage(
            workspace,
            module,
            inp,
            weight,
            device_index=3,
            world_size=4,
            rows=128,
        )
        for weight in weights
    ]

    reused = backend._descriptor_storage(
        workspace,
        module,
        inp,
        weights[0],
        device_index=3,
        world_size=4,
        rows=128,
    )
    assert reused is descriptors[0]
    assert len(set(descriptors)) == len(allocations) == len(preparations) == 80
    assert len(workspace.descriptor_cache) == 80


def test_validate_inputs_uses_the_exact_passed_subgroup(monkeypatch):
    backend = _backend()
    subgroup = SimpleNamespace(group_name="tp-group")
    device = torch.device("cuda:3")
    inp = SimpleNamespace(
        device=device,
        dtype=torch.bfloat16,
        ndim=2,
        shape=(128, 8192),
        is_contiguous=lambda: True,
    )
    weight = SimpleNamespace(
        device=device,
        dtype=torch.bfloat16,
        ndim=2,
        shape=(8192, 2048),
        is_contiguous=lambda: True,
    )
    seen_groups = []
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: "nccl")

    def get_world_size(group):
        seen_groups.append(group)
        return 4

    def get_rank(group):
        seen_groups.append(group)
        return 2

    monkeypatch.setattr(backend.dist, "get_world_size", get_world_size)
    monkeypatch.setattr(backend.dist, "get_rank", get_rank)
    monkeypatch.setattr(backend.symm_mem, "get_backend", lambda device: "NVSHMEM")
    monkeypatch.setattr(backend, "_target_arch", lambda device: "sm_100a")

    assert backend._validate_inputs(inp, weight, subgroup) == (
        3,
        2,
        4,
        "tp-group",
    )
    assert seen_groups == [subgroup, subgroup]


def test_validate_inputs_rejects_unsupported_subgroup_size(monkeypatch):
    backend = _backend()
    subgroup = SimpleNamespace(group_name="tp-group")
    device = torch.device("cuda:0")
    inp = SimpleNamespace(
        device=device,
        dtype=torch.float16,
        ndim=2,
        shape=(128, 8192),
        is_contiguous=lambda: True,
    )
    weight = SimpleNamespace(
        device=device,
        dtype=torch.float16,
        ndim=2,
        shape=(8192, 2048),
        is_contiguous=lambda: True,
    )
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: 3)
    monkeypatch.setattr(backend.dist, "get_rank", lambda group: 0)

    with pytest.raises(ValueError, match="world size 2 or 4"):
        backend._validate_inputs(inp, weight, subgroup)
