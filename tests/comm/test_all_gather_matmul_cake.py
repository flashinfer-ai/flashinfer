import copy
import hashlib
import importlib
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch


def _backend():
    return importlib.import_module(
        "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul"
    )


def test_backend_entrypoint_rejects_non_cake_route():
    backend = _backend()

    with pytest.raises(ValueError, match="exactly 'cake'"):
        backend.all_gather_matmul_cake(object(), object(), object(), backend="auto")


def _manifest(backend, source: bytes, arch: str):
    return {
        "schema_version": 1,
        "arch": arch,
        "compile_flags": ["--use_fast_math"],
        "tma_abi": "pointer",
        "kernel_count": 8,
        "launch": backend._launch_contract(source),
        "constraints": copy.deepcopy(backend._CONSTRAINTS),
        "kernel_symbols": list(backend._KERNEL_SYMBOLS),
        "route_coverage": copy.deepcopy(backend._ROUTE_COVERAGE),
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def _write_bundle(tmp_path: Path, backend, arch: str = "sm_100a"):
    source = (
        b"#define SMEM_TOTAL 197632\n" * 4
        + b'extern "C" __global__ void generated() {}\n'
    )
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
    assert manifest["launch"]["main"]["dynamic_smem_bytes"] == 197632


def test_host_launch_consumes_manifest_smem(tmp_path, monkeypatch):
    backend = _backend()
    source_path, _ = _write_bundle(tmp_path, backend)
    monkeypatch.setattr(backend, "_source_dir", lambda: tmp_path)
    _, manifest = backend._program_source("sm_100a")

    rendered = backend._render_host_source("test_module", manifest)

    assert "constexpr int32_t kMainSmemBytes = 197632;" in rendered
    assert "CAKE_MAIN_SMEM_BYTES" not in rendered
    assert source_path.read_bytes().count(b"#define SMEM_TOTAL 197632") == 4


@pytest.mark.parametrize(
    "source",
    [
        b"#define SMEM_TOTAL 197632\n" * 3,
        b"#define SMEM_TOTAL 197632\n" * 3 + b"#define SMEM_TOTAL 196608\n",
    ],
)
def test_source_launch_contract_rejects_missing_or_divergent_main_smem(source):
    backend = _backend()

    with pytest.raises(RuntimeError, match="one uniform SMEM_TOTAL"):
        backend._launch_contract(source)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_packaged_program_has_self_contained_pointer_abi(arch):
    backend = _backend()

    source_path, _ = backend._program_source(arch)
    source = source_path.read_bytes()

    assert (
        source.count(b"struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };")
        == 1
    )
    assert source.count(b"CakeTensorMap const*") == 12
    assert backend._resolved_main_smem_bytes(source) == 197632


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.update(schema_version=2),
        lambda manifest: manifest.update(compile_flags=[]),
        lambda manifest: manifest.update(tma_abi="grid_constant"),
        lambda manifest: manifest["kernel_symbols"].reverse(),
        lambda manifest: manifest["constraints"].update(world_sizes=[2]),
        lambda manifest: manifest["launch"]["main"].update(block_threads=128),
        lambda manifest: manifest["launch"]["main"].update(dynamic_smem_bytes=196608),
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


def test_descriptor_cache_reuses_each_layer_mapping(monkeypatch):
    backend = _backend()

    class WeakTensor:
        pass

    inp = WeakTensor()
    scratch = object()
    weights = [WeakTensor() for _ in range(80)]
    allocations = []
    preparations = []
    workspace = SimpleNamespace(
        scratch=scratch,
        descriptor_cache=OrderedDict(),
    )

    monkeypatch.setattr(backend, "_tensor_fingerprint", lambda tensor: id(tensor))

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
    assert all(len(args) == 6 for args in preparations)


def test_descriptor_cache_is_bounded_lru(monkeypatch):
    backend = _backend()

    class WeakTensor:
        def __init__(self, fingerprint):
            self.fingerprint = fingerprint

    inp = WeakTensor("input")
    weights = [WeakTensor(f"weight-{index}") for index in range(3)]
    workspace = SimpleNamespace(
        scratch=WeakTensor("scratch"),
        descriptor_cache=OrderedDict(),
    )
    monkeypatch.setattr(
        backend, "_tensor_fingerprint", lambda tensor: tensor.fingerprint
    )
    allocations = []
    preparations = []

    def allocate(*args, **kwargs):
        descriptor = object()
        allocations.append(descriptor)
        return descriptor

    monkeypatch.setattr(backend.torch, "empty", allocate)
    monkeypatch.setattr(backend, "_DESCRIPTOR_CACHE_MAX_ENTRIES", 2)
    module = SimpleNamespace(
        prepare_descriptors=lambda *args: preparations.append(args)
    )
    descriptors = [
        backend._descriptor_storage(
            workspace,
            module,
            inp,
            weight,
            device_index=0,
            world_size=2,
            rows=128,
        )
        for weight in weights[:2]
    ]
    assert backend._descriptor_storage(
        workspace,
        module,
        inp,
        weights[0],
        device_index=0,
        world_size=2,
        rows=128,
    ) is descriptors[0]
    third = backend._descriptor_storage(
        workspace,
        module,
        inp,
        weights[2],
        device_index=0,
        world_size=2,
        rows=128,
    )

    first_key = ("input", "scratch", "weight-0")
    second_key = ("input", "scratch", "weight-1")
    third_key = ("input", "scratch", "weight-2")
    assert list(workspace.descriptor_cache) == [first_key, third_key]
    assert second_key not in workspace.descriptor_cache
    assert backend._descriptor_storage(
        workspace,
        module,
        inp,
        weights[0],
        device_index=0,
        world_size=2,
        rows=128,
    ) is descriptors[0]
    assert third is not descriptors[0]
    assert len(allocations) == len(preparations) == 3


def test_descriptor_cache_reuses_recycled_fingerprint(monkeypatch):
    backend = _backend()

    class TensorFingerprint:
        def __init__(self, fingerprint):
            self.fingerprint = fingerprint

    first = TensorFingerprint("input")
    replacement = TensorFingerprint("input")
    weight = TensorFingerprint("weight")
    workspace = SimpleNamespace(
        scratch=TensorFingerprint("scratch"),
        descriptor_cache=OrderedDict(),
    )
    monkeypatch.setattr(
        backend, "_tensor_fingerprint", lambda tensor: tensor.fingerprint
    )
    allocations = []
    preparations = []

    def allocate(*args, **kwargs):
        descriptor = object()
        allocations.append(descriptor)
        return descriptor

    monkeypatch.setattr(backend.torch, "empty", allocate)
    module = SimpleNamespace(
        prepare_descriptors=lambda *args: preparations.append(args)
    )
    descriptors = backend._descriptor_storage(
        workspace,
        module,
        first,
        weight,
        device_index=0,
        world_size=2,
        rows=128,
    )

    assert backend._descriptor_storage(
        workspace,
        module,
        replacement,
        weight,
        device_index=0,
        world_size=2,
        rows=128,
    ) is descriptors
    assert len(allocations) == len(preparations) == 1


def test_descriptor_prepare_failure_does_not_install_cache_entry(monkeypatch):
    backend = _backend()
    workspace = SimpleNamespace(scratch="scratch", descriptor_cache=OrderedDict())
    monkeypatch.setattr(backend, "_tensor_fingerprint", lambda tensor: tensor)
    monkeypatch.setattr(backend.torch, "empty", lambda *args, **kwargs: object())

    def prepare(*args):
        raise RuntimeError("prepare failed")

    module = SimpleNamespace(prepare_descriptors=prepare)

    with pytest.raises(RuntimeError, match="prepare failed"):
        backend._descriptor_storage(
            workspace,
            module,
            "input",
            "weight",
            device_index=0,
            world_size=2,
            rows=128,
        )

    assert workspace.descriptor_cache == OrderedDict()


def test_concurrent_descriptor_prepare_returns_one_cached_entry(monkeypatch):
    backend = _backend()
    workspace = SimpleNamespace(scratch="scratch", descriptor_cache=OrderedDict())
    monkeypatch.setattr(backend, "_tensor_fingerprint", lambda tensor: tensor)
    allocations = []
    preparations = []
    call_barrier = Barrier(2, timeout=30)

    def allocate(*args, **kwargs):
        descriptor = object()
        allocations.append(descriptor)
        return descriptor

    def prepare(*args):
        preparations.append(args)

    monkeypatch.setattr(backend.torch, "empty", allocate)
    module = SimpleNamespace(prepare_descriptors=prepare)

    def resolve():
        call_barrier.wait()
        return backend._descriptor_storage(
            workspace,
            module,
            "input",
            "weight",
            device_index=0,
            world_size=2,
            rows=128,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: resolve(), range(2)))

    assert results[0] is results[1]
    assert len(workspace.descriptor_cache) == 1
    assert len(allocations) == len(preparations)
    assert len(preparations) in (1, 2)


def test_consecutive_calls_return_fresh_outputs_without_overwriting(monkeypatch):
    backend = _backend()
    backend._LAUNCH_STATES.clear()
    backend._WORKSPACES.clear()
    input_streams = []
    weight_streams = []
    descriptor_streams = []
    inp = SimpleNamespace(
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        shape=(128, 8192),
        record_stream=lambda stream: input_streams.append(stream),
    )
    weight = SimpleNamespace(
        record_stream=lambda stream: weight_streams.append(stream),
    )
    group = object()

    class FakeOutput:
        def __init__(self, pointer):
            self.pointer = pointer
            self.value = None

        def data_ptr(self):
            return self.pointer

    class FakeEvent:
        def record(self, stream):
            pass

    class FakeStream:
        cuda_stream = 17

        def wait_event(self, event):
            pass

        def wait_stream(self, stream):
            pass

    class FakeSignal:
        def zero_(self):
            pass

    class FakeScratchHandle:
        def get_signal_pad(self, *args):
            return FakeSignal()

    outputs = []
    launches = []

    def allocate(*shape, **kwargs):
        output = FakeOutput(1000 + len(outputs))
        outputs.append(output)
        return output

    def run_main(*args):
        output = args[3]
        output.value = len(launches) + 1
        launches.append(output)

    module = SimpleNamespace(run_barrier=lambda *args: None, run_main=run_main)
    main_stream = FakeStream()
    comm_stream = FakeStream()
    descriptors = SimpleNamespace(
        record_stream=lambda stream: descriptor_streams.append(stream)
    )
    workspace = SimpleNamespace(
        scratch=SimpleNamespace(shape=(1, 128, 8192), dtype=torch.bfloat16),
        scratch_handle=FakeScratchHandle(),
        comm_stream=comm_stream,
        descriptor_cache=OrderedDict(),
    )
    monkeypatch.setattr(
        backend, "_validate_inputs", lambda *args: (0, 0, 1, "tp-group")
    )
    monkeypatch.setattr(backend, "_target_arch", lambda device: "sm_100a")
    monkeypatch.setattr(backend, "_load_program", lambda arch: module)
    monkeypatch.setattr(
        backend,
        "_ensure_launch_state",
        lambda state, **kwargs: setattr(state, "flag_peers", object()),
    )
    monkeypatch.setattr(
        backend.symm_mem,
        "rendezvous",
        lambda *args, **kwargs: pytest.fail("input must not be rendezvoused"),
    )
    monkeypatch.setattr(backend, "_workspace", lambda **kwargs: workspace)
    monkeypatch.setattr(
        backend,
        "_descriptor_storage",
        lambda *args, **kwargs: descriptors,
    )
    monkeypatch.setattr(backend.torch, "empty", allocate)
    monkeypatch.setattr(
        backend.torch.cuda, "current_stream", lambda device: main_stream
    )
    monkeypatch.setattr(backend.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(backend.torch.cuda, "Event", lambda **kwargs: FakeEvent())

    first = backend.all_gather_matmul_cake(inp, weight, group, backend="cake")
    second = backend.all_gather_matmul_cake(inp, weight, group, backend="cake")

    assert first.data_ptr() != second.data_ptr()
    assert first.value == 1
    assert second.value == 2
    assert launches == [first, second]
    assert input_streams == [main_stream, comm_stream, main_stream, comm_stream]
    assert weight_streams == [main_stream, main_stream]
    assert descriptor_streams == [main_stream, main_stream]


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
