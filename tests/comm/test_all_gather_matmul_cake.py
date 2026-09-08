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


class _FakeDescriptorStorage:
    def __init__(self, device):
        self.device = device
        self.copies = []

    def copy_(self, source, *, non_blocking):
        self.copies.append((source, non_blocking))
        return self


def _mock_descriptor_allocations(monkeypatch, backend):
    allocations = []

    def allocate(*args, **kwargs):
        storage = _FakeDescriptorStorage(kwargs["device"])
        allocations.append((storage, args, kwargs))
        return storage

    monkeypatch.setattr(backend.torch, "empty", allocate)
    monkeypatch.setattr(backend.torch.cuda, "stream", lambda stream: nullcontext())
    return allocations


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
        "kernel_count": 12,
        "launch": backend._launch_contract(source),
        "constraints": backend._constraints_for_arch(arch),
        "kernel_symbols": list(backend._KERNEL_SYMBOLS),
        "route_coverage": copy.deepcopy(backend._ROUTE_COVERAGE),
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def _write_bundle(tmp_path: Path, backend, arch: str = "sm_100a"):
    source = (
        b"#define SMEM_TOTAL 197632\n" * 6
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
    assert "kPackedQkvExperimentSupported =\n    false;" in rendered
    assert "CAKE_MAIN_SMEM_BYTES" not in rendered
    assert "CAKE_PACKED_QKV_EXPERIMENT_SUPPORTED" not in rendered
    assert "main_cuda_stream != 0" not in rendered
    assert "comm_cuda_stream != 0 && bridge_cuda_event != 0" in rendered
    assert source_path.read_bytes().count(b"#define SMEM_TOTAL 197632") == 6


def test_host_descriptor_encoder_writes_cpu_staging_only(tmp_path, monkeypatch):
    backend = _backend()
    _write_bundle(tmp_path, backend)
    monkeypatch.setattr(backend, "_source_dir", lambda: tmp_path)
    _, manifest = backend._program_source("sm_100a")

    rendered = backend._render_host_source("test_module", manifest)

    assert "#include <cstring>" in rendered
    assert (
        'CheckCpuTensor(host_descriptor_storage, "host_descriptor_storage");'
        in rendered
    )
    assert '<< name << " must be a CPU tensor";' in rendered
    assert "host_descriptor_storage must have uint8 dtype" in rendered
    assert (
        "std::memcpy(host_descriptor_storage.data_ptr(), maps.data(), sizeof(maps));"
        in rendered
    )
    assert "cuMemcpyHtoD" not in rendered


def test_host_prepared_launcher_has_fixed_tp8_peer_abi(tmp_path, monkeypatch):
    backend = _backend()
    _write_bundle(tmp_path, backend, arch="sm_103a")
    monkeypatch.setattr(backend, "_source_dir", lambda: tmp_path)
    _, manifest = backend._program_source("sm_103a")

    rendered = backend._render_host_source("test_module", manifest)

    assert "TensorView peer_scratch_6, TensorView peer_signal_6" in rendered
    assert "std::array<const TensorView*, 7> peer_scratch" in rendered
    assert "std::array<const TensorView*, 7> peer_signal" in rendered
    assert "std::array<int64_t, 7> expected_peer_scratch" in rendered
    assert "std::array<int64_t, 7> expected_peer_signal" in rendered
    assert "world_size == 4 && weight.size(1) == 2560" in rendered
    assert "world_size == 8 && weight.size(1) == 1280" in rendered
    assert "int64_t ready_target, int64_t main_cuda_stream" in rendered
    assert "static_cast<uint32_t>(ready_target)" in rendered
    assert "cuMemsetD32Async" not in rendered


@pytest.mark.parametrize(
    "source",
    [
        b"#define SMEM_TOTAL 197632\n" * 5,
        b"#define SMEM_TOTAL 197632\n" * 5 + b"#define SMEM_TOTAL 196608\n",
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
    assert source.count(b"CakeTensorMap const*") == 18
    assert backend._resolved_main_smem_bytes(source) == 197632


def test_packaged_bf16_ws4_derives_private_packed_width_from_grid():
    backend = _backend()

    source_path, manifest = backend._program_source("sm_103a")
    source = source_path.read_text(encoding="utf-8")
    function = source.split("kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4", 1)[
        1
    ].split("} // extern", 1)[0]

    n_tiles = "(num_bids / (((M < 2432) ? M : 2432) / 128))"
    assert f"active_tiles = chunk_tiles_m * {n_tiles}" in function
    assert f"active_tiles_1 = chunk_tiles_m_1 * {n_tiles}" in function
    assert f"active_tiles_2 = chunk_tiles_m_2 * {n_tiles}" in function
    packed_width = "(num_bids / (((M < 2432) ? M : 2432) / 128) * 256)"
    assert f"(out_m + epi_tid) * {packed_width}" in function
    assert manifest["constraints"]["n_by_world_size"] == {
        "2": [2048],
        "4": [2048],
        "8": [2048],
    }
    assert manifest["constraints"]["prepared_packed_qkv"] == {
        "dtypes": ["bfloat16"],
        "n_by_world_size": {"4": [2560], "8": [1280]},
    }
    assert manifest["launch"]["main"]["grid_x"] == ("(min(M, 2432) / 128) * (N / 256)")
    rendered = backend._render_host_source("test_module", manifest)
    assert "kPackedQkvExperimentSupported =\n    true;" in rendered
    assert (
        "kPackedQkvExperimentSupported && world_size == 8 &&\n"
        "                      dtype_code == 0" in rendered
    )

    sm100_source, sm100_manifest = backend._program_source("sm_100a")
    assert sm100_source.read_bytes() == source_path.read_bytes()
    assert "prepared_packed_qkv" not in sm100_manifest["constraints"]


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
    preparations = []
    main_stream = object()
    publication_streams = []
    workspace = SimpleNamespace(
        scratch=scratch,
        descriptor_cache=OrderedDict(),
    )

    monkeypatch.setattr(backend, "_tensor_fingerprint", lambda tensor: id(tensor))

    def prepare(*args):
        preparations.append(args)

    allocations = _mock_descriptor_allocations(monkeypatch, backend)
    monkeypatch.setattr(
        backend.torch.cuda,
        "stream",
        lambda stream: (
            publication_streams.append(stream),
            nullcontext(),
        )[1],
    )
    module = SimpleNamespace(prepare_descriptors=prepare)
    descriptors = [
        backend._descriptor_storage(
            workspace,
            module,
            inp,
            weight,
            device_index=3,
            main_stream=main_stream,
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
        main_stream=main_stream,
        world_size=4,
        rows=128,
    )
    assert reused is descriptors[0]
    assert len(set(descriptors)) == len(preparations) == 80
    assert len(allocations) == 160
    assert len(workspace.descriptor_cache) == 80
    assert all(len(args) == 6 for args in preparations)
    assert all(args[3].device == "cpu" for args in preparations)
    assert all(args[3].copies == [] for args in preparations)
    assert publication_streams == [main_stream] * 80
    assert all(
        kwargs["device"] == "cpu" and kwargs["pin_memory"] is True
        for _, _, kwargs in allocations[::2]
    )
    assert all(kwargs["device"] == 3 for _, _, kwargs in allocations[1::2])
    assert all(
        descriptor.copies == [(preparation[3], True)]
        for descriptor, preparation in zip(descriptors, preparations, strict=True)
    )


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
    preparations = []
    main_stream = object()
    allocations = _mock_descriptor_allocations(monkeypatch, backend)
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
            main_stream=main_stream,
            world_size=2,
            rows=128,
        )
        for weight in weights[:2]
    ]
    assert (
        backend._descriptor_storage(
            workspace,
            module,
            inp,
            weights[0],
            device_index=0,
            main_stream=main_stream,
            world_size=2,
            rows=128,
        )
        is descriptors[0]
    )
    third = backend._descriptor_storage(
        workspace,
        module,
        inp,
        weights[2],
        device_index=0,
        main_stream=main_stream,
        world_size=2,
        rows=128,
    )

    first_key = ("input", "scratch", "weight-0")
    second_key = ("input", "scratch", "weight-1")
    third_key = ("input", "scratch", "weight-2")
    assert list(workspace.descriptor_cache) == [first_key, third_key]
    assert second_key not in workspace.descriptor_cache
    assert (
        backend._descriptor_storage(
            workspace,
            module,
            inp,
            weights[0],
            device_index=0,
            main_stream=main_stream,
            world_size=2,
            rows=128,
        )
        is descriptors[0]
    )
    assert third is not descriptors[0]
    assert len(allocations) == 6
    assert len(preparations) == 3


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
    preparations = []
    main_stream = object()
    allocations = _mock_descriptor_allocations(monkeypatch, backend)
    module = SimpleNamespace(
        prepare_descriptors=lambda *args: preparations.append(args)
    )
    descriptors = backend._descriptor_storage(
        workspace,
        module,
        first,
        weight,
        device_index=0,
        main_stream=main_stream,
        world_size=2,
        rows=128,
    )

    assert (
        backend._descriptor_storage(
            workspace,
            module,
            replacement,
            weight,
            device_index=0,
            main_stream=main_stream,
            world_size=2,
            rows=128,
        )
        is descriptors
    )
    assert len(allocations) == 2
    assert len(preparations) == 1


def test_descriptor_prepare_failure_does_not_install_cache_entry(monkeypatch):
    backend = _backend()
    workspace = SimpleNamespace(scratch="scratch", descriptor_cache=OrderedDict())
    monkeypatch.setattr(backend, "_tensor_fingerprint", lambda tensor: tensor)
    _mock_descriptor_allocations(monkeypatch, backend)

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
            main_stream=object(),
            world_size=2,
            rows=128,
        )

    assert workspace.descriptor_cache == OrderedDict()


def test_concurrent_descriptor_prepare_returns_one_cached_entry(monkeypatch):
    backend = _backend()
    workspace = SimpleNamespace(scratch="scratch", descriptor_cache=OrderedDict())
    monkeypatch.setattr(backend, "_tensor_fingerprint", lambda tensor: tensor)
    preparations = []
    call_barrier = Barrier(2, timeout=30)
    main_stream = object()

    def prepare(*args):
        preparations.append(args)

    allocations = _mock_descriptor_allocations(monkeypatch, backend)
    module = SimpleNamespace(prepare_descriptors=prepare)

    def resolve():
        call_barrier.wait()
        return backend._descriptor_storage(
            workspace,
            module,
            "input",
            "weight",
            device_index=0,
            main_stream=main_stream,
            world_size=2,
            rows=128,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: resolve(), range(2)))

    assert results[0] is results[1]
    assert len(workspace.descriptor_cache) == 1
    assert len(allocations) == 2 * len(preparations)
    assert len(preparations) in (1, 2)


def test_prepared_descriptor_cache_preserves_ready_stream(monkeypatch):
    backend = _backend()
    workspace = SimpleNamespace(
        scratch="scratch", prepared_descriptor_cache=OrderedDict()
    )
    monkeypatch.setattr(backend, "_tensor_fingerprint", lambda tensor: tensor)
    allocations = _mock_descriptor_allocations(monkeypatch, backend)
    preparations = []
    events = []

    class ReadyEvent:
        def __init__(self, **kwargs):
            self.recorded_streams = []
            events.append(self)

        def record(self, stream):
            self.recorded_streams.append(stream)

    monkeypatch.setattr(backend.torch.cuda, "Event", ReadyEvent)
    module = SimpleNamespace(
        prepare_descriptors=lambda *args: preparations.append(args)
    )
    first_stream = SimpleNamespace(cuda_stream=17)
    second_stream = SimpleNamespace(cuda_stream=19)
    kwargs = {
        "device_index": 0,
        "world_size": 4,
        "rows": 128,
        "scratch_fingerprint": "scratch",
        "weight_fingerprint": "weight",
    }

    first = backend._prepared_descriptor_storage(
        workspace, module, "input", "weight", main_stream=first_stream, **kwargs
    )
    second = backend._prepared_descriptor_storage(
        workspace, module, "input", "weight", main_stream=second_stream, **kwargs
    )

    assert second is first
    assert first.ready_stream == 17
    assert first.ready_event.recorded_streams == [first_stream]
    assert len(events) == len(preparations) == 1
    assert len(allocations) == 2


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
        shape=(8192, 2048),
        record_stream=lambda stream: weight_streams.append(stream),
    )
    group = object()
    lifecycle = []

    class FakeOutput:
        def __init__(self, pointer):
            self.pointer = pointer
            self.value = None

        def data_ptr(self):
            return self.pointer

    class FakeEvent:
        def record(self, stream):
            lifecycle.append("tail-record")

        def synchronize(self):
            lifecycle.append("tail-sync")

    initialization_event = object()

    class FakeStream:
        cuda_stream = 17

        def synchronize(self):
            lifecycle.append("current-sync")

        def wait_event(self, event):
            assert event is initialization_event
            lifecycle.append("initialization-wait")

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
    barriers = []

    def allocate(*shape, **kwargs):
        output = FakeOutput(1000 + len(outputs))
        outputs.append(output)
        return output

    def run_main(*args):
        output = args[3]
        output.value = len(launches) + 1
        launches.append((output, args[6], args[-1]))

    module = SimpleNamespace(
        run_barrier=lambda *args: barriers.append(args[-1]), run_main=run_main
    )
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

    def ensure_launch_state(state, **kwargs):
        if state.flags is None:
            lifecycle.append("launch-state")
            state.flags = object()
            state.flag_peers = object()
            state.initialization_event = initialization_event
            state.initialization_stream = 23

    monkeypatch.setattr(backend, "_ensure_launch_state", ensure_launch_state)
    monkeypatch.setattr(
        backend.symm_mem,
        "rendezvous",
        lambda *args, **kwargs: pytest.fail("input must not be rendezvoused"),
    )

    def create_workspace(**kwargs):
        lifecycle.append("workspace")
        key = (
            kwargs["device_index"],
            id(kwargs["group"]),
            kwargs["group_name"],
            kwargs["dtype"],
            kwargs["world_size"],
            kwargs["rows"],
        )
        backend._WORKSPACES[key] = workspace
        return workspace

    monkeypatch.setattr(backend, "_workspace", create_workspace)
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
    assert launches == [(first, 1, 17), (second, 2, 17)]
    assert barriers == [17, 17]
    assert input_streams == [main_stream, comm_stream, main_stream, comm_stream]
    assert weight_streams == [main_stream, main_stream]
    assert descriptor_streams == [main_stream, main_stream]
    assert lifecycle == [
        "current-sync",
        "launch-state",
        "workspace",
        "initialization-wait",
        "tail-record",
        "initialization-wait",
        "tail-record",
    ]


def test_ensure_launch_state_records_initialization_after_device_state(monkeypatch):
    backend = _backend()
    lifecycle = []

    class FakeFlags:
        def zero_(self):
            lifecycle.append("flags-zero")

    class FakePeer:
        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            lifecycle.append(f"peer-{self.pointer}")
            return self.pointer

    class FakeHandle:
        rank = 2
        world_size = 4

        def get_buffer(self, peer, shape, dtype, offset):
            assert shape == (2,)
            assert dtype == torch.uint32
            assert offset == 0
            return FakePeer(1000 + peer)

    class FakeEvent:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": False}
            lifecycle.append("event-create")

        def record(self, stream):
            lifecycle.append(("event-record", stream.cuda_stream))

    class FakeStream:
        cuda_stream = 29

    flags = FakeFlags()
    handle = FakeHandle()
    peer_tensor = object()
    monkeypatch.setattr(
        backend.symm_mem,
        "empty",
        lambda *args, **kwargs: (lifecycle.append("flags-allocate"), flags)[1],
    )
    monkeypatch.setattr(
        backend.symm_mem,
        "rendezvous",
        lambda value, *, group: (
            lifecycle.append(("rendezvous", value is flags, group)),
            handle,
        )[1],
    )
    monkeypatch.setattr(
        backend.torch,
        "tensor",
        lambda values, *, dtype, device: (
            lifecycle.append(("peer-upload", tuple(values), dtype, device)),
            peer_tensor,
        )[1],
    )
    monkeypatch.setattr(backend.torch.cuda, "Event", FakeEvent)

    state = backend._LaunchState()
    stream = FakeStream()
    backend._ensure_launch_state(
        state,
        device_index=3,
        rank=2,
        world_size=4,
        group_name="tp-group",
        main_stream=stream,
    )

    assert state.flags is flags
    assert state.flag_handle is handle
    assert state.flag_peers is peer_tensor
    assert state.initialization_stream == 29
    assert lifecycle == [
        "flags-allocate",
        ("rendezvous", True, "tp-group"),
        "flags-zero",
        "peer-1000",
        "peer-1001",
        "peer-1002",
        "peer-1003",
        ("peer-upload", (1000, 1001, 1002, 1003), torch.int64, 3),
        "event-create",
        ("event-record", 29),
    ]


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


def test_validate_inputs_accepts_exact_tp8_width(monkeypatch):
    backend = _backend()
    subgroup = SimpleNamespace(group_name="tp8-group")
    device = torch.device("cuda:7")
    inp = SimpleNamespace(
        device=device,
        dtype=torch.bfloat16,
        ndim=2,
        shape=(512, 8192),
        is_contiguous=lambda: True,
    )
    weight = SimpleNamespace(
        device=device,
        dtype=torch.bfloat16,
        ndim=2,
        shape=(8192, 2048),
        is_contiguous=lambda: True,
    )
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(backend.dist, "get_rank", lambda group: 7)
    monkeypatch.setattr(backend.symm_mem, "get_backend", lambda device: "NVSHMEM")
    monkeypatch.setattr(backend, "_target_arch", lambda device: "sm_103a")

    assert backend._validate_inputs(inp, weight, subgroup) == (
        7,
        7,
        8,
        "tp8-group",
    )


@pytest.mark.parametrize(
    ("world_size", "rank", "n"),
    [(4, 2, 2560), (8, 7, 1280)],
)
def test_validate_inputs_keeps_packed_qkv_routes_private(
    monkeypatch, world_size, rank, n
):
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
        shape=(8192, n),
        is_contiguous=lambda: True,
    )
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: world_size)
    monkeypatch.setattr(backend.dist, "get_rank", lambda group: rank)
    monkeypatch.setattr(backend.symm_mem, "get_backend", lambda device: "NVSHMEM")
    monkeypatch.setattr(backend, "_target_arch", lambda device: "sm_103a")

    with pytest.raises(ValueError, match=f"an N supported by world_size={world_size}"):
        backend._validate_inputs(inp, weight, subgroup)
    assert backend._validate_inputs(
        inp, weight, subgroup, packed_qkv_experiment=True
    ) == (3, rank, world_size, "tp-group")
    assert "_all_gather_matmul_cake_packed_qkv_sm103_tp4" not in backend.__all__
    assert "_prepare_all_gather_matmul_cake_packed_qkv_sm103" not in backend.__all__


@pytest.mark.parametrize(
    ("arch", "dtype", "world_size", "n", "message"),
    [
        ("sm_100a", torch.bfloat16, 4, 2560, "requires SM103 and bfloat16"),
        ("sm_103a", torch.float16, 4, 2560, "requires SM103 and bfloat16"),
        ("sm_103a", torch.bfloat16, 2, 2560, "requires exact K=8192"),
        ("sm_103a", torch.bfloat16, 4, 1280, "requires exact K=8192"),
        ("sm_103a", torch.bfloat16, 8, 2560, "requires exact K=8192"),
    ],
)
def test_validate_inputs_rejects_other_packed_qkv_routes(
    monkeypatch, arch, dtype, world_size, n, message
):
    backend = _backend()
    subgroup = SimpleNamespace(group_name="tp-group")
    device = torch.device("cuda:0")
    inp = SimpleNamespace(
        device=device,
        dtype=dtype,
        ndim=2,
        shape=(128, 8192),
        is_contiguous=lambda: True,
    )
    weight = SimpleNamespace(
        device=device,
        dtype=dtype,
        ndim=2,
        shape=(8192, n),
        is_contiguous=lambda: True,
    )
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: world_size)
    monkeypatch.setattr(backend.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(backend.symm_mem, "get_backend", lambda device: "NVSHMEM")
    monkeypatch.setattr(backend, "_target_arch", lambda device: arch)

    with pytest.raises(ValueError, match=message):
        backend._validate_inputs(inp, weight, subgroup, packed_qkv_experiment=True)


class _FakePreparedTensor:
    def __init__(self, pointer, shape, dtype, device, *, contiguous=True):
        self.pointer = pointer
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.ndim = len(shape)
        self.contiguous = contiguous
        self.recorded_streams = []

    def data_ptr(self):
        return self.pointer

    def stride(self):
        stride = 1
        result = []
        for dim in reversed(self.shape):
            result.append(stride)
            stride *= dim
        return tuple(reversed(result))

    def is_contiguous(self):
        return self.contiguous

    def record_stream(self, stream):
        self.recorded_streams.append(stream)

    def __getitem__(self, index):
        if isinstance(index, int):
            shape = self.shape[1:]
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.shape[0])
            shape = (len(range(start, stop, step)), *self.shape[1:])
        else:
            raise TypeError(f"unsupported fake tensor index {index!r}")
        return _FakePreparedTensor(
            self.pointer + 1,
            shape,
            self.dtype,
            self.device,
            contiguous=self.contiguous,
        )


def _fake_prepared_packed_qkv(
    monkeypatch,
    backend,
    *,
    world_size=4,
    rank=2,
    device_index=3,
    n=None,
    handle_world_size=None,
    bad_peer_scratch=False,
):
    backend._LAUNCH_STATES.clear()
    backend._WORKSPACES.clear()
    if n is None:
        n = 2560 if world_size == 4 else 1280
    if handle_world_size is None:
        handle_world_size = world_size
    device = torch.device("cuda", device_index)
    inp = _FakePreparedTensor(1001, (128, 8192), torch.bfloat16, device)
    weight = _FakePreparedTensor(2001, (8192, n), torch.bfloat16, device)
    scratch = _FakePreparedTensor(3001, (world_size, 128, 8192), torch.bfloat16, device)
    group = SimpleNamespace(group_name="tp-group")
    module = SimpleNamespace(name="bound-module")

    class FakeEvent:
        cuda_event = 29

        def __init__(self, **kwargs):
            self.recorded_streams = []

        def record(self, stream):
            self.recorded_streams.append(stream)

    class FakeStream:
        cuda_stream = 23

    initialization_event = FakeEvent()
    state = backend._LaunchState(
        flags=object(),
        flag_peers=object(),
        initialization_event=initialization_event,
        initialization_stream=0,
    )

    class FakeScratchHandle:
        def __init__(self):
            self.rank = rank
            self.world_size = handle_world_size
            self.calls = []

        def get_signal_pad(self, peer, shape, dtype, offset):
            self.calls.append(("signal", peer, tuple(shape), dtype, offset))
            return _FakePreparedTensor(4000 + peer, tuple(shape), dtype, device)

        def get_remote_tensor(self, peer, shape, dtype):
            self.calls.append(("remote", peer, tuple(shape), dtype))
            remote_shape = tuple(shape)
            if bad_peer_scratch:
                remote_shape = (remote_shape[0], 127, remote_shape[2])
            return _FakePreparedTensor(5000 + peer, remote_shape, dtype, device)

    scratch_handle = FakeScratchHandle()
    workspace = backend._Workspace(
        scratch=scratch,
        scratch_handle=scratch_handle,
        comm_stream=FakeStream(),
        bridge_event=FakeEvent(),
    )
    state_key = (device_index, id(group), "tp-group")
    workspace_key = (
        device_index,
        id(group),
        "tp-group",
        torch.bfloat16,
        world_size,
        128,
    )
    backend._LAUNCH_STATES[state_key] = state
    backend._WORKSPACES[workspace_key] = workspace
    validation_calls = []
    arch_calls = []
    module_calls = []
    descriptor_calls = []

    def validate(*args, **kwargs):
        validation_calls.append((args, kwargs))
        return device_index, rank, world_size, "tp-group"

    monkeypatch.setattr(backend, "_validate_inputs", validate)
    monkeypatch.setattr(
        backend,
        "_target_arch",
        lambda bound_device: (arch_calls.append(bound_device), "sm_103a")[1],
    )
    monkeypatch.setattr(
        backend,
        "_load_program",
        lambda arch: (module_calls.append(arch), module)[1],
    )
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda device_index: SimpleNamespace(cuda_stream=0),
    )
    monkeypatch.setattr(backend.torch.cuda, "Event", FakeEvent)
    descriptor = _FakePreparedTensor(
        6001,
        (backend._DESCRIPTOR_COUNT * backend._TENSOR_MAP_BYTES,),
        torch.uint8,
        device,
    )
    descriptor_entry = SimpleNamespace(
        descriptors=descriptor,
        ready_event=FakeEvent(),
        ready_stream=0,
    )
    monkeypatch.setattr(
        backend,
        "_prepared_descriptor_storage",
        lambda *args, **kwargs: (
            descriptor_calls.append((args, kwargs)),
            descriptor_entry,
        )[1],
    )
    launcher = backend._prepare_all_gather_matmul_cake_packed_qkv_sm103(
        inp, weight, group
    )
    calls = SimpleNamespace(
        validation=validation_calls,
        arch=arch_calls,
        module=module_calls,
        descriptor=descriptor_calls,
        descriptor_entry=descriptor_entry,
    )
    return launcher, inp, weight, group, state, workspace, module, calls


def test_prepare_packed_qkv_binds_host_identity_once(monkeypatch):
    backend = _backend()
    launcher, inp, weight, group, state, workspace, module, calls = (
        _fake_prepared_packed_qkv(monkeypatch, backend)
    )

    assert launcher.group is group
    assert launcher.group_id == id(group)
    assert launcher.group_name == "tp-group"
    assert launcher.rank == 2
    assert launcher.world_size == 4
    assert launcher.device_index == 3
    assert launcher.device == torch.device("cuda:3")
    assert launcher.arch == "sm_103a"
    assert launcher.dtype == torch.bfloat16
    assert launcher.rows == 128
    assert launcher.output_n == 2560
    assert launcher.module is module
    assert launcher.state is state
    assert launcher.workspace is workspace
    assert launcher.weight is weight
    assert launcher.weight_fingerprint == backend._tensor_fingerprint(weight)
    assert launcher.scratch_fingerprint == backend._tensor_fingerprint(
        workspace.scratch
    )
    assert launcher.chunk_size == 128
    assert launcher.num_chunks == 1
    assert launcher.chunk_plan == ((0, 128),)
    assert launcher.signal_pad.shape == (4, 1)
    assert launcher.signal_pad_ptr == 4002
    assert len(launcher.peer_routes) == 3
    assert launcher.peer_scratch_ptrs == (5004, 5001, 5002)
    assert launcher.peer_signal_ptrs == (4004, 4001, 4002)
    assert launcher.native_peer_args == (
        launcher.peer_routes[0][0],
        launcher.peer_routes[0][1],
        launcher.peer_routes[1][0],
        launcher.peer_routes[1][1],
        launcher.peer_routes[2][0],
        launcher.peer_routes[2][1],
        launcher.peer_routes[2][0],
        launcher.peer_routes[2][1],
        launcher.peer_routes[2][0],
        launcher.peer_routes[2][1],
        launcher.peer_routes[2][0],
        launcher.peer_routes[2][1],
        launcher.peer_routes[2][0],
        launcher.peer_routes[2][1],
    )
    assert launcher.native_expected_peer_args == (
        5004,
        4004,
        5001,
        4001,
        5002,
        4002,
        5002,
        4002,
        5002,
        4002,
        5002,
        4002,
        5002,
        4002,
    )
    assert [call[:2] for call in workspace.scratch_handle.calls] == [
        ("signal", 2),
        ("remote", 3),
        ("signal", 3),
        ("remote", 0),
        ("signal", 0),
        ("remote", 1),
        ("signal", 1),
    ]
    assert len(calls.validation) == 1
    assert calls.validation[0][0] == (inp, weight, group)
    assert calls.validation[0][1] == {"packed_qkv_experiment": True}
    assert calls.arch == [torch.device("cuda:3")]
    assert calls.module == ["sm_103a"]
    assert len(calls.descriptor) == 1
    assert "_prepare_all_gather_matmul_cake_packed_qkv_sm103" not in backend.__all__
    with pytest.raises(AttributeError):
        launcher.peer_routes = ()


def test_prepare_packed_qkv_tp8_binds_and_submits_all_seven_peers(monkeypatch):
    backend = _backend()
    launcher, inp, weight, _, state, workspace, module, _ = _fake_prepared_packed_qkv(
        monkeypatch,
        backend,
        world_size=8,
        rank=7,
        device_index=7,
    )

    assert launcher.world_size == 8
    assert launcher.rank == 7
    assert launcher.device == torch.device("cuda:7")
    assert launcher.output_n == 1280
    assert launcher.signal_pad.shape == (8, 1)
    assert launcher.signal_pad_ptr == 4007
    assert len(launcher.peer_routes) == 7
    assert launcher.peer_scratch_ptrs == tuple(range(5001, 5008))
    assert launcher.peer_signal_ptrs == tuple(range(4001, 4008))
    assert len(launcher.native_peer_args) == 14
    assert launcher.native_expected_peer_args == tuple(
        pointer
        for pair in zip(
            launcher.peer_scratch_ptrs,
            launcher.peer_signal_ptrs,
            strict=True,
        )
        for pointer in pair
    )
    assert [call[:2] for call in workspace.scratch_handle.calls] == [
        ("signal", 7),
        ("remote", 0),
        ("signal", 0),
        ("remote", 1),
        ("signal", 1),
        ("remote", 2),
        ("signal", 2),
        ("remote", 3),
        ("signal", 3),
        ("remote", 4),
        ("signal", 4),
        ("remote", 5),
        ("signal", 5),
        ("remote", 6),
        ("signal", 6),
    ]

    output = _FakePreparedTensor(
        7001,
        (launcher.world_size * launcher.rows, launcher.output_n),
        torch.bfloat16,
        launcher.device,
    )
    allocations = []
    monkeypatch.setattr(
        backend.torch,
        "empty",
        lambda *args, **kwargs: (allocations.append((args, kwargs)), output)[1],
    )
    submissions = []
    module.run_prepared_packed_qkv = lambda *args: submissions.append(args)

    assert launcher(inp) is output
    assert allocations == [((8 * 128, 1280), {"dtype": torch.bfloat16, "device": 7})]
    assert len(submissions) == 1
    submission = submissions[0]
    assert submission[7:21] == launcher.native_peer_args
    assert submission[21:31] == (8, 7, 128, 0, 1, 0, 23, 29, 3001, 4007)
    assert submission[31:] == launcher.native_expected_peer_args
    assert state.next_phase == 1
    assert state.ready_epoch == 1

    assert launcher(inp) is output
    assert allocations == [
        ((8 * 128, 1280), {"dtype": torch.bfloat16, "device": 7}),
        ((8 * 128, 1280), {"dtype": torch.bfloat16, "device": 7}),
    ]
    assert len(submissions) == 2
    second_submission = submissions[1]
    assert second_submission[7:21] == launcher.native_peer_args
    assert second_submission[21:31] == (
        8,
        7,
        128,
        1,
        2,
        0,
        23,
        29,
        3001,
        4007,
    )
    assert second_submission[31:] == launcher.native_expected_peer_args
    assert state.next_phase == 0
    assert state.ready_epoch == 2


def test_prepared_packed_qkv_hot_path_uses_one_native_submission(monkeypatch):
    backend = _backend()
    launcher, inp, weight, _, state, workspace, module, _ = _fake_prepared_packed_qkv(
        monkeypatch, backend
    )
    output = _FakePreparedTensor(
        7001,
        (launcher.world_size * launcher.rows, launcher.output_n),
        torch.bfloat16,
        launcher.device,
    )
    monkeypatch.setattr(backend.torch, "empty", lambda *args, **kwargs: output)
    submissions = []
    module.run_prepared_packed_qkv = lambda *args: submissions.append(args)

    class TailEvent:
        def __init__(self, **kwargs):
            self.recorded_streams = []

        def record(self, stream):
            self.recorded_streams.append(stream)

    monkeypatch.setattr(backend.torch.cuda, "Event", TailEvent)

    result = launcher(inp)

    assert result is output
    assert len(submissions) == 1
    submission = submissions[0]
    assert submission[0] is inp
    assert submission[1] is workspace.scratch
    assert submission[2] is weight
    assert submission[3] is output
    descriptor = submission[4]
    assert descriptor.data_ptr() == 6001
    assert submission[5] is launcher.signal_pad
    assert submission[6] is state.flag_peers
    assert submission[7:21] == launcher.native_peer_args
    assert submission[21:] == (
        4,
        2,
        128,
        0,
        1,
        0,
        23,
        29,
        3001,
        4002,
        *launcher.native_expected_peer_args,
    )
    assert state.next_phase == 1
    assert state.ready_epoch == 1
    assert state.tail_stream == 0
    assert state.tail_event.recorded_streams
    main_stream = descriptor.recorded_streams[0]
    assert inp.recorded_streams == [main_stream, workspace.comm_stream]
    assert weight.recorded_streams == [main_stream]
    assert descriptor.recorded_streams == [main_stream]


def test_prepared_packed_qkv_first_other_stream_waits_for_initialization_and_descriptor(
    monkeypatch,
):
    backend = _backend()
    launcher, inp, _, _, state, _, module, calls = _fake_prepared_packed_qkv(
        monkeypatch, backend
    )
    output = _FakePreparedTensor(
        7001,
        (launcher.world_size * launcher.rows, launcher.output_n),
        torch.bfloat16,
        launcher.device,
    )
    monkeypatch.setattr(backend.torch, "empty", lambda *args, **kwargs: output)
    module.run_prepared_packed_qkv = lambda *args: None

    class FirstCallStream:
        cuda_stream = 19

        def __init__(self):
            self.waited_events = []

        def wait_event(self, event):
            self.waited_events.append(event)

    class TailEvent:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            pass

    first_call_stream = FirstCallStream()
    monkeypatch.setattr(
        backend.torch.cuda, "current_stream", lambda device_index: first_call_stream
    )
    monkeypatch.setattr(backend.torch.cuda, "Event", TailEvent)

    launcher(inp)

    assert first_call_stream.waited_events == [
        state.initialization_event,
        calls.descriptor_entry.ready_event,
    ]
    assert state.tail_stream == 19


def test_prepared_packed_qkv_first_call_waits_for_descriptor_and_tail(monkeypatch):
    backend = _backend()
    launcher, inp, _, _, state, _, module, calls = _fake_prepared_packed_qkv(
        monkeypatch, backend
    )
    output = _FakePreparedTensor(
        7001,
        (launcher.world_size * launcher.rows, launcher.output_n),
        torch.bfloat16,
        launcher.device,
    )
    monkeypatch.setattr(backend.torch, "empty", lambda *args, **kwargs: output)
    module.run_prepared_packed_qkv = lambda *args: None

    class FirstCallStream:
        cuda_stream = 19

        def __init__(self):
            self.waited_events = []

        def wait_event(self, event):
            self.waited_events.append(event)

    class PriorTail:
        def record(self, stream):
            pass

    first_call_stream = FirstCallStream()
    prior_tail = PriorTail()
    state.tail_event = prior_tail
    state.tail_stream = 23
    monkeypatch.setattr(
        backend.torch.cuda, "current_stream", lambda device_index: first_call_stream
    )

    launcher(inp)

    assert first_call_stream.waited_events == [
        state.initialization_event,
        calls.descriptor_entry.ready_event,
        prior_tail,
    ]
    assert state.tail_stream == 19


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda inp: setattr(inp, "device", torch.device("cuda:2")),
            "inp device changed",
        ),
        (
            lambda inp: setattr(inp, "dtype", torch.float16),
            "inp dtype changed",
        ),
        (
            lambda inp: setattr(inp, "shape", (256, 8192)),
            "inp must have shape",
        ),
        (
            lambda inp: setattr(inp, "contiguous", False),
            "inp must be contiguous",
        ),
    ],
)
def test_prepared_packed_qkv_hot_input_misuse_fails_closed(
    monkeypatch, mutation, message
):
    backend = _backend()
    launcher, inp, *_ = _fake_prepared_packed_qkv(monkeypatch, backend)
    mutation(inp)

    with pytest.raises(ValueError, match=message):
        launcher._validate_hot_input(inp)


def test_prepared_packed_qkv_bound_weight_drift_fails_closed(monkeypatch):
    backend = _backend()
    launcher, inp, weight, *_ = _fake_prepared_packed_qkv(monkeypatch, backend)
    weight.pointer += 1

    with pytest.raises(RuntimeError, match="bound weight contract changed"):
        launcher._validate_hot_input(inp)


def test_prepared_packed_qkv_bound_group_drift_fails_closed(monkeypatch):
    backend = _backend()
    launcher, inp, *_ = _fake_prepared_packed_qkv(monkeypatch, backend)
    object.__setattr__(launcher, "group", object())

    with pytest.raises(RuntimeError, match="group identity changed"):
        launcher._validate_hot_input(inp)


def test_prepared_packed_qkv_poisoned_state_fails_before_hot_submission(monkeypatch):
    backend = _backend()
    launcher, inp, _, _, state, *_ = _fake_prepared_packed_qkv(monkeypatch, backend)
    state.poisoned = True

    with pytest.raises(RuntimeError, match="state is poisoned"):
        launcher(inp)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"handle_world_size": 3}, "workspace topology does not match group"),
        ({"bad_peer_scratch": True}, "peer 3 scratch rank slice shape"),
    ],
)
def test_prepare_packed_qkv_prebound_views_and_topology_fail_closed(
    monkeypatch, kwargs, message
):
    backend = _backend()

    with pytest.raises(RuntimeError, match=message):
        _fake_prepared_packed_qkv(monkeypatch, backend, **kwargs)

    assert len(backend._LAUNCH_STATES) == 1
    assert next(iter(backend._LAUNCH_STATES.values())).poisoned


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

    with pytest.raises(ValueError, match="world size 2, 4, or 8"):
        backend._validate_inputs(inp, weight, subgroup)
