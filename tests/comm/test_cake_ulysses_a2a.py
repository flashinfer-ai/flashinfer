"""The Cake Ulysses all-to-all package: its portfolio through the public communicator, and the package route.

The backend assertions import the package only through :func:`tests.test_helpers.cake_package.cake_package`
(flashinfer-ai/flashinfer#6553); the multi-GPU portfolio runs through the public communicator so the public
contract (caller-owned outputs, NCCL-equal results) is what is checked.
"""

import re
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.comm.cake_ulysses_a2a_fixture import (
    NVLINK_KERNELS,
    cake_route_available,
    prepare,
    shapes,
)
from tests.comm.test_ulysses_a2a import multi_process_parallel
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_ulysses")
cake_build = cake_package("cake_ulysses", "cake_build")
loader = cake_package("cake_ulysses", "cake_jit")
backend = cake_package("cake_ulysses", "cake_backend")


# ---- the portfolio through the public communicator ---------------------------------


def _correctness_worker(world_size, rank, port, nvlink_kernel):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        for shape in shapes(world_size):
            case = prepare(shape, backend="nvlink", nvlink_kernel=nvlink_kernel)
            try:
                assert case.nvlink_kernel == nvlink_kernel
                case.check()
            finally:
                case.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("nvlink_kernel", NVLINK_KERNELS)
@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_generated_ulysses_portfolio(world_size, nvlink_kernel):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if nvlink_kernel == "cake":
        if (
            torch.cuda.get_device_capability(0)
            not in pkg.SUPPORTED_COMPUTE_CAPABILITIES
        ):
            pytest.skip("the Cake portfolio targets Blackwell SM100/SM103")
        if not cake_route_available(world_size):
            pytest.skip(
                "build.json delivers no library for this device yet (the export run fills it)"
            )
    multi_process_parallel(world_size, _correctness_worker, (nvlink_kernel,))


# ---- the package route (CPU) -------------------------------------------------------


@pytest.mark.parametrize(
    "capability,arch",
    [
        ((10, 0), "sm_100a"),
        ((10, 3), "sm_103a"),
        ((9, 0), None),
        ((12, 0), None),
    ],
)
def test_arch_for_capability_names_the_exported_targets(capability, arch):
    assert loader.arch_for_capability(capability) == arch
    assert pkg.arch_for_capability(capability) == arch
    if arch is None:
        assert not loader.route_applies(
            device_capability=capability, dtype="bf16", world_size=2
        )
    else:
        assert arch in loader.ARCHES
        assert capability in pkg.SUPPORTED_COMPUTE_CAPABILITIES


def test_records_cover_the_portfolio_once():
    """One record per (dtype, world size, mode), each carrying the host plan of its collective."""
    expected = {
        loader.record_name(dtype, world, mode)
        for dtype in loader.DTYPES
        for world in loader.WORLD_SIZES
        for mode in loader.MODES
    }
    assert set(loader.MODULES) == expected
    assert len(loader.MODULES) == 24
    libraries = set()
    for name, record in loader.MODULES.items():
        direction = "scatter" if record["mode"] == 0 else "gather"
        assert name == f"{direction}_{record['dtype']}_w{record['world_size']}"
        assert (
            loader.record_for(record["dtype"], record["world_size"], record["mode"])
            is record
        )
        assert record["library"] == record["kernel"]
        assert re.fullmatch(r"cake_ulysses_a2a_[0-9a-f]{20}", record["library"])
        assert record["kernel_symbol"] == f"kernel_{record['library']}"
        libraries.add(record["library"])
        assert list(record["arches"]) == list(loader.ARCHES)
        assert record["ffi_entry"] == "run"
        launch = record["launch"]
        assert tuple(launch["block"]) == (512, 1, 1)
        assert tuple(launch["cluster"]) == (1, 1, 1)
        assert launch["dynamic_smem_bytes"] == 0
        assert launch["use_pdl"] is False and launch["cooperative"] is False
        assert loader.launch_grid(record) == (148, 1, 1)
        plan = [tuple(pair) for pair in record["arg_plan"]]
        assert plan[-3:] == [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]
        kinds = {kind for kind, _ in plan[:-3]}
        assert kinds <= {"buffer", "parameter", "peer_table"}
        logical = record["logical"]
        assert {key for _, key in plan[:-3]} == set(logical)
        vocabulary = (
            set(loader.LOGICAL_TENSORS)
            | set(loader.LOGICAL_PEER_TABLES)
            | set(loader.LOGICAL_SCALARS)
        )
        assert set(logical.values()) <= vocabulary
        for kind, key in plan[:-3]:
            if kind == "peer_table":
                assert logical[key] in loader.LOGICAL_PEER_TABLES
            elif kind == "buffer":
                assert logical[key] in loader.LOGICAL_TENSORS
            else:
                assert logical[key] in loader.LOGICAL_SCALARS
    # every program is one (dtype, world, mode): no two records share a library
    assert len(libraries) == 24
    with pytest.raises(ValueError):
        loader.record_name("f64", 2, 0)
    with pytest.raises(ValueError):
        loader.record_name("bf16", 3, 0)
    with pytest.raises(ValueError):
        loader.record_name("bf16", 2, 2)


def test_plan_arguments_follow_the_record():
    record = loader.record_for("bf16", 4, 1)
    values = {
        "inp": "INP",
        "out": "OUT",
        "staging_peers": "STAGING",
        "signals_peers": "SIGNALS",
        "signals_self": "SELF",
        "rank": 3,
        "B": 1,
        "S_local": 5,
        "H_local": 6,
        "D": 128,
    }
    args = loader.plan_arguments(record, values)
    assert args == [
        "INP",
        "OUT",
        "STAGING",
        "SIGNALS",
        "SELF",
        3,
        1,
        5,
        6,
        128,
        148,
        1,
        1,
    ]
    with pytest.raises(ValueError, match="needs 'out'"):
        loader.plan_arguments(
            record, {key: value for key, value in values.items() if key != "out"}
        )


def test_peer_tables_and_signal_constants():
    table = backend.peer_table([11, 22])
    assert table.dtype == torch.int64 and table.device.type == "cpu"
    assert table.is_contiguous() and table.numel() == loader.PEER_TABLE_CAPACITY == 8
    assert table.tolist() == [11, 22, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        backend.peer_table([11, 0])
    with pytest.raises(ValueError):
        backend.peer_table(list(range(1, 10)))
    assert loader.SIGNAL_WORDS == 864 and loader.SIGNAL_BYTES == 3456
    assert pkg.SIGNAL_BYTES == loader.SIGNAL_BYTES
    assert backend.dtype_name(torch.bfloat16) == "bf16"
    assert backend.dtype_name(torch.float16) == "f16"
    assert backend.dtype_name(torch.float32) == "f32"
    with pytest.raises(ValueError):
        backend.dtype_name(torch.int32)


def test_route_requires_delivered_libraries(monkeypatch):
    """A route exists only where the manifest delivers both the scatter and the gather library."""
    scatter = loader.library_name(loader.record_name("bf16", 2, 0), "sm_100a")
    gather = loader.library_name(loader.record_name("bf16", 2, 1), "sm_100a")
    monkeypatch.setattr(cake_build, "libraries_by_name", lambda: {})
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2
    )
    monkeypatch.setattr(cake_build, "libraries_by_name", lambda: {scatter: {}})
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2
    )
    monkeypatch.setattr(
        cake_build, "libraries_by_name", lambda: {scatter: {}, gather: {}}
    )
    assert loader.route_applies(device_capability=(10, 0), dtype="bf16", world_size=2)
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=4
    )
    assert not loader.route_applies(
        device_capability=(10, 3), dtype="bf16", world_size=2
    )
    assert not loader.route_applies(
        device_capability=(9, 0), dtype="bf16", world_size=2
    )
    with pytest.raises(RuntimeError):
        loader.library_name(loader.record_name("bf16", 2, 0), "sm_90a")


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only.
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES)
    package = Path(cake_build.__file__).parent
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
    libraries = cake_build.libraries_by_name()
    if not libraries:
        pytest.skip("build.json delivers no library yet (the export run fills it)")
    referenced = set()
    for name, record in loader.MODULES.items():
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)


# ---- the package's view of the signal allocation (one GPU) --------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_signal_view_aliases_the_allocation_without_a_copy():
    device_index = torch.cuda.current_device()
    buffer = torch.zeros(loader.SIGNAL_WORDS, dtype=torch.uint32, device="cuda")
    view = backend.signal_view(buffer.data_ptr(), device_index)
    assert view.dtype == torch.uint32 and view.shape == (loader.SIGNAL_WORDS,)
    assert view.is_contiguous() and view.device == buffer.device
    assert view.data_ptr() == buffer.data_ptr()
    view.fill_(7)
    torch.cuda.synchronize()
    assert torch.equal(buffer, view)
    view.zero_()
    torch.cuda.synchronize()
    assert int(buffer.sum()) == 0
    del view
    with pytest.raises(ValueError):
        backend.signal_view(0, device_index)
