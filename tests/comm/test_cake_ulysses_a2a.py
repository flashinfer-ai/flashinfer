"""The Cake Ulysses all-to-all package: its portfolio through the public communicator, and the package route.

The backend assertions import the package only through :func:`tests.test_helpers.cake_package.cake_package`
(flashinfer-ai/flashinfer#6553); the multi-GPU portfolio runs through the public communicator so the public
contract (caller-owned outputs, NCCL-equal results) is what is checked.
"""

import ctypes
import gc
import re
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.comm.cake_ulysses_a2a_fixture import (
    NVLINK_KERNELS,
    PreparedCase,
    cake_direct_available,
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


def _refresh(case, step):
    """New input values and poisoned outputs, so a call that writes nothing or the wrong output fails."""
    generator = torch.Generator(device="cuda").manual_seed(
        7919 * step + case.communicator.rank
    )
    for source in case.inputs:
        source.copy_(
            torch.randn(
                source.shape, dtype=source.dtype, device="cuda", generator=generator
            )
        )
    for output in case.outputs:
        output.fill_(float("nan"))


def _registered_worker(world_size, rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        for index, shape in enumerate(shapes(world_size)):
            case = prepare(
                shape, backend="nvlink", nvlink_kernel="cake", registered=True
            )
            communicator = case.communicator
            try:
                assert case.nvlink_kernel == "cake" and case.registered
                assert {output.data_ptr() for output in case.outputs} == set(
                    communicator._nvlink_outputs
                )
                _refresh(case, 0)
                case.check()
                # caller-owned outputs on the same communicator take the staged
                # route; the two routes alternate on disjoint words of one
                # signal allocation
                staged = PreparedCase(
                    communicator,
                    case.inputs,
                    tuple(torch.empty_like(output) for output in case.outputs),
                    None,
                    "cake",
                )
                for step in range(1, 3):
                    _refresh(staged, step)
                    staged.check()
                    _refresh(case, step)
                    case.check()
                if index == 0:
                    # the direct route under CUDA graph capture and replay
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        case.run()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        case.run()
                    for step in range(3, 6):
                        _refresh(case, step)
                        graph.replay()
                        case.check(run=False)
                    del graph
                    torch.cuda.synchronize()
                    dist.barrier()
                    # an output registered under inference mode keeps no base
                    # view; releasing it after close() must stay safe
                    with torch.inference_mode():
                        late = communicator.allocate_output(
                            case.inputs[0], "scatter_heads"
                        )
                    assert late.data_ptr() in communicator._nvlink_outputs
            finally:
                case.close()
            # close releases every registered allocation and peer mapping
            assert not communicator._nvlink_outputs
            assert not communicator._exports and not communicator._imports
            if index == 0:
                del late
                gc.collect()
        # the portable kernel registers nothing: allocate_output stays a plain allocation
        portable = prepare(
            shapes(world_size)[0],
            backend="nvlink",
            nvlink_kernel="portable",
            registered=True,
        )
        try:
            assert not portable.communicator._nvlink_outputs
            portable.check()
        finally:
            portable.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_registered_outputs_take_the_direct_route(world_size):
    """Registered outputs: the portfolio bit-exact, alternation with caller-owned outputs, graph replay, close."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if torch.cuda.get_device_capability(0) not in pkg.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("the Cake portfolio targets Blackwell SM100/SM103")
    if not cake_direct_available(world_size):
        pytest.skip("build.json delivers no direct-route library for this device yet")
    multi_process_parallel(world_size, _registered_worker, ())


def _offset_input(dimensions, dtype, offset):
    """A contiguous CUDA tensor that starts ``offset`` elements into its storage."""
    numel = 1
    for extent in dimensions:
        numel *= extent
    storage = torch.empty(numel + offset, dtype=dtype, device="cuda")
    return storage[offset:].view(dimensions)


def _unaligned_rank_worker(world_size, rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        for shape in shapes(world_size):
            if shape["performance"]:
                continue
            dtype = getattr(torch, shape["dtype"])
            batch, sequence, heads, head_dim = (
                shape[key] for key in ("B", "S_local", "H", "D")
            )
            local_shape = (batch, sequence, heads, head_dim)
            global_shape = (batch, sequence * world_size, heads // world_size, head_dim)
            offset = 1 if rank == 1 else 0
            inputs = tuple(
                _offset_input(dimensions, dtype, offset)
                for dimensions in (local_shape, local_shape, local_shape, global_shape)
            )
            if rank == 1:
                assert all(source.data_ptr() % 16 for source in inputs)
            case = prepare(
                shape,
                backend="nvlink",
                nvlink_kernel="cake",
                inputs=inputs,
                registered=True,
            )
            try:
                assert case.nvlink_kernel == "cake" and case.registered
                for step in range(2):
                    _refresh(case, step)
                    case.check()
            finally:
                case.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_registered_outputs_with_one_unaligned_rank(world_size):
    """Rank 1 passes every input one element into its storage, so its streams take the
    element path while its peers' take the 16-byte vector path; every correctness row
    of the world completes bit-exact on the direct route."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if torch.cuda.get_device_capability(0) not in pkg.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("the Cake portfolio targets Blackwell SM100/SM103")
    if not cake_direct_available(world_size):
        pytest.skip("build.json delivers no direct-route library for this device yet")
    multi_process_parallel(world_size, _unaligned_rank_worker, ())


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
    """One record per (route, dtype, world size, mode), each carrying the host plan of its collective."""
    expected = {
        loader.record_name(dtype, world, mode, direct=direct)
        for direct in (False, True)
        for dtype in loader.DTYPES
        for world in loader.WORLD_SIZES
        for mode in loader.MODES
    }
    assert set(loader.MODULES) == expected
    assert len(loader.MODULES) == 48
    libraries = set()
    for name, record in loader.MODULES.items():
        direct = name.startswith("direct_")
        direction = "scatter" if record["mode"] == 0 else "gather"
        assert name == (
            f"{'direct_' if direct else ''}{direction}_{record['dtype']}_w{record['world_size']}"
        )
        assert (
            loader.record_for(
                record["dtype"], record["world_size"], record["mode"], direct=direct
            )
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
        # the staged route reads the staging tables, the direct route the output's
        tables = [logical[key] for kind, key in plan if kind == "peer_table"]
        assert tables == [
            "out_peers" if direct else "staging_peers",
            "signals_peers",
        ]
    # every program is one (route, dtype, world, mode): no two records share a library
    assert len(libraries) == 48
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
    direct = loader.record_for("bf16", 4, 1, direct=True)
    with pytest.raises(ValueError, match="needs 'out_peers'"):
        loader.plan_arguments(direct, values)
    args = loader.plan_arguments(direct, {**values, "out_peers": "OUTPEERS"})
    assert args == [
        "INP",
        "OUT",
        "OUTPEERS",
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
    # the direct route needs its own two libraries
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2, direct=True
    )
    direct = {
        loader.library_name(
            loader.record_name("bf16", 2, mode, direct=True), "sm_100a"
        ): {}
        for mode in loader.MODES
    }
    monkeypatch.setattr(cake_build, "libraries_by_name", lambda: direct)
    assert loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2, direct=True
    )
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2
    )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_device_view_aliases_an_output_allocation(dtype):
    device_index = torch.cuda.current_device()
    buffer = torch.randn(1000, device="cuda").to(dtype)
    view = backend.device_view(buffer.data_ptr(), device_index, dtype, 640)
    assert view.dtype == dtype and view.shape == (640,) and view.is_contiguous()
    assert view.data_ptr() == buffer.data_ptr() and view.device == buffer.device
    assert torch.equal(view, buffer[:640])
    view.fill_(3)
    torch.cuda.synchronize()
    assert torch.equal(buffer[:640], torch.full_like(view, 3))
    with pytest.raises(ValueError):
        backend.device_view(buffer.data_ptr(), device_index, torch.int64, 8)
    with pytest.raises(ValueError):
        backend.device_view(buffer.data_ptr(), device_index, dtype, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_device_view_structures_outlive_a_base_dropped_under_inference_mode():
    """A view narrowed under inference mode records no base: the DLPack structures live with the storage."""
    device_index = torch.cuda.current_device()
    buffer = torch.zeros(1024, device="cuda", dtype=torch.bfloat16)

    def narrowed():
        with torch.inference_mode():
            view = backend.device_view(
                buffer.data_ptr(), device_index, torch.bfloat16, 1024
            )
            return view.narrow(0, 0, 512).view(16, 32)

    out = narrowed()
    gc.collect()
    # refill freed small Python allocations with nonzero bytes: structures freed
    # with the base would now hold a garbage deleter
    churn = [(ctypes.c_ubyte * 64).from_buffer_copy(b"\xff" * 64) for _ in range(4096)]
    with torch.inference_mode():
        out.fill_(5)
    torch.cuda.synchronize()
    assert int((buffer[:512] == 5).sum()) == 512
    assert int(buffer[512:].count_nonzero()) == 0
    del out
    gc.collect()  # releases the storage through the view's deleter
    del churn
    assert int((buffer[:512] == 5).sum()) == 512
