# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Host-model contracts for inline V4.1 combined sparse routing.

Run with --confcutdir=tests/experimental to exclude root CUDA collection probes.
Only the actual selected definitions are AST-loaded, without package imports or
CuTe decorators. NumPy supplies signed Int32/Int64 and Boolean complement
semantics; explicit virtual pointers, tables and pools provide bounded read
traces without allocating accelerator storage. This is not CuTe lowering, GPU
execution, producer-address lowering or full decode numerical qualification.
"""

import ast
import builtins
import importlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[2]
_PRIVATE = _ROOT / "flashinfer/experimental/deepseek_v41/prims_ts/_sparse_views.py"
_CORE = (
    _ROOT / "flashinfer/attention/prims_ts/kernels/mla_decode/helpers/sparse_indices.py"
)
_PRODUCER = _PRIVATE.with_name("resources.py")
_INT_MIN = -(1 << 31)
_INT_MAX = (1 << 31) - 1
_WIDTH_EXTENTS = (
    (0, 128),
    (64, 256),
    (128, 256),
    (192, 384),
    (256, 384),
    (320, 512),
    (384, 512),
    (448, 640),
    (512, 640),
)
_ACCELERATOR_ROOTS = {
    "torch",
    "cutlass",
    "flashinfer",
    "cuda",
    "nvidia",
    "triton",
    "cupy",
}


def _accelerator_modules():
    return {name for name in sys.modules if name.split(".")[0] in _ACCELERATOR_ROOTS}


@pytest.fixture(autouse=True)
def no_accelerator_work(monkeypatch):
    """Never import accelerator packages, including through importlib's API.

    A test runner may already have imported them. Guard existing CUDA/JIT entry
    points without importing anything to install those guards, and require that
    no accelerator module or CUDA visibility change occurs during each test.
    """
    modules_before = _accelerator_modules()
    visibility = {
        name: (name in os.environ, os.getenv(name))
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
        )
    }
    attempts = []

    def fail(label):
        attempts.append(label)
        raise AssertionError(
            "sparse-view host test attempted accelerator work: " + label
        )

    def forbidden(label):
        def reject(*args, **kwargs):
            fail(label)

        return reject

    for module_name, names in (
        (
            "torch.cuda",
            (
                "init",
                "_lazy_init",
                "is_available",
                "device_count",
                "current_device",
                "get_device_capability",
                "get_device_properties",
                "synchronize",
                "current_stream",
                "default_stream",
                "is_current_stream_capturing",
                "device",
            ),
        ),
        ("cutlass.cute.runtime", ("load_module",)),
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            for name in names:
                if name in vars(module):
                    monkeypatch.setattr(
                        module, name, forbidden(module_name + "." + name)
                    )
    cuda = sys.modules.get("torch.cuda")
    initialized = vars(cuda).get("_initialized") if cuda is not None else None
    for module_name in (
        "torch._C",
        "cuda.bindings.driver",
        "cuda.bindings.runtime",
        "cuda.cuda",
        "cuda.cudart",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            for name, value in tuple(vars(module).items()):
                prefix = "_cuda_" if module_name == "torch._C" else "cu"
                if (
                    name.startswith(prefix)
                    and callable(value)
                    and not isinstance(value, type)
                ):
                    monkeypatch.setattr(
                        module, name, forbidden(module_name + "." + name)
                    )
    cute = sys.modules.get("cutlass.cute")
    if cute is not None and "compile" in vars(cute):

        class CompileGuard:
            def __getitem__(self, frontend):
                return self

            def __call__(self, *args, **kwargs):
                fail("cute.compile")

        monkeypatch.setattr(cute, "compile", CompileGuard())

    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in _ACCELERATOR_ROOTS:
            fail("import " + name)
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, package=None):
        if (
            name.lstrip(".").split(".")[0] in _ACCELERATOR_ROOTS
            or (package or "").split(".")[0] in _ACCELERATOR_ROOTS
        ):
            fail("importlib " + name)
        return original_import_module(name, package)

    class ImportGuard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in _ACCELERATOR_ROOTS:
                fail("module finder " + fullname)
            return None

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    monkeypatch.setattr(sys, "meta_path", [ImportGuard(), *sys.meta_path])
    try:
        yield
    finally:
        assert not attempts, f"intercepted accelerator attempts: {attempts}"
        assert _accelerator_modules() == modules_before
        assert visibility == {
            name: (name in os.environ, os.getenv(name)) for name in visibility
        }
        if cuda is not None:
            assert vars(cuda).get("_initialized") == initialized


def _load_definitions(path, names, namespace):
    """Keep production bodies intact; exclude imports and remove decorators."""
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    for definition in definitions:
        for node in ast.walk(definition):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                node.decorator_list = []
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )


class _Pointer:
    """Element arithmetic over a virtual Int64 byte address, never real memory."""

    def __init__(self, table, address):
        self.table = table
        self.address = np.int64(address)

    def toint(self):
        return self.address

    def __add__(self, elements):
        return _Pointer(self.table, self.address + np.int64(elements) * np.int64(4))

    def load(self, *, count, alignment):
        assert int(self.address) % alignment == 0
        self.table.vector_loads.append((int(self.address), count, alignment))
        return tuple(
            self.table.read((self + offset).address) for offset in range(count)
        )


class _IndexTable:
    """A small real table with possibly enormous virtual inter-request strides."""

    def __init__(self, rows, *, row_stride, base):
        self.rows = tuple(tuple(np.int32(value) for value in row) for row in rows)
        width = len(self.rows[0])
        assert all(len(row) == width for row in self.rows)
        assert row_stride >= width
        # Batch is a runtime scalar, as in the actual fake-tensor declaration.
        self.shape = (np.int32(len(self.rows)), width)
        self.stride = (np.int64(row_stride), 1)
        self.base = np.int64(base)
        self.iterator = _Pointer(self, self.base)
        self.allowed = None
        self.reset()

    def reset(self):
        self.reads = []
        self.vector_loads = []

    def read(self, address):
        offset = int(address) - int(self.base)
        assert offset >= 0 and offset % 4 == 0, "invalid virtual table pointer"
        row, column = divmod(offset // 4, int(self.stride[0]))
        assert 0 <= row < len(self.rows), "read outside combined-table batch"
        assert 0 <= column < self.shape[1], "read outside combined-table columns"
        coordinate = (row, column)
        assert self.allowed is None or coordinate in self.allowed, (
            f"read outside the declared visible prefix: {coordinate}"
        )
        self.reads.append(coordinate)
        return self.rows[row][column]


class _RowTensor:
    def __init__(self, pointer, layout):
        assert layout.stride == (1,)
        self.iterator = pointer
        self.shape = layout.shape

    def __getitem__(self, column):
        assert 0 <= int(column) < int(self.shape[0]), "read outside row tensor"
        return self.iterator.table.read((self.iterator + column).address)


class _LengthVector:
    def __init__(self, values):
        self.values = tuple(np.int32(value) for value in values)
        self.shape = (np.int32(len(self.values)),)
        self.reads = []

    def __getitem__(self, row):
        row = int(row)
        assert 0 <= row < len(self.values)
        self.reads.append(row)
        return self.values[row]


class _VirtualPool:
    def __init__(self, rows, width):
        self.shape = (
            (np.int64(rows * width),) if width == 288 else (1, 512, np.int64(rows))
        )


class _HostMemory:
    def table(self, rows, *, row_stride=None, base=None):
        stride = len(rows[0]) if row_stride is None else row_stride
        self.current_table = _IndexTable(
            rows, row_stride=stride, base=(1 << 36) if base is None else base
        )
        return self.current_table

    def make_ptr(self, dtype, address, *, assumed_align):
        assert dtype is np.int32 and assumed_align == 4
        return _Pointer(self.current_table, address)

    @staticmethod
    def pointer(iterator, *, dtype):
        assert isinstance(iterator, _Pointer) and dtype is np.int32
        return iterator

    @staticmethod
    def array(dtype, count, *, space):
        assert dtype is np.int32 and space == "rmem"
        return [np.int32(0) for _ in range(count)]


class _Constexpr:
    def __class_getitem__(cls, annotation):
        return annotation


@pytest.fixture
def sparse_host(no_accelerator_work):
    memory = _HostMemory()
    cutlass = SimpleNamespace(
        Array=memory.array,
        AddressSpace=SimpleNamespace(rmem="rmem"),
        Pointer=memory.pointer,
        range_constexpr=range,
        const_expr=bool,
        Constexpr=_Constexpr,
    )
    cute = SimpleNamespace(
        math=SimpleNamespace(min=min, max=max),
        make_ptr=memory.make_ptr,
        make_layout=lambda shape, *, stride: SimpleNamespace(
            shape=shape, stride=stride
        ),
        make_tensor=_RowTensor,
        size=lambda pool: math.prod(int(dimension) for dimension in pool.shape),
    )
    namespace = dict(
        __name__="_deepseek_v41_sparse_views_host",
        cutlass=cutlass,
        cute=cute,
        Int32=np.int32,
        Int64=np.int64,
    )
    _load_definitions(
        _CORE, ("SparseKVIndices", "SparseSourceTile", "SparseKVLengths"), namespace
    )
    _load_definitions(
        _PRIVATE,
        ("CombinedSparseRouteView", "CombinedSparseTile", "bind_combined_sparse_views"),
        namespace,
    )
    return SimpleNamespace(memory=memory, **namespace)


def _case(
    host,
    k,
    seq_lens,
    topk_lens,
    *,
    rows=None,
    swa_rows=64,
    main_rows=64,
    request=0,
    row_stride=None,
    base=None,
):
    if rows is None:
        rows = [[column % 31 for column in range(128 + k)] for _ in seq_lens]
    assert len(rows) == len(seq_lens) == len(topk_lens)
    assert all(len(row) == 128 + k for row in rows)
    table = host.memory.table(rows, row_stride=row_stride, base=base)
    seq, topk = _LengthVector(seq_lens), _LengthVector(topk_lens)
    # The private binder's source order is main, SWA, not SWA, main.
    pools = (_VirtualPool(main_rows, 288), _VirtualPool(swa_rows, 512))
    extent = dict(_WIDTH_EXTENTS)[k]
    capacity = max(256, extent)
    request = np.int32(request)
    routes, lengths = host.bind_combined_sparse_views(
        table, seq, topk, pools, capacity, request=request
    )
    return SimpleNamespace(
        routes=routes,
        lengths=lengths,
        table=table,
        seq=seq,
        topk=topk,
        pools=pools,
        request=request,
        capacity=capacity,
        extent=extent,
    )


def _masked_rows(capacity):
    return [_INT_MAX] * 128 + [-1] * (capacity - 128)


def _valid_prefix_expectation(case, swa_length, main_length):
    """Expected prefixes for fixtures whose IDs are all known valid.

    ID validity is deliberately not reimplemented here. Adversarial IDs have
    literal expected rows/masks in their own tests below.
    """
    expected = _masked_rows(case.capacity)
    invalid = [True] * case.capacity
    columns = list(range(swa_length)) + list(range(128, 128 + main_length))
    row = case.table.rows[int(case.request)]
    expected[:swa_length] = [int(value) for value in row[:swa_length]]
    expected[128 : 128 + main_length] = [
        int(value) - (1 << 31) for value in row[128 : 128 + main_length]
    ]
    for column in columns:
        invalid[column] = False
    return expected, invalid, columns


def _check_routes(case, expected, invalid, columns):
    """Exercise actual scalar gather, softmax mask and quad gather bodies."""
    assert len(expected) == len(invalid) == case.capacity
    table, request = case.table, case.request
    reads = [(int(request), column) for column in columns]
    table.allowed = set(reads)
    table.reset()
    gathered = [case.routes[np.int32(token), request] for token in range(case.capacity)]
    assert all(isinstance(value, np.int32) for value in gathered)
    assert [int(value) for value in gathered] == expected
    assert table.reads == reads
    table.reset()
    masks = [
        case.routes.is_invalid(np.int32(token), request)
        for token in range(case.capacity)
    ]
    assert all(isinstance(value, np.bool_) for value in masks)
    assert [bool(value) for value in masks] == invalid
    assert table.reads == reads
    for method in ("load_quad", "load_full_quad"):
        table.reset()
        quads = []
        for token in range(0, case.capacity, 4):
            tile = case.routes.select_source_tile(np.int32(token), request)
            quads.extend(getattr(tile, method)(np.int32(token)))
        assert [int(value) for value in quads] == expected
        assert table.reads == reads
    # A bound CTA snapshots the two live lengths once, not once per slot.
    assert case.seq.reads == case.topk.reads == [int(request)]


def test_host_scalars_preserve_signed_wrap_and_boolean_complement():
    with np.errstate(over="ignore"):
        assert int(np.int32(_INT_MIN) - np.int32(128)) == _INT_MAX - 127
    valid = np.int32(0) >= np.int32(0)
    assert isinstance(valid, np.bool_)
    assert bool(~valid) is False
    assert bool(~(np.int32(-1) >= np.int32(0))) is True
    assert int(np.int64(_INT_MAX + 65)) == _INT_MAX + 65


def test_binder_rejects_persistent_request_before_touching_inputs(sparse_host):
    poison = object()
    with pytest.raises(ValueError, match="nonpersistent request"):
        sparse_host.bind_combined_sparse_views(poison, poison, poison, poison, 256)


@pytest.mark.parametrize("k,extent", _WIDTH_EXTENTS)
@pytest.mark.parametrize(
    "visibility", ("zero", "int_min", "partial", "tail", "saturated")
)
def test_all_widths_have_fixed_extents_and_bounded_identical_masks(
    sparse_host, k, extent, visibility
):
    seq, topk, swa_length, main_length = {
        "zero": (0, 128, 0, 0),
        "int_min": (_INT_MIN, _INT_MIN, 0, 0),
        "partial": (5, 131, 5, 3 if k else 0),
        "tail": (127, 127 + k, 127, k - 1 if k else 0),
        "saturated": (_INT_MAX, _INT_MAX, 128, k),
    }[visibility]
    case = _case(sparse_host, k, (seq,), (topk,))
    routes = case.routes
    assert isinstance(routes, sparse_host.CombinedSparseRouteView)
    assert isinstance(routes, sparse_host.SparseKVIndices)
    assert routes.bind_query_row(case.request) is routes
    assert isinstance(case.lengths, sparse_host.SparseKVLengths)
    assert routes.indices is case.table
    assert routes.shape == (max(256, extent), np.int32(1))
    assert case.lengths.shape == (np.int32(1),)
    assert isinstance(case.lengths.shape[0], np.int32)
    assert int(case.lengths[np.int32(0)]) == extent
    head = routes.select_source_tile(np.int32(0), case.request)
    main = routes.select_source_tile(np.int32(128), case.request, for_mask=True)
    assert isinstance(head, sparse_host.CombinedSparseTile)
    assert isinstance(head, sparse_host.SparseSourceTile)
    assert not head.valid_prefix and not main.valid_prefix
    assert (int(head.origin), int(main.origin)) == (0, 128)
    assert (int(head.length), int(main.length)) == (swa_length, main_length)
    assert (int(head.tag), int(main.tag)) == (0, _INT_MIN)
    assert head.indices.shape == main.indices.shape == (128 + k,)
    _check_routes(case, *_valid_prefix_expectation(case, swa_length, main_length))


@pytest.mark.parametrize("seq,swa_length", ((0, 0), (3, 3), (_INT_MAX, 128)))
def test_k0_forced_masked_main_loop_never_reads_column128(sparse_host, seq, swa_length):
    case = _case(sparse_host, 0, (seq,), (_INT_MAX,))
    assert case.table.shape[1] == 128
    assert case.routes.shape[0] == 256
    assert int(case.lengths[case.request]) == 128
    main = case.routes.select_source_tile(np.int32(128), case.request)
    case.table.allowed = set()
    assert int(main.length) == 0 and int(main.origin) == 128
    assert int(main.load_row(np.int32(128))) == -1
    assert [int(value) for value in main.load_quad(np.int32(128))] == [-1] * 4
    assert bool(main.is_invalid(np.int32(128), case.request)) is True
    assert not case.table.reads and not case.table.vector_loads
    _check_routes(case, *_valid_prefix_expectation(case, swa_length, 0))


@pytest.mark.parametrize(
    "seq,topk,swa_length,main_length",
    (
        (_INT_MIN, _INT_MIN, 0, 0),
        (_INT_MIN, 131, 0, 3),
        (3, _INT_MIN, 3, 0),
        (-1, 127, 0, 0),
        (0, 128, 0, 0),
        (129, 129, 128, 1),
        (128, 192, 128, 64),
        (_INT_MAX, _INT_MAX, 128, 64),
    ),
)
def test_length_clamping_precedes_main_subtraction(
    sparse_host, seq, topk, swa_length, main_length
):
    case = _case(sparse_host, 64, (seq,), (topk,))
    assert (int(case.routes.sl), int(case.routes.cl)) == (swa_length, main_length)
    _check_routes(case, *_valid_prefix_expectation(case, swa_length, main_length))


@pytest.mark.parametrize("source,origin", (("swa", 0), ("main", 128)))
@pytest.mark.parametrize(
    "start_delta,visible,active_offsets",
    (
        (0, 1, (0,)),
        (0, 2, (0, 1)),
        (0, 3, (0, 1, 2)),
        (-1, 3, (1, 2, 3)),
        (2, 3, (0,)),
        (3, 3, ()),
        (-4, 3, ()),
        (0, 4, (0, 1, 2, 3)),
    ),
)
def test_partial_quad_only_loads_live_columns_and_matches_scalar_masks(
    sparse_host, source, origin, start_delta, visible, active_offsets
):
    row = [1] * 192
    row[origin : origin + 4] = [9, 4, 12, 6]
    case = _case(
        sparse_host,
        64,
        (visible if source == "swa" else 0,),
        (128 + visible if source == "main" else 128,),
        rows=[row],
    )
    tile = case.routes.select_source_tile(np.int32(origin), case.request)
    assert int(tile.origin) == origin
    start = origin + start_delta
    expected = [_INT_MAX if source == "swa" else -1] * 4
    invalid = [True] * 4
    columns = [start + offset for offset in active_offsets]
    for offset in active_offsets:
        expected[offset] = row[start + offset] - ((1 << 31) if source == "main" else 0)
        invalid[offset] = False
    reads = [(0, column) for column in columns]
    case.table.allowed = set(reads)
    case.table.reset()
    assert [int(value) for value in tile.load_quad(np.int32(start))] == expected
    assert case.table.reads == reads
    assert len(case.table.vector_loads) == int(start_delta == 0 and visible == 4)
    case.table.reset()
    assert [int(tile.load_row(np.int32(start + j))) for j in range(4)] == expected
    assert case.table.reads == reads
    case.table.reset()
    masks = [tile.is_invalid(np.int32(start + j), case.request) for j in range(4)]
    assert all(isinstance(value, np.bool_) for value in masks)
    assert [bool(value) for value in masks] == invalid
    assert case.table.reads == reads


def test_signed_holes_duplicates_and_capacity_edges_share_the_softmax_mask(sparse_host):
    row = [1] * 192
    row[:8] = [3, _INT_MIN, 3, -1, 15, 16, _INT_MAX, 0]
    row[128:136] = [2, _INT_MIN, 2, -1, 6, 7, _INT_MAX, 0]
    case = _case(sparse_host, 64, (8,), (136,), rows=[row], swa_rows=16, main_rows=7)
    expected = _masked_rows(case.capacity)
    expected[:8] = [3, _INT_MAX, 3, _INT_MAX, 15, _INT_MAX, _INT_MAX, 0]
    # Signed ID validation must precede source tagging. An original INT_MIN is
    # a hole, whereas a valid original main ID0 legitimately routes as INT_MIN.
    expected[128:136] = [
        _INT_MIN + 2,
        -1,
        _INT_MIN + 2,
        -1,
        _INT_MIN + 6,
        -1,
        -1,
        _INT_MIN,
    ]
    invalid = [True] * case.capacity
    invalid[:8] = invalid[128:136] = [
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
    ]
    _check_routes(case, expected, invalid, list(range(8)) + list(range(128, 136)))
    assert (int(case.routes.sc), int(case.routes.cc)) == (16, 7)


def test_intmax_stays_a_sentinel_with_virtual_pools_larger_than_int32(sparse_host):
    large_id = (1 << 24) + 3
    row = [1] * 192
    row[:6] = row[128:134] = [_INT_MIN, -1, _INT_MAX, _INT_MAX - 1, large_id, 0]
    case = _case(
        sparse_host,
        64,
        (6,),
        (134,),
        rows=[row],
        swa_rows=_INT_MAX + 19,
        main_rows=_INT_MAX + 65,
    )
    assert isinstance(case.routes.sc, np.int64) and isinstance(case.routes.cc, np.int64)
    assert (int(case.routes.sc), int(case.routes.cc)) == (_INT_MAX + 19, _INT_MAX + 65)
    assert all(
        math.prod(int(dim) for dim in pool.shape) > (1 << 31) for pool in case.pools
    )
    # The surviving IDs name virtual cache rows beyond a 2GiB byte offset.
    # No physical pool is allocated or read; producer lowering is not tested.
    assert large_id * 288 > (1 << 31) and large_id * 512 > (1 << 31)
    expected = _masked_rows(case.capacity)
    expected[:6] = [_INT_MAX, _INT_MAX, _INT_MAX, _INT_MAX - 1, large_id, 0]
    expected[128:134] = [-1, -1, -1, -2, _INT_MIN + large_id, _INT_MIN]
    invalid = [True] * case.capacity
    invalid[:6] = invalid[128:134] = [True, True, True, False, False, False]
    _check_routes(case, expected, invalid, list(range(6)) + list(range(128, 134)))


def test_zero_pool_capacity_invalidates_even_nonnegative_live_ids(sparse_host):
    case = _case(sparse_host, 64, (4,), (132,), swa_rows=0, main_rows=0)
    _check_routes(
        case,
        _masked_rows(case.capacity),
        [True] * case.capacity,
        list(range(4)) + list(range(128, 132)),
    )


@pytest.mark.parametrize("row_stride", (337, (1 << 29) + 19))
def test_original_batched_row_stride_and_int64_virtual_row_addresses(
    sparse_host, row_stride
):
    rows = [
        [17 + 56 * batch + column % 17 for column in range(320)] for batch in range(3)
    ]
    case = _case(
        sparse_host,
        192,
        (0, 1, 7),
        (128, 129, 133),
        rows=rows,
        swa_rows=256,
        main_rows=256,
        request=2,
        row_stride=row_stride,
        base=4096,
    )
    assert isinstance(case.routes.row, np.int32)
    assert case.routes.shape == (384, np.int32(3))
    assert case.lengths.shape == (np.int32(3),)
    assert int(case.lengths[np.int32(0)]) == int(case.lengths[np.int32(2)]) == 384
    address = 4096 + 2 * row_stride * 4
    for origin in (0, 128):
        tile = case.routes.select_source_tile(np.int32(origin), case.request)
        assert isinstance(tile.indices.iterator.toint(), np.int64)
        assert int(tile.indices.iterator.toint()) == address
    if row_stride > (1 << 29):
        assert address > (1 << 32)
    expected, invalid, columns = _valid_prefix_expectation(case, 7, 5)
    _check_routes(case, expected, invalid, columns)
    assert (int(case.routes.sl), int(case.routes.cl)) == (7, 5)


@pytest.mark.parametrize("kind", ("route", "tile", "length"))
def test_structural_hooks_follow_core_sparse_length_tuple_convention(
    sparse_host, monkeypatch, kind
):
    """Check hook delegation, not a fabricated claim about actual MLIR leaves."""
    case = _case(sparse_host, 64, (3,), (131,))
    if kind == "route":
        original = case.routes
        table = sparse_host.memory.table([[2] * 192 for _ in range(3)])
        replacement = (
            table,
            np.int32(2),
            np.int32(2),
            np.int32(1),
            np.int64(23),
            np.int64(17),
        )
    elif kind == "tile":
        original = case.routes.select_source_tile(np.int32(128), case.request)
        replacement = (
            original.indices,
            np.int32(128),
            np.int32(1),
            np.int64(17),
            np.int32(_INT_MIN),
        )
    else:
        original = case.lengths
        replacement = (None, None, np.int32(384), np.int32(5))
    flat_values = object()
    extract = Mock(return_value=flat_values)
    reconstruct = Mock(return_value=replacement)
    monkeypatch.setattr(
        sparse_host.cutlass, "extract_mlir_values", extract, raising=False
    )
    monkeypatch.setattr(
        sparse_host.cutlass, "new_from_mlir_values", reconstruct, raising=False
    )
    assert original.__extract_mlir_values__() is flat_values
    rebuilt = original.__new_from_mlir_values__(flat_values)
    extract.assert_called_once_with(original.values)
    reconstruct.assert_called_once_with(original.values, flat_values)
    assert type(rebuilt) is type(original) and rebuilt is not original
    assert rebuilt.values is replacement
    if kind == "route":
        assert rebuilt.capacity == original.capacity
        assert rebuilt.shape == (original.capacity, np.int32(3))
        assert rebuilt.row == np.int32(2)
    elif kind == "length":
        assert rebuilt.shape == (np.int32(5),)
        assert isinstance(rebuilt.shape[0], np.int32)
        assert int(rebuilt[np.int32(4)]) == 384


def test_mixed_producer_hook_forwards_original_binder_arguments_without_imports():
    result = object()
    bind = Mock(return_value=result)
    namespace = {"Fp4Fp8KvProducer": object, "bind_combined_sparse_views": bind}
    _load_definitions(_PRODUCER, ("MixedFp8KvProducer",), namespace)
    producer = namespace["MixedFp8KvProducer"]()
    indices, seq, topk, main, swa = (object() for _ in range(5))
    arguments = (indices, seq, topk, (main, swa), 384, np.int32(2))
    assert producer.bind_sparse_views(*arguments) is result
    bind.assert_called_once_with(*arguments)
