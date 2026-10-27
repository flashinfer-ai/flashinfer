"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Behavioural tests of the Cake fused FP8 row-wise GEMM + all-reduce backend that
need no GPU: the route table and the frozen program record, workspace and
launch geometry, operand validation, the symmetric-memory rendezvous guard,
workspace allocation (readiness words in the signal pad or in a dedicated
symmetric buffer), the capacity-bound prepared launcher, the launch argument
binding, CUDA-graph capture rules and the public dispatch. The four-GPU
numerical test lives in ``test_fp8_rowwise_gemm_allreduce_cake_e2e.py``.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch
from flashinfer.jit import cake_fp8_rowwise_gemm_allreduce as loader

backend = importlib.import_module(
    "flashinfer.comm.fp8_rowwise_gemm_allreduce.cake_fp8_rowwise_gemm_allreduce"
)

N = 3072
K_L = 768
FP8 = torch.float8_e4m3fn


# --------------------------------------------------------------------------- #
# Route table and frozen record
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_route_table_delivers_one_fused_program_for_world_size_four():
    program = loader.main_program(4)
    assert program in loader.PROGRAMS
    assert set(loader.ROUTES) == {"main_ws4_p2p"}
    assert set(loader.ROUTES.values()) == set(loader.PROGRAMS) == {program}
    assert loader.SUPPORTED_WORLD_SIZES == (4,)
    for world_size in (1, 2, 8):
        with pytest.raises(ValueError, match=f"main_ws{world_size}_p2p"):
            loader.main_program(world_size)


@pytest.mark.cpu_only
def test_program_record_carries_the_fused_launcher_contract():
    record = loader.PROGRAMS[loader.main_program(4)]
    # The device unit and its thin tvm-ffi launcher, both directly under the package.
    assert len(record["sources"]) == 2
    for relative in record["sources"]:
        assert relative.startswith("csrc/cake_fp8_rowwise_gemm_allreduce/")
        assert relative.count("/") == 2
    assert record["arches"] == ["sm_100a"]
    assert loader.launch_block(loader.main_program(4)) == (loader.THREADS, 1, 1)
    assert (
        loader.cluster_dims(loader.main_program(4))
        == loader.CLUSTER_DIMS
        == (
            2,
            1,
            1,
        )
    )
    smem = loader.dynamic_smem_bytes(loader.main_program(4))
    assert 0 < smem <= loader.MAX_DYNAMIC_SMEM_BYTES
    args = loader.launcher_args(loader.main_program(4))
    assert args[-3:] == ("grid_x", "grid_y", "grid_z")
    assert len(set(args)) == len(args)
    assert set(args) == {
        "A",
        "W",
        "OUT",
        "out_ptr",
        "sa",
        "sw",
        "ctl",
        "S_mc",
        "S",
        "flags",
        "M",
        "N",
        "K",
        "grid_n",
        "pair_tiles",
        "K_tiles",
        "C",
        "C_cap",
        "S_bytes",
        "flag_words",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "S_peers",
        "flags_peers",
        "grid_x",
        "grid_y",
        "grid_z",
    }


@pytest.mark.cpu_only
def test_frozen_trace_time_choices_are_recorded_not_exposed():
    assert loader.COMM_MODE == "p2p"
    assert loader.TRANSPORT == "pull"
    assert loader.TIMEOUT_CYCLES_LOG2 == 34
    assert loader.NUM_STAGES == 4
    assert loader.EXCHANGE_MODE == "X16"
    assert loader.ARCH_FLAGS.keys() == {"sm_100a"}
    assert loader.CAPABILITY_ARCH == {(10, 0): "sm_100a"}


@pytest.mark.cpu_only
def test_spec_names_carry_the_exact_architecture(monkeypatch):
    specs = []

    def gen_jit_spec(**kwargs):
        specs.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(loader, "gen_jit_spec", gen_jit_spec)
    monkeypatch.setattr(
        loader, "_source_path", lambda relative: f"/csrc/{relative.rsplit('/', 1)[-1]}"
    )
    monkeypatch.setattr(loader, "_source_dir", lambda: loader.Path("/csrc/pkg"))
    loader.spec.cache_clear()
    try:
        spec = loader.spec(loader.main_program(4), "sm_100a")
        assert (
            spec.name
            == f"cake_fp8_rowwise_gemm_allreduce_sm_100a_{loader.main_program(4)}"
        )
        assert spec.extra_cuda_cflags[: len(loader.sm100a_nvcc_flags)] == list(
            loader.sm100a_nvcc_flags
        )
        assert spec.extra_cuda_cflags[len(loader.sm100a_nvcc_flags) :] == list(
            loader.COMPILE_FLAGS
        )
        assert spec.use_fast_math == ("--use_fast_math" in loader.COMPILE_FLAGS)
        assert len(spec.sources) == 2
        with pytest.raises(ValueError, match="sm_103a"):
            loader.spec(loader.main_program(4), "sm_103a")
    finally:
        loader.spec.cache_clear()
    assert len(specs) == 1


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows, expected",
    [(1, 128), (127, 128), (128, 128), (129, 256), (1024, 1024), (4133, 4224)],
)
def test_ceil128_rounds_up_to_the_chunk(rows, expected):
    assert loader.ceil128(rows) == expected
    assert loader.chunk_count(rows) == expected // 128
    assert loader.chunk_capacity(rows) == expected // 128


@pytest.mark.cpu_only
def test_ceil128_rejects_non_positive_rows():
    for rows in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            loader.ceil128(rows)


@pytest.mark.cpu_only
def test_workspace_bytes_is_two_n_times_four_slots_times_ceil128_rows():
    # C_cap=8, 4 slots (3 sender + 1 result), 12 column tiles, 64 KiB tiles: 24 MiB.
    assert loader.workspace_bytes(1024, 3072, 4) == 25165824
    assert loader.workspace_bytes(1024, 3072, 4) == 2 * 3072 * 4 * 1024
    assert loader.workspace_bytes(1, 256, 4) == 4 * loader.TILE_BYTES
    assert loader.workspace_bytes(129, 256, 4) == 2 * 4 * loader.TILE_BYTES
    assert loader.TILE_BYTES == 65536
    assert loader.slots_per_chunk(4) == 4
    with pytest.raises(ValueError, match="world size"):
        loader.slots_per_chunk(8)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "max_rows, flag_bytes", [(1024, 64), (8192, 512), (65536, 4096), (1, 8)]
)
def test_flag_words_are_two_per_chunk(max_rows, flag_bytes):
    assert loader.flag_bytes(max_rows) == flag_bytes
    assert loader.flag_words(max_rows) == flag_bytes // 4
    assert loader.FLAG_WORD_BYTES_PER_CHUNK == 8


@pytest.mark.cpu_only
def test_ctl_words_follow_the_control_block_layout():
    # arr[C_cap] | red_done[C_cap] | cta_arrivals[1024] | exit | EP | steal[C_cap]
    expected_words = (
        2 * 8 + 1024 + 2 + 8
    )  # arr | red_done | cta_arrivals | exit ticket + EP | steal
    assert expected_words == 1050
    assert loader.ctl_words(8) == expected_words
    assert loader.ctl_words(1) == 1029
    assert loader.MAX_CTAS == 1024 and loader.CTL_EXTRA_WORDS == 2
    with pytest.raises(ValueError, match="positive"):
        loader.ctl_words(0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "m, n, expected",
    [
        (1, 3072, 12),
        (256, 3072, 12),
        (257, 3072, 24),
        (65536, 3072, 3072),
        (37, 256, 1),
    ],
)
def test_pair_tiles_count_two_cta_row_pairs_times_column_tiles(m, n, expected):
    assert loader.pair_tiles(m, n) == expected
    assert loader.grid_n(n) == n // 256


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "m, n, max_clusters, expected",
    [
        (1, 3072, 74, (24, 1, 1)),
        (257, 3072, 74, (48, 1, 1)),
        (65536, 3072, 74, (148, 1, 1)),
        (65536, 3072, 70, (140, 1, 1)),
        (37, 256, 74, (2, 1, 1)),
    ],
)
def test_launch_grid_is_two_ctas_per_co_resident_cluster_bounded_by_the_pair_tiles(
    m, n, max_clusters, expected
):
    assert loader.launch_grid(m, n, max_clusters=max_clusters) == expected


@pytest.mark.cpu_only
def test_launch_grid_requires_a_positive_cluster_count():
    with pytest.raises(ValueError, match="max_clusters"):
        loader.launch_grid(1, 256, max_clusters=0)


@pytest.mark.cpu_only
def test_validate_shape_enforces_the_operand_contract():
    assert loader.validate_shape(1, 256, 128) == (1, 256, 128)
    assert loader.validate_shape(4133, 3072, 768) == (4133, 3072, 768)
    assert loader.validate_shape(1 << 20, 1024, 128) == (1 << 20, 1024, 128)
    with pytest.raises(ValueError, match="M must be positive"):
        loader.validate_shape(0, 256, 128)
    for n in (128, 1000, 1281, 3073):
        with pytest.raises(ValueError, match="multiple of 256"):
            loader.validate_shape(1, n, 128)
    for k_l in (0, 64, 769):
        with pytest.raises(ValueError, match="multiple of 128"):
            loader.validate_shape(1, 256, k_l)
    with pytest.raises(ValueError, match=r"2\*\*30"):
        loader.validate_shape((1 << 20) + 1, 1024, 128)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _fake_group(world_size, rank, name="g"):
    return SimpleNamespace(group_name=name, _world_size=world_size, _rank=rank)


def _patch_distributed(
    monkeypatch,
    *,
    world_size=4,
    rank=0,
    backend_name="nccl",
    capability=(10, 0),
    symm_backend="CUDA",
):
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: backend_name)
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: group._world_size)
    monkeypatch.setattr(backend.dist, "get_rank", lambda group: group._rank)
    monkeypatch.setattr(backend.symm_mem, "get_backend", lambda device: symm_backend)
    monkeypatch.setattr(
        loader,
        "device_facts",
        lambda index: loader.DeviceFacts(
            capability=capability,
            arch=loader.CAPABILITY_ARCH.get(capability),
            sm_count=148,
        ),
    )


class _CudaLike:
    """Tensor metadata stand-in: shape, dtype, contiguity and a CUDA device without a GPU."""

    def __init__(self, shape, dtype, *, contiguous=True, index=0):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", index)
        self._contiguous = contiguous

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def is_contiguous(self):
        return self._contiguous

    def view(self, dtype):
        return ("view", dtype, self)


def _operands(m, n=N, k_l=K_L, *, index=0):
    return (
        _CudaLike((m, k_l), FP8, index=index),
        _CudaLike((m,), torch.float32, index=index),
        _CudaLike((n, k_l), FP8, index=index),
        _CudaLike((n,), torch.float32, index=index),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [1, 37, 1024, 4133])
@pytest.mark.parametrize("n", [256, 3072, 14336])
@pytest.mark.parametrize("k_l", [128, 768, 1024])
def test_validation_admits_any_rows_and_the_tile_multiples(monkeypatch, rows, n, k_l):
    _patch_distributed(monkeypatch, rank=2)
    call = backend._validate(*_operands(rows, n, k_l), _fake_group(4, 2))
    assert (call.m, call.n, call.k_l) == (rows, n, k_l)
    assert (call.rank, call.world_size, call.arch) == (2, 4, "sm_100a")
    assert (call.device_index, call.group_name) == (0, "g")


@pytest.mark.cpu_only
def test_validation_rejects_dtype_rank_and_shape_mismatches(monkeypatch):
    _patch_distributed(monkeypatch)
    group = _fake_group(4, 0)
    a, sa, w, sw = _operands(256)
    with pytest.raises(ValueError, match="a must have dtype"):
        backend._validate(_CudaLike((256, K_L), torch.bfloat16), sa, w, sw, group)
    with pytest.raises(ValueError, match="w must have dtype"):
        backend._validate(a, sa, _CudaLike((N, K_L), torch.int8), sw, group)
    with pytest.raises(ValueError, match="a_scale must have dtype"):
        backend._validate(a, _CudaLike((256,), torch.bfloat16), w, sw, group)
    with pytest.raises(ValueError, match="w_scale must be 1-D"):
        backend._validate(a, sa, w, _CudaLike((N, 1), torch.float32), group)
    with pytest.raises(ValueError, match=r"w must be \[N, K_l\]"):
        backend._validate(a, sa, _CudaLike((N, 2 * K_L), FP8), sw, group)
    with pytest.raises(ValueError, match=r"a_scale must be \[M\]"):
        backend._validate(a, _CudaLike((255,), torch.float32), w, sw, group)
    with pytest.raises(ValueError, match=r"w_scale must be \[N\]"):
        backend._validate(a, sa, w, _CudaLike((N + 1,), torch.float32), group)
    with pytest.raises(ValueError, match="multiple of 256"):
        backend._validate(*_operands(256, 1000), group)
    with pytest.raises(ValueError, match="multiple of 128"):
        backend._validate(*_operands(256, N, 700), group)
    with pytest.raises(ValueError, match=r"2\*\*30"):
        backend._validate(*_operands((1 << 20) + 1, 1024), group)


@pytest.mark.cpu_only
def test_validation_rejects_strided_and_foreign_device_operands(monkeypatch):
    _patch_distributed(monkeypatch)
    group = _fake_group(4, 0)
    a, sa, w, sw = _operands(256)
    with pytest.raises(ValueError, match="a must be contiguous"):
        backend._validate(
            _CudaLike((256, K_L), FP8, contiguous=False), sa, w, sw, group
        )
    with pytest.raises(ValueError, match="w must be contiguous"):
        backend._validate(a, sa, _CudaLike((N, K_L), FP8, contiguous=False), sw, group)
    with pytest.raises(ValueError, match="share one CUDA device"):
        backend._validate(a, sa, _CudaLike((N, K_L), FP8, index=1), sw, group)
    cpu = SimpleNamespace(
        shape=(256, K_L),
        ndim=2,
        dtype=FP8,
        device=torch.device("cpu"),
        is_contiguous=lambda: True,
    )
    with pytest.raises(ValueError, match="CUDA tensor"):
        backend._validate(cpu, sa, w, sw, group)


@pytest.mark.cpu_only
def test_validation_rejects_other_groups_world_sizes_and_backends(monkeypatch):
    _patch_distributed(monkeypatch, backend_name="gloo")
    with pytest.raises(ValueError, match="NCCL"):
        backend._validate(*_operands(256), _fake_group(4, 0))
    _patch_distributed(monkeypatch)
    for world_size in (1, 2, 8):
        with pytest.raises(ValueError, match="world size 4"):
            backend._validate(*_operands(256), _fake_group(world_size, 0))
    _patch_distributed(monkeypatch, capability=(9, 0))
    with pytest.raises(ValueError, match="requires SM100, got SM90"):
        backend._validate(*_operands(256), _fake_group(4, 0))
    _patch_distributed(monkeypatch, capability=(10, 3))
    with pytest.raises(ValueError, match="requires SM100, got SM103"):
        backend._validate(*_operands(256), _fake_group(4, 0))


@pytest.mark.cpu_only
@pytest.mark.parametrize("symm_backend", ["CUDA", "NVSHMEM", "cuda", "nvshmem"])
def test_validation_admits_the_cuda_and_nvshmem_symmetric_memory_backends(
    monkeypatch, symm_backend
):
    _patch_distributed(monkeypatch, symm_backend=symm_backend)
    device = torch.device("cuda", 0)
    assert backend.symmetric_memory_backend(device) == symm_backend.upper()
    assert backend.supports_symmetric_memory_backend(device)
    backend._validate(*_operands(256), _fake_group(4, 0))


@pytest.mark.cpu_only
@pytest.mark.parametrize("symm_backend", ["NCCL", None, "XPU"])
def test_validation_rejects_other_symmetric_memory_backends_without_selecting_one(
    monkeypatch, symm_backend
):
    _patch_distributed(monkeypatch, symm_backend=symm_backend)
    selected = []
    monkeypatch.setattr(backend.symm_mem, "set_backend", selected.append)
    device = torch.device("cuda", 0)
    assert not backend.supports_symmetric_memory_backend(device)
    with pytest.raises(ValueError, match="symmetric-memory backends"):
        backend._validate(*_operands(256), _fake_group(4, 0))
    assert selected == []


@pytest.mark.cpu_only
def test_group_name_is_required():
    with pytest.raises(ValueError, match="group_name"):
        backend._group_name(SimpleNamespace())


def _call(m=256, n=N, k_l=K_L, *, rank=1, index=0):
    return backend._Call(index, rank, 4, "g", "sm_100a", m, n, k_l)


@pytest.mark.cpu_only
def test_out_must_match_the_result_contract():
    call = _call(250)
    like = _CudaLike((250, K_L), FP8)
    with pytest.raises(ValueError, match="bfloat16"):
        backend._output(_CudaLike((250, N), torch.float16), call, like)
    with pytest.raises(ValueError, match=r"\[250, 3072\]"):
        backend._output(_CudaLike((256, N), torch.bfloat16), call, like)
    with pytest.raises(ValueError, match=r"\[250, 3072\]"):
        backend._output(
            _CudaLike((250, N), torch.bfloat16, contiguous=False), call, like
        )
    with pytest.raises(ValueError, match="device"):
        backend._output(_CudaLike((250, N), torch.bfloat16, index=1), call, like)
    out = _CudaLike((250, N), torch.bfloat16)
    assert backend._output(out, call, like) is out


# --------------------------------------------------------------------------- #
# Entry points and dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_backend_entrypoints_reject_non_cake_routes():
    with pytest.raises(ValueError, match="exactly 'cake'"):
        backend.fp8_rowwise_gemm_allreduce_cake(
            *_operands(256), _fake_group(4, 0), backend="auto"
        )
    with pytest.raises(ValueError, match="exactly 'cake'"):
        backend.prepare_fp8_rowwise_gemm_allreduce_cake(
            *_operands(256), _fake_group(4, 0), backend="auto"
        )


@pytest.mark.cpu_only
def test_public_entrypoint_routes_auto_and_cake_to_the_backend(monkeypatch):
    dispatcher = importlib.import_module(
        "flashinfer.comm.fp8_rowwise_gemm_allreduce.fp8_rowwise_gemm_allreduce"
    )
    calls = []
    result = object()

    def fake_backend(a, a_scale, w, w_scale, group, *, backend, out):
        calls.append((a, a_scale, w, w_scale, group, backend, out))
        return result

    monkeypatch.setattr(backend, "fp8_rowwise_gemm_allreduce_cake", fake_backend)
    a, sa, w, sw, group, out = (object() for _ in range(6))
    assert dispatcher.fp8_rowwise_gemm_allreduce(a, sa, w, sw, group) is result
    assert (
        dispatcher.fp8_rowwise_gemm_allreduce(
            a, sa, w, sw, group, backend="cake", out=out
        )
        is result
    )
    assert calls == [
        (a, sa, w, sw, group, "cake", None),
        (a, sa, w, sw, group, "cake", out),
    ]
    with pytest.raises(ValueError, match="'auto' or 'cake'"):
        dispatcher.fp8_rowwise_gemm_allreduce(a, sa, w, sw, group, backend="trtllm")


@pytest.mark.cpu_only
def test_public_prepare_forwards_the_row_capacity(monkeypatch):
    dispatcher = importlib.import_module(
        "flashinfer.comm.fp8_rowwise_gemm_allreduce.fp8_rowwise_gemm_allreduce"
    )
    calls = []
    launcher = object()

    def fake_prepare(a, a_scale, w, w_scale, group, *, backend, max_rows):
        calls.append((a, a_scale, w, w_scale, group, backend, max_rows))
        return launcher

    monkeypatch.setattr(
        backend, "prepare_fp8_rowwise_gemm_allreduce_cake", fake_prepare
    )
    a, sa, w, sw, group = (object() for _ in range(5))
    assert (
        dispatcher.prepare_fp8_rowwise_gemm_allreduce(a, sa, w, sw, group) is launcher
    )
    assert (
        dispatcher.prepare_fp8_rowwise_gemm_allreduce(
            a, sa, w, sw, group, backend="cake", max_rows=4133
        )
        is launcher
    )
    assert calls == [
        (a, sa, w, sw, group, "cake", None),
        (a, sa, w, sw, group, "cake", 4133),
    ]
    with pytest.raises(ValueError, match="'auto' or 'cake'"):
        dispatcher.prepare_fp8_rowwise_gemm_allreduce(a, sa, w, sw, group, backend="x")


@pytest.mark.cpu_only
def test_package_exports_the_public_names():
    package = importlib.import_module("flashinfer.comm.fp8_rowwise_gemm_allreduce")
    comm = importlib.import_module("flashinfer.comm")
    dispatcher = importlib.import_module(
        "flashinfer.comm.fp8_rowwise_gemm_allreduce.fp8_rowwise_gemm_allreduce"
    )
    assert package.fp8_rowwise_gemm_allreduce is dispatcher.fp8_rowwise_gemm_allreduce
    assert comm.fp8_rowwise_gemm_allreduce is dispatcher.fp8_rowwise_gemm_allreduce
    assert (
        comm.prepare_fp8_rowwise_gemm_allreduce
        is dispatcher.prepare_fp8_rowwise_gemm_allreduce
    )


@pytest.mark.cpu_only
def test_public_entrypoints_carry_the_flashinfer_api_decorator():
    dispatcher = importlib.import_module(
        "flashinfer.comm.fp8_rowwise_gemm_allreduce.fp8_rowwise_gemm_allreduce"
    )
    decorated = set()
    for node in ast.parse(inspect.getsource(dispatcher)).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        names = set()
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            names.add(getattr(target, "id", getattr(target, "attr", None)))
        if "flashinfer_api" in names:
            decorated.add(node.name)
    assert decorated == {
        "fp8_rowwise_gemm_allreduce",
        "prepare_fp8_rowwise_gemm_allreduce",
    }


# --------------------------------------------------------------------------- #
# Symmetric memory: rendezvous guard and workspace allocation
# --------------------------------------------------------------------------- #


class _FakeHandle:
    """Symmetric-memory handle stand-in: topology, sizes and CPU peer buffers / pads."""

    def __init__(self, *, rank, world_size, buffer_size, signal_pad_size=0):
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self.signal_pad_size = signal_pad_size
        self.pads = {}

    def get_buffer(self, peer, shape, dtype, offset):
        assert offset == 0
        return torch.empty(shape, dtype=dtype)

    def get_signal_pad(self, peer, shape, dtype, offset):
        assert offset == 0 and dtype == torch.uint32
        if peer not in self.pads:
            self.pads[peer] = torch.full(shape, 7, dtype=dtype)
        return self.pads[peer]


@pytest.mark.cpu_only
def test_rendezvous_guard_checks_topology_and_size(monkeypatch):
    tensor = torch.empty(64, dtype=torch.uint8)
    handles = []
    monkeypatch.setattr(backend.symm_mem, "rendezvous", lambda t, group: handles.pop(0))
    good = _FakeHandle(rank=1, world_size=4, buffer_size=64)
    handles.append(good)
    assert backend._rendezvous(_call(), tensor, "receive workspace") is good
    handles.append(_FakeHandle(rank=0, world_size=4, buffer_size=64))
    with pytest.raises(RuntimeError, match="topology"):
        backend._rendezvous(_call(), tensor, "receive workspace")
    handles.append(_FakeHandle(rank=1, world_size=4, buffer_size=63))
    with pytest.raises(backend.StaleSymmetricHandleError, match="pytorch#192579"):
        backend._rendezvous(_call(), tensor, "receive workspace")
    assert issubclass(backend.StaleSymmetricHandleError, RuntimeError)


def _patch_allocation(monkeypatch, *, handles, max_clusters=74, loaded=None):
    """CPU stand-ins for every device-side step of ``_allocate_workspace``."""

    events = []
    allocated = []

    def empty(numel, *, dtype, device):
        tensor = torch.empty(numel, dtype=dtype)
        allocated.append((numel, dtype, device))
        return tensor

    def rendezvous(call, tensor, what):
        events.append(("rendezvous", what))
        return handles.pop(0)

    monkeypatch.setattr(backend.symm_mem, "empty", empty)
    monkeypatch.setattr(backend, "_rendezvous", rendezvous)
    monkeypatch.setattr(
        backend.torch.cuda, "device", lambda index: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(
        backend.torch.cuda, "synchronize", lambda device: events.append(("sync",))
    )
    monkeypatch.setattr(
        backend.dist, "barrier", lambda group: events.append(("barrier", group))
    )
    monkeypatch.setattr(
        loader,
        "load",
        lambda program, arch: events.append(("load", program, arch)) or loaded,
    )
    monkeypatch.setattr(
        loader,
        "cluster_capacity",
        lambda index, program: events.append(("capacity", index, program))
        or max_clusters,
    )
    return events, allocated


@pytest.mark.cpu_only
def test_allocation_places_the_readiness_words_in_the_signal_pad_when_they_fit(
    monkeypatch,
):
    s_bytes = loader.workspace_bytes(1024, N, 4)
    handle = _FakeHandle(
        rank=1, world_size=4, buffer_size=s_bytes, signal_pad_size=65536
    )
    module = object()
    events, allocated = _patch_allocation(monkeypatch, handles=[handle], loaded=module)
    group = _fake_group(4, 1)
    workspace = backend._allocate_workspace(_call(1024), group, max_rows=1024)
    program = loader.main_program(4)
    assert events == [
        ("load", program, "sm_100a"),
        ("rendezvous", "receive workspace"),
        ("capacity", 0, program),
        ("sync",),
        ("barrier", group),
    ]
    assert allocated == [(s_bytes, torch.uint8, torch.device("cuda", 0))]
    assert workspace.module is module and workspace.program == program
    assert (workspace.rank, workspace.world_size, workspace.n) == (1, 4, N)
    assert (workspace.max_rows, workspace.rows_capacity) == (1024, 1024)
    assert (workspace.chunk_capacity, workspace.slots, workspace.grid_n) == (8, 4, 12)
    assert workspace.s_bytes == s_bytes == workspace.s.numel()
    assert workspace.flags_in_signal_pad and workspace.flags_buffer is None
    assert workspace.flags_handle is None and workspace.signal_pad_size == 65536
    # This rank's pad words are zeroed; the peers' pads are not touched.
    assert workspace.flags is handle.pads[1]
    assert workspace.flags.numel() == 16 and int(workspace.flags.sum()) == 0
    assert all(int(handle.pads[peer].sum()) == 7 * 16 for peer in (0, 2, 3))
    assert len(workspace.s_peers) == 4 and len(workspace.flags_peers) == 4
    assert workspace.ctl.dtype == torch.uint32
    assert workspace.ctl.numel() == loader.ctl_words(8) == 1050
    assert int(workspace.ctl.sum()) == 0
    assert workspace.max_clusters == 74 and workspace.captured is False


@pytest.mark.cpu_only
def test_allocation_uses_a_dedicated_symmetric_buffer_when_the_pad_is_too_small(
    monkeypatch,
):
    s_bytes = loader.workspace_bytes(8192, N, 4)
    words = loader.flag_words(8192)
    s_handle = _FakeHandle(
        rank=0, world_size=4, buffer_size=s_bytes, signal_pad_size=256
    )
    flag_handle = _FakeHandle(rank=0, world_size=4, buffer_size=4 * words)
    events, allocated = _patch_allocation(monkeypatch, handles=[s_handle, flag_handle])
    group = _fake_group(4, 0)
    workspace = backend._allocate_workspace(_call(8192, rank=0), group, max_rows=8192)
    assert [event for event in events if event[0] == "rendezvous"] == [
        ("rendezvous", "receive workspace"),
        ("rendezvous", "readiness words"),
    ]
    assert allocated == [
        (s_bytes, torch.uint8, torch.device("cuda", 0)),
        (words, torch.uint32, torch.device("cuda", 0)),
    ]
    assert not workspace.flags_in_signal_pad
    assert workspace.flags is workspace.flags_buffer
    assert workspace.s_handle is s_handle and workspace.flags_handle is flag_handle
    assert workspace.flags.numel() == words == 128 and int(workspace.flags.sum()) == 0
    assert workspace.ctl.numel() == loader.ctl_words(64)
    assert events[-2:] == [("sync",), ("barrier", group)]


@pytest.mark.cpu_only
def test_allocation_refuses_inside_capture_and_an_oversubscribed_grid(monkeypatch):
    s_bytes = loader.workspace_bytes(128, N, 4)
    handle = _FakeHandle(
        rank=1, world_size=4, buffer_size=s_bytes, signal_pad_size=65536
    )
    events, _ = _patch_allocation(monkeypatch, handles=[handle], max_clusters=513)
    with pytest.raises(RuntimeError, match="per-CTA arrival words"):
        backend._allocate_workspace(_call(128), _fake_group(4, 1), max_rows=128)
    assert ("barrier", _fake_group(4, 1)) not in events
    monkeypatch.setattr(backend.torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside CUDA-graph capture"):
        backend._allocate_workspace(_call(128), _fake_group(4, 1), max_rows=128)


# --------------------------------------------------------------------------- #
# One-shot workspace cache, prepared launcher, launch
# --------------------------------------------------------------------------- #


def _workspace(*, max_rows, n=N, rank=1, max_clusters=74, program=None, module=None):
    c_cap = loader.chunk_capacity(max_rows)
    return backend._Workspace(
        rank=rank,
        world_size=4,
        device_index=0,
        arch="sm_100a",
        n=n,
        program=program or loader.main_program(4),
        module=module,
        max_rows=max_rows,
        rows_capacity=loader.ceil128(max_rows),
        chunk_capacity=c_cap,
        slots=4,
        grid_n=loader.grid_n(n),
        s_bytes=loader.workspace_bytes(max_rows, n, 4),
        s="S",
        s_handle=None,
        s_peers="S_peers",
        flags="flags",
        flags_peers="flags_peers",
        flags_in_signal_pad=True,
        signal_pad_size=65536,
        flags_buffer=None,
        flags_handle=None,
        ctl="ctl",
        max_clusters=max_clusters,
    )


@pytest.mark.cpu_only
def test_one_shot_cache_serves_a_fitting_workspace_without_a_collective(monkeypatch):
    group = _fake_group(4, 1)
    call = _call(1000)
    key = (0, id(group), "g", N)
    state = backend._LaunchState(rank=1, world_size=4)
    workspace = _workspace(max_rows=1024)
    monkeypatch.setitem(backend._WORKSPACES, key, workspace)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setitem(backend._LAUNCH_STATES, (0, id(group), "g"), state)

    def no_collective(*args, **kwargs):
        raise AssertionError("no allocation expected")

    monkeypatch.setattr(backend, "_allocate_workspace", no_collective)
    assert backend._prepare_one_shot(call, group) == (state, workspace)
    assert backend._workspace_of(group, 0, N) is workspace
    with pytest.raises(
        RuntimeError, match="no fused FP8 GEMM \\+ all-reduce workspace"
    ):
        backend._workspace_of(group, 0, 2 * N)


@pytest.mark.cpu_only
def test_one_shot_cache_refuses_to_grow_a_captured_workspace_before_any_collective(
    monkeypatch,
):
    group = _fake_group(4, 1)
    key = (0, id(group), "g", N)
    state = backend._LaunchState(rank=1, world_size=4)
    workspace = _workspace(max_rows=128)
    workspace.captured = True
    monkeypatch.setitem(backend._WORKSPACES, key, workspace)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setitem(backend._LAUNCH_STATES, (0, id(group), "g"), state)

    def no_collective(*args, **kwargs):
        raise AssertionError("no allocation expected")

    monkeypatch.setattr(backend, "_allocate_workspace", no_collective)
    with pytest.raises(RuntimeError, match="captured into a CUDA graph"):
        backend._prepare_one_shot(_call(129), group)
    # A refused growth is a caller error; the launch state stays usable.
    assert state.poisoned is False
    assert backend._prepare_one_shot(_call(128), group) == (state, workspace)


@pytest.mark.cpu_only
def test_one_shot_growth_retires_the_previous_workspace_and_poisons_on_failure(
    monkeypatch,
):
    group = _fake_group(4, 1)
    key = (0, id(group), "g", N)
    state = backend._LaunchState(rank=1, world_size=4)
    old = _workspace(max_rows=128)
    grown = _workspace(max_rows=256)
    monkeypatch.setitem(backend._WORKSPACES, key, old)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setitem(backend._LAUNCH_STATES, (0, id(group), "g"), state)
    monkeypatch.setattr(backend, "_RETIRED", [])
    synced = []
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda index: SimpleNamespace(synchronize=lambda: synced.append(index)),
    )
    requests = []

    def allocate(call, g, *, max_rows):
        requests.append((call.m, g, max_rows))
        if max_rows > 256:
            raise RuntimeError("peer went away")
        return grown

    monkeypatch.setattr(backend, "_allocate_workspace", allocate)
    assert backend._prepare_one_shot(_call(200), group) == (state, grown)
    assert requests == [(200, group, 256)] and synced == [0]
    assert backend._WORKSPACES[key] is grown and list(backend._RETIRED) == [old]
    with pytest.raises(RuntimeError, match="peer went away"):
        backend._prepare_one_shot(_call(300), group)
    assert state.poisoned and "peer went away" in state.poison_reason
    with pytest.raises(RuntimeError, match="poisoned"):
        backend._prepare_one_shot(_call(1), group)
    # The grown generation is still published; the failed one never was.
    assert backend._WORKSPACES[key] is grown


@pytest.mark.cpu_only
def test_launch_state_is_cached_per_device_and_group(monkeypatch):
    group = _fake_group(4, 2)
    monkeypatch.setattr(backend, "_LAUNCH_STATES", {})
    first = backend._launch_state(_call(rank=2), group)
    assert backend._launch_state(_call(rank=2), group) is first
    assert (first.rank, first.world_size, first.calls) == (2, 4, 0)
    with pytest.raises(RuntimeError, match="topology changed"):
        backend._launch_state(_call(rank=3), group)


@pytest.mark.cpu_only
def test_prepared_launcher_serves_every_row_count_up_to_its_capacity(monkeypatch):
    _patch_distributed(monkeypatch, rank=3)
    group = _fake_group(4, 3)
    sample = _operands(1)
    call = backend._validate(*sample, group)
    launcher = backend._PreparedLauncher(
        group=group,
        group_id=id(group),
        call=call,
        max_rows=4133,
        device=sample[0].device,
        state=backend._LaunchState(rank=3, world_size=4),
        workspace=_workspace(max_rows=4133, rank=3),
    )
    assert (launcher.n, launcher.k_l, launcher.max_rows) == (N, K_L, 4133)
    assert launcher.workspace_bytes == loader.workspace_bytes(4133, N, 4)
    assert launcher.flags_in_signal_pad and launcher.max_active_clusters == 74
    for rows in (1, 129, 4133):
        bound = launcher._validate_call(*_operands(rows))
        assert (bound.m, bound.n, bound.k_l) == (rows, N, K_L)
    with pytest.raises(ValueError, match=r"\[capacity_exceeded\] M=4134"):
        launcher._validate_call(*_operands(4134))
    with pytest.raises(ValueError, match=r"\[N_mismatch\]"):
        launcher._validate_call(*_operands(1, 2 * N))
    with pytest.raises(ValueError, match=r"\[K_l_mismatch\]"):
        launcher._validate_call(*_operands(1, N, 2 * K_L))
    with pytest.raises(ValueError, match="contiguous"):
        a, sa, w, sw = _operands(1)
        launcher._validate_call(_CudaLike((1, K_L), FP8, contiguous=False), sa, w, sw)
    with pytest.raises(ValueError, match="device changed"):
        launcher._validate_call(*_operands(1, index=1))


@pytest.mark.cpu_only
def test_prepare_requires_the_capacity_to_cover_the_sample(monkeypatch):
    _patch_distributed(monkeypatch)
    with pytest.raises(ValueError, match="max_rows=100 is smaller"):
        backend.prepare_fp8_rowwise_gemm_allreduce_cake(
            *_operands(256), _fake_group(4, 0), backend="cake", max_rows=100
        )


@pytest.mark.cpu_only
def test_capacity_agreement_names_every_rank_that_differs(monkeypatch):
    table = {0: 4133, 1: 4133, 2: 4133, 3: 4160}

    def fake_all_gather(outputs, value, group):
        assert group.group_name == "g" and value.dtype == torch.int64
        for rank, output in enumerate(outputs):
            output.copy_(torch.tensor([table[rank]], dtype=torch.int64))

    monkeypatch.setattr(backend.dist, "all_gather", fake_all_gather)
    cpu = torch.device("cpu")
    with pytest.raises(ValueError, match="rank 2: 4133, rank 3: 4160"):
        backend._agree_capacity(_fake_group(4, 2), 4133, world_size=4, device=cpu)
    table[3] = 4133
    backend._agree_capacity(_fake_group(4, 2), 4133, world_size=4, device=cpu)


@pytest.mark.cpu_only
def test_prepare_agrees_the_capacity_before_any_allocation(monkeypatch):
    _patch_distributed(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(backend, "_LAUNCH_STATES", {})
    agreed = []

    def disagree(group, capacity, *, world_size, device):
        agreed.append((capacity, world_size, device))
        raise ValueError("max_rows must be identical on every rank of the group")

    monkeypatch.setattr(backend, "_agree_capacity", disagree)
    monkeypatch.setattr(
        backend, "_allocate_workspace", lambda *args, **kwargs: pytest.fail("allocated")
    )
    with pytest.raises(ValueError, match="identical on every rank"):
        backend.prepare_fp8_rowwise_gemm_allreduce_cake(
            *_operands(256), _fake_group(4, 0), backend="cake", max_rows=4133
        )
    assert agreed == [(4133, 4, torch.device("cuda", 0))]
    assert backend._LAUNCH_STATES == {}


@pytest.mark.cpu_only
def test_launch_refuses_a_workspace_smaller_than_the_call_without_poisoning():
    state = backend._LaunchState(rank=1, world_size=4)
    with pytest.raises(RuntimeError, match="holds 128 rows, the call needs 256"):
        backend._launch(state, _workspace(max_rows=128), _call(129), *([None] * 5))
    with pytest.raises(RuntimeError, match="another N"):
        backend._launch(state, _workspace(max_rows=128), _call(1, 2 * N), *([None] * 5))
    state.calls = backend.EPOCH_MAX_CALLS
    with pytest.raises(RuntimeError, match="epoch exhausted"):
        backend._launch(state, _workspace(max_rows=128), _call(1), *([None] * 5))
    assert state.poisoned is False
    assert backend.EPOCH_MAX_CALLS == 2**32 - 2


class _Module:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def run(self, *args):
        self.calls.append(args)
        if self.error is not None:
            raise self.error


def _launch_fixture(monkeypatch, *, capturing=False, error=None):
    module = _Module(error)
    workspace = _workspace(max_rows=1024, module=module)
    state = backend._LaunchState(rank=1, world_size=4)
    tails = []
    monkeypatch.setattr(backend, "_current_stream_handle", lambda index: 4242)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: capturing
    )
    monkeypatch.setattr(
        backend,
        "_remember_tail",
        lambda state, handle, index: tails.append((handle, index)),
    )
    monkeypatch.setattr(backend.tvm_ffi, "use_torch_stream", contextlib.nullcontext)
    return state, workspace, module, tails


@pytest.mark.cpu_only
def test_launch_binds_every_argument_by_name_in_the_recorded_order(monkeypatch):
    state, workspace, module, tails = _launch_fixture(monkeypatch)
    a, sa, w, sw = _operands(1000)
    out = _CudaLike((1000, N), torch.bfloat16)
    backend._launch(state, workspace, _call(1000), a, sa, w, sw, out)
    names = loader.launcher_args(workspace.program)
    assert len(module.calls) == 1 and len(module.calls[0]) == len(names)
    bound = dict(zip(names, module.calls[0], strict=True))
    assert bound["A"] == ("view", torch.uint8, a) and bound["W"] == (
        "view",
        torch.uint8,
        w,
    )
    assert bound["OUT"] is out and bound["out_ptr"] is out
    assert bound["sa"] is sa and bound["sw"] is sw
    assert bound["ctl"] == "ctl" and bound["S_mc"] == "ctl"
    assert bound["S"] == "S" and bound["S_peers"] == "S_peers"
    assert bound["flags"] == "flags"
    assert bound["flags_peers"] == "flags_peers" and bound["pg_flags"] == "flags_peers"
    assert (bound["M"], bound["N"], bound["K"]) == (1000, N, K_L)
    assert (bound["grid_n"], bound["pair_tiles"], bound["K_tiles"]) == (12, 48, 6)
    assert (bound["C"], bound["C_cap"]) == (8, 8)
    assert bound["S_bytes"] == loader.workspace_bytes(1024, N, 4)
    assert bound["flag_words"] == 16
    assert (bound["pg_world"], bound["pg_rank"]) == (4, 1)
    # 48 pair tiles on 74 co-resident clusters: 96 CTAs.
    assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (96, 1, 1)
    assert state.calls == 1 and state.captured_calls == 0
    assert tails == [(4242, 0)] and workspace.captured is False


@pytest.mark.cpu_only
def test_captured_launch_marks_the_workspace_and_skips_the_tail(monkeypatch):
    state, workspace, module, tails = _launch_fixture(monkeypatch, capturing=True)
    a, sa, w, sw = _operands(37)
    out = _CudaLike((37, N), torch.bfloat16)
    backend._launch(state, workspace, _call(37), a, sa, w, sw, out)
    assert len(module.calls) == 1
    assert workspace.captured is True
    assert (state.calls, state.captured_calls) == (1, 1)
    assert tails == []


@pytest.mark.cpu_only
def test_launch_failure_poisons_the_state(monkeypatch):
    state, workspace, module, _ = _launch_fixture(
        monkeypatch, error=RuntimeError("boom")
    )
    a, sa, w, sw = _operands(1)
    out = _CudaLike((1, N), torch.bfloat16)
    with pytest.raises(RuntimeError, match="boom"):
        backend._launch(state, workspace, _call(1), a, sa, w, sw, out)
    assert state.poisoned and "boom" in state.poison_reason
    assert state.calls == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        backend._launch(state, workspace, _call(1), a, sa, w, sw, out)
    assert len(module.calls) == 1


@pytest.mark.cpu_only
def test_launch_refuses_unknown_launcher_arguments(monkeypatch):
    state, workspace, module, _ = _launch_fixture(monkeypatch)
    monkeypatch.setattr(
        loader,
        "launcher_args",
        lambda program: ("A", "mystery", "grid_x", "grid_y", "grid_z"),
    )
    a, sa, w, sw = _operands(1)
    with pytest.raises(RuntimeError, match=r"unknown arguments \['mystery'\]"):
        backend._launch(
            state, workspace, _call(1), a, sa, w, sw, _CudaLike((1, N), torch.bfloat16)
        )
    assert module.calls == [] and state.poisoned


@pytest.mark.cpu_only
def test_cross_stream_join_waits_on_the_previous_tail_and_skips_capture(monkeypatch):
    state = backend._LaunchState(rank=1, world_size=4)
    waited = []
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda index: SimpleNamespace(wait_stream=waited.append, cuda_stream=4242),
    )
    assert backend._join_previous_tail(state, 4242, 0, capturing=False) is False
    previous = SimpleNamespace(cuda_stream=4242)
    state.tail_stream = previous
    assert backend._join_previous_tail(state, 4242, 0, capturing=False) is False
    other = SimpleNamespace(cuda_stream=17)
    state.tail_stream = other
    assert backend._join_previous_tail(state, 4242, 0, capturing=True) is False
    assert backend._join_previous_tail(state, 4242, 0, capturing=False) is True
    assert waited == [other]
    backend._remember_tail(state, 4242, 0)
    assert state.tail_handle == 4242 and state.tail_stream.cuda_stream == 4242
