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

Behavioural tests of the Cake all-gather matmul backend that need no GPU:
route-table coverage, launch geometry, input validation and the public
dispatch. The multi-GPU numerical test lives in
``test_all_gather_matmul_cake_e2e.py``.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from flashinfer.jit import cake_all_gather_matmul as loader

backend = importlib.import_module(
    "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul"
)


def test_route_table_covers_every_world_size_dtype_and_phase():
    for phase in (0, 1):
        assert loader.barrier_program(phase) in loader.PROGRAMS
    assert loader.barrier_program(0) != loader.barrier_program(1)
    for world_size in loader.SUPPORTED_WORLD_SIZES:
        for dtype_name in loader.SUPPORTED_DTYPES.values():
            assert loader.main_program(world_size, dtype_name) in loader.PROGRAMS
    assert loader.fused_peer_copy_program() in loader.PROGRAMS
    assert set(loader.ROUTES.values()) == set(loader.PROGRAMS)


def test_every_program_lists_one_device_and_one_binding_source():
    for program, row in loader.PROGRAMS.items():
        assert len(row["sources"]) == 2, program
        assert all(
            source.startswith("csrc/cake_all_gather_matmul/")
            for source in row["sources"]
        )
        assert len(row["block"]) == 3 and row["block"][0] > 0
        assert row["dynamic_smem_bytes"] >= 0
        assert row["arches"] and set(row["arches"]) <= set(loader.ARCH_FLAGS), program


def test_fused_copy_program_is_delivered_for_its_routed_architecture_only():
    fused = loader.fused_peer_copy_program()
    assert loader.PROGRAMS[fused]["arches"] == ["sm_103a"]
    for phase in (0, 1):
        assert loader.PROGRAMS[loader.barrier_program(phase)]["arches"] == [
            "sm_100a",
            "sm_103a",
        ]
    for world_size in loader.SUPPORTED_WORLD_SIZES:
        for dtype_name in loader.SUPPORTED_DTYPES.values():
            assert loader.PROGRAMS[loader.main_program(world_size, dtype_name)][
                "arches"
            ] == ["sm_100a", "sm_103a"]
    loader.spec.cache_clear()
    try:
        with pytest.raises(ValueError, match="sm_100a"):
            loader.spec(fused, "sm_100a")
    finally:
        loader.spec.cache_clear()


def test_barrier_and_fused_copy_programs_use_static_shared_memory_only():
    for phase in (0, 1):
        program = loader.barrier_program(phase)
        assert loader.launch_block(program) == (32, 1, 1)
        assert loader.dynamic_smem_bytes(program) == 0
    fused = loader.fused_peer_copy_program()
    assert loader.launch_block(fused) == (128, 1, 1)
    assert loader.dynamic_smem_bytes(fused) == 0


def test_main_programs_share_one_block_and_dynamic_shared_memory_contract():
    blocks = set()
    smem = set()
    for world_size in loader.SUPPORTED_WORLD_SIZES:
        for dtype_name in loader.SUPPORTED_DTYPES.values():
            program = loader.main_program(world_size, dtype_name)
            blocks.add(loader.launch_block(program))
            smem.add(loader.dynamic_smem_bytes(program))
    assert blocks == {(loader.MAIN_THREADS, 1, 1)}
    assert len(smem) == 1 and smem.pop() > 0


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (128, (128, 1)),
        (2432, (2432, 1)),
        (2560, (2432, 2)),
        (16384, (2432, 7)),
        (19456, (2432, 8)),
    ],
)
def test_chunk_plan_pushes_at_most_nineteen_row_blocks_per_chunk(rows, expected):
    assert loader.chunk_plan(rows) == expected


@pytest.mark.parametrize(
    ("rows", "n", "partitions", "expected"),
    [
        (16384, 2048, 1, (19 * 8, 1, 1)),
        (512, 2048, 1, (4 * 8, 1, 1)),
        (512, 1280, 4, (4 * 5, 4, 1)),
        (512, 2560, 1, (4 * 10, 1, 1)),
    ],
)
def test_main_grid_covers_the_first_chunk_tiles(rows, n, partitions, expected):
    assert loader.main_grid(rows, n, peer_partitions=partitions) == expected


@pytest.mark.parametrize(
    ("arch", "dtype_name", "world_size", "rows", "n", "expected"),
    [
        ("sm_103a", "bfloat16", 8, 512, 1280, True),
        ("sm_100a", "bfloat16", 8, 512, 1280, False),
        ("sm_103a", "float16", 8, 512, 1280, False),
        ("sm_103a", "bfloat16", 4, 512, 2560, False),
        ("sm_103a", "bfloat16", 8, 1024, 1280, False),
        ("sm_103a", "bfloat16", 8, 512, 2048, False),
    ],
)
def test_fused_peer_copy_is_the_exact_sm103_tp8_packed_qkv_route(
    arch, dtype_name, world_size, rows, n, expected
):
    assert (
        loader.uses_fused_peer_copy(
            arch=arch, dtype_name=dtype_name, world_size=world_size, rows=rows, n=n
        )
        is expected
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_barrier_flag_pad_has_two_epochs_and_two_mailbox_banks_per_phase(world_size):
    assert loader.barrier_flag_words(world_size) == 2 + 4 * world_size


def test_spec_names_carry_the_exact_architecture(monkeypatch):
    # Build-target discovery reads FLASHINFER_CUDA_ARCH_LIST when set; pin it so
    # the test does not depend on the visible devices of the test host.
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a 10.3a")
    loader.spec.cache_clear()
    try:
        program = loader.main_program(2, "bfloat16")
        for arch in loader.PROGRAMS[program]["arches"]:
            spec = loader.spec(program, arch)
            assert spec.name == f"{program}_{arch}"
            assert loader.ARCH_FLAGS[arch][0] in spec.extra_cuda_cflags
        assert loader.spec(program, "sm_100a") is not loader.spec(program, "sm_103a")
    finally:
        loader.spec.cache_clear()


def _fake_group(world_size, rank, name="fake_group"):
    return SimpleNamespace(group_name=name, _world_size=world_size, _rank=rank)


def _patch_distributed(
    monkeypatch, *, world_size, rank, backend_name="nccl", arch="sm_100a"
):
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: backend_name)
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: group._world_size)
    monkeypatch.setattr(backend.dist, "get_rank", lambda group: group._rank)
    monkeypatch.setattr(backend.symm_mem, "get_backend", lambda device: "NVSHMEM")
    monkeypatch.setattr(
        loader,
        "device_facts",
        lambda index: loader.DeviceFacts(
            capability={"sm_100a": (10, 0), "sm_103a": (10, 3)}[arch],
            arch=arch,
            sm_count=148,
        ),
    )


def _meta_pair(rows, n, dtype=torch.bfloat16):
    inp = torch.empty(rows, 8192, dtype=dtype, device="meta")
    w = torch.empty(8192, n, dtype=dtype, device="meta")
    return inp, w


class _CudaLike:
    """Tensor metadata stand-in: shape, stride, dtype and a CUDA device without a GPU."""

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


def test_validation_admits_the_exported_widths_per_world_size(monkeypatch):
    for world_size, widths in backend.SUPPORTED_N_BY_WORLD_SIZE.items():
        _patch_distributed(monkeypatch, world_size=world_size, rank=0)
        for n in widths:
            call = backend._validate(
                _CudaLike((256, 8192), torch.bfloat16),
                _CudaLike((8192, n), torch.bfloat16),
                _fake_group(world_size, 0),
                packed_qkv_only=False,
            )
            assert (call.world_size, call.n, call.arch) == (world_size, n, "sm_100a")


@pytest.mark.parametrize(
    ("world_size", "n"),
    [(2, 1280), (2, 2560), (4, 1280), (4, 2560), (8, 1280), (8, 4096)],
)
def test_one_shot_validation_rejects_widths_outside_the_exported_routes(
    monkeypatch, world_size, n
):
    _patch_distributed(monkeypatch, world_size=world_size, rank=0)
    with pytest.raises(ValueError, match="N supported by"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, n), torch.bfloat16),
            _fake_group(world_size, 0),
            packed_qkv_only=False,
        )


def test_validation_rejects_strided_views_instead_of_copying(monkeypatch):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(ValueError, match="contiguous"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16, contiguous=False),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
            packed_qkv_only=False,
        )


def test_validation_rejects_rows_that_are_not_a_multiple_of_128(monkeypatch):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(ValueError, match="multiple of 128"):
        backend._validate(
            _CudaLike((200, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
            packed_qkv_only=False,
        )


def test_validation_rejects_unsupported_world_sizes_and_backends(monkeypatch):
    _patch_distributed(monkeypatch, world_size=3, rank=0)
    with pytest.raises(ValueError, match="world size 2, 4, or 8"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(3, 0),
            packed_qkv_only=False,
        )
    _patch_distributed(monkeypatch, world_size=2, rank=0, backend_name="gloo")
    with pytest.raises(ValueError, match="NCCL"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
            packed_qkv_only=False,
        )


def test_validation_rejects_devices_outside_the_exported_architectures(monkeypatch):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    monkeypatch.setattr(
        loader,
        "device_facts",
        lambda index: loader.DeviceFacts(capability=(9, 0), arch=None, sm_count=132),
    )
    with pytest.raises(ValueError, match="SM100 or SM103"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
            packed_qkv_only=False,
        )


@pytest.mark.parametrize(
    ("arch", "world_size", "n", "dtype", "ok"),
    [
        ("sm_100a", 8, 1280, torch.bfloat16, True),
        ("sm_103a", 8, 1280, torch.bfloat16, True),
        ("sm_103a", 4, 2560, torch.bfloat16, True),
        ("sm_100a", 4, 2560, torch.bfloat16, False),
        ("sm_100a", 8, 2048, torch.bfloat16, False),
        ("sm_103a", 4, 1280, torch.bfloat16, False),
        ("sm_100a", 2, 2048, torch.bfloat16, False),
        ("sm_103a", 8, 1280, torch.float16, False),
    ],
)
def test_prepared_packed_qkv_route_is_bf16_tp8_n1280_or_sm103_tp4_n2560(
    monkeypatch, arch, world_size, n, dtype, ok
):
    _patch_distributed(monkeypatch, world_size=world_size, rank=0, arch=arch)
    inp = _CudaLike((512, 8192), dtype)
    w = _CudaLike((8192, n), dtype)
    if ok:
        call = backend._validate(
            inp, w, _fake_group(world_size, 0), packed_qkv_only=True
        )
        assert (call.world_size, call.n, call.rows) == (world_size, n, 512)
    else:
        with pytest.raises(ValueError):
            backend._validate(inp, w, _fake_group(world_size, 0), packed_qkv_only=True)


def test_out_must_match_the_gathered_output_contract():
    call = backend._Call(0, 0, 2, "g", "bfloat16", "sm_100a", 256, 2048)
    like = _CudaLike((256, 8192), torch.bfloat16)
    with pytest.raises(ValueError, match=r"\[512, 2048\]"):
        backend._output(_CudaLike((256, 2048), torch.bfloat16), call, like)
    with pytest.raises(ValueError, match="dtype"):
        backend._output(_CudaLike((512, 2048), torch.float16), call, like)
    out = _CudaLike((512, 2048), torch.bfloat16)
    assert backend._output(out, call, like) is out


def test_backend_entrypoint_rejects_non_cake_route():
    inp, w = _meta_pair(256, 2048)
    with pytest.raises(ValueError, match="exactly 'cake'"):
        backend.all_gather_matmul_cake(inp, w, _fake_group(2, 0), backend="auto")


def test_public_entrypoint_routes_backend_cake_to_the_backend(monkeypatch):
    dispatcher = importlib.import_module(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul"
    )
    calls = []
    result = object()

    def fake_backend(inp, w, group, *, backend, verbose):
        calls.append((inp, w, group, backend, verbose))
        return result

    monkeypatch.setattr(backend, "all_gather_matmul_cake", fake_backend)
    inp, w, group = object(), object(), object()
    assert dispatcher.all_gather_matmul(inp, w, group, backend="cake") is result
    assert calls == [(inp, w, group, "cake", False)]


def test_cross_stream_join_records_a_fresh_event_and_skips_capture():
    """The join must never reuse an event that a CUDA graph capture re-recorded."""

    assert not hasattr(backend._LaunchState, "tail_event")

    class _Stream:
        def __init__(self, handle):
            self.cuda_stream = handle
            self.joined = []

        def wait_stream(self, other):
            self.joined.append(other)

    state = backend._LaunchState(rank=0, world_size=2)
    first, second, third = _Stream(11), _Stream(22), _Stream(33)
    assert backend._join_previous_tail(state, first, capturing=False) is False
    state.tail_stream = first
    assert backend._join_previous_tail(state, first, capturing=False) is False
    assert backend._join_previous_tail(state, second, capturing=True) is False
    assert backend._join_previous_tail(state, second, capturing=False) is True
    assert second.joined == [first] and first.joined == []
    state.tail_stream = second
    assert backend._join_previous_tail(state, third, capturing=True) is False
    assert third.joined == []


def test_barrier_launches_inside_the_main_stream_binding_of_the_call_device(
    monkeypatch,
):
    """The tensor-less barrier launcher gets no stream from its arguments; the backend binds one."""

    bindings = []

    class _Binding:
        def __init__(self, device, stream):
            self.device, self.stream, self.active = device, stream, False

        def __enter__(self):
            self.active = True
            bindings.append(self)
            return self

        def __exit__(self, *exc):
            self.active = False
            return False

    class _Barrier:
        def __init__(self):
            self.runs = []

        def run(self, *args):
            self.runs.append((args, [binding.active for binding in bindings]))

    monkeypatch.setattr(
        backend.tvm_ffi,
        "use_raw_stream",
        lambda device, stream: _Binding(device, stream),
    )
    call = backend._Call(
        device_index=1,
        rank=1,
        world_size=2,
        group_name="g",
        dtype_name="bfloat16",
        arch="sm_100a",
        rows=256,
        n=2048,
    )
    state = backend._LaunchState(rank=1, world_size=2, flag_peers=("peer-table",))
    barrier = _Barrier()
    backend._run_barrier(barrier, call, SimpleNamespace(cuda_stream=4242), state)
    assert barrier.runs == [((1, 2, 1, ("peer-table",), 1, 1, 1), [True])]
    assert [
        (binding.device, binding.stream, binding.active) for binding in bindings
    ] == [(backend.tvm_ffi.device("cuda", 1), 4242, False)]
