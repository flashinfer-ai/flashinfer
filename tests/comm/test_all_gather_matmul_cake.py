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
route-table coverage, launch geometry, operand classification, input
validation and the public dispatch. The multi-GPU numerical test lives in
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


def _main_programs():
    for world_size in loader.SUPPORTED_WORLD_SIZES:
        for dtype_name in loader.SUPPORTED_DTYPES.values():
            for b_layout in loader.B_LAYOUTS:
                yield world_size, dtype_name, b_layout


def test_route_table_covers_every_world_size_dtype_layout_and_phase():
    for phase in (0, 1):
        assert loader.barrier_program(phase) in loader.PROGRAMS
    assert loader.barrier_program(0) != loader.barrier_program(1)
    mains = set()
    for world_size, dtype_name, b_layout in _main_programs():
        program = loader.main_program(world_size, dtype_name, b_layout)
        assert program in loader.PROGRAMS
        mains.add(program)
    # One main kernel per (world size, dtype, weight layout): twelve distinct programs.
    assert len(mains) == 12
    assert loader.peer_push_program() in loader.PROGRAMS
    assert set(loader.ROUTES.values()) == set(loader.PROGRAMS)


def test_every_program_lists_its_device_source_only():
    for program, row in loader.PROGRAMS.items():
        assert len(row["sources"]) == 1, program
        assert row["sources"][0].startswith("csrc/cake_all_gather_matmul/")
        assert row["sources"][0].endswith("_kernel.cu")
        assert len(row["block"]) == 3 and row["block"][0] > 0
        assert row["dynamic_smem_bytes"] >= 0
        assert row["arches"] and set(row["arches"]) <= set(loader.ARCH_FLAGS), program


def test_every_route_has_one_host_sequence_per_architecture():
    expected = set()
    for arch in loader.ARCH_FLAGS:
        for world_size, dtype_name, b_layout in _main_programs():
            name = loader.sequence_name(world_size, dtype_name, b_layout, arch)
            expected.add(name)
            row = loader.SEQUENCES[name]
            assert row["arches"] == [arch]
            assert (row["world_size"], row["dtype"], row["b_layout"]) == (
                world_size,
                dtype_name,
                b_layout,
            )
            programs = loader.sequence_programs(name)
            # The sequence compiles the route's device units (one each) with
            # its launcher; the compile order carries no meaning.
            assert len(row["sources"]) == len(programs) + 1
            assert set(row["sources"][:-1]) == {
                loader.PROGRAMS[program]["sources"][0] for program in programs
            }
            assert row["sources"][-1].startswith(
                "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_"
            )
            assert row["sources"][-1].endswith(f"{name}.cu")
            # Every sequence module compiles the SM push kernel: the route is per call (rows, world size).
            assert loader.peer_push_program() in programs
    assert set(loader.SEQUENCES) == expected
    with pytest.raises(ValueError, match="host sequence"):
        loader.sequence_name(16, "bfloat16", "n_major", "sm_100a")


def test_every_program_is_delivered_for_both_architectures():
    push = loader.peer_push_program()
    assert loader.PROGRAMS[push]["arches"] == ["sm_100a", "sm_103a"]
    for phase in (0, 1):
        assert loader.PROGRAMS[loader.barrier_program(phase)]["arches"] == [
            "sm_100a",
            "sm_103a",
        ]
    for world_size, dtype_name, b_layout in _main_programs():
        assert loader.PROGRAMS[loader.main_program(world_size, dtype_name, b_layout)][
            "arches"
        ] == ["sm_100a", "sm_103a"]
    sm103_sequence = loader.sequence_name(8, "bfloat16", "n_major", "sm_103a")
    loader.spec.cache_clear()
    try:
        with pytest.raises(ValueError, match="sm_100a"):
            loader.spec(sm103_sequence, "sm_100a")
    finally:
        loader.spec.cache_clear()


def test_barrier_and_push_programs_use_static_shared_memory_only():
    for phase in (0, 1):
        program = loader.barrier_program(phase)
        assert loader.launch_block(program) == (32, 1, 1)
        assert loader.dynamic_smem_bytes(program) == 0
    push = loader.peer_push_program()
    assert loader.launch_block(push) == (128, 1, 1)
    assert loader.dynamic_smem_bytes(push) == 0


def test_main_programs_share_one_block_and_dynamic_shared_memory_contract():
    blocks = set()
    smem = set()
    for world_size, dtype_name, b_layout in _main_programs():
        program = loader.main_program(world_size, dtype_name, b_layout)
        blocks.add(loader.launch_block(program))
        smem.add(loader.dynamic_smem_bytes(program))
    assert blocks == {(loader.MAIN_THREADS, 1, 1)}
    assert len(smem) == 1 and smem.pop() > 0


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (1, 128),
        (125, 128),
        (128, 128),
        (129, 256),
        (1025, 1152),
        (19456, 19456),
    ],
)
def test_padded_rows_rounds_up_to_the_mma_tile(rows, expected):
    assert loader.padded_rows(rows) == expected


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (125, (128, 128, 1)),
        (128, (128, 128, 1)),
        (1025, (1152, 1152, 1)),
        (2432, (2432, 2432, 1)),
        (2433, (2560, 2432, 2)),
        (2560, (2560, 2432, 2)),
        (16384, (16384, 2432, 7)),
        (19456, (19456, 2432, 8)),
    ],
)
def test_chunk_plan_pads_rows_and_pushes_at_most_nineteen_row_blocks_per_chunk(
    rows, expected
):
    assert loader.chunk_plan(rows) == expected


@pytest.mark.parametrize(
    ("rows", "n", "world_size", "sm_count", "expected"),
    [
        (125, 1280, 8, 148, (40, 1, 1)),  # 8 x 1 x 5 tiles < SM count
        (512, 1280, 8, 148, (148, 1, 1)),  # 8 x 4 x 5 = 160 tiles: one CTA per SM
        (512, 2560, 4, 148, (148, 1, 1)),
        (125, 7168, 2, 148, (56, 1, 1)),  # 2 x 1 x 28
        (1025, 14336, 4, 148, (148, 1, 1)),
        (125, 1280, 8, 160, (40, 1, 1)),
    ],
)
def test_main_grid_is_one_cta_per_sm_bounded_by_the_total_tiles(
    rows, n, world_size, sm_count, expected
):
    assert (
        loader.main_grid(rows, n, world_size=world_size, sm_count=sm_count) == expected
    )


def test_main_grid_requires_a_positive_world_size_and_sm_count():
    with pytest.raises(ValueError, match="sm_count"):
        loader.main_grid(512, 1280, world_size=8, sm_count=0)
    with pytest.raises(ValueError, match="world_size"):
        loader.main_grid(512, 1280, world_size=0, sm_count=148)


@pytest.mark.parametrize(
    ("world_size", "rows", "cols", "expected"),
    [
        (8, 125, 1280, True),
        (8, 125, 7168, True),  # 128 padded rows serve N up to 10240
        (8, 125, 10240, True),
        (8, 125, 10496, False),  # one tile too wide
        (8, 512, 1280, True),
        (8, 512, 4096, True),  # 512 padded rows serve N up to 4096
        (8, 512, 7168, False),  # the GEMM would wait behind the SM push: copy engines
        (8, 513, 1280, False),  # pads to 640 rows
        (8, 1024, 1280, False),
        (4, 500, 2560, True),
        (4, 125, 14336, False),
        (4, 512, 14336, False),
        (4, 256, 8192, True),  # 256 padded rows serve N up to 8192
        (4, 256, 8448, False),
        (4, 1025, 2048, False),
        (2, 125, 1280, False),  # a single peer: the copy engine wins
        (2, 512, 2048, False),
    ],
)
def test_sm_push_serves_at_most_512_padded_rows_within_the_width_ceiling_at_ws4_and_ws8(
    world_size, rows, cols, expected
):
    assert loader.uses_sm_push(rows=rows, world_size=world_size, cols=cols) is expected
    assert loader.SM_PUSH_MAX_ROWS == 512
    assert (loader.SM_PUSH_COLS_INTERCEPT, loader.SM_PUSH_COLS_PER_ROW) == (12288, 16)
    assert [loader.sm_push_max_cols(rows) for rows in (128, 256, 384, 512)] == [
        10240,
        8192,
        6144,
        4096,
    ]


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_barrier_flag_pad_has_two_epochs_and_two_mailbox_banks_per_phase(world_size):
    assert loader.barrier_flag_words(world_size) == 2 + 4 * world_size


def test_spec_names_carry_the_exact_architecture(monkeypatch):
    # Build-target discovery reads FLASHINFER_CUDA_ARCH_LIST when set; pin it so
    # the test does not depend on the visible devices of the test host.
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a 10.3a")
    loader.spec.cache_clear()
    try:
        specs = {}
        for arch in ("sm_100a", "sm_103a"):
            sequence = loader.sequence_name(2, "bfloat16", "n_major", arch)
            spec = loader.spec(sequence, arch)
            assert spec.name == f"{loader.SOURCE_PACKAGE}_sequence_{sequence}"
            assert sequence.endswith(arch)
            assert loader.ARCH_FLAGS[arch][0] in spec.extra_cuda_cflags
            assert [str(path).split("/")[-1] for path in spec.sources] == [
                source.split("/")[-1]
                for source in loader.SEQUENCES[sequence]["sources"]
            ]
            specs[arch] = spec
        assert specs["sm_100a"] is not specs["sm_103a"]
    finally:
        loader.spec.cache_clear()


@pytest.mark.parametrize("n", [256, 1280, 2048, 7168, 14336])
def test_weight_layout_is_classified_from_strides_without_copies(n):
    n_major = torch.empty(8192, n, dtype=torch.bfloat16, device="meta")
    k_major = torch.empty(n, 8192, dtype=torch.bfloat16, device="meta").t()
    assert loader.weight_layout(n_major) == "n_major"
    assert loader.weight_layout(k_major) == "k_major"
    assert loader.weight_tma_source(n_major, "n_major").shape == (1, 8192, n)
    assert loader.weight_tma_source(k_major, "k_major").shape == (1, n, 8192)
    assert loader.weight_tma_source(k_major, "k_major").is_contiguous()


def test_weight_layout_rejects_other_stride_patterns():
    padded = torch.empty(8192, 4096, dtype=torch.bfloat16, device="meta")[:, :2048]
    with pytest.raises(ValueError, match="strides"):
        loader.weight_layout(padded)
    with pytest.raises(ValueError, match=r"\[8192, N\]"):
        loader.weight_layout(
            torch.empty(4096, 2048, dtype=torch.bfloat16, device="meta")
        )


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
    """Tensor metadata stand-in: shape, strides, dtype and a CUDA device without a GPU."""

    def __init__(self, shape, dtype, *, contiguous=True, index=0, stride=None):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", index)
        self._contiguous = contiguous
        if stride is None:
            stride = []
            acc = 1
            for dim in reversed(self._shape):
                stride.append(acc)
                acc *= dim
            stride = tuple(reversed(stride))
        self._stride = tuple(stride)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def stride(self):
        return self._stride

    def is_contiguous(self):
        return self._contiguous

    def t(self):
        return _CudaLike(
            self._shape[::-1],
            self.dtype,
            contiguous=self._contiguous and self.ndim < 2,
            index=self.device.index,
            stride=self._stride[::-1],
        )

    def view(self, *shape):
        # Metadata-only reshape of a dense (possibly transposed-to-dense) view.
        return _CudaLike(shape, self.dtype, index=self.device.index)


def _k_major_weight(n, dtype=torch.bfloat16):
    return _CudaLike((8192, n), dtype, contiguous=False, stride=(1, 8192))


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 125, 512, 1025, 16384])
@pytest.mark.parametrize("n", [256, 1280, 2048, 7168, 14336])
def test_validation_admits_any_rows_and_any_width_multiple_of_256_in_both_layouts(
    monkeypatch, world_size, rows, n
):
    _patch_distributed(monkeypatch, world_size=world_size, rank=0)
    for weight, layout in (
        (_CudaLike((8192, n), torch.bfloat16), "n_major"),
        (_k_major_weight(n), "k_major"),
    ):
        call = backend._validate(
            _CudaLike((rows, 8192), torch.bfloat16), weight, _fake_group(world_size, 0)
        )
        assert (call.world_size, call.rows, call.n, call.b_layout, call.arch) == (
            world_size,
            rows,
            n,
            layout,
            "sm_100a",
        )


@pytest.mark.parametrize("n", [128, 1000, 1281, 2049])
def test_validation_rejects_widths_that_are_not_a_multiple_of_256(monkeypatch, n):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(ValueError, match="multiple of 256"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, n), torch.bfloat16),
            _fake_group(2, 0),
        )


def test_validation_rejects_strided_inputs_and_padded_weights_instead_of_copying(
    monkeypatch,
):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(ValueError, match="contiguous"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16, contiguous=False),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )
    with pytest.raises(ValueError, match="strides"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16, contiguous=False, stride=(4096, 1)),
            _fake_group(2, 0),
        )


def test_validation_rejects_wrong_k_and_empty_inputs(monkeypatch):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(ValueError, match="K=8192"):
        backend._validate(
            _CudaLike((256, 4096), torch.bfloat16),
            _CudaLike((4096, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )
    with pytest.raises(ValueError, match="positive"):
        backend._validate(
            _CudaLike((0, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )


def test_validation_rejects_unsupported_world_sizes_and_backends(monkeypatch):
    _patch_distributed(monkeypatch, world_size=3, rank=0)
    with pytest.raises(ValueError, match="world size 2, 4, or 8"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(3, 0),
        )
    _patch_distributed(monkeypatch, world_size=2, rank=0, backend_name="gloo")
    with pytest.raises(ValueError, match="NCCL"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
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
        )


def test_out_must_match_the_gathered_output_contract():
    call = backend._Call(0, 0, 2, "g", "bfloat16", "sm_100a", 250, 2048, "n_major")
    like = _CudaLike((250, 8192), torch.bfloat16)
    with pytest.raises(ValueError, match=r"\[500, 2048\]"):
        backend._output(_CudaLike((250, 2048), torch.bfloat16), call, like)
    with pytest.raises(ValueError, match="dtype"):
        backend._output(_CudaLike((500, 2048), torch.float16), call, like)
    out = _CudaLike((500, 2048), torch.bfloat16)
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


def test_public_prepare_forwards_the_row_capacity(monkeypatch):
    dispatcher = importlib.import_module(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul"
    )
    calls = []
    launcher = object()

    def fake_prepare(inp, w, group, *, max_rows, verbose):
        calls.append((inp, w, group, max_rows, verbose))
        return launcher

    monkeypatch.setattr(backend, "_prepare_all_gather_matmul_cake", fake_prepare)
    inp, w, group = object(), object(), object()
    assert dispatcher.prepare_all_gather_matmul(inp, w, group) is launcher
    assert (
        dispatcher.prepare_all_gather_matmul(
            inp, w, group, backend="cake", max_rows=2048
        )
        is launcher
    )
    assert calls == [(inp, w, group, None, False), (inp, w, group, 2048, False)]
    with pytest.raises(ValueError, match="'auto' or 'cake'"):
        dispatcher.prepare_all_gather_matmul(inp, w, group, backend="cutile")


def test_prepared_launcher_serves_every_row_count_up_to_its_capacity(monkeypatch):
    _patch_distributed(monkeypatch, world_size=8, rank=3, arch="sm_103a")
    weight = _k_major_weight(1280)
    weight.data_ptr = lambda: 0x1000
    sample = _CudaLike((512, 8192), torch.bfloat16)
    call = backend._validate(sample, weight, _fake_group(8, 3))
    state = backend._LaunchState(rank=3, world_size=8)
    workspace = backend._Workspace(
        dtype=torch.bfloat16, rank=3, world_size=8, device_index=0, pitch=2048
    )
    group = _fake_group(8, 3)
    launcher = backend._PreparedLauncher(
        group=group,
        group_id=id(group),
        call=call,
        max_rows=2048,
        device=sample.device,
        dtype=sample.dtype,
        weight=weight,
        weight_fingerprint=backend._fingerprint(weight),
        state=state,
        workspace=workspace,
    )
    # The frozen launcher resolves the weight's tensor-map source view once.
    assert launcher.weight_source is not None
    assert tuple(launcher.weight_source.shape) == (1, 1280, 8192)
    for rows in (1, 125, 512, 1025, 2048):
        bound = launcher._validate_input(_CudaLike((rows, 8192), torch.bfloat16))
        assert (bound.rows, bound.n, bound.b_layout) == (rows, 1280, "k_major")
    with pytest.raises(ValueError, match=r"\[1, 2048\]"):
        launcher._validate_input(_CudaLike((2049, 8192), torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        launcher._validate_input(
            _CudaLike((512, 8192), torch.bfloat16, contiguous=False)
        )
    with pytest.raises(ValueError, match="dtype"):
        launcher._validate_input(_CudaLike((512, 8192), torch.float16))
    with pytest.raises(ValueError, match=r"\[M, 8192\]"):
        launcher._validate_input(_CudaLike((512, 4096), torch.bfloat16))


def test_launch_refuses_a_workspace_smaller_than_the_padded_rows():
    state = backend._LaunchState(rank=0, world_size=2)
    workspace = backend._Workspace(
        dtype=torch.bfloat16, rank=0, world_size=2, device_index=0, pitch=128
    )
    call = backend._Call(0, 0, 2, "g", "bfloat16", "sm_100a", 129, 2048, "n_major")
    with pytest.raises(
        RuntimeError, match="holds 128 rows per peer, the call needs 256"
    ):
        backend._launch(state, workspace, call, None, None, None)
    # A capacity shortfall is a caller error, not a failed collective.
    assert state.poisoned is False


def test_cross_stream_join_records_a_fresh_event_and_skips_capture(monkeypatch):
    """The join must never reuse an event that a CUDA graph capture re-recorded."""

    assert not hasattr(backend._LaunchState, "tail_event")

    class _Stream:
        def __init__(self, handle):
            self.cuda_stream = handle
            self.joined = []

        def wait_stream(self, other):
            self.joined.append(other)

    first, second, third = _Stream(11), _Stream(22), _Stream(33)
    current = {11: first, 22: second, 33: third}
    monkeypatch.setattr(
        backend.torch.cuda, "current_stream", lambda index: current[handles[-1]]
    )
    handles = [11]
    state = backend._LaunchState(rank=0, world_size=2)
    assert backend._join_previous_tail(state, 11, 0, capturing=False) is False
    backend._remember_tail(state, 11, 0)
    assert state.tail_stream is first and state.tail_handle == 11
    assert backend._join_previous_tail(state, 11, 0, capturing=False) is False
    handles.append(22)
    assert backend._join_previous_tail(state, 22, 0, capturing=True) is False
    assert backend._join_previous_tail(state, 22, 0, capturing=False) is True
    assert second.joined == [first] and first.joined == []
    backend._remember_tail(state, 22, 0)
    assert state.tail_stream is second
    handles.append(33)
    assert backend._join_previous_tail(state, 33, 0, capturing=True) is False
    assert third.joined == []
    # Re-remembering the same stream keeps the cached torch stream object.
    backend._remember_tail(state, 22, 0)
    assert state.tail_stream is second


class _Recording:
    def __init__(self, *, index=0):
        self.device = torch.device("cuda", index)
        self.streams = []

    def record_stream(self, stream):
        self.streams.append(stream)


def _launch_fixture(monkeypatch, *, rows, n, world_size, arch, b_layout="n_major"):
    call = backend._Call(0, 1, world_size, "g", "bfloat16", arch, rows, n, b_layout)
    state = backend._LaunchState(rank=1, world_size=world_size, flag_peers=("flags",))
    workspace = backend._Workspace(
        dtype=torch.bfloat16,
        rank=1,
        world_size=world_size,
        device_index=0,
        pitch=max(2048, backend.loader.padded_rows(rows)),
    )
    workspace.scratch = "scratch"
    workspace.signal_pad = "signal_pad"
    workspace.comm_stream = "comm_stream"
    workspace.comm_handle = 77
    workspace.peer_scratch_ptrs = ("peer_scratch",)
    workspace.peer_signal_ptrs = ("peer_signals",)
    workspace.push_buffers = ("payload", "signals", "counters")

    class _Sequence:
        def __init__(self):
            self.calls = []

        def run(self, *args):
            self.calls.append(("run", args))

        def run_push(self, *args):
            self.calls.append(("run_push", args))

    sequence = _Sequence()
    workspace.sequences = {b_layout: sequence}
    monkeypatch.setattr(backend, "_current_stream_handle", lambda index: 4242)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(backend, "_remember_tail", lambda state, handle, index: None)
    return call, state, workspace, sequence


def test_launch_is_one_host_sequence_call_with_the_workspace_tables(monkeypatch):
    call, state, workspace, sequence = _launch_fixture(
        monkeypatch, rows=1025, n=2048, world_size=4, arch="sm_100a"
    )
    inp = _Recording()
    backend._launch(state, workspace, call, inp, "w_source", "out")
    grid = backend.loader.main_grid(
        1025, 2048, world_size=4, sm_count=backend.loader.device_facts(0).sm_count
    )
    assert sequence.calls == [
        (
            "run",
            (
                inp,
                "scratch",
                "w_source",
                "out",
                "signal_pad",
                ("flags",),
                ("peer_scratch",),
                ("peer_signals",),
                1,
                1025,
                workspace.pitch,
                0,
                1,
                *grid,
                4242,
                77,
            ),
        )
    ]
    # The pushes read ``inp`` on the communication stream.
    assert inp.streams == ["comm_stream"]
    assert (state.next_phase, state.ready_epoch) == (1, 1)
    backend._launch(state, workspace, call, inp, "w_source", "out")
    assert sequence.calls[-1][1][11:13] == (1, 2)
    assert (state.next_phase, state.ready_epoch) == (0, 2)
    assert state.poisoned is False


def test_small_rows_use_the_sm_push_sequence_entry_on_the_callers_stream(monkeypatch):
    call, state, workspace, sequence = _launch_fixture(
        monkeypatch, rows=512, n=1280, world_size=8, arch="sm_103a"
    )
    inp = _Recording()
    backend._launch(state, workspace, call, inp, "w_source", "out")
    (entry, args) = sequence.calls[0]
    assert entry == "run_push"
    assert args[6:9] == ("payload", "signals", "counters")
    assert args[12:14] == (0, 1)
    # No communication stream is involved on the SM push route.
    assert inp.streams == []
    assert state.next_phase == 1 and state.ready_epoch == 1


def test_launch_refuses_an_unprepared_layout_without_poisoning(monkeypatch):
    call, state, workspace, _sequence = _launch_fixture(
        monkeypatch, rows=256, n=2048, world_size=2, arch="sm_100a"
    )
    workspace.sequences = {}
    with pytest.raises(RuntimeError, match="host sequence"):
        backend._launch(state, workspace, call, _Recording(), "w_source", "out")
    assert state.poisoned is False


def test_sequence_failure_poisons_the_launch_state(monkeypatch):
    call, state, workspace, sequence = _launch_fixture(
        monkeypatch, rows=256, n=2048, world_size=2, arch="sm_100a"
    )

    def failing(*args):
        raise RuntimeError("boom")

    sequence.run = failing
    with pytest.raises(RuntimeError, match="boom"):
        backend._launch(state, workspace, call, _Recording(), "w_source", "out")
    assert state.poisoned is True
