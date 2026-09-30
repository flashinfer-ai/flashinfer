from __future__ import annotations

from contextlib import nullcontext
import inspect

import pytest
import torch

from b12x.comm.pcie import _twoshot_cute
from b12x.comm.pcie.pcie_oneshot import _OwnedSharedBuffer
from b12x.comm.pcie.pcie_twoshot import (
    PCIeTwoShotSP,
    _SIGNAL_BYTES,
    _make_layout,
    _pad_scalar_peer_ptrs,
)


def test_twoshot_signal_layout_matches_native_struct() -> None:
    # self_counter[64][16] followed by two 128-byte-strided peer flag arrays.
    assert _SIGNAL_BYTES == 266_240
    assert _SIGNAL_BYTES % 256 == 0


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_twoshot_staging_layout_is_aligned(world_size: int) -> None:
    layout = _make_layout(4096, 6144, world_size)
    assert layout.signal_bytes == _SIGNAL_BYTES
    assert layout.pack_stride % 16 == 0
    assert layout.scale_offset % 256 == 0
    assert layout.scale_stride % 64 == 0
    assert layout.slot_bytes % 256 == 0
    assert layout.slab_bytes == layout.signal_bytes + 2 * layout.slot_bytes


@pytest.mark.parametrize(
    ("max_rows", "row_elems", "world_size"),
    [(0, 16, 2), (3, 16, 2), (2, 15, 2), (2, 16, 6)],
)
def test_twoshot_layout_rejects_invalid_contracts(
    max_rows: int,
    row_elems: int,
    world_size: int,
) -> None:
    with pytest.raises(ValueError):
        _make_layout(max_rows, row_elems, world_size)


def test_twoshot_barrier_intrinsics_pin_native_ptx_modifiers() -> None:
    assert "ld.relaxed.sys.global.u32" in inspect.getsource(
        _twoshot_cute._ld_relaxed_sys_u32
    )
    assert "st.relaxed.sys.global.u32" in inspect.getsource(
        _twoshot_cute._st_relaxed_sys_u32
    )
    assert "fence.sc.sys" in inspect.getsource(_twoshot_cute._fence_sc_sys)


def test_twoshot_payload_paths_pin_native_ptx_address_spaces() -> None:
    assert "ld.global.nc.v4.u32" in inspect.getsource(
        _twoshot_cute.ld_global_nc_v4_u32
    )
    assert "ld.v4.b32" in inspect.getsource(_twoshot_cute._ld_generic_v4_u32)
    assert "st.v4.b32" in inspect.getsource(_twoshot_cute._st_generic_v4_u32)
    assert "st.b32" in inspect.getsource(_twoshot_cute._st_generic_u32)
    kernel_source = inspect.getsource(_twoshot_cute._TwoShotLaunch.kernel)
    assert "ld_generic_f32" in kernel_source


def test_twoshot_all_gather_pack_loop_stays_rolled_with_int64_offsets() -> None:
    source = inspect.getsource(_twoshot_cute._TwoShotLaunch.kernel)
    assert "iteration_count = Int32(" in source
    assert "for iteration in cutlass.range(" in source
    assert "unroll=1" in source
    # The compact loop counter is never used directly as an address offset.
    # Widen before scaling so production-sized buffers cannot wrap in i32.
    assert "first_index + Int64(iteration) * Int64(" in source


def test_twoshot_graph_slot_retires_epoch_before_remote_push() -> None:
    source = inspect.getsource(_twoshot_cute._TwoShotLaunch.kernel)
    assert "wait_for_prior_consumer" not in source
    assert "_GRAPH_EPOCH_OFFSET" in source
    assert "Uint32(self._slot_bias)" in source
    barrier = inspect.getsource(_twoshot_cute._TwoShotLaunch._barrier)
    assert "graph_epoch_arrive_serialized(" not in barrier
    assert "graph_epoch_arrive_serialized(" in source
    assert "self._threads - 1" in source
    assert _twoshot_cute._GRAPH_EPOCH_OFFSET == 4096 + 4
    assert _twoshot_cute._GRAPH_BLOCKS_ARRIVED_OFFSET == 4096 + 8
    assert _twoshot_cute._GRAPH_BLOCKS_ARRIVED_OFFSET < 4096 + 32 * 4
    # Retire only after a graph-only CTA barrier proves every thread observed
    # the slot, but before phase one so its atomic latency overlaps the push.
    generation_load = source.index("generation = ld_relaxed_gpu_u32(")
    graph_barrier = source.index("cute.arch.barrier()", generation_load)
    arrive = source.index("graph_epoch_arrive_serialized(", graph_barrier)
    remote_push = source.index("# Push remote shards", arrive)
    assert generation_load < graph_barrier < arrive < remote_push
    assert source.count("self._barrier(signals, local_rank)") == 1
    assert "_finish_device_slot_round" not in source


def test_twoshot_serialized_epoch_arrival_uses_one_relaxed_atomic() -> None:
    source = inspect.getsource(
        _twoshot_cute.graph_epoch_arrive_serialized
    )
    assert source.count("atom.relaxed.gpu.global.inc.u32") == 1
    assert "atom.global.add" not in source
    assert "ld.relaxed.gpu.global.u32" in source
    assert "st.global.u32 [$0], generation;" in source
    assert "fence." not in source


def test_twoshot_graph_slot_size_is_a_precomputed_kernel_argument() -> None:
    call_source = inspect.getsource(_twoshot_cute._TwoShotLaunch.__call__)
    kernel_source = inspect.getsource(_twoshot_cute._TwoShotLaunch.kernel)
    launch_source = inspect.getsource(PCIeTwoShotSP._launch)
    assert "slot_bytes: Int64" in call_source
    assert "slot_bytes: Int64" in kernel_source
    assert "self._slot_bytes" in launch_source
    assert "_IPC_SLAB_ALIGNMENT - 1" not in kernel_source


def test_twoshot_resolves_unindexed_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    pool = PCIeTwoShotSP._from_prepared_factory(
        rank=0,
        world_size=2,
        device=torch.device("cuda"),
        signal_ptrs=(0,) * 8,
        staging_ptrs=((0,) * 8, (0,) * 8),
        owned_buffers=(),
        ipc=None,
        exchange_group=None,
        max_rows=2,
        row_elems=16,
        pack_stride=1,
        scale_offset=256,
        scale_stride=64,
    )
    assert pool.device == torch.device("cuda", 3)
    assert pool._slot_bytes == 768


def test_twoshot_explicit_close_unmaps_before_free() -> None:
    events: list[tuple[str, int]] = []

    class _IPC:
        def cudaIpcCloseMemHandle(self, pointer: int) -> None:
            events.append(("unmap", pointer))

        def cudaFree(self, pointer: int) -> None:
            events.append(("free", pointer))

    pool = object.__new__(PCIeTwoShotSP)
    pool.device = torch.device("cpu")
    pool.exchange_group = None
    pool._ipc = _IPC()
    pool._owned_buffers = [
        _OwnedSharedBuffer(
            local_ptr=101,
            peer_ptrs=(101, 202),
            remote_ptrs=(202,),
        )
    ]
    pool._signal_ptrs = (101, 202) + (101,) * 6
    pool._staging_ptrs = ((301,) * 8, (401,) * 8)
    pool._closed = False
    pool._ipc_imports_closed = False
    pool._ipc_exports_freed = False

    pool.close()
    pool.close()

    assert events == [("unmap", 202), ("free", 101)]
    assert pool._closed
    assert pool._owned_buffers == []


def test_twoshot_rejects_grid_larger_than_signal_capacity() -> None:
    pool = object.__new__(PCIeTwoShotSP)
    pool.row_elems = 16
    pool._pack_stride = 64
    pool._scale_stride = 64
    with pytest.raises(ValueError, match=r"block_limit must be in \[1, 64\]"):
        pool._launch(
            "reduce_scatter",
            None,
            None,
            None,
            rows_per_rank=1,
            threads=32,
            block_limit=65,
        )


def test_twoshot_rejects_threads_above_native_launch_bound() -> None:
    pool = object.__new__(PCIeTwoShotSP)
    with pytest.raises(ValueError, match=r"warp-aligned value in \[32, 512\]"):
        pool._launch(
            "all_gather",
            None,
            None,
            None,
            rows_per_rank=1,
            threads=544,
            block_limit=1,
        )


def test_twoshot_odd_eager_to_graph_transition_seeds_slot_bias(
    monkeypatch,
) -> None:
    class _Pointer:
        def __init__(self, address: int) -> None:
            self._address = address

        def data_ptr(self) -> int:
            return self._address

    captures = iter((False, True))
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot._is_current_stream_capturing",
        lambda _device: next(captures),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    getters: list[tuple[bool, int]] = []
    launches: list[tuple[int, ...]] = []

    def _get_launcher(
        _operation,
        _world_size,
        _rank,
        device_slot_selection,
        slot_bias,
        _threads,
        _row_elems,
        _device_index,
    ):
        getters.append((device_slot_selection, slot_bias))

        def _launch(*args) -> None:
            launches.append(tuple(args))

        return _launch

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot.get_twoshot_launcher",
        _get_launcher,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot.is_twoshot_launcher_prepared",
        lambda *args: True,
    )

    pool = object.__new__(PCIeTwoShotSP)
    pool.device = torch.device("cuda", 0)
    pool.rank = 0
    pool.world_size = 2
    pool.row_elems = 16
    pool._pack_stride = 64
    pool._scale_offset = 256
    pool._scale_stride = 64
    pool._slot_bytes = 768
    pool._slot = 0
    pool._device_slot_selection = False
    pool._device_slot_bias = 0
    pool._capture_context_depth = 1
    pool._signal_ptrs = (100, 101) + (100,) * 6
    pool._staging_ptrs = (
        (200, 201) + (200,) * 6,
        (300, 301) + (300,) * 6,
    )
    payload, scale, output = _Pointer(400), _Pointer(500), _Pointer(600)

    pool._launch(
        "reduce_scatter",
        payload,
        scale,
        output,
        rows_per_rank=1,
        threads=32,
        block_limit=1,
    )
    pool._launch(
        "reduce_scatter",
        payload,
        scale,
        output,
        rows_per_rank=1,
        threads=32,
        block_limit=1,
    )

    assert getters == [(False, 0), (True, 0), (True, 1), (True, 1)]
    assert launches[0][2][0] == 200  # eager invocation consumed slot zero
    assert launches[1][2][0] == 200  # graph always receives slot zero
    assert launches[0][9] == 768
    assert launches[1][9] == 768
    assert pool._slot == 1
    assert pool._device_slot_selection
    assert pool._device_slot_bias == 1


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_twoshot_scalar_pointer_abi_pads_with_rank_local_address(
    world_size: int,
) -> None:
    pointers = tuple(100 + rank for rank in range(world_size))
    rank = world_size - 1
    padded = _pad_scalar_peer_ptrs(
        pointers,
        rank=rank,
        world_size=world_size,
    )
    assert padded[:world_size] == pointers
    assert padded[world_size:] == (pointers[rank],) * (8 - world_size)
