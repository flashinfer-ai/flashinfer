"""Block storage expands exact scale bytes into caller-owned native scratch."""

import numpy as np
import pytest
import torch

from b12x._lib.quant.csf_blocks import (
    Nvfp4BlockPlane,
    Nvfp4CompactBlockPlane,
    upload_csf_blocks,
    upload_csf_block_payload,
)
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
from b12x.loader._csf_blocks import export_csf_blocks
from ..conftest import require_b12x


def planes_and_native(
    rows, columns, experts, *, dense, empty=False, quantization="nvfp4"
):
    bits = 4 if quantization == "nvfp4" else 1
    fields = 8 // bits
    rng = np.random.default_rng(rows * columns)
    planes, native = [], []
    for _ in range(experts):
        bases = rng.integers(0, 257 - (1 << bits), rows, dtype=np.uint8)
        offsets = rng.integers(0, 1 << bits, (rows, columns), dtype=np.uint8)
        logical = bases[:, None] + offsets
        selected = rng.random(logical.shape) < (
            0.0 if empty else 1.0 if dense else 0.07
        )
        if not empty:
            selected[31::32, :] = True
        logical[selected] = rng.integers(0, 256, int(selected.sum()), dtype=np.uint8)
        positions = np.flatnonzero(selected).astype(np.uint32)
        records = positions | (logical.ravel()[positions].astype(np.uint32) << 24)
        padded_offsets = np.pad(offsets, ((0, 0), (0, -columns % fields)))
        codes = np.bitwise_or.reduce(
            padded_offsets.reshape(rows, -1, fields)
            << (np.arange(fields, dtype=np.uint8) * bits),
            axis=2,
        )
        fixed = np.concatenate(
            (bases.reshape(-1, 16), codes.reshape(rows // 16, -1)), 1
        )
        planes.append(
            export_csf_blocks(
                fixed, records, rows=rows, columns=columns, quantization=quantization
            )
        )
        r, c = (rows + 127) // 128 * 128, (columns + 3) // 4 * 4
        expanded = np.zeros((r, c), dtype=np.uint8)
        expanded[:rows] = bases[:, None]
        expanded[:rows, :columns] = logical
        expanded = expanded.reshape(r // 128, 4, 32, c // 4, 4)
        native.append(expanded.transpose(0, 3, 2, 1, 4).copy().reshape(r, c))
    return tuple(planes), torch.from_numpy(np.stack(native))


@pytest.mark.parametrize("rows,columns", [(320, 160), (640, 160), (2560, 10), (128, 9)])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize(
    "compact",
    [None, (8, True), (32, True), (8, False), (16, False), (32, False), (64, False)],
)
def test_native_block_expansion_preserves_every_scale_byte(
    rows, columns, dense, compact
):
    device = require_b12x()
    planes, expected = planes_and_native(rows, columns, 3, dense=dense)
    first = (
        Nvfp4CompactBlockPlane(
            upload_csf_block_payload(planes, device),
            groups_per_task=compact[0],
            shared_exceptions=compact[1],
        )
        if compact
        else Nvfp4BlockPlane(upload_csf_blocks(planes, device))
    )
    output = torch.empty_like(expected, device=device).view(torch.float8_e4m3fn)
    other = torch.empty_like(output)
    decoder = Nvfp4CsfDecoder.prepare(first, first, output, other)
    decoder.decode_all(output, other)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.view(torch.uint8).cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(other.view(torch.uint8).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("entries", [5, 65, 129])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "compact", [None, (8, True), (32, True), (8, False), (32, False)]
)
def test_routed_block_expansion_replays_with_poisoned_scratch(entries, dtype, compact):
    device = require_b12x()
    sources, outputs, expected = [], [], []
    for rows, columns in ((320, 160), (2560, 10)):
        planes, native = planes_and_native(rows, columns, 8, dense=False)
        sources.append(
            Nvfp4CompactBlockPlane(
                upload_csf_block_payload(planes, device),
                groups_per_task=compact[0],
                shared_exceptions=compact[1],
            )
            if compact
            else Nvfp4BlockPlane(upload_csf_blocks(planes, device))
        )
        outputs.append(
            torch.empty_like(native, device=device).view(torch.float8_e4m3fn)
        )
        expected.append(native)
    ids_cpu = torch.tensor([0, 3, 3, -1, 7, 8], dtype=dtype).repeat((entries + 5) // 6)[
        :entries
    ]
    ids = ids_cpu.to(device)
    barriers = tuple(
        torch.empty(16, dtype=torch.int32, device=device) for _ in range(2)
    )
    decoder = Nvfp4CsfDecoder.prepare(*sources, *outputs)
    decoder.decode(ids, *outputs, barriers=barriers)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decoder.decode(ids, *outputs, barriers=barriers)
    for turn in range(3):
        changed = torch.where(
            (ids_cpu >= 0) & (ids_cpu < 8), (ids_cpu + turn) % 8, ids_cpu
        )
        ids.copy_(changed)
        for tensor in outputs:
            tensor.view(torch.uint8).fill_(0xD3)
        for barrier in barriers:
            barrier.fill_(42)
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        assert all(torch.count_nonzero(barrier).item() == 0 for barrier in barriers)
        active = (
            set(range(8)) if entries >= 128 else set(changed.tolist()) & set(range(8))
        )
        for output, native in zip(outputs, expected, strict=True):
            result = output.view(torch.uint8).cpu()
            for expert in range(8):
                target = (
                    native[expert]
                    if expert in active
                    else torch.full_like(native[expert], 0xD3)
                )
                torch.testing.assert_close(result[expert], target, rtol=0, atol=0)
    graph.reset()


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("rows,columns", [(320, 10), (256, 132), (128, 9)])
@pytest.mark.parametrize("dense", [False, True])
def test_prepared_group_index_matches_exception_words(
    quantization, rows, columns, dense
):
    device = require_b12x()
    planes, native = planes_and_native(
        rows, columns, 3, dense=dense, quantization=quantization
    )
    stored = upload_csf_blocks(planes, device)
    layout = stored.geometry
    raw = stored.storage.cpu().numpy()
    metadata = (
        raw[layout.metadata : layout.records].view("<u4").reshape(3, layout.groups, 8)
    )
    records = raw[layout.records :].view("<u4")
    expected = native.numpy().reshape(
        3, (rows + 127) // 128, (columns + 3) // 4, 32, 4, 4
    )
    expected = (
        expected.transpose(0, 1, 2, 4, 3, 5).copy().reshape(3, layout.groups, 128, 4)
    )
    expected = np.bitwise_or.reduce(
        expected.astype(np.uint32) << (8 * np.arange(4)), axis=3
    )
    for expert, plane in enumerate(planes):
        np.testing.assert_array_equal(
            raw[layout.bases + expert * rows : layout.bases + (expert + 1) * rows],
            plane.bases,
        )
        np.testing.assert_array_equal(
            raw[
                layout.codes + expert * layout.code_bytes : layout.codes
                + (expert + 1) * layout.code_bytes
            ],
            plane.codes,
        )
        for group, index in enumerate(metadata[expert]):
            selected = np.unpackbits(
                index[2:6].view(np.uint8), bitorder="little"
            ).astype(bool)
            counts = selected.reshape(4, 32).sum(axis=1)
            prefix = np.concatenate(([0], np.cumsum(counts[:-1])))
            np.testing.assert_array_equal(index[1:2].view(np.uint8), prefix)
            assert int(index[6] - index[0]) == int(counts.sum())
            assert index[7] == 1
            np.testing.assert_array_equal(
                records[index[0] : index[6]], expected[expert, group, selected]
            )


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize(
    "rows,columns",
    [
        (32, 1),
        (96, 3),
        (320, 5),
        (320, 10),
        (320, 160),
        (4096, 128),
        (4608, 160),
        (5120, 18),
    ],
)
@pytest.mark.parametrize("shared_exceptions", [False, True])
@pytest.mark.parametrize("native_codes", [False, True])
@pytest.mark.parametrize("group_record_bytes", [0, 3, 4])
def test_compact_offset_widths_into_native_byte_grid(
    quantization,
    rows,
    columns,
    shared_exceptions,
    native_codes,
    group_record_bytes,
):
    """Unpack either offset width into the F8_128x4 byte grid before consumer conversion."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32

    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x._lib.quant.csf_blocks import CompactBlockStoragePlane
    from b12x._lib.utils import current_cuda_stream, make_ptr

    device = require_b12x()
    planes, expected = planes_and_native(
        rows, columns, 2, dense=False, quantization=quantization
    )
    payload = upload_csf_block_payload(
        planes,
        device,
        native_codes=native_codes,
        group_record_bytes=group_record_bytes,
    )
    payload.validate()
    geometry = (
        expected.shape[1],
        expected.shape[2],
        payload.geometry.bits,
        (8 if group_record_bytes == 3 else 6 if group_record_bytes == 4 else 4)
        + int(native_codes),
        rows,
        columns,
        2,
        8,
        int(shared_exceptions),
        0,
        1,
    )

    class ExpandFields:
        def __init__(self):
            self.reader = CompactBlockStoragePlane(geometry)

        @cute.jit
        def __call__(self, storage: cute.Pointer, output: cute.Pointer, stream):
            self.kernel(storage, output).launch(
                grid=(2 * self.reader.tasks, 1, 1), block=(256, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(self, storage, output):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            self.reader.decode(
                (storage, output),
                Int32(block) // Int32(self.reader.tasks),
                Int32(block) % Int32(self.reader.tasks),
                Int32(tid),
            )

    pointer = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    program = b12x_compile(
        ExpandFields(),
        pointer,
        pointer,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quant.csf_compact_fields", 1, geometry
        ),
    )
    output = torch.empty_like(expected, device=device)
    program(
        make_ptr(
            cutlass.Uint8,
            payload.storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        current_cuda_stream(),
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize(
    "destination,rows,columns,sections",
    [
        (1, 640, 160, 1),
        (1, 320, 10, 1),
        (1, 128, 2, 1),
        (2, 256, 160, 1),
        (2, 512, 8, 1),
        (2, 576, 18, 2),
        (2, 160, 3, 1),
        (2, 320, 5, 2),
        (3, 1152, 160, 2),
        (3, 5120, 18, 1),
        (3, 384, 6, 2),
        (3, 320, 5, 2),
        (3, 128, 7, 2),
    ],
)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("groups", [1, 2, 4, 8])
def test_compact_expansion_writes_native_layouts(
    quantization,
    destination,
    rows,
    columns,
    sections,
    shared,
    dense,
    groups,
):
    from b12x.moe._shared.kernels.w4a16.prepare import _permute_packed_scales

    device = require_b12x()
    sources, swizzled = planes_and_native(
        rows, columns, 2, dense=dense, quantization=quantization
    )
    _, pr, pc = swizzled.shape
    logical = (
        swizzled.reshape(2, pr // 128, pc // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(2, pr, pc)[:, :rows, :columns]
        .contiguous()
    )
    if destination == 1:
        group_size = 16 if quantization == "nvfp4" else 32
        reference = torch.stack(
            [
                _permute_packed_scales(
                    plane.T.contiguous(),
                    size_k=columns * group_size,
                    size_n=rows,
                    group_size=group_size,
                )
                .reshape(-1, 4)[:, [0, 2, 1, 3]]
                .reshape(columns, rows)
                for plane in logical
            ]
        )
    elif destination == 2:
        section_rows = rows // sections
        padded_half = (section_rows + 127) // 128 * 128
        padded_rows = (padded_half * sections + 255) // 256 * 256
        padded_cols = (columns + 3) // 4 * 4
        grid = torch.zeros((2, padded_rows, padded_cols), dtype=torch.uint8)
        for section in range(sections):
            grid[
                :,
                section * padded_half : section * padded_half + section_rows,
                :columns,
            ] = logical[:, section * section_rows : (section + 1) * section_rows]
        reference = (
            grid.reshape(2, padded_rows // 256, 256, padded_cols // 4, 4)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(2, padded_rows, padded_cols)
        )
    else:
        chunks = []
        section_rows = rows // sections
        for section in range(sections):
            grid = logical[:, section * section_rows : (section + 1) * section_rows]
            for r in range(0, section_rows, 128):
                for c in range(0, columns, 4):
                    chunks.append(grid[:, r : r + 128, c : c + 4].reshape(2, -1))
        reference = torch.cat(chunks, dim=1).reshape(2, rows, columns)
    payload = upload_csf_block_payload(
        sources,
        device,
        native_codes=True,
        group_record_bytes=3,
        destination=destination,
        group_rows=rows // sections,
    )
    plane = Nvfp4CompactBlockPlane(
        payload,
        layout=int(destination == 1),
        shared_exceptions=shared,
        groups_per_task=groups,
    )
    outputs = tuple(torch.empty_like(reference, device=device) for _ in range(2))
    decoder = Nvfp4CsfDecoder.prepare(plane, plane, *outputs)
    decoder.decode_all(*outputs)
    for output in outputs:
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
    ids = torch.tensor([0, 0, -1], dtype=torch.int32, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decoder.decode(ids, *outputs)
    for expert in (0, 1, -1):
        ids.fill_(expert)
        for output in outputs:
            output.fill_(0xD3)
        allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert (
            torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
        )
        expected = torch.full_like(reference, 0xD3)
        if expert >= 0:
            expected[expert] = reference[expert]
        for output in outputs:
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    graph.reset()
