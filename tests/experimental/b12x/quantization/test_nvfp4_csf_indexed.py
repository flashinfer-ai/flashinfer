"""Indexed expansion matches an independent NVFP4 byte oracle."""

import numpy as np
import pytest
import torch

from b12x._lib.quant.nvfp4_csf_indexed import (
    prepare_indexed_scales,
)
from b12x._lib.quant.nvfp4_csf import make_nvfp4_csf_batch
from b12x._lib.runtime_control import kernel_resolution_guard
from .test_nvfp4_csf import _fixture


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows,columns", [(128, 4), (640, 20), (1024, 256)])
def test_indexed_expansion_routes_tail_and_frozen_replay(ids_dtype, rows, columns):
    from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder

    first, expected13 = _fixture(rows, columns, 0, "cuda")
    second, expected2 = _fixture(256, 16, 0, "cuda")
    outputs = tuple(
        torch.empty_like(t).view(torch.float8_e4m3fn) for t in (expected13, expected2)
    )
    decoder = Nvfp4CsfDecoder.prepare(
        first,
        second,
        *outputs,
        indexed_scales=tuple(prepare_indexed_scales(p) for p in (first, second)),
    )
    barriers = tuple(
        torch.empty(97, dtype=torch.int32, device="cuda") for _ in range(2)
    )
    for count in (1, 5, 31, 63, 64, 128):
        routes = torch.ones(count, dtype=ids_dtype, device="cuda")
        if count > 1:
            routes[0] = -1
            routes[-1] = 2**32 + 3 if ids_dtype == torch.int64 else 4
        active = [1] if count < 64 else [0, 1, 2, 3]
        with kernel_resolution_guard("Indexed NVFP4 expansion with mutable routes"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                decoder.decode(routes, *outputs, barriers=barriers)
            for _ in range(2):
                routes.copy_(routes.flip(0))
                for tensor in outputs:
                    tensor.view(torch.uint8).fill_(0xD6)
                for tensor in barriers:
                    tensor.fill_(123)
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                )
                for output, expected in zip(
                    outputs, (expected13, expected2), strict=True
                ):
                    assert torch.equal(
                        output.view(torch.uint8)[active], expected[active]
                    )
                    if count < 64:
                        assert (output.view(torch.uint8)[[0, 2, 3]] == 0xD6).all()
                assert all(torch.count_nonzero(t) == 0 for t in barriers)
            graph.reset()


def _exception_fixture(rows, columns, period):
    fixed, exceptions, logical = [], [], []
    for expert in range(4):
        base = np.full(rows, 10 + expert, dtype=np.uint8)
        codes = (np.arange(rows * columns) % 16).astype(np.uint8).reshape(rows, columns)
        values = base[:, None] + codes
        positions = (
            np.arange(0, rows * columns, period, dtype=np.uint32)
            if period
            else np.empty(0, dtype=np.uint32)
        )
        values.reshape(-1)[positions] = (128 + positions % 128).astype(np.uint8)
        packed = codes[:, ::2] | (codes[:, 1::2] << 4)
        fixed.append(
            np.concatenate(
                (base.reshape(-1, 16), packed.reshape(rows // 16, -1)), axis=1
            )
        )
        exceptions.append(
            (positions | (values.reshape(-1)[positions].astype(np.uint32) << 24))
            .astype("<u4")
            .view(np.uint8)
        )
        logical.append(values)
    batch = make_nvfp4_csf_batch(
        fixed, exceptions, rows=rows, columns=columns, device="cuda"
    )
    expected = (
        torch.from_numpy(np.stack(logical))
        .cuda()
        .reshape(4, rows // 128, 4, 32, columns // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(4, rows, columns)
    )
    return batch, expected


@pytest.mark.parametrize("rows,columns", [(128, 4), (256, 20), (640, 16)])
@pytest.mark.parametrize("period", [0, 1, 3, 17, 97])
def test_indexed_vectors_cover_exception_masks(period, rows, columns):
    from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder

    batch, expected = _exception_fixture(rows, columns, period)
    plane = prepare_indexed_scales(batch)
    outputs = tuple(
        torch.empty_like(expected).view(torch.float8_e4m3fn) for _ in range(2)
    )
    decoder = Nvfp4CsfDecoder.prepare(
        batch, batch, *outputs, indexed_scales=(plane, plane)
    )
    routes = torch.tensor([3, -1, 1, 3], device="cuda", dtype=torch.int64)
    with kernel_resolution_guard("Indexed vectors with dense exception words"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decoder.decode(routes, *outputs)
        for active in ([1, 3], [0, 2]):
            routes.copy_(
                torch.tensor(
                    [active[0], -1, active[1], active[0]],
                    device="cuda",
                    dtype=torch.int64,
                )
            )
            for output in outputs:
                output.view(torch.uint8).fill_(0xD6)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            for output in outputs:
                assert torch.equal(output.view(torch.uint8)[active], expected[active])
                inactive = sorted(set(range(4)) - set(active))
                assert (output.view(torch.uint8)[inactive] == 0xD6).all()
        graph.reset()


def test_indexed_vectors_use_64bit_expert_offsets():
    from b12x._lib.quant.nvfp4_csf import Nvfp4CsfBatch, Nvfp4CsfDecoder
    from b12x._lib.quant.nvfp4_csf_indexed import IndexedNvfp4Scales

    # The selected expert's fixed stream starts beyond 2 GiB and its output
    # starts at 4 GiB. Only its storage and an adjacent output guard are touched.
    experts, selected, rows = 4097, 4096, 128
    batches, planes, outputs = [], [], []
    for columns in (8192, 4):
        fixed_bytes = rows * (1 + columns // 2)
        metadata_bytes = rows * columns // 512 * 32
        storage = torch.empty(
            1024 + experts * (fixed_bytes + metadata_bytes),
            device="cuda",
            dtype=torch.uint8,
        )
        fixed = storage[1024 : 1024 + experts * fixed_bytes].view(
            experts, 1, fixed_bytes
        )
        fixed[selected].zero_()
        fixed[selected, 0, :128].fill_(10)
        metadata = 1024 + experts * fixed_bytes + selected * metadata_bytes
        storage[metadata : metadata + metadata_bytes].zero_()
        batch = Nvfp4CsfBatch(
            fixed,
            torch.empty(0, device="cuda", dtype=torch.uint8),
            torch.zeros((experts, 2), device="cuda", dtype=torch.int64),
            rows,
            columns,
        )
        batch.validate()
        batches.append(batch)
        planes.append(IndexedNvfp4Scales(storage, rows, columns, experts))
        outputs.append(
            torch.empty(
                (experts, rows, columns), device="cuda", dtype=torch.float8_e4m3fn
            )
        )
    decoder = Nvfp4CsfDecoder.prepare(*batches, *outputs, indexed_scales=tuple(planes))
    routes = torch.tensor(
        [selected, -1, selected, 2**32 + selected], device="cuda", dtype=torch.int64
    )
    with kernel_resolution_guard("Expert scale offsets above 4 GiB"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decoder.decode(routes, *outputs)
        for _ in range(2):
            for output in outputs:
                output.view(torch.uint8)[selected - 1 :].fill_(0xD6)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            for output in outputs:
                assert (output.view(torch.uint8)[selected] == 10).all()
                assert (output.view(torch.uint8)[selected - 1] == 0xD6).all()
        graph.reset()
