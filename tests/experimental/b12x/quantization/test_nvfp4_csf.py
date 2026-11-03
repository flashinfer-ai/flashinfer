"""Lossless scale expansion preserves bytes, routes and captured addresses."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from b12x._lib.quant.nvfp4_csf import (
    Nvfp4CsfDecoder,
    compile_nvfp4_csf_pair,
    decode_nvfp4_csf_pair,
    make_nvfp4_csf_batch,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from ..conftest import require_b12x


@pytest.mark.parametrize("partial_overlap", [False, True])
def test_pair_rejects_overlapping_output_storage(partial_overlap):
    batch, reference = _fixture(128, 16, 0, require_b12x())
    storage = torch.empty(
        reference.numel() + 16, dtype=torch.uint8, device=reference.device
    )
    first = storage[: reference.numel()].reshape_as(reference).view(torch.float8_e4m3fn)
    offset = 16 if partial_overlap else 0
    second = storage[offset : offset + reference.numel()].reshape_as(first)
    with pytest.raises(ValueError, match="must not overlap"):
        Nvfp4CsfDecoder.prepare(batch, batch, first, second.view(torch.float8_e4m3fn))
    ids = torch.zeros(1, dtype=torch.int32, device=reference.device)
    with pytest.raises(ValueError, match="must not overlap"):
        decode_nvfp4_csf_pair(batch, batch, ids, first, second)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_pair_rejects_projections_on_different_devices():
    first, ref13 = _fixture(128, 16, 0, "cuda:0")
    second, ref2 = _fixture(128, 16, 0, "cuda:1")
    outputs = (
        torch.empty_like(ref13).view(torch.float8_e4m3fn),
        torch.empty_like(ref2).view(torch.float8_e4m3fn),
    )
    with pytest.raises(ValueError, match="one CUDA device"):
        Nvfp4CsfDecoder.prepare(first, second, *outputs)
    ids = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    with pytest.raises(ValueError, match="one CUDA device"):
        decode_nvfp4_csf_pair(first, second, ids, *outputs)


def _fixture(rows, columns, codec, device):
    fixed, exceptions, logical = [], [], []
    for expert in range(4):
        # Each row has a known palette and exceptions across CTA boundaries.
        bases = ((np.arange(rows) + expert * 31) % 231).astype(np.uint8)
        offsets = (np.arange(rows * columns).reshape(rows, columns) % 16).astype(
            np.uint8
        )
        source = bases[:, None] + offsets
        source.flat[::97] = (np.arange(source.size)[::97] % 256).astype(np.uint8)
        if codec == 0:
            difference = source.astype(np.int16) - bases[:, None]
            outside = (difference < 0) | (difference > 15)
            difference[outside] = 0
            difference = difference.astype(np.uint8)
            packed = difference[:, ::2] | (difference[:, 1::2] << 4)
            stream = np.concatenate(
                (bases.reshape(-1, 16), packed.reshape(rows // 16, -1)), axis=1
            )
            positions = np.flatnonzero(outside).astype(np.uint32)
            values = source.ravel()[positions].astype(np.uint32)
        else:
            bases >>= 3
            high = source >> 3
            upper = high == bases[:, None] + 1
            outside = (high != bases[:, None]) & ~upper
            selectors = np.packbits(upper, axis=1, bitorder="little")
            low = (source & 7).reshape(rows, columns // 8, 8).astype(np.uint32)
            words = np.zeros((rows, columns // 8), dtype=np.uint32)
            for lane in range(8):
                words |= low[:, :, lane] << (3 * lane)
            mantissas = np.stack(
                [words & 255, (words >> 8) & 255, words >> 16], axis=-1
            ).astype(np.uint8)
            stream = np.concatenate(
                (
                    bases.reshape(-1, 16),
                    selectors.reshape(rows // 16, -1),
                    mantissas.reshape(rows // 16, -1),
                ),
                axis=1,
            )
            positions = np.flatnonzero(outside).astype(np.uint32)
            values = high.ravel()[positions].astype(np.uint32)
        words = positions | (values << (19 if codec == 2 else 24))
        if codec == 2:
            records = np.stack(
                [words & 255, (words >> 8) & 255, words >> 16], axis=-1
            ).astype(np.uint8)
        else:
            records = words.astype("<u4").view(np.uint8)
        fixed.append(stream)
        exceptions.append(records)
        logical.append(source)
    batch = make_nvfp4_csf_batch(
        fixed, exceptions, rows=rows, columns=columns, codec=codec, device=device
    )
    reference = torch.from_numpy(np.stack(logical)).to(device)
    reference = reference.reshape(4, rows // 128, 4, 32, columns // 4, 4)
    reference = reference.permute(0, 1, 4, 3, 2, 5).contiguous()
    return batch, reference.reshape(4, rows, columns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "codec,geometry",
    [(c, g) for c in (0, 1, 2) for g in ((128, 16, 256, 8), (2048, 256, 4096, 64))]
    + [(0, (640, 160, 2560, 20))],
)
def test_pair_exact_routes_and_poisoned_graph(codec, ids_dtype, geometry):
    device = require_b12x()
    r13, c13, r2, c2 = geometry
    first, expected13 = _fixture(r13, c13, codec, device)
    second, expected2 = _fixture(r2, c2, codec, device)
    out13, out2 = torch.empty_like(expected13), torch.empty_like(expected2)
    invalid = 2**32 + 3 if ids_dtype == torch.int64 else 4
    routes = torch.tensor([3, 1, 3, -1, invalid], dtype=ids_dtype, device=device)
    program = compile_nvfp4_csf_pair(
        first.geometry, second.geometry, ids_dtype == torch.int64
    )

    def run(ids, mode=0):
        decode_nvfp4_csf_pair(
            first, second, ids, out13, out2, mode=mode, program=program
        )

    def check(active):
        torch.cuda.synchronize()
        for output, expected in ((out13, expected13), (out2, expected2)):
            for expert in range(4):
                if expert in active:
                    assert torch.equal(output[expert], expected[expert])
                else:
                    assert bool((output[expert] == 0xD6).all())

    out13.fill_(0xD6)
    out2.fill_(0xD6)
    run(routes)
    check({1, 3})
    with kernel_resolution_guard("NVFP4-CSF retained decoder"):
        for count in (1, 3, 5):
            run(routes[:count])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(routes)
        for values, active in [([2, 0, 2, -1, invalid], {0, 2}), ([-1] * 5, set())]:
            routes.copy_(torch.tensor(values, dtype=ids_dtype, device=device))
            out13.fill_(0xD6)
            out2.fill_(0xD6)
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            check(active)
            assert torch.cuda.memory_allocated() == allocated
        graph.reset()
        for mode, values, active in [
            (1, [2, 0], {0, 2}),
            (2, [0, 4, 0, 8], {1, 3}),
            (2, [0, 0, 0, 0], set()),
            (3, [0], {0, 1, 2, 3}),
        ]:
            out13.fill_(0xD6)
            out2.fill_(0xD6)
            run(torch.tensor(values, dtype=ids_dtype, device=device), mode)
            check(active)
