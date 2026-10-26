"""Compressed E8M0 expansion matches native W4A8 packing and graph semantics."""

import numpy as np
import pytest
import torch

from b12x._lib.quant.mxfp4_csf import Mxfp4CsfDecoder, repack_mxfp4_csf_batch
from b12x._lib.quant.x4t_scales import make_x4t_scale_batch
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe.fused_moe._impl import (
    _e8m0_scale_to_w4a8_n64_sfb_inplace,
    _e8m0_scale_to_w4a8_sfb_inplace,
)
from ..conftest import require_b12x


@pytest.mark.parametrize("partial_overlap", [False, True])
def test_preparation_rejects_overlapping_scale_storage(partial_overlap):
    batch, _ = fixture(128, 16)
    plane = repack_mxfp4_csf_batch(batch, compact=False, group_rows=128)
    size = plane.num_experts * plane.rows * plane.columns
    storage = torch.empty(size + 16, dtype=torch.uint8, device=batch.fixed.device)
    first = storage[:size]
    offset = 16 if partial_overlap else 0
    with pytest.raises(ValueError, match="must not overlap"):
        Mxfp4CsfDecoder.prepare(plane, plane, first, storage[offset : offset + size])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_preparation_rejects_projections_on_different_devices():
    from dataclasses import fields, replace

    batch, _ = fixture(128, 16)
    first = repack_mxfp4_csf_batch(batch, compact=False, group_rows=128)
    second = replace(
        first,
        **{
            f.name: getattr(first, f.name).to("cuda:1")
            for f in fields(first)
            if isinstance(getattr(first, f.name), torch.Tensor)
        },
    )
    size = first.num_experts * first.rows * first.columns
    outputs = (
        torch.empty(size, dtype=torch.uint8, device="cuda:0"),
        torch.empty(size, dtype=torch.uint8, device="cuda:1"),
    )
    with pytest.raises(ValueError, match="one CUDA device"):
        Mxfp4CsfDecoder.prepare(first, second, *outputs)


def fixture(rows, columns, *, experts=8, finite_scales=False):
    device = require_b12x()
    rng = np.random.default_rng(12987 + rows + columns)
    fixed, exceptions, grids = [], [], []
    for _ in range(experts):
        bases = rng.integers(
            120 if finite_scales else 0,
            126 if finite_scales else 255,
            rows,
            dtype=np.uint8,
        )
        bits = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
        logical = bases[:, None] + bits
        positions = np.unique(
            np.concatenate(
                (
                    np.arange(0, rows * columns, 97),
                    [0, rows * columns - 1, rows * columns // 2],
                )
            )
        ).astype(np.uint32)
        values = rng.integers(
            118 if finite_scales else 0,
            128 if finite_scales else 256,
            len(positions),
            dtype=np.uint32,
        )
        logical.ravel()[positions] = values
        selectors = np.packbits(bits, axis=1, bitorder="little")
        fixed.append(
            torch.from_numpy(
                np.concatenate(
                    (bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1
                )
            )
        )
        exceptions.append(torch.from_numpy(positions | (values << 24)))
        grids.append(logical)
    return make_x4t_scale_batch(
        fixed, exceptions, rows=rows, columns=columns, device=device
    ), torch.from_numpy(np.stack(grids)).to(device)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "rows,columns,group,compact,rotation",
    [
        (1152, 160, 576, True, 576),  # DS4.1 TP4 gate/up N64 tails
        (5120, 18, 5120, True, 0),  # DS4.1 TP4 down K64 tails
        (2304, 160, 1152, False, 1152),
        (5120, 36, 5120, False, 0),
        (576, 160, 288, False, 288),  # independently padded gated halves
        (5120, 9, 5120, False, 0),  # padded scale columns
        (384, 112, 192, True, 192),  # Kimi TP16
        (3584, 6, 3584, True, 0),
    ],
)
def test_native_layout_and_poisoned_graph(
    rows, columns, group, compact, rotation, dtype
):
    batch, logical = fixture(rows, columns)
    kwargs = dict(weight_E=8, rows=rows, k_dim=columns * 32, row_rotation=rotation)
    if compact:
        expected = _e8m0_scale_to_w4a8_n64_sfb_inplace(
            logical.clone(), group_rows=group, **kwargs
        )
    else:
        expected = _e8m0_scale_to_w4a8_sfb_inplace(
            logical.clone(),
            gated_half_rows=group if rows == 2 * group else None,
            **kwargs,
        )
    plane = repack_mxfp4_csf_batch(
        batch, compact=compact, group_rows=group, row_rotation=rotation
    )
    first, second = torch.empty_like(expected), torch.empty_like(expected)
    decoder = Mxfp4CsfDecoder.prepare(plane, plane, first, second)
    invalid = 2**32 + 3 if dtype == torch.int64 else 8
    routes = torch.tensor([3, 1, 3, -1, invalid], dtype=dtype, device=logical.device)

    def check(active):
        torch.cuda.synchronize()
        for out in (first, second):
            for expert in range(8):
                if expert in active:
                    assert torch.equal(out[expert], expected[expert])
                else:
                    assert (out[expert].view(torch.uint8) == 0xD6).all()

    with kernel_resolution_guard("MXFP4-CSF native scale replay"):
        for count in (0, 1, 3, 5):
            decoder.decode(routes[:count], first, second)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decoder.decode(routes, first, second)
        for values, active in (([2, 0, 2, -1, invalid], {0, 2}), ([-1] * 5, set())):
            routes.copy_(torch.tensor(values, dtype=dtype, device=logical.device))
            first.view(torch.uint8).fill_(0xD6)
            second.view(torch.uint8).fill_(0xD6)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            check(active)
        graph.reset()
        decoder.decode(
            torch.arange(8, dtype=dtype, device=logical.device), first, second
        )
        check(set(range(8)))
