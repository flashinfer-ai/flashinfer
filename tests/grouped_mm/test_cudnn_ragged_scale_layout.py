# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ragged block-scale storage through the public grouped GEMM graph path."""

import pytest
import torch

from flashinfer.grouped_mm import grouped_mm_fp4, grouped_mm_mxfp8
from flashinfer.grouped_mm.cudnn.core import (
    _plan_indices,
    _run_cudnn_moe_block_scale_grouped_gemm_fp4,
    _run_cudnn_moe_block_scale_grouped_gemm_mxfp8,
)


def _pack_bytes(logical):
    """Independent F8_128x4 byte-address oracle, outside the timed path."""
    rows, cols = logical.shape
    ncols = (cols + 3) // 4
    out = torch.zeros((rows + 127) // 128 * ncols * 512, dtype=torch.uint8)
    for row in range(rows):
        for col in range(cols):
            address = (
                ((row // 128) * ncols + col // 4) * 512
                + (row % 32) * 16
                + ((row % 128) // 32) * 4
                + col % 4
            )
            out[address] = logical[row, col]
    return out


@pytest.mark.parametrize("kind", ["mxfp8", "mxfp4", "nvfp4"])
def test_ragged_scale_capacity_and_capture(kind, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    torch.manual_seed(183)
    bounds = [0, 5, 5, 134, 137]
    m, e, k, n = 137, 4, 128, 128
    block = 16 if kind == "nvfp4" else 32
    sk = k // block
    # 640 rows cover every possible partition of 137 rows into four groups.
    capacity_rows = 128 * (min(m, e) + (m - min(m, e)) // 128)
    if kind == "nvfp4":
        sa = (2.0 ** torch.randint(-2, 2, (m, sk))).to(torch.float8_e4m3fn)
        sb = (2.0 ** torch.randint(-2, 2, (e, n, sk))).to(torch.float8_e4m3fn)
        sa_bytes, sb_bytes = sa.view(torch.uint8), sb.view(torch.uint8)
    else:
        sa_bytes = torch.randint(125, 129, (m, sk), dtype=torch.uint8)
        sb_bytes = torch.randint(125, 129, (e, n, sk), dtype=torch.uint8)
        sa, sb = (
            sa_bytes.view(torch.float8_e8m0fnu),
            sb_bytes.view(torch.float8_e8m0fnu),
        )
    exact = torch.cat(
        [
            _pack_bytes(sa_bytes[l:r])
            for l, r in zip(bounds[:-1], bounds[1:], strict=True)
        ]
    )
    padded = torch.zeros(capacity_rows * sk, dtype=torch.uint8)
    padded[: exact.numel()] = exact
    sfa = padded.view(capacity_rows, sk).cuda()
    sfb = torch.stack([_pack_bytes(s) for s in sb_bytes]).view(e, n, sk).cuda()
    if kind == "nvfp4":
        sfa, sfb = sfa.view(torch.float8_e4m3fn), sfb.view(torch.float8_e4m3fn)
    if kind == "mxfp8":
        a = torch.randint(-3, 4, (m, k), device="cuda").to(torch.float8_e4m3fn)
        b = torch.randint(-3, 4, (e, n, k), device="cuda").to(torch.float8_e4m3fn)
        api = grouped_mm_mxfp8
        prepare = _run_cudnn_moe_block_scale_grouped_gemm_mxfp8
        kwargs = {}

        def unpack(t):
            return t.float()

    else:
        a = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8)
        b = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )

        def unpack(t):
            return torch.stack(
                [lut[(t & 15).long()], lut[(t >> 4).long()]], -1
            ).flatten(-2)

        api = grouped_mm_fp4
        prepare = _run_cudnn_moe_block_scale_grouped_gemm_fp4
        kwargs = {"block_size": block}
    offsets = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    graph = prepare(
        a, b, sfa, sfb, offsets, out=out, tactic=0, _prepare_only=True, **kwargs
    )
    tactic = next(t for t in _plan_indices(graph) if t[0] == 20400)

    def reference():
        da = unpack(a) * sa.cuda().float().repeat_interleave(block, 1)
        db = unpack(b) * sb.cuda().float().repeat_interleave(block, 2)
        return torch.cat(
            [
                da[l:r] @ db[j].T
                for j, (l, r) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
            ]
        ).to(out.dtype)

    def run():
        return api(a, b, sfa, sfb, offsets, out=out, tactic=tactic, **kwargs)

    run()
    torch.testing.assert_close(out, reference(), atol=0.125, rtol=0.01)
    # Graph callers may pass an opaque flat SF blob even though the graph
    # descriptors are rank three. Keep capacity and the accepted rank bridge.
    from flashinfer.grouped_mm.cudnn.core import _CUDNN_UIDs, _plan_index

    flat_pack = {
        _CUDNN_UIDs.TOKEN.value: a.unsqueeze(0),
        _CUDNN_UIDs.WEIGHT.value: b.transpose(1, 2),
        _CUDNN_UIDs.TOKEN_SCALE_FACTOR.value: sfa.flatten(),
        _CUDNN_UIDs.WEIGHT_SCALE_FACTOR.value: sfb.flatten(),
        _CUDNN_UIDs.FIRST_TOKEN_OFFSET.value: offsets[:-1].view(e, 1, 1),
        _CUDNN_UIDs.OUTPUT.value: out.unsqueeze(0),
    }
    workspace = torch.empty(
        graph.get_workspace_size(), dtype=torch.uint8, device="cuda"
    )
    out.fill_(float("nan"))
    graph.execute_plan_at_index(flat_pack, workspace, _plan_index(graph, tactic))
    torch.testing.assert_close(out, reference(), atol=0.125, rtol=0.01)
    previous = out.clone()
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        run()
    if kind == "mxfp8":
        a.copy_((-a.float()).to(a.dtype))
    else:
        a.bitwise_xor_(0x88)
    out.fill_(float("nan"))
    cg.replay()
    torch.testing.assert_close(out, reference(), atol=0.125, rtol=0.01)
    assert not torch.equal(out, previous)
    # Exact current routing needs 512 rows, but the no-D2H contract requires
    # enough capacity for all partitions. Keep that guard rather than bypass it.
    with pytest.raises(ValueError, match="packed blob"):
        api(
            a,
            b,
            sfa[: exact.numel() // sk],
            sfb,
            offsets,
            out=out,
            tactic=tactic,
            **kwargs,
        )
