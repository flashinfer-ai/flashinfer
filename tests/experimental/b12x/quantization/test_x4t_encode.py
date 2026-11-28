"""The X4T encoder is lossless and matches the reference base/selector rule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from b12x._lib.quant import x4t_encode
from b12x._lib.quant.x4t_encode import encode_x4t_scales

from ..conftest import require_b12x


def _reference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference encoder for one expert ``uint8 [rows, cols]``."""

    rows, cols = values.shape
    row_bytes = (cols + 7) // 8
    keys = (np.arange(rows, dtype=np.int64)[:, None] << 8) | values
    counts = np.bincount(keys.reshape(-1), minlength=rows << 8).reshape(rows, 256)
    bases = (counts[:, :255] + counts[:, 1:]).argmax(1).astype(np.uint8)
    offsets = values.astype(np.int16) - bases[:, None]
    selected = offsets == 1
    outliers = (offsets != 0) & ~selected
    bits = np.packbits(selected, axis=1, bitorder="little")
    assert bits.shape[1] == row_bytes
    fixed = np.concatenate(
        (bases.reshape(-1, 16), bits.reshape(-1, 16 * row_bytes)), axis=1
    )
    positions = np.flatnonzero(outliers.reshape(-1)).astype(np.uint32)
    words = (values.reshape(-1)[positions].astype(np.uint32) << 24) | positions
    return fixed, words


def _data(kind: str, experts: int, rows: int, cols: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(rows * 1000 + cols * 7 + experts)
    if kind == "random":
        return torch.randint(
            0, 256, (experts, rows, cols), dtype=torch.uint8, generator=gen
        )
    if kind == "equal":
        base = torch.randint(
            0, 255, (experts, rows, 1), dtype=torch.uint8, generator=gen
        )
        return base.expand(experts, rows, cols).contiguous()
    if kind == "ties":
        # Two equally frequent windows {10,11} and {20,21}; the lowest base wins.
        pattern = torch.tensor([10, 11, 10, 11, 20, 21, 20, 21], dtype=torch.uint8)
        return (
            pattern.repeat(cols // 8 + 1)[:cols]
            .expand(experts, rows, cols)
            .contiguous()
        )
    if kind == "extreme":
        choices = torch.tensor([0, 1, 253, 254, 255], dtype=torch.uint8)
        pick = torch.randint(0, 5, (experts, rows, cols), generator=gen)
        return choices[pick]
    base = torch.randint(60, 90, (experts, rows, 1), dtype=torch.uint8, generator=gen)
    out = base + torch.randint(
        0, 2, (experts, rows, cols), dtype=torch.uint8, generator=gen
    )
    sparse = torch.rand((experts, rows, cols), generator=gen) < 0.02
    noise = torch.randint(
        0, 256, (experts, rows, cols), dtype=torch.uint8, generator=gen
    )
    return torch.where(sparse, noise, out)


KINDS = ["typical", "equal", "ties", "extreme", "random"]


def _check(scales: torch.Tensor) -> None:
    fixed, exceptions = encode_x4t_scales(scales)
    experts, rows, cols = scales.shape
    assert len(fixed) == len(exceptions) == experts
    raw = scales.view(torch.uint8).numpy()
    for e in range(experts):
        ref_fixed, ref_words = _reference(raw[e])
        assert fixed[e].dtype == torch.uint8 and fixed[e].device.type == "cpu"
        assert tuple(fixed[e].shape) == (rows // 16, 16 * (1 + (cols + 7) // 8))
        assert exceptions[e].dtype == torch.uint32 and exceptions[e].ndim == 1
        assert exceptions[e].device.type == "cpu"
        np.testing.assert_array_equal(fixed[e].numpy(), ref_fixed)
        words = exceptions[e].to(torch.int64).numpy()
        np.testing.assert_array_equal(words, ref_words.astype(np.int64))
        positions = words & x4t_encode.X4T_POSITION_MASK
        assert (np.diff(positions) > 0).all()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cols", [1, 7, 8, 9, 64, 255])
@pytest.mark.parametrize("rows", [16, 48, 256])
@pytest.mark.parametrize("experts", [1, 3])
def test_matches_reference(kind, cols, rows, experts):
    _check(_data(kind, experts, rows, cols))


def test_ties_pick_lowest_base():
    scales = _data("ties", 1, 16, 8)
    fixed, exceptions = encode_x4t_scales(scales)
    assert (fixed[0][0, :16] == 10).all()
    assert exceptions[0].numel() == 16 * 4  # the 20/21 window is all exceptions


def test_accepts_e8m0_dtype():
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch.float8_e8m0fnu unavailable")
    raw = _data("typical", 2, 32, 9)
    as_e8m0 = raw.view(torch.float8_e8m0fnu)
    got = encode_x4t_scales(as_e8m0)
    want = encode_x4t_scales(raw)
    for a, b in zip(got, want, strict=True):
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)


def test_noncontiguous_input():
    raw = _data("typical", 2, 32, 16)
    view = raw.transpose(1, 2).contiguous().transpose(1, 2)
    assert not view.is_contiguous()
    got = encode_x4t_scales(view)
    want = encode_x4t_scales(raw)
    for a, b in zip(got, want, strict=True):
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)


def _set_slab_rows(monkeypatch, slab: int, cols: int) -> None:
    per_row = (
        x4t_encode._HISTOGRAM_BYTES_PER_ROW + cols * x4t_encode._TEMP_BYTES_PER_SCALE
    )
    monkeypatch.setattr(x4t_encode, "_SLAB_BUDGET_BYTES", slab * per_row)
    assert x4t_encode._slab_rows(48, cols) == slab


def _assert_same(got, want) -> None:
    for a, b in zip(got, want, strict=True):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)


@pytest.mark.parametrize("kind", ["typical", "random", "extreme"])
@pytest.mark.parametrize("slab", [16, 32, 64, 144])
def test_slabbing_is_invisible(monkeypatch, kind, slab):
    # rows=48: slab 16 aligns to tiles, 32 splits inside experts and straddles
    # expert boundaries, 64 and 144 cover several experts per slab.
    scales = _data(kind, 5, 48, 9)
    want = encode_x4t_scales(scales)
    _set_slab_rows(monkeypatch, slab, 9)
    got = encode_x4t_scales(scales)
    _assert_same(got, want)
    _check_planes(scales, got)
    for plane in got[0] + got[1]:
        assert plane.untyped_storage().nbytes() == plane.numel() * plane.element_size()


def _check_planes(scales, planes) -> None:
    raw = scales.view(torch.uint8).numpy()
    for e in range(scales.shape[0]):
        ref_fixed, ref_words = _reference(raw[e])
        np.testing.assert_array_equal(planes[0][e].numpy(), ref_fixed)
        np.testing.assert_array_equal(
            planes[1][e].to(torch.int64).numpy(), ref_words.astype(np.int64)
        )


def test_tall_expert_is_processed_in_slabs(monkeypatch):
    # A 2**20-row expert at the position limit exceeds the default slab.
    assert x4t_encode._slab_rows(2**20, 16) < 2**20
    assert x4t_encode._slab_rows(2**24, 1) < 2**24
    scales = _data("typical", 1, 4096, 3)
    want = encode_x4t_scales(scales)
    _set_slab_rows(monkeypatch, 16, 3)
    calls = []
    original = x4t_encode._encode_slab
    monkeypatch.setattr(
        x4t_encode,
        "_encode_slab",
        lambda *args: calls.append(args[1]) or original(*args),
    )
    got = encode_x4t_scales(scales)
    assert len(calls) == 4096 // 16
    _assert_same(got, want)
    _check_planes(scales, got)


def test_histogram_counted_in_row_estimate():
    assert x4t_encode._HISTOGRAM_BYTES_PER_ROW == (256 + 255) * 4 + 8
    assert x4t_encode._slab_rows(16, 1) % 16 == 0


def test_exception_heavy_slabs_fit_the_budget(monkeypatch):
    # Every row holds bytes 0..254 (253 exceptions per row, the most a row can
    # have at cols=255), so each slab is dominated by the exception stage.
    rows, cols = 64, 255
    scales = (
        (torch.arange(rows * cols, dtype=torch.int64) % cols)
        .to(torch.uint8)
        .reshape(1, rows, cols)
        .roll(shifts=3, dims=2)
    )
    assert encode_x4t_scales(scales)[1][0].numel() == rows * 253
    budget = 4 << 20
    monkeypatch.setattr(x4t_encode, "_SLAB_BUDGET_BYTES", budget)
    slab = x4t_encode._slab_rows(rows, cols)
    per_row = x4t_encode._HISTOGRAM_BYTES_PER_ROW + cols * 26
    assert slab == budget // per_row // 16 * 16
    # Worst-case exception stage: 25 B per exception plus the fixed plane.
    exception_stage = 25 * 253 + 1 + (cols + 7) // 8
    assert x4t_encode._TEMP_BYTES_PER_SCALE * cols >= exception_stage
    assert slab * per_row <= budget and 16 <= slab < 300 * rows
    calls = []
    original = x4t_encode._encode_slab
    monkeypatch.setattr(
        x4t_encode,
        "_encode_slab",
        lambda *args: calls.append(args[2]) or original(*args),
    )
    big = scales.expand(300, rows, cols).contiguous()
    got = encode_x4t_scales(big)
    assert len(calls) > 1
    _check_planes(big, got)


def test_planes_own_their_storage():
    fixed, exceptions = encode_x4t_scales(_data("random", 4, 48, 9))
    for plane in fixed + exceptions:
        assert plane.untyped_storage().nbytes() == plane.numel() * plane.element_size()


@pytest.mark.parametrize(
    "scales, error",
    [
        (torch.zeros(16, 4, dtype=torch.uint8), ValueError),
        (torch.zeros(0, 16, 4, dtype=torch.uint8), ValueError),
        (torch.zeros(1, 24, 4, dtype=torch.uint8), ValueError),
        (torch.zeros(1, 0, 4, dtype=torch.uint8), ValueError),
        (torch.zeros(1, 16, 0, dtype=torch.uint8), ValueError),
        (torch.zeros(1, 16, 256, dtype=torch.uint8), ValueError),
        (torch.empty(1, 2**20 + 16, 16, dtype=torch.uint8), ValueError),
        (torch.zeros(1, 16, 4, dtype=torch.int16), TypeError),
        (torch.zeros(1, 16, 4, dtype=torch.float32), TypeError),
    ],
)
def test_validation(scales, error):
    with pytest.raises(error):
        encode_x4t_scales(scales)


def test_public_wrapper_returns_csf_planes():
    from b12x.moe.fused_moe.weights import CsfScalePlanes, encode_mxfp4_csf_scales

    scales = _data("typical", 2, 32, 9)
    planes = encode_mxfp4_csf_scales(scales)
    assert isinstance(planes, CsfScalePlanes)
    fixed, exceptions = encode_x4t_scales(scales)
    for a, b in zip(planes.fixed + planes.exceptions, fixed + exceptions, strict=True):
        assert torch.equal(a, b)


def test_position_limit_boundary_accepted():
    scales = torch.full((1, 2**20, 16), 7, dtype=torch.uint8)
    scales[0, -1, -1] = 200
    _, exceptions = encode_x4t_scales(scales)
    assert exceptions[0].tolist() == [(200 << 24) | (2**24 - 1)]


@pytest.mark.parametrize("kind", ["typical", "random"])
@pytest.mark.parametrize("rows,cols", [(64, 12), (256, 255)])
def test_cuda_matches_cpu(kind, rows, cols):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    scales = _data(kind, 3, rows, cols)
    cpu = encode_x4t_scales(scales)
    cuda = encode_x4t_scales(scales.cuda())
    for a, b in zip(cuda[0] + cuda[1], cpu[0] + cpu[1], strict=True):
        assert a.device.type == "cpu"
        assert torch.equal(a, b)


@pytest.mark.parametrize("kind", ["typical", "random"])
@pytest.mark.parametrize("rows,cols", [(64, 12), (256, 255)])
def test_cuda_round_trips_through_decoder(kind, rows, cols):
    device = require_b12x()
    from b12x._lib.quant import x4t_scales
    from b12x._lib.quant.x4t_scales import decode_x4t_scales, make_x4t_scale_batch
    from b12x.moe import fused_moe

    assert x4t_encode.X4T_TILE_ROWS == x4t_scales.X4T_TILE_ROWS
    assert x4t_encode.X4T_POSITION_MASK == x4t_scales.X4T_POSITION_MASK

    experts = 3
    scales = _data(kind, experts, rows, cols)
    fixed, exceptions = encode_x4t_scales(scales.to(device))
    planes = fused_moe.encode_mxfp4_csf_scales(scales.to(device))
    assert isinstance(planes, fused_moe.CsfScalePlanes)
    assert all(torch.equal(a, b) for a, b in zip(planes.fixed, fixed, strict=True))

    batch = make_x4t_scale_batch(
        fixed, exceptions, rows=rows, columns=cols, device=device
    )
    out = torch.empty((experts, rows, cols), dtype=torch.uint8, device=device)
    expert_ids = torch.arange(experts, dtype=torch.int32, device=device)
    decode_x4t_scales(batch, expert_ids, out)
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), scales)


def test_cuda_rejects_graph_capture(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    scales = _data("typical", 1, 16, 8).cuda()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="graph capture"):
        encode_x4t_scales(scales)
