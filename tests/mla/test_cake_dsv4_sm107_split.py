"""GPU tests of the SM107 split-KV launch of the CAKE DSv4 FP8/H128 decode route.

On SM107, FP8 launches of 128 heads over items 1025-1152 wide (nine sparse
tiles) with at most 17 metadata rows and PDL run a split-KV producer and a
merge kernel (:func:`flashinfer.mla.cake_dsv4._k64_split`).  These tests call
``trtllm_batch_decode_sparse_mla_dsv4(backend="cake")`` on such launches and
check that the launch splits as planned, that the output matches the fp32
reference of the hardening suite at its contract tolerance (rows at every
piece edge included, and through separate SWA / compressed tables and a lens
offset, bit for bit equal to the combined call), that masked columns
contribute nothing (bit-identical output under a poisoned page), and that
repeated calls are bitwise identical (eager and CUDA graph replay, with the
piece buffers poisoned before every call).  Launches outside the split
rule (18 rows, PDL off, a workspace below the split carve, items 1024 wide,
the K=64 launch gate false) must run the route's own launch bit for bit.
Skips without an SM107 GPU.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pathlib

import pytest
import torch

from flashinfer.mla import cake_dsv4 as cake
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from flashinfer.mla.cake_dsv4 import (
    cake_dsv4_workspace_layout,
    cake_dsv4_workspace_reset,
    get_cake_dsv4_workspace_bytes,
)
from flashinfer.utils import get_compute_capability


def _load_hardening_module():
    try:
        return importlib.import_module("tests.mla.test_cake_dsv4_hardening")
    except ModuleNotFoundError:
        path = pathlib.Path(__file__).resolve().parent / "test_cake_dsv4_hardening.py"
        spec = importlib.util.spec_from_file_location(
            "_cake_dsv4_hardening_helpers", path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


hardening = _load_hardening_module()

HEADS = 128
DTYPE = torch.float8_e4m3fn
# Compressed columns of the routed items: 128 SWA + 1024 = 1152 (nine tiles).
COMPRESSED_WIDTH = 1024
WIDTH = hardening.SWA_WINDOW + COMPRESSED_WIDTH
TILE = 128
# Routed row counts and the pieces per row the SM107 rule plans for them.
ROUTED = [
    (1, 9),
    (2, 9),
    (4, 9),
    (8, 5),
    (12, 3),
    (13, 3),
    (14, 3),
    (15, 3),
    (16, 2),
    (17, 2),
]
REPEATS = 20


def _skip_unless_sm107() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CAKE DSv4 SM107 split tests")
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (10, 7):
        pytest.skip(
            "the CAKE DSv4 split-KV launch runs on SM107 only, "
            f"got SM{compute_capability[0]}{compute_capability[1]}"
        )
    if not cake._k64_sibling_launchable("sm_107a", torch.device("cuda:0")):
        pytest.skip(
            "this process cannot launch the oversized-shared-memory K=64 programs"
        )


def _ragged_q_lens(rows: int) -> list:
    pattern = (1, 3, 7, 2, 5, 4, 6)
    q_lens: list = []
    while sum(q_lens) < rows:
        q_lens.append(min(pattern[len(q_lens) % len(pattern)], rows - sum(q_lens)))
    return q_lens


def _short_cap(splits: int) -> int:
    """The most compressed columns of a row whose tiles fit one piece
    (``ceil(9 / s)`` tiles, the first of them the SWA tile)."""
    return (-(-9 // splits) - 1) * TILE


def _boundary_lens(splits: int) -> list:
    """Compressed columns of rows at the piece edges of ``splits`` pieces of
    ``k = ceil(9 / s)`` tiles each (the SWA tile first): rows of exactly k
    tiles (piece 0 alone), of k + 1 tiles (one column, then one full tile in
    piece 1), of 2k and 2k + 1 tiles where those fit in nine (a later piece
    without a tile, then with one), and of nine tiles."""
    tiles = -(-9 // splits)
    lens = [(tiles - 1) * TILE, (tiles - 1) * TILE + 1, tiles * TILE]
    if 2 * tiles < 9:
        lens += [(2 * tiles - 1) * TILE, (2 * tiles - 1) * TILE + 1]
    return lens + [COMPRESSED_WIDTH - 1, COMPRESSED_WIDTH]


def _compressed_lens(lengths: str, rows: int, splits: int) -> list:
    if lengths == "full":
        return [COMPRESSED_WIDTH] * rows
    if lengths == "mixed":
        return hardening._mixed_compressed_lens(rows, COMPRESSED_WIDTH)
    if lengths == "short":
        cap = _short_cap(splits)
        pattern = (37, 0, 131, 1, 253, 65, 511, 129, 3, 512, 255, 63)
        return [min(cap, pattern[i % len(pattern)]) for i in range(rows)]
    if lengths == "boundary":
        pattern = _boundary_lens(splits)
        return [pattern[i % len(pattern)] for i in range(rows)]
    raise ValueError(lengths)


def _layout_shape(layout: str, rows: int, *, short: bool):
    """(seq_lens, q_lens, kind) of ``rows`` metadata rows; ``short`` puts some
    requests below the 128-token SWA window."""
    if layout == "dense-q1":
        q_lens = [1] * rows
    elif layout == "dense-q4":
        assert rows % 4 == 0, rows
        q_lens = [4] * (rows // 4)
    elif layout == "ragged":
        q_lens = _ragged_q_lens(rows)
    else:
        raise ValueError(layout)
    short_seq = (1, 5, 64, 127, 130, 300)
    seq_lens = [
        max(q, short_seq[i % len(short_seq)]) if short else 2048 + 37 * i
        for i, q in enumerate(q_lens)
    ]
    return seq_lens, q_lens, "ragged" if layout == "ragged" else "dense"


def _case(
    rows: int,
    splits: int,
    *,
    layout: str = "dense-q1",
    lengths: str = "full",
    compressed_width: int = COMPRESSED_WIDTH,
):
    seq_lens, q_lens, kind = _layout_shape(layout, rows, short=lengths == "short")
    lens = _compressed_lens(lengths, rows, splits)
    case = hardening._build_case(
        num_heads=HEADS,
        dtype=DTYPE,
        seq_lens=seq_lens,
        q_lens=q_lens,
        layout=kind,
        compressed_width=compressed_width,
        compressed_lens=[min(n, compressed_width) for n in lens],
    )
    assert case.rows == rows
    return case


def _workspace(case, num_bytes=None) -> torch.Tensor:
    if num_bytes is None:
        num_bytes = get_cake_dsv4_workspace_bytes(
            case.rows, HEADS, int(case.sparse_indices.shape[1]), DTYPE
        )
        assert num_bytes >= case.requirement().single_launch_bytes
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device="cuda:0")
    cake_dsv4_workspace_reset(workspace)
    return workspace


def _split_carve(rows: int, splits: int) -> int:
    return cake_dsv4_workspace_layout(rows, HEADS, splits).total_bytes


def _poison_pieces(workspace: torch.Tensor, rows: int, splits: int) -> None:
    """NaN bytes over the partial O and LSE regions of a split launch."""
    layout = cake_dsv4_workspace_layout(rows, HEADS, splits)
    start = layout.partial_o[0]
    stop = layout.partial_lse[0] + layout.partial_lse[1]
    workspace[start:stop].fill_(0xFF)


def _record_splits(monkeypatch) -> list:
    """Record the split decision of every whole launch."""
    seen: list = []
    decide = cake._k64_split

    def record(*args, **kwargs):
        split = decide(*args, **kwargs)
        seen.append(split)
        return split

    monkeypatch.setattr(cake, "_k64_split", record)
    return seen


def _main_launch(monkeypatch, case, workspace: torch.Tensor, **run_kwargs):
    """The route's own launch of ``case`` (the split decision forced off)."""
    with monkeypatch.context() as patch:
        patch.setattr(cake, "_k64_split", lambda *args, **kwargs: None)
        out = case.run(workspace=workspace, out=case.nan_out(), **run_kwargs)
        torch.cuda.synchronize()
    return out


_ROUTED_CASES = [
    pytest.param(rows, splits, layout, lengths, id=f"t{rows}-{layout}-{lengths}")
    for rows, splits in ROUTED
    for layout in ("dense-q1", "dense-q4", "ragged")
    if layout != "dense-q4" or rows % 4 == 0
    for lengths in ("full", "mixed", "short", "boundary")
]


@pytest.mark.parametrize("rows,splits,layout,lengths", _ROUTED_CASES)
def test_split_launch_matches_reference(monkeypatch, rows, splits, layout, lengths):
    """The launch splits into ``splits`` pieces per row, matches the reference
    and is bit-identical with the masked columns' page poisoned."""
    _skip_unless_sm107()
    case = _case(rows, splits, layout=layout, lengths=lengths)
    workspace = _workspace(case)
    seen = _record_splits(monkeypatch)
    clean = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    hardening._assert_valid(clean, case)
    case.fill_poison(hardening.POISON)
    poisoned = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(clean, poisoned), "masked columns leak into the split output"


@pytest.mark.parametrize("rows,splits", [(1, 9), (4, 9), (8, 5), (15, 3), (17, 2)])
@pytest.mark.parametrize("sink", ["none", "large", "empty_piece_zero"])
def test_split_launch_sinks(monkeypatch, rows, splits, sink):
    """No sink, a sink far above the scores, and rows whose first piece has no
    valid column (it carries the sink alone)."""
    _skip_unless_sm107()
    case = _case(rows, splits, layout="ragged")
    if sink == "none":
        case = dataclasses.replace(case, sinks=None)
    elif sink == "large":
        case.sinks.fill_(8.0)
    else:
        first_piece = -(-9 // splits) * TILE
        case.sparse_indices[:, :first_piece] = -1
    workspace = _workspace(case)
    seen = _record_splits(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    hardening._assert_valid(out, case)


@pytest.mark.parametrize("rows,splits", [(2, 9), (8, 5), (15, 3), (17, 2)])
def test_split_launch_zero_length_rows(monkeypatch, rows, splits):
    """Rows without an attendable column (zero active length, or every index
    ``-1``) write zeros; the other rows of the launch match the reference."""
    _skip_unless_sm107()
    case = _case(rows, splits, layout="ragged", lengths="mixed")
    zero_len = list(range(0, rows, 3))
    all_invalid = list(range(1, rows, 4))
    case.sparse_topk_lens[zero_len] = 0
    case.sparse_indices[all_invalid] = -1
    workspace = _workspace(case)
    seen = _record_splits(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    empty = sorted(set(zero_len) | set(all_invalid))
    assert bool(out[empty].eq(0).all()), "rows without columns must be zero"
    hardening._assert_valid(out, case)


LENS_OFFSET = 37


def _run_metadata_form(
    case, form: str, *, workspace: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``case`` with PDL through another form of its metadata: ``separate``
    passes the SWA and the compressed table as tensors of their own (row
    strides 128 and 1024) with compressed-only lengths
    (``extra_sparse_topk_lens``, read with a lens offset of 128);
    ``lens_offset`` passes the combined table with its lengths lowered by
    ``LENS_OFFSET`` and ``sparse_topk_lens_offset=LENS_OFFSET``."""
    swa = hardening.SWA_WINDOW
    if form == "separate":
        tables = dict(
            sparse_indices=case.sparse_indices[:, :swa].contiguous(),
            extra_sparse_indices=case.sparse_indices[:, swa:].contiguous(),
            extra_sparse_topk_lens=(case.sparse_topk_lens - swa).contiguous(),
        )
    elif form == "lens_offset":
        tables = dict(
            sparse_indices=case.sparse_indices,
            sparse_topk_lens=(case.sparse_topk_lens - LENS_OFFSET).contiguous(),
            sparse_topk_lens_offset=LENS_OFFSET,
        )
    else:
        raise ValueError(form)
    result = trtllm_batch_decode_sparse_mla_dsv4(
        query=case.query,
        swa_kv_cache=case.swa_kv_cache,
        workspace_buffer=workspace,
        compressed_kv_cache=case.compressed_kv_cache,
        seq_lens=case.seq_lens,
        out=out,
        bmm1_scale=case.scale(case.sm_scale),
        bmm2_scale=case.scale(1.0),
        sinks=case.sinks,
        kv_layout="HND",
        cum_seq_lens_q=case.cum_seq_lens_q,
        max_q_len=case.max_q_len if case.ragged else None,
        enable_pdl=True,
        backend="cake",
        **tables,
    )
    return result.reshape(-1, case.num_heads, 512)


@pytest.mark.parametrize("rows,splits", [(8, 5), (13, 3)])
@pytest.mark.parametrize("form", ["separate", "lens_offset"])
def test_split_launch_metadata_forms(monkeypatch, rows, splits, form):
    """Separate SWA and compressed tables (compressed-only lengths, lens offset
    128) and a combined table with a lens offset split like the combined call,
    match the reference and equal the combined call bit for bit, with rows at
    every piece edge (a later piece without a tile included)."""
    _skip_unless_sm107()
    case = _case(rows, splits, layout="ragged", lengths="boundary")
    workspace = _workspace(case)
    seen = _record_splits(monkeypatch)
    combined = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    other = _run_metadata_form(case, form, workspace=workspace, out=case.nan_out())
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)] * 2
    hardening._assert_valid(other, case)
    assert torch.equal(other, combined), (
        f"the {form} form differs from the combined call"
    )


@pytest.mark.parametrize("rows,splits", [(1, 9), (8, 5), (15, 3), (17, 2)])
def test_split_launch_is_deterministic(monkeypatch, rows, splits):
    """Eager calls and CUDA graph replays of one split launch, with the piece
    buffers and the output poisoned before each, are bitwise identical; inside
    the graph the producer reads the query the copy right before it writes."""
    _skip_unless_sm107()
    case = _case(rows, splits, layout="ragged", lengths="mixed")
    workspace = _workspace(case)
    seen = _record_splits(monkeypatch)
    run = dict(workspace=workspace, float_scales=True, enable_pdl=True)
    first = case.run(out=case.nan_out(), **run)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    hardening._assert_valid(first, case)
    for repeat in range(REPEATS):
        _poison_pieces(workspace, rows, splits)
        again = case.run(out=case.nan_out(), **run)
        torch.cuda.synchronize()
        assert torch.equal(again, first), f"eager repeat {repeat} differs"

    source = case.query.clone()
    out = case.nan_out()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.query.copy_(source)
        case.run(out=out, **run)
    torch.cuda.synchronize()
    assert seen[-1] == (0, rows, splits)
    for repeat in range(REPEATS):
        case.query.zero_()
        out.fill_(float("nan"))
        _poison_pieces(workspace, rows, splits)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.reshape(first.shape), first), (
            f"graph replay {repeat} differs from the eager call"
        )


_UNSPLIT_FORMS = ["rows18", "pdl_off", "small_workspace", "width1024", "gate_false"]


@pytest.mark.parametrize("form", _UNSPLIT_FORMS)
def test_unsplit_launches_run_the_route_as_main(monkeypatch, form):
    """Launches outside the split rule run the route's own launch: no split,
    output bit-identical to the launch with the split decision forced off."""
    _skip_unless_sm107()
    rows = 18 if form == "rows18" else 8
    case = _case(
        rows,
        5,
        layout="ragged",
        lengths="mixed",
        compressed_width=896 if form == "width1024" else COMPRESSED_WIDTH,
    )
    enable_pdl = form != "pdl_off"
    full = _workspace(case)
    workspace = (
        _workspace(case, _split_carve(rows, 5) - 1)
        if form == "small_workspace"
        else full
    )
    if form == "gate_false":
        monkeypatch.setattr(cake, "_k64_sibling_launchable", lambda arch, device: False)
    seen = _record_splits(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=enable_pdl)
    torch.cuda.synchronize()
    assert seen == [None]
    main = _main_launch(monkeypatch, case, full, enable_pdl=enable_pdl)
    assert torch.equal(out, main)
    hardening._assert_valid(out, case)
