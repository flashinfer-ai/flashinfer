"""GPU tests of the SM107 split-KV launch of the CAKE DSv4 FP8/H128 decode route.

On SM107, FP8 launches of 128 heads over items 1025-1152 wide (nine sparse
tiles) with at most 20 metadata rows and PDL run a split-KV producer and a
merge kernel (:func:`flashinfer.mla.cake_dsv4._k64_split`); launches of
three pieces per row and 16-20 rows merge with the merge built for three
pieces (one warp per row and head, two heads per CTA, grid ``(rows, 64)``),
those of nine and five pieces with the merge built for their count alone
and those of three at 9-15 rows with the merge of any count (grid
``(rows, 128)``) (:func:`flashinfer.mla.cake_dsv4._k64_split_merge`).
These tests call ``trtllm_batch_decode_sparse_mla_dsv4(backend="cake")`` on
such launches and check that the launch splits and merges as planned, that
the output matches the fp32 reference of the hardening suite at its contract
tolerance (rows at every piece edge included, and through separate SWA /
compressed tables and a lens offset, bit for bit equal to the combined
call), that masked columns contribute nothing (bit-identical output under a
poisoned page), and that repeated calls are bitwise identical (eager and
CUDA graph replay, with the piece buffers poisoned before every call).
Every launch that merges with the merge built for its count alone writes,
bit for bit, what the merge of any count writes for it (its key removed from
the registry, as on FlashInfer main before those merges).
Launches outside the split rule (21 rows, PDL off, a workspace below the
split carve at 8 rows, at 17 and at 20, items 1024 wide, the K=64 launch
gate false) must run the route's own launch bit for bit.
FP8 launches of 64 heads over items of four to nine tiles with PDL split by
the rows and pieces of ``_K64_SPLIT_RULES_H64``; the ``test_h64_*`` tests
check the same contract at both ends of every pieces band of those rules
(``H64_ROUTED``) and at a partial-tile width of every tile count
(``H64_PARTIAL``, through both compressed page sizes), with the piece buffers
NaN-poisoned and the bytes past the split carve and past the output guarded,
and run 64-head launches outside the rules as the route's own launch.
Skips without an SM107 GPU.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pathlib

import pytest
import torch

from flashinfer.jit import cake_dsv4 as jit
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
# 1 and 4, 5 and 8 rows bound the launches that merge with the merge for nine
# and five pieces alone, 9 and 15 rows those that merge with the merge of any
# count, 16 and 20 rows those that merge with the three-piece merge.
ROUTED = [
    (1, 9),
    (2, 9),
    (4, 9),
    (5, 5),
    (8, 5),
    (9, 3),
    (12, 3),
    (13, 3),
    (14, 3),
    (15, 3),
    (16, 3),
    (17, 3),
    (18, 3),
    (19, 3),
    (20, 3),
]
REPEATS = 20
# The merge built for three pieces per row (one warp per row and head, two
# heads per CTA): the 128-head launches of three pieces per row and at least
# MERGE_S3_ROWS rows merge with it.
MERGE_S3 = f"{cake._K64_SPLIT_MERGE}_s3"
MERGE_S3_ROWS = 16
# The merges built for nine and five pieces alone (one CTA per row and head,
# the per-element operations of the merge of any count): the 128-head
# launches of nine and five pieces merge with them.
MERGE_C = {splits: f"{cake._K64_SPLIT_MERGE}_c{splits}" for splits in (9, 5)}
# Every row count whose launch merges with the merge built for its count
# alone, with its pieces per row.
PER_COUNT = [
    *((rows, 9) for rows in range(1, 5)),
    *((rows, 5) for rows in range(5, 9)),
]


def _merge_of(rows: int, splits: int, num_heads: int = HEADS) -> tuple:
    """The merge key and grid of a split launch of ``rows`` rows of ``splits``
    pieces each."""
    if num_heads == HEADS and splits == 3 and rows >= MERGE_S3_ROWS:
        return (MERGE_S3, (rows, num_heads // 2, 1))
    if num_heads == HEADS and splits in MERGE_C:
        return (MERGE_C[splits], (rows, num_heads, 1))
    return (cake._K64_SPLIT_MERGE, (rows, num_heads, 1))


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


def _item_tiles(item_width: int) -> int:
    """Sparse tiles of an item of ``item_width`` compressed columns (the SWA
    tile first)."""
    return -(-item_width // TILE) + 1


def _short_cap(splits: int, item_width: int = COMPRESSED_WIDTH) -> int:
    """The most compressed columns of a row whose tiles fit one piece
    (``ceil(n / s)`` of the item's ``n`` tiles, the first of them the SWA
    tile)."""
    return (-(-_item_tiles(item_width) // splits) - 1) * TILE


def _boundary_lens(splits: int, item_width: int = COMPRESSED_WIDTH) -> list:
    """Compressed columns of rows at the piece edges of ``splits`` pieces of
    ``k = ceil(n / s)`` tiles each (the SWA tile first): rows of exactly k
    tiles (piece 0 alone), of k + 1 tiles (one column, then one full tile in
    piece 1), of 2k and 2k + 1 tiles where those fit in the item's ``n`` (a
    later piece without a tile, then with one), and of all ``item_width``
    compressed columns (and one fewer)."""
    item_tiles = _item_tiles(item_width)
    tiles = -(-item_tiles // splits)
    lens = [(tiles - 1) * TILE, (tiles - 1) * TILE + 1, tiles * TILE]
    if 2 * tiles < item_tiles:
        lens += [(2 * tiles - 1) * TILE, (2 * tiles - 1) * TILE + 1]
    return lens + [item_width - 1, item_width]


def _compressed_lens(
    lengths: str, rows: int, splits: int, item_width: int = COMPRESSED_WIDTH
) -> list:
    if lengths == "full":
        return [item_width] * rows
    if lengths == "mixed":
        return hardening._mixed_compressed_lens(rows, item_width)
    if lengths == "short":
        cap = _short_cap(splits, item_width)
        pattern = (37, 0, 131, 1, 253, 65, 511, 129, 3, 512, 255, 63)
        return [min(cap, pattern[i % len(pattern)]) for i in range(rows)]
    if lengths == "boundary":
        pattern = _boundary_lens(splits, item_width)
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
    num_heads: int = HEADS,
    item_width: int = COMPRESSED_WIDTH,
    compressed_page: int = 64,
):
    seq_lens, q_lens, kind = _layout_shape(layout, rows, short=lengths == "short")
    lens = _compressed_lens(lengths, rows, splits, item_width)
    case = hardening._build_case(
        num_heads=num_heads,
        dtype=DTYPE,
        seq_lens=seq_lens,
        q_lens=q_lens,
        layout=kind,
        compressed_width=compressed_width,
        compressed_page=compressed_page,
        compressed_lens=[min(n, compressed_width) for n in lens],
    )
    assert case.rows == rows
    return case


def _workspace(case, num_bytes=None) -> torch.Tensor:
    if num_bytes is None:
        num_bytes = get_cake_dsv4_workspace_bytes(
            case.rows, case.num_heads, int(case.sparse_indices.shape[1]), DTYPE
        )
        assert num_bytes >= case.requirement().single_launch_bytes
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device="cuda:0")
    cake_dsv4_workspace_reset(workspace)
    return workspace


def _split_carve(rows: int, splits: int, num_heads: int = HEADS) -> int:
    return cake_dsv4_workspace_layout(rows, num_heads, splits).total_bytes


def _poison_pieces(
    workspace: torch.Tensor, rows: int, splits: int, num_heads: int = HEADS
) -> None:
    """NaN bytes over the partial O and LSE regions of a split launch."""
    layout = cake_dsv4_workspace_layout(rows, num_heads, splits)
    start = layout.partial_o[0]
    stop = layout.partial_lse[0] + layout.partial_lse[1]
    workspace[start:stop].fill_(0xFF)


class _Splits(list):
    """The split decision of every whole launch, in call order, and in
    ``merges`` the (key, grid) of every split merge the calls prepare."""

    def __init__(self) -> None:
        super().__init__()
        self.merges: list = []


def _record_splits(monkeypatch) -> _Splits:
    """Record the split decision of every whole launch and the key and grid
    of every split merge it launches."""
    seen = _Splits()
    decide = cake._k64_split
    prepare = cake._prepare_variant
    merges = {cake._K64_SPLIT_MERGE, *cake._K64_SPLIT_COUNT_MERGES}

    def record(*args, **kwargs):
        split = decide(*args, **kwargs)
        seen.append(split)
        return split

    def record_merge(variant, *, grid, **kwargs):
        if variant in merges:
            seen.merges.append((variant, tuple(grid)))
        return prepare(variant, grid=grid, **kwargs)

    monkeypatch.setattr(cake, "_k64_split", record)
    monkeypatch.setattr(cake, "_prepare_variant", record_merge)
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
    """The launch splits into ``splits`` pieces per row and merges with the
    merge of its rows and pieces (``_merge_of``), matches the reference and is
    bit-identical with the masked columns' page poisoned."""
    _skip_unless_sm107()
    case = _case(rows, splits, layout=layout, lengths=lengths)
    workspace = _workspace(case)
    seen = _record_splits(monkeypatch)
    clean = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    assert seen.merges == [_merge_of(rows, splits)]
    hardening._assert_valid(clean, case)
    case.fill_poison(hardening.POISON)
    poisoned = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(clean, poisoned), "masked columns leak into the split output"


@pytest.mark.parametrize(
    "rows,splits", [(1, 9), (4, 9), (8, 5), (15, 3), (16, 3), (17, 3), (20, 3)]
)
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
    assert seen.merges == [_merge_of(rows, splits)]
    hardening._assert_valid(out, case)


@pytest.mark.parametrize("rows,splits", [(2, 9), (8, 5), (15, 3), (17, 3), (20, 3)])
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
    assert seen.merges == [_merge_of(rows, splits)]
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


@pytest.mark.parametrize("rows,splits", [(8, 5), (13, 3), (17, 3), (20, 3)])
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
    assert seen.merges == [_merge_of(rows, splits)] * 2
    hardening._assert_valid(other, case)
    assert torch.equal(other, combined), (
        f"the {form} form differs from the combined call"
    )


@pytest.mark.parametrize(
    "rows,splits",
    [(1, 9), (4, 9), (5, 5), (8, 5), (15, 3), (16, 3), (17, 3), (18, 3), (20, 3)],
)
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
    assert seen.merges == [_merge_of(rows, splits)]
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
    assert seen.merges[-1] == _merge_of(rows, splits)
    for repeat in range(REPEATS):
        case.query.zero_()
        out.fill_(float("nan"))
        _poison_pieces(workspace, rows, splits)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.reshape(first.shape), first), (
            f"graph replay {repeat} differs from the eager call"
        )


def _any_count_merge_launch(
    monkeypatch, case, workspace: torch.Tensor, splits: int
) -> torch.Tensor:
    """``case`` with PDL, merged with the merge of any count: the key of the
    merge built for ``splits`` pieces alone removed from the SM107 registry
    (as on FlashInfer main before those merges), the piece buffers poisoned
    first."""
    with monkeypatch.context() as patch:
        patch.delitem(jit.VARIANTS["sm_107a"], MERGE_C[splits])
        _poison_pieces(workspace, case.rows, splits)
        out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
        torch.cuda.synchronize()
    return out


_PER_COUNT_CASES = [
    pytest.param(rows, splits, layout, lengths, id=f"t{rows}-{layout}-{lengths}")
    for rows, splits in PER_COUNT
    for layout in ("dense-q1", "dense-q4", "ragged")
    if layout != "dense-q4" or rows % 4 == 0
    for lengths in ("full", "mixed", "short", "boundary")
]


@pytest.mark.parametrize("rows,splits,layout,lengths", _PER_COUNT_CASES)
def test_per_count_merge_matches_the_merge_of_any_count(
    monkeypatch, rows, splits, layout, lengths
):
    """A launch that merges with the merge built for its count alone (nine
    pieces at 1-4 rows, five at 5-8) matches the reference and writes, bit
    for bit, the output of the same launch merged with the merge of any count
    over the same grid."""
    _skip_unless_sm107()
    case = _case(rows, splits, layout=layout, lengths=lengths)
    workspace = _workspace(case)
    seen = _record_splits(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    hardening._assert_valid(out, case)
    main = _any_count_merge_launch(monkeypatch, case, workspace, splits)
    assert seen == [(0, rows, splits)] * 2
    assert seen.merges == [
        (MERGE_C[splits], (rows, HEADS, 1)),
        (cake._K64_SPLIT_MERGE, (rows, HEADS, 1)),
    ]
    assert torch.equal(out.view(torch.int16), main.view(torch.int16)), (
        "the per-count merge differs from the merge of any count"
    )


_UNSPLIT_FORMS = [
    "rows21",
    "pdl_off",
    "small_workspace",
    "small_workspace_t17",
    "small_workspace_t20",
    "width1024",
    "gate_false",
]


@pytest.mark.parametrize("form", _UNSPLIT_FORMS)
def test_unsplit_launches_run_the_route_as_main(monkeypatch, form):
    """Launches outside the split rule run the route's own launch: no split,
    output bit-identical to the launch with the split decision forced off.
    A 17- or 20-row launch with a workspace one byte below its three-piece
    carve runs the route's own launch too (no fallback to fewer pieces)."""
    _skip_unless_sm107()
    rows = {
        "rows21": 21,
        "small_workspace_t17": 17,
        "small_workspace_t20": 20,
    }.get(form, 8)
    splits = 3 if form in ("small_workspace_t17", "small_workspace_t20") else 5
    case = _case(
        rows,
        splits,
        layout="ragged",
        lengths="mixed",
        compressed_width=896 if form == "width1024" else COMPRESSED_WIDTH,
    )
    enable_pdl = form != "pdl_off"
    full = _workspace(case)
    workspace = (
        _workspace(case, _split_carve(rows, splits) - 1)
        if form.startswith("small_workspace")
        else full
    )
    if form == "gate_false":
        monkeypatch.setattr(cake, "_k64_sibling_launchable", lambda arch, device: False)
    seen = _record_splits(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=enable_pdl)
    torch.cuda.synchronize()
    assert seen == [None]
    assert seen.merges == []
    main = _main_launch(monkeypatch, case, full, enable_pdl=enable_pdl)
    assert torch.equal(out, main)
    hardening._assert_valid(out, case)


# 64-head launches: (item tiles, rows, pieces per row) at both ends of every
# pieces band of the SM107 64-head rules (``_K64_SPLIT_RULES_H64``), over
# items of whole tiles (``tiles * 128`` wide).
H64_HEADS = 64
H64_ROUTED = [
    (4, 1, 4),
    (4, 4, 4),
    (5, 1, 5),
    (5, 9, 5),
    (6, 1, 6),
    (6, 9, 6),
    (6, 10, 3),
    (6, 18, 3),
    (7, 1, 7),
    (7, 9, 7),
    (7, 10, 4),
    (7, 17, 4),
    (8, 1, 8),
    (8, 7, 8),
    (8, 8, 4),
    (8, 15, 4),
    (8, 16, 3),
    (8, 20, 3),
    (9, 1, 9),
    (9, 6, 9),
    (9, 7, 5),
    (9, 12, 5),
    (9, 13, 3),
    (9, 20, 3),
]
# Items whose last tile is partial (``width % 128`` columns), one or two per
# tile count: (item tiles, rows, pieces per row, width).
H64_PARTIAL = [
    (4, 4, 4, 388),
    (5, 9, 5, 516),
    (6, 18, 3, 644),
    (7, 10, 4, 772),
    (8, 8, 4, 900),
    (9, 6, 9, 1028),
    (9, 13, 3, 1028),
]
# Bytes behind the split carve that a 64-head split launch must leave alone.
GUARD_BYTES = 4096
GUARD = 0xA5
# Value of the output row behind the last metadata row (exact in bf16).
OUT_GUARD = -3.0


def _h64_case(tiles: int, rows: int, splits: int, *, width: int = 0, **kwargs):
    """A 64-head case over items ``width`` wide (default ``tiles * 128``)
    whose compressed lengths follow that width."""
    width = width or tiles * TILE
    assert -(-width // TILE) == tiles, (width, tiles)
    return _case(
        rows,
        splits,
        compressed_width=width - TILE,
        num_heads=H64_HEADS,
        item_width=width - TILE,
        **kwargs,
    )


def _h64_guarded_workspace(case, rows: int, splits: int):
    """The split carve, which the requirement reports, plus ``GUARD_BYTES``
    of ``GUARD`` behind it, with the piece buffers NaN-poisoned."""
    carve = _split_carve(rows, splits, H64_HEADS)
    assert case.requirement().single_launch_bytes == carve
    workspace = _workspace(case, carve + GUARD_BYTES)
    workspace[carve:].fill_(GUARD)
    _poison_pieces(workspace, rows, splits, H64_HEADS)
    return workspace, carve


def _assert_guard(workspace: torch.Tensor, carve: int) -> None:
    assert bool(workspace[carve:].eq(GUARD).all()), "the split wrote past its carve"


def _h64_guarded_out(case):
    """A NaN output and the row of ``OUT_GUARD`` right behind it in the same
    buffer (returned second), which the launch must leave alone."""
    size = case.query.numel()
    buffer = torch.full(
        (size + case.num_heads * 512,),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    buffer[size:].fill_(OUT_GUARD)
    return buffer[:size].view(case.query.shape), buffer[size:]


def _assert_out_guard(tail: torch.Tensor) -> None:
    assert bool(tail.eq(OUT_GUARD).all()), "the launch wrote past its output"


_H64_CASES = [
    pytest.param(
        tiles,
        rows,
        splits,
        tiles * TILE,
        64,
        layout,
        lengths,
        id=f"n{tiles}-t{rows}-{layout}-{lengths}",
    )
    for tiles, rows, splits in H64_ROUTED
    for layout in ("dense-q1", "dense-q4", "ragged")
    if layout != "dense-q4" or rows % 4 == 0
    for lengths in ("full", "mixed", "short", "boundary")
] + [
    pytest.param(
        tiles,
        rows,
        splits,
        width,
        page,
        layout,
        lengths,
        id=f"n{tiles}-w{width}-p{page}-t{rows}-{layout}-{lengths}",
    )
    for tiles, rows, splits, width in H64_PARTIAL
    for layout, page in (("dense-q1", 64), ("ragged", 2))
    for lengths in ("full", "mixed", "short", "boundary")
]


@pytest.mark.parametrize("tiles,rows,splits,width,page,layout,lengths", _H64_CASES)
def test_h64_split_launch_matches_reference(
    monkeypatch, tiles, rows, splits, width, page, layout, lengths
):
    """A 64-head launch over ``width``-wide items (compressed page ``page``)
    splits into ``splits`` pieces per row, merges with the merge of any count
    (no 64-head launch takes a per-count merge), writes a NaN-filled output
    that matches the reference from NaN-poisoned piece buffers, leaves the
    bytes past its carve and past its output alone and is bit-identical with
    the masked columns' page poisoned."""
    _skip_unless_sm107()
    case = _h64_case(
        tiles,
        rows,
        splits,
        width=width,
        layout=layout,
        lengths=lengths,
        compressed_page=page,
    )
    workspace, carve = _h64_guarded_workspace(case, rows, splits)
    seen = _record_splits(monkeypatch)
    out, tail = _h64_guarded_out(case)
    clean = case.run(workspace=workspace, out=out, enable_pdl=True)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    assert seen.merges == [(cake._K64_SPLIT_MERGE, (rows, H64_HEADS, 1))]
    _assert_guard(workspace, carve)
    _assert_out_guard(tail)
    hardening._assert_valid(clean, case)
    case.fill_poison(hardening.POISON)
    _poison_pieces(workspace, rows, splits, H64_HEADS)
    out, tail = _h64_guarded_out(case)
    poisoned = case.run(workspace=workspace, out=out, enable_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(clean, poisoned), "masked columns leak into the split output"
    _assert_guard(workspace, carve)
    _assert_out_guard(tail)


@pytest.mark.parametrize(
    "tiles,rows,splits", [(9, 1, 9), (9, 12, 5), (8, 8, 4), (6, 10, 3), (4, 4, 4)]
)
@pytest.mark.parametrize("sink", ["none", "large", "empty_piece_zero"])
def test_h64_split_launch_sinks(monkeypatch, tiles, rows, splits, sink):
    """64-head launches with no sink, a sink far above the scores, and rows
    whose first piece has no valid column (it carries the sink alone): the
    producer reads the sinks and writes the LSE of the 64 heads only."""
    _skip_unless_sm107()
    case = _h64_case(tiles, rows, splits, layout="ragged")
    if sink == "none":
        case = dataclasses.replace(case, sinks=None)
    elif sink == "large":
        case.sinks.fill_(8.0)
    else:
        first_piece = -(-tiles // splits) * TILE
        case.sparse_indices[:, :first_piece] = -1
    workspace, carve = _h64_guarded_workspace(case, rows, splits)
    seen = _record_splits(monkeypatch)
    out, tail = _h64_guarded_out(case)
    out = case.run(workspace=workspace, out=out, enable_pdl=True)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    hardening._assert_valid(out, case)
    _assert_guard(workspace, carve)
    _assert_out_guard(tail)


@pytest.mark.parametrize("tiles,rows,splits", [(9, 2, 9), (7, 10, 4), (5, 8, 5)])
def test_h64_split_launch_zero_length_rows(monkeypatch, tiles, rows, splits):
    """64-head rows without an attendable column (zero active length, or every
    index ``-1``) write zeros; the other rows of the launch match the
    reference."""
    _skip_unless_sm107()
    case = _h64_case(tiles, rows, splits, layout="ragged", lengths="mixed")
    zero_len = list(range(0, rows, 3))
    all_invalid = list(range(1, rows, 4))
    case.sparse_topk_lens[zero_len] = 0
    case.sparse_indices[all_invalid] = -1
    workspace, carve = _h64_guarded_workspace(case, rows, splits)
    seen = _record_splits(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    empty = sorted(set(zero_len) | set(all_invalid))
    assert bool(out[empty].eq(0).all()), "rows without columns must be zero"
    hardening._assert_valid(out, case)
    _assert_guard(workspace, carve)


@pytest.mark.parametrize(
    "tiles,rows,splits,width",
    [(5, 8, 5, 640), (7, 10, 4, 896), (9, 13, 3, 1152), (6, 18, 3, 644)],
)
@pytest.mark.parametrize("form", ["separate", "lens_offset"])
def test_h64_split_launch_metadata_forms(monkeypatch, tiles, rows, splits, width, form):
    """64-head launches through separate SWA and compressed tables (row stride
    ``width - 128``, compressed-only lengths) and through a lens offset split
    like the combined call, match the reference and equal the combined call
    bit for bit, with rows at every piece edge."""
    _skip_unless_sm107()
    case = _h64_case(
        tiles, rows, splits, width=width, layout="ragged", lengths="boundary"
    )
    workspace, carve = _h64_guarded_workspace(case, rows, splits)
    seen = _record_splits(monkeypatch)
    combined = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    other = _run_metadata_form(case, form, workspace=workspace, out=case.nan_out())
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)] * 2
    hardening._assert_valid(other, case)
    assert torch.equal(other, combined), (
        f"the {form} form differs from the combined call"
    )
    _assert_guard(workspace, carve)


@pytest.mark.parametrize(
    "tiles,rows,splits",
    [(9, 1, 9), (9, 12, 5), (8, 20, 3), (7, 17, 4), (5, 8, 5), (4, 4, 4)],
)
def test_h64_split_launch_is_deterministic(monkeypatch, tiles, rows, splits):
    """Eager calls and CUDA graph replays of one 64-head split launch, with
    the piece buffers and the output poisoned before each, are bitwise
    identical and never write past the carve."""
    _skip_unless_sm107()
    case = _h64_case(tiles, rows, splits, layout="ragged", lengths="mixed")
    workspace, carve = _h64_guarded_workspace(case, rows, splits)
    seen = _record_splits(monkeypatch)
    run = dict(workspace=workspace, float_scales=True, enable_pdl=True)
    first = case.run(out=case.nan_out(), **run)
    torch.cuda.synchronize()
    assert seen == [(0, rows, splits)]
    hardening._assert_valid(first, case)
    for repeat in range(REPEATS):
        _poison_pieces(workspace, rows, splits, H64_HEADS)
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
        _poison_pieces(workspace, rows, splits, H64_HEADS)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.reshape(first.shape), first), (
            f"graph replay {repeat} differs from the eager call"
        )
    _assert_guard(workspace, carve)


# (item tiles, rows): one row past the last split row of every tile count.
_H64_PAST_RULE = [(4, 5), (5, 10), (6, 19), (7, 18), (8, 21), (9, 21)]
_H64_UNSPLIT_FORMS = [
    *(("rows_past_rule", tiles, rows) for tiles, rows in _H64_PAST_RULE),
    ("pdl_off", 9, 8),
    ("small_workspace", 9, 8),
    ("three_tiles", 3, 8),
    ("gate_false", 9, 8),
]


@pytest.mark.parametrize("form,tiles,rows", _H64_UNSPLIT_FORMS)
def test_h64_unsplit_launches_run_the_route_as_main(monkeypatch, form, tiles, rows):
    """64-head launches outside the rules (one row past the last split row of
    every tile count, PDL off, a workspace one byte below the split carve,
    three-tile items, the K=64 gate false) run the route's own launch: no
    split, output bit-identical to the launch with the split decision forced
    off."""
    _skip_unless_sm107()
    case = _h64_case(tiles, rows, 5, layout="ragged", lengths="mixed")
    enable_pdl = form != "pdl_off"
    full = _workspace(case)
    workspace = (
        _workspace(case, _split_carve(rows, 5, H64_HEADS) - 1)
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
