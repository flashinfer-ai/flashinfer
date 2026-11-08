"""GPU tests of the SM107 FP8/H64 ragged prefill row of the CAKE DSv4 host.

Two ragged requests of up to 257 query tokens over topk4x items 640 wide (128
SWA + 512 compressed slots, the FlashMLA MODEL1 varlen prefill row) run the
source-exact program on SM100 and SM103.  SM107 builds no such program and
runs them on the multi-tile M64 program, as it runs the dense twin of the row
(the same requests without ``cum_seq_lens_q``).  These tests call
``trtllm_batch_decode_sparse_mla_dsv4(backend="cake")`` on 257 + 257,
257 + 100 and 129 + 257 tokens at contexts 4096 and 16384 and check the route
and the launched program, the output against the fp32 reference of the
hardening suite at its contract tolerance, that masked columns contribute
nothing (bit-identical output under a poisoned page), that PDL launches equal
plain ones bit for bit, that 257 + 257 tokens equal the dense twin bit for bit,
that row-tiled launches equal the single launch, and that eager calls and CUDA
graph replays are bitwise identical.  Both multi-tile programs run: the
four-stage sibling (where this process can launch it) and the two-stage
program.

The 64-head MTP decode rows of the four-stage sibling's TMEM-batch window (a
longest request of 2-4 tokens, 108 rows up to two per SM, items up to 768 wide;
dense and ragged calls) run the TMEM-batch sibling: those tests check the
launched program, the output against the same reference, that masked columns
contribute nothing, that it equals the four-stage program's output bit for bit
and that eager calls and CUDA graph replays are bitwise identical; rows just
outside the window launch the four-stage program.

The 64-head decode rows of the L2 prefetch rule (few rows over items 512 or
640 wide with compressed pages of 64 rows, ``_KV4_L2PF_MAX_TOKENS``) run the
four-stage sibling's L2 prefetch sibling on devices of 212 or 216 SMs (the
four-stage program elsewhere): those tests check the launched program, the
output against the same reference, that masked columns contribute nothing
and that it equals the four-stage program's output bit for bit; the first
rows past each ceiling, 3-token calls and the c128 layer launch the
four-stage program.  One-token calls of both one-token rules (the
L2 prefetch rule and the TMEM-batch sibling's decode window) also run the
rule's program as ragged calls and as the chunks of row-tiled calls (every
chunk after the first a ragged launch over all the call's requests), with the
four-stage program's bits.  Skips without an SM107 GPU.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pathlib

import pytest
import torch

from flashinfer.mla import cake_dsv4 as cake
from flashinfer.mla.cake_dsv4 import (
    cake_dsv4_workspace_layout,
    cake_dsv4_workspace_reset,
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

HEADS = 64
DTYPE = torch.float8_e4m3fn
# Compressed columns of the routed items: 128 SWA + 512 = 640 (topk4x, page 64).
COMPRESSED_WIDTH = 512
M64_ROUTE = "fp8_h64_prefill_source_persistent_m64_multi_tile"
Q_LENS = [
    pytest.param((257, 257), id="q257-257"),
    pytest.param((257, 100), id="q257-100"),
    pytest.param((129, 257), id="q129-257"),
]
CONTEXTS = [4096, 16384]
PROGRAMS = ["kv4", "two-stage"]
REPEATS = 5


def _skip_unless_sm107() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CAKE DSv4 SM107 prefill tests")
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (10, 7):
        pytest.skip(
            "the FP8/H64 ragged prefill row runs the M64 program on SM107 only, "
            f"got SM{compute_capability[0]}{compute_capability[1]}"
        )


def _program(monkeypatch, program: str) -> str:
    """The variant every launch of the route runs: the four-stage sibling
    (skipped where this process cannot launch it) or the two-stage program
    (the sibling's launch gate forced off)."""
    if program == "kv4":
        if not cake._kv4_sibling_launchable("sm_107a", torch.device("cuda:0")):
            pytest.skip("this process cannot launch the four-stage M64 program")
        return f"{M64_ROUTE}_kv4"
    monkeypatch.setattr(cake, "_kv4_sibling_launchable", lambda arch, device: False)
    return M64_ROUTE


def _record_launches(monkeypatch) -> list:
    """Record the variant of every prepared launch."""
    launched: list = []
    prepare = cake._prepare_variant

    def record(variant, **kwargs):
        launched.append(variant)
        return prepare(variant, **kwargs)

    monkeypatch.setattr(cake, "_prepare_variant", record)
    return launched


def _case(q_lens, context: int, *, lengths: str = "full"):
    rows = sum(q_lens)
    if lengths == "full":
        compressed_lens = [COMPRESSED_WIDTH] * rows
    else:
        compressed_lens = hardening._mixed_compressed_lens(rows, COMPRESSED_WIDTH)
    case = hardening._build_case(
        num_heads=HEADS,
        dtype=DTYPE,
        seq_lens=[context + 37 * i for i in range(len(q_lens))],
        q_lens=list(q_lens),
        layout="ragged",
        compressed_width=COMPRESSED_WIDTH,
        compressed_page=64,
        compressed_lens=compressed_lens,
    )
    assert case.rows == rows and case.max_q_len == 257
    assert int(case.sparse_indices.shape[1]) == hardening.SWA_WINDOW + 512
    return case


def _dense_twin(case):
    """The rows of ``case`` (equal requests) as a dense ``[B, Q]`` call."""
    query = case.query.view(case.batch_size, case.max_q_len, HEADS, 512)
    return dataclasses.replace(case, query=query, cum_seq_lens_q=None)


def _workspace(case) -> torch.Tensor:
    """A workspace of exactly the call's single-launch requirement."""
    num_bytes = case.requirement().single_launch_bytes
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device="cuda:0")
    cake_dsv4_workspace_reset(workspace)
    return workspace


@pytest.mark.parametrize("lengths", ["full", "mixed"])
@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("q_lens", Q_LENS)
def test_ragged_prefill_row_matches_reference(
    monkeypatch, q_lens, context, program, lengths
):
    """The row runs the multi-tile M64 program, matches the reference, is
    bit-identical with the masked columns' page poisoned and equals the PDL
    launch without PDL bit for bit."""
    _skip_unless_sm107()
    expected = _program(monkeypatch, program)
    case = _case(q_lens, context, lengths=lengths)
    req = case.requirement()
    assert (req.route, req.num_splits) == (M64_ROUTE, 1)
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    pdl = hardening._assert_poison_inert(case, workspace, enable_pdl=True)
    plain = case.run(workspace=workspace, out=case.nan_out())
    torch.cuda.synchronize()
    assert launched == [expected] * 3
    assert torch.equal(plain, pdl), "the PDL launch differs from the plain one"


@pytest.mark.parametrize("lengths", ["full", "mixed"])
@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("context", CONTEXTS)
def test_ragged_prefill_row_equals_the_dense_twin(
    monkeypatch, context, program, lengths
):
    """2 x 257 ragged tokens and the same rows as a dense ``[2, 257]`` call run
    the same program and give the same bits."""
    _skip_unless_sm107()
    expected = _program(monkeypatch, program)
    case = _case((257, 257), context, lengths=lengths)
    twin = _dense_twin(case)
    assert twin.requirement().route == case.requirement().route == M64_ROUTE
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    ragged = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    dense = twin.run(workspace=workspace, out=twin.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [expected] * 2
    hardening._assert_valid(ragged, case)
    assert torch.equal(ragged, dense), "the ragged row differs from the dense twin"


@pytest.mark.parametrize("program", PROGRAMS)
def test_ragged_prefill_row_tiled_launches_match_the_single_launch(
    monkeypatch, program
):
    """A workspace of 128 rows tiles the 357 rows into 128 + 128 + 101 (the
    third chunk crosses the request boundary): the same bits as one launch."""
    _skip_unless_sm107()
    expected = _program(monkeypatch, program)
    case = _case((257, 100), 4096, lengths="mixed")
    req = case.requirement()
    small_bytes = cake_dsv4_workspace_layout(
        128, HEADS, 1, num_query_offsets=case.batch_size + 1
    ).total_bytes
    assert req.rows_per_launch(small_bytes) == 128
    small = torch.zeros(small_bytes, dtype=torch.uint8, device="cuda:0")
    calls = hardening._count_dispatches(monkeypatch)
    launched = _record_launches(monkeypatch)
    single = case.run(workspace=_workspace(case), out=case.nan_out(), enable_pdl=True)
    tiled = case.run(workspace=small, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert calls == [(M64_ROUTE, rows) for rows in (357, 128, 128, 101)]
    assert launched == [expected] * 4
    hardening._assert_valid(tiled, case)
    assert torch.equal(tiled, single), "row-tiled launches differ from one launch"


@pytest.mark.parametrize("program", PROGRAMS)
def test_ragged_prefill_row_is_deterministic(monkeypatch, program):
    """Eager calls and CUDA graph replays with PDL are bitwise identical; inside
    the graph the program reads the query the copy right before it writes.
    Float scales throughout: a scale tensor made inside the capture would copy
    from host memory."""
    _skip_unless_sm107()
    expected = _program(monkeypatch, program)
    case = _case((257, 100), 16384, lengths="mixed")
    workspace = _workspace(case)
    run = dict(workspace=workspace, float_scales=True, enable_pdl=True)
    launched = _record_launches(monkeypatch)
    first = case.run(out=case.nan_out(), **run)
    torch.cuda.synchronize()
    hardening._assert_valid(first, case)
    for repeat in range(REPEATS):
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
    # One launch per eager call and one captured; replays prepare nothing.
    assert launched == [expected] * (REPEATS + 2)
    for repeat in range(REPEATS):
        case.query.zero_()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.reshape(first.shape), first), (
            f"graph replay {repeat} differs from the eager call"
        )


# --------------------------------------------------------------------------- #
# TMEM-batch sibling: 64-head MTP decode rows of the four-stage program       #
# --------------------------------------------------------------------------- #

KV4_VARIANT = f"{M64_ROUTE}_kv4"
BCL_VARIANT = f"{KV4_VARIANT}_bcl"
DECODE_CONTEXT = 1024
# The context of the first request of a "mixed" row: shorter than the SWA
# window, so its rows mask SWA slots.
SHORT_CONTEXT = 61
# (q_len, rows, width, compressed page) of rows inside the window; a tuple
# q_len is the request lengths of a ragged call, cycled (the window reads the
# longest); "2sm" is two rows per SM of the device, the window's upper edge.
BCL_ROWS = [
    pytest.param(4, 108, 640, 64, id="c4w640-q4-t108"),
    pytest.param(4, 120, 384, 2, id="c128w384-q4-t120"),
    pytest.param(4, 192, 640, 2, id="c128w640-q4-t192"),
    pytest.param(4, 120, 768, 64, id="c4w768-q4-t120"),
    pytest.param(4, 256, 640, 64, id="c4w640-q4-t256"),
    pytest.param(3, 120, 640, 64, id="c4w640-q3-t120"),
    pytest.param(3, 192, 768, 64, id="c4w768-q3-t192"),
    pytest.param(2, 108, 640, 64, id="c4w640-q2-t108"),
    pytest.param(2, 200, 640, 64, id="c4w640-q2-t200"),
    pytest.param(4, "2sm", 640, 64, id="c4w640-q4-t2sm"),
    pytest.param((4, 1, 3, 2), 120, 640, 64, id="c4w640-q4132-t120"),
]
# Rows just outside it: 107 rows and two rows per SM plus one (ragged calls
# whose last request is shorter), 96 rows, items 772 wide, one- and five-token
# requests.
EDGE_ROWS = [
    pytest.param(4, 107, 640, 64, id="c4w640-q4-t107"),
    pytest.param(4, 96, 640, 64, id="c4w640-q4-t96"),
    pytest.param(4, "2sm+1", 640, 64, id="c4w640-q4-t2sm+1"),
    pytest.param(4, 108, 772, 64, id="c4w772-q4-t108"),
    pytest.param(1, 108, 640, 64, id="c4w640-q1-t108"),
    pytest.param(5, 110, 640, 64, id="c4w640-q5-t110"),
    # Just outside the one-token decode window: 175 and 209 rows, 160 and
    # 256 rows, and the c128 layer (compressed pages of 2 rows) at 192 rows.
    pytest.param(1, 175, 1152, 64, id="c4w1152-q1-t175"),
    pytest.param(1, 209, 1152, 64, id="c4w1152-q1-t209"),
    pytest.param(1, 160, 1152, 64, id="c4w1152-q1-t160"),
    pytest.param(1, 256, 1152, 64, id="c4w1152-q1-t256"),
    pytest.param(1, 192, 1152, 2, id="c128w1152-q1-t192"),
]
# Rows of the one-token decode window (``_KV4_BCL_DECODE_WINDOWS``: 64
# heads, items 1152 wide, compressed pages of 64 rows, 176-208 rows), which
# takes the TMEM-batch sibling on devices of 212 or 216 SMs only.
DECODE_WINDOW_ROWS = [
    pytest.param(1, 176, 1152, 64, id="c4w1152-q1-t176"),
    pytest.param(1, 192, 1152, 64, id="c4w1152-q1-t192"),
    pytest.param(1, 208, 1152, 64, id="c4w1152-q1-t208"),
]


def _require_tmem_batch_program() -> None:
    """Skip where this process cannot launch the four-stage program (the
    TMEM-batch sibling takes its launch check); SM107 must route the sibling
    to a program of its own."""
    if not cake._kv4_sibling_launchable("sm_107a", torch.device("cuda:0")):
        pytest.skip("this process cannot launch the four-stage M64 program")
    from flashinfer.jit.cake_dsv4 import VARIANTS

    programs = VARIANTS["sm_107a"]
    four_stage = programs[KV4_VARIANT]
    assert programs.get(BCL_VARIANT, four_stage) != four_stage, (
        "SM107 does not route the TMEM-batch sibling to a program of its own"
    )


def _decode_rows(rows) -> int:
    """``rows``, with "2sm" / "2sm+1" read on the device."""
    sm_count = cake._device_sm_count(torch.device("cuda:0"))
    return {"2sm": 2 * sm_count, "2sm+1": 2 * sm_count + 1}.get(rows, rows)


def _decode_case(
    q_len, rows, width: int, page: int, *, lengths: str = "full", layout: str = ""
):
    """``rows`` query rows in requests of ``q_len`` tokens (a ragged call whose
    last request is shorter where ``q_len`` does not divide them; a tuple
    ``q_len`` gives the request lengths of a ragged call, cycled) over items
    ``width`` wide, the compressed slots on pages of ``page`` rows.  "full"
    attends every column; "mixed" masks columns of both kinds behind the
    poison pages: the first request's context is shorter than the SWA window
    (stale SWA slots beyond it) and the compressed active counts are mixed,
    incl. 0 and the full width.  ``layout`` "ragged" passes the request
    offsets of equal requests too."""
    rows = _decode_rows(rows)
    pattern = q_len if isinstance(q_len, tuple) else (q_len,)
    q_lens, left = [], rows
    while left:
        q_lens.append(min(pattern[len(q_lens) % len(pattern)], left))
        left -= q_lens[-1]
    seq_lens = [DECODE_CONTEXT + 37 * i for i in range(len(q_lens))]
    compressed_width = width - hardening.SWA_WINDOW
    mixed = lengths == "mixed"
    if mixed:
        seq_lens[0] = SHORT_CONTEXT
    case = hardening._build_case(
        num_heads=HEADS,
        dtype=DTYPE,
        seq_lens=seq_lens,
        q_lens=q_lens,
        layout=layout or ("ragged" if len(set(q_lens)) > 1 else "dense"),
        compressed_width=compressed_width,
        compressed_page=page,
        compressed_lens=(
            hardening._mixed_compressed_lens(rows, compressed_width) if mixed else None
        ),
        swa_tail="stale" if mixed else "minus_one",
    )
    assert case.rows == rows and case.max_q_len == max(pattern)
    if mixed:
        assert int((case.visible() < hardening.SWA_WINDOW).sum()) > 0
        assert int((case.sparse_topk_lens < width).sum()) > 0
    return case


@pytest.mark.parametrize("lengths", ["full", "mixed"])
@pytest.mark.parametrize("q_len,rows,width,page", BCL_ROWS)
def test_mtp_decode_rows_in_the_window_run_the_tmem_batch_program(
    monkeypatch, q_len, rows, width, page, lengths
):
    """The row runs the TMEM-batch sibling, matches the reference, is
    bit-identical with the masked columns' pages poisoned, equals the launch
    without PDL bit for bit and equals the four-stage program's output bit for
    bit."""
    _skip_unless_sm107()
    _require_tmem_batch_program()
    case = _decode_case(q_len, rows, width, page, lengths=lengths)
    req = case.requirement()
    assert (req.route, req.num_splits) == (M64_ROUTE, 1)
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    pdl = hardening._assert_poison_inert(case, workspace, enable_pdl=True)
    plain = case.run(workspace=workspace, out=case.nan_out())
    torch.cuda.synchronize()
    assert launched == [BCL_VARIANT] * 3
    assert torch.equal(plain, pdl), "the PDL launch differs from the plain one"
    monkeypatch.setattr(cake, "_kv4_bcl_variant", lambda variant, **kwargs: variant)
    four_stage = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [BCL_VARIANT] * 3 + [KV4_VARIANT]
    assert torch.equal(four_stage, pdl), (
        "the TMEM-batch program differs from the four-stage program"
    )


@pytest.mark.parametrize("q_len,rows,width,page", EDGE_ROWS)
def test_mtp_decode_rows_outside_the_window_run_the_four_stage_program(
    monkeypatch, q_len, rows, width, page
):
    """The row runs the four-stage program and matches the reference."""
    _skip_unless_sm107()
    _require_tmem_batch_program()
    case = _decode_case(q_len, rows, width, page)
    assert case.requirement().route == M64_ROUTE
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [KV4_VARIANT]
    hardening._assert_valid(out, case)


def test_tmem_batch_program_is_deterministic(monkeypatch):
    """Eager calls and CUDA graph replays of the TMEM-batch program with PDL
    (two rows per SM in 4-token requests over items 640 wide, masked columns
    of both kinds) are bitwise identical; inside the graph the program reads
    the query the copy right before it writes."""
    _skip_unless_sm107()
    _require_tmem_batch_program()
    case = _decode_case(4, "2sm", 640, 64, lengths="mixed")
    workspace = _workspace(case)
    run = dict(workspace=workspace, float_scales=True, enable_pdl=True)
    launched = _record_launches(monkeypatch)
    first = case.run(out=case.nan_out(), **run)
    torch.cuda.synchronize()
    hardening._assert_valid(first, case)
    for repeat in range(REPEATS):
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
    # One launch per eager call and one captured; replays prepare nothing.
    assert launched == [BCL_VARIANT] * (REPEATS + 2)
    for repeat in range(REPEATS):
        case.query.zero_()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.reshape(first.shape), first), (
            f"graph replay {repeat} differs from the eager call"
        )


def _decode_window_program() -> str:
    """The program a one-token decode-window row runs on this device: the
    TMEM-batch sibling on 212 or 216 SMs, the four-stage program elsewhere."""
    sm_count = cake._device_sm_count(torch.device("cuda:0"))
    if sm_count in cake._KV4_BCL_DECODE_SM_COUNTS:
        return BCL_VARIANT
    return KV4_VARIANT


@pytest.mark.parametrize("lengths", ["full", "mixed"])
@pytest.mark.parametrize("q_len,rows,width,page", DECODE_WINDOW_ROWS)
def test_decode_window_rows_run_the_tmem_batch_program(
    monkeypatch, q_len, rows, width, page, lengths
):
    """On 212 or 216 SMs the row runs the TMEM-batch sibling (elsewhere the
    four-stage program), matches the reference, is bit-identical with the
    masked columns' pages poisoned, equals the launch without PDL bit for bit
    and equals the four-stage program's output bit for bit."""
    _skip_unless_sm107()
    _require_tmem_batch_program()
    program = _decode_window_program()
    case = _decode_case(q_len, rows, width, page, lengths=lengths)
    req = case.requirement()
    assert (req.route, req.num_splits) == (M64_ROUTE, 1)
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    pdl = hardening._assert_poison_inert(case, workspace, enable_pdl=True)
    plain = case.run(workspace=workspace, out=case.nan_out())
    torch.cuda.synchronize()
    assert launched == [program] * 3
    assert torch.equal(plain, pdl), "the PDL launch differs from the plain one"
    monkeypatch.setattr(cake, "_kv4_bcl_variant", lambda variant, **kwargs: variant)
    four_stage = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [program] * 3 + [KV4_VARIANT]
    assert torch.equal(four_stage, pdl), (
        "the TMEM-batch program differs from the four-stage program"
    )


def test_tmem_batch_decode_window_program_is_deterministic(monkeypatch):
    """Eager calls and CUDA graph replays of a one-token decode-window row
    (192 rows over items 1152 wide, masked columns of both kinds, PDL on) are
    bitwise identical and run the device's decode-window program."""
    _skip_unless_sm107()
    _require_tmem_batch_program()
    program = _decode_window_program()
    case = _decode_case(1, 192, 1152, 64, lengths="mixed")
    workspace = _workspace(case)
    run = dict(workspace=workspace, float_scales=True, enable_pdl=True)
    launched = _record_launches(monkeypatch)
    first = case.run(out=case.nan_out(), **run)
    torch.cuda.synchronize()
    hardening._assert_valid(first, case)
    for repeat in range(REPEATS):
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
    assert launched == [program] * (REPEATS + 2)
    for repeat in range(REPEATS):
        case.query.zero_()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.reshape(first.shape), first), (
            f"graph replay {repeat} differs from the eager call"
        )


# --------------------------------------------------------------------------- #
# L2 prefetch sibling: few-row 64-head decode of the c4 layer                 #
# --------------------------------------------------------------------------- #

L2PF_VARIANT = f"{KV4_VARIANT}_l2pf"
# (q_len, rows, width, compressed page) of rows of the L2 prefetch rule: the
# lowest four-stage rows and the ceilings of each (width, request length).
L2PF_ROWS = [
    pytest.param(1, 32, 640, 64, id="c4w640-q1-t32"),
    pytest.param(1, 47, 640, 64, id="c4w640-q1-t47"),
    pytest.param(4, 32, 640, 64, id="c4w640-q4-t32"),
    pytest.param(4, 64, 640, 64, id="c4w640-q4-t64"),
    pytest.param(1, 41, 512, 64, id="c4w512-q1-t41"),
    pytest.param(4, 56, 512, 64, id="c4w512-q4-t56"),
]
# Rows the host sends to the four-stage program outside the rule: one request
# above each ceiling, 3-token requests and the c128 layer (compressed pages of
# 2 rows).
L2PF_EDGE_ROWS = [
    pytest.param(1, 58, 640, 64, id="c4w640-q1-t58"),
    pytest.param(4, 68, 640, 64, id="c4w640-q4-t68"),
    pytest.param(1, 42, 512, 64, id="c4w512-q1-t42"),
    pytest.param(4, 60, 512, 64, id="c4w512-q4-t60"),
    pytest.param(3, 60, 640, 64, id="c4w640-q3-t60"),
    pytest.param(1, 40, 640, 2, id="c128w640-q1-t40"),
]


def _require_l2pf_program() -> None:
    """Skip where this process cannot launch the four-stage program (the L2
    prefetch sibling takes its launch check); SM107 must route the sibling to
    a program of its own."""
    if not cake._kv4_sibling_launchable("sm_107a", torch.device("cuda:0")):
        pytest.skip("this process cannot launch the four-stage M64 program")
    from flashinfer.jit.cake_dsv4 import VARIANTS

    programs = VARIANTS["sm_107a"]
    four_stage = programs[KV4_VARIANT]
    assert programs.get(L2PF_VARIANT, four_stage) != four_stage, (
        "SM107 does not route the L2 prefetch sibling to a program of its own"
    )


def _l2pf_program() -> str:
    """The program a row of the L2 prefetch rule runs on this device: the L2
    prefetch sibling on 212 or 216 SMs, the four-stage program elsewhere."""
    sm_count = cake._device_sm_count(torch.device("cuda:0"))
    if sm_count in cake._KV4_L2PF_SM_COUNTS:
        return L2PF_VARIANT
    return KV4_VARIANT


@pytest.mark.parametrize("lengths", ["full", "mixed"])
@pytest.mark.parametrize("q_len,rows,width,page", L2PF_ROWS)
def test_l2pf_rows_run_the_l2_prefetch_program(
    monkeypatch, q_len, rows, width, page, lengths
):
    """On 212 or 216 SMs the row runs the L2 prefetch sibling (elsewhere the
    four-stage program), matches the reference, is bit-identical with the
    masked columns' pages poisoned, equals the launch without PDL bit for bit
    and equals the four-stage program's output bit for bit."""
    _skip_unless_sm107()
    _require_l2pf_program()
    program = _l2pf_program()
    case = _decode_case(q_len, rows, width, page, lengths=lengths)
    req = case.requirement()
    assert (req.route, req.num_splits) == (M64_ROUTE, 1)
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    pdl = hardening._assert_poison_inert(case, workspace, enable_pdl=True)
    plain = case.run(workspace=workspace, out=case.nan_out())
    torch.cuda.synchronize()
    assert launched == [program] * 3
    assert torch.equal(plain, pdl), "the PDL launch differs from the plain one"
    monkeypatch.setattr(cake, "_kv4_l2pf_variant", lambda variant, **kwargs: variant)
    four_stage = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [program] * 3 + [KV4_VARIANT]
    assert torch.equal(four_stage, pdl), (
        "the L2 prefetch program differs from the four-stage program"
    )


@pytest.mark.parametrize("q_len,rows,width,page", L2PF_EDGE_ROWS)
def test_l2pf_edge_rows_run_the_four_stage_program(
    monkeypatch, q_len, rows, width, page
):
    """The row runs the four-stage program and matches the reference."""
    _skip_unless_sm107()
    _require_l2pf_program()
    case = _decode_case(q_len, rows, width, page)
    assert case.requirement().route == M64_ROUTE
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [KV4_VARIANT]
    hardening._assert_valid(out, case)


# --------------------------------------------------------------------------- #
# One-token rows of both rules as ragged launches                             #
# --------------------------------------------------------------------------- #

# (rows, width) of one-token ragged calls inside a rule: the L2 prefetch rule
# (items 640 and 512 wide) and the TMEM-batch sibling's decode window (items
# 1152 wide).
ONE_TOKEN_RAGGED_ROWS = [
    pytest.param(32, 640, id="c4w640-q1-t32"),
    pytest.param(47, 640, id="c4w640-q1-t47"),
    pytest.param(41, 512, id="c4w512-q1-t41"),
    pytest.param(192, 1152, id="c4w1152-q1-t192"),
]
# (rows, width, chunk) of one-token calls outside both rules that a workspace
# of ``chunk`` rows tiles into chunks inside one; every chunk after the first
# runs as ragged rows over all the call's requests (more than 64).
ONE_TOKEN_TILED_ROWS = [
    pytest.param(96, 640, 32, id="c4w640-q1-t96-x32"),
    pytest.param(120, 512, 40, id="c4w512-q1-t120-x40"),
    pytest.param(384, 1152, 192, id="c4w1152-q1-t384-x192"),
]


def _one_token_rule(width: int) -> tuple[str, str]:
    """The rule of one-token rows over items ``width`` wide: the program its
    rows run on this device and the host function that applies it."""
    if width == 1152:
        _require_tmem_batch_program()
        return _decode_window_program(), "_kv4_bcl_variant"
    _require_l2pf_program()
    return _l2pf_program(), "_kv4_l2pf_variant"


@pytest.mark.parametrize("rows,width", ONE_TOKEN_RAGGED_ROWS)
def test_one_token_ragged_rows_of_the_rules_equal_the_four_stage_program(
    monkeypatch, rows, width
):
    """A ragged call of one-token requests inside a rule runs the rule's
    program on 212 or 216 SMs (elsewhere the four-stage program), matches the
    reference and equals the four-stage program's output bit for bit."""
    _skip_unless_sm107()
    program, rule = _one_token_rule(width)
    case = _decode_case(1, rows, width, 64, lengths="mixed", layout="ragged")
    assert case.cum_seq_lens_q is not None and case.max_q_len == 1
    assert case.requirement().route == M64_ROUTE
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [program]
    hardening._assert_valid(out, case)
    monkeypatch.setattr(cake, rule, lambda variant, **kwargs: variant)
    four_stage = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [program, KV4_VARIANT]
    assert torch.equal(out, four_stage), (
        "the ragged one-token row differs from the four-stage program"
    )


@pytest.mark.parametrize("rows,width,chunk", ONE_TOKEN_TILED_ROWS)
def test_one_token_row_tiled_chunks_run_the_rules(monkeypatch, rows, width, chunk):
    """A workspace of ``chunk`` rows tiles a one-token call outside both rules
    into chunks inside one: the single launch runs the four-stage program,
    every chunk the rule's program on 212 or 216 SMs (elsewhere the
    four-stage program), with the bits of the single launch."""
    _skip_unless_sm107()
    program, _ = _one_token_rule(width)
    case = _decode_case(1, rows, width, 64, lengths="mixed")
    assert case.batch_size == rows > 64 and case.max_q_len == 1
    req = case.requirement()
    assert req.route == M64_ROUTE
    small_bytes = cake_dsv4_workspace_layout(
        chunk, HEADS, 1, num_query_offsets=case.batch_size + 1
    ).total_bytes
    assert req.rows_per_launch(small_bytes) == chunk
    small = torch.zeros(small_bytes, dtype=torch.uint8, device="cuda:0")
    launched = _record_launches(monkeypatch)
    single = case.run(workspace=_workspace(case), out=case.nan_out(), enable_pdl=True)
    tiled = case.run(workspace=small, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [KV4_VARIANT] + [program] * (rows // chunk)
    hardening._assert_valid(tiled, case)
    assert torch.equal(tiled, single), "row-tiled launches differ from one launch"
