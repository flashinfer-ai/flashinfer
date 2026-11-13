"""GPU tests of the SM107 V-block sibling of the CAKE DSv4 FP8 low-head prefill route.

On SM107 the low-head prefill route (the SWA-only FP8 rows of 8, 16, 32 and 64
heads below 257 query tokens per request) runs a launch of at most 27 query
rows (scaled down with the SM count below 216 SMs: 26 on 212 and 208) on its
V-block sibling ``fp8_lowhead_prefill_vblock``: four CTAs per row, each one
128-column V block of the output, with the early-PV sibling's output bits.  A
launch above that ceiling runs the early-PV sibling (up to 64 rows) or the
route's program.  These tests call
``trtllm_batch_decode_sparse_mla_dsv4(backend="cake")`` and check the launched
program and its grid, the output against the fp32 reference of the hardening
suite at its contract tolerance, that masked columns contribute nothing
(bit-identical output under a poisoned page), that PDL launches equal plain
ones bit for bit, that the V-block program equals the early-PV program bit for
bit on the same call (at and above the ceiling too), that launches at the
ceiling, one row above it and of 27 and 28 rows take the program the ceiling
names, and that eager calls and CUDA graph replays are bitwise identical.
Skips without an SM107 GPU.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

import pytest
import torch

from flashinfer.mla import cake_dsv4 as cake
from flashinfer.mla.cake_dsv4 import cake_dsv4_workspace_reset
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

DTYPE = torch.float8_e4m3fn
ROUTE = "fp8_lowhead_prefill"
OVERLAP = f"{ROUTE}_overlap"
VBLOCK = f"{ROUTE}_vblock"
# The V-block sibling's row ceiling and the SM count it is set for, and the
# early-PV sibling's ceiling (the host's _VBLOCK_SIBLING_* / _OVERLAP_SIBLING_*).
VBLOCK_MAX_ROWS = 27
OVERLAP_MAX_ROWS = 64
MEASURED_SMS = 216
CONTEXT = 1024
# The context of the first request of a "mixed" row: shorter than the SWA
# window, so its rows mask SWA slots (stale rows of the poison page).
SHORT_CONTEXT = 61
# (heads, request query lengths) of launches up to the ceiling of every SM
# count; "ceil" is one-row requests up to the ceiling of this device.
ROWS = [
    pytest.param(64, [1] * 8, id="h64-q1-t8"),
    pytest.param(64, [4] * 6, id="h64-q4-t24"),
    pytest.param(64, [5, 4, 3], id="h64-q543-t12"),
    pytest.param(32, [4] * 3, id="h32-q4-t12"),
    pytest.param(16, [4] * 6, id="h16-q4-t24"),
    pytest.param(8, [4] * 3, id="h8-q4-t12"),
    pytest.param(64, "ceil", id="h64-q1-tceil"),
    pytest.param(8, "ceil", id="h8-q1-tceil"),
]
# Launches at the ceiling and around it: the ceiling, one row above it, and 27
# and 28 rows (the ceiling and one above on 216 SMs).
EDGES = ["ceil", "ceil+1", 27, 28]
REPEATS = 5


def _skip_unless_sm107() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CAKE DSv4 SM107 low-head tests")
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (10, 7):
        pytest.skip(
            "the V-block sibling runs on SM107 only, "
            f"got SM{compute_capability[0]}{compute_capability[1]}"
        )


def _sm_count() -> int:
    return cake._device_sm_count(torch.device("cuda:0"))


def _ceiling() -> int:
    """The most rows of a launch the V-block sibling takes on this device."""
    return VBLOCK_MAX_ROWS * min(_sm_count(), MEASURED_SMS) // MEASURED_SMS


def _expected_variant(rows: int) -> str:
    """The variant a low-head prefill launch of ``rows`` rows runs here."""
    sm_count = min(_sm_count(), MEASURED_SMS)
    if rows <= VBLOCK_MAX_ROWS * sm_count // MEASURED_SMS:
        return VBLOCK
    if rows <= OVERLAP_MAX_ROWS * sm_count // MEASURED_SMS:
        return OVERLAP
    return ROUTE


def _expected_launch(rows: int) -> tuple:
    """(variant, grid, work items) of a low-head prefill launch of ``rows``."""
    variant = _expected_variant(rows)
    items = (4 if variant == VBLOCK else 2) * rows
    return variant, (items, 1, 1), items


def _require_vblock_program() -> None:
    """SM107 routes the V-block sibling to a program of its own, other than
    the route's and the early-PV sibling's, with the route's arguments."""
    from flashinfer.jit.cake_dsv4 import VARIANTS, get_cake_dsv4_spec

    programs = VARIANTS["sm_107a"]
    assert len({programs[ROUTE], programs[OVERLAP], programs[VBLOCK]}) == 3
    assert (
        get_cake_dsv4_spec(VBLOCK, arch="sm_107a")["arg_plan"]
        == get_cake_dsv4_spec(ROUTE, arch="sm_107a")["arg_plan"]
    )


def _record_launches(monkeypatch) -> list:
    """Record (variant, grid, work items) of every prepared launch."""
    launched: list = []
    prepare = cake._prepare_variant

    def record(variant, **kwargs):
        items = int(kwargs["values"]["total_work_items"])
        launched.append((variant, tuple(kwargs["grid"]), items))
        return prepare(variant, **kwargs)

    monkeypatch.setattr(cake, "_prepare_variant", record)
    return launched


def _q_lens(q_lens) -> list:
    """Request query lengths: "ceil" / "ceil+1" are one-row requests up to the
    device's ceiling and one row above it; an int is that many one-row
    requests."""
    if isinstance(q_lens, list):
        return q_lens
    rows = {"ceil": _ceiling(), "ceil+1": _ceiling() + 1}.get(q_lens, q_lens)
    return [1] * int(rows)


def _case(num_heads: int, q_lens, *, lengths: str = "full"):
    """One SWA-only FP8 call of the low-head prefill route.  "full" gives every
    request a long context; "mixed" gives the first one a context shorter than
    the SWA window, whose masked slots hold stale rows of the poison page."""
    q_lens = _q_lens(q_lens)
    seq_lens = [CONTEXT + 37 * i for i in range(len(q_lens))]
    mixed = lengths == "mixed"
    if mixed:
        seq_lens[0] = SHORT_CONTEXT
    case = hardening._build_case(
        num_heads=num_heads,
        dtype=DTYPE,
        seq_lens=seq_lens,
        q_lens=q_lens,
        layout="ragged" if len(set(q_lens)) > 1 else "dense",
        swa_tail="stale" if mixed else "minus_one",
    )
    assert case.rows == sum(q_lens)
    assert int(case.sparse_indices.shape[1]) == hardening.SWA_WINDOW
    req = case.requirement()
    assert (req.route, req.num_splits) == (ROUTE, 1), req.route
    if mixed:
        assert int((case.visible() < hardening.SWA_WINDOW).sum()) > 0
    return case


def _workspace(case) -> torch.Tensor:
    """A workspace of exactly the call's single-launch requirement."""
    num_bytes = case.requirement().single_launch_bytes
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device="cuda:0")
    cake_dsv4_workspace_reset(workspace)
    return workspace


def test_the_ceiling_is_the_hosts():
    """The host's V-block ceiling and its pick on this device are the ones
    these tests expect."""
    _skip_unless_sm107()
    _require_vblock_program()
    assert cake._VBLOCK_SIBLING_MAX_TOKENS == {"sm_107a": VBLOCK_MAX_ROWS}
    assert cake._VBLOCK_SIBLING_MEASURED_SMS == {"sm_107a": MEASURED_SMS}
    assert cake._OVERLAP_SIBLING_MAX_TOKENS == {"sm_107a": OVERLAP_MAX_ROWS}
    assert cake._VBLOCK_ITEMS_PER_TOKEN == 4
    device = torch.device("cuda:0")
    for rows in range(1, 2 * OVERLAP_MAX_ROWS):
        picked = cake._vblock_variant(
            ROUTE, arch="sm_107a", num_query_tokens=rows, device=device
        )
        assert picked == (VBLOCK if _expected_variant(rows) == VBLOCK else ROUTE)


@pytest.mark.parametrize("lengths", ["full", "mixed"])
@pytest.mark.parametrize("num_heads,q_lens", ROWS)
def test_rows_up_to_the_ceiling_run_the_vblock_program(
    monkeypatch, num_heads, q_lens, lengths
):
    """The launch runs the V-block program over four CTAs and work items per
    row, matches the reference, is bit-identical with the masked columns' page
    poisoned, equals the launch without PDL bit for bit and equals the early-PV
    program's output on the same call bit for bit."""
    _skip_unless_sm107()
    _require_vblock_program()
    case = _case(num_heads, q_lens, lengths=lengths)
    expected = _expected_launch(case.rows)
    assert expected[0] == VBLOCK, (case.rows, _ceiling())
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    pdl = hardening._assert_poison_inert(case, workspace, enable_pdl=True)
    plain = case.run(workspace=workspace, out=case.nan_out())
    torch.cuda.synchronize()
    assert launched == [expected] * 3
    assert torch.equal(plain, pdl), "the PDL launch differs from the plain one"
    monkeypatch.setattr(cake, "_vblock_variant", lambda route, **kwargs: route)
    early = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    rows = case.rows
    assert launched == [expected] * 3 + [(OVERLAP, (2 * rows, 1, 1), 2 * rows)]
    assert torch.equal(early, pdl), "the V-block program differs from the early-PV one"


@pytest.mark.parametrize("num_heads", [64, 8])
@pytest.mark.parametrize("rows", EDGES)
def test_launches_around_the_ceiling_take_the_program_it_names(
    monkeypatch, rows, num_heads
):
    """A launch at the ceiling runs the V-block program, one above it the
    early-PV program, each matching the reference; raising the host's ceiling
    runs the V-block program on the same call with the same bits."""
    _skip_unless_sm107()
    _require_vblock_program()
    case = _case(num_heads, rows, lengths="mixed")
    expected = _expected_launch(case.rows)
    assert expected[0] in (VBLOCK, OVERLAP), expected
    if rows in ("ceil", "ceil+1"):
        assert expected[0] == (VBLOCK if rows == "ceil" else OVERLAP)
    workspace = _workspace(case)
    launched = _record_launches(monkeypatch)
    out = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    assert launched == [expected]
    hardening._assert_valid(out, case)
    monkeypatch.setitem(cake._VBLOCK_SIBLING_MAX_TOKENS, "sm_107a", OVERLAP_MAX_ROWS)
    raised = case.run(workspace=workspace, out=case.nan_out(), enable_pdl=True)
    torch.cuda.synchronize()
    rows = case.rows
    assert launched == [expected, (VBLOCK, (4 * rows, 1, 1), 4 * rows)]
    assert torch.equal(raised, out), "the V-block program differs from the early-PV one"


def test_vblock_program_is_deterministic(monkeypatch):
    """Eager calls and CUDA graph replays of the V-block program with PDL (24
    rows of 64 heads in 4-row requests, masked SWA slots) are bitwise
    identical; inside the graph the program reads the query the copy right
    before it writes.  Float scales throughout: a scale tensor made inside the
    capture would copy from host memory."""
    _skip_unless_sm107()
    _require_vblock_program()
    case = _case(64, [4] * 6, lengths="mixed")
    expected = _expected_launch(case.rows)
    assert expected[0] == VBLOCK
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
