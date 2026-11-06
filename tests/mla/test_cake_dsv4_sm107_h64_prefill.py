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
program.  Skips without an SM107 GPU.
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
