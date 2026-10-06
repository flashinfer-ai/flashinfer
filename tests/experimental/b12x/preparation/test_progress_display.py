"""Rendering and accounting checks for the rank-zero preparation dashboard."""
import io
from collections import Counter
import time

import pytest

from b12x.preparation.types import PreparationProgress
from b12x.preparation._progress import PreparationDisplay, _bar, _us

rich = pytest.importorskip("rich")
from rich.console import Console  # noqa: E402


def _progress(**overrides):
    base = dict(
        running=False, pending_compilation=False, ready_collectives=(), done=False,
        phase="planning", component_id="", request_name="", completed_requests=0,
        total_requests=6, candidate_count=0, candidates_prepared=0, measured_candidates=0,
        completed_rounds=0, total_rounds=0, latest_round_us=(), cache_hits=0,
        compilations=0, active_compilations=0, elapsed_seconds=0.0, tuning_stopped=False,
    )
    base.update(overrides)
    sources = ("cached", "fixed", "tuned", "cached", "default", "fixed")
    base.setdefault("selection_counts", tuple(sorted(
        Counter(sources[:base["completed_requests"]]).items()
    )))
    return PreparationProgress(**base)


def _timeline():
    """Two cached, two fixed, one tuned, and one stopped request."""
    yield _progress(phase="planning", total_requests=0)
    yield _progress(phase="planning", cache_hits=2)
    common = dict(cache_hits=2)
    # Request 0: cached lookup from planning (zero candidates).
    yield _progress(phase="selecting", component_id="gemm.mm", request_name="mm_a", **common)
    yield _progress(phase="priming", component_id="gemm.mm", request_name="mm_a", running=True, **common)
    yield _progress(phase="priming", component_id="gemm.mm", request_name="mm_a", completed_requests=1, **common)
    # Request 1: single-candidate space compiled in the pool.
    yield _progress(phase="selecting", component_id="norm.mhc", request_name="mhc", candidate_count=1, completed_requests=1, **common)
    yield _progress(phase="compiling", component_id="norm.mhc", request_name="mhc", candidate_count=1, completed_requests=1,
                    pending_compilation=True, compilations=3, active_compilations=2, **common)
    yield _progress(phase="priming", component_id="norm.mhc", request_name="mhc", candidate_count=1, completed_requests=2,
                    compilations=3, **common)
    # Request 2: a four-candidate race over three rounds.
    race = dict(component_id="attention.dense_mla", request_name="mla_q1", candidate_count=4, compilations=7, **common)
    yield _progress(phase="selecting", completed_requests=2, **race)
    yield _progress(phase="preparing candidates", completed_requests=2, candidates_prepared=2, **race)
    yield _progress(phase="calibrating", completed_requests=2, candidates_prepared=4, **race)
    rounds = ((50.0, 41.0, 66.0, 45.0), (52.0, 40.0, 64.0, 44.0), (51.0, 42.0, 65.0, 46.0))
    for turn, latest in enumerate(rounds, start=1):
        yield _progress(phase="autotuning", completed_requests=2, candidates_prepared=4, completed_rounds=turn,
                        total_rounds=3, latest_round_us=latest, **race)
    yield _progress(phase="priming", completed_requests=2, candidates_prepared=4, completed_rounds=3, total_rounds=3,
                    latest_round_us=rounds[-1], measured_candidates=4, **race)
    yield _progress(phase="priming", completed_requests=3, candidates_prepared=4, measured_candidates=4, **race)
    # Request 3: a lookup hit during selection (cache hits grow while the request is active).
    yield _progress(phase="selecting", component_id="moe.fused_moe", request_name="moe", candidate_count=6,
                    completed_requests=3, measured_candidates=4, cache_hits=3, compilations=7)
    yield _progress(phase="priming", component_id="moe.fused_moe", request_name="moe", candidate_count=6,
                    completed_requests=4, measured_candidates=4, cache_hits=3, compilations=7)
    # Request 4: tuning stopped before the race, defaults chosen.
    yield _progress(phase="selecting", component_id="sequence.ple", request_name="ple", candidate_count=8,
                    completed_requests=4, measured_candidates=4, cache_hits=3, compilations=7, tuning_stopped=True)
    yield _progress(phase="priming", component_id="sequence.ple", request_name="ple", candidate_count=8,
                    completed_requests=5, measured_candidates=4, cache_hits=3, compilations=7, tuning_stopped=True)
    # Request 5: waits for other ranks, then completes as a fixed selection.
    class Requirement:
        key = ("allreduce", 8)
    yield _progress(phase="waiting for ranks", component_id="comm.pcie", request_name="pcie", candidate_count=1,
                    completed_requests=5, measured_candidates=4, cache_hits=3, compilations=7,
                    ready_collectives=(Requirement(),), tuning_stopped=True)
    yield _progress(phase="priming", component_id="comm.pcie", request_name="pcie", candidate_count=1,
                    completed_requests=6, measured_candidates=4, cache_hits=3, compilations=7, tuning_stopped=True)
    yield _progress(phase="finishing", completed_requests=6, measured_candidates=4, cache_hits=3, compilations=7,
                    tuning_stopped=True)
    yield _progress(phase="ready", done=True, completed_requests=6, measured_candidates=4, cache_hits=3,
                    compilations=7, tuning_stopped=True)


def _attach(width, height):
    display = PreparationDisplay(global_rank=0, stream=io.StringIO())
    display._started = time.monotonic()
    display._request_started = display._started
    console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor",
                      width=width, height=height, record=True)
    display._console = console
    return display, console


def _render_text(display, console):
    console.print(display._render())
    return console.export_text(clear=True)


@pytest.mark.parametrize("width,height", [(140, 40), (100, 16), (84, 12), (80, 24), (60, 24), (40, 8)])
def test_every_phase_renders_within_console_width(width, height):
    display, console = _attach(width, height)
    for progress in _timeline():
        display.update(progress)
        text = _render_text(display, console)
        assert text.strip(), progress.phase
        assert max(len(line) for line in text.splitlines()) <= width, (progress.phase, text)
    display.close(failed=True)
    text = _render_text(display, console)
    assert "FAILED" in text or "failed" in text
    assert max(len(line) for line in text.splitlines()) <= width


def test_reported_outcomes_and_measured_lanes_render_from_snapshots():
    display, console = _attach(140, 40)
    seen_lanes = None
    for progress in _timeline():
        display.update(progress)
        if progress.phase == "autotuning" and progress.completed_rounds == 3:
            seen_lanes = display._frame.lanes
    assert seen_lanes is not None
    assert [lane.index for lane in seen_lanes] == [1, 3, 0, 2]
    assert seen_lanes[0].median_us == pytest.approx(41.0)
    assert seen_lanes[0].history == (41.0, 40.0, 42.0)
    frame = display._frame
    assert dict(frame.progress.selection_counts) == {
        "cached": 2, "fixed": 2, "tuned": 1, "default": 1,
    }
    assert frame.widest is not None and frame.widest[1] == "attention.dense_mla"
    assert frame.widest[0] == pytest.approx(65.0 / 41.0)
    text = _render_text(display, console)
    assert "kernels ready" in text
    assert "6 requests" in text
    assert "2 tuned" not in text and "1 tuned" in text


def test_race_history_resets_between_requests():
    display, _ = _attach(140, 40)
    race = dict(component_id="attention.paged", request_name="paged", candidate_count=2, total_requests=2)
    display.update(_progress(phase="autotuning", completed_rounds=1, total_rounds=2, latest_round_us=(10.0, 20.0), **race))
    display.update(_progress(phase="autotuning", completed_rounds=2, total_rounds=2, latest_round_us=(12.0, 18.0), **race))
    assert display._frame.lanes[0].history == (10.0, 12.0)
    display.update(_progress(phase="selecting", component_id="attention.qsa", request_name="qsa", candidate_count=3,
                             completed_requests=1, total_requests=2, measured_candidates=2))
    assert display._frame.lanes == ()
    display.update(_progress(phase="autotuning", component_id="attention.qsa", request_name="qsa", candidate_count=3,
                             completed_requests=1, total_requests=2, measured_candidates=2, completed_rounds=1,
                             total_rounds=2, latest_round_us=(5.0, 7.0, 6.0)))
    assert [lane.index for lane in display._frame.lanes] == [0, 2, 1]


def test_pipe_output_is_plain_milestone_text():
    stream = io.StringIO()
    with PreparationDisplay(global_rank=0, stream=stream) as display:
        for progress in _timeline():
            display.update(progress)
    lines = stream.getvalue().splitlines()
    assert lines and all(line.startswith("b12x ") for line in lines)
    assert all(character.isprintable() for line in lines for character in line)
    assert lines[-1].startswith("b12x ready: 6/6 ready")
    assert any("tuning stopped" in line for line in lines)


def test_nonzero_ranks_are_silent():
    stream = io.StringIO()
    with PreparationDisplay(global_rank=1, stream=stream) as display:
        for progress in _timeline():
            display.update(progress)
    assert stream.getvalue() == ""


def test_display_guards():
    display = PreparationDisplay(global_rank=0, stream=io.StringIO())
    with pytest.raises(RuntimeError):
        display.update(_progress())
    with display, pytest.raises(TypeError):
        display.update(object())
    with pytest.raises(RuntimeError):
        display.__enter__()
    with pytest.raises(ValueError):
        PreparationDisplay(global_rank=-1)


def test_bar_and_number_helpers():
    assert _bar(0.5, 10).cell_len == 10
    assert _bar(1.0, 7).plain == "█" * 7
    assert _bar(0.0, 4).plain == "━━━━"
    assert _us(41.26) == "41.3 µs" and _us(512.0) == "512 µs" and _us(2500.0) == "2.50 ms"


def test_batch_change_resets_measurements_when_intermediate_rounds_are_not_reported():
    display, console = _attach(160, 40)
    common = dict(
        phase="autotuning", component_id="norm.mhc", request_name="mhc.post_pre.m24",
        candidate_count=272, batch_candidates=33, completed_rounds=2, total_rounds=3,
        tuning_rank=2,
    )
    display.update(_progress(batch_index=2, latest_round_us=(10.0, 20.0), **common))
    display.update(_progress(batch_index=3, latest_round_us=(40.0, 30.0), **common))
    assert display._frame.lanes[0].history == (30.0,)
    assert display._frame.lanes[1].history == (40.0,)
    assert "rank 2 batch 3" in _render_text(display, console)


def test_progress_bar_measures_candidate_work_independently_of_request_count():
    display, console = _attach(180, 40)
    display.update(_progress(
        phase="autotuning", component_id="norm.mhc", request_name="mhc.post_pre.m4096",
        completed_requests=1, total_requests=100, measured_candidates=900, total_candidates=1000,
        candidate_count=43, global_candidate_count=172, candidate_sharded=True,
        batch_candidates=20, batch_index=2, completed_rounds=1, total_rounds=3,
    ))
    text = _render_text(display, console)
    assert "900 / 1000 candidates measured" in text
    assert "90%" in text
    assert "20 in batch" in text
    assert "43 on rank / 172 total" in text
    assert "1 / 100 requests" not in text


def test_planning_does_not_present_request_fraction_as_remaining_work():
    display, console = _attach(140, 40)
    display.update(_progress(completed_requests=1, total_requests=100))
    text = _render_text(display, console)
    assert "Counting candidates" in text
    assert "1 / 100 requests ready" in text
    assert "%" not in text


@pytest.mark.parametrize("width", [60, 84])
def test_escape_hint_and_cancellation_status(width):
    stream = io.StringIO()
    display = PreparationDisplay(global_rank=0, stream=stream, cancel_available=True)
    with display:
        display._console = Console(file=stream, width=width, height=24, color_system=None)
        display.update(_progress(phase="autotuning"))
        display._console.print(display._render())
        assert "Press ESC to use default tuning" in stream.getvalue()
        stream.seek(0)
        stream.truncate()
        display.tuning_stopped()
        display._console.print(display._render())
        assert "Press ESC to use default tuning" not in stream.getvalue()
        assert "stopping" in stream.getvalue()


def test_results_use_selections_when_completion_snapshots_are_skipped():
    display, console = _attach(160, 40)
    display.update(_progress(phase="autotuning", candidate_count=64))
    display.update(_progress(
        phase="ready", done=True, completed_requests=298, total_requests=298,
        measured_candidates=64000,
        selection_counts=(("tuned", 279), ("fixed", 1), ("default", 17), ("override", 1)),
    ))
    text = _render_text(display, console)
    assert "279 tuned" in text and "1 fixed" in text
    assert "17 default" in text and "1 override" in text
    assert "279 fixed" not in text
