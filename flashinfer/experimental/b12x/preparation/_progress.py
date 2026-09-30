"""Rank-zero live dashboard for kernel preparation, with plain milestone lines for pipes.

The dashboard uses fixed rows for overall progress, request identity, activity,
measurements, and cumulative work. Color marks status; bars change only when
progress changes.
All accounting happens on the caller's thread inside ``update`` and is published
as one immutable frame; rich's refresh thread only reads that frame and the clock,
so it never touches a session, CUDA, a compiler, or a measurement object.
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, replace

from .types import PreparationProgress

_STAGES = ("PLAN", "BUILD", "TUNE", "PRIME", "READY")
_PHASE_STAGE = {
    "planning": "PLAN", "selecting": "PLAN", "compiling": "BUILD",
    "preparing candidates": "TUNE", "calibrating": "TUNE", "autotuning": "TUNE",
    "priming": "PRIME", "warming heuristics": "PRIME", "finishing": "PRIME", "ready": "READY", "waiting for ranks": "WAIT",
}
_PHASE_LABEL = {
    "planning": "Planning requests", "selecting": "Selecting configuration",
    "compiling": "Compiling kernels", "preparing candidates": "Preparing candidates",
    "calibrating": "Building measurement graphs", "autotuning": "Measuring candidates",
    "priming": "Priming kernels", "warming heuristics": "Compiling and warming heuristic kernels",
    "finishing": "Finalizing preparation",
    "ready": "All kernels ready", "waiting for ranks": "Waiting for ranks",
}
_INK = "#d8dfe5"
_DIM = "#939faa"
_SIGNAL = "#83b9ad"
_TRACK = "#46515b"
_WARNING = "#d6b77b"
_ERROR = "#df9292"
_STAGE_STYLE = {stage: _SIGNAL for stage in _STAGES}
_STAGE_STYLE.update(WAIT=_WARNING, FAILED=_ERROR)
_OUTCOME_PRIORITY = ("tuned", "cached", "fixed", "default", "override")
_EIGHTHS = "▏▎▍▌▋▊▉█"
_RACE_PHASES = frozenset({"preparing candidates", "calibrating", "autotuning"})


def _plain(value):
    return "".join(character if character.isprintable() else " " for character in value)


def _duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _us(value):
    if value >= 1000:
        return f"{value / 1000:.2f} ms"
    if value >= 100:
        return f"{value:.0f} µs"
    return f"{value:.1f} µs"


def _bar(fraction, cells):
    from rich.text import Text

    cells = max(1, cells)
    eighths = round(min(1.0, max(0.0, fraction)) * cells * 8)
    full, remainder = divmod(eighths, 8)
    text = Text("█" * full, style=_SIGNAL, no_wrap=True)
    if remainder:
        text.append(_EIGHTHS[remainder - 1], style=_SIGNAL)
    text.append("━" * (cells - full - bool(remainder)), style=_TRACK)
    return text


@dataclass(frozen=True)
class _Lane:
    index: int
    median_us: float
    history: tuple[float, ...]


@dataclass(frozen=True)
class _Frame:
    progress: PreparationProgress
    stage: str
    lanes: tuple[_Lane, ...] = ()
    previous_medians: tuple[tuple[int, float], ...] = ()
    lanes_changed_at: float = 0.0
    compile_rate: tuple[int, ...] = ()
    peak_active: int = 0
    widest: tuple[float, str] | None = None
    request_started: float = 0.0
    failed: bool = False


class PreparationDisplay:
    """Live dashboard for global rank zero; plain milestone lines when piped.

    Only the coordinator for global rank zero creates a live renderer. The
    refresh thread reads immutable frames, never a session, CUDA, a compiler,
    or a measurement object. Pipe output is plain milestone text.
    """

    def __init__(self, *, global_rank: int, stream=None, title="b12x / kernel autotuning", cancel_available=False):
        if type(global_rank) is not int or global_rank < 0:
            raise ValueError("progress display requires a nonnegative global rank")
        self._enabled = global_rank == 0
        self._title = _plain(title)
        self._cancel_available = cancel_available
        self._tuning_stopped = False
        self._stream = sys.stderr if stream is None else stream
        self._frame = _Frame(PreparationProgress(False, False, (), False), "PLAN")
        self._live = None
        self._console = None
        self._started = None
        self._ended = None
        self._closed = False
        self._last_log = 0.0
        self._logged = set()
        self._last_stop = False
        self._signature = None
        self._frame_at = 0.0
        self._request_key = None
        self._request_started = 0.0
        self._race_history = []
        self._race_rounds = 0
        self._lanes = ()
        self._previous_medians = ()
        self._lanes_changed_at = 0.0
        self._compile_seen = 0
        self._compile_buckets = {}
        self._peak_active = 0
        self._widest = None

    def __enter__(self):
        if self._started is not None or self._closed:
            raise RuntimeError("preparation display is single-use")
        self._started = time.monotonic()
        self._request_started = self._started
        if self._enabled and self._stream.isatty():
            from rich.console import Console

            self._attach(Console(file=self._stream))
        return self

    def _attach(self, console):
        from rich.live import Live

        self._console = console
        self._live = Live(
            console=console, get_renderable=self._render, refresh_per_second=2,
            screen=False, transient=False, redirect_stdout=True, redirect_stderr=True,
            vertical_overflow="crop",
        )
        self._live.start(refresh=True)

    def write_output(self, line):
        """Write a complete startup log line above the live panel."""
        if self._live is not None:
            from rich.text import Text

            self._console.print(Text.from_ansi(line), highlight=False)
        else:
            self._stream.write(line + "\n")
            self._stream.flush()

    def tuning_stopped(self):
        self._tuning_stopped = True
        self._cancel_available = False

    def update(self, progress: PreparationProgress):
        if self._started is None or self._closed:
            raise RuntimeError("progress updates require an active display context")
        if not isinstance(progress, PreparationProgress):
            raise TypeError("display requires a PreparationProgress snapshot")
        if not self._enabled:
            return
        now = time.monotonic()
        signature = (
            progress.phase, progress.component_id, progress.request_name, progress.completed_requests,
            progress.candidate_count, progress.candidates_prepared, progress.completed_rounds,
            progress.measured_candidates, progress.cache_hits, progress.compilations,
            progress.active_compilations, progress.tuning_stopped, progress.done,
            progress.batch_index, progress.batch_candidates, progress.tuning_rank,
            progress.total_candidates, progress.global_candidate_count, progress.selection_counts,
        )
        if signature == self._signature and now - self._frame_at < 0.25 and not progress.done:
            return
        self._signature = signature
        self._frame_at = now
        self._account(progress, now)
        self._frame = _Frame(
            progress, _PHASE_STAGE.get(progress.phase, "PLAN"),
            lanes=self._lanes, previous_medians=self._previous_medians,
            lanes_changed_at=self._lanes_changed_at,
            compile_rate=self._rate(now), peak_active=self._peak_active, widest=self._widest,
            request_started=self._request_started,
        )
        if progress.done:
            self._ended = now
        if self._live is not None:
            if progress.done:
                self._live.stop()
            return
        milestone = (progress.component_id, progress.phase)
        if (milestone not in self._logged or progress.done
                or progress.tuning_stopped != self._last_stop or now - self._last_log >= 10):
            self._logged.add(milestone)
            self._last_log = now
            self._last_stop = progress.tuning_stopped
            self._write_milestone()

    def _account(self, progress, now):
        previous = self._frame.progress
        tuned = dict(progress.selection_counts).get("tuned", 0)
        previous_tuned = dict(previous.selection_counts).get("tuned", 0)
        if tuned > previous_tuned and len(self._lanes) > 1:
            spread = self._lanes[-1].median_us / self._lanes[0].median_us
            if self._widest is None or spread > self._widest[0]:
                self._widest = (spread, previous.component_id)
        key = (progress.component_id, progress.request_name)
        if key != self._request_key:
            self._request_key = key
            self._request_started = now
            self._reset_race()
        if (progress.batch_index != previous.batch_index
                or progress.tuning_rank != previous.tuning_rank
                or progress.completed_rounds < self._race_rounds):
            self._reset_race()
        if progress.completed_rounds > self._race_rounds:
            values = progress.latest_round_us
            if len(values) != len(self._race_history):
                self._race_history = [[] for _ in values]
            for series, value in zip(self._race_history, values, strict=True):
                if math.isfinite(value):
                    series.append(value)
            self._race_rounds = progress.completed_rounds
            self._previous_medians = tuple((lane.index, lane.median_us) for lane in self._lanes)
            lanes = []
            for index, series in enumerate(self._race_history):
                if series and all(math.isfinite(value) and value > 0 for value in series):
                    lanes.append(_Lane(index, statistics.median(series), tuple(series[-8:])))
            self._lanes = tuple(sorted(lanes, key=lambda lane: (lane.median_us, lane.index)))
            self._lanes_changed_at = now
        built = progress.compilations - self._compile_seen
        if built > 0:
            bucket = int((now - self._started) / 2)
            self._compile_buckets[bucket] = self._compile_buckets.get(bucket, 0) + built
            self._compile_seen = progress.compilations
            for stale in [key for key in self._compile_buckets if key < bucket - 24]:
                del self._compile_buckets[stale]
        self._peak_active = max(self._peak_active, progress.active_compilations)

    def _reset_race(self):
        self._race_history = []
        self._race_rounds = 0
        self._lanes = ()
        self._previous_medians = ()

    def _rate(self, now):
        if not self._compile_buckets:
            return ()
        bucket = int((now - self._started) / 2)
        return tuple(self._compile_buckets.get(index, 0) for index in range(bucket - 23, bucket + 1))

    def _elapsed(self):
        return (self._ended if self._ended is not None else time.monotonic()) - self._started

    def _write_milestone(self):
        p = self._frame.progress
        phase = "failed" if self._frame.failed else p.phase
        component = f" {p.component_id}" if p.component_id else ""
        candidates = f", candidates {p.candidates_prepared}/{p.candidate_count} prepared" if p.candidate_count else ""
        batch = f", rank {p.tuning_rank} batch {p.batch_index}" if p.batch_index else ""
        rounds = f", round {p.completed_rounds}/{p.total_rounds}" if p.total_rounds else ""
        stopped = ", tuning stopped; completing required preparation" if p.tuning_stopped and not p.done else ""
        line = (
            f"b12x {phase}{component}: {p.completed_requests}/{p.total_requests} ready"
            f"{candidates}{batch}{rounds}, {p.measured_candidates} measured, {p.cache_hits} cached, "
            f"{p.compilations} compilations, {_duration(self._elapsed())}{stopped}"
        )
        self._stream.write(_plain(line) + "\n")
        self._stream.flush()

    # Rendering runs on rich's refresh thread and reads only the published frame.

    def _render(self):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        frame = self._frame
        p = frame.progress
        if self._stream.isatty():
            try:
                size = os.get_terminal_size(self._stream.fileno())
            except (AttributeError, OSError, ValueError):
                pass
            else:
                if size.columns > 0 and size.lines > 0:
                    self._console.size = size
        width, height = self._console.width, self._console.height
        if width < 76 or height < 10:
            return self._render_line(frame, width)
        inner = width - 6
        rows = [
            self._rail_row(frame),
            self._progress_row(frame, inner),
            Text("─" * inner, style=_TRACK),
            self._field("REQUEST", p.component_id or "—", p.request_name),
            self._activity_row(frame),
            self._detail_row(frame),
            Text("─" * inner, style=_TRACK),
            self._summary_row(frame),
        ]
        grid = Table.grid(expand=True)
        grid.add_column(no_wrap=True, overflow="ellipsis")
        for row in rows:
            grid.add_row(row)
        return Panel(
            grid, box=box.SQUARE, border_style=_TRACK, padding=(0, 2), expand=True,
            title=Text(f" {self._title} ", style=f"bold {_INK}"),
            title_align="left",
            subtitle=Text("Press ESC to use default tuning", style=_DIM)
            if self._cancel_available and not p.tuning_stopped and not p.done else None,
            subtitle_align="right",
        )

    def _two(self, left, right):
        from rich.table import Table

        grid = Table.grid(padding=(0, 2), expand=True)
        grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        grid.add_column(no_wrap=True, justify="right")
        grid.add_row(left, right)
        return grid

    def _field(self, label, value, detail=""):
        from rich.text import Text

        text = Text(f"{label:<10}", style=_DIM, no_wrap=True, overflow="ellipsis")
        text.append(_plain(value), style=_INK)
        if detail:
            text.append("  /  ", style=_TRACK)
            text.append(_plain(detail), style=_DIM)
        return text

    def _rail_row(self, frame):
        from rich.text import Text

        rail = Text(no_wrap=True)
        for index, stage in enumerate(_STAGES):
            if index:
                rail.append("  /  ", style=_TRACK)
            active = stage == frame.stage and not frame.failed
            rail.append(stage, style=f"bold {_SIGNAL}" if active else _DIM)
        clock = Text(f"elapsed {_duration(self._elapsed())}", style=_DIM, no_wrap=True)
        return self._two(rail, clock)

    def _progress_row(self, frame, inner):
        from rich.text import Text

        p = frame.progress
        if p.total_candidates:
            fraction = p.measured_candidates / p.total_candidates
            label = Text(
                f"{p.measured_candidates} / {p.total_candidates} candidates measured",
                style=_INK, no_wrap=True,
            )
            label.append(f"   {fraction:4.0%}", style=_SIGNAL)
            return self._two(_bar(fraction, max(8, inner - label.cell_len - 2)), label)
        detail = "Counting candidates" if p.total_candidates is None and not p.done else "No candidate races"
        return self._two(Text(detail, style=_DIM), Text(
            f"{p.completed_requests} / {p.total_requests} requests ready", style=_INK, no_wrap=True,
        ))

    def _activity_row(self, frame):
        p = frame.progress
        if frame.failed:
            text = self._field("STATUS", "Preparation failed", f"during {p.phase}")
            text.stylize(_ERROR, 10)
            return text
        if (p.tuning_stopped or self._tuning_stopped) and not p.done:
            detail = "compiling and warming heuristics" if p.tuning_stopped else "stopping autotuning across ranks"
            text = self._field("STATUS", "Tuning stopped", detail)
            text.stylize(_WARNING, 10)
            return text
        detail = ""
        if p.phase == "autotuning":
            count = p.batch_candidates or p.candidate_count
            detail = f"round {p.completed_rounds} / {p.total_rounds}  ·  {count} in batch"
            if p.active_count and p.active_count < count:
                detail += f"  ·  {p.active_count} timed"
        elif p.phase == "preparing candidates":
            detail = f"{p.candidates_prepared} / {p.candidate_count} prepared"
        elif p.phase == "calibrating":
            detail = f"{p.batch_candidates or p.candidate_count} candidates"
        elif p.phase == "compiling":
            detail = f"{p.active_compilations} active  ·  {p.compilations} built"
        if p.batch_index and p.phase in _RACE_PHASES:
            detail = f"rank {p.tuning_rank} batch {p.batch_index}  ·  {detail}"
            if p.candidate_sharded:
                detail += f"  ·  {p.candidate_count} on rank / {p.global_candidate_count} total"
        return self._field("STATUS", _PHASE_LABEL.get(p.phase, p.phase), detail)

    def _detail_row(self, frame):
        p = frame.progress
        if frame.lanes and p.phase in ("autotuning", "priming", "compiling"):
            leader = frame.lanes[0]
            detail = f"#{leader.index} median"
            for lane in frame.lanes[1:3]:
                delta = 100 * (lane.median_us / leader.median_us - 1)
                detail += f"  ·  #{lane.index} {delta:+.0f}%"
            if len(frame.lanes) > 3:
                detail += f"  ·  {len(frame.lanes) - 3} more"
            return self._field("BEST", _us(leader.median_us), detail)
        if p.phase == "waiting for ranks":
            keys = ", ".join(_plain(str(item.key)) for item in p.ready_collectives[:2])
            return self._field("SYNC", keys or "Waiting for other ranks")
        if p.phase in _RACE_PHASES:
            return self._field("BEST", "Awaiting measurements")
        if p.done or frame.failed:
            selected = dict(p.selection_counts)
            counts = [f"{selected[outcome]} {outcome}" for outcome in _OUTCOME_PRIORITY
                      if selected.get(outcome, 0)]
            return self._field("RESULT", "  ·  ".join(counts) or "No selection results")
        return self._field("MEASURED", f"{p.measured_candidates} candidates")

    def _summary_row(self, frame):
        p = frame.progress
        return self._field("TOTAL", f"{p.completed_requests} requests",
                           f"{p.compilations} compiled  ·  {p.cache_hits} cached  ·  {p.measured_candidates} measured")

    def _render_line(self, frame, width):
        from rich.text import Text

        p = frame.progress
        stage = "FAILED" if frame.failed else frame.stage
        color = _STAGE_STYLE[stage]
        line = Text("b12x  ", style=f"bold {_INK}", no_wrap=True, overflow="ellipsis")
        line.append(stage, style=f"bold {color}")
        show_hint = self._cancel_available and not p.done and not p.tuning_stopped
        if not show_hint:
            if p.total_candidates:
                line.append(f"  {p.measured_candidates}/{p.total_candidates} candidates", style=_INK)
            else:
                line.append(f"  {p.completed_requests}/{p.total_requests} ready", style=_INK)
        if (p.tuning_stopped or self._tuning_stopped) and not p.done:
            line.append("  warming heuristics" if p.tuning_stopped else "  stopping tuning", style=_WARNING)
        elif self._cancel_available and not p.done:
            line.append("  Press ESC to use default tuning", style=_DIM)
        elif p.phase == "autotuning":
            line.append(f"  round {p.completed_rounds}/{p.total_rounds}", style=_DIM)
        if width >= 60 and p.component_id and not p.done:
            line.append(f"  {_plain(p.component_id)}", style=_DIM)
        return self._two(line, Text(_duration(self._elapsed()), style=_DIM))

    def close(self, *, failed=False):
        if self._closed:
            return
        if failed:
            self._frame = replace(self._frame, failed=True)
        if self._started is not None and self._ended is None:
            self._ended = time.monotonic()
        try:
            if self._live is not None:
                self._live.stop()
            elif self._enabled and failed and self._started is not None:
                self._write_milestone()
        finally:
            self._closed = True

    def __exit__(self, kind, value, traceback):
        self.close(failed=kind is not None)
        return False
