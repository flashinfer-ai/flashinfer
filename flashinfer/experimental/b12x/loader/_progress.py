"""Checkpoint loading display using the kernel-preparation dashboard's styling."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import sys
from threading import RLock
import time

from b12x.preparation._progress import (
    _bar, _duration, _plain, _DIM, _ERROR, _INK, _SIGNAL, _TRACK,
)

_STAGES = ("ROUTE", "SETUP", "PLAN", "READ", "SYNC", "READY")
_PHASES = {
    "routing": ("ROUTE", "Routing checkpoint shards"),
    "prepare": ("SETUP", "Preparing shared reads"),
    "plan": ("PLAN", "Planning shared reads"),
    "execute": ("READ", "Reading and scattering weights"),
    "unmap": ("SYNC", "Completing weight transfers"),
    "ready": ("READY", "Weights loaded"),
    "failed": ("FAILED", "Weight loading failed"),
}


@dataclass(frozen=True)
class _Frame:
    phase: str = "routing"
    shards: int = 0
    total: int = 0
    filename: str = ""
    selected_bytes: int = 0
    summary: tuple[tuple[str, int | float], ...] = ()


class CheckpointDisplay:
    """Publish immutable snapshots; the refresh thread reads only UI state."""

    def __init__(self, *, enabled=True, stream=None):
        self._enabled = enabled
        self._stream = sys.stderr if stream is None else stream
        self._frame = _Frame()
        self._console = self._live = None
        self._started = self._ended = None
        self._last_stage = None
        self._last_log = 0.0
        self._phase_started = None
        self._phase_seconds = {}
        self._output_lock = RLock()
        self._pending_output = []

    def __enter__(self):
        self._started = self._phase_started = time.monotonic()
        if self._enabled and self._stream.isatty():
            from rich.console import Console
            from rich.live import Live

            with self._output_lock:
                self._console = Console(file=self._stream)
                for line in self._pending_output:
                    self._console.print(line, markup=False, highlight=False)
                self._pending_output.clear()
                self._live = Live(
                    console=self._console, get_renderable=self._render,
                    refresh_per_second=8, transient=False, vertical_overflow="crop",
                )
                self._live.start(refresh=True)
        return self

    def write_output(self, line):
        from rich.text import Text

        with self._output_lock:
            if self._console is None:
                self._pending_output.append(line)
            else:
                self._console.print(Text.from_ansi(line), highlight=False)

    def source(self, shards):
        self._publish(phase="routing", total=self._frame.total + shards)

    def file(self, filename):
        self._publish(filename=_plain(filename))

    def advance(self, selected_bytes):
        self._publish(shards=self._frame.shards + 1, selected_bytes=selected_bytes)

    def phase(self, phase):
        self._record_phase(time.monotonic())
        self._publish(phase=phase)

    def _record_phase(self, now):
        phase = self._frame.phase
        self._phase_seconds[phase] = self._phase_seconds.get(phase, 0) + now - self._phase_started
        self._phase_started = now

    def complete(self, summary):
        self._ended = time.monotonic()
        self._record_phase(self._ended)
        stages = {phase: self._phase_seconds.get(phase, 0)
                  for phase in ("prepare", "plan", "execute", "unmap")}
        summary = dict(summary, routing_seconds=max(0, summary["load_seconds"] - sum(stages.values())),
                       **{phase + "_seconds": seconds for phase, seconds in stages.items()})
        self._publish(phase="ready", summary=tuple(summary.items()))

    def _elapsed(self):
        return (self._ended if self._ended is not None else time.monotonic()) - self._started

    def _publish(self, **changes):
        self._frame = replace(self._frame, **changes)
        if not self._enabled or self._live is not None:
            return
        frame = self._frame
        stage, label = _PHASES[frame.phase]
        now = time.monotonic()
        routed = "shards" in changes and frame.shards == frame.total
        if stage != self._last_stage or routed or now - self._last_log >= 10:
            self._last_stage, self._last_log = stage, now
            summary = dict(frame.summary)
            if summary:
                detail = self._result(summary) + "; " + self._timing(summary) + "; " + self._breakdown(summary)
            else:
                detail = (f"{frame.shards}/{frame.total} shards routed, "
                          f"{frame.selected_bytes / 1e9:.2f} GB selected on rank 0, "
                          f"{_duration(self._elapsed())}")
            self._stream.write(f"b12x {label.lower()}: {detail}\n")
            self._stream.flush()

    @staticmethod
    def _result(summary):
        scope = f"across {summary['ranks']} TP ranks" if summary["ranks"] > 1 else "on rank 0"
        return (f"{summary['payload_bytes'] / 1e9:.2f} GB selected {scope}"
                f" · {summary['physical_bytes'] / 1e9:.2f} GB physical reads")

    @staticmethod
    def _transfer_rate(summary):
        seconds = summary.get("shared_transfer_seconds", 0)
        if seconds > 0 and "shared_physical_bytes" in summary:
            return summary["shared_physical_bytes"] / seconds / 1e9
        return None

    @classmethod
    def _timing(cls, summary):
        text = f"{summary['load_seconds']:.2f} s total loading"
        if "shared_transfer_seconds" in summary:
            text += f" · {summary['shared_transfer_seconds']:.2f} s shared read + scatter"
        rate = cls._transfer_rate(summary)
        if rate is not None:
            text += f" · ≈{rate:.1f} GB/s"
        return text

    @staticmethod
    def _breakdown(summary):
        return " · ".join(f"{_PHASES[phase][0]} {summary[phase + '_seconds']:.2f}s"
                          for phase in ("routing", "prepare", "plan", "execute", "unmap"))

    def _render(self):
        from rich import box
        from rich.panel import Panel
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        frame = self._frame
        stage, label = _PHASES[frame.phase]
        try:
            size = os.get_terminal_size(self._stream.fileno())
        except (AttributeError, OSError, ValueError):
            pass
        else:
            if size.columns > 0 and size.lines > 0:
                self._console.size = size
        width = self._console.width
        color = _ERROR if stage == "FAILED" else _SIGNAL
        phase_elapsed = (self._ended if self._ended is not None else time.monotonic()) - self._phase_started
        if width < 76 or self._console.height < 10:
            line = Text("b12x  ", style=f"bold {_INK}", no_wrap=True, overflow="ellipsis")
            line.append(stage, style=f"bold {color}")
            if frame.summary:
                summary = dict(frame.summary)
                line.append(f"  {summary['load_seconds']:.2f}s total", style=_DIM)
                rate = self._transfer_rate(summary)
                if rate is not None:
                    line.append(f" · ≈{rate:.1f} GB/s", style=_INK)
                return line
            detail = (f"{frame.shards}/{frame.total} shards" if frame.total else "reading metadata") if frame.phase == "routing" else label.lower()
            line.append(f"  {detail}  {_duration(phase_elapsed)}", style=_DIM)
            return line
        rail = Text(no_wrap=True)
        for index, item in enumerate(_STAGES):
            if index:
                rail.append("  /  ", style=_TRACK)
            rail.append(item, style=f"bold {_SIGNAL}" if item == stage else _DIM)
        rail.append(f"    elapsed {_duration(self._elapsed())}", style=_DIM)
        if frame.phase == "routing" and frame.total:
            detail = f"{frame.shards} / {frame.total} shards routed"
        elif frame.phase in ("ready", "failed"):
            detail = label.lower()
        else:
            detail = f"{stage.lower()} · {_duration(phase_elapsed)}"
        bar_width = max(8, width - len(detail) - 8)
        if frame.phase == "routing" and frame.total:
            bar = _bar(frame.shards / frame.total, bar_width)
        elif frame.phase in ("ready", "failed"):
            bar = _bar(int(frame.phase == "ready"), bar_width)
        else:
            bar = ProgressBar(total=None, pulse=True, width=bar_width,
                              style=_TRACK, pulse_style=_SIGNAL,
                              animation_time=phase_elapsed)
        activity = Table.grid(padding=(0, 1))
        activity.add_row(bar, Text(detail, style=color, no_wrap=True))
        grid = Table.grid(expand=True)
        grid.add_column(no_wrap=True, overflow="ellipsis")
        grid.add_row(rail)
        grid.add_row(activity)
        grid.add_row(Text("─" * (width - 6), style=_TRACK))
        status = Text("STATUS    ", style=_DIM)
        status.append(label, style=color)
        grid.add_row(status)
        summary = dict(frame.summary)
        if summary:
            grid.add_row(Text(self._result(summary), style=_INK))
            grid.add_row(Text(self._timing(summary), style=_DIM))
            grid.add_row(Text(self._breakdown(summary), style=_DIM))
        else:
            source = frame.filename or "Reading checkpoint metadata"
            if frame.phase != "routing":
                source = f"{frame.total} checkpoint shards"
            grid.add_row(Text(f"SOURCE    {source}", style=_DIM))
            grid.add_row(Text(f"SELECTED  {frame.selected_bytes / 1e9:.2f} GB on rank 0", style=_INK))
        return Panel(grid, box=box.SQUARE, border_style=_TRACK, padding=(0, 2),
                     title=Text(" b12x / weight loading ", style=f"bold {_INK}"), title_align="left")

    def stop(self):
        if self._live is not None:
            self._live.stop()

    def __exit__(self, kind, *_):
        if self._ended is None:
            self._ended = time.monotonic()
        try:
            if kind is not None:
                self.phase("failed")
        finally:
            self.stop()
