"""Weight-loading presentation must distinguish routed shards from completed I/O."""

import io

import pytest
from rich.console import Console

from b12x.loader._progress import CheckpointDisplay


@pytest.mark.parametrize("width,height", [(120, 30), (100, 20), (80, 10), (60, 8), (40, 8)])
def test_loading_phases_fit_and_do_not_complete_at_the_end_of_routing(width, height):
    with CheckpointDisplay(stream=io.StringIO()) as display:
        console = Console(file=io.StringIO(), width=width, height=height, record=True)
        display._console = console
        display.source(2)
        display.file("weights-with-a-very-long-name-00001-of-00002.safetensors")
        display.advance(7_000_000_000)
        display.advance(15_000_000_000)
        for phase in ("routing", "prepare", "plan", "execute", "unmap"):
            display.phase(phase)
            console.print(display._render())
            text = console.export_text(clear=True)
            assert max(map(len, text.splitlines())) <= width
            assert "Weights loaded" not in text
            if phase != "routing":
                assert "shards routed" not in text
        display.complete(dict(ranks=4, payload_bytes=59_000_000_000,
                              physical_bytes=30_000_000_000, load_seconds=19,
                              shared_physical_bytes=29_000_000_000,
                              shared_transfer_seconds=6))
        console.print(display._render())
        text = console.export_text(clear=True)
        assert max(map(len, text.splitlines())) <= width
        if width >= 80:
            assert "b12x / weight loading" in text
            assert "Weights loaded" in text
            assert "59.00 GB selected across 4 TP ranks" in text
            assert "19.00 s total loading" in text
            assert "6.00 s shared read + scatter" in text
            assert "≈4.8 GB/s" in text


def test_redirected_output_is_plain_and_disabled_output_is_silent():
    for enabled in (False, True):
        output = io.StringIO()
        with CheckpointDisplay(enabled=enabled, stream=output) as display:
            display.source(2)
            display.advance(1)
            display.advance(2)
            display.phase("execute")
            assert "weights loaded" not in output.getvalue()
            display.complete(dict(ranks=1, payload_bytes=2, physical_bytes=4096, load_seconds=1))
        text = output.getvalue()
        assert "\x1b" not in text
        if enabled:
            assert "2/2 shards routed" in text
            assert "b12x weights loaded:" in text
            assert "GB/s" not in text
        else:
            assert not text


def test_failed_loading_never_reports_success():
    output = io.StringIO()
    with pytest.raises(ValueError, match="read failed"):
        with CheckpointDisplay(stream=output) as display:
            display.source(1)
            display.advance(8)
            raise ValueError("read failed")
    assert "weight loading failed" in output.getvalue()
    assert "weights loaded" not in output.getvalue()


def test_phase_times_account_for_total_including_collective_waits(monkeypatch):
    now = 0.0
    monkeypatch.setattr("b12x.loader._progress.time.monotonic", lambda: now)
    with CheckpointDisplay(stream=io.StringIO()) as display:
        now = 7.0
        display.phase("prepare")
        now = 11.0
        display.phase("plan")
        now = 13.0
        display.phase("execute")
        now = 19.0
        display.phase("unmap")
        now = 19.5
        display.complete(dict(ranks=4, payload_bytes=1, physical_bytes=1,
                              load_seconds=19, shared_transfer_seconds=5.7))
    summary = dict(display._frame.summary)
    expected = dict(routing_seconds=6.5, prepare_seconds=4, plan_seconds=2,
                    execute_seconds=6, unmap_seconds=0.5)
    assert {key: summary[key] for key in expected} == expected
    assert sum(expected.values()) == summary["load_seconds"]
