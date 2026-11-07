"""Settling near-tied tactics with autotune(tie_tolerance=...) (issue #7884)."""

import pytest
import torch

from flashinfer import autotune
from flashinfer.autotuner import AutoTuner, TuningConfig

from .utils import DummyRunner, reset_autotuner

INPUTS = [torch.empty((16, 32), dtype=torch.float32)]


def _choose(monkeypatch, times, **autotune_kwargs):
    """Tune with each tactic's measurements in call order (the last repeats;
    an exception is raised by its call). Returns (tactic, profiled tactics)."""
    remaining = {tactic: list(seq) for tactic, seq in times.items()}
    calls = []

    def profile(self, runner, inputs, tactic, *args, **kwargs):
        calls.append(tactic)
        seq = remaining[tactic]
        value = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", profile)
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=tuple(times))
    with autotune(True, **autotune_kwargs):
        _, tactic = tuner.choose_one("near_tie_op", [runner], TuningConfig(), INPUTS)
    return tactic, calls


@pytest.mark.parametrize(
    "times, tolerance, kept, retimed",
    [
        # Without a tolerance the fastest stands.
        ({0: [1.003], 1: [1.0], 2: [1.001]}, 0, 1, []),
        # 0 and 2 are within 0.5% of the fastest, 1, and 3 is not: the earliest
        # is re-timed against 1, alternating over two rounds, and kept.
        ({0: [1.004], 1: [1.0], 2: [1.002], 3: [1.5]}, 0.005, 0, [0, 1, 1, 0]),
        # Within the tolerance by chance: it re-times 2% slower.
        ({0: [1.004, 1.02], 1: [1.0]}, 0.005, 1, [0, 1, 1, 0]),
        # The fastest comes first, or wins clearly: nothing is re-timed.
        ({0: [1.0], 1: [1.002], 2: [1.1]}, 0.005, 0, []),
        ({0: [1.2], 1: [1.0], 2: [1.1]}, 0.005, 1, []),
        # A contender whose re-timing fails, out of memory included, is not kept.
        ({0: [1.001, RuntimeError("lost")], 1: [1.0]}, 0.005, 1, [0, 1, 1, 0]),
        (
            {0: [1.001, torch.cuda.OutOfMemoryError("re-timing")], 1: [1.0]},
            0.005,
            1,
            [0, 1, 1, 0],
        ),
    ],
)
def test_near_tie_settling(monkeypatch, times, tolerance, kept, retimed):
    tactic, calls = _choose(monkeypatch, times, tie_tolerance=tolerance)
    assert tactic == kept
    assert calls == [*times, *retimed]


def test_at_most_four_contenders_are_re_timed(monkeypatch):
    count = AutoTuner._TIE_MAX_CONTENDERS + 2
    fastest = count - 1
    # Every tactic before the fastest, the last one, measures within the
    # tolerance once and re-times outside it.
    times = {t: [1.003, 1.02] for t in range(fastest)}
    times[fastest] = [1.0]

    tactic, calls = _choose(monkeypatch, times, tie_tolerance=0.005)
    assert tactic == fastest
    expected = []
    for t in range(AutoTuner._TIE_MAX_CONTENDERS):
        expected += [t, fastest, fastest, t]
    assert calls[count:] == expected


def test_tie_tolerance_is_inherited_and_validated():
    tuner = reset_autotuner()
    with autotune(True, tie_tolerance=0.01):
        with autotune(True, cuda_graph_profile_replays=2):
            assert tuner._tie_tolerance == 0.01
        with autotune(True, tie_tolerance=0):
            assert tuner._tie_tolerance == 0
    assert tuner._tie_tolerance == 0
    for bad in (-0.01, 1.0, float("nan")):
        with pytest.raises(ValueError), autotune(True, tie_tolerance=bad):
            pass
