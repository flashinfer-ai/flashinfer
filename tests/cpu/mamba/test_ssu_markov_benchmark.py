# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def markov(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmarks"))
    return importlib.import_module("ssu_markov")


@pytest.mark.parametrize("draft_length", [0, 1, 3, 7])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.8109348502411851, 0.99, 1.0])
def test_acceptance_fit_roundtrip(markov, draft_length, p):
    mean = sum(p**k for k in range(draft_length + 1))
    fitted, histogram = markov.fit_acceptance(draft_length, mean)
    assert sum(histogram) == pytest.approx(1.0)
    assert sum(a * mass for a, mass in enumerate(histogram, 1)) == pytest.approx(mean)
    if draft_length:
        assert fitted == pytest.approx(p, abs=1e-12)
        assert histogram[-1] == pytest.approx(p**draft_length)


@pytest.mark.parametrize(
    "draft,mean", [(-1, 1), (0, 2), (7, 0.9), (7, 8.1), (7, float("nan"))]
)
def test_acceptance_fit_invalid(markov, draft, mean):
    with pytest.raises(ValueError):
        markov.fit_acceptance(draft, mean)


def test_acceptance_csv_counts(markov, tmp_path):
    csv = tmp_path / "histogram.csv"
    csv.write_text("accepted_tokens,probability\n1,1\n2,2\n3,1\n")
    assert markov.load_acceptance_csv(csv) == (0.25, 0.5, 0.25)


def test_stationary_metadata_law(markov):
    _, acceptance = markov.fit_acceptance(7, 4.3)
    law = markov.stationary_workload(acceptance)
    assert sum(mass for _, _, mass in law.joint) == pytest.approx(1.0)
    assert law.balance_residual < 1e-12
    assert law.flush_share == pytest.approx(0.37043333661416344)
    assert law.draws(1, 64, 7601) == law.draws(1, 64, 7601)
    assert len({row[0].history for row in law.draws(1, 64, 7601)}) > 1


def test_iqm_uses_whole_pass_means(markov):
    pairs = [
        dict(round=i, us_saved=d, fractional_improvement=d / 200)
        for i, d in enumerate([1, 2, 3, 100])
    ]
    result = markov.paired_pass_iqm(pairs)
    assert result["iqm_us_saved"] == 2.5
    assert result["paired_pass_count"] == 4


def test_course_means_are_averaged_before_trimming(markov):
    benchmark = importlib.import_module("bench_checkpointing_ssu_markov")
    courses = [dict(A0=a, A1=a, B0=10.0, B1=10.0) for a in (10.0, 10.0, 10.0, 110.0)]
    result = benchmark.summarize([courses] * 4)
    assert result["effects"]["AB"]["iqm_us_saved"] == 25.0
