from dataclasses import dataclass

import pytest

from tests.test_helpers.fuzz_ledger import Finding, FuzzLedger


@dataclass(frozen=True)
class Config:
    value: int


def _matches_one(cfg):
    return cfg.value == 1


def test_fuzz_ledger_requires_tracking_issue():
    with pytest.raises(ValueError, match="tracking issue"):
        FuzzLedger("test", (Finding(_matches_one, "missing issue"),))


def test_fuzz_ledger_quarantine_precedes_tolerated_overlap():
    tolerated = Finding(_matches_one, "wrong answer #100")
    quarantined = Finding(
        _matches_one,
        "illegal access #101",
        quarantine=True,
        backend="broken",
    )
    ledger = FuzzLedger("test", (tolerated, quarantined))
    cfg = Config(1)

    assert ledger.find(cfg) is tolerated
    assert ledger.find(cfg, backend="healthy") is tolerated
    assert ledger.find(cfg, backend="broken") is quarantined


def test_fuzz_ledger_backend_scope_is_exact():
    scoped = Finding(_matches_one, "backend bug #102", backend="broken")
    ledger = FuzzLedger("test", (scoped,))
    cfg = Config(1)

    assert ledger.find(cfg) is None
    assert ledger.find(cfg, backend="healthy") is None
    assert ledger.find(cfg, backend="broken") is scoped


def test_fuzz_ledger_global_quarantine_xfails_before_launch():
    ledger = FuzzLedger(
        "test",
        (Finding(_matches_one, "crash #103", quarantine=True),),
    )

    with pytest.raises(pytest.xfail.Exception, match="quarantined"):
        ledger.xfail_if_quarantined(Config(1))


def test_fuzz_ledger_reports_accumulated_expected_failures_as_xfail():
    finding = Finding(_matches_one, "known mismatch #104", backend="broken")
    ledger = FuzzLedger("test", (finding,))

    ledger.report_expected_failures([], context="nothing failed")
    with pytest.raises(pytest.xfail.Exception, match="broken.*#104"):
        ledger.report_expected_failures(
            [(finding, "broken backend")],
            context="completed healthy backends",
        )


def test_fuzz_ledger_unexpected_pass_is_strict_failure():
    finding = Finding(_matches_one, "known mismatch #105")
    ledger = FuzzLedger("test", (finding,))

    with pytest.raises(pytest.fail.Exception, match="unexpectedly PASSED"):
        ledger.flag_xpass(finding, "backend")
