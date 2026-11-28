"""Host tests exercise the built-in selector, policies and offline tuning on
their own; the shipped anchor table is covered by test_tuned_table.py."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_tuned_table(request, monkeypatch):
    if request.module.__name__.endswith("test_tuned_table"):
        return
    from moe_sm120_nvfp4_split import tuned_table

    monkeypatch.setattr(tuned_table, "_load", lambda: dict(schema=1, tables=[]))
