from __future__ import annotations

import pytest

from b12x.comm.pcie._launch_geometry import (
    DIAGNOSTIC_GEOMETRY_ENV,
    require_diagnostic_geometry,
)


def test_reviewed_launch_geometry_needs_no_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(DIAGNOSTIC_GEOMETRY_ENV, raising=False)

    require_diagnostic_geometry(
        "oneshot",
        threads=(256, 256),
        block_limit=(8, 8),
    )


def test_nondefault_launch_geometry_is_diagnostic_only(monkeypatch) -> None:
    monkeypatch.delenv(DIAGNOSTIC_GEOMETRY_ENV, raising=False)

    with pytest.raises(ValueError, match="non-default launch geometry.*diagnostic-only"):
        require_diagnostic_geometry(
            "twoshot",
            threads=(256, 512),
            block_limit=(64, 64),
        )


@pytest.mark.parametrize("enabled", ["1", "true", "YES", "on"])
def test_shared_opt_in_allows_diagnostic_geometry(monkeypatch, enabled: str) -> None:
    monkeypatch.setenv(DIAGNOSTIC_GEOMETRY_ENV, enabled)

    require_diagnostic_geometry(
        "dcp_topk",
        threads=(128, 512),
        block_limit=(32, 128),
    )
