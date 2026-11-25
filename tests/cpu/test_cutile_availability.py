# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Optional cuTile installation must not block other MoE backends."""

import pytest


@pytest.mark.parametrize("missing_parent", ["cuda", "cuda.tile"])
def test_missing_cutile_parent_declines(monkeypatch, missing_parent):
    from flashinfer.cutile import cutile_common

    def absent(name):
        assert name == "cuda.tile.tune"
        raise ModuleNotFoundError(name=missing_parent)

    monkeypatch.setattr(cutile_common.importlib.util, "find_spec", absent)
    monkeypatch.setattr(
        cutile_common,
        "_find_tileiras_binary",
        lambda: pytest.fail("compiler probe ran without the optional Python package"),
    )
    assert cutile_common.is_cuda_tile_available() is False
