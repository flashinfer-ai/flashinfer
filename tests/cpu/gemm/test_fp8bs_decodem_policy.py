# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy
from importlib import resources

import pytest

from flashinfer.gemm import fp8bs_decodem_sm90 as mod


OP = "fp8_blockscale_gemm_sm90"
PAIRS = {
    (8192, 1536),
    (2048, 7168),
    (7168, 2048),
    (2048, 2048),
    (2624, 6144),
    (6144, 2048),
}


def test_packaged_domain():
    op = mod._dispatch_table()["ops"][OP]
    assert set(map(tuple, op["measured_nk_classes"])) == PAIRS
    assert op["alignment"] == {"n_multiple_of": 64, "k_multiple_of": 128}
    for n, k in PAIRS:
        for m in range(1, 25):
            assert mod._auto_refusal(m, n, k) is None
        for m in (0, 25):
            assert mod._auto_refusal(m, n, k) == "m_band"
    for n, k in ((64, 16384), (8192, 7168), (16384, 128)):
        assert mod._auto_refusal(13, n, k) == "unmeasured_nk"
    assert mod._auto_refusal(13, 2625, 6144) == "shape_alignment"


@pytest.fixture
def manifest(monkeypatch, tmp_path):
    table = deepcopy(mod._dispatch_table())
    path = tmp_path / mod._DISPATCH_FILE
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    monkeypatch.setattr(mod, "_dispatch_table", mod._dispatch_table.__wrapped__)
    return table, path


@pytest.mark.parametrize(
    "field,value",
    [
        ("m_band", [24, 1]),
        ("m_band", [True, 24]),
        ("m_band", [1, 25]),
        ("alignment", {"n_multiple_of": 0, "k_multiple_of": 128}),
        ("alignment", {"n_multiple_of": 64, "k_multiple_of": 64}),
        ("measured_nk_classes", []),
        ("measured_nk_classes", [[64]]),
        ("measured_nk_classes", [[64, 128.0]]),
        ("measured_nk_classes", [[65, 128]]),
        ("measured_nk_classes", [[64, 128], [64, 128]]),
    ],
)
def test_invalid_manifest_fails_closed(manifest, field, value):
    table, path = manifest
    assert mod._auto_refusal(13, 2048, 2048) == "no_dispatch_table"
    table["ops"][OP][field] = value
    path.write_text(json.dumps(table))
    assert mod._auto_refusal(13, 2048, 2048) == "no_dispatch_table"


def test_valid_narrower_policy_is_read(manifest):
    table, path = manifest
    table["ops"][OP].update(
        m_band=[5, 8],
        alignment={"n_multiple_of": 128, "k_multiple_of": 256},
        measured_nk_classes=[[2048, 2048]],
    )
    path.write_text(json.dumps(table))
    assert mod._auto_refusal(4, 2048, 2048) == "m_band"
    assert mod._auto_refusal(6, 2048, 2048) is None
    assert mod._auto_refusal(6, 2624, 6144) == "shape_alignment"
    assert mod._auto_refusal(6, 8192, 1536) == "unmeasured_nk"
