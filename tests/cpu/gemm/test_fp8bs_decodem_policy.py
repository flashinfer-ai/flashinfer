# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import nullcontext
from copy import deepcopy
from importlib import resources
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_native_stream_decision(monkeypatch, backend):
    x = SimpleNamespace(shape=(12, 6144), device=0)
    w = SimpleNamespace(shape=(2624, 6144))
    ws, out = object(), object()
    native = Mock(spec_set=["try_run"])
    native.try_run.side_effect = [True, False, RuntimeError("occupancy failed"), True]
    monkeypatch.setattr(mod, "_get_module", lambda: native)
    monkeypatch.setattr(mod, "_eligibility", lambda *args: (True, "ok"))
    monkeypatch.setattr(mod.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        mod,
        "_stats",
        {
            "jit_build": "built",
            "fallback_reasons": {},
            "backend_dispatches": {"specialized": 0, "deepgemm": 0},
        },
    )

    def call():
        return mod._fp8bs_decodem_try_dispatch(x, w, None, ws, out, backend)

    assert call() is out
    if backend == "auto":
        assert call() is None
    else:
        with pytest.raises(NotImplementedError, match="stream_cluster_capacity"):
            call()
    with pytest.raises(RuntimeError, match="occupancy failed"):
        call()
    assert call() is out
    assert native.try_run.call_count == 4
    native.try_run.assert_called_with(x, w, ws, mod._QUANT_EPS, out)
    assert mod._stats["backend_dispatches"] == {"specialized": 2, "deepgemm": 0}
    assert mod._stats["fallback_reasons"] == {f"{backend}: stream_cluster_capacity": 1}
    assert mod._stats["jit_build"] == "built"


@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_previous_prebuilt_module_falls_back_without_launch(monkeypatch, backend):
    x = SimpleNamespace(shape=(4, 2048), device=0)
    w = SimpleNamespace(shape=(2048, 2048))
    ws, out = object(), object()
    previous = Mock(spec_set=["run"])
    current = Mock(spec_set=["try_run"])
    current.try_run.side_effect = [True, AttributeError("native execution error")]
    monkeypatch.setattr(
        mod, "_get_module", Mock(side_effect=[previous, current, current])
    )
    monkeypatch.setattr(mod, "_eligibility", lambda *args: (True, "ok"))
    monkeypatch.setattr(mod.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        mod,
        "_stats",
        {
            "jit_build": "built",
            "fallback_reasons": {},
            "backend_dispatches": {"specialized": 0, "deepgemm": 0},
        },
    )

    def call():
        return mod._fp8bs_decodem_try_dispatch(x, w, None, ws, out, backend)

    if backend == "auto":
        assert call() is None
    else:
        with pytest.raises(NotImplementedError, match="older prebuilt module.*rebuild"):
            call()
    previous.run.assert_not_called()
    assert mod._stats["jit_build"] == "built"
    assert mod._stats["fallback_reasons"] == {
        f"{backend}: incompatible_prebuilt_module": 1
    }
    # A replacement provider works, and errors from its actual invocation
    # propagate rather than being mistaken for a missing export.
    assert call() is out
    with pytest.raises(AttributeError, match="native execution error"):
        call()
    assert mod._stats["backend_dispatches"] == {"specialized": 1, "deepgemm": 0}
