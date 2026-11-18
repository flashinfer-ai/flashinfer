"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---------------------------------------------------------------------------
The KDA ucache verify+flush tests, re-run on ``backend="umma"`` (the tcgen05 /
TMEM kernel in flashinfer/experimental/kda_ucache_flush/kda_replay_mtp_umma.py),
plus UMMA-only tests.

How the reuse works: the star import below re-collects every test of
test_kda_ucache_flush.py in this module. Those functions keep the base
module's globals, and every one of them reaches the kernel through
``_base._load_flush()``, which binds ``**_base._BACKEND_KW``. The autouse
fixture patches that dict to ``{"backend": "umma"}`` for every test collected
here, so the base file's oracle, cases and tolerances apply unchanged.

Run:
  pytest tests/experimental/test_kda_ucache_flush_umma.py -v
"""

from __future__ import annotations

import pytest
import torch

from . import test_kda_ucache_flush as _base
from .test_kda_ucache_flush import *  # noqa: F401,F403  re-collects all base cases
from .test_kda_ucache_flush import _load_flush, _make_case

# Base tests the UMMA backend cannot run yet at the current stage. MUST be
# empty once the kernel covers T in {4, 8} x w_ring in {16, 32} (stage S3).
# S0/S1: no kernel yet -> every base test is pending.
_PENDING_NAMES: set = {n for n in dir(_base) if n.startswith("test_")}


def _PENDING_PARAMS(params: dict) -> bool:
    """Per-parameter pending filter (e.g. ``params.get("t") == 8``)."""
    return False


@pytest.fixture(autouse=True)
def _umma_backend(request, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("KDA UMMA backend needs SM10x (tcgen05)")
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    if request.node.originalname in _PENDING_NAMES or _PENDING_PARAMS(params):
        pytest.skip("not yet supported by the UMMA backend at this stage")
    monkeypatch.setattr(_base, "_BACKEND_KW", {"backend": "umma"})


# ---------------------------------------------------------------------------
# UMMA-only tests
# ---------------------------------------------------------------------------
def _call(case, backend, t, w_ring=16, pdl_trigger=False):
    from flashinfer.kda_decode import kda_delta_rule_mtp_ucache_flush as f

    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    flush_min = min(w_ring - t + 1, 33 - 2 * t)
    return f(
        A_log,
        g,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=hl,
        cache_base=cb,
        flush_min=flush_min,
        restart_hist_on_flush=False,
        w_ring=w_ring,
        pdl_trigger=pdl_trigger,
        backend=backend,
    )


def test_umma_backend_dispatch():
    """backend validation, 'umma' availability errors, and which kernel
    'auto' / 'hmma' / 'umma' actually launch (via the UMMA launch counter)."""
    from flashinfer.experimental.kda_ucache_flush import (
        kda_decode_bf16_wy_ucache_flush as disp,
    )
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    def launches() -> int:
        return um._LAUNCHES

    dev = torch.device("cuda")
    mod = _load_flush()  # pinned to backend="umma" by the autouse fixture
    assert mod.kda_delta_rule_mtp_ucache_flush.keywords == {"backend": "umma"}

    # Unknown backend: ValueError naming the accepted values.
    case = _make_case(4, [0, 3, 12, 13], seed=7)
    with pytest.raises(ValueError, match="backend must be 'auto', 'hmma' or 'umma'"):
        _call(case, "bogus", t=4)

    # pdl_trigger is not supported by the UMMA kernel yet (until stage S4).
    assert disp.kda_umma_unavailable_reason(dev, 4, 16, True) is not None
    case = _make_case(4, [0, 3, 12, 13], seed=7)
    with pytest.raises(ValueError, match="backend='umma' unavailable"):
        _call(case, "umma", t=4, pdl_trigger=True)

    for t in (4, 8):
        for w_ring in (16, 32):
            fm = min(w_ring - t + 1, 33 - 2 * t)
            hist = [0, fm - 1, fm, 1]
            why = disp.kda_umma_unavailable_reason(dev, t, w_ring, False)

            # Explicit 'umma': raises with the reason, or launches UMMA.
            n0 = launches()
            case = _make_case(4, hist, seed=11, t=t)
            if why is not None:
                with pytest.raises(ValueError, match="backend='umma' unavailable"):
                    _call(case, "umma", t=t, w_ring=w_ring)
                assert launches() == n0
            else:
                _call(case, "umma", t=t, w_ring=w_ring)
                assert launches() == n0 + 1

            # 'hmma' never launches UMMA.
            n0 = launches()
            _call(_make_case(4, hist, seed=11, t=t), "hmma", t=t, w_ring=w_ring)
            assert launches() == n0

            # 'auto' launches UMMA iff available and the auto policy is on.
            n0 = launches()
            _call(_make_case(4, hist, seed=11, t=t), "auto", t=t, w_ring=w_ring)
            expect = int(why is None and disp._AUTO_UMMA)
            assert launches() == n0 + expect


def test_umma_backend_validation_precedes_dispatch():
    """Contract violations raise the shared validation error on every
    backend value, before the backend split: 'umma' rejects bad inputs the
    same way 'hmma' does, whatever its availability."""
    for backend in ("hmma", "umma", "auto", "bogus"):
        case = list(_make_case(4, [0, 3, 12, 13], seed=3))
        case[11] = torch.tensor([0, 3, 12, 17], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="hist_len out of legal range"):
            _call(tuple(case), backend, t=4)
