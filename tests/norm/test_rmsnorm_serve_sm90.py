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
"""

import pytest
import torch

import flashinfer
from flashinfer.norm import RMSNORM_BACKENDS
from flashinfer.norm.kernels.rmsnorm_serve_sm90 import (
    _dispatch_table,
    _reset_rmsnorm_serve_sm90_stats,
    _rmsnorm_serve_sm90_stats,
)

DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
VARIANT_ENV = "FLASHINFER_SM90_SERVE_RMSNORM"  # "0" = deprecated kill alias


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(VARIANT_ENV, raising=False)
    _reset_rmsnorm_serve_sm90_stats()
    yield
    _reset_rmsnorm_serve_sm90_stats()


def _sm90_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")


def _ref(x, w, eps=1e-6):
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rms * w.float()).to(x.dtype)


# --------------------------------------------------------------------------
# Backend surface (#5667 pattern, GPU-free)
# --------------------------------------------------------------------------


def test_backends_tuple_is_plain_strings_with_specialized():
    assert RMSNORM_BACKENDS == ("auto", "specialized", "cuda", "cutedsl")
    assert all(isinstance(b, str) for b in RMSNORM_BACKENDS)


def test_dispatch_table_loads_and_matches_guard_constants():
    table = _dispatch_table()
    assert table is not None
    assert table["kill_switch"] == DISABLE_ENV
    assert "specialized" in table["backends"]
    op = table["ops"]["rmsnorm_serve"]
    assert op["envelope"]["max_tokens"] == 8192
    assert op["envelope"]["hidden_exact_allowlist"] == [512, 2048, 6144]
    band = op["envelope"]["hidden_generic_band"]
    assert (band["min"], band["max"], band["multiple_of"]) == (512, 6144, 8)


def test_unknown_backend_raises():
    x = torch.zeros((4, 512), dtype=torch.bfloat16)
    w = torch.zeros((512,), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="never silently swapped"):
        flashinfer.norm.rmsnorm(x, w, backend="fa2")


def test_stats_structure():
    stats = _rmsnorm_serve_sm90_stats()
    assert set(stats["backend_dispatches"]) == {"specialized", "cuda", "cutedsl"}
    assert set(stats["variant_dispatches"]) == {"cutedsl", "cuda"}
    assert stats["backends"] == RMSNORM_BACKENDS
    assert "fallback_reasons" in stats and "dispatch_table_loaded" in stats


# --------------------------------------------------------------------------
# Device tests (SM90)
# --------------------------------------------------------------------------


def _case(tokens=64, hidden=2048, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((tokens, hidden), generator=g, device=device) * 0.5).to(
        torch.bfloat16
    )
    w = (torch.randn((hidden,), generator=g, device=device) * 0.5).to(torch.bfloat16)
    return x, w


def _matched(y, ref, rtol=0.02, atol=0.01):
    yf, rf = y.float(), ref.float()
    return float(((yf - rf).abs() <= (atol + rtol * rf.abs())).float().mean().item())


# Shape grid: the cutedsl plan's launch regimes (H=512 single-warp, H=2048,
# H=6144 12-warp, H=520 non-allowlist generic-band) at small and large token
# counts, including both sides of the cuda variant's 6144 split boundary.
@pytest.mark.parametrize(
    "tokens,hidden",
    [(64, 2048), (2, 2048), (16, 512), (8, 520), (64, 6144), (65, 6144)],
)
def test_specialized_dispatches_and_matches_stock(tokens, hidden):
    _sm90_or_skip()
    x, w = _case(tokens=tokens, hidden=hidden)
    y_spec = flashinfer.norm.rmsnorm(x, w, backend="specialized")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert stats["variant_dispatches"]["cutedsl"] == 1, stats
    y_stock = flashinfer.norm.rmsnorm(x, w, backend="cutedsl")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert stats["backend_dispatches"]["cutedsl"] == 1, stats
    ref = _ref(x, w)
    assert _matched(y_spec, ref) >= 0.999
    assert _matched(y_stock, ref) >= 0.999
    # determinism
    y2 = flashinfer.norm.rmsnorm(x, w, backend="specialized")
    assert torch.equal(y_spec, y2)


def test_explicit_incumbent_backends_are_never_swapped():
    """backend="cuda"/"cutedsl" on an allowlisted shape must never reach the
    specialized kernels, and each pins its own implementation."""
    _sm90_or_skip()
    x, w = _case(tokens=16, hidden=512)
    for incumbent in ("cuda", "cutedsl", "cuda"):
        flashinfer.norm.rmsnorm(x, w, backend=incumbent)
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["variant_dispatches"]["cutedsl"] == 0, stats
    assert stats["variant_dispatches"]["cuda"] == 0, stats
    assert stats["backend_dispatches"]["cuda"] == 2, stats
    assert stats["backend_dispatches"]["cutedsl"] == 1, stats


def test_auto_routes_on_allowlist_only():
    _sm90_or_skip()
    x, w = _case(tokens=32, hidden=2048)
    y = flashinfer.norm.rmsnorm(x, w)  # auto default
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert _matched(y, _ref(x, w)) >= 0.999
    # off-allowlist H (not a multiple of 8): auto -> stock path
    _reset_rmsnorm_serve_sm90_stats()
    x2, w2 = _case(tokens=8, hidden=516)
    y2 = flashinfer.norm.rmsnorm(x2, w2)
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert any("off the allowlist" in r for r in stats["fallback_reasons"]), stats
    assert _matched(y2, _ref(x2, w2)) >= 0.999


def test_explicit_specialized_off_envelope_raises():
    _sm90_or_skip()
    x, w = _case(tokens=4, hidden=516)  # H off both variants' envelopes
    with pytest.raises(NotImplementedError, match="never falls back"):
        flashinfer.norm.rmsnorm(x, w, backend="specialized")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert any("off the allowlist" in r for r in stats["fallback_reasons"]), stats
    # the same call under auto is served by the incumbent path
    assert _matched(flashinfer.norm.rmsnorm(x, w), _ref(x, w)) >= 0.999


@pytest.mark.parametrize("env,value", [(DISABLE_ENV, "1"), (VARIANT_ENV, "0")])
def test_kill_switch_and_deprecated_alias_force_incumbent(env, value, monkeypatch):
    _sm90_or_skip()
    x, w = _case()
    monkeypatch.setenv(env, value)
    y = flashinfer.norm.rmsnorm(x, w, backend="auto")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert any("kill switch" in r for r in stats["fallback_reasons"]), stats
    assert _matched(y, _ref(x, w)) >= 0.999
    # explicit "specialized" under the kill switch raises (never falls back)
    with pytest.raises(NotImplementedError, match="never falls back"):
        flashinfer.norm.rmsnorm(x, w, backend="specialized")


# Shape grid: every launch regime of the cuda specialist — H=512
# (launch_exact<32,2>), H=2048 2-token (osplit2) and >2-token (launch_exact),
# H=6144 on both sides of the 64-token split boundary (osplit8 /
# launch_exact<384,2>).
@pytest.mark.parametrize(
    "tokens,hidden",
    [(32, 2048), (2, 2048), (16, 512), (64, 6144), (65, 6144)],
)
def test_cuda_variant_behind_the_one_specialized_backend(tokens, hidden, monkeypatch):
    _sm90_or_skip()
    monkeypatch.setenv(VARIANT_ENV, "cuda")
    x, w = _case(tokens=tokens, hidden=hidden)  # on the cuda exact allowlist
    y = flashinfer.norm.rmsnorm(x, w, backend="specialized")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert stats["variant_dispatches"]["cuda"] == 1, stats
    assert _matched(y, _ref(x, w)) >= 0.999


def test_cuda_variant_off_exact_list_raises_explicit_serves_auto(monkeypatch):
    # off the cuda exact list, explicit "specialized" raises (never a silent
    # variant swap or fallback); auto routes the call to the stock path
    _sm90_or_skip()
    monkeypatch.setenv(VARIANT_ENV, "cuda")
    x, w = _case(tokens=8, hidden=1024)
    with pytest.raises(NotImplementedError, match="never falls back"):
        flashinfer.norm.rmsnorm(x, w, backend="specialized")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["variant_dispatches"]["cuda"] == 0, stats
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert _matched(flashinfer.norm.rmsnorm(x, w), _ref(x, w)) >= 0.999


@pytest.mark.parametrize("tokens,hidden", [(2, 2048), (64, 6144)])
def test_cuda_variant_in_place_declined(tokens, hidden, monkeypatch):
    """out aliasing input must not reach the cuda variant's split kernels
    (several CTAs per row: a write could land before another CTA's read):
    explicit "specialized" raises, auto serves via the incumbent path."""
    _sm90_or_skip()
    monkeypatch.setenv(VARIANT_ENV, "cuda")
    x, w = _case(tokens=tokens, hidden=hidden)
    ref = _ref(x, w)
    with pytest.raises(NotImplementedError, match="never falls back"):
        flashinfer.norm.rmsnorm(x, w, out=x, backend="specialized")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["variant_dispatches"]["cuda"] == 0, stats
    assert any("in-place" in r for r in stats["fallback_reasons"]), stats
    y = flashinfer.norm.rmsnorm(x, w, out=x, backend="auto")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["variant_dispatches"]["cuda"] == 0, stats
    assert y.data_ptr() == x.data_ptr()
    assert _matched(y, ref) >= 0.999


def test_cutedsl_in_place_is_supported():
    """The cutedsl variant is one CTA per row with register staging, so exact
    out-is-input aliasing stays on the specialized path and is correct."""
    _sm90_or_skip()
    x, w = _case(tokens=16, hidden=2048)
    ref = _ref(x, w)
    y = flashinfer.norm.rmsnorm(x, w, out=x, backend="specialized")
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["variant_dispatches"]["cutedsl"] == 1, stats
    assert y.data_ptr() == x.data_ptr()
    assert _matched(y, ref) >= 0.999


def test_stats_census_sums_to_calls(monkeypatch):
    """Every call is attributed to exactly one backend, fallbacks included:
    calls == specialized + cuda + cutedsl."""
    _sm90_or_skip()
    x, w = _case(tokens=16, hidden=2048)
    flashinfer.norm.rmsnorm(x, w, backend="specialized")  # specialized
    flashinfer.norm.rmsnorm(x, w, backend="cutedsl")  # explicit incumbent
    x2, w2 = _case(tokens=8, hidden=516)
    flashinfer.norm.rmsnorm(x2, w2, backend="specialized")  # guard fallback
    monkeypatch.setenv(DISABLE_ENV, "1")
    flashinfer.norm.rmsnorm(x, w)  # kill-switch fallback
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["calls"] == 4, stats
    assert (
        stats["backend_dispatches"]["specialized"]
        + stats["backend_dispatches"]["cuda"]
        + stats["backend_dispatches"]["cutedsl"]
        == stats["calls"]
    ), stats


def test_use_cuda_norm_env_still_pins_legacy_cuda_kernels(monkeypatch):
    """FLASHINFER_USE_CUDA_NORM=1 keeps its pre-existing meaning under
    backend="auto": the legacy CUDA JIT norm kernels serve the call and the
    specialized dispatch is not consulted."""
    _sm90_or_skip()
    monkeypatch.setattr(flashinfer.norm, "_USE_CUDA_NORM", True)
    x, w = _case(tokens=16, hidden=2048)  # on the auto allowlist
    y = flashinfer.norm.rmsnorm(x, w)
    stats = _rmsnorm_serve_sm90_stats()
    assert stats["calls"] == 0, stats
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert _matched(y, _ref(x, w)) >= 0.999
