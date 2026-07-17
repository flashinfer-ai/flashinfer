"""Knob cache: offline-tuned winners resolved via pure lookup.

CPU tests cover the cache file semantics (round trip, bucket selection,
fallback, corruption tolerance, disable switch); the GPU test verifies
``get_symm_buffer_for_mega_moe(knobs=None)`` actually picks a cached entry up
into the compiled config. The hot path must be a pure lookup: knobs="auto"
is a collective multi-minute compile+timing sweep, unusable in a serving
engine.
"""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")


_KEY = dict(
    dtype="nvfp4",
    world_size=4,
    hidden=7168,
    intermediate=4096,
    num_experts=256,
    topk=8,
    combine_dtype="bf16",
)
_KNOBS = {
    "mma_tiler_mnk": (256, 128, 256),
    "cluster_shape_mnk": (2, 1, 1),
    "flag_batch": 16,
    "token_back_mode": "reuse_dispatch_warps",
}


def _cache_env(monkeypatch, tmp_path):
    path = tmp_path / "knobs.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    return path


def test_record_lookup_roundtrip_restores_tuples(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    path = _cache_env(monkeypatch, tmp_path)
    written = record_knobs(
        _KNOBS, max_tokens=2048, device="testgpu", p50_us=585.0, **_KEY
    )
    assert written == str(path)
    got = lookup_knobs(max_tokens=2048, device="testgpu", **_KEY)
    assert got == _KNOBS
    assert isinstance(got["mma_tiler_mnk"], tuple)  # JSON lists -> tuples


def test_lookup_bucket_selection(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    _cache_env(monkeypatch, tmp_path)
    small = {**_KNOBS, "flag_batch": 4}
    large = {**_KNOBS, "flag_batch": 8}
    record_knobs(small, max_tokens=512, device="testgpu", **_KEY)
    record_knobs(large, max_tokens=2048, device="testgpu", **_KEY)

    # Below both buckets -> smallest bucket at or above the request.
    assert lookup_knobs(max_tokens=100, device="testgpu", **_KEY) == small
    # Between buckets -> next bucket up.
    assert lookup_knobs(max_tokens=1024, device="testgpu", **_KEY) == large
    # Above every bucket -> largest recorded.
    assert lookup_knobs(max_tokens=8192, device="testgpu", **_KEY) == large


def test_lookup_misses_on_any_key_mismatch(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    _cache_env(monkeypatch, tmp_path)
    record_knobs(_KNOBS, max_tokens=2048, device="testgpu", **_KEY)
    for field, wrong in (
        ("hidden", 4096),
        ("topk", 6),
        ("world_size", 8),
        ("dtype", "mxfp8_e4m3"),
        ("combine_dtype", "nvfp4"),
    ):
        key = {**_KEY, field: wrong}
        assert lookup_knobs(max_tokens=2048, device="testgpu", **key) is None
    assert lookup_knobs(max_tokens=2048, device="othergpu", **_KEY) is None


def test_record_upserts_same_key(monkeypatch, tmp_path):
    import json

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    path = _cache_env(monkeypatch, tmp_path)
    record_knobs({**_KNOBS, "flag_batch": 1}, max_tokens=2048, device="testgpu", **_KEY)
    record_knobs({**_KNOBS, "flag_batch": 2}, max_tokens=2048, device="testgpu", **_KEY)
    got = lookup_knobs(max_tokens=2048, device="testgpu", **_KEY)
    assert got is not None and got["flag_batch"] == 2
    data = json.loads(path.read_text())
    assert len(data["entries"]) == 1


def test_resolve_falls_back_to_heuristic(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import resolve_knobs
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.tuner import default_knobs

    _cache_env(monkeypatch, tmp_path)  # empty cache
    knobs, source = resolve_knobs(max_tokens=2048, **_KEY)
    assert source == "heuristic"
    assert knobs == default_knobs(2048)
    # mxfp8 kinds route to the mxfp8 heuristic table.
    knobs, source = resolve_knobs(max_tokens=64, **{**_KEY, "dtype": "mxfp8_e4m3"})
    assert source == "heuristic"
    assert knobs == default_knobs(64, dtype="mxfp8")


def test_resolve_prefers_cache_hit(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        record_knobs,
        resolve_knobs,
    )

    _cache_env(monkeypatch, tmp_path)
    with mock.patch(
        "flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.knob_cache."
        "_current_device_name",
        return_value="testgpu",
    ):
        record_knobs(_KNOBS, max_tokens=2048, **_KEY)
        knobs, source = resolve_knobs(max_tokens=2048, **_KEY)
    assert source == "cache"
    assert knobs == _KNOBS


def test_cache_disable_switch(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    assert record_knobs(_KNOBS, max_tokens=2048, device="testgpu", **_KEY) is None
    assert lookup_knobs(max_tokens=2048, device="testgpu", **_KEY) is None


def test_corrupt_cache_file_warns_and_misses(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import lookup_knobs

    path = _cache_env(monkeypatch, tmp_path)
    path.write_text("{not json")
    with pytest.warns(RuntimeWarning, match="unreadable"):
        assert lookup_knobs(max_tokens=2048, device="testgpu", **_KEY) is None


def test_backend_warns_on_auto_knobs():
    from flashinfer.moe_ep import Nvfp4CutedslMegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    with pytest.warns(UserWarning, match="offline"):
        create_mega_kernel(
            Nvfp4CutedslMegaMoeConfig(intermediate_size=128, top_k=2, knobs="auto")
        )


@pytest.mark.arch_blackwell
def test_symm_buffer_resolves_cached_knobs(monkeypatch, tmp_path):
    """knobs=None buffer creation must pick up the recorded winner."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        record_knobs,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _cache_env(monkeypatch, tmp_path)
    hidden, intermediate2x, num_experts, topk, max_tokens = 2048, 2048, 4, 4, 64
    cached = {
        "mma_tiler_mnk": (256, 128, 256),
        "cluster_shape_mnk": (2, 1, 1),
        "group_hint": 128,
        "flag_batch": 16,
        "epi_flag_batch": (1, 2),
        "token_back_mode": "standalone_warps",
        "load_balance_mode": "atomic_counter",
    }
    record_knobs(
        cached,
        dtype="nvfp4",
        world_size=1,
        hidden=hidden,
        intermediate=intermediate2x,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
    )
    buf = get_symm_buffer_for_mega_moe(
        num_experts, max_tokens, topk, hidden, intermediate2x, 0, 1
    )
    try:
        cfg = buf._frontend.config
        assert cfg.flag_batch == 16
        assert cfg.group_hint == 128
        assert cfg.token_back_mode == "standalone_warps"
        assert cfg.epi_flag_batch == (1, 2)
    finally:
        buf.destroy()
