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

pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")


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
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
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
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
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
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
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

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
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


def test_ikr_and_deterministic_winners_are_separate_entries(monkeypatch, tmp_path):
    """The two objectives are tuned separately, so neither may evict the other."""
    import json

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    path = _cache_env(monkeypatch, tmp_path)
    ikr = {**_KNOBS, "flag_batch": 8, "in_kernel_fc2_reduce": True}
    record_knobs(_KNOBS, max_tokens=2048, device="testgpu", **_KEY)
    record_knobs(ikr, max_tokens=2048, device="testgpu", **_KEY)
    assert len(json.loads(path.read_text())["entries"]) == 2

    # A reproducible session must not be served the ikr winner: its other
    # knobs were only ever measured alongside ikr.
    assert lookup_knobs(max_tokens=2048, device="testgpu", **_KEY) == _KNOBS
    assert (
        lookup_knobs(
            max_tokens=2048,
            device="testgpu",
            enable_in_kernel_fc2_reduce=True,
            **_KEY,
        )
        == ikr
    )


def test_permitted_session_falls_back_to_a_deterministic_entry(monkeypatch, tmp_path):
    """The permission is a ceiling, so a non-ikr winner stays usable."""
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    _cache_env(monkeypatch, tmp_path)
    record_knobs(_KNOBS, max_tokens=2048, device="testgpu", **_KEY)
    assert (
        lookup_knobs(
            max_tokens=2048,
            device="testgpu",
            enable_in_kernel_fc2_reduce=True,
            **_KEY,
        )
        == _KNOBS
    )


@pytest.mark.parametrize("dtype", ["nvfp4", "bf16_nvfp4"])
def test_resolve_ignores_an_ikr_entry_for_a_deterministic_session(
    monkeypatch, tmp_path, dtype
):
    """Falling back to the heuristic beats serving a knob set never measured."""
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        default_knobs,
        record_knobs,
        resolve_knobs,
    )

    key = {**_KEY, "dtype": dtype}
    _cache_env(monkeypatch, tmp_path)
    with mock.patch(
        "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.knob_cache."
        "_current_device_name",
        return_value="testgpu",
    ):
        record_knobs({**_KNOBS, "in_kernel_fc2_reduce": True}, max_tokens=2048, **key)
        knobs, source = resolve_knobs(max_tokens=2048, **key)
        permitted, permitted_source = resolve_knobs(
            max_tokens=2048, enable_in_kernel_fc2_reduce=True, **key
        )
    assert (source, knobs) == ("heuristic", default_knobs(2048, dtype=dtype))
    assert permitted_source == "cache"
    assert permitted["in_kernel_fc2_reduce"] is True


def test_resolve_falls_back_to_heuristic(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import resolve_knobs
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        default_knobs,
    )

    _cache_env(monkeypatch, tmp_path)  # empty cache
    knobs, source = resolve_knobs(max_tokens=2048, **_KEY)
    assert source == "heuristic"
    assert knobs == default_knobs(2048)
    # mxfp8 kinds route to the mxfp8 heuristic table.
    knobs, source = resolve_knobs(max_tokens=64, **{**_KEY, "dtype": "mxfp8_e4m3"})
    assert source == "heuristic"
    assert knobs == default_knobs(64, dtype="mxfp8")


def test_resolve_prefers_cache_hit(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        record_knobs,
        resolve_knobs,
    )

    _cache_env(monkeypatch, tmp_path)
    with mock.patch(
        "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.knob_cache."
        "_current_device_name",
        return_value="testgpu",
    ):
        record_knobs(_KNOBS, max_tokens=2048, **_KEY)
        knobs, source = resolve_knobs(max_tokens=2048, **_KEY)
    assert source == "cache"
    assert knobs == _KNOBS


def test_cache_disable_switch(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    assert record_knobs(_KNOBS, max_tokens=2048, device="testgpu", **_KEY) is None
    assert lookup_knobs(max_tokens=2048, device="testgpu", **_KEY) is None


def test_corrupt_cache_file_warns_and_misses(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import lookup_knobs

    path = _cache_env(monkeypatch, tmp_path)
    path.write_text("{not json")
    with pytest.warns(RuntimeWarning, match="unreadable"):
        assert lookup_knobs(max_tokens=2048, device="testgpu", **_KEY) is None


def test_backend_warns_on_auto_knobs():
    from flashinfer.moe_ep import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    with pytest.warns(UserWarning, match="offline"):
        create_mega_kernel(
            Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=128, top_k=2, knobs="auto"
            )
        )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "mode,mma_m,explicit",
    [("nvfp4", 256, False), ("bf16_nvfp4", 128, False), ("bf16_nvfp4", 128, True)],
)
def test_symm_buffer_resolves_cached_knobs(
    monkeypatch, tmp_path, mode, mma_m, explicit
):
    """Cached and explicit partial tiles derive the same CTA instruction mode."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from flashinfer.utils import get_compute_capability

    cap = get_compute_capability(torch.device("cuda"))
    if cap[0] != 10:
        pytest.skip(f"needs sm_100/sm_103; got sm_{cap[0]}{cap[1]}")

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        record_knobs,
    )

    if mode == "bf16_nvfp4":
        from flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4 import (
            get_symm_buffer_for_bf16_nvfp4_mega_moe as get_symm_buffer_for_mega_moe,
        )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _cache_env(monkeypatch, tmp_path)
    hidden, intermediate2x, num_experts, topk, max_tokens = 2048, 2048, 4, 4, 64
    token_back_mode = {
        "nvfp4": "standalone_warps",
        "bf16_nvfp4": "reuse_dispatch_warps",
    }[mode]
    cached = {
        "mma_tiler_mnk": (mma_m, 128, 256),
        "cluster_shape_mnk": (2, 1, 1),
        "group_hint": 128,
        "flag_batch": 16,
        "epi_flag_batch": (1, 2),
        "token_back_mode": token_back_mode,
        "load_balance_mode": "atomic_counter",
    }
    record_knobs(
        {**cached, "flag_batch": 8} if explicit else cached,
        dtype=mode,
        world_size=1,
        hidden=hidden,
        intermediate=intermediate2x,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
    )
    knobs = cached if explicit else None
    buf = get_symm_buffer_for_mega_moe(
        num_experts, max_tokens, topk, hidden, intermediate2x, 0, 1, knobs=knobs
    )
    try:
        cfg = buf._frontend.config
        assert cfg.mma_tiler_mnk == cached["mma_tiler_mnk"]
        assert cfg.use_2cta_instrs == (mma_m == 256)
        assert cfg.flag_batch == 16
        assert cfg.group_hint == 128
        assert cfg.token_back_mode == token_back_mode
        assert cfg.epi_flag_batch == (1, 2)
    finally:
        buf.destroy()


def test_routing_weight_placement_partitions_cache(monkeypatch, tmp_path):
    import json

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import knob_cache

    path = _cache_env(monkeypatch, tmp_path)
    monkeypatch.setattr(knob_cache, "_current_device_name", lambda: "testgpu")
    key = dict(max_tokens=2048, **{**_KEY, "dtype": "bf16_nvfp4"})
    knob_cache.record_knobs(_KNOBS, **key)
    assert knob_cache.lookup_knobs(**key) == _KNOBS
    assert knob_cache.lookup_knobs(apply_topk_in_fc1=False, **key) == _KNOBS
    assert knob_cache.lookup_knobs(apply_topk_in_fc1=True, **key) is None

    fc1_knobs = {**_KNOBS, "flag_batch": 8}
    knob_cache.record_knobs(fc1_knobs, apply_topk_in_fc1=True, **key)
    replacement = {**_KNOBS, "flag_batch": 2}
    knob_cache.record_knobs(replacement, apply_topk_in_fc1=False, **key)
    assert knob_cache.resolve_knobs(**key) == (replacement, "cache")
    assert knob_cache.resolve_knobs(apply_topk_in_fc1=True, **key) == (
        fc1_knobs,
        "cache",
    )
    entries = json.loads(path.read_text())["entries"]
    assert len(entries) == 2
    assert {entry["apply_topk_in_fc1"] for entry in entries} == {False, True}
