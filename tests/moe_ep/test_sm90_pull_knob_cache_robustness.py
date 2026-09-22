"""Robustness contracts for the shared SM90 persistent knob cache."""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    knob_cache,
    hopper_mxfp4,
    mxfp4_tuner,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
    hopper_mxfp4_cache_provenance_sha256,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)


_KEY = {
    "dtype": (
        "sm90_w_mxfp4_e2m1_k32_a_fp8_e4m3_per_token_full_hidden_"
        "humming_v1_fold_m64_k128_gateup8_packedk2_residual64_"
        "swapab_fused_layout_v2"
    ),
    "fp8_scale_mode": "mxfp4_hybrid",
    "world_size": 4,
    "hidden": 7168,
    "intermediate": 3072,
    "num_experts": 384,
    "topk": 6,
}
_MXFP4_HARDWARE = {
    "compute_capability": (9, 0),
    "sm_count": 132,
}
_FUSED_BLOCK_PROVENANCE_SHA256 = hopper_mxfp4_cache_provenance_sha256(
    routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
)


def _entry(*, knobs: dict[str, Any], gate_up_clamp: float | None | object) -> dict:
    entry = {
        "device": "NVIDIA H200",
        **_KEY,
        "compute_capability": [9, 0],
        "sm_count": 132,
        "tuning_provenance_sha256": _FUSED_BLOCK_PROVENANCE_SHA256,
        "max_tokens": 512,
        "knobs": knobs,
        "p50_us": 1.0,
        "source": "test",
        "tuned_at": "2026-08-28T00:00:00",
    }
    if gate_up_clamp is not _MISSING:
        entry["gate_up_clamp"] = gate_up_clamp
    return entry


_MISSING = object()


def _lookup(**overrides: Any) -> dict[str, Any] | None:
    routing_profile = overrides.get(
        "routing_profile", SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    )
    tuning_provenance_sha256 = overrides.pop(
        "tuning_provenance_sha256",
        hopper_mxfp4_cache_provenance_sha256(
            routing_profile=routing_profile,
        ),
    )
    return knob_cache.lookup_knobs(
        max_tokens=512,
        device="NVIDIA H200",
        **_MXFP4_HARDWARE,
        tuning_provenance_sha256=tuning_provenance_sha256,
        **{**_KEY, **overrides},
    )


def test_legacy_entry_without_clamp_is_none_only(tmp_path, monkeypatch) -> None:
    """A pre-axis entry remains usable only for the historical None clamp."""
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    legacy = {"mma_tiler_mnk": [256, 32, 256], "group_hint": 330}
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [_entry(knobs=legacy, gate_up_clamp=_MISSING)],
            }
        )
    )

    assert _lookup(gate_up_clamp=None) == {
        "mma_tiler_mnk": (256, 32, 256),
        "group_hint": 330,
    }
    assert _lookup(gate_up_clamp=10.0) is None


def test_clamp_values_are_distinct_cache_keys(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    unclamped = {"mma_tiler_mnk": (256, 16, 256), "group_hint": 264}
    clamp10 = {"mma_tiler_mnk": (256, 32, 256), "group_hint": 330}

    assert knob_cache.record_knobs(
        unclamped,
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=None,
        tuning_provenance_sha256=_FUSED_BLOCK_PROVENANCE_SHA256,
        **_MXFP4_HARDWARE,
        **_KEY,
    )
    assert knob_cache.record_knobs(
        clamp10,
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=10.0,
        tuning_provenance_sha256=_FUSED_BLOCK_PROVENANCE_SHA256,
        **_MXFP4_HARDWARE,
        **_KEY,
    )

    assert _lookup(gate_up_clamp=None) == unclamped
    assert _lookup(gate_up_clamp=10.0) == clamp10
    assert _lookup(gate_up_clamp=9.0) is None
    assert len(json.loads(path.read_text())["entries"]) == 2


def test_mxfp4_legacy_routing_entry_matches_only_legacy_profile(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    legacy = {"mma_tiler_mnk": [256, 32, 256], "group_hint": 330}
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [_entry(knobs=legacy, gate_up_clamp=10.0)],
            }
        )
    )

    expected = {
        "mma_tiler_mnk": (256, 32, 256),
        "group_hint": 330,
    }
    assert _lookup(gate_up_clamp=10.0) == expected
    assert (
        _lookup(
            gate_up_clamp=10.0,
            routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        )
        is None
    )
    assert (
        _lookup(
            gate_up_clamp=10.0,
            routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        )
        is None
    )


def test_mxfp4_routing_profile_upsert_preserves_profileless_entry(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    _entry(
                        knobs={"group_hint": 111},
                        gate_up_clamp=10.0,
                    )
                ],
            }
        )
    )
    block = {"group_hint": 222}
    exact = {"group_hint": 333}
    common = dict(
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=10.0,
        **_MXFP4_HARDWARE,
        **_KEY,
    )

    assert knob_cache.record_knobs(
        block,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        tuning_provenance_sha256=hopper_mxfp4_cache_provenance_sha256(
            routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        ),
        **common,
    )
    entries = json.loads(path.read_text())["entries"]
    assert len(entries) == 2
    block_entries = [
        entry
        for entry in entries
        if entry.get("routing_profile") == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    ]
    assert len(block_entries) == 1

    assert knob_cache.record_knobs(
        exact,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        tuning_provenance_sha256=hopper_mxfp4_cache_provenance_sha256(
            routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        ),
        **common,
    )
    assert len(json.loads(path.read_text())["entries"]) == 3
    assert _lookup(gate_up_clamp=10.0) == {"group_hint": 111}
    assert (
        _lookup(
            gate_up_clamp=10.0,
            routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        )
        == block
    )
    assert (
        _lookup(
            gate_up_clamp=10.0,
            routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        )
        == exact
    )


def test_fp8_none_profile_keeps_v1_entry_shape_and_does_not_claim_legacy_profile(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    fp8_key = {
        **_KEY,
        "dtype": "fp8_e4m3",
        "fp8_scale_mode": "per_tensor",
    }
    knobs = {"flag_batch": 4}
    common = dict(
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=None,
        **fp8_key,
    )
    assert knob_cache.record_knobs(
        knobs,
        routing_profile=None,
        tuning_provenance_sha256=_FUSED_BLOCK_PROVENANCE_SHA256,
        **common,
    )
    entry = json.loads(path.read_text())["entries"][0]
    assert "routing_profile" not in entry
    assert "tuning_provenance_sha256" not in entry
    assert knob_cache.lookup_knobs(**common) == knobs
    assert (
        knob_cache.lookup_knobs(
            **common,
            routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        )
        is None
    )


def _concurrent_record_worker(
    cache_path: str,
    barrier: multiprocessing.synchronize.Barrier,
    topk: int,
) -> None:
    """Force both legacy writers to read before either replaces the file.

    With the production sidecar lock, writer one owns the lock and times out
    at this test barrier; writer two then reads writer one's committed entry.
    Without a lock both read the empty file together and one update is lost.
    """
    os.environ["FLASHINFER_MOE_EP_KNOB_CACHE"] = cache_path
    original_load = knob_cache._load_entries

    def synchronized_load(path: str) -> list[dict[str, Any]]:
        entries = original_load(path)
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=1.0)
        return entries

    knob_cache._load_entries = synchronized_load
    written = knob_cache.record_knobs(
        {"mma_tiler_mnk": (256, 32, 256), "group_hint": 300 + topk},
        dtype=_KEY["dtype"],
        fp8_scale_mode=_KEY["fp8_scale_mode"],
        world_size=_KEY["world_size"],
        hidden=_KEY["hidden"],
        intermediate=_KEY["intermediate"],
        num_experts=_KEY["num_experts"],
        topk=topk,
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=10.0,
        tuning_provenance_sha256=_FUSED_BLOCK_PROVENANCE_SHA256,
        **_MXFP4_HARDWARE,
    )
    if written != cache_path:
        raise RuntimeError(f"cache write failed: {written!r}")


def test_concurrent_writers_do_not_lose_distinct_entries(tmp_path) -> None:
    path = tmp_path / "cache.json"
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_concurrent_record_worker,
            args=(str(path), barrier, topk),
        )
        for topk in (6, 7)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert not process.is_alive()
        assert process.exitcode == 0

    data = json.loads(Path(path).read_text())
    assert data["version"] == 1
    assert len(data["entries"]) == 2
    assert {entry["topk"] for entry in data["entries"]} == {6, 7}


def test_mxfp4_legacy_entry_without_hardware_identity_fails_closed(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    legacy = _entry(knobs={"group_hint": 111}, gate_up_clamp=10.0)
    legacy.pop("compute_capability")
    legacy.pop("sm_count")
    path.write_text(json.dumps({"version": 1, "entries": [legacy]}))

    assert _lookup(gate_up_clamp=10.0) is None


def test_mxfp4_legacy_entry_without_tuning_provenance_fails_closed(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    legacy = _entry(knobs={"group_hint": 111}, gate_up_clamp=10.0)
    legacy.pop("tuning_provenance_sha256")
    path.write_text(json.dumps({"version": 1, "entries": [legacy]}))

    assert _lookup(gate_up_clamp=10.0) is None


def test_mxfp4_tuning_provenance_is_exact_and_source_is_diagnostic_only(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    stale_provenance = ("0" if _FUSED_BLOCK_PROVENANCE_SHA256[0] != "0" else "1") + (
        _FUSED_BLOCK_PROVENANCE_SHA256[1:]
    )
    knobs = {"group_hint": 111}
    common = dict(
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=10.0,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        source="same-diagnostic-source",
        **_MXFP4_HARDWARE,
        **_KEY,
    )
    assert knob_cache.record_knobs(
        knobs,
        tuning_provenance_sha256=stale_provenance,
        **common,
    )

    assert (
        _lookup(
            gate_up_clamp=10.0,
            routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        )
        is None
    )
    assert (
        _lookup(
            gate_up_clamp=10.0,
            routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
            tuning_provenance_sha256=stale_provenance,
        )
        == knobs
    )


def test_mxfp4_current_provenance_selects_new_row_over_stale_row(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    stale_provenance = ("0" if _FUSED_BLOCK_PROVENANCE_SHA256[0] != "0" else "1") + (
        _FUSED_BLOCK_PROVENANCE_SHA256[1:]
    )
    common = dict(
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=10.0,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        source="same-diagnostic-source",
        **_MXFP4_HARDWARE,
        **_KEY,
    )
    assert knob_cache.record_knobs(
        {"group_hint": 111},
        tuning_provenance_sha256=stale_provenance,
        **common,
    )
    assert knob_cache.record_knobs(
        {"group_hint": 222},
        tuning_provenance_sha256=_FUSED_BLOCK_PROVENANCE_SHA256,
        **common,
    )

    assert len(json.loads(path.read_text())["entries"]) == 2
    assert _lookup(
        gate_up_clamp=10.0,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    ) == {"group_hint": 222}


def test_mxfp4_callers_must_supply_valid_tuning_provenance(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    common = dict(
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=10.0,
        **_MXFP4_HARDWARE,
        **_KEY,
    )
    with pytest.warns(RuntimeWarning, match="tuning-provenance"):
        assert knob_cache.lookup_knobs(**common) is None
    with pytest.warns(RuntimeWarning, match="tuning-provenance"):
        assert knob_cache.record_knobs({"group_hint": 111}, **common) is None
    assert not path.exists()


def test_mxfp4_hardware_identity_is_an_exact_cache_axis(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    knobs = {"mma_tiler_mnk": (256, 32, 256), "group_hint": 330}
    common = dict(
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=10.0,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        tuning_provenance_sha256=_FUSED_BLOCK_PROVENANCE_SHA256,
        **_KEY,
    )
    assert knob_cache.record_knobs(knobs, **_MXFP4_HARDWARE, **common)

    assert knob_cache.lookup_knobs(**_MXFP4_HARDWARE, **common) == knobs
    assert (
        knob_cache.lookup_knobs(compute_capability=(9, 1), sm_count=132, **common)
        is None
    )
    assert (
        knob_cache.lookup_knobs(compute_capability=(9, 0), sm_count=78, **common)
        is None
    )
    assert (
        knob_cache.lookup_knobs(
            compute_capability=(9, 0),
            sm_count=132,
            **{**common, "device": "NVIDIA H20"},
        )
        is None
    )


def test_cache_dtype_identities_spell_both_operand_formats():
    fused = hopper_mxfp4._MXFP4_TUNING_DTYPE_ID
    assert fused == (
        "sm90_w_mxfp4_e2m1_k32_a_fp8_e4m3_per_token_full_hidden_humming_v1_"
        "fold_m64_k128_gateup8_packedk2_residual64_swapab_fused_layout_v2_"
        "fused_local_v2"
    )


def test_persistent_cache_isolates_format_versions(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "FLASHINFER_MOE_EP_KNOB_CACHE",
        str(tmp_path / "cache.json"),
    )
    key = dict(
        fp8_scale_mode="mxfp4_hybrid",
        world_size=4,
        hidden=7168,
        intermediate=3072,
        num_experts=384,
        topk=6,
        max_tokens=64,
        device="NVIDIA H200",
        compute_capability=(9, 0),
        sm_count=132,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    )
    tactic = mxfp4_tuner.hopper_mxfp4_default_tactic(key["max_tokens"])
    dtype = hopper_mxfp4._MXFP4_TUNING_DTYPE_ID + "_different_format"
    provenance = hopper_mxfp4_cache_provenance_sha256(
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    )
    assert knob_cache.record_knobs(
        tactic,
        dtype=dtype,
        tuning_provenance_sha256=provenance,
        **key,
    )
    assert (
        knob_cache.lookup_knobs(
            dtype=dtype,
            tuning_provenance_sha256=provenance,
            **key,
        )
        == tactic
    )
    assert (
        knob_cache.lookup_knobs(
            dtype=hopper_mxfp4._MXFP4_TUNING_DTYPE_ID,
            tuning_provenance_sha256=hopper_mxfp4_cache_provenance_sha256(
                routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
            ),
            **key,
        )
        is None
    )
    # Ordinary SM90 FP8 shares the physical JSON file, but neither its dtype
    # nor its scale ABI may consume an MXFP4 entry.
    fp8_key = {**key, "fp8_scale_mode": "per_tensor"}
    assert (
        knob_cache.lookup_knobs(
            dtype="fp8_e4m3",
            **fp8_key,
        )
        is None
    )


def test_routing_profiles_have_separate_cache_records(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "cache.json"))
    common = dict(
        **_KEY,
        **_MXFP4_HARDWARE,
        max_tokens=512,
        device="NVIDIA H200",
        gate_up_clamp=10.0,
        # Hold provenance fixed here to isolate the routing-profile key axis.
        tuning_provenance_sha256=_FUSED_BLOCK_PROVENANCE_SHA256,
    )
    profiles = (
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    for profile in profiles:
        assert knob_cache.record_knobs(
            {"profile": profile}, routing_profile=profile, **common
        )
    for profile in profiles:
        assert knob_cache.lookup_knobs(routing_profile=profile, **common) == {
            "profile": profile
        }
