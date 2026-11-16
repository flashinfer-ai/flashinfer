# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Portable W8A8 admission, resolved-plan identity and persistent-cache checks."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from flashinfer import moe_ep
from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp8_batch_invariant import (
    MXFP8_BATCH_INVARIANT_VERSION,
    mxfp8_batch_invariant_default_knobs,
    mxfp8_batch_invariant_fingerprint,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

pytestmark = pytest.mark.cpu_only
PROVIDERS = ("legacy", "standard", "genphase", "107-standard", "107-genphase")
KINDS = ("mxfp8_e4m3", "mxfp8_e5m2")


def _public(provider, kind="mxfp8_e4m3", **overrides):
    cls = (
        moe_ep.Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
        if provider.startswith("107")
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    )
    kwargs = dict(intermediate_size=384, top_k=3, kind=kind, batch_invariant=True)
    if provider == "standard":
        kwargs["kernel_name"] = "sm100_mxfp8_mxfp8_bf16_cutedsl_next"
    if "genphase" in provider:
        kwargs["kernel_variant"] = "genphase"
        if provider.startswith("107"):
            kwargs.update(cluster_shape_mn=(4, 1), fc2_use_bulk=True)
    kwargs.update(overrides)
    return cls(**kwargs)


def _backend(provider, kind="mxfp8_e4m3", **kwargs):
    backend = create_mega_kernel(_public(provider, kind, **kwargs))
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    return backend


def _fleet(capacity=17, hidden=384):
    return moe_ep.FleetParams(
        num_experts=4, max_tokens_per_rank=capacity, token_hidden_size=hidden
    )


def _resolved(backend, fleet):
    if backend._config.kernel_name == "sm100_mxfp8_mxfp8_bf16_cutedsl":
        getter = (
            backend._genphase_config
            if backend._config.kernel_variant == "genphase"
            else backend._legacy_config
        )
    else:
        getter = getattr(backend, "_resolved_config", None) or backend._workspace_config
    return getter(fleet)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("kind", KINDS)
def test_receipt_matches_actual_plan_and_all_admitted_candidates(
    provider, kind, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    backend = _backend(provider, kind, gate_up_clamp=2.5)
    fleet = _fleet()
    cfg = _resolved(backend, fleet)
    arch = "sm_107" if provider.startswith("107") else "sm_100"
    expected = mxfp8_batch_invariant_fingerprint(
        backend._config, hidden=384, num_experts=4, arch=arch
    )
    assert backend.batch_invariant_fingerprint(fleet) == expected
    assert mxfp8_batch_invariant_fingerprint(cfg, arch=arch) == expected
    assert backend.batch_invariant_fingerprint(_fleet(257)) == expected
    assert cfg.input_norm_const == 1.0
    if provider == "legacy":
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
            mxfp8_candidates,
            with_knobs,
        )

        assert cfg.mma_tiler_mnk == (256, 256, 128)
        assert cfg.has_3_global_scales is False
        candidates = mxfp8_candidates(batch_invariant=True)
        plans = [with_knobs(cfg, knobs) for knobs in candidates]
        assert len(plans) == 4
    elif provider.startswith("107"):
        from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.autotune import (
            is_valid_sm107,
            sm107_candidates,
        )

        candidates = sm107_candidates(
            kind, kernel_variant=cfg.kernel_variant, batch_invariant=True
        )
        plans = [replace(cfg, **k) for k in candidates if is_valid_sm107(k, cfg)]
    else:
        from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import tuning

        generator = (
            tuning.sm100_genphase_candidates
            if provider == "genphase"
            else tuning.sm100_standard_candidates
        )
        candidates = (
            generator(cfg, sm_count=148) if provider == "genphase" else generator(cfg)
        )
        plans = [replace(cfg, **k) for k in candidates]
    assert plans
    assert {mxfp8_batch_invariant_fingerprint(plan, arch=arch) for plan in plans} == {
        expected
    }
    if provider != "legacy":
        assert (cfg.has_fc1_alpha, cfg.has_fc2_alpha, cfg.has_fc1_norm_const) == (
            False,
            False,
            False,
        )
        assert cfg.resolved_mma_tiler_mnk == (256, 128, 128)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 33},
        {"top_k": True},
        {"intermediate_size": 192},
        {"batch_invariant": 1},
        {"activation": "situ"},
        {"knobs": {"unknown": True}},
        {"knobs": {"mma_tiler_mnk": (256, 64, 128)}},
    ],
)
def test_public_rejects_unadmitted_arithmetic_before_allocation(provider, overrides):
    with pytest.raises(ValueError):
        _public(provider, **overrides)


def test_legacy_pins_flags_carriers_and_early_weights(monkeypatch):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import with_knobs

    with pytest.raises(ValueError, match="early"):
        _public("legacy", apply_routing_weights_before_fc2=False)
    cfg = _resolved(_backend("legacy"), _fleet())
    for knobs in (
        {"token_back_mode": "standalone_warps"},
        {"flag_batch": 8},
        {"epi_flag_batch": (2, 4)},
        {"in_kernel_fc2_reduce": True},
    ):
        with pytest.raises(ValueError):
            with_knobs(cfg, knobs)
    with pytest.raises(ValueError, match="hidden"):
        _backend("legacy").batch_invariant_fingerprint(_fleet(hidden=192))


def test_genphase_rejects_unsupported_public_carriers_and_flags():
    for knobs in (
        {"token_back_mode": "reuse_dispatch_warps"},
        {"epi_flag_batches": (1, 1)},
        {"token_in_flag_batch": 2},
    ):
        with pytest.raises(ValueError):
            _public("107-genphase", knobs=knobs)
        with pytest.raises(ValueError, match="unvalidated"):
            _public("genphase", knobs=knobs)


def test_w8a8_cannot_select_mxfp4_weight_provider():
    with pytest.raises(ValueError, match="MXFP8-weight"):
        _public("standard", kernel_name="sm100_mxfp8_mxfp4_bf16_cutedsl")
    for kind in ("nvfp4", "mxfp4"):
        with pytest.raises(ValueError, match="unsupported"):
            _public("standard", kind=kind)


def test_phase_interleave_requires_atomic_work_ids():
    with pytest.raises(ValueError, match="atomic_counter"):
        _public("107-standard", schedule_policy=("phase_interleave", None))


def test_legacy_autotune_records_effective_clamp_and_filters_explicit_bad_plan(
    monkeypatch,
):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import (
        autotune,
        knob_cache,
    )

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    cfg = _resolved(_backend("legacy", "mxfp8_e5m2", gate_up_clamp=1.0), _fleet())
    frontend = SimpleNamespace(config=cfg)
    frontend.set_gate_up_clamp = lambda value: setattr(
        frontend, "config", replace(frontend.config, gate_up_clamp=value)
    )
    recorded = []
    monkeypatch.setattr(
        knob_cache,
        "record_knobs",
        lambda knobs, **kwargs: recorded.append((knobs, kwargs)),
    )
    safe = autotune.mxfp8_candidates(batch_invariant=True)[0]

    def tune(_frontend, _launch, candidates, **kwargs):
        candidates = kwargs["candidate_filter"](candidates)
        assert candidates == [safe]
        autotune._apply_config_overrides(_frontend, kwargs["config_overrides"])
        kwargs["on_winner"](safe, 1e-6)
        return safe

    monkeypatch.setattr(autotune, "autotune_knobs", tune)
    with pytest.warns(RuntimeWarning, match="flag_batch"):
        autotune.autotune_mxfp8_mega_moe(
            None,
            None,
            None,
            SimpleNamespace(_frontend=frontend),
            gate_up_clamp=3.0,
            candidates=[{**safe, "flag_batch": 8}, safe],
        )
    assert recorded[0][1]["gate_up_clamp"] == 3.0
    assert recorded[0][1]["batch_invariant"] is True
    assert recorded[0][1]["apply_routing_weights_before_fc2"] is True
    assert recorded[0][1]["intermediate"] == 384


@pytest.mark.parametrize("provider", PROVIDERS)
def test_native_metadata_cannot_add_scalar_multiplications(provider, monkeypatch):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    cfg = _resolved(_backend(provider), _fleet())
    field = "has_3_global_scales" if provider == "legacy" else "has_fc1_alpha"
    with pytest.raises(ValueError):
        replace(cfg, **{field: True})
    with pytest.raises(ValueError):
        replace(cfg, input_norm_const=2.0)


def test_auto_contract_isolates_dtype_provider_input_boundary_and_weight_placement():
    a = moe_ep.MegaConfig(megakernel=_public("standard"))
    moe_ep.AutoConfig(candidates=(a, replace(a)))
    for other in (
        _public("standard", "mxfp8_e5m2"),
        _public("legacy"),
        _public("genphase"),
        _public("107-standard"),
        _public("standard", gate_up_clamp=1.0),
        _public("standard", apply_routing_weights_before_fc2=False),
        _public("standard", batch_invariant=False),
    ):
        with pytest.raises(ValueError, match="cannot mix|same calculation"):
            moe_ep.AutoConfig(candidates=(a, moe_ep.MegaConfig(megakernel=other)))
    with pytest.raises(ValueError, match="same calculation"):
        moe_ep.AutoConfig(
            candidates=(a, replace(a, quantize_input=not a.quantize_input))
        )


@pytest.mark.parametrize("kind", KINDS)
def test_legacy_disk_cache_roundtrip_and_invalid_plan_rejection(
    kind, tmp_path, monkeypatch
):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import knob_cache

    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    monkeypatch.setattr(knob_cache, "_current_device_name", lambda: "test")
    key = dict(
        dtype=kind,
        world_size=1,
        hidden=384,
        intermediate=384,
        num_experts=4,
        topk=3,
        max_tokens=17,
        batch_invariant=True,
        apply_routing_weights_before_fc2=True,
    )
    knobs = mxfp8_batch_invariant_default_knobs("legacy", quant_kind=kind)
    knob_cache.record_knobs(knobs, **key)
    assert knob_cache.lookup_knobs(**key) == knobs
    original = json.loads(path.read_text())
    entry = original["entries"][0]
    assert entry["batch_invariant_version"] == MXFP8_BATCH_INVARIANT_VERSION
    for field, value in (
        ("batch_invariant_fingerprint", None),
        ("batch_invariant_version", 0),
        ("knobs", {**entry["knobs"], "flag_batch": 8}),
    ):
        broken = deepcopy(original)
        broken["entries"][0][field] = value
        path.write_text(json.dumps(broken))
        assert knob_cache.lookup_knobs(**key) is None
    path.write_text(json.dumps(original))
    assert knob_cache.lookup_knobs(**{**key, "batch_invariant": False}) is None
    assert (
        knob_cache.lookup_knobs(**{**key, "dtype": KINDS[1 - KINDS.index(kind)]})
        is None
    )


@pytest.mark.parametrize("provider", ("standard", "genphase"))
@pytest.mark.parametrize("kind", KINDS)
def test_blackwell_cache_isolates_kind_receipt_and_policy(
    provider, kind, tmp_path, monkeypatch
):
    from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        knob_cache,
        tuning,
    )

    cfg = _resolved(_backend(provider, kind), _fleet())
    knobs = {k: getattr(cfg, k) for k in tuning.knob_keys(cfg)}
    record = knob_cache.make_record(knobs, cfg, live_tokens=17, p50_us=1, device="test")
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    path.write_text(json.dumps(dict(version=KNOB_CACHE_VERSION, entries=[record])))
    assert len(knob_cache._entries(cfg, device="test")) == 1
    assert knob_cache._entries(replace(cfg, batch_invariant=False), device="test") == []
    for field, value in (
        ("batch_invariant_fingerprint", "wrong"),
        ("batch_invariant_version", 0),
    ):
        broken = deepcopy(record)
        broken["problem"][field] = value
        path.write_text(json.dumps(dict(version=KNOB_CACHE_VERSION, entries=[broken])))
        assert knob_cache._entries(cfg, device="test") == []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("variant", ("inference", "genphase"))
def test_rubin_cache_roundtrip_and_fail_closed(kind, variant):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import knob_cache

    knobs = mxfp8_batch_invariant_default_knobs(
        "genphase" if variant == "genphase" else "standard", quant_kind=kind
    )
    key = dict(
        dtype=kind,
        kernel_variant=variant,
        world_size=1,
        hidden=384,
        intermediate=384,
        num_experts=4,
        topk=3,
        max_tokens=17,
        batch_invariant=True,
    )
    record = knob_cache.make_record(knobs, **key)
    assert knob_cache.lookup_knobs(entries=[record], **key) == knobs
    for field, value in (
        ("batch_invariant_fingerprint", None),
        ("batch_invariant_version", 0),
        ("knobs", {**record["knobs"], "token_in_flag_batch": 4}),
        (
            "knobs",
            {
                **record["knobs"],
                "schedule_policy": ["phase_interleave", None],
                "work_id_mode": "grid_stride",
            },
        ),
    ):
        assert (
            knob_cache.lookup_knobs(entries=[{**record, field: value}], **key) is None
        )
    assert (
        knob_cache.lookup_knobs(entries=[record], **{**key, "batch_invariant": False})
        is None
    )


@pytest.mark.parametrize("arch", ("sm100", "sm107"))
@pytest.mark.parametrize("kind", KINDS)
def test_input_stage_pins_floor_scales_and_format_without_fallback(
    arch, kind, monkeypatch
):
    staging = importlib.import_module(
        f"flashinfer.moe_ep.backends.mega.kernel.{arch}.mxfp8_mxfp8_bf16_cutedsl.staging"
    )
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as shim

    class CudaMetadataTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    live, capacity, hidden = 17, 20, 384
    x = torch.zeros((live, hidden), dtype=torch.bfloat16).as_subclass(
        CudaMetadataTensor
    )
    ids = torch.zeros((live, 3), dtype=torch.int64).as_subclass(CudaMetadataTensor)
    scores = torch.zeros((live, 3), dtype=torch.float32).as_subclass(CudaMetadataTensor)
    out = torch.empty((capacity, hidden), dtype=torch.uint8)
    sf = torch.empty((capacity, 128), dtype=torch.uint8)
    out_ids = torch.empty((capacity, 3), dtype=torch.int32)
    out_scores = torch.empty((capacity, 3), dtype=torch.float32)
    calls = []
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "0")
    monkeypatch.setattr(
        shim, "fused_quant_stage", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(shim, "note_staged_tokens", lambda *args: None)
    staging.stage_mega_moe_inputs(
        x, scores, ids, out, sf, out_ids, out_scores, kind=kind, batch_invariant=True
    )
    assert len(calls) == 1
    assert calls[0][0][4].shape == (capacity, hidden // 32)
    assert calls[0][0][4].stride() == sf.stride()
    assert calls[0][1]["batch_invariant"] is True
    assert calls[0][1]["mxfp8_normal_scales"] is True
    assert calls[0][1]["batch_invariant_quant_kind"] == kind
    with pytest.raises(ValueError, match="contiguous|alignment|aligned"):
        staging.stage_mega_moe_inputs(
            x[:, ::2],
            scores,
            ids,
            out,
            sf,
            out_ids,
            out_scores,
            kind=kind,
            batch_invariant=True,
        )
    assert len(calls) == 1
