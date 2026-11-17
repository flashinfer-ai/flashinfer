# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Portable admission/cache/metadata checks for the MXFP4 BI frontends."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json

import pytest
import torch

from flashinfer import moe_ep
from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp4_batch_invariant import (
    MXFP4_BATCH_INVARIANT_VERSION,
    mxfp4_batch_invariant_default_knobs,
    mxfp4_batch_invariant_fingerprint,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

pytestmark = pytest.mark.cpu_only
PROVIDERS = (
    "100-standard",
    "100-genphase",
    "100-next-genphase",
    "107-mixed-standard",
    "107-mixed-genphase",
    "107-pure-standard",
    "107-pure-genphase",
)


def _public(provider, **overrides):
    kwargs = dict(intermediate_size=256, top_k=3, batch_invariant=True)
    if provider.startswith("100"):
        if provider == "100-standard":
            cls = moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
        else:
            cls = moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
            kwargs.update(kind="mxfp4_mxfp8", kernel_variant="genphase")
            if "next" in provider:
                kwargs["kernel_name"] = "sm100_mxfp8_mxfp8_bf16_cutedsl_next"
    else:
        cls = (
            moe_ep.Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
            if "pure" in provider
            else moe_ep.Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
        )
        if "genphase" in provider:
            kwargs.update(
                kernel_variant="genphase", fc2_use_bulk=True, cluster_shape_mn=(4, 1)
            )
    kwargs.update(overrides)
    return cls(**kwargs)


def _backend(provider, **kwargs):
    backend = create_mega_kernel(_public(provider, **kwargs))
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    return backend


def _resolved(backend, capacity=17):
    fleet = moe_ep.FleetParams(
        num_experts=4, max_tokens_per_rank=capacity, token_hidden_size=384
    )
    getter = getattr(
        backend,
        "_resolved_config",
        getattr(
            backend, "_workspace_config", getattr(backend, "_genphase_config", None)
        ),
    )
    return getter(fleet), fleet


@pytest.mark.parametrize("provider", PROVIDERS)
def test_public_receipt_matches_pinned_workspace_and_all_candidates(
    provider, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    backend = _backend(
        provider, gate_up_clamp=2.5, apply_routing_weights_before_fc2=False
    )
    cfg, fleet = _resolved(backend)
    arch = "sm_107" if provider.startswith("107") else "sm_100"
    expected = mxfp4_batch_invariant_fingerprint(
        backend._config, hidden=384, num_experts=4, arch=arch
    )
    assert backend.batch_invariant_fingerprint(fleet) == expected
    assert mxfp4_batch_invariant_fingerprint(cfg, arch=arch) == expected
    assert (cfg.has_fc1_alpha, cfg.has_fc2_alpha, cfg.has_fc1_norm_const) == (
        False,
        False,
        False,
    )
    assert cfg.input_norm_const == 1.0
    assert cfg.resolved_mma_tiler_mnk == (256, 128, 256 if "pure" in provider else 128)
    cfg2, _ = _resolved(backend, capacity=257)
    assert mxfp4_batch_invariant_fingerprint(cfg2, arch=arch) == expected
    if arch == "sm_107":
        from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.autotune import (
            sm107_candidates,
            is_valid_sm107,
        )

        candidates = [
            k
            for k in sm107_candidates(
                cfg.quant_kind, kernel_variant=cfg.kernel_variant, batch_invariant=True
            )
            if is_valid_sm107(k, cfg)
        ]
    else:
        from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import (
            sm100_standard_candidates,
            sm100_genphase_candidates,
        )

        candidates = (
            sm100_genphase_candidates(cfg, sm_count=148, live_tokens=17)
            if "genphase" in provider
            else sm100_standard_candidates(cfg)
        )
    assert candidates
    for knobs in candidates:
        assert (
            mxfp4_batch_invariant_fingerprint(replace(cfg, **knobs), arch=arch)
            == expected
        )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "overrides",
    [
        {"intermediate_size": 192},
        {"top_k": 33},
        {"top_k": True},
        {"apply_routing_weights_before_fc2": 1},
        {"combine_dtype": "fp32"},
        {"activation": "situ", "situ_beta": -1.0, "situ_linear_beta": 1.0},
        {"knobs": {"unknown_future_knob": False}},
        {"knobs": {"mma_tiler_mnk": (256, 64, 128)}},
        {"knobs": {"epi_flag_batches": (1, 1)}},
    ],
)
def test_unsupported_modes_fail_before_allocation(provider, overrides):
    with pytest.raises((TypeError, ValueError)):
        _public(provider, **overrides)


def test_unknown_kind_and_auto_variant_are_not_admitted():
    from types import SimpleNamespace
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.block_scaled_batch_invariant import (
        validate_block_scaled_batch_invariant,
    )

    with pytest.raises(ValueError, match="unsupported"):
        validate_block_scaled_batch_invariant(SimpleNamespace(batch_invariant=True))
    with pytest.raises(ValueError, match="cross-variant"):
        _public("107-mixed-standard", kernel_variant="auto", knobs="auto")
    for provider in PROVIDERS:
        field = (
            "enable_in_kernel_fc2_reduce"
            if provider.startswith("100")
            else "in_kernel_fc2_reduce"
        )
        with pytest.raises(ValueError):
            _public(provider, **{field: True})


def test_auto_contract_isolates_kind_provider_clamp_and_weight_placement():
    a = moe_ep.MegaConfig(megakernel=_public("107-mixed-standard"))
    moe_ep.AutoConfig(candidates=(a, replace(a)))
    for other in (
        _public("107-pure-standard"),
        _public("107-mixed-genphase"),
        _public("107-mixed-standard", gate_up_clamp=2.5),
        _public("107-mixed-standard", apply_routing_weights_before_fc2=False),
        _public("107-mixed-standard", batch_invariant=False),
    ):
        with pytest.raises(ValueError, match="cannot mix|same calculation"):
            moe_ep.AutoConfig(candidates=(a, moe_ep.MegaConfig(megakernel=other)))


@pytest.mark.parametrize("provider", ["100-standard", "100-genphase"])
def test_blackwell_cache_separates_policy_and_receipt(provider, tmp_path, monkeypatch):
    from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        knob_cache,
        tuning,
    )

    cfg, _ = _resolved(_backend(provider))
    knobs = {k: getattr(cfg, k) for k in tuning.knob_keys(cfg)}
    record = knob_cache.make_record(knobs, cfg, live_tokens=17, p50_us=1, device="test")
    assert record["problem"]["batch_invariant_version"] == MXFP4_BATCH_INVARIANT_VERSION
    path = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    path.write_text(json.dumps(dict(version=KNOB_CACHE_VERSION, entries=[record])))
    assert len(knob_cache._entries(cfg, device="test")) == 1
    assert knob_cache._entries(replace(cfg, batch_invariant=False), device="test") == []
    for key, value in (
        ("batch_invariant_fingerprint", "wrong"),
        ("batch_invariant_version", 0),
    ):
        broken = deepcopy(record)
        broken["problem"][key] = value
        path.write_text(json.dumps(dict(version=KNOB_CACHE_VERSION, entries=[broken])))
        assert knob_cache._entries(cfg, device="test") == []
    broken = deepcopy(record)
    broken["knobs"]["mma_tiler_mnk"] = [256, 64, 128]
    assert knob_cache._decode(broken, cfg) is None


@pytest.mark.parametrize("kind", ["mxfp4", "mxfp4_mxfp8"])
@pytest.mark.parametrize("variant", ["inference", "genphase"])
def test_rubin_cache_roundtrip_and_fail_closed(kind, variant):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import knob_cache

    implementation = "standard" if variant == "inference" else variant
    knobs = mxfp4_batch_invariant_default_knobs(
        implementation, quant_kind=kind, arch="sm_107"
    )
    key = dict(
        dtype=kind,
        kernel_variant=variant,
        world_size=1,
        hidden=384,
        intermediate=256,
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
        ("knobs", {**record["knobs"], "reduce_topk_in_kernel": True}),
        ("knobs", {**record["knobs"], "token_in_flag_batch": 4}),
    ):
        assert (
            knob_cache.lookup_knobs(entries=[{**record, field: value}], **key) is None
        )
    assert (
        knob_cache.lookup_knobs(entries=[record], **{**key, "batch_invariant": False})
        is None
    )


@pytest.mark.parametrize(
    "arch,kind",
    [("sm100", "mxfp4_mxfp8"), ("sm107", "mxfp4"), ("sm107", "mxfp4_mxfp8")],
)
def test_fixed_fused_staging_with_padded_sf_and_no_stride_fallback(
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
    out = torch.empty(
        (capacity, hidden // 2 if kind == "mxfp4" else hidden), dtype=torch.uint8
    )
    sf = torch.empty((capacity, 128), dtype=torch.uint8)
    out_ids = torch.empty((capacity, 3), dtype=torch.int32)
    out_scores = torch.empty((capacity, 3), dtype=torch.float32)
    calls = []
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "0")
    monkeypatch.setenv("FLASHINFER_SM107_MEGA_CHECK_ROUTES", "0")
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
