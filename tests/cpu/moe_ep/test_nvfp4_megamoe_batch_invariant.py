# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU admission, numerical identity and persisted-plan checks for NVFP4 BI."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from flashinfer.moe_ep import (
    AutoConfig,
    BootstrapConfig,
    FinalizeMode,
    FleetParams,
    MegaConfig,
    Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig as BlackwellConfig,
    Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig as RubinConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
    NVFP4_BATCH_INVARIANT_VERSION,
    nvfp4_batch_invariant_default_knobs,
    nvfp4_batch_invariant_fingerprint,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

pytestmark = pytest.mark.cpu_only


def _public(provider="legacy", **kwargs):
    if provider.startswith("sm107"):
        cls = RubinConfig
        if provider.endswith("genphase"):
            kwargs = dict(
                kernel_variant="genphase",
                cluster_shape_mn=(4, 1),
                fc2_use_bulk=True,
                **kwargs,
            )
    else:
        cls = BlackwellConfig
        if provider != "legacy":
            kwargs = dict(kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next", **kwargs)
        if provider == "genphase":
            kwargs = dict(kernel_variant="genphase", **kwargs)
    return cls(intermediate_size=256, top_k=2, batch_invariant=True, **kwargs)


def _backend(provider, **kwargs):
    backend = create_mega_kernel(_public(provider, **kwargs))
    backend.bind_ep_bootstrap(
        BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    return backend


def _fleet(capacity=16):
    return FleetParams(
        num_experts=4, max_tokens_per_rank=capacity, token_hidden_size=256
    )


@pytest.mark.parametrize(
    "provider", ["legacy", "standard", "genphase", "sm107-standard", "sm107-genphase"]
)
def test_public_receipt_matches_actual_workspace_arithmetic(provider, monkeypatch):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    backend = _backend(provider, gate_up_clamp=7.0, input_norm_const=2.0)
    fleet = _fleet()
    config = backend._config
    arch = "sm_107" if provider.startswith("sm107") else "sm_100"
    expected = nvfp4_batch_invariant_fingerprint(
        config, hidden=256, num_experts=4, arch=arch, public_inputs=True
    )
    if provider == "legacy":
        knobs = backend._workspace_knobs(fleet)
        assert knobs["mma_tiler_mnk"] == (256, 128, 256)
    else:
        actual = (
            backend._resolved_config(fleet)
            if provider.startswith("sm107")
            else backend._workspace_config(fleet)
        )
        assert actual.batch_invariant
        assert actual.input_norm_const == 2.0
        assert (
            actual.has_fc1_alpha,
            actual.has_fc2_alpha,
            actual.has_fc1_norm_const,
        ) == (True, True, True)
        assert nvfp4_batch_invariant_fingerprint(actual, arch=arch) == expected
    assert backend.batch_invariant_fingerprint(fleet) == expected
    assert backend.batch_invariant_fingerprint(_fleet(64)) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_in_kernel_fc2_reduce": True},
        {"combine_dtype": "fp32"},
        {"finalize_mode": FinalizeMode.CAKE},
        {"finalize_mode": FinalizeMode.NONE, "combine_dtype": "mxfp8"},
        {"activation": "situ", "situ_beta": -1.0},
        {"swiglu_alpha": 1.702},
        {"enable_fc1_activation_per_token_scale": 1},
        {"knobs": {"unknown_future_knob": False}},
        {"knobs": {"mma_tiler_mnk": (256, 256, 256)}},
        {"knobs": {"in_kernel_fc2_reduce": True}},
    ],
)
def test_public_rejects_unsupported_arithmetic_before_allocation(kwargs):
    with pytest.raises(ValueError):
        _public(**kwargs)


def test_rubin_rejects_cross_variant_auto_and_wrong_geometry():
    with pytest.raises(ValueError, match="kernel_variant='auto'"):
        _public("sm107-standard", kernel_variant="auto", knobs="auto")
    with pytest.raises(ValueError, match="mma_tiler_mnk"):
        _public("sm107-standard", mma_tiler_mnk=(256, 256, 256))


@pytest.mark.parametrize("cls", [BlackwellConfig, RubinConfig])
@pytest.mark.parametrize("top_k", [0, 33, True, 2.0])
def test_top_k_rejected_before_backend_allocation(cls, top_k):
    with pytest.raises(ValueError, match="top_k"):
        cls(intermediate_size=192, top_k=top_k, batch_invariant=True, knobs="auto")


@pytest.mark.parametrize("cls", [BlackwellConfig, RubinConfig])
@pytest.mark.parametrize("intermediate", [0, -64, 96, True, 192.0])
def test_intermediate_rejected_before_backend_allocation(cls, intermediate):
    with pytest.raises(ValueError, match="intermediate"):
        cls(intermediate_size=intermediate, top_k=3, batch_invariant=True, knobs="auto")


@pytest.mark.parametrize("cls", [BlackwellConfig, RubinConfig])
@pytest.mark.parametrize("norm", [1e-50, 1e39, 0.0, float("inf"), float("nan"), True])
def test_norm_rejected_in_actual_fp32_domain(cls, norm):
    with pytest.raises(ValueError, match="input_norm_const"):
        cls(intermediate_size=192, top_k=3, batch_invariant=True, input_norm_const=norm)


@pytest.mark.parametrize("cls", [BlackwellConfig, RubinConfig])
def test_routing_weight_placement_requires_bool(cls):
    with pytest.raises(ValueError, match="must be a bool"):
        cls(
            intermediate_size=192,
            top_k=3,
            batch_invariant=True,
            apply_routing_weights_before_fc2=1,
        )


@pytest.mark.parametrize(
    "provider,knobs",
    [
        ("legacy", {"flag_batch": 4}),
        ("legacy", {"epi_flag_batch": (2, 4)}),
        ("standard", {"epi_flag_batches": (1, 1)}),
        ("standard", {"token_in_flag_batch": 4}),
        ("sm107-standard", {"epi_flag_batches": (1, 1)}),
        ("sm107-standard", {"token_in_flag_batch": 4}),
    ],
)
def test_only_validated_default_flag_batches_are_admitted(provider, knobs):
    with pytest.raises(ValueError, match="does not admit"):
        _public(provider, knobs=knobs)


def test_legacy_full_fc1_width_must_describe_specified_intermediate():
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Config,
    )

    with pytest.raises(ValueError, match="post-activation intermediate"):
        MegaMoENvfp4Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=16,
            num_topk=3,
            num_total_experts=4,
            hidden=384,
            intermediate=192,
            batch_invariant=True,
            **nvfp4_batch_invariant_default_knobs("legacy"),
        )


@pytest.mark.parametrize("provider", ["legacy", "standard"])
def test_ordered_finalize_modes_share_receipt(provider):
    default = _public(provider, finalize_mode=FinalizeMode.DEFAULT)
    # Public tensor validation is independent of the arithmetic receipt. A real
    # PERSISTENT execution additionally supplies a CUDA int32 live-count tensor.
    persistent = deepcopy(default)
    persistent.finalize_mode = FinalizeMode.PERSISTENT
    assert nvfp4_batch_invariant_fingerprint(
        default, hidden=256, num_experts=4, arch="sm_100", public_inputs=True
    ) == nvfp4_batch_invariant_fingerprint(
        persistent, hidden=256, num_experts=4, arch="sm_100", public_inputs=True
    )


@pytest.mark.parametrize("carrier", ["reuse_dispatch_warps", "standalone_warps"])
def test_genphase_rejects_unsupported_carrier(carrier):
    with pytest.raises(ValueError, match="unvalidated tuning knobs|token_back_mode"):
        _public("genphase", knobs={"token_back_mode": carrier})
    with pytest.raises(ValueError, match="token_back_mode"):
        _public("sm107-genphase", token_back_mode=carrier)


def test_auto_selector_requires_same_nvfp4_numerical_class():
    a = MegaConfig(megakernel=_public())
    AutoConfig(candidates=(a, replace(a)))
    for other in (
        MegaConfig(megakernel=replace(a.megakernel, batch_invariant=False)),
        MegaConfig(megakernel=_public("standard")),
        MegaConfig(megakernel=_public(apply_routing_weights_before_fc2=False)),
        MegaConfig(megakernel=_public(gate_up_clamp=7.0)),
        MegaConfig(megakernel=_public(input_norm_const=2.0)),
    ):
        with pytest.raises(ValueError, match="cannot mix|same calculation"):
            AutoConfig(candidates=(a, other))


def test_legacy_cache_isolates_arithmetic_and_rejects_corrupted_plan(
    tmp_path, monkeypatch
):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import knob_cache

    path = tmp_path / "knobs.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    key = dict(
        dtype="nvfp4",
        world_size=1,
        hidden=256,
        intermediate=512,
        num_experts=4,
        topk=2,
        max_tokens=16,
        batch_invariant=True,
        apply_routing_weights_before_fc2=True,
    )
    knobs = nvfp4_batch_invariant_default_knobs("legacy")
    knob_cache.record_knobs(knobs, **key)
    assert knob_cache.lookup_knobs(**key) == knobs
    assert knob_cache.lookup_knobs(**{**key, "batch_invariant": False}) is None
    assert knob_cache.lookup_knobs(**{**key, "input_norm_const": 2.0}) is None
    valid = json.loads(path.read_text())
    for field, value in (
        ("batch_invariant_fingerprint", "wrong-receipt"),
        ("batch_invariant_version", NVFP4_BATCH_INVARIANT_VERSION - 1),
        ("knobs", {**valid["entries"][0]["knobs"], "in_kernel_fc2_reduce": True}),
        ("knobs", {**valid["entries"][0]["knobs"], "flag_batch": 4}),
    ):
        broken = deepcopy(valid)
        broken["entries"][0][field] = value
        path.write_text(json.dumps(broken))
        assert knob_cache.lookup_knobs(**key) is None


def _native_config(arch):
    if arch == "sm_107":
        from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
            Sm107BlockScaledMoeConfig as Config,
        )
    else:
        from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe import (
            Sm100BlockScaledMoeConfig as Config,
        )
    return Config(
        num_total_experts=4,
        max_tokens_per_rank=16,
        num_topk=2,
        hidden=256,
        intermediate=256,
        rank=0,
        world_size=1,
        quant_kind="nvfp4",
        batch_invariant=True,
        **nvfp4_batch_invariant_default_knobs("standard", arch=arch),
    )


@pytest.mark.parametrize("arch", ["sm_100", "sm_107"])
def test_native_requires_real_expert_scalar_inputs(arch):
    cfg = _native_config(arch)
    for name in ("has_fc1_alpha", "has_fc2_alpha", "has_fc1_norm_const"):
        with pytest.raises(ValueError, match=name):
            replace(cfg, **{name: False})


def test_blackwell_next_cache_contains_policy_and_receipt(tmp_path, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        knob_cache,
        tuning,
    )

    cfg = _native_config("sm_100")
    key = knob_cache._key(cfg, device="test")
    assert key["problem"]["batch_invariant_version"] == NVFP4_BATCH_INVARIANT_VERSION
    assert key["problem"][
        "batch_invariant_fingerprint"
    ] == nvfp4_batch_invariant_fingerprint(cfg, arch="sm_100")
    assert key != knob_cache._key(replace(cfg, batch_invariant=False), device="test")
    candidates = tuning.sm100_standard_candidates(cfg)
    assert candidates
    for knobs in candidates:
        assert (
            nvfp4_batch_invariant_fingerprint(replace(cfg, **knobs), arch="sm_100")
            == key["problem"]["batch_invariant_fingerprint"]
        )
    from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION

    path = tmp_path / "next-knobs.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    record = knob_cache.make_record(
        candidates[0], cfg, live_tokens=16, p50_us=1.0, device="test"
    )
    path.write_text(json.dumps(dict(version=KNOB_CACHE_VERSION, entries=[record])))
    entries = knob_cache._entries(cfg, device="test")
    assert len(entries) == 1
    assert knob_cache._decode(entries[0], cfg) == candidates[0]
    for name, bad_value in (
        ("batch_invariant_fingerprint", "wrong"),
        ("batch_invariant_version", 0),
    ):
        bad = deepcopy(record)
        bad["problem"][name] = bad_value
        path.write_text(json.dumps(dict(version=KNOB_CACHE_VERSION, entries=[bad])))
        assert knob_cache._entries(cfg, device="test") == []
    bad = deepcopy(record)
    bad["knobs"]["mma_tiler_mnk"] = [256, 256, 128]
    assert knob_cache._decode(bad, cfg) is None
    bad = deepcopy(record)
    bad["knobs"]["epi_flag_batches"] = [1, 1]
    assert knob_cache._decode(bad, cfg) is None


def test_rubin_cache_isolates_receipt_and_rejects_bad_cached_geometry():
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import knob_cache

    knobs = nvfp4_batch_invariant_default_knobs("standard", arch="sm_107")
    key = dict(
        dtype="nvfp4",
        world_size=1,
        hidden=256,
        intermediate=256,
        num_experts=4,
        topk=2,
        max_tokens=16,
        batch_invariant=True,
    )
    record = knob_cache.make_record(knobs, **key)
    assert knob_cache.lookup_knobs(entries=[record], **key) == knobs
    for field, value in (
        ("batch_invariant_fingerprint", None),
        ("batch_invariant_version", NVFP4_BATCH_INVARIANT_VERSION - 1),
        ("knobs", {**record["knobs"], "mma_tiler_mnk": [256, 256, 256]}),
        ("knobs", {**record["knobs"], "token_in_flag_batch": 4}),
    ):
        bad = {**record, field: value}
        assert knob_cache.lookup_knobs(entries=[bad], **key) is None
    assert (
        knob_cache.lookup_knobs(entries=[record], **{**key, "batch_invariant": False})
        is None
    )


def test_rubin_bi_staging_uses_logical_sf_view_of_padded_native_workspace(monkeypatch):
    """Exercise the host path using CPU storage with CUDA metadata, without a GPU."""
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl import (
        staging,
    )
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as shim

    class CudaMetadataTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    tokens, capacity, hidden, top_k = 257, 320, 384, 3
    x = torch.zeros((tokens, hidden), dtype=torch.bfloat16).as_subclass(
        CudaMetadataTensor
    )
    ids = torch.zeros((tokens, top_k), dtype=torch.int64).as_subclass(
        CudaMetadataTensor
    )
    weights = torch.zeros((tokens, top_k), dtype=torch.float32).as_subclass(
        CudaMetadataTensor
    )
    out_x = torch.empty((capacity, hidden // 2), dtype=torch.uint8)
    out_sf = torch.empty((capacity, 128), dtype=torch.float8_e4m3fn)
    out_ids = torch.empty((capacity, top_k), dtype=torch.int32)
    out_weights = torch.empty((capacity, top_k), dtype=torch.float32)
    calls = []
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "0")
    monkeypatch.setenv("FLASHINFER_SM107_MEGA_CHECK_ROUTES", "0")
    monkeypatch.setattr(
        shim, "fused_quant_stage", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(shim, "note_staged_tokens", lambda *args: None)
    assert (
        staging.stage_mega_moe_inputs(
            x,
            weights,
            ids,
            out_x,
            out_sf,
            out_ids,
            out_weights,
            input_norm_const=1.7,
            batch_invariant=True,
        )
        == tokens
    )
    assert len(calls) == 1
    logical_sf = calls[0][0][4]
    assert logical_sf.shape == (capacity, hidden // 16)
    assert logical_sf.stride() == out_sf.stride()
    assert logical_sf.data_ptr() == out_sf.data_ptr()
    assert calls[0][1]["batch_invariant"] is True
    with pytest.raises(ValueError, match="aligned contiguous"):
        staging.stage_mega_moe_inputs(
            x[:, ::2],
            weights,
            ids,
            out_x,
            out_sf,
            out_ids,
            out_weights,
            batch_invariant=True,
        )
    assert len(calls) == 1
