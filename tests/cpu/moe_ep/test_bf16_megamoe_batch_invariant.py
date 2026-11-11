"""Host-side admission and cache boundaries of batch-invariant BF16 MegaMoE."""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import pytest

from flashinfer.moe_ep import (
    BootstrapConfig,
    FinalizeMode,
    FleetParams,
    Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
    BF16_BATCH_INVARIANT_VERSION,
    bf16_batch_invariant_signature,
)
from flashinfer.moe_ep.batch_invariant.spec import (
    ALLOWED_PLAN_CONFIGS,
    ALLOWED_PLANS,
    SPEC_VERSION,
    spec_fingerprint,
)
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.bf16 import (
    MegaMoEBf16Config,
    MegaMoEBf16Frontend,
)

pytestmark = pytest.mark.cpu_only


def _config(**overrides):
    return MegaMoEBf16Config(
        **{
            "rank": 0,
            "world_size": 1,
            "num_tokens_per_rank": 257,
            "num_topk": 2,
            "num_total_experts": 2,
            "hidden": 256,
            "intermediate": 256,
            "batch_invariant": True,
            **overrides,
        }
    )


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (10, 7), (12, 0)])
def test_device_guard_scopes_the_numerical_contract(monkeypatch, capability):
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        validate_bf16_batch_invariant_device,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)

    def device_capability(device):
        assert device == 7
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", device_capability)
    if capability in ((10, 0), (10, 3), (10, 7)):
        validate_bf16_batch_invariant_device()
    else:
        with pytest.raises(ValueError, match="batch_invariant"):
            validate_bf16_batch_invariant_device()


def test_device_guard_allows_host_only_config_validation(monkeypatch):
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        validate_bf16_batch_invariant_device,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: pytest.fail("initialized CUDA on host")
    )
    validate_bf16_batch_invariant_device()


@pytest.mark.parametrize(
    "target", ("sm_107a", "sm_107f", "sm_100f", "sm_100a", "string")
)
def test_sm107_compile_target_uses_captured_native_enum(monkeypatch, target):
    import torch
    from cutlass.base_dsl import Arch
    from cutlass.cutlass_dsl import CuTeDSL

    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        validate_bf16_batch_invariant_compile_target,
    )

    try:
        Arch["sm_107a"]
        captured = "sm_107a" if target == "string" else Arch[target]
    except KeyError:
        pytest.skip("the installed CuTe DSL does not expose this native SM107 target")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_107a")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 7))
    monkeypatch.setattr(
        CuTeDSL, "_get_dsl", lambda: SimpleNamespace(get_arch_enum=lambda: captured)
    )
    if target == "sm_107a":
        validate_bf16_batch_invariant_compile_target()
    else:
        with pytest.raises(ValueError, match="captured.*sm_107a"):
            validate_bf16_batch_invariant_compile_target()


@pytest.mark.parametrize(
    "configured,target,cute_imported,accepted",
    [
        (None, "sm_107a", False, True),
        ("sm_100f", "sm_100f", False, False),
        ("sm_107f", "sm_107f", False, False),
        (None, "sm_100f", True, False),
        (None, "sm_107a", True, True),
    ],
)
def test_sm107_default_selection_preserves_explicit_and_captured_targets(
    monkeypatch, configured, target, cute_imported, accepted
):
    import os
    import sys

    import torch
    import cutlass.cute as cute
    from cutlass.base_dsl import Arch
    from cutlass.cutlass_dsl import CuTeDSL

    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        validate_bf16_batch_invariant_compile_target,
    )

    try:
        Arch["sm_107a"]
        captured = Arch[target]
    except KeyError:
        pytest.skip("the installed CuTe DSL does not expose this native SM107 target")
    if configured is None:
        monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    else:
        monkeypatch.setenv("CUTE_DSL_ARCH", configured)
    if cute_imported:
        monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    else:
        monkeypatch.delitem(sys.modules, "cutlass.cute", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 7))

    def captured_dsl():
        expected = "sm_107a" if configured is None and not cute_imported else configured
        assert os.environ.get("CUTE_DSL_ARCH") == expected
        return SimpleNamespace(get_arch_enum=lambda: captured)

    monkeypatch.setattr(CuTeDSL, "_get_dsl", captured_dsl)
    if accepted:
        validate_bf16_batch_invariant_compile_target()
    else:
        with pytest.raises(ValueError, match="captured.*sm_107a"):
            validate_bf16_batch_invariant_compile_target()


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
def test_sm107_target_check_preserves_legacy_target_behavior(monkeypatch, capability):
    import torch
    from cutlass.cutlass_dsl import CuTeDSL

    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        validate_bf16_batch_invariant_compile_target,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(
        CuTeDSL, "_get_dsl", lambda: pytest.fail("checked legacy DSL target")
    )
    validate_bf16_batch_invariant_compile_target()


def test_sm107_wrong_compile_target_fails_before_collective_allocation(monkeypatch):
    import torch
    from cutlass.base_dsl import Arch
    from cutlass.cutlass_dsl import CuTeDSL

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import bf16

    try:
        Arch["sm_107a"]
    except KeyError:
        pytest.skip("the installed CuTe DSL does not expose native SM107")
    frontend = MegaMoEBf16Frontend(_config())
    monkeypatch.setattr(frontend, "_compile_key", lambda: ())
    monkeypatch.setattr(
        frontend,
        "_release_workspace",
        lambda: pytest.fail("freed collective workspace"),
    )
    monkeypatch.setattr(
        bf16,
        "sym_zeros",
        lambda *_a, **_kw: pytest.fail("allocated collective workspace"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 7))
    monkeypatch.setattr(
        CuTeDSL, "_get_dsl", lambda: SimpleNamespace(get_arch_enum=lambda: Arch.sm_100f)
    )
    with pytest.raises(ValueError, match="captured.*sm_107a"):
        frontend._ensure_compiled(None)


def test_smem_capacity_uses_explicit_arch_or_current_device(monkeypatch):
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        bf16_batch_invariant_smem_capacity,
    )

    for arch, expected in (("sm_100", 232448), ("sm_103", 232448), ("sm_107", 334848)):
        assert bf16_batch_invariant_smem_capacity(arch) == expected
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bf16_batch_invariant_smem_capacity() == 232448
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 7))
    assert bf16_batch_invariant_smem_capacity() == 334848
    assert bf16_batch_invariant_smem_capacity("sm_100") == 232448
    for arch in ("sm_120", "sm_100f", "sm_107a", []):
        with pytest.raises(ValueError, match="architecture"):
            bf16_batch_invariant_smem_capacity(arch)


def test_sm107_resource_filter_and_receipt_share_the_same_arch(monkeypatch):
    import flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant as policy
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import autotune
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.tuner import (
        is_valid_bf16_for_config,
    )

    base = _config()
    knobs = {"mma_tiler_mnk": (128, 256, 256)}
    public = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256, top_k=2, batch_invariant=True, knobs=knobs
    )
    monkeypatch.setattr(
        autotune,
        "autotune_knobs",
        lambda _frontend, _launch, candidates, **_: candidates,
    )
    for current_arch, expected in (
        ("sm_100", False),
        ("sm_103", False),
        ("sm_107", True),
    ):
        monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: current_arch)
        assert is_valid_bf16_for_config(base, knobs) is expected
        candidates = autotune.autotune_bf16_mega_moe(
            None, None, None, SimpleNamespace(_frontend=MegaMoEBf16Frontend(base))
        )
        assert (
            any(
                candidate["mma_tiler_mnk"] == knobs["mma_tiler_mnk"]
                for candidate in candidates
            )
            is expected
        )
    # Explicit receipt arch owns admission even on a different current device.
    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_100")
    assert policy.bf16_batch_invariant_fingerprint(
        public, hidden=256, num_experts=2, arch="sm_107"
    ) == spec_fingerprint(256, 256, 2, 2, arch="sm_107")
    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_107")
    with pytest.raises(ValueError, match="shared-memory bytes"):
        policy.bf16_batch_invariant_fingerprint(
            public, hidden=256, num_experts=2, arch="sm_100"
        )


def test_cache_separates_sm107_math_even_with_identical_device_names(
    monkeypatch, tmp_path
):
    import flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant as policy
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import knob_cache

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    key = dict(
        dtype="bf16",
        world_size=1,
        hidden=256,
        intermediate=256,
        num_experts=2,
        topk=2,
        max_tokens=257,
        device="same-name",
        batch_invariant=True,
        apply_routing_weights_before_fc2=True,
    )
    original = {"mma_tiler_mnk": (256, 256, 64)}
    sm107 = {"mma_tiler_mnk": (128, 256, 256)}
    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_100")
    knob_cache.record_knobs(original, **key)
    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_107")
    assert knob_cache.lookup_knobs(**key) is None
    knob_cache.record_knobs(sm107, **key)
    assert knob_cache.lookup_knobs(**key) == sm107
    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_100")
    assert knob_cache.lookup_knobs(**key) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_in_kernel_fc2_reduce", True),
        ("in_kernel_fc2_reduce", True),
        ("use_custom_finalize", True),
        ("clc_bundle_size", 2),
        ("num_sched_stages", 2),
        ("group_hint", 123),
        ("group_hint", 512),
        ("flag_batch", 3),
        ("flag_batch", 2),
        ("epi_flag_batch", (2, 4)),
        ("enable_iket", True),
    ],
)
def test_low_level_config_rejects_unvalidated_mode(field, value):
    with pytest.raises(ValueError, match="batch_invariant"):
        _config(**{field: value})


@pytest.mark.parametrize(
    "knobs",
    [
        {"in_kernel_fc2_reduce": True, "token_back_mode": "reuse_dispatch_warps"},
        {"mma_tiler_mnk": (128, 64, 64)},
        {"use_2cta_instrs": False},
        {"force_static_sched": False},
        {"clc_bundle_size": 2},
        {"future_numerical_knob": True},
        {"apply_routing_weights_before_fc2": False},
    ],
)
def test_pinned_and_reapplied_knobs_cannot_bypass_admission(knobs):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.tuner import (
        is_valid_bf16_for_config,
        with_checked_knobs,
        with_knobs,
    )

    config = _config()
    assert not is_valid_bf16_for_config(config, knobs)
    for apply in (with_knobs, with_checked_knobs):
        with pytest.raises(ValueError):
            apply(config, knobs)
    with pytest.raises(ValueError):
        MegaMoEBf16Frontend(config).apply_knobs(knobs)
    with pytest.raises(ValueError):
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256, top_k=2, batch_invariant=True, knobs=knobs
        )


def test_signature_pins_arithmetic_and_excludes_batch_and_scheduling():
    base = _config()
    signature = bf16_batch_invariant_signature(base)
    for capacity in (1, 31, 129, 257):
        for schedule in ("static", "atomic_counter"):
            for carrier in ("epi_warps", "standalone_warps", "reuse_dispatch_warps"):
                assert (
                    bf16_batch_invariant_signature(
                        dataclasses.replace(
                            base,
                            num_tokens_per_rank=capacity,
                            load_balance_mode=schedule,
                            token_back_mode=carrier,
                        )
                    )
                    == signature
                )
    for changes in (
        {"apply_routing_weights_before_fc2": False},
        {"gate_up_clamp": 2.0},
        {"intermediate": 512},
    ):
        assert (
            bf16_batch_invariant_signature(dataclasses.replace(base, **changes))
            != signature
        )
    plain = dataclasses.replace(base, batch_invariant=False)
    assert (
        MegaMoEBf16Frontend(base)._compile_key()
        != MegaMoEBf16Frontend(plain)._compile_key()
    )


def test_geometry_variants_preserve_the_receipt_but_compile_separately():
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.autotune import (
        bf16_candidates,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.tuner import (
        is_valid_bf16_for_config,
        with_checked_knobs,
    )

    base = _config(hidden=224)
    signature = bf16_batch_invariant_signature(base)
    receipt = MegaMoEBf16Frontend(base).batch_invariant_fingerprint
    # The implementation policy changes, but the previously published math
    # specification and the SM100 receipt remain byte-for-byte unchanged.
    assert BF16_BATCH_INVARIANT_VERSION == 3
    assert SPEC_VERSION == 1
    assert spec_fingerprint(256, 256, 2, 2, arch="sm_100") == (
        "megamoe-bf16-40e431db7187f5ed"
    )
    candidates = bf16_candidates(batch_invariant=True)
    assert len(candidates) == len(ALLOWED_PLAN_CONFIGS) * len(ALLOWED_PLANS)
    assert candidates[0]["mma_tiler_mnk"] == (256, 256, 64)
    keys = set()
    admitted = 0
    for knobs in candidates:
        if not is_valid_bf16_for_config(base, knobs):
            continue
        admitted += 1
        config = with_checked_knobs(base, knobs)
        frontend = MegaMoEBf16Frontend(config)
        assert bf16_batch_invariant_signature(config) == signature
        assert frontend.batch_invariant_fingerprint == receipt
        keys.add(frontend._compile_key())
    assert len(keys) == admitted
    assert {
        candidate["mma_tiler_mnk"]
        for candidate in candidates
        if is_valid_bf16_for_config(base, candidate)
    } == {geometry.mma_tiler_mnk for geometry in ALLOWED_PLAN_CONFIGS}


@pytest.mark.parametrize("geometry", ALLOWED_PLAN_CONFIGS)
def test_geometry_overrides_derive_cta_and_cluster_consistently(geometry):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.tuner import (
        is_valid_bf16_for_config,
        with_checked_knobs,
        with_knobs,
    )

    base = _config(hidden=224)
    knobs = {"mma_tiler_mnk": geometry.mma_tiler_mnk}
    public = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256, top_k=2, batch_invariant=True, knobs=knobs
    )
    assert public.knobs == knobs
    assert is_valid_bf16_for_config(base, knobs)
    for apply in (with_knobs, with_checked_knobs):
        config = apply(base, knobs)
        for name, value in dataclasses.asdict(geometry).items():
            assert getattr(config, name) == value
    frontend = MegaMoEBf16Frontend(base)
    frontend.apply_knobs(knobs)
    assert frontend.config == with_checked_knobs(base, knobs)
    # Applying another tile must not retain a previous tile's CTA topology.
    frontend.apply_knobs({"mma_tiler_mnk": (256, 256, 64)})
    assert frontend.config == base


@pytest.mark.parametrize(
    "knobs",
    [
        {"mma_tiler_mnk": (128, 128, 64), "cluster_shape_mnk": (2, 1, 1)},
        {"mma_tiler_mnk": (128, 128, 64), "use_2cta_instrs": True},
        {"mma_tiler_mnk": (256, 128, 64), "cluster_shape_mnk": (1, 1, 1)},
        {"mma_tiler_mnk": (256, 128, 64), "use_2cta_instrs": False},
        {"mma_tiler_mnk": (128, 128, 64), "use_2cta_instrs": 0},
        {"mma_tiler_mnk": [128, 128, 64]},
    ],
)
def test_explicit_geometry_conflicts_fail_closed(knobs):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.tuner import (
        is_valid_bf16_for_config,
        with_knobs,
    )

    assert not is_valid_bf16_for_config(_config(), knobs)
    with pytest.raises(ValueError, match="batch_invariant"):
        with_knobs(_config(), knobs)
    with pytest.raises(ValueError, match="batch_invariant"):
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256, top_k=2, batch_invariant=True, knobs=knobs
        )


@pytest.mark.parametrize(
    "hidden,intermediate,stage_k,compatible",
    [
        (32, 64, 64, True),
        (96, 128, 128, True),
        (224, 256, 256, True),
        (32, 128, 128, False),
        (96, 64, 128, False),
        (96, 256, 256, False),
        (256, 192, 128, False),
    ],
)
def test_tile_staging_cannot_extend_canonical_padding(
    hidden, intermediate, stage_k, compatible
):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.tuner import (
        is_valid_bf16_for_config,
        with_checked_knobs,
    )

    base = _config(hidden=hidden, intermediate=intermediate)
    knobs = {"mma_tiler_mnk": (128, 128, stage_k)}
    assert is_valid_bf16_for_config(base, knobs) is compatible
    if compatible:
        config = with_checked_knobs(base, knobs)
        assert config.mma_tiler_mnk[2] == stage_k
    else:
        with pytest.raises(ValueError, match="rounding each to 64"):
            with_checked_knobs(base, knobs)
        with pytest.raises(ValueError, match="rounding each to 64"):
            dataclasses.replace(
                base, **knobs, cluster_shape_mnk=(1, 1, 1), use_2cta_instrs=False
            )


def test_autotune_filters_geometry_by_shape_and_keeps_k64_fallback(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import autotune

    monkeypatch.setattr(
        autotune,
        "autotune_knobs",
        lambda _frontend, _launch, candidates, **_: candidates,
    )
    for hidden, intermediate, expected_k in ((32, 64, {64}), (96, 128, {64, 128})):
        buffer = SimpleNamespace(
            _frontend=MegaMoEBf16Frontend(
                _config(hidden=hidden, intermediate=intermediate)
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidates = autotune.autotune_bf16_mega_moe(None, None, None, buffer)
        assert not caught
        assert {candidate["mma_tiler_mnk"][2] for candidate in candidates} == expected_k
        assert len(candidates) == 4 * len(expected_k) * len(ALLOWED_PLANS)
        assert all(not candidate["in_kernel_fc2_reduce"] for candidate in candidates)


@pytest.mark.parametrize(
    "hidden,intermediate,tile",
    [(32, 64, (128, 128, 128)), (256, 256, (128, 256, 256))],
)
def test_default_plan_filter_is_quiet_but_explicit_invalid_plans_warn(
    monkeypatch, hidden, intermediate, tile
):
    import flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant as policy
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import autotune

    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_100")
    monkeypatch.setattr(
        autotune,
        "autotune_knobs",
        lambda _frontend, _launch, candidates, **_: candidates,
    )
    buffer = SimpleNamespace(
        _frontend=MegaMoEBf16Frontend(_config(hidden=hidden, intermediate=intermediate))
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        defaults = autotune.autotune_bf16_mega_moe(None, None, None, buffer)
    assert not caught
    assert defaults and not any(plan["mma_tiler_mnk"] == tile for plan in defaults)
    safe, unsafe = {}, {"mma_tiler_mnk": tile}
    with pytest.warns(RuntimeWarning, match="dropped 1/2"):
        explicit = autotune.autotune_bf16_mega_moe(
            None, None, None, buffer, candidates=[unsafe, safe]
        )
    assert explicit == [safe]


def test_resource_admission_depends_on_shape_experts_and_token_back(monkeypatch):
    import flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant as policy
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        bf16_plan_smem_bytes,
    )
    from flashinfer.moe_ep.batch_invariant.spec import PlanConfig
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.tuner import (
        is_valid_bf16_for_config,
        with_checked_knobs,
    )

    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_100")
    geometry = PlanConfig((128, 256, 256), (1, 1, 1), False)
    knobs = dataclasses.asdict(geometry)
    # The same geometry fits with H=224 and fails by 48 bytes with H=256.
    assert bf16_plan_smem_bytes(224, 2, geometry) == 232240
    assert bf16_plan_smem_bytes(256, 2, geometry) == 232496
    assert is_valid_bf16_for_config(_config(hidden=224), knobs)
    assert not is_valid_bf16_for_config(_config(), knobs)
    assert not is_valid_bf16_for_config(
        _config(hidden=224), {**knobs, "token_back_mode": "standalone_warps"}
    )
    assert not is_valid_bf16_for_config(
        _config(hidden=224, num_total_experts=256), knobs
    )
    with pytest.raises(ValueError, match="shared-memory bytes"):
        with_checked_knobs(_config(), knobs)


@pytest.mark.parametrize("source", ("pinned", "cache"))
@pytest.mark.parametrize(
    "tile,cluster,use_2cta,smem_bytes",
    [
        ((256, 128, 64), (2, 1, 1), True, 214064),
        ((128, 128, 64), (1, 1, 1), False, 222256),
    ],
)
def test_factory_admits_selected_plan_when_default_exceeds_smem(
    monkeypatch, tmp_path, source, tile, cluster, use_2cta, smem_bytes
):
    import torch

    import flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant as policy
    from flashinfer.moe_ep.batch_invariant.spec import PlanConfig
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import bf16, knob_cache

    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_100")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    monkeypatch.setattr(knob_cache, "_current_device_name", lambda: "testgpu")
    capacity = policy.bf16_batch_invariant_smem_capacity("sm_100")
    assert policy.bf16_plan_smem_bytes(21504, 4, PlanConfig()) == 238640 > capacity
    assert (
        policy.bf16_plan_smem_bytes(21504, 4, PlanConfig(tile, cluster, use_2cta))
        == smem_bytes
        < capacity
    )

    knobs = {"mma_tiler_mnk": tile}
    if source == "cache":
        knob_cache.record_knobs(
            knobs,
            dtype="bf16",
            world_size=1,
            hidden=21504,
            intermediate=256,
            num_experts=4,
            topk=2,
            max_tokens=1,
            device="testgpu",
            batch_invariant=True,
            apply_routing_weights_before_fc2=True,
        )

    def fake_zeros(shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        tensor._mega_plain_alloc = True
        return tensor

    monkeypatch.setattr(bf16, "sym_zeros", fake_zeros)
    workspace = bf16.get_symm_buffer_for_bf16_mega_moe(
        4,
        1,
        2,
        21504,
        256,
        0,
        1,
        batch_invariant=True,
        knobs=knobs if source == "pinned" else None,
    )
    try:
        config = workspace._frontend.config
        assert config.mma_tiler_mnk == tile
        assert config.cluster_shape_mnk == cluster
        assert config.use_2cta_instrs is use_2cta
        assert config.batch_invariant is True
        assert workspace.x.shape == workspace.output_activation.shape == (1, 21504)
    finally:
        workspace.destroy()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"hidden": 256}, "explicit config fields"),
        ({"batch_invariant": False}, "explicit config fields"),
        ({"apply_topk_in_fc1": False}, "explicit config fields"),
        ({"future_numerical_knob": True}, "future_numerical_knob"),
        ({"use_2cta_instrs": False}, "use_2cta_instrs"),
        ({"mma_tiler_mnk": (256, 256, 64)}, "shared-memory bytes"),
    ],
)
def test_factory_rejects_invalid_selected_plan_before_allocation(
    monkeypatch, overrides, message
):
    import flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant as policy
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import bf16

    monkeypatch.setattr(policy, "bf16_batch_invariant_arch", lambda: "sm_100")
    monkeypatch.setattr(
        bf16,
        "sym_zeros",
        lambda *_a, **_kw: pytest.fail("allocated before rejecting selected knobs"),
    )
    with pytest.raises(ValueError, match=message):
        bf16.get_symm_buffer_for_bf16_mega_moe(
            4,
            1,
            2,
            21504,
            256,
            0,
            1,
            batch_invariant=True,
            knobs={"mma_tiler_mnk": (256, 128, 64), **overrides},
        )


@pytest.mark.parametrize(
    "hidden,intermediate,tile,message",
    [
        (32, 128, (128, 128, 128), "rounding each to 64"),
        (256, 64, (128, 128, 256), "rounding each to 64"),
        (256, 256, (128, 256, 256), "shared-memory bytes"),
    ],
)
def test_public_receipt_checks_pinned_plan_with_complete_dimensions(
    hidden, intermediate, tile, message
):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        bf16_batch_invariant_fingerprint,
    )

    public = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=intermediate,
        top_k=2,
        batch_invariant=True,
        knobs={"mma_tiler_mnk": tile},
    )
    # Public construction can defer checks until FleetParams supplies H/E;
    # once those values are available, metadata must reject unusable plans.
    with pytest.raises(ValueError, match=message):
        bf16_batch_invariant_fingerprint(
            public, hidden=hidden, num_experts=2, arch="sm_100"
        )


def test_public_receipt_for_valid_pinned_geometry_is_still_canonical():
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        bf16_batch_invariant_fingerprint,
    )

    public = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256,
        top_k=2,
        batch_invariant=True,
        knobs={"mma_tiler_mnk": (128, 256, 256)},
    )
    assert bf16_batch_invariant_fingerprint(
        public, hidden=224, num_experts=2, arch="sm_100"
    ) == spec_fingerprint(224, 256, 2, 2, arch="sm_100")


def test_new_geometry_cache_records_invalidate_v2_and_roundtrip_tuples(
    monkeypatch, tmp_path
):
    import json
    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import bf16, knob_cache

    cache_path = tmp_path / "knobs.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(cache_path))
    monkeypatch.setattr(knob_cache, "_current_device_name", lambda: "testgpu")
    key = dict(
        dtype="bf16",
        world_size=1,
        hidden=256,
        intermediate=256,
        num_experts=2,
        topk=2,
        max_tokens=257,
        device="testgpu",
        batch_invariant=True,
        apply_routing_weights_before_fc2=True,
    )
    knobs = {
        "mma_tiler_mnk": (128, 128, 128),
        "cluster_shape_mnk": (1, 1, 1),
        "use_2cta_instrs": False,
    }
    knob_cache.record_knobs(knobs, **key, batch_invariant_version=2)
    assert knob_cache.lookup_knobs(**key) is None
    old = json.loads(cache_path.read_text())["entries"][0]
    knob_cache.record_knobs(knobs, **key)
    assert knob_cache.lookup_knobs(**key) == knobs
    current = next(
        entry
        for entry in json.loads(cache_path.read_text())["entries"]
        if entry["batch_invariant_version"] == 3
    )
    assert old["batch_invariant_fingerprint"] == current["batch_invariant_fingerprint"]
    assert isinstance(current["knobs"]["mma_tiler_mnk"], list)

    def fake_zeros(shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        tensor._mega_plain_alloc = True
        return tensor

    monkeypatch.setattr(bf16, "sym_zeros", fake_zeros)
    workspace = bf16.get_symm_buffer_for_bf16_mega_moe(
        2, 257, 2, 256, 256, 0, 1, batch_invariant=True
    )
    try:
        assert workspace._frontend.config.mma_tiler_mnk == knobs["mma_tiler_mnk"]
        assert workspace._frontend.config.use_2cta_instrs is False
    finally:
        workspace.destroy()


def test_shape_incompatible_cached_geometry_fails_before_workspace_allocation(
    monkeypatch, tmp_path
):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import bf16, knob_cache

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    monkeypatch.setattr(knob_cache, "_current_device_name", lambda: "testgpu")
    knob_cache.record_knobs(
        {"mma_tiler_mnk": (128, 128, 128)},
        dtype="bf16",
        world_size=1,
        hidden=32,
        intermediate=64,
        num_experts=2,
        topk=2,
        max_tokens=257,
        device="testgpu",
        batch_invariant=True,
        apply_routing_weights_before_fc2=True,
    )
    monkeypatch.setattr(
        bf16,
        "sym_zeros",
        lambda *_a, **_kw: pytest.fail("allocated incompatible geometry"),
    )
    with pytest.raises(ValueError, match="unsupported BF16 MegaMoE knobs"):
        bf16.get_symm_buffer_for_bf16_mega_moe(
            2, 257, 2, 32, 64, 0, 1, batch_invariant=True
        )


@pytest.mark.parametrize(
    "options",
    [
        {"finalize_mode": FinalizeMode.NONE},
        {"enable_in_kernel_fc2_reduce": True},
        {"knobs": "unknown"},
        {"batch_invariant": 1},
        {"kernel_name": "sm100_nvfp4_nvfp4_bf16_cutedsl"},
    ],
)
def test_public_config_rejects_incompatible_contract(options):
    with pytest.raises(ValueError):
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            **{
                "intermediate_size": 256,
                "top_k": 2,
                "batch_invariant": True,
                **options,
            }
        )


def test_public_config_construction_keeps_kernel_tree_lazy():
    import subprocess
    import sys

    code = """
import sys
from flashinfer.moe_ep import Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig

Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
    intermediate_size=256, top_k=2, batch_invariant=True
)
prefix = 'flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe'
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "changes",
    [
        {"enable_in_kernel_fc2_reduce": True},
        {"knobs": {"future_numerical_knob": True}},
        {"kernel_name": "sm100_nvfp4_nvfp4_bf16_cutedsl"},
    ],
)
def test_mutated_public_config_is_rechecked_before_backend_creation(changes):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.backend import (
        Bf16CutedslMegaKernelBackend,
    )

    config = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256, top_k=2, batch_invariant=True
    )
    for name, value in changes.items():
        setattr(config, name, value)
    with pytest.raises(ValueError, match="batch_invariant"):
        Bf16CutedslMegaKernelBackend(config)


@pytest.mark.parametrize("candidate", (None, 0, 1))
@pytest.mark.parametrize(
    "other",
    (
        "ordinary",
        "split",
        "weight_placement",
        "clamp",
        "top_k",
        "intermediate",
        "unknown_backend",
    ),
)
def test_outer_selector_rejects_different_calculations_even_when_pinned(
    other, candidate
):
    from flashinfer.moe_ep import AutoConfig, MegaConfig, SplitConfig

    base = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256, top_k=2, batch_invariant=True
    )
    changes = {
        "ordinary": {"batch_invariant": False},
        "weight_placement": {"apply_routing_weights_before_fc2": False},
        "clamp": {"gate_up_clamp": 2.0},
        "top_k": {"top_k": 1},
        "intermediate": {"intermediate_size": 512},
    }
    if other == "split":
        alternative = SplitConfig()
    elif other == "unknown_backend":
        alternative = MegaConfig(
            megakernel=SimpleNamespace(
                batch_invariant=True, kernel_name="unvalidated_backend"
            )
        )
    else:
        alternative = MegaConfig(megakernel=dataclasses.replace(base, **changes[other]))
    with pytest.raises(ValueError):
        AutoConfig(
            candidates=(MegaConfig(megakernel=base), alternative), candidate=candidate
        )


@pytest.mark.parametrize("candidate", (None, 0, 1))
def test_outer_selector_accepts_schedule_variants_of_one_calculation(candidate):
    from flashinfer.moe_ep import AutoConfig, MegaConfig

    left = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256,
        top_k=2,
        batch_invariant=True,
        gate_up_clamp=2.0,
        knobs={"load_balance_mode": "static"},
    )
    right = dataclasses.replace(
        left,
        gate_up_clamp=None,
        activation_clamp=2.0,
        knobs={
            "load_balance_mode": "atomic_counter",
            "token_back_mode": "reuse_dispatch_warps",
        },
    )
    config = AutoConfig(
        candidates=(MegaConfig(megakernel=left), MegaConfig(megakernel=right)),
        candidate=candidate,
    )
    assert len(config.candidates) == 2
    assert config.candidate == candidate


def test_autotune_only_receives_admitted_candidates(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import autotune

    safe = {
        "load_balance_mode": "atomic_counter",
        "token_back_mode": "reuse_dispatch_warps",
    }
    unsafe = {"clc_bundle_size": 2}
    monkeypatch.setattr(
        autotune,
        "autotune_knobs",
        lambda _frontend, _launch, candidates, **_kwargs: candidates,
    )
    buffer = SimpleNamespace(_frontend=MegaMoEBf16Frontend(_config()))
    with pytest.warns(RuntimeWarning):
        candidates = autotune.autotune_bf16_mega_moe(
            None, None, None, buffer, candidates=[unsafe, safe]
        )
    assert candidates == [safe]
    with pytest.raises(ValueError, match="no valid BF16"):
        autotune.autotune_bf16_mega_moe(None, None, None, buffer, candidates=[unsafe])


def test_cache_separates_legacy_policy_versions_and_weight_placement(
    monkeypatch, tmp_path
):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import knob_cache

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    key = dict(
        dtype="bf16",
        world_size=1,
        hidden=256,
        intermediate=256,
        num_experts=2,
        topk=2,
        max_tokens=257,
        device="testgpu",
    )
    plain = {"token_back_mode": "epi_warps"}
    current = {"token_back_mode": "reuse_dispatch_warps"}
    previous = {"token_back_mode": "standalone_warps"}
    knob_cache.record_knobs(plain, **key)
    assert knob_cache.lookup_knobs(**key, batch_invariant=True) is None
    knob_cache.record_knobs(
        previous, **key, batch_invariant=True, batch_invariant_version=0
    )
    assert knob_cache.lookup_knobs(**key, batch_invariant=True) is None
    knob_cache.record_knobs(current, **key, batch_invariant=True)
    assert knob_cache.lookup_knobs(**key, batch_invariant=True) == current
    assert knob_cache.lookup_knobs(**key) == plain
    assert (
        knob_cache.lookup_knobs(
            **key, batch_invariant=True, apply_routing_weights_before_fc2=True
        )
        is None
    )
    knob_cache.record_knobs(
        previous, **key, batch_invariant=True, apply_routing_weights_before_fc2=True
    )
    assert knob_cache.lookup_knobs(**key, batch_invariant=True) == current
    assert (
        knob_cache.lookup_knobs(
            **key,
            batch_invariant=True,
            batch_invariant_version=BF16_BATCH_INVARIANT_VERSION,
            apply_routing_weights_before_fc2=True,
        )
        == previous
    )


def test_poisoned_cache_fails_before_allocating_or_launching(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import bf16, knob_cache

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    monkeypatch.setattr(knob_cache, "_current_device_name", lambda: "testgpu")
    knob_cache.record_knobs(
        {"future_numerical_knob": True},
        dtype="bf16",
        world_size=1,
        hidden=256,
        intermediate=256,
        num_experts=2,
        topk=2,
        max_tokens=257,
        device="testgpu",
        batch_invariant=True,
        apply_routing_weights_before_fc2=True,
    )
    monkeypatch.setattr(
        bf16,
        "sym_zeros",
        lambda *_a, **_kw: pytest.fail("allocated before rejecting poisoned cache"),
    )
    with pytest.raises(ValueError, match="future_numerical_knob"):
        bf16.get_symm_buffer_for_bf16_mega_moe(
            2, 257, 2, 256, 256, 0, 1, batch_invariant=True
        )


def test_public_backend_propagates_and_isolates_policy(monkeypatch):
    import torch

    import flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe as package
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.backend import (
        Bf16CutedslMegaKernelBackend,
    )

    fleet = FleetParams(num_experts=2, max_tokens_per_rank=257, token_hidden_size=256)
    captured = {}

    def fake_factory(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(package, "get_symm_buffer_for_bf16_mega_moe", fake_factory)
    # Pool identity only needs a stable device id; never create a CUDA context.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    public = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256,
        top_k=2,
        batch_invariant=True,
        apply_routing_weights_before_fc2=False,
        knobs={},
    )
    backend = Bf16CutedslMegaKernelBackend(public)
    backend.bind_ep_bootstrap(
        BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    backend._allocate_workspace(fleet)
    assert captured["batch_invariant"] is True
    assert captured["apply_routing_weights_before_fc2"] is False
    key = backend._tuning_key(fleet)
    assert key["batch_invariant"] is True
    assert key["batch_invariant_version"] == BF16_BATCH_INVARIANT_VERSION
    assert key["apply_routing_weights_before_fc2"] is False
    plain = Bf16CutedslMegaKernelBackend(
        dataclasses.replace(public, batch_invariant=False)
    )
    plain.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False))
    assert backend._workspace_pool_key(fleet) != plain._workspace_pool_key(fleet)
    assert backend._tuning_key(fleet) != plain._tuning_key(fleet)


def test_cache_requires_the_arithmetic_fingerprint(monkeypatch, tmp_path):
    import json
    import torch

    from flashinfer.batch_invariant import fingerprint
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import knob_cache

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "receipts.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    key = dict(
        dtype="bf16",
        world_size=1,
        hidden=256,
        intermediate=256,
        num_experts=2,
        topk=2,
        max_tokens=257,
        device="testgpu",
        batch_invariant=True,
        apply_routing_weights_before_fc2=True,
    )
    knobs = {"token_back_mode": "epi_warps"}
    knob_cache.record_knobs(knobs, **key)
    data = json.loads(path.read_text())
    expected = fingerprint(
        "megamoe_bf16",
        hidden=256,
        intermediate=256,
        top_k=2,
        num_experts=2,
        arch="sm_100",
    )
    assert data["entries"][0]["batch_invariant_fingerprint"] == expected
    assert knob_cache.lookup_knobs(**key) == knobs
    for bad in ("wrong-arithmetic", None):
        data["entries"][0]["batch_invariant_fingerprint"] = bad
        path.write_text(json.dumps(data))
        assert knob_cache.lookup_knobs(**key) is None
    del data["entries"][0]["batch_invariant_fingerprint"]
    path.write_text(json.dumps(data))
    assert knob_cache.lookup_knobs(**key) is None


def test_cache_isolates_fp32_clamp_bits(monkeypatch, tmp_path):
    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import knob_cache

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "clamps.json"))
    key = dict(
        dtype="bf16",
        world_size=1,
        hidden=256,
        intermediate=256,
        num_experts=2,
        topk=2,
        max_tokens=257,
        device="testgpu",
        batch_invariant=True,
    )
    positive = {"token_back_mode": "epi_warps"}
    negative = {"token_back_mode": "reuse_dispatch_warps"}
    knob_cache.record_knobs(positive, **key, gate_up_clamp=0.0)
    knob_cache.record_knobs(negative, **key, gate_up_clamp=-2.0)
    assert knob_cache.lookup_knobs(**key, gate_up_clamp=0.0) == positive
    assert knob_cache.lookup_knobs(**key, gate_up_clamp=-0.0) == positive
    assert knob_cache.lookup_knobs(**key, gate_up_clamp=-2.0) == negative
    assert knob_cache.lookup_knobs(**key, gate_up_clamp=2.0) == negative
    assert knob_cache.lookup_knobs(**key, gate_up_clamp=None) is None


def test_frontend_receipt_tracks_clamp_bits_but_not_schedule(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    frontend = MegaMoEBf16Frontend(_config(gate_up_clamp=0.0))
    original = frontend.batch_invariant_fingerprint
    frontend.apply_knobs({"load_balance_mode": "atomic_counter"})
    assert frontend.batch_invariant_fingerprint == original
    frontend.set_gate_up_clamp(-0.0)
    assert frontend.batch_invariant_fingerprint == original
    frontend.set_gate_up_clamp(-2.0)
    assert frontend.batch_invariant_fingerprint != original
    changed = frontend.batch_invariant_fingerprint
    frontend.set_gate_up_clamp(2.0)
    assert frontend.batch_invariant_fingerprint == changed
