"""Versioned static / dynamic cutover registry.

The dispatch resolves the routed-pair cutover of the static band in this order: environment override, registry entry
of the full key ``(quant_mode, activation, E, H, I_true, top_k, sm_count, wrapper capacity)``, density rule (1024-pair
floor, 8/16 rows per expert with a 17-row band for eligible merged-three schedules;
flat 1024 for single-slice extents). Only measured
keys are in the registry and every key part is exact: a missing part (including the wrapper capacity) or a near miss on
any part falls back to the density rule.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the SM count of the cutover key needs a CUDA device",
)

Q35 = dict(
    quant_mode="nvfp4",
    num_experts=256,
    intermediate_size=512,
    hidden_size=2048,
    activation="silu",
    num_topk=8,
    capacity_tokens=8192,
)
Q38 = dict(
    quant_mode="nvfp4",
    num_experts=512,
    intermediate_size=320,
    hidden_size=2560,
    activation="silu",
    num_topk=10,
    capacity_tokens=8192,
)
SM = md.get_num_sm(torch.device("cuda")) if torch.cuda.is_available() else 0
measured_device = pytest.mark.skipif(
    SM != 110, reason="the measured entries are for the 110-SM SM120 device"
)


@pytest.fixture(autouse=True)
def _clean_cutover_state(monkeypatch):
    for name in (
        "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
        "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
        "B12X_DYNAMIC_STATIC_CUTOVER_PAIRS",
        "B12X_LEVEL10_STATIC_CUTOVER_PAIRS",
    ):
        monkeypatch.delenv(name, raising=False)
    md._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    yield
    md._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()


def test_registry_is_keyed_by_the_full_exact_key():
    for key, pairs in md._STATIC_CUTOVER_REGISTRY.items():
        assert len(key) == 8 and isinstance(pairs, int) and pairs > 0
        quant, act, e, h, i, k, sm, capacity = key
        assert quant in ("nvfp4", "mxfp4") and isinstance(act, str)
        assert all(isinstance(v, int) for v in (e, h, i, k, sm, capacity))
        # every entry was measured at the wrapper capacity of the deployment scope
        assert capacity == 8192


@measured_device
def test_exact_key_hits_return_the_measured_entry():
    assert md._get_static_compact_cutover_pairs("fp4", **Q35) == 1792
    assert (
        md._get_static_compact_cutover_pairs("fp4", **Q38) == 8704
    )  # merged-three policy, not a model-specific registry entry


@measured_device
@pytest.mark.parametrize("capacity_tokens", [None, 0, 1024, 4096, 8191, 8193, 16384])
def test_unmeasured_or_missing_capacity_falls_back_to_the_density_rule(capacity_tokens):
    """The registry entries were measured at capacity 8192 only; the lookup is fail-closed
    on the capacity part instead of widening the measurement to a class."""
    assert (
        md._get_static_compact_cutover_pairs(
            "fp4", **{**Q35, "capacity_tokens": capacity_tokens}
        )
        == 2048
    )
    assert (
        md._static_cutover_registry_lookup(
            "nvfp4", "silu", 256, 2048, 512, 8, 110, capacity_tokens
        )
        is None
    )


@measured_device
def test_near_misses_fall_back_to_the_density_rule():
    # different hidden size (Qwen3.5-122B TP2 shares E / I / top-k with 35B): 8 rows x 256 experts
    assert (
        md._get_static_compact_cutover_pairs("fp4", **{**Q35, "hidden_size": 3072})
        == 2048
    )
    # different activation / top-k
    assert (
        md._get_static_compact_cutover_pairs("fp4", **{**Q35, "activation": "relu2"})
        == 2048
    )
    assert md._get_static_compact_cutover_pairs("fp4", **{**Q35, "num_topk": 4}) == 2048
    # another SM count is another device
    assert (
        md._static_cutover_registry_lookup(
            "nvfp4", "silu", 256, 2048, 512, 8, 148, 8192
        )
        is None
    )
    # a partial key (no hidden size / activation / capacity) never consults the registry
    assert (
        md._get_static_compact_cutover_pairs(
            "fp4", quant_mode="nvfp4", num_experts=256, intermediate_size=512
        )
        == 2048
    )
    # the generic recipe keeps its flat boundary
    assert (
        md._get_static_compact_cutover_pairs("fp4", **{**Q35, "quant_mode": "mxfp4"})
        == 640
    )


@measured_device
def test_wrappers_of_different_capacity_resolve_independently_in_one_process():
    """The resolution cache is keyed by the capacity too: a measured-capacity wrapper and
    a smaller one alternate without one result leaking into the other."""
    assert md._get_static_compact_cutover_pairs("fp4", **Q35) == 1792
    assert (
        md._get_static_compact_cutover_pairs("fp4", **{**Q35, "capacity_tokens": 4096})
        == 2048
    )  # density rule (8 padded rows x 256) at the unmeasured capacity
    assert md._get_static_compact_cutover_pairs("fp4", **Q38) == 8704
    assert (
        md._get_static_compact_cutover_pairs("fp4", **{**Q38, "capacity_tokens": 4096})
        == 8704
    )  # the merged-three schedule does not depend on a model/capacity registry hit


@pytest.mark.parametrize("intermediates", [(512, 256), (256, 512)])
def test_intermediate_extents_resolve_independently_in_one_process(intermediates):
    # Same padding/single-slice class, but only I512 has a registry entry.
    expected = {512: 1792, 256: 2048}
    for intermediate in intermediates * 2:
        assert (
            md._get_static_compact_cutover_pairs(
                "fp4", **{**Q35, "intermediate_size": intermediate}, sm_count=110
            )
            == expected[intermediate]
        )


def test_environment_override_wins_over_the_registry(monkeypatch):
    monkeypatch.setenv("FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS", "4096")
    md._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    assert md._get_static_compact_cutover_pairs("fp4", **Q35) == 4096


@measured_device
def test_backend_selection_uses_the_registry_only_with_the_full_key():
    common = dict(
        num_topk=8,
        activation_precision="fp4",
        quant_mode="nvfp4",
        num_experts=256,
        intermediate_size=512,
        hidden_size=2048,
        activation="silu",
        capacity_tokens=8192,
    )
    assert (
        md.select_sm120_moe_backend(num_tokens=224, **common) == "static"
    )  # 1792 pairs = the registry boundary (r=7)
    assert (
        md.select_sm120_moe_backend(num_tokens=225, **common) == "dynamic"
    )  # 1800 pairs: dynamic under r=7
    # the same shape at another capacity or without a capacity keeps the density rule (2048)
    assert (
        md.select_sm120_moe_backend(
            num_tokens=225, **{**common, "capacity_tokens": 4096}
        )
        == "static"
    )
    assert (
        md.select_sm120_moe_backend(
            num_tokens=225, **{**common, "capacity_tokens": None}
        )
        == "static"
    )
    q38 = dict(
        num_topk=10,
        activation_precision="fp4",
        quant_mode="nvfp4",
        num_experts=512,
        intermediate_size=320,
        hidden_size=2560,
        activation="silu",
        capacity_tokens=8192,
    )
    assert md.select_sm120_moe_backend(num_tokens=819, **q38) == "static"
    assert md.select_sm120_moe_backend(num_tokens=820, **q38) == "static"
    assert md.select_sm120_moe_backend(num_tokens=870, **q38) == "static"
    assert md.select_sm120_moe_backend(num_tokens=871, **q38) == "dynamic"
    assert (
        md.select_sm120_moe_backend(num_tokens=820, **{**q38, "capacity_tokens": 4096})
        == "static"
    )
    partial = dict(
        num_topk=8,
        activation_precision="fp4",
        quant_mode="nvfp4",
        num_experts=256,
        intermediate_size=512,
    )
    assert (
        md.select_sm120_moe_backend(num_tokens=225, **partial) == "static"
    )  # density rule (2048) without the full key


@pytest.mark.parametrize("intermediate", [272, 288, 304, 320, 352, 384])
@pytest.mark.parametrize("experts", [256, 512, 1024])
def test_merged_three_policy_is_shape_based(intermediate, experts):
    # Deliberately not the Q38 registry key; no H or capacity supplied.
    args = dict(
        quant_mode="nvfp4",
        num_experts=experts,
        intermediate_size=intermediate,
        activation="silu",
        num_topk=4,
    )
    assert md._get_static_compact_cutover_pairs(**args) == 17 * experts


@pytest.mark.parametrize("intermediate", [128, 256, 272, 320, 384, 400, 512])
@pytest.mark.parametrize("routed_pairs", [1919, 1920])
def test_merged_groups_uses_slice_count_and_routing_density(intermediate, routed_pairs):
    """Only three-slice extents at the routing-density floor use merged groups."""
    expected = 256 < intermediate <= 384 and routed_pairs >= 1920
    assert md._static_merged_groups(intermediate, routed_pairs) is expected


def test_partial_keys_do_not_alias_activation_or_slice_count():
    args = dict(
        quant_mode="nvfp4",
        num_experts=512,
        intermediate_size=320,
        activation="silu",
        num_topk=4,
    )
    for _ in range(2):
        assert md._get_static_compact_cutover_pairs(**args) == 8704
        assert (
            md._get_static_compact_cutover_pairs(**{**args, "activation": "relu2"})
            == 8192
        )
        assert md._get_static_compact_cutover_pairs(**{**args, "num_topk": 1}) == 8192
        assert (
            md._get_static_compact_cutover_pairs(**{**args, "intermediate_size": 448})
            == 8192
        )


def test_workspace_allocation_covers_the_new_boundary():
    from flashinfer import B12xMoEWrapper

    args = dict(
        num_experts=512,
        top_k=10,
        hidden_size=256,
        intermediate_size=320,
        use_cuda_graph=True,
        max_num_tokens=8192,
    )
    w = B12xMoEWrapper(**args)
    assert w._static_workspace.max_rows == 8704
    assert w._dynamic_workspace is not None
    assert (
        md.select_sm120_moe_backend(
            num_tokens=870, num_topk=w.top_k, **w._dispatch_kwargs
        )
        == "static"
    )
    assert (
        md.select_sm120_moe_backend(
            num_tokens=871, num_topk=w.top_k, **w._dispatch_kwargs
        )
        == "dynamic"
    )
