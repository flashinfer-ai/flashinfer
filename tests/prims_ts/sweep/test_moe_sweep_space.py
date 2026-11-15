# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace
import random
from types import SimpleNamespace

import pytest

from flashinfer.prims_ts.batched_gemm.config_types import (
    CtaRasterOrder,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
)
from flashinfer.prims_ts.moe.tools.sweep import space
from flashinfer.prims_ts.moe.tools.sweep.space import (
    MOVE_KINDS,
    Candidate,
    ConfigContext,
    StageLimits,
    applicable_move_kinds,
    enumerate_stage_neighbors,
    enumerate_incumbents,
    filter_legal_move_kinds,
    fixed_seed_variants,
    generate_move,
    materialize_candidate,
    repair_structural_candidate,
    warp_group_register_conflicts,
)
from flashinfer.tllm_enums import ActivationType


def _context(fc="fc1"):
    return ConfigContext(
        sm_arch=100,
        fc=fc,
        activation_type=int(ActivationType.Swiglu),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1 if fc == "fc1" else DType.BF16),
        top_k=6,
        num_local_experts=96,
        dtype_label="NVFP4xNVFP4",
    )


def test_config_context_restores_hashable_out_hiddens_after_json_roundtrip():
    payload = {**_context().__dict__, "out_hiddens": [768, 1536]}

    restored = ConfigContext(**payload)

    assert restored.out_hiddens == (768, 1536)
    hash(restored)


def test_move_applicability_matches_sweep_context():
    fc1_moves = applicable_move_kinds(_context("fc1"))
    fc2_moves = applicable_move_kinds(_context("fc2"))

    assert {"warps", "route_warps"} <= set(fc1_moves)
    assert {"warps", "route_warps"}.isdisjoint(fc2_moves)
    assert "large_carveout" not in fc1_moves

    sm107_context = ConfigContext(**{**_context().__dict__, "sm_arch": 107})
    assert "large_carveout" in applicable_move_kinds(sm107_context)


@pytest.mark.parametrize(
    ("num_tokens", "expected_applicable"),
    ((2048, True), (3072, False)),
)
def test_along_n_grid_limit_applicability(num_tokens, expected_applicable):
    candidate = Candidate(options={}, tile_n=32)
    materialized = SimpleNamespace(
        candidate=candidate,
        kwargs={
            "tile_n": 32,
            "mma_n": 32,
            "tile_scheduler": int(TileScheduler.PERSISTENT),
            "cta_raster_order": int(CtaRasterOrder.ALONG_N),
            "use_early_exit": 1,
        },
    )
    shape = SimpleNamespace(
        selected_tile_ns=(32,),
        fc2_n=3584,
        timing_num_tokens=num_tokens,
        local_num_experts=896,
        top_k=16,
    )

    applicable, reason = space.is_applicable(
        materialized,
        _context("fc2"),
        shape,
    )

    assert applicable is expected_applicable
    if expected_applicable:
        assert reason == ""
    else:
        assert reason.startswith("grid_limit: ")
        assert "gridDim.z=67312" in reason


def test_register_harmonization_repairs_loader_group_conflict():
    context = _context()
    parent = enumerate_incumbents(context, (128,))[0]
    bad = None
    for field in (
        "load_a_regs",
        "load_b_regs",
        "load_sfa_regs",
        "load_sfb_regs",
    ):
        candidate = Candidate(
            options={**parent.options, field: 232},
            tile_n=parent.tile_n,
            move_kind="skeleton",
            parent_ids=(parent.candidate_id,),
        )
        cfg = space.make_config(**space.candidate_kwargs(candidate, context))
        if warp_group_register_conflicts(cfg):
            bad = candidate
            break

    assert bad is not None
    repaired_options = space._harmonize_register_options(bad, context)
    repaired = Candidate(
        options=repaired_options,
        tile_n=bad.tile_n,
        move_kind=bad.move_kind,
        parent_ids=bad.parent_ids,
    )
    repaired_cfg = space.make_config(**space.candidate_kwargs(repaired, context))
    assert warp_group_register_conflicts(repaired_cfg) == []


def test_cluster_move_prunes_unsupported_four_wide_swap_ab_shapes(monkeypatch):
    context = ConfigContext(
        **{
            **_context().__dict__,
            "sm_arch": 107,
            "out_hiddens": (2304, 1152),
        }
    )
    parent = enumerate_incumbents(context, (128,))[0]
    parent = replace(
        parent,
        options={
            **parent.options,
            "umma_cta_count": 1,
            "cluster_dim_x": 1,
            "cluster_dim_y": 1,
            "fallback_cluster_dim_x": 0,
            "fallback_cluster_dim_y": 1,
            "mma_m": 128,
        },
    )
    monkeypatch.setattr(
        space,
        "_supports_any_out_hidden",
        lambda candidate, _context: int(candidate.options["cluster_dim_x"]) != 4,
    )
    monkeypatch.setattr(
        space,
        "_ensure_config_task_manager_buildable",
        lambda *_args, **_kwargs: None,
    )

    child = generate_move(
        parent,
        "cluster",
        random.Random(7),
        context=context,
    )

    assert child.options["cluster_dim_x"] == 2


@pytest.mark.parametrize(
    ("move_kind", "probe", "reason"),
    (
        (
            "cluster",
            SimpleNamespace(
                mma_n=8,
                use_tile256_tmem_overlap=False,
            ),
            "cluster requires 2-CTA MMA",
        ),
        (
            "tile_k",
            SimpleNamespace(
                mma_n=256,
                use_tile256_tmem_overlap=True,
                is_mx_mma=False,
                tile_k=256,
            ),
            "no supported tile_k alternative",
        ),
        (
            "tile_k_stages",
            SimpleNamespace(
                mma_n=256,
                use_tile256_tmem_overlap=True,
                is_mx_mma=False,
                tile_k=256,
            ),
            "no supported tile_k alternative",
        ),
        (
            "tmem_stages",
            SimpleNamespace(
                mma_n=256,
                use_tile256_tmem_overlap=True,
            ),
            "fixed accumulator and fused scale-factor staging",
        ),
    ),
)
def test_parent_unsupported_moves_are_not_applicable(
    monkeypatch, move_kind, probe, reason
):
    parent = Candidate(options={}, tile_n=probe.mma_n)
    monkeypatch.setattr(space, "candidate_kwargs", lambda *_args: {})
    monkeypatch.setattr(space, "make_config", lambda **_kwargs: probe)

    assert move_kind not in applicable_move_kinds(
        _context(), (move_kind,), parent=parent
    )
    if move_kind == "tmem_stages":
        assert reason in space._unsupported_move_reason(probe, move_kind)
        return
    with pytest.raises(ValueError, match=reason):
        generate_move(
            parent,
            move_kind,
            random.Random(0),
            context=_context(),
        )


@pytest.mark.parametrize("move_kind", ("warps", "route_warps"))
def test_routing_move_rejects_fc_without_routing(move_kind):
    with pytest.raises(
        ValueError,
        match=rf"{move_kind} only varies routed-operand loading; fc2 has no routing",
    ):
        generate_move(
            Candidate(options={}, tile_n=128),
            move_kind,
            random.Random(0),
            context=_context("fc2"),
        )


def test_enumerates_individual_fc_rows_and_round_trips_signatures():
    context = _context()
    candidates = enumerate_incumbents(context, (128,))
    assert candidates
    assert all(candidate.tile_n == 128 for candidate in candidates)

    materialized = materialize_candidate(
        candidates[0],
        context,
        (7168, 7296),
        check_task_manager=False,
    )
    assert set(materialized.resolved_hashes) == {7168, 7296}
    assert materialized.source_hash
    assert materialized.tmem_columns > 0


def test_moves_are_deterministic_and_keep_mapper_rows():
    parent = enumerate_incumbents(_context(), (128,))[0]
    left = generate_move(parent, "flags", random.Random(7))
    right = generate_move(parent, "flags", random.Random(7))
    assert left.candidate_id == right.candidate_id
    assert left.parent_ids == (parent.candidate_id,)
    assert left.move_kind == "flags"


def test_skeleton_move_keeps_block_scaled_sf_routing_enabled():
    context = _context()
    parent = enumerate_incumbents(context, (128,))[0]
    options = dict(parent.options)
    options["route_act"] = "tma"
    options.pop("route_sfs_act", None)
    implicit = replace(parent, options=options)

    child = generate_move(
        implicit,
        "skeleton",
        _ScriptedRandom("route_act", "ldgsts"),
        context=context,
    )

    assert child.options["route_act"] == "ldgsts"
    assert child.options["route_sfs_act"] == "ldgsts"
    assert space.candidate_kwargs(child, context)["route_sfs_act"] == int(
        RouteImpl.LDGSTS
    )


def test_crossover_keeps_block_scaled_sf_routing_enabled():
    context = _context()
    incumbents = enumerate_incumbents(context, (128,))
    parent_options = dict(incumbents[0].options)
    parent_options["route_act"] = "tma"
    parent_options.pop("route_sfs_act", None)
    parent = replace(incumbents[0], options=parent_options)
    other_options = dict(incumbents[-1].options)
    other_options["route_act"] = "ldgsts"
    other_options.pop("route_sfs_act", None)
    other = replace(incumbents[-1], options=other_options)

    child = generate_move(
        parent,
        "crossover",
        random.Random(0),
        other_parent=other,
        context=context,
    )

    assert child.options["route_act"] == "ldgsts"
    assert child.options["route_sfs_act"] == "ldgsts"


@pytest.mark.parametrize("disabled", (False, "none"))
def test_explicit_sf_routing_opt_out_is_preserved(disabled):
    context = _context()
    parent = enumerate_incumbents(context, (128,))[0]
    options = {
        **parent.options,
        "route_act": "ldgsts",
        "route_sfs_act": disabled,
        "sf_layout_b": "128x4",
    }
    opted_out = replace(parent, options=options)

    kwargs = space.candidate_kwargs(opted_out, context)
    materialized = materialize_candidate(
        opted_out,
        context,
        (7168,),
        check_task_manager=False,
    )

    assert kwargs["route_sfs_act"] == int(RouteImpl.NONE)
    assert opted_out.options["route_sfs_act"] == disabled
    assert materialized.candidate.options["route_sfs_act"] == disabled


def test_epilogue_and_warp_moves_stay_structurally_legal(monkeypatch):
    context = _context()
    parent = enumerate_incumbents(context, (128,))[0]
    task_manager_probes = []
    monkeypatch.setattr(
        space,
        "_ensure_config_task_manager_buildable",
        lambda *args, **kwargs: task_manager_probes.append((args, kwargs)),
    )
    epilogue = generate_move(
        parent,
        "epilogue",
        _ScriptedRandom(32, random_value=0.0),
        context=context,
    )
    assert epilogue.tile_n % int(epilogue.options["epi_tile_n"]) == 0

    for seed in range(20):
        child = generate_move(
            parent,
            "warps",
            random.Random(seed),
            context=context,
        )
        materialize_candidate(
            child,
            context,
            (7168,),
            check_task_manager=False,
        )
    assert task_manager_probes


def test_warp_move_uses_first_task_manager_buildable_choice(monkeypatch):
    parent = Candidate(
        options={"num_load_a_warps": 1},
        tile_n=128,
    )
    probe = SimpleNamespace(
        is_swap_ab=False,
        has_routed_sfs=True,
        has_routed_act=True,
        fuse_operand_sf_loads=False,
    )
    attempted = []

    monkeypatch.setattr(
        space,
        "candidate_kwargs",
        lambda candidate, _context: dict(candidate.options),
    )

    def make_probe(**kwargs):
        return SimpleNamespace(
            **probe.__dict__,
            compile_signature=lambda: kwargs,
        )

    monkeypatch.setattr(space, "make_config", make_probe)
    monkeypatch.setattr(space, "compute_warp_layout", lambda _cfg: None)
    monkeypatch.setattr(space, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(
        space,
        "_normalize_candidate_sf_routing",
        lambda candidate, _context: candidate,
    )

    def ensure_buildable(spec, **_kwargs):
        warps = int(spec.kwargs["num_load_a_warps"])
        attempted.append(warps)
        if warps == 1:
            raise ValueError("register group mismatch")

    monkeypatch.setattr(
        space,
        "_ensure_config_task_manager_buildable",
        ensure_buildable,
    )

    child = generate_move(
        parent,
        "warps",
        _ScriptedRandom("num_load_a_warps"),
        context=_context(),
    )

    assert attempted == [2]
    assert child.options["num_load_a_warps"] == 2


def test_flags_move_skips_task_manager_unbuildable_choices(monkeypatch):
    context = _context()
    parent = enumerate_incumbents(context, (16,))[0]
    target_unroll = not bool(parent.options.get("use_unroll_loop_2x_for_mma", False))
    attempted = []

    def ensure_buildable(spec, **_kwargs):
        attempted.append(dict(spec.kwargs))
        if len(attempted) == 1:
            raise ValueError("SMEM usage exceeds hardware capacity")

    monkeypatch.setattr(
        space,
        "_ensure_config_task_manager_buildable",
        ensure_buildable,
    )

    child = generate_move(
        parent,
        "flags",
        _PreferFieldRandom(
            "use_unroll_loop_2x_for_mma",
            parent.options,
            preferred=target_unroll,
        ),
        context=context,
    )

    assert len(attempted) > 1
    assert bool(attempted[0]["use_unroll_loop_2x_for_mma"]) == target_unroll
    assert child.candidate_id != parent.candidate_id


def test_flags_move_rejects_when_task_manager_rejects_every_choice(monkeypatch):
    context = _context()
    parent = enumerate_incumbents(context, (16,))[0]

    def reject_all(_spec, **_kwargs):
        raise ValueError("SMEM usage exceeds hardware capacity")

    monkeypatch.setattr(
        space,
        "_ensure_config_task_manager_buildable",
        reject_all,
    )

    with pytest.raises(ValueError, match="no effective flags mutation"):
        generate_move(parent, "flags", random.Random(7), context=context)


def test_fc2_rejects_rows_that_cannot_match_production_mapper():
    context = _context("fc2")
    parent = enumerate_incumbents(context, (128,))[0]
    invalid = replace(
        parent,
        options={**parent.options, "route_act": "tma"},
    )

    with pytest.raises(ValueError, match="production MoE mapper"):
        materialize_candidate(
            invalid,
            context,
            (3072,),
            check_task_manager=False,
        )


def test_smem_headroom_rejects_generated_children_but_not_incumbents(monkeypatch):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_kernel

    context = _context()
    incumbent = enumerate_incumbents(context, (128,))[0]
    child = generate_move(incumbent, "flags", random.Random(7))
    capacity = (228 - 1) * 1024
    usage = capacity - space.SWEEP_SMEM_HEADROOM_BYTES + 1
    monkeypatch.setattr(
        batched_gemm_kernel,
        "build_batched_gemm_task_manager",
        lambda **kwargs: SimpleNamespace(
            _smem_allocator=SimpleNamespace(
                total_smem_bytes=usage, barrier_smem_bytes=0
            )
        ),
    )

    assert materialize_candidate(incumbent, context, (7168,)).smem_bytes == usage
    with pytest.raises(ValueError, match="sweep limit"):
        materialize_candidate(child, context, (7168,))


class _ScriptedRandom:
    def __init__(self, *choices, random_value=0.0):
        self.choices = list(choices)
        self.random_value = random_value

    def choice(self, values):
        value = self.choices.pop(0)
        assert value in values
        return value

    def sample(self, values, count):
        return list(values)[:count]

    def random(self):
        return self.random_value

    def shuffle(self, values):
        pass


class _PreferFieldRandom:
    def __init__(self, field, parent_options, *, preferred=None):
        self.field = field
        self.parent_options = parent_options
        self.preferred = preferred

    def choice(self, values):
        return self.field if self.field in values else values[0]

    def random(self):
        return 0.0

    def shuffle(self, values):
        if self.preferred is not None:
            values.sort(key=lambda trial: trial.get(self.field) != self.preferred)
        else:
            values.sort(
                key=lambda trial: (
                    trial.get(self.field) == self.parent_options.get(self.field)
                )
            )


def test_stage_groups_link_resources_that_must_move_together():
    options = {
        "umma_cta_count": 2,
        "fuse_operand_sf_loads": True,
        "num_stages_a": 2,
        "num_stages_b": 2,
        "num_stages_smem_sfa": 1,
        "num_stages_tmem_sfa": 2,
        "num_stages_tmem_sfb": 2,
    }

    assert ("num_stages_a", "num_stages_b") in space._stage_groups(options)
    assert (
        "num_stages_tmem_sfa",
        "num_stages_tmem_sfb",
    ) in space._stage_groups(options, tmem=True)


def test_stage_neighbor_enumeration_is_finite_legal_and_distinct():
    parent = Candidate(
        options={
            "num_stages_a": 2,
            "num_stages_b": 2,
            "num_stages_smem_sfa": 1,
        },
        tile_n=128,
    )

    limits = StageLimits(inputs=3, accumulator=3, c_smem=3, workid=3)
    one = enumerate_stage_neighbors(parent, "stages_one", stage_limits=limits)
    trade = enumerate_stage_neighbors(parent, "stages_trade", stage_limits=limits)
    tmem_parent = Candidate(
        options={
            "num_stages_tmem_acc": 2,
            "num_stages_tmem_sfa": 1,
        },
        tile_n=128,
    )
    tmem = enumerate_stage_neighbors(tmem_parent, "tmem_stages", stage_limits=limits)

    assert one
    assert trade
    assert any(
        candidate.repair_kind
        and candidate.repair_kind.startswith("tmem_neighbor_trade:")
        for candidate in tmem
    )
    assert len({candidate.candidate_id for candidate in one}) == len(one)
    assert all(candidate.candidate_id != parent.candidate_id for candidate in one)
    assert any(
        candidate.options["num_stages_a"] == 1
        and candidate.options["num_stages_smem_sfa"] == 2
        for candidate in trade
    )


def test_stage_limits_bound_each_resource_class_and_inherited_options():
    limits = StageLimits(inputs=16, accumulator=10, c_smem=4, workid=8)
    finalized = space._finalize_options(
        {
            "num_stages_a": 32,
            "num_stages_smem_sfa": 32,
            "num_stages_tmem_sfa": 32,
            "num_stages_tmem_acc": 32,
            "num_stages_c_smem": 32,
            "num_stages_workid": 32,
        },
        stage_limits=limits,
        max_sf_ring_ratio=4,
    )

    assert finalized["num_stages_a"] == 16
    assert finalized["num_stages_smem_sfa"] == 16
    assert finalized["num_stages_tmem_sfa"] == 16
    assert finalized["num_stages_tmem_acc"] == 10
    assert finalized["num_stages_c_smem"] == 4
    assert finalized["num_stages_workid"] == 8

    parent = Candidate(
        options={
            "num_stages_tmem_acc": 10,
            "num_stages_tmem_sfa": 16,
            "num_stages_tmem_sfb": 15,
        },
        tile_n=128,
    )
    neighbors = enumerate_stage_neighbors(
        parent,
        "tmem_stages",
        stage_limits=limits,
    )
    assert neighbors
    assert all(
        int(candidate.options["num_stages_tmem_acc"]) <= 10
        and int(candidate.options["num_stages_tmem_sfa"]) <= 16
        and int(candidate.options["num_stages_tmem_sfb"]) <= 16
        for candidate in neighbors
    )
    epilogue = generate_move(
        Candidate(
            options={
                "tile_n": 8,
                "epi_tile_n": 8,
                "num_stages_c_smem": 4,
            },
            tile_n=8,
        ),
        "epilogue",
        random.Random(0),
        stage_limits=limits,
    )
    assert epilogue.options["num_stages_c_smem"] == 3


def test_move_legality_filter_rejects_exhausted_stage_moves():
    parent = Candidate(
        options={"num_stages_a": 1},
        tile_n=128,
    )
    legality = filter_legal_move_kinds(
        parent,
        ("stages_all", "stages_trade", "register_split"),
        stage_limits=StageLimits(inputs=1, accumulator=1, c_smem=1, workid=1),
    )

    assert legality.legal == ("register_split",)
    assert legality.rejected == {
        "stages_all": "no_legal_neighbor",
        "stages_trade": "no_legal_neighbor",
    }


def test_flags_searches_scheduler_and_fixed_seeds_cover_ring_fills():
    parent = Candidate(
        options={
            "tile_scheduler": "static",
            "umma_cta_count": 2,
            "num_stages_a": 2,
            "num_stages_b": 2,
        },
        tile_n=128,
    )
    child = generate_move(
        parent,
        "flags",
        _PreferFieldRandom("tile_scheduler", parent.options),
    )
    assert child.options["tile_scheduler"] == "persistent"

    persistent = Candidate(
        options={
            **parent.options,
            "tile_scheduler": "persistent",
            "use_clc_fast_drain": True,
            "cta_raster_order": 1,
        },
        tile_n=128,
    )
    static = generate_move(
        persistent,
        "flags",
        _PreferFieldRandom("tile_scheduler", persistent.options),
    )
    assert static.options["tile_scheduler"] == "static"
    assert static.options["use_clc_fast_drain"] is False
    assert static.options["cta_raster_order"] == 0
    fast_drain = generate_move(
        parent,
        "flags",
        _PreferFieldRandom(
            "use_clc_fast_drain",
            parent.options,
            preferred=True,
        ),
    )
    assert fast_drain.options["use_clc_fast_drain"] is True
    assert fast_drain.options["tile_scheduler"] == "persistent"
    crossed = generate_move(
        persistent,
        "crossover",
        random.Random(0),
        other_parent=parent,
    )
    assert crossed.options["tile_scheduler"] == "static"
    assert crossed.options["use_clc_fast_drain"] is False
    assert crossed.options["cta_raster_order"] == 0

    seeds = fixed_seed_variants([parent], limit=1)
    repairs = {seed.repair_kind for seed in seeds}
    assert "stages_all:max" in repairs
    assert "stages_all_then_one:num_stages_a,num_stages_b:fill" in repairs
    direct_store_max_seeds = [
        seed
        for seed in seeds
        if seed.repair_kind == "stages_all:max"
        and seed.options.get("use_tma_store") is False
    ]
    assert len(direct_store_max_seeds) == 1
    sm107_seeds = fixed_seed_variants([parent], limit=1, sm_arch=107)
    assert any(seed.move_kind == "large_carveout" for seed in sm107_seeds)
    tmem_seeds = [
        seed for seed in sm107_seeds if seed.repair_kind.endswith(":tmem_fill")
    ]
    assert len(tmem_seeds) * 2 == len(sm107_seeds)


def test_repair_honors_stage_bounds_and_sm107_carveout(monkeypatch):
    def fake_materialize(candidate, _context, _ks, *, check_task_manager):
        assert check_task_manager
        stages = sum(
            int(value)
            for field, value in candidate.options.items()
            if field.startswith("num_stages_") and "tmem" not in field
        )
        return SimpleNamespace(
            smem_bytes=stages * 100 * 1024,
            smem_regime_kib=328 if stages >= 3 else 228,
        )

    monkeypatch.setattr(space, "materialize_candidate", fake_materialize)
    base = Candidate(
        options={"num_stages_a": 1},
        tile_n=128,
        move_kind="large_carveout",
    )
    sm100 = repair_structural_candidate(
        base,
        _context(),
        (7168,),
        repair_index=0,
        stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
    )
    sm107 = repair_structural_candidate(
        base,
        ConfigContext(**{**_context().__dict__, "sm_arch": 107}),
        (7168,),
        repair_index=0,
        stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
    )
    sm107_small = repair_structural_candidate(
        base,
        ConfigContext(**{**_context().__dict__, "sm_arch": 107}),
        (7168,),
        repair_index=1,
        stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
    )
    assert sm100.options["num_stages_a"] == 2
    assert sm107.options["num_stages_a"] == 3
    assert sm107_small.options["num_stages_a"] == 2


def test_structural_repair_can_fill_each_tmem_budget_strategy(monkeypatch):
    def fake_materialize(candidate, _context, _ks, *, check_task_manager):
        assert check_task_manager
        tmem_columns = sum(
            int(value)
            for field, value in candidate.options.items()
            if field.startswith("num_stages_tmem_")
        )
        if tmem_columns > 6:
            raise ValueError("TMEM capacity")
        return SimpleNamespace(
            smem_bytes=1024,
            smem_regime_kib=227,
            tmem_columns=tmem_columns,
        )

    monkeypatch.setattr(space, "materialize_candidate", fake_materialize)
    base = Candidate(
        options={
            "num_stages_a": 1,
            "num_stages_tmem_acc": 1,
            "num_stages_tmem_sfa": 1,
            "num_stages_tmem_sfb": 1,
        },
        tile_n=128,
        move_kind="cluster",
    )

    no_fill = repair_structural_candidate(
        base,
        _context(),
        (7168,),
        repair_index=0,
        stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
        fill_tmem=False,
    )
    accumulator = repair_structural_candidate(
        base,
        _context(),
        (7168,),
        repair_index=0,
        stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
        fill_tmem=True,
    )
    capped_accumulator = repair_structural_candidate(
        base,
        _context(),
        (7168,),
        repair_index=0,
        stage_limits=StageLimits(inputs=8, accumulator=2, c_smem=8, workid=8),
        fill_tmem=True,
    )
    scale_factors = repair_structural_candidate(
        base,
        _context(),
        (7168,),
        repair_index=1,
        stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
        fill_tmem=True,
    )
    all_rings = repair_structural_candidate(
        base,
        _context(),
        (7168,),
        repair_index=2,
        stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
        fill_tmem=True,
    )

    assert no_fill.options["num_stages_tmem_acc"] == 1
    assert accumulator.options["num_stages_tmem_acc"] == 4
    assert capped_accumulator.options["num_stages_tmem_acc"] == 2
    assert scale_factors.options["num_stages_tmem_sfa"] == 2
    assert scale_factors.options["num_stages_tmem_sfb"] == 2
    assert all_rings.options["num_stages_tmem_acc"] == 2
    assert all_rings.options["num_stages_tmem_sfa"] == 2
    assert all_rings.options["num_stages_tmem_sfb"] == 2
    assert accumulator.repair_kind.endswith(":tmem_fill")


@pytest.mark.parametrize(
    ("move_kind", "repair_kind", "options"),
    (
        (
            "tile_k_stages",
            "tile_k_stages:inverse",
            {"tile_k": 256, "num_stages_a": 3},
        ),
    ),
)
def test_compound_repair_preserves_concrete_stage_neighbor(
    monkeypatch, move_kind, repair_kind, options
):
    monkeypatch.setattr(
        space,
        "materialize_candidate",
        lambda *_args, **_kwargs: SimpleNamespace(
            smem_bytes=1024,
            smem_regime_kib=227,
            tmem_columns=4,
        ),
    )
    candidate = Candidate(
        options=options,
        tile_n=128,
        move_kind=move_kind,
        repair_kind=repair_kind,
    )

    repaired = repair_structural_candidate(
        candidate,
        _context(),
        (7168,),
        repair_index=0,
        stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
    )

    assert repaired.options == candidate.options


def test_compound_repair_rejects_instead_of_breaking_coupling(monkeypatch):
    attempts = []

    def reject(candidate, *_args, **_kwargs):
        attempts.append(dict(candidate.options))
        raise ValueError("resource budget")

    monkeypatch.setattr(space, "materialize_candidate", reject)
    candidate = Candidate(
        options={"tile_k": 256, "num_stages_a": 4},
        tile_n=128,
        move_kind="tile_k_stages",
        repair_kind="tile_k_stages:inverse",
    )

    with pytest.raises(ValueError, match="resource budget"):
        repair_structural_candidate(
            candidate,
            _context(),
            (7168,),
            repair_index=0,
            stage_limits=StageLimits(inputs=8, accumulator=8, c_smem=8, workid=8),
        )

    assert attempts == [candidate.options]


def test_moves_that_toggle_umma_cta_count_keep_mma_m_legal():
    single = Candidate(
        options={"umma_cta_count": 1, "mma_m": 128, "num_stages_a": 2},
        tile_n=32,
    )
    paired = Candidate(
        options={"umma_cta_count": 2, "mma_m": 256, "num_stages_a": 2},
        tile_n=32,
    )

    to_paired = generate_move(
        single,
        "skeleton",
        _PreferFieldRandom("umma_cta_count", single.options),
    )
    assert to_paired.options["umma_cta_count"] == 2
    assert to_paired.options["mma_m"] == 256
    to_single = generate_move(
        paired,
        "skeleton",
        _PreferFieldRandom("umma_cta_count", paired.options),
    )
    assert to_single.options["umma_cta_count"] == 1
    assert to_single.options["mma_m"] == 128

    clustered = generate_move(single, "cluster", _ScriptedRandom(2, 0))
    assert clustered.options["mma_m"] == 256
    token_clustered = [
        generate_move(
            single,
            "cluster",
            random.Random(seed),
            enable_cluster_y=True,
        )
        for seed in range(32)
    ]
    assert any(
        int(candidate.options.get("cluster_dim_y", 1)) > 1
        for candidate in token_clustered
    )
    x_only_clustered = [
        generate_move(single, "cluster", random.Random(seed)) for seed in range(16)
    ]
    assert all(
        int(candidate.options.get("cluster_dim_y", 1)) == 1
        for candidate in x_only_clustered
    )
    crossed = generate_move(single, "crossover", random.Random(0), other_parent=paired)
    assert crossed.options["mma_m"] == 256
    assert "random_restart" not in MOVE_KINDS


def test_discrete_moves_change_the_effective_config(monkeypatch):
    context = _context()
    incumbents = enumerate_incumbents(context, (128,))
    parent = incumbents[0]
    other = next(
        (
            candidate
            for candidate in incumbents[1:]
            if space.structural_options_key(candidate)
            != space.structural_options_key(parent)
        ),
        None,
    )
    monkeypatch.setattr(
        space,
        "_ensure_config_task_manager_buildable",
        lambda *args, **kwargs: None,
    )
    parent_key = space._effective_key(parent.options, parent, context)

    for move_kind in (
        "tile_k",
        "epilogue",
        "warps",
        "flags",
        "skeleton",
        "cluster",
        "crossover",
    ):
        if move_kind == "crossover" and other is None:
            continue
        kwargs = {"other_parent": other} if move_kind == "crossover" else {}
        for seed in range(30):
            try:
                child = generate_move(
                    parent,
                    move_kind,
                    random.Random(seed),
                    context=context,
                    **kwargs,
                )
            except ValueError:
                continue
            assert space._effective_key(child.options, parent, context) != parent_key


def test_nvfp4_tile_k_move_skips_illegal_half_tile():
    context = _context()
    parent = enumerate_incumbents(context, (128,))[0]
    parent = replace(
        parent,
        options={**parent.options, "tile_k": 256},
    )

    child = generate_move(
        parent,
        "tile_k",
        _ScriptedRandom(),
        context=context,
    )

    assert child.options["tile_k"] == 512


def test_tile_k_move_rescales_live_tmem_sf_stages():
    parent = Candidate(
        options={
            "tile_k": 256,
            "num_stages_tmem_sfa": 3,
            "num_stages_tmem_sfb": 5,
        },
        tile_n=128,
    )

    child = generate_move(
        parent,
        "tile_k",
        _PreferFieldRandom(
            "num_stages_tmem_sfa",
            parent.options,
            preferred=2,
        ),
    )

    assert child.options["tile_k"] == 512
    assert child.options["num_stages_tmem_sfa"] == 2
    assert child.options["num_stages_tmem_sfb"] == 3


def test_tile_k_stages_rounds_odd_stage_counts():
    parent = Candidate(
        options={
            "tile_k": 256,
            "num_stages_a": 3,
            "num_stages_b": 5,
            "num_stages_tmem_sfa": 3,
        },
        tile_n=128,
    )

    child = generate_move(
        parent,
        "tile_k_stages",
        _PreferFieldRandom("tile_k", parent.options, preferred=512),
    )

    assert child.options["tile_k"] == 512
    assert child.options["num_stages_a"] in (1, 2)
    assert child.options["num_stages_b"] in (2, 3)
    assert child.options["num_stages_tmem_sfa"] in (1, 2)


def test_epilogue_move_never_redraws_current_value(monkeypatch):
    context = _context()
    parent = enumerate_incumbents(context, (128,))[0]
    monkeypatch.setattr(
        space,
        "_ensure_config_task_manager_buildable",
        lambda *args, **kwargs: None,
    )

    for seed in range(20):
        child = generate_move(
            parent,
            "epilogue",
            random.Random(seed),
            context=context,
        )
        changed_epilogue_tile = child.options.get("epi_tile_n") != parent.options.get(
            "epi_tile_n"
        )
        changed_stages = child.options.get("num_stages_c_smem") != parent.options.get(
            "num_stages_c_smem"
        )
        changed_store_path = child.options.get(
            "use_tma_store", True
        ) != parent.options.get("use_tma_store", True)
        assert changed_epilogue_tile or changed_stages or changed_store_path


def test_epilogue_move_toggles_tma_store_and_restores_c_smem_stages(
    monkeypatch,
):
    context = _context("fc2")
    parent = enumerate_incumbents(context, (128,))[0]
    original_stages = int(parent.options["num_stages_c_smem"])
    monkeypatch.setattr(
        space,
        "_ensure_config_task_manager_buildable",
        lambda *args, **kwargs: None,
    )

    direct_store = generate_move(
        parent,
        "epilogue",
        _PreferFieldRandom(
            "use_tma_store",
            parent.options,
            preferred=False,
        ),
        context=context,
    )
    direct_signature = space.make_config(
        **space.candidate_kwargs(direct_store, context)
    ).compile_signature()
    assert direct_store.options["use_tma_store"] is False
    assert direct_signature["num_stages_c_smem"] == 0

    tma_store = generate_move(
        direct_store,
        "epilogue",
        _PreferFieldRandom(
            "use_tma_store",
            direct_store.options,
            preferred=True,
        ),
        context=context,
    )
    assert tma_store.options["use_tma_store"] is True
    assert tma_store.options["num_stages_c_smem"] == original_stages


def test_direct_store_option_round_trips_through_mapper():
    context = _context("fc2")
    parent = enumerate_incumbents(context, (128,))[0]
    direct_store = replace(
        parent,
        options={**parent.options, "use_tma_store": False},
    )

    kwargs = space.candidate_kwargs(direct_store, context)
    assert kwargs["use_tma_store"] == 0
    space.validate_config(space.make_config(**kwargs))


def test_skeleton_disabling_fused_sf_copy_clears_dependents():
    parent = Candidate(
        options={
            "fuse_sf_copy_to_mma": True,
            "fuse_operand_sf_loads": True,
            "use_max_tmem_overlap": True,
        },
        tile_n=128,
    )

    child = generate_move(
        parent,
        "skeleton",
        _PreferFieldRandom("fuse_sf_copy_to_mma", parent.options),
    )

    assert child.options["fuse_sf_copy_to_mma"] is False
    assert child.options["fuse_operand_sf_loads"] is False
    assert child.options["use_max_tmem_overlap"] is False


def test_fc2_skeleton_prunes_routing_only_and_unsupported_trials(
    monkeypatch,
):
    parent = Candidate(options={}, tile_n=128)
    probe = SimpleNamespace(
        mma_n=128,
        smem_sfa_layout=int(SfLayout.R128c4),
        smem_sfb_layout=int(SfLayout.R128c4),
        tile_n=128,
        has_input_block_scaling=True,
    )
    captured = []

    def pick_first(_parent, trials, *_args, **_kwargs):
        captured.extend(trials)
        return captured[0]

    monkeypatch.setattr(space, "candidate_kwargs", lambda *_args: {})
    monkeypatch.setattr(space, "make_config", lambda **_kwargs: probe)
    monkeypatch.setattr(space, "_effective_signature", lambda *_args: {})
    monkeypatch.setattr(space, "_pick_effective", pick_first)
    monkeypatch.setattr(
        space,
        "_effective_key",
        lambda options, *_args: space._digest(dict(options)),
    )
    monkeypatch.setattr(
        space,
        "_normalize_candidate_sf_routing",
        lambda candidate, _context: candidate,
    )

    generate_move(
        parent,
        "skeleton",
        random.Random(0),
        context=_context("fc2"),
    )

    assert captured
    assert all("route_act" not in trial for trial in captured)
    assert all("route_sfs_act" not in trial for trial in captured)
    assert all("fuse_operand_sf_loads" not in trial for trial in captured)
    assert all(not bool(trial.get("use_max_tmem_overlap")) for trial in captured)


def test_fc2_tile8_skeleton_has_no_supported_trial(monkeypatch):
    parent = Candidate(options={}, tile_n=8)
    probe = SimpleNamespace(
        mma_n=8,
        smem_sfa_layout=int(SfLayout.R8c4),
        smem_sfb_layout=int(SfLayout.R8c4),
        tile_n=8,
        has_input_block_scaling=True,
    )
    monkeypatch.setattr(space, "candidate_kwargs", lambda *_args: {})
    monkeypatch.setattr(space, "make_config", lambda **_kwargs: probe)
    monkeypatch.setattr(space, "_effective_signature", lambda *_args: {})

    with pytest.raises(ValueError, match="unsupported for this parent"):
        generate_move(
            parent,
            "skeleton",
            random.Random(0),
            context=_context("fc2"),
        )


def test_all_illegal_trials_report_illegal_rejection(monkeypatch):
    parent = Candidate(options={"tile_k": 256}, tile_n=128)
    root_cause = ValueError("TMEM footprint exceeds capacity")
    monkeypatch.setattr(
        space,
        "_effective_key",
        lambda *_args, **_kwargs: "parent",
    )
    monkeypatch.setattr(
        space,
        "_effective_signature_result",
        lambda *_args, **_kwargs: (None, root_cause),
    )

    with pytest.raises(ValueError, match="every trial illegal") as exc_info:
        space._pick_effective(
            parent,
            ({"tile_k": 512},),
            _context(),
            random.Random(0),
            move_kind="tile_k",
            stage_limits=StageLimits(),
            max_sf_ring_ratio=4,
        )

    assert "TMEM footprint exceeds capacity" in str(exc_info.value)
    assert space._move_rejection_reason(exc_info.value) == "illegal"


def test_warp_move_never_returns_current_count():
    parent = Candidate(
        options={"num_load_a_warps": 1},
        tile_n=128,
    )

    child = generate_move(
        parent,
        "warps",
        _PreferFieldRandom("num_load_a_warps", parent.options),
    )

    assert child.options["num_load_a_warps"] in (2, 4)


def test_register_split_is_a_dedicated_move():
    parent = Candidate(
        options={
            "load_regs": 48,
            "mma_regs": 48,
            "epilogue_regs": 160,
        },
        tile_n=128,
    )

    child = generate_move(parent, "register_split", random.Random(0))

    assert child.options["load_regs"] != 48
    assert child.options["mma_regs"] != 48
    assert child.options["epilogue_regs"] != 160
    assert (
        child.options["load_regs"]
        + child.options["mma_regs"]
        + child.options["epilogue_regs"]
        == 256
    )
    assert child.move_kind == "register_split"


@pytest.mark.parametrize(
    ("move_kind", "options", "expected_families"),
    (
        (
            "route_warps",
            {"route_act": "tma", "num_load_a_warps": 1},
            (
                ("route_act", "route_sfs_act"),
                (
                    "num_load_a_warps",
                    "num_load_b_warps",
                    "num_load_sfa_warps",
                    "num_load_sfb_warps",
                ),
            ),
        ),
        (
            "tile_k_stages",
            {"tile_k": 128, "num_stages_a": 2},
            (("tile_k",), ("num_stages_a",)),
        ),
    ),
)
def test_compound_moves_change_both_knob_families(
    move_kind, options, expected_families
):
    parent = Candidate(options=options, tile_n=128)

    child = generate_move(parent, move_kind, random.Random(9))

    assert all(
        any(child.options.get(field) != parent.options.get(field) for field in family)
        for family in expected_families
    )
    assert child.parent_ids == (parent.candidate_id,)
    assert child.move_kind == move_kind


def test_tile_k_stages_preserves_exact_inverse_footprints():
    tile_parent = Candidate(
        options={
            "tile_k": 128,
            "num_stages_a": 2,
            "num_stages_b": 4,
        },
        tile_n=128,
    )
    tiled = generate_move(
        tile_parent,
        "tile_k_stages",
        random.Random(4),
    )
    for field in ("num_stages_a", "num_stages_b"):
        assert int(tiled.options["tile_k"]) * int(tiled.options[field]) == int(
            tile_parent.options["tile_k"]
        ) * int(tile_parent.options[field])


def test_route_warps_tunes_the_matching_loader():
    parent = Candidate(
        options={
            "route_act": "tma",
            "route_sfs_act": "tma",
            "num_load_a_warps": 1,
            "num_load_sfa_warps": 1,
        },
        tile_n=128,
    )

    child = generate_move(
        parent,
        "route_warps",
        _PreferFieldRandom("route_act", parent.options, preferred="ldgsts"),
    )

    assert child.options["route_act"] == "ldgsts"
    assert child.options["num_load_a_warps"] in (2, 4)
    assert child.options["num_load_sfa_warps"] == 1


def test_crossover_tunes_one_structural_donor_family():
    parent = Candidate(
        options={
            "route_act": "tma",
            "tile_scheduler": "static",
            "num_stages_a": 2,
        },
        tile_n=128,
    )
    donor = Candidate(
        options={
            "route_act": "ldgsts",
            "tile_scheduler": "persistent",
            "num_stages_a": 9,
        },
        tile_n=128,
    )

    child = generate_move(
        parent,
        "crossover",
        _PreferFieldRandom("route_act", parent.options, preferred="ldgsts"),
        other_parent=donor,
    )

    assert child.options["route_act"] == "ldgsts"
    assert child.options["num_stages_a"] == 9
    assert child.repair_kind == "crossover:donor_tuned"


def test_crossover_falls_back_to_structural_donor(monkeypatch):
    parent = Candidate(
        options={
            "route_act": "tma",
            "num_stages_a": 2,
        },
        tile_n=128,
    )
    donor = Candidate(
        options={
            "route_act": "ldgsts",
            "num_stages_a": 9,
        },
        tile_n=128,
    )
    calls = []

    def pick_once_then_fallback(
        _parent,
        trials,
        _context,
        _rng,
        **_kwargs,
    ):
        trials = list(trials)
        calls.append(trials)
        if len(calls) == 1:
            raise ValueError("donor tuning exceeds budget")
        return trials[0]

    monkeypatch.setattr(space, "_pick_effective", pick_once_then_fallback)

    child = generate_move(
        parent,
        "crossover",
        random.Random(0),
        other_parent=donor,
    )

    assert len(calls) == 2
    assert child.options["route_act"] == "ldgsts"
    assert child.options["num_stages_a"] == 2
    assert child.repair_kind == "crossover:donor_repaired"


def test_crossover_cluster_donor_includes_y_dimensions():
    parent = Candidate(
        options={
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
            "cluster_dim_y": 1,
            "fallback_cluster_dim_x": 0,
            "fallback_cluster_dim_y": 1,
        },
        tile_n=128,
    )
    donor = Candidate(
        options={
            **parent.options,
            "cluster_dim_y": 2,
        },
        tile_n=128,
    )
    assert space.structural_options_key(parent) != space.structural_options_key(donor)
    with pytest.raises(ValueError, match="no effective crossover mutation"):
        generate_move(
            parent,
            "crossover",
            random.Random(0),
            other_parent=donor,
        )

    child = generate_move(
        parent,
        "crossover",
        _PreferFieldRandom(
            "cluster_dim_y",
            parent.options,
            preferred=2,
        ),
        other_parent=donor,
        enable_cluster_y=True,
    )

    assert child.options["cluster_dim_y"] == 2


def test_crossover_accepts_stage_only_donor_difference():
    parent = Candidate(
        options={"tile_scheduler": "persistent", "num_stages_a": 2},
        tile_n=128,
    )
    other = Candidate(
        options={"tile_scheduler": "persistent", "num_stages_a": 4},
        tile_n=128,
    )

    child = generate_move(
        parent,
        "crossover",
        random.Random(0),
        other_parent=other,
    )

    assert child.options["num_stages_a"] == 4
    assert child.repair_kind == "crossover:donor_tuned"


@pytest.mark.parametrize("move_kind", MOVE_KINDS)
def test_every_move_kind_generates_a_lineaged_child(move_kind):
    incumbents = enumerate_incumbents(_context(), (128,))
    parent = incumbents[0]
    other = incumbents[-1]
    kwargs = {"other_parent": other} if move_kind == "crossover" else {}
    try:
        child = generate_move(parent, move_kind, random.Random(13), **kwargs)
    except ValueError:
        return
    assert child.move_kind == move_kind
    assert parent.candidate_id in child.parent_ids
