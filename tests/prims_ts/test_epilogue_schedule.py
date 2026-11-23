# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Preload schedule lifetimes, configuration compatibility, and cache identity."""

from dataclasses import replace

import pytest

from flashinfer.prims_ts.batched_gemm import epilogue_schedule as schedules


def _config(preset=schedules.FINALIZE_FULL_TILE_4W, **overrides):
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic

    cfg = map_trtllm_nvfp4_moe_tactic([512, 0], sm_arch=107).fc2.cfg.build()
    options = dict(
        preset.overrides(),
        num_stages_a=2,
        num_stages_b=2,
        num_stages_smem_sfa=2,
        num_stages_smem_sfb=2,
    )
    options.update(overrides)
    return replace(cfg, **options)


@pytest.mark.parametrize(
    "preset",
    [
        schedules.PLAIN_PRELOAD_4W,
        schedules.PLAIN_PRELOAD_8W,
        schedules.PLAIN_PREFIX_8W,
        schedules.FINALIZE_PRELOAD_8W,
        schedules.FINALIZE_PREFIX_8W,
        schedules.FINALIZE_FULL_TILE_4W,
    ],
)
def test_preset_preserves_legacy_schedule_and_cache_key(preset):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.moe.compile_cache import stable_config_hash

    explicit = _config(preset)
    legacy = replace(
        explicit,
        epilogue_tmem_load_width=0,
        epilogue_tmem_load_ahead=-1,
        epilogue_smem_staging=0,
        epilogue_tmem_release=0,
    )
    validate_config(explicit, problem_mnk=(8192, 1024, 512))
    assert explicit.preload_epilogue_plan == legacy.preload_epilogue_plan
    assert stable_config_hash(explicit) == stable_config_hash(legacy)
    plan = explicit.preload_epilogue_plan
    # Every accumulator element is loaded exactly once, including mixed reads.
    addresses = [i * width + j for i, width in plan.tmem_reads for j in range(width)]
    assert addresses == list(range(len(addresses)))
    assert len(addresses) * plan.warps * 32 == 256 * 256
    assert len(plan.preload_destinations) == 8
    assert (
        plan.preload_destinations.count("registers") * (32 if plan.warps == 4 else 16)
        == plan.retained_words
    )
    assert plan.barrier_threads == (0 if plan.thread_owns_scratch else plan.warps * 32)


def test_load_ahead_and_release_are_distinct_compilations():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.moe.compile_cache import stable_config_hash

    pipelined = _config(schedules.FINALIZE_PRELOAD_8W)
    serial = replace(pipelined, epilogue_tmem_load_ahead=0)
    late = replace(
        serial, epilogue_tmem_release=int(schedules.TmemRelease.AFTER_OUTPUT)
    )
    for cfg in (pipelined, serial, late):
        validate_config(cfg, problem_mnk=(8192, 1024, 512))
    assert len({stable_config_hash(cfg) for cfg in (pipelined, serial, late)}) == 3
    assert late.preload_epilogue_plan.release_after_output
    assert late.preload_epilogue_plan.fragments_before_release == 8
    assert serial.preload_epilogue_plan.preload_destinations == ("registers",) * 8


@pytest.mark.parametrize(
    "overrides",
    [
        {"epilogue_tmem_load_width": 16},
        {"epilogue_tmem_load_width": 32},
        {"epilogue_tmem_load_ahead": 2},
        {"epilogue_tmem_load_ahead": 1},
        {"epilogue_smem_staging": 1},
        {"epilogue_smem_staging": 3},
        {"epilogue_tmem_release": 3},
        {"dense_preload_epilogue": 0},
        {"num_stages_c_smem": 1},
    ],
)
def test_reject_unsupported_data_lifetimes(overrides):
    with pytest.raises(ValueError):
        schedules.resolve_preload_epilogue(_config(**overrides))


def test_cooperative_prefix_cannot_request_unimplemented_read_ahead():
    with pytest.raises(ValueError, match="load_ahead"):
        schedules.resolve_preload_epilogue(
            _config(schedules.FINALIZE_PREFIX_8W, epilogue_tmem_load_ahead=1)
        )


@pytest.mark.parametrize("value", [{}, {"load_fragments": 1}, "", None, 0, set()])
def test_epilogue_steps_reject_non_arrays(value):
    with pytest.raises(ValueError, match="tuple or list"):
        schedules.normalize_epilogue_steps(value)
    assert schedules.normalize_epilogue_steps(()) == ()


def test_schedule_controls_cli_and_source_key():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )
    from flashinfer.prims_ts.moe.compile_cache import _kernel_source_files
    from flashinfer.prims_ts.moe.config_mapper import _RUBIN_JSON_INT_OPTIONS

    fields = dict(
        epilogue_tmem_load_width=32,
        epilogue_tmem_load_ahead=0,
        epilogue_smem_staging=1,
        epilogue_tmem_release=2,
    )
    args = [
        arg
        for name, value in fields.items()
        for arg in ("--" + name.replace("_", "-"), str(value))
    ]
    overrides = _parse_overrides(build_arg_parser().parse_args(args))
    assert {name: overrides[name] for name in fields} == fields
    assert fields.keys() <= _RUBIN_JSON_INT_OPTIONS
    assert any(
        path.endswith("/epilogue_schedule.py") for path in _kernel_source_files()
    )


def _step_config(steps, *, warps=8, stages=2, finalize=True):
    return _config(
        dense_epilogue_warps=warps,
        epilogue_regs=248 if warps == 4 else 168,
        num_stages_c_smem=stages,
        fuse_moe_finalize=int(finalize),
        epilogue_steps=steps,
        epilogue_tmem_load_width=0,
        epilogue_tmem_load_ahead=-1,
        epilogue_smem_staging=0,
        epilogue_tmem_release=0,
    )


def _verify_repeated_tile_lifetimes(plan):
    """Adversarial model: asynchronous reads never finish unless waited on.

    Check each individual SMEM write against *all* pending groups, and verify
    the contents consumed by output operations. Repeat the generated body,
    including startup with empty queues and steady-state reuse.
    """
    op = schedules.EpilogueOp
    queue = []
    scratch = {}
    unit = plan.output_unit
    for tile in range(5):
        issuing = set()
        seen = []
        pending_reads = set()
        issued_reads = {}
        completed_reads = {}
        registers = set()
        released = False
        for action in plan.actions:
            if action.op == op.READ:
                assert not released
                assert action.read_id not in issued_reads
                issued_reads[action.read_id] = (
                    action.fragment,
                    action.offset,
                    action.width,
                )
                pending_reads.add(action.read_id)
            elif action.op == op.READ_WAIT:
                completed_reads.update(
                    (read_id, issued_reads[read_id]) for read_id in pending_reads
                )
                pending_reads.clear()
            elif action.op == op.CONVERT:
                assert completed_reads.pop(action.read_id) == (
                    action.fragment,
                    action.offset,
                    action.width,
                )
                if plan.gated:
                    for fragment in range(
                        action.fragment, action.fragment + action.width // 64
                    ):
                        for word in range(plan.words_per_fragment):
                            registers.add((fragment, word))
                else:
                    for word in range(
                        action.offset // 2, (action.offset + action.width) // 2
                    ):
                        registers.add((action.fragment, word))
            elif action.op == op.STAGE:
                assert not any(action.slot in group for group in queue)
                for word in range(plan.words_per_fragment):
                    registers.remove((action.fragment, word))
                scratch[action.slot, action.fragment % unit] = (tile, action.fragment)
            elif action.op == op.EMIT:
                for f in range(action.fragment, action.fragment + unit):
                    assert scratch[action.slot, f % unit] == (tile, f)
                    seen.append(f)
                issuing.add(action.slot)
            elif action.op == op.COMMIT:
                queue.append(issuing)
                issuing = set()
            elif action.op == op.SOURCE_WAIT:
                # wait_group N leaves at most the newest N groups pending.
                queue = queue[-action.pending :] if action.pending else []
            elif action.op == op.RELEASE:
                assert not released and not pending_reads
                released = True
        assert not issuing
        assert released and not registers and not completed_reads
        assert seen == list(range(plan.fragment_count))


def test_lifetime_audit_rejects_missing_final_commit():
    cfg = _step_config(
        schedules.epilogue_steps_for_prefix(0, finalize=True), finalize=True
    )
    plan = cfg.preload_epilogue_plan
    _verify_repeated_tile_lifetimes(plan)
    last_commit = max(
        i
        for i, action in enumerate(plan.actions)
        if action.op == schedules.EpilogueOp.COMMIT
    )
    with pytest.raises(AssertionError):
        _verify_repeated_tile_lifetimes(
            replace(
                plan,
                actions=plan.actions[:last_commit] + plan.actions[last_commit + 1 :],
            )
        )


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("finalize", [False, True])
@pytest.mark.parametrize("ahead", [0, 1])
def test_explicit_prefix_lifetimes(warps, stages, finalize, ahead):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    valid = 0
    for prefix in range(9):
        cfg = _step_config(
            schedules.epilogue_steps_for_prefix(
                prefix, finalize=finalize, read_ahead=ahead
            ),
            warps=warps,
            stages=stages,
            finalize=finalize,
        )
        try:
            plan = cfg.preload_epilogue_plan
        except ValueError:
            # Four-warps cannot retain all eight fragments. Prefixes that
            # occupy more than the allocated slots need earlier emission.
            assert (warps == 4 and prefix == 0) or prefix > stages * (
                4 if finalize else 1
            )
            continue
        validate_config(cfg, problem_mnk=(8192, 1024, 512))
        _verify_repeated_tile_lifetimes(plan)
        valid += 1
    assert valid


def test_group_waits_follow_groups_not_slot_count():
    Step = schedules.EpilogueStep
    full_tile = (Step(load_fragments=1, stage_fragments=1),) * 8 + (
        Step(release_tmem=True),
        Step(emit_fragments=8),
    )
    one_group = _step_config(full_tile, warps=4).preload_epilogue_plan
    two_groups = _step_config(
        schedules.epilogue_steps_for_prefix(8), warps=4
    ).preload_epilogue_plan
    waits = lambda plan: [
        a.pending for a in plan.actions if a.op == schedules.EpilogueOp.SOURCE_WAIT
    ]
    assert waits(one_group) == [0]  # One group reads both slots.
    assert waits(two_groups) == [1, 0]  # Each group reads one slot.
    for plan in (one_group, two_groups):
        _verify_repeated_tile_lifetimes(plan)
        assert plan.barrier_threads == 0
        assert all(a.op != schedules.EpilogueOp.BARRIER for a in plan.actions)

    plain_steps = (
        Step(load_fragments=2, stage_fragments=2),
        Step(emit_fragments=2),
    ) * 4 + (Step(release_tmem=True),)
    plain = _step_config(plain_steps, finalize=False).preload_epilogue_plan
    assert waits(plain) == [0, 0, 0, 0]
    assert plain.barrier_threads == 256
    _verify_repeated_tile_lifetimes(plain)


@pytest.mark.parametrize(
    "steps",
    [
        [dict(load_fragments=8), dict(release_tmem=True)],
        [dict(load_fragments=4), dict(release_tmem=True)],
        [dict(stage_fragments=1)],
        [dict(load_fragments=4, stage_fragments=4), dict(emit_fragments=2)],
        [
            dict(load_fragments=8, stage_fragments=8),
            dict(release_tmem=True),
            dict(emit_fragments=8),
        ],
        [dict(load_fragments=1, release_tmem=True)],
        [dict(load_fragments=1, emit_fragments=1)],
        [dict(load_fragments=-1)],
        [dict(load_fragments=True)],
        [dict(load_fragments=1.5)],
        [dict(load_fragments=1, read_ahead=2)],
        [dict(stage_fragments=1, read_ahead=1)],
        [dict(release_tmem=1)],
        [dict(unknown=1)],
        [{}],
    ],
)
def test_explicit_schedule_rejects_invalid_transfers(steps):
    with pytest.raises(ValueError):
        _ = _step_config(steps, stages=1).preload_epilogue_plan


def test_explicit_schedule_json_identity_and_legacy_conflict():
    from dataclasses import asdict
    from flashinfer.prims_ts.moe.compile_cache import stable_config_hash

    steps = schedules.epilogue_steps_for_prefix(2, read_ahead=1)
    cfg = _step_config(steps)
    restored = replace(cfg, epilogue_steps=[asdict(s) for s in steps])
    assert restored.epilogue_steps == steps
    assert stable_config_hash(restored) == stable_config_hash(cfg)
    assert stable_config_hash(
        replace(cfg, epilogue_steps=schedules.epilogue_steps_for_prefix(2))
    ) != stable_config_hash(cfg)
    for override in (
        dict(dense_preload_store_prefix_fragments=2),
        dict(epilogue_tmem_load_ahead=0),
        dict(epilogue_tmem_release=1),
        dict(dense_preload_epilogue=0),
    ):
        with pytest.raises(ValueError):
            _ = replace(cfg, **override).preload_epilogue_plan


def test_bounded_search_covers_resources_and_safe_lifetimes():
    from flashinfer.prims_ts.batched_gemm.epilogue_search import iter_epilogue_configs

    candidates = list(iter_epilogue_configs(_config(), max_candidates=32))
    assert len(candidates) == 32
    assert {(c.dense_epilogue_warps, c.num_stages_c_smem) for c in candidates} == {
        (4, 1),
        (4, 2),
        (8, 1),
        (8, 2),
    }
    for cfg in candidates:
        _verify_repeated_tile_lifetimes(cfg.preload_epilogue_plan)
    # Search beyond the curated family and verify generated combinations too.
    exhaustive = list(
        iter_epilogue_configs(
            _config(), warps=(8,), stages=(1,), exhaustive=True, max_candidates=40
        )
    )
    assert len(exhaustive) == 40
    for cfg in exhaustive:
        _verify_repeated_tile_lifetimes(cfg.preload_epilogue_plan)


def test_explicit_steps_cli():
    import json
    from dataclasses import asdict
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    defaults = _parse_overrides(build_arg_parser().parse_args([]))
    assert defaults["epilogue_steps"] == ()
    steps = schedules.epilogue_steps_for_prefix(2)
    args = build_arg_parser().parse_args(
        ["--epilogue-steps", json.dumps([asdict(s) for s in steps])]
    )
    assert (
        schedules.normalize_epilogue_steps(_parse_overrides(args)["epilogue_steps"])
        == steps
    )


def test_moe_preserves_explicit_epilogue_resource_profile():
    from flashinfer.prims_ts.moe.runner import _fused_finalize_config

    cfg = _step_config(schedules.epilogue_steps_for_prefix(2), warps=4, stages=1)
    selected = _fused_finalize_config(replace(cfg, fuse_moe_finalize=0))
    assert selected == cfg
    _verify_repeated_tile_lifetimes(selected.preload_epilogue_plan)


def test_search_register_budgets_have_distinct_configurations():
    from flashinfer.prims_ts.batched_gemm.epilogue_search import iter_epilogue_configs
    from flashinfer.prims_ts.moe.compile_cache import stable_config_hash

    candidates = list(
        iter_epilogue_configs(
            _config(),
            warps=(4,),
            stages=(2,),
            register_budgets={4: (248, 256)},
            max_candidates=2,
        )
    )
    assert {cfg.epilogue_regs for cfg in candidates} == {248, 256}
    assert len({stable_config_hash(cfg) for cfg in candidates}) == 2


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("ahead", [0, 1])
def test_fc1_step_lifetimes(warps, stages, ahead):
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    base = map_trtllm_nvfp4_moe_tactic([512, 1], sm_arch=107).fc1.cfg.build()
    for prefix in range(min(32 // warps, stages * 2) + 1):
        cfg = replace(
            base,
            dense_preload_epilogue=1,
            dense_epilogue_warps=warps,
            num_stages_c_smem=stages,
            epilogue_regs=248 if warps == 4 else 168,
            epilogue_steps=schedules.epilogue_steps_for_prefix(
                prefix, finalize=False, gated=True, warps=warps, read_ahead=ahead
            ),
        )
        validate_config(cfg)
        plan = cfg.preload_epilogue_plan
        assert plan.gated and plan.words_per_fragment == 4 and plan.output_unit == 2
        assert plan.fragment_count * (warps // 4) == 8
        assert plan.retained_words == (plan.fragment_count - prefix) * 4
        _verify_repeated_tile_lifetimes(plan)
        with pytest.raises(ValueError, match="interleaved"):
            _ = replace(cfg, gated_weight_group_size=64).preload_epilogue_plan


def test_fc1_search_validates_and_deduplicates():
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic
    from flashinfer.prims_ts.batched_gemm.epilogue_search import iter_epilogue_configs

    base = map_trtllm_nvfp4_moe_tactic([1024, 0], sm_arch=107).fc1.cfg.build()
    candidates = list(iter_epilogue_configs(base, max_candidates=32))
    assert 4 <= len(candidates) <= 32
    first = candidates[0]
    assert first.dense_epilogue_warps == base.dense_epilogue_warps
    assert first.num_stages_c_smem == base.num_stages_c_smem
    assert first.epilogue_steps == schedules.fc1_streaming_steps(
        warps=base.dense_epilogue_warps
    )
    assert {c.dense_epilogue_warps for c in candidates} == {4, 8}
    assert {c.num_stages_c_smem for c in candidates} == {1, 2}
    for cfg in candidates:
        _verify_repeated_tile_lifetimes(cfg.preload_epilogue_plan)


def test_fc1_search_rejects_total_register_overflow():
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic
    from flashinfer.prims_ts.batched_gemm.epilogue_search import iter_epilogue_configs

    base = map_trtllm_nvfp4_moe_tactic([512, 1], sm_arch=107).fc1.cfg.build()
    candidates = list(
        iter_epilogue_configs(
            base,
            warps=(8,),
            stages=(2,),
            register_budgets={8: (144, 168)},
            max_candidates=64,
        )
    )
    assert candidates and {c.epilogue_regs for c in candidates} == {144}


@pytest.mark.parametrize("warps", [4, 8])
def test_fc1_fragment_mapping_covers_accumulator_once(warps):
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic

    base = map_trtllm_nvfp4_moe_tactic([512, 1], sm_arch=107).fc1.cfg.build()
    cfg = replace(
        base,
        dense_preload_epilogue=1,
        dense_epilogue_warps=warps,
        epilogue_regs=248 if warps == 4 else 144,
        epilogue_steps=schedules.epilogue_steps_for_prefix(
            0, finalize=False, gated=True, warps=warps
        ),
    )
    plan = cfg.preload_epilogue_plan
    indices = [
        plan.fc1_fragment_index(f, g)
        for f in range(plan.fragment_count)
        for g in range(warps // 4)
    ]
    assert sorted(indices) == list(range(8))
    for group in range(warps // 4):
        for fragment in range(0, plan.fragment_count, 2):
            first = plan.fc1_fragment_index(fragment, group)
            second = plan.fc1_fragment_index(fragment + 1, group)
            assert first // 4 == second // 4 and second == first + 1


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("batch", [1, 2])
def test_fc1_planned_streaming_matches_legacy_ownership(warps, stages, batch):
    from collections import Counter
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic

    base = map_trtllm_nvfp4_moe_tactic([512, 1], sm_arch=107).fc1.cfg.build()
    cfg = replace(
        base,
        dense_preload_epilogue=1,
        dense_epilogue_warps=warps,
        epilogue_regs=144 if warps == 8 else 248,
        num_stages_c_smem=stages,
        epilogue_steps=schedules.fc1_streaming_steps(
            warps=warps, batch_fragments=batch
        ),
    )
    plan = cfg.preload_epilogue_plan
    op = schedules.EpilogueOp
    assert plan.barrier_threads == 128
    assert not plan.retained_words and not plan.after_release
    counts = Counter(a.op for a in plan.actions)
    units = plan.fragment_count // 2
    assert counts[op.READ_WAIT] == plan.fragment_count // batch
    assert counts[op.BARRIER] == units * 2  # source reuse + publication per box
    assert counts[op.EMIT] == counts[op.COMMIT] == units
    assert {a.width for a in plan.actions if a.op == op.READ} == {batch * 64}
    body = plan.fc1_rolled_actions
    assert body and sum(a.op == op.EMIT for a in body) == 1
    assert sum(a.op == op.CONVERT for a in body) == 2 // batch
    assert {a.pending for a in body if a.op == op.SOURCE_WAIT} == {stages - 1}
    _verify_repeated_tile_lifetimes(plan)


def test_fc1_rolling_does_not_fold_grouped_emission_or_early_release():
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic

    base = map_trtllm_nvfp4_moe_tactic([512, 1], sm_arch=107).fc1.cfg.build()
    for steps in (
        schedules.epilogue_steps_for_prefix(0, finalize=False, gated=True),
        (schedules.EpilogueStep(load_fragments=2, stage_fragments=2),) * 2
        + (
            schedules.EpilogueStep(emit_fragments=4),
            schedules.EpilogueStep(release_tmem=True),
        ),
    ):
        cfg = replace(base, dense_preload_epilogue=1, epilogue_steps=steps)
        assert not cfg.preload_epilogue_plan.fc1_rolled_actions
        _verify_repeated_tile_lifetimes(cfg.preload_epilogue_plan)


def test_legacy_preset_clears_explicit_steps():
    cfg = _step_config(schedules.epilogue_steps_for_prefix(2, finalize=False), warps=8)
    legacy = replace(cfg, **schedules.PLAIN_PRELOAD_8W.overrides())
    assert not legacy.epilogue_steps
    assert (
        legacy.preload_epilogue_plan
        == _config(schedules.PLAIN_PRELOAD_8W).preload_epilogue_plan
    )


@pytest.mark.parametrize("streaming", [0, 1])
@pytest.mark.parametrize("group", [0, 1])
def test_fc1_emit_uses_streaming_cache_policy(monkeypatch, streaming, group):
    from types import SimpleNamespace
    from flashinfer.prims_ts.batched_gemm import gmem_c_resources as module
    from tests.prims_ts.python_body import python_body

    policy, calls = object(), []
    monkeypatch.setattr(module, "Int32", int)
    monkeypatch.setattr(module, "_streaming_c_cache_policy", lambda: policy)
    monkeypatch.setattr(module.cute.arch, "warp_idx", lambda: group * 4)
    monkeypatch.setattr(module.cute.arch, "lane_idx", lambda: 0)
    monkeypatch.setattr(
        module.prims,
        "cp_async_bulk_tensor_global_shared_cta",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    cfg = SimpleNamespace(
        dense_streaming_c=streaming,
        tile_m=256,
        tile_n=256,
        num_bytes_c_tma_store_per_group=4096,
        warp_layout=SimpleNamespace(epilogue_warp_idx=0),
        preload_epilogue_plan=schedules.EpilogueStepPlan(
            steps=(),
            actions=(),
            warps=8,
            finalize=False,
            scratch_stages=1,
            retained_fragments=(),
            peak_payload_words=0,
            gated=True,
        ),
    )
    resource = SimpleNamespace(
        cfg=cfg,
        problem_n=1024,
        tma_c_desc="descriptor",
        sC=SimpleNamespace(subview=lambda offset: offset),
    )
    stage = SimpleNamespace(work_tile=SimpleNamespace(tile_idx=(1, 2, 0)))
    action = SimpleNamespace(op=schedules.EpilogueOp.EMIT, fragment=0)
    python_body(module.GmemCResource._fc1_step_output_action)(
        resource, stage, action, None
    )
    assert calls == [
        (
            ("descriptor", group * 4096, (256 + group * 64, 256)),
            dict(l2_cache_hint=policy if streaming else None),
        )
    ]
