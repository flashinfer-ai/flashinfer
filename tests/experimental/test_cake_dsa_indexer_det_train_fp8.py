"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Contract tests of the FP8 form of the experimental Cake DSA indexer top-k
# backend: float8_e4m3fn q / k with per-key FP32 scales, selected by the input
# dtype (the BF16 form stays the default).  The host-only tests (scan key
# grammar, the FP8 dispatch record and kind mapping, reachability of every FP8
# key, validation, BF16 plan and workspace identity) run without a GPU; the
# operator tests need a compute capability 10.0 / 10.3 / 10.7 device with
# registered FP8 programs and skip otherwise.
#
# Like ``test_cake_dsa_indexer_det_train.py`` this suite names the package once, through
# ``tests.cake.test_helpers.cake_package``, and imports nothing else from flashinfer.

import copy
import importlib.util
import itertools

import pytest
import torch

from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref
from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_dsa_indexer_det_train", "cake_backend")
cake_jit = cake_package("cake.cake_dsa_indexer_det_train", "cake_jit")
cake_policy = cake_package("cake.cake_dsa_indexer_det_train", "cake_policy")
HEAD_DIM = cake_backend.HEAD_DIM
NUM_HEADS = cake_backend.NUM_HEADS
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
arch_for = cake_backend.arch_for
cake_dsa_indexer_det_train_topk_workspace_size = (
    cake_backend.cake_dsa_indexer_det_train_topk_workspace_size
)
generated_program_available = cake_backend.generated_program_available
plan_dsa_indexer_det_topk = cake_backend.plan_dsa_indexer_det_topk
policy_for = cake_backend.policy_for
prepare_dsa_indexer_det_topk = cake_backend.prepare_dsa_indexer_det_topk
validate_dsa_indexer_det_inputs = cake_backend.validate_dsa_indexer_det_inputs
BF16 = cake_policy.BF16
FP8 = cake_policy.FP8
FP8_KIND_FALLBACK = cake_policy.FP8_KIND_FALLBACK
MERGE_KEY = cake_policy.MERGE_KEY
POLICY_FIELDS = cake_policy.POLICY_FIELDS
DispatchPolicy = cake_policy.DispatchPolicy
fp8_policy_record = cake_policy.fp8_policy_record
fp8_program_choice = cake_policy.fp8_program_choice
fp8_scan_programs = cake_policy.fp8_scan_programs
sample_class_for = cake_policy.sample_class_for
scan_key = cake_policy.scan_key
scan_key_fields = cake_policy.scan_key_fields
scan_key_form = cake_policy.scan_key_form
select_fp8_program = cake_policy.select_fp8_program
select_program = cake_policy.select_program
stage_slug = cake_policy.stage_slug
E4M3 = torch.float8_e4m3fn

# The SM counts each architecture ships with (one persistent CTA per SM).
_GRIDS = {"sm_100a": (148,), "sm_103a": (148, 152, 160), "sm_107a": (212,)}
# DeepGEMM's fp8_mqa_logits accumulates in four K = 32 steps, the order of the sm_100a / sm_103a FP8 programs.
_K32_ARCHES = ("sm_100a", "sm_103a")
# The architectures that register an FP8 wide program; elsewhere an FP8 wide decision runs l6 (``FP8_KIND_FALLBACK``).
_FP8_WIDE_ARCHES = ("sm_100a", "sm_103a", "sm_107a")


def _spec_fp8_keys(arch: str) -> set[str]:
    """The FP8 scan keys the export registers per architecture: narrow, l6, split_narrow, split_l6 and wide, each
    with its :lsf copy, on sm_100a / sm_103a; narrow, l6 and its snake copy, split_narrow and wide, without :lsf
    copies, on sm_107a.  A stand-in registry for the host tests while a checkout registers none; the dispatch reads
    the tile unroll from the keys, so two unrolls appear."""
    if arch == "sm_107a":
        kinds, lsf = (("narrow", 2), ("l6", 1), ("split_narrow", 1)), (False,)
    else:
        kinds = (("narrow", 1), ("l6", 1), ("split_narrow", 1), ("split_l6", 1))
        lsf = (False, True)
    if arch in _FP8_WIDE_ARCHES:
        kinds += (("wide", 2),)
    keys = {
        scan_key(kind, unroll, False, True, copy, form=FP8)
        for (kind, unroll), copy in itertools.product(kinds, lsf)
    }
    if arch == "sm_107a":
        keys.add(scan_key("l6", 1, True, True, False, form=FP8))
    return keys


def _registries():
    """``(label, arch, scan keys, FP8 keys)``: the spec's key set over the registered BF16 keys of every architecture,
    and the checkout's own registry (its FP8 keys empty where it registers none: the test skips)."""
    out = []
    for arch in sorted(cake_jit.PROGRAM_KEYS or {}):
        keys = cake_jit.PROGRAM_KEYS[arch]
        bf16 = {k for k in keys if scan_key_form(k) != FP8}
        out.append(("spec", arch, bf16 | _spec_fp8_keys(arch), _spec_fp8_keys(arch)))
        registered = {k for k in keys if scan_key_form(k) == FP8}
        out.append(("registry", arch, set(keys), registered))
    return out


def _fp8_policy(arch: str) -> DispatchPolicy:
    return DispatchPolicy.from_record(
        fp8_policy_record(cake_jit.POLICY[arch], cake_jit.POLICY_FP8[arch])
    )


def _fp8_programs_without_wide(keys) -> dict:
    """``fp8_scan_programs(keys)`` with the FP8 wide programs hidden: a wide decision then runs the l6 program
    (``FP8_KIND_FALLBACK``, ``fallback_knobs``) as on an architecture that registers none."""
    return {
        kind: unrolls
        for kind, unrolls in fp8_scan_programs(keys).items()
        if kind[0] != "wide"
    }


def _geometries(policy, *, steps_t=16, steps_m=14):
    """A geometric (queries, mean unit tiles) grid as in the BF16 suite's dense grid (coarser), plus the FP8 bench
    shapes."""
    tile = policy.tile_keys
    queries = sorted(
        {1, 2, 3, 4, 2048, 6018, 8192, 16384}
        | {int(round(2 ** (17 * i / steps_t))) for i in range(steps_t + 1)}
    )
    tiles = sorted(
        {0, 1, 2} | {int(round(2 ** (13 * i / steps_m))) for i in range(steps_m + 1)}
    )
    for S, ratio in ((1, 1), (1, 2), (4, 1), (8, 1), (64, 1)):
        for T in queries:
            for m in tiles:
                Tkv = max(1, int(round(m * tile * S + T / (2 * ratio))))
                for top_k in (1, 256, 512, 1024, 2048, 4096):
                    yield T, Tkv, S, ratio, top_k


# ---------------------------------------------------------------------------
# Host-only: key grammar, FP8 record, dispatch, validation, BF16 identity
# ---------------------------------------------------------------------------


def test_fp8_scan_key_grammar_keeps_the_lsf_suffix_last():
    assert scan_key("narrow", 1, False, True, form=FP8) == "scan:narrow:u1:s0:f1:fp8"
    lsf = scan_key("split_l6", 2, False, True, True, form=FP8)
    assert lsf == "scan:split_l6:u2:s0:f1:fp8:lsf" and lsf.split(":")[-1] == "lsf"
    # the BF16 keys are unchanged
    assert scan_key("narrow", 1, False, True) == "scan:narrow:u1:s0:f1"
    assert scan_key("wide", 2, True, False, True) == "scan:wide:u2:s1:f0:lsf"
    with pytest.raises(ValueError, match="form"):
        scan_key("narrow", 1, False, True, form="fp16")
    for physical, unroll, snake, fit, form, copy_ in itertools.product(
        ("narrow", "split_l6", "pair_wide"),
        (1, 2),
        (False, True),
        (False, True),
        cake_policy.FORMS,
        (False, True),
    ):
        key = scan_key(physical, unroll, snake, fit, copy_, form=form)
        fields = scan_key_fields(key)
        assert fields == cake_policy.ScanKeyFields(
            physical, unroll, snake, fit, form, copy_
        ), key
        assert scan_key_form(key) == form and stage_slug(key).isidentifier(), key
        assert (key.split(":")[-1] == "lsf") == copy_, key
    for other in (MERGE_KEY, "finalize:t256", "finalize_rank:t256:w8192", "scan:x"):
        assert scan_key_fields(other) is None and scan_key_form(other) is None
    assert scan_key_fields("scan:narrow:u1:s0:f1:lsf:fp8") is None  # the tag order


def test_fp8_scan_programs_index_the_fp8_keys_only():
    keys = {
        "scan:narrow:u1:s0:f0",
        "scan:narrow:u1:s0:f1:lsf",
        "scan:narrow:u1:s0:f1:fp8",
        "scan:narrow:u2:s0:f1:fp8",
        "scan:narrow:u1:s0:f1:fp8:lsf",
        "scan:l6:u1:s1:f1:fp8",  # a snake copy: not indexed, select_fp8_program swaps it in
        "scan:l6:u1:s0:f0:fp8",  # plain fit form: not an FP8 build
        MERGE_KEY,
    }
    assert fp8_scan_programs(keys) == {
        ("narrow", False): {
            1: "scan:narrow:u1:s0:f1:fp8",
            2: "scan:narrow:u2:s0:f1:fp8",
        },
        ("narrow", True): {1: "scan:narrow:u1:s0:f1:fp8:lsf"},
    }
    assert fp8_scan_programs({k for k in keys if scan_key_form(k) != FP8}) == {}


def test_fp8_record_replaces_only_its_fields_and_leaves_the_bf16_record():
    """``POLICY_FP8`` covers the architectures of ``POLICY``, each naming its own ``static_scan_kinds``; the FP8 record
    is the BF16 record with exactly those fields replaced, and neither registry table changes."""
    if not cake_jit.POLICY:
        pytest.skip("placeholder registry")
    assert (
        set(cake_jit.POLICY_FP8)
        == set(cake_jit.POLICY)
        == set(SUPPORTED_COMPUTE_CAPABILITIES.values())
    )
    before = copy.deepcopy((cake_jit.POLICY, cake_jit.POLICY_FP8))
    for arch, record in cake_jit.POLICY.items():
        fields = cake_jit.POLICY_FP8[arch]
        assert "static_scan_kinds" in fields and set(fields) <= set(POLICY_FIELDS), arch
        fp8 = fp8_policy_record(record, fields)
        # the FP8 record keeps the BF16 record's field order (the registry's, not necessarily POLICY_FIELDS')
        assert tuple(fp8) == tuple(record) and set(record) == set(POLICY_FIELDS), arch
        assert {f: fp8[f] for f in fields} == fields, arch
        assert {f: fp8[f] for f in POLICY_FIELDS if f not in fields} == {
            f: record[f] for f in POLICY_FIELDS if f not in fields
        }, arch
        DispatchPolicy.from_record(fp8)  # a valid record
    assert before == (cake_jit.POLICY, cake_jit.POLICY_FP8)
    assert {
        arch: fields["static_scan_kinds"]
        for arch, fields in cake_jit.POLICY_FP8.items()
    } == {
        "sm_100a": [],
        "sm_103a": [],
        "sm_107a": ["l6", "narrow", "split_narrow", "wide"],
    }
    # the FP8 records' own rules: every split field pinned (no split-resident rule); sm_107a keeps no split-fit
    # rule (no FP8 split program is measured under one), and no FP8 call walks its key tiles in descending order
    # (the descending-walk programs are BF16 programs); the sm_107a FP8 record sets neither BF16 sampling rule (the
    # #7549 rules are measured on the BF16 programs) and runs no longest-first copy (its static programs draw no
    # tickets to order; the BF16 record runs the copies of its l6 ticket programs)
    split_fields = (
        "split_max",
        "split_min_range_tiles",
        "split_wave_rule",
        "split_wave_scope",
        "split_fit_rule",
        "split_resident_rule",
        "split_units",
    )
    assert {
        arch: {
            f: v
            for f, v in fields.items()
            if f != "static_scan_kinds" and f not in split_fields
        }
        for arch, fields in cake_jit.POLICY_FP8.items()
    } == {
        "sm_100a": {
            "wide_wave_rule": [1.7, 1.3, 84.0],
            "desc_walk_kinds": None,
            "snake_packed_rule": None,
            "snake_l6_fallback_waves": None,
        },
        "sm_103a": {
            "wide_wave_rule": [
                1.7,
                1.15,
                64.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.26,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ],
            "lsf_single_rule": [64.0],
            "desc_walk_kinds": None,
            "snake_packed_rule": None,
            "snake_l6_fallback_waves": None,
        },
        "sm_107a": {
            "sample_class_rule": None,
            "sample_long_rule": None,
            "lsf_programs": False,
            "snake_cap_rule": [1, 40],
            "desc_walk_kinds": None,
            "snake_packed_rule": None,
            "snake_l6_fallback_waves": None,
        },
    }
    assert {
        arch: {f: fields[f] for f in split_fields}
        for arch, fields in cake_jit.POLICY_FP8.items()
    } == {
        "sm_100a": dict(
            split_max=16,
            split_min_range_tiles=32,
            split_wave_rule=None,
            split_wave_scope=None,
            split_fit_rule=[4096],
            split_resident_rule=None,
            split_units=["l6", "narrow"],
        ),
        "sm_103a": dict(
            split_max=16,
            split_min_range_tiles=32,
            split_wave_rule=None,
            split_wave_scope=None,
            split_fit_rule=[4096],
            split_resident_rule=None,
            split_units=["l6", "narrow"],
        ),
        "sm_107a": dict(
            split_max=16,
            split_min_range_tiles=32,
            split_wave_rule=[1000, 0.12],
            split_wave_scope=[["narrow"], 1, True],
            split_fit_rule=None,
            split_resident_rule=None,
            split_units=["narrow", "wide"],
        ),
    }
    # the FP8 record never inherits the BF16 programs' static kinds, and takes only record fields
    with pytest.raises(ValueError, match="static_scan_kinds"):
        fp8_policy_record(cake_jit.POLICY["sm_107a"], {})
    with pytest.raises(ValueError, match="static_scan_kinds"):
        fp8_policy_record(
            cake_jit.POLICY["sm_100a"], {"static_scan_kinds": [], "unknown": 1}
        )


def test_fp8_policy_fails_closed_without_fp8_programs(monkeypatch):
    if not cake_jit.POLICY:
        pytest.skip("placeholder registry")
    with pytest.raises(NotImplementedError, match="registered"):
        policy_for("sm_90a", FP8)
    with pytest.raises(ValueError, match="dtype"):
        cake_backend.form_for(torch.float8_e5m2)
    assert cake_backend.form_for(torch.bfloat16) == BF16
    assert cake_backend.form_for(E4M3) == FP8
    for arch in cake_jit.POLICY:
        if cake_backend.fp8_programs_for(arch):
            assert policy_for(arch, FP8) == _fp8_policy(arch)
            continue
        # this checkout registers no FP8 program for the architecture: an FP8 plan raises by name, a BF16 one plans
        with pytest.raises(NotImplementedError, match="float8_e4m3fn"):
            plan_dsa_indexer_det_topk(
                64, 4096, 1, top_k=256, arch=arch, num_sms=148, dtype=E4M3
            )
        plan_dsa_indexer_det_topk(64, 4096, 1, top_k=256, arch=arch, num_sms=148)
    # the record lookup itself, without the process cache: no FP8 program, no FP8 policy
    monkeypatch.setattr(cake_backend, "fp8_programs_for", lambda arch: {})
    for arch in cake_jit.POLICY:
        with pytest.raises(NotImplementedError, match="float8_e4m3fn"):
            cake_backend._policy_for.__wrapped__(arch, FP8)
        assert cake_backend._policy_for.__wrapped__(arch, BF16) == policy_for(arch)
        with pytest.raises(ValueError, match="form"):
            cake_backend._policy_for.__wrapped__(arch, "fp16")


def test_fp8_program_choice_follows_the_key():
    if not cake_jit.POLICY:
        pytest.skip("placeholder registry")
    policy = _fp8_policy("sm_100a")
    keys = set(cake_jit.PROGRAM_KEYS.get("sm_100a", {})) | _spec_fp8_keys("sm_100a")
    base = select_program(
        policy,
        num_queries=2048,
        num_keys=16384,
        num_segments=1,
        ratio=1,
        top_k=2048,
        grid=148,
        scan_keys=keys,
    )
    unit = {
        "narrow": policy.block_q_narrow,
        "wide": policy.block_q_wide,
        "l6": policy.block_q_l6,
    }
    # both key sets carry the wide program
    for key in sorted(_spec_fp8_keys("sm_100a") | _spec_fp8_keys("sm_103a")):
        fields = scan_key_fields(key)
        c = fp8_program_choice(policy, base, key)
        split = fields.physical.startswith("split_")
        assert c.scan_key == key and not c.pair and c.grid_ctas == base.grid
        assert c.split == split and c.physical_kind == fields.physical
        assert c.unit == fields.physical.removeprefix("split_")
        assert c.block_q == unit[c.unit] and c.lsf == fields.lsf
        assert (c.tile_unroll, c.snake, c.sample_fit) == (
            fields.tile_unroll,
            False,
            True,
        )
        assert c.n_split == (max(2, base.n_split) if split else 1)
        assert c.merge_key == (MERGE_KEY if split else None)
        assert c.finalize_key == base.finalize_key and c.cand_cap == base.cand_cap
        assert (
            c.workspace_bytes
            == (c.grid_ctas * c.block_q * c.cand_cap + c.staging_entries)
            * policy.candidate_entry_bytes
            + cake_policy.TICKET_TAIL_BYTES
        )
        assert c.staging_entries == (2048 * c.n_split * 2048 if split else 0)
    assert (
        fp8_program_choice(policy, base, "scan:split_narrow:u1:s0:f1:fp8", 5).n_split
        == 5
    )
    with pytest.raises(ValueError, match="n_split"):
        fp8_program_choice(policy, base, "scan:split_narrow:u1:s0:f1:fp8", 1)
    with pytest.raises(ValueError, match="not an FP8"):
        fp8_program_choice(policy, base, "scan:narrow:u1:s0:f1")
    with pytest.raises(ValueError, match="no FP8 program"):
        fp8_program_choice(policy, base, "scan:pair_l6:u1:s0:f1:fp8")
    with pytest.raises(NotImplementedError, match="no registered FP8 program"):
        select_fp8_program(
            policy,
            scan_keys=keys,
            fp8_programs={},
            num_queries=2048,
            num_keys=16384,
            num_segments=1,
            ratio=1,
            top_k=2048,
            grid=148,
        )


@pytest.mark.parametrize(
    "label, arch, keys, fp8_keys",
    _registries(),
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_fp8_dispatch_maps_the_decision_under_the_fp8_record_and_reaches_every_fp8_key(
    label, arch, keys, fp8_keys
):
    """The FP8 dispatch runs the BF16 decision procedure under the FP8 record and maps its kind to a registered FP8
    kind (``FP8_KIND_FALLBACK``); on a geometry grid every program it selects is an FP8 key of the registry, never a
    CTA pair or the plain fit form, the split, unroll, snake and :lsf choices follow that decision where an FP8 program
    carries them (the :lsf copy also on a one-segment wide call within the record's lsf_single_rule), the sampled first
    threshold and the check period are those of the kind that runs, and every FP8 key
    is selected by some call."""
    if not fp8_keys:
        pytest.skip(f"this checkout registers no FP8 scan program for {arch}")
    policy = _fp8_policy(arch)
    programs = fp8_scan_programs(keys)
    fp8_kinds = {kind for kind, _lsf in programs}
    reached = set()
    for grid in _GRIDS[arch]:
        for T, Tkv, S, ratio, top_k in _geometries(policy):
            bounds = [None] + ([-(-Tkv // S)] if S > 1 else [])
            for bound, sample in itertools.product(bounds, (None, 0)):
                call = dict(
                    num_queries=T,
                    num_keys=Tkv,
                    num_segments=S,
                    ratio=ratio,
                    top_k=top_k,
                    grid=grid,
                    sample_tiles_max=sample,
                    max_seqlen_k=bound,
                    scan_keys=keys,
                )
                decided = select_program(policy, **call)
                c = select_fp8_program(policy, fp8_programs=programs, **call)
                where = (arch, grid, T, Tkv, S, ratio, top_k, bound, sample)
                assert c.scan_key in fp8_keys, (*where, c.scan_key)
                assert not c.pair and c.sample_fit, where
                # the snake order where the decision takes it, the registry has the snake copy of the program and
                # the call stays within the record's snake_cap_rule
                plain = c.scan_key.replace(":s1:", ":s0:", 1)
                assert c.snake == (
                    decided.snake
                    and plain.replace(":s0:", ":s1:", 1) in keys
                    and policy.snake_copy_fits(T, S, c.block_q, grid)
                ), (*where, c.scan_key)
                assert c.grid_ctas == grid and c.finalize_key in keys, where
                physical = decided.physical_kind
                if physical not in fp8_kinds:
                    physical = FP8_KIND_FALLBACK[physical]
                assert c.physical_kind == physical, (*where, decided.scan_key)
                assert (c.split, c.n_split) == (decided.split, decided.n_split), where
                lsf = decided.lsf or (
                    physical == "wide" and policy.lsf_single_fits(T, Tkv, S, ratio)
                )
                assert c.lsf == (lsf and (physical, True) in programs), where
                unrolls = programs[(physical, c.lsf)]
                assert c.tile_unroll == (
                    decided.tile_unroll
                    if decided.tile_unroll in unrolls
                    else min(unrolls)
                ), where
                tiles = (
                    policy.sample_tiles_for(T, Tkv, S, ratio)
                    if sample is None
                    else sample
                )
                assert (
                    c.sample_tiles_max,
                    c.sample_shift_permille,
                ) == sample_class_for(
                    policy,
                    physical,
                    T,
                    Tkv,
                    S,
                    ratio,
                    tiles,
                    policy.sample_shift_for(top_k),
                    tiles_knob=sample is not None,
                    shift_knob=False,
                ), where
                assert c.check_period == policy.check_period_for(
                    top_k, c.cand_cap, c.kind
                ), where
                if physical == decided.physical_kind:
                    assert (
                        c.sample_tiles_max,
                        c.sample_shift_permille,
                        c.check_period,
                    ) == (
                        decided.sample_tiles_max,
                        decided.sample_shift_permille,
                        decided.check_period,
                    ), where
                if arch == "sm_107a":
                    assert not c.lsf, where
                reached.add(c.scan_key)
    assert reached == fp8_keys, sorted(fp8_keys - reached)


def test_fp8_snake_marked_sm107_call_runs_the_registered_snake_copy():
    """On sm_107a a call the snake rule marks (half a query per key or more, 32 mean unit tiles or more) runs the snake
    copy of the FP8 l6 program where the registry has it and the call is one document within the FP8 record's
    snake_cap_rule waves, and the plain program without the copy, over packed documents or beyond the waves; a call
    the rule leaves runs the plain program."""
    arch, twin = "sm_107a", "scan:l6:u1:s1:f1:fp8"
    keys = set(cake_jit.PROGRAM_KEYS.get(arch, {}))
    if twin not in keys:
        pytest.skip(f"this checkout registers no FP8 snake program for {arch}")
    policy = _fp8_policy(arch)
    marked = dict(
        num_queries=16384, num_keys=16384, num_segments=1, ratio=1, top_k=2048, grid=212
    )
    left = dict(marked, num_queries=1024, num_keys=8192)
    assert select_program(policy, scan_keys=keys, **marked).snake
    assert not select_program(policy, scan_keys=keys, **left).snake
    c = select_fp8_program(policy, scan_keys=keys, **marked)
    assert c.scan_key == twin and c.snake, c.scan_key
    c = select_fp8_program(policy, scan_keys=keys - {twin}, **marked)
    assert c.scan_key == "scan:l6:u1:s0:f1:fp8" and not c.snake, c.scan_key
    c = select_fp8_program(policy, scan_keys=keys, **left)
    assert ":s1:" not in c.scan_key and not c.snake, c.scan_key
    # marked calls beyond the cap keep the plain order: 131072 queries (103 waves of 6-query units on 212 CTAs) and
    # two packed 16384-query documents
    long = dict(marked, num_queries=131072, num_keys=131072)
    packed = dict(marked, num_queries=32768, num_keys=32768, num_segments=2)
    for call in (long, packed):
        assert select_program(policy, scan_keys=keys, **call).snake, call
        assert not policy.snake_copy_fits(
            call["num_queries"], call["num_segments"], 6, 212
        ), call
        c = select_fp8_program(policy, scan_keys=keys, **call)
        assert c.scan_key == "scan:l6:u1:s0:f1:fp8" and not c.snake, (call, c.scan_key)


def test_fp8_one_document_sm103_wide_call_runs_the_lsf_copy_within_the_rule():
    """On sm_103a a one-document FP8 call of the wide unit runs the :lsf copy of the FP8 wide program from the FP8
    record's lsf_single_rule mean unit tiles on (the plan leaves the segment order plain on one segment), and the plain
    program below them or where the registry has no copy; packed calls keep the plan's segment order, and the BF16
    dispatch keeps the plain order on one segment."""
    arch, plain, copy = (
        "sm_103a",
        "scan:wide:u2:s0:f1:fp8",
        "scan:wide:u2:s0:f1:fp8:lsf",
    )
    keys = set(cake_jit.PROGRAM_KEYS.get(arch, {}))
    if not {plain, copy} <= keys:
        pytest.skip(
            f"this checkout registers no FP8 wide program with its :lsf copy for {arch}"
        )
    policy = _fp8_policy(arch)
    assert policy.lsf_single_rule == [64.0]
    # mean unit tiles (Tkv - T / 2) / 128: 544 and 56; on 152 CTAs the wide unit needs 7 waves (wide_wave_rule): long
    # (an l6 call) needs 9 of the l6 unit (>= 1.15 x 7) over the 88.5-tile minimum (64 + 3.5 x (14 - 7)), short (top_k
    # 256, a narrow call) 14 of the narrow unit (>= 1.7 x 7)
    long = dict(
        num_queries=8192, num_keys=73728, num_segments=1, ratio=1, top_k=2048, grid=152
    )
    short = dict(long, num_keys=11264, top_k=256)
    for call in (long, short):
        decided = select_program(policy, scan_keys=keys, **call)
        assert decided.physical_kind == "wide" and not decided.lsf, (
            call,
            decided.scan_key,
        )
        assert not select_program(policy_for(arch), scan_keys=keys, **call).lsf, call
    c = select_fp8_program(policy, scan_keys=keys, **long)
    assert c.scan_key == copy and c.lsf, c.scan_key
    c = select_fp8_program(policy, scan_keys=keys - {copy}, **long)
    assert c.scan_key == plain and not c.lsf, c.scan_key
    c = select_fp8_program(policy, scan_keys=keys, **short)
    assert c.scan_key == plain and not c.lsf, c.scan_key
    # four equal packed documents of 120 mean unit tiles (top_k 256): the plan keeps the plain order, and so does the
    # FP8 dispatch
    packed = dict(long, num_keys=65536, num_segments=4, max_seqlen_k=16384, top_k=256)
    decided = select_program(policy, scan_keys=keys, **packed)
    assert decided.physical_kind == "wide" and not decided.lsf, decided.scan_key
    c = select_fp8_program(policy, scan_keys=keys, **packed)
    assert c.scan_key == plain and not c.lsf, c.scan_key


@pytest.mark.parametrize(
    "arch, grid", [("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 212)]
)
def test_fp8_wide_call_runs_the_wide_or_l6_program(arch, grid):
    """s1_8k_on_64k (8192 queries over 73728 keys, one segment, top_k 2048): the BF16 record plans a wide unit (on
    sm_107a the sampled call drops its CTA pair, ``wide_pair_snake_only``), and the FP8 form runs its wide program (8
    queries per unit) where the architecture registers one (``_FP8_WIDE_ARCHES``: sm_100a, sm_103a and sm_107a; on
    sm_103a its :lsf copy, ``lsf_single_rule``), else its l6 program (``FP8_KIND_FALLBACK``), unpaired and unsplit,
    rather than the narrow one."""
    keys = set((cake_jit.PROGRAM_KEYS or {}).get(arch) or ())
    if not any(scan_key_form(k) == FP8 for k in keys):
        pytest.skip(f"this checkout registers no FP8 scan program for {arch}")
    call = dict(
        num_queries=8192,
        num_keys=73728,
        num_segments=1,
        ratio=1,
        top_k=2048,
        grid=grid,
        scan_keys=keys,
    )
    bf16 = select_program(DispatchPolicy.from_record(cake_jit.POLICY[arch]), **call)
    assert bf16.physical_kind in ("wide", "pair_wide"), bf16.scan_key
    fp8 = select_fp8_program(_fp8_policy(arch), **call)
    key, block_q = (
        ("scan:wide:u2:s0:f1:fp8", 8)
        if arch in _FP8_WIDE_ARCHES
        else ("scan:l6:u1:s0:f1:fp8", 6)
    )
    if (
        arch == "sm_103a"
    ):  # one document of 544 mean unit tiles: the :lsf copy (lsf_single_rule)
        key += ":lsf"
    assert fp8.scan_key == key, fp8.scan_key
    assert (fp8.block_q, fp8.pair, fp8.n_split, fp8.grid_ctas) == (
        block_q,
        False,
        1,
        grid,
    )


def test_fp8_wide_fallback_takes_the_sampling_class_of_l6():
    """s2_4x2k_packed on sm_100a (four segments of 2048 queries over 229376 keys, the longest 100352): the BF16 record
    plans a wide unit with the wide-unit sample shift of ``sample_class_rule``; with the FP8 wide programs hidden (as
    where none is registered), the FP8 l6 program that runs instead takes the l6 kind's shift, as an l6 decision of the
    call would (sm_100a's record keys no check period on the kind: the next test covers sm_103a's).  Explicit knobs
    stay."""
    keys = set((cake_jit.PROGRAM_KEYS or {}).get("sm_100a") or ())
    if not any(scan_key_form(k) == FP8 for k in keys):
        pytest.skip("this checkout registers no FP8 scan program for sm_100a")
    policy = _fp8_policy("sm_100a")
    programs = _fp8_programs_without_wide(keys)
    call = dict(
        num_queries=8192,
        num_keys=229376,
        num_segments=4,
        ratio=1,
        top_k=2048,
        grid=148,
        max_seqlen_k=100352,
        scan_keys=keys,
    )
    decided = select_program(policy, **call)
    assert decided.physical_kind == "wide", decided.scan_key
    assert decided.sample_shift_permille == policy.sample_class_rule[1]
    c = select_fp8_program(policy, fp8_programs=programs, **call)
    assert c.scan_key == "scan:l6:u1:s0:f1:fp8:lsf", c.scan_key
    assert c.sample_shift_permille == policy.sample_shift_for(2048)
    assert c.sample_tiles_max == decided.sample_tiles_max
    assert c.check_period == policy.check_period_for(2048, c.cand_cap, "l6")
    knobbed = select_fp8_program(
        policy, fp8_programs=programs, sample_shift_permille=500, check_period=8, **call
    )
    assert (knobbed.sample_shift_permille, knobbed.check_period) == (500, 8)


def test_fp8_wide_fallback_takes_the_check_period_of_l6_on_sm103a():
    """sm_103a's record overrides the wide kind's check period (``check_period_kind_overrides``): with sm_103a's FP8
    wide programs hidden (as where none is registered), the FP8 l6 program run for a wide decision takes the l6 period
    instead.  The unsampled wide decisions of the geometry grid whose two periods differ must exist, and each one runs
    with the l6 period."""
    keys = set((cake_jit.PROGRAM_KEYS or {}).get("sm_103a") or ())
    if not any(scan_key_form(k) == FP8 for k in keys):
        pytest.skip("this checkout registers no FP8 scan program for sm_103a")
    policy = _fp8_policy("sm_103a")
    programs = _fp8_programs_without_wide(keys)
    seen = 0
    for T, Tkv, S, ratio, top_k in _geometries(policy):
        call = dict(
            num_queries=T,
            num_keys=Tkv,
            num_segments=S,
            ratio=ratio,
            top_k=top_k,
            grid=152,
            sample_tiles_max=0,
            scan_keys=keys,
        )
        decided = select_program(policy, **call)
        if decided.physical_kind not in ("wide", "pair_wide"):
            continue
        wide = policy.check_period_for(top_k, decided.cand_cap, decided.kind)
        l6 = policy.check_period_for(top_k, decided.cand_cap, "l6")
        if wide == l6:
            continue
        c = select_fp8_program(policy, fp8_programs=programs, **call)
        where = (T, Tkv, S, ratio, top_k)
        assert decided.check_period == wide, where
        assert (c.physical_kind, c.check_period) == ("l6", l6), where
        seen += 1
    assert seen, (
        "no unsampled wide decision of the grid has a check period other than l6's"
    )


def test_bf16_plans_and_workspace_bounds_are_the_unchanged_procedure():
    """A call without the FP8 dtype plans exactly what ``select_program`` gives under the architecture's BF16 record,
    explicitly or by default, and its workspace bound is unchanged."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        bf16_policy = DispatchPolicy.from_record(cake_jit.POLICY[arch])
        assert policy_for(arch) == bf16_policy and policy_for(arch, BF16) is policy_for(
            arch
        )
        for grid in _GRIDS[arch]:
            for T, Tkv, S, ratio, top_k in itertools.islice(
                _geometries(bf16_policy, steps_t=8, steps_m=7), 0, None, 7
            ):
                want = [
                    select_program(
                        bf16_policy,
                        num_queries=T,
                        num_keys=Tkv,
                        num_segments=S,
                        ratio=ratio,
                        top_k=top_k,
                        grid=grid,
                        sample_tiles_max=sample,
                        outputs_aligned=True,
                        scan_keys=keys,
                    )
                    for sample in (None, 0)
                ]
                common = dict(top_k=top_k, ratio=ratio, arch=arch, num_sms=grid)
                for sample, expected in zip((None, 0), want, strict=True):
                    got = plan_dsa_indexer_det_topk(
                        T, Tkv, S, sample_tiles_max=sample, **common
                    )
                    assert got == expected, (arch, grid, T, Tkv, S, top_k, sample)
                    assert scan_key_form(got.scan_key) == BF16
                    assert (
                        plan_dsa_indexer_det_topk(
                            T,
                            Tkv,
                            S,
                            sample_tiles_max=sample,
                            dtype=torch.bfloat16,
                            **common,
                        )
                        == expected
                    )
                nbytes = max(int(c.workspace_bytes) for c in want)
                assert (
                    cake_dsa_indexer_det_train_topk_workspace_size(T, Tkv, S, **common)
                    == nbytes
                )
                assert (
                    cake_dsa_indexer_det_train_topk_workspace_size(
                        T, Tkv, S, dtype=torch.bfloat16, **common
                    )
                    == nbytes
                )


def _fp8_host_inputs(total_q=6, total_k=10):
    q = torch.zeros((total_q, NUM_HEADS, HEAD_DIM), dtype=torch.uint8).view(E4M3)
    k = torch.zeros((total_k, HEAD_DIM), dtype=torch.uint8).view(E4M3)
    return dict(
        q=q,
        k=k,
        w=torch.zeros((total_q, NUM_HEADS), dtype=torch.float32),
        cu_seqlens_q=torch.tensor((0, 2, total_q), dtype=torch.int32),
        cu_seqlens_k=torch.tensor((0, 3, total_k), dtype=torch.int32),
        q_causal_offsets=None,
        top_k=4,
        softmax_scale=1.0,
        ratio=1,
        k_scale=torch.ones(total_k, dtype=torch.float32),
    )


def _validate(h):
    return validate_dsa_indexer_det_inputs(
        h["q"],
        h["k"],
        h["w"],
        h["cu_seqlens_q"],
        h["cu_seqlens_k"],
        h["q_causal_offsets"],
        h["top_k"],
        h["softmax_scale"],
        h["ratio"],
        k_scale=h["k_scale"],
    )


def _bf16(h):
    return dict(
        h,
        q=torch.zeros(h["q"].shape, dtype=torch.bfloat16),
        k=torch.zeros(h["k"].shape, dtype=torch.bfloat16),
        k_scale=None,
    )


def test_fp8_validation_reaches_the_device_rule_for_host_tensors():
    with pytest.raises(ValueError, match="CUDA"):
        _validate(_fp8_host_inputs())
    # the trainer's row-strided key view: 704 one-byte elements per row
    h = _fp8_host_inputs()
    h["k"] = torch.zeros((10, ref.TRAINER_KEY_ROW_STRIDE), dtype=torch.uint8).view(
        E4M3
    )[:, :HEAD_DIM]
    with pytest.raises(ValueError, match="CUDA"):
        _validate(h)
    with pytest.raises(ValueError, match="CUDA"):
        _validate(_bf16(_fp8_host_inputs()))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda h: h.update(k_scale=None), "need k_scale"),
        (lambda h: h.update(k_scale=torch.ones(9)), r"k_scale must be \[Tkv"),
        (lambda h: h.update(k_scale=torch.ones(10, 1)), r"k_scale must be \[Tkv"),
        (lambda h: h.update(k_scale=torch.ones(10, dtype=torch.float64)), "float32"),
        (lambda h: h.update(k_scale=torch.ones(20)[::2]), "contiguous"),
        (
            lambda h: h.update(k=torch.zeros((10, HEAD_DIM), dtype=torch.bfloat16)),
            "k must be float8_e4m3fn",
        ),
        (
            lambda h: h.update(
                k=torch.zeros((10, HEAD_DIM), dtype=torch.uint8).view(torch.float8_e5m2)
            ),
            "k must be float8_e4m3fn",
        ),
        (
            lambda h: h.update(q=h["q"].view(torch.uint8).view(torch.float8_e5m2)),
            "q must be bfloat16 or float8_e4m3fn",
        ),
        (
            lambda h: h.update(
                k=torch.zeros((10, 136), dtype=torch.uint8).view(E4M3)[:, :HEAD_DIM]
            ),
            "row stride",
        ),
        (lambda h: h.update(q=h["q"][:, :16]), r"q must be \[T"),
    ],
)
def test_fp8_validation_rejects(mutate, match):
    h = _fp8_host_inputs()
    mutate(h)
    with pytest.raises(ValueError, match=match):
        _validate(h)


def test_bf16_validation_rejects_fp8_only_arguments_and_keeps_its_messages():
    h = _bf16(_fp8_host_inputs())
    with pytest.raises(ValueError, match="bfloat16 q / k take none"):
        _validate(dict(h, k_scale=torch.ones(10)))
    with pytest.raises(ValueError, match="k must be bfloat16, got torch.float8_e4m3fn"):
        _validate(dict(h, k=_fp8_host_inputs()["k"]))
    with pytest.raises(ValueError, match="bfloat16"):
        _validate(dict(h, q=h["q"].float()))


def test_fp8_prepare_rules_on_host_tensors():
    """The FP8 form folds the softmax scale into ``w``: a given ``softmax_scale`` is rejected before any device work,
    and ``None`` reaches the device rule (the programs take 1.0); a BF16 call with ``k_scale`` is rejected."""
    h = _fp8_host_inputs()
    args = [h.pop(name) for name in ("q", "k", "w", "cu_seqlens_q", "cu_seqlens_k")]
    for name in ("q_causal_offsets", "softmax_scale", "top_k", "ratio"):
        h.pop(name)
    with pytest.raises(ValueError, match="softmax_scale=None"):
        prepare_dsa_indexer_det_topk(*args, softmax_scale=0.5, **h)
    with pytest.raises(ValueError, match="CUDA"):
        prepare_dsa_indexer_det_topk(*args, **h)
    bf16 = _bf16(_fp8_host_inputs())
    bf16_args = [bf16[name] for name in ("q", "k", "w", "cu_seqlens_q", "cu_seqlens_k")]
    with pytest.raises(ValueError, match="k_scale"):
        prepare_dsa_indexer_det_topk(*bf16_args, k_scale=torch.ones(10))
    with pytest.raises(ValueError, match="CUDA"):
        prepare_dsa_indexer_det_topk(*bf16_args)


# ---------------------------------------------------------------------------
# Operator (GPU)
# ---------------------------------------------------------------------------


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_fp8_program():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda"), dtype=E4M3):
        pytest.skip("generated FP8 DSA indexer programs not registered for this device")


def _run(inputs: ref.IndexerInputs, **extra):
    out = cake_backend.cake_dsa_indexer_det_train_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        **extra,
    )
    assert isinstance(out, tuple) and len(out) == 2
    return out


def _prepare(inputs: ref.IndexerInputs, **extra):
    return prepare_dsa_indexer_det_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        **extra,
    )


def _plan(inputs: ref.IndexerInputs, **extra):
    return plan_dsa_indexer_det_topk(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=inputs.device,
        dtype=inputs.q.dtype,
        **extra,
    )


def _cloned(out):
    return out[0].clone(), out[1].clone()


def _judge_ok(result, reference) -> ref.Judgement:
    verdict = ref.judge(result, reference)
    assert verdict.passed, str(verdict)
    return verdict


def _scan_argument(runner, name: str):
    (scan,) = [launch for launch in runner.launches if launch.role == "scan"]
    plan = cake_backend.ARG_PLANS["scan"]
    (index,) = [i for i, (_kind, arg) in enumerate(plan) if arg == name]
    return scan.arguments[index]


# s1_8k_on_64k (8192 queries over 73728 keys, one segment, top_k 2048): the BF16 planner decides a wide unit, which the
# FP8 wide program runs where the architecture registers one
_S1 = dict(seg_q_len=[8192], seg_k_len=[73728], top_k=2048)


def _assert_fp8_wide_on_wide_arches(plan):
    """A wide-decision case (``_S1``) must run the FP8 wide program on the architectures of ``_FP8_WIDE_ARCHES``."""
    if arch_for(torch.device("cuda")) in _FP8_WIDE_ARCHES:
        assert plan.physical_kind == "wide", plan.scan_key


_ACCURACY = {
    "single": dict(seg_q_len=[512], seg_k_len=[8192], top_k=2048),
    "s1_8k_on_64k": _S1,
    "split_tail": dict(seg_q_len=[4], seg_k_len=[22018], top_k=2048),
    "packed_unequal": dict(
        seg_q_len=[600, 400, 300], seg_k_len=[600, 12000, 5000], top_k=1024
    ),
    "empty_segments": dict(seg_q_len=[100, 0, 50], seg_k_len=[0, 300, 4000], top_k=256),
    "ratio2_offsets": dict(
        seg_q_len=[64, 64],
        seg_k_len=[2048, 4096],
        top_k=512,
        ratio=2,
        q_causal_offsets=[100, -3],
    ),
    "top_k_1": dict(seg_q_len=[256], seg_k_len=[3000], top_k=1),
    "top_k_4096": dict(seg_q_len=[300], seg_k_len=[9000], top_k=4096),
    "strided_k": dict(
        seg_q_len=[256],
        seg_k_len=[5000],
        top_k=2048,
        k_row_stride=ref.TRAINER_KEY_ROW_STRIDE,
    ),
}


@pytest.mark.parametrize("name", sorted(_ACCURACY))
def test_fp8_random_accuracy_within_bound(name):
    """FP8 rows against the FP64 score of the e4m3 inputs: every returned score within ``gamma_n * A`` and the
    selected ids exact up to near ties; repeated calls are bitwise identical."""
    _require_fp8_program()
    inputs = ref.make_fp8_inputs(seed=31, device="cuda", label=name, **_ACCURACY[name])
    plan = _plan(inputs)
    assert scan_key_form(plan.scan_key) == FP8 and not plan.pair
    if _ACCURACY[name] is _S1:
        _assert_fp8_wide_on_wide_arches(plan)
    out = _cloned(_run(inputs))
    torch.cuda.synchronize()
    _judge_ok(out, ref.select_reference(inputs))
    assert ref.same_bits(_run(inputs), out)


def test_fp8_empty_rows_and_no_keys_are_all_padding():
    _require_fp8_program()
    inputs = ref.make_fp8_inputs([5], [0], top_k=16, device="cuda")
    indices, scores = _run(inputs)
    assert (indices == -1).all() and torch.isneginf(scores).all()
    empty = ref.make_fp8_inputs([0], [100], top_k=16, device="cuda")
    indices, scores = _run(empty)
    assert indices.shape == (0, 16) and scores.shape == (0, 16)


# two segments of unequal key length, each with more units than CTAs: the longest-first order and the ticket draws
# that continue into the shorter segment differ from the plain order's
_UNEQUAL2 = dict(seg_q_len=[4096, 4096], seg_k_len=[200000, 20000], top_k=2048)
# the rows of ``_UNEQUAL2`` judged against the reference, (segment, first row in the segment, rows): the first and the
# last rows of each segment
_UNEQUAL2_JUDGED = [(0, 0, 64), (0, 4032, 64), (1, 0, 64), (1, 4032, 64)]


def _row_blocks_reference(inputs, blocks):
    """The positions in ``inputs`` of the rows ``blocks`` (segment, first row in the segment, rows) and their
    reference: every block is one segment of a smaller problem over its segment's keys, with the causal offset
    moved by the block's first row, so each row sees the keys it sees in ``inputs``."""
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    offsets = inputs.offsets()
    rows = torch.cat(
        [
            torch.arange(
                q_starts[s] + first, q_starts[s] + first + n, device=inputs.device
            )
            for s, first, n in blocks
        ]
    )
    keys = [(k_starts[s], k_starts[s + 1]) for s, _first, _n in blocks]
    sub = ref.build_inputs(
        inputs.q.index_select(0, rows),
        torch.cat([inputs.k[a:b] for a, b in keys]),
        inputs.w.index_select(0, rows),
        [n for _s, _first, n in blocks],
        [inputs.seg_k_len[s] for s, _first, _n in blocks],
        top_k=inputs.top_k,
        softmax_scale=inputs.softmax_scale,
        q_causal_offsets=[offsets[s] + first for s, first, _n in blocks],
        ratio=inputs.ratio,
        k_scale=torch.cat([inputs.k_scale[a:b] for a, b in keys]),
    )
    return rows, ref.select_reference(sub)


@pytest.mark.parametrize(
    "shape",
    [
        dict(seg_q_len=[700, 300, 200], seg_k_len=[700, 9000, 4000], top_k=1024),
        _S1,
        _UNEQUAL2,
    ],
    ids=["packed", "s1_8k_on_64k", "unequal_two_segments"],
)
def test_fp8_dispatch_invariance_over_every_registered_program(monkeypatch, shape):
    """Every FP8 program of the device's architecture, forced on the same call (the planned capacity, check period,
    sampling and finalize; the split and unit geometry of the forced key), returns the dispatched program's ids and
    score bits: within one architecture the dispatched FP8 program never changes a result.  The packed call is judged
    against the reference, ``_UNEQUAL2`` on the rows of ``_UNEQUAL2_JUDGED``; on ``_S1`` and ``_UNEQUAL2`` every unit
    kind outnumbers the CTAs, so a drawing program draws tickets, and on ``_UNEQUAL2`` the draws cross from one segment
    into the other in both segment orders."""
    _require_fp8_program()
    arch = arch_for(torch.device("cuda"))
    policy = policy_for(arch, FP8)
    keys = sorted(
        key for key in cake_jit.PROGRAM_KEYS.get(arch, {}) if scan_key_form(key) == FP8
    )
    inputs = ref.make_fp8_inputs(seed=41, device="cuda", **shape)
    rows, reference = None, None
    if shape is _UNEQUAL2:
        rows, reference = _row_blocks_reference(inputs, _UNEQUAL2_JUDGED)
    elif shape is not _S1:
        reference = ref.select_reference(inputs)
    original = cake_backend.plan_dsa_indexer_det_topk
    for sample in (None, 0):
        extra = {} if sample is None else {"sample_tiles_max": sample}
        expected = _cloned(_run(inputs, **extra))
        torch.cuda.synchronize()
        if rows is not None:
            _judge_ok(tuple(x.index_select(0, rows) for x in expected), reference)
        elif reference is not None:
            _judge_ok(expected, reference)
        for key in keys:

            def forced(*args, _key=key, **kwargs):
                return fp8_program_choice(policy, original(*args, **kwargs), _key)

            monkeypatch.setattr(cake_backend, "plan_dsa_indexer_det_topk", forced)
            runner = _prepare(inputs, **extra)
            assert runner.choice.scan_key == key and runner.form == FP8
            assert runner.programs["scan"] == cake_jit.PROGRAM_KEYS[arch][key]
            got = _cloned(runner())
            torch.cuda.synchronize()
            assert ref.same_bits(got, expected), (arch, key, sample)
            monkeypatch.setattr(cake_backend, "plan_dsa_indexer_det_topk", original)


def test_fp8_prepared_runner_binds_k_scale():
    """The runner binds ``k_scale`` with ``q`` / ``k``: values written into it are read by the next run (doubling every
    scale doubles every score exactly and keeps the ids); a BF16 runner binds the never-read placeholder."""
    _require_fp8_program()
    inputs = ref.make_fp8_inputs(
        [300, 200], [3000, 6000], top_k=512, seed=43, device="cuda"
    )
    runner = _prepare(inputs)
    assert runner.form == FP8
    assert _scan_argument(runner, "K_scale").data_ptr() == inputs.k_scale.data_ptr()
    assert (
        runner.tensors["Q"].dtype == torch.uint8
        and runner.tensors["K"].dtype == torch.uint8
    )
    first = _cloned(runner())
    torch.cuda.synchronize()
    _judge_ok(first, ref.select_reference(inputs))
    inputs.k_scale.mul_(2.0)
    second = _cloned(runner())
    torch.cuda.synchronize()
    assert torch.equal(second[0], first[0])
    assert torch.equal(second[1], first[1] * 2.0)
    bf16 = ref.dequantized_bf16(inputs)
    bf16_runner = _prepare(bf16)
    assert bf16_runner.form == BF16
    assert _scan_argument(bf16_runner, "K_scale") is cake_backend._unread_placeholder(
        bf16_runner.device_index, torch.float32
    )


def test_fp8_ticket_decision_is_per_form_and_exact_on_a_poisoned_workspace():
    """The FP8 form decides its ticket tail by its own record: on sm_107a every FP8 program keeps the static unit
    enumeration (no tail, no ticket block, the device's placeholder); elsewhere a drawing FP8 program gets the tail
    (zeroed by the scan binding on a caller-owned buffer).  Rows stay bit-exact on a caller-owned buffer poisoned
    everywhere.  The last shape (``_S1``) runs the FP8 wide program where the architecture registers one, and its
    units outnumber the CTAs, so that program draws tickets on sm_100a / sm_103a (on sm_107a it keeps the static unit
    enumeration like every FP8 program there)."""
    _require_fp8_program()
    device = torch.device("cuda", torch.cuda.current_device())
    arch = arch_for(device)
    static_kinds = set(policy_for(arch, FP8).static_scan_kinds)
    shapes = (
        (8192, 32768, 1),
        (4096, 16384, 2),
        (2048, 65536, 1),
        (4, 22018, 1),
        (8192, 73728, 1),
    )
    for (T, Tkv, S), sample in itertools.product(shapes, (None, 0)):
        inputs = ref.make_fp8_inputs(
            [T // S] * S, [Tkv // S] * S, top_k=2048, seed=47, device="cuda"
        )
        extra = {} if sample is None else {"sample_tiles_max": sample}
        expected = _cloned(_run(inputs, **extra))
        nbytes = cake_dsa_indexer_det_train_topk_workspace_size(
            T, Tkv, S, top_k=2048, device=device, dtype=E4M3
        )
        shared = torch.full((nbytes,), 7, dtype=torch.uint8, device=device)
        runner = _prepare(inputs, workspace_buffer=shared, **extra)
        if (T, Tkv, S, sample) == (8192, 73728, 1, None):
            _assert_fp8_wide_on_wide_arches(runner.choice)
        static_kind = runner.choice.physical_kind in static_kinds
        reads = cake_backend.reads_ticket_word(runner.programs["scan"], arch)
        if arch == "sm_107a":
            assert static_kind, runner.choice.scan_key
        # a static kind has no tail; a program reading its TicketWord draws from a block of its own instead
        assert (runner.ticket_tail is None) == (static_kind or reads), (
            runner.choice.scan_key
        )
        assert (runner.ticket_block is not None) == (
            not static_kind and not runner._tail_static and reads
        )
        if runner.ticket_block is None:
            assert _scan_argument(
                runner, "TicketWord"
            ) is cake_backend._ticket_placeholder(runner.device_index)
        for _ in range(2):
            out = runner()
            torch.cuda.synchronize()
            assert ref.same_bits(out, expected), (
                runner.choice.scan_key,
                T,
                Tkv,
                S,
                sample,
            )
        if runner.ticket_tail is not None and not runner._tail_static:
            assert not runner.ticket_tail[: 8 * 16].any()


@pytest.mark.parametrize(
    "shape",
    [dict(seg_q_len=[900, 300], seg_k_len=[900, 7000], top_k=2048), _S1],
    ids=["packed", "s1_8k_on_64k"],
)
def test_fp8_workspace_bound_is_respected(shape):
    _require_fp8_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = ref.make_fp8_inputs(seed=53, device="cuda", **shape)
    if shape is _S1:
        _assert_fp8_wide_on_wide_arches(_plan(inputs))
    nbytes = cake_dsa_indexer_det_train_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=device,
        dtype=E4M3,
    )
    assert nbytes == max(
        int(_plan(inputs, sample_tiles_max=s).workspace_bytes) for s in (None, 0)
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    runner = _prepare(inputs, workspace_buffer=workspace)
    assert runner.workspace_bytes <= nbytes
    runner()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    out = runner()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    _judge_ok(out, ref.select_reference(inputs))
    with pytest.raises(ValueError, match="workspace_buffer needs"):
        _prepare(inputs, workspace_buffer=workspace[: runner.workspace_bytes - 8])


def test_fp8_prepared_runner_graph_capture_and_replay():
    _require_fp8_program()
    inputs = ref.make_fp8_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=59, device="cuda"
    )
    eager = _cloned(_run(inputs))
    runner = _prepare(inputs)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            runner()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner()
    graph.replay()
    torch.cuda.synchronize()
    assert ref.same_bits(captured, eager)
    # new values in the bound tensors (k_scale included): the replay reads them
    fresh = ref.make_fp8_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=61, device="cuda"
    )
    inputs.q.view(torch.uint8).copy_(fresh.q.view(torch.uint8))
    inputs.k.view(torch.uint8).copy_(fresh.k.view(torch.uint8))
    inputs.w.copy_(fresh.w)
    inputs.k_scale.copy_(fresh.k_scale)
    graph.replay()
    torch.cuda.synchronize()
    assert ref.same_bits(_cloned(captured), _run(fresh))


def test_fp8_k32_logits_match_deep_gemm_bitwise(monkeypatch):
    """Diagnostic: on sm_100a / sm_103a the FP8 programs accumulate in DeepGEMM's four K = 32 steps, so every score
    returned by the dispatched program, and by every FP8 program of the architecture forced on the same call, equals
    the stock DeepGEMM FP8 MQA logit of its (query, key) pair bit for bit (up to the sign of zero).  The logits come
    from ``deep_gemm.fp8_mqa_logits``, or in a build without it from ``deep_gemm.fp8_fp4_mqa_logits`` with
    the same e4m3 ``q`` / ``k``, FP32 per-key scales and FP32 logits.  Skips on sm_107a (two K = 64 steps; last-bit
    differences allowed), where DeepGEMM does not import, where the build has neither entry point, where an entry
    point takes other arguments, and where ``fp8_fp4_mqa_logits`` rejects FP32 per-key scales on this architecture
    (DeepGEMM builds that take MX block scale factors and return BF16 logits there: a different input format)."""
    _require_fp8_program()
    arch = arch_for(torch.device("cuda"))
    if arch not in _K32_ARCHES:
        pytest.skip(f"{arch} accumulates in K = 64 steps: no bitwise DeepGEMM claim")
    if importlib.util.find_spec("deep_gemm") is None:
        pytest.skip("deep_gemm is not installed")
    try:
        import deep_gemm
    except (ImportError, OSError) as exc:  # installed, but its extension does not load
        pytest.skip(f"deep_gemm does not import: {exc}")

    inputs = ref.make_fp8_inputs([512], [8192], top_k=2048, seed=67, device="cuda")
    indices, scores = _run(inputs)
    visible = ref.visible_rows(inputs).to(torch.int32)
    starts = torch.zeros_like(visible)
    kv = (inputs.k.contiguous(), inputs.k_scale)
    if hasattr(deep_gemm, "fp8_mqa_logits"):
        try:
            logits = deep_gemm.fp8_mqa_logits(
                inputs.q, kv, inputs.w, starts, visible, clean_logits=False
            )
        except TypeError as err:  # an API this diagnostic does not know
            pytest.skip(f"deep_gemm.fp8_mqa_logits is not callable as expected: {err}")
    elif hasattr(deep_gemm, "fp8_fp4_mqa_logits"):
        try:
            logits = deep_gemm.fp8_fp4_mqa_logits(
                q=(inputs.q, None),
                kv=kv,
                weights=inputs.w,
                cu_seq_len_k_start=starts,
                cu_seq_len_k_end=visible,
                clean_logits=False,
                max_seqlen_k=0,
                logits_dtype=torch.float32,
            )
        except TypeError as err:  # an API this diagnostic does not know
            pytest.skip(
                f"deep_gemm.fp8_fp4_mqa_logits is not callable as expected: {err}"
            )
        except RuntimeError as err:  # DeepGEMM's own argument checks
            if "Assertion" not in str(err):
                raise
            pytest.skip(
                f"this deep_gemm build's fp8_fp4_mqa_logits takes no FP32 per-key scales on {arch}: {err}"
            )
    else:
        pytest.skip(
            "this deep_gemm build has neither fp8_mqa_logits nor fp8_fp4_mqa_logits"
        )
    logits = logits[:, : inputs.num_keys].float()

    def check(indices, scores, program):
        valid = indices >= 0
        picked = logits.gather(1, indices.clamp_min(0).to(torch.int64))
        got = torch.where(valid, scores + 0.0, torch.zeros_like(scores))
        want = torch.where(valid, picked + 0.0, torch.zeros_like(picked))
        mismatch = got.view(torch.int32) != want.view(torch.int32)
        assert not mismatch.any(), (
            f"{program}: {int(mismatch.sum())} scores differ from the DeepGEMM logits; first at "
            f"{mismatch.nonzero()[0].tolist()}"
        )

    check(indices, scores, "the dispatched program")
    # every FP8 program forced on the call as in the dispatch invariance test (the planned capacity, check period,
    # sampling and finalize; the split and unit geometry of the forced key)
    policy = policy_for(arch, FP8)
    keys = sorted(
        key
        for unrolls in cake_backend.fp8_programs_for(arch).values()
        for key in unrolls.values()
    )
    original = cake_backend.plan_dsa_indexer_det_topk
    for key in keys:

        def forced(*args, _key=key, **kwargs):
            return fp8_program_choice(policy, original(*args, **kwargs), _key)

        monkeypatch.setattr(cake_backend, "plan_dsa_indexer_det_topk", forced)
        runner = _prepare(inputs)
        assert runner.choice.scan_key == key and runner.form == FP8
        check(*runner(), key)
        monkeypatch.setattr(cake_backend, "plan_dsa_indexer_det_topk", original)
