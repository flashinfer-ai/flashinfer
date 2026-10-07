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

# Contract tests of the experimental Cake DSA indexer top-k backend
# (flashinfer-ai/flashinfer#5676).  The host-side tests (registry, dispatch
# policy, validation, binding, the reference itself) run without a GPU; the
# operator tests need a compute capability 10.0 / 10.3 / 10.7 device with
# registered generated programs and skip otherwise.  Opt-in heavy variant:
# ``FLASHINFER_CAKE_DSA_INDEXER_FULL=1`` runs the recorded packed key geometry
# through the FP64 reference (minutes on B200).

import contextlib
import os
import warnings
from types import SimpleNamespace

import pytest
import torch
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_dsa_indexer import cake_backend, cake_jit, cake_policy
from flashinfer.experimental.cake_dsa_indexer.cake_backend import (
    ABI_CONTRACT,
    CONTRACT_SCALARS,
    CONTRACT_TENSORS,
    HEAD_DIM,
    MAX_TOP_K,
    NUM_HEADS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    ZERO_SIGN_POLICIES,
    arch_for,
    bind_program,
    dsa_indexer_workspace_size,
    generated_program_available,
    plan_dsa_indexer_topk,
    policy_for,
    prepare_dsa_indexer_topk,
    program_for,
    record_zero_sign_policy,
    validate_dsa_indexer_inputs,
    visible_key_count,
    visible_key_counts,
)
from flashinfer.experimental.cake_dsa_indexer.cake_policy import (
    MERGE_KEY,
    POLICY_FIELDS,
    DispatchPolicy,
    finalize_grid,
    finalize_key,
    scan_key,
    select_program,
    stage_slug,
)

from tests.test_helpers import cake_dsa_indexer_reference as ref

FULL = os.environ.get("FLASHINFER_CAKE_DSA_INDEXER_FULL", "0") not in (
    "",
    "0",
    "false",
    "no",
)

# Dispatch policies in the record form (the registry record carries the
# authoritative values of the exported programs; these two drive the host-only
# tests of the policy evaluation itself).  ``NARROW_PAIR_POLICY`` is the shape
# of a B200-class rule set, ``SNAKE_UNROLL_POLICY`` the shape of a rule set with
# the snake unit order, the tile-loop unroll and the two-way wave split.
_COMMON = dict(
    tile_keys=128,
    block_q_narrow=4,
    block_q_wide=8,
    block_q_l6=6,
    candidate_entry_bytes=8,
    candidate_multiplier=4,
    candidate_slack=128,
    cand_mult_rule=[384.0, 8],
    cand_cap_floor=8192,
    split_max=32,
    split_min_range_tiles=64,
    snake_default=False,
    tile_unroll_default=1,
    tile_unroll_factor=2,
    sample_fit_max_mean_tiles=512,
    sample_tiles_max=32,
    sample_tiles_short_units=16,
    sample_dispatch_mean_tiles_max=640,
    sample_tiles_tiny_units=8,
    sample_dispatch_tiny_tiles_max=128,
    sample_shift_permille=250,
    check_period_max=32,
    check_period_knob_max=64,
    check_period_cap_divisor=512,
    check_period_kind_overrides={},
    finalize_items=8,
    finalize_threads_fit=[32, 64, 128],
    finalize_threads_small=256,
    finalize_top_k_small=2048,
    finalize_threads=512,
    finalize_fit=True,
    finalize_exact_key_bits=True,
    rank_finalize=True,
    rank_window_variants=[8192, 16384, 65536, 131072, 262144, 524288],
    rank_top_k_min=1025,
    rank_staged=False,
    rank_seg_window=False,
    rank_window_variants_seg_only=[131072],
    rank_two_level=False,
    rank_two_level_window_variants=[131072, 262144, 524288, 1048576],
    rank_slab_window_max=65536,
    rank_bulk_io=False,
    rank_bulk_align_bytes=16,
    rank_t16=False,
    rank_t16_slots=[4096],
    rank_persist_max_k=0,
    rank_persist_ctas_per_sm=4,
)
NARROW_PAIR_POLICY = DispatchPolicy.from_record(
    dict(
        _COMMON,
        l6_rule=[256.0, 512],
        wide_rule=[6.0, 300.0],
        split_wave_rule=None,
        pair_rule=[128.0, 256.0, 512],
        snake_rule=None,
        tile_unroll_rule=None,
        rank_rule=[524288, 262144],
        rank_staged_rule=[262144],
        rank_two_level_rule=[524288],
        rank_two_level_staged_rule=[524288],
    )
)
SNAKE_UNROLL_POLICY = DispatchPolicy.from_record(
    dict(
        _COMMON,
        l6_rule=[64.0, 0],
        wide_rule=[2.0, 250.0],
        split_wave_rule=[1000, 0.12],
        pair_rule=[0.0, 4096.0, 512],
        snake_rule=[
            ["narrow", "wide", "l6", "pair_narrow", "pair_wide", "pair_l6"],
            32.0,
            128.0,
            0.5,
        ],
        tile_unroll_rule=[
            ["narrow", "wide", "split_wide", "pair_narrow", "pair_wide"],
            512,
            True,
        ],
        rank_rule=[524288, 262144],
        rank_staged_rule=[262144],
        rank_two_level_rule=[524288],
        rank_two_level_staged_rule=[524288],
    )
)
P1_CU_SEQLENS_Q = (0, 1777, 3532, 5655, 7765, 9888, 11998, 14121, 16231)
P1_CU_SEQLENS_K = (0, 1777, 58619, 60742, 128665, 130788, 198711, 200834, 268757)
P2_T = 16172


def _lengths(cu):
    return [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)]


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated DSA indexer programs not registered for this device")


@contextlib.contextmanager
def _quiet_experimental():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


def _run(inputs: ref.IndexerInputs, **extra):
    out = cake_backend.dsa_indexer_topk(
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


def _plan(inputs: ref.IndexerInputs, **extra):
    return plan_dsa_indexer_topk(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=inputs.device,
        **extra,
    )


def _judge_ok(result, reference, **kw) -> ref.Judgement:
    if kw.get("exact_bits") and "zero_sign_policy" not in kw:
        kw["zero_sign_policy"] = record_zero_sign_policy()
    verdict = ref.judge(result, reference, **kw)
    assert verdict.passed, str(verdict)
    return verdict


def _independent_membership_check(
    inputs: ref.IndexerInputs, indices: torch.Tensor
) -> None:
    """Visibility and segment membership from the scalar rule in Python (no tensors shared with the reference)."""
    ids = indices.cpu().tolist()
    K = inputs.top_k
    offsets = inputs.offsets()
    starts = inputs.query_starts()
    for s in range(inputs.num_segments):
        for u in range(inputs.seg_q_len[s]):
            visible = visible_key_count(
                offsets[s], u, inputs.ratio, inputs.seg_k_len[s]
            )
            n = min(K, visible)
            row = ids[starts[s] + u]
            assert all(0 <= j < visible for j in row[:n]), (s, u, row[:8])
            assert all(j == -1 for j in row[n:]), (s, u)
            assert row[:n] == sorted(set(row[:n])), (s, u)


# ---------------------------------------------------------------------------
# Host-only: registry, dispatch policy, validation, binding
# ---------------------------------------------------------------------------


def test_public_api_is_marked_experimental():
    from flashinfer.dsa_indexer import dsa_indexer_topk

    assert dsa_indexer_topk.is_experimental


def test_registry_records_are_well_formed():
    if not cake_jit.PROGRAM_KEYS:
        # placeholder registry: every architecture reports the missing programs by name
        for arch in cake_jit.ARCHES:
            with pytest.raises(NotImplementedError, match="5676"):
                program_for(arch, "scan:narrow:u1:s0:f0")
            with pytest.raises(NotImplementedError, match="5676"):
                policy_for(arch)
        return
    assert cake_jit.ABI == ABI_CONTRACT
    assert record_zero_sign_policy() in ZERO_SIGN_POLICIES
    assert set(cake_jit.ARG_PLANS) == set(cake_jit.STAGES)
    assert (
        {"scan", "merge", "finalize"}
        <= set(cake_jit.STAGES)
        <= {"scan", "merge", "finalize", "finalize_rank"}
    )
    for role, plan in cake_jit.ARG_PLANS.items():
        for kind, arg in plan:
            assert kind in (
                "buffer",
                "tma_buffer",
                "raw_pointer",
                "nullable_raw_pointer",
                "parameter",
                "grid",
            ), (role, kind, arg)
            if kind != "grid":
                key = cake_backend.CONTRACT_ALIASES.get(arg, arg)
                assert (
                    key in CONTRACT_TENSORS
                    or key in CONTRACT_SCALARS
                    or key in ("num_queries", "num_keys")
                ), (role, arg)
    root = cake_jit.__file__.rsplit("/", 1)[0] + "/csrc"
    for name, record in cake_jit.PROGRAMS.items():
        assert record["role"] in cake_jit.STAGES, name
        assert record["arches"] and set(record["arches"]) <= set(cake_jit.ARCHES), name
        assert all(
            os.path.isfile(os.path.join(root, src)) for src in record["sources"]
        ), name
        assert (
            len(record["launch"]["block"]) == 3
            and len(record["launch"]["cluster"]) == 3
        ), name
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        assert arch in cake_jit.ARCHES
        policy = policy_for(arch)
        assert set(cake_jit.POLICY[arch]) == set(POLICY_FIELDS)
        scan_keys = sorted(k for k in keys if k.startswith("scan:"))
        finalize_keys = sorted(k for k in keys if k.startswith("finalize:"))
        rank_keys = sorted(k for k in keys if k.startswith("finalize_rank:"))
        assert scan_keys and finalize_keys and MERGE_KEY in keys, arch
        assert set(keys) == set(scan_keys) | set(finalize_keys) | set(rank_keys) | {
            MERGE_KEY
        }, arch
        # the rank finalize programs are registered exactly where the architecture's rule admits a call
        assert bool(rank_keys) == (
            policy.rank_finalize and policy.rank_rule is not None
        ), arch
        for key in rank_keys:
            _role, threads, window, *form = key.split(":")
            # lever FRB: the bulk row I/O carries a trailing ``:bulk``; lever FRT: the 16-item thread form ``:i16`` -- both forms of the
            # staged program (``threads`` is the launched count); lever FRS: ``:staged``; lever FR2: the two-level form ``:two``
            tags = list(form)
            # lever FRP-K: the persistent two-row pipelined form of a bulk program carries a trailing ``:persist`` (sm_107a, top_k <= cap)
            persist = bool(tags) and tags[-1] == "persist"
            if persist:
                tags.pop()
            bulk = bool(tags) and tags[-1] == "bulk"
            if bulk:
                tags.pop()
            i16 = bool(tags) and tags[-1] == "i16"
            if i16:
                tags.pop()
            assert tags in ([], ["staged"], ["two"], ["two", "staged"]), key
            assert not ((bulk or i16) and "staged" not in tags), key
            assert not (persist and not bulk), key
            assert not (persist and policy.rank_persist_max_k <= 0), key
            if bulk:  # the plain twin for a caller's unaligned outputs is registered alongside
                assert key[: -len(":bulk:persist" if persist else ":bulk")] in keys, key
            variants = (
                policy.rank_two_level_window_variants
                if "two" in form
                else policy.rank_window_variants
            )
            assert (
                threads.startswith("t")
                and window.startswith("w")
                and int(window[1:]) in variants
            ), key
            assert int(window[1:]) % (32 * int(threads[1:])) == 0, key
            assert ("two" in form) == policy.rank_two_level_form(int(window[1:])), key
        for key, name in keys.items():
            record = cake_jit.PROGRAMS[name]
            assert arch in record["arches"], (arch, key, name)
            assert record["role"] == key.split(":", 1)[0], (arch, key, name)
            assert stage_slug(key).isidentifier(), key
            if key.startswith("scan:pair_"):
                assert record["launch"]["cluster"] == [2, 1, 1], (arch, key)
        assert set(cake_jit.PROGRAM_LEVERS[arch]) == set(scan_keys), arch
        for top_k in (1, 256, 1024, 2048, 4096):
            assert finalize_key(policy.finalize_threads_for(top_k)) in keys, (
                arch,
                top_k,
            )  # the CUB programs stay registered
        for top_k, num_keys, segments, bound in (
            (2048, 8192, 1, None),
            (2048, 268757, 8, None),
            (
                2048,
                268757,
                8,
                67923,
            ),  # lever RW: the P1 row with its longest key segment as the bound
            (4096, 65536, 1, None),
            (2048, 524288, 1, None),  # lever FR2: a single 2^19 segment
        ):
            window = policy.finalize_rank_for(
                top_k, num_keys, segments, max_seqlen_k=bound
            )
            if window is not None:
                threads = policy.finalize_threads_for(top_k)
                two = policy.rank_two_level_form(window)
                staged = policy.finalize_rank_staged_for(window, threads, two)
                t16 = policy.finalize_rank_t16_for(threads, staged)
                launched, items = (
                    policy.rank_t16_form(threads)
                    if t16
                    else (threads, policy.finalize_items)
                )
                bulk = policy.finalize_rank_bulk_io_for(top_k, staged)
                # lever FRP-K: the dispatched bulk form carries ``:persist`` where the rule admits the call (never its plain twin)
                persistent = policy.finalize_rank_persistent_for(
                    top_k, staged, two, bulk, t16
                )
                for bulk_form in sorted(
                    {bulk, False}
                ):  # the dispatched form and (lever FRB) its plain twin
                    assert (
                        finalize_key(
                            launched,
                            window,
                            staged,
                            two,
                            items=items,
                            bulk=bulk_form,
                            persistent=bool(persistent and bulk_form),
                        )
                        in keys
                    ), (arch, top_k, num_keys, segments, bound, bulk_form)
        # the merge program is one text: every architecture registers the same merge program name
        assert cake_jit.PROGRAMS[keys[MERGE_KEY]]["role"] == "merge"


def test_policy_record_round_trip_and_validation():
    record = NARROW_PAIR_POLICY.record()
    assert tuple(record) == POLICY_FIELDS
    assert DispatchPolicy.from_record(record) == NARROW_PAIR_POLICY
    with pytest.raises(ValueError):
        DispatchPolicy.from_record({"tile_keys": 128})
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, tile_keys=0))
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, finalize_fit=1))
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, rank_top_k_min=0))
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, sample_tiles_tiny_units=0))
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, rank_window_variants=[16384, 8192]))
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, rank_rule=[524288]))
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, rank_staged=1))
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, rank_staged_rule=[0]))
    with pytest.raises(ValueError):
        DispatchPolicy.from_record(dict(record, rank_staged_rule=[262144, 1]))
    for bad in (
        dict(rank_seg_window=1),
        dict(rank_window_variants_seg_only=[4096]),  # not a rank_window_variants entry
        dict(rank_window_variants_seg_only=[131072, 8192]),  # not ascending
        dict(rank_two_level=1),
        dict(rank_two_level_window_variants=[]),
        dict(
            rank_two_level_window_variants=[65536, 131072]
        ),  # not above rank_slab_window_max
        dict(rank_two_level_window_variants=[262144, 131072]),
        dict(rank_slab_window_max=0),
        dict(rank_two_level_rule=[0]),
        dict(rank_two_level_rule=[524288, 1]),
        dict(rank_two_level_staged_rule=[0]),
        dict(rank_bulk_io=1),  # round 38: lever FRB / lever FRT fields
        dict(rank_t16=1),
        dict(rank_bulk_align_bytes=0),
        dict(rank_bulk_align_bytes=6),
        dict(rank_t16_slots=[4096, 2048]),
        dict(rank_t16_slots=[4100]),
        dict(rank_t16_slots=4096),
        dict(rank_persist_max_k=-1),  # round 39: lever FRP-K fields
        dict(rank_persist_max_k=2048.5),
        dict(rank_persist_ctas_per_sm=0),
    ):
        with pytest.raises(ValueError):
            DispatchPolicy.from_record(dict(record, **bad))


def test_policy_capacity_period_and_finalize():
    p = NARROW_PAIR_POLICY
    assert p.candidate_capacity(2048) == 8192
    assert p.candidate_capacity(1) == 256  # max(4, 1 + 128) rounded up to a tile
    assert p.candidate_capacity(4096) == 16384
    assert p.candidate_capacity(2048, multiplier=2) == 4096
    assert p.check_period_for(2048, 8192) == 16
    assert p.check_period_limit(2048, 8192) == 32
    assert p.check_period_for(1, 256) == 1
    assert p.check_period_for(2048, 16384) == 32
    wider = DispatchPolicy.from_record(
        dict(p.record(), check_period_kind_overrides={"wide": 32})
    )
    assert (
        wider.check_period_for(2048, 8192, "wide") == 32
        and wider.check_period_for(2048, 8192, "narrow") == 16
    )
    assert (
        wider.check_period_for(256, 8192, "wide") == 32
    )  # clamped to the exactness bound min(64, 62, 32)
    # candidate multiplier: 4 for short units, 8 for long units, raised to the 8192-slot floor for small top_k
    assert p.cand_cap_multiplier_for(65536, 65536, 1, 1, 2048) == 4
    assert p.cand_cap_multiplier_for(8192, 1048576, 1, 1, 2048) == 8
    assert p.cand_cap_multiplier_for(65536, 65536, 1, 1, 256) == 32
    assert p.cand_cap_multiplier_for(65536, 65536, 1, 1, 4096) == 4
    # right-sized finalize programs for small top_k, 256 threads up to 2048, 512 above
    assert [p.finalize_threads_for(k) for k in (1, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 4096)] == [
        32, 32, 64, 64, 128, 128, 256, 256, 512, 512,
    ]  # fmt: skip
    assert p.finalize_threads_admissible(512, 1) and p.finalize_threads_admissible(
        256, 2048
    )
    assert not p.finalize_threads_admissible(
        256, 2049
    ) and not p.finalize_threads_admissible(32, 257)
    assert not p.finalize_threads_admissible(1024, 1)
    assert (
        p.finalize_key_bits_for(65536) == 16
        and p.finalize_key_bits_for(65537) == 17
        and p.finalize_key_bits_for(1) == 1
    )
    # the rank finalize: the smallest bitmap window holding Tkv and one word per thread, within the pool bound (Tkv) and the mean
    # segment bound (Tkv / S); top_k below the floor, thread counts without a rank form and the switched-off policy keep CUB
    assert (
        p.finalize_rank_for(2048, 8192, 1) == 8192
        and p.finalize_rank_for(2048, 8193, 1) == 16384
    )
    assert (
        p.finalize_rank_for(2048, 65536, 1) == 65536
        and p.finalize_rank_for(2048, 262144, 1) == 262144
    )
    assert (
        p.finalize_rank_for(2048, 262145, 1) is None
        and p.finalize_rank_for(2048, 524288, 1) is None
    )  # single 2^19-key segments
    assert (
        p.finalize_rank_for(2048, 262145, 2) == 524288
        and p.finalize_rank_for(2048, 268757, 8) == 524288
    )  # packed segments
    assert (
        p.finalize_rank_for(2048, 524289, 8) is None
        and p.finalize_rank_for(2048, 1 << 20, 1) is None
    )  # above the pool bound
    assert (
        p.finalize_rank_for(1024, 8192, 1) is None
        and p.finalize_rank_for(1025, 8192, 1) == 8192
        and p.finalize_rank_for(256, 65536, 1) is None
    )
    assert (
        p.finalize_rank_for(4096, 8192, 1) == 16384
        and p.finalize_rank_for(4096, 65536, 1) == 65536
    )  # 512 threads: one word per thread
    assert (
        p.finalize_rank_for(2048, 8192, 1, 512) == 16384
        and p.finalize_rank_for(2048, 8192, 1, 1024) is None
        and p.finalize_rank_for(2048, 8192, 1, 128) is None
    )
    assert (
        p.finalize_rank_for(4096, 8192, 1, 256) is None
    )  # 256 threads do not hold 4096 entries (the admissibility check refuses the knob)
    assert (
        DispatchPolicy.from_record(
            dict(p.record(), rank_finalize=False)
        ).finalize_rank_for(2048, 8192, 1)
        is None
    )
    assert (
        DispatchPolicy.from_record(dict(p.record(), rank_rule=None)).finalize_rank_for(
            2048, 8192, 1
        )
        is None
    )
    assert (
        DispatchPolicy.from_record(
            dict(p.record(), rank_rule=[262144, 262144])
        ).finalize_rank_for(2048, 268757, 8)
        is None
    )
    # lever FRS: the staged form of the rank finalize where the switch is on and the window is within the staged bound (two CTAs per SM)
    assert (
        p.finalize_rank_staged_for(8192, 256) is False
        and p.finalize_rank_staged_for(None, 256) is False
    )  # switched off: the direct form
    s = DispatchPolicy.from_record(dict(p.record(), rank_staged=True))
    assert (
        s.finalize_rank_staged_for(8192, 256)
        and s.finalize_rank_staged_for(262144, 256)
        and s.finalize_rank_staged_for(262144, 512)
        and s.finalize_rank_staged_for(16384, 512)
    )
    assert (
        not s.finalize_rank_staged_for(524288, 256)
        and not s.finalize_rank_staged_for(524288, 512)
        and not s.finalize_rank_staged_for(None, 256)
    )
    assert DispatchPolicy.from_record(
        dict(p.record(), rank_staged=True, rank_staged_rule=[524288])
    ).finalize_rank_staged_for(524288, 512)
    assert not DispatchPolicy.from_record(
        dict(p.record(), rank_staged=True, rank_staged_rule=None)
    ).finalize_rank_staged_for(8192, 256)
    assert (
        finalize_key(256, 8192, True) == "finalize_rank:t256:w8192:staged"
        and finalize_key(256, 8192) == "finalize_rank:t256:w8192"
        and finalize_key(256) == "finalize:t256"
    )
    with pytest.raises(ValueError):
        finalize_key(256, None, True)
    # lever FRP-K: the persistent form of a staged bulk program carries ``:persist`` after ``:bulk``; never without the bulk form or on CUB
    assert (
        finalize_key(256, 8192, True, bulk=True, persistent=True)
        == "finalize_rank:t256:w8192:staged:bulk:persist"
    )
    with pytest.raises(ValueError):
        finalize_key(256, 8192, True, persistent=True)
    with pytest.raises(ValueError):
        finalize_key(256, persistent=True)
    # lever RW: the pool is the call's max_seqlen_k where the switch is on (the segment-only 2^17 variant joins the choice); the Tkv
    # pool otherwise; a bound above Tkv or below the mean segment raises whether or not the switch is on
    r = DispatchPolicy.from_record(dict(p.record(), rank_seg_window=True))
    assert (
        r.finalize_rank_for(2048, 268757, 8, max_seqlen_k=67923) == 131072
        and p.finalize_rank_for(2048, 268757, 8, max_seqlen_k=67923) == 524288
        and r.finalize_rank_for(2048, 268757, 8) == 524288
        and r.finalize_rank_for(2048, 268757, 8, max_seqlen_k=268757) is None
    )  # a bound equal to Tkv is tested exactly against the 262144 segment limit (the Tkv pool keeps the mean proxy)
    assert (
        r.finalize_rank_for(2048, 1 << 20, 2, max_seqlen_k=1 << 19) is None
        and r.finalize_rank_for(2048, 1 << 20, 4, max_seqlen_k=262144) == 262144
        and r.finalize_rank_for(2048, 100000, 1, max_seqlen_k=100000) == 131072
        and p.finalize_rank_for(2048, 100000, 1, max_seqlen_k=100000) == 262144
        and r.finalize_rank_for(2048, 100000, 1) == 262144
    )  # the bound itself is the segment bound and admits the 2^17 variant even when it equals Tkv; the Tkv pool (switch off or
    # no bound) keeps the round-35 inventory and skips the 2^17 slab variant
    for policy_, bad in ((r, 8193), (p, 8193), (r, 0)):
        with pytest.raises(ValueError, match="max_seqlen_k"):
            policy_.finalize_rank_for(2048, 8192, 1, max_seqlen_k=bad)
    with pytest.raises(ValueError, match="max_seqlen_k"):
        p.finalize_rank_for(2048, 8192, 2, max_seqlen_k=4095)
    # lever FR2: the pools above the slab bound take the two-level program with no segment bound, up to the two-level pool bound
    t = DispatchPolicy.from_record(dict(p.record(), rank_two_level=True))
    assert (
        t.finalize_rank_for(2048, 262145, 1) == 524288
        and t.finalize_rank_for(2048, 524288, 1) == 524288
        and t.finalize_rank_for(2048, 100000, 8) == 131072
        and t.finalize_rank_for(2048, 65536, 1) == 65536
        and t.finalize_rank_for(2048, 65537, 1) == 131072
        and t.finalize_rank_for(2048, 1 << 20, 1) is None
        and t.finalize_rank_for(1024, 100000, 1) is None
    )
    assert (
        t.rank_two_level_form(524288)
        and t.rank_two_level_form(131072)
        and not t.rank_two_level_form(65536)
        and not t.rank_two_level_form(None)
        and not p.rank_two_level_form(524288)
    )
    assert (
        DispatchPolicy.from_record(
            dict(p.record(), rank_two_level=True, rank_two_level_rule=[262144])
        ).finalize_rank_for(2048, 262145, 1)
        is None
    )
    assert (
        DispatchPolicy.from_record(
            dict(p.record(), rank_two_level=True, rank_two_level_rule=None)
        ).finalize_rank_for(2048, 100000, 8)
        == 262144
    )  # a two-level rule of None (an arch without a two-level entry) keeps the slab rule for the pool, as the kernel does
    rs = DispatchPolicy.from_record(
        dict(p.record(), rank_seg_window=True, rank_two_level=True)
    )
    assert rs.finalize_rank_for(
        2048, 268757, 8, max_seqlen_k=67923
    ) == 131072 and rs.rank_two_level_form(
        131072
    )  # both levers: the P rows' bound reaches the two-level 2^17 pool
    ts = DispatchPolicy.from_record(
        dict(p.record(), rank_two_level=True, rank_staged=True)
    )
    assert (
        ts.finalize_rank_staged_for(524288, 256, True)
        and ts.finalize_rank_staged_for(131072, 512, True)
        and not ts.finalize_rank_staged_for(524288, 256)
        and not ts.finalize_rank_staged_for(1048576, 256, True)
    )
    assert not DispatchPolicy.from_record(
        dict(
            p.record(),
            rank_two_level=True,
            rank_staged=True,
            rank_two_level_staged_rule=None,
        )
    ).finalize_rank_staged_for(131072, 256, True)
    assert (
        finalize_key(256, 131072, False, True) == "finalize_rank:t256:w131072:two"
        and finalize_key(256, 131072, True, True)
        == "finalize_rank:t256:w131072:two:staged"
    )
    with pytest.raises(ValueError):
        finalize_key(256, None, False, True)
    assert (
        p.sample_tiles_for(65536, 65536, 1, 1) == 16
        and p.sample_tiles_for(8192, 1048576, 1, 1) == 32
    )
    assert (
        p.sample_tiles_for(8192, 8192, 1, 1) == 8
        and p.sample_tiles_for(0, 16384, 1, 1) == 8
        and p.sample_tiles_for(0, 16512, 1, 1) == 16
    )  # the tiny-unit tier up to 128 mean tiles
    assert p.mean_unit_tiles(65536, 65536, 1, 1) == 256.0


def _select(policy, T, Tkv, S=1, K=2048, ratio=1, grid=148, **knobs):
    return select_program(
        policy,
        num_queries=T,
        num_keys=Tkv,
        num_segments=S,
        ratio=ratio,
        top_k=K,
        grid=grid,
        **knobs,
    )


def test_round38_rank_forms_policy_and_select():
    """Levers FRT (16-item thread form) and FRB (bulk row I/O) of the staged rank finalize: the record switches, the rules, the
    key tags and the dispatch -- the production slot count keeps the 16-item form to the 512-thread (top_k > 2048) staged form;
    the bulk form needs top_k % 4 == 0 and aligned outputs, else the plain twin program (same results)."""
    p = NARROW_PAIR_POLICY
    assert (
        p.rank_bulk_io is False
        and p.rank_t16 is False
        and p.rank_bulk_align_bytes == 16
        and p.rank_t16_slots == [4096]
    )
    assert (
        p.finalize_rank_t16_for(512, True) is False
        and p.finalize_rank_bulk_io_for(2048, True) is False
    )
    s = DispatchPolicy.from_record(
        dict(p.record(), rank_staged=True, rank_t16=True, rank_bulk_io=True)
    )
    assert (
        s.finalize_rank_t16_for(512, True)
        and not s.finalize_rank_t16_for(256, True)
        and not s.finalize_rank_t16_for(512, False)
    )
    assert s.rank_t16_form(512) == (256, 16) and s.rank_t16_form(256) == (128, 16)
    assert (
        s.finalize_rank_bulk_io_for(2048, True)
        and s.finalize_rank_bulk_io_for(4096, True)
        and not s.finalize_rank_bulk_io_for(2047, True)
    )
    assert not s.finalize_rank_bulk_io_for(
        2048, False
    ) and not s.finalize_rank_bulk_io_for(2048, True, outputs_aligned=False)
    # K = 4096 within the staged bound: 256 launched threads x 16 slots with bulk I/O; K = 2048: the 8-item form with bulk I/O
    c = _select(s, 8192, 65536, K=4096)
    assert (c.rank_threads, c.rank_items, c.rank_bulk_io, c.finalize_threads) == (
        256,
        16,
        True,
        512,
    )
    assert (
        c.finalize_key == "finalize_rank:t256:w65536:staged:i16:bulk"
        and c.keys[-1] == c.finalize_key
    )
    c = _select(s, 8192, 8192)
    assert (c.rank_threads, c.rank_items, c.rank_bulk_io) == (
        256,
        8,
        True,
    ) and c.finalize_key == "finalize_rank:t256:w8192:staged:bulk"
    # unaligned caller outputs: the plain twin; top_k off the granule: no bulk form; the direct form above the staged bound: neither
    assert (
        select_program(
            s,
            num_queries=8192,
            num_keys=8192,
            num_segments=1,
            ratio=1,
            top_k=2048,
            grid=148,
            outputs_aligned=False,
        ).finalize_key
        == "finalize_rank:t256:w8192:staged"
    )
    assert (
        _select(s, 8192, 8192, K=2047).finalize_key == "finalize_rank:t256:w8192:staged"
    )
    c = _select(s, 16231, 268757, S=8)
    assert c.finalize_key == "finalize_rank:t256:w524288" and (
        c.rank_threads,
        c.rank_items,
        c.rank_bulk_io,
    ) == (256, 8, False)
    c = _select(s, 8192, 524288)  # a CUB call carries the plain thread form
    assert c.finalize_key == "finalize:t256" and (
        c.rank_threads,
        c.rank_items,
        c.rank_bulk_io,
    ) == (256, 8, False)


def test_select_program_narrow_pair_policy():
    p = NARROW_PAIR_POLICY
    # one causal 8k document: three-warpgroup unit, buffer-fitting units sample their first threshold, the tiny-unit sample cap
    c = _select(p, 8192, 8192)
    assert (
        c.kind,
        c.block_q,
        c.n_split,
        c.pair,
        c.snake,
        c.tile_unroll,
        c.sample_fit,
    ) == ("l6", 6, 1, False, False, 1, True)
    assert (
        c.scan_key == "scan:l6:u1:s0:f1"
        and c.merge_key is None
        and c.finalize_key == "finalize_rank:t256:w8192"
    )
    assert (
        c.keys == ("scan:l6:u1:s0:f1", "finalize_rank:t256:w8192")
        and c.rank_window == 8192
        and c.finalize_role == "finalize_rank"
    )
    assert (
        c.rank_staged is False
    )  # lever FRS off (the production default): the direct rank program
    assert (
        c.cand_cap == 8192
        and c.check_period == 16
        and c.sample_tiles_max == 8
        and c.key_bits == 13
    )
    assert c.workspace_bytes == 148 * 6 * 8192 * 8
    # one causal 64k document: the CTA-pair three-warpgroup program
    c = _select(p, 65536, 65536)
    assert c.scan_key == "scan:pair_l6:u1:s0:f1" and c.pair and c.grid_ctas == 148
    assert (
        _select(p, 65536, 65536, grid=37).grid_ctas == 36
    )  # the pair needs an even grid
    # 64 tail queries over 1M keys: the narrow program split nine ways, merged before the finalize
    c = _select(p, 64, 1048576)
    assert (c.kind, c.physical_kind, c.block_q, c.n_split) == (
        "split",
        "split_narrow",
        4,
        9,
    )
    assert c.scan_key == "scan:split_narrow:u1:s0:f0" and c.merge_key == MERGE_KEY
    assert c.cand_cap_multiplier == 8 and c.cand_cap == 16384 and c.check_period == 32
    assert c.workspace_bytes == (148 * 4 * 16384 + 64 * 9 * 2048) * 8
    assert (
        c.keys == ("scan:split_narrow:u1:s0:f0", "merge", "finalize:t256")
        and c.rank_window is None
        and c.finalize_role == "finalize"
    )
    # 8192 tail queries over 1M keys: the wide unit, long-unit capacity, no sampled fit; the 1M-key window keeps the CUB finalize
    c = _select(p, 8192, 1048576)
    assert (
        c.scan_key == "scan:wide:u1:s0:f0"
        and c.block_q == 8
        and c.n_split == 1
        and not c.pair
    )
    assert (
        c.cand_cap == 16384
        and c.check_period == 32
        and c.sample_tiles_max == 32
        and c.finalize_key == "finalize:t256"
    )
    # 8192 tail queries over 512k keys (one segment above the mean-segment bound) keep CUB; the packed P1 geometry takes the 2^19 pool
    assert (
        _select(p, 8192, 524288).finalize_key == "finalize:t256"
        and _select(p, 8192, 262144).finalize_key == "finalize_rank:t256:w262144"
    )
    assert _select(p, 16231, 268757, S=8).finalize_key == "finalize_rank:t256:w524288"
    assert (
        _select(p, 16231, 268757, S=8, K=4096).finalize_key
        == "finalize_rank:t512:w524288"
    )
    # top_k 256 at 64k: no three-warpgroup / pair floor, the wide unit, the capacity floor, the 32-thread finalize
    c = _select(p, 65536, 65536, K=256)
    assert (
        c.scan_key == "scan:wide:u1:s0:f1"
        and c.cand_cap_multiplier == 32
        and c.cand_cap == 8192
    )
    assert c.finalize_key == "finalize:t32" and c.check_period == 16
    # explicit knobs (the 512-thread rank form holds one bitmap word per thread: its smallest window is 2^14)
    c = _select(
        p,
        8192,
        8192,
        candidate_multiplier=2,
        check_period=1,
        sample_tiles_max=0,
        finalize_threads=512,
    )
    assert (c.cand_cap, c.check_period, c.sample_tiles_max, c.finalize_key) == (
        4096,
        1,
        0,
        "finalize_rank:t512:w16384",
    )
    assert (
        _select(p, 8192, 1048576, finalize_threads=512).finalize_key == "finalize:t512"
    )
    # lever FRS switched on: the staged form within the staged bound, the direct form above it, the CUB calls untouched
    s = DispatchPolicy.from_record(dict(p.record(), rank_staged=True))
    c = _select(s, 8192, 8192)
    assert (
        c.rank_staged
        and c.finalize_key == "finalize_rank:t256:w8192:staged"
        and c.finalize_role == "finalize_rank"
        and c.keys[-1] == c.finalize_key
    )
    assert (
        _select(s, 8192, 262144).finalize_key == "finalize_rank:t256:w262144:staged"
        and _select(s, 16231, 268757, S=8).finalize_key == "finalize_rank:t256:w524288"
    )
    assert (
        not _select(s, 16231, 268757, S=8, K=4096).rank_staged
        and not _select(s, 8192, 1048576).rank_staged
        and _select(s, 8192, 1048576).finalize_key == "finalize:t256"
    )
    assert (
        _select(s, 8192, 8192, finalize_threads=512).finalize_key
        == "finalize_rank:t512:w16384:staged"
    )
    # levers RW + FR2 (round 37) on the staged policy: the P1 row's bound reaches the staged two-level 2^17 program, the same call
    # without the bound the staged two-level 2^19 program; a bound that cannot hold every key segment is rejected
    s2 = DispatchPolicy.from_record(
        dict(s.record(), rank_seg_window=True, rank_two_level=True)
    )
    c = _select(s2, 16231, 268757, S=8, max_seqlen_k=67923)
    assert (
        c.max_seqlen_k == 67923
        and c.rank_window == 131072
        and c.rank_two_level
        and c.rank_staged
        and c.finalize_key == "finalize_rank:t256:w131072:two:staged"
        and c.finalize_role == "finalize_rank"
        and c.keys[-1] == c.finalize_key
    )
    c0 = _select(s2, 16231, 268757, S=8)
    assert (
        c0.max_seqlen_k is None
        and c0.finalize_key == "finalize_rank:t256:w524288:two:staged"
    )
    assert (
        _select(s2, 8192, 524288).finalize_key
        == "finalize_rank:t256:w524288:two:staged"
    )
    assert _select(s2, 8192, 1048576).finalize_key == "finalize:t256"
    assert (
        _select(s, 16231, 268757, S=8, max_seqlen_k=67923).finalize_key
        == "finalize_rank:t256:w524288"
    )  # the switches off: the bound is validated and ignored
    for bad in (1000, 268758):
        with pytest.raises(ValueError, match="max_seqlen_k"):
            _select(s2, 16231, 268757, S=8, max_seqlen_k=bad)
    with pytest.raises(ValueError, match="check_period"):
        _select(p, 8192, 8192, check_period=33)
    with pytest.raises(ValueError, match="finalize_threads"):
        _select(p, 8192, 8192, finalize_threads=32)
    with pytest.raises(ValueError, match="grid_ctas"):
        _select(p, 8192, 8192, grid=0)


def test_select_program_snake_unroll_policy():
    p = SNAKE_UNROLL_POLICY
    # one causal 8k document: pair three-warpgroup program in snake order (the unroll yields to the snake order)
    c = _select(p, 8192, 8192, grid=212)
    assert c.scan_key == "scan:pair_l6:u1:s1:f1" and c.snake and c.tile_unroll == 1
    # 4096 tail queries over 128k keys: the wide unit split two ways by the wave rule, unrolled by two
    c = _select(p, 4096, 131072, grid=212)
    assert (c.kind, c.physical_kind, c.block_q, c.n_split, c.snake, c.tile_unroll) == (
        "split",
        "split_wide",
        8,
        2,
        False,
        2,
    )
    assert c.scan_key == "scan:split_wide:u2:s0:f0" and c.merge_key == MERGE_KEY
    # the same call on a grid the ragged last round does not idle keeps the unsplit wide program
    assert _select(p, 4096, 131072, grid=256).n_split == 1
    # 64 tail queries over 1M keys: thirteen narrow splits on 212 CTAs
    assert _select(p, 64, 1048576, grid=212).n_split == 13
    # compressed keys (ratio 4) use the second snake floor and the unit-cost spread rule
    c = _select(p, 65536, 16384, ratio=4, grid=212)
    assert c.kind in ("l6", "pair_l6") and not c.snake
    # a short-query / long-key call never snakes (spread ~ 0) and unrolls its narrow pair program
    c = _select(p, 2048, 262144, grid=212)
    assert not c.snake and c.tile_unroll == (
        2 if c.kind in ("narrow", "wide", "pair_narrow", "pair_wide") else 1
    )


# The persistent grids the export froze the dispatch at (one CTA per SM of the production parts: B200; B300, GB300 as
# deployed, the full B300 die; R200).  The registry does not carry them: the host reads the device's SM count and fails
# closed by name on a program outside the architecture's registry.
_FROZEN_GRIDS = {"sm_100a": (148,), "sm_103a": (148, 152, 160), "sm_107a": (212,)}


def _dense_geometries(policy, *, steps_t=32, steps_m=28):
    """A geometric (queries, mean unit tiles) grid: 1..2^17 queries and 0..2^13 mean key tiles per unit at about 1.4x
    spacing, the production call patterns, the top_k values on both sides of the policy's floors, plus the review's
    geometry (6018 x 11754 keys)."""
    tile = policy.tile_keys
    queries = sorted(
        {1, 2, 3, 6018}
        | {int(round(2 ** (17 * i / steps_t))) for i in range(steps_t + 1)}
    )
    tiles = sorted(
        {0, 1, 2} | {int(round(2 ** (13 * i / steps_m))) for i in range(steps_m + 1)}
    )
    for S, ratio in ((1, 1), (1, 2), (8, 1), (64, 1)):
        for T in queries:
            for m in tiles:
                Tkv = max(1, int(round(m * tile * S + T / (2 * ratio))))
                for top_k in (1, 2, 256, 511, 512, 1000, 1024, 1025, 2048, 4096):
                    yield T, Tkv, S, ratio, top_k


def test_policy_selects_registered_programs_on_a_dense_geometry_grid():
    """Every program the frozen policy selects on a dense geometry grid is registered for the architecture: the host
    fails closed by name on an unregistered program, so a selectable but unregistered program is a production failure
    (round 2 review: ``scan:narrow:u1:s1:f1`` on sm_107a at 6018 queries x 11754 keys was selectable but unregistered)."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        policy = policy_for(arch)
        for grid in _FROZEN_GRIDS[arch]:
            for T, Tkv, S, ratio, top_k in _dense_geometries(policy):
                bounds = [None]
                if (
                    top_k >= policy.rank_top_k_min
                ):  # lever RW: S equal segments and one long segment
                    bounds += sorted({-(-Tkv // S), Tkv})
                for bound in bounds:
                    choice = select_program(
                        policy,
                        num_queries=T,
                        num_keys=Tkv,
                        num_segments=S,
                        ratio=ratio,
                        top_k=top_k,
                        grid=grid,
                        max_seqlen_k=bound,
                    )
                    selected = [choice.scan_key, choice.finalize_key] + (
                        [MERGE_KEY] if choice.n_split > 1 else []
                    )
                    for key in selected:
                        assert key in keys, (
                            arch,
                            grid,
                            T,
                            Tkv,
                            S,
                            ratio,
                            top_k,
                            bound,
                            key,
                        )


def test_program_keys_and_slugs():
    assert scan_key("narrow", 1, False, True) == "scan:narrow:u1:s0:f1"
    assert stage_slug("scan:split_wide:u2:s0:f0") == "scan_split_wide_u2_s0_f0"
    assert finalize_key(256) == "finalize:t256" and stage_slug(MERGE_KEY) == "merge"
    assert (
        finalize_key(256, 8192) == "finalize_rank:t256:w8192"
        and stage_slug(finalize_key(512, 65536)) == "finalize_rank_t512_w65536"
        and finalize_key(256, 131072, two_level=True)
        == "finalize_rank:t256:w131072:two"
        and stage_slug(finalize_key(256, 131072, True, True))
        == "finalize_rank_t256_w131072_two_staged"
    )
    assert (
        cake_policy.finalize_role(None) == "finalize"
        and cake_policy.finalize_role(8192) == "finalize_rank"
    )
    assert (
        cake_policy.physical_kind("split", "l6") == "split_l6"
        and cake_policy.physical_kind("pair_wide", "wide") == "pair_wide"
    )


@pytest.mark.parametrize(
    "spec",
    [
        dict(seg_q_len=[5, 3], seg_k_len=[9, 3], ratio=1, offsets=None),
        dict(seg_q_len=[4, 6], seg_k_len=[10, 2], ratio=1, offsets=[-3, 4]),
        dict(seg_q_len=[7, 1, 0], seg_k_len=[4, 1, 5], ratio=2, offsets=None),
        dict(seg_q_len=[9], seg_k_len=[6], ratio=3, offsets=[-7]),
    ],
)
def test_visible_key_counts_match_scalar_rule(spec):
    counts = visible_key_counts(
        spec["seg_q_len"],
        spec["seg_k_len"],
        ratio=spec["ratio"],
        q_causal_offsets=spec["offsets"],
    ).tolist()
    offsets = cake_backend.effective_offsets(
        spec["seg_q_len"], spec["seg_k_len"], spec["offsets"], spec["ratio"]
    )
    expected = [
        visible_key_count(offsets[s], u, spec["ratio"], spec["seg_k_len"][s])
        for s in range(len(spec["seg_q_len"]))
        for u in range(spec["seg_q_len"][s])
    ]
    assert counts == expected
    helper = ref.visible_rows(
        ref.build_inputs(
            torch.zeros(
                (sum(spec["seg_q_len"]), NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16
            ),
            torch.zeros((sum(spec["seg_k_len"]), HEAD_DIM), dtype=torch.bfloat16),
            torch.zeros((sum(spec["seg_q_len"]), NUM_HEADS)),
            spec["seg_q_len"],
            spec["seg_k_len"],
            top_k=4,
            softmax_scale=1.0,
            q_causal_offsets=spec["offsets"],
            ratio=spec["ratio"],
        )
    ).tolist()
    assert helper == expected


def _host_inputs(total_q=6, total_k=10, segments=(0, 2, 6), key_segments=(0, 3, 10)):
    q = torch.zeros((total_q, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    k = torch.zeros((total_k, HEAD_DIM), dtype=torch.bfloat16)
    w = torch.zeros((total_q, NUM_HEADS), dtype=torch.float32)
    cu_q = torch.tensor(segments, dtype=torch.int32)
    cu_k = torch.tensor(key_segments, dtype=torch.int32)
    return SimpleNamespace(
        q=q,
        k=k,
        w=w,
        cu_q=cu_q,
        cu_k=cu_k,
        offsets=None,
        top_k=4,
        scale=0.5,
        ratio=1,
        max_seqlen_k=None,
    )


def _validate(h):
    return validate_dsa_indexer_inputs(
        h.q,
        h.k,
        h.w,
        h.cu_q,
        h.cu_k,
        h.offsets,
        h.top_k,
        h.scale,
        h.ratio,
        max_seqlen_k=h.max_seqlen_k,
    )


def test_validate_reaches_the_device_rule_for_host_tensors():
    # shapes, dtypes, strides and scalars pass; host tensors fail only the device rule
    with pytest.raises(ValueError, match="CUDA"):
        _validate(_host_inputs())
    for bound in (
        5,
        10,
    ):  # a bound holding every key segment (10 keys in 2 segments) passes to the device rule
        h = _host_inputs()
        h.max_seqlen_k = bound
        with pytest.raises(ValueError, match="CUDA"):
            _validate(h)
    h = _host_inputs()
    h.k = torch.zeros((10, ref.TRAINER_KEY_ROW_STRIDE), dtype=torch.bfloat16)[
        :, :HEAD_DIM
    ]  # trainer view
    with pytest.raises(ValueError, match="CUDA"):
        _validate(h)
    h = _host_inputs()
    h.offsets = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="CUDA"):
        _validate(h)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda h: setattr(h, "q", h.q[:, :, :64]), r"q must be"),
        (lambda h: setattr(h, "q", h.q.float()), "bfloat16"),
        (
            lambda h: setattr(h, "q", h.q.transpose(0, 1).contiguous().transpose(0, 1)),
            "contiguous",
        ),
        (lambda h: setattr(h, "k", h.k[:, :64]), r"k must be"),
        (lambda h: setattr(h, "k", h.k.float()), "bfloat16"),
        (
            lambda h: setattr(
                h, "k", torch.zeros((10, 130), dtype=torch.bfloat16)[:, :128]
            ),
            "row stride",
        ),
        (
            lambda h: setattr(h, "k", h.k.t().contiguous().t()),
            "unit inner stride|row stride",
        ),
        (lambda h: setattr(h, "w", h.w[:, :16]), r"w must be"),
        (lambda h: setattr(h, "w", h.w.to(torch.bfloat16)), "float32"),
        (lambda h: setattr(h, "cu_q", h.cu_q.to(torch.int64)), "int32"),
        (lambda h: setattr(h, "cu_k", h.cu_k[:-1]), "S \\+ 1"),
        (
            lambda h: setattr(h, "offsets", torch.zeros(3, dtype=torch.int64)),
            "q_causal_offsets must be",
        ),
        (lambda h: setattr(h, "offsets", torch.zeros(2, dtype=torch.int32)), "int64"),
        (lambda h: setattr(h, "top_k", 0), "top_k"),
        (lambda h: setattr(h, "top_k", MAX_TOP_K + 1), "top_k"),
        (lambda h: setattr(h, "top_k", 2.0), "top_k"),
        (lambda h: setattr(h, "ratio", 0), "ratio"),
        (lambda h: setattr(h, "scale", 0.0), "softmax_scale"),
        (lambda h: setattr(h, "scale", float("nan")), "softmax_scale"),
        (lambda h: setattr(h, "max_seqlen_k", 2.0), "max_seqlen_k"),
        (lambda h: setattr(h, "max_seqlen_k", -1), "max_seqlen_k"),
        (lambda h: setattr(h, "max_seqlen_k", 11), "max_seqlen_k"),  # above Tkv = 10
        (
            lambda h: setattr(h, "max_seqlen_k", 4),
            "max_seqlen_k",
        ),  # 4 x 2 segments < 10 keys
    ],
)
def test_validate_rejects(mutate, match):
    h = _host_inputs()
    mutate(h)
    with pytest.raises(ValueError, match=match):
        _validate(h)


def test_bind_program_orders_arguments_and_fails_closed(monkeypatch):
    calls = []
    fake_module = SimpleNamespace(run=lambda *args: calls.append(args))
    monkeypatch.setattr(cake_backend, "load_program", lambda program, arch: fake_module)
    monkeypatch.setattr(cake_backend, "FFI_ENTRY", "run")
    plan = [
        ["tma_buffer", "K"],
        ["buffer", "indices"],  # alias of Indices
        ["parameter", "topk"],  # alias of top_k
        ["parameter", "softmax_scale"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ]
    monkeypatch.setitem(cake_backend.ARG_PLANS, "scan", plan)
    monkeypatch.setitem(
        cake_backend.PROGRAMS,
        "fake",
        {
            "role": "scan",
            "sources": [],
            "arches": ["sm_100a"],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
    )
    values = {
        "K": "k-view",
        "Indices": "idx",
        "top_k": 7,
        "softmax_scale": 0.25,
        "Q": "unused",
    }
    launch = bind_program("fake", "sm_100a", "scan", values, (148, 1, 1))
    assert launch.arguments == ("k-view", "idx", 7, 0.25, 148, 1, 1)
    assert launch.grid == (148, 1, 1)
    launch()
    assert calls == [launch.arguments]
    monkeypatch.setitem(
        cake_backend.ARG_PLANS, "scan", plan + [["parameter", "mystery_knob"]]
    )
    with pytest.raises(KeyError, match="mystery_knob"):
        bind_program("fake", "sm_100a", "scan", values, (148, 1, 1))
    monkeypatch.setitem(cake_backend.ARG_PLANS, "scan", [["buffer", "Cand"]])
    with pytest.raises(KeyError, match="Cand"):
        bind_program("fake", "sm_100a", "scan", {"Cand": None}, (1, 1, 1))
    with pytest.raises(ValueError, match="merge"):
        bind_program("fake", "sm_100a", "merge", values, (1, 1, 1))


# ---------------------------------------------------------------------------
# Host-only: the reference itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed,top_k,ratio,offsets",
    [(0, 5, 1, None), (1, 1, 1, [2, -1]), (2, 3, 2, None), (3, 20, 1, None)],
)
def test_reference_matches_bruteforce(seed, top_k, ratio, offsets):
    inputs = ref.make_random_inputs(
        [9, 6],
        [11, 7],
        top_k=top_k,
        seed=seed,
        device="cpu",
        ratio=ratio,
        q_causal_offsets=offsets,
        softmax_scale=0.3,
    )
    result = ref.select_reference(inputs, order="sequential")
    ids, scores = ref.bruteforce_rows(inputs)
    assert result.indices.tolist() == ids
    got = result.scores_fp64.tolist()
    for row_got, row_exp in zip(got, scores, strict=True):
        for a, b in zip(row_got, row_exp, strict=True):
            assert a == b or abs(a - b) <= 1e-12 * max(1.0, abs(b)), (a, b)
    assert not ref.structure_failures(inputs, result.indices, result.scores)
    assert ref.judge((result.indices, result.scores), result, exact_bits=True).passed


def test_reference_sequential_order_is_partition_invariant():
    inputs = ref.make_random_inputs([40, 30], [70, 50], top_k=16, seed=5, device="cpu")
    a = ref.select_reference(inputs, order="sequential", query_chunk=7, key_chunk=9)
    b = ref.select_reference(inputs, order="sequential", query_chunk=64, key_chunk=4096)
    assert ref.same_bits((a.indices, a.scores), (b.indices, b.scores))
    assert torch.equal(a.scores_fp64, b.scores_fp64)


@pytest.mark.parametrize("name", list(ref.EXACT_CASES))
def test_exact_cases_match_closed_form(name):
    inputs = ref.make_exact_case(name, "cpu")
    result = ref.select_reference(inputs, order="sequential")
    score = ref.exact_closed_form(inputs)
    ids = result.indices.tolist()
    s32 = result.scores.tolist()
    s64 = result.scores_fp64.tolist()
    for t in range(inputs.num_queries):
        for c, j in enumerate(ids[t]):
            if j < 0:
                break
            expected = score(t, j)
            assert s64[t][c] == expected and s32[t][c] == expected, (name, t, c, j)
    if name == "signed_zeros":
        bits = result.scores.view(torch.int32)
        zero = (result.scores == 0) & (result.indices >= 0)
        assert bool(zero.any(dim=1).all()), (
            "zero-score keys must reach the selector boundary on every row"
        )
        # the program's positive-accumulator reduction yields +0.0 for every zero score; an IEEE sequential
        # head sum would give -0.0 on the all-negative-weight (even) rows, so the bit check is discriminating
        assert bool((bits[zero] == 0).all())
        even = torch.arange(inputs.num_queries) % 2 == 0
        assert bool(zero[even].any()) and bool(zero[~even].any())
        with pytest.raises(NotImplementedError):
            ref.judge(
                (result.indices, result.scores),
                result,
                exact_bits=True,
                zero_sign_policy="ieee_sum",
            )


# the scaled geometry of the exact cases, small enough for the sequential CPU reference
_SCALED_EXACT = dict(num_queries=24, num_keys=1024)


@pytest.mark.parametrize("name", list(ref.EXACT_CASES))
def test_scaled_exact_cases_match_closed_form(name):
    inputs = ref.make_exact_case(name, "cpu", **_SCALED_EXACT)
    assert (inputs.seg_q_len, inputs.seg_k_len, inputs.offsets()) == (
        [24],
        [1024],
        [1000],
    )
    assert 256 <= inputs.top_k <= 257
    result = ref.select_reference(inputs, order="sequential")
    score = ref.exact_closed_form(inputs)
    ids = result.indices.tolist()
    s32 = result.scores.tolist()
    s64 = result.scores_fp64.tolist()
    visible = result.visible.tolist()
    for t in range(inputs.num_queries):
        selected = ids[t][: int(result.selected_count[t])]
        assert len(selected) == inputs.top_k and all(j >= 0 for j in selected)
        chosen = [score(t, j) for j in selected]
        for c, (j, expected) in enumerate(zip(selected, chosen, strict=True)):
            assert s64[t][c] == expected and s32[t][c] == expected, (name, t, c, j)
        rest = [score(t, j) for j in range(visible[t]) if j not in set(selected)]
        if name == "distinct":
            assert len(set(chosen)) == len(chosen)
            if visible[t] == 1024:
                assert len({score(t, j) for j in range(1024)}) == 1024
        elif name == "cutoff_ties":
            if visible[t] == 1024:
                assert min(chosen) == max(rest), "the cut must fall inside a tie group"
        elif name == "all_equal":
            assert len(set(chosen)) == 1 and set(rest) == set(chosen)
        elif name == "negative":
            assert all((v < 0) if t % 3 == 0 else (v > 0) for v in chosen)
        elif name == "few_winners_large_tie":
            base = score(t, 0)
            winners = {w for w in (3, 232, 511, 512, 1023) if w < visible[t]}
            assert {j for j in selected if score(t, j) != base} == winners
            assert all(v == base for v in rest)
    if name == "signed_zeros":
        bits = result.scores.view(torch.int32)
        zero = (result.scores == 0) & (result.indices >= 0)
        assert bool(zero.any(dim=1).all()), (
            "zero-score keys must reach the selector boundary on every row"
        )
        assert bool((bits[zero] == 0).all())
        even = torch.arange(inputs.num_queries) % 2 == 0
        assert bool(zero[even].any()) and bool(zero[~even].any())
        assert bool(((result.scores > 0) & (result.indices >= 0))[~even].any())
        assert not bool(((result.scores > 0) & (result.indices >= 0))[even].any())


def test_scaled_exact_cases_reject_unusable_geometries():
    with pytest.raises(ValueError):
        ref.make_exact_case("distinct", "cpu", num_queries=8)
    with pytest.raises(ValueError):
        ref.make_exact_case("distinct", "cpu", num_queries=9, num_keys=8)
    with pytest.raises(ValueError):
        ref.make_exact_case("distinct", "cpu", num_queries=8, num_keys=65536)
    with pytest.raises(
        ValueError
    ):  # the first row would see too few keys for the zero boundary
        ref.make_exact_case("signed_zeros", "cpu", num_queries=1024, num_keys=1024)


@pytest.mark.parametrize("name", list(ref.PACKED_GEOMETRIES))
def test_packed_geometry_extra_segments_keep_the_declared_segments(name):
    base = ref.make_packed_inputs(name, "cpu")
    inputs = ref.make_packed_inputs(
        name, "cpu", extra_segments=2, extra_segment_queries=8, extra_segment_keys=64
    )
    n = base.num_segments
    assert inputs.seg_q_len == base.seg_q_len + [8, 8]
    assert inputs.seg_k_len == base.seg_k_len + [64, 64]
    assert (inputs.top_k, inputs.ratio, inputs.softmax_scale) == (
        base.top_k,
        base.ratio,
        base.softmax_scale,
    )
    assert inputs.offsets()[:n] == base.offsets()
    assert inputs.offsets()[n:] == [base.ratio * 64 - 8] * 2
    assert (inputs.q_causal_offsets is None) == (
        base.q_causal_offsets is None and base.ratio == 1
    )
    visible = ref.visible_rows(inputs, device="cpu").tolist()
    assert visible[: base.num_queries] == ref.visible_rows(base, device="cpu").tolist()
    extra = visible[base.num_queries :]
    assert extra[7] == 64 and extra[15] == 64 and min(extra) >= 56


def test_judge_rejects_a_clearly_better_lacking_key_and_bad_bits():
    inputs = ref.make_exact_case("distinct", "cpu")
    reference = ref.select_reference(inputs, order="sequential")
    ids, scores = reference.indices.clone(), reference.scores.clone()
    assert ref.judge((ids, scores), reference, exact_bits=True).passed
    # swap the weakest selected key of row 0 for an unselected, lower-scoring visible key
    row = 0
    n = int(reference.selected_count[row])
    chosen = set(ids[row, :n].tolist())
    visible = int(reference.visible[row])
    unselected = [j for j in range(visible) if j not in chosen]
    assert unselected
    worst_slot = min(range(n), key=lambda c: (float(scores[row, c]), int(ids[row, c])))
    ids[row, worst_slot] = unselected[0]
    scores[row, worst_slot] = ref.score_pairs(
        inputs, torch.tensor([row]), torch.tensor([unselected[0]])
    )[1][0]
    order = torch.argsort(
        torch.where(ids[row] >= 0, ids[row], torch.full_like(ids[row], 2**31 - 1))
    )
    ids[row], scores[row] = ids[row][order], scores[row][order]
    verdict = ref.judge((ids, scores), reference)
    assert not verdict.passed and any(
        "clearly better" in f for f in verdict.failures
    ), str(verdict)
    # a wrong score bit pattern on a correct id
    ids, scores = reference.indices.clone(), reference.scores.clone()
    scores[1, 0] = scores[1, 0] + 1.0
    verdict = ref.judge((ids, scores), reference)
    assert not verdict.passed and any(
        "outside the bound" in f for f in verdict.failures
    ), str(verdict)
    # a structural error: a duplicate id
    ids, scores = reference.indices.clone(), reference.scores.clone()
    ids[2, 1] = ids[2, 0]
    assert ref.structure_failures(inputs, ids, scores)


# ---------------------------------------------------------------------------
# Operator: exact semantics, causality, accuracy, near ties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(ref.EXACT_CASES))
def test_exact_semantics(name):
    _require_program()
    inputs = ref.make_exact_case(name, "cuda")
    reference = ref.select_reference(inputs, order="sequential")
    verdict = _judge_ok(_run(inputs), reference, exact_bits=True)
    assert verdict.rows_identical == inputs.num_queries


def test_signed_zero_bits_follow_the_documented_policy():
    _require_program()
    inputs = ref.make_exact_case("signed_zeros", "cuda")
    indices, scores = _run(inputs)
    reference = ref.select_reference(inputs, order="sequential")
    assert torch.equal(
        indices, reference.indices
    )  # +0.0 == -0.0 for ranking: the id rule decides inside the zero group
    policy = record_zero_sign_policy()
    zero = (scores == 0) & (indices >= 0)
    assert bool(zero.any())
    bits = scores.view(torch.int32)
    if policy != "positive_accumulator":
        pytest.fail(f"the reference has no model for zero-sign policy {policy!r}")
    assert bool((bits[zero] == 0).all()), (
        "the chains start from +0.0 with a fixed FMA order: every zero score is +0.0"
    )
    assert torch.equal(bits[zero], reference.scores.view(torch.int32)[zero])


@pytest.mark.parametrize("name", list(ref.PACKED_GEOMETRIES))
def test_packed_causality(name):
    _require_program()
    inputs = ref.make_packed_inputs(name, "cuda")
    reference = ref.select_reference(inputs)
    out = _run(inputs)
    _independent_membership_check(inputs, out[0])
    _judge_ok(out, reference)


@pytest.mark.parametrize("seed,peaked", [(1, False), (2, False), (3, True)])
def test_random_accuracy_within_bound(seed, peaked):
    _require_program()
    inputs = ref.make_random_inputs(
        [1100, 900, 1000],
        [1100, 20000, 1000],
        top_k=2048,
        seed=seed,
        device="cuda",
        peaked=peaked,
    )
    reference = ref.select_reference(inputs)
    verdict = _judge_ok(_run(inputs), reference)
    assert (
        verdict.max_fp32_error_over_bound <= 1.0
        and verdict.max_fp64_error_over_bound <= 1.0
    )


def test_near_tie_duplicated_keys_prefer_larger_ids():
    """Replicated keys give bit-identical scores under any deterministic scheme: the larger id must win."""
    _require_program()
    n_distinct, reps, lq, K = 300, 8, 48, 2048
    base = ref.make_random_inputs([lq], [n_distinct], top_k=K, seed=21, device="cuda")
    lk = n_distinct * reps
    group = (
        torch.randperm(
            lk, generator=torch.Generator(device="cuda").manual_seed(5), device="cuda"
        )
        % n_distinct
    )
    k = base.k[group].contiguous()
    inputs = ref.build_inputs(
        base.q,
        k,
        base.w,
        [lq],
        [lk],
        top_k=K,
        softmax_scale=base.softmax_scale,
        label="near_tie_duplicates",
    )
    reference = ref.select_reference(inputs, order="sequential")
    out = _run(inputs)
    _judge_ok(out, reference)
    ids = out[0].cpu()
    group_cpu = group.cpu()
    visible = ref.visible_rows(inputs, device="cpu")
    for t in range(inputs.num_queries):
        n = min(K, int(visible[t]))
        sel = ids[t, :n]
        sel_groups = group_cpu[sel]
        for g in sel_groups.unique().tolist():
            members = (group_cpu[: int(visible[t])] == g).nonzero().squeeze(1)
            chosen = sel[sel_groups == g]
            if (
                chosen.numel() < members.numel()
            ):  # the cutoff splits this replica group: the largest ids are chosen
                assert torch.equal(
                    chosen.sort().values, members.sort().values[-chosen.numel() :]
                ), (t, g)


def test_near_tie_perturbed_keys_disagree_only_within_uncertainty():
    _require_program()
    inputs = ref.make_random_inputs([64], [6000], top_k=2048, seed=22, device="cuda")
    k = inputs.k
    src = torch.arange(0, 3000, device="cuda")
    k[src + 3000] = k[src]
    col = k[src + 3000, 0].contiguous()
    k[src + 3000, 0] = (col.view(torch.int16) + 1).view(
        torch.bfloat16
    )  # one BF16 ulp on one dimension
    reference = ref.select_reference(inputs)
    verdict = _judge_ok(_run(inputs), reference)
    assert (
        verdict.rows_identical + verdict.rows_within_uncertainty == inputs.num_queries
    )


# ---------------------------------------------------------------------------
# Operator: dynamic shapes, dispatch coverage, repeatability, partition independence
# ---------------------------------------------------------------------------


def _recorded_inputs(name: str):
    if name == "P1":
        seg_q, seg_k = _lengths(P1_CU_SEQLENS_Q), _lengths(P1_CU_SEQLENS_K)
    else:  # the P1 pattern scaled to the second recorded token count (its boundaries were not recorded)
        seg_q = _lengths(P1_CU_SEQLENS_Q)
        seg_q = [max(1, round(v * P2_T / sum(seg_q))) for v in seg_q]
        seg_q[-1] += P2_T - sum(seg_q)
        seg_k = list(seg_q)
    if not FULL:
        seg_k = list(
            seg_q
        )  # keys per segment = queries per segment keeps the FP64 reference to seconds
    return ref.make_random_inputs(
        seg_q, seg_k, top_k=2048, seed=7571, device="cuda", label=name
    )


@pytest.mark.parametrize("name", ["P1", "P2"])
def test_recorded_token_counts(name):
    _require_program()
    inputs = _recorded_inputs(name)
    assert inputs.num_queries in (16231, P2_T)
    reference = ref.select_reference(inputs)
    _judge_ok(_run(inputs), reference)


_BOUNDARY = [
    (5, lk, 32)
    for lk in (
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
    )
]
_BOUNDARY += [
    (n, n, 64)
    for n in (1, 2, 3, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025)
]
_BOUNDARY += [
    (16, 5000, k)
    for k in (
        1,
        2,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        2047,
        2048,
        2049,
        4095,
        4096,
    )
]


@pytest.mark.parametrize("lq,lk,top_k", _BOUNDARY)
def test_tile_boundary_neighbours(lq, lk, top_k):
    _require_program()
    inputs = ref.make_boundary_inputs(lq, lk, top_k=top_k, device="cuda")
    _judge_ok(_run(inputs), ref.select_reference(inputs))


# Geometries that reach the host dispatch's program kinds (which kind a geometry takes depends on the
# architecture's policy; every kind computes the same result).  The 64-query / 256k-key row splits its key ranges
# on every registered architecture and so exercises the merge stage; the 16k causal document reaches the CTA-pair
# program where the architecture's rule admits it.
_DISPATCH_ROWS = [
    ([64], [262144], 2048),  # 16 narrow units over 2047 tiles: key-range split + merge
    (
        [16384],
        [16384],
        2048,
    ),  # one causal document of 128 mean tiles: three-warpgroup / pair programs
    (
        [4096],
        [65536],
        2048,
    ),  # context-parallel tail: narrow or wide unit by the architecture's rule
    ([4096], [4096], 256),  # small top_k: capacity floor, right-sized finalize
]


@pytest.mark.parametrize("seg_q,seg_k,top_k", _DISPATCH_ROWS)
def test_dispatch_rows_are_exact_and_plan_matches_runner(seg_q, seg_k, top_k):
    _require_program()
    inputs = ref.make_random_inputs(
        seg_q, seg_k, top_k=top_k, seed=90 + top_k % 7, device="cuda"
    )
    bound = max(inputs.seg_k_len)  # the caller's bound on every key segment (lever RW)
    plan = _plan(inputs, max_seqlen_k=bound)
    runner = prepare_dsa_indexer_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        max_seqlen_k=bound,
    )
    assert runner.choice == plan and plan.max_seqlen_k == bound
    assert _plan(inputs).max_seqlen_k is None
    assert runner.stages == (
        ("scan", "merge", plan.finalize_role)
        if plan.split
        else ("scan", plan.finalize_role)
    )
    assert set(runner.programs) == set(runner.stages)
    assert runner.workspace_bytes == plan.workspace_bytes
    out = runner()
    _judge_ok((out[0].clone(), out[1].clone()), ref.select_reference(inputs))
    if seg_q == [64]:
        assert plan.split and plan.n_split >= 2


def test_changing_shapes_in_sequence():
    _require_program()
    sequence = [
        ([37], [1000], 64),
        ([2049], [2049], 2048),
        ([3, 700, 1], [3, 9000, 1], 512),
        ([16231 // 8] * 2, [16231 // 8, 20000], 2048),
        ([129], [129], 1),
    ]
    for i, (lq, lk, k) in enumerate(sequence):
        inputs = ref.make_random_inputs(lq, lk, top_k=k, seed=30 + i, device="cuda")
        _judge_ok(_run(inputs), ref.select_reference(inputs))


def test_max_seqlen_k_bound_changes_the_program_never_a_bit():
    """Lever RW: the caller's bound on every key segment sizes the rank finalize's bitmap pool -- a smaller program for packed
    segments -- and the results are bitwise identical with the bound, with a looser bound and without it; a bound that cannot hold
    every segment is rejected."""
    _require_program()
    inputs = ref.make_random_inputs(
        [50, 300, 50, 300], [100, 9000, 100, 9000], top_k=2048, seed=37, device="cuda"
    )
    tight = max(inputs.seg_k_len)
    plan_none, plan_tight, plan_loose = (
        _plan(inputs),
        _plan(inputs, max_seqlen_k=tight),
        _plan(inputs, max_seqlen_k=inputs.num_keys),
    )
    assert plan_tight.max_seqlen_k == tight and plan_none.max_seqlen_k is None
    assert plan_loose.max_seqlen_k == inputs.num_keys
    if (
        plan_none.rank_window is not None
        and policy_for(arch_for(inputs.device)).rank_seg_window
    ):
        assert (
            plan_tight.rank_window is not None
            and plan_tight.rank_window < plan_none.rank_window
        )
        assert plan_tight.finalize_key != plan_none.finalize_key
    base = _run(inputs)
    _judge_ok(base, ref.select_reference(inputs))
    assert ref.same_bits(base, _run(inputs, max_seqlen_k=tight))
    assert ref.same_bits(base, _run(inputs, max_seqlen_k=inputs.num_keys))
    for bad in (
        tight - 1 if tight * (inputs.num_segments - 1) < inputs.num_keys else 0,
        inputs.num_keys + 1,
    ):
        with pytest.raises(ValueError, match="max_seqlen_k"):
            _run(inputs, max_seqlen_k=bad)


def test_repeatability_bitwise():
    _require_program()
    inputs = ref.make_random_inputs(
        [700, 300], [700, 9000], top_k=2048, seed=40, device="cuda", peaked=True
    )
    a, b, c = _run(inputs), _run(inputs), _run(inputs)
    assert ref.same_bits(a, b) and ref.same_bits(a, c)


PARTITION_KNOBS = (
    {"grid_ctas": 37},
    {"candidate_multiplier": 2},
    {"check_period": 1},
    {"sample_tiles_max": 0},
    {"sample_shift_permille": -500},
    {
        "finalize_threads": 512
    },  # the 512-thread finalize program (the rank form where the policy admits the call)
)


@pytest.mark.parametrize(
    "knobs", PARTITION_KNOBS, ids=lambda d: "_".join(f"{k}={v}" for k, v in d.items())
)
def test_partition_knobs_do_not_change_results(knobs):
    _require_program()
    inputs = ref.make_random_inputs(
        [1500, 600], [1500, 30000], top_k=2048, seed=41, device="cuda"
    )
    base = _run(inputs)
    _judge_ok(base, ref.select_reference(inputs))
    other = _run(inputs, **knobs)
    assert ref.same_bits(base, other), f"partition knob {knobs} changed the result"


# ---------------------------------------------------------------------------
# Operator: robustness, graph capture, workspace, public entry
# ---------------------------------------------------------------------------


def test_strided_k_view_matches_contiguous_bitwise():
    _require_program()
    seg_q = [max(1, v // 16) for v in _lengths(P1_CU_SEQLENS_Q)]
    seg_k = [max(1, v // 16) for v in _lengths(P1_CU_SEQLENS_K)]
    strided = ref.make_random_inputs(
        seg_q,
        seg_k,
        top_k=256,
        seed=7701,
        device="cuda",
        k_row_stride=ref.TRAINER_KEY_ROW_STRIDE,
    )
    assert (
        strided.k.stride(0) == ref.TRAINER_KEY_ROW_STRIDE
        and not strided.k.is_contiguous()
    )
    contiguous = ref.build_inputs(
        strided.q,
        strided.k.contiguous(),
        strided.w,
        seg_q,
        seg_k,
        top_k=256,
        softmax_scale=strided.softmax_scale,
    )
    a, b = _run(strided), _run(contiguous)
    assert ref.same_bits(a, b)
    _judge_ok(a, ref.select_reference(contiguous))


def test_nonfinite_inputs_return_with_intact_structure():
    _require_program()
    inputs = ref.make_nonfinite_inputs("cuda")
    indices, scores = _run(inputs)
    torch.cuda.synchronize()
    assert (
        tuple(indices.shape) == (inputs.num_queries, inputs.top_k)
        and indices.dtype == torch.int32
    )
    assert (
        tuple(scores.shape) == (inputs.num_queries, inputs.top_k)
        and scores.dtype == torch.float32
    )
    reference = ref.select_reference(inputs)
    _judge_ok((indices, scores), reference, rows=ref.clean_rows(inputs))
    visible = ref.visible_rows(inputs)
    for t in ref.dirty_rows(inputs).tolist():
        row = indices[t].tolist()
        valid = [j for j in row if j >= 0]
        assert len(valid) == len(set(valid)) and all(
            j < int(visible[t]) for j in valid
        ), t
        assert valid == sorted(valid), t


def test_prepared_runner_graph_capture_and_replay():
    _require_program()
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=50, device="cuda"
    )
    eager = _run(inputs)
    runner = prepare_dsa_indexer_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            runner()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner()
    graph.replay()
    torch.cuda.synchronize()
    assert ref.same_bits(captured, eager)
    fresh = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=51, device="cuda"
    )
    inputs.q.copy_(fresh.q)
    inputs.k.copy_(fresh.k)
    inputs.w.copy_(fresh.w)
    graph.replay()
    torch.cuda.synchronize()
    _judge_ok((captured[0].clone(), captured[1].clone()), ref.select_reference(inputs))


def test_workspace_bound_is_respected():
    _require_program()
    inputs = ref.make_random_inputs(
        [900, 300], [900, 7000], top_k=2048, seed=60, device="cuda"
    )
    device = torch.device("cuda", torch.cuda.current_device())
    nbytes = dsa_indexer_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=device,
    )
    plan = _plan(inputs)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert nbytes == plan.workspace_bytes and plan.grid == sms
    assert (
        nbytes
        == (plan.grid_ctas * plan.block_q * plan.cand_cap + plan.staging_entries) * 8
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    indices = torch.empty(
        (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=device
    )
    scores = torch.empty(
        (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=device
    )
    runner = prepare_dsa_indexer_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        workspace_buffer=workspace,
        indices=indices,
        scores=scores,
    )
    assert runner.workspace_bytes == nbytes
    runner()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    out = runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert (
        after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0
    ), "the prepared launch allocated"
    assert (
        out[0].data_ptr() == indices.data_ptr()
        and out[1].data_ptr() == scores.data_ptr()
    )
    _judge_ok(out, ref.select_reference(inputs))
    with pytest.raises(ValueError, match="workspace_buffer needs"):
        prepare_dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            workspace_buffer=workspace[: nbytes - 8],
        )


def test_empty_rows_and_no_keys_are_all_padding():
    _require_program()
    device = torch.device("cuda")
    q = torch.zeros((0, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    k = torch.zeros((5, HEAD_DIM), dtype=torch.bfloat16, device=device)
    w = torch.zeros((0, NUM_HEADS), dtype=torch.float32, device=device)
    cu = torch.tensor([0, 0], dtype=torch.int32, device=device)
    indices, scores = cake_backend.dsa_indexer_topk(
        q, k, w, cu, torch.tensor([0, 5], dtype=torch.int32, device=device), top_k=8
    )
    assert tuple(indices.shape) == (0, 8) and tuple(scores.shape) == (0, 8)
    inputs = ref.make_random_inputs([3, 4], [0, 0], top_k=8, seed=1, device="cuda")
    indices, scores = _run(inputs)
    assert bool((indices == -1).all()) and bool(torch.isneginf(scores).all())
    _judge_ok((indices, scores), ref.select_reference(inputs))


def test_public_entry_point_matches_backend_and_warns_once():
    _require_program()
    from flashinfer.dsa_indexer import dsa_indexer_topk, dsa_indexer_topk_workspace_size

    inputs = ref.make_random_inputs(
        [200, 300], [200, 2500], top_k=128, seed=70, device="cuda"
    )
    expected = _run(inputs)
    with pytest.warns(ExperimentalWarning):
        got = dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
        )
    assert ref.same_bits(got, expected)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        again = dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            max_seqlen_q=max(inputs.seg_q_len),
            max_seqlen_k=max(inputs.seg_k_len),
        )
    assert ref.same_bits(again, expected)
    assert dsa_indexer_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    ) == dsa_indexer_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    )
    with pytest.raises(ValueError, match="backend"), _quiet_experimental():
        dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            backend="other",
        )


def test_persistent_rank_finalize_follows_the_architecture_cap():
    """Lever FRP-K (round 39): the persistent two-row pipelined rank finalize serves the staged bulk slab calls up to the
    architecture's ``rank_persist_max_k`` (sm_107a: 2048) and launches ``min(rows, rank_persist_ctas_per_sm x SMs)`` CTAs; every
    other architecture and every larger top_k keeps the one-row program (one CTA per row)."""
    for arch in cake_jit.ARCHES:
        policy = policy_for(arch)
        for top_k in (1024, 2048, 4096):
            choice = select_program(
                policy,
                num_queries=8192,
                num_keys=8192,
                num_segments=1,
                ratio=1,
                top_k=top_k,
                grid=148,
            )
            expected = (
                policy.rank_persist_max_k > 0
                and top_k <= policy.rank_persist_max_k
                and choice.rank_staged
                and choice.rank_bulk_io
                and not choice.rank_two_level
            )
            assert choice.rank_persistent is bool(expected), (arch, top_k)
            assert choice.finalize_key.endswith(":persist") is bool(expected), (
                arch,
                top_k,
            )
            if expected:
                assert choice.rank_persist_ctas == policy.rank_persist_ctas_per_sm
                assert finalize_grid(choice, 148) == min(
                    8192, policy.rank_persist_ctas_per_sm * 148
                )
                assert finalize_grid(choice, 10_000) == 8192
            else:
                assert (
                    choice.rank_persist_ctas == 0 and finalize_grid(choice, 148) == 8192
                )
            assert choice.finalize_key in cake_jit.PROGRAM_KEYS[arch], (
                arch,
                top_k,
                choice.finalize_key,
            )
        # the admissible 512-thread knob at top_k <= cap resolves to the 16-item form, which keeps the one-row program: a program
        # every architecture registers (the persistent 16-item form is never catalogued)
        knob = select_program(
            policy,
            num_queries=1500,
            num_keys=30000,
            num_segments=1,
            ratio=1,
            top_k=2048,
            finalize_threads=512,
            grid=148,
        )
        assert knob.rank_window is not None and knob.rank_items == 16, (
            arch,
            knob.finalize_key,
        )
        assert not knob.rank_persistent and knob.rank_persist_ctas == 0, (
            arch,
            knob.finalize_key,
        )
        assert knob.finalize_key.endswith(":i16:bulk"), (arch, knob.finalize_key)
        assert knob.finalize_key in cake_jit.PROGRAM_KEYS[arch], (
            arch,
            knob.finalize_key,
        )
        off = DispatchPolicy.from_record(dict(policy.record(), rank_persist_max_k=0))
        assert not off.finalize_rank_persistent_for(2048, True, False, True, False)
        on = DispatchPolicy.from_record(dict(policy.record(), rank_persist_max_k=2048))
        assert on.finalize_rank_persistent_for(2048, True, False, True, False)
        assert not on.finalize_rank_persistent_for(4096, True, False, True, False)
        assert not on.finalize_rank_persistent_for(
            2048, True, True, True, False
        )  # the two-level pools keep the one-row program
        assert not on.finalize_rank_persistent_for(
            2048, True, False, True, True
        )  # the 16-item thread form keeps the one-row program
        assert not on.finalize_rank_persistent_for(
            2048, True, False, False, False
        )  # a form of the bulk program
        assert not on.finalize_rank_persistent_for(2048, False, False, False, False)
