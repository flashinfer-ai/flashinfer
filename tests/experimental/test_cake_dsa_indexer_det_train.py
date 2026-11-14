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
#
# This suite exercises the package alone: it names the package once, through
# ``tests.cake.test_helpers.cake_package``, and imports nothing else from flashinfer,
# so a build that vendors the package under another import path runs it
# unchanged after pointing that table entry at its package.  The flashinfer
# public entry points (``flashinfer.cake_dsa_indexer_det_train``) are covered by
# ``test_cake_dsa_indexer_det_train_api.py`` and their #8928 naming rules by
# ``test_cake_dsa_indexer_det_train_names.py``.

import dataclasses
import itertools
import json
import os
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref
from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_dsa_indexer_det_train", "cake_backend")
cake_build = cake_package("cake.cake_dsa_indexer_det_train", "cake_build")
cake_jit = cake_package("cake.cake_dsa_indexer_det_train", "cake_jit")
cake_policy = cake_package("cake.cake_dsa_indexer_det_train", "cake_policy")
ABI_CONTRACT = cake_backend.ABI_CONTRACT
CONTRACT_SCALARS = cake_backend.CONTRACT_SCALARS
CONTRACT_TENSORS = cake_backend.CONTRACT_TENSORS
HEAD_DIM = cake_backend.HEAD_DIM
MAX_TOP_K = cake_backend.MAX_TOP_K
NUM_HEADS = cake_backend.NUM_HEADS
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
ZERO_SIGN_POLICIES = cake_backend.ZERO_SIGN_POLICIES
arch_for = cake_backend.arch_for
bind_program = cake_backend.bind_program
cake_dsa_indexer_det_train_topk_workspace_size = (
    cake_backend.cake_dsa_indexer_det_train_topk_workspace_size
)
generated_program_available = cake_backend.generated_program_available
plan_dsa_indexer_det_topk = cake_backend.plan_dsa_indexer_det_topk
policy_for = cake_backend.policy_for
prepare_dsa_indexer_det_topk = cake_backend.prepare_dsa_indexer_det_topk
program_for = cake_backend.program_for
record_zero_sign_policy = cake_backend.record_zero_sign_policy
validate_dsa_indexer_det_inputs = cake_backend.validate_dsa_indexer_det_inputs
visible_key_count = cake_backend.visible_key_count
visible_key_counts = cake_backend.visible_key_counts
MERGE_KEY = cake_policy.MERGE_KEY
POLICY_FIELDS = cake_policy.POLICY_FIELDS
SAMPLE_SHIFT_PERMILLE_BOUNDS = cake_policy.SAMPLE_SHIFT_PERMILLE_BOUNDS
DispatchPolicy = cake_policy.DispatchPolicy
finalize_grid = cake_policy.finalize_grid
finalize_key = cake_policy.finalize_key
scan_key = cake_policy.scan_key
scan_key_fields = cake_policy.scan_key_fields
scan_key_form = cake_policy.scan_key_form
select_program = cake_policy.select_program
stage_slug = cake_policy.stage_slug

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
    cand_cap_floor_rule=None,
    candidate_multiplier_unsampled=4,
    cand_mult_rule_unsampled=[384.0, 8],
    cand_cap_floor_unsampled=8192,
    cand_cap_ceiling=None,
    cand_cap_ceiling_unsampled=None,
    wide_first_rule=None,
    wide_wave_rule=None,
    desc_walk_kinds=None,
    split_max=32,
    split_min_range_tiles=64,
    split_units=["l6", "narrow", "wide"],
    split_wave_scope=None,
    split_fit_rule=None,
    split_resident_rule=None,
    l6_solo_rule=None,
    l6_solo_unsampled_max_queries=None,
    wide_pair_snake_only=False,
    snake_default=False,
    snake_cap_rule=None,
    snake_packed_rule=None,
    snake_l6_fallback_waves=None,
    tile_unroll_default=1,
    tile_unroll_factor=2,
    sample_fit_max_mean_tiles=512,
    sample_fit_kinds=["l6", "narrow", "pair_l6", "pair_narrow", "pair_wide", "wide"],
    sample_tiles_max=32,
    sample_tiles_short_units=16,
    sample_dispatch_mean_tiles_max=640,
    sample_tiles_tiny_units=8,
    sample_dispatch_tiny_tiles_max=128,
    sample_shift_permille=250,
    sample_shift_rule=None,
    sample_class_rule=None,
    sample_long_rule=None,
    lsf_programs=False,
    lsf_single_rule=None,
    static_scan_kinds=[],
    far_sample=0,
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
    rank_persist_spare_sms=0,
    merge_rank=False,
)
NARROW_PAIR_POLICY = DispatchPolicy.from_record(
    dict(
        _COMMON,
        l6_rule=[256.0, 512],
        wide_rule=[6.0, 300.0],
        split_wave_rule=None,
        pair_rule=[128.0, 256.0, 512],
        pair_units=["l6"],
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
        pair_units=["l6", "narrow", "wide"],
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


def _plan(inputs: ref.IndexerInputs, **extra):
    return plan_dsa_indexer_det_topk(
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
                "zero",
                "grid",
            ), (role, kind, arg)
            if kind != "grid":
                key = cake_backend.CONTRACT_ALIASES.get(arg, arg)
                assert (
                    key in CONTRACT_TENSORS
                    or key in CONTRACT_SCALARS
                    or key in ("num_queries", "num_keys")
                ), (role, arg)
    # build.json (``cake_build``) is the only record of what to compile; the registry names its
    # entries ``<library>_<arch>``
    package = cake_jit.__file__.rsplit("/", 1)[0]
    manifest = cake_build.manifest()
    libraries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert len(libraries) == len(manifest["libraries"])  # names unique
    assert set(cake_jit.ARCHES) <= set(manifest["arch_flags"])
    for name, record in cake_jit.PROGRAMS.items():
        assert record["role"] in cake_jit.STAGES, name
        assert record["arches"] and set(record["arches"]) <= set(cake_jit.ARCHES), name
        assert record["library"] == name
        for arch in record["arches"]:
            entry = libraries[f"{record['library']}_{arch}"]
            assert entry["arch"] == arch, name
            assert entry["sources"] and all(
                os.path.isfile(os.path.join(package, src)) for src in entry["sources"]
            ), name
            assert len(entry["closure_sha256"]) == 64, name
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        assert arch in cake_jit.ARCHES
        policy = policy_for(arch)
        assert set(cake_jit.POLICY[arch]) == set(POLICY_FIELDS)
        scan_keys = sorted(k for k in keys if k.startswith("scan:"))
        finalize_keys = sorted(k for k in keys if k.startswith("finalize:"))
        rank_keys = sorted(k for k in keys if k.startswith("finalize_rank:"))
        merge_rank_keys = sorted(k for k in keys if k.startswith("merge:rank:"))
        assert scan_keys and finalize_keys and MERGE_KEY in keys, arch
        assert set(keys) == set(scan_keys) | set(finalize_keys) | set(rank_keys) | set(
            merge_rank_keys
        ) | {MERGE_KEY}, arch
        # the fused merge + rank finalize programs: one merge program per slab rank window, registered exactly where
        # the record turns merge_rank on
        assert bool(merge_rank_keys) == bool(policy.merge_rank), arch
        for key in merge_rank_keys:
            window = int(key[len("merge:rank:w") :])
            assert window in policy.rank_window_variants, (arch, key)
            assert window <= policy.rank_slab_window_max, (arch, key)
            assert cake_jit.PROGRAMS[keys[key]]["role"] == "merge", (arch, key)
        # every scan key follows the key grammar; an FP8 key (``:fp8`` before ``:lsf``) runs the sampled-fit form, in
        # the plain unit order or as the snake copy of a registered plain FP8 program, and the BF16 record's rules
        # below are the BF16 programs' (the FP8 form has a record of its own, ``policy_for(arch, "fp8")``)
        for key in scan_keys:
            fields = scan_key_fields(key)
            assert fields is not None, (arch, key)
            assert (key.split(":")[-1] == "lsf") == fields.lsf, (arch, key)
            if fields.form == cake_policy.FP8:
                assert fields.sample_fit, (arch, key)
                assert not fields.snake or key.replace(":s1:", ":s0:", 1) in keys, (
                    arch,
                    key,
                )
        scan_keys = [k for k in scan_keys if scan_key_form(k) == cake_policy.BF16]
        # the rank finalize programs are registered exactly where the architecture's rule admits a call
        assert bool(rank_keys) == (
            policy.rank_finalize and policy.rank_rule is not None
        ), arch
        # the CTA pair serves exactly the unit geometries whose pair program the architecture registers
        assert policy.pair_units == sorted(
            {
                k.split(":")[1][len("pair_") :]
                for k in scan_keys
                if k.startswith("scan:pair_")
            }
        ), arch
        # the unpaired three-warpgroup preference needs the unpaired l6 program
        if (
            policy.l6_solo_rule is not None
            or policy.l6_solo_unsampled_max_queries is not None
        ):
            assert "scan:l6:u1:s0:f0" in keys, arch
        # the key-range split serves exactly the unit geometries whose split program the architecture registers
        assert policy.split_units == sorted(
            {
                k.split(":")[1][len("split_") :]
                for k in scan_keys
                if k.startswith("scan:split_")
            }
        ), arch
        # the wide-first precedence reaches the unrolled wide program in both fit forms
        if policy.wide_first_rule is not None:
            assert {"scan:wide:u2:s0:f0", "scan:wide:u2:s0:f1"} <= set(keys), arch
        # the sampled rank shift rule: a top_k floor a call can reach, a shift the default does not already give
        if policy.sample_shift_rule is not None:
            assert 1 <= policy.sample_shift_rule[0] <= MAX_TOP_K, arch
            assert policy.sample_shift_rule[1] != policy.sample_shift_permille, arch
        # a split needs grid // units >= 2 (rounds <= 0.5) and a wide unit needs rounds >= wide_rule[0], so without
        # the two-way wave split the wide split program is unreachable where wide_rule[0] > 0.5 and need not ship
        assert "scan:split_wide:u1:s0:f0" in keys or (
            policy.split_wave_rule is None and policy.wide_rule[0] > 0.5
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


_UNIT_SOURCE = re.compile(
    r"^csrc/cake_dsa_indexer_det_topk/"
    r"(cake_dsa_indexer_det_topk_[0-9a-f]{20})_(kernel|binding)\.cu$"
)
_PROGRAM_IDENTIFIER = re.compile(r"cake_dsa_indexer_det_topk_[0-9a-f]{20}")
_KNOB_DEFINE = re.compile(r"^-D(CAKE_INDEXER_[A-Z0-9_]+)=(-?[0-9]+)$")
_KERNEL_DEFINES = ("CAKE_PROGRAM_KERNEL",)
_BINDING_DEFINES = (
    "CAKE_KERNEL_SYMBOL",
    "CAKE_PROGRAM_SHIM",
    "CAKE_BINDING_THREADS",
    "CAKE_BINDING_SMEM",
)


def _unit(entry, index, kind):
    # The exported unit whose file the entry compiles as its kernel (index 0) or
    # binding (index 1) translation unit.
    match = _UNIT_SOURCE.match(entry["sources"][index])
    assert match is not None and match.group(2) == kind, (entry["name"], kind)
    return match.group(1)


def _defaults(text, macro):
    # The ``#ifndef`` default of ``macro`` in ``text`` (the program the shared unit
    # was exported for): a list of at most one value.
    return re.findall(
        rf"^#ifndef {macro}\n#define {macro} (\S+)\n#endif$", text, re.MULTILINE
    )


def _defines(entry):
    # The ``-D`` definitions of the entry's compile line, by macro.
    found = {}
    for flag in entry["nvcc_flags"]:
        if flag.startswith("-D"):
            macro, _, value = flag[2:].partition("=")
            assert macro not in found, (entry["name"], flag)
            found[macro] = value
    return found


def test_shared_kernel_sources_keep_distinct_symbols():
    # Programs whose kernel text differs only in a few constants (a scan lever-twin
    # group, the rank-finalize programs of one window form) build one kernel
    # translation unit: it names its kernel ``CAKE_PROGRAM_KERNEL`` and reads those
    # constants from ``CAKE_INDEXER_<KNOB>``, each with an ``#ifndef`` default
    # naming the program the unit was exported for; every other entry passes its
    # own values with ``-D``.  A program with a unit of its own spells its symbol.
    # All libraries of one architecture load into one process, so every entry must
    # still produce ``kernel_<program>``, and every definition an entry passes must
    # be one its source reads.
    package = cake_jit.__file__.rsplit("/", 1)[0]
    symbols = {}
    for entry in cake_build.manifest()["libraries"]:
        name, arch = entry["name"], entry["arch"]
        assert name.endswith(f"_{arch}"), name
        program = name[: -len(arch) - 1]
        unit = _unit(entry, 0, "kernel")
        with open(os.path.join(package, entry["sources"][0])) as source:
            text = source.read()
        defines = _defines(entry)
        for macro, value in defines.items():
            if macro in _BINDING_DEFINES:
                continue
            if macro not in _KERNEL_DEFINES:
                assert _KNOB_DEFINE.match(f"-D{macro}={value}"), (name, macro)
            assert len(_defaults(text, macro)) == 1, (name, macro)
        defaults = _defaults(text, "CAKE_PROGRAM_KERNEL")
        if not defaults:
            # a unit of its own: the source spells the program's symbol
            assert unit == program and "CAKE_PROGRAM_KERNEL" not in defines, name
            assert f"kernel_{program}(" in text, name
            symbol = f"kernel_{program}"
        else:
            # a shared unit: its own program takes the default, every other program
            # names its symbol on the compile line
            assert defaults == [f"kernel_{unit}"], (name, defaults)
            assert "CAKE_PROGRAM_KERNEL(" in text, name
            assert ("CAKE_PROGRAM_KERNEL" in defines) == (unit != program), name
            symbol = defines.get("CAKE_PROGRAM_KERNEL", defaults[0])
        assert symbol == f"kernel_{program}", (name, symbol)
        symbols.setdefault(program, set()).add(symbol)
    assert set(symbols) == set(cake_jit.PROGRAMS)
    assert len({s for group in symbols.values() for s in group}) == len(symbols)


_CHECK_OPEN = re.compile(r"\bTVM_FFI_CHECK(?:_CUDA_ERROR)?\(")


def _check_conditions(text):
    # Every ``TVM_FFI_CHECK(`` condition of ``text`` as ``#cond`` stringizes it
    # (unexpanded): balanced parentheses, string literals skipped, multi-line aware.
    conditions = []
    for match in _CHECK_OPEN.finditer(text):
        start = match.end()
        depth, quoted, i = 1, False, start
        while depth and i < len(text):
            c = text[i]
            if quoted:
                if c == "\\":
                    i += 1
                elif c == '"':
                    quoted = False
            elif c == '"':
                quoted = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        assert depth == 0, text[start : start + 80]
        conditions.append(text[start : i - 1])
    return conditions


def test_shared_binding_sources_keep_distinct_namespaces():
    # Programs whose launcher text differs only in the kernel symbol, the host-shim
    # namespace, the block size and the dynamic shared memory build one binding
    # translation unit: it declares and launches ``CAKE_KERNEL_SYMBOL`` with
    # ``CAKE_BINDING_THREADS`` / ``CAKE_BINDING_SMEM`` and exports ``run`` from
    # namespace ``CAKE_PROGRAM_SHIM``, each with an ``#ifndef`` default naming the
    # program the file was exported for; every other entry passes its own values
    # with ``-D``.  A program with a binding of its own spells its names.  All
    # libraries of one architecture load into one process, so every entry must
    # still export a distinct namespace, and no check condition may spell a macro
    # (``TVM_FFI_CHECK`` stringizes its condition unexpanded).
    package = cake_jit.__file__.rsplit("/", 1)[0]
    namespaces = {}
    for entry in cake_build.manifest()["libraries"]:
        name, arch = entry["name"], entry["arch"]
        program = name[: -len(arch) - 1]
        unit = _unit(entry, 1, "binding")
        with open(os.path.join(package, entry["sources"][1])) as source:
            text = source.read()
        lines = text.splitlines()
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_cuda_version, " in text, name
        assert not [
            c for c in _check_conditions(text) if "CAKE_" in c or "SMEM_TOTAL" in c
        ], name
        defines = _defines(entry)
        defaults = {macro: _defaults(text, macro) for macro in _BINDING_DEFINES}
        if not any(defaults.values()):
            # a binding of its own: the source spells the program's names
            assert unit == program, name
            assert not [m for m in _BINDING_DEFINES if m in defines], (name, defines)
            namespace = f"cake_binding_kernel_{program}"
            assert lines.count(f"namespace {namespace} {{") == 1, name
            assert f'extern "C" __global__ void kernel_{program}(' in text, name
            assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, {namespace}::Run);" in text
        else:
            assert all(len(found) == 1 for found in defaults.values()), (name, defaults)
            assert defaults["CAKE_KERNEL_SYMBOL"] == [f"kernel_{unit}"], name
            assert defaults["CAKE_PROGRAM_SHIM"] == [f"cake_binding_kernel_{unit}"]
            assert lines.count("namespace CAKE_PROGRAM_SHIM {") == 1, name
            assert 'extern "C" __global__ void CAKE_KERNEL_SYMBOL(' in text, name
            assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_PROGRAM_SHIM::Run);" in text
            # every other mention of a program identifier goes through the macros
            assert not [
                line
                for line in lines
                if _PROGRAM_IDENTIFIER.search(line)
                and not line.startswith(("#define CAKE_", "//"))
            ], name
            if unit == program:
                assert "CAKE_KERNEL_SYMBOL" not in defines, name
                assert "CAKE_PROGRAM_SHIM" not in defines, name
            else:
                # another program's binding: symbol and namespace come from the
                # compile line
                assert defines.get("CAKE_KERNEL_SYMBOL") == f"kernel_{program}", name
                assert "CAKE_PROGRAM_SHIM" in defines, name
            namespace = defines.get(
                "CAKE_PROGRAM_SHIM", defaults["CAKE_PROGRAM_SHIM"][0]
            )
        assert namespace == f"cake_binding_kernel_{program}", (name, namespace)
        namespaces.setdefault(program, set()).add(namespace)
    assert set(namespaces) == set(cake_jit.PROGRAMS)
    assert all(len(found) == 1 for found in namespaces.values()), namespaces
    assert len({next(iter(found)) for found in namespaces.values()}) == len(namespaces)


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
        dict(rank_persist_spare_sms=-1),  # the spare SMs of the persistent grid (#7548)
        dict(rank_persist_spare_sms=0.5),
        dict(far_sample=2),  # the far-sample first threshold switch: 0 or 1
        dict(far_sample=-1),
        dict(far_sample=True),
        dict(candidate_multiplier_unsampled=0),  # the unsampled capacity values
        dict(cand_mult_rule_unsampled=384),
        dict(cand_cap_floor_unsampled=8192.5),
        # the capacity ceilings: positive, at or above the floor
        dict(cand_cap_ceiling=0),
        dict(cand_cap_ceiling=4096),
        dict(cand_cap_ceiling=32768.5),
        dict(cand_cap_ceiling_unsampled=4096),
        # the kinds with a sampled-fit program: an ascending list of physical kinds
        dict(sample_fit_kinds="l6"),
        dict(sample_fit_kinds=["split"]),
        dict(sample_fit_kinds=["wide", "l6"]),
        dict(sample_fit_kinds=["l6", "l6"]),
        # the wide-first precedence floor: [min mean unit tiles] or None
        dict(wide_first_rule=512),
        dict(wide_first_rule=[]),
        dict(wide_first_rule=[512.0, 1]),
        dict(wide_first_rule=[-1.0]),
        dict(wide_first_rule=[True]),
        # the wave rule: [min narrow / wide wave ratio, min l6 / wide wave ratio, min mean unit tiles], or that with
        # [top_k (an int >= 1), exponent, packed minimum (>= the one-segment one), rise per wave, waves (an int >= 1),
        # low wave ratio (>= element 1), its tile coefficient, its packed coefficient, the one-document cap
        # coefficient, the cap's lower bound from two wide waves, the wide waves (an int >= 1) from which the next
        # element is that lower bound, that lower bound], each >= 0, or None
        dict(wide_wave_rule=1.6),
        dict(wide_wave_rule=[]),
        dict(wide_wave_rule=[1.6, 1.3]),
        dict(wide_wave_rule=[1.6, 1.3, 80.0, 2048]),
        dict(wide_wave_rule=[1.6, 1.3, 80.0, 2048, 1]),
        dict(wide_wave_rule=[1.6, 1.3, 80.0, 2048, 1.1, 112.0, 3.5, 14]),
        dict(
            wide_wave_rule=[1.7, 1.15, 64.0, 2048, 1.1, 112.0, 3.5, 14, 1.26, 40.0, 9.0]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                1.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                0,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048.0,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                True,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                -1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                64.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                -3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                0,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14.0,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.2,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                -40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                True,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                -9.0,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                True,
                8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                -8.0,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                True,
                55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                -55.0,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                True,
                8,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8.0,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                0,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                True,
                60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                -60.0,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                True,
            ]
        ),
        dict(
            wide_wave_rule=[
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
                1.0,
            ]
        ),
        dict(wide_wave_rule=[-1.0, 1.3, 80.0]),
        dict(wide_wave_rule=[1.6, 1.3, True]),
        # the descending-walk kinds: [unsampled kinds, sampled kinds], ascending lists of physical scan kinds, or None
        dict(desc_walk_kinds="wide"),
        dict(desc_walk_kinds=[]),
        dict(desc_walk_kinds=[["wide"]]),
        dict(desc_walk_kinds=[["wide"], [], []]),
        dict(desc_walk_kinds=[["wide_desc"], []]),
        dict(desc_walk_kinds=[["wide", "wide"], []]),
        dict(desc_walk_kinds=[["wide", "l6"], []]),
        dict(desc_walk_kinds=[[1], []]),
        # the units that split: an ascending list of unit geometries (may be empty)
        dict(split_units="l6"),
        dict(split_units=["split_l6"]),
        dict(split_units=["wide", "l6"]),
        dict(split_units=["l6", "l6"]),
        # the split-resident rule: [chunk, fit, resident, max splits, min splits], positive integers, 2 <= min <= max
        dict(split_resident_rule=4096),
        dict(split_resident_rule=[4096, 32768, 49152, 32]),
        dict(split_resident_rule=[4096, 32768, 49152, 32, 8, 1]),
        dict(split_resident_rule=[0, 32768, 49152, 32, 8]),
        dict(split_resident_rule=[4096, 32768, 49152.0, 32, 8]),
        dict(split_resident_rule=[4096, 32768, 49152, 32, True]),
        dict(split_resident_rule=[4096, 32768, 49152, 32, 1]),
        dict(split_resident_rule=[4096, 32768, 49152, 8, 32]),
        # the unpaired three-warpgroup preference: [max mean unit tiles >= 0, min segments >= 1] or None
        dict(l6_solo_rule=128),
        dict(l6_solo_rule=[]),
        dict(l6_solo_rule=[128.0]),
        dict(l6_solo_rule=[128.0, 2, 1]),
        dict(l6_solo_rule=[-1.0, 2]),
        dict(l6_solo_rule=[128.0, 0]),
        dict(l6_solo_rule=[128.0, 2.0]),
        dict(l6_solo_rule=[True, 2]),
        # its query bound with the sampled first threshold off: an integer >= 1 or None
        dict(l6_solo_unsampled_max_queries=0),
        dict(l6_solo_unsampled_max_queries=16384.0),
        dict(l6_solo_unsampled_max_queries=True),
        dict(l6_solo_unsampled_max_queries=[16384]),
        # the FP8 snake cap: [max segments >= 1, max waves >= 1] or None
        dict(snake_cap_rule=40),
        dict(snake_cap_rule=[]),
        dict(snake_cap_rule=[1]),
        dict(snake_cap_rule=[1, 40, 1]),
        dict(snake_cap_rule=[0, 40]),
        dict(snake_cap_rule=[1, 0]),
        dict(snake_cap_rule=[1.0, 40]),
        dict(snake_cap_rule=[True, 40]),
        # the FP8 one-segment longest-first rule: [min mean unit tiles >= 0] or None
        dict(lsf_single_rule=64.0),
        dict(lsf_single_rule=[]),
        dict(lsf_single_rule=[64.0, 1]),
        dict(lsf_single_rule=[-1.0]),
        dict(lsf_single_rule=[True]),
        dict(lsf_single_rule=["64"]),
        # the snake order of packed calls: [ascending program kinds but split, min mean tiles sampled, min mean tiles
        # unsampled or None, max waves >= 1 or None]
        dict(snake_packed_rule=48.0),
        dict(snake_packed_rule=[]),
        dict(snake_packed_rule=[["l6"], 48.0, None]),
        dict(snake_packed_rule=[["l6"], 48.0, None, 7, 1]),
        dict(snake_packed_rule=["l6", 48.0, None, 7]),
        dict(snake_packed_rule=[[], 48.0, None, 7]),
        dict(snake_packed_rule=[["split_l6"], 48.0, None, 7]),
        dict(snake_packed_rule=[["split"], 48.0, None, 7]),
        dict(snake_packed_rule=[["pair_l6", "l6"], 48.0, None, 7]),
        dict(snake_packed_rule=[["l6"], -1.0, None, 7]),
        dict(snake_packed_rule=[["l6"], None, None, 7]),
        dict(snake_packed_rule=[["l6"], True, None, 7]),
        dict(snake_packed_rule=[["l6"], 48.0, -1.0, 7]),
        dict(snake_packed_rule=[["l6"], 48.0, False, 7]),
        dict(snake_packed_rule=[["l6"], 48.0, None, 0]),
        dict(snake_packed_rule=[["l6"], 48.0, None, 7.0]),
        dict(snake_packed_rule=[["l6"], 48.0, None, True]),
        # the waves of the plain-order l6 fallback with sampling off: an integer >= 1 or None
        dict(snake_l6_fallback_waves=0),
        dict(snake_l6_fallback_waves=310.0),
        dict(snake_l6_fallback_waves=True),
        dict(snake_l6_fallback_waves=[310]),
        # the scope of the two-way wave split: [ascending unit geometries, max segments >= 1, sampled only] or None
        dict(split_wave_scope="narrow"),
        dict(split_wave_scope=[]),
        dict(split_wave_scope=[["narrow"], 1]),
        dict(split_wave_scope=[["narrow"], 1, True, 0]),
        dict(split_wave_scope=[[], 1, True]),
        dict(split_wave_scope=["narrow", 1, True]),
        dict(split_wave_scope=[["split_narrow"], 1, True]),
        dict(split_wave_scope=[["wide", "narrow"], 1, True]),
        dict(split_wave_scope=[["narrow", "narrow"], 1, True]),
        dict(split_wave_scope=[["narrow"], 0, True]),
        dict(split_wave_scope=[["narrow"], 1.0, True]),
        dict(split_wave_scope=[["narrow"], True, True]),
        dict(split_wave_scope=[["narrow"], 1, 1]),
        # the wide CTA pair on the snake walk only (sampled calls): a bool
        dict(wide_pair_snake_only=1),
        dict(wide_pair_snake_only=None),
        # the static scan kinds: an ascending list of physical scan kinds
        dict(static_scan_kinds="l6"),
        dict(static_scan_kinds=["split"]),
        dict(static_scan_kinds=["pair_l6", "l6"]),
        dict(static_scan_kinds=["l6", "l6"]),
        # the units the CTA pair serves: an ascending list of unit geometries, not empty while pair_rule is set
        dict(pair_units="l6"),
        dict(pair_units=["pair_l6"]),
        dict(pair_units=["split"]),
        dict(pair_units=["wide", "l6"]),
        dict(pair_units=["l6", "l6"]),
        dict(pair_units=[]),
        # the sampled rank shift rule: [min top_k, shift permille] or None, the shift inside the knob's bound
        dict(sample_shift_rule=4096),
        dict(sample_shift_rule=[]),
        dict(sample_shift_rule=[4096]),
        dict(sample_shift_rule=[4096, 500, 1]),
        dict(sample_shift_rule=[-1, 500]),
        dict(sample_shift_rule=[0, 500]),
        dict(sample_shift_rule=[4096, 20000]),
        dict(sample_shift_rule=[4096, -1001]),
        dict(sample_shift_rule=["a", 500]),
        dict(sample_shift_rule=[4096.0, 500]),
        dict(sample_shift_rule=[4096, 500.0]),
        dict(sample_shift_rule=[True, 500]),
        # the top_k-keyed capacity floor: [min top_k >= 1, floor slots >= 1] or None, the floor within the ceiling
        dict(cand_cap_floor_rule=16384),
        dict(cand_cap_floor_rule=[]),
        dict(cand_cap_floor_rule=[1024]),
        dict(cand_cap_floor_rule=[1024, 16384, 1]),
        dict(cand_cap_floor_rule=[0, 16384]),
        dict(cand_cap_floor_rule=[1024, 0]),
        dict(cand_cap_floor_rule=[1024.0, 16384]),
        dict(cand_cap_floor_rule=[1024, 16384.0]),
        dict(cand_cap_floor_rule=[True, 16384]),
        dict(cand_cap_floor_rule=[1024, 16384], cand_cap_ceiling=8192),
    ):
        with pytest.raises(ValueError):
            DispatchPolicy.from_record(dict(record, **bad))
    ruled = DispatchPolicy.from_record(dict(record, cand_cap_floor_rule=[1024, 16384]))
    assert ruled.cand_cap_floor_rule == [1024, 16384]
    assert ruled.record()["cand_cap_floor_rule"] == [1024, 16384]
    assert DispatchPolicy.from_record(ruled.record()) == ruled
    assert (
        DispatchPolicy.from_record(
            dict(record, cand_cap_floor_rule=[1024, 16384], cand_cap_ceiling=16384)
        ).cand_cap_ceiling
        == 16384
    )  # a ceiling at the rule's floor is well-formed
    first = DispatchPolicy.from_record(dict(record, wide_first_rule=[192.0]))
    assert first.wide_first_rule == [192.0]
    assert first.record()["wide_first_rule"] == [192.0]
    assert DispatchPolicy.from_record(first.record()) == first
    waved = DispatchPolicy.from_record(dict(record, wide_wave_rule=[1.6, 1.3, 80.0]))
    assert waved.wide_wave_rule == [1.6, 1.3, 80.0]
    assert waved.record()["wide_wave_rule"] == [1.6, 1.3, 80.0]
    assert DispatchPolicy.from_record(waved.record()) == waved
    capped = DispatchPolicy.from_record(dict(record, snake_cap_rule=(1, 40)))
    assert capped.record()["snake_cap_rule"] == [1, 40]
    assert DispatchPolicy.from_record(capped.record()) == capped
    # waves of 6-query units on 212 CTAs: 32768 queries take 26, 65536 take 52
    assert capped.snake_copy_fits(32768, 1, 6, 212)
    assert not capped.snake_copy_fits(65536, 1, 6, 212)
    assert not capped.snake_copy_fits(32768, 2, 6, 212)
    assert DispatchPolicy.from_record(record).snake_copy_fits(1 << 20, 64, 6, 212)
    single = DispatchPolicy.from_record(dict(record, lsf_single_rule=(64,)))
    assert single.record()["lsf_single_rule"] == [64]
    assert DispatchPolicy.from_record(single.record()) == single
    # mean unit tiles (Tkv - T / (2 ratio)) / S / 128: 16384 x 16384 take 64, 16384 x 15360 take 56
    assert single.lsf_single_fits(16384, 16384, 1, 1)
    assert not single.lsf_single_fits(16384, 15360, 1, 1)
    assert not single.lsf_single_fits(32768, 32768, 2, 1)
    assert not DispatchPolicy.from_record(record).lsf_single_fits(1024, 1 << 20, 1, 1)
    walked = DispatchPolicy.from_record(dict(record, desc_walk_kinds=[["wide"], []]))
    assert walked.desc_walk_kinds == [["wide"], []]
    assert walked.record()["desc_walk_kinds"] == [["wide"], []]
    assert DispatchPolicy.from_record(walked.record()) == walked
    assert walked.desc_walk_for("wide", sampled=False)
    assert not walked.desc_walk_for("wide", sampled=True)
    assert not walked.desc_walk_for("l6", sampled=False)
    packed = DispatchPolicy.from_record(
        dict(record, snake_packed_rule=(("l6", "pair_l6"), 48, None, 7))
    )
    assert packed.record()["snake_packed_rule"] == [["l6", "pair_l6"], 48, None, 7]
    assert DispatchPolicy.from_record(packed.record()) == packed
    fallback = DispatchPolicy.from_record(dict(record, snake_l6_fallback_waves=310))
    assert fallback.record()["snake_l6_fallback_waves"] == 310
    assert DispatchPolicy.from_record(fallback.record()) == fallback
    # waves of the planned program: 8-query CTA pairs on 212 CTAs (106 slots) take 310 waves for 262144 queries and 619
    # for 524288; the unpaired 6-query unit takes 207 for 262144; only with the sampled first threshold off
    assert fallback.snake_l6_fallback_for(262144, 8, True, 212, sampled=False)
    assert not fallback.snake_l6_fallback_for(524288, 8, True, 212, sampled=False)
    assert fallback.snake_l6_fallback_for(262144, 6, False, 212, sampled=False)
    assert not fallback.snake_l6_fallback_for(262144, 8, True, 212, sampled=True)
    assert not DispatchPolicy.from_record(record).snake_l6_fallback_for(
        4096, 8, True, 212, sampled=False
    )
    scaled = DispatchPolicy.from_record(
        dict(
            record,
            wide_wave_rule=(
                1.6,
                1.3,
                80.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                8.0,
                55.0,
                8,
                60.0,
            ),
        )
    )
    assert scaled.record()["wide_wave_rule"] == [
        1.6,
        1.3,
        80.0,
        2048,
        1.1,
        112.0,
        3.5,
        14,
        1.3,
        40.0,
        9.0,
        8.0,
        55.0,
        8,
        60.0,
    ]
    assert DispatchPolicy.from_record(scaled.record()) == scaled
    scoped = DispatchPolicy.from_record(
        dict(record, split_wave_scope=(("narrow", "wide"), 2, False))
    )
    assert scoped.split_wave_scope == [["narrow", "wide"], 2, False]
    assert DispatchPolicy.from_record(scoped.record()) == scoped
    bare = DispatchPolicy.from_record(dict(record, pair_rule=None, pair_units=[]))
    assert bare.pair_units == []  # no pair rule: an empty unit list is well-formed
    shifted = DispatchPolicy.from_record(dict(record, sample_shift_rule=[4096, 500]))
    assert shifted.sample_shift_rule == [4096, 500]
    assert shifted.record()["sample_shift_rule"] == [4096, 500]
    assert DispatchPolicy.from_record(shifted.record()) == shifted
    # both ends of the knob's bound are admissible shifts
    for shift in SAMPLE_SHIFT_PERMILLE_BOUNDS:
        edge = DispatchPolicy.from_record(dict(record, sample_shift_rule=[1, shift]))
        assert edge.sample_shift_for(1) == shift


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
    # the unsampled capacity values stand in when the sampled first threshold is off
    raised = DispatchPolicy.from_record(
        dict(p.record(), candidate_multiplier=16, cand_cap_floor=16384)
    )
    assert raised.cand_cap_multiplier_for(65536, 65536, 1, 1, 2048) == 16
    assert raised.cand_cap_multiplier_for(65536, 65536, 1, 1, 2048, sampled=False) == 4
    assert raised.cand_cap_multiplier_for(65536, 65536, 1, 1, 256, sampled=False) == 32
    # the top_k-keyed floor: 8192 slots below the rule's top_k, 16384 from it on; the unsampled floor and the long-unit
    # multiplier untouched
    ruled = DispatchPolicy.from_record(
        dict(p.record(), cand_cap_floor_rule=[1024, 16384])
    )
    assert [ruled.cand_cap_floor_for(k) for k in (256, 512, 1023, 1024, 2048)] == [
        8192, 8192, 8192, 16384, 16384,
    ]  # fmt: skip
    assert [
        ruled.candidate_capacity(
            k, ruled.cand_cap_multiplier_for(65536, 65536, 1, 1, k)
        )
        for k in (256, 512, 1024, 2048, 4096)
    ] == [8192, 8192, 16384, 16384, 16384]
    assert [
        ruled.candidate_capacity(
            k, ruled.cand_cap_multiplier_for(8192, 1048576, 1, 1, k)
        )
        for k in (256, 512, 2048, 4096)
    ] == [8192, 8192, 16384, 32768]
    assert ruled.cand_cap_multiplier_for(65536, 65536, 1, 1, 512, sampled=False) == 16
    assert p.cand_cap_floor_for(2048) == 8192  # no rule: the plain floor on every top_k
    geometry = dict(
        num_queries=65536, num_keys=65536, num_segments=1, ratio=1, top_k=2048, grid=148
    )
    sampled = cake_policy.select_program(raised, **geometry)
    unsampled = cake_policy.select_program(raised, sample_tiles_max=0, **geometry)
    assert sampled.cand_cap == 32768 and sampled.sample_tiles_max > 0
    assert unsampled.cand_cap == 8192 and unsampled.sample_tiles_max == 0
    assert (
        unsampled.cand_cap
        == cake_policy.select_program(p, sample_tiles_max=0, **geometry).cand_cap
    )
    assert (unsampled.scan_key, unsampled.n_split) == (
        sampled.scan_key,
        sampled.n_split,
    )
    # the ceiling lowers the multiplier once multiplier x top_k would exceed it: 16 x top_k up to 32768 slots, long units
    # and the floor untouched
    capped = DispatchPolicy.from_record(
        dict(
            p.record(),
            candidate_multiplier=16,
            cand_cap_floor=16384,
            cand_cap_ceiling=32768,
        )
    )
    assert capped.cand_cap_multiplier_for(65536, 65536, 1, 1, 2048) == 16
    assert capped.cand_cap_multiplier_for(65536, 65536, 1, 1, 4096) == 8
    assert capped.cand_cap_multiplier_for(65536, 65536, 1, 1, 3000) == 10
    assert capped.cand_cap_multiplier_for(65536, 65536, 1, 1, 256) == 64
    assert capped.cand_cap_multiplier_for(8192, 1048576, 1, 1, 2048) == 8
    assert capped.cand_cap_multiplier_for(65536, 65536, 1, 1, 2048, sampled=False) == 4
    assert [
        capped.candidate_capacity(
            k, capped.cand_cap_multiplier_for(65536, 65536, 1, 1, k)
        )
        for k in (256, 2048, 3000, 4096)
    ] == [16384, 32768, 30080, 32768]
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


# lever LSF on sm_107a: the longest-first copies its registry carries, those of the two l6 ticket programs (the plain
# l6 programs drawing from their TicketWord, which serve its packed calls)
_SM107_LSF_COPIES = frozenset({"scan:l6:u1:s0:f0:lsf", "scan:l6:u1:s0:f1:lsf"})


def _lsf_key(keys, key: str) -> str:
    """The key a packed call whose key segments the host cannot tell equal runs for the decided program ``key`` on an
    ``lsf_programs`` architecture: its longest-first copy where ``keys`` registers one, else ``key`` itself."""
    return key + ":lsf" if key + ":lsf" in keys else key


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
    assert c.workspace_bytes == 148 * 6 * 8192 * 8 + cake_policy.TICKET_TAIL_BYTES
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
    assert (
        c.workspace_bytes
        == (148 * 4 * 16384 + 64 * 9 * 2048) * 8 + cake_policy.TICKET_TAIL_BYTES
    )
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
        far_sample=1,
        finalize_threads=512,
    )
    assert (
        c.cand_cap,
        c.check_period,
        c.sample_tiles_max,
        c.far_sample,
        c.finalize_key,
    ) == (
        4096,
        1,
        0,
        1,
        "finalize_rank:t512:w16384",
    )
    assert _select(p, 8192, 8192).far_sample == p.far_sample == 0  # record default
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
    with pytest.raises(ValueError, match="far_sample"):
        _select(p, 8192, 8192, far_sample=2)


def test_wide_first_precedence_and_pair_units_guard():
    """``wide_first_rule`` gives a wide-eligible call of at least its mean tiles the wide program before the
    three-warpgroup rule; ``pair_units`` keeps the CTA pair on the units whose pair program the architecture
    registers.  The fixture takes the sm_103a rule shape (``l6_rule`` to 1024 tiles, ``pair_rule`` 64..1024 tiles,
    floor 512 tiles) on a 148-CTA grid; ``NARROW_PAIR_POLICY`` has no unroll rule, so its wide keys are ``u1``."""
    first = DispatchPolicy.from_record(
        dict(
            NARROW_PAIR_POLICY.record(),
            l6_rule=[1024.0, 512],
            pair_rule=[64.0, 1024.0, 512],
            wide_first_rule=[512.0],
        )
    )
    # 128k causal document: 512.0 mean tiles, 16384 / 148 = 110.7 wide rounds -> wide first, no pair (the unit is
    # not in pair_units); the long-unit multiplier 8 x 2048 = 16384 slots fit the 512-tile unit (offset 0) -> f1
    c = _select(first, 131072, 131072)
    assert c.scan_key == "scan:wide:u1:s0:f1" and c.block_q == 8 and not c.pair
    assert c.grid_ctas == 148 and c.cand_cap == 16384
    assert c.workspace_bytes == 148 * 8 * 16384 * 8 + cake_policy.TICKET_TAIL_BYTES
    # 64k causal document: 256 < 512 keeps the three-warpgroup pair (multiplier 4 x 2048 = 8192 slots, fit)
    assert _select(first, 65536, 65536).scan_key == "scan:pair_l6:u1:s0:f1"
    # 4096 tail queries over 128k keys: 1008 tiles but 512 / 148 = 3.46 rounds < 6 -- ineligible for wide, so the
    # precedence never demotes the call; the pair stays (the guard against any narrow demotion)
    assert _select(first, 4096, 131072).scan_key == "scan:pair_l6:u1:s0:f0"
    # the context-parallel tail of 496 tiles: the same ineligible pair (the 480-tile offset leaves no fitting unit)
    assert _select(first, 4096, 65536).scan_key == "scan:pair_l6:u1:s0:f0"
    # 8k causal document: 32 tiles, work 6.9 x 32 = 221 < 300 -> ineligible, below the pair floor: plain l6
    assert _select(first, 8192, 8192).scan_key == "scan:l6:u1:s0:f1"
    # top_k 256 at 64k: no three-warpgroup floor, the wide unit as before (the precedence adds no new wide call)
    assert _select(first, 65536, 65536, K=256).kind == "wide"
    # the guard: the same record with the wide unit admitted to the pair composes pair_wide on the wide-first call,
    # a program sm_103a does not register -- the field, not the pair interval, keeps the dispatch on shipped programs
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    open_ = DispatchPolicy.from_record(dict(first.record(), pair_units=["l6", "wide"]))
    assert _select(open_, 131072, 131072).scan_key == "scan:pair_wide:u1:s0:f1"
    with pytest.raises(NotImplementedError, match="pair_wide"):
        program_for("sm_103a", "scan:pair_wide:u1:s0:f1")
    # a hostile record (no l6 rule, the pair interval open, the floor at zero) still selects sm_103a programs only
    hostile = DispatchPolicy.from_record(
        dict(
            first.record(),
            l6_rule=None,
            pair_rule=[0.0, 4096.0, 512],
            wide_first_rule=[0.0],
        )
    )
    registered = cake_jit.PROGRAM_KEYS["sm_103a"]
    for T, Tkv, S, ratio, top_k in _dense_geometries(hostile):
        c = _select(hostile, T, Tkv, S=S, K=top_k, ratio=ratio)
        assert c.kind not in ("pair_wide", "pair_narrow"), (T, Tkv, S, ratio, top_k)
        assert c.scan_key in registered, (T, Tkv, S, ratio, top_k, c.scan_key)
    hostile_open = DispatchPolicy.from_record(
        dict(hostile.record(), pair_units=["l6", "wide"])
    )
    assert _select(hostile_open, 131072, 131072).kind == "pair_wide"


def test_wide_wave_rule_compares_waves_of_the_persistent_grid():
    """``wide_wave_rule`` replaces ``wide_rule`` and ``wide_first_rule``: the wide unit takes a call when the unit the
    other rules leave needs at least the record's ratio of the wide unit's waves of the grid (``ceil(units / grid)``),
    against the three-warpgroup unit also from a mean unit tile count on.  The fixture takes ``NARROW_PAIR_POLICY``
    (``l6_rule`` to 256 tiles at top_k >= 512) with the rule [1.7, 1.3, 84] on a 148-CTA grid."""
    waved = DispatchPolicy.from_record(
        dict(NARROW_PAIR_POLICY.record(), wide_wave_rule=[1.7, 1.3, 84.0])
    )

    def unit(T, Tkv, K=2048):
        return waved.unit_block_q_for(T, Tkv, 1, 1, 148, K)

    # 4096 queries over 65536 keys (496 mean tiles): narrow 1024 units = 7 waves, wide 512 = 4 waves, 7 >= 6.8 -> wide
    assert unit(4096, 65536) == 8
    # 2560 queries over 49152 keys: narrow 640 units = 5 waves, wide 320 = 3 waves, 5 < 5.1 -> narrow
    assert unit(2560, 49152) == 4
    # 16384 queries over 20480 keys (96 mean tiles, three-warpgroup): l6 2731 units = 19 waves, wide 2048 = 14
    # waves, 19 >= 18.2 from the 84-tile floor on -> wide; at 18432 keys (80 mean tiles) below the floor -> l6
    assert unit(16384, 20480) == 8
    assert unit(16384, 18432) == 6
    # 8k causal document (32 mean tiles): l6 10 waves against wide 7, 10 >= 9.1 but below the floor -> l6
    assert unit(8192, 8192) == 6
    # top_k 256 has no three-warpgroup unit: 712 queries over 4096 keys compare narrow 2 waves with wide 1 -> wide
    assert unit(712, 4096, K=256) == 8
    # the record without the field keeps the rounds-and-work rule: 512 / 148 = 3.5 rounds < 6 -> narrow
    assert NARROW_PAIR_POLICY.unit_block_q_for(4096, 65536, 1, 1, 148, 2048) == 4
    # a fifteen-element rule [.., 84, 2048, 1.1, 112, 3.5, 14, 1.3, 40, 9, 1e6, 1e6, 8, 1e6] (no low wave-ratio
    # minimum: its ratio is the second element's; no one-document cap: 1e6 is above every minimum): the same
    # decisions on one document of at least 14 wide waves at top_k <= 2048 (16384 queries: 14 wide waves, 19 l6
    # waves)
    scaled = DispatchPolicy.from_record(
        dict(
            NARROW_PAIR_POLICY.record(),
            wide_wave_rule=[
                1.7,
                1.3,
                84.0,
                2048,
                1.1,
                112.0,
                3.5,
                14,
                1.3,
                40.0,
                9.0,
                1e6,
                1e6,
                8,
                1e6,
            ],
        )
    )

    def scaled_unit(T, Tkv, K=2048, S=1):
        return scaled.unit_block_q_for(T, Tkv, S, 1, 148, K)

    for T, Tkv in ((16384, 20480), (16384, 18432), (8192, 8192), (4096, 65536)):
        assert scaled_unit(T, Tkv) == unit(T, Tkv)
    assert scaled_unit(16384, 18432, K=1024) == unit(16384, 18432, K=1024) == 6
    # 8192 queries take 7 wide waves (10 l6 waves): the floor rises by 3.5 x 7 to 108.5, so 96 mean tiles keep l6
    # where the three-element rule takes the wide unit, and 128 take it
    assert unit(8192, 16384) == 8
    assert scaled_unit(8192, 16384) == 6
    assert scaled_unit(8192, 20480) == 8
    # two packed documents of 16384 queries in all need 112 mean tiles: 104 keep l6, 112 take the wide unit
    assert scaled_unit(16384, 34816, S=2) == 6
    assert scaled_unit(16384, 36864, S=2) == 8
    # above top_k 2048 the floor grows as (top_k / 2048) ** 1.1: 131.2 at 3072, 180.1 at 4096
    assert scaled_unit(16384, 24576, K=3072) == 6
    assert scaled_unit(16384, 26624, K=3072) == 8
    assert unit(16384, 20480, K=4096) == 8
    assert scaled_unit(16384, 20480, K=4096) == 6
    assert scaled_unit(16384, 32768, K=4096) == 8
    # [1.7, 1.15, 64, 2048, 1.1, 112, 3.5, 14, 1.26, 40, 9, 8, 55, 8, 60]: below a wave ratio W = 1.26 the minimum is at
    # least 40 / (W - 1) mean tiles, on a packed call also 9 / (W - 1) ** 2.  4096 queries take 4 wide waves against 5
    # l6 waves (W = 1.25), so they need 160 mean tiles (packed too: 9 x 16 = 144) where 4096 x 16384 (112 tiles) clear
    # the 99-tile minimum of 4 wide waves; 8520 queries take 8 wide waves against 10 l6 waves (W = 1.25) and need 160
    # too; 6144 queries take 6 wide waves against 7 l6 waves (W = 7 / 6) and need 240, on two documents 324.  From
    # W = 1.26 on, one document of fewer than 14 wide waves needs at most 8 / (W - 1) ** 2, from two wide waves on at
    # most the larger of that and 55, from 8 wide waves on of that and 60: 8192 queries (7 wide waves, 10 l6 waves,
    # W = 10 / 7: 43.6) need 55 instead of 88.5, 2048 queries (2 wide waves, 3 l6 waves: 32) 55 instead of 106 but
    # 112 on two documents, 11776 queries (10 wide waves, 14 l6 waves, W = 1.4: 50) 60 instead of 78, 9216 queries
    # (8 wide waves, 11 l6 waves, W = 1.375: 56.9) 60 instead of 85, and 1024 queries (1 wide wave, 2 l6 waves) 8
    # instead of 109.5; 16384 queries (14 wide waves, 19 l6 waves) keep the 64-tile minimum
    low = DispatchPolicy.from_record(
        dict(
            NARROW_PAIR_POLICY.record(),
            l6_rule=[1024, 512],
            wide_wave_rule=[
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
        )
    )

    def low_unit(T, Tkv, K=2048, S=1):
        return low.unit_block_q_for(T, Tkv, S, 1, 148, K)

    assert low_unit(4096, 16384) == 6
    assert low_unit(4096, 2048 + 159 * 128) == 6
    assert low_unit(4096, 2048 + 160 * 128) == 8
    assert low_unit(4096, 2048 + 159 * 256, S=2) == 6
    assert low_unit(4096, 2048 + 160 * 256, S=2) == 8
    assert low_unit(8520, 4260 + 159 * 128) == 6
    assert low_unit(8520, 4260 + 160 * 128) == 8
    assert low_unit(6144, 3072 + 239 * 128) == 6
    assert low_unit(6144, 3072 + 240 * 128) == 8
    assert low_unit(6144, 3072 + 323 * 256, S=2) == 6
    assert low_unit(6144, 3072 + 324 * 256, S=2) == 8
    assert low_unit(8192, 4096 + 54 * 128) == 6
    assert low_unit(8192, 4096 + 55 * 128) == 8
    assert low_unit(2048, 1024 + 54 * 128) == 6
    assert low_unit(2048, 1024 + 55 * 128) == 8
    assert low_unit(11776, 5888 + 55 * 128) == 6
    assert low_unit(11776, 5888 + 59 * 128) == 6
    assert low_unit(11776, 5888 + 60 * 128) == 8
    # the top_k factor applies after the cap: 60 x 2 ** 1.1 = 128.6 at top_k 4096 (78 x 2 ** 1.1 = 167.2 without it)
    assert low_unit(11776, 5888 + 128 * 128, K=4096) == 6
    assert low_unit(11776, 5888 + 129 * 128, K=4096) == 8
    assert low_unit(9216, 4608 + 59 * 128) == 6
    assert low_unit(9216, 4608 + 60 * 128) == 8
    assert low_unit(2048, 1024 + 111 * 256, S=2) == 6
    assert low_unit(2048, 1024 + 112 * 256, S=2) == 8
    assert low_unit(1024, 512 + 7 * 128) == 6
    assert low_unit(1024, 512 + 8 * 128) == 8
    assert low_unit(16384, 8192 + 63 * 128) == 6
    assert low_unit(16384, 8192 + 64 * 128) == 8


def test_sample_shift_rule_precedence():
    """``sample_shift_rule`` replaces the sampled rank shift from its ``top_k`` floor on; below the floor, and on a
    record without a rule, ``sample_shift_permille`` stands; an explicit ``sample_shift_permille`` knob overrides both.
    The shift is a launch scalar only: the programs, capacity and workspace of a call do not depend on it."""
    p = NARROW_PAIR_POLICY
    assert p.sample_shift_rule is None
    assert p.sample_shift_for(2048) == 250 and p.sample_shift_for(4096) == 250
    shifted = DispatchPolicy.from_record(
        dict(p.record(), sample_shift_rule=[4096, 500])
    )
    assert shifted.sample_shift_for(4095) == 250
    assert shifted.sample_shift_for(4096) == 500
    for K, shift in ((2048, 250), (4096, 500)):
        base, rule = _select(p, 8192, 8192, K=K), _select(shifted, 8192, 8192, K=K)
        assert base.sample_shift_permille == 250
        assert rule.sample_shift_permille == shift
        assert (rule.keys, rule.cand_cap, rule.workspace_bytes) == (
            base.keys,
            base.cand_cap,
            base.workspace_bytes,
        )
        # the knob wins over the rule and over the default alike
        for q in (p, shifted):
            knob = _select(q, 8192, 8192, K=K, sample_shift_permille=-500)
            assert knob.sample_shift_permille == -500


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
    (round 2 review: ``scan:narrow:u1:s1:f1`` on sm_107a at 6018 queries x 11754 keys was selectable but unregistered).
    Both sample modes: the unsampled plan family (``sample_tiles_max=0``, the ``*_unsampled`` capacity values) is
    otherwise unseen by the grid.  With and without the registry: ``l6_solo_rule`` is on only with it, and without it a
    packed call names the :lsf copy of its program, which the backend runs only where the registry has it."""
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
                    for sample_tiles_max, scan_keys in itertools.product(
                        (None, 0), (None, keys)
                    ):
                        choice = select_program(
                            policy,
                            num_queries=T,
                            num_keys=Tkv,
                            num_segments=S,
                            ratio=ratio,
                            top_k=top_k,
                            grid=grid,
                            sample_tiles_max=sample_tiles_max,
                            max_seqlen_k=bound,
                            scan_keys=scan_keys,
                        )
                        selected = [choice.scan_key, choice.finalize_key] + (
                            [MERGE_KEY] if choice.n_split > 1 else []
                        )
                        # without a registry the plan names the :lsf copy wherever the record asks for it; the backend
                        # passes the registry, which runs the program itself where its copy is not registered (sm_107a)
                        if (
                            scan_keys is None
                            and arch == "sm_107a"
                            and choice.scan_key not in keys
                        ):
                            selected[0] = choice.scan_key.removesuffix(":lsf")
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
                                sample_tiles_max,
                                scan_keys is None,
                                key,
                            )


def test_split_units_keep_every_split_plan_registered():
    """A unit outside ``split_units`` never splits: on sm_107a (no ``split_l6`` program) an l6 unit whose mean is exactly
    2 x ``split_min_range_tiles`` tiles (T = 2, Tkv = 8193: 64.0 mean tiles, the l6 rule's bound) stays one unsplit (paired) unit,
    where the split rule alone gives two splits; every plan of a geometry grid around the split and l6 bounds runs a
    registered scan program on every architecture and SM count."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    p = policy_for("sm_107a")
    assert "l6" not in p.split_units
    assert p.split_rule_for(2, 8193, 1, 1, 212, p.block_q_l6) == 2
    assert p.n_split_for(2, 8193, 1, 1, 212, p.block_q_l6) == 1
    c = select_program(
        p, num_queries=2, num_keys=8193, num_segments=1, ratio=1, top_k=2048, grid=212
    )
    assert (c.kind, c.n_split) == (
        "pair_l6",
        1,
    ) and c.scan_key in cake_jit.PROGRAM_KEYS["sm_107a"]
    # without the guard (every unit in split_units) the same call composes the unregistered split_l6 key
    loose = DispatchPolicy.from_record(
        dict(p.record(), split_units=["l6", "narrow", "wide"])
    )
    assert (
        select_program(
            loose,
            num_queries=2,
            num_keys=8193,
            num_segments=1,
            ratio=1,
            top_k=2048,
            grid=212,
        ).scan_key
        == "scan:split_l6:u1:s0:f0"
    )
    for arch, grids in (
        ("sm_100a", (148, 152)),
        ("sm_103a", (148, 152)),
        ("sm_107a", (208, 212, 216)),
    ):
        policy, keys = policy_for(arch), cake_jit.PROGRAM_KEYS[arch]
        for grid in grids:
            for T in (1, 2, 4, 6, 16, 64, 256):
                for tiles in (63, 64, 65, 96, 128, 129, 256, 1024):
                    for S, delta in ((1, 0), (1, 64), (2, 0)):
                        Tkv = S * tiles * 128 + T // 2 + delta
                        for sample in (None, 0):
                            c = select_program(
                                policy,
                                num_queries=T,
                                num_keys=Tkv,
                                num_segments=S,
                                ratio=1,
                                top_k=2048,
                                grid=grid,
                                sample_tiles_max=sample,
                                scan_keys=keys,
                            )
                            assert c.scan_key in keys, (
                                arch,
                                grid,
                                T,
                                Tkv,
                                S,
                                sample,
                                c.scan_key,
                            )


def test_l6_solo_rule_prefers_the_unpaired_l6_unit_where_registered_and_unsplit():
    """sm_107a's ``l6_solo_rule`` [128.0, 2]: a call that does not split takes the three-warpgroup unit without a CTA pair
    with sampling off always, and sampled with >= 2 segments or <= 128 mean unit tiles; never where the l6 unit would
    split, where the plan splits, or where the program it picks (here ``scan:l6:u1:s1:f0``) is not registered.  The
    workspace bound covers both sampling modes."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    p, keys = policy_for("sm_107a"), cake_jit.PROGRAM_KEYS["sm_107a"]
    assert p.l6_solo_rule == [128.0, 2]
    plain = DispatchPolicy.from_record(dict(p.record(), l6_solo_rule=None))

    def plans(T, Tkv, S, sample):
        kw = dict(
            num_queries=T,
            num_keys=Tkv,
            num_segments=S,
            ratio=1,
            top_k=2048,
            grid=212,
            sample_tiles_max=sample,
        )
        return select_program(p, scan_keys=keys, **kw), select_program(
            plain, scan_keys=keys, **kw
        )

    def solo(c):
        return (c.kind, c.block_q, c.pair, c.n_split, c.grid_ctas) == (
            "l6",
            6,
            False,
            1,
            212,
        ) and c.scan_key in keys

    # one causal 64k document (256 mean tiles): sampled keeps the snake pair wide program, unsampled takes l6
    rule, base = plans(65536, 65536, 1, None)
    assert rule == base and base.scan_key == "scan:pair_wide:u1:s1:f1"
    rule, base = plans(65536, 65536, 1, 0)
    assert (
        solo(rule) and rule.scan_key == "scan:l6:u1:s1:f1" and base.kind == "pair_wide"
    )
    # eight causal 8k documents (32 mean tiles, >= 2 segments): the pair l6 program becomes unpaired in both modes (in the
    # plain unit order: snake_packed_rule), in its longest-first copy (the snake pair l6 program has none)
    for sample in (None, 0):
        rule, base = plans(65536, 65536, 8, sample)
        assert solo(rule), sample
        assert rule.scan_key == _lsf_key(keys, "scan:l6:u1:s0:f1"), sample
        assert base.scan_key == "scan:pair_l6:u1:s1:f1", sample
    # one sampled document of 100 mean tiles (<= 128): unpaired l6
    rule, base = plans(16384, 16384 // 2 + 100 * 128, 1, None)
    assert solo(rule) and base.kind != "l6"
    # one causal 512k document (2048 mean tiles, no sampled fit): the unpaired l6 key would be scan:l6:u1:s1:f0, which is not
    # registered, so the unsampled call keeps the pair wide program (619 waves, beyond snake_l6_fallback_waves)
    assert "scan:l6:u1:s1:f0" not in keys
    rule, base = plans(524288, 524288, 1, 0)
    assert rule == base and base.scan_key == "scan:pair_wide:u1:s1:f0"
    # a split call keeps its split (4 queries over 22018 keys: 12 narrow splits, split_fit_rule)
    rule, base = plans(4, 22018, 1, 0)
    assert rule == base and (base.kind, base.n_split) == ("split", 12)
    # 600 queries over 100 mean tiles: the l6 unit would split two ways (no split_l6 program), so the plan stays
    assert p.split_rule_for(600, 600 // 2 + 100 * 128, 1, 1, 212, p.block_q_l6) == 2
    rule, base = plans(600, 600 // 2 + 100 * 128, 1, 0)
    assert rule == base and base.kind != "l6"
    # without a registry the rule stays off: the unsampled 64k document keeps the pair wide program
    assert (
        select_program(
            p,
            num_queries=65536,
            num_keys=65536,
            num_segments=1,
            ratio=1,
            top_k=2048,
            grid=212,
            sample_tiles_max=0,
        ).scan_key
        == "scan:pair_wide:u1:s1:f1"
    )
    # the backend plan passes the registry's keys; the workspace bound is the larger of the two sampling modes: 1024 queries
    # over 200 mean tiles keep the four-query pair narrow unit sampled and take the six-query l6 unit unsampled
    geometry = (1024, 1024 // 2 + 200 * 128, 1)
    sampled = plan_dsa_indexer_det_topk(
        *geometry, top_k=2048, arch="sm_107a", num_sms=212
    )
    unsampled = plan_dsa_indexer_det_topk(
        *geometry, top_k=2048, arch="sm_107a", num_sms=212, sample_tiles_max=0
    )
    assert (sampled.kind, sampled.block_q) == ("pair_narrow", 4)
    assert solo(unsampled) and unsampled.scan_key == "scan:l6:u1:s0:f0"
    assert unsampled.workspace_bytes > sampled.workspace_bytes
    assert (
        cake_dsa_indexer_det_train_topk_workspace_size(
            *geometry, top_k=2048, arch="sm_107a", num_sms=212
        )
        == unsampled.workspace_bytes
    )
    # the other architectures carry no sampled rule; their unsampled bound (16384 queries) leaves a 64k-query call alone:
    # both sampling modes keep the same unit and pair
    for arch in ("sm_100a", "sm_103a"):
        assert policy_for(arch).l6_solo_rule is None
        assert policy_for(arch).l6_solo_unsampled_max_queries == 16384
        a = plan_dsa_indexer_det_topk(
            65536, 65536, 1, top_k=2048, arch=arch, num_sms=148
        )
        b = plan_dsa_indexer_det_topk(
            65536, 65536, 1, top_k=2048, arch=arch, num_sms=148, sample_tiles_max=0
        )
        assert (a.kind, a.block_q, a.pair) == (b.kind, b.block_q, b.pair), arch


def test_desc_walk_kinds_runs_the_descending_wide_twin_with_sampling_off():
    """sm_103a ``desc_walk_kinds`` ``[["wide"], []]``: a call with the sampled first threshold off whose plan is the
    single-CTA wide program runs its descending-walk twin (key kind ``wide_desc``: the same unit, split, tile unroll, fit
    and longest-first form; the same ids and score bits); sampled calls, every other kind and sm_100a / sm_107a (field
    None) keep their program.  On a grid of shapes in both modes and on every frozen SM count, each plan equals the plan
    of the record without the field except for exactly that key; the sampled list ``[["wide"], ["wide"]]`` moves sampled
    wide calls the same way; without a registry the field is inert."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for arch in ("sm_100a", "sm_107a"):
        assert policy_for(arch).desc_walk_kinds is None
        assert not any(":wide_desc:" in k for k in cake_jit.PROGRAM_KEYS[arch])
    p, keys = policy_for("sm_103a"), cake_jit.PROGRAM_KEYS["sm_103a"]
    assert p.desc_walk_kinds == [["wide"], []]
    desc = sorted(k for k in keys if k.startswith("scan:wide_desc:"))
    assert len(desc) == 8
    assert all(k.replace("scan:wide_desc:", "scan:wide:") in keys for k in desc)
    plain = DispatchPolicy.from_record(dict(p.record(), desc_walk_kinds=None))
    both = DispatchPolicy.from_record(
        dict(p.record(), desc_walk_kinds=[["wide"], ["wide"]])
    )

    def plan(policy, T, Tkv, S, sample, grid, top_k=2048, scan_keys=keys):
        return select_program(
            policy,
            num_queries=T,
            num_keys=Tkv,
            num_segments=S,
            ratio=1,
            top_k=top_k,
            grid=grid,
            sample_tiles_max=sample,
            scan_keys=scan_keys,
        )

    def twin(choice):
        key = choice.scan_key.replace("scan:wide:", "scan:wide_desc:", 1)
        if choice.physical_kind != "wide" or key not in keys:
            return choice
        return dataclasses.replace(choice, scan_key=key)

    moved = {False: 0, True: 0}
    for q, k, S, top_k, sample, grid in itertools.product(
        (1024, 4096, 8192, 16384),
        (32768, 131072, 262144),
        (1, 2, 8),
        (512, 2048),
        (None, 0),
        _FROZEN_GRIDS["sm_103a"],
    ):
        T, Tkv = q * S, k * S
        if q > k or T > 131072:
            continue
        base = plan(plain, T, Tkv, S, sample, grid, top_k)
        sampled = base.sample_tiles_max > 0
        want = base if sampled else twin(base)
        assert plan(p, T, Tkv, S, sample, grid, top_k) == want, (T, Tkv, S, sample)
        assert plan(both, T, Tkv, S, sample, grid, top_k) == twin(base)
        assert plan(p, T, Tkv, S, sample, grid, top_k, scan_keys=None) == plan(
            plain, T, Tkv, S, sample, grid, top_k, scan_keys=None
        )
        moved[sampled] += twin(base) != base
    assert moved[False] > 0 and moved[True] > 0, moved


def test_l6_solo_unsampled_max_queries_on_sm_100a_and_sm_103a():
    """sm_100a / sm_103a ``l6_solo_unsampled_max_queries`` 16384 (no ``l6_solo_rule``): a call with the sampled first
    threshold off and at most 16384 queries whose plan is the CTA-pair l6 unit without a split runs the unpaired l6 unit;
    a wide or narrow plan keeps its unit
    (the descending three-warpgroup program; same ids and score bits, round-19 lane Q screen: GB200 sink calls of 32k+
    queries lose with it, every call up to 16384 queries gains or holds).  On a grid of shapes in both modes and on every
    frozen SM count, each plan equals the plan of the record without the field except exactly there; sampled calls and
    sm_107a (field None) are unchanged; without a registry the preference stays off."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    assert policy_for("sm_107a").l6_solo_unsampled_max_queries is None
    for arch in ("sm_100a", "sm_103a"):
        p, keys = policy_for(arch), cake_jit.PROGRAM_KEYS[arch]
        assert p.l6_solo_rule is None and p.l6_solo_unsampled_max_queries == 16384
        plain = DispatchPolicy.from_record(
            dict(p.record(), l6_solo_unsampled_max_queries=None)
        )
        unpaired = DispatchPolicy.from_record(dict(plain.record(), pair_rule=None))

        def plan(policy, T, Tkv, S, sample, grid, top_k=2048, scan_keys=keys):
            return select_program(
                policy,
                num_queries=T,
                num_keys=Tkv,
                num_segments=S,
                ratio=1,
                top_k=top_k,
                grid=grid,
                sample_tiles_max=sample,
                scan_keys=scan_keys,
            )

        for grid in _FROZEN_GRIDS[arch]:
            # eight packed 2048 x 32768 segments (16384 queries, 248 mean tiles): pair l6 sampled, unpaired l6 unsampled
            base = plan(plain, 16384, 262144, 8, 0, grid)
            got = plan(p, 16384, 262144, 8, 0, grid)
            assert (base.kind, base.n_split) == ("pair_l6", 1), (arch, grid)
            assert (got.kind, got.block_q, got.pair, got.n_split) == ("l6", 6, False, 1)
            assert got.scan_key in keys and got.grid_ctas == grid, (arch, grid)
            assert plan(p, 16384, 262144, 8, None, grid) == plan(
                plain, 16384, 262144, 8, None, grid
            )
            # 16385 queries: past the bound, the pair stays; without a registry the preference stays off
            assert plan(p, 16385, 262144 + 8, 8, 0, grid) == plan(
                plain, 16385, 262144 + 8, 8, 0, grid
            )
            assert plan(p, 16384, 262144, 8, 0, grid, scan_keys=None) == plan(
                plain, 16384, 262144, 8, 0, grid, scan_keys=None
            )
        moved = 0
        for q, k, S, top_k, sample, grid in itertools.product(
            (1024, 2048, 4096, 8192, 16384, 32768),
            (8192, 16384, 32768, 65536, 131072),
            (1, 2, 4, 8),
            (512, 2048),
            (None, 0),
            _FROZEN_GRIDS[arch],
        ):
            T, Tkv = q * S, k * S
            if q > k or T > 65536:
                continue
            base = plan(plain, T, Tkv, S, sample, grid, top_k)
            got = plan(p, T, Tkv, S, sample, grid, top_k)
            want = base
            if (
                sample == 0
                and T <= 16384
                and (base.kind, base.n_split) == ("pair_l6", 1)
            ):
                want = plan(unpaired, T, Tkv, S, sample, grid, top_k)
                assert (want.kind, want.n_split) == ("l6", 1), (arch, T, Tkv, S)
                moved += 1
            assert got == want, (arch, T, Tkv, S, top_k, sample, grid, got.scan_key)
            assert got.scan_key in keys, (arch, T, Tkv, S, got.scan_key)
        assert moved, arch


def test_split_fit_rule_takes_fitting_splits_in_whole_merge_chunks():
    """``split_fit_rule`` [4096] on every architecture (sm_100a and sm_103a with their ``split_resident_rule`` off): a
    split call whose mean unit keys fit ceil(keys / top_k) <= split_max splits takes that count rounded up to whole
    4096-entry merge chunks; a call that needs more keeps the floor rule, and so does a call the floor rule does not
    split."""
    for arch, grid in (("sm_100a", 152), ("sm_103a", 152), ("sm_107a", 212)):
        p = policy_for(arch)
        assert p.split_fit_rule == [4096], arch
        p = DispatchPolicy.from_record(dict(p.record(), split_resident_rule=None))
        # 4 queries x 22018 keys: floor rule 5, fit ceil(22016 / 2048) = 11 -> 12 (two splits per chunk)
        assert p.split_rule_for(4, 22018, 1, 1, grid, p.block_q_l6, True) == 5
        assert p.split_rule_for(4, 22018, 1, 1, grid, p.block_q_l6, True, 2048) == 12
        assert p.split_rule_for(4, 30466, 1, 1, grid, p.block_q_l6, False, 2048) == 16
        # 41986 keys need 21 splits > split_max: the floor rule's 10
        assert p.split_rule_for(4, 41986, 1, 1, grid, p.block_q_l6, True, 2048) == 10
        # top_k 4096: one split per chunk, ceil(22016 / 4096) = 6
        assert p.split_rule_for(4, 22018, 1, 1, grid, p.block_q_l6, True, 4096) == 6
        # the floor rule does not split 8192 keys: no fit split either
        assert p.split_rule_for(1, 8192, 1, 1, grid, p.block_q_l6, True, 2048) == 1


def test_split_resident_rule_keeps_the_merge_resident():
    """``split_resident_rule`` [4096, 32768, 49152, 32, 8] on sm_100a and sm_103a (sm_107a: None, ``split_fit_rule``): a
    call the floor rule splits takes ``min(grid // units, 49152 // top_k, 32, max(8, n_fit if n_fit x top_k <= 32768
    else 32))`` splits, at least 2; ``n_fit`` = ceil(mean unit keys / top_k) in whole 4096-entry merge chunks.  A call
    the floor rule does not split, or one without ``top_k``, keeps the floor rule."""
    assert policy_for("sm_107a").split_resident_rule is None
    for arch in ("sm_100a", "sm_103a"):
        p = policy_for(arch)
        assert p.split_resident_rule == [4096, 32768, 49152, 32, 8], arch
        for grid in (148, 152):

            def n(T, Tkv, block_q=p.block_q_l6, top_k=None, sampled=True):
                return p.split_rule_for(T, Tkv, 1, 1, grid, block_q, sampled, top_k)

            # 4 queries x 22018 keys (one l6 unit): n_fit 11 -> 12 at top_k 2048 (24576 entries)
            assert n(4, 22018) == 5 and n(4, 22018, top_k=2048) == 12
            assert n(4, 30466, top_k=2048, sampled=False) == 16
            # 41986 keys: n_fit 22 x 2048 > 32768 entries, so 32, capped at the 24 splits the merge keeps resident
            assert n(4, 41986, top_k=2048) == 24
            # top_k 4096: 6 fitting splits lifted to 8; 65536 keys: 16 fitting splits, 12 resident
            assert n(4, 22018, top_k=4096) == 8 and n(4, 65536, top_k=4096) == 12
            # top_k 256 / 512: n_fit 64 / 32 (16384 entries) capped at 32 splits
            assert n(1, 16384, p.block_q_narrow, 256) == 32
            assert n(1, 16384, p.block_q_narrow, 512) == 32
            # 64 queries (16 narrow units): the grid caps the count
            assert n(64, 16384, p.block_q_narrow, 256) == grid // 16
            # the floor rule does not split 8192 keys: no split either
            assert n(1, 8192, top_k=2048) == 1
        # the FP8 record keeps every split field of the measured FP8 plans (no split-resident rule)
        fp8 = cake_policy.fp8_policy_record(p.record(), cake_jit.POLICY_FP8[arch])
        assert fp8["split_resident_rule"] is None and fp8["split_fit_rule"] == [4096]


def test_split_wave_scope_and_wide_pair_snake_only_on_sm_107a():
    """sm_107a's ``split_wave_scope`` [["narrow"], 1, True]: the two-way wave split of ``split_wave_rule`` stays on sampled
    single-segment calls of the narrow unit only (never with the sampled first threshold off, and never for the
    hypothetical l6 split of the ``l6_solo_rule`` check); ``wide_pair_snake_only``: a sampled call runs the unpaired wide
    program where the wide CTA pair would not walk the snake order.  On a grid of shapes in both modes every plan is the
    plan of the record with both fields off after two substitutions: outside the kept narrow wave split, the plan of that
    record without ``split_wave_rule``; then, sampled, a non-snake ``pair_wide`` plan becomes the plan without
    ``pair_rule`` where that program is registered.  The other architectures leave both fields off."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    p, keys = policy_for("sm_107a"), cake_jit.PROGRAM_KEYS["sm_107a"]
    assert p.split_wave_scope == [["narrow"], 1, True] and p.wide_pair_snake_only
    for arch in ("sm_100a", "sm_103a"):
        other = policy_for(arch)
        assert other.split_wave_scope is None and not other.wide_pair_snake_only, arch
    plain = DispatchPolicy.from_record(
        dict(p.record(), split_wave_scope=None, wide_pair_snake_only=False)
    )

    def with_(policy, **override):
        return DispatchPolicy.from_record(dict(policy.record(), **override))

    def plan(policy, T, Tkv, S, sample, top_k=2048, grid=212):
        return select_program(
            policy,
            num_queries=T,
            num_keys=Tkv,
            num_segments=S,
            ratio=1,
            top_k=top_k,
            grid=grid,
            sample_tiles_max=sample,
            scan_keys=keys,
        )

    def expected(*args, **kw):
        base = plan(plain, *args, **kw)
        sampled = base.sample_tiles_max > 0
        S = args[2]
        kept = sampled and S <= 1 and base.split and base.unit == "narrow"
        record = plain if kept else with_(plain, split_wave_rule=None)
        cur = plan(record, *args, **kw)
        if sampled and cur.kind == "pair_wide" and not cur.snake:
            alt = plan(with_(record, pair_rule=None), *args, **kw)
            if alt.scan_key in keys:
                cur = alt
        return base, cur

    # anchors (top_k 2048, 212 SMs): one 1024-query document over 128k keys keeps its two-way narrow split sampled and
    # takes the unpaired l6 unit unsampled
    rule = plan(p, 1024, 131072, 1, None)
    assert (rule.kind, rule.unit, rule.n_split) == ("split", "narrow", 2)
    assert plan(plain, 1024, 131072, 1, 0).n_split == 2
    assert plan(p, 1024, 131072, 1, 0).scan_key == "scan:l6:u1:s0:f0"
    # two such documents: the narrow wave split goes in both modes (the l6 solo preference then applies; the
    # longest-first copy of its program)
    assert plan(plain, 2048, 262144, 2, None).n_split == 2
    for sample in (None, 0):
        rule = plan(p, 2048, 262144, 2, sample)
        assert rule.n_split == 1, sample
        assert rule.scan_key == _lsf_key(keys, "scan:l6:u1:s0:f0"), sample
    # a 4096-query document over 128k keys: the wide wave split goes; sampled the call then runs the unpaired wide
    # program (its CTA pair does not walk the snake order), unsampled the l6 unit
    base = plan(plain, 4096, 131072, 1, None)
    assert (base.kind, base.unit) == ("split", "wide")
    assert plan(p, 4096, 131072, 1, None).scan_key == "scan:wide:u2:s0:f0"
    assert plan(p, 4096, 131072, 1, 0).scan_key == "scan:l6:u1:s0:f0"
    # a 4096-query document over 64k keys: sampled pair_wide without the snake order -> unpaired wide; unsampled unchanged
    assert plan(plain, 4096, 65536, 1, None).scan_key == "scan:pair_wide:u2:s0:f0"
    assert plan(p, 4096, 65536, 1, None).scan_key == "scan:wide:u2:s0:f0"
    assert plan(p, 4096, 65536, 1, 0) == plan(plain, 4096, 65536, 1, 0)
    # without a registry the unpaired wide preference stays off (its program may not be registered: scan:wide:u2:s0:f1 is not)
    bare = select_program(
        p,
        num_queries=4096,
        num_keys=65536,
        num_segments=1,
        ratio=1,
        top_k=2048,
        grid=212,
    )
    assert bare.scan_key == "scan:pair_wide:u2:s0:f0"
    # the snake pair wide program stays (one causal 64k document, sampled)
    assert plan(p, 65536, 65536, 1, None).scan_key == "scan:pair_wide:u1:s1:f1"
    counts = dict(kept=0, dropped=0, unpaired=0, same=0)
    for q, k, S, top_k, sample in itertools.product(
        (4, 64, 256, 1024, 2048, 4096, 8192),
        (16384, 65536, 131072, 262144),
        (1, 2, 4),
        (256, 2048),
        (None, 0),
    ):
        if q > k:
            continue
        args = (q * S, k * S, S, sample)
        base, want = expected(*args, top_k=top_k)
        got = plan(p, *args, top_k=top_k)
        assert got == want, (args, top_k, got.scan_key, want.scan_key)
        assert got.scan_key in keys, (args, top_k, got.scan_key)
        if got == base:
            counts["kept" if base.split else "same"] += 1
        elif base.split and not got.split:
            counts["dropped"] += 1
        elif base.kind == "pair_wide" and got.kind == "wide":
            counts["unpaired"] += 1
    assert counts["kept"] and counts["dropped"] and counts["unpaired"], counts


def test_snake_packed_rule_and_snake_l6_fallback_waves_on_sm_107a():
    """sm_107a's ``snake_packed_rule`` [["l6"], 48.0, None, 7]: an l6 call of two or more segments walks the snake order
    only sampled and from 48 mean unit tiles on, or within 7 waves of the l6 persistent grid (one document keeps
    ``snake_rule`` alone); ``snake_l6_fallback_waves`` 310: a
    sampling-off call the l6 preference moves runs ``scan:l6:u1:s0:f0`` where ``scan:l6:u1:s1:f0`` is not registered and
    the planned program needs at most 310 waves of its persistent grid.  On a grid of shapes in both modes every plan is
    the plan of the record with both fields off but for the unit order of an l6 program and, with sampling off, the
    unpaired plain-order l6 unit of the fallback.  The other architectures and every FP8 record leave both fields off."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    p, keys = policy_for("sm_107a"), cake_jit.PROGRAM_KEYS["sm_107a"]
    assert p.snake_packed_rule == [["l6"], 48.0, None, 7]
    assert p.snake_l6_fallback_waves == 310
    for arch in ("sm_100a", "sm_103a"):
        other = policy_for(arch)
        assert other.snake_packed_rule is None, arch
        assert other.snake_l6_fallback_waves is None, arch
    for arch, fields in cake_jit.POLICY_FP8.items():
        assert fields["snake_packed_rule"] is None, arch
        assert fields["snake_l6_fallback_waves"] is None, arch
    off = DispatchPolicy.from_record(
        dict(p.record(), snake_packed_rule=None, snake_l6_fallback_waves=None)
    )
    assert "scan:l6:u1:s1:f0" not in keys

    def plan(policy, T, Tkv, S, sample, top_k=2048, grid=212):
        return select_program(
            policy,
            num_queries=T,
            num_keys=Tkv,
            num_segments=S,
            ratio=1,
            top_k=top_k,
            grid=grid,
            sample_tiles_max=sample,
            scan_keys=keys,
        )

    # packed equal documents: eight 8k documents (32 mean tiles, 52 waves) leave the snake order in both modes; two 16k
    # documents (64 mean tiles, 26 waves) keep it sampled only; two 4k-query documents over 8k keys each (48 mean tiles)
    # and two 4k documents over 6k keys each (32 mean tiles) need 7 waves of 6-query units on 212 CTAs: both keep it
    # (packed calls without max_seqlen_k: the plain-order l6 programs run their longest-first copies, the snake-order
    # ones have none)
    for sample in (None, 0):
        assert plan(off, 65536, 65536, 8, sample).scan_key == "scan:l6:u1:s1:f1"
        assert plan(p, 65536, 65536, 8, sample).scan_key == _lsf_key(
            keys, "scan:l6:u1:s0:f1"
        )
        assert plan(p, 8192, 16384, 2, sample).scan_key == "scan:l6:u1:s1:f1"
        assert plan(p, 8192, 12288, 2, sample).scan_key == "scan:l6:u1:s1:f1"
    assert plan(p, 32768, 32768, 2, None).scan_key == "scan:l6:u1:s1:f1"
    assert plan(p, 32768, 32768, 2, 0).scan_key == _lsf_key(keys, "scan:l6:u1:s0:f1")
    assert p.snake_for("l6", 8192, 16384, 2, 1, sampled=False, grid=212)
    assert not p.snake_for(
        "l6", 8192, 16384, 2, 1, sampled=False
    )  # no grid: no waves clause
    assert not p.snake_for(
        "l6", 16384, 32768, 2, 1, sampled=False, grid=212
    )  # 13 waves
    # one document keeps the snake order in both modes
    for sample in (None, 0):
        assert plan(p, 8192, 8192, 1, sample).scan_key == "scan:l6:u1:s1:f1"
    # sampling off, one causal document: 16k queries over 32k keys (20 waves of the pair wide plan) and 256k queries
    # (310 waves) take the plain-order l6 program; 512k queries (619 waves) keep the snake pair wide program
    for T, Tkv in ((16384, 32768), (262144, 262144)):
        assert plan(off, T, Tkv, 1, 0).scan_key == "scan:pair_wide:u1:s1:f0", T
        got = plan(p, T, Tkv, 1, 0)
        assert (got.scan_key, got.kind, got.pair, got.snake) == (
            "scan:l6:u1:s0:f0",
            "l6",
            False,
            False,
        ), T
        assert plan(p, T, Tkv, 1, None) == plan(off, T, Tkv, 1, None), T
    assert plan(p, 524288, 524288, 1, 0) == plan(off, 524288, 524288, 1, 0)
    # four 16k-query documents over 32k keys each: sampling off the packed rule already takes the plain-order l6 key
    assert plan(p, 65536, 131072, 4, 0).scan_key == _lsf_key(keys, "scan:l6:u1:s0:f0")
    counts = dict(same=0, order=0, fallback=0)
    for q, k, S, top_k, sample in itertools.product(
        (4, 64, 256, 1024, 2048, 4096, 8192, 16384),
        (8192, 16384, 32768, 65536, 131072, 262144),
        (1, 2, 4, 8),
        (512, 2048),
        (None, 0),
    ):
        if q > k:
            continue
        args = (q * S, k * S, S, sample)
        base = plan(off, *args, top_k=top_k)
        got = plan(p, *args, top_k=top_k)
        assert got.scan_key in keys, (args, top_k, got.scan_key)
        if got == base:
            counts["same"] += 1
        elif base.kind == "l6" and got.kind == "l6":
            # the unit order alone: a packed l6 call below the rule's floor (in the longest-first copy of the
            # plain-order program where the registry has it; the snake-order program has none)
            assert S >= 2 and base.snake and not got.snake, (args, top_k)
            key = _lsf_key(keys, base.scan_key.replace(":s1:", ":s0:"))
            assert got == replace(
                base, snake=False, scan_key=key, lsf=key.endswith(":lsf")
            ), (args, top_k)
            counts["order"] += 1
        else:
            # the plain-order l6 fallback (sampling off) or the packed rule's l6 key now registered for the l6 preference
            assert sample == 0 and base.snake and not got.snake, (args, top_k)
            assert (got.kind, got.pair, got.n_split) == ("l6", False, 1), (args, top_k)
            counts["fallback"] += 1
    assert counts["same"] and counts["order"] and counts["fallback"], counts


def test_ticket_units_bound_covers_every_segment_split_and_static_path_matches_the_grid():
    """``ticket_units_bound`` (shape only) is at least the kernel's ``n_total`` -- the sum over segments of
    ``ceil(l_s / block_q)``, times ``n_split`` on a split plan, ``ceil(blocks / 2)`` per segment on a CTA pair -- for every
    segment-length vector the shape admits (random ones, and the extremal one that reaches it on the unpaired plans);
    ``ticket_static_path`` compares it with the launched grid (half of it on a CTA pair).  Static and drawing plans both
    occur on every architecture."""
    import random

    rng = random.Random(7042)

    def n_total(c, lens):
        bq = c.block_q
        blocks = [-(-int(n) // bq) for n in lens]
        if c.split:
            return sum(blocks) * c.n_split
        if c.pair:
            return sum((b + 1) // 2 for b in blocks)
        return sum(blocks)

    for arch, grids in _FROZEN_GRIDS.items():
        policy = policy_for(arch)
        keys = cake_jit.PROGRAM_KEYS.get(arch) if cake_jit.PROGRAM_KEYS else None
        static = drawing = 0
        for grid, q, k, S, top_k, sample in itertools.product(
            grids,
            (4, 64, 300, 1024, 4096),
            (8192, 22018, 131072),
            (1, 3, 16),
            (256, 2048),
            (None, 0),
        ):
            if q > k:
                continue
            T, Tkv = q * S, k * S
            c = select_program(
                policy,
                num_queries=T,
                num_keys=Tkv,
                num_segments=S,
                ratio=1,
                top_k=top_k,
                grid=grid,
                sample_tiles_max=sample,
                scan_keys=keys,
            )
            bound = cake_policy.ticket_units_bound(c)
            agents = c.grid_ctas // 2 if c.pair else c.grid_ctas
            assert cake_policy.ticket_static_path(c) == (bound <= agents)
            static += bound <= agents
            drawing += bound > agents
            for _ in range(16):
                total = rng.randint(0, T)
                cuts = sorted(rng.randint(0, total) for _ in range(S - 1))
                lens = [b - a for a, b in zip([0, *cuts], [*cuts, total], strict=True)]
                assert n_total(c, lens) <= bound, (arch, grid, T, S, lens)
            extremal = [1] * (S - 1) + [T - (S - 1)]
            assert n_total(c, extremal) <= bound
            if not c.pair:
                assert n_total(c, extremal) == bound, (arch, grid, T, S)
        assert static and drawing, (arch, static, drawing)
    # the split tail rows (4 queries over 22-42k keys) are static on every architecture
    for arch, grids in _FROZEN_GRIDS.items():
        keys = cake_jit.PROGRAM_KEYS.get(arch) if cake_jit.PROGRAM_KEYS else None
        for grid in grids:
            for Tkv in (22018, 30466, 41986):
                c = select_program(
                    policy_for(arch),
                    num_queries=4,
                    num_keys=Tkv,
                    num_segments=1,
                    ratio=1,
                    top_k=2048,
                    grid=grid,
                    scan_keys=keys,
                )
                assert cake_policy.ticket_static_path(c), (arch, grid, Tkv)


def test_sample_fit_kinds_name_exactly_the_registered_sampled_fit_programs():
    """``sample_fit_kinds`` is the registry's own list: every scan kind with a registered ``:f1`` program and no other, so
    the dispatch never composes a sampled-fit key the architecture does not ship (round 3: a 32768-slot capacity made
    split units of 22k-42k keys fit the buffer and selected ``scan:split_l6:u1:s0:f1``, which has no program).  A
    descending-walk twin (key kind ``<kind>_desc``) is not a decided kind: its ascending program carries the form."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        registered = sorted(
            {
                key.split(":")[1]
                for key in keys
                if key.startswith("scan:")
                and key.endswith(":f1")
                and not key.split(":")[1].endswith("_desc")
            }
        )
        assert policy_for(arch).sample_fit_kinds == registered, arch
    # a split unit that fits a 32768-slot buffer keeps the plain split program; the gate, not the geometry, decides
    # (12 splits: split_fit_rule gives each split at most top_k keys)
    p = policy_for("sm_103a")
    geometry = dict(
        num_queries=4, num_keys=22018, num_segments=1, ratio=1, top_k=2048, grid=148
    )
    c = select_program(p, candidate_multiplier=16, **geometry)
    assert (c.scan_key, c.n_split, c.cand_cap) == ("scan:split_l6:u1:s0:f0", 12, 32768)
    assert c.physical_kind == "split_l6" and not c.sample_fit
    assert (
        select_program(p, **geometry).cand_cap == 32768
    )  # the record's own multiplier reaches the same plan
    wider = DispatchPolicy.from_record(
        dict(p.record(), sample_fit_kinds=sorted(p.sample_fit_kinds + ["split_l6"]))
    )
    c = select_program(wider, candidate_multiplier=16, **geometry)
    assert c.scan_key == "scan:split_l6:u1:s0:f1" and c.sample_fit
    assert p.sample_fit_for(4, 22018, 1, 1, 32768) and not p.sample_fit_for(
        4, 22018, 1, 1, 32768, "split_l6"
    )


def test_shipped_capacity_rules_by_architecture():
    """sm_103a: short units (under 384 mean tiles) take 16 x top_k slots up to the 32768-slot ceiling, long units keep
    8 x top_k above its 16384-slot floor, which applies on every top_k; sm_100a keeps 4 x top_k above its floor, which
    is 16384 slots from top_k 1024 on (``cand_cap_floor_rule``) and 8192 below it, so its top_k 256 / 512 calls take
    8192-slot buffers; the unsampled values (``sample_tiles_max=0``) are the 8192-slot rule on both."""
    short = dict(
        num_queries=65536, num_keys=65536, num_segments=1, ratio=1, grid=148
    )  # 256 mean tiles
    long = dict(
        num_queries=16384, num_keys=270336, num_segments=2, ratio=1, grid=148
    )  # 1024 mean tiles
    p103, p100 = policy_for("sm_103a"), policy_for("sm_100a")
    caps = lambda p, g, **kw: [
        select_program(p, top_k=k, **g, **kw).cand_cap
        for k in (256, 512, 1024, 2048, 4096)
    ]  # noqa: E731
    assert caps(p103, short) == [16384, 16384, 16384, 32768, 32768]
    assert caps(p103, long) == [16384, 16384, 16384, 16384, 32768]
    assert caps(p100, short) == [8192, 8192, 16384, 16384, 16384]
    assert caps(p100, long) == [8192, 8192, 16384, 16384, 32768]
    assert (
        caps(p103, short, sample_tiles_max=0)
        == caps(p100, short, sample_tiles_max=0)
        == [8192, 8192, 8192, 8192, 16384]
    )
    assert (
        caps(p103, long, sample_tiles_max=0)
        == caps(p100, long, sample_tiles_max=0)
        == [8192, 8192, 8192, 16384, 32768]
    )
    assert cake_jit.POLICY["sm_100a"]["cand_cap_floor"] == 8192
    assert cake_jit.POLICY["sm_100a"]["cand_cap_floor_rule"] == [1024, 16384]
    assert cake_jit.POLICY["sm_103a"]["cand_cap_floor"] == 16384
    assert cake_jit.POLICY["sm_103a"]["cand_cap_floor_rule"] is None
    assert cake_jit.POLICY["sm_107a"]["cand_cap_floor"] == 8192
    assert cake_jit.POLICY["sm_107a"]["cand_cap_floor_rule"] is None


def test_shipped_floor_rule_keeps_the_pooled_workspace_small():
    """flashinfer-ai/flashinfer#7549 (minor): on sm_100a the ratio-4 pooled selection (top-512 pools over a 4-token pool
    layout) of a packed batch, here the CP16 zigzag rank-0 geometry of the Z3cp row with its keys pooled 4:1, takes
    8192-slot candidate buffers (57 MiB on 152 SMs with six queries per unit, the pre-16384-floor plan: check period 16),
    where the 16384-slot floor on every top_k needed 114 MiB and check period 32; ``candidate_multiplier=32`` restores
    that plan; a top_k 2048 call of the same geometry keeps the previous record's 16384 slots; ``cake_dsa_indexer_det_train_topk_workspace_size``
    agrees.  sm_103a sets no rule: the same call keeps its 16384-slot buffers (114 MiB, check period 32), the plan of the
    previous package there."""
    pooled = dict(
        num_queries=10624, num_keys=43824, num_segments=16, ratio=4, top_k=512
    )
    tail = cake_policy.TICKET_TAIL_BYTES
    p = policy_for("sm_100a")
    new = select_program(p, grid=152, **pooled)
    assert new.kind == "l6" and new.block_q == 6 and new.grid_ctas == 152
    assert (new.cand_cap, new.check_period) == (8192, 16)
    assert new.workspace_bytes == 152 * 6 * 8192 * 8 + tail == 59768832 + tail
    today = select_program(p, grid=152, candidate_multiplier=32, **pooled)
    assert (today.cand_cap, today.check_period) == (16384, 32)
    assert today.workspace_bytes == 152 * 6 * 16384 * 8 + tail == 119537664 + tail
    assert (today.scan_key, today.finalize_key) == (new.scan_key, new.finalize_key)
    previous = DispatchPolicy.from_record(
        dict(p.record(), cand_cap_floor=16384, cand_cap_floor_rule=None)
    )  # the record before the rule: the 16384-slot floor on every top_k
    before = select_program(previous, grid=152, **pooled)
    assert (before.cand_cap, before.check_period, before.workspace_bytes) == (
        today.cand_cap, today.check_period, today.workspace_bytes,
    )  # fmt: skip
    large = dict(
        pooled, top_k=2048
    )  # above the rule's top_k: the previous record's plan
    assert (
        select_program(p, grid=152, **large).cand_cap
        == select_program(previous, grid=152, **large).cand_cap
        == 16384
    )
    assert (
        cake_dsa_indexer_det_train_topk_workspace_size(
            10624, 43824, 16, top_k=512, ratio=4, arch="sm_100a", num_sms=152
        )
        == new.workspace_bytes
    )
    assert (
        cake_dsa_indexer_det_train_topk_workspace_size(
            10624,
            43824,
            16,
            top_k=512,
            ratio=4,
            arch="sm_100a",
            num_sms=152,
            candidate_multiplier=32,
        )
        == today.workspace_bytes
    )
    p103 = policy_for("sm_103a")
    assert p103.cand_cap_floor_rule is None and p103.cand_cap_floor == 16384
    kept = select_program(p103, grid=152, **pooled)
    assert kept.kind == "l6" and kept.block_q == 6 and kept.grid_ctas == 152
    assert (kept.cand_cap, kept.check_period) == (16384, 32)
    assert kept.workspace_bytes == 152 * 6 * 16384 * 8 + tail == 119537664 + tail
    assert (
        cake_dsa_indexer_det_train_topk_workspace_size(
            10624, 43824, 16, top_k=512, ratio=4, arch="sm_103a", num_sms=152
        )
        == kept.workspace_bytes
    )


def test_shipped_unit_rules_by_architecture():
    """sm_103a (``wide_first_rule [512.0]``, ``pair_units ["l6"]``): a wide-eligible call of 512 or more mean tiles takes the
    unrolled wide program before the three-warpgroup rule, in both sample modes; below the floor and on calls short of six
    wide rounds the three-warpgroup pair stays, and the 32-tile document keeps plain l6.  sm_100a and sm_107a carry no floor
    (``None``) and list every registered pair unit, so their plans are the previous package's by code path."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    p103, p100, p107 = (
        policy_for("sm_103a"),
        policy_for("sm_100a"),
        policy_for("sm_107a"),
    )
    for grid in (152, 148):
        for knobs in ({}, {"sample_tiles_max": 0}):
            # 128k causal document: 512.0 mean tiles, 16384 / grid >= 107 rounds -> wide first; 8 x 2048 slots fit -> f1
            assert (
                _select(p103, 131072, 131072, grid=grid, **knobs).scan_key
                == "scan:wide:u2:s0:f1"
            ), (grid, knobs)
            # 262k causal document: 1024.0 tiles -> wide first; above the 512-tile fit bound -> f0
            assert (
                _select(p103, 262144, 262144, grid=grid, **knobs).scan_key
                == "scan:wide:u2:s0:f0"
            ), (grid, knobs)
            # 16k queries over two 135k-key segments: (270336 - 8192) / 2 / 128 = 1024.0 tiles, 13.5 rounds -> wide first
            # at top_k 2048 (16384 slots) and 4096 (the 32768-slot ceiling); two segments and no max_seqlen_k: the
            # longest-first copy (:lsf), and the program without it once max_seqlen_k says the segments are equal
            for K in (2048, 4096):
                assert (
                    _select(p103, 16384, 270336, S=2, K=K, grid=grid, **knobs).scan_key
                    == "scan:wide:u2:s0:f0:lsf"
                ), (grid, knobs, K)
                assert (
                    _select(
                        p103,
                        16384,
                        270336,
                        S=2,
                        K=K,
                        grid=grid,
                        max_seqlen_k=135168,
                        **knobs,
                    ).scan_key
                    == "scan:wide:u2:s0:f0"
                ), (grid, knobs, K)
            # 64k causal document: 256 tiles, below the floor -> the three-warpgroup pair as before
            assert (
                _select(p103, 65536, 65536, grid=grid, **knobs).scan_key
                == "scan:pair_l6:u1:s0:f1"
            ), (grid, knobs)
            # context-parallel tail: 496 tiles but 512 / grid = 3.4 rounds < 6 -> ineligible, the pair stays (no fit:
            # the 480-tile offset exceeds the 127 fitting tiles)
            assert (
                _select(p103, 4096, 65536, grid=grid, **knobs).scan_key
                == "scan:pair_l6:u1:s0:f0"
            ), (grid, knobs)
            # 8k causal document: 32 tiles, 1024 / grid = 6.7 rounds but work 6.7 x 32 = 216 < 300 -> ineligible, below
            # the 64-tile pair floor -> plain l6
            assert (
                _select(p103, 8192, 8192, grid=grid, **knobs).scan_key
                == "scan:l6:u1:s0:f1"
            ), (grid, knobs)
            # sm_100a: the previous package's plans (l6 to 256 tiles, pair 128..256, no floor)
            assert (
                _select(p100, 65536, 65536, grid=grid, **knobs).scan_key
                == "scan:pair_l6:u1:s0:f1"
            ), (grid, knobs)
            assert (
                _select(p100, 16384, 270336, S=2, grid=grid, **knobs).scan_key
                == "scan:wide:u2:s0:f0:lsf"
            ), (grid, knobs)
            assert (
                _select(p100, 4096, 65536, grid=grid, **knobs).scan_key
                == "scan:narrow:u1:s0:f0"
            ), (grid, knobs)
    # the identity guard of the untimed records: no floor, every registered pair unit listed
    assert p100.wide_first_rule is None and p100.pair_units == ["l6"]
    assert p107.wide_first_rule is None
    assert p107.pair_units == ["l6", "narrow", "wide"]
    assert p103.wide_first_rule == [512.0] and p103.pair_units == ["l6"]


def test_shipped_sample_shift_rule_by_architecture():
    """sm_103a (``sample_shift_rule [4096, 375]``): a ``top_k = 4096`` call shifts its sampled rank by 375 permille, a
    ``top_k = 2048`` call keeps the 250 default, in both sample modes, and an explicit ``sample_shift_permille`` knob wins
    on both; sm_100a and sm_107a carry no rule (``None``) and keep 250 at every ``top_k``, except where their class rules
    shift a sampled two-segment call (sm_107a also by its long-unit rule).  The backend's plan carries the same scalar
    (host-only: ``arch`` and the SM count given)."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    p103, p100, p107 = (
        policy_for("sm_103a"),
        policy_for("sm_100a"),
        policy_for("sm_107a"),
    )
    assert p103.sample_shift_rule == [4096, 375]
    assert p100.sample_shift_rule is None and p107.sample_shift_rule is None
    # the 64-tile document, the 512-tile document and the two-segment 1024-tile call: on sm_103a the three-warpgroup
    # pair and both wide fit forms
    rows = ((16384, 16384, 1), (131072, 131072, 1), (16384, 270336, 2))
    for grid in (152, 148):
        for knobs in ({}, {"sample_tiles_max": 0}):
            for T, Tkv, S in rows:
                # sample_class_rule [8, 500] (sm_103a / sm_100a): a sampled call of two or more segments on a wide-unit
                # program shifts 500 permille, ahead of sample_shift_rule; on sm_107a the two-segment call shifts 500
                # permille too (sample_long_rule on a long-unit l6 program, sample_class_rule [32, 500] on a wide one)
                cls = 500 if S >= 2 and not knobs else None
                for K, shift in ((2048, 250), (4096, 375)):
                    where = (grid, knobs, T, Tkv, S, K)
                    c = _select(p103, T, Tkv, S=S, K=K, grid=grid, **knobs)
                    assert c.sample_shift_permille == (cls or shift), where
                    for p in (p100, p107):
                        c = _select(p, T, Tkv, S=S, K=K, grid=grid, **knobs)
                        assert c.sample_shift_permille == (cls or 250), where
                    forced = dict(knobs, sample_shift_permille=-500)
                    c = _select(p103, T, Tkv, S=S, K=K, grid=grid, **forced)
                    assert c.sample_shift_permille == -500, where

    # the backend's plan carries the same scalar
    for arch, K, knobs, shift in (
        ("sm_103a", 4096, {}, 375),
        ("sm_103a", 2048, {}, 250),
        ("sm_100a", 4096, {}, 250),
        ("sm_107a", 4096, {}, 250),
        ("sm_103a", 4096, {"sample_shift_permille": 250}, 250),
    ):
        plan = plan_dsa_indexer_det_topk(
            16384, 16384, 1, top_k=K, arch=arch, num_sms=152, **knobs
        )
        assert plan.sample_shift_permille == shift, (arch, K, knobs)


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
    return validate_dsa_indexer_det_inputs(
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
            "library": "fake",
            "arches": ["sm_100a"],
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
# program where the architecture's rule admits it; the 8k-query / 72k-key row reaches the wide-first precedence
# where the architecture's record sets a floor; the 16k causal document at top_k 4096 reaches the sampled rank
# shift rule where the architecture's record sets one.
_DISPATCH_ROWS = [
    ([64], [262144], 2048),  # 16 narrow units over 2047 tiles: key-range split + merge
    # one causal document of 64 mean tiles: three-warpgroup / pair programs (pair_l6 on sm_103a: at its 64-tile pair
    # floor, below the wide-first floor)
    ([16384], [16384], 2048),
    # context-parallel tail of 496 tiles: narrow, the three-warpgroup pair (sm_103a: 3.4 wide rounds at 148-152 SMs,
    # ineligible for the wide-first precedence) or the CTA-pair wide program by the architecture's rule
    ([4096], [65536], 2048),
    # one causal 8k query block over 72k keys, 544 mean tiles: wide-first on sm_103a, wide on sm_100a, CTA-pair wide on
    # sm_107a
    ([8192], [73728], 2048),
    # the 64-tile document at top_k 4096: the sampled rank shift rule on sm_103a (375 permille from top_k 4096 on;
    # the 32768-slot ceiling), the 250 default elsewhere; same bits for every shift
    ([16384], [16384], 4096),
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
    runner = prepare_dsa_indexer_det_topk(
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
    if plan.split and plan.merge_rank:
        assert runner.stages == ("scan", "merge")
    elif plan.split:
        assert runner.stages == ("scan", "merge", plan.finalize_role)
    else:
        assert runner.stages == ("scan", plan.finalize_role)
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


def test_sm107a_sampling_rules_cover_the_7549_rank_geometries():
    """#7549 on sm_107a: the packed calls of short query segments and few mean unit tiles (the 22-segment packed rank
    inputs) sample ``sample_class_rule[0]`` tiles at its shift -- not a call whose units average no more tiles than that
    (the ratio-4 pooled packed rows keep the tiny defaults) -- and the long-unit l6 calls (the two-segment 262144-key
    document ranks) take ``sample_long_rule``'s shift at the long-unit tile count, for every ``top_k``.  A long-unit call
    of another kind keeps the long-unit defaults, explicit knobs stay, and sampling off stays off.  The other
    architectures set no long-unit rule, and the sm_107a FP8 record sets neither rule."""
    assert policy_for("sm_100a").sample_long_rule is None
    assert policy_for("sm_103a").sample_long_rule is None
    assert policy_for("sm_107a", "fp8").sample_class_rule is None
    assert policy_for("sm_107a", "fp8").sample_long_rule is None
    arch, grid = "sm_107a", _FROZEN_GRIDS["sm_107a"][0]
    policy = policy_for(arch)
    class_rule, long_rule = policy.sample_class_rule, policy.sample_long_rule
    assert class_rule is not None and long_rule is not None

    def plan(T, Tkv, S, top_k=2048, **knobs):
        return plan_dsa_indexer_det_topk(
            T, Tkv, S, top_k=top_k, arch=arch, num_sms=grid, **knobs
        )

    def sampling(choice):
        return choice.sample_tiles_max, choice.sample_shift_permille

    assert sampling(plan(17536, 289344, 22)) == tuple(class_rule)
    # the ratio-4 pooled packed call (top-512 pools, about 21 mean unit tiles) keeps the tiny defaults: the class
    # rule's larger sample would cover its whole units; sm_100a / sm_103a sample the tiny default count under theirs
    pooled = plan(10624, 43824, 16, top_k=512, ratio=4)
    assert policy.mean_unit_tiles(10624, 43824, 16, 4) <= class_rule[0]
    assert sampling(pooled) == (
        policy.sample_tiles_tiny_units,
        policy.sample_shift_for(512),
    )
    for other in ("sm_100a", "sm_103a"):
        assert (
            policy_for(other).sample_class_rule[0]
            == policy_for(other).sample_tiles_tiny_units
        )
    for top_k in (256, 2048, 4096):
        doc = plan(16384, 270336, 2, top_k=top_k)
        assert doc.physical_kind in long_rule[0], top_k
        assert sampling(doc) == (policy.sample_tiles_max, long_rule[1]), top_k
    wide = plan(4096, 131072, 1)
    assert wide.physical_kind not in long_rule[0]
    assert sampling(wide) == (policy.sample_tiles_max, policy.sample_shift_for(2048))
    assert sampling(plan(16384, 270336, 2, sample_tiles_max=16)) == (16, long_rule[1])
    assert sampling(plan(16384, 270336, 2, sample_shift_permille=100)) == (
        policy.sample_tiles_max,
        100,
    )
    assert plan(16384, 270336, 2, sample_tiles_max=0).sample_tiles_max == 0
    assert plan(17536, 289344, 22, sample_tiles_max=0).sample_tiles_max == 0


def test_far_sample_knob_is_validated_on_the_host():
    """``far_sample`` is ``0`` or ``1`` and defaults to the architecture's record; the check reads no device."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for arch in cake_jit.ARCHES:
        grid = _FROZEN_GRIDS[arch][0]
        plan = plan_dsa_indexer_det_topk(
            1024, 4096, 1, top_k=256, arch=arch, num_sms=grid
        )
        assert plan.far_sample in (0, 1)
        assert plan.far_sample == policy_for(arch).far_sample
        for value in (0, 1):
            assert (
                plan_dsa_indexer_det_topk(
                    1024, 4096, 1, top_k=256, arch=arch, num_sms=grid, far_sample=value
                ).far_sample
                == value
            )
        for bad in (2, -1, True):
            with pytest.raises(ValueError, match="far_sample"):
                plan_dsa_indexer_det_topk(
                    1024, 4096, 1, top_k=256, arch=arch, num_sms=grid, far_sample=bad
                )


PARTITION_KNOBS = (
    {"grid_ctas": 2},
    {"grid_ctas": 37},
    {"candidate_multiplier": 2},
    {"check_period": 1},
    {"sample_tiles_max": 0},
    {"sample_shift_permille": -500},
    {"far_sample": 1},
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


# The exact cases (cutoff ties, signed zeros, large tie groups, ...) under the partition and sampling knobs: the
# id rule (a tied score keeps the larger id) and the zero rule (+0.0 == -0.0) select the same keys with the same
# score bits however the keys are split across CTAs, buffered, checked or sampled, and every row stays in
# positional order (ascending ids, padding only at the tail).
EXACT_CASE_KNOBS = (
    {"grid_ctas": 2},
    {"grid_ctas": 37},
    {"candidate_multiplier": 2},
    {"check_period": 1},
    {"sample_tiles_max": 0},
    {"sample_shift_permille": -500},
    {"far_sample": 1},
)


def _assert_positional_order(ids: torch.Tensor) -> None:
    valid = ids >= 0
    flags = valid.to(torch.int32)
    assert torch.equal(flags, flags.cummin(dim=1).values), "padding before a valid id"
    assert bool(((ids[:, 1:] > ids[:, :-1]) | ~valid[:, 1:]).all()), "ids not ascending"


@pytest.mark.parametrize("name", list(ref.EXACT_CASES))
def test_exact_cases_are_knob_invariant(name):
    _require_program()
    inputs = ref.make_exact_case(name, "cuda")
    base = _run(inputs)
    _judge_ok(base, ref.select_reference(inputs, order="sequential"), exact_bits=True)
    _assert_positional_order(base[0])
    for knobs in EXACT_CASE_KNOBS:
        other = _run(inputs, **knobs)
        assert ref.same_bits(base, other), f"{name}: knob {knobs} changed the result"


# -- the attention-sink + local-band distribution (flashinfer-ai/flashinfer#7042) ----------------------------------
# Small context-parallel geometries whose long units overflow the candidate buffer and run the sampled first
# threshold (the issue's regime: the nearest key tile of every unit -- the diagonal -- holds a hot band of top-k
# keys), judged like the random rows against the FP64 / FP32 reference under both settings of the far-sample
# switch and compared bitwise with the sampled threshold switched off and with the forced re-stream.  Host plan,
# hand-evaluated from the registry policies (the test asserts the host-visible parts): narrow units (l6 on sm_107a for ``cp16_16k``), ``cand_cap`` 8192,
# ``check_period`` 16; ``cp16_16k`` (mean unit tiles 64) samples 8 tiles at stride 16 on its 125-128-tile units,
# ``tail_30848`` (mean 239) samples 16 tiles at stride 15 / 16 on its 238-241-tile units.  The 241-tile units of
# ``tail_30848`` satisfy ``(n_tiles - 1) % stride == 0``, so both tile orders sample the same tiles, the diagonal
# and the sink tile included.  The 16384-token document of ``cp16_16k`` is the issue's 262144-token zigzag at 1/16
# (``cp16_16k`` = ``cp_zigzag_rank0(16384)``: queries ``[0, 512)`` and ``[15872, 16384)``).
SINK_GEOMETRIES = {
    "cp16_16k": dict(seg_q_len=[512, 512], seg_k_len=[512, 16384], top_k=256),
    "tail_30848": dict(seg_q_len=[512], seg_k_len=[30848], top_k=256),
}


def _longest_unit_samples(inputs, plan) -> bool:
    """The kernel's ``sample_plan`` conditions for the longest unit, from host-visible numbers only: its key tiles
    overflow the candidate buffer and the plan admits at least two sample tiles."""
    tile = policy_for(arch_for(inputs.device)).tile_keys
    tiles = -(-max(inputs.seg_k_len) // tile)
    max_tiles = min(plan.sample_tiles_max, plan.cand_cap // tile - plan.check_period)
    return tiles * tile + tile > plan.cand_cap and max_tiles >= 2


@pytest.mark.parametrize("far_sample", [0, 1])
@pytest.mark.parametrize("name", list(SINK_GEOMETRIES))
def test_sink_distribution_is_exact_and_knob_invariant(name, far_sample):
    _require_program()
    inputs = ref.make_sink_inputs(
        **SINK_GEOMETRIES[name], seed=ref.SINK_SEED, device="cuda", label=name
    )
    plan = _plan(inputs, far_sample=far_sample)
    assert plan.sample_tiles_max > 0 and _longest_unit_samples(inputs, plan), plan
    assert plan.far_sample == far_sample
    base = _run(inputs, far_sample=far_sample)
    verdict = _judge_ok(base, ref.select_reference(inputs))
    assert (
        verdict.max_fp32_error_over_bound <= 1.0
        and verdict.max_fp64_error_over_bound <= 1.0
    )
    # the sink: most rows select key 0 (the issue reports 94-100 %; an iid row selects it with probability
    # ~ top_k / visible)
    assert float((base[0] == 0).any(dim=1).float().mean()) > 0.5
    nosample = _run(inputs, sample_tiles_max=0)
    assert ref.same_bits(base, nosample), (
        "sample_tiles_max=0 changed the result on the sink distribution"
    )
    restream = _run(inputs, sample_shift_permille=-500)
    assert ref.same_bits(base, restream), (
        "the forced re-stream changed the result on the sink distribution"
    )


def test_sink_issue_geometry_full():
    """The issue's call: one 262144-token document, CP16 zigzag rank 0, top-k 2048, ``max_seqlen_k`` 262144.
    ``FLASHINFER_CAKE_DSA_INDEXER_FULL=1`` only: 2.1 G visible pairs through the matmul-order reference (minutes)."""
    _require_program()
    if not FULL:
        pytest.skip("set FLASHINFER_CAKE_DSA_INDEXER_FULL=1 for the 262k sink geometry")
    inputs = ref.make_sink_inputs(
        [8192, 8192],
        [8192, 262144],
        top_k=2048,
        seed=ref.SINK_SEED,
        device="cuda",
        label="Z1",
    )
    plan = _plan(inputs, max_seqlen_k=262144)
    # unsplit units, 32 sample tiles, 16384-entry candidate buffers checked every 32 tiles: the plan the issue
    # prints (the unit kind is the architecture's: wide on sm_100a, the l6 CTA pair on sm_103a)
    assert plan.n_split == 1 and _longest_unit_samples(inputs, plan), plan
    assert plan.sample_tiles_max == 32 and plan.cand_cap == 16384, plan
    assert plan.check_period == 32, plan
    base = _run(inputs, max_seqlen_k=262144)
    _judge_ok(base, ref.select_reference(inputs))
    nosample = _run(inputs, max_seqlen_k=262144, sample_tiles_max=0)
    assert ref.same_bits(base, nosample)
    far = _run(inputs, max_seqlen_k=262144, far_sample=1)
    assert ref.same_bits(base, far)
    unbounded = _run(inputs)
    assert ref.same_bits(base, unbounded)


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
    runner = prepare_dsa_indexer_det_topk(
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
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(runner)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner()
    graph.replay()
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(runner)
    assert ref.same_bits(captured, eager)
    fresh = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=51, device="cuda"
    )
    inputs.q.copy_(fresh.q)
    inputs.k.copy_(fresh.k)
    inputs.w.copy_(fresh.w)
    graph.replay()
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(runner)
    _judge_ok((captured[0].clone(), captured[1].clone()), ref.select_reference(inputs))


# the first 16 int64 of the ticket tail: the ticket word and its padding; the rest of the tail holds the
# longest-first segment order, which every :lsf scan launch writes at its prologue before reading it (also the :lsf
# copy of a program that draws from its TicketWord, whose runner has no tail of its own)
_TICKET_WORD_BYTES = 8 * 16


def _assert_ticket_tail_zero(runner) -> None:
    """The ticket word reads 0 after every completed run (every completed scan launch leaves it 0; a
    backend-owned tail is zeroed once at prepare, a caller-owned one by the scan binding before every scan launch); None
    for a program without a tail (``static_scan_kinds``).  The segment order behind it is not checked: an :lsf launch
    leaves it written.  A static call (``runner._tail_static``, ``cake_policy.ticket_static_path``) never touches the word,
    which then holds whatever the buffer held: nothing to check.  A runner whose scan program reads its ``TicketWord``
    argument has no tail (its :lsf copy writes the segment order behind the workspace's ticket word before it reads
    it): its ticket block's word reads 0 instead."""
    if runner.ticket_tail is not None and not runner._tail_static:
        words = runner.ticket_tail[:_TICKET_WORD_BYTES]
        assert not words.any(), words[:8].tolist()
    if runner.ticket_block is not None:
        word = runner.ticket_block[: cake_backend.TICKET_WORD_ZERO_BYTES]
        assert not word.any(), word.tolist()


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


def _cloned(out) -> tuple[torch.Tensor, torch.Tensor]:
    return out[0].clone(), out[1].clone()


def test_prepared_runner_ticket_tail_is_zero_at_prepare_after_runs_and_after_a_failed_launch():
    """Backend-owned workspace: the tail is zeroed once at prepare, no memset in run(); a run that raised
    while launching marks it dirty and the next run restores it."""
    _require_program()
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=52, device="cuda"
    )
    runner = _prepare(inputs)
    if runner.ticket_tail is None:
        pytest.skip(
            "no ticket tail for this program (a static_scan_kinds program, or one drawing from its TicketWord block)"
        )
    assert not runner._tail_shared  # backend-owned: nothing else writes the tail
    assert not runner._tail_static  # the call draws tickets
    # the scan binding zeroes nothing: the device's one empty region, no per-call view
    assert _scan_zero_region(runner) is cake_backend._empty_region(runner.device_index)
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(runner)  # zeroed once, at prepare
    expected = runner()
    expected = (expected[0].clone(), expected[1].clone())
    torch.cuda.synchronize()
    # left at 0 by the scan, no memset in run() on this path
    _assert_ticket_tail_zero(runner)
    # a launch sequence that raises marks the tail dirty; the next run restores the word before it launches
    launches = runner.launches

    def failing() -> None:
        raise RuntimeError("boom")

    runner.launches = (failing,)
    runner.ticket_tail.fill_(7)  # what an aborted launch could leave behind
    with pytest.raises(RuntimeError, match="boom"):
        runner()
    assert runner._tail_dirty
    runner.launches = launches
    out = runner()
    torch.cuda.synchronize()
    assert not runner._tail_dirty
    _assert_ticket_tail_zero(runner)
    assert ref.same_bits(out, expected)


def _workspace_bytes(inputs: ref.IndexerInputs, device: torch.device) -> int:
    return cake_dsa_indexer_det_train_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=device,
    )


def _scan_zero_region(runner) -> torch.Tensor:
    """The ``zero_region`` argument the runner's scan launch passes its binding."""
    (scan,) = [launch for launch in runner.launches if launch.role == "scan"]
    plan = cake_backend.ARG_PLANS["scan"]
    (index,) = [i for i, (kind, _name) in enumerate(plan) if kind == "zero"]
    return scan.arguments[index]


def _caller_owned_runner(inputs: ref.IndexerInputs, workspace: torch.Tensor):
    """A runner on the caller-owned ``workspace`` (skips where its program has no ticket tail)."""
    runner = _prepare(inputs, workspace_buffer=workspace)
    if runner.ticket_tail is None:
        pytest.skip(
            "no ticket tail for this program (a static_scan_kinds program, or one drawing from its TicketWord block)"
        )
    assert runner._tail_shared
    # the scan binding zeroes the tail right before every scan launch; nothing on a static call
    zero_region = _scan_zero_region(runner)
    if runner._tail_static:
        assert zero_region is cake_backend._empty_region(runner.device_index)
    else:
        assert (zero_region.data_ptr(), zero_region.numel()) == (
            runner.ticket_tail.data_ptr(),
            cake_policy.TICKET_TAIL_BYTES,
        )
    return runner


def test_caller_owned_workspace_shared_by_two_geometries_keeps_every_run_exact():
    """Two prepared runners of different geometries share one caller-owned buffer; the larger geometry's
    candidate region covers the smaller one's ticket tail, so every run of B dirties A's tail.  Alternating
    A, B, A, B: every run's bits equal the same geometry's exclusive-workspace run and the tail of the
    runner that ran reads 0 afterwards."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs_a = ref.make_random_inputs(
        [300, 200], [300, 6000], top_k=256, seed=70, device="cuda"
    )
    # top_k 4096: where the candidate capacity has a floor (sm_107a: 8192 slots), top_k 256 and 2048 both sit on it, and
    # with the same unit the two bounds would be equal
    inputs_b = ref.make_random_inputs(
        [900, 300], [900, 7000], top_k=4096, seed=71, device="cuda"
    )
    bytes_a = _workspace_bytes(inputs_a, device)
    bytes_b = _workspace_bytes(inputs_b, device)
    # B's candidate region covers A's tail
    assert bytes_b > bytes_a + cake_policy.TICKET_TAIL_BYTES, (bytes_a, bytes_b)
    expected_a = _cloned(_run(inputs_a))  # exclusive, backend-owned workspaces
    expected_b = _cloned(_run(inputs_b))
    shared = torch.empty(bytes_b, dtype=torch.uint8, device=device)
    runner_a = _caller_owned_runner(inputs_a, shared)
    runner_b = _caller_owned_runner(inputs_b, shared)
    assert runner_a.workspace_bytes == bytes_a and runner_b.workspace_bytes == bytes_b
    tail_a = runner_a.ticket_tail
    assert (
        tail_a.data_ptr() >= shared.data_ptr()
        and tail_a.data_ptr() + cake_policy.TICKET_TAIL_BYTES
        <= shared.data_ptr() + bytes_b - cake_policy.TICKET_TAIL_BYTES
    )  # A's tail lies inside B's candidate region
    for step, (runner, expected) in enumerate(
        [(runner_a, expected_a), (runner_b, expected_b)] * 2
    ):
        out = runner()
        torch.cuda.synchronize()
        assert ref.same_bits(out, expected), step
        _assert_ticket_tail_zero(runner)


def test_caller_owned_workspace_tail_written_between_runs_is_restored_before_the_next_run():
    """The direct hazard: on a caller-owned workspace another kernel writes the ticket word between two
    runs of a cached runner; the next run's scan binding zeroes it first and the run stays bit-exact, and the tail
    reads 0 after."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=72, device="cuda"
    )
    workspace = torch.empty(
        _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
    )
    # whatever the buffer held: prepare zeroes nothing, the scan binding does
    workspace.fill_(0xFF)
    runner = _caller_owned_runner(inputs, workspace)
    assert not runner._tail_static  # the call draws tickets
    expected = _cloned(runner())
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(runner)
    _judge_ok(expected, ref.select_reference(inputs))
    for fill in (7, 0xFF, 1):
        # what another kernel could leave in a shared buffer; not a failed launch,
        # so the scan binding's zero_region alone restores it
        runner.ticket_tail.fill_(fill)
        assert not runner._tail_dirty
        out = runner()
        torch.cuda.synchronize()
        assert ref.same_bits(out, expected), fill
        _assert_ticket_tail_zero(runner)


def test_eager_entry_zeroes_a_caller_owned_tail_once_per_call(monkeypatch):
    """The eager entry with a caller-owned workspace: the scan binding zeroes the ticket tail right before the scan
    launch and no host-side memset runs (neither in ``prepare`` nor in the run); a tail another kernel left non-zero
    before the call is restored, so every call stays bit-exact and leaves the word 0.  A prepared runner on the same
    buffer restores it on every run the same way."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=75, device="cuda"
    )
    workspace = torch.empty(
        _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
    )
    expected = _cloned(_run(inputs))
    runners, host_memsets = [], []
    prepare = cake_backend._prepare_runner

    def counting_prepare(*args, **kwargs):
        assert kwargs.get("one_shot") is True  # the eager entry's runner runs once
        runner = prepare(*args, **kwargs)
        if runner.ticket_tail is not None:
            tail = runner.ticket_tail
            zero = tail.zero_
            tail.zero_ = lambda: (host_memsets.append(1), zero())[1]
        runners.append(runner)
        return runner

    monkeypatch.setattr(cake_backend, "_prepare_runner", counting_prepare)
    for fill in (7, 0xFF):
        workspace.fill_(
            fill
        )  # what another kernel could leave in the buffer, the tail included
        out = cake_backend.cake_dsa_indexer_det_train_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            workspace_buffer=workspace,
        )
        torch.cuda.synchronize()
        assert ref.same_bits(out, expected), fill
        _assert_ticket_tail_zero(runners[-1])
    if runners[-1].ticket_block is not None:
        # a program that reads its TicketWord: the one launch draws from the tail's first bytes, its scan binding
        # zeroes the 8-byte word (not the whole tail), and no host-side memset runs
        runner = runners[-1]
        block_at = runner.workspace_bytes - cake_policy.TICKET_TAIL_BYTES
        assert runner.ticket_block.data_ptr() == workspace.data_ptr() + block_at
        zero_region = _scan_zero_region(runner)
        assert (zero_region.data_ptr(), zero_region.numel()) == (
            runner.ticket_block.data_ptr(),
            cake_backend.TICKET_WORD_ZERO_BYTES,
        )
        assert not host_memsets and runner._ticket_restore is None
        return
    if runners[-1].ticket_tail is None:
        pytest.skip(
            "no ticket tail for this program (a static_scan_kinds program, or one drawing from its TicketWord block)"
        )
    assert runners[-1]._tail_shared and not host_memsets  # the scan binding zeroes it
    assert not runners[-1]._tail_static  # the call draws tickets
    runner = runners[-1]
    runner.ticket_tail.fill_(7)
    out = runner()  # the prepared runner's scan binding restores a shared tail
    torch.cuda.synchronize()
    assert ref.same_bits(out, expected) and not host_memsets
    _assert_ticket_tail_zero(runner)


def test_caller_owned_workspace_graph_replay_restores_the_tail():
    """A graph capturing a caller-owned runner carries the scan binding's tail memset node: corrupting the tail
    between replays leaves every replay bit-exact and the tail 0 afterwards."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=73, device="cuda"
    )
    workspace = torch.empty(
        _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
    )
    runner = _caller_owned_runner(inputs, workspace)
    # the call draws tickets: the graph carries the memset node
    assert not runner._tail_static
    eager = _cloned(_run(inputs))
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
    for fill in (7, 0xFF):
        runner.ticket_tail.fill_(fill)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        _assert_ticket_tail_zero(runner)
        assert ref.same_bits(captured, eager), fill
    fresh = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=74, device="cuda"
    )
    inputs.q.copy_(fresh.q)
    inputs.k.copy_(fresh.k)
    inputs.w.copy_(fresh.w)
    runner.ticket_tail.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(runner)
    _judge_ok(_cloned(captured), ref.select_reference(inputs))


def test_static_scan_kind_runner_has_no_ticket_tail_and_stays_exact_on_a_poisoned_workspace():
    """A program of the record's ``static_scan_kinds`` keeps the static enumeration on every call, also where the
    call's units exceed the grid: its runner gets no ticket tail (no memset at prepare or run) and its rows stay
    bit-exact on a caller-owned buffer poisoned everywhere.  A drawing program on the same architecture keeps its tail.
    No BF16 record names a static kind now, so this skips; the FP8 form's static programs (sm_107a) are covered by
    ``test_fp8_ticket_decision_is_per_form_and_exact_on_a_poisoned_workspace``."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    arch = arch_for(device)
    static_kinds = policy_for(arch).static_scan_kinds
    if not static_kinds:
        pytest.skip("every scan program of this architecture draws tickets")
    found = {}
    for (T, Tkv, S), sample in itertools.product(
        (
            (8192, 32768, 1),
            (8192, 131072, 4),
            (4096, 16384, 2),
            (2048, 65536, 1),
            (32768, 32768, 1),
            (16384, 131072, 1),
            (65536, 65536, 8),
        ),
        (None, 0),
    ):
        choice = cake_backend.plan_dsa_indexer_det_topk(
            T, Tkv, S, top_k=2048, sample_tiles_max=sample
        )
        if cake_policy.ticket_static_path(choice):
            continue  # one unit per CTA at most: no draw either way
        static = choice.physical_kind in static_kinds
        found.setdefault(static, (T, Tkv, S, sample))
    assert True in found, (
        "no geometry runs a static scan kind with more units than CTAs"
    )
    assert False in found, (
        "no geometry runs a drawing scan program with more units than CTAs"
    )
    for static, (T, Tkv, S, sample) in sorted(found.items()):
        inputs = ref.make_random_inputs(
            [T // S] * S, [Tkv // S] * S, top_k=2048, seed=78, device="cuda"
        )
        extra = {} if sample is None else {"sample_tiles_max": sample}
        expected = _cloned(_run(inputs, **extra))
        shared = torch.empty(
            _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
        )
        shared.fill_(7)
        runner = _prepare(inputs, workspace_buffer=shared, **extra)
        assert (runner.choice.physical_kind in static_kinds) == static
        # a drawing program keeps a ticket word of its own: the workspace's tail, or its ticket block where it reads
        # its TicketWord argument
        assert (runner.ticket_tail is None and runner.ticket_block is None) == static, (
            runner.choice.scan_key
        )
        for _ in range(2):
            out = runner()
            torch.cuda.synchronize()
            assert ref.same_bits(out, expected), runner.choice.scan_key
        _assert_ticket_tail_zero(runner)


def test_static_call_skips_the_tail_memset_and_never_touches_the_ticket_word():
    """A call whose shape bounds its work units by the grid (``ticket_static_path``; 4 queries over 22018 keys) takes the
    scan's static fast path, which never reads or writes the ticket word.  On a caller-owned workspace neither prepare,
    a run nor the eager entry zeroes its tail: a poisoned word stays as it was and every run is bit-exact.  A drawing
    runner on the same buffer still zeroes its own tail before its run, and a failed launch still marks the static
    runner's tail dirty, so its next run restores it."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs_s = ref.make_random_inputs([4], [22018], top_k=2048, seed=76, device="cuda")
    inputs_d = ref.make_random_inputs(
        [2048, 2048], [2048, 16384], top_k=1024, seed=77, device="cuda"
    )
    expected_s = _cloned(_run(inputs_s))
    expected_d = _cloned(_run(inputs_d))
    shared = torch.empty(
        max(_workspace_bytes(inputs_s, device), _workspace_bytes(inputs_d, device)),
        dtype=torch.uint8,
        device=device,
    )
    shared.fill_(7)  # what another kernel could leave in the buffer, the tails included
    runner = _caller_owned_runner(inputs_s, shared)
    assert runner._tail_static and cake_policy.ticket_static_path(runner.choice)
    torch.cuda.synchronize()
    word = runner.ticket_tail[:_TICKET_WORD_BYTES]
    assert (word == 7).all()  # prepare issued no memset
    memsets = []
    zero = runner.ticket_tail.zero_
    runner.ticket_tail.zero_ = lambda: (memsets.append(1), zero())[1]
    for fill in (7, 0xFF, 1):
        runner.ticket_tail.fill_(fill)
        out = runner()
        torch.cuda.synchronize()
        assert ref.same_bits(out, expected_s), fill
        assert (word == fill).all(), fill  # untouched by the scan
    assert not memsets
    out = cake_backend.cake_dsa_indexer_det_train_topk(
        inputs_s.q,
        inputs_s.k,
        inputs_s.w,
        inputs_s.cu_seqlens_q,
        inputs_s.cu_seqlens_k,
        **inputs_s.kwargs(),
        workspace_buffer=shared,
    )
    torch.cuda.synchronize()
    # the eager entry: no memset either
    assert ref.same_bits(out, expected_s) and (word == 1).all()
    drawing = _caller_owned_runner(inputs_d, shared)
    assert not drawing._tail_static
    drawing.ticket_tail.fill_(7)
    out = drawing()
    torch.cuda.synchronize()
    assert ref.same_bits(out, expected_d)
    _assert_ticket_tail_zero(drawing)  # zeroed by its scan binding, left 0 by its scan
    out = runner()  # the static runner after the drawing one wrote the buffer
    torch.cuda.synchronize()
    assert ref.same_bits(out, expected_s) and not memsets
    launches = runner.launches

    def failing() -> None:
        raise RuntimeError("boom")

    runner.launches = (failing,)
    with pytest.raises(RuntimeError, match="boom"):
        runner()
    assert runner._tail_dirty
    runner.launches = launches
    out = runner()
    torch.cuda.synchronize()
    assert memsets == [1] and not runner._tail_dirty and not word.any()
    assert ref.same_bits(out, expected_s)


# ---------------------------------------------------------------------------
# TicketWord: a scan program that reads its ticket block (registry ``ticket_arg_arches``)
# ---------------------------------------------------------------------------

_TICKET_GEOMETRIES = (
    ([600, 400], [600, 12000], 1024),
    ([2048, 2048], [2048, 16384], 1024),
    ([8192], [32768], 2048),
)


def _scan_ticket_word(runner):
    """The ``TicketWord`` argument the runner's scan launch passes its binding."""
    (scan,) = [launch for launch in runner.launches if launch.role == "scan"]
    plan = cake_backend.ARG_PLANS["scan"]
    (index,) = [i for i, (_kind, name) in enumerate(plan) if name == "TicketWord"]
    return scan.arguments[index]


def _ticket_word_inputs(seed: int):
    """Inputs whose prepared runner draws from its own ticket block on this device (skips where no scan program of the
    architecture reads its TicketWord, or none of the geometries dispatches one that draws)."""
    _require_program()
    arch = arch_for(torch.device("cuda"))
    if not any(
        arch in r.get("ticket_arg_arches", ()) for r in cake_jit.PROGRAMS.values()
    ):
        pytest.skip(
            f"no scan program reads its TicketWord on {arch}: they draw from the workspace's tail"
        )
    for seg_q, seg_k, top_k in _TICKET_GEOMETRIES:
        inputs = ref.make_random_inputs(
            seg_q, seg_k, top_k=top_k, seed=seed, device="cuda"
        )
        runner = _prepare(inputs)
        if runner.ticket_block is not None:
            return inputs
    pytest.skip("no geometry dispatches a TicketWord program that draws")


def test_ticket_word_runner_owns_its_block_and_ignores_a_poisoned_caller_workspace():
    """A prepared runner whose scan program reads its TicketWord gets a 16-byte block of its own, zeroed once at
    prepare, never the shared placeholder and never the workspace's tail: poisoning the whole caller-owned workspace
    (the tail's ticket word included, and the segment order an :lsf copy writes there before it reads it) between runs
    leaves every run bit-equal to a backend-owned runner, no run issues a memset, and the block's word reads 0 after
    every run."""
    inputs = _ticket_word_inputs(80)
    device = torch.device("cuda", torch.cuda.current_device())
    own = _prepare(inputs)
    expected = _cloned(own())
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(own)
    _judge_ok(expected, ref.select_reference(inputs))
    workspace = torch.empty(
        _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
    )
    workspace.fill_(0xFF)
    runner = _prepare(inputs, workspace_buffer=workspace)
    assert runner.ticket_tail is None and runner.ticket_block is not None
    assert cake_backend.reads_ticket_word(runner.programs["scan"], runner.arch)
    word = _scan_ticket_word(runner)
    assert word is not cake_backend._ticket_placeholder(runner.device_index)
    assert (
        word.dtype == torch.uint32 and word.data_ptr() == runner.ticket_block.data_ptr()
    )
    assert runner.ticket_block.numel() == cake_backend.TICKET_BLOCK_BYTES
    assert not (
        workspace.data_ptr()
        <= runner.ticket_block.data_ptr()
        < workspace.data_ptr() + workspace.numel()
    )
    # the scan binding zeroes nothing on a run: the device's empty region
    assert _scan_zero_region(runner) is cake_backend._empty_region(runner.device_index)
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(runner)  # zeroed once, at prepare
    generator = torch.Generator(device=device).manual_seed(81)
    tail_at = runner.workspace_bytes - cake_policy.TICKET_TAIL_BYTES
    for step in range(6):
        out = runner()
        torch.cuda.synchronize()
        assert ref.same_bits(out, expected), step
        _assert_ticket_tail_zero(runner)
        # what another kernel could leave in the shared buffer: random bytes, a random tail word
        workspace.copy_(
            torch.randint(
                0,
                256,
                workspace.shape,
                generator=generator,
                device=device,
                dtype=torch.uint8,
            )
        )
        workspace[tail_at : tail_at + 4] = torch.tensor(
            [1 + step, 2, 3, 4], dtype=torch.uint8, device=device
        )


# the packed calls of the sm_107a longest-first copies (``_SM107_LSF_COPIES``): unequal key segments on each copy, and
# near-equal ones (the longest at most 5/4 of the shortest), on which the copy keeps the argument order
_SM107_LSF_GEOMETRIES = (
    ([1500, 1000], [1500, 12000], 1024, "scan:l6:u1:s0:f1:lsf"),
    ([1024, 1024], [8192, 24576], 1024, "scan:l6:u1:s0:f0:lsf"),
    ([1024, 1024], [16384, 18432], 1024, None),
)


@pytest.mark.parametrize("geometry", range(len(_SM107_LSF_GEOMETRIES)))
def test_sm107_longest_first_copy_runs_on_a_poisoned_caller_workspace(geometry):
    """On sm_107a a packed call of the two l6 ticket programs runs their longest-first copy, which draws from its
    TicketWord block and writes the segment order into the caller-owned workspace's ticket tail before any role reads
    it: with the whole workspace poisoned before the first run and refilled with random bytes between runs, every run
    is bit-equal to a backend-owned runner's, which passes the reference judge."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    arch = arch_for(device)
    if arch != "sm_107a":
        pytest.skip(
            f"the longest-first copies of the TicketWord programs ship on sm_107a, not {arch}"
        )
    assert set(cake_jit.PROGRAM_KEYS[arch]) >= _SM107_LSF_COPIES
    seg_q, seg_k, top_k, key = _SM107_LSF_GEOMETRIES[geometry]
    inputs = ref.make_random_inputs(
        seg_q, seg_k, top_k=top_k, seed=88 + geometry, device="cuda"
    )
    own = _prepare(inputs)
    assert own.choice.scan_key in (_SM107_LSF_COPIES if key is None else {key}), (
        own.choice.scan_key
    )
    expected = _cloned(own())
    torch.cuda.synchronize()
    _judge_ok(expected, ref.select_reference(inputs))
    workspace = torch.empty(
        _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
    )
    workspace.fill_(0xFF)
    runner = _prepare(inputs, workspace_buffer=workspace)
    assert (
        runner.choice.scan_key == own.choice.scan_key
        and runner.ticket_block is not None
    )
    assert cake_backend.reads_ticket_word(runner.programs["scan"], runner.arch)
    generator = torch.Generator(device=device).manual_seed(89)
    for step in range(4):
        out = runner()
        torch.cuda.synchronize()
        assert ref.same_bits(out, expected), step
        workspace.copy_(
            torch.randint(
                0,
                256,
                workspace.shape,
                generator=generator,
                device=device,
                dtype=torch.uint8,
            )
        )


def test_ticket_word_runner_restores_its_word_after_a_failed_launch():
    """A run that raised while launching may leave the block's word non-zero: the next run launches the scan through
    the binding call whose ``zero_region`` is the block's 8-byte word, on the run's stream, once; later runs pass the
    empty region again."""
    inputs = _ticket_word_inputs(82)
    runner = _prepare(inputs)
    expected = _cloned(runner())
    torch.cuda.synchronize()
    restore = runner._ticket_restore
    assert (
        restore is not None
        and restore.role == "scan"
        and runner.launches[0].role == "scan"
    )
    plan = cake_backend.ARG_PLANS["scan"]
    (zero_at,) = [i for i, (kind, _name) in enumerate(plan) if kind == "zero"]
    region = restore.arguments[zero_at]
    assert (region.data_ptr(), region.numel()) == (
        runner.ticket_block.data_ptr(),
        cake_backend.TICKET_WORD_ZERO_BYTES,
    )
    launches = runner.launches

    def failing() -> None:
        raise RuntimeError("boom")

    runner.launches = (failing, *launches[1:])
    with pytest.raises(RuntimeError, match="boom"):
        runner()
    assert runner._tail_dirty
    runner.launches = launches
    runner.ticket_block[:4].fill_(7)  # what an aborted launch could leave behind
    used = []
    restore_entry = restore.entry
    object.__setattr__(
        restore, "entry", lambda *args: (used.append(1), restore_entry(*args))[1]
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = runner()  # the restoring launch zeroes the word on this stream
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    assert used == [1] and not runner._tail_dirty
    assert ref.same_bits(out, expected)
    _assert_ticket_tail_zero(runner)
    out = runner()
    torch.cuda.synchronize()
    assert used == [1] and ref.same_bits(out, expected)


def test_ticket_word_runners_sharing_a_workspace_keep_their_own_blocks():
    """Two TicketWord runners of different geometries on one caller-owned buffer, run in turn: each draws from its own
    block, so every run is bit-equal to the same geometry's backend-owned run."""
    inputs_a = _ticket_word_inputs(83)
    device = torch.device("cuda", torch.cuda.current_device())
    inputs_b = ref.make_random_inputs(
        [900, 300], [900, 7000], top_k=4096, seed=84, device="cuda"
    )
    expected_a = _cloned(_prepare(inputs_a)())
    expected_b = _cloned(_prepare(inputs_b)())
    shared = torch.empty(
        max(_workspace_bytes(inputs_a, device), _workspace_bytes(inputs_b, device)),
        dtype=torch.uint8,
        device=device,
    )
    shared.fill_(7)
    runner_a = _prepare(inputs_a, workspace_buffer=shared)
    runner_b = _prepare(inputs_b, workspace_buffer=shared)
    assert runner_a.ticket_block is not None
    if runner_b.ticket_block is not None:
        assert runner_a.ticket_block.data_ptr() != runner_b.ticket_block.data_ptr()
    for step, (runner, expected) in enumerate(
        [(runner_a, expected_a), (runner_b, expected_b)] * 2
    ):
        out = runner()
        torch.cuda.synchronize()
        assert ref.same_bits(out, expected), step
        _assert_ticket_tail_zero(runner)


def test_ticket_word_runner_prepared_on_one_stream_runs_on_another():
    """The block is zeroed at prepare on the preparing stream; after the caller synchronizes, a run on another stream
    draws from a zero word."""
    inputs = _ticket_word_inputs(85)
    expected = _cloned(_prepare(inputs)())
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    stream_a.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream_a):
        runner = _prepare(inputs)
    torch.cuda.synchronize()
    stream_b.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream_b):
        out = runner()
        again = _cloned(runner())
    torch.cuda.current_stream().wait_stream(stream_b)
    torch.cuda.synchronize()
    assert ref.same_bits(out, expected) and ref.same_bits(again, expected)
    _assert_ticket_tail_zero(runner)


@pytest.mark.parametrize("warm", [False, True])
def test_ticket_word_runner_graph_capture_before_and_after_first_use(warm):
    """A TicketWord runner captured before its first run and after it: the graph carries no memset node, and replays
    stay bit-exact with the caller-owned workspace poisoned between them."""
    inputs = _ticket_word_inputs(86)
    device = torch.device("cuda", torch.cuda.current_device())
    expected = _cloned(_prepare(inputs)())
    workspace = torch.empty(
        _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
    )
    workspace.fill_(0xFF)
    runner = _prepare(inputs, workspace_buffer=workspace)
    if warm:
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
    for fill in (7, 0xFF, 0):
        workspace.fill_(fill)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert ref.same_bits(captured, expected), fill
        _assert_ticket_tail_zero(runner)


def test_ticket_word_eager_entry_draws_from_the_tail_word_it_zeroes():
    """The eager entry on either workspace: its one scan launch draws from the tail's first bytes, which its binding
    zeroes (8 bytes) right before the launch; a poisoned caller-owned buffer leaves every call bit-exact."""
    inputs = _ticket_word_inputs(87)
    device = torch.device("cuda", torch.cuda.current_device())
    expected = _cloned(_prepare(inputs)())
    assert ref.same_bits(_cloned(_run(inputs)), expected)  # backend-owned workspace
    workspace = torch.empty(
        _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
    )
    for fill in (0xFF, 7, 1):
        workspace.fill_(fill)
        out = _run(inputs, workspace_buffer=workspace)
        torch.cuda.synchronize()
        assert ref.same_bits(out, expected), fill
        tail_at = _workspace_bytes(inputs, device) - cake_policy.TICKET_TAIL_BYTES
        assert not workspace[
            tail_at : tail_at + cake_backend.TICKET_WORD_ZERO_BYTES
        ].any(), fill


def test_static_ticket_word_calls_get_the_placeholder_and_shape_alone_decides():
    """A call that never draws (``ticket_static_path``, or a ``static_scan_kinds`` program) passes the device's zeroed
    placeholder as its TicketWord and gets no block; the decision is the shape's alone, so the same shapes with other
    segment contents decide the same way."""
    _require_program()
    inputs = ref.make_random_inputs([4], [22018], top_k=2048, seed=88, device="cuda")
    runner = _prepare(inputs)
    assert cake_policy.ticket_static_path(runner.choice)
    assert runner.ticket_block is None and runner._ticket_restore is None
    assert _scan_ticket_word(runner) is cake_backend._ticket_placeholder(
        runner.device_index
    )
    expected = _cloned(runner())
    torch.cuda.synchronize()
    placeholder = torch.from_dlpack(
        cake_backend._ticket_placeholder(runner.device_index)
    )
    assert (
        placeholder.numel() * placeholder.element_size()
        == cake_backend.TICKET_BLOCK_BYTES
    )
    a = _prepare(
        ref.make_random_inputs(
            [2048, 2048], [2048, 16384], top_k=1024, seed=89, device="cuda"
        )
    )
    # the same T, Tkv and S with other segment lengths and values
    b = _prepare(
        ref.make_random_inputs(
            [3000, 1096], [9000, 9384], top_k=1024, seed=90, device="cuda"
        )
    )
    if (a.choice.scan_key, a.choice.grid_ctas) == (
        b.choice.scan_key,
        b.choice.grid_ctas,
    ):
        assert (a._tail_static, a.ticket_block is None) == (
            b._tail_static,
            b.ticket_block is None,
        )
    for r in (a, b):
        static = (
            r._tail_static
            or r.choice.physical_kind in policy_for(r.arch, r.form).static_scan_kinds
        )
        assert (r.ticket_block is not None) == (
            not static and cake_backend.reads_ticket_word(r.programs["scan"], r.arch)
        )
        if r.ticket_block is None:
            assert _scan_ticket_word(r) is cake_backend._ticket_placeholder(
                r.device_index
            )
    assert ref.same_bits(runner(), expected)
    torch.cuda.synchronize()
    assert not placeholder.view(torch.uint8).any()  # no launch writes it


def test_workspace_bound_is_respected():
    _require_program()
    inputs = ref.make_random_inputs(
        [900, 300], [900, 7000], top_k=2048, seed=60, device="cuda"
    )
    device = torch.device("cuda", torch.cuda.current_device())
    nbytes = cake_dsa_indexer_det_train_topk_workspace_size(
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
        + cake_policy.TICKET_TAIL_BYTES
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    indices = torch.empty(
        (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=device
    )
    scores = torch.empty(
        (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=device
    )
    runner = prepare_dsa_indexer_det_topk(
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
        prepare_dsa_indexer_det_topk(
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
    indices, scores = cake_backend.cake_dsa_indexer_det_train_topk(
        q, k, w, cu, torch.tensor([0, 5], dtype=torch.int32, device=device), top_k=8
    )
    assert tuple(indices.shape) == (0, 8) and tuple(scores.shape) == (0, 8)
    inputs = ref.make_random_inputs([3, 4], [0, 0], top_k=8, seed=1, device="cuda")
    indices, scores = _run(inputs)
    assert bool((indices == -1).all()) and bool(torch.isneginf(scores).all())
    _judge_ok((indices, scores), ref.select_reference(inputs))


def test_persistent_rank_finalize_follows_the_architecture_cap():
    """Lever FRP-K (round 39): the persistent two-row pipelined rank finalize serves the staged bulk slab calls up to the
    architecture's ``rank_persist_max_k`` (sm_107a: 2048) and launches ``min(rows, rank_persist_ctas_per_sm x (SMs -
    rank_persist_spare_sms))`` CTAs; every
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
                assert choice.rank_persist_spare == policy.rank_persist_spare_sms
                assert finalize_grid(choice, 148) == min(
                    8192,
                    policy.rank_persist_ctas_per_sm
                    * (148 - policy.rank_persist_spare_sms),
                )
                assert finalize_grid(choice, 10_000) == 8192
            else:
                assert (
                    choice.rank_persist_ctas == 0
                    and choice.rank_persist_spare == 0
                    and finalize_grid(choice, 148) == 8192
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


_MERGE_RANK_CALLS = (
    # (num_queries, num_keys, num_segments): split calls whose finalize is the slab rank form (windows 16384 and 65536)
    (4, 22018, 1),
    (1, 16384, 1),
    (64, 16384, 1),
    (128, 32768, 2),
)


def test_persistent_rank_grid_spares_sms_for_another_stream():
    """#7548: where the record spares SMs (sm_107a: one), the persistent rank grid of a call with enough rows is
    ``rank_persist_ctas_per_sm x (SMs - spare)``, a multiple of the SMs that remain.  The :persist programs run 256 threads
    per CTA, so a 1024-thread SM107 SM holds 4 of them at a time; with ``rank_persist_ctas_per_sm`` a multiple of 4, a block
    of another stream on the spare SM leaves the static row walk whole waves of the other SMs instead of a short extra
    wave."""
    assert policy_for("sm_107a").rank_persist_spare_sms == 1
    for arch in cake_jit.ARCHES:
        policy = policy_for(arch)
        if policy.rank_persist_max_k <= 0:
            assert policy.rank_persist_spare_sms == 0, arch
            continue
        for sms in (148, 212):
            choice = select_program(
                policy,
                num_queries=8192,
                num_keys=8192,
                num_segments=1,
                ratio=1,
                top_k=2048,
                grid=sms,
            )
            assert choice.rank_persistent, (arch, sms)
            spare = policy.rank_persist_spare_sms
            grid = finalize_grid(choice, sms)
            assert grid == policy.rank_persist_ctas_per_sm * (sms - spare), (arch, sms)
            assert grid < policy.rank_persist_ctas_per_sm * sms, (arch, sms)
            # CTAs one 1024-thread SM107 SM holds at a time divide the CTAs per SM: whole waves
            assert choice.rank_threads == 256, (arch, choice.finalize_key)
            assert (
                policy.rank_persist_ctas_per_sm % (1024 // choice.rank_threads) == 0
            ), arch
        # a device of no more SMs than the spare count keeps one SM's grid
        assert (
            finalize_grid(choice, policy.rank_persist_spare_sms)
            == policy.rank_persist_ctas_per_sm
        )


def test_merge_rank_fuses_the_slab_rank_finalize_into_split_merges():
    """A split call whose finalize is the slab rank form runs the merge of key ``merge:rank:w<window>`` and no finalize
    program where the architecture's record turns ``merge_rank`` on; the finalize key it replaces is unchanged, a record
    with ``merge_rank`` off keeps the merge + finalize programs, and the FP8 form never fuses."""
    for arch in cake_jit.ARCHES:
        policy = policy_for(arch)
        off = DispatchPolicy.from_record(dict(policy.record(), merge_rank=False))
        keys = cake_jit.PROGRAM_KEYS[arch]
        grid = 212 if arch == "sm_107a" else 148
        for T, Tkv, S in _MERGE_RANK_CALLS:
            call = dict(
                num_queries=T,
                num_keys=Tkv,
                num_segments=S,
                ratio=1,
                top_k=2048,
                grid=grid,
                scan_keys=keys,
            )
            fused, plain = select_program(policy, **call), select_program(off, **call)
            assert plain.split and not plain.merge_rank, (arch, T, Tkv, S)
            assert plain.keys == (plain.scan_key, MERGE_KEY, plain.finalize_key)
            assert fused.merge_rank is bool(policy.merge_rank), (arch, T, Tkv, S)
            if not fused.merge_rank:
                assert fused == plain
                continue
            window = fused.rank_window
            assert window is not None and window <= policy.rank_slab_window_max
            assert (
                fused.merge_key == f"merge:rank:w{window}" and fused.merge_key in keys
            )
            assert fused.keys == (fused.scan_key, fused.merge_key)
            assert fused.finalize_key == plain.finalize_key
            assert (
                dataclasses.replace(fused, merge_key=MERGE_KEY, merge_rank=False)
                == plain
            )
            fp8 = plan_dsa_indexer_det_topk(
                T,
                Tkv,
                S,
                top_k=2048,
                arch=arch,
                num_sms=grid,
                dtype=torch.float8_e4m3fn,
            )
            assert not fp8.merge_rank and fp8.merge_key in (None, MERGE_KEY), arch
        # a call above the fused programs' top_k bound, an unsplit call and a two-level window keep the finalize program
        for call in (
            dict(num_queries=4, num_keys=22018, num_segments=1, top_k=4096),
            dict(num_queries=8192, num_keys=8192, num_segments=1, top_k=2048),
            dict(num_queries=4, num_keys=300000, num_segments=1, top_k=2048),
        ):
            c = select_program(policy, ratio=1, grid=grid, scan_keys=keys, **call)
            assert not c.merge_rank and c.keys[-1] == c.finalize_key, (arch, call)


def test_merge_rank_runs_are_bitwise_the_merge_plus_finalize_runs(monkeypatch):
    """The fused merge writes the bits of the merge + slab rank finalize pair: every output of a split call is bitwise the
    same with the architecture's record and with ``merge_rank`` off, and the fused call launches no finalize program."""
    _require_program()
    arch = arch_for(torch.device("cuda"))
    if not policy_for(arch).merge_rank:
        pytest.skip(f"merge_rank is off on {arch}")
    rows = (
        ([4], [22018]),
        ([64], [16384]),
        ([64, 64], [16384, 16384]),
        ([3, 5], [7000, 30000]),
    )
    for seed, (seg_q, seg_k) in enumerate(rows):
        inputs = ref.make_random_inputs(
            seg_q, seg_k, top_k=2048, seed=700 + seed, device="cuda"
        )
        runner = _prepare(inputs)
        if not runner.choice.split or not runner.choice.merge_rank:
            continue
        assert runner.stages == ("scan", "merge"), (seg_q, seg_k, runner.stages)
        fused = _cloned(runner())
        with monkeypatch.context() as m:
            original = cake_backend.policy_for

            def merge_rank_off(arch, form=cake_policy.BF16):
                return DispatchPolicy.from_record(
                    dict(original(arch, form).record(), merge_rank=False)
                )

            m.setattr(cake_backend, "policy_for", merge_rank_off)
            plain_runner = _prepare(inputs)
            assert plain_runner.stages == (
                "scan",
                "merge",
                plain_runner.choice.finalize_role,
            )
            plain = _cloned(plain_runner())
        assert torch.equal(fused[0], plain[0]), (seg_q, seg_k)
        assert torch.equal(fused[1].view(torch.int32), plain[1].view(torch.int32)), (
            seg_q,
            seg_k,
        )
        _judge_ok(fused, ref.select_reference(inputs))


def test_device_facts_are_queried_once_per_device(monkeypatch):
    """Compute capability and SM count are process-immutable: one ``get_device_properties`` per device index, the
    index-less device forms still resolve the current device on every call, and a CPU device is still refused."""
    _require_program()
    index = torch.cuda.current_device()
    device = torch.device("cuda", index)
    capability = tuple(torch.cuda.get_device_capability(device))
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    monkeypatch.setattr(cake_backend, "_DEVICE_FACTS", {})
    properties = mock.Mock(wraps=torch.cuda.get_device_properties)
    capabilities = mock.Mock(wraps=torch.cuda.get_device_capability)
    current = mock.Mock(wraps=torch.cuda.current_device)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capabilities)
    monkeypatch.setattr(torch.cuda, "current_device", current)
    for dev in (device, torch.device("cuda"), None, index, f"cuda:{index}"):
        for _ in range(3):
            assert arch_for(dev) == SUPPORTED_COMPUTE_CAPABILITIES[capability]
            plan = plan_dsa_indexer_det_topk(1024, 4096, 1, top_k=256, device=dev)
            assert plan.grid == sms
            assert (
                cake_dsa_indexer_det_train_topk_workspace_size(
                    1024, 4096, 1, top_k=256, device=dev
                )
                == plan.workspace_bytes
            )
    assert properties.call_count == 1 and capabilities.call_count == 0
    assert {index: (capability, sms)} == cake_backend._DEVICE_FACTS
    # the index-less forms resolve the current device per call
    assert current.call_count >= 6
    with pytest.raises(ValueError, match="cuda device"):
        arch_for(torch.device("cpu"))


def test_prepared_calls_perform_no_device_queries_after_warm_up(monkeypatch):
    """After the first call on a device the eager and the prepared paths run without a device-property query."""
    _require_program()
    inputs = ref.make_random_inputs(
        [200, 300], [200, 2500], top_k=128, seed=70, device="cuda"
    )
    expected = _run(inputs)
    boom = mock.Mock(side_effect=AssertionError("device query on the hot path"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    again = _run(inputs)
    runner = prepare_dsa_indexer_det_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
    )
    captured = runner()
    torch.cuda.synchronize()
    assert boom.call_count == 0
    assert ref.same_bits(again, expected) and ref.same_bits(captured, expected)


def test_policy_is_validated_once_per_architecture(monkeypatch):
    """The registry record is process-immutable: ``policy_for`` validates it once per architecture, the planning
    path re-validates nothing, and an unregistered architecture keeps raising by name."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    validate = mock.Mock(wraps=DispatchPolicy.from_record)
    monkeypatch.setattr(DispatchPolicy, "from_record", validate)
    for arch in cake_jit.ARCHES:
        policy = policy_for(arch)
        validate.reset_mock()
        assert policy_for(arch) is policy
        plan_dsa_indexer_det_topk(
            1024, 4096, 1, top_k=256, arch=arch, num_sms=_FROZEN_GRIDS[arch][0]
        )
        assert validate.call_count == 0, arch
    for _ in range(2):
        with pytest.raises(NotImplementedError, match="5676"):
            policy_for("sm_90a")


def test_lsf_copy_runs_on_packed_calls_of_unequal_key_segments():
    """Lever LSF (#7549): an architecture whose policy sets ``lsf_programs`` (a ticket architecture) ships
    longest-first segment order copies of its scan programs (key suffix ``:lsf``, each a program of its own): sm_100a
    and sm_103a of every scan program, sm_107a of its two l6 ticket programs (``_SM107_LSF_COPIES``); the host runs the
    copy on packed calls whose key segments are not all equal and the program without it on one-segment calls and on
    packed calls of equal key segments (``num_keys == S x max_seqlen_k``)."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        scan = {k for k in keys if k.startswith("scan:")}
        copies = {k for k in scan if k.endswith(":lsf")}
        lsf = policy_for(arch).lsf_programs
        every = {k + ":lsf" for k in scan - copies}
        shipped = _SM107_LSF_COPIES if arch == "sm_107a" else every
        assert copies == (shipped if lsf else set()), arch
        assert copies <= every, arch  # every copy has its program
        assert all(keys[k] != keys[k[: -len(":lsf")]] for k in copies), arch
        # the copy reorders the ticket draws, so only a ticket architecture can ship it
        assert not lsf or arch in cake_jit.TICKET_TAIL_ARCHES, arch
    p = policy_for("sm_100a")
    one = _select(p, 65536, 65536)
    assert not one.lsf and not one.scan_key.endswith(":lsf")
    equal = _select(p, 8 * 32768, 8 * 32768, S=8, max_seqlen_k=32768)
    assert not equal.lsf and not equal.scan_key.endswith(":lsf")
    unequal = _select(p, 8 * 32768, 8 * 32768, S=8, max_seqlen_k=65536)
    assert unequal.lsf and unequal.scan_key == equal.scan_key + ":lsf"
    assert unequal.scan_key == (
        scan_key(
            unequal.physical_kind,
            unequal.tile_unroll,
            unequal.snake,
            unequal.sample_fit,
            True,
        )
    )
    assert _select(p, 8 * 32768, 8 * 32768, S=8).lsf  # no bound: the host cannot tell
    off = DispatchPolicy.from_record(dict(p.record(), lsf_programs=False))
    assert not _select(off, 8 * 32768, 8 * 32768, S=8, max_seqlen_k=65536).lsf


def test_lsf_copy_runs_where_the_registry_has_it_and_never_changes_the_plan():
    """Lever LSF where the registry carries the copies of some programs only (sm_107a: its two l6 ticket programs): a
    packed call whose key segments the host cannot tell equal runs the longest-first copy of its decided program where
    the registry has it and the program itself elsewhere; either way the plan is the plan of the record with
    ``lsf_programs`` off, because the rules that ask the registry for a program look up its key without the suffix.
    Shown on the sm_107a registry with and without the two copies, and on sm_100a / sm_103a, whose registries carry
    every copy, so a call there never falls back.  The backend plan (the registry's own keys): four 16k-query documents
    over 32k keys run ``scan:l6:u1:s0:f0:lsf`` where that copy is registered, two 4k-query documents over 8k keys run
    ``scan:l6:u1:s1:f1``, which has no copy."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    # (T, Tkv, S, sample_tiles_max): the packed sm_107a calls whose programs the tests above pin -- eight causal 8k
    # documents (the unpaired l6 program, sampled fit), four 16k-query documents over 32k keys and two 1024-query
    # documents over 128k keys (the unpaired l6 program, plain fit), two 4k-query documents over 8k keys (the snake l6
    # program) -- and a grid of packed calls
    packed = [
        (q * S, k * S, S, sample)
        for q, k, S, sample in itertools.product(
            (4, 64, 1024, 4096), (8192, 32768, 131072), (2, 4), (None, 0)
        )
        if q <= k
    ]
    geometries = [
        (65536, 65536, 8, None),
        (65536, 65536, 8, 0),
        (65536, 131072, 4, 0),
        (2048, 262144, 2, None),
        (8192, 16384, 2, None),
        *packed,
    ]
    for arch, grid in (("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 212)):
        p, registry = policy_for(arch), set(cake_jit.PROGRAM_KEYS[arch])
        off = DispatchPolicy.from_record(dict(p.record(), lsf_programs=False))
        assert p.lsf_programs, arch
        plain = {k for k in registry if not k.endswith(":lsf")}
        key_sets = (
            [plain, plain | _SM107_LSF_COPIES] if arch == "sm_107a" else [registry]
        )
        for keys in key_sets:
            reached, kept = set(), 0
            for (T, Tkv, S, sample), top_k in itertools.product(
                geometries, (512, 2048)
            ):
                call = dict(
                    num_queries=T,
                    num_keys=Tkv,
                    num_segments=S,
                    ratio=1,
                    top_k=top_k,
                    grid=grid,
                    sample_tiles_max=sample,
                    scan_keys=keys,
                )
                where = (arch, len(keys), T, Tkv, S, top_k, sample)
                base, got = select_program(off, **call), select_program(p, **call)
                assert base.scan_key in keys, where
                copy = base.scan_key + ":lsf"
                if copy in keys:
                    assert got == replace(base, lsf=True, scan_key=copy), where
                    reached.add(copy)
                else:
                    assert got == base, where
                    kept += 1
                # equal key segments (num_keys == S x max_seqlen_k): the program itself
                bounded = dict(call, max_seqlen_k=Tkv // S)
                assert select_program(p, **bounded) == select_program(off, **bounded), (
                    where
                )
            if arch == "sm_107a":
                assert reached == keys & _SM107_LSF_COPIES and kept, arch
            else:
                assert reached and not kept, arch
    keys = cake_jit.PROGRAM_KEYS["sm_107a"]
    common = dict(top_k=2048, arch="sm_107a", num_sms=212)
    four = plan_dsa_indexer_det_topk(65536, 131072, 4, sample_tiles_max=0, **common)
    assert four.scan_key == _lsf_key(keys, "scan:l6:u1:s0:f0")
    assert four.lsf == four.scan_key.endswith(":lsf")
    equal = plan_dsa_indexer_det_topk(
        65536, 131072, 4, sample_tiles_max=0, max_seqlen_k=32768, **common
    )
    assert (equal.scan_key, equal.lsf) == ("scan:l6:u1:s0:f0", False)
    snake = plan_dsa_indexer_det_topk(8192, 16384, 2, **common)
    assert (snake.scan_key, snake.lsf) == ("scan:l6:u1:s1:f1", False)


def test_ticket_arg_arches_name_exactly_the_drawing_programs_of_their_architectures():
    """``ticket_arg_arches`` (written by the exporter from each program's source build) lists, per scan program, the
    architectures on which it draws from its TicketWord: on sm_107a every drawing scan program, on sm_100a / sm_103a
    none (they draw from the workspace's tail).  Both ways: no program of a ``static_scan_kinds`` kind reads it, an
    ``:lsf`` copy reads it only as the copy of a program that does (sm_107a's l6 ticket programs, whose copies keep the
    segment order in the workspace's tail), and every other scan program of an architecture that has a reader reads
    it -- so the host, which decides by the record alone (``reads_ticket_word``), never hands a drawing program the
    placeholder."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for name, record in cake_jit.PROGRAMS.items():
        reads = record.get("ticket_arg_arches")
        if reads is not None:
            assert record["role"] == "scan" and reads, name
            assert list(reads) == [a for a in record["arches"] if a in reads], name
    readers = {}
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        for key, name in keys.items():
            if not key.startswith("scan:"):
                continue
            # the static kinds of the key's form (on sm_107a every FP8 kind is static)
            static = set(policy_for(arch, scan_key_form(key)).static_scan_kinds)
            parts = key.split(":")
            reads = cake_backend.reads_ticket_word(name, arch)
            assert reads == (
                arch in cake_jit.PROGRAMS[name].get("ticket_arg_arches", ())
            ), (arch, key)
            if reads:
                assert parts[1] not in static, (arch, key)
            if parts[-1] == "lsf":
                plain = keys[key[: -len(":lsf")]]
                assert reads == cake_backend.reads_ticket_word(plain, arch), (arch, key)
            readers.setdefault(arch, []).append((reads, parts[1] in static))
    assert {arch: any(r for r, _ in rows) for arch, rows in readers.items()} == {
        "sm_100a": False,
        "sm_103a": False,
        "sm_107a": True,
    }
    for arch, rows in readers.items():
        if any(r for r, _ in rows):
            assert all(r != s for r, s in rows), (
                arch
            )  # every drawing program reads it, no static one does
    assert cake_jit.ARG_PLANS["scan"][-5] == ["buffer", "TicketWord"]
    assert cake_jit.ARG_PLANS["scan"][-6] == ["buffer", "K_scale"]
    assert all(
        ["buffer", "TicketWord"] not in plan and ["buffer", "K_scale"] not in plan
        for role, plan in cake_jit.ARG_PLANS.items()
        if role != "scan"
    )


def test_static_scan_kinds_are_registered_ticket_architecture_kinds():
    """Every architecture is a ticket architecture; a record's ``static_scan_kinds`` names the physical scan kinds whose
    programs keep the static enumeration there (no BF16 record names one; sm_107a's FP8 record does), each a kind the
    architecture registers, and a kind with an ``:lsf`` copy is never static (the copy reorders ticket draws)."""
    assert tuple(cake_jit.POLICY) == cake_jit.TICKET_TAIL_ARCHES
    assert {arch: policy_for(arch).static_scan_kinds for arch in cake_jit.POLICY} == {
        "sm_100a": [],
        "sm_103a": [],
        "sm_107a": [],
    }
    # prepare decides a call's tail from this per-architecture set (made once per process)
    for arch in cake_jit.POLICY:
        assert cake_backend.ticket_tail_skips(arch) == frozenset(
            policy_for(arch).static_scan_kinds
        )
    assert cake_backend.ticket_tail_skips("sm_90a") is None
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        scan = [k.split(":") for k in keys if k.startswith("scan:")]
        kinds = {parts[1] for parts in scan}
        lsf_kinds = {parts[1] for parts in scan if parts[-1] == "lsf"}
        for kind in policy_for(arch).static_scan_kinds:
            assert kind in kinds and kind not in lsf_kinds, (arch, kind)
        # the FP8 form decides its tail by its own record (``POLICY_FP8``): every static kind is an FP8 kind
        # the architecture registers without an :lsf copy
        if not cake_backend.fp8_programs_for(arch):
            continue
        fp8 = [
            scan_key_fields(k)
            for k in keys
            if k.startswith("scan:") and scan_key_form(k) == cake_policy.FP8
        ]
        fp8_static = policy_for(arch, cake_policy.FP8).static_scan_kinds
        assert cake_backend.ticket_tail_skips(arch, cake_policy.FP8) == frozenset(
            fp8_static
        )
        for kind in fp8_static:
            assert kind in {f.physical for f in fp8}, (arch, kind)
            assert kind not in {f.physical for f in fp8 if f.lsf}, (arch, kind)


# ---------------------------------------------------------------------------
# Eager entry: nothing launched before the scan
# ---------------------------------------------------------------------------

# a static call on every architecture (4 queries, split plan) and a call that draws tickets from a caller-owned buffer
_EAGER_GEOMETRIES = (([4], [22018], 2048), ([600, 400], [600, 12000], 1024))


def _device_ops(fn, tmp_path, name: str) -> list[str]:
    """The categories (``kernel`` / ``gpu_memset`` / ``gpu_memcpy``) of the device activities ``fn`` issues, in start
    order, from a profiler trace of one call (the stream is synchronized inside the trace)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = tmp_path / f"{name}.json"
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    ops = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")
        ),
        key=lambda e: e["ts"],
    )
    assert ops, "the profiler recorded no device activity"
    return [e["cat"] for e in ops]


def _expected_ops(runner) -> list[str]:
    """The device activities of one run: the scan binding's ``zero_region`` memset where it zeroes bytes, then one
    kernel per launch."""
    # a torch view, or the device's empty tvm_ffi region
    region = _scan_zero_region(runner)
    memset = ["gpu_memset"] if torch.Size(tuple(region.shape)).numel() else []
    return memset + ["kernel"] * len(runner.launches)


class _EagerCall:
    """One eager call with a caller-owned workspace and outputs; records the runner of every call."""

    def __init__(self, monkeypatch, inputs, device):
        self.inputs = inputs
        self.workspace = torch.empty(
            _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
        )
        self.indices = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=device
        )
        self.scores = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=device
        )
        self.runners = []
        prepare = cake_backend._prepare_runner

        def recording(*args, **kwargs):
            runner = prepare(*args, **kwargs)
            self.runners.append(runner)
            return runner

        monkeypatch.setattr(cake_backend, "_prepare_runner", recording)

    def __call__(self):
        i = self.inputs
        return cake_backend.cake_dsa_indexer_det_train_topk(
            i.q,
            i.k,
            i.w,
            i.cu_seqlens_q,
            i.cu_seqlens_k,
            **i.kwargs(),
            workspace_buffer=self.workspace,
            indices=self.indices,
            scores=self.scores,
        )


def test_offsets_placeholder_is_one_never_written_tensor_per_device():
    """A call without ``q_causal_offsets`` passes the device's offsets placeholder (one int64 element, ``has_offsets``
    = 0), and every call the device's ``K_scale`` placeholder (one float32 element, read by no BF16 program): the
    same tensors for every runner and eager call of the device, never written by any launch."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    holder = cake_backend._offsets_placeholder(device.index)
    assert holder is cake_backend._offsets_placeholder(device.index)
    assert holder.dtype == torch.int64 and holder.numel() == 1
    assert holder.device == device
    scales = cake_backend._unread_placeholder(device.index, torch.float32)
    assert scales.dtype == torch.float32 and scales.numel() == 1
    before = holder.clone()
    for n, (seg_q, seg_k, top_k) in enumerate(_EAGER_GEOMETRIES):
        inputs = ref.make_random_inputs(
            seg_q, seg_k, top_k=top_k, seed=93 + n, device="cuda"
        )
        assert inputs.q_causal_offsets is None
        runner = _prepare(inputs)
        assert runner.tensors["q_offsets"] is holder
        assert runner.tensors["K_scale"] is scales
        expected = _cloned(runner())
        for _ in range(2):
            assert ref.same_bits(runner(), expected)
            assert ref.same_bits(_run(inputs), expected)
        torch.cuda.synchronize()
        _judge_ok(expected, ref.select_reference(inputs))
    assert torch.equal(holder, before)


@pytest.mark.parametrize("geometry", range(len(_EAGER_GEOMETRIES)))
def test_eager_entry_launches_nothing_before_its_scan(monkeypatch, tmp_path, geometry):
    """One eager call issues exactly its runner's launches, the scan first (after its binding's ``zero_region``
    memset where that zeroes bytes): no fill or copy of its own before them."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    seg_q, seg_k, top_k = _EAGER_GEOMETRIES[geometry]
    inputs = ref.make_random_inputs(
        seg_q, seg_k, top_k=top_k, seed=95 + geometry, device="cuda"
    )
    call = _EagerCall(monkeypatch, inputs, device)
    expected = _cloned(call())  # loads the programs and makes the device's placeholders
    ops = _device_ops(call, tmp_path, f"eager{geometry}")
    runner = call.runners[-1]
    assert ops == _expected_ops(runner), (ops, runner.stages)
    assert runner.tensors["q_offsets"] is cake_backend._offsets_placeholder(
        device.index
    )
    assert ref.same_bits((call.indices, call.scores), expected)


@pytest.mark.parametrize("first_use", ["before_capture", "in_capture"])
def test_eager_entry_graph_capture_holds_only_the_runner_launches(
    monkeypatch, tmp_path, first_use
):
    """A CUDA graph capturing one eager call holds exactly the runner's launches, whether the device's offsets
    placeholder was made before the capture or first needed inside it (then the capture's own tensor, not kept);
    replays and later eager calls stay bit-exact, also after new values are copied into the bound inputs."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    seg_q, seg_k, top_k = _EAGER_GEOMETRIES[0]
    inputs = ref.make_random_inputs(seg_q, seg_k, top_k=top_k, seed=97, device="cuda")
    call = _EagerCall(monkeypatch, inputs, device)
    expected = _cloned(call())  # loads the programs and makes the device's placeholders
    kept = cake_backend._offsets_placeholder(device.index)
    if first_use == "in_capture":
        monkeypatch.setattr(cake_backend, "_UNREAD_PLACEHOLDERS", {})
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    runner = call.runners[-1]
    if first_use == "in_capture":
        assert runner.tensors["q_offsets"] is not kept
        # made during the capture: not kept
        assert cake_backend._UNREAD_PLACEHOLDERS == {}
    else:
        assert runner.tensors["q_offsets"] is kept
    ops = _device_ops(graph.replay, tmp_path, f"graph_{first_use}")
    assert ops == _expected_ops(runner), (ops, runner.stages)
    assert ref.same_bits(captured, expected)
    fresh = ref.make_random_inputs(seg_q, seg_k, top_k=top_k, seed=98, device="cuda")
    inputs.q.copy_(fresh.q)
    inputs.k.copy_(fresh.k)
    inputs.w.copy_(fresh.w)
    graph.replay()
    torch.cuda.synchronize()
    replayed = _cloned(captured)
    assert ref.same_bits(replayed, _run(fresh))
    assert ref.same_bits(call(), replayed)
    torch.cuda.synchronize()
    _judge_ok(replayed, ref.select_reference(inputs))
