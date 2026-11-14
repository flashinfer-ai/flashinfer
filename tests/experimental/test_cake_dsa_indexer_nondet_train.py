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

# Contract tests of the experimental Cake non-deterministic DSA indexer top-k backend
# (flashinfer-ai/flashinfer#5676 under the set-equality semantics of flashinfer-ai/flashinfer#8946, point 4).  The
# host-side tests (registry, dispatch policy, validation, binding, the deterministic-algorithms alert) run without a
# GPU; the operator tests need a compute capability 10.0 / 10.3 / 10.7 device with registered generated programs and
# skip otherwise.  Opt-in heavy variant: ``FLASHINFER_CAKE_DSA_INDEXER_FULL=1`` runs the recorded packed key geometry
# and the issue's sink geometry through the FP64 reference (minutes on B200).
#
# The package returns the selected entries of a row in no specified slot order, with the row's padding in any of its
# slots.  The operator tests compare results as sets: ``_canonical`` puts every row in the order of the shared
# reference (the valid slots by ascending id with their scores, then the padding), a per-row permutation that keeps
# each row's (id, score bits) pairs and valid count, so the reference's judgement (``ref.judge``) and bit identity
# (``ref.same_bits``) apply unchanged; ``_same_sets`` compares two results by their per-row valid counts and sorted
# (id, score bits) pairs.  No test compares the raw slot order of two calls.
#
# This suite exercises the package alone: it names the package once, through ``tests.cake.test_helpers.cake_package``, and
# imports nothing else from flashinfer, so a build that vendors the package under another import path runs it
# unchanged after pointing that table entry at its package.  The flashinfer public entry points
# (``flashinfer.cake_dsa_indexer_nondet_train``) are covered by ``test_cake_dsa_indexer_nondet_train_api.py`` and
# their #8928 naming rules by ``test_cake_dsa_indexer_nondet_train_names.py``; the parity with the deterministic package
# (registry, host plans, device rows) by ``test_cake_dsa_indexer_nondet_train_vs_det.py``.

import contextlib
import dataclasses
import itertools
import json
import os
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref
from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_backend")
cake_build = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_build")
cake_jit = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_jit")
cake_policy = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_policy")
ABI_CONTRACT = cake_backend.ABI_CONTRACT
BF16_ONLY = cake_backend.BF16_ONLY
CONTRACT_SCALARS = cake_backend.CONTRACT_SCALARS
CONTRACT_TENSORS = cake_backend.CONTRACT_TENSORS
HEAD_DIM = cake_backend.HEAD_DIM
MAX_TOP_K = cake_backend.MAX_TOP_K
NUM_HEADS = cake_backend.NUM_HEADS
RUN_ENTRY = cake_backend.RUN_ENTRY
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
TOPK_ENTRY = cake_backend.TOPK_ENTRY
ZERO_SIGN_POLICIES = cake_backend.ZERO_SIGN_POLICIES
arch_for = cake_backend.arch_for
bind_program = cake_backend.bind_program
cake_dsa_indexer_nondet_train_topk_workspace_size = (
    cake_backend.cake_dsa_indexer_nondet_train_topk_workspace_size
)
generated_program_available = cake_backend.generated_program_available
plan_dsa_indexer_nondet_topk = cake_backend.plan_dsa_indexer_nondet_topk
policy_for = cake_backend.policy_for
prepare_dsa_indexer_nondet_topk = cake_backend.prepare_dsa_indexer_nondet_topk
program_for = cake_backend.program_for
record_zero_sign_policy = cake_backend.record_zero_sign_policy
validate_dsa_indexer_nondet_inputs = cake_backend.validate_dsa_indexer_nondet_inputs
visible_key_count = cake_backend.visible_key_count
visible_key_counts = cake_backend.visible_key_counts
MERGE_KEY = cake_policy.MERGE_KEY
POLICY_FIELDS = cake_policy.POLICY_FIELDS
TICKET_TAIL_BYTES = cake_policy.TICKET_TAIL_BYTES
DispatchPolicy = cake_policy.DispatchPolicy
ProgramChoice = cake_policy.ProgramChoice
scan_key = cake_policy.scan_key
scan_key_fields = cake_policy.scan_key_fields
select_program = cake_policy.select_program
stage_slug = cake_policy.stage_slug

FULL = os.environ.get("FLASHINFER_CAKE_DSA_INDEXER_FULL", "0") not in (
    "",
    "0",
    "false",
    "no",
)

P1_CU_SEQLENS_Q = (0, 1777, 3532, 5655, 7765, 9888, 11998, 14121, 16231)
P1_CU_SEQLENS_K = (0, 1777, 58619, 60742, 128665, 130788, 198711, 200834, 268757)
P2_T = 16172
# the persistent grids (SM counts) of the devices of each architecture
_FROZEN_GRIDS = {"sm_100a": (148,), "sm_103a": (148, 152, 160), "sm_107a": (212,)}
# the plain split merge: the deterministic package's merge program under this package's prefix, with its twenty-hex
# identity
MERGE_PROGRAM = "cake_dsa_indexer_nondet_topk_b8440d1d1d54917a47d7"
_PROGRAM_NAME = re.compile(r"cake_dsa_indexer_nondet_topk_[0-9a-f]{20}")
# a word of the deterministic package's ordering pass (the finalize, the rank finalize, the rank-fused merge) or of its
# FP8 form: no role, key, program, policy field or module attribute of this package carries one
_ORDERING_OR_FP8 = re.compile(r"finalize|rank|fp8", re.IGNORECASE)
ALERT = "does not have a deterministic implementation"
# sorts after every int32 id: the padding of a row goes behind its valid slots
_PAD_SORT_KEY = 1 << 32


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
    out = cake_backend.cake_dsa_indexer_nondet_train_topk(
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
    return plan_dsa_indexer_nondet_topk(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=inputs.device,
        **extra,
    )


def _prepare(inputs: ref.IndexerInputs, **extra):
    return prepare_dsa_indexer_nondet_topk(
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


# ---------------------------------------------------------------------------
# Rows as sets
# ---------------------------------------------------------------------------


def _row_pairs(result) -> torch.Tensor:
    """int64 ``[T, K]``: every slot as ``id << 32 | score bits``, each row sorted -- the row's multiset of (id, score
    bits) pairs, its padding pairs included."""
    ids, scores = result
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((ids.to(torch.int64) << 32) | bits).sort(dim=1).values


def _valid_counts(indices: torch.Tensor) -> torch.Tensor:
    """int64 ``[T]``: the valid (non-negative) ids of every row."""
    return (indices >= 0).sum(dim=1)


def _canonical(result) -> tuple[torch.Tensor, torch.Tensor]:
    """``result`` with every row in the shared reference's slot order: the valid slots by ascending id with their
    scores, then the padding (any negative id keeps its score and goes behind them).  A stable per-row permutation of
    the slots, so every row keeps its (id, score bits) pairs and its valid count (checked once by
    ``test_set_comparison_ignores_slot_order_and_keeps_every_pair``): it cannot hide a duplicate, a missing entry, a
    changed bit or a bad padding slot, which the structure checks of ``ref.judge`` then see."""
    ids, scores = result
    key = torch.where(ids >= 0, ids.to(torch.int64), _PAD_SORT_KEY)
    order = key.sort(dim=1, stable=True).indices
    return ids.gather(1, order), scores.gather(1, order)


def _same_sets(a, b) -> bool:
    """Set identity of two ``(indices, scores)`` results: per row the same valid count and the same (id, score bits)
    pairs; the slot order and the positions of the padding are free."""
    ia, sa = a
    ib, sb = b
    if tuple(ia.shape) != tuple(ib.shape) or tuple(sa.shape) != tuple(sb.shape):
        return False
    return bool(torch.equal(_valid_counts(ia), _valid_counts(ib))) and bool(
        torch.equal(_row_pairs(a), _row_pairs(b))
    )


def _judge_ok(result, reference, **kw) -> ref.Judgement:
    if kw.get("exact_bits") and "zero_sign_policy" not in kw:
        kw["zero_sign_policy"] = record_zero_sign_policy()
    verdict = ref.judge(_canonical(result), reference, **kw)
    assert verdict.passed, str(verdict)
    return verdict


def _assert_sets_repeat(call, expected, calls: int = 5) -> None:
    """``calls`` further results of ``call`` select ``expected``'s set per row (pairs and valid count; no slot order
    is compared).  Each result is copied before the next call, so a runner writing into the same outputs works too."""
    for i in range(calls):
        assert _same_sets(_cloned(call()), expected), i


def _split_grids(inputs: ref.IndexerInputs, **extra) -> dict[str, int]:
    """The persistent grid sizes that force the key-range split: ``"split"`` the largest grid (from the SM count down
    to 2) on which the host plan of ``inputs`` splits the key ranges (``n_split > 1``), ``"unsplit"`` the largest on
    which it does not; a form no grid gives is absent.  ``grid_ctas`` enters the split rule
    (``DispatchPolicy.n_split_for``): a grid of more CTAs per query unit splits longer key ranges into more parts."""
    sms = torch.cuda.get_device_properties(inputs.device).multi_processor_count
    found: dict[str, int] = {}
    for grid in range(sms, 1, -1):
        plan = _plan(inputs, grid_ctas=grid, **extra)
        found.setdefault("split" if plan.n_split > 1 else "unsplit", grid)
        if len(found) == 2:
            break
    return found


def _independent_membership_check(inputs: ref.IndexerInputs, result) -> None:
    """Visibility and segment membership from the scalar rule in Python (no tensors shared with the reference), on
    the set of every row (its canonical order)."""
    ids = _canonical(result)[0].cpu().tolist()
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


@contextlib.contextmanager
def _torch_deterministic(*, warn_only: bool):
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)


def test_set_comparison_ignores_slot_order_and_keeps_every_pair():
    """The comparison helpers on a reference result (CPU): permuting the slots of every row (the padding included)
    changes neither the canonical form nor the pairs, and the canonical form keeps every row's pairs and valid count; a
    changed score bit (the lowest or the sign), two ids' scores swapped, a duplicated id, a dropped entry, a padding
    slot with a finite score or a padding id other than -1 changes the pairs and fails the judgement or the structure
    checks of the canonical form."""
    inputs = ref.make_random_inputs([9, 6], [11, 7], top_k=8, seed=3, device="cpu")
    reference = ref.select_reference(inputs, order="sequential")
    expected = (reference.indices, reference.scores)
    assert bool((_valid_counts(reference.indices) < inputs.top_k).any())  # padded rows
    gen = torch.Generator().manual_seed(5)
    order = torch.stack(
        [torch.randperm(inputs.top_k, generator=gen) for _ in range(inputs.num_queries)]
    )
    shuffled = expected[0].gather(1, order), expected[1].gather(1, order)
    assert not ref.same_bits(shuffled, expected)  # the slot order moved
    assert _same_sets(_canonical(shuffled), shuffled)
    assert ref.same_bits(_canonical(shuffled), expected)
    assert _same_sets(shuffled, expected)
    assert ref.judge(_canonical(shuffled), reference, exact_bits=True).passed
    r, c = (shuffled[0] >= 0).nonzero()[0].tolist()
    shuffled_bits = shuffled[1].view(torch.int32)
    # a second valid slot of row r with other score bits
    c2 = next(
        j
        for j in range(inputs.top_k)
        if j != c
        and int(shuffled[0][r, j]) >= 0
        and int(shuffled_bits[r, j]) != int(shuffled_bits[r, c])
    )
    # a padding slot
    pr, pc = (shuffled[0] < 0).nonzero()[0].tolist()
    # one score bit flipped: the lowest, then the sign
    for bit in (1, -0x80000000):
        ids, scores = _cloned(shuffled)
        scores.view(torch.int32)[r, c] ^= bit
        assert not _same_sets((ids, scores), expected), bit
        verdict = ref.judge(_canonical((ids, scores)), reference, exact_bits=True)
        assert not verdict.passed, bit
    # the scores of two ids of one row swapped
    ids, scores = _cloned(shuffled)
    bits = scores.view(torch.int32)
    a, b = int(bits[r, c]), int(bits[r, c2])
    bits[r, c], bits[r, c2] = b, a
    assert torch.equal(_valid_counts(ids), _valid_counts(expected[0]))
    assert not _same_sets((ids, scores), expected)
    assert not ref.judge(_canonical((ids, scores)), reference, exact_bits=True).passed
    # a padding slot with a finite score, and a padding id other than -1
    for pad_id, pad_score in ((-1, 0.0), (-2, float("-inf"))):
        ids, scores = _cloned(shuffled)
        ids[pr, pc], scores[pr, pc] = pad_id, pad_score
        assert torch.equal(_valid_counts(ids), _valid_counts(expected[0]))
        assert not _same_sets((ids, scores), expected), pad_id
        assert ref.structure_failures(inputs, *_canonical((ids, scores))), pad_id
    # one id duplicated within its row
    ids, scores = _cloned(shuffled)
    others = [j for j in range(inputs.top_k) if j != c and int(ids[r, j]) >= 0]
    ids[r, others[0]], scores[r, others[0]] = ids[r, c], scores[r, c]
    assert not _same_sets((ids, scores), expected)
    assert ref.structure_failures(inputs, *_canonical((ids, scores)))
    # one entry dropped to padding
    ids, scores = _cloned(shuffled)
    ids[r, c], scores[r, c] = -1, float("-inf")
    assert not torch.equal(_valid_counts(ids), _valid_counts(expected[0]))
    assert not _same_sets((ids, scores), expected)
    assert ref.structure_failures(inputs, *_canonical((ids, scores)))


# ---------------------------------------------------------------------------
# Host-only: registry, dispatch policy, validation, binding
# ---------------------------------------------------------------------------


def test_registry_records_are_well_formed():
    assert cake_jit.ABI == ABI_CONTRACT
    assert record_zero_sign_policy() in ZERO_SIGN_POLICIES
    # the scan and the plain split merge, positionally as the deterministic package binds them; no ordering pass
    assert cake_jit.STAGES == ["scan", "merge"]
    assert set(cake_jit.ARG_PLANS) == set(cake_jit.STAGES)
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
    assert [name for _, name in cake_jit.ARG_PLANS["merge"]] == [
        "Staging",
        "Indices",
        "Scores",
        "top_k",
        "n_split",
        "grid_x",
        "grid_y",
        "grid_z",
    ]
    # build.json (``cake_build``) is the only record of what to compile; the registry names its entries
    # ``<library>_<arch>``, and the manifest compiles the registered programs and nothing else
    package = cake_jit.__file__.rsplit("/", 1)[0]
    manifest = cake_build.manifest()
    libraries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert len(libraries) == len(manifest["libraries"])  # names unique
    assert set(cake_jit.ARCHES) <= set(manifest["arch_flags"])
    assert set(libraries) == {
        f"{record['library']}_{arch}"
        for record in cake_jit.PROGRAMS.values()
        for arch in record["arches"]
    }
    for name, record in cake_jit.PROGRAMS.items():
        assert _PROGRAM_NAME.fullmatch(name), name
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
    # one merge program: the deterministic package's plain split merge
    assert [n for n, r in cake_jit.PROGRAMS.items() if r["role"] == "merge"] == [
        MERGE_PROGRAM
    ]
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        assert arch in cake_jit.ARCHES
        policy = policy_for(arch)
        assert set(cake_jit.POLICY[arch]) == set(POLICY_FIELDS)
        scan_keys = sorted(k for k in keys if k.startswith("scan:"))
        assert scan_keys and set(keys) == set(scan_keys) | {MERGE_KEY}, arch
        assert keys[MERGE_KEY] == MERGE_PROGRAM, arch
        # every scan key follows the key grammar of the BF16 form
        for key in scan_keys:
            fields = scan_key_fields(key)
            assert fields is not None, (arch, key)
            assert (key.split(":")[-1] == "lsf") == fields.lsf, (arch, key)
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
        for key, name in keys.items():
            record = cake_jit.PROGRAMS[name]
            assert arch in record["arches"], (arch, key, name)
            assert record["role"] == key.split(":", 1)[0], (arch, key, name)
            assert stage_slug(key).isidentifier(), key


def test_no_role_key_program_or_name_of_an_ordering_pass_or_the_fp8_form():
    """This package ships the deterministic package's BF16 scan programs and its plain split merge only: no role, key,
    argument, policy field, plan field or module attribute names the finalize, the rank finalize, the rank-fused merge
    or the FP8 form, and the scan key grammar has no FP8 form."""
    names = [
        *cake_jit.STAGES,
        *cake_jit.ARG_PLANS,
        *[name for plan in cake_jit.ARG_PLANS.values() for _, name in plan],
        *[key for keys in cake_jit.PROGRAM_KEYS.values() for key in keys],
        *[field for record in cake_jit.POLICY.values() for field in record],
        *POLICY_FIELDS,
        *[f.name for f in dataclasses.fields(DispatchPolicy)],
        *[f.name for f in dataclasses.fields(ProgramChoice)],
        *cake_policy.ScanKeyFields._fields,
        *[field for record in cake_jit.PROGRAMS.values() for field in record],
    ]
    assert not [n for n in names if _ORDERING_OR_FP8.search(n)]
    for module in (cake_jit, cake_policy, cake_backend):
        found = [n for n in vars(module) if _ORDERING_OR_FP8.search(n)]
        assert not found, (module.__name__, found)
    # the deterministic package's FP8 scan keys (``:fp8`` before ``:lsf``) are not keys of this grammar
    for key in ("scan:l6:u1:s0:f1:fp8", "scan:l6:u1:s0:f1:fp8:lsf"):
        assert scan_key_fields(key) is None, key
    assert scan_key_fields("scan:l6:u1:s0:f1:lsf") == cake_policy.ScanKeyFields(
        "l6", 1, False, True, True
    )


_UNIT_SOURCE = re.compile(
    r"^csrc/cake_dsa_indexer_nondet_topk/"
    r"(cake_dsa_indexer_nondet_topk_[0-9a-f]{20})_(kernel|binding)\.cu$"
)
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
    # group) build one kernel translation unit: it names its kernel
    # ``CAKE_PROGRAM_KERNEL`` and reads those constants from ``CAKE_INDEXER_<KNOB>``,
    # each with an ``#ifndef`` default naming the program the unit was exported for;
    # every other entry passes its own values with ``-D``.  A program with a unit of
    # its own spells its symbol.  All libraries of one architecture load into one
    # process, so every entry must still produce ``kernel_<program>``, and every
    # definition an entry passes must be one its source reads.
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
                if _PROGRAM_NAME.search(line)
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


def test_policy_records_round_trip_and_reject_foreign_fields():
    """Every architecture's registry record carries exactly ``POLICY_FIELDS`` and round-trips through
    ``DispatchPolicy``; a record with a field of the deterministic package's ordering pass or FP8 form, without one
    of its own fields, or with a malformed ``sample_long_rule`` is rejected."""
    for arch, raw in cake_jit.POLICY.items():
        policy = policy_for(arch)
        record = policy.record()
        assert tuple(record) == POLICY_FIELDS
        assert DispatchPolicy.from_record(record) == policy
        assert DispatchPolicy.from_record(dict(raw)) == policy
        for foreign in (
            dict(finalize_threads=512),
            dict(rank_finalize=True),
            dict(merge_rank=False),
        ):
            with pytest.raises(ValueError):
                DispatchPolicy.from_record(dict(record, **foreign))
        with pytest.raises(ValueError):
            DispatchPolicy.from_record(
                {k: v for k, v in record.items() if k != "split_units"}
            )
        with pytest.raises(ValueError):
            DispatchPolicy.from_record(dict(record, far_sample=2))
        # a malformed long rule (kinds not a flat list of physical kind names) is a ValueError, not a TypeError
        for kinds in ([["l6"]], ["l6", 6], ["no_such_kind"], []):
            with pytest.raises(ValueError):
                DispatchPolicy.from_record(dict(record, sample_long_rule=[kinds, 500]))


def test_split_calls_run_the_plain_merge_and_unsplit_calls_the_scan_alone():
    """``ProgramChoice.keys`` is a call's launch order: the scan, then ``MERGE_KEY`` where the plan splits the key
    ranges (``n_split > 1``) and the split staging sits behind the candidate buffers; an unsplit call runs the scan
    alone without staging.  Every key is registered, both kinds of plan occur on every architecture, and the backend's
    host plan is the policy's with the registered scan keys."""
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        policy = policy_for(arch)
        seen = set()
        for grid, (T, Tkv, S), sample in itertools.product(
            _FROZEN_GRIDS[arch],
            (
                (4, 22018, 1),
                (64, 32768, 1),
                (64, 262144, 1),
                (4096, 65536, 1),
                (16384, 16384, 1),
                (8 * 2048, 8 * 32768, 8),
            ),
            (None, 0),
        ):
            c = _select(
                policy, T, Tkv, S=S, grid=grid, sample_tiles_max=sample, scan_keys=keys
            )
            if c.n_split > 1:
                assert c.split and c.merge_key == MERGE_KEY, c
                assert c.keys == (c.scan_key, MERGE_KEY), c
                assert c.staging_entries == T * c.n_split * c.top_k, c
            else:
                assert not c.split and c.merge_key is None, c
                assert c.keys == (c.scan_key,) and c.staging_entries == 0, c
            assert all(key in keys for key in c.keys), (arch, c.keys)
            assert (
                c.workspace_bytes
                == (c.scan_entries + c.staging_entries) * policy.candidate_entry_bytes
                + TICKET_TAIL_BYTES
            )
            plan = plan_dsa_indexer_nondet_topk(
                T, Tkv, S, top_k=2048, arch=arch, num_sms=grid, sample_tiles_max=sample
            )
            assert plan == c, (arch, grid, T, Tkv, S, sample)
            seen.add(c.n_split > 1)
        assert seen == {True, False}, arch


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
    fails closed by name on an unregistered program, so a selectable but unregistered program is a production failure.
    Both sample modes (the unsampled plan family is otherwise unseen by the grid), with and without the registry
    (``l6_solo_rule`` is on only with it; without it a packed call names the longest-first copy of its program, which
    the backend runs only where the registry has it), and on packed calls with the bound of equal key segments (the
    plain-order program instead of the longest-first copy); a split plan adds the plain merge, and nothing else."""
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        policy = policy_for(arch)
        for grid in _FROZEN_GRIDS[arch]:
            for T, Tkv, S, ratio, top_k in _dense_geometries(policy):
                bounds = [None] + ([-(-Tkv // S)] if S > 1 else [])
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
                        selected = [choice.scan_key] + (
                            [MERGE_KEY] if choice.n_split > 1 else []
                        )
                        assert list(choice.keys) == selected
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
    assert c.keys == (c.scan_key,)
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
                            assert all(key in keys for key in c.keys), (
                                arch,
                                grid,
                                T,
                                Tkv,
                                S,
                                sample,
                                c.keys,
                            )


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
        keys = cake_jit.PROGRAM_KEYS[arch]
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
        keys = cake_jit.PROGRAM_KEYS[arch]
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
    the dispatch never composes a sampled-fit key the architecture does not ship.  A descending-walk twin (key kind
    ``<kind>_desc``) is not a decided kind: its ascending program carries the form."""
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
    assert c.keys == ("scan:split_l6:u1:s0:f0", MERGE_KEY)
    wider = DispatchPolicy.from_record(
        dict(p.record(), sample_fit_kinds=sorted(p.sample_fit_kinds + ["split_l6"]))
    )
    c = select_program(wider, candidate_multiplier=16, **geometry)
    assert c.scan_key == "scan:split_l6:u1:s0:f1" and c.sample_fit


def test_lsf_copy_runs_on_packed_calls_of_unequal_key_segments():
    """Lever LSF (#7549): an architecture whose policy sets ``lsf_programs`` (a ticket architecture) ships
    longest-first segment order copies of its scan programs (key suffix ``:lsf``, each a program of its own): sm_100a
    and sm_103a of every scan program, sm_107a of its two l6 ticket programs (``_SM107_LSF_COPIES``); the host runs the
    copy on packed calls whose key segments are not all equal and the program without it on one-segment calls and on
    packed calls of equal key segments (``num_keys == S x max_seqlen_k``)."""
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
    ``scan:l6:u1:s1:f1``, which has no copy (the deterministic package's plans of these calls)."""
    # (T, Tkv, S, sample_tiles_max): packed sm_107a calls of the unpaired l6 program in the sampled-fit form (eight
    # causal 8k documents) and the plain form (four 16k-query documents over 32k keys, two 1024-query documents over
    # 128k keys), of the snake l6 program (two 4k-query documents over 8k keys), and a grid of packed calls
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
                    assert got == dataclasses.replace(base, lsf=True, scan_key=copy), (
                        where
                    )
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
    four = plan_dsa_indexer_nondet_topk(65536, 131072, 4, sample_tiles_max=0, **common)
    assert four.scan_key == _lsf_key(keys, "scan:l6:u1:s0:f0")
    assert four.lsf == four.scan_key.endswith(":lsf")
    equal = plan_dsa_indexer_nondet_topk(
        65536, 131072, 4, sample_tiles_max=0, max_seqlen_k=32768, **common
    )
    assert (equal.scan_key, equal.lsf) == ("scan:l6:u1:s0:f0", False)
    snake = plan_dsa_indexer_nondet_topk(8192, 16384, 2, **common)
    assert (snake.scan_key, snake.lsf) == ("scan:l6:u1:s1:f1", False)


def test_ticket_arg_arches_name_exactly_the_drawing_programs_of_their_architectures():
    """``ticket_arg_arches`` (written by the exporter from each program's source build) lists, per scan program, the
    architectures on which it draws from its TicketWord: on sm_107a every drawing scan program, on sm_100a / sm_103a
    none (they draw from the workspace's tail).  Both ways: no program of a ``static_scan_kinds`` kind reads it, an
    ``:lsf`` copy reads it only as the copy of a program that does (sm_107a's l6 ticket programs, whose copies keep the
    segment order in the workspace's tail), and every other scan program of an architecture that has a reader reads
    it -- so the host, which decides by the record alone (``reads_ticket_word``), never hands a drawing program the
    placeholder."""
    for name, record in cake_jit.PROGRAMS.items():
        reads = record.get("ticket_arg_arches")
        if reads is not None:
            assert record["role"] == "scan" and reads, name
            assert list(reads) == [a for a in record["arches"] if a in reads], name
    readers = {}
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        static = set(policy_for(arch).static_scan_kinds)
        for key, name in keys.items():
            if not key.startswith("scan:"):
                continue
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
            # every drawing program reads it, no static one does
            assert all(r != s for r, s in rows), arch
    assert cake_jit.ARG_PLANS["scan"][-5] == ["buffer", "TicketWord"]
    assert cake_jit.ARG_PLANS["scan"][-6] == ["buffer", "K_scale"]
    assert all(
        ["buffer", "TicketWord"] not in plan and ["buffer", "K_scale"] not in plan
        for role, plan in cake_jit.ARG_PLANS.items()
        if role != "scan"
    )


def test_static_scan_kinds_are_registered_ticket_architecture_kinds():
    """Every architecture is a ticket architecture; a record's ``static_scan_kinds`` names the physical scan kinds whose
    programs keep the static enumeration there, each a kind the architecture registers, and a kind with an ``:lsf``
    copy is never static (the copy reorders ticket draws).  No record names one: every scan program draws its work
    units, the sm_107a l6 and CTA-pair l6 programs from their ``TicketWord`` block (flashinfer-ai/flashinfer#7548)."""
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
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        scan = [k.split(":") for k in keys if k.startswith("scan:")]
        kinds = {parts[1] for parts in scan}
        lsf_kinds = {parts[1] for parts in scan if parts[-1] == "lsf"}
        for kind in policy_for(arch).static_scan_kinds:
            assert kind in kinds and kind not in lsf_kinds, (arch, kind)


def test_sm107a_sampling_rules_cover_the_7549_rank_geometries():
    """#7549 on sm_107a: the packed calls of short query segments and few mean unit tiles (the 22-segment packed rank
    inputs) sample ``sample_class_rule[0]`` tiles at its shift (a call whose mean unit holds no more tiles keeps the
    tiny-unit tiles), and the long-unit l6 calls (the two-segment 262144-key
    document ranks) take ``sample_long_rule``'s shift at the long-unit tile count, for every ``top_k``.  A long-unit call
    of another kind keeps the long-unit defaults, explicit knobs stay, and sampling off stays off.  The other
    architectures set no long-unit rule."""
    assert policy_for("sm_100a").sample_long_rule is None
    assert policy_for("sm_103a").sample_long_rule is None
    arch, grid = "sm_107a", _FROZEN_GRIDS["sm_107a"][0]
    policy = policy_for(arch)
    class_rule, long_rule = policy.sample_class_rule, policy.sample_long_rule
    assert class_rule is not None and long_rule is not None

    def plan(T, Tkv, S, top_k=2048, **knobs):
        return plan_dsa_indexer_nondet_topk(
            T, Tkv, S, top_k=top_k, arch=arch, num_sms=grid, **knobs
        )

    def sampling(choice):
        return choice.sample_tiles_max, choice.sample_shift_permille

    assert sampling(plan(17536, 289344, 22)) == tuple(class_rule)
    # a packed call whose mean unit holds no more tiles than the rule's keeps the tiny-unit tiles at the rule's shift
    assert policy.mean_unit_tiles(10624, 43824, 16, 4) <= class_rule[0]
    assert sampling(plan(10624, 43824, 16, top_k=512, ratio=4)) == (
        policy.sample_tiles_tiny_units,
        class_rule[1],
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
    for arch in cake_jit.ARCHES:
        grid = _FROZEN_GRIDS[arch][0]
        plan = plan_dsa_indexer_nondet_topk(
            1024, 4096, 1, top_k=256, arch=arch, num_sms=grid
        )
        assert plan.far_sample in (0, 1)
        assert plan.far_sample == policy_for(arch).far_sample
        for value in (0, 1):
            assert (
                plan_dsa_indexer_nondet_topk(
                    1024, 4096, 1, top_k=256, arch=arch, num_sms=grid, far_sample=value
                ).far_sample
                == value
            )
        for bad in (2, -1, True):
            with pytest.raises(ValueError, match="far_sample"):
                plan_dsa_indexer_nondet_topk(
                    1024, 4096, 1, top_k=256, arch=arch, num_sms=grid, far_sample=bad
                )


def test_policy_is_validated_once_per_architecture(monkeypatch):
    """The registry record is process-immutable: ``policy_for`` validates it once per architecture, the planning
    path re-validates nothing, and an unregistered architecture keeps raising by name."""
    validate = mock.Mock(wraps=DispatchPolicy.from_record)
    monkeypatch.setattr(DispatchPolicy, "from_record", validate)
    for arch in cake_jit.ARCHES:
        policy = policy_for(arch)
        validate.reset_mock()
        assert policy_for(arch) is policy
        plan_dsa_indexer_nondet_topk(
            1024, 4096, 1, top_k=256, arch=arch, num_sms=_FROZEN_GRIDS[arch][0]
        )
        assert validate.call_count == 0, arch
    for _ in range(2):
        with pytest.raises(NotImplementedError, match="5676"):
            policy_for("sm_90a")
    with pytest.raises(NotImplementedError, match="5676"):
        program_for("sm_90a", "scan:narrow:u1:s0:f0")
    with pytest.raises(NotImplementedError, match="finalize:t256"):
        program_for("sm_100a", "finalize:t256")


def test_plan_and_workspace_bound_take_the_bf16_form_only():
    """The host plan and the workspace bound take ``dtype=torch.bfloat16`` (the default); the FP8 form of the
    deterministic package raises with the package's BF16-only text and any other dtype raises by name, both before the
    device is consulted (``arch`` / ``num_sms`` given here, no device read).  The bound is the larger of the sampled
    and the unsampled plans; an ``max_seqlen_k`` that bounds no key segment is rejected by the plan."""
    for arch in cake_jit.ARCHES:
        grid = _FROZEN_GRIDS[arch][0]
        call = dict(top_k=2048, arch=arch, num_sms=grid)
        for kw in (dict(call), dict(call, dtype=torch.bfloat16)):
            assert plan_dsa_indexer_nondet_topk(4096, 65536, 2, **kw) == (
                plan_dsa_indexer_nondet_topk(4096, 65536, 2, **call)
            )
        assert cake_dsa_indexer_nondet_train_topk_workspace_size(
            4096, 65536, 2, **call
        ) == max(
            plan_dsa_indexer_nondet_topk(
                4096, 65536, 2, sample_tiles_max=sample, **call
            ).workspace_bytes
            for sample in (None, 0)
        )
        for fn in (
            plan_dsa_indexer_nondet_topk,
            cake_dsa_indexer_nondet_train_topk_workspace_size,
        ):
            with pytest.raises(ValueError, match="BF16 form only"):
                fn(4096, 65536, 2, dtype=torch.float8_e4m3fn, **call)
            with pytest.raises(ValueError, match="bfloat16"):
                fn(4096, 65536, 2, dtype=torch.float16, **call)
        for bad in (65537, 32767):  # above Tkv; below ceil(Tkv / S)
            with pytest.raises(ValueError, match="max_seqlen_k"):
                plan_dsa_indexer_nondet_topk(4096, 65536, 2, max_seqlen_k=bad, **call)


def test_select_program_rejects_an_invalid_bound_and_knobs():
    """The policy's own dispatch validates the caller's bound on every key segment: the P1 row's bound (67923) is
    kept on the plan, a bound that cannot hold every key segment (below ``ceil(Tkv / S)``, or above ``Tkv``) is
    rejected; the check period, the grid and the far-sample switch are validated likewise."""
    for arch in cake_jit.ARCHES:
        p = policy_for(arch)
        c = _select(p, 16231, 268757, S=8, max_seqlen_k=67923)
        assert c.max_seqlen_k == 67923
        assert _select(p, 16231, 268757, S=8).max_seqlen_k is None
        for bad in (1000, 268758):
            with pytest.raises(ValueError, match="is not an upper bound"):
                _select(p, 16231, 268757, S=8, max_seqlen_k=bad)
        for bad in (0, 1 << 20):
            with pytest.raises(ValueError, match="check_period"):
                _select(p, 8192, 8192, check_period=bad)
        with pytest.raises(ValueError, match="grid_ctas"):
            _select(p, 8192, 8192, grid=0)
        with pytest.raises(ValueError, match="far_sample"):
            _select(p, 8192, 8192, far_sample=2)


def test_program_keys_and_slugs():
    assert scan_key("narrow", 1, False, True) == "scan:narrow:u1:s0:f1"
    assert scan_key("l6", 1, False, False, True) == "scan:l6:u1:s0:f0:lsf"
    assert stage_slug("scan:split_wide:u2:s0:f0") == "scan_split_wide_u2_s0_f0"
    assert stage_slug(MERGE_KEY) == "merge"
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
    return validate_dsa_indexer_nondet_inputs(
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
        # the FP8 form of the deterministic package: this package offers the BF16 form only
        (lambda h: setattr(h, "q", h.q.to(torch.float8_e4m3fn)), "BF16 form only"),
        (lambda h: setattr(h, "k", h.k.to(torch.float8_e4m3fn)), "BF16 form only"),
        (
            lambda h: (
                setattr(h, "q", h.q.to(torch.float8_e4m3fn)),
                setattr(h, "k", h.k.to(torch.float8_e4m3fn)),
            ),
            "BF16 form only",
        ),
    ],
)
def test_validate_rejects(mutate, match):
    h = _host_inputs()
    mutate(h)
    with pytest.raises(ValueError, match=match):
        _validate(h)


def test_fp8_rejection_names_the_deterministic_package():
    """The BF16-only text tells the caller where the FP8 form lives."""
    assert "BF16 form only" in BF16_ONLY
    assert "flashinfer.cake_dsa_indexer_det_train" in BF16_ONLY
    h = _host_inputs()
    h.q, h.k = h.q.to(torch.float8_e4m3fn), h.k.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError) as info:
        _validate(h)
    assert BF16_ONLY in str(info.value)


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
    # no ordering role to bind
    with pytest.raises(ValueError, match="role must be one of"):
        bind_program("fake", "sm_100a", "finalize", values, (1, 1, 1))


# ---------------------------------------------------------------------------
# Host-only: torch.use_deterministic_algorithms
# ---------------------------------------------------------------------------


def _alert_match(entry: str) -> str:
    """The ``match`` pattern of the alert naming ``entry`` (the run entry's name holds regex metacharacters)."""
    return re.escape(f"{entry} {ALERT}")


@pytest.mark.parametrize("warn_only", [None, False, True])
@pytest.mark.parametrize("entry", ["topk", "run"])
def test_alert_follows_torch_deterministic_mode(entry, warn_only):
    """The alert helper itself (CPU): nothing outside the mode; ``RuntimeError`` under it; a ``UserWarning`` naming
    the entry with ``warn_only=True``.  The eager entry alerts under its own name, every run of a prepared runner under
    the runner's; preparing has no entry name.  The message names no other package to fall back to."""
    alert = cake_backend._alert_nondeterministic
    name = TOPK_ENTRY if entry == "topk" else RUN_ENTRY
    assert (TOPK_ENTRY, RUN_ENTRY) == (
        "cake_dsa_indexer_nondet_train_topk",
        "DSAIndexerNondetRunner.run (prepare_dsa_indexer_nondet_topk)",
    )
    assert not hasattr(cake_backend, "PREPARE_ENTRY")
    if warn_only is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            alert(name)
        return
    with _torch_deterministic(warn_only=warn_only):
        if warn_only:
            with pytest.warns(UserWarning, match=_alert_match(name)) as record:
                alert(name)
            assert len(record) == 1
            message = str(record[0].message)
        else:
            with pytest.raises(RuntimeError, match=_alert_match(name)) as info:
                alert(name)
            message = str(info.value)
    assert "_det_train" not in message


@pytest.mark.parametrize("warn_only", [False, True])
def test_workspace_bound_never_consults_torch_deterministic_mode(warn_only):
    """The workspace bound is host arithmetic: under ``torch.use_deterministic_algorithms(True)`` (strict or
    ``warn_only``) it neither raises nor warns and returns the bound outside the mode (``arch`` / ``num_sms`` given:
    no device read)."""
    for arch, grids in _FROZEN_GRIDS.items():
        call = dict(top_k=2048, arch=arch, num_sms=grids[0])
        expected = cake_dsa_indexer_nondet_train_topk_workspace_size(
            16231, 268757, 8, **call
        )
        with _torch_deterministic(warn_only=warn_only), warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                cake_dsa_indexer_nondet_train_topk_workspace_size(
                    16231, 268757, 8, **call
                )
                == expected
            )
            plan_dsa_indexer_nondet_topk(16231, 268757, 8, **call)


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
    indices, scores = _canonical(_run(inputs))
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
    _independent_membership_check(inputs, out)
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


def _duplicated_key_inputs(lq: int, n_distinct: int, reps: int, seed: int):
    """One causal segment of ``lq`` queries over ``n_distinct * reps`` keys, each distinct key replicated ``reps`` times
    at random positions; returns the inputs and the replica group of every key."""
    K = 2048
    base = ref.make_random_inputs([lq], [n_distinct], top_k=K, seed=seed, device="cuda")
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
    return inputs, group


def _assert_cut_groups_keep_the_larger_ids(inputs, out, group) -> int:
    """Where the cutoff of a row splits a replica group (bit-identical scores), the row keeps the group's largest
    ids; returns the number of such cut groups."""
    K = inputs.top_k
    ids = _canonical(out)[0].cpu()
    group_cpu = group.cpu()
    visible = ref.visible_rows(inputs, device="cpu")
    cut = 0
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
                cut += 1
                assert torch.equal(
                    chosen.sort().values, members.sort().values[-chosen.numel() :]
                ), (t, g)
    return cut


def test_near_tie_duplicated_keys_prefer_larger_ids():
    """Replicated keys give bit-identical scores under any scheme of a fixed reduction: the larger id must win."""
    _require_program()
    inputs, group = _duplicated_key_inputs(48, 300, 8, seed=21)
    reference = ref.select_reference(inputs, order="sequential")
    out = _run(inputs)
    _judge_ok(out, reference)
    _assert_cut_groups_keep_the_larger_ids(inputs, out, group)


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
# Operator: dynamic shapes, dispatch coverage, set repeatability, partition independence
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
# architecture's policy; every kind selects the same set).  The 64-query / 256k-key row splits its key ranges
# on every registered architecture and so exercises the plain merge; the 16k causal document reaches the CTA-pair
# program where the architecture's rule admits it; the 8k-query / 72k-key row reaches the wide-first precedence
# where the architecture's record sets a floor; the 16k causal document at top_k 4096 reaches the sampled rank
# shift rule where the architecture's record sets one.
_DISPATCH_ROWS = [
    # 16 narrow units over 2047 tiles: key-range split + plain merge
    ([64], [262144], 2048),
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
    # the 32768-slot ceiling), the 250 default elsewhere; the same set for every shift
    ([16384], [16384], 4096),
    ([4096], [4096], 256),  # small top_k: the capacity floor
]


@pytest.mark.parametrize("seg_q,seg_k,top_k", _DISPATCH_ROWS)
def test_dispatch_rows_are_exact_and_plan_matches_runner(seg_q, seg_k, top_k):
    _require_program()
    inputs = ref.make_random_inputs(
        seg_q, seg_k, top_k=top_k, seed=90 + top_k % 7, device="cuda"
    )
    bound = max(inputs.seg_k_len)  # the caller's bound on every key segment
    plan = _plan(inputs, max_seqlen_k=bound)
    runner = _prepare(inputs, max_seqlen_k=bound)
    assert runner.choice == plan and plan.max_seqlen_k == bound
    assert _plan(inputs).max_seqlen_k is None
    # the scan and, on a split call, the plain merge: never an ordering pass
    assert runner.stages == (("scan", "merge") if plan.split else ("scan",))
    assert tuple(launch.role for launch in runner.launches) == runner.stages
    # every run of a prepared runner that launches alerts under torch's deterministic algorithms
    assert runner._alert_entry == RUN_ENTRY
    assert runner.programs == {
        role: program_for(runner.arch, key)
        for role, key in zip(runner.stages, plan.keys, strict=True)
    }
    assert runner.workspace_bytes == plan.workspace_bytes
    expected = _cloned(runner())
    _judge_ok(expected, ref.select_reference(inputs))
    _assert_sets_repeat(runner, expected)
    if seg_q == [64]:
        assert plan.split and plan.n_split >= 2


# flashinfer-ai/flashinfer#7549's ratio-4 pooled selection (top-512 over 4-token pools: every key segment holds
# ``ceil(L / 4)`` pools and the offsets stay in query tokens) at a scale the reference judges in seconds: one 8192-token
# document, packed documents, and rank 0 of a 16-way context-parallel zigzag split of an 8192-token document (its head
# chunk ``[0, 256)`` and its tail chunk ``[7936, 8192)``, which passes its token offset; ``ratio > 1`` makes the default
# offset 0).
POOLED_ROWS = {
    "document": ([8192], [2048], None),
    "packed": ([2048, 4096, 6144], [512, 1024, 1536], None),
    "cp_zigzag": ([256, 256], [64, 2048], [0, 7936]),
}


@pytest.mark.parametrize("name", list(POOLED_ROWS))
def test_ratio4_pooled_top512_rows(name):
    _require_program()
    seg_q, seg_k, offsets = POOLED_ROWS[name]
    inputs = ref.make_random_inputs(
        seg_q,
        seg_k,
        top_k=512,
        ratio=4,
        q_causal_offsets=offsets,
        seed=7604,
        device="cuda",
        label=f"pooled_{name}",
    )
    out = _cloned(_run(inputs))
    verdict = _judge_ok(out, ref.select_reference(inputs))
    assert (
        verdict.max_fp32_error_over_bound <= 1.0
        and verdict.max_fp64_error_over_bound <= 1.0
    )
    _assert_sets_repeat(lambda: _run(inputs), out)


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


def test_max_seqlen_k_bound_changes_the_program_never_the_set():
    """The caller's bound on every key segment tells the dispatch a packed call of equal key segments, which runs the
    scan program without the longest-first segment order where the architecture ships that copy (of the decided
    program); the selected sets are the same with the bound, with a looser bound and without it, and a bound that
    cannot hold every segment is rejected."""
    _require_program()
    inputs = ref.make_random_inputs(
        [300] * 4, [9000] * 4, top_k=2048, seed=37, device="cuda"
    )
    tight = max(inputs.seg_k_len)
    plan_none, plan_tight, plan_loose = (
        _plan(inputs),
        _plan(inputs, max_seqlen_k=tight),
        _plan(inputs, max_seqlen_k=inputs.num_keys),
    )
    assert plan_tight.max_seqlen_k == tight and plan_none.max_seqlen_k is None
    assert plan_loose.max_seqlen_k == inputs.num_keys
    assert not plan_tight.lsf  # num_keys == S x max_seqlen_k: equal key segments
    arch = arch_for(inputs.device)
    copy = plan_tight.scan_key + ":lsf"
    if policy_for(arch).lsf_programs and copy in cake_jit.PROGRAM_KEYS[arch]:
        assert plan_none.lsf and plan_loose.lsf
        assert plan_none.scan_key == plan_loose.scan_key == copy
    else:
        # no copy of the decided program (sm_107a registers those of its l6 ticket programs only)
        assert plan_none == dataclasses.replace(plan_tight, max_seqlen_k=None)
    base = _run(inputs)
    _judge_ok(base, ref.select_reference(inputs))
    assert _same_sets(base, _run(inputs, max_seqlen_k=tight))
    assert _same_sets(base, _run(inputs, max_seqlen_k=inputs.num_keys))
    for bad in (tight - 1, inputs.num_keys + 1):
        with pytest.raises(ValueError, match="max_seqlen_k"):
            _run(inputs, max_seqlen_k=bad)


# a packed peaked call, and a split call: 64 queries over 32768 keys on the largest grid that splits its key ranges
# (``_split_grids``; the device's own grid may or may not split this geometry)
_REPEAT_GEOMETRIES = (
    ([700, 300], [700, 9000], 2048, True, False),
    ([64], [32768], 2048, False, True),
)


@pytest.mark.parametrize("seg_q,seg_k,top_k,peaked,split", _REPEAT_GEOMETRIES)
def test_repeatability_as_sets(seg_q, seg_k, top_k, peaked, split):
    """Five eager calls and five runs of one prepared runner: every row's sorted (id, score bits) pairs and its valid
    count repeat bit for bit, and the count is ``min(top_k, visible)``.  The slot order of a row may change between
    calls, so no call is compared with another slot by slot; the first is judged against the reference.  The split
    geometry runs on a grid asserted (on the host plan) to split its key ranges."""
    _require_program()
    inputs = ref.make_random_inputs(
        seg_q, seg_k, top_k=top_k, seed=40, device="cuda", peaked=peaked
    )
    extra = {}
    if split:
        grid = _split_grids(inputs).get("split")
        assert grid is not None, "no grid splits the key ranges"
        extra = {"grid_ctas": grid}
        assert _plan(inputs, **extra).n_split > 1
    runner = _prepare(inputs, **extra)
    if split:
        assert runner.stages == ("scan", "merge")
    runs = [_cloned(_run(inputs, **extra)) for _ in range(5)]
    runs += [_cloned(runner()) for _ in range(5)]
    first = runs[0]
    _judge_ok(first, ref.select_reference(inputs))
    counts = _valid_counts(first[0])
    assert torch.equal(counts, ref.visible_rows(inputs).clamp(max=top_k))
    pairs = _row_pairs(first)
    for i, out in enumerate(runs[1:], start=1):
        assert torch.equal(_valid_counts(out[0]), counts), i
        assert torch.equal(_row_pairs(out), pairs), i


def test_forced_split_and_unsplit_plans_select_the_same_sets():
    """The key-range split is forced through the persistent grid, which enters the split rule
    (``DispatchPolicy.n_split_for``): for 64 queries over 32768 keys a grid of many CTAs per query unit splits the key
    ranges and a small grid does not.  Both grids are found and asserted on the host plan before anything runs; the
    split runner launches the scan and the plain merge, the unsplit one the scan alone, and both select the reference's
    set per row (as does the eager entry on the split grid and on the device's own grid)."""
    _require_program()
    inputs = ref.make_random_inputs([64], [32768], top_k=2048, seed=120, device="cuda")
    grids = _split_grids(inputs)
    assert set(grids) == {"split", "unsplit"}, grids
    plans = {form: _plan(inputs, grid_ctas=grid) for form, grid in grids.items()}
    assert plans["split"].n_split > 1 and plans["split"].split
    assert plans["unsplit"].n_split == 1 and not plans["unsplit"].split
    runs = {}
    for form, grid in grids.items():
        runner = _prepare(inputs, grid_ctas=grid)
        assert runner.choice == plans[form]
        assert runner.stages == (("scan", "merge") if form == "split" else ("scan",))
        runs[form] = _cloned(runner())
    reference = ref.select_reference(inputs)
    for out in runs.values():
        _judge_ok(out, reference)
    assert _same_sets(runs["split"], runs["unsplit"])
    assert _same_sets(_run(inputs, grid_ctas=grids["split"]), runs["unsplit"])
    assert _same_sets(_run(inputs), runs["unsplit"])


# split calls whose rows include padded ones (visible < top_k): a short second packed segment, and a causal offset
# that leaves every row of the segment fewer keys than top_k
_PADDED_SPLIT_GEOMETRIES = {
    "short_segment": dict(seg_q=[64, 16], seg_k=[32768, 1500], offsets=None),
    "causal_offset": dict(seg_q=[64], seg_k=[32768], offsets=[1000]),
}
_SENTINEL_ID, _SENTINEL_SCORE = 12345678, 7.0


@pytest.mark.parametrize("name", list(_PADDED_SPLIT_GEOMETRIES))
def test_forced_split_writes_every_slot_of_padded_rows(name):
    """On a grid asserted to split the key ranges, a call with padded rows writes every slot of caller-owned outputs
    pre-filled with sentinels: valid slots hold the reference's set, every other slot exactly ``(-1, -inf)``.  Five
    runs of the prepared runner and an eager call, each over fresh sentinels, select the same sets."""
    _require_program()
    geometry = _PADDED_SPLIT_GEOMETRIES[name]
    inputs = ref.make_random_inputs(
        geometry["seg_q"],
        geometry["seg_k"],
        top_k=2048,
        q_causal_offsets=geometry["offsets"],
        seed=121,
        device="cuda",
        label=f"padded_split_{name}",
    )
    grid = _split_grids(inputs).get("split")
    assert grid is not None, "no grid splits the key ranges"
    plan = _plan(inputs, grid_ctas=grid)
    assert plan.n_split > 1 and plan.split
    K = inputs.top_k
    visible = ref.visible_rows(inputs)
    assert bool((visible < K).any()) and bool((visible > 0).any()), visible
    device = inputs.device
    indices = torch.empty((inputs.num_queries, K), dtype=torch.int32, device=device)
    scores = torch.empty((inputs.num_queries, K), dtype=torch.float32, device=device)
    runner = _prepare(inputs, grid_ctas=grid, indices=indices, scores=scores)
    assert runner.stages == ("scan", "merge")

    def fresh(call):
        indices.fill_(_SENTINEL_ID)
        scores.fill_(_SENTINEL_SCORE)
        out = _cloned(call())
        assert not bool((out[0] == _SENTINEL_ID).any()), "an id slot was not written"
        padding = out[0] < 0
        assert bool((out[0][padding] == -1).all()), "padding id other than -1"
        assert bool(torch.isneginf(out[1][padding]).all()), "padding without -inf"
        assert torch.equal(_valid_counts(out[0]), visible.clamp(max=K))
        return out

    first = fresh(runner)
    _judge_ok(first, ref.select_reference(inputs))
    for step in range(4):
        assert _same_sets(fresh(runner), first), step
    eager = fresh(lambda: _run(inputs, grid_ctas=grid, indices=indices, scores=scores))
    assert _same_sets(eager, first)


def test_forced_split_keeps_the_larger_ids_of_a_cut_tie_group():
    """Replicated keys (bit-identical scores) through a forced key-range split: 64 queries over 4681 distinct keys
    replicated 7 times (32767 keys; 2048 is not a multiple of 7, so the cutoff cuts replica groups).  The split call
    is judged against the sequential reference, and every cut group keeps its largest ids, as on the unsplit call."""
    _require_program()
    inputs, group = _duplicated_key_inputs(64, 4681, 7, seed=122)
    grids = _split_grids(inputs)
    assert set(grids) == {"split", "unsplit"}, grids
    assert _plan(inputs, grid_ctas=grids["split"]).n_split > 1
    reference = ref.select_reference(inputs, order="sequential")
    split = _cloned(_run(inputs, grid_ctas=grids["split"]))
    _judge_ok(split, reference)
    assert _assert_cut_groups_keep_the_larger_ids(inputs, split, group) > 0
    unsplit = _run(inputs, grid_ctas=grids["unsplit"])
    assert _same_sets(split, unsplit)


PARTITION_KNOBS = (
    {"grid_ctas": 2},
    {"grid_ctas": 37},
    {"candidate_multiplier": 2},
    {"check_period": 1},
    {"sample_tiles_max": 0},
    {"sample_shift_permille": -500},
    {"far_sample": 1},
)


@pytest.mark.parametrize(
    "knobs", PARTITION_KNOBS, ids=lambda d: "_".join(f"{k}={v}" for k, v in d.items())
)
def test_partition_knobs_do_not_change_the_sets(knobs):
    _require_program()
    inputs = ref.make_random_inputs(
        [1500, 600], [1500, 30000], top_k=2048, seed=41, device="cuda"
    )
    base = _run(inputs)
    _judge_ok(base, ref.select_reference(inputs))
    other = _run(inputs, **knobs)
    assert _same_sets(base, other), f"partition knob {knobs} changed the selected sets"


# The exact cases (cutoff ties, signed zeros, large tie groups, ...) under the partition and sampling knobs: the
# id rule (a tied score keeps the larger id) and the zero rule (+0.0 == -0.0) select the same keys with the same
# score bits however the keys are split across CTAs, buffered, checked or sampled.
EXACT_CASE_KNOBS = PARTITION_KNOBS


@pytest.mark.parametrize("name", list(ref.EXACT_CASES))
def test_exact_cases_are_knob_invariant(name):
    _require_program()
    inputs = ref.make_exact_case(name, "cuda")
    reference = ref.select_reference(inputs, order="sequential")
    base = _run(inputs)
    _judge_ok(base, reference, exact_bits=True)
    for knobs in EXACT_CASE_KNOBS:
        other = _run(inputs, **knobs)
        assert _same_sets(base, other), f"{name}: knob {knobs} changed the sets"


# -- the attention-sink + local-band distribution (flashinfer-ai/flashinfer#7042) ----------------------------------
# Small context-parallel geometries whose long units overflow the candidate buffer and run the sampled first
# threshold (the issue's regime: the nearest key tile of every unit -- the diagonal -- holds a hot band of top-k
# keys), judged like the random rows against the FP64 / FP32 reference under both settings of the far-sample
# switch and compared as sets with the sampled threshold switched off and with the forced re-stream.
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
    _assert_sets_repeat(lambda: _run(inputs, far_sample=far_sample), base)
    nosample = _run(inputs, sample_tiles_max=0)
    assert _same_sets(base, nosample), (
        "sample_tiles_max=0 changed the sets on the sink distribution"
    )
    restream = _run(inputs, sample_shift_permille=-500)
    assert _same_sets(base, restream), (
        "the forced re-stream changed the sets on the sink distribution"
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
    # prints (the unit kind is the architecture's)
    assert plan.n_split == 1 and _longest_unit_samples(inputs, plan), plan
    assert plan.sample_tiles_max == 32 and plan.cand_cap == 16384, plan
    assert plan.check_period == 32, plan
    base = _run(inputs, max_seqlen_k=262144)
    _judge_ok(base, ref.select_reference(inputs))
    nosample = _run(inputs, max_seqlen_k=262144, sample_tiles_max=0)
    assert _same_sets(base, nosample)
    far = _run(inputs, max_seqlen_k=262144, far_sample=1)
    assert _same_sets(base, far)
    unbounded = _run(inputs)
    assert _same_sets(base, unbounded)


# ---------------------------------------------------------------------------
# Operator: robustness, graph capture, workspace
# ---------------------------------------------------------------------------


def test_strided_k_view_matches_contiguous_sets():
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
    assert _same_sets(a, b)
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
    # the non-finite rows, as sets: unique visible ids (in any slot order)
    for t in ref.dirty_rows(inputs).tolist():
        row = indices[t].tolist()
        valid = [j for j in row if j >= 0]
        assert len(valid) == len(set(valid)) and all(
            j < int(visible[t]) for j in valid
        ), t
    # every padding slot of every row is exactly (-1, -inf): no other negative id, no finite or NaN score
    padding = indices < 0
    assert bool((indices[padding] == -1).all())
    assert bool(torch.isneginf(scores[padding]).all())
    # the sets of every row, the non-finite ones included, repeat over five more calls
    _assert_sets_repeat(lambda: _run(inputs), (indices, scores))


def test_prepared_runner_graph_capture_and_replay():
    _require_program()
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=50, device="cuda"
    )
    eager = _run(inputs)
    runner = _prepare(inputs)
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
    assert _same_sets(captured, eager)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert _same_sets(captured, eager)
    fresh = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=51, device="cuda"
    )
    inputs.q.copy_(fresh.q)
    inputs.k.copy_(fresh.k)
    inputs.w.copy_(fresh.w)
    graph.replay()
    torch.cuda.synchronize()
    _assert_ticket_tail_zero(runner)
    _judge_ok(_cloned(captured), ref.select_reference(inputs))
    assert _same_sets(captured, _run(fresh))


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
    expected = _cloned(runner())
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
    assert _same_sets(out, expected)


def _workspace_bytes(inputs: ref.IndexerInputs, device: torch.device) -> int:
    return cake_dsa_indexer_nondet_train_topk_workspace_size(
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
            TICKET_TAIL_BYTES,
        )
    return runner


def test_caller_owned_workspace_shared_by_two_geometries_keeps_every_set():
    """Two prepared runners of different geometries share one caller-owned buffer; the larger geometry's
    candidate region covers the smaller one's ticket tail, so every run of B dirties A's tail.  Alternating
    A, B, A, B: every run selects the sets of the same geometry's exclusive-workspace run and the tail of the
    runner that ran reads 0 afterwards.  The eager entry reuses the same buffer across calls of both geometries."""
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
    assert bytes_b > bytes_a + TICKET_TAIL_BYTES, (bytes_a, bytes_b)
    expected_a = _cloned(_run(inputs_a))  # exclusive, backend-owned workspaces
    expected_b = _cloned(_run(inputs_b))
    _judge_ok(expected_a, ref.select_reference(inputs_a))
    _judge_ok(expected_b, ref.select_reference(inputs_b))
    shared = torch.empty(bytes_b, dtype=torch.uint8, device=device)
    runner_a = _caller_owned_runner(inputs_a, shared)
    runner_b = _caller_owned_runner(inputs_b, shared)
    assert runner_a.workspace_bytes == bytes_a and runner_b.workspace_bytes == bytes_b
    tail_a = runner_a.ticket_tail
    assert (
        tail_a.data_ptr() >= shared.data_ptr()
        and tail_a.data_ptr() + TICKET_TAIL_BYTES
        <= shared.data_ptr() + bytes_b - TICKET_TAIL_BYTES
    )  # A's tail lies inside B's candidate region
    for step, (runner, expected) in enumerate(
        [(runner_a, expected_a), (runner_b, expected_b)] * 2
    ):
        out = runner()
        torch.cuda.synchronize()
        assert _same_sets(out, expected), step
        _assert_ticket_tail_zero(runner)
    for step, (inputs, expected) in enumerate(
        [(inputs_a, expected_a), (inputs_b, expected_b)] * 2
    ):
        out = _run(inputs, workspace_buffer=shared)
        torch.cuda.synchronize()
        assert _same_sets(out, expected), step


def test_caller_owned_workspace_tail_written_between_runs_is_restored_before_the_next_run():
    """The direct hazard: on a caller-owned workspace another kernel writes the ticket word between two
    runs of a cached runner; the next run's scan binding zeroes it first, the run selects the same sets, and the tail
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
        assert _same_sets(out, expected), fill
        _assert_ticket_tail_zero(runner)


def test_eager_entry_zeroes_a_caller_owned_tail_once_per_call(monkeypatch):
    """The eager entry with a caller-owned workspace: the scan binding zeroes the ticket tail right before the scan
    launch and no host-side memset runs (neither in ``prepare`` nor in the run); a tail another kernel left non-zero
    before the call is restored, so every call selects the same sets and leaves the word 0."""
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
        out = _run(inputs, workspace_buffer=workspace)
        torch.cuda.synchronize()
        assert _same_sets(out, expected), fill
        _assert_ticket_tail_zero(runners[-1])
    if runners[-1].ticket_block is not None:
        # a program that reads its TicketWord: the one launch draws from the tail's first bytes, its scan binding
        # zeroes the 8-byte word (not the whole tail), and no host-side memset runs
        runner = runners[-1]
        block_at = runner.workspace_bytes - TICKET_TAIL_BYTES
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


def test_caller_owned_workspace_graph_replay_restores_the_tail():
    """A graph capturing a caller-owned runner carries the scan binding's tail memset node: corrupting the tail
    between replays leaves every replay's sets unchanged and the tail 0 afterwards."""
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
        assert _same_sets(captured, eager), fill
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


def test_static_scan_kind_runner_has_no_ticket_tail_and_keeps_its_sets_on_a_poisoned_workspace():
    """A program of the record's ``static_scan_kinds`` keeps the static enumeration on every call, also where the
    call's units exceed the grid: its runner gets no ticket tail (no memset at prepare or run) and its rows keep their
    sets on a caller-owned buffer poisoned everywhere.  A drawing program on the same architecture keeps its tail.  No
    record names a static kind now (every scan program draws), so this runs only on a record that does."""
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
        choice = plan_dsa_indexer_nondet_topk(
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
            assert _same_sets(out, expected), runner.choice.scan_key
        _assert_ticket_tail_zero(runner)


def test_static_call_skips_the_tail_memset_and_never_touches_the_ticket_word():
    """A call whose shape bounds its work units by the grid (``ticket_static_path``; 4 queries over 22018 keys, a split
    plan) takes the scan's static fast path, which never reads or writes the ticket word.  On a caller-owned workspace
    neither prepare, a run nor the eager entry zeroes its tail: a poisoned word stays as it was and every run keeps its
    sets.  A drawing runner on the same buffer still zeroes its own tail before its run, and a failed launch still
    marks the static runner's tail dirty, so its next run restores it."""
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
    assert runner.stages == ("scan", "merge")  # the split plan: the plain merge
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
        assert _same_sets(out, expected_s), fill
        assert (word == fill).all(), fill  # untouched by the scan
    assert not memsets
    out = _run(inputs_s, workspace_buffer=shared)
    torch.cuda.synchronize()
    # the eager entry: no memset either
    assert _same_sets(out, expected_s) and (word == 1).all()
    drawing = _caller_owned_runner(inputs_d, shared)
    assert not drawing._tail_static
    drawing.ticket_tail.fill_(7)
    out = drawing()
    torch.cuda.synchronize()
    assert _same_sets(out, expected_d)
    _assert_ticket_tail_zero(drawing)  # zeroed by its scan binding, left 0 by its scan
    out = runner()  # the static runner after the drawing one wrote the buffer
    torch.cuda.synchronize()
    assert _same_sets(out, expected_s) and not memsets
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
    assert _same_sets(out, expected_s)


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
    """Inputs whose prepared runner draws from its own ticket block on this device.  Skips where no scan program of
    the architecture reads its TicketWord; fails where some do (sm_107a, ``test_ticket_arg_arches_...``) but none of
    the geometries dispatches one that draws."""
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
    pytest.fail(
        f"{arch} registers TicketWord scan programs but no geometry of _TICKET_GEOMETRIES dispatches one that draws"
    )


def test_ticket_word_runner_owns_its_block_and_ignores_a_poisoned_caller_workspace():
    """A prepared runner whose scan program reads its TicketWord gets a 16-byte block of its own, zeroed once at
    prepare, never the shared placeholder and never the workspace's tail: poisoning the whole caller-owned workspace
    (the tail's ticket word included, and the segment order an :lsf copy writes there before it reads it) between runs
    leaves every run's sets equal to a backend-owned runner's, and the block's word reads 0 after every run."""
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
    tail_at = runner.workspace_bytes - TICKET_TAIL_BYTES
    for step in range(6):
        out = runner()
        torch.cuda.synchronize()
        assert _same_sets(out, expected), step
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
    has the same sets as a backend-owned runner's, which passes the reference judge."""
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
        assert _same_sets(out, expected), step
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
    the binding call whose ``zero_region`` is the block's 8-byte word, once; later runs pass the empty region again."""
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
    out = runner()  # the restoring launch zeroes the word on this stream
    torch.cuda.synchronize()
    assert used == [1] and not runner._tail_dirty
    assert _same_sets(out, expected)
    _assert_ticket_tail_zero(runner)
    out = runner()
    torch.cuda.synchronize()
    assert used == [1] and _same_sets(out, expected)


def test_ticket_word_runners_sharing_a_workspace_keep_their_own_blocks():
    """Two TicketWord runners of different geometries on one caller-owned buffer, run in turn: each draws from its own
    block, so every run selects the sets of the same geometry's backend-owned run."""
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
        assert _same_sets(out, expected), step
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
        out = _cloned(runner())
        again = _cloned(runner())
    torch.cuda.current_stream().wait_stream(stream_b)
    torch.cuda.synchronize()
    assert _same_sets(out, expected) and _same_sets(again, expected)
    _assert_ticket_tail_zero(runner)


@pytest.mark.parametrize("warm", [False, True])
def test_ticket_word_runner_graph_capture_before_and_after_first_use(warm):
    """A TicketWord runner captured before its first run and after it: the graph carries no memset node, and replays
    keep their sets with the caller-owned workspace poisoned between them."""
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
        assert _same_sets(captured, expected), fill
        _assert_ticket_tail_zero(runner)


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
            or r.choice.physical_kind in policy_for(r.arch).static_scan_kinds
        )
        assert (r.ticket_block is not None) == (
            not static and cake_backend.reads_ticket_word(r.programs["scan"], r.arch)
        )
        if r.ticket_block is None:
            assert _scan_ticket_word(r) is cake_backend._ticket_placeholder(
                r.device_index
            )
    assert _same_sets(runner(), expected)
    torch.cuda.synchronize()
    assert not placeholder.view(torch.uint8).any()  # no launch writes it


def test_ticket_word_eager_entry_draws_from_the_tail_word_it_zeroes():
    """The eager entry on either workspace: its one scan launch draws from the tail's first bytes, which its binding
    zeroes (8 bytes) right before the launch; a poisoned caller-owned buffer leaves every call's sets unchanged."""
    inputs = _ticket_word_inputs(87)
    device = torch.device("cuda", torch.cuda.current_device())
    expected = _cloned(_prepare(inputs)())
    assert _same_sets(_cloned(_run(inputs)), expected)  # backend-owned workspace
    workspace = torch.empty(
        _workspace_bytes(inputs, device), dtype=torch.uint8, device=device
    )
    tail_at = _workspace_bytes(inputs, device) - TICKET_TAIL_BYTES
    for fill in (0xFF, 7, 1):
        workspace.fill_(fill)
        out = _run(inputs, workspace_buffer=workspace)
        torch.cuda.synchronize()
        assert _same_sets(out, expected), fill
        assert not workspace[
            tail_at : tail_at + cake_backend.TICKET_WORD_ZERO_BYTES
        ].any(), fill


def test_workspace_bound_is_respected():
    """The bound of the call's geometry suffices: a prepared runner on caller-owned workspace and outputs launches
    without an allocation, writes the caller's outputs, and a buffer one int64 short is refused before anything runs."""
    _require_program()
    inputs = ref.make_random_inputs(
        [900, 300], [900, 7000], top_k=2048, seed=60, device="cuda"
    )
    device = torch.device("cuda", torch.cuda.current_device())
    nbytes = _workspace_bytes(inputs, device)
    plan = _plan(inputs)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert nbytes >= plan.workspace_bytes and plan.grid == sms
    assert nbytes == max(
        plan.workspace_bytes, _plan(inputs, sample_tiles_max=0).workspace_bytes
    )
    assert (
        plan.workspace_bytes
        == (plan.grid_ctas * plan.block_q * plan.cand_cap + plan.staging_entries) * 8
        + TICKET_TAIL_BYTES
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    indices = torch.empty(
        (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=device
    )
    scores = torch.empty(
        (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=device
    )
    runner = _prepare(
        inputs, workspace_buffer=workspace, indices=indices, scores=scores
    )
    assert runner.workspace_bytes == plan.workspace_bytes
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
    short = workspace[: plan.workspace_bytes - 8]
    for call in (
        lambda: _prepare(inputs, workspace_buffer=short),
        lambda: _run(inputs, workspace_buffer=short),
    ):
        with pytest.raises(ValueError, match="workspace_buffer needs"):
            call()


@pytest.mark.parametrize("warn_only", [None, False])
def test_workspace_checks_precede_the_deterministic_alert(warn_only):
    """A caller-owned workspace one int64 short of the plan, or not 8-byte aligned, raises ``ValueError`` from the
    eager entry and from prepare, also under ``torch.use_deterministic_algorithms(True)``: the buffer is checked
    before the eager entry's alert, so the caller sees the argument error rather than PyTorch's ``RuntimeError``."""
    _require_program()
    inputs = ref.make_random_inputs(
        [900, 300], [900, 7000], top_k=2048, seed=61, device="cuda"
    )
    device = torch.device("cuda", torch.cuda.current_device())
    need = _plan(inputs).workspace_bytes
    short = torch.empty(need - 8, dtype=torch.uint8, device=device)
    padded = torch.empty(need + 8, dtype=torch.uint8, device=device)
    misaligned = padded[1 : 1 + need]
    assert misaligned.data_ptr() % 8 and misaligned.numel() == need
    mode = (
        contextlib.nullcontext()
        if warn_only is None
        else _torch_deterministic(warn_only=warn_only)
    )
    with mode:
        for buffer, match in (
            (short, "workspace_buffer needs"),
            (misaligned, "8-byte aligned"),
        ):
            for call in (_run, _prepare):
                with pytest.raises(ValueError, match=match):
                    call(inputs, workspace_buffer=buffer)


def test_empty_rows_and_no_keys_are_all_padding():
    """A call without queries returns ``[0, top_k]`` outputs; a call without keys fills every row with padding.
    Neither launches a program, so neither the eager entry nor a run of its prepared runner (no alert entry) alerts
    under ``torch.use_deterministic_algorithms(True)``."""
    _require_program()
    device = torch.device("cuda")
    q = torch.zeros((0, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    k = torch.zeros((5, HEAD_DIM), dtype=torch.bfloat16, device=device)
    w = torch.zeros((0, NUM_HEADS), dtype=torch.float32, device=device)
    cu = torch.tensor([0, 0], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 5], dtype=torch.int32, device=device)
    indices, scores = cake_backend.cake_dsa_indexer_nondet_train_topk(
        q, k, w, cu, cu_k, top_k=8
    )
    assert tuple(indices.shape) == (0, 8) and tuple(scores.shape) == (0, 8)
    inputs = ref.make_random_inputs([3, 4], [0, 0], top_k=8, seed=1, device="cuda")
    indices, scores = _run(inputs)
    assert bool((indices == -1).all()) and bool(torch.isneginf(scores).all())
    _judge_ok((indices, scores), ref.select_reference(inputs))
    with _torch_deterministic(warn_only=False):
        indices, scores = cake_backend.cake_dsa_indexer_nondet_train_topk(
            q, k, w, cu, cu_k, top_k=8
        )
        assert tuple(indices.shape) == (0, 8)
        indices, scores = _run(inputs)
        runner = _prepare(inputs)
        assert runner.stages == () and runner._alert_entry is None
        indices, scores = runner()
    torch.cuda.synchronize()
    assert bool((indices == -1).all()) and bool(torch.isneginf(scores).all())


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
            plan = plan_dsa_indexer_nondet_topk(1024, 4096, 1, top_k=256, device=dev)
            assert plan.grid == sms
            assert cake_dsa_indexer_nondet_train_topk_workspace_size(
                1024, 4096, 1, top_k=256, device=dev
            ) == max(
                plan.workspace_bytes,
                plan_dsa_indexer_nondet_topk(
                    1024, 4096, 1, top_k=256, device=dev, sample_tiles_max=0
                ).workspace_bytes,
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
    expected = _cloned(_run(inputs))
    boom = mock.Mock(side_effect=AssertionError("device query on the hot path"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    again = _run(inputs)
    captured = _prepare(inputs)()
    torch.cuda.synchronize()
    assert boom.call_count == 0
    assert _same_sets(again, expected) and _same_sets(captured, expected)


@pytest.mark.parametrize("split", [False, True])
def test_eager_call_and_prepared_run_do_not_synchronize_with_the_host(split):
    """Under ``torch.cuda.set_sync_debug_mode("error")`` (a torch operation that synchronizes with the host raises) one
    eager call and one run of a prepared runner, after a warm-up, complete: a drawing call, and a call on a grid that
    splits its key ranges (the plain merge).  Both select the warm-up's sets."""
    _require_program()
    if split:
        inputs = ref.make_random_inputs(
            [64], [32768], top_k=2048, seed=131, device="cuda"
        )
        grid = _split_grids(inputs).get("split")
        assert grid is not None, "no grid splits the key ranges"
        extra = {"grid_ctas": grid}
    else:
        inputs = ref.make_random_inputs(
            [600, 400], [600, 12000], top_k=1024, seed=130, device="cuda"
        )
        extra = {}
    expected = _cloned(_run(inputs, **extra))
    runner = _prepare(inputs, **extra)
    if split:
        assert runner.stages == ("scan", "merge")
    runner()
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = _run(inputs, **extra)
        prepared = runner()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert _same_sets(eager, expected) and _same_sets(prepared, expected)


# ---------------------------------------------------------------------------
# torch.use_deterministic_algorithms: the whole call is the step whose slot order varies, so the eager entry raises
# PyTorch's alert after its argument checks (the caller's workspace included) and before its first allocation or launch,
# and every run of a prepared runner that launches before its first launch (with warn_only=True each warns once and
# runs); preparing a runner, the workspace bound and the replay of a captured graph never alert
# ---------------------------------------------------------------------------


def _device_ops(fn, tmp_path, name: str, *, expect_activity: bool = True) -> list[str]:
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
    if expect_activity:
        assert ops, "the profiler recorded no device activity"
    return [e["cat"] for e in ops]


def _sentinel_storage(inputs: ref.IndexerInputs, device: torch.device):
    """A caller-owned workspace and caller-owned outputs for ``inputs``, filled with sentinels."""
    T, K = inputs.num_queries, inputs.top_k
    workspace = torch.full(
        (_workspace_bytes(inputs, device),), 0x5A, dtype=torch.uint8, device=device
    )
    indices = torch.full((T, K), 7, dtype=torch.int32, device=device)
    scores = torch.full((T, K), 3.0, dtype=torch.float32, device=device)
    return workspace, indices, scores


def _untouched(workspace, indices, scores) -> bool:
    torch.cuda.synchronize()
    return (
        bool((workspace == 0x5A).all())
        and bool((indices == 7).all())
        and bool((scores == 3.0).all())
    )


def _strict_alerts(calls, tmp_path, name: str) -> tuple[list[str], int]:
    """Runs every ``(entry, call)`` of ``calls`` under ``torch.use_deterministic_algorithms(True)`` inside one profiler
    trace, each expected to raise PyTorch's ``RuntimeError`` naming its entry, followed by one known kernel (an in-place
    add on a tensor made before the trace: the positive control that shows the trace records this process's launches);
    returns the device activities of the trace and the number of allocations made.  No message names a deterministic
    package to fall back to."""
    messages = []
    control = torch.zeros(1, dtype=torch.float32, device="cuda")

    def strict():
        with _torch_deterministic(warn_only=False):
            for entry, call in calls:
                with pytest.raises(RuntimeError, match=_alert_match(entry)) as info:
                    call()
                messages.append(str(info.value))
        control.add_(1.0)

    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    ops = _device_ops(strict, tmp_path, name)
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"] - before
    assert float(control) == 1.0
    assert len(messages) == len(calls)
    assert not [m for m in messages if "_det_train" in m]
    return ops, allocated


def test_torch_deterministic_mode_eager_entry_raises_before_the_first_allocation_or_launch(
    tmp_path,
):
    """Under ``torch.use_deterministic_algorithms(True)`` the eager entry, with caller-owned outputs and workspace and
    with its own, raises PyTorch's ``RuntimeError`` naming the entry after its argument checks: no allocation, no
    device activity, the caller's outputs and workspace untouched.  Outside the mode the same call runs."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=110, device="cuda"
    )
    # loads the programs and makes the device's placeholders
    expected = _cloned(_run(inputs))
    workspace, indices, scores = _sentinel_storage(inputs, device)
    ops, allocated = _strict_alerts(
        (
            (
                TOPK_ENTRY,
                lambda: _run(
                    inputs, workspace_buffer=workspace, indices=indices, scores=scores
                ),
            ),
            (TOPK_ENTRY, lambda: _run(inputs)),
        ),
        tmp_path,
        "strict_eager",
    )
    assert ops == ["kernel"], ops  # the positive control's kernel alone
    assert allocated == 0, "the eager entry allocated before its alert"
    assert _untouched(workspace, indices, scores)
    out = _run(inputs, workspace_buffer=workspace, indices=indices, scores=scores)
    assert _same_sets(out, expected)


def test_torch_deterministic_mode_prepares_and_every_run_raises_before_its_first_launch(
    tmp_path,
):
    """Preparing never consults the mode: under ``torch.use_deterministic_algorithms(True)``
    :func:`prepare_dsa_indexer_nondet_topk` returns a runner without an alert and leaves caller-owned outputs and
    workspace untouched.  Every run of a runner that launches raises PyTorch's ``RuntimeError`` naming the runner's
    entry before its first launch: no allocation, no device activity, outputs and workspace untouched, the runner not
    marked dirty.  A runner prepared outside the mode raises the same once the mode is on; both run once it is off."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=113, device="cuda"
    )
    expected = _cloned(_run(inputs))
    workspace, indices, scores = _sentinel_storage(inputs, device)
    with _torch_deterministic(warn_only=False), warnings.catch_warnings():
        warnings.simplefilter("error")
        caller = _prepare(
            inputs, workspace_buffer=workspace, indices=indices, scores=scores
        )
    assert caller._alert_entry == RUN_ENTRY
    assert _untouched(workspace, indices, scores)  # prepare wrote no caller-owned byte
    own = _prepare(inputs)  # outside the mode
    assert own._alert_entry == RUN_ENTRY
    ops, allocated = _strict_alerts(
        ((RUN_ENTRY, caller), (RUN_ENTRY, caller.run), (RUN_ENTRY, own)),
        tmp_path,
        "strict_runs",
    )
    assert ops == ["kernel"], ops  # the positive control's kernel alone
    assert allocated == 0, "a run allocated before its alert"
    assert _untouched(workspace, indices, scores)
    assert not caller._tail_dirty and not own._tail_dirty  # nothing was launched
    for runner in (caller, own):
        out = runner()
        torch.cuda.synchronize()
        assert _same_sets(out, expected)


def _ours(record, entry: str) -> list:
    """The alerts of ``entry`` among the recorded warnings (other operations' warnings are ignored)."""
    return [w for w in record if str(w.message).startswith(f"{entry} {ALERT}")]


def _points_at_this_file(warning) -> bool:
    """The alert points at the test's own call, not at a frame inside the package or flashinfer."""
    return Path(warning.filename).name == Path(__file__).name


def test_torch_deterministic_warn_only_warns_once_per_call_and_runs():
    """With ``warn_only=True`` every eager call warns once under the entry's name (its one-shot runner does not warn
    again) and every run of a prepared runner once under the runner's, each pointing at its caller, and every call
    runs: the selected sets are those outside the mode.  Preparing a runner and the workspace bound never warn."""
    _require_program()
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=111, device="cuda"
    )
    expected = _cloned(_run(inputs))
    nbytes = _workspace_bytes(inputs, device)
    outs = []
    with _torch_deterministic(warn_only=True):
        for _ in range(2):
            with pytest.warns(UserWarning, match=ALERT) as record:
                outs.append(_cloned(_run(inputs)))
            alerts = _ours(record, TOPK_ENTRY)
            assert len(alerts) == 1, [str(w.message) for w in record]
            assert not _ours(record, RUN_ENTRY), [str(w.message) for w in record]
            assert _points_at_this_file(alerts[0]), alerts[0].filename
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            runner = _prepare(inputs)
            assert _workspace_bytes(inputs, device) == nbytes
        for _ in range(2):
            with pytest.warns(UserWarning, match=ALERT) as record:
                outs.append(_cloned(runner()))
            alerts = _ours(record, RUN_ENTRY)
            assert len(alerts) == 1, [str(w.message) for w in record]
            assert not _ours(record, TOPK_ENTRY), [str(w.message) for w in record]
            assert _points_at_this_file(alerts[0]), alerts[0].filename
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        assert _same_sets(out, expected), i
    _judge_ok(outs[-1], ref.select_reference(inputs))


def test_torch_deterministic_mode_graph_replay_is_host_free():
    """A CUDA graph capturing a prepared runner's run holds its launches only: the alert is host code, so a replay
    under ``torch.use_deterministic_algorithms(True)`` neither raises nor warns and selects the eager call's sets,
    while a direct run of the same runner under the mode raises."""
    _require_program()
    inputs = ref.make_random_inputs(
        [600, 400], [600, 12000], top_k=1024, seed=112, device="cuda"
    )
    expected = _cloned(_run(inputs))
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
    with _torch_deterministic(warn_only=False), warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match=_alert_match(RUN_ENTRY)):
            runner()
    _assert_ticket_tail_zero(runner)
    assert _same_sets(captured, expected)


# ---------------------------------------------------------------------------
# Eager entry: nothing launched before the scan
# ---------------------------------------------------------------------------

# a static call on every architecture (4 queries, split plan) and a call that draws tickets from a caller-owned buffer
_EAGER_GEOMETRIES = (([4], [22018], 2048), ([600, 400], [600, 12000], 1024))


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
        return _run(
            self.inputs,
            workspace_buffer=self.workspace,
            indices=self.indices,
            scores=self.scores,
        )


def test_offsets_placeholder_is_one_never_written_tensor_per_device():
    """A call without ``q_causal_offsets`` passes the device's offsets placeholder (one int64 element, ``has_offsets``
    = 0), and every call the device's ``K_scale`` placeholder (one float32 element, read by no program of this
    package): the same tensors for every runner and eager call of the device, never written by any launch."""
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
            assert _same_sets(runner(), expected)
            assert _same_sets(_run(inputs), expected)
        torch.cuda.synchronize()
        _judge_ok(expected, ref.select_reference(inputs))
    assert torch.equal(holder, before)


@pytest.mark.parametrize("geometry", range(len(_EAGER_GEOMETRIES)))
def test_eager_entry_launches_nothing_before_its_scan(monkeypatch, tmp_path, geometry):
    """One eager call issues exactly its runner's launches, the scan first (after its binding's ``zero_region``
    memset where that zeroes bytes) and the plain merge on a split call: no fill, copy or ordering pass of its own."""
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
    assert runner.stages == (("scan", "merge") if runner.choice.split else ("scan",))
    # the eager entry alerted before it allocated: its one-shot runner does not alert again
    assert runner._alert_entry is None
    assert runner.tensors["q_offsets"] is cake_backend._offsets_placeholder(
        device.index
    )
    assert _same_sets((call.indices, call.scores), expected)


@pytest.mark.parametrize("first_use", ["before_capture", "in_capture"])
def test_eager_entry_graph_capture_holds_only_the_runner_launches(
    monkeypatch, tmp_path, first_use
):
    """A CUDA graph capturing one eager call holds exactly the runner's launches, whether the device's offsets
    placeholder was made before the capture or first needed inside it (then the capture's own tensor, not kept);
    replays and later eager calls select the same sets, also after new values are copied into the bound inputs."""
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
    assert _same_sets(captured, expected)
    fresh = ref.make_random_inputs(seg_q, seg_k, top_k=top_k, seed=98, device="cuda")
    inputs.q.copy_(fresh.q)
    inputs.k.copy_(fresh.k)
    inputs.w.copy_(fresh.w)
    graph.replay()
    torch.cuda.synchronize()
    replayed = _cloned(captured)
    assert _same_sets(replayed, _run(fresh))
    assert _same_sets(call(), replayed)
    torch.cuda.synchronize()
    _judge_ok(replayed, ref.select_reference(inputs))
