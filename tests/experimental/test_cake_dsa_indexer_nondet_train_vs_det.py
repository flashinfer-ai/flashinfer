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

# Parity of the experimental Cake non-deterministic DSA indexer top-k package with the deterministic one
# (flashinfer-ai/flashinfer#8946, point 4): the non-deterministic package runs the deterministic package's BF16 scan
# programs and its plain split merge under the same host decisions, without the ordering pass.  So
#
# * its registry is the deterministic registry's BF16 scan keys plus the plain ``merge`` key, every program the
#   deterministic program of the same twenty-hex identity under the ``cake_dsa_indexer_nondet_topk`` prefix, and its
#   dispatch records are the deterministic records restricted to the fields it keeps;
# * every host plan has the deterministic plan's fields (scan program, split, grid, unit, capacity, check period,
#   sampling, segment order, pair, staging and workspace bytes, ticket decisions), the deterministic package's
#   rank-fused merge being the plain merge here;
# * on sm_107a the sampled first threshold's launch scalars (tile count and rank shift) are this package's own
#   (``_OWN_SAMPLING``: its ``sample_class_rule`` and the ``sample_long_rule`` the deterministic record has no field
#   for), so that no #7549 rank input runs slower with sampling on than off (flashinfer-ai/flashinfer#7549).  Whether a
#   call samples, and every other decision, stays the deterministic one;
# * every row on a device is the deterministic row in another slot order: the valid slots in ascending id order
#   (``_canonical``) are bit-identical to the deterministic row, ids, score bits, valid count and tail padding.
#
# The host tests run without a GPU; the device tests need a compute capability 10.0 / 10.3 / 10.7 device with the
# generated programs of both packages registered and skip otherwise.  Both packages are named through
# ``tests.cake.test_helpers.cake_package``, as their own suites do.

import dataclasses
import itertools

import pytest
import torch

from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref
from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_backend")
cake_jit = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_jit")
cake_policy = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_policy")
det_backend = cake_package("cake.cake_dsa_indexer_det_train", "cake_backend")
det_jit = cake_package("cake.cake_dsa_indexer_det_train", "cake_jit")
det_policy = cake_package("cake.cake_dsa_indexer_det_train", "cake_policy")

MERGE_KEY = cake_policy.MERGE_KEY
NONDET_PREFIX = "cake_dsa_indexer_nondet_topk_"
DET_PREFIX = "cake_dsa_indexer_det_topk_"
# the persistent grids (SM counts) of the devices of each architecture
_FROZEN_GRIDS = {"sm_100a": (148,), "sm_103a": (148, 152, 160), "sm_107a": (212,)}
# every plan field this package keeps; ``merge_key`` is compared on its own (the deterministic package may name its
# rank-fused merge where this one runs the plain merge)
_PLAN_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(cake_policy.ProgramChoice)
    if f.name != "merge_key"
)
# sorts after every int32 id: the padding of a row goes behind its valid slots
_PAD_SORT_KEY = 1 << 32

P1_CU_SEQLENS_Q = (0, 1777, 3532, 5655, 7765, 9888, 11998, 14121, 16231)
P1_CU_SEQLENS_K = (0, 1777, 58619, 60742, 128665, 130788, 198711, 200834, 268757)
P2_T, P2_TKV = 16172, 267520


# the policy fields this package sets on its own: the deterministic record has no ``sample_long_rule``, and on sm_107a
# the sampled tile count and rank shift follow this package's class and long-unit rules (#7549)
_OWN_FIELDS = ("sample_long_rule",)
_OWN_SAMPLING = {"sm_107a": ("sample_class_rule", "sample_long_rule")}


def _nondet_name(det_name: str) -> str:
    """The non-deterministic program of a deterministic program: the same twenty-hex identity, the other prefix."""
    assert det_name.startswith(DET_PREFIX), det_name
    return NONDET_PREFIX + det_name[len(DET_PREFIX) :]


def _det_name(name: str) -> str:
    """The deterministic program a program of this package comes from (the inverse of ``_nondet_name``)."""
    assert name.startswith(NONDET_PREFIX), name
    return DET_PREFIX + name[len(NONDET_PREFIX) :]


def _bf16_keys(keys: dict) -> dict:
    """The keys of a deterministic registry that this package ships: the BF16 scan keys (no ``fp8`` part) and the
    plain merge."""
    return {
        key: name
        for key, name in keys.items()
        if (key.startswith("scan:") and "fp8" not in key.split(":")) or key == MERGE_KEY
    }


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0)
        in cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_programs():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    device = torch.device("cuda")
    if not cake_backend.generated_program_available(device):
        pytest.skip("generated non-deterministic programs not registered")
    if not det_backend.generated_program_available(device):
        pytest.skip("generated deterministic programs not registered")


# ---------------------------------------------------------------------------
# Rows as sets
# ---------------------------------------------------------------------------


def _row_pairs(result) -> torch.Tensor:
    """int64 ``[T, K]``: every slot as ``id << 32 | score bits``, each row sorted."""
    ids, scores = result
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((ids.to(torch.int64) << 32) | bits).sort(dim=1).values


def _same_sets(a, b) -> bool:
    """Per row the same valid count and the same (id, score bits) pairs; slot order and padding positions free."""
    if tuple(a[0].shape) != tuple(b[0].shape) or tuple(a[1].shape) != tuple(b[1].shape):
        return False
    return bool(torch.equal((a[0] >= 0).sum(dim=1), (b[0] >= 0).sum(dim=1))) and bool(
        torch.equal(_row_pairs(a), _row_pairs(b))
    )


def _canonical(result) -> tuple[torch.Tensor, torch.Tensor]:
    """``result`` with every row in the deterministic package's slot order: the valid slots by ascending id with their
    scores, then the padding (a stable per-row permutation)."""
    ids, scores = result
    key = torch.where(ids >= 0, ids.to(torch.int64), _PAD_SORT_KEY)
    order = key.sort(dim=1, stable=True).indices
    return ids.gather(1, order), scores.gather(1, order)


def _cloned(out) -> tuple[torch.Tensor, torch.Tensor]:
    return out[0].clone(), out[1].clone()


# ---------------------------------------------------------------------------
# Host: registry
# ---------------------------------------------------------------------------


def test_registry_constants_and_argument_plans_are_the_deterministic_ones():
    """The ABI, numerics, FFI entry, architectures and ticket architectures are the deterministic package's; the scan
    and merge argument plans are its plans, positionally."""
    for name in ("ABI", "NUMERICS", "FFI_ENTRY", "ARCHES", "TICKET_TAIL_ARCHES"):
        assert getattr(cake_jit, name) == getattr(det_jit, name), name
    assert set(cake_jit.STAGES) <= set(det_jit.STAGES)
    for role in cake_jit.STAGES:
        assert cake_jit.ARG_PLANS[role] == det_jit.ARG_PLANS[role], role


def test_program_keys_are_the_deterministic_bf16_scan_keys_and_the_plain_merge():
    """Per architecture the keys are the deterministic registry's BF16 scan keys and ``merge``, each naming the
    deterministic program of the same twenty-hex identity under this package's prefix; every program record has the
    deterministic record's role, architectures and ticket-argument architectures, and reads its TicketWord exactly
    where the deterministic program does."""
    assert set(cake_jit.PROGRAM_KEYS) == set(det_jit.PROGRAM_KEYS)
    shipped = set()
    for arch, det_keys in det_jit.PROGRAM_KEYS.items():
        expected = {
            key: _nondet_name(name) for key, name in _bf16_keys(det_keys).items()
        }
        assert MERGE_KEY in expected, arch
        assert cake_jit.PROGRAM_KEYS[arch] == expected, arch
        shipped |= set(expected.values())
        for key, det_name in _bf16_keys(det_keys).items():
            if key.startswith("scan:"):
                assert cake_backend.reads_ticket_word(
                    _nondet_name(det_name), arch
                ) == det_backend.reads_ticket_word(det_name, arch), (arch, key)
        assert cake_backend.ticket_tail_skips(arch) == det_backend.ticket_tail_skips(
            arch
        ), arch
    assert set(cake_jit.PROGRAMS) == shipped
    for name, record in cake_jit.PROGRAMS.items():
        det_record = det_jit.PROGRAMS[_det_name(name)]
        for field in ("role", "arches"):
            assert record.get(field) == det_record.get(field), (name, field)
        assert record.get("ticket_arg_arches") == det_record.get("ticket_arg_arches"), (
            name
        )
        assert record["library"] == name


def test_policy_records_are_the_deterministic_records_restricted():
    """Every architecture's dispatch record holds exactly the fields this package's policy reads, with the
    deterministic record's values: the same scan, capacity, check-period, split, pair, sampling and ticket rules,
    but the sm_107a sampling rules of this package (``_OWN_SAMPLING``); ``sample_long_rule`` is set on sm_107a
    only."""
    assert set(cake_jit.POLICY) == set(det_jit.POLICY)
    for arch, record in cake_jit.POLICY.items():
        det_record = det_jit.POLICY[arch]
        assert set(record) == set(cake_policy.POLICY_FIELDS), arch
        shared = set(cake_policy.POLICY_FIELDS) - set(_OWN_FIELDS)
        assert shared <= set(det_record), arch
        own = _OWN_SAMPLING.get(arch, ())
        assert (record["sample_long_rule"] is not None) == (
            "sample_long_rule" in own
        ), arch
        expected = {
            field: det_record[field] if field in shared else record[field]
            for field in record
            if field not in own
        }
        assert {f: v for f, v in record.items() if f not in own} == expected, arch


# ---------------------------------------------------------------------------
# Host: plans
# ---------------------------------------------------------------------------


def _assert_same_decisions(arch, plan, det_plan, where) -> None:
    # sm_107a: the sampled tile count and rank shift are this package's own (launch scalars only; ``_OWN_SAMPLING``),
    # whether the call samples is not
    own = ("sample_tiles_max", "sample_shift_permille") if arch in _OWN_SAMPLING else ()
    assert (plan.sample_tiles_max > 0) == (det_plan.sample_tiles_max > 0), where
    for field in _PLAN_FIELDS:
        if field in own:
            continue
        assert getattr(plan, field) == getattr(det_plan, field), (
            where,
            field,
            getattr(plan, field),
            getattr(det_plan, field),
        )
    if plan.merge_key is None:
        assert det_plan.merge_key is None, (where, det_plan.merge_key)
    else:
        # the plain merge; the deterministic package may run its rank-fused merge (``merge:rank:<window>``) there
        assert plan.merge_key == MERGE_KEY, where
        assert det_plan.merge_key == MERGE_KEY or det_plan.merge_key.startswith(
            MERGE_KEY + ":rank:"
        ), (where, det_plan.merge_key)
    assert cake_policy.ticket_static_path(plan) == det_policy.ticket_static_path(
        det_plan
    ), where
    assert cake_policy.ticket_units_bound(plan) == det_policy.ticket_units_bound(
        det_plan
    ), where
    assert cake_backend.program_for(arch, plan.scan_key) == _nondet_name(
        det_backend.program_for(arch, det_plan.scan_key)
    ), where


def _plans(arch, T, Tkv, S, **call):
    plan = cake_backend.plan_dsa_indexer_nondet_topk(T, Tkv, S, arch=arch, **call)
    det_plan = det_backend.plan_dsa_indexer_det_topk(T, Tkv, S, arch=arch, **call)
    return plan, det_plan


def _geometries(tile: int, steps_t: int = 8, steps_m: int = 7):
    """A geometric (queries, mean unit tiles) grid: 1..2^17 queries and 0..2^13 mean key tiles per unit, the recorded
    and the review's query counts, and the top_k values on both sides of the policies' floors."""
    queries = sorted(
        {1, 2, 3, 64, 6018, 16231}
        | {int(round(2 ** (17 * i / steps_t))) for i in range(steps_t + 1)}
    )
    tiles = sorted(
        {0, 1, 2, 64}
        | {int(round(2 ** (13 * i / steps_m))) for i in range(steps_m + 1)}
    )
    for S, ratio in ((1, 1), (1, 2), (8, 1), (64, 1)):
        for T in queries:
            for m in tiles:
                Tkv = max(1, int(round(m * tile * S + T / (2 * ratio))))
                for top_k in (1, 256, 512, 1024, 2048, 4096):
                    yield T, Tkv, S, ratio, top_k


def test_host_plans_are_the_deterministic_plans():
    """On a geometry grid, every SM count of every architecture, both sample modes and with and without the bound of
    equal key segments, the plan of a BF16 call has every field of the deterministic plan of the same call: scan
    program, split, grid, unit, capacity, check period, sampling, segment order, pair, staging and workspace bytes,
    ticket decisions (static path, units bound) and the scan program's identity."""
    for arch in cake_jit.ARCHES:
        tile = cake_backend.policy_for(arch).tile_keys
        for grid in _FROZEN_GRIDS[arch]:
            for T, Tkv, S, ratio, top_k in _geometries(tile):
                bounds = [None] + ([-(-Tkv // S)] if S > 1 else [])
                for bound, sample in itertools.product(bounds, (None, 0)):
                    call = dict(
                        top_k=top_k,
                        ratio=ratio,
                        num_sms=grid,
                        sample_tiles_max=sample,
                        max_seqlen_k=bound,
                    )
                    plan, det_plan = _plans(arch, T, Tkv, S, **call)
                    _assert_same_decisions(
                        arch, plan, det_plan, (arch, grid, T, Tkv, S, call)
                    )


_KNOBS = (
    {"grid_ctas": 2},
    {"grid_ctas": 37},
    {"candidate_multiplier": 2},
    {"check_period": 1},
    {"sample_tiles_max": 0},
    {"sample_shift_permille": -500},
    {"far_sample": 0},
    {"far_sample": 1},
)
_KNOB_GEOMETRIES = (
    (64, 262144, 1, 2048),
    (4, 22018, 1, 2048),
    (4096, 65536, 1, 2048),
    (16231, 268757, 8, 2048),
    (16384, 16384, 1, 4096),
    (8192, 8192, 1, 256),
)


def test_host_plans_under_the_partition_knobs_are_the_deterministic_plans():
    """The partition and sampling knobs (``grid_ctas``, ``candidate_multiplier``, ``check_period``,
    ``sample_tiles_max``, ``sample_shift_permille``, ``far_sample``) enter both packages' plans alike."""
    for arch in cake_jit.ARCHES:
        grid = _FROZEN_GRIDS[arch][0]
        for (T, Tkv, S, top_k), knobs in itertools.product(_KNOB_GEOMETRIES, _KNOBS):
            call = dict(top_k=top_k, num_sms=grid, **knobs)
            plan, det_plan = _plans(arch, T, Tkv, S, **call)
            _assert_same_decisions(arch, plan, det_plan, (arch, T, Tkv, S, knobs))


# ---------------------------------------------------------------------------
# Device: rows
# ---------------------------------------------------------------------------


def _run(backend_entry, inputs: ref.IndexerInputs, **extra):
    out = backend_entry(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        **extra,
    )
    assert isinstance(out, tuple) and len(out) == 2
    return _cloned(out)


def _run_both(inputs: ref.IndexerInputs, **extra):
    """``(non-deterministic, deterministic)`` results of one call (eager entries, the same knobs)."""
    nondet = _run(cake_backend.cake_dsa_indexer_nondet_train_topk, inputs, **extra)
    det = _run(det_backend.cake_dsa_indexer_det_train_topk, inputs, **extra)
    torch.cuda.synchronize()
    return nondet, det


def _device_plans(inputs: ref.IndexerInputs, **extra):
    call = dict(top_k=inputs.top_k, ratio=inputs.ratio, device=inputs.device, **extra)
    plan = cake_backend.plan_dsa_indexer_nondet_topk(
        inputs.num_queries, inputs.num_keys, inputs.num_segments, **call
    )
    det_plan = det_backend.plan_dsa_indexer_det_topk(
        inputs.num_queries, inputs.num_keys, inputs.num_segments, **call
    )
    return plan, det_plan


def _split_grid(inputs: ref.IndexerInputs, split: bool) -> int:
    """The largest persistent grid (from the SM count down to 2) on which the host plan of ``inputs`` splits its key
    ranges (``split``) or does not."""
    sms = torch.cuda.get_device_properties(inputs.device).multi_processor_count
    for grid in range(sms, 1, -1):
        plan, _ = _device_plans(inputs, grid_ctas=grid)
        if (plan.n_split > 1) == split:
            return grid
    pytest.fail(f"no grid {'splits' if split else 'leaves unsplit'} the key ranges")


def _lengths(cu):
    return [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)]


def _scaled(lengths, total):
    """``lengths`` scaled to sum to ``total`` (every length at least 1)."""
    out = [max(1, round(v * total / sum(lengths))) for v in lengths]
    out[-1] += total - sum(out)
    return out


def _random(seg_q, seg_k, top_k, seed, **kw):
    def build():
        inputs = ref.make_random_inputs(
            seg_q, seg_k, top_k=top_k, seed=seed, device="cuda", **kw
        )
        return inputs, {}

    return build


def _forced(seg_q, seg_k, seed, split: bool):
    def build():
        inputs = ref.make_random_inputs(
            seg_q, seg_k, top_k=2048, seed=seed, device="cuda"
        )
        return inputs, {"grid_ctas": _split_grid(inputs, split)}

    return build


def _packed(name):
    return lambda: (ref.make_packed_inputs(name, "cuda"), {})


def _exact(name):
    return lambda: (ref.make_exact_case(name, "cuda"), {})


def _sink(seg_q_len, seg_k_len, top_k, far_sample):
    def build():
        inputs = ref.make_sink_inputs(
            seg_q_len, seg_k_len, top_k=top_k, seed=ref.SINK_SEED, device="cuda"
        )
        return inputs, {"far_sample": far_sample}

    return build


_P1_Q, _P1_K = _lengths(P1_CU_SEQLENS_Q), _lengths(P1_CU_SEQLENS_K)
# name -> build() -> (inputs, knobs).  Random rows (iid, peaked, causal offsets on both sides, ratio 2), the packed
# causality geometries of the reference (offsets, ratio 2, empty segments, singletons, short rows, top_k 1 and 4096),
# forced split and unsplit calls (padded rows included), the ratio-4 pooled top-512 rows of
# flashinfer-ai/flashinfer#7549, top_k 4096, the attention-sink rows of flashinfer-ai/flashinfer#7042 under both
# far-sample settings, the exact semantic cases (cutoff ties, signed zeros, all-equal scores, ...) and the two
# recorded token counts at their full key counts (no reference is computed here).
_PARITY_CASES = {
    "random": _random([1100, 900, 1000], [1100, 20000, 1000], 2048, 1),
    "random_peaked": _random(
        [1100, 900, 1000], [1100, 20000, 1000], 2048, 3, peaked=True
    ),
    "causal_offsets": _random(
        [300, 200, 100], [2000, 900, 4000], 512, 5, q_causal_offsets=[-50, 700, 3900]
    ),
    "ratio2": _random([800, 300], [400, 3000], 256, 6, ratio=2),
    **{f"packed_{name}": _packed(name) for name in ref.PACKED_GEOMETRIES},
    "forced_split": _forced([64], [32768], 120, split=True),
    "forced_unsplit": _forced([64], [32768], 120, split=False),
    "forced_split_padded": _forced([64, 16], [32768, 1500], 121, split=True),
    "pooled_document": _random([8192], [2048], 512, 7604, ratio=4),
    "pooled_packed": _random([2048, 4096, 6144], [512, 1024, 1536], 512, 7604, ratio=4),
    "pooled_cp_zigzag": _random(
        [256, 256], [64, 2048], 512, 7604, ratio=4, q_causal_offsets=[0, 7936]
    ),
    "top_k_4096": _random([16384], [16384], 4096, 7),
    "sink_cp16_16k": _sink([512, 512], [512, 16384], 256, 0),
    "sink_cp16_16k_far": _sink([512, 512], [512, 16384], 256, 1),
    "sink_tail_30848": _sink([512], [30848], 256, 0),
    "sink_tail_30848_far": _sink([512], [30848], 256, 1),
    **{f"exact_{name}": _exact(name) for name in ref.EXACT_CASES},
    "recorded_P1": _random(_P1_Q, _P1_K, 2048, 7571),
    "recorded_P2": _random(_scaled(_P1_Q, P2_T), _scaled(_P1_K, P2_TKV), 2048, 7572),
}


@pytest.mark.parametrize("case", list(_PARITY_CASES))
def test_rows_are_the_deterministic_rows_in_another_slot_order(case):
    """On identical inputs and knobs, both packages plan the same scan program and split, and every row of this
    package, its valid slots put in ascending id order, is bit-identical to the deterministic row: ids, score bits,
    valid count and tail padding."""
    _require_programs()
    inputs, extra = _PARITY_CASES[case]()
    plan, det_plan = _device_plans(inputs, **extra)
    assert (plan.scan_key, plan.n_split) == (det_plan.scan_key, det_plan.n_split)
    if case.startswith("forced_"):
        assert (plan.n_split > 1) == ("unsplit" not in case), plan.n_split
    if case == "forced_split_padded":
        assert bool((ref.visible_rows(inputs) < inputs.top_k).any())
    nondet, det = _run_both(inputs, **extra)
    assert ref.same_bits(_canonical(nondet), det), case


def test_nonfinite_rows_are_the_deterministic_rows():
    """NaN queries, infinite keys and overflowing products (``ref.make_nonfinite_inputs``): the finite segment's rows
    are bit-identical to the deterministic rows in ascending id order, and every row, the non-finite ones included, has
    the deterministic row's (id, score bits) pairs and valid count.  The non-finite rows are compared as sets: the
    deterministic contract for them is unique visible ids in ascending order, with no reference that judges them, so
    this comparison does not depend on where their padding sits."""
    _require_programs()
    inputs = ref.make_nonfinite_inputs("cuda")
    nondet, det = _run_both(inputs)
    clean = ref.clean_rows(inputs)
    canonical = _canonical(nondet)
    assert ref.same_bits(
        (canonical[0][clean], canonical[1][clean]), (det[0][clean], det[1][clean])
    )
    assert _same_sets(nondet, det)
