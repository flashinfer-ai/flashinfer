"""Deterministic case registry and shard plan for the sampling quality fuzzer (gh #3605).

Everything the two test layers parametrize over is declared here, once, with a stable case id.
The registry is a pure function of the module constants -- no RNG, no device, no kernel -- so

* the shard plan is an exact cover of the round and any single case replays alone on one GPU
  (``FLASHINFER_SAMPLING_FUZZ_ONLY_CASE=<id>``),
* the statistical budget's ``K`` -- the number of (config x effective class) comparisons -- is
  declared before any draw happens and pinned by a CPU test against the registry.

Case ids are ``<layer>/<family>/<detail>`` and must stay unique: a duplicate id would silently
drop a case from the shard plan, so the CPU test asserts uniqueness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tests.sampling import sampling_reference as ref
from tests.sampling.sampling_stats import ALPHA, N_TRIALS, half_width

BASE_SEED = int(os.environ.get("FLASHINFER_SAMPLING_FUZZ_SEED", "0"))
ONLY_CASE = os.environ.get("FLASHINFER_SAMPLING_FUZZ_ONLY_CASE", "")
SHARD = os.environ.get("FLASHINFER_SAMPLING_FUZZ_SHARD", "")
# Rows per kernel call in the distribution layer: one call is `rows` independent trials, each
# row on its own Philox subsequence, which is what makes them draws rather than a replay.
TRIAL_ROWS = 4096
# K: the number of (config x effective class) comparisons the round performs.  Declared here,
# checked against the registry by tests/sampling/test_sampling_contract_cpu.py, and recorded
# with the calibration in NOTES.md.  Changing the case list means changing this number.
DECLARED_COMPARISONS = 44

# --------------------------------------------------------------------------------------
# Small dyadic distributions: every value is a sum of powers of two, so the float32 kernel and
# the float64 oracle see bit-identical inputs and filter boundaries are never ambiguous.
# --------------------------------------------------------------------------------------
UNIFORM8 = (0.125,) * 8
BIMODAL8 = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0078125)
SKEW8 = (0.5, 0.1875, 0.125, 0.09375, 0.046875, 0.0234375, 0.01171875, 0.01171875)
SPARSE8 = (0.5, 0.0, 0.25, 0.0, 0.0, 0.125, 0.0, 0.125)
TIE4 = (0.25, 0.25, 0.25, 0.25)
# Rows whose nucleus / relative-threshold sets differ, for the per-row parameter keying cases:
# row 1 has a singleton top-1 set, row 0 keeps everything at top-2.
KEY_ROWS = ((0.5, 0.25, 0.25), (0.25, 0.25, 0.5))
KEY_ROWS_K = (1.0, 2.0)
# One-hot rows: the only token a sampler may return.
ONEHOT_ROWS = ((0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


@dataclass(frozen=True)
class SupportCase:
    """A sampling call whose oracle is a support set plus its renormalized distribution."""

    cid: str
    api: str
    probs: Tuple[Tuple[float, ...], ...]
    spec: ref.FilterSpec = ref.FilterSpec()
    indices: Optional[Tuple[int, ...]] = None
    index_dtype: torch.dtype = torch.int32
    keying: str = ref.KEYING_ROW
    logits: Optional[Tuple[float, ...]] = None
    calls: int = 1
    note: str = ""
    kind: str = field(default="support", init=False)

    @property
    def batch(self) -> int:
        """Output rows per call: the index count when rows are mapped, else the prob rows."""
        return len(self.indices) if self.indices is not None else len(self.probs)


@dataclass(frozen=True)
class FilterCase:
    """A filter-only call (renormalize / mask) whose oracle is a value matrix."""

    cid: str
    api: str
    values: Tuple[Tuple[float, ...], ...]
    spec: ref.FilterSpec
    note: str = ""
    kind: str = field(default="filter", init=False)


@dataclass(frozen=True)
class DistCase:
    """A distribution-layer case: ``N_TRIALS`` independent draws against a declared target."""

    cid: str
    api: str
    probs: Tuple[Tuple[float, ...], ...]
    spec: ref.FilterSpec = ref.FilterSpec()
    note: str = ""
    kind: str = field(default="dist", init=False)


def _support_cases() -> List[SupportCase]:
    # 4096 independent rows for the single-distribution cases: with `indices` each output row
    # runs on its own Philox subsequence, so one call is 4096 independent draws of that row.
    draws = tuple([0] * TRIAL_ROWS)

    def single(cid, api, row, spec, note, indices=draws, calls=1, index_dtype=torch.int32):
        return SupportCase(
            cid, api, (row,), spec, indices=indices, index_dtype=index_dtype, calls=calls, note=note
        )

    one_hot_specs = (
        ("sampling_from_probs", ref.FilterSpec()),
        ("top_k_sampling_from_probs", ref.FilterSpec(top_k=2)),
        ("top_p_sampling_from_probs", ref.FilterSpec(top_p=0.9)),
        ("min_p_sampling_from_probs", ref.FilterSpec(min_p=0.5)),
        ("top_k_top_p_sampling_from_probs", ref.FilterSpec(top_k=2, top_p=0.9)),
        ("sampling_from_logits", ref.FilterSpec()),
        ("top_k_top_p_sampling_from_logits", ref.FilterSpec(top_k=2, top_p=0.9)),
    )
    cases: List[SupportCase] = [
        SupportCase(
            f"det/onehot/{api}",
            api,
            ONEHOT_ROWS,
            spec,
            indices=(0, 1) * 16,
            note="one-hot rows: only the single legal token may come back",
        )
        for api, spec in one_hot_specs
    ]
    # Extreme but legal logits (A2-11): the dominant token wins, and equal logits stay uniform.
    cases += [
        SupportCase(
            "struct/extreme/logits_scale",
            "sampling_from_logits",
            ((1.0, 0.0, 0.0, 0.0),),
            logits=(1e30, -1e30, 0.0, 0.0),
            indices=draws,
            note="extreme but legal logits: the dominant token wins in float32",
        ),
        SupportCase(
            "struct/extreme/logits_equal",
            "sampling_from_logits",
            ((0.25,) * 4,),
            logits=(0.0, 0.0, 0.0, 0.0),
            indices=draws,
            note="all-equal logits: uniform over the whole row",
        ),
    ]
    # Ties and boundaries (A2-04/A2-05/A2-06/A2-11): retention is value based, so ties at a
    # boundary are kept together -- torch.topk's tie order is NOT a documented contract.
    cases += [
        single(
            "struct/tie/equal_probs_k2",
            "top_k_sampling_from_probs",
            TIE4,
            ref.FilterSpec(top_k=2),
            "4-way tie at the top-k boundary: all four tokens are retained",
        ),
        single(
            "struct/tie/bimodal_k2",
            "top_k_sampling_from_probs",
            BIMODAL8,
            ref.FilterSpec(top_k=2),
            "k=2 keeps only the two strictly-larger values",
        ),
        single(
            "struct/boundary/top_p_exact",
            "top_p_sampling_from_probs",
            (0.5, 0.25, 0.25),
            ref.FilterSpec(top_p=0.5),
            "tail mass exactly top_p is excluded: the nucleus is the single top token",
        ),
        single(
            "struct/boundary/top_p_over",
            "top_p_sampling_from_probs",
            (0.5, 0.25, 0.25),
            ref.FilterSpec(top_p=0.75),
            "once the cumulative mass passes top_p the tied boundary tokens are retained",
        ),
        single(
            "struct/boundary/min_p_exact",
            "min_p_sampling_from_probs",
            (0.5, 0.25, 0.25),
            ref.FilterSpec(min_p=0.5),
            "min-p is inclusive: a token exactly at max(p)*min_p stays",
        ),
        single(
            "struct/boundary/min_p_above",
            "min_p_sampling_from_probs",
            (0.5, 0.25, 0.25),
            ref.FilterSpec(min_p=0.51),
            "just above the tie: only the top token survives",
        ),
        single(
            "struct/order/top_k_first",
            "top_k_top_p_sampling_from_probs",
            BIMODAL8,
            ref.FilterSpec(top_k=4, top_p=0.9, order=ref.ORDER_TOP_K_FIRST),
            "top_k_first retains a strict subset of the joint support here",
        ),
        single(
            "struct/order/joint",
            "top_k_top_p_sampling_from_probs",
            BIMODAL8,
            ref.FilterSpec(top_k=4, top_p=0.9, order=ref.ORDER_JOINT),
            "joint keeps the tied tokens top_k_first drops",
        ),
        single(
            "struct/order/logits_top_k_first",
            "top_k_top_p_sampling_from_logits",
            BIMODAL8,
            ref.FilterSpec(top_k=4, top_p=0.9, order=ref.ORDER_TOP_K_FIRST),
            "logits entry point masks, softmaxes and samples in one call",
        ),
    ]
    # Per-row parameters are keyed by the probability row they describe (A2-08).  Each of these
    # runs a swapped indices mapping with a per-row parameter tensor: the singleton support
    # belongs to the source row, so a block-index keying lets tokens from row 1's tie group into
    # an output row that may only emit the top token of row 0.
    cases += [
        SupportCase(
            "struct/keying/top_k_row",
            "top_k_sampling_from_probs",
            KEY_ROWS,
            ref.FilterSpec(top_k=KEY_ROWS_K),
            indices=(1, 0),
            calls=64,
            note="per-row top-k tensor with a swapped indices mapping",
        ),
        SupportCase(
            "struct/keying/min_p_row",
            "min_p_sampling_from_probs",
            KEY_ROWS,
            ref.FilterSpec(min_p=(0.5, 1.0)),
            indices=(1, 0),
            calls=64,
            note="per-row min-p tensor with a swapped indices mapping",
        ),
        SupportCase(
            "struct/keying/top_k_int64",
            "top_k_sampling_from_probs",
            KEY_ROWS,
            ref.FilterSpec(top_k=KEY_ROWS_K),
            indices=(1, 0),
            index_dtype=torch.int64,
            calls=64,
            note="int64 indices must not reinterpret the per-row top-k tensor",
        ),
        SupportCase(
            "struct/keying/row_reuse",
            "top_k_sampling_from_probs",
            KEY_ROWS,
            ref.FilterSpec(top_k=KEY_ROWS_K),
            indices=(1, 0) * 8,
            calls=8,
            note="one probability row reused by many outputs",
        ),
    ]
    return cases


def _dist_cases() -> List[DistCase]:
    return [
        DistCase(
            "dist/uniform/from_probs", "sampling_from_probs", (UNIFORM8,), note="uniform 8"
        ),
        DistCase(
            "dist/uniform/from_logits",
            "sampling_from_logits",
            (UNIFORM8,),
            note="equal logits => uniform",
        ),
        DistCase(
            "dist/bimodal/from_probs", "sampling_from_probs", (BIMODAL8,), note="dyadic bimodal"
        ),
        DistCase(
            "dist/sparse/from_probs",
            "sampling_from_probs",
            (SPARSE8,),
            note="half the row is exactly zero: those classes are unreachable",
        ),
        DistCase(
            "dist/skew/top_k",
            "top_k_sampling_from_probs",
            (SKEW8,),
            ref.FilterSpec(top_k=3),
            note="top-k nucleus renormalized",
        ),
        DistCase(
            "dist/skew/top_p",
            "top_p_sampling_from_probs",
            (SKEW8,),
            ref.FilterSpec(top_p=0.6),
            note="top-p nucleus renormalized",
        ),
        DistCase(
            "dist/skew/min_p",
            "min_p_sampling_from_probs",
            (SKEW8,),
            ref.FilterSpec(min_p=0.1),
            note="min-p retention is inclusive",
        ),
        DistCase(
            "dist/skew/top_k_first",
            "top_k_top_p_sampling_from_probs",
            (SKEW8,),
            ref.FilterSpec(top_k=4, top_p=0.85, order=ref.ORDER_TOP_K_FIRST),
            note="nucleus of the renormalized top-k set (strict subset of joint)",
        ),
        DistCase(
            "dist/skew/joint",
            "top_k_top_p_sampling_from_probs",
            (SKEW8,),
            ref.FilterSpec(top_k=4, top_p=0.85, order=ref.ORDER_JOINT),
            note="intersection of the two filters (keeps the token top_k_first drops)",
        ),
    ]


def _filter_cases() -> List[FilterCase]:
    return [
        FilterCase(
            "struct/renorm/top_k",
            "top_k_renorm_probs",
            (BIMODAL8, SKEW8),
            ref.FilterSpec(top_k=3),
            note="values below the k-th are zero, the rest renormalize",
        ),
        FilterCase(
            "struct/renorm/top_k_tie",
            "top_k_renorm_probs",
            (TIE4,),
            ref.FilterSpec(top_k=2),
            note="ties at the k-th value are all kept",
        ),
        FilterCase(
            "struct/renorm/top_k_all",
            "top_k_renorm_probs",
            (SKEW8,),
            ref.FilterSpec(top_k=8),
            note="k == vocab renormalizes by the row sum",
        ),
        FilterCase(
            "struct/renorm/top_p",
            "top_p_renorm_probs",
            (BIMODAL8, SKEW8),
            ref.FilterSpec(top_p=0.6),
            note="nucleus renormalized",
        ),
        FilterCase(
            "struct/renorm/top_p_exact",
            "top_p_renorm_probs",
            ((0.5, 0.25, 0.25),),
            ref.FilterSpec(top_p=0.5),
            note="boundary nucleus",
        ),
        FilterCase(
            "struct/mask/top_k",
            "top_k_mask_logits",
            (BIMODAL8, SKEW8),
            ref.FilterSpec(top_k=3),
            note="masked entries are exactly -inf",
        ),
        FilterCase(
            "struct/mask/top_k_tie",
            "top_k_mask_logits",
            (TIE4,),
            ref.FilterSpec(top_k=2),
            note="ties at the k-th value stay unmasked",
        ),
    ]


SUPPORT_CASES: Tuple[SupportCase, ...] = tuple(_support_cases())
DIST_CASES: Tuple[DistCase, ...] = tuple(_dist_cases())
FILTER_CASES: Tuple[FilterCase, ...] = tuple(_filter_cases())
ALL_CASES: Tuple[object, ...] = SUPPORT_CASES + FILTER_CASES + DIST_CASES
ALL_IDS: Tuple[str, ...] = tuple(c.cid for c in ALL_CASES)


def validate_only_case(spec: Optional[str] = None) -> None:
    """Reject an unknown replay target at load time instead of collecting nothing.

    Like :func:`parse_shard`, ``None`` reads the module-level ``ONLY_CASE`` at call time.
    """
    raw = ONLY_CASE if spec is None else spec
    if raw and raw not in ALL_IDS:
        raise ValueError(f"FLASHINFER_SAMPLING_FUZZ_ONLY_CASE={raw} is not a declared case id")


validate_only_case()

# Draws reserved per output row by each sampling API (flashinfer/sampling.py passes batch_size
# or batch_size * 32 to get_seed_and_offset).  The fuzzer pins both the value and the resulting
# generator-offset advance, so this table is the RNG-consumption contract.
ROWS_PER_DRAW = {
    "sampling_from_probs": 1,
    "sampling_from_logits": 1,
    "top_k_sampling_from_probs": 32,
    "top_p_sampling_from_probs": 32,
    "min_p_sampling_from_probs": 32,
    "top_k_top_p_sampling_from_probs": 32,
    "top_k_top_p_sampling_from_logits": 32,
}


def dist_logits(case: DistCase) -> torch.Tensor:
    """Logits whose softmax is the case distribution (logits entry points only)."""
    probs = torch.tensor(case.probs, dtype=torch.float64)
    if not bool((probs > 0).all()):
        raise ValueError(f"{case.cid}: logits cases need a strictly positive distribution")
    return torch.log(probs)


def dist_target(case: DistCase) -> Tuple[torch.Tensor, torch.Tensor]:
    """(support, target) for a distribution case, in float64 on the CPU."""
    if case.api in ("sampling_from_logits", "top_k_top_p_sampling_from_logits"):
        return ref.target_distribution(ref.reference_softmax(dist_logits(case)), case.spec)
    return ref.target_distribution(torch.tensor(case.probs, dtype=torch.float64), case.spec)


def round_comparisons() -> Tuple[int, Dict[str, int]]:
    """Declared ``K``: (config x effective class) comparisons over the distribution layer."""
    per_case = {case.cid: ref.effective_classes(dist_target(case)[1]) for case in DIST_CASES}
    return sum(per_case.values()), per_case


def declared_half_width() -> float:
    return half_width(N_TRIALS, round_comparisons()[0], ALPHA)


# --------------------------------------------------------------------------------------
# Sharding: shard i of n owns the cases at indices i, i+n, i+2n, ... of the declared order, so
# the plan is an exact cover, needs no RNG to reproduce, and any case replays on its own.
# --------------------------------------------------------------------------------------
def parse_shard(spec: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """Parse ``<shard>/<count>``; ``None`` means "read the module-level SHARD right now".

    The env hooks are read at call time rather than bound as default arguments: a default is
    evaluated once at import, so ``select()`` would keep using the value the process started
    with and a test (or a driver) that sets ``SHARD`` later would be silently ignored.
    """
    raw = SHARD if spec is None else spec
    if not raw:
        return None
    index, _, total = raw.partition("/")
    shard, count = int(index), int(total)
    if not 0 <= shard < count:
        raise ValueError(f"shard {shard} out of range for {count} shards")
    return shard, count


def shard_ids(shard: int, count: int, ids: Sequence[str] = ALL_IDS) -> List[str]:
    return [cid for i, cid in enumerate(ids) if i % count == shard]


def select(cases: Sequence, ids: Optional[Sequence[str]] = None) -> List:
    """Apply the single-case replay hook, then the shard filter, to a case list.

    A case list that owns none of the selected cases comes back empty, so a shard runs only the
    cases it owns and ``ONLY_CASE`` narrows every layer to that one case.
    """
    chosen = list(cases)
    if ONLY_CASE:
        chosen = [c for c in chosen if c.cid == ONLY_CASE]
    shard = parse_shard()
    if shard is not None:
        owned = set(shard_ids(shard[0], shard[1], list(ids if ids is not None else ALL_IDS)))
        chosen = [c for c in chosen if c.cid in owned]
    return chosen


def replay_command(cid: str) -> str:
    return (
        f"FLASHINFER_SAMPLING_FUZZ_ONLY_CASE={cid} "
        "pytest -q tests/sampling/test_sampling_quality_fuzz.py"
    )
