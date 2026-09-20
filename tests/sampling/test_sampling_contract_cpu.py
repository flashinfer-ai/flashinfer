"""CPU layer of the sampling quality fuzzer (gh #3605): hand-solvable oracle cases, the
declared statistical budget, and the shard plan.

Needs torch (CPU) and the oracle only -- never a GPU, never a kernel.  These are the families a
GPU cannot make more convincing:

* **A2-01 hand-solvable oracle**: distributions small enough that every filter, support and
  renormalized value is written out by hand, so "the oracle is wrong" is caught before any
  kernel result is judged against it.
* **A2-05/A2-06/A2-07/A2-11 semantics**: the equality and tie rules of each filter, and the fact
  that ``top_k_first`` and ``joint`` are different filters needing separate references.
* **A2-13 shard plan**: the partition is an exact cover, ids are unique, a case replays alone.
* **A2-14 statistics**: false-positive rate and detection power of the acceptance rule measured
  by pure reference sampling, plus an injected wrong distribution and a permuted row mapping.
"""

from __future__ import annotations

import pytest
import torch

from tests.sampling import sampling_cases as cases
from tests.sampling import sampling_reference as ref
from tests.sampling import sampling_stats as stats


# --------------------------------------------------------------------------------------
# A2-01: hand-solvable oracle
# --------------------------------------------------------------------------------------
def test_hand_solvable_softmax():
    # The oracle's contract is float64 (see sampling_reference's module docstring), so the hand
    # tables are written in float64 as well: the comparison then holds exactly, at atol=1e-15,
    # instead of trading a dtype mismatch for a looser tolerance.
    logits = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    got = ref.reference_softmax(logits)
    e1, e2 = 2.718281828459045, 7.389056098930650
    expected = torch.tensor(
        [
            [1.0 / (1.0 + e1 + e2), e1 / (1.0 + e1 + e2), e2 / (1.0 + e1 + e2)],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, expected, atol=1e-15, rtol=0)
    torch.testing.assert_close(got.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    # temperature divides the logits, per row or as a scalar -- what the sampling module's
    # softmax does before its online normalizer.
    torch.testing.assert_close(
        ref.reference_softmax(logits, temperature=0.5),
        ref.reference_softmax(logits / 0.5),
        atol=1e-15,
        rtol=0,
    )
    torch.testing.assert_close(
        ref.reference_softmax(logits, temperature=torch.tensor([0.5, 2.0])),
        ref.reference_softmax(logits / torch.tensor([[0.5], [2.0]])),
        atol=1e-15,
        rtol=0,
    )


def test_hand_solvable_top_k_and_top_p():
    """p = (1/2, 1/4, 1/8, 1/8): every answer below is computable without a kernel.

    All expectations are float64: dyadic probabilities are exact in either precision, and the
    oracle (float64) then matches them with no tolerance at all.
    """
    p = torch.tensor([[0.5, 0.25, 0.125, 0.125]], dtype=torch.float64)
    support, target = ref.target_distribution(p, ref.FilterSpec(top_k=2))
    assert support.tolist() == [[True, True, False, False]]
    torch.testing.assert_close(
        target,
        torch.tensor([[2 / 3, 1 / 3, 0.0, 0.0]], dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    # k = 3 reaches the two-way tie at 1/8, and a value-based top-k keeps both.
    support, target = ref.target_distribution(p, ref.FilterSpec(top_k=3))
    assert support.tolist() == [[True, True, True, True]]
    torch.testing.assert_close(target, p, atol=0, rtol=0)
    # top_p = 3/4: the strictly-larger tail of 1/8 is 3/4, which is not < 3/4, so the nucleus
    # stops at the two larger tokens -- the boundary token is excluded when the cumulative mass
    # lands exactly on top_p ...
    support, target = ref.target_distribution(p, ref.FilterSpec(top_p=0.75))
    assert support.tolist() == [[True, True, False, False]]
    torch.testing.assert_close(
        target,
        torch.tensor([[2 / 3, 1 / 3, 0.0, 0.0]], dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    # ... and included once the cumulative mass passes it.
    support, _ = ref.target_distribution(p, ref.FilterSpec(top_p=0.875))
    assert support.tolist() == [[True, True, True, True]]
    support, _ = ref.target_distribution(p, ref.FilterSpec(top_p=1.0))
    assert support.tolist() == [[True, True, True, True]]
    support, target = ref.target_distribution(p, ref.FilterSpec(top_p=0.0))
    assert not bool(support.any()) and float(target.sum()) == 0.0
    # The target depends only on the retained SET, never on how the set was reached.
    k_first, t_first = ref.target_distribution(
        p, ref.FilterSpec(top_k=3, top_p=0.75, order=ref.ORDER_TOP_K_FIRST)
    )
    joint, t_joint = ref.target_distribution(
        p, ref.FilterSpec(top_k=3, top_p=0.75, order=ref.ORDER_JOINT)
    )
    assert k_first.tolist() == joint.tolist()
    torch.testing.assert_close(t_first, t_joint, atol=0, rtol=0)


def test_hand_solvable_min_p_is_inclusive():
    """p = (1/2, 1/4, 1/4): min-p keeps the exact relative threshold, top-p does not."""
    p = torch.tensor([[0.5, 0.25, 0.25]])
    support, _ = ref.target_distribution(p, ref.FilterSpec(min_p=0.5))
    assert support.tolist() == [[True, True, True]], (
        "min_p is inclusive at max(p) * min_p"
    )
    support, _ = ref.target_distribution(p, ref.FilterSpec(min_p=0.500001))
    assert support.tolist() == [[True, False, False]]
    support, _ = ref.target_distribution(p, ref.FilterSpec(top_p=0.5))
    assert support.tolist() == [[True, False, False]], "top-p excludes the equal tail"
    support, target = ref.target_distribution(p, ref.FilterSpec(min_p=2.0))
    assert not bool(support.any()) and float(target.sum()) == 0.0
    # The threshold is relative to each row's own maximum.
    two = torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]])
    support, _ = ref.target_distribution(two, ref.FilterSpec(min_p=1.0))
    assert support.tolist() == [[True, False, False], [False, False, True]]


def test_hand_solvable_top_k_first_vs_joint():
    """A2-07: the two filter orders are different filters with different references."""
    p = torch.tensor([cases.BIMODAL8], dtype=torch.float64)
    k, top_p = 4, 0.9
    k_first, t_first = ref.target_distribution(
        p, ref.FilterSpec(top_k=k, top_p=top_p, order=ref.ORDER_TOP_K_FIRST)
    )
    joint, t_joint = ref.target_distribution(
        p, ref.FilterSpec(top_k=k, top_p=top_p, order=ref.ORDER_JOINT)
    )
    # top_k_first measures the nucleus against the top-k set renormalized to 1 (mass 0.9375),
    # joint measures it against the raw row, whose nucleus reaches one more tied token.
    assert k_first.tolist() == [[True, True, True, False, False, False, False, False]]
    assert joint.tolist() == [[True, True, True, True, False, False, False, False]]
    assert bool((joint & ~k_first).any()), "this case must separate the two orders"
    kept_k = torch.tensor(
        [[0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    kept_joint = torch.tensor(
        [[0.5, 0.25, 0.125, 0.0625, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    torch.testing.assert_close(t_first, kept_k / 0.875, atol=0, rtol=0)
    torch.testing.assert_close(t_joint, kept_joint / 0.9375, atol=0, rtol=0)


def test_hand_solvable_per_row_parameter_keying():
    """A per-row parameter belongs to the probability row it describes, not the output row."""
    two = torch.tensor(cases.KEY_ROWS)
    indices = torch.tensor([1, 0])
    support, _ = ref.target_distribution(
        two, ref.FilterSpec(top_k=cases.KEY_ROWS_K), indices=indices
    )
    assert support.tolist() == [[True, True, True], [True, False, False]]
    # The rejected keying (parameter index == output row) answers differently, which is what
    # makes the GPU keying cases able to detect that defect.
    wrong, _ = ref.target_distribution(
        two,
        ref.FilterSpec(top_k=cases.KEY_ROWS_K),
        indices=indices,
        keying=ref.KEYING_BLOCK,
    )
    assert wrong.tolist() != support.tolist()
    support, _ = ref.target_distribution(
        two, ref.FilterSpec(min_p=(0.5, 1.0)), indices=indices
    )
    assert support.tolist() == [[False, False, True], [True, True, True]]


def test_hand_solvable_inverse_cdf_never_draws_zero_classes():
    target = torch.tensor([[0.5, 0.0, 0.125, 0.375]], dtype=torch.float64)
    draws = ref.inverse_cdf_sample(target, 512, torch.Generator().manual_seed(7))
    assert draws.shape == (1, 512)
    assert set(draws.unique().tolist()) <= {0, 2, 3}


# --------------------------------------------------------------------------------------
# A2-13: shard plan (exact cover, unique ids, single-case replay)
# --------------------------------------------------------------------------------------
def test_case_ids_are_unique_and_labelled():
    assert len(cases.ALL_IDS) == len(set(cases.ALL_IDS))
    for case in cases.ALL_CASES:
        assert case.cid.count("/") == 2, case.cid
        assert case.cid.split("/")[0] in ("det", "struct", "dist"), case.cid


def test_shard_plan_is_an_exact_cover():
    shards = 4
    per_shard = [cases.shard_ids(i, shards) for i in range(shards)]
    flat = [cid for group in per_shard for cid in group]
    assert sorted(flat) == sorted(cases.ALL_IDS), (
        "a shard must not drop or duplicate a case"
    )
    # The plan is a pure function of (shard index, shard count): re-running a shard resumes it.
    assert per_shard == [cases.shard_ids(i, shards) for i in range(shards)]


def test_single_case_replay_selects_exactly_one(monkeypatch):
    """The replay hook and the shard filter are read live: exactly one case, or none.

    A case is selected by ``ONLY_CASE`` and then kept only by the shard that owns it, so a
    resumed or re-run shard sees the same single case and every other shard sees nothing.
    """
    target = cases.SUPPORT_CASES[0]
    owning = next(i for i in range(4) if target.cid in cases.shard_ids(i, 4))
    monkeypatch.setattr(cases, "ONLY_CASE", target.cid)
    monkeypatch.setattr(cases, "SHARD", "")
    assert [c.cid for c in cases.select(cases.SUPPORT_CASES)] == [target.cid], (
        "no shard filter"
    )
    monkeypatch.setattr(cases, "SHARD", f"{owning}/4")
    assert [c.cid for c in cases.select(cases.SUPPORT_CASES)] == [target.cid]
    monkeypatch.setattr(cases, "SHARD", f"{(owning + 1) % 4}/4")
    assert cases.select(cases.SUPPORT_CASES) == [], (
        "a shard must drop the cases it does not own"
    )
    # An unknown case id is rejected at load time, not silently collected as nothing.
    with pytest.raises(ValueError, match="not a declared case id"):
        cases.validate_only_case("no/such/case")
    cases.validate_only_case(cases.ALL_IDS[-1])


def test_declared_budget_matches_the_registry():
    """K is declared in advance, so it must equal the comparison count of the declared cases."""
    k, per_case = cases.round_comparisons()
    assert len(per_case) == len(cases.DIST_CASES)
    assert all(count > 0 for count in per_case.values()), per_case
    assert k == cases.DECLARED_COMPARISONS, (
        "K moved: update DECLARED_COMPARISONS, the calibration record in NOTES.md and the "
        "budget table together with the case list"
    )
    assert cases.declared_half_width() == stats.half_width(stats.N_TRIALS, k)


# --------------------------------------------------------------------------------------
# A2-14: calibration by pure reference sampling
# --------------------------------------------------------------------------------------
REFERENCE_P = (0.28, 0.22, 0.17, 0.12, 0.09, 0.07, 0.05, 0.0)


def test_budget_calibration_controls_false_positives_and_detects_bias(capsys):
    """The round's acceptance rule, calibrated by reference sampling.

    NOTES.md records the same measurement at the round's own N/K; this runs a smaller N so it
    stays a fast CPU check, and asserts the two properties the round relies on: the whole-round
    false-positive rate stays inside the budget, and a bias of two half-widths -- and the class
    permutation a wrong row mapping produces -- is detected.
    """
    k, _ = cases.round_comparisons()
    n_trials = 50_000
    hw = stats.half_width(n_trials, k)
    cal = stats.calibrate(
        REFERENCE_P,
        n_trials=n_trials,
        n_comparisons=k,
        reps=25,
        deltas=(0.0, hw, 2.0 * hw, stats.INJECTED_DELTA),
    )
    print(f"calibration: {cal.as_dict()}")
    # alpha = 0.01 is the whole-round budget; a finite calibration estimates it, so the
    # assertion allows 3x slack on the ESTIMATOR, not on the rule.
    assert cal.fp_rate <= 3 * stats.ALPHA, cal.as_dict()
    assert cal.per_comparison_fp <= 3 * stats.ALPHA / k, cal.as_dict()
    assert cal.power_at(stats.INJECTED_DELTA) >= 0.99, cal.as_dict()
    assert cal.power_at(2.0 * hw) >= 0.5, cal.as_dict()
    assert cal.power_permutation >= 0.99, cal.as_dict()
    # A bias well below the half-width is deliberately NOT claimed to be detectable; that
    # conservatism is what keeps the false-positive budget honest.
    assert cal.detection_floor >= 0.5 * hw, cal.as_dict()


def test_oracle_rejects_wrong_distribution_and_wrong_row_mapping():
    """A2-14: the injected defects the round must catch, judged at the round's own budget."""
    k, _ = cases.round_comparisons()
    n_trials = 200_000
    hw = stats.half_width(n_trials, k)
    rng = torch.Generator().manual_seed(20260919)
    target = torch.tensor([REFERENCE_P], dtype=torch.float64)
    observed = ref.frequencies(
        ref.inverse_cdf_sample(target, n_trials, rng), len(REFERENCE_P)
    )
    assert stats.violations(observed[0].tolist(), REFERENCE_P, hw) == []
    # An exactly-zero class is unreachable: a structural check, not a statistical one.
    assert float(observed[0][-1]) == 0.0
    # A wrong distribution (mass swapped between two classes) is rejected ...
    biased = torch.tensor([stats.swap_mass(REFERENCE_P, stats.INJECTED_DELTA)])
    observed_bias = ref.frequencies(
        ref.inverse_cdf_sample(biased, n_trials, rng), len(REFERENCE_P)
    )
    assert stats.violations(observed_bias[0].tolist(), REFERENCE_P, hw) != []
    # ... and so is a permuted row mapping (the wrong-reference signature of a swapped row).
    assert stats.violations(observed[0].flip(0).tolist(), REFERENCE_P, hw) != []
