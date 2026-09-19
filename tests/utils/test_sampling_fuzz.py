"""Seeded correctness fuzzer for flashinfer's probability-sampling entry points.

Covers ``sampling_from_probs``, ``top_k`` / ``top_p`` / ``top_k_top_p_sampling_from_probs``
(both ``filter_apply_order`` values), and the ``indices``, per-request-threshold and
``seed`` / ``offset`` / ``generator`` paths those APIs expose. The oracle is an
independent float64 PyTorch reference -- it never calls flashinfer and does not mirror
the kernel's rejection loop.

The reference brackets each row instead of pinning it to one answer, because the
documented contract leaves two things to the implementation:

* ``top_k_renorm_probs`` promises "keep the top-k probabilities"; ``test_sampling.py``
  reads that as "at least k, at most everything tied with the k-th largest". So the
  may-draw mask keeps every class at least as likely as the k-th largest, while the
  must-draw mask only requires the classes that every such implementation has to keep.
  Both are computed on the float32 values that are handed to the kernel, compared
  exactly: top-k is a rank test, so no tolerance belongs here.
* The top-p nucleus is a cumulative-mass comparison the kernel performs in float32,
  so the mass threshold -- and only that -- carries a float32 boundary allowance.

Two layers run over that bracket. ``test_sampling_fuzz`` draws a few thousand samples
per seeded case and checks that every draw lands in the may-draw support, that every
class the must-draw support requires with a high enough expected count is drawn, that
the shape / dtype / index range are right, and that a fixed seed/offset (or generator
state) replays exactly. ``test_sampling_distribution`` runs a Pearson chi-square
goodness-of-fit with an effect-size gate over a frozen set of constructed
distributions, to catch a kernel whose support is right but whose probability mass is
wrong; the family-wise false-positive budget is nominal (asymptotic chi-square, split
by Bonferroni), and the machinery is checked against ``torch.multinomial`` first.

The reference, the statistical machinery and the generator are exercised on CPU; only
the tests that call flashinfer need a GPU. A failing case prints a single-seed repro
command.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest
import torch

from tests.test_helpers.fuzz_ledger import Finding, FuzzLedger

_DEFAULT_NUM_CASES = 240
NUM_CASES = int(
    os.environ.get("FLASHINFER_SAMPLING_FUZZ_NUM_CASES", str(_DEFAULT_NUM_CASES))
)
BASE_SEED = int(os.environ.get("FLASHINFER_SAMPLING_FUZZ_SEED", "0"))
# Comma-separated seeds -> run only those cases; the repro command printed on failure
# uses this so a single seed reproduces one case exactly.
_ONLY_SEEDS = os.environ.get("FLASHINFER_SAMPLING_FUZZ_ONLY_SEED", "")

# Only the tests that launch a kernel need CUDA; the reference, the goodness-of-fit
# machinery and the case generator are plain CPU tensor code and run anywhere.
_GPU_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# The sampling entry points document float32 in / int32 out; the renorm helpers also take
# fp16/bf16, but the sampling path is float32, so a wider dtype axis would be out of contract.
_DTYPE = torch.float32
_DEVICE = "cuda:0"
_SAMPLES_PER_CASE = 4096


def _flashinfer():
    """Import lazily so the CPU-only tests in this file collect without CUDA."""
    import flashinfer

    return flashinfer


def _fp32_mass_eps(vocab: int) -> float:
    """Boundary allowance for a float32 cumulative sum over up to ``vocab`` terms.

    float32 has a unit roundoff of 2**-24; a running sum of ``vocab`` terms carries a
    worst-case absolute error of about ``vocab * 2**-24``. Only the top-p mass
    threshold is loosened by this. It is an engineering allowance for these small
    rows, not a proven error bound for the kernel at large vocabularies.
    """
    return vocab * 2.0**-24


# ---------------------------------------------------------------------------
# Independent reference (float64, CPU). Every mask is computed on the exact values the
# kernel sees, so a float32 tie stays a tie: float64 normalization divides a row by one
# positive constant, which preserves both order and equality.
# ---------------------------------------------------------------------------
def reference_normalize(probs: torch.Tensor) -> torch.Tensor:
    """Row-normalize to a distribution in float64."""
    p = probs.detach().to("cpu", torch.float64)
    return p / p.sum(dim=-1, keepdim=True)


def reference_top_k_pivot(probs_row: torch.Tensor, top_k: int) -> torch.Tensor:
    """The k-th largest probability in the row (the top-k threshold value)."""
    vocab = probs_row.numel()
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return torch.sort(probs_row, descending=True).values[min(top_k, vocab) - 1]


def reference_top_k_support(probs_row: torch.Tensor, top_k: int) -> torch.Tensor:
    """Every class at least as likely as the k-th largest (the largest legal support).

    No tolerance: top-k is a rank test over the stored float32 values and the kernel
    compares them exactly.
    """
    return probs_row >= reference_top_k_pivot(probs_row, top_k)


def reference_top_k_bounds(
    probs_row: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(may_keep, must_keep)`` for a top-k filter.

    ``may_keep`` is the tie-inclusive support. ``must_keep`` drops the classes tied
    with the pivot when the implementation is free to choose among them: with ``g``
    classes strictly above the pivot and ``t`` tied with it, any implementation that
    keeps at least ``k`` classes must keep all of ``g`` and ``k - g`` of the ``t``,
    so the tie is only forced when ``t <= k - g``.
    """
    pivot = reference_top_k_pivot(probs_row, top_k)
    above = probs_row > pivot
    may = probs_row >= pivot
    tied = may & ~above
    if int(tied.sum()) <= top_k - int(above.sum()):
        return may, may
    return may, above


def reference_top_p_support(
    probs_row: torch.Tensor, top_p: float, shift: float = 0.0
) -> torch.Tensor:
    """Inclusive top-p nucleus: keep a class iff the mass strictly above it is < top_p.

    Written over the strictly-greater mass so equal-valued classes are kept together.
    ``shift`` loosens (or tightens) the cumulative-mass threshold for the fp32 boundary.
    """
    return _mass_strictly_above(probs_row) < top_p + shift


def _mass_strictly_above(probs_row: torch.Tensor) -> torch.Tensor:
    """For each class, the total mass of the classes strictly more likely than it."""
    return torch.where(
        probs_row.unsqueeze(0) > probs_row.unsqueeze(1),
        probs_row.unsqueeze(0),
        torch.zeros_like(probs_row).unsqueeze(0),
    ).sum(dim=1)


def _renorm_on(probs_row: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    kept = torch.where(keep, probs_row, torch.zeros_like(probs_row))
    return kept / kept.sum()


def reference_top_k_filter(probs_row: torch.Tensor, top_k: int) -> torch.Tensor:
    return _renorm_on(probs_row, reference_top_k_support(probs_row, top_k))


def reference_top_p_filter(probs_row: torch.Tensor, top_p: float) -> torch.Tensor:
    return _renorm_on(probs_row, reference_top_p_support(probs_row, top_p))


def reference_top_k_renorm_sums(
    probs_row: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(smallest, largest)`` mass a legal top-k filter can leave to renormalize by.

    ``top_k_first`` runs top-p on the renormalized row, so which tied classes the
    top-k stage kept moves the top-p boundary. Keeping the fewest legal classes gives
    the largest renormalized masses (the tightest nucleus), keeping all of them the
    smallest (the widest nucleus).
    """
    pivot = reference_top_k_pivot(probs_row, top_k)
    above = probs_row > pivot
    tied = (probs_row >= pivot) & ~above
    n_tied_kept = min(int(tied.sum()), max(top_k - int(above.sum()), 0))
    s_min = probs_row[above].sum() + n_tied_kept * pivot
    s_max = probs_row[probs_row >= pivot].sum()
    return s_min, s_max


def reference_top_k_top_p_filter_joint(
    probs_row: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    keep = reference_top_k_support(probs_row, top_k) & reference_top_p_support(
        probs_row, top_p
    )
    return _renorm_on(probs_row, keep)


def reference_top_k_top_p_filter_top_k_first(
    probs_row: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    renorm_k = reference_top_k_filter(probs_row, top_k)
    return _renorm_on(renorm_k, reference_top_p_support(renorm_k, top_p))


def reference_support_bounds(
    api: str,
    probs_row: torch.Tensor,
    top_k: Optional[int],
    top_p: Optional[float],
    order: str,
    shift: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(may_draw, must_draw)`` for one row.

    ``may_draw`` is a superset of every support the documented contract permits, so a
    draw outside it is a defect. ``must_draw`` is a subset of all of them, so a class
    it contains has to be reachable. They differ exactly where the contract leaves a
    choice: classes tied with the k-th largest probability, and the class that crosses
    the top-p mass threshold within the float32 allowance ``shift``.

    Both are intersected with the positive-mass classes, so a zero-probability class is
    excluded even for a no-op threshold.
    """
    positive = probs_row > 0
    if api == "sampling_from_probs":
        return positive, positive
    if api == "top_k":
        may, must = reference_top_k_bounds(probs_row, top_k)
        return may & positive, must & positive
    if api == "top_p":
        return (
            reference_top_p_support(probs_row, top_p, shift) & positive,
            reference_top_p_support(probs_row, top_p, -shift) & positive,
        )
    k_may, k_must = reference_top_k_bounds(probs_row, top_k)
    if order == "joint":
        return (
            k_may & reference_top_p_support(probs_row, top_p, shift) & positive,
            k_must & reference_top_p_support(probs_row, top_p, -shift) & positive,
        )
    # top_k_first: top-p runs on the row the top-k stage renormalized. Every class the
    # top-k stage can keep is at least as likely as the pivot, so the mass strictly
    # above it is the same whichever tied classes were dropped; only the renormalizing
    # sum moves, and the two sums above bracket it.
    s_min, s_max = reference_top_k_renorm_sums(probs_row, top_k)
    greater = _mass_strictly_above(probs_row)
    return (
        k_may & (greater / s_max < top_p + shift) & positive,
        k_must & (greater / s_min < top_p - shift) & positive,
    )


def reference_target_distribution(
    api: str,
    probs_row: torch.Tensor,
    top_k: Optional[int],
    top_p: Optional[float],
    order: str,
) -> torch.Tensor:
    """Exact post-filter distribution, for rows whose support is unambiguous."""
    if api == "sampling_from_probs":
        return probs_row / probs_row.sum()
    if api == "top_k":
        return reference_top_k_filter(probs_row, top_k)
    if api == "top_p":
        return reference_top_p_filter(probs_row, top_p)
    if order == "joint":
        return reference_top_k_top_p_filter_joint(probs_row, top_k, top_p)
    return reference_top_k_top_p_filter_top_k_first(probs_row, top_k, top_p)


# ---------------------------------------------------------------------------
# Goodness-of-fit machinery. Chi-square survival function via the regularized upper
# incomplete gamma (no SciPy). Low-expected tail bins are merged so the asymptotic
# approximation holds.
# ---------------------------------------------------------------------------
_MIN_EXPECTED = 10.0


@dataclass(frozen=True)
class GofResult:
    statistic: float
    dof: int
    p_value: float
    tvd: float
    max_resid: float


def chi_square_gof(counts: torch.Tensor, target: torch.Tensor) -> GofResult:
    """Pearson goodness-of-fit of ``counts`` against ``target``, tail bins merged.

    The p-value is the nominal asymptotic chi-square tail, not an exact finite-sample
    false-positive rate. Anything that would make the statistic meaningless -- a
    malformed target, non-integer or negative counts, draws on a zero-target class, or
    a merge that leaves no degrees of freedom -- raises instead of returning a number
    a caller would compare against a threshold.
    """
    counts = counts.to("cpu", torch.float64)
    target = target.to("cpu", torch.float64)
    if counts.ndim != 1 or target.ndim != 1:
        raise ValueError("counts and target must be 1-D")
    if counts.shape != target.shape or counts.numel() == 0:
        raise ValueError("counts and target must be nonempty vectors of equal length")
    if not bool(torch.isfinite(counts).all() and torch.isfinite(target).all()):
        raise ValueError("counts and target must be finite")
    if bool((counts < 0).any() or (target < 0).any()):
        raise ValueError("counts and target must be nonnegative")
    if not torch.equal(counts, counts.round()):
        raise ValueError("counts must be integers")
    if abs(target.sum().item() - 1.0) > 1e-10:
        raise ValueError("target must sum to one")
    n = counts.sum()
    if n.item() < _MIN_EXPECTED:
        raise ValueError("not enough draws for the configured minimum expected count")
    support = target > 0
    if bool((counts[~support] > 0).any()):
        raise ValueError("counts include draws on zero-target classes")

    obs, exp = counts[support], target[support] * n
    order = torch.argsort(exp)
    obs, exp = obs[order].tolist(), exp[order].tolist()
    merged_o: List[float] = []
    merged_e: List[float] = []
    acc_o = acc_e = 0.0
    for o_i, e_i in zip(obs, exp, strict=True):
        acc_o += o_i
        acc_e += e_i
        if acc_e >= _MIN_EXPECTED:
            merged_o.append(acc_o)
            merged_e.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0 and merged_e:
        merged_o[-1] += acc_o
        merged_e[-1] += acc_e
    elif acc_e > 0:
        merged_o.append(acc_o)
        merged_e.append(acc_e)
    o = torch.tensor(merged_o, dtype=torch.float64)
    e = torch.tensor(merged_e, dtype=torch.float64)
    assert o.shape == e.shape and o.ndim == 1 and o.numel() >= 1
    dof = o.numel() - 1
    if dof == 0:
        raise ValueError(
            "no degrees of freedom after merging: the target has a single bin with "
            "expected count >= the minimum, so the test carries no information"
        )
    stat = (((o - e) ** 2) / e).sum()
    p_value = torch.special.gammaincc(
        torch.tensor(dof / 2.0, dtype=torch.float64), stat / 2.0
    ).item()
    tvd = 0.5 * (counts / n - target).abs().sum().item()
    max_resid = ((o - e) / e.sqrt()).abs().max().item()
    result = GofResult(stat.item(), dof, p_value, tvd, max_resid)
    assert math.isfinite(result.statistic) and 0.0 <= result.p_value <= 1.0
    assert math.isfinite(result.tvd) and math.isfinite(result.max_resid)
    return result


# ---------------------------------------------------------------------------
# Case model and input construction.
# ---------------------------------------------------------------------------
_VOCAB = [1, 2, 3, 17, 257]
_SHAPES = [
    "sparse",
    "one_hot",
    "bimodal",
    "long_tail",
    "near_uniform",
    "strict_decreasing",
    "duplicate_plateau",
]
_APIS = ["sampling_from_probs", "top_k", "top_p", "top_k_top_p"]


@dataclass
class Cfg:
    seed: int
    api: str
    order: str  # "joint" / "top_k_first" for top_k_top_p, else "-"
    shapes: List[str]  # one per distribution
    vocab: int
    top_k: List[int] = field(default_factory=list)  # per distribution (empty if unused)
    top_p: List[float] = field(default_factory=list)
    per_request: bool = False  # thresholds as (batch,) tensors vs a python scalar
    use_indices: bool = False  # draw through the indices remap
    p_boundary: str = (
        "-"  # "below" / "at" / "above" a cumulative-mass boundary, else "-"
    )

    @property
    def num_dist(self) -> int:
        return len(self.shapes)

    @property
    def label(self) -> str:
        thr = ""
        if self.top_k:
            thr += "_k" + ("pr" if self.per_request else str(self.top_k[0]))
        if self.top_p:
            thr += "_p" + ("pr" if self.per_request else f"{self.top_p[0]:g}")
        mode = "".join(
            t
            for t in (
                self.order if self.order != "-" else "",
                f"b{self.p_boundary}" if self.p_boundary != "-" else "",
                "idx" if self.use_indices else "",
            )
            if t
        )
        mode = f"_{mode}" if mode else ""
        return f"{self.api}_d{self.num_dist}_v{self.vocab}{thr}{mode}_s{self.seed}"


def _row(shape: str, vocab: int, rng: random.Random) -> torch.Tensor:
    """One unnormalized, nonnegative row (float64) with the requested structure."""
    g = torch.Generator().manual_seed(rng.getrandbits(63))
    if shape == "one_hot":
        row = torch.zeros(vocab, dtype=torch.float64)
        row[rng.randrange(vocab)] = 1.0
    elif shape == "sparse":  # a few positive classes, the rest exactly zero
        row = torch.zeros(vocab, dtype=torch.float64)
        for j in torch.randperm(vocab, generator=g)[: max(1, vocab // 4)].tolist():
            row[j] = torch.rand(1, generator=g, dtype=torch.float64).item() + 1e-2
    elif shape == "bimodal":
        row = torch.full((vocab,), 1e-3, dtype=torch.float64)
        for j in torch.randperm(vocab, generator=g)[: max(1, vocab // 8)].tolist():
            row[j] += 1.0
    elif shape == "long_tail":
        row = 1.0 / (torch.arange(vocab, dtype=torch.float64) + 1.0)
    elif shape == "near_uniform":
        row = 1.0 + 0.01 * torch.rand(vocab, generator=g, dtype=torch.float64)
    elif shape == "strict_decreasing":
        row = torch.tensor([0.7**j for j in range(vocab)], dtype=torch.float64)
    elif shape == "duplicate_plateau":  # deliberate boundary ties
        row = torch.rand(vocab, generator=g, dtype=torch.float64) + 0.1
        if vocab >= 3:
            row[: max(2, vocab // 3)] = row[0]
    else:
        raise ValueError(shape)
    return row


def _boundary_top_p(row_norm: torch.Tensor, where: str, rng: random.Random) -> float:
    """A top_p placed just below / at / just above one of the row's cumulative masses."""
    csum = torch.cumsum(torch.sort(row_norm, descending=True).values, 0)
    # interior boundaries only (the last one is 1.0); fall back to a mid value if none.
    interior = csum[:-1]
    if interior.numel() == 0:
        return 0.5
    b = interior[rng.randrange(interior.numel())].item()
    margin = 10.0 * _fp32_mass_eps(row_norm.numel())
    if where == "below":
        return max(1e-3, b - margin)
    if where == "above":
        return min(1.0, b + margin)
    return b  # "at"


def _gen(seed: int) -> Cfg:
    rng = random.Random(seed)
    vocab = rng.choice(_VOCAB)
    api = rng.choice(_APIS)
    num_dist = rng.choice([1, 2, 4]) if vocab > 1 else 1
    shapes = [
        rng.choice([s for s in _SHAPES if not (s == "one_hot" and vocab == 1)])
        for _ in range(num_dist)
    ]
    use_indices = rng.random() < 0.5
    per_request = num_dist > 1 and rng.random() < 0.5
    order = "-"
    top_k: List[int] = []
    top_p: List[float] = []
    p_boundary = "-"
    if api in ("top_k", "top_k_top_p"):
        # k stays in the documented (0, num_classes] range, including vocab == 1.
        choices = sorted({1, min(2, vocab), max(1, vocab - 1), vocab})
        top_k = [rng.choice(choices) for _ in range(num_dist if per_request else 1)]
    if api in ("top_p", "top_k_top_p"):
        # a boundary-placed p exercises the fp32 threshold; otherwise a plain value.
        if api == "top_p" and rng.random() < 0.5:
            p_boundary = rng.choice(["below", "at", "above"])
            row0 = _row(shapes[0], vocab, random.Random(seed))
            # Place the boundary on the row the kernel actually receives.
            row0 = reference_normalize((row0 / row0.sum()).to(_DTYPE))
            top_p = [_boundary_top_p(row0, p_boundary, rng)]
            per_request = False
        else:
            top_p = [
                rng.choice([0.2, 0.5, 0.8, 0.95, 1.0])
                for _ in range(num_dist if per_request else 1)
            ]
    if api == "top_k_top_p":
        order = rng.choice(["joint", "top_k_first"])
    return Cfg(
        seed=seed,
        api=api,
        order=order,
        shapes=shapes,
        vocab=vocab,
        top_k=top_k,
        top_p=top_p,
        per_request=per_request,
        use_indices=use_indices,
        p_boundary=p_boundary,
    )


if _ONLY_SEEDS.strip():
    _CONFIGS = [_gen(int(s)) for s in _ONLY_SEEDS.split(",") if s.strip()]
else:
    _CONFIGS = [_gen(BASE_SEED + i) for i in range(NUM_CASES)]


def _indexed_per_request_top_k(cfg: Cfg) -> bool:
    """The combination gh #5339 gets wrong: per-row top_k reached through indices."""
    return cfg.api == "top_k" and cfg.per_request and cfg.use_indices


# Tracked wrong-answer findings for this API, applied through the shared ledger.
# These cases still run: they are kept to one output per probs row (see
# _draw_samples), which keeps the buggy top_k_arr[output_position] read inside the
# array, so no unpatched build is asked to read out of bounds. Once gh #5340 lands,
# test_per_request_top_k_follows_indices fails with "remove its ledger entry";
# delete the Finding below and both it and the generated cases are asserted normally.
_LEDGER = FuzzLedger(
    "sampling",
    findings=(
        Finding(
            match=_indexed_per_request_top_k,
            reason=(
                "gh #5339: top_k_sampling_from_probs reads a per-request top_k by "
                "output position instead of by the probs row indices selects; fixed "
                "by gh #5340. Probe: test_per_request_top_k_follows_indices."
            ),
        ),
    ),
)

# One output per probs row keeps top_k_arr[output_position] in bounds on a build
# without gh #5340; repeat the launch instead of widening it.
_ROW_LIMITED_LAUNCHES = 64


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------
def _describe(cfg: Cfg) -> str:
    cc = torch.cuda.get_device_capability(0)
    return (
        f"CONFIG {cfg.label}\n"
        f"  api={cfg.api} order={cfg.order} shapes={cfg.shapes} vocab={cfg.vocab}\n"
        f"  top_k={cfg.top_k} top_p={cfg.top_p} per_request={cfg.per_request} "
        f"indices={cfg.use_indices} p_boundary={cfg.p_boundary}\n"
        f"  dtype={_DTYPE} device={torch.cuda.get_device_name(0)} sm={cc[0]}{cc[1]} "
        f"seed={cfg.seed}"
    )


def _repro(cfg: Cfg) -> str:
    return (
        f"REPRO: FLASHINFER_SAMPLING_FUZZ_ONLY_SEED={cfg.seed} "
        "python -m pytest -s tests/utils/test_sampling_fuzz.py::test_sampling_fuzz"
    )


class _CaseFailure(Exception):
    """An invariant this case violated, held so the ledger can be consulted."""


def _fail(cfg: Cfg, why: str) -> None:
    raise _CaseFailure("\n".join([why, _describe(cfg), _repro(cfg)]))


# ---------------------------------------------------------------------------
# Case runner.
# ---------------------------------------------------------------------------
def _build_probs(cfg: Cfg, rng: random.Random) -> torch.Tensor:
    # The sampling entry points require a contiguous probs; we stay inside that domain.
    probs64 = torch.stack([_row(s, cfg.vocab, rng) for s in cfg.shapes])
    probs64 = probs64 / probs64.sum(dim=-1, keepdim=True)
    return probs64.to(_DTYPE).to(_DEVICE)


def _threshold(values: List, which: str, cfg: Cfg):
    """Scalar when one value is configured, else a per-distribution tensor.

    A per-request threshold tensor is indexed by the probs row: the C++ entry point
    requires its length to equal the number of probs rows, and the kernel pairs each
    output with its row's threshold through indices.
    """
    if not cfg.per_request:
        return values[0]
    dt = (
        torch.int64 if which == "k" else _DTYPE
    )  # per-request top_k is int64 (see test_sampling.py)
    return torch.tensor(values, dtype=dt, device=_DEVICE)


def _call(cfg: Cfg, probs, k, p, indices, seed, offset, generator=None):
    sampling = _flashinfer().sampling
    if cfg.api == "sampling_from_probs":
        return sampling.sampling_from_probs(
            probs, indices=indices, seed=seed, offset=offset, generator=generator
        )
    if cfg.api == "top_k":
        return sampling.top_k_sampling_from_probs(
            probs, k, indices=indices, seed=seed, offset=offset, generator=generator
        )
    if cfg.api == "top_p":
        return sampling.top_p_sampling_from_probs(
            probs, p, indices=indices, seed=seed, offset=offset, generator=generator
        )
    return sampling.top_k_top_p_sampling_from_probs(
        probs,
        k,
        p,
        indices=indices,
        filter_apply_order=cfg.order,
        seed=seed,
        offset=offset,
        generator=generator,
    )


@_GPU_ONLY
@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_sampling_fuzz(cfg: Cfg):
    # Consulted before any CUDA work, so a quarantined config could never reach a
    # kernel launch. Nothing is quarantined today; the entry below is a tolerated
    # wrong answer, applied after the checks run.
    _LEDGER.xfail_if_quarantined(cfg)
    finding = _LEDGER.find(cfg)
    try:
        _run_case(cfg)
    except _CaseFailure as exc:
        if finding is not None:
            pytest.xfail(f"[sampling] {cfg.label}: {finding.reason}\n{exc}")
        pytest.fail(str(exc), pytrace=False)
    # A passing case is not reported as an unexpected pass: one generated case is not
    # guaranteed to exercise the tracked defect. test_per_request_top_k_follows_indices
    # is the probe that fails loudly once the entry is stale.


def _run_case(cfg: Cfg) -> None:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    probs = _build_probs(cfg, rng)
    probs64 = reference_normalize(probs)
    nd = cfg.num_dist
    seed = 0xA5A5 + cfg.seed

    # Per-distribution reference. may_draw: a draw may not land outside it.
    # must_draw: every firmly-included class with enough expected draws must appear.
    # floor: a lower bound on each class's probability -- renormalizing on the largest
    # legal support gives the largest denominator, so no boundary or tie choice can
    # make the real probability smaller than this.
    shift = _fp32_mass_eps(cfg.vocab)
    may = torch.zeros(nd, cfg.vocab, dtype=torch.bool)
    must = torch.zeros(nd, cfg.vocab, dtype=torch.bool)
    floor = torch.zeros(nd, cfg.vocab, dtype=torch.float64)
    for d in range(nd):
        k = cfg.top_k[d if cfg.per_request else 0] if cfg.top_k else None
        p = cfg.top_p[d if cfg.per_request else 0] if cfg.top_p else None
        may[d], must[d] = reference_support_bounds(
            cfg.api, probs64[d], k, p, cfg.order, shift
        )
        floor[d] = _renorm_on(probs64[d], may[d])
    may, must, floor = may.to(_DEVICE), must.to(_DEVICE), floor.to(_DEVICE)

    # Draw a few thousand samples so "never outside the support" has real power.
    samples, dist_idx = _draw_samples(cfg, probs, seed)
    if samples.shape != dist_idx.shape:
        _fail(
            cfg,
            f"output shape {tuple(samples.shape)}, expected {tuple(dist_idx.shape)}",
        )
    if samples.dtype != torch.int32:
        _fail(cfg, f"output dtype {samples.dtype}, expected int32")
    if not (torch.all(samples >= 0) and torch.all(samples < cfg.vocab)):
        _fail(cfg, f"sample index out of range [0,{cfg.vocab})")

    ok = may[dist_idx.long(), samples.long()]
    if not bool(ok.all()):
        i = int((~ok).nonzero()[0, 0].item())
        d, tok = int(dist_idx[i].item()), int(samples[i].item())
        _fail(
            cfg,
            f"dist {d}: drew class {tok} (p={probs64[d, tok].item():.3e}) outside the "
            f"support (|support|={int(may[d].sum().item())})",
        )

    # Mirror: a required class with a high enough expected count must appear. The count
    # uses the probability floor above, so a boundary or tie choice cannot inflate it.
    # A floor-expected count of 64 bounds one class's miss probability by exp(-64)
    # under independent draws; at most num_dist * vocab classes are checked.
    n_per_row = torch.bincount(dist_idx.long(), minlength=nd)
    obs = torch.zeros(nd * cfg.vocab, dtype=torch.long, device=_DEVICE)
    obs.scatter_add_(
        0,
        dist_idx.long() * cfg.vocab + samples.long(),
        torch.ones_like(samples, dtype=torch.long),
    )
    obs = obs.view(nd, cfg.vocab)
    expected = floor * n_per_row.unsqueeze(1)
    required = must & (expected >= 64)
    missing = required & (obs == 0)
    if bool(missing.any()):
        d, tok = (int(x) for x in missing.nonzero()[0])
        _fail(
            cfg,
            f"dist {d}: class {tok} is required by the support with at least "
            f"~{expected[d, tok].item():.0f} expected draws but was never drawn "
            f"({int(n_per_row[d].item())} draws for this row)",
        )

    # Same seed/offset replays exactly on the same path.
    replay, _ = _draw_samples(cfg, probs, seed, single_chunk=True)
    first, _ = _draw_samples(cfg, probs, seed, single_chunk=True)
    if not torch.equal(first, replay):
        _fail(cfg, "same seed/offset did not replay identically")

    # Same generator state -> identical output (documented deterministic path).
    if cfg.use_indices:
        _check_generator_replay(cfg, probs)


def _draw_samples(cfg: Cfg, probs, seed, single_chunk: bool = False):
    """Draw samples and return (samples, dist_idx) on device.

    indices path: one launch of _SAMPLES_PER_CASE outputs mapped back to the num_dist
    rows, so out_batch != num_dist -- except for the configs the ledger tracks, which
    stay at one output per row and repeat the launch instead. non-indices path: tile
    the rows so a single launch still draws thousands of samples per row (each output
    has its own block index, so the draws are independent even for repeated rows).
    """
    nd = cfg.num_dist
    if cfg.use_indices:
        # Reversing the rows makes even the one-output-per-row launch a real remap.
        order = torch.arange(nd - 1, -1, -1, device=_DEVICE)
        k = _threshold(cfg.top_k, "k", cfg) if cfg.top_k else None
        p = _threshold(cfg.top_p, "p", cfg) if cfg.top_p else None
        if _indexed_per_request_top_k(cfg):
            launches = 1 if single_chunk else _ROW_LIMITED_LAUNCHES
            idx = order.to(torch.int32)
            chunks = [
                _call(cfg, probs, k, p, idx, seed + i, 0) for i in range(launches)
            ]
            torch.cuda.synchronize()
            return torch.cat(chunks), order.repeat(launches)
        reps = 1 if single_chunk else -(-_SAMPLES_PER_CASE // nd)
        dist_idx = order.repeat(reps)
        s = _call(cfg, probs, k, p, dist_idx.to(torch.int32), seed, 0)
        torch.cuda.synchronize()
        return s, dist_idx
    reps = 1 if single_chunk else -(-_SAMPLES_PER_CASE // nd)
    dist_idx = torch.arange(nd, device=_DEVICE).repeat(reps)
    k = _threshold(cfg.top_k, "k", cfg) if cfg.top_k else None
    p = _threshold(cfg.top_p, "p", cfg) if cfg.top_p else None
    # per-request thresholds ride along the tiled rows; a scalar stays scalar.
    kt = k.repeat(reps) if torch.is_tensor(k) else k
    pt = p.repeat(reps) if torch.is_tensor(p) else p
    s = _call(cfg, probs.repeat(reps, 1), kt, pt, None, seed, 0)
    torch.cuda.synchronize()
    return s, dist_idx


def _check_generator_replay(cfg: Cfg, probs) -> None:
    nd = cfg.num_dist
    dist_idx = torch.arange(nd - 1, -1, -1, device=_DEVICE).to(torch.int32)
    k = _threshold(cfg.top_k, "k", cfg) if cfg.top_k else None
    p = _threshold(cfg.top_p, "p", cfg) if cfg.top_p else None
    g1 = torch.Generator(_DEVICE).manual_seed(1234)
    g2 = torch.Generator(_DEVICE).manual_seed(1234)
    s1 = _call(cfg, probs, k, p, dist_idx, None, None, generator=g1)
    s2 = _call(cfg, probs, k, p, dist_idx, None, None, generator=g2)
    torch.cuda.synchronize()
    if not torch.equal(s1, s2):
        _fail(cfg, "same generator state did not produce identical output")


# ---------------------------------------------------------------------------
# Dedicated probe for the tracked per-request top_k / indices defect.
# ---------------------------------------------------------------------------
# Two identical rows, one output per row, so the buggy read stays in bounds. Row 1
# takes k=4 (everything), row 0 takes k=1 (the argmax only), and indices swaps them.
_PROBE_ROW = [0.5, 0.25, 0.125, 0.125]
_PROBE_K = [1, 4]
_PROBE_INDICES = [1, 0]
_PROBE_LAUNCHES = 256
_PROBE_CFG = Cfg(
    seed=-1,
    api="top_k",
    order="-",
    shapes=["probe", "probe"],
    vocab=len(_PROBE_ROW),
    top_k=list(_PROBE_K),
    per_request=True,
    use_indices=True,
)


def _run_indexed_top_k_probe() -> Optional[str]:
    """Return None when per-request top_k follows indices, else what went wrong.

    Output 0 reads row 1 with k=4 and must reach all four classes; output 1 reads
    row 0 with k=1 and must always be the argmax. Reading k by output position
    swaps the two, so output 1 leaves class 0 with probability 1/2 per launch and
    output 0 never leaves it. Over 256 launches a wrong kernel escapes with
    probability 2**-256, and a correct one is misreported with probability at most
    4 * (7/8)**256 ~ 5e-15.
    """
    sampling = _flashinfer().sampling
    probs = torch.tensor([_PROBE_ROW, _PROBE_ROW], dtype=_DTYPE, device=_DEVICE)
    top_k = torch.tensor(_PROBE_K, dtype=torch.int64, device=_DEVICE)
    indices = torch.tensor(_PROBE_INDICES, dtype=torch.int32, device=_DEVICE)
    seen = torch.zeros(2, len(_PROBE_ROW), dtype=torch.bool, device=_DEVICE)
    lanes = torch.arange(2, device=_DEVICE)
    for launch in range(_PROBE_LAUNCHES):
        out = sampling.top_k_sampling_from_probs(
            probs, top_k, indices=indices, seed=0x5339 + launch, offset=0
        )
        seen[lanes, out.long()] = True
    torch.cuda.synchronize()
    wide, narrow = seen[0].tolist(), seen[1].tolist()
    if not all(wide):
        return f"output 0 (row 1, k=4) reached only {wide}, expected every class"
    if narrow != [True, False, False, False]:
        return f"output 1 (row 0, k=1) reached {narrow}, expected the argmax only"
    return None


@_GPU_ONLY
def test_per_request_top_k_follows_indices():
    """A per-request top_k must be read from the probs row, not the output position.

    While the ledger tracks gh #5339 this is an expected failure; once the kernel is
    fixed the ledger's own xpass rule turns the pass into a loud failure, and the
    maintainer removes the Finding from _LEDGER (the generated
    top_k/per-request/indices cases then stop being tolerated and are asserted like
    every other case).
    """
    finding = _LEDGER.find(_PROBE_CFG)
    detail = _run_indexed_top_k_probe()
    if finding is not None:
        if detail is not None:
            pytest.xfail(f"[sampling] {finding.reason}\n{detail}")
        _LEDGER.flag_xpass(finding, "per-request top_k through indices")
    assert detail is None, detail


# ---------------------------------------------------------------------------
# Bounded distributional check. Frozen configs over strictly-decreasing, well-separated
# distributions (no small target bins). A config fails only when the p-value is past the
# nominal Bonferroni threshold and an effect size is past a declared floor.
# ---------------------------------------------------------------------------
ALPHA_FAMILY = 1e-5
SAMPLE_COUNT = 1 << 15
TVD_FLOOR = 0.02
RESID_FLOOR = 6.0


def _geometric(vocab: int, active: int, ratio: float) -> torch.Tensor:
    row = torch.zeros(vocab, dtype=torch.float64)
    row[:active] = torch.tensor([ratio**j for j in range(active)], dtype=torch.float64)
    return row / row.sum()


# (api, order, top_k, top_p, vocab, active, ratio). Thresholds fall strictly between
# cumulative masses so the target support is unambiguous; the last four are no-op
# thresholds whose target is the unfiltered distribution.
_STAT_CONFIGS = [
    ("sampling_from_probs", "-", None, None, 4, 4, 0.6),
    ("sampling_from_probs", "-", None, None, 8, 8, 0.7),
    ("top_k", "-", 3, None, 16, 8, 0.6),
    ("top_k", "-", 5, None, 32, 10, 0.7),
    ("top_p", "-", None, 0.8, 12, 8, 0.55),
    ("top_p", "-", None, 0.9, 20, 10, 0.6),
    ("top_k_top_p", "joint", 6, 0.85, 24, 12, 0.6),
    ("top_k_top_p", "top_k_first", 4, 0.8, 16, 9, 0.6),
    ("top_k", "-", 8, None, 8, 8, 0.7),  # top_k == vocab, no-op
    ("top_p", "-", None, 1.0, 8, 8, 0.7),  # top_p == 1, no-op
    ("top_k_top_p", "joint", 8, 1.0, 8, 8, 0.7),  # both no-op
    ("top_k_top_p", "top_k_first", 8, 1.0, 8, 8, 0.7),  # both no-op
]
_ALPHA_PER = ALPHA_FAMILY / len(_STAT_CONFIGS)


def _stat_row(vocab: int, active: int, ratio: float) -> torch.Tensor:
    """The float32 row the kernel is handed, so the reference sees the same values."""
    return _geometric(vocab, active, ratio).to(_DTYPE)


def _draw_counts(api, order, top_k, top_p, probs_row, seed) -> torch.Tensor:
    sampling = _flashinfer().sampling
    vocab = probs_row.numel()
    probs = probs_row.to(_DTYPE).to(_DEVICE).unsqueeze(0)
    indices = torch.zeros(SAMPLE_COUNT, dtype=torch.int32, device=_DEVICE)
    if api == "sampling_from_probs":
        s = sampling.sampling_from_probs(probs, indices=indices, seed=seed, offset=0)
    elif api == "top_k":
        s = sampling.top_k_sampling_from_probs(
            probs, top_k, indices=indices, seed=seed, offset=0
        )
    elif api == "top_p":
        s = sampling.top_p_sampling_from_probs(
            probs, top_p, indices=indices, seed=seed, offset=0
        )
    else:
        s = sampling.top_k_top_p_sampling_from_probs(
            probs,
            top_k,
            top_p,
            indices=indices,
            filter_apply_order=order,
            seed=seed,
            offset=0,
        )
    torch.cuda.synchronize()
    return torch.bincount(s.long(), minlength=vocab).cpu()


@_GPU_ONLY
@pytest.mark.parametrize(
    "sc",
    _STAT_CONFIGS,
    ids=[f"{c[0]}_{c[1]}_v{c[4]}_k{c[2]}_p{c[3]}" for c in _STAT_CONFIGS],
)
def test_sampling_distribution(sc):
    api, order, top_k, top_p, vocab, active, ratio = sc
    probs_row = _stat_row(vocab, active, ratio)
    target = reference_target_distribution(
        api, reference_normalize(probs_row), top_k, top_p, order
    )

    counts = _draw_counts(api, order, top_k, top_p, probs_row, seed=0xBEEF)
    drew_forbidden = int(counts[target == 0].sum().item())
    assert drew_forbidden == 0, (
        f"{api}/{order} v{vocab}: {drew_forbidden} draws on zero-target classes\n"
        f"  counts={counts.tolist()} target={target.tolist()}"
    )

    res = chi_square_gof(counts, target)
    significant = res.p_value < _ALPHA_PER
    large_effect = res.tvd > TVD_FLOOR or res.max_resid > RESID_FLOOR
    assert not (significant and large_effect), (
        f"distribution mismatch for {api}/{order} v{vocab}:\n"
        f"  chi2={res.statistic:.2f} dof={res.dof} p={res.p_value:.3e} "
        f"(alpha_per={_ALPHA_PER:.2e}) tvd={res.tvd:.4f} max_resid={res.max_resid:.2f}\n"
        f"  target={[round(x, 4) for x in target.tolist()]}\n"
        f"  freq={[round(x, 4) for x in (counts / counts.sum()).tolist()]}"
    )


def test_statistical_machinery():
    """Sanity-check the gof gate: it must not fire on correct data and must fire on a bias
    above the declared detection floor. Both halves are deterministic under a fixed
    generator; this is an observation at these settings, not a proof of the nominal
    rate. The multi-seed calibration lives in the offline sweep, not here."""
    targets = [
        _geometric(4, 4, 0.6),
        _geometric(8, 8, 0.7),
        _geometric(10, 10, 0.6),
        _geometric(6, 6, 0.55),
    ]
    gen = torch.Generator().manual_seed(20240919)

    false_positives = 0
    for t in range(400):
        target = targets[t % len(targets)]
        counts = torch.bincount(
            torch.multinomial(target, SAMPLE_COUNT, replacement=True, generator=gen),
            minlength=target.numel(),
        )
        res = chi_square_gof(counts, target)
        if res.p_value < _ALPHA_PER and (
            res.tvd > TVD_FLOOR or res.max_resid > RESID_FLOOR
        ):
            false_positives += 1
    assert false_positives == 0, f"gate fired on correct data: {false_positives}/400"

    # A bias with TVD ~0.03 (above TVD_FLOOR) must trip the gate every time.
    target = _geometric(8, 8, 0.7)
    biased = target.clone()
    biased[0] += 0.03
    biased[-1] = max(biased[-1].item() - 0.03, 1e-6)
    biased /= biased.sum()
    caught = 0
    for _ in range(50):
        counts = torch.bincount(
            torch.multinomial(biased, SAMPLE_COUNT, replacement=True, generator=gen),
            minlength=target.numel(),
        )
        res = chi_square_gof(counts, target)
        if res.p_value < _ALPHA_PER and (
            res.tvd > TVD_FLOOR or res.max_resid > RESID_FLOOR
        ):
            caught += 1
    assert caught == 50, f"gate missed an above-floor bias: {caught}/50"


def test_reference_self_consistency():
    """The reference's own algebra, against hand-worked cases (no GPU)."""
    row = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05], dtype=torch.float64)
    assert torch.allclose(reference_top_k_filter(row, 5), row)
    assert reference_top_p_support(row, 1.0).all()
    assert reference_top_k_support(row, 1).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    # p=0.75 keeps {0.4, 0.3, 0.15} (cumulative 0.85 crosses), excludes the tail.
    assert reference_top_p_support(row, 0.75).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    joint = reference_top_k_top_p_filter_joint(row, 2, 0.9)
    assert (joint > 0).tolist() == [True, True, False, False, False]
    # Boundary ties may be kept together, and the choice is the implementation's.
    tied = torch.tensor([0.3, 0.3, 0.3, 0.1], dtype=torch.float64)
    may, must = reference_top_k_bounds(tied, 1)
    assert may.tolist() == [True, True, True, False]
    assert must.tolist() == [False, False, False, False]
    # Two tied classes and room for exactly two: both are then required.
    may, must = reference_top_k_bounds(
        torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64), 3
    )
    assert may.tolist() == must.tolist() == [True, True, True]


@pytest.mark.parametrize(
    "api,order",
    [("top_k", "-"), ("top_k_top_p", "joint"), ("top_k_top_p", "top_k_first")],
)
@pytest.mark.parametrize(
    "row,k,may,must",
    [
        # A unique k-th class is required, however small the gap above it.
        ([0.5, 0.25, 0.125, 0.125], 2, [1, 1, 0, 0], [1, 1, 0, 0]),
        # Two classes tied at the pivot with room for one: either may be kept.
        ([0.5, 0.25, 0.125, 0.125], 3, [1, 1, 1, 1], [1, 1, 0, 0]),
        # ... and with room for both, both are required.
        ([0.5, 0.25, 0.125, 0.125], 4, [1, 1, 1, 1], [1, 1, 1, 1]),
    ],
)
def test_top_k_boundary_is_exact(api, order, row, k, may, must):
    """A sampler that drops the unique k-th class must be caught.

    Loosening the top-k pivot by the float32 mass allowance would leave the second
    case's pivot class out of the must-draw mask, and the first case's too, so a
    kernel that kept only k-1 classes would pass. top-k compares stored values
    exactly, so the pivot moves only for a real tie.
    """
    p = reference_normalize(torch.tensor(row, dtype=_DTYPE))
    shift = _fp32_mass_eps(len(row))
    got_may, got_must = reference_support_bounds(api, p, k, 1.0, order, shift)
    assert [int(x) for x in got_may.tolist()] == may
    assert [int(x) for x in got_must.tolist()] == must


def test_dropping_the_unique_kth_class_is_detected():
    """The must-draw mask plus the expected-count floor flags an under-including kernel."""
    row = reference_normalize(torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=_DTYPE))
    shift = _fp32_mass_eps(row.numel())
    may, must = reference_support_bounds("top_k", row, 2, None, "-", shift)
    floor = _renorm_on(row, may)
    draws = 4096
    # A kernel that keeps only the argmax: class 1 is required and never drawn.
    observed = torch.tensor([draws, 0, 0, 0])
    missing = must & (floor * draws >= 64) & (observed == 0)
    assert missing.tolist() == [False, True, False, False]


def test_top_k_first_bounds_track_the_renormalized_row():
    """top_k_first runs top-p on the renormalized row, so the tie choice moves it."""
    # k=3 over [0.4, 0.3, 0.15, 0.15] keeps 0.4 and 0.3 plus one of the two 0.15s
    # (mass 0.85) or both (mass 1.0). Class 1 then carries 0.4/1.0 or 0.4/0.85 of the
    # mass above it, which straddles p=0.45: it is allowed but not required, purely
    # because of which tied class the top-k stage kept.
    row = reference_normalize(torch.tensor([0.4, 0.3, 0.15, 0.15], dtype=_DTYPE))
    may, must = reference_support_bounds(
        "top_k_top_p", row, 3, 0.45, "top_k_first", _fp32_mass_eps(row.numel())
    )
    assert may.tolist() == [True, True, False, False]
    assert must.tolist() == [True, False, False, False]


def test_frozen_distribution_targets_are_unambiguous():
    """Every frozen config has one legal support, so its exact target is well defined."""
    for api, order, top_k, top_p, vocab, active, ratio in _STAT_CONFIGS:
        row = reference_normalize(_stat_row(vocab, active, ratio))
        may, must = reference_support_bounds(
            api, row, top_k, top_p, order, _fp32_mass_eps(vocab)
        )
        assert torch.equal(may, must), (
            f"{api}/{order} v{vocab} has an ambiguous support"
        )
        target = reference_target_distribution(api, row, top_k, top_p, order)
        assert torch.equal(target > 0, may), f"{api}/{order} v{vocab} target/support"
        # The chi-square needs degrees of freedom at these sample counts.
        counts = (target * SAMPLE_COUNT).round().to(torch.int64)
        counts[target.argmax()] += SAMPLE_COUNT - int(counts.sum())
        assert chi_square_gof(counts, target).dof >= 1


def test_generated_thresholds_are_in_domain():
    """Generation stays in the API's documented k/p domain, including vocab == 1."""
    for seed in range(_DEFAULT_NUM_CASES):
        cfg = _gen(seed)
        assert all(1 <= k <= cfg.vocab for k in cfg.top_k)
        assert all(0 < p <= 1 for p in cfg.top_p)
        assert len(cfg.top_k) in (0, 1, cfg.num_dist)
        assert len(cfg.top_p) in (0, 1, cfg.num_dist)


def test_indexed_per_request_top_k_is_generated_and_kept_in_bounds():
    """The combination gh #5339 gets wrong is generated rather than silently skipped."""
    tracked = [
        c
        for c in (_gen(i) for i in range(_DEFAULT_NUM_CASES))
        if _indexed_per_request_top_k(c)
    ]
    assert tracked, "no generated case covers per-request top_k with indices"
    # These cases and the probe hand the kernel one output per probs row, so the read
    # gh #5339 gets wrong stays inside the top_k array on an unpatched build.
    assert len(_PROBE_INDICES) == len(_PROBE_K) == 2
    for cfg in tracked + [_PROBE_CFG]:
        finding = _LEDGER.find(cfg)
        if finding is not None:
            assert "#5339" in finding.reason
            # Not quarantined: the case runs, so the entry can be seen to go stale.
            assert not finding.quarantine


def test_gof_rejects_invalid_inputs():
    """Invalid statistics raise instead of returning a number a caller would compare."""
    bad = [
        ([0, 0], [0.5, 0.5]),  # no draws
        ([10, -1], [0.5, 0.5]),  # negative count
        ([10, 10], [float("nan"), 0.5]),  # non-finite target
        ([10, 10], [0.2, 0.2]),  # target does not sum to one
        ([10.5, 9.5], [0.5, 0.5]),  # non-integer counts
        ([10, 10], [1.1, -0.1]),  # negative target
        ([99, 1], [1.0, 0.0]),  # draw on a zero-target class
        ([100, 0], [1.0, 0.0]),  # single bin: no degrees of freedom
    ]
    for counts, target in bad:
        with pytest.raises(ValueError):
            chi_square_gof(
                torch.tensor(counts, dtype=torch.float64),
                torch.tensor(target, dtype=torch.float64),
            )


def test_gof_result_shapes_and_range():
    """A valid call returns finite scalars and a p-value in [0, 1]."""
    target = _geometric(8, 8, 0.7)
    counts = (target * SAMPLE_COUNT).round().to(torch.int64)
    counts[0] += SAMPLE_COUNT - int(counts.sum())
    res = chi_square_gof(counts, target)
    assert isinstance(res.dof, int) and res.dof >= 1
    assert 0.0 <= res.p_value <= 1.0
    assert res.tvd >= 0.0 and res.max_resid >= 0.0
    assert res.statistic >= 0.0
